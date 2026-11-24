"""
Persistent storage for trainings.

One row per training (one launch of a Kubernetes training Job). Two backends
share the same SQL and are selected with the DB_BACKEND setting:

* ``sqlite``   - a file inside the kmlops container (no extra dependencies).
* ``postgres`` - an external PostgreSQL server (needs ``psycopg``).

No sample or seed data is ever inserted: rows only come from POST /experiments.
"""
import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, List, Optional

from src.config import Settings
from src.schemas import ACTIVE_STATUSES, ALL_STATUSES


class ActiveTrainingExists(Exception):
    """Raised when an experiment already has a training in progress."""


class ActiveModelTrainingExists(ActiveTrainingExists):
    """Raised when the model (model_name) already has a training in progress, in any experiment."""


COLUMNS = (
    "training_id",
    "experiment_name",
    "status",
    "status_reason",
    "model_type",
    "model_name",
    "config",
    "dependencies",
    "csv_filename",
    "minio_path",
    "job_name",
    "mlflow_run_id",
    "created_at",
    "updated_at",
)
JSON_COLUMNS = ("config", "dependencies")
UPDATABLE_COLUMNS = ("status", "status_reason", "job_name", "mlflow_run_id", "minio_path")

ONE_ACTIVE_INDEX = "trainings_one_active_idx"
ONE_ACTIVE_PER_MODEL_INDEX = "trainings_one_active_per_model_idx"

_status_list = ", ".join(f"'{s}'" for s in ALL_STATUSES)
_active_list = ", ".join(f"'{s}'" for s in ACTIVE_STATUSES)

SCHEMA = (
    f"""
    CREATE TABLE IF NOT EXISTS trainings (
        training_id     TEXT PRIMARY KEY,
        experiment_name TEXT NOT NULL,
        status          TEXT NOT NULL CHECK (status IN ({_status_list})),
        status_reason   TEXT,
        model_type      TEXT,
        model_name      TEXT,
        config          TEXT,
        dependencies    TEXT NOT NULL DEFAULT '[]',
        csv_filename    TEXT,
        minio_path      TEXT,
        job_name        TEXT,
        mlflow_run_id   TEXT,
        created_at      BIGINT NOT NULL,
        updated_at      BIGINT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS trainings_experiment_idx ON trainings (experiment_name, created_at)",
    # At most one training in progress per experiment, enforced by the database.
    f"""
    CREATE UNIQUE INDEX IF NOT EXISTS {ONE_ACTIVE_INDEX} ON trainings (experiment_name)
    WHERE status IN ({_active_list})
    """,
)

# At most one training in progress per model (model_name), in any experiment. Together with the
# feature check of POST /experiments it guarantees that two trainings with different features
# cannot become versions of the same model at the same time. Rows without model_name (NULL) never
# conflict. Created apart from SCHEMA because existing data may violate it (see TrainingStore.init).
MODEL_INDEX = f"""
    CREATE UNIQUE INDEX IF NOT EXISTS {ONE_ACTIVE_PER_MODEL_INDEX} ON trainings (model_name)
    WHERE status IN ({_active_list})
    """


def now_ms() -> int:
    return int(time.time() * 1000)


def _decode(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    record = dict(row)
    for col in JSON_COLUMNS:
        raw = record.get(col)
        record[col] = json.loads(raw) if raw is not None else None
    if record.get("dependencies") is None:
        record["dependencies"] = []
    return record


class TrainingStore:
    """Backend-independent operations. Subclasses implement `_execute`.

    SQL is written with `?` placeholders; backends translate them if needed.
    """

    def _execute(self, sql: str, params: Iterable[Any] = (), fetch: bool = False) -> Any:
        """Runs one statement. Returns a list of dict rows if `fetch`, else the rowcount."""
        raise NotImplementedError

    def _active_conflict(self, exc: Exception) -> Optional[str]:
        """ONE_ACTIVE_INDEX or ONE_ACTIVE_PER_MODEL_INDEX if `exc` is a violation of it, else None."""
        raise NotImplementedError

    def init(self) -> None:
        for statement in SCHEMA:
            self._execute(statement)
        try:
            self._execute(MODEL_INDEX)
        except Exception as exc:
            # Only possible if two trainings of the same model are in progress right now (created
            # before this index existed). Failing would stop kmlops, and with it the poller that
            # finishes those trainings, so it starts without the index and retries on next startup.
            print(
                f"WARNING: could not create {ONE_ACTIVE_PER_MODEL_INDEX} ({exc}). Several trainings of the same "
                "model are in progress; until kmlops restarts after they finish, two trainings of one model "
                "can run at the same time."
            )

    def close(self) -> None:
        pass

    # ---- writes -------------------------------------------------------------

    def insert(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Inserts a training. Raises ActiveTrainingExists if the experiment already has one in progress,
        or ActiveModelTrainingExists if its model_name has one in progress in another experiment."""
        ts = now_ms()
        values = dict(record)
        values.setdefault("created_at", ts)
        values.setdefault("updated_at", ts)
        for col in JSON_COLUMNS:
            if col in values:
                values[col] = json.dumps(values[col] if values[col] is not None else ([] if col == "dependencies" else None))
        cols = [c for c in COLUMNS if c in values]
        sql = f"INSERT INTO trainings ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
        try:
            self._execute(sql, [values[c] for c in cols])
        except Exception as exc:
            conflict = self._active_conflict(exc)
            if conflict == ONE_ACTIVE_INDEX:
                raise ActiveTrainingExists(record["experiment_name"]) from exc
            if conflict == ONE_ACTIVE_PER_MODEL_INDEX:
                # Both indexes can be violated at once; the database may report either.
                if self.active(record["experiment_name"]):
                    raise ActiveTrainingExists(record["experiment_name"]) from exc
                raise ActiveModelTrainingExists(record.get("model_name")) from exc
            raise
        created = self.get(record["training_id"])
        assert created is not None
        return created

    def update(self, training_id: str, expected_status: Optional[Iterable[str]] = None, **fields: Any) -> bool:
        """Updates some fields of a training.

        If `expected_status` is given, the update only happens when the current status is one of
        them (compare-and-set), so concurrent writers (poller, cancel, launcher) never overwrite
        each other. Returns True if a row was updated.
        """
        unknown = set(fields) - set(UPDATABLE_COLUMNS)
        if unknown:
            raise ValueError(f"Cannot update columns: {sorted(unknown)}")
        sets = [f"{col} = ?" for col in fields] + ["updated_at = ?"]
        params: List[Any] = list(fields.values()) + [now_ms(), training_id]
        sql = f"UPDATE trainings SET {', '.join(sets)} WHERE training_id = ?"
        if expected_status is not None:
            expected = list(expected_status)
            sql += f" AND status IN ({', '.join('?' for _ in expected)})"
            params += expected
        return self._execute(sql, params) > 0

    def delete(self, training_id: str) -> bool:
        return self._execute("DELETE FROM trainings WHERE training_id = ?", [training_id]) > 0

    def delete_by_name(self, experiment_name: str) -> int:
        return self._execute("DELETE FROM trainings WHERE experiment_name = ?", [experiment_name])

    # ---- reads --------------------------------------------------------------

    def get(self, training_id: str) -> Optional[Dict[str, Any]]:
        rows = self._execute("SELECT * FROM trainings WHERE training_id = ?", [training_id], fetch=True)
        return _decode(rows[0]) if rows else None

    def list_all(self) -> List[Dict[str, Any]]:
        """All trainings, newest first."""
        rows = self._execute("SELECT * FROM trainings ORDER BY created_at DESC", fetch=True)
        return [r for r in (_decode(row) for row in rows) if r is not None]

    def list_by_name(self, experiment_name: str) -> List[Dict[str, Any]]:
        """Trainings of one experiment, newest first."""
        rows = self._execute(
            "SELECT * FROM trainings WHERE experiment_name = ? ORDER BY created_at DESC",
            [experiment_name],
            fetch=True,
        )
        return [r for r in (_decode(row) for row in rows) if r is not None]

    def latest(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        trainings = self.list_by_name(experiment_name)
        return trainings[0] if trainings else None

    def latest_model_training(self, model_name: str, statuses: Iterable[str]) -> Optional[Dict[str, Any]]:
        """Newest training of a model (any experiment) in one of `statuses`."""
        wanted = list(statuses)
        sql = (
            f"SELECT * FROM trainings WHERE model_name = ? AND status IN ({', '.join('?' for _ in wanted)})"
            " ORDER BY created_at DESC LIMIT 1"
        )
        rows = self._execute(sql, [model_name, *wanted], fetch=True)
        return _decode(rows[0]) if rows else None

    def active(self, experiment_name: Optional[str] = None) -> List[Dict[str, Any]]:
        placeholders = ", ".join("?" for _ in ACTIVE_STATUSES)
        sql = f"SELECT * FROM trainings WHERE status IN ({placeholders})"
        params: List[Any] = list(ACTIVE_STATUSES)
        if experiment_name is not None:
            sql += " AND experiment_name = ?"
            params.append(experiment_name)
        rows = self._execute(sql + " ORDER BY created_at DESC", params, fetch=True)
        return [r for r in (_decode(row) for row in rows) if r is not None]


class SQLiteStore(TrainingStore):
    def __init__(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        self._lock = threading.Lock()
        # isolation_level=None -> autocommit; every statement is its own transaction.
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")

    def _execute(self, sql: str, params: Iterable[Any] = (), fetch: bool = False) -> Any:
        with self._lock:
            cursor = self._conn.execute(sql, list(params))
            if fetch:
                return [dict(row) for row in cursor.fetchall()]
            return cursor.rowcount

    def _active_conflict(self, exc: Exception) -> Optional[str]:
        # SQLite names the columns, not the index: "UNIQUE constraint failed: trainings.model_name".
        if not isinstance(exc, sqlite3.IntegrityError):
            return None
        if "trainings.experiment_name" in str(exc):
            return ONE_ACTIVE_INDEX
        if "trainings.model_name" in str(exc):
            return ONE_ACTIVE_PER_MODEL_INDEX
        return None

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class PostgresStore(TrainingStore):
    def __init__(self, settings: Settings) -> None:
        # Imported lazily so the SQLite backend does not need psycopg.
        import psycopg
        from psycopg.rows import dict_row

        self._psycopg = psycopg
        self._dict_row = dict_row
        self._conninfo = dict(
            host=settings.postgres_host,
            port=settings.postgres_port,
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            connect_timeout=10,
        )

    def _execute(self, sql: str, params: Iterable[Any] = (), fetch: bool = False) -> Any:
        # Our SQL never contains a literal '?', so translating placeholders is safe.
        pg_sql = sql.replace("?", "%s")
        with self._psycopg.connect(**self._conninfo, autocommit=True, row_factory=self._dict_row) as conn:
            with conn.cursor() as cursor:
                cursor.execute(pg_sql, list(params))
                if fetch:
                    return [dict(row) for row in cursor.fetchall()]
                return cursor.rowcount

    def _active_conflict(self, exc: Exception) -> Optional[str]:
        if not isinstance(exc, self._psycopg.errors.UniqueViolation):
            return None
        name = getattr(exc.diag, "constraint_name", None)
        return name if name in (ONE_ACTIVE_INDEX, ONE_ACTIVE_PER_MODEL_INDEX) else None


_store: Optional[TrainingStore] = None


def init_store(settings: Settings) -> TrainingStore:
    """Creates the configured store and its schema. Called once at startup."""
    global _store
    if settings.db_backend == "postgres":
        store: TrainingStore = PostgresStore(settings)
        where = f"postgres://{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    else:
        store = SQLiteStore(settings.sqlite_path)
        where = f"sqlite:{settings.sqlite_path}"
    store.init()
    _store = store
    print(f"Training store ready ({where})")
    return store


def get_store() -> TrainingStore:
    if _store is None:
        raise RuntimeError("The training store has not been initialised (init_store was not called)")
    return _store


def close_store() -> None:
    global _store
    if _store is not None:
        _store.close()
        _store = None
