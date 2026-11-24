import json
from types import SimpleNamespace

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

import src.api.experiments as api_module
import src.services.experiments as experiments_module
import src.services.feature_schema as feature_module
from src.db.store import (
    ONE_ACTIVE_PER_MODEL_INDEX,
    SCHEMA,
    ActiveModelTrainingExists,
    ActiveTrainingExists,
    SQLiteStore,
)
from src.services.feature_schema import InvalidFeatures, feature_signature

DEFAULT_INPUTS = (("temp", "number"), ("rpm", "number"))
DEFAULT_OUTPUTS = (("vibration", "number"),)


def ludwig_config(inputs=DEFAULT_INPUTS, outputs=DEFAULT_OUTPUTS, **extra):
    return {
        "input_features": [{"name": n, "type": t} for n, t in inputs],
        "output_features": [{"name": n, "type": t} for n, t in outputs],
        **extra,
    }


@pytest.fixture
def store(monkeypatch):
    s = SQLiteStore(":memory:")
    s.init()
    launched = []
    monkeypatch.setattr(experiments_module, "get_store", lambda: s)
    monkeypatch.setattr(experiments_module, "upload_csv_to_minio", lambda content, tid, filename: f"{tid}/{filename}")
    monkeypatch.setattr(experiments_module, "train_model_job", lambda **kwargs: launched.append(kwargs))
    s.launched = launched
    yield s
    s.close()


def add_training(store, training_id, status="completed", experiment="exp-a", model=None, config=None, created_at=1):
    store.insert({
        "training_id": training_id,
        "experiment_name": experiment,
        "status": status,
        "model_type": "ecd",
        "model_name": model or experiment,
        "config": config if config is not None else ludwig_config(),
        "dependencies": [],
        "created_at": created_at,
    })


def post(client, config, experiment="exp-a", **extra):
    return client.post(
        "/api/mlops/experiments",
        files={"csv_file": ("data.csv", b"temp,rpm,vibration\n1,2,3\n", "text/csv")},
        data={"model_type": "ecd", "config_json": json.dumps(config), "experiment_name": experiment, **extra},
    )


# ---- fake MLflow registry ------------------------------------------------------


class FakeMlflowClient:
    def __init__(self, models=None, runs=None, error=None):
        self.models = models or {}  # name -> list of versions
        self.runs = runs or {}  # run_id -> params
        self.error = error

    def get_registered_model(self, name):
        if self.error is not None:
            raise self.error
        if name not in self.models:
            raise MlflowException("not found", error_code=RESOURCE_DOES_NOT_EXIST)
        return SimpleNamespace(name=name, latest_versions=self.models[name])

    def get_run(self, run_id):
        if run_id not in self.runs:
            raise MlflowException("not found", error_code=RESOURCE_DOES_NOT_EXIST)
        return SimpleNamespace(data=SimpleNamespace(params=self.runs[run_id]))


def version(number, run_id, source="mlflow-artifacts:/1/run/artifacts/ludwig-model"):
    return SimpleNamespace(version=str(number), run_id=run_id, source=source)


def ludwig_params(inputs=DEFAULT_INPUTS, outputs=DEFAULT_OUTPUTS):
    """As Ludwig logs them: the repr of the config lists."""
    config = ludwig_config(inputs, outputs)
    return {"input_features": repr(config["input_features"]), "output_features": repr(config["output_features"])}


@pytest.fixture
def mlflow_registry(monkeypatch):
    """Enables MLflow and installs a fake registry. Returns a setter."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.invalid")
    monkeypatch.setattr(experiments_module, "prepare_experiment_for_training", lambda name: None)
    artifacts = {}

    def fake_download(artifact_uri, dst_path):
        if artifact_uri not in artifacts:
            raise OSError("artifact not found")
        path = f"{dst_path}/model_hyperparameters.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(artifacts[artifact_uri], f)
        return path

    monkeypatch.setattr(feature_module.mlflow.artifacts, "download_artifacts", fake_download)

    def _set(**kwargs):
        client = FakeMlflowClient(**kwargs)
        monkeypatch.setattr(feature_module, "MlflowClient", lambda: client)
        return artifacts

    return _set


# ---- feature_signature ---------------------------------------------------------


def test_signature_ignores_order_and_other_settings():
    a = ludwig_config(inputs=(("temp", "number"), ("rpm", "number")))
    b = ludwig_config(inputs=(("rpm", "NUMBER"), ("temp", "number")), trainer={"epochs": 5})
    b["input_features"][0]["encoder"] = {"type": "dense"}
    assert feature_signature(a) == feature_signature(b)


def test_old_numerical_type_is_number():
    assert feature_signature(ludwig_config(inputs=(("rpm", "numerical"),))) == feature_signature(
        ludwig_config(inputs=(("rpm", "number"),))
    )


@pytest.mark.parametrize(
    "config",
    [
        [],
        {},
        {"input_features": [], "output_features": [{"name": "y", "type": "number"}]},
        {"input_features": [{"name": "x", "type": "number"}]},
        {"input_features": [{"name": "x"}], "output_features": [{"name": "y", "type": "number"}]},
        {"input_features": ["x"], "output_features": [{"name": "y", "type": "number"}]},
        {
            "input_features": [{"name": "x", "type": "number"}, {"name": "x", "type": "category"}],
            "output_features": [{"name": "y", "type": "number"}],
        },
    ],
)
def test_invalid_features(config):
    with pytest.raises(InvalidFeatures):
        feature_signature(config)


# ---- store --------------------------------------------------------------------


def test_latest_model_training_filters_status_and_model(store):
    add_training(store, "old", created_at=1)
    add_training(store, "new", experiment="exp-b", model="exp-a", created_at=2)
    add_training(store, "failed", status="failed", created_at=3)
    add_training(store, "other", experiment="exp-c", created_at=4)
    assert store.latest_model_training("exp-a", ("completed",))["training_id"] == "new"
    assert store.latest_model_training("exp-a", ("failed",))["training_id"] == "failed"
    assert store.latest_model_training("missing", ("completed",)) is None


def test_one_active_training_per_model_in_any_experiment(store):
    add_training(store, "t1", status="training", experiment="exp-a", model="shared")
    with pytest.raises(ActiveModelTrainingExists):
        add_training(store, "t2", status="queued", experiment="exp-b", model="shared")
    # Finished trainings and other models do not conflict.
    add_training(store, "t3", status="completed", experiment="exp-b", model="shared")
    add_training(store, "t4", status="queued", experiment="exp-b", model="other")


def test_same_experiment_conflict_is_reported_as_experiment(store):
    add_training(store, "t1", status="training")
    with pytest.raises(ActiveTrainingExists) as info:
        add_training(store, "t2", status="queued")
    assert not isinstance(info.value, ActiveModelTrainingExists)


def test_init_tolerates_existing_duplicates_of_the_model_index():
    s = SQLiteStore(":memory:")
    for statement in SCHEMA:  # an older database, without the per-model index
        s._execute(statement)
    for tid, exp in (("t1", "exp-a"), ("t2", "exp-b")):
        s._execute(
            "INSERT INTO trainings (training_id, experiment_name, status, model_name, created_at, updated_at) "
            "VALUES (?, ?, 'training', 'shared', 1, 1)",
            [tid, exp],
        )
    s.init()  # does not raise: kmlops must start to finish those trainings
    indexes = s._execute("SELECT name FROM sqlite_master WHERE type = 'index'", fetch=True)
    assert ONE_ACTIVE_PER_MODEL_INDEX not in {row["name"] for row in indexes}
    s.close()


# ---- POST /api/mlops/experiments: local reference -----------------------------


def test_first_training_accepts_any_features(client, store):
    assert post(client, ludwig_config()).status_code == 200


def test_model_name_defaults_to_experiment_name(client, store):
    assert post(client, ludwig_config(), experiment="exp-z").status_code == 200
    assert store.launched[0]["model_name"] == "exp-z"
    assert store.latest("exp-z")["model_name"] == "exp-z"


def test_explicit_model_name_is_used(client, store):
    assert post(client, ludwig_config(), experiment="exp-z", model_name="my-model").status_code == 200
    assert store.launched[0]["model_name"] == "my-model"
    assert store.latest("exp-z")["model_name"] == "my-model"


def test_several_experiments_can_feed_one_model(client, store):
    add_training(store, "t1", experiment="exp-a", model="shared")
    assert post(client, ludwig_config(), experiment="exp-b", model_name="shared").status_code == 200


def test_different_features_for_a_model_used_in_another_experiment_is_409(client, store):
    add_training(store, "t1", experiment="exp-a", model="shared")
    response = post(client, ludwig_config(outputs=(("fault_type", "category"),)), experiment="exp-b", model_name="shared")
    assert response.status_code == 409
    assert "model 'shared'" in response.json()["detail"]
    assert "experiment 'exp-a'" in response.json()["detail"]
    assert store.list_by_name("exp-b") == []


def test_model_in_progress_in_another_experiment_is_409(client, store):
    add_training(store, "t1", status="training", experiment="exp-a", model="shared")
    response = post(client, ludwig_config(), experiment="exp-b", model_name="shared")
    assert response.status_code == 409
    assert "Model 'shared' already has a training in progress" in response.json()["detail"]


def test_one_experiment_can_hold_several_models(client, store):
    add_training(store, "t1", experiment="exp-a", model="model-1")
    config = ludwig_config(outputs=(("fault_type", "category"),))
    assert post(client, config, experiment="exp-a", model_name="model-2").status_code == 200


def test_same_features_is_a_new_version(client, store):
    add_training(store, "t1")
    config = ludwig_config(inputs=(("rpm", "number"), ("temp", "number")), trainer={"epochs": 50})
    assert post(client, config).status_code == 200


@pytest.mark.parametrize(
    "config, fragment",
    [
        (ludwig_config(outputs=(("fault_type", "category"),)), "output_features"),
        (ludwig_config(inputs=(("temp", "number"),)), "input_features missing ['rpm']"),
        (ludwig_config(inputs=(("temp", "number"), ("rpm", "number"), ("load", "number"))), "['load']"),
        (ludwig_config(outputs=(("vibration", "category"),)), "'vibration' has type 'category' instead of 'number'"),
    ],
)
def test_different_features_is_409(client, store, config, fragment):
    add_training(store, "t1")
    response = post(client, config)
    assert response.status_code == 409
    assert "model 'exp-a'" in response.json()["detail"]
    assert fragment in response.json()["detail"]
    assert len(store.list_by_name("exp-a")) == 1


def test_other_model_is_independent(client, store):
    add_training(store, "t1")
    assert post(client, ludwig_config(outputs=(("fault_type", "category"),)), experiment="exp-b").status_code == 200


def test_failed_and_cancelled_trainings_are_not_reference(client, store):
    add_training(store, "f1", status="failed", created_at=1)
    add_training(store, "c1", status="cancelled", created_at=2)
    assert post(client, ludwig_config(outputs=(("fault_type", "category"),))).status_code == 200


def test_latest_version_is_the_reference(client, store):
    add_training(store, "t1", config=ludwig_config(outputs=(("old_target", "number"),)), created_at=1)
    add_training(store, "t2", created_at=2)
    assert post(client, ludwig_config()).status_code == 200


def test_invalid_features_is_422(client, store):
    response = post(client, {"input_features": [{"name": "x", "type": "number"}]})
    assert response.status_code == 422
    assert "output_features" in response.json()["detail"]


# ---- POST /api/mlops/experiments: MLflow registry reference -------------------


def test_mlflow_only_model_accepts_same_features(client, store, mlflow_registry):
    mlflow_registry(models={"exp-a": [version(3, "r3")]}, runs={"r3": ludwig_params()})
    assert post(client, ludwig_config()).status_code == 200


def test_mlflow_only_model_rejects_different_features(client, store, mlflow_registry):
    mlflow_registry(models={"exp-a": [version(3, "r3")]}, runs={"r3": ludwig_params()})
    response = post(client, ludwig_config(outputs=(("fault_type", "category"),)))
    assert response.status_code == 409
    assert "model version 3" in response.json()["detail"]
    assert store.list_by_name("exp-a") == []


def test_latest_registered_version_is_the_reference(client, store, mlflow_registry):
    mlflow_registry(
        models={"exp-a": [version(1, "r1"), version(2, "r2")]},
        runs={"r1": ludwig_params(outputs=(("old_target", "number"),)), "r2": ludwig_params()},
    )
    assert post(client, ludwig_config()).status_code == 200


def test_falls_back_to_model_hyperparameters_artifact(client, store, mlflow_registry):
    source = "mlflow-artifacts:/1/r1/artifacts/ludwig-model"
    artifacts = mlflow_registry(models={"exp-a": [version(1, "r1", source)]}, runs={"r1": {}})
    artifacts[f"{source}/model/model_hyperparameters.json"] = ludwig_config()
    assert post(client, ludwig_config()).status_code == 200
    assert post(client, ludwig_config(inputs=(("temp", "number"),))).status_code == 409


def test_truncated_params_fall_back_to_artifact(client, store, mlflow_registry):
    source = "mlflow-artifacts:/1/r1/artifacts/ludwig-model"
    params = {"input_features": "[{'name': 'temp', 'ty", "output_features": "[{'name': 'vibration', 'type': 'number'}]"}
    artifacts = mlflow_registry(models={"exp-a": [version(1, "r1", source)]}, runs={"r1": params})
    artifacts[f"{source}/model/model_hyperparameters.json"] = ludwig_config()
    assert post(client, ludwig_config(outputs=(("other", "number"),))).status_code == 409


def test_version_with_unknown_features_is_not_checked(client, store, mlflow_registry):
    mlflow_registry(models={"exp-a": [version(1, "r1", "models:/m-123")]}, runs={"r1": {}})
    assert post(client, ludwig_config(outputs=(("anything", "binary"),))).status_code == 200


def test_unregistered_model_has_no_reference(client, store, mlflow_registry):
    mlflow_registry(models={})
    assert post(client, ludwig_config()).status_code == 200


def test_local_and_mlflow_references_must_both_match(client, store, mlflow_registry):
    add_training(store, "t1")
    mlflow_registry(models={"exp-a": [version(1, "r1")]}, runs={"r1": ludwig_params(outputs=(("x", "number"),))})
    assert post(client, ludwig_config()).status_code == 409


def test_mlflow_error_is_502_and_creates_nothing(client, store, mlflow_registry):
    mlflow_registry(error=MlflowException("connection refused"))
    response = post(client, ludwig_config())
    assert response.status_code == 502
    assert "MLflow" in response.json()["detail"]
    assert store.list_by_name("exp-a") == []


def test_api_reports_mlflow_disabled_check_uses_only_local(client, store):
    # MLflow disabled by default in the tests: the registry is never queried.
    assert api_module.mlflow_enabled() is False
    add_training(store, "t1")
    assert post(client, ludwig_config()).status_code == 200
