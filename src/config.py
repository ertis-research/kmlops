"""
Centralised configuration for kmlops.

Every setting comes from environment variables (a local ``.env`` file in
development, a Secret/ConfigMap in Kubernetes). Required variables have no
default: if one is missing, kmlops refuses to start and names the variable.
"""
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Versions of the packages the training pod installs. Keep them in sync with
# requirements.txt (a test checks it); this is the only place with these defaults.
DEFAULT_TRAINING_MLFLOW_VERSION = "3.18.0"
DEFAULT_TRAINING_MINIO_VERSION = "7.2.20"

# These values end up inside a shell command, so they are validated strictly.
_VERSION_RE = re.compile(r"[0-9][0-9A-Za-z.+!_-]*")
_PACKAGE_NAME_RE = re.compile(r"[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?")
# Kubernetes: Secret name = DNS-1123 subdomain, Secret data key = [-._a-zA-Z0-9]+ (both <= 253 chars).
_DNS_SUBDOMAIN_RE = re.compile(r"[a-z0-9]([-a-z0-9.]*[a-z0-9])?")
_SECRET_KEY_RE = re.compile(r"[-._a-zA-Z0-9]+")
_K8S_NAME_MAX_LENGTH = 253


class ConfigError(RuntimeError):
    """Raised when a required setting is missing or invalid."""


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


def _optional(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name, "").strip()
    return value if value else default


def _int(name: str, default: int, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    raw = _optional(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"Environment variable {name} must be an integer, got '{raw}'")
    if minimum is not None and value < minimum:
        raise ConfigError(f"Environment variable {name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"Environment variable {name} must be <= {maximum}, got {value}")
    return value


def _bool(name: str, default: bool) -> bool:
    raw = _optional(name)
    if raw is None:
        return default
    if raw.lower() in ("1", "true", "yes", "on"):
        return True
    if raw.lower() in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"Environment variable {name} must be true/false, got '{raw}'")


def _version(name: str, default: str) -> str:
    value = _optional(name, default) or default
    if not _VERSION_RE.fullmatch(value):
        raise ConfigError(f"Invalid {name}={value!r}: expected a version such as {default}")
    return value


def _pip_extra_packages(name: str) -> tuple[str, ...]:
    """Parses a comma-separated list of ``name==version`` requirements."""
    raw = _optional(name)
    if raw is None:
        return ()
    packages = []
    for item in (part.strip() for part in raw.split(",")):
        if not item:
            continue
        package, sep, version = item.partition("==")
        if not sep or not _PACKAGE_NAME_RE.fullmatch(package) or not _VERSION_RE.fullmatch(version):
            raise ConfigError(
                f"Invalid {name} entry {item!r}: expected 'name==version' (e.g. scikit-learn==1.5.2)"
            )
        packages.append(item)
    return tuple(packages)


def _k8s_identifier(name: str, value: str, pattern: "re.Pattern[str]", what: str) -> str:
    if len(value) > _K8S_NAME_MAX_LENGTH or not pattern.fullmatch(value):
        raise ConfigError(f"Invalid {name}={value!r}: not a valid {what}")
    return value


@dataclass(frozen=True)
class Settings:
    # MinIO
    minio_url: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool
    bucket_name: str
    # Optional Secret the training Jobs read the MinIO credentials from (secretKeyRef)
    # instead of receiving them in plain text in the Job spec.
    minio_credentials_secret: Optional[str]
    minio_credentials_access_key_key: str
    minio_credentials_secret_key_key: str

    # Kubernetes
    kube_config_path: str
    k8s_namespace: str
    training_image: str
    training_job_ttl_seconds: int
    training_job_backoff_limit: int
    training_mlflow_version: str
    training_minio_version: str
    training_pip_extra_packages: tuple[str, ...]

    # MLflow (optional: without it kmlops only tracks local trainings)
    mlflow_tracking_uri: Optional[str]

    # Status tracking
    status_poll_interval_seconds: int
    launch_timeout_seconds: int

    # Storage
    db_backend: str
    sqlite_path: str
    postgres_host: Optional[str]
    postgres_port: int
    postgres_db: Optional[str]
    postgres_user: Optional[str]
    postgres_password: Optional[str]

    @property
    def mlflow_enabled(self) -> bool:
        return bool(self.mlflow_tracking_uri)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    db_backend = (_optional("DB_BACKEND", "sqlite") or "sqlite").lower()
    if db_backend not in ("sqlite", "postgres"):
        raise ConfigError(f"DB_BACKEND must be 'sqlite' or 'postgres', got '{db_backend}'")

    minio_credentials_secret = _optional("MINIO_CREDENTIALS_SECRET")
    if minio_credentials_secret is not None:
        _k8s_identifier("MINIO_CREDENTIALS_SECRET", minio_credentials_secret, _DNS_SUBDOMAIN_RE, "Kubernetes Secret name")

    settings = Settings(
        minio_url=_required("MINIO_URL"),
        minio_access_key=_required("MINIO_ACCESS_KEY"),
        minio_secret_key=_required("MINIO_SECRET_KEY"),
        minio_secure=_bool("MINIO_SECURE", True),
        bucket_name=_required("BUCKET_NAME"),
        minio_credentials_secret=minio_credentials_secret,
        minio_credentials_access_key_key=_k8s_identifier(
            "MINIO_CREDENTIALS_ACCESS_KEY_KEY",
            _optional("MINIO_CREDENTIALS_ACCESS_KEY_KEY", "MINIO_ACCESS_KEY") or "",
            _SECRET_KEY_RE,
            "Secret key name",
        ),
        minio_credentials_secret_key_key=_k8s_identifier(
            "MINIO_CREDENTIALS_SECRET_KEY_KEY",
            _optional("MINIO_CREDENTIALS_SECRET_KEY_KEY", "MINIO_SECRET_KEY") or "",
            _SECRET_KEY_RE,
            "Secret key name",
        ),
        kube_config_path=os.path.expanduser(_optional("KUBERNETES_CONFIG_PATH", "~/.kube/config") or ""),
        k8s_namespace=_required("KUBERNETES_NAMESPACE"),
        training_image=_optional("TRAINING_IMAGE", "ludwigai/ludwig:latest") or "",
        training_job_ttl_seconds=_int("TRAINING_JOB_TTL_SECONDS", 60, minimum=0),
        training_job_backoff_limit=_int("TRAINING_JOB_BACKOFF_LIMIT", 1, minimum=0),
        training_mlflow_version=_version("TRAINING_MLFLOW_VERSION", DEFAULT_TRAINING_MLFLOW_VERSION),
        training_minio_version=_version("TRAINING_MINIO_VERSION", DEFAULT_TRAINING_MINIO_VERSION),
        training_pip_extra_packages=_pip_extra_packages("TRAINING_PIP_EXTRA_PACKAGES"),
        mlflow_tracking_uri=_optional("MLFLOW_TRACKING_URI"),
        status_poll_interval_seconds=_int("STATUS_POLL_INTERVAL_SECONDS", 10, minimum=1),
        launch_timeout_seconds=_int("LAUNCH_TIMEOUT_SECONDS", 300, minimum=1),
        db_backend=db_backend,
        sqlite_path=_optional("SQLITE_PATH", "/data/kmlops.db") or "",
        postgres_host=_optional("POSTGRES_HOST"),
        postgres_port=_int("POSTGRES_PORT", 5432, minimum=1, maximum=65535),
        postgres_db=_optional("POSTGRES_DB"),
        postgres_user=_optional("POSTGRES_USER"),
        postgres_password=_optional("POSTGRES_PASSWORD"),
    )

    if db_backend == "postgres":
        for name, value in (
            ("POSTGRES_HOST", settings.postgres_host),
            ("POSTGRES_DB", settings.postgres_db),
            ("POSTGRES_USER", settings.postgres_user),
            ("POSTGRES_PASSWORD", settings.postgres_password),
        ):
            if not value:
                raise ConfigError(f"Missing required environment variable for DB_BACKEND=postgres: {name}")

    return settings


def config_warnings(settings: Settings) -> list[str]:
    """Returns human-readable warnings about a valid but questionable configuration."""
    warnings = []
    if not settings.minio_credentials_secret:
        warnings.append(
            "MINIO_CREDENTIALS_SECRET is not set: the MinIO credentials are copied in plain text into the "
            "spec of the training Jobs (visible with 'kubectl get job -o yaml'). Set MINIO_CREDENTIALS_SECRET "
            "to the name of a Secret holding them so the Jobs reference it with secretKeyRef instead."
        )
    return warnings
