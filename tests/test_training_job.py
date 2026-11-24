"""Tests of how the training Job is built: credentials, pinned versions and config validation."""
import json
import re
from pathlib import Path

import pytest
from kubernetes import client

from src.config import (
    DEFAULT_TRAINING_MINIO_VERSION,
    DEFAULT_TRAINING_MLFLOW_VERSION,
    ConfigError,
    config_warnings,
    get_settings,
)
from src.training.launcher import build_training_job

ACCESS_KEY = "test-access-key"
SECRET_KEY = "test-secret-key"


def _job():
    return build_training_job("abc123", {"model_type": "ecd"}, "path/data.csv", "exp", "model")


def _container(job):
    return job.spec.template.spec.containers[0]


def _env(job):
    return {e.name: e for e in _container(job).env}


def _command(job):
    return _container(job).args[0]


def _serialized(job) -> str:
    return json.dumps(client.ApiClient().sanitize_for_serialization(job))


def _assert_secret_ref(env_var, name, key):
    assert env_var.value is None
    ref = env_var.value_from.secret_key_ref
    assert ref.name == name
    assert ref.key == key


def test_without_credentials_secret_values_are_literal(monkeypatch):
    monkeypatch.delenv("MINIO_CREDENTIALS_SECRET", raising=False)
    env = _env(_job())
    assert env["MINIO_ACCESS_KEY"].value == ACCESS_KEY
    assert env["MINIO_ACCESS_KEY"].value_from is None
    assert env["MINIO_SECRET_KEY"].value == SECRET_KEY
    assert env["MINIO_SECRET_KEY"].value_from is None


def test_env_order_is_kept(monkeypatch):
    monkeypatch.setenv("MINIO_CREDENTIALS_SECRET", "my-minio-secret")
    names = [e.name for e in _container(_job()).env]
    assert names[:4] == ["MINIO_URL", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "MINIO_SECURE"]


def test_credentials_secret_uses_secret_key_ref_with_default_keys(monkeypatch):
    monkeypatch.setenv("MINIO_CREDENTIALS_SECRET", "my-minio-secret")
    job = _job()
    env = _env(job)
    for e in _container(job).env:
        assert e.value not in (ACCESS_KEY, SECRET_KEY)
    _assert_secret_ref(env["MINIO_ACCESS_KEY"], "my-minio-secret", "MINIO_ACCESS_KEY")
    _assert_secret_ref(env["MINIO_SECRET_KEY"], "my-minio-secret", "MINIO_SECRET_KEY")
    serialized = _serialized(job)
    assert SECRET_KEY not in serialized
    assert ACCESS_KEY not in serialized
    assert env["MINIO_URL"].value == "minio.invalid:9000"  # the rest stays literal


def test_credentials_secret_with_custom_keys(monkeypatch):
    monkeypatch.setenv("MINIO_CREDENTIALS_SECRET", "my-minio-secret")
    monkeypatch.setenv("MINIO_CREDENTIALS_ACCESS_KEY_KEY", "accessKey")
    monkeypatch.setenv("MINIO_CREDENTIALS_SECRET_KEY_KEY", "secretKey")
    job = _job()
    env = _env(job)
    _assert_secret_ref(env["MINIO_ACCESS_KEY"], "my-minio-secret", "accessKey")
    _assert_secret_ref(env["MINIO_SECRET_KEY"], "my-minio-secret", "secretKey")
    for e in _container(job).env:
        assert e.value not in (ACCESS_KEY, SECRET_KEY)
    assert SECRET_KEY not in _serialized(job)


def test_default_pip_command_pins_versions():
    command = _command(_job())
    assert f'pip install -q "minio=={DEFAULT_TRAINING_MINIO_VERSION}" "mlflow=={DEFAULT_TRAINING_MLFLOW_VERSION}"' in command
    assert 'pip install -q "minio==7.2.20" "mlflow==3.18.0" && cat << \'EOF\' > run.py\n' in command


def test_default_versions_match_requirements_txt():
    requirements = (Path(__file__).parent.parent / "requirements.txt").read_text(encoding="utf-8")

    def pinned(package):
        match = re.search(rf"^{package}==(\S+)\s*$", requirements, re.MULTILINE | re.IGNORECASE)
        assert match, f"{package} is not pinned in requirements.txt"
        return match.group(1)

    assert pinned("minio") == DEFAULT_TRAINING_MINIO_VERSION
    assert pinned("mlflow") == DEFAULT_TRAINING_MLFLOW_VERSION


def test_custom_versions(monkeypatch):
    monkeypatch.setenv("TRAINING_MLFLOW_VERSION", "3.1.0")
    monkeypatch.setenv("TRAINING_MINIO_VERSION", "7.2.15")
    assert 'pip install -q "minio==7.2.15" "mlflow==3.1.0" &&' in _command(_job())


@pytest.mark.parametrize("var", ["TRAINING_MLFLOW_VERSION", "TRAINING_MINIO_VERSION"])
@pytest.mark.parametrize("value", ["3.18.0; rm -rf /", "1 2", "$(whoami)", "latest", '3.18.0"', "`id`", "a&&b", "3.18.0|x", "<3"])
def test_invalid_versions_are_rejected(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    with pytest.raises(ConfigError, match=var):
        get_settings()


def test_extra_packages(monkeypatch):
    monkeypatch.setenv("TRAINING_PIP_EXTRA_PACKAGES", "scikit-learn==1.5.2, pandas==2.2.3,")
    assert get_settings().training_pip_extra_packages == ("scikit-learn==1.5.2", "pandas==2.2.3")
    assert '"mlflow==3.18.0" "scikit-learn==1.5.2" "pandas==2.2.3" &&' in _command(_job())


def test_no_extra_packages_by_default(monkeypatch):
    monkeypatch.delenv("TRAINING_PIP_EXTRA_PACKAGES", raising=False)
    assert get_settings().training_pip_extra_packages == ()


@pytest.mark.parametrize("value", ["pandas", "pandas>=2", "pandas==2; rm -rf /", "pan das==1", "pandas==", "==1.0", "pandas==1.0 2", "a==1,b"])
def test_invalid_extra_packages_are_rejected(monkeypatch, value):
    monkeypatch.setenv("TRAINING_PIP_EXTRA_PACKAGES", value)
    with pytest.raises(ConfigError, match="TRAINING_PIP_EXTRA_PACKAGES"):
        get_settings()


def test_invalid_credentials_secret_name(monkeypatch):
    monkeypatch.setenv("MINIO_CREDENTIALS_SECRET", "My_Secret")
    with pytest.raises(ConfigError, match="MINIO_CREDENTIALS_SECRET"):
        get_settings()


@pytest.mark.parametrize("var", ["MINIO_CREDENTIALS_ACCESS_KEY_KEY", "MINIO_CREDENTIALS_SECRET_KEY_KEY"])
def test_invalid_credentials_key(monkeypatch, var):
    monkeypatch.setenv(var, "bad key")
    with pytest.raises(ConfigError, match=var):
        get_settings()


def test_config_warnings(monkeypatch):
    monkeypatch.delenv("MINIO_CREDENTIALS_SECRET", raising=False)
    warnings = config_warnings(get_settings())
    assert len(warnings) == 1
    assert "MINIO_CREDENTIALS_SECRET" in warnings[0]

    get_settings.cache_clear()
    monkeypatch.setenv("MINIO_CREDENTIALS_SECRET", "my-minio-secret")
    assert config_warnings(get_settings()) == []
