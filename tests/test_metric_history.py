"""GET /api/mlops/experiments/{name}/runs/{run_id}/metrics/history"""
import math
from types import SimpleNamespace

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST

URL = "/api/mlops/experiments/X/runs/r1/metrics/history"


def metric(key, value, step, timestamp):
    return SimpleNamespace(key=key, value=value, step=step, timestamp=timestamp)


HISTORY = {
    "loss": [
        metric("loss", 0.5, 2, 3000),
        metric("loss", math.nan, 1, 2000),
        metric("loss", 0.9, 0, 1000),
        metric("loss", 0.7, 1, 1500),
    ],
    "acc": [metric("acc", 0.1, 0, 1000)],
}


class FakeClient:
    run_experiment_id = "1"
    lifecycle_stage = "active"
    get_run_error = None
    history_error = None

    def get_run(self, run_id):
        if self.get_run_error is not None:
            raise self.get_run_error
        return SimpleNamespace(
            info=SimpleNamespace(
                experiment_id=self.run_experiment_id, lifecycle_stage=self.lifecycle_stage, run_id=run_id
            ),
            data=SimpleNamespace(metrics={"loss": 0.5, "acc": 0.1}),
        )

    def get_metric_history(self, run_id, key):
        if self.history_error is not None:
            raise self.history_error
        return list(HISTORY.get(key, []))


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr("src.api.experiments.mlflow_enabled", lambda: True)
    monkeypatch.setattr("src.services.runs.get_active_experiment", lambda name: SimpleNamespace(experiment_id="1"))
    FakeClient.run_experiment_id = "1"
    FakeClient.lifecycle_stage = "active"
    FakeClient.get_run_error = None
    FakeClient.history_error = None
    monkeypatch.setattr("src.services.runs.MlflowClient", FakeClient)
    return FakeClient


def test_sorted_points_without_nan(client, setup):
    r = client.get(URL, params={"keys": ["loss", "acc", "loss"]})
    assert r.status_code == 200
    body = r.json()
    assert body["run_id"] == "r1"
    assert list(body["metrics"]) == ["loss", "acc"]
    assert body["metrics"]["loss"] == [
        {"step": 0, "value": 0.9, "timestamp": 1000},
        {"step": 1, "value": 0.7, "timestamp": 1500},
        {"step": 2, "value": 0.5, "timestamp": 3000},
    ]
    assert body["metrics"]["acc"] == [{"step": 0, "value": 0.1, "timestamp": 1000}]


def test_keys_omitted_returns_all_run_metrics(client, setup):
    r = client.get(URL)
    assert r.status_code == 200
    assert set(r.json()["metrics"]) == {"loss", "acc"}


def test_missing_key_gives_empty_list(client, setup):
    r = client.get(URL, params={"keys": ["nope"]})
    assert r.status_code == 200
    assert r.json()["metrics"] == {"nope": []}


def test_run_from_other_experiment_404(client, setup):
    setup.run_experiment_id = "2"
    assert client.get(URL).status_code == 404


def test_deleted_run_404(client, setup):
    setup.lifecycle_stage = "deleted"
    assert client.get(URL).status_code == 404


def test_missing_experiment_404(client, setup, monkeypatch):
    monkeypatch.setattr("src.services.runs.get_active_experiment", lambda name: None)
    r = client.get(URL)
    assert r.status_code == 404
    assert r.json()["detail"] == "There is no experiment with name X"


def test_too_many_keys_422(client, setup):
    r = client.get(URL, params={"keys": [f"k{i}" for i in range(101)]})
    assert r.status_code == 422


def test_nonexistent_run_404(client, setup):
    setup.get_run_error = MlflowException("no run", error_code=RESOURCE_DOES_NOT_EXIST)
    assert client.get(URL).status_code == 404


def test_mlflow_not_configured_500(client):
    r = client.get(URL)
    assert r.status_code == 500
    assert r.json()["detail"] == "MLflow tracking URI is not configured"


def test_other_mlflow_error_500(client, setup):
    setup.history_error = MlflowException("boom", error_code=INTERNAL_ERROR)
    r = client.get(URL, params={"keys": ["loss"]})
    assert r.status_code == 500
    assert r.json()["detail"].startswith("Error reading metric history from MLflow:")
