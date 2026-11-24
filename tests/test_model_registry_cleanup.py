from types import SimpleNamespace

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

import src.integrations.mlflow_tracking as tracking


class Page(list):
    def __init__(self, items, token=None):
        super().__init__(items)
        self.token = token


class FakeRegistryClient:
    """Runs of experiments and model versions registered from them."""

    def __init__(self, runs_by_experiment, versions, page_size=1000):
        self.runs = runs_by_experiment  # experiment_id -> [run_id]
        self.versions = versions  # list of (model name, version, run_id)
        self.page_size = page_size
        self.deleted_models = []
        self.deleted_runs = []

    def search_runs(self, experiment_ids, run_view_type=None, page_token=None, **kwargs):
        ids = [r for e in experiment_ids for r in self.runs.get(e, [])]
        start = int(page_token or 0)
        end = start + self.page_size
        page = [SimpleNamespace(info=SimpleNamespace(run_id=r)) for r in ids[start:end]]
        return Page(page, token=str(end) if end < len(ids) else None)

    def search_model_versions(self, filter_string):
        run_id = filter_string.split("'")[1]
        return [SimpleNamespace(name=n, version=v) for n, v, r in self.versions if r == run_id]

    def delete_model_version(self, name, version):
        self.versions = [(n, v, r) for n, v, r in self.versions if (n, v) != (name, version)]

    def get_registered_model(self, name):
        if name in self.deleted_models:
            raise MlflowException("not found", error_code=RESOURCE_DOES_NOT_EXIST)
        return SimpleNamespace(latest_versions=[v for n, v, r in self.versions if n == name])

    def delete_registered_model(self, name):
        self.deleted_models.append(name)

    def set_terminated(self, run_id, status):
        pass

    def delete_run(self, run_id):
        self.deleted_runs.append(run_id)


@pytest.fixture
def registry(monkeypatch):
    def _set(**kwargs):
        client = FakeRegistryClient(**kwargs)
        monkeypatch.setattr(tracking, "MlflowClient", lambda: client)
        return client
    return _set


def test_versions_of_the_experiment_and_empty_models_are_deleted(registry):
    client = registry(
        runs_by_experiment={"1": ["r1", "r2", "r3"]},
        versions=[("model-a", "1", "r1"), ("model-a", "2", "r2"), ("model-b", "1", "r3")],
    )
    result = tracking.delete_experiment_model_versions("1")
    assert sorted(result["model_versions"]) == ["model-a/1", "model-a/2", "model-b/1"]
    assert result["registered_models"] == ["model-a", "model-b"]
    assert client.versions == []


def test_model_shared_with_another_experiment_keeps_its_other_versions(registry):
    client = registry(
        runs_by_experiment={"1": ["r1"], "2": ["r2"]},
        versions=[("shared", "1", "r1"), ("shared", "2", "r2")],
    )
    result = tracking.delete_experiment_model_versions("1")
    assert result == {"model_versions": ["shared/1"], "registered_models": []}
    assert client.versions == [("shared", "2", "r2")]
    assert client.deleted_models == []


def test_all_pages_of_runs_are_visited(registry):
    client = registry(
        runs_by_experiment={"1": ["r1", "r2", "r3"]},
        versions=[("m", "1", "r1"), ("m", "2", "r3")],
        page_size=2,
    )
    result = tracking.delete_experiment_model_versions("1")
    assert sorted(result["model_versions"]) == ["m/1", "m/2"]
    assert client.deleted_models == ["m"]


def test_experiment_without_versions_deletes_nothing(registry):
    client = registry(runs_by_experiment={"1": ["r1"]}, versions=[("other", "1", "r9")])
    assert tracking.delete_experiment_model_versions("1") == {"model_versions": [], "registered_models": []}
    assert client.deleted_models == []


def test_discard_run_deletes_the_model_left_empty(registry):
    client = registry(runs_by_experiment={}, versions=[("m", "1", "r1")])
    run = SimpleNamespace(info=SimpleNamespace(run_id="r1", status="FINISHED"))
    assert tracking.discard_run(run) == ["m/1"]
    assert client.deleted_models == ["m"]
    assert client.deleted_runs == ["r1"]
