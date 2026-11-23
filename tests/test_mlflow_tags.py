from types import SimpleNamespace

import pytest

from src.integrations import mlflow_tracking as tracking
from src.integrations.mlflow_tracking import TRAINING_ID_TAG, find_training_runs

TRAINING = {"training_id": "abc123", "created_at": 1000}


def make_run(run_id, tags=None):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id),
        data=SimpleNamespace(tags=tags or {}),
    )


class FakeClient:
    """search_runs understands only the two filters used by find_training_runs."""

    def __init__(self, runs):
        self.runs = runs

    def search_runs(self, experiment_ids, filter_string="", **kwargs):
        if filter_string.startswith("tags."):
            tag = filter_string.split("`")[1]
            value = filter_string.split("'")[1]
            return [r for r in self.runs if r.data.tags.get(tag) == value]
        return list(self.runs)  # attributes.start_time >= ...: every run started after creation


@pytest.fixture
def client(monkeypatch):
    def install(runs):
        fake = FakeClient(runs)
        monkeypatch.setattr(tracking, "MlflowClient", lambda: fake)
        return fake

    return install


def test_tag_name():
    assert TRAINING_ID_TAG == "kmlops.training_id"


def test_finds_runs_by_training_id_tag(client):
    client([
        make_run("r1", {TRAINING_ID_TAG: "abc123"}),
        make_run("other", {TRAINING_ID_TAG: "zzz"}),
        make_run("untagged"),
    ])
    assert [r.info.run_id for r in find_training_runs("1", TRAINING)] == ["r1"]


def test_falls_back_to_untagged_runs_only(client):
    client([make_run("other", {TRAINING_ID_TAG: "zzz"}), make_run("untagged")])
    assert [r.info.run_id for r in find_training_runs("1", TRAINING)] == ["untagged"]
