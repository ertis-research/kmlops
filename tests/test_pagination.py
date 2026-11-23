from types import SimpleNamespace

import pandas as pd
import pytest

import src.api.experiments as api_module
import src.services.experiments as experiments_module
import src.services.runs as runs_module
from src.services.pagination import paginate
from tests.conftest import make_training


# ---- paginate unit tests ------------------------------------------------------


def test_first_and_second_page():
    items = list(range(25))
    p1 = paginate(items, 1, 10)
    assert p1["results"] == list(range(10))
    assert p1["pagination"]["current_page"] == 1
    assert p1["pagination"]["has_next"] is True
    assert p1["pagination"]["has_prev"] is False
    p2 = paginate(items, 2, 10)
    assert p2["results"] == list(range(10, 20))
    assert p2["pagination"]["has_prev"] is True


def test_partial_last_page():
    p = paginate(list(range(25)), 3, 10)
    assert p["results"] == [20, 21, 22, 23, 24]
    assert p["pagination"]["total_pages"] == 3
    assert p["pagination"]["has_next"] is False
    assert p["pagination"]["has_prev"] is True


def test_exact_last_page():
    p = paginate(list(range(20)), 2, 10)
    assert p["results"] == list(range(10, 20))
    assert p["pagination"]["total_pages"] == 2
    assert p["pagination"]["has_next"] is False


def test_out_of_range_page_is_empty_not_clamped():
    p = paginate(list(range(5)), 4, 2)
    assert p["results"] == []
    assert p["pagination"]["current_page"] == 4
    assert p["pagination"]["total_pages"] == 3
    assert p["pagination"]["has_next"] is False
    assert p["pagination"]["has_prev"] is True


def test_empty_list():
    p = paginate([], 1, 10)
    assert p["results"] == []
    assert p["pagination"]["total_items"] == 0
    assert p["pagination"]["total_pages"] == 0
    assert p["pagination"]["has_next"] is False
    assert p["pagination"]["has_prev"] is False


def test_page_below_one_raises():
    with pytest.raises(ValueError):
        paginate([1], 0, 10)


# ---- GET /api/mlops/experiments ----------------------------------------------


class ListStore:
    def __init__(self, trainings):
        self._trainings = trainings

    def list_all(self):
        return list(self._trainings)


def _trainings(n):
    out = []
    for i in range(n):
        t = make_training(status="completed", training_id=f"id{i:03d}")
        t["experiment_name"] = f"exp-{i:03d}"
        out.append(t)
    return out


@pytest.fixture
def set_experiments(monkeypatch):
    def _set(n):
        monkeypatch.setattr(api_module, "mlflow_enabled", lambda: False)
        monkeypatch.setattr(experiments_module, "get_store", lambda: ListStore(_trainings(n)))
    return _set


def test_experiments_page_1(client, set_experiments):
    set_experiments(25)
    body = client.get("/api/mlops/experiments?page=1").json()
    assert len(body["results"]) == 20
    assert body["pagination"]["current_page"] == 1
    assert body["pagination"]["total_items"] == 25
    assert body["pagination"]["total_pages"] == 2
    assert body["pagination"]["has_next"] is True
    assert body["pagination"]["has_prev"] is False


def test_experiments_page_omitted_is_page_1(client, set_experiments):
    set_experiments(25)
    body = client.get("/api/mlops/experiments").json()
    assert len(body["results"]) == 20
    assert body["pagination"]["current_page"] == 1


def test_experiments_page_2(client, set_experiments):
    set_experiments(25)
    body = client.get("/api/mlops/experiments?page=2").json()
    assert len(body["results"]) == 5
    assert body["pagination"]["has_next"] is False
    assert body["pagination"]["has_prev"] is True


def test_experiments_out_of_range(client, set_experiments):
    set_experiments(25)
    body = client.get("/api/mlops/experiments?page=3").json()
    assert body["results"] == []
    assert body["pagination"]["current_page"] == 3
    assert body["pagination"]["has_next"] is False
    assert body["pagination"]["has_prev"] is True


@pytest.mark.parametrize("page", ["0", "-1"])
def test_experiments_invalid_page_is_422(client, set_experiments, page):
    set_experiments(25)
    assert client.get(f"/api/mlops/experiments?page={page}").status_code == 422


def test_experiments_empty_store(client, set_experiments):
    set_experiments(0)
    body = client.get("/api/mlops/experiments").json()
    assert body["results"] == []
    assert body["pagination"]["total_items"] == 0
    assert body["pagination"]["total_pages"] == 0
    assert body["pagination"]["has_next"] is False
    assert body["pagination"]["has_prev"] is False


# ---- GET /api/mlops/experiments/{name}/runs ----------------------------------


class NoLatestStore:
    def latest(self, experiment_name):
        return None


def _runs_df(n):
    return pd.DataFrame(
        {
            "run_id": [f"run{i:03d}" for i in range(n)],
            "status": ["FINISHED"] * n,
            "experiment_id": ["1"] * n,
            "artifact_uri": [f"s3://b/1/run{i:03d}/artifacts" for i in range(n)],
            "start_time": [1_700_000_000_000 + i for i in range(n)],
            "end_time": [1_700_000_001_000 + i for i in range(n)],
            "metrics.x": [float(i) for i in range(n)],
            "tags.mlflow.runName": [f"name-{i}" for i in range(n)],
        }
    )


@pytest.fixture
def set_runs(monkeypatch):
    def _set(n):
        monkeypatch.setattr(api_module, "mlflow_enabled", lambda: True)
        monkeypatch.setattr(runs_module, "get_active_experiment", lambda name: SimpleNamespace(experiment_id="1"))
        monkeypatch.setattr(runs_module, "get_store", lambda: NoLatestStore())
        monkeypatch.setattr(
            runs_module, "mlflow", SimpleNamespace(search_runs=lambda **kwargs: _runs_df(n))
        )
    return _set


URL = "/api/mlops/experiments/my-exp/runs"


def test_runs_page_1_and_default(client, set_runs):
    set_runs(25)
    for url in (f"{URL}?page=1", URL):
        body = client.get(url).json()
        assert len(body["results"]) == 20
        assert body["pagination"]["current_page"] == 1
        assert body["pagination"]["total_pages"] == 2
        assert body["pagination"]["has_prev"] is False


def test_runs_page_2(client, set_runs):
    set_runs(25)
    body = client.get(f"{URL}?page=2").json()
    assert len(body["results"]) == 5
    assert body["pagination"]["has_next"] is False
    assert body["pagination"]["has_prev"] is True


def test_runs_out_of_range(client, set_runs):
    set_runs(25)
    body = client.get(f"{URL}?page=3").json()
    assert body["results"] == []
    assert body["pagination"]["current_page"] == 3
    assert body["pagination"]["has_next"] is False


def test_runs_page_0_is_422(client, set_runs):
    set_runs(25)
    assert client.get(f"{URL}?page=0").status_code == 422


def test_runs_empty(client, set_runs):
    set_runs(0)
    body = client.get(URL).json()
    assert body["results"] == []
    assert body["pagination"]["total_items"] == 0
    assert body["pagination"]["total_pages"] == 0
    assert body["pagination"]["has_next"] is False
    assert body["pagination"]["has_prev"] is False
