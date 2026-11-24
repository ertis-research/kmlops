from types import SimpleNamespace

import pandas as pd

import src.api.experiments as api_module
import src.services.runs as runs_module


class NoLatestStore:
    def latest(self, experiment_name):
        return None


def _mixed_runs_df():
    """Like mlflow.search_runs for two runs with different output features: each run gets
    NaN/None in the metric, param and tag columns of the other one."""
    return pd.DataFrame(
        {
            "run_id": ["run-vibration", "run-fault"],
            "status": ["FINISHED", "FINISHED"],
            "experiment_id": ["1", "1"],
            "artifact_uri": ["s3://b/1/a", "s3://b/1/b"],
            "start_time": [pd.Timestamp("2026-11-24T10:00:00Z"), pd.Timestamp("2026-11-24T11:00:00Z")],
            "end_time": [pd.Timestamp("2026-11-24T10:05:00Z"), pd.NaT],
            "metrics.test_metrics.vibration.r2": [0.9, float("nan")],
            "metrics.test_metrics.fault_type.accuracy": [float("nan"), 0.8],
            "metrics.inf_metric": [float("inf"), 1.0],
            "params.output": ["vibration", None],
            "tags.only_fault": [None, "yes"],
        }
    )


def test_runs_with_different_metrics_do_not_break_json(client, monkeypatch):
    monkeypatch.setattr(api_module, "mlflow_enabled", lambda: True)
    monkeypatch.setattr(runs_module, "get_active_experiment", lambda name: SimpleNamespace(experiment_id="1"))
    monkeypatch.setattr(runs_module, "get_store", lambda: NoLatestStore())
    monkeypatch.setattr(runs_module, "mlflow", SimpleNamespace(search_runs=lambda **kwargs: _mixed_runs_df()))

    response = client.get("/api/mlops/experiments/my-exp/runs")

    assert response.status_code == 200
    vibration, fault = response.json()["results"]
    assert vibration["metrics"] == {"test_metrics.vibration.r2": 0.9}
    assert vibration["params"] == {"output": "vibration"}
    assert vibration["tags"] == {}
    assert fault["metrics"] == {"test_metrics.fault_type.accuracy": 0.8, "inf_metric": 1.0}
    assert fault["params"] == {}
    assert fault["tags"] == {"only_fault": "yes"}
    assert fault["end_time"] is None
