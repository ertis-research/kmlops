"""Tests for the pure helpers of the training pod script (no ludwig / minio needed)."""
from collections import namedtuple
from dataclasses import dataclass
from types import SimpleNamespace

from src.training import trainer_script as ts

TrainerMetric = namedtuple("TrainerMetric", ["epoch", "step", "value"])


def _stats():
    return {
        "training": {
            "combined": {"loss": [1.0, 0.8, 0.6]},
            "failure": {"loss": [1.1, 0.9, 0.7]},
        },
        "validation": {
            "failure": {
                "accuracy": [0.5, 0.6, 0.7],
                "roc_auc": [0.6, float("nan"), float("inf")],
                "loss": [0.9, "abc", 0.5],
            },
        },
        "test": {"failure": {"accuracy": 0.8}},
    }


def test_flatten_dict_stats():
    metrics, warnings = ts.flatten_train_stats(_stats())
    assert metrics == [
        ("train_metrics.combined.loss", 1.0, 0),
        ("train_metrics.combined.loss", 0.8, 1),
        ("train_metrics.combined.loss", 0.6, 2),
        ("train_metrics.failure.loss", 1.1, 0),
        ("train_metrics.failure.loss", 0.9, 1),
        ("train_metrics.failure.loss", 0.7, 2),
        ("validation_metrics.failure.accuracy", 0.5, 0),
        ("validation_metrics.failure.accuracy", 0.6, 1),
        ("validation_metrics.failure.accuracy", 0.7, 2),
        ("validation_metrics.failure.roc_auc", 0.6, 0),
        ("validation_metrics.failure.loss", 0.9, 0),
        ("validation_metrics.failure.loss", 0.5, 2),
        ("test_metrics.failure.accuracy", 0.8, 0),
    ]
    assert len(warnings) == 3
    assert all(w.startswith("Warning: skipping metric") and "ERROR" not in w for w in warnings)


def test_flatten_trainer_metric_uses_epoch():
    stats = {"validation": {"failure": {"accuracy": [TrainerMetric(1, 10, 0.5), TrainerMetric(2, 20, 0.7)]}}}
    metrics, warnings = ts.flatten_train_stats(stats)
    assert metrics == [
        ("validation_metrics.failure.accuracy", 0.5, 1),
        ("validation_metrics.failure.accuracy", 0.7, 2),
    ]
    assert warnings == []


def test_flatten_plain_tuple_entries():
    stats = {"training": {"failure": {"loss": [(3, 30, 0.4)]}}}
    assert ts.flatten_train_stats(stats)[0] == [("train_metrics.failure.loss", 0.4, 3)]


def test_flatten_attribute_style_and_wrapper():
    data = {"failure": {"accuracy": [TrainerMetric(1, 5, 0.9)]}}
    obj = SimpleNamespace(training=None, validation=data, test=None)
    expected = [("validation_metrics.failure.accuracy", 0.9, 1)]
    assert ts.flatten_train_stats(obj)[0] == expected
    assert ts.flatten_train_stats(ts.StatsWrapper({"validation": data}))[0] == expected


def test_flatten_empty_and_bool():
    assert ts.flatten_train_stats(None) == ([], [])
    assert ts.flatten_train_stats({"training": None, "validation": None}) == ([], [])
    metrics, warnings = ts.flatten_train_stats({"test": {"f": {"m": [True]}}})
    assert metrics == [] and len(warnings) == 1


def test_ensure_split_default_added():
    cfg = ts.ensure_split({"input_features": []})
    assert cfg["preprocessing"]["split"] == {"type": "random", "probabilities": [0.7, 0.2, 0.1]}


def test_ensure_split_respects_user_split():
    split = {"type": "fixed", "column": "s"}
    cfg = ts.ensure_split({"preprocessing": {"split": split}})
    assert cfg["preprocessing"]["split"] == split


def test_ensure_split_keeps_other_preprocessing_keys():
    cfg = ts.ensure_split({"preprocessing": {"sample_ratio": 0.5}})
    assert cfg["preprocessing"]["sample_ratio"] == 0.5
    assert cfg["preprocessing"]["split"]["type"] == "random"


class RecordingClient:
    def __init__(self):
        self.calls = []

    def log_batch(self, run_id, metrics=()):
        self.calls.append((run_id, list(metrics)))


def test_log_split_metrics_logs_history():
    client = RecordingClient()
    ts.log_split_metrics(client, "run-1", _stats())
    assert all(run_id == "run-1" for run_id, _ in client.calls)
    logged = [m for _, batch in client.calls for m in batch]
    assert len(logged) == 13
    got = {(m.key, m.step): m.value for m in logged}
    assert got[("validation_metrics.failure.accuracy", 2)] == 0.7
    assert got[("test_metrics.failure.accuracy", 0)] == 0.8
    assert all(len(batch) <= 1000 for _, batch in client.calls)


def test_log_split_metrics_chunks_batches():
    client = RecordingClient()
    stats = {"training": {"f": {"loss": [float(i) for i in range(2500)]}}}
    ts.log_split_metrics(client, "r", stats)
    assert [len(b) for _, b in client.calls] == [1000, 1000, 500]


def test_log_split_metrics_never_raises(capsys):
    class Broken:
        def log_batch(self, *a, **k):
            raise RuntimeError("ERROR boom")

    ts.log_split_metrics(Broken(), "run-1", _stats())
    out = capsys.readouterr().out
    assert "Warning: could not log training metrics to MLflow" in out
    assert "ERROR" not in out


@dataclass
class FakeTrainingStats:
    training: dict
    validation: dict
    test: dict


@dataclass
class FakeTrainingResults:
    train_stats: FakeTrainingStats
    preprocessed_data: object
    output_directory: str


def test_extract_train_stats_from_training_results():
    stats = FakeTrainingStats(
        training={"failure": {"loss": [1.0, 0.5]}},
        validation={"failure": {"accuracy": [0.6, 0.7]}},
        test={"failure": {"accuracy": [0.8]}},
    )
    results = FakeTrainingResults(stats, None, "out")
    assert ts.extract_train_stats(results) is stats
    metrics, warnings = ts.flatten_train_stats(ts.extract_train_stats(results))
    assert {m[0] for m in metrics} == {
        "train_metrics.failure.loss",
        "validation_metrics.failure.accuracy",
        "test_metrics.failure.accuracy",
    }
    assert len(metrics) == 5 and warnings == []


def test_extract_train_stats_tuple_and_dict():
    stats = _stats()
    assert ts.extract_train_stats((stats, "pre", "out")) is stats
    assert ts.extract_train_stats(stats) is stats


def test_log_split_metrics_warns_when_no_metrics(capsys):
    client = RecordingClient()
    ts.log_split_metrics(client, "run-1", FakeTrainingResults(None, None, "out"))
    out = capsys.readouterr().out
    assert "Warning: no training metrics found in the Ludwig training statistics (got FakeTrainingResults)." in out
    assert "ERROR" not in out
    assert client.calls == []


@dataclass
class FakeTrainingStatsFreq:
    training: dict
    validation: dict
    test: dict
    evaluation_frequency: object = None


def _full_dict():
    return {
        "training": {"failure": {"loss": [1.0, 0.5]}},
        "validation": {"failure": {"accuracy": [0.6, 0.7]}},
        "test": {"failure": {"accuracy": [0.8]}},
    }


def test_stats_for_contrib_dataclass():
    full = _full_dict()
    stats = FakeTrainingStatsFreq(full["training"], full["validation"], full["test"], evaluation_frequency="freq")
    out = ts.stats_for_contrib(stats)
    assert out is not stats
    assert out.training == {} and out.validation == {} and out.test == {}
    assert out.evaluation_frequency == "freq"
    assert stats.validation == full["validation"] and stats.training == full["training"]
    assert len(ts.flatten_train_stats(stats)[0]) == 5
    assert ts.flatten_train_stats(out)[0] == []


def test_stats_for_contrib_dict_and_wrapper():
    full = _full_dict()
    out = ts.stats_for_contrib(full)
    assert out == {"training": {}, "validation": {}, "test": {}}
    assert full == _full_dict()
    assert len(ts.flatten_train_stats(full)[0]) == 5

    wrapper = ts.StatsWrapper(full)
    wrapped = ts.stats_for_contrib(wrapper)
    assert isinstance(wrapped, ts.StatsWrapper) and wrapped is not wrapper
    assert wrapped.training == {} and wrapped.validation == {} and wrapped.test == {}
    assert wrapper.validation == _full_dict()["validation"]


def test_stats_for_contrib_generic_object():
    obj = SimpleNamespace(training={"f": {"m": [1.0]}}, validation={"f": {"m": [1.0]}}, test=None, other=1)
    out = ts.stats_for_contrib(obj)
    assert out.training == {} and out.validation == {} and out.test == {} and out.other == 1
    assert obj.training == {"f": {"m": [1.0]}}
