# Multiclass: fault type

1000 rows of synthetic vibration/current readings classified into none, bearing, misalignment or imbalance. Uses an explicit split to check that a user split is respected.

## Columns

- `vibration_x`, `vibration_y`, `temperature`, `current`, `rpm` (number)
- `fault_type` (category, output): none, bearing, misalignment, imbalance

## Generation

`python examples/generate_examples.py` (stdlib only, seed 42). Each class gets a noisy score from related inputs and the highest wins.

## Ludwig config

`ludwig_config.json` (split: explicit `preprocessing.split` random 0.8/0.1/0.1, which kmlops must not override):

```json
{
  "input_features": [
    {"name": "vibration_x", "type": "number"},
    {"name": "vibration_y", "type": "number"},
    {"name": "temperature", "type": "number"},
    {"name": "current", "type": "number"},
    {"name": "rpm", "type": "number"}
  ],
  "output_features": [{"name": "fault_type", "type": "category"}],
  "preprocessing": {"split": {"type": "random", "probabilities": [0.8, 0.1, 0.1]}},
  "trainer": {"epochs": 10, "early_stop": 10}
}
```

## Launch against kmlops

```bash
curl -X POST "http://localhost:8000/api/mlops/experiments" \
  -F "csv_file=@examples/multiclass_fault_type/data.csv" \
  -F "model_type=ludwig" \
  -F "experiment_name=examples-multiclass-fault-type" \
  -F "model_name=examples-multiclass-fault-type" \
  -F "config_json=<examples/multiclass_fault_type/ludwig_config.json"
```

## Expected MLflow metrics

Run `GET /api/mlops/experiments/{name}/runs` and `.../runs/{run_id}/metrics/history`. Keys:

- `model.*` (3 model-level keys)
- `train_metrics.fault_type.*`, `validation_metrics.fault_type.*`, `test_metrics.fault_type.*`
- `train_metrics.combined.loss`, `validation_metrics.combined.loss`, `test_metrics.combined.loss`

Metrics: multiclass classification (loss, accuracy, hits@k, ...). No bare `training.*`, `validation.*` or `test.*` keys.
