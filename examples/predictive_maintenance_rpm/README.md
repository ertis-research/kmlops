# Predictive maintenance (rpm)

Small 60-row dataset (the main README POST example): engine rpm and whether failure is imminent. The first 8 rows are the original hand-written sample; the rest were appended (seed 7). A first attempt with only the 8 rows failed in dflow ("Training dataset has only 2 rows following preprocessing, need at least 3 to compute metrics"), so keep at least a few dozen rows.

## Columns

- `rpm` (number): rotational speed.
- `imminent_failure` (binary, output): True when rpm is high.

## Generation

Hand-written 8 rows plus 52 rows appended with a one-off seeded script (not produced by `generate_examples.py`).

## Ludwig config

`ludwig_config.json` (split: default 0.7/0.2/0.1 applied by dflow):

```json
{
  "input_features": [{"name": "rpm", "type": "number"}],
  "output_features": [{"name": "imminent_failure", "type": "binary"}],
  "trainer": {"epochs": 10, "early_stop": 10}
}
```

## Launch against dflow

```bash
curl -X POST "http://localhost:8000/api/mlops/experiments" \
  -F "csv_file=@examples/predictive_maintenance_rpm/data.csv" \
  -F "model_type=ludwig" \
  -F "experiment_name=examples-predictive-maintenance-rpm" \
  -F "model_name=examples-predictive-maintenance-rpm" \
  -F "config_json=<examples/predictive_maintenance_rpm/ludwig_config.json"
```

## Expected MLflow metrics

Run `GET /api/mlops/experiments/{name}/runs` and `.../runs/{run_id}/metrics/history`. Keys:

- `model.*` (3 model-level keys)
- `train_metrics.imminent_failure.*`, `validation_metrics.imminent_failure.*`, `test_metrics.imminent_failure.*`
- `train_metrics.combined.loss`, `validation_metrics.combined.loss`, `test_metrics.combined.loss`

Metrics: binary classification (loss, accuracy, ROC AUC, ...). No bare `training.*`, `validation.*` or `test.*` keys.
