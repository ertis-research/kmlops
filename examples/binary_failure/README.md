# Binary failure

1000 rows of synthetic machine readings with a binary failure label (about 25% positive).

## Columns

- `rpm`, `temperature`, `vibration`, `pressure` (number)
- `machine_type` (category: A, B, C)
- `failure` (binary, output)

## Generation

`python examples/generate_examples.py` (stdlib only, seed 42). `failure` is the top 25% of a noisy linear score of the inputs plus a machine_type offset, so it is learnable but not perfectly separable.

## Ludwig config

`ludwig_config.json` (split: default 0.7/0.2/0.1 applied by dflow):

```json
{
  "input_features": [
    {"name": "rpm", "type": "number"},
    {"name": "temperature", "type": "number"},
    {"name": "vibration", "type": "number"},
    {"name": "pressure", "type": "number"},
    {"name": "machine_type", "type": "category"}
  ],
  "output_features": [{"name": "failure", "type": "binary"}],
  "trainer": {"epochs": 10, "early_stop": 10}
}
```

## Launch against dflow

```bash
curl -X POST "http://localhost:8000/api/mlops/experiments" \
  -F "csv_file=@examples/binary_failure/data.csv" \
  -F "model_type=ludwig" \
  -F "experiment_name=examples-binary-failure" \
  -F "model_name=examples-binary-failure" \
  -F "config_json=<examples/binary_failure/ludwig_config.json"
```

## Expected MLflow metrics

Run `GET /api/mlops/experiments/{name}/runs` and `.../runs/{run_id}/metrics/history`. Keys:

- `model.*` (3 model-level keys)
- `train_metrics.failure.*`, `validation_metrics.failure.*`, `test_metrics.failure.*`
- `train_metrics.combined.loss`, `validation_metrics.combined.loss`, `test_metrics.combined.loss`

Metrics: binary classification (loss, accuracy, ROC AUC, ...). No bare `training.*`, `validation.*` or `test.*` keys.
