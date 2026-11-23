# Regression: wind turbine power output

1000 rows of synthetic wind turbine data with a numeric power output (kW).

## Columns

- `wind_speed`, `air_density`, `blade_pitch`, `temperature` (number)
- `power_output` (number, output)

## Generation

`python examples/generate_examples.py` (stdlib only, seed 42). Cubic in wind speed scaled by air density, saturating at rated power, reduced by blade pitch and high temperature, plus Gaussian noise (sd 60).

## Ludwig config

`ludwig_config.json` (split: default 0.7/0.2/0.1 applied by dflow):

```json
{
  "input_features": [
    {"name": "wind_speed", "type": "number"},
    {"name": "air_density", "type": "number"},
    {"name": "blade_pitch", "type": "number"},
    {"name": "temperature", "type": "number"}
  ],
  "output_features": [{"name": "power_output", "type": "number"}],
  "trainer": {"epochs": 20, "early_stop": 20}
}
```

## Launch against dflow

```bash
curl -X POST "http://localhost:8000/api/mlops/experiments" \
  -F "csv_file=@examples/regression_power_output/data.csv" \
  -F "model_type=ludwig" \
  -F "experiment_name=examples-regression-power-output" \
  -F "model_name=examples-regression-power-output" \
  -F "config_json=<examples/regression_power_output/ludwig_config.json"
```

## Expected MLflow metrics

Run `GET /api/mlops/experiments/{name}/runs` and `.../runs/{run_id}/metrics/history`. Keys:

- `model.*` (3 model-level keys)
- `train_metrics.power_output.*`, `validation_metrics.power_output.*`, `test_metrics.power_output.*`
- `train_metrics.combined.loss`, `validation_metrics.combined.loss`, `test_metrics.combined.loss`

Metrics: regression (loss, mean squared/absolute error, r2, ...). No bare `training.*`, `validation.*` or `test.*` keys.
