# Changelog

All notable changes to this project are documented here, following [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Per-split training metrics logged to MLflow (`train_metrics.*`, `validation_metrics.*` and `test_metrics.*`), with per-epoch history.
- `GET /api/mlops/experiments/{experiment_name}/runs/{run_id}/metrics/history` endpoint.
- `MINIO_CREDENTIALS_SECRET`, `MINIO_CREDENTIALS_ACCESS_KEY_KEY` and `MINIO_CREDENTIALS_SECRET_KEY_KEY` to make training Jobs read the MinIO credentials from a Kubernetes Secret.
- `TRAINING_MLFLOW_VERSION`, `TRAINING_MINIO_VERSION` and `TRAINING_PIP_EXTRA_PACKAGES` to control what the training pod installs.

### Changed
- `model_name` of `POST /api/mlops/experiments` defaults to `experiment_name` instead of `ludwig_model`, so unrelated experiments no longer become versions of the same model by default.
- **BREAKING:** `POST /api/mlops/experiments` requires the same input and output features (names and types) as the existing versions of the model (`model_name`). These are the latest completed or in-progress training of the model in the kmlops database, in any experiment, and the latest version in the MLflow Model Registry. If they differ, it returns 409 listing the differences. It returns 422 if `config_json` has no valid `input_features` / `output_features`, and 502 if MLflow cannot be read.
- `DELETE /api/mlops/experiments/{name}` also deletes the model versions registered by the runs of the experiment and the registered models left without versions (a model with versions from other experiments keeps those). It returns 502 if the Model Registry cannot be cleaned, keeping the rows and the MLflow experiment so it can be retried. It also cleans the versions left by experiments deleted before. Cancelling a training also deletes its registered model if it is left without versions.
- **BREAKING:** at most one training in progress per model (`model_name`), in any experiment, enforced by a unique index. A second one returns 409. If existing data violates the index at startup, kmlops starts without it, logs a warning and retries on the next startup.
- The training pod installs pinned versions (`minio==7.2.20`, `mlflow==3.18.0` by default) instead of the latest ones.
- kmlops logs a warning at startup when `MINIO_CREDENTIALS_SECRET` is not set.
- **BREAKING:** list pagination is now 1-based. `page` defaults to 1 and `page=0` (or negative) gives 422. Out-of-range pages return empty `results` instead of the last page. Affects `GET /api/mlops/experiments` and `GET /api/mlops/experiments/{name}/runs`. Clients that sent `page - 1` must now send `page`.
- Default random 0.7/0.2/0.1 split when the Ludwig config does not define one.
- Ludwig's own `validation.*` / `test.*` metrics are no longer logged.
- `data/` renamed to `examples/`, now with four example datasets.

### Fixed
- `GET /api/mlops/experiments/{name}/runs` returned 500 (`Out of range float values are not JSON compliant: nan`) when the runs of an experiment had different metrics, params or tags (e.g. different output features). Keys a run does not have are now omitted, and so are non-finite metric values.

### Security
- When `MINIO_CREDENTIALS_SECRET` is set, the MinIO credentials no longer appear in the spec of the training Jobs (they are read from the Secret with `secretKeyRef`). The manifest deployment (`k8s/deployment.yaml`) sets it.
