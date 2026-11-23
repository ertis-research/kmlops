# Changelog

All notable changes to this project are documented here, following [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Per-split training metrics logged to MLflow (`train_metrics.*`, `validation_metrics.*` and `test_metrics.*`), with per-epoch history.
- `GET /api/mlops/experiments/{experiment_name}/runs/{run_id}/metrics/history` endpoint.

### Changed
- **BREAKING:** list pagination is now 1-based. `page` defaults to 1 and `page=0` (or negative) gives 422. Out-of-range pages return empty `results` instead of the last page. Affects `GET /api/mlops/experiments` and `GET /api/mlops/experiments/{name}/runs`. Clients that sent `page - 1` must now send `page`.
- Default random 0.7/0.2/0.1 split when the Ludwig config does not define one.
- Ludwig's own `validation.*` / `test.*` metrics are no longer logged.
- `data/` renamed to `examples/`, now with four example datasets.
