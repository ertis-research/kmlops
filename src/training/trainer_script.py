"""Training script run INSIDE the training pod (not imported by kmlops).

kmlops writes this file into the Job container and runs it with python. Every per-training
value arrives through environment variables, so names or configs with quotes cannot break
(or inject into) the code."""
import copy
import dataclasses
import json
import math
import os
import sys
import time

DEFAULT_SPLIT = {'type': 'random', 'probabilities': [0.7, 0.2, 0.1]}
SPLIT_PREFIX = {'training': 'train', 'validation': 'validation', 'test': 'test'}
BATCH_SIZE = 1000


def fail(phase, err):
    # kmlops reads the last 'ERROR' line of these logs as the failure reason.
    print(f'ERROR {phase}: {err}', flush=True)
    sys.exit(1)


def _clean(text):
    # Warnings must never contain 'ERROR': kmlops would take them as the failure reason.
    return str(text).replace('ERROR', 'error')


class StatsWrapper:
    def __init__(self, stats):
        self.stats = stats

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return self.stats.get(name)


CONTRIB_SPLITS = ('training', 'validation', 'test')


def stats_for_contrib(train_stats):
    """Returns a COPY of the training stats with the per-split metrics emptied.

    Ludwig's log_training_run would log its own validation.<feature>.<metric> / test.<...> keys,
    duplicating our <split>_metrics.* ones; with empty splits it only registers the model and logs
    model.*. The original is never mutated. On any failure the original is returned."""
    try:
        if isinstance(train_stats, StatsWrapper):
            return StatsWrapper(stats_for_contrib(train_stats.stats))
        if isinstance(train_stats, dict):
            copied = dict(train_stats)
            for split in CONTRIB_SPLITS:
                copied[split] = {}
            return copied
        if dataclasses.is_dataclass(train_stats) and not isinstance(train_stats, type):
            try:
                return dataclasses.replace(train_stats, **{s: {} for s in CONTRIB_SPLITS})
            except Exception:
                pass
        copied = copy.copy(train_stats)
        for split in CONTRIB_SPLITS:
            if hasattr(copied, split):
                setattr(copied, split, {})
        return copied
    except Exception as e:
        print(f'Warning: could not prepare training stats for Ludwig MLflow logging: {_clean(e)}', flush=True)
        return train_stats


def ensure_split(config):
    """Makes the dataset split explicit (Ludwig only reports validation/test metrics if it exists)."""
    preprocessing = config.get('preprocessing') or {}
    if not preprocessing.get('split'):
        preprocessing = dict(preprocessing)
        preprocessing['split'] = {'type': 'random', 'probabilities': list(DEFAULT_SPLIT['probabilities'])}
        config['preprocessing'] = preprocessing
    return config


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_entry(entry, index):
    """Returns (value, step) from a number, an object with .value/.epoch or an (epoch, step, value) tuple."""
    if hasattr(entry, 'value'):
        epoch = getattr(entry, 'epoch', None)
        return entry.value, (epoch if epoch is not None else index)
    if isinstance(entry, (tuple, list)) and len(entry) == 3:
        return entry[2], entry[0]
    return entry, index


def extract_train_stats(results):
    """Returns the training stats from what model.train() returned.

    Ludwig >= 0.10 returns a TrainingResults dataclass (not a tuple); older versions return a tuple.
    Never index or unpack TrainingResults: that emits a DeprecationWarning."""
    if hasattr(results, 'train_stats'):
        return results.train_stats
    if isinstance(results, tuple):
        return results[0]
    return results


def flatten_train_stats(train_stats):
    """Returns (metrics, warnings): metrics is a list of (key, value, step)."""
    metrics, warnings = [], []
    if train_stats is None:
        return metrics, warnings
    for split, prefix in SPLIT_PREFIX.items():
        if isinstance(train_stats, dict):
            data = train_stats.get(split)
        else:
            data = getattr(train_stats, split, None)
        if not isinstance(data, dict):
            continue
        for feature, feature_metrics in data.items():
            if not isinstance(feature_metrics, dict):
                continue
            for metric, entries in feature_metrics.items():
                key = f'{prefix}_metrics.{feature}.{metric}'
                if hasattr(entries, 'value') or not isinstance(entries, (list, tuple)):
                    entries, scalar = [entries], True
                else:
                    scalar = False
                for index, entry in enumerate(entries):
                    value, step = _parse_entry(entry, index)
                    if not _is_number(value) or not math.isfinite(value):
                        warnings.append(f'Warning: skipping metric {_clean(key)} (step {_clean(step)}): '
                                        f'invalid value {_clean(repr(value))}')
                        continue
                    try:
                        step = 0 if scalar and not _is_number(step) else int(step)
                    except (TypeError, ValueError):
                        step = index
                    metrics.append((key, float(value), step))
    return metrics, warnings


def log_split_metrics(client, run_id, train_stats):
    """Logs the per-split metric history to the MLflow run. Never raises."""
    try:
        from mlflow.entities import Metric

        metrics, warnings = flatten_train_stats(train_stats)
        for w in warnings:
            print(w, flush=True)
        if not metrics:
            print('Warning: no training metrics found in the Ludwig training statistics '
                  f'(got {_clean(type(train_stats).__name__)}).', flush=True)
        timestamp = int(time.time() * 1000)
        batch = [Metric(key, value, timestamp, step) for key, value, step in metrics]
        for i in range(0, len(batch), BATCH_SIZE):
            client.log_batch(run_id, metrics=batch[i:i + BATCH_SIZE])
        print(f'Logged {len(batch)} training metrics to MLflow.', flush=True)
    except Exception as e:
        print(f'Warning: could not log training metrics to MLflow: {_clean(e)}', flush=True)


def main():
    from minio import Minio
    from ludwig.api import LudwigModel
    import mlflow
    from mlflow.tracking import MlflowClient

    training_id = os.environ['KMLOPS_TRAINING_ID']
    experiment_name = os.environ['KMLOPS_EXPERIMENT_NAME']
    model_name = os.environ['KMLOPS_MODEL_NAME']
    bucket_name = os.environ['KMLOPS_BUCKET_NAME']
    minio_path = os.environ['KMLOPS_MINIO_PATH']
    started_at_ms = int(time.time() * 1000)


    print('Starting Advanced MLOps Job in Kubernetes...', flush=True)

    # 1. Establish connections and download the dataset
    try:
        minio_client = Minio(
            os.environ['MINIO_URL'],
            access_key=os.environ['MINIO_ACCESS_KEY'],
            secret_key=os.environ['MINIO_SECRET_KEY'],
            secure=os.environ.get('MINIO_SECURE', 'true').lower() == 'true',
        )
        mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
        mlflow.set_experiment(experiment_name)

        print('Downloading dataset from MinIO...', flush=True)
        response = minio_client.get_object(bucket_name, minio_path)
        with open('dataset.csv', 'wb') as f:
            f.write(response.read())
    except Exception as e:
        fail('IN SETUP PHASE', e)

    # 2. Train
    try:
        print('Parsing Ludwig Configuration...', flush=True)
        ludwig_config = json.loads(os.environ['KMLOPS_LUDWIG_CONFIG'])
        ludwig_config['model_name'] = model_name
        ensure_split(ludwig_config)

        print('Initializing Ludwig Model...', flush=True)
        model = LudwigModel(config=ludwig_config)

        print('Starting Training...', flush=True)
        results = model.train(dataset='dataset.csv')
        train_stats = extract_train_stats(results)
    except Exception as e:
        fail('DURING TRAINING', e)

    # 3. Log and register in MLflow
    print('Training finished. Logging to MLflow...', flush=True)
    run_id = None
    try:
        from ludwig.contribs.mlflow.mlflow3 import log_training_run

        stats_to_log = StatsWrapper(train_stats) if isinstance(train_stats, dict) else train_stats
        # Emptied copy: Ludwig only registers the model and logs model.*; split metrics go through
        # log_split_metrics below with the real stats.
        log_training_run(model=model, train_stats=stats_for_contrib(stats_to_log), config=ludwig_config)
        last_run = mlflow.last_active_run()
        run_id = last_run.info.run_id if last_run else None
    except Exception as mlflow_err:
        print('Warning: MLflow contrib logging failed, trying direct model logging:', str(mlflow_err), flush=True)
        try:
            from ludwig.contribs.mlflow.model import log_model
            with mlflow.start_run() as run:
                run_id = run.info.run_id
                log_model(model, artifact_path='model', registered_model_name=model_name)
        except Exception as e:
            fail('DURING MLFLOW LOGGING', e)

    if run_id is None:
        # Fallback: the newest run of this experiment created by this pod.
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string=f'attributes.start_time >= {started_at_ms}',
            order_by=['attributes.start_time DESC'],
            max_results=1,
            output_format='list',
        )
        run_id = runs[0].info.run_id if runs else None

    if run_id is None:
        fail('DURING MLFLOW LOGGING', 'the model was trained but no MLflow run was created')

    log_split_metrics(MlflowClient(), run_id, train_stats)

    try:
        # Must match TRAINING_ID_TAG in src/integrations/mlflow_tracking.py (this script cannot import src).
        MlflowClient().set_tag(run_id, 'kmlops.training_id', training_id)
    except Exception as e:
        print('Warning: could not tag the MLflow run:', str(e), flush=True)

    print('SUCCESS: Model trained and registered in MLflow (run ' + run_id + ').', flush=True)


if __name__ == '__main__':
    main()
