"""Helpers to relate kmlops trainings with MLflow experiments and runs."""
from typing import Any, Dict, Iterable, List, Optional

import mlflow
from mlflow.entities import Experiment as MlflowExperiment
from mlflow.entities import Run as MlflowRun
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.tracking import MlflowClient

# Tag set by the training pod on the run it registers. The pod script (trainer_script.py) cannot
# import this module, so it hardcodes the same name: keep both in sync.
TRAINING_ID_TAG = "kmlops.training_id"

ACTIVE_RUN_STATUSES = ("RUNNING", "SCHEDULED")


def get_active_experiment(experiment_name: str) -> Optional[MlflowExperiment]:
    """MLflow experiment by name, ignoring soft-deleted ones."""
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None or exp.lifecycle_stage != "active":
        return None
    return exp


def find_training_runs(experiment_id: str, training: Dict[str, Any]) -> List[MlflowRun]:
    """Runs created by one kmlops training.

    Runs are matched by the `kmlops.training_id` tag. Runs without that tag (created before the tag
    existed, or still being logged) are matched if they started after the training was created.
    At most one training per experiment can be active, so this cannot pick runs of another training.
    """
    client = MlflowClient()
    tagged = client.search_runs(
        [experiment_id],
        filter_string=f"tags.`{TRAINING_ID_TAG}` = '{training['training_id']}'",
    )
    if tagged:
        return list(tagged)

    created_at = int(training["created_at"])
    candidates = client.search_runs(
        [experiment_id],
        filter_string=f"attributes.start_time >= {created_at}",
    )
    return [run for run in candidates if TRAINING_ID_TAG not in run.data.tags]


def find_finished_training_run(training: Dict[str, Any]) -> Optional[MlflowRun]:
    exp = get_active_experiment(training["experiment_name"])
    if exp is None:
        return None
    for run in find_training_runs(exp.experiment_id, training):
        if run.info.status == "FINISHED":
            return run
    return None


def has_finished_runs(experiment_id: str, exclude_run_ids: Optional[List[str]] = None) -> bool:
    """True if the experiment has at least one FINISHED (non-deleted) run, i.e. a model version."""
    exclude = set(exclude_run_ids or [])
    runs = MlflowClient().search_runs(
        [experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    return any(run.info.run_id not in exclude for run in runs)


def discard_run(run: MlflowRun) -> List[str]:
    """Marks a run as KILLED if it is still running, deletes the model versions registered from it
    and deletes the run. Returns the deleted model versions as 'name/version'."""
    client = MlflowClient()
    run_id = run.info.run_id
    if run.info.status in ACTIVE_RUN_STATUSES:
        client.set_terminated(run_id, "KILLED")

    deleted_versions = _delete_run_model_versions(client, run_id)
    delete_models_without_versions({v.split("/", 1)[0] for v in deleted_versions}, client)

    client.delete_run(run_id)
    return deleted_versions


def _delete_run_model_versions(client: MlflowClient, run_id: str) -> List[str]:
    """Deletes the model versions registered from a run. Returns them as 'name/version'."""
    deleted: List[str] = []
    for version in client.search_model_versions(f"run_id='{run_id}'"):
        client.delete_model_version(version.name, version.version)
        deleted.append(f"{version.name}/{version.version}")
    return deleted


def delete_models_without_versions(model_names: Iterable[str], client: Optional[MlflowClient] = None) -> List[str]:
    """Deletes the registered models of `model_names` that have no version left. A model that still
    has versions (e.g. registered from another experiment) is kept. Returns the deleted names."""
    client = client or MlflowClient()
    deleted: List[str] = []
    for name in sorted(set(model_names)):
        try:
            model = client.get_registered_model(name)
        except MlflowException as e:
            if e.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
                continue
            raise
        if not model.latest_versions:
            client.delete_registered_model(name)
            deleted.append(name)
    return deleted


def delete_experiment_model_versions(experiment_id: str) -> Dict[str, List[str]]:
    """Deletes the model versions registered from any run of an experiment (deleted runs included),
    then the registered models left without versions.

    Returns {"model_versions": ['name/version', ...], "registered_models": ['name', ...]}.
    """
    client = MlflowClient()
    versions: List[str] = []
    page_token = None
    while True:
        runs = client.search_runs([experiment_id], run_view_type=ViewType.ALL, page_token=page_token)
        for run in runs:
            versions += _delete_run_model_versions(client, run.info.run_id)
        page_token = runs.token
        if not page_token:
            break
    models = delete_models_without_versions({v.split("/", 1)[0] for v in versions}, client)
    return {"model_versions": versions, "registered_models": models}


def prepare_experiment_for_training(experiment_name: str) -> None:
    """MLflow only soft-deletes experiments, and a training cannot log into a deleted experiment.
    If the name belongs to a deleted experiment, restore it and delete its old runs, so the new
    training starts clean (the user had already deleted that model)."""
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None or exp.lifecycle_stage != "deleted":
        return
    client.restore_experiment(exp.experiment_id)
    for run in client.search_runs([exp.experiment_id], run_view_type=ViewType.ACTIVE_ONLY):
        client.delete_run(run.info.run_id)
    print(f"MLflow: restored deleted experiment '{experiment_name}' and cleared its old runs")
