import json
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile
from kubernetes.client.rest import ApiException

from src.config import get_settings
from src.db.store import ActiveModelTrainingExists, ActiveTrainingExists
from src.integrations.minio_storage import MINIO_ERRORS
from src.services import experiments, runs, trainings
from src.services.feature_schema import FeatureCheckUnavailable, IncompatibleFeatures, InvalidFeatures
from src.services.trainings import NoActiveTraining

router = APIRouter()

MAX_METRIC_KEYS = 100


def mlflow_enabled() -> bool:
    return get_settings().mlflow_enabled


@router.get("/")
def home():
    """Root endpoint to verify the API is running."""
    return {"message": "Welcome to the kmlops API. Go to /docs to view the interface."}

@router.get("/api/mlops/experiments")
def get_experiments(
    page: int = Query(1, ge=1, description="Page number, starting at 1"),
    pageSize: int = Query(20, ge=1, le=1000, description="Items per page"),
    search: str = '',
):
    """
    Returns one entry per experiment (model), merging local trainings and MLflow history.
    Paginated with 1-based `page`; an out-of-range page returns empty `results`.
    `status` is one of: queued, launching_job, training, completed, failed
    (`status_reason` explains failures).
    """

    return experiments.get_all_experiments(mlflow_enabled(), page, pageSize, search)

@router.get("/api/mlops/experiments/{experiment_name}")
async def get_experiment(experiment_name: str):
    try:
        return await experiments.get_experiment(experiment_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"There is no experiment with name {str(experiment_name)}")

@router.get("/api/mlops/experiments/{experiment_name}/logs")
async def get_training_logs(experiment_name: str, tail_lines: int = Query(500, ge=1, le=5000)):
    """Returns the last `tail_lines` log lines of the training pod of an experiment."""
    try:
        return await trainings.get_training_logs(experiment_name, tail_lines)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"There is no experiment with name {str(experiment_name)}")
    except ApiException as e:
        raise HTTPException(status_code=502, detail=f"Kubernetes error while reading training logs: {e.status} {e.reason}")

@router.post("/api/mlops/experiments")
async def create_experiment(
    background_tasks: BackgroundTasks,
    csv_file: UploadFile = File(...),
    model_type: str = Form(...),
    config_json: str = Form(...),
    dependencies: str = Form("[]"),
    experiment_name: str = Form("kmlops-default-experiment"),
    model_name: Optional[str] = Form(None, description="Name of the model in the MLflow Model Registry. Defaults to experiment_name."),
):
    """
    Receives the data file, model type, configuration, and dependencies to create a new experiment.
    Uploads the data to MinIO and triggers a Kubernetes Job in the background.
    The trained model is registered in MLflow as a new version of `model_name` (default: the
    experiment name). Several experiments can feed the same model.
    Returns 409 if the experiment, or the model in any experiment, already has a training in
    progress, or if the input/output features (name and type) differ from those of the existing
    versions of the model. Returns 422 if config_json does not define valid input_features and
    output_features. Returns 502 if the existing versions cannot be read from MLflow.
    """
    model_name = (model_name or "").strip() or experiment_name

    # 1. Parse and validate JSON inputs
    try:
        config_dict = json.loads(config_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="config_json is not a valid JSON string")

    try:
        dependencies_list = json.loads(dependencies)
    except json.JSONDecodeError:
        # Fallback for comma-separated strings
        if dependencies.strip():
            dependencies_list = [dep.strip() for dep in dependencies.split(",")]
        else:
            dependencies_list = []

    try:
        experiment = await experiments.create_experiment(background_tasks, csv_file, model_type, experiment_name, model_name, config_dict, dependencies_list)
    except ActiveModelTrainingExists:
        raise HTTPException(
            status_code=409,
            detail=f"Model '{model_name}' already has a training in progress in another experiment. "
            "Cancel it or wait until it finishes.",
        )
    except ActiveTrainingExists:
        raise HTTPException(
            status_code=409,
            detail=f"Experiment '{experiment_name}' already has a training in progress. Cancel it or wait until it finishes.",
        )
    except InvalidFeatures as e:
        raise HTTPException(status_code=422, detail=str(e))
    except IncompatibleFeatures as e:
        raise HTTPException(status_code=409, detail=str(e))
    except FeatureCheckUnavailable as e:
        raise HTTPException(status_code=502, detail=str(e))
    experiment_id = experiment["internal_id"]
    return {
        "message": f"Experiment {experiment_id} successfully created",
        "experiment_id": experiment_id,
        "details": experiment
    }


@router.delete("/api/mlops/experiments/{experiment_name}")
def delete_experiment(experiment_name: str):
    """Deletes an experiment by its NAME from the local DB, MLflow and MinIO (training CSVs of all its trainings).

    The model versions registered from its runs are deleted from the MLflow Model Registry, and so
    are the registered models left without versions (a model with versions from other experiments
    keeps those). If a training is in progress, its Kubernetes Job is stopped first. If that fails,
    or if the MinIO data or the model versions cannot be deleted, the rows and the MLflow experiment
    are kept and 502 is returned so the request can be retried.
    """
    try:
        deleted = experiments.delete_experiment(experiment_name, mlflowb=mlflow_enabled())
    except ApiException as e:
        raise HTTPException(status_code=502, detail=f"Kubernetes error while deleting the training job: {e.status} {e.reason}")
    except MINIO_ERRORS as e:
        raise HTTPException(status_code=502, detail=f"MinIO error while deleting training data: {e}")
    except experiments.ModelRegistryError as e:
        raise HTTPException(status_code=502, detail=f"MLflow error while deleting the registered model versions: {e}")

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_name}' not found in local DB or MLflow")

    return {"message": f"Experiment '{experiment_name}' successfully deleted"}

@router.post("/api/mlops/experiments/{experiment_name}/cancel")
def cancel_training(experiment_name: str):
    """
    Stops the training in progress (queued, launching_job or training) of an experiment and removes it:
    deletes the Kubernetes Job and its pods, kills and deletes its MLflow run and deletes its dataset
    from MinIO. If the model has no previous versions the whole experiment is deleted; otherwise only
    this training is removed and the existing versions are kept.
    If a step fails, `cleanup_complete` is false: the training stays as 'cancelled' and the cleanup is
    retried automatically (calling this endpoint again also retries it).
    """
    try:
        return trainings.cancel_training(experiment_name)
    except NoActiveTraining:
        raise HTTPException(status_code=404, detail=f"No training in progress for experiment '{experiment_name}'")

@router.get("/api/mlops/experiments/{experiment_name}/runs")
def get_runs(
    experiment_name: str,
    page: int = Query(1, ge=1, description="Page number, starting at 1"),
    pageSize: int = Query(20, ge=1, le=1000, description="Items per page"),
    search: str = '',
):
    """
    Returns the list of runs for a given experiment name from MLflow.
    Includes metrics, parameters, and tags for each run.
    Paginated with 1-based `page`; an out-of-range page returns empty `results`.
    """
    if not mlflow_enabled():
        raise HTTPException(status_code=500, detail="MLflow tracking URI is not configured")

    try:
        result = runs.get_runs(experiment_name, page, pageSize, search)
        return result
    except KeyError:
        raise HTTPException(status_code=404, detail=f"There is no experiment with name {str(experiment_name)}")
    except Exception as e:
        print(f"Error searching runs in MLflow for '{experiment_name}': {str(e)}")
        # If it's a known error from mlflow or pandas, we catch it
        raise HTTPException(status_code=500, detail=f"Error searching runs in MLflow: {str(e)}")

@router.get("/api/mlops/experiments/{experiment_name}/runs/{run_id}/metrics/history")
def get_metric_history(experiment_name: str, run_id: str, keys: Optional[List[str]] = Query(None)):
    """
    Returns the step-by-step history of the metrics of a run from MLflow.
    `keys` is optional and repeatable (`?keys=a&keys=b`); if omitted, all the metrics of the run are returned.
    """
    if keys is not None and len(set(keys)) > MAX_METRIC_KEYS:
        raise HTTPException(status_code=422, detail=f"At most {MAX_METRIC_KEYS} keys can be requested")

    if not mlflow_enabled():
        raise HTTPException(status_code=500, detail="MLflow tracking URI is not configured")

    try:
        return runs.get_metric_history(experiment_name, run_id, keys)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"There is no experiment with name {str(experiment_name)}")
    except runs.RunNotFound:
        raise HTTPException(status_code=404, detail=f"There is no run {run_id} in experiment {experiment_name}")
    except Exception as e:
        print(f"Error reading metric history of run '{run_id}' in '{experiment_name}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading metric history from MLflow: {str(e)}")
