# kMLOps (Kubernetes MLOps)

**From a CSV file to a tracked, registered model with a single API call.**

kMLOps is a lightweight MLOps service that turns your Kubernetes cluster into a model training platform. Upload a dataset, describe the model you want in a declarative [Ludwig](https://ludwig.ai) configuration, and kMLOps takes care of the rest: it runs the training as an isolated Kubernetes Job, records every metric, parameter and artifact in [MLflow](https://mlflow.org), and registers the resulting model, ready to be served.

It is also a single entry point to your whole MLflow tracking server. Experiments created outside kMLOps are listed, inspected and managed through the same REST API, so frontends and other services never need to talk to MLflow directly.

> [!NOTE]
> kMLOps is the MLOps component of **[OpenTwinsV2](https://github.com/ertis-research/opentwins)**, the open-source platform for building digital twins, where it powers model training and experiment tracking. It has no dependency on OpenTwins and can be deployed on its own in any Kubernetes cluster.

---

## Key Features

* **No-code model training:** Define models declaratively through Ludwig JSON configurations (binary and multi-class classification, regression and more), without writing deep learning code.
* **Kubernetes-native execution:** Every training runs in its own isolated, ephemeral Job, keeping heavy workloads off the API and scaling with your cluster.
* **Full traceability with MLflow:** Final and per-epoch metrics for the train, validation and test splits, parameters, artifacts and model versions are recorded automatically.
* **Training lifecycle control:** Follow the status of each training, read its logs or cancel it through the API.
* **Unified experiment management:** List, search, inspect and delete experiments and runs, whether they were trained by kMLOps or logged to MLflow by any other tool.
* **Learning curves out of the box:** Fetch the per-epoch history of any run metric to plot training and validation curves.
* **S3-compatible dataset storage:** Datasets are stored in MinIO, so every training can be traced back to the exact data it used.
* **Frontend-ready API:** Asynchronous, paginated REST endpoints with CORS enabled, designed to be consumed directly by web applications.

---

## Tech Stack

* **API Framework:** FastAPI (Python)
* **Data Storage:** MinIO
* **Container Orchestration:** Kubernetes (K8s)
* **Machine Learning Engine:** Ludwig AI
* **Tracking & Registry:** MLflow

---

## Prerequisites

To run this project, the following dependencies must be configured in your environment:

1. Python 3.10+
2. Access to a Kubernetes cluster (with a valid `~/.kube/config` file).
3. An operational MinIO server with access credentials.
4. An operational MLflow Tracking server.

---

## Local Installation and Deployment

### 1. Clone the repository

```bash
git clone https://github.com/ertis-research/kmlops.git
cd kmlops
```

### 2. Create and activate a virtual environment

```bash
py -3.12 -m venv .venv
```

#### On Windows

```powershell
.\.venv\Scripts\activate
```

#### On Linux / macOS

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env` in the project root and fill in your values:

```bash
cp .env.example .env
```

The minimum configuration looks like this:

```env
MINIO_URL=minio.your-domain.com
MINIO_ACCESS_KEY=your_access_key
MINIO_SECRET_KEY=your_secret_key
BUCKET_NAME=your_bucket_name
KUBERNETES_CONFIG_PATH=~/.kube/config
KUBERNETES_NAMESPACE=<YOUR_K8S_NAMESPACE>
MLFLOW_TRACKING_URI=http://your-mlflow-host:5000

# Training store (see "Training store" below)
DB_BACKEND=sqlite
SQLITE_PATH=./data/kmlops.db
```

All configuration comes from environment variables. If a required variable is missing, kmlops does not start and the log names the variable.

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `MINIO_URL`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` | yes | | MinIO endpoint and credentials |
| `BUCKET_NAME` | yes | | Bucket for the uploaded datasets |
| `MINIO_SECURE` | no | `true` | Use HTTPS for MinIO |
| `KUBERNETES_NAMESPACE` | yes | | Namespace where training Jobs run |
| `KUBERNETES_CONFIG_PATH` | no | `~/.kube/config` | Kubeconfig, used only outside the cluster |
| `MLFLOW_TRACKING_URI` | no | | MLflow server. Without it, only local trainings are tracked |
| `TRAINING_IMAGE` | no | `ludwigai/ludwig:latest` | Image of the training Jobs |
| `TRAINING_JOB_TTL_SECONDS` | no | `60` | Kubernetes removes finished Jobs after this time |
| `TRAINING_JOB_BACKOFF_LIMIT` | no | `1` | Retries of a failed training pod |
| `STATUS_POLL_INTERVAL_SECONDS` | no | `10` | How often training statuses are refreshed |
| `LAUNCH_TIMEOUT_SECONDS` | no | `300` | A training still `queued`/`launching_job` after this time is marked `failed` |
| `DB_BACKEND` | no | `sqlite` | `sqlite` or `postgres` |
| `SQLITE_PATH` | no | `/data/kmlops.db` | SQLite file (when `DB_BACKEND=sqlite`) |
| `POSTGRES_HOST`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` | if `postgres` | | PostgreSQL connection |
| `POSTGRES_PORT` | no | `5432` | PostgreSQL port |

### Training store

kmlops keeps its trainings in a SQL database, so they survive restarts. Choose the backend with `DB_BACKEND`:

* **`sqlite`** (default): a file inside the kmlops container, with no extra services. In Kubernetes, `k8s/deployment.yaml` mounts the `kmlops-data` volume at `/data`. Without a volume (for example `docker run` without `-v`), data is lost when the container is removed. Only one replica can be used.
* **`postgres`**: an external PostgreSQL server (the database must exist; kmlops creates its table at startup). Use it to share the store or to run several replicas. In Kubernetes, create the optional `kmlops-postgres` Secret, which the deployment loads automatically:

```bash
kubectl -n <YOUR_K8S_NAMESPACE> create secret generic kmlops-postgres \
  --from-literal=DB_BACKEND=postgres \
  --from-literal=POSTGRES_HOST=<host> --from-literal=POSTGRES_DB=<db> \
  --from-literal=POSTGRES_USER=<user> --from-literal=POSTGRES_PASSWORD=<password>
```

### 5. Start the API

```bash
python -m uvicorn src.main:app --reload --port 8000
```

### Running tests

The tests use dummy settings and mocks, so they need neither a `.env` file nor Postgres, MinIO, MLflow or Kubernetes.

```bash
python -m venv .venv            # or reuse the one from step 2
# activate it (Windows: .venv\Scripts\activate, Linux/macOS: source .venv/bin/activate)
pip install -r requirements-dev.txt
pytest
```

## Docker and Kubernetes Deployment

### 1. Build docker image

```bash
docker build -t <YOUR_REGISTRY>/<YOUR_PROJECT>/kmlops:latest ./
```

### 2. Push docker image into registry

```bash
docker push <YOUR_REGISTRY>/<YOUR_PROJECT>/kmlops:latest
```

After the image is pushed, it can be initialized using docker or it can be deployed on Kubernetes if wanted.

### 3. Deployment

Depending on the user's needs, it can be either locally deployed using Docker or use a Kubernetes pod:

#### 3.1. Docker deployment

```bash
docker run -p 8000:8000 <YOUR_REGISTRY>/<YOUR_PROJECT>/kmlops:latest
```

#### 3.2. Kubernetes deployment

`k8s/deploy.sh` deploys kmlops from your `.env` file (see `.env.example`). It:
0. builds the image from the repository and pushes it as `KMLOPS_IMAGE` (`docker build` + `docker push`). The push uses your own `docker login` on the registry, since `REGISTRY_*` is a pull-only account;
1. creates or updates the `kmlops-minio-credentials` Secret with kmlops's configuration;
2. creates or updates the `kmlops-registry-creds` pull Secret, if `REGISTRY_*` are set;
3. applies `k8s/rbac.yaml` and `k8s/deployment.yaml` in the `KUBERNETES_NAMESPACE` namespace, using `KMLOPS_IMAGE` as the image;
4. restarts kmlops so it reads the new configuration and pulls the image again.

```bash
docker login <YOUR_REGISTRY>    # once, with an account that can push to the image's project
DRY_RUN=1 k8s/deploy.sh         # validate against the cluster, change nothing, build nothing
k8s/deploy.sh                   # build, push and deploy (or k8s/deploy.sh path/to/other.env)
SKIP_BUILD=1 k8s/deploy.sh      # deploy the image already in the registry (e.g. only .env changed)
```

It needs bash, `kubectl` and Docker (on Windows, run it from Git Bash with Docker Desktop running). Credentials never go into git or into the manifests. The Secrets are written with `create`/`replace`, not `kubectl apply`, so their values are not copied into the `last-applied-configuration` annotation.

What it takes from `.env`:

| Variable | Used for |
| --- | --- |
| `KUBERNETES_NAMESPACE` | The namespace where kmlops is deployed and where it launches the training Jobs |
| `KMLOPS_IMAGE` | The kmlops image that the script builds, pushes and deploys |
| `REGISTRY_SERVER`, `REGISTRY_USERNAME`, `REGISTRY_PASSWORD` | The `kmlops-registry-creds` pull Secret. Required the first time: `deploy.sh` fails if `kmlops-registry-creds` does not exist in the namespace and these are not set. An older `model-registry-creds` Secret is no longer reused. Once `kmlops-registry-creds` exists, you can omit them to keep it |
| Everything else | The `kmlops-minio-credentials` Secret, i.e. kmlops's configuration. `SQLITE_PATH` and `KUBERNETES_CONFIG_PATH` are local-only and are not copied: in the cluster kmlops uses `/data/kmlops.db` (the persistent volume) and its service account |

**Registry credentials.** They need **pull permission on the registry project where the image is**; otherwise the pod stays in `ImagePullBackOff` with `401 Unauthorized`. Harbor robot accounts are usually limited to their own project. For example, a robot of project `project-a` can pull from `<registry>/project-a/...` but not from `<registry>/project-b/...`. Push the image to the robot's project, or give the robot pull permission on the image's project. Use a robot account, not a personal user.

**Without the script.** The manifests need both Secrets to exist and `image:` to be edited by hand:

```bash
kubectl -n <ns> create secret docker-registry kmlops-registry-creds   --docker-server=<registry> --docker-username='<robot>' --docker-password='<token>'
kubectl -n <ns> create secret generic kmlops-minio-credentials --from-env-file=<env file without SQLITE_PATH>
kubectl apply -n <ns> -f k8s/rbac.yaml -f k8s/deployment.yaml
```

Pod status can be checked using the command:

```bash
kubectl get pods -n <YOUR_K8S_NAMESPACE>
```

Once deployed, Kubernetes will expose the API through a Service. To find the API adress and port to access the API, run:

```bash
kubectl get svc -n <YOUR_K8S_NAMESPACE>
```

---

## API Endpoints (Prefix: `/api/mlops`)

The API follows a modular path structure. Interactive documentation is available at `http://localhost:8000/docs`.

### 1. List Experiments

* **Method:** `GET`
* **Path:** `/api/mlops/experiments`
* **Description:** Returns one entry per experiment (model), merging local trainings and MLflow history. `GET /api/mlops/experiments/{experiment_name}` returns a single experiment with the same fields.
* **Query params:**
  * `page` (default `1`, 1-based; `0` or negative gives 422)
  * `pageSize` (default `20`, max `1000`)
  * `search` (optional, matches the experiment name)
* **Response:** `results` plus a `pagination` object. A page beyond the last one returns an empty `results` (it is not clamped to the last page).

```json
{
  "results": [{ "experiment_name": "my-experiment", "status": "completed" }],
  "pagination": { "total_items": 25, "total_pages": 2, "current_page": 1, "page_size": 20, "has_next": true, "has_prev": false }
}
```

#### Training status

Each experiment has a `status` from this closed list and, when it failed, a `status_reason`:

| `status` | Meaning |
| --- | --- |
| `queued` | Request accepted and dataset stored; the Kubernetes Job has not been requested yet |
| `launching_job` | The Kubernetes Job is being created |
| `training` | The Job is running (or its pod is pending). `status_reason` explains a stuck pod, e.g. `ImagePullBackOff` |
| `completed` | The training finished and the model is registered in MLflow |
| `failed` | The training failed. `status_reason` gives the cause: the last `ERROR` line of the pod logs, the Job failure message, or e.g. `The training job no longer exists` |
| `cancelled` | Returned by the cancel endpoint. It only appears in listings while the cleanup of a cancellation failed and is waiting to be retried (`status_reason` says why) |

kmlops refreshes the status of active trainings in the background from the Kubernetes Job and the MLflow run. When the latest training of an experiment has a finished MLflow run, the experiment is `completed`, so the listing never contradicts MLflow.

### 2. Create Experiment

* **Method:** `POST`
* **Path:** `/api/mlops/experiments`
* **Description:** Initializes a new training job.
* **Multipart Fields:**
    | Field             | Description                                                        |
    | ----------------- | ------------------------------------------------------------------ |
    | `csv_file`        | Dataset file to be processed (e.g. `examples/predictive_maintenance_rpm/data.csv`) |
    | `model_type`      | Internal reference algorithm (e.g. `Ludwig_Test`)                  |
    | `config_json`     | Stringified JSON object containing Ludwig configuration parameters |
    | `dependencies`    | (Optional) JSON list or comma-separated string of dependencies     |
    | `experiment_name` | Logical grouping name in MLflow (Default: `kmlops-default-experiment`) |
    | `model_name`      | Final model artifact name in the MLflow Model Registry (Default: `ludwig_model`) |

#### Example cURL Request

```bash
curl -X POST "http://localhost:8000/api/mlops/experiments" \
  -F "csv_file=@examples/predictive_maintenance_rpm/data.csv" \
  -F "model_type=Ludwig_Test" \
  -F "experiment_name=Predictive_Maintenance" \
  -F "model_name=Engine_Predictor" \
  -F 'config_json={
    "input_features": [{"name": "rpm", "type": "number"}],
    "output_features": [{"name": "imminent_failure", "type": "binary"}],
    "trainer": {"epochs": 10}
  }'
```

Returns `409 Conflict` if the experiment already has a training in progress.

### 3. Delete Experiment

* **Method:** `DELETE`
* **Path:** `/api/mlops/experiments/{experiment_name}`
* **Description:** Removes the experiment from the local database and the MLflow Tracking Server, and deletes the uploaded datasets (training CSVs) of all its trainings from MinIO. If a training is in progress, its Kubernetes Job is stopped first.
* **Errors:** `404` if the experiment does not exist. `502` if the Kubernetes Job of the training in progress could not be deleted, or if the datasets could not be deleted from MinIO; nothing else is deleted in that case, so the request can be retried.

### 4. Cancel Training

* **Method:** `POST`
* **Path:** `/api/mlops/experiments/{experiment_name}/cancel`
* **Description:** Stops the training in progress (`queued`, `launching_job` or `training`) and removes it:
    - deletes the Kubernetes Job and its pods;
    - marks its MLflow run as `KILLED` and deletes it, with any model version registered from it;
    - deletes its dataset from MinIO.

  If the model has no previous versions, the whole experiment is deleted (as with `DELETE`). Otherwise only this training is removed and the existing versions are kept.
* **Responses:** `404` if there is no training in progress for that name. `200` otherwise:

```json
{
  "message": "Training 'ab12cd34' of 'X' cancelled; experiment deleted because it had no versions",
  "experiment_name": "X",
  "training_id": "ab12cd34",
  "status": "cancelled",
  "experiment_deleted": true,
  "cleanup_complete": true,
  "deleted": {"k8s_job": "kmlops-train-job-ab12cd34", "mlflow_runs": [], "model_versions": [], "minio_objects": 1, "local_record": true},
  "errors": []
}
```

If a step fails (for example, MinIO unreachable), `cleanup_complete` is `false` and `errors` says which one. In that case:
- the training stays as `cancelled` and the experiment is not deleted;
- kmlops retries the cleanup automatically, and calling the endpoint again also retries it.

Cancelling only ever deletes the cancelled training itself. If a new training of the same model has already started, it is not touched.

### 5. Get Runs

* **Method:** `GET`
* **Path:** `/api/mlops/experiments/{experiment_name}/runs`
* **Description:** Retrieves all training runs associated with the experiment name from MLflow, including metrics and parameters.
* **Query params:**
  * `page` (default `1`, 1-based; `0` or negative gives 422)
  * `pageSize` (default `20`, max `1000`)
  * `search` (optional, matches run id or run name)
* **Response:** `results` plus a `pagination` object. A page beyond the last one returns an empty `results` (it is not clamped to the last page).

```json
{
  "results": [{ "run_id": "...", "status": "FINISHED", "metrics": {}, "params": {}, "tags": {} }],
  "pagination": { "total_items": 25, "total_pages": 2, "current_page": 1, "page_size": 20, "has_next": true, "has_prev": false }
}
```

### 5b. Get Metric History

* **Method:** `GET`
* **Path:** `/api/mlops/experiments/{experiment_name}/runs/{run_id}/metrics/history`
* **Query param:** `keys` (optional, repeatable, e.g. `?keys=loss&keys=accuracy`, max `100`): metrics to return. If omitted, all the metrics of the run are returned. Duplicates are ignored.
* **Description:** Returns the step-by-step history of the metrics of a run from MLflow. Points are sorted by `step` (then `timestamp`, in ms); non-finite values are skipped; a requested key that the run does not have gives `[]`.
* **Responses:**
    - `200`: the history.
    - `404`: the experiment is not in MLflow, or the run does not exist, is deleted or belongs to another experiment.
    - `422`: more than 100 keys.
    - `500`: MLflow is not configured, or MLflow returned an error.

```json
{
  "run_id": "3f2a...",
  "metrics": {
    "loss": [{"step": 0, "value": 0.9, "timestamp": 1760000000000}, {"step": 1, "value": 0.7, "timestamp": 1760000001000}]
  }
}
```

### 6. Get Training Logs

* **Method:** `GET`
* **Path:** `/api/mlops/experiments/{experiment_name}/logs`
* **Query param:** `tail_lines` (default `500`, range `1`-`5000`): number of log lines to return, counted from the end.
* **Description:** Returns the logs of the training pod of the experiment (the training in progress, or else the latest one). It is intended to show the training progress live in the OpenTwins UI / Grafana, by polling it.
* **Responses:**
    - `200`: the logs, or `available: false` with a `reason` (see below).
    - `404`: the experiment does not exist.
    - `502`: Kubernetes returned an error while reading the logs.

```json
{
  "experiment_name": "X",
  "training_id": "ab12cd34",
  "job_name": "kmlops-train-job-ab12cd34",
  "pod_name": "kmlops-train-job-ab12cd34-xk2lp",
  "status": "training",
  "pod_phase": "Running",
  "logs": "Epoch 1/10 ...\n",
  "available": true,
  "reason": null
}
```

Cases that return `available: false` (with `logs` empty):
- `"The training has not started yet."`: the training is `queued` or `launching_job`.
- `"The training pod is starting."`: the pod exists but its container has not started yet (the pod waiting reason is appended in parentheses when known).
- `"The training pod no longer exists; its logs are not kept."`: the Job or its pod has been deleted.
- `"This model was not trained by kmlops, or its training is no longer kept."`: the experiment only exists in MLflow.

> **Note:** because of the TTL (`TRAINING_JOB_TTL_SECONDS`), the Job and its pods are deleted after the training finishes, so the logs of finished trainings are not kept.

---

## Architecture Workflow

```text
1. API receives dataset + training configuration.
2. CSV file is uploaded to MinIO.
3. Kubernetes provisions a Ludwig training Job.
4. Pod downloads dataset and executes training.
5. MLflow logs metrics and registers the trained model.
```

### Detailed Execution Flow

1. **Ingestion:** API receives the dataset and training JSON. The CSV is uploaded to MinIO under `<experiment_id>/filename.csv`.
2. **Orchestration:** A Kubernetes Job is created using the `TRAINING_IMAGE` image (`ludwigai/ludwig:latest` by default).
3. **Training:** The Pod executes a generated Python script that:
    - Downloads the dataset from MinIO.
    - Initializes the `LudwigModel`.
    - Starts the training pipeline.
4. **Traceability:** Upon completion, training statistics and the model artifact are automatically logged to MLflow via integrated callbacks.

---

## Project Structure

```text
.
├── src/
│   ├── main.py                # FastAPI app: lifespan (store, MinIO, status poller), CORS, routers
│   ├── config.py              # Settings read from environment variables
│   ├── schemas.py             # Schemas and the training status vocabulary
│   ├── api/
│   │   └── experiments.py     # HTTP routes (/api/mlops/experiments...)
│   ├── services/              # Business logic behind the routes
│   │   ├── experiments.py     # List/get/create/delete experiments, status resolution
│   │   ├── trainings.py       # Training pod logs and cancellation
│   │   ├── runs.py            # MLflow runs of an experiment
│   │   └── pagination.py      # Shared pagination helper
│   ├── training/              # Lifecycle of a training in the background
│   │   ├── launcher.py        # Builds and creates the Kubernetes training Job
│   │   ├── trainer_script.py  # Script run inside the training pod (Ludwig + MLflow)
│   │   ├── status.py          # Background refresh of training statuses
│   │   └── cancellation.py    # Cleanup of cancelled trainings
│   ├── integrations/          # Clients for external services
│   │   ├── k8s.py             # Kubernetes API: Jobs, pods, failure reasons
│   │   ├── minio_storage.py   # MinIO dataset storage
│   │   └── mlflow_tracking.py # MLflow helpers (runs of a training, cleanup)
│   └── db/
│       └── store.py           # Training store (SQLite or PostgreSQL)
├── tests/             # pytest suite (no external services needed)
├── examples/          # Sample datasets and Ludwig configs (see examples/README.md)
│   ├── generate_examples.py  # Deterministic dataset generator
│   └── <dataset>/     # data.csv, ludwig_config.json, README.md
├── docs/              # Documentation and runbooks
├── k8s/               # Kubernetes manifests and deploy script
│   ├── deploy.sh       # Builds, pushes and deploys kmlops from .env (image + Secrets + manifests)
│   ├── deployment.yaml # Deployment, Service and SQLite volume
│   └── rbac.yaml       # Permissions for training Jobs and pod logs
├── requirements.txt
├── .env.example       # Configuration template (copy to .env)
├── .dockerignore
├── Dockerfile
└── README.md
```

---

## Future Improvements

* Add authentication and role-based access.
* Support distributed training workloads.
* Add GPU scheduling support in Kubernetes.
* Integrate hyperparameter optimization.
* Add experiment monitoring dashboard.

---

## License

This project is licensed under the MIT License.
