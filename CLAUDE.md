# kmlops

kMLOps (Kubernetes MLOps) is a FastAPI service that trains Ludwig models on Kubernetes and tracks them in MLflow. It is the MLOps component of OpenTwins, and it can also be deployed standalone, so its API must stay generic: OpenTwins-specific conventions (for example "one experiment = one model") belong in the OpenTwins frontend, not here.

How a training works: `POST /api/mlops/experiments` stores the CSV in MinIO and records the training in the database. Then it launches a Kubernetes Job that installs pinned `minio`/`mlflow` packages and runs `src/training/trainer_script.py` inside the Ludwig image. The pod trains, logs metrics to MLflow and registers a new version of `model_name` in the MLflow Model Registry. A background poller (`src/training/status.py`) keeps the training status in sync with the Job and the MLflow run.

The user communicates in Spanish: answer in Spanish. Code, docs and commit messages are in English (see below).

## Project layout

- `src/main.py`: app and lifespan (settings, store, bucket, status poller, startup warnings).
- `src/config.py`: every setting, from environment variables, validated with `ConfigError` (`_required`, `_optional`, `_int`, `_bool`, `_version`...).
- `src/api/`: HTTP layer only (FastAPI routers). It maps service exceptions to HTTP errors.
- `src/services/`: use cases (experiments, runs, trainings, feature checks, pagination).
- `src/training/`: Job building and launching (`launcher.py`), status polling, cancellation, and the pod script (`trainer_script.py`). The pod script cannot import `src`: constants shared with it (for example the `kmlops.training_id` tag) are duplicated and must be kept in sync.
- `src/integrations/`: Kubernetes (`k8s.py`), MinIO (`minio_storage.py`) and MLflow (`mlflow_tracking.py`) helpers.
- `src/db/store.py`: training store with SQLite and PostgreSQL backends sharing the same SQL. Database invariants live there as unique indexes, for example one active training per experiment and per model.
- `src/schemas.py`: response types and training statuses.
- `k8s/`: manifests (`deployment.yaml`, `rbac.yaml`) and `deploy.sh`.
- `docs/runbook.md`: operations and troubleshooting. `examples/`: datasets and Ludwig configs.
- `tests/`: pytest suite.

## Design decisions

Project-wide decisions. Follow them in new code, and ask the user before changing any of them.

**Scope and data model**
1. **The kmlops API is generic.** OpenTwins-specific conventions, such as "one experiment = one model", are enforced by the OpenTwins frontend, not by kmlops.
2. **Experiments and registered models are independent.** Every successful training registers a new version of `model_name` in the MLflow Model Registry. `model_name` defaults to `experiment_name`. Several experiments can feed one model, and one experiment can train several models.
3. **All the versions of a model have the same input and output features (name and type).** Feature order, per-feature settings and the rest of the config can change. A new training is checked (`src/services/feature_schema.py`) against two references:
   - the latest completed or in-progress training of the model in the database, in any experiment;
   - the latest version in the MLflow Model Registry, whose features come from the Ludwig run params or, if missing, from the model's `model_hyperparameters.json`.

   A version whose features cannot be read is not checked. If MLflow is unreachable, the request fails with 502 and nothing is created.
4. **Failed and cancelled trainings never count as model versions.**

**Consistency and errors**

5. **Concurrency rules are enforced by the database with unique indexes**, not by application locks. Currently: one training in progress per experiment and one per model.
6. **Every operation that fails halfway can be retried.** A failure of an external system returns 502 and keeps what is needed to retry. The deletion order is: Kubernetes Job, MinIO data, MLflow Model Registry, local rows, MLflow experiment.
7. **Deleting an experiment deletes the model versions it registered, and the registered models left without versions.** A model that also has versions from other experiments keeps those. Cancelling a training does not leave empty registered models either.
8. **Startup never fails because of existing data**, because the poller must be able to finish active trainings. An invalid configuration does stop startup, with a `ConfigError` that names the variable.

**Security and configuration**

9. **Credentials reach the training pods through `secretKeyRef`**, never as plain values in the Job spec. This is why kmlops does not need permission to read Secrets.
10. **The packages installed in the training pod have pinned versions.** Their defaults are the versions in `requirements.txt`, defined in one place (`src/config.py`) and kept in sync by a test. Any value that reaches a shell command is strictly validated.
11. **Everything that depends on the installation is configurable through environment variables** (Secret names and keys, versions...), because the Helm chart passes them as values. Nothing of that kind is hardcoded.

**API responses**

12. **Each run returns only its own keys.** Missing metrics, params and tags are omitted instead of returned as `null` or `"None"`, and NaN and infinite values are never returned.
13. **Pagination is 1-based.** `page` defaults to 1, `page=0` returns 422, and an out-of-range page returns `results: []`.
14. **Training metrics are logged as `<split>_metrics.<feature>.<metric>`** (split: train, validation, test), with `step` = epoch. NaN and infinite values are not logged.

## Agents

There are three project agents in `.claude/agents/`, which is tracked in git: the rest of `.claude/` (local settings) is ignored. Agent definitions are tracked files, so they follow the sensitive information rules below. They are loaded when the session starts, so a new or edited agent is available from the next session.

- **explore** (read-only, haiku): finds code, explains modules and summarizes files. Use it to gather context before a change, and ask for exact `file:line` references and code excerpts.
- **coder** (writes code, sonnet): implements a change from precise instructions. It never commits or deploys.
- **tester** (end user, sonnet): tests the *deployed* API end to end with `curl`, as a user or the frontend would. It cannot read the code, touch files or use `kubectl`, and it may train models. It reports a table of OK / FAIL / INCONCLUSIVE results.

How to use them when the user asks for it:
1. Send **explore** first to collect the context (relevant functions, tests, docs).
2. Give **coder** a concrete design: which files to change, what to change, names, error codes, the tests to write, and the docs to update. Repeat the coding standards below that apply.
3. **Review the result yourself**: read the diff (`git diff`), run the full test suite, and fix or send back whatever is wrong. Never report an agent's work as done without checking it.
4. After the user deploys, run **tester**. Give it:
   - the API URL, taken from `CLAUDE.local.md`;
   - a unique prefix for everything it creates (for example `tester-<MMDD>-`);
   - a step-by-step plan with the expected status codes;
   - a scratchpad directory for temporary files.

   It must not touch resources it did not create, and it deletes its own at the end. Afterwards, check what it cannot see, read-only: the Job spec, the database and the MLflow registry.

## Git

- **Never commit, never create branches, never push.** Leave all changes uncommitted on `main`; the user commits.
- When a task is done, give the user a ready-to-use commit message in professional English following Conventional Commits (`feat`, `fix`, `refactor`, `docs`, `test`, `chore`, with an optional scope such as `feat(api):`, and `!` for breaking changes). It has:
  - a short imperative subject (72 characters at most);
  - a body explaining what changed and why, grouped by area as in the existing history;
  - a `BREAKING CHANGE:` footer when the API or configuration changes incompatibly;
  - the list of files to include.
- Do not add `Co-Authored-By` lines unless the user asks.

## Deployment and environments

- **Never deploy** (`k8s/deploy.sh`, `kubectl apply`/`rollout`, image builds or pushes) unless the user explicitly asks. The user usually deploys.
- Read-only inspection of the test environment is allowed: `kubectl get`/`logs`/`describe`, and `kubectl exec` for read-only queries (MLflow client calls, `sqlite3` opened with `mode=ro`). Anything that modifies the cluster, MLflow, MinIO or the database needs the user's approval, except resources that a test created and that belong to that test.
- Never delete or change other people's experiments or models, even when they look like leftovers: report them and ask.
- Real environment values (API URL, namespace, registry) are in `CLAUDE.local.md`.
- kmlops is also installed with the OpenTwins Helm chart, which lives in another repository. When a change adds or changes environment variables, RBAC permissions, Secrets, ports, volumes or migrations, tell the user what the chart needs.

## Sensitive information

- Never write URLs or IPs of real environments, hostnames, namespaces, credentials, tokens, passwords, emails or personal paths in tracked files: code, docs, `CLAUDE.md`, `.claude/agents/`, tests, examples or commit messages.
- Use placeholders (for example `<kmlops-url>`) or environment variables instead.
- Keep local values in `CLAUDE.local.md` and secrets in `.env`. Both are git-ignored. Do not read `.env` unless necessary; when you need one value, extract only that variable.
- Training Jobs must not carry credentials in plain text in their spec: use `secretKeyRef` (see `MINIO_CREDENTIALS_SECRET`).
- Before finishing a task, check the diff for sensitive data.

## Coding standards

- **English everywhere**: identifiers, comments, docstrings, log and error messages, docs. Use professional wording.
- **Docstrings**: every new or changed module, class and function has a docstring that explains what it does. One line is enough when the behavior is obvious. When it is not, add details on parameters, return values and the exceptions it raises.
- **Conventions**:
  - Python 3.12 and PEP 8: snake_case functions and variables, PascalCase classes, `_private` helpers, `UPPER_CASE` constants.
  - Type hints throughout. New code uses built-in generics and `X | None`; do not rewrite existing `Optional`/`Dict` code only for style.
  - Exception names describe the condition (`ActiveTrainingExists`, `IncompatibleFeatures`, `ConfigError`); `*Error` is fine for failures.
- **Reuse before writing**: look for an existing helper first: `src/config.py` validators, `src/integrations/*`, `src/services/pagination.py`, `src/db/store.py` queries, test fixtures in `tests/conftest.py`. Extract shared logic instead of duplicating it.
- **Layers**: the API layer parses the request and maps exceptions to HTTP errors; services hold the logic; integrations talk to external systems. Blocking calls (MLflow, database) inside `async` endpoints go through `asyncio.to_thread`.
- **Size and structure**: keep functions short and focused, and modules cohesive. Split a module when it mixes responsibilities, but do not create a file per function: group related code by responsibility, as in the existing `src/` subpackages.
- **Readability over cleverness**: clear names, no dead code, comments explain *why*, not *what* the code already says.
- **Errors**: endpoints raise `HTTPException(status_code, detail=<clear message>)`, so clients receive `{"detail": str}`. The usual mapping:
  - 404: not found.
  - 409: conflict with the current state (training in progress, incompatible features).
  - 422: invalid input.
  - 502: failure of an external system (Kubernetes, MinIO, MLflow).

  A request that fails halfway leaves things in a state where it can be retried. Expected states are not errors.
- **Configuration**: every new setting goes through `src/config.py`, with a default when it is optional and validation with a clear `ConfigError` (strict validation when the value reaches a shell command). Document it in the `README.md` variables table and in `.env.example`.
- **Database**: schema changes must work on both SQLite and PostgreSQL and on existing databases (`CREATE ... IF NOT EXISTS`). Startup must not fail because of existing data: kmlops has to start so that the poller can finish active trainings.
- **MLflow data**: runs of one experiment may have different metrics, params and tags (`mlflow.search_runs` fills the missing ones with NaN/None). Never return NaN or infinity in a response.
- **Tests**: every change comes with tests, and the full suite must pass.
  - Tests are pytest functions in `tests/test_<area>.py`.
  - Shared setup goes in fixtures in `tests/conftest.py` (`client`, `set_store`, `k8s`, `FakeStore`, `make_training`). The suite sets dummy environment variables, and the settings cache is cleared for every test.
  - Use `monkeypatch` on the module attribute that the code under test uses (for example `experiments_module.get_store`), `parametrize` for variants, and a real `SQLiteStore(":memory:")` when SQL behavior matters.
  - Fake external clients (MLflow, Kubernetes, MinIO); tests never touch real services.
  - New tests and fixtures have a one-line docstring when the name is not self-explanatory.
  - Run the suite with `.venv\Scripts\python.exe -m pytest -q`.
- **Docs**: update `README.md` (endpoints, errors, variables table), `CHANGELOG.md` (Keep a Changelog, under `## [Unreleased]`, marking incompatible changes with **BREAKING**) and, when relevant, `docs/runbook.md`, together with the code.
- **RBAC**: the Role for training Jobs is in `k8s/rbac.yaml`. Check it before relying on new Kubernetes permissions; referencing a Secret from a pod does not need permission to read Secrets.
