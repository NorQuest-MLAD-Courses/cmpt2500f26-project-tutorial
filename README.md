# Telco Customer Churn Prediction

Predicts whether a telecom customer will churn (cancel their service) from account and service details, using the IBM Telco Customer Churn dataset (7,043 customers, 21 columns).

The project includes a command-line training pipeline (preprocess, train, evaluate, predict), automated tests with pytest, YAML configuration, experiment tracking with MLflow, data and model versioning with DVC and DagsHub, a hyperparameter search that produces two deployment-ready models, a Flask REST API that serves both models, Docker images, a Docker Compose stack, deployment to Google Cloud Run, CI/CD with GitHub Actions, Prometheus metrics, Grafana dashboards, alert rules, and data drift detection with Evidently.

## Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Testing](#testing)
- [Experiment tracking (MLflow)](#experiment-tracking-mlflow)
- [Data versioning (DVC)](#data-versioning-dvc)
- [REST API](#rest-api)
- [Docker](#docker)
- [Docker Compose](#docker-compose)
- [Cloud deployment (Google Cloud Run)](#cloud-deployment-google-cloud-run)
- [CI/CD (GitHub Actions)](#cicd-github-actions)
- [Monitoring](#monitoring)
- [Data drift detection](#data-drift-detection)
- [Project structure](#project-structure)
- [Make targets](#make-targets)
- [References](#references)

## Requirements

- Python 3.11 or newer
- Git
- A [DagsHub](https://dagshub.com/) account and repository (DVC remote storage)
- Docker with Docker Compose
- A [Docker Hub](https://hub.docker.com/) account (to publish images)
- A Google Cloud project with billing enabled and the `gcloud` CLI (see the course's GCP setup guide, `lab_5_gcp_setup_guide.md`)

GitHub Codespaces provides Python, Git and Docker out of the box.

## Setup

Clone the repository and change into it:

```bash
git clone <your-repository-url>
cd <your-repository>
```

Then run the one-command setup:

```bash
make setup
```

This creates the virtual environment in `.venv/`, installs `requirements.txt`, asks once for your DagsHub token (stored only in the git-ignored `.dvc/config.local`), and runs `dvc pull` to download the data and the trained models. It also installs the Google Cloud CLI if it is missing.

To recreate only the virtual environment, run `make venv`.

Commands in this README call tools through `.venv/bin/...`, so you do not need to activate the environment.

## Usage

Run the pipeline stages in order:

```bash
make preprocess  # clean data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv -> data/processed/churn_processed.csv
make train       # train a pipeline -> models/model.pkl (+ MLflow run)
make evaluate    # accuracy, classification report and confusion matrix on the test split
make predict     # predictions for the rows of the processed CSV
```

### Hyperparameter tuning

```bash
make tune
```

Searches over gradient boosting, random forest and logistic regression settings, logs every candidate to MLflow (experiment `churn-tuning`), and saves the two best pipelines by F1 score as `models/model_v1.pkl` and `models/model_v2.pkl`. These pipelines include their own encoding, so they accept raw (cleaned, not encoded) customer data. The API serves them.

The two models are tracked with DVC. After a new search you want to keep, run `.venv/bin/dvc add models/model_v1.pkl models/model_v2.pkl`, commit the two `.dvc` files and run `make dvc-push`.

While it runs, `tune.py` publishes training metrics for Prometheus on port 8002 (`--metrics-port` changes it).

### Command-line options

Every script reads `config/default.yaml` (or the file given with `--config`). Options left out fall back to the config file, so a flag only overrides one setting for one run. Example:

```bash
.venv/bin/python src/train.py --test-size 0.25 --random-state 7 --experiment-name feature-test
```

`src/preprocess.py`:

| Option | Default | Description |
|---|---|---|
| `--config` | `config/default.yaml` | Path to YAML config file |
| `--input` | from config | Override raw data path |
| `--output` | from config | Override output path |

`src/train.py`:

| Option | Default | Description |
|---|---|---|
| `--config` | `config/default.yaml` | Path to YAML config file |
| `--data` | from config | Override processed data path |
| `--model-out` | from config | Override model output path |
| `--test-size` | from config | Override the test-set fraction from the config |
| `--random-state` | from config | Override the random seed from the config |
| `--experiment-name` | from config | Override the MLflow experiment name from the config |

`src/evaluate.py`:

| Option | Default | Description |
|---|---|---|
| `--config` | `config/default.yaml` | Path to YAML config file |
| `--data` | from config | Override processed data path |
| `--model` | from config | Override model path |
| `--test-size` | from config | Override the test-set fraction from the config |
| `--random-state` | from config | Override the random seed from the config |

`src/predict.py`:

| Option | Default | Description |
|---|---|---|
| `--config` | `config/default.yaml` | Path to YAML config file |
| `--data` | from config | Override input data path |
| `--model` | from config | Override model path |

`src/tune.py`:

| Option | Default | Description |
|---|---|---|
| `--config` | `config/default.yaml` | Path to YAML config file |
| `--experiment-name` | `churn-tuning` | MLflow experiment to log the search runs to |
| `--metrics-port` | `8002` | Port for the Prometheus training-metrics server |

## Configuration

Settings live in `config/default.yaml`:

| Key | Value |
|---|---|
| `paths.raw_data` | `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| `paths.processed_data` | `data/processed/churn_processed.csv` |
| `paths.model` | `models/model.pkl` |
| `features.numerical` | `tenure`, `MonthlyCharges`, `TotalCharges` |
| `training.test_size` | `0.3` |
| `training.random_state` | `40` |
| `model.name` | `GradientBoostingClassifier` |
| `model.params.random_state` | `40` |
| `mlflow.tracking_uri` | `sqlite:///mlflow.db` |
| `mlflow.experiment_name` | `churn-prediction` |

To try different settings, copy the file (for example to `config/experiment.yaml`), edit the copy, and pass it with `--config config/experiment.yaml`. Committing the copy keeps a record of the experiment.

## Testing

```bash
make test
```

Runs 29 tests with pytest (configured in `pytest.ini`):

- `tests/test_api.py`: 11 tests. Automated tests for the Flask API endpoints.
- `tests/test_config.py`: 2 tests. Tests for the config loader.
- `tests/test_drift.py`: 2 tests. Tests for the drift simulation in src/drift.py.
- `tests/test_monitoring.py`: 2 tests. Tests for the training monitor in src/utils/monitoring.py.
- `tests/test_preprocess.py`: 4 tests. Minimal tests for the preprocessing module.
- `tests/test_train.py`: 5 tests. Tests for the training module.
- `tests/test_tune.py`: 3 tests. Tests for the hyperparameter-tuning module.

The API tests load `models/model_v1.pkl` and `models/model_v2.pkl`, so get the models first (`make dvc-pull` or `make tune`).

## Experiment tracking (MLflow)

Every run of `train.py` and every candidate in `tune.py` is logged to MLflow: parameters, metrics (accuracy and F1) and the trained pipeline as an MLflow logged model.

- Run metadata goes to the SQLite database `mlflow.db`; model files go to `mlruns/`. Both are git-ignored.
- Runs are grouped by experiment: `churn-prediction` (from `config/default.yaml`) and `churn-tuning` (for `tune.py`). Use `--experiment-name <name>` to log a run to a different experiment.

Browse the runs with:

```bash
make mlflow-ui
```

and open http://127.0.0.1:5000 (in Codespaces, use the forwarded port 5000 in the Ports tab).
The API also uses port 5000, so stop one before starting the other.

Inside the Docker Compose stack, training logs to the MLflow server container instead (see [Docker Compose](#docker-compose)).

## Data versioning (DVC)

Large files are versioned with [DVC](https://doc.dvc.org/) and stored on DagsHub; Git only keeps the small `.dvc` pointer files. Tracked with DVC: `data/raw/`, `data/processed/`, `models/model_v1.pkl` and `models/model_v2.pkl`.

```bash
make dvc-pull     # download the tracked files
make dvc-push     # upload new versions after `dvc add`
```

Your DagsHub credentials live in `.dvc/config.local`, which is git-ignored. In a new Codespace, run `make setup` (or the two `dvc remote modify origin --local` commands) to restore them.

## REST API

Start the development server:

```bash
make api
```

It listens on http://127.0.0.1:5000. In the Docker image the API runs under Gunicorn instead of Flask's development server.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health Check |
| GET | `/info` | API Information |
| POST | `/v1/predict` | Predict churn using model v1 |
| POST | `/v2/predict` | Predict churn using model v2 |
| GET | `/metrics` | Expose Prometheus metrics |
| GET | `/apidocs/` | Interactive Swagger UI (generated from the route docstrings) |

Example request:

```bash
curl -X POST http://127.0.0.1:5000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "MonthlyCharges": 59.95, "TotalCharges": 720.5, "Contract": "One year", "PaymentMethod": "Electronic check", "OnlineSecurity": "No", "TechSupport": "No", "InternetService": "DSL", "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No", "PaperlessBilling": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "No", "StreamingTV": "No", "StreamingMovies": "No"}'
```

Response:

```json
{"prediction": "No", "probability": 0.94, "model_version": "v1"}
```

Send a JSON list of customers instead of one object to get a list of predictions. Missing fields, wrong types or an empty body return HTTP 400 with an `error` message.

The API logs to the console and to `logs/api.log` (the folder is set by `LOG_DIR`).

## Docker

Build and run the API as a container:

```bash
make docker-build   # build the image churn-api
make docker-run     # run it in the background on port 5000
make docker-stop    # stop and remove the container
```

The image installs `requirements.txt`, copies `src/`, `models/` and `config/`, and serves the API with Gunicorn. `.dockerignore` keeps the virtual environment, data and other local files out of the build.

## Docker Compose

Start the whole stack:

```bash
make compose-up     # build and start all services
make compose-down   # stop and remove the containers
```

| Service | URL | Purpose |
|---|---|---|
| `api` | http://localhost:5000 | Prediction API |
| `mlflow` | http://localhost:5001 | MLflow tracking server and UI |
| `prometheus` | http://localhost:9090 | Metrics collection and alerting (`prom/prometheus:v3.13.4`) |
| `grafana` | http://localhost:3000 | Dashboards (login admin / admin) (`grafana/grafana:12.4.12`) |

In Codespaces, open each port from the Ports tab.

Train or tune inside the stack. The scripts then log to the MLflow server at `http://mlflow:5001`:

```bash
make compose-train
make compose-tune
```

The MLflow server keeps its database and model files in the named volume `mlflow-data`, separate from your local `mlflow.db`. To send a local run to the server, set `MLFLOW_TRACKING_URI=http://localhost:5001` for that command.

`make compose-drift` runs the drift report inside the API container (see [Data drift detection](#data-drift-detection)).

### Publishing images

```bash
docker login
make docker-build
make compose-up
DOCKER_USER=<your-dockerhub-username> make docker-tag
DOCKER_USER=<your-dockerhub-username> make docker-push
```

This publishes `churn-api` and `churn-mlflow` to your Docker Hub account.

## Cloud deployment (Google Cloud Run)

Deploy the image you published to Docker Hub:

```bash
make docker-build
DOCKER_USER=<your-dockerhub-username> make docker-tag
DOCKER_USER=<your-dockerhub-username> make docker-push
DOCKER_USER=<your-dockerhub-username> make deploy
```

`make deploy` runs `gcloud run deploy` for the service `churn-api` in `us-central1`. It prints a public HTTPS URL; check it with `curl <url>/health`. Cloud Run scales to zero when idle. One-time Google Cloud setup is described in the course's GCP setup guide (`lab_5_gcp_setup_guide.md`).

## CI/CD (GitHub Actions)

| Workflow | Runs on | What it does |
|---|---|---|
| `.github/workflows/ci.yml` (CI — Tests) | push to `main`, pull_request to `main` | runs the tests |
| `.github/workflows/deploy.yml` (CD — Deploy to Cloud Run) | push to `main` | runs the tests, then builds the image, pushes it to Artifact Registry and deploys to Cloud Run |

Add these repository secrets under Settings, Secrets and variables, Actions:

- `DAGSHUB_TOKEN`: your DagsHub token, used to `dvc pull` the trained models (they are not in Git)
- `GCP_PROJECT_ID`: your Google Cloud project ID
- `GCP_SA_KEY`: the JSON key of a Google Cloud service account allowed to push to Artifact Registry and deploy to Cloud Run

## Monitoring

### Prometheus

Prometheus (http://localhost:9090) scrapes these targets (Status, Targets shows their state):

- `churn-api` at `api:5000`: API metrics: request counts and latency per route, plus `prediction_requests_total`, `prediction_duration_seconds` and `active_models_loaded`
- `model-training` at `trainer:8002`: training metrics from `tune.py`; UP only while `make compose-tune` runs

The API exposes its metrics at `/metrics`.

### Grafana

Grafana (http://localhost:3000, login admin / admin) is provisioned automatically with Prometheus as its data source and these dashboards in the folder ML Monitoring:

- API Performance: Request Rate (requests/sec), Response Latency (p50 / p95 / p99), Error Rate (4xx + 5xx / total), Prediction Requests by Model Version, Prediction Latency by Model Version
- Training Monitoring: Training vs Validation Accuracy, Training vs Validation F1, CPU Usage During Training, Memory Usage During Training, Top Feature Importances, Training Epochs Completed

### Alerts

Alert rules in `monitoring/rules/ml_alerts.yml` (see http://localhost:9090/alerts):

| Alert | Fires when |
|---|---|
| `HighPredictionErrorRate` | High prediction error rate (> 10%) (for 2m) |
| `SlowPredictionResponse` | Slow prediction response time (for 5m) |
| `APIDown` | Churn API is down (for 1m) |
| `HighTrainingMemory` | High memory usage during training (for 5m) |

## Data drift detection

```bash
make drift           # run locally
make compose-drift   # run inside the API container
```

Compares a reference dataset with current data using [Evidently](https://docs.evidentlyai.com/) and writes an HTML report to `reports/` (git-ignored). Options:

| Option | Default | Description |
|---|---|---|
| `--config` | `config/default.yaml` | Path to YAML config file |
| `--current` | not set | Path to current (production) data CSV. If not provided, the script simulates drift by perturbing the reference data. |
| `--output` | `reports/drift_report.html` | Path to save the HTML report. |

## Project structure

```
.
├── .dvc/
│   └── config                                       # DVC remote settings (committed; keys go in the git-ignored config.local)
├── .github/
│   └── workflows/
│       ├── ci.yml                                   # CI: run tests on every push and pull request
│       └── deploy.yml                               # CD: test, build, push and deploy on push to main
├── config/
│   └── default.yaml                                 # Central configuration
├── data/
│   ├── processed/                                   # generated by `make preprocess` (git-ignored, tracked by DVC)
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv     # Raw dataset (tracked by DVC)
│   ├── processed.dvc                                # DVC pointer for data/processed/
│   └── raw.dvc                                      # DVC pointer for data/raw/
├── models/                                          # trained models (`*.pkl` are git-ignored)
│   ├── model_v1.pkl.dvc                             # DVC pointer for the best tuned model
│   └── model_v2.pkl.dvc                             # DVC pointer for the second-best tuned model
├── monitoring/
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   ├── api-dashboard.json                   # API Performance dashboard
│   │   │   └── training-dashboard.json              # Training Monitoring dashboard
│   │   └── provisioning/
│   │       ├── dashboards/
│   │       │   └── dashboards.yml                   # Grafana dashboard provider
│   │       └── datasources/
│   │           └── prometheus.yml                   # Grafana data source
│   ├── rules/
│   │   └── ml_alerts.yml                            # Prometheus alert rules
│   └── prometheus.yml                               # Prometheus scrape configuration
├── notebooks/
│   └── proof_of_concept.ipynb                       # Original proof-of-concept notebook
├── src/
│   ├── utils/
│   │   ├── __init__.py                              # Marks the folder as a Python package
│   │   ├── config.py                                # Load a YAML configuration file and return it as a dictionary
│   │   └── monitoring.py                            # Prometheus metrics for training processes
│   ├── __init__.py                                  # Marks the folder as a Python package
│   ├── app.py                                       # Flask REST API for the churn prediction service
│   ├── drift.py                                     # Generate a data drift report comparing reference and current data
│   ├── evaluate.py                                  # Load a saved pipeline and evaluate it on the test split
│   ├── predict.py                                   # Load a saved pipeline and print predictions for new data
│   ├── preprocess.py                                # Load raw data, clean it, encode categoricals, and save processed data
│   ├── train.py                                     # Load processed data, split, train a pipeline, save it, and log to MLflow
│   └── tune.py                                      # Hyperparameter search across model families
├── tests/
│   ├── __init__.py                                  # Marks the folder as a Python package
│   ├── test_api.py                                  # Automated tests for the Flask API endpoints
│   ├── test_config.py                               # Tests for the config loader
│   ├── test_drift.py                                # Tests for the drift simulation in src/drift.py
│   ├── test_monitoring.py                           # Tests for the training monitor in src/utils/monitoring.py
│   ├── test_preprocess.py                           # Minimal tests for the preprocessing module
│   ├── test_train.py                                # Tests for the training module
│   └── test_tune.py                                 # Tests for the hyperparameter-tuning module
├── .dockerignore                                    # Files kept out of Docker builds
├── .dvcignore                                       # Files DVC should ignore
├── .gitignore                                       # Files Git should not track
├── docker-compose.yml                               # Multi-container stack
├── Dockerfile                                       # API image (Gunicorn)
├── Dockerfile.mlflow                                # MLflow tracking server image
├── Makefile                                         # Short commands for every task (`make <target>`)
├── pytest.ini                                       # pytest settings
├── README.md                                        # This file
└── requirements.txt                                 # Python dependencies
```

Files under `data/` and the tuned models are downloaded with `make dvc-pull`.

## Make targets

| Target | What it does |
|---|---|
| `make venv` | Creates the project's virtual environment in .venv/ and installs every package listed in requirements.txt into it. |
| `make setup` | One-command setup for a fresh clone or a new Codespace. |
| `make preprocess` | Cleans the raw CSV and writes the processed data to data/processed/. |
| `make train` | Trains the model on the processed data and saves it in models/. |
| `make tune` | Runs the hyperparameter search in src/tune.py, logs every candidate to MLflow and saves the two best pipelines as models/model_v1.pkl and models/model_v2.pkl (the two API versions). |
| `make evaluate` | Loads the saved model and prints its scores on the test split. |
| `make predict` | Loads the saved model and makes predictions for new customers. |
| `make test` | Runs the automated tests in tests/ with pytest (settings in pytest.ini). |
| `make api` | Starts the Flask API on your machine (outside Docker) for quick testing. |
| `make mlflow-ui` | Opens the MLflow web interface so you can browse and compare runs. |
| `make dvc-push` | Uploads DVC-tracked files to the DagsHub remote. |
| `make dvc-pull` | Downloads the DVC-tracked files that match the current .dvc files in Git. |
| `make docker-build` | Builds a Docker image from the Dockerfile in this folder (the final "." is the build context) and names it churn-api with -t. |
| `make docker-run` | Starts a container from the churn-api image. |
| `make docker-stop` | Stops the churn-api container and then removes it, so the name is free for the next `make docker-run`. |
| `make compose-up` | Starts every service in docker-compose.yml. |
| `make compose-down` | Stops and removes the Compose containers and network. |
| `make compose-train` | Runs the training script inside a one-off container built from the api service, so it talks to the MLflow container over the Compose network. |
| `make compose-tune` | Runs the tuning script inside a one-off api container (removed afterwards by --rm), logging to the MLflow container. |
| `make docker-tag` | (or. |
| `make docker-push` | Uploads both tagged images to Docker Hub. |
| `make deploy` | Deploys the API image from Docker Hub to Google Cloud Run. |
| `make drift` | Runs the Evidently data-drift check (src/drift.py) on your machine and writes an HTML report to reports/. |
| `make compose-drift` | Runs the same drift check inside a one-off api container. |
| `make clean` | Deletes generated files so you can start again from a clean state. |

## References

- [Telco Customer Churn dataset (IBM sample data, via Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [Original notebook: Customer Churn Prediction by Bharti Prasad (Kaggle)](https://www.kaggle.com/code/bhartiprasad17/customer-churn-prediction)
- [Python virtual environments (venv)](https://docs.python.org/3/library/venv.html)
- [pandas documentation](https://pandas.pydata.org/docs/)
- [scikit-learn: Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html)
- [pytest documentation](https://docs.pytest.org/)
- [argparse (command-line options)](https://docs.python.org/3/library/argparse.html)
- [PyYAML documentation](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [skops (safe scikit-learn model persistence used by MLflow)](https://skops.readthedocs.io/en/stable/)
- [DVC documentation](https://doc.dvc.org/)
- [DagsHub storage](https://dagshub.com/docs/feature_guide/dagshub_storage/)
- [Flask documentation](https://flask.palletsprojects.com/)
- [Flasgger (Swagger UI for Flask)](https://github.com/flasgger/flasgger)
- [Gunicorn](https://docs.gunicorn.org/en/stable/)
- [Docker documentation](https://docs.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- [Google Cloud Run](https://docs.cloud.google.com/run/docs)
- [GitHub Actions](https://docs.github.com/en/actions)
- [Prometheus documentation](https://prometheus.io/docs/)
- [prometheus_flask_exporter](https://github.com/rycus86/prometheus_flask_exporter)
- [Grafana provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/)
- [Evidently](https://docs.evidentlyai.com/)

The dataset is IBM sample data distributed on Kaggle by user blastchar. The proof-of-concept notebook is by Bharti Prasad (Kaggle user bhartiprasad17); it is the starting point that this project refactors, not original course work.
