# Telco Customer Churn Prediction

Predicts whether a telecom customer will churn (cancel their service) from account and service details, using the IBM Telco Customer Churn dataset (7,043 customers, 21 columns).

The project includes a command-line training pipeline (preprocess, train, evaluate, predict), automated tests with pytest, YAML configuration, and experiment tracking with MLflow.

## Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Testing](#testing)
- [Experiment tracking (MLflow)](#experiment-tracking-mlflow)
- [Project structure](#project-structure)
- [Make targets](#make-targets)
- [References](#references)

## Requirements

- Python 3.11 or newer
- Git

GitHub Codespaces provides Python and Git out of the box.

## Setup

Clone the repository and change into it:

```bash
git clone <your-repository-url>
cd <your-repository>
```

Create the virtual environment and install the dependencies:

```bash
make venv
```

Commands in this README call tools through `.venv/bin/...`, so you do not need to activate the environment.

The raw dataset is `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Usage

Run the pipeline stages in order:

```bash
make preprocess  # clean data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv -> data/processed/churn_processed.csv
make train       # train a pipeline -> models/model.pkl (+ MLflow run)
make evaluate    # accuracy, classification report and confusion matrix on the test split
make predict     # predictions for the rows of the processed CSV
```

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

Runs 7 tests with pytest (configured in `pytest.ini`):

- `tests/test_config.py`: 2 tests. Tests for the config loader.
- `tests/test_preprocess.py`: 4 tests. Minimal tests for the preprocessing module.
- `tests/test_train.py`: 1 test. Minimal tests for the training module.

## Experiment tracking (MLflow)

Every run of `train.py` is logged to MLflow: parameters, metrics (accuracy and F1) and the trained pipeline as an MLflow logged model.

- Run metadata goes to the SQLite database `mlflow.db`; model files go to `mlruns/`. Both are git-ignored.
- Runs are grouped by experiment: `churn-prediction` (from `config/default.yaml`). Use `--experiment-name <name>` to log a run to a different experiment.

Browse the runs with:

```bash
make mlflow-ui
```

and open http://127.0.0.1:5000 (in Codespaces, use the forwarded port 5000 in the Ports tab).

## Project structure

```
.
├── config/
│   └── default.yaml                                 # Central configuration
├── data/
│   ├── processed/                                   # generated by `make preprocess` (git-ignored)
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv     # Raw dataset
├── models/                                          # trained models (`*.pkl` are git-ignored)
├── notebooks/
│   └── proof_of_concept.ipynb                       # Original proof-of-concept notebook
├── src/
│   ├── utils/
│   │   ├── __init__.py                              # Marks the folder as a Python package
│   │   └── config.py                                # Load a YAML configuration file and return it as a dictionary
│   ├── evaluate.py                                  # Load a saved pipeline and evaluate it on the test split
│   ├── predict.py                                   # Load a saved pipeline and print predictions for new data
│   ├── preprocess.py                                # Load raw data, clean it, encode categoricals, and save processed data
│   └── train.py                                     # Load processed data, split, train a pipeline, save it, and log to MLflow
├── tests/
│   ├── test_config.py                               # Tests for the config loader
│   ├── test_preprocess.py                           # Minimal tests for the preprocessing module
│   └── test_train.py                                # Minimal tests for the training module
├── .gitignore                                       # Files Git should not track
├── Makefile                                         # Short commands for every task (`make <target>`)
├── pytest.ini                                       # pytest settings
├── README.md                                        # This file
└── requirements.txt                                 # Python dependencies
```

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` and the notebook are committed with the code.

## Make targets

| Target | What it does |
|---|---|
| `make venv` | Creates the project's virtual environment in .venv/ and installs every package listed in requirements.txt into it. |
| `make preprocess` | Cleans the raw CSV and writes the processed data to data/processed/. |
| `make train` | Trains the model on the processed data and saves it in models/. |
| `make evaluate` | Loads the saved model and prints its scores on the test split. |
| `make predict` | Loads the saved model and makes predictions for new customers. |
| `make test` | Runs the automated tests in tests/ with pytest (settings in pytest.ini). |
| `make mlflow-ui` | Opens the MLflow web interface so you can browse and compare runs. |
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

The dataset is IBM sample data distributed on Kaggle by user blastchar. The proof-of-concept notebook is by Bharti Prasad (Kaggle user bhartiprasad17); it is the starting point that this project refactors, not original course work.
