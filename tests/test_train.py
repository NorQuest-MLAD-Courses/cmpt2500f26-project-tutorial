"""Tests for the training module.

This file checks these things in ``src/train.py``:

* ``build_pipeline`` returns a scikit-learn pipeline that can be fitted and
  used for prediction;
* ``parse_args`` leaves the split settings to the config by default and
  accepts overrides on the command line;
* ``main`` trains on a small fake dataset, saves the pipeline and logs one
  run to MLflow;
* the ``MLFLOW_TRACKING_URI`` environment variable, when set, takes the
  place of the tracking URI in the config.

Run it from the project root with::

    pytest

The tests use small random (synthetic) data instead of the real processed
CSV. They check that the code is wired together correctly, not how accurate
the model is, so random numbers with the right column names are enough. This
also keeps the tests fast and lets them run on a machine that has no data
files.

The MLflow tests write to a temporary SQLite database that pytest deletes
afterwards, so they never touch your own mlflow.db.
"""

import numpy as np
import pandas as pd
import os
import yaml
from mlflow.tracking import MlflowClient

# `train` lives in src/. pytest.ini sets `pythonpath = . src`, so src/ is on
# the import path while tests run and this plain import works.
from train import build_pipeline, parse_args, main

# The shipped config file, found relative to this test file so the tests do
# not depend on the folder pytest is started from.
DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml")

def test_pipeline_fit_predict():
    """The pipeline should fit and predict on a trivial dataset."""
    # A seeded random generator gives the same "random" data on every run,
    # so the test result is reproducible.
    rng = np.random.RandomState(42)
    n = 100
    # Build 100 fake customers. Three numeric columns match the ones the
    # pipeline scales; SeniorCitizen and gender are 0/1 integers, as they are
    # after label encoding, and pass through the pipeline unscaled.
    X = pd.DataFrame({
        "tenure": rng.randint(1, 72, n).astype(float),
        "MonthlyCharges": rng.uniform(20, 100, n),
        "TotalCharges": rng.uniform(100, 5000, n),
        "SeniorCitizen": rng.randint(0, 2, n),
        "gender": rng.randint(0, 2, n),
    })
    # Random 0/1 labels stand in for the Churn target.
    y = rng.randint(0, 2, n)

    # Tell the pipeline which columns are numeric (to be standardised).
    # build_pipeline takes a dict of classifier settings, the same shape as
    # `model.params` in config/default.yaml, and passes it on with
    # GradientBoostingClassifier(**model_params). Here we only fix the seed.
    model_params = {"random_state": 42}
    pipeline = build_pipeline(["tenure", "MonthlyCharges", "TotalCharges"], model_params)
    pipeline.fit(X, y)
    yhat = pipeline.predict(X)

    # One prediction per input row, and every prediction is a valid class.
    assert len(yhat) == n
    assert set(yhat).issubset({0, 1})


def test_parse_args_defaults(monkeypatch):
    """With no options, parse_args leaves the split settings to the config."""
    # argparse reads the command line from sys.argv. Under pytest, sys.argv
    # holds pytest's own arguments, which parse_args would reject. The
    # built-in `monkeypatch` fixture replaces sys.argv for this test only
    # (as if we had run `python train.py` with no options) and restores the
    # original value automatically when the test ends.
    monkeypatch.setattr("sys.argv", ["train.py"])
    args = parse_args()
    # The defaults are None, which main() reads as "use the value from
    # config/default.yaml". (In Step 4 the defaults were 0.30 and 40; since
    # Step 5 those numbers live in the config file instead.)
    assert args.config == "config/default.yaml"
    assert args.test_size is None
    assert args.random_state is None
    # No --experiment-name either: the experiment name comes from the config.
    assert args.experiment_name is None


def test_parse_args_overrides(monkeypatch):
    """Options given on the command line override the config values."""
    # The same as typing:
    #   python train.py --test-size 0.25 --random-state 7 --experiment-name feature-test
    monkeypatch.setattr("sys.argv", ["train.py", "--test-size", "0.25",
                                     "--random-state", "7", "--experiment-name", "feature-test"])
    args = parse_args()
    # argparse turns "--test-size" into the attribute name test_size and
    # converts the text "0.25" to a float because of type=float.
    assert args.test_size == 0.25
    assert args.random_state == 7
    assert args.experiment_name == "feature-test"


def make_processed_csv(path, n=200):
    """Write a small fake processed dataset (all numeric, plus Churn) to path."""
    # The same columns as the pipeline test above, plus the Churn label that
    # main() separates from the features. 200 rows leaves enough of both
    # classes for the stratified train/test split.
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "tenure": rng.randint(1, 72, n).astype(float),
        "MonthlyCharges": rng.uniform(20, 100, n),
        "TotalCharges": rng.uniform(100, 5000, n),
        "SeniorCitizen": rng.randint(0, 2, n),
        "gender": rng.randint(0, 2, n),
        "Churn": rng.randint(0, 2, n),
    })
    df.to_csv(path, index=False)


def write_config(path, tracking_uri):
    """Copy config/default.yaml to path, with a different MLflow tracking URI."""
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["mlflow"]["tracking_uri"] = tracking_uri
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)


def run_train(monkeypatch, tmp_path, config_path):
    """Run train.py's main() on the fake data, with every output in tmp_path."""
    data = tmp_path / "processed.csv"
    make_processed_csv(data)
    monkeypatch.setattr("sys.argv", [
        "train.py", "--config", str(config_path), "--data", str(data),
        "--model-out", str(tmp_path / "model.pkl"),
        "--experiment-name", "pytest-run",
    ])
    # MLflow stores model files under ./mlruns of the current folder, so we
    # run from the temporary folder to keep the project folder clean.
    monkeypatch.chdir(tmp_path)
    main()



def test_main_logs_run_to_mlflow(monkeypatch, tmp_path):
    """main() trains, saves the pipeline and logs one MLflow run."""
    # A throwaway SQLite tracking store in pytest's temporary folder
    # (tmp_path is a built-in fixture: a fresh empty folder for each test).
    # tmp_path is an absolute path, so the URI has four slashes.
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    config_path = tmp_path / "config.yaml"
    write_config(config_path, tracking_uri)
    # This test is about the config file, so make sure an MLFLOW_TRACKING_URI
    # left in your shell (see Step 14) does not send the run somewhere else.
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    run_train(monkeypatch, tmp_path, config_path)

    # The pipeline was saved, and exactly one run was logged, with both
    # metrics.
    assert (tmp_path / "model.pkl").exists()
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("pytest-run")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert "accuracy" in runs[0].data.metrics
    assert "f1_score" in runs[0].data.metrics


def count_runs(tracking_uri):
    """Number of runs in the pytest-run experiment of a tracking store."""
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("pytest-run")
    if experiment is None:
        return 0
    return len(client.search_runs([experiment.experiment_id]))


def test_tracking_uri_env_overrides_config(monkeypatch, tmp_path):
    """MLFLOW_TRACKING_URI, when set, wins over the tracking URI in the config."""
    # Two separate stores: one named in the config file, one in the
    # environment variable (as docker-compose.yml sets it in the api
    # container). The run must land in the second one.
    config_uri = f"sqlite:///{tmp_path / 'from_config.db'}"
    env_uri = f"sqlite:///{tmp_path / 'from_env.db'}"
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config_uri)
    # monkeypatch.setenv sets the variable for this test only.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", env_uri)

    run_train(monkeypatch, tmp_path, config_path)

    assert count_runs(env_uri) == 1
    # The config's store was never used, so its database file was never
    # created.
    assert not (tmp_path / "from_config.db").exists()
