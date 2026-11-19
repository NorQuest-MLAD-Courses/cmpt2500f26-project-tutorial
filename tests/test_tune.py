"""Tests for the hyperparameter-tuning module.

This file checks these things in ``src/tune.py``:

* ``build_deployment_pipeline`` builds a pipeline that takes the raw text
  values the API receives (for example "One year") and can score a value it
  never saw during training;
* ``parse_args`` logs to the churn-tuning experiment by default.
* ``feature_importances`` maps every input column to a non-negative
  importance, for both tree models and logistic regression.

Run it from the project root with::

    pytest

Like the training tests, these use small random data with the right column
names, so they run in a second and need no data files.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sys, os

# The module under test lives in src/, which is not on Python's import path
# by default. Prepending it to sys.path lets us write `from tune import ...`.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tune import build_deployment_pipeline, parse_args, feature_importances, NUM_COLS, CAT_COLS


def make_raw_data(n=200):
    """Fake customers in the raw format: numbers and text categories."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame({col: rng.uniform(0, 100, n) for col in NUM_COLS})
    # Two made-up text values per categorical column are enough to check
    # that the OrdinalEncoder turns text into numbers inside the pipeline.
    for col in CAT_COLS:
        X[col] = rng.choice(["Yes", "No"], n)
    y = rng.randint(0, 2, n)
    return X, y


def test_deployment_pipeline_accepts_raw_data():
    """The pipeline fits on raw data and scores new rows, unseen values too."""
    X, y = make_raw_data()
    pipeline = build_deployment_pipeline(LogisticRegression(max_iter=1000),
                                         NUM_COLS, CAT_COLS)
    pipeline.fit(X, y)

    yhat = pipeline.predict(X)
    assert len(yhat) == len(X)
    assert set(yhat).issubset({0, 1})

    # A customer with a category value the model never saw in training. The
    # encoder was built with handle_unknown="use_encoded_value", so it maps
    # the value to -1 instead of raising an error. The API depends on this.
    new_row = X.iloc[[0]].copy()
    new_row[CAT_COLS[0]] = "a value never seen"
    probability = pipeline.predict_proba(new_row)[0, 1]
    assert 0.0 <= probability <= 1.0


def test_parse_args_defaults(monkeypatch):
    """With no options, tuning runs log to the churn-tuning experiment."""
    # Replace sys.argv as in tests/test_train.py, as if we had run
    # `python src/tune.py` with no options.
    monkeypatch.setattr("sys.argv", ["tune.py"])
    args = parse_args()
    assert args.experiment_name == "churn-tuning"
    # The training-metrics server for Prometheus listens on port 8002.
    assert args.metrics_port == 8002


def test_feature_importances_cover_every_column():
    """feature_importances returns one non-negative value per input column."""
    X, y = make_raw_data()
    # A tree model has feature_importances_; logistic regression has coef_,
    # which the helper turns into absolute values. Check both kinds.
    for classifier in (RandomForestClassifier(n_estimators=10, random_state=0),
                       LogisticRegression(max_iter=1000)):
        pipeline = build_deployment_pipeline(classifier, NUM_COLS, CAT_COLS)
        pipeline.fit(X, y)
        importances = feature_importances(pipeline)
        assert set(importances) == set(NUM_COLS + CAT_COLS)
        assert all(value >= 0 for value in importances.values())
