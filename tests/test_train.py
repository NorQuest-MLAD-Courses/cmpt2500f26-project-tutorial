"""Minimal tests for the training module.

This file checks that ``build_pipeline`` in ``src/train.py`` returns a
scikit-learn pipeline that can be fitted and used for prediction. Run it from
the project root with::

    pytest

The test uses small random (synthetic) data instead of the real processed
CSV. It is checking that the pipeline is wired together correctly (the
scaler, the pass-through columns and the classifier), not how accurate the
model is, so random numbers with the right column names are enough. This also
keeps the test fast and lets it run on a machine that has no data files.
"""

import numpy as np
import pandas as pd
import sys, os

# The module under test lives in src/, which is not on Python's import path
# by default. Prepending it to sys.path lets us write `from train import ...`.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train import build_pipeline


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
