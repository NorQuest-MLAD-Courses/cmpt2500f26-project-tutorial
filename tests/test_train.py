"""Minimal tests for the training module."""

import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train import build_pipeline


def test_pipeline_fit_predict():
    """The pipeline should fit and predict on a trivial dataset."""
    rng = np.random.RandomState(42)
    n = 100
    X = pd.DataFrame({
        "tenure": rng.randint(1, 72, n).astype(float),
        "MonthlyCharges": rng.uniform(20, 100, n),
        "TotalCharges": rng.uniform(100, 5000, n),
        "SeniorCitizen": rng.randint(0, 2, n),
        "gender": rng.randint(0, 2, n),
    })
    y = rng.randint(0, 2, n)

    model_params = {"random_state": 42}
    pipeline = build_pipeline(["tenure", "MonthlyCharges", "TotalCharges"], model_params)
    pipeline.fit(X, y)
    yhat = pipeline.predict(X)

    assert len(yhat) == n
    assert set(yhat).issubset({0, 1})
