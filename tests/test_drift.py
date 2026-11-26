"""Tests for the drift simulation in src/drift.py.

``simulate_drift`` makes a changed copy of the data so that the Evidently
report has some drift to find. These tests check that it changes the three
columns the way drift.py describes and leaves the original data alone. They
do not build the HTML report, which is slow and needs no unit test.

Run it from the project root with::

    pytest
"""

import numpy as np
import pandas as pd

# `drift` lives in src/. pytest.ini sets `pythonpath = . src`, so src/ is on
# the import path while tests run and this plain import works.
from drift import simulate_drift


def make_data(n=500):
    """Fake cleaned customers with the three columns simulate_drift changes."""
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "tenure": rng.randint(1, 72, n),
        "MonthlyCharges": rng.uniform(20, 100, n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
    })


def test_simulate_drift_shifts_the_data():
    """Charges go up, tenure goes down and more customers are month-to-month."""
    df = make_data()
    drifted = simulate_drift(df)

    assert drifted.shape == df.shape
    assert drifted["MonthlyCharges"].mean() > df["MonthlyCharges"].mean()
    assert drifted["tenure"].mean() < df["tenure"].mean()
    # tenure is clipped so that no customer drops below one month.
    assert drifted["tenure"].min() >= 1
    share = lambda d: (d["Contract"] == "Month-to-month").mean()
    assert share(drifted) > share(df)


def test_simulate_drift_is_reproducible_and_copies():
    """The same seed gives the same result, and the input is not modified."""
    df = make_data()
    original = df.copy()
    first = simulate_drift(df, random_state=1)
    second = simulate_drift(df, random_state=1)
    # assert_frame_equal fails with a readable message if the frames differ.
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(df, original)
