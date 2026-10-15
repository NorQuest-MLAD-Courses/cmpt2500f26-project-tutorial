"""Minimal tests for the preprocessing module.

These are unit tests for ``load_and_clean`` and ``encode_categoricals`` in
``src/preprocess.py``. Run them from the project root with::

    pytest

pytest finds this file because its name starts with ``test_`` and it sits in
the ``tests/`` folder named in ``pytest.ini``. Every function whose name starts
with ``test_`` is run as a separate test; a test passes if none of its
``assert`` statements fail.

The tests never touch the real Telco CSV. Each one writes a tiny synthetic CSV
(three customers) into a temporary folder. Synthetic data keeps the tests
fast, makes them independent of files that may not be present (the raw data
is not stored in Git), and lets us include the exact edge cases we want to
check, such as a customer with ``tenure == 0`` and a blank ``TotalCharges``.
"""

import pandas as pd
import sys, os

# Allow imports from src/.
# The modules under test live in src/, which is not on Python's import path
# by default. Prepending it to sys.path lets us write `from preprocess import ...`
# below, exactly as the scripts in src/ import each other.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import load_and_clean, encode_categoricals


def _make_sample_csv(tmp_path):
    """Create a tiny CSV that mimics the raw data format.

    The leading underscore marks this as a helper, so pytest does not collect
    it as a test. ``tmp_path`` is passed in from the calling test (see below).
    Returns the path of the written CSV as a string.
    """
    # Three rows with the same columns as the raw Telco file. The values are
    # chosen to exercise the cleaning rules:
    #   - row 2 has tenure 0 and a blank TotalCharges (" "), as in the real data,
    #     so load_and_clean should drop it;
    #   - TotalCharges is stored as text, as in the raw CSV, so the cleaning
    #     step must convert it to numbers.
    data = {
        "customerID": ["0001", "0002", "0003"],
        "gender": ["Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0],
        "tenure": [1, 0, 24],
        "MonthlyCharges": [29.85, 56.95, 53.85],
        "TotalCharges": ["29.85", " ", "1370.4"],
        "Churn": ["No", "Yes", "No"],
        "Partner": ["Yes", "No", "No"],
        "Dependents": ["No", "No", "Yes"],
        "PhoneService": ["No", "Yes", "Yes"],
        "MultipleLines": ["No phone service", "No", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "DSL"],
        "OnlineSecurity": ["No", "No", "Yes"],
        "OnlineBackup": ["Yes", "No", "No"],
        "DeviceProtection": ["No", "Yes", "Yes"],
        "TechSupport": ["No", "No", "Yes"],
        "StreamingTV": ["No", "No", "No"],
        "StreamingMovies": ["No", "No", "Yes"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "Yes", "No"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
        ],
    }
    # Write the CSV inside the temporary folder; pytest deletes old temporary
    # folders automatically, so nothing is left behind in the project.
    path = tmp_path / "sample.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
# Each test takes a `tmp_path` argument. tmp_path is a built-in pytest
# *fixture*: when pytest sees a test parameter with that name, it creates a
# fresh, empty temporary directory (a pathlib.Path) for that test and passes
# it in. Each test therefore gets its own clean folder to write files into.


def test_load_and_clean_drops_customer_id(tmp_path):
    """customerID is an identifier, not a feature, so cleaning must remove it."""
    path = _make_sample_csv(tmp_path)
    df = load_and_clean(path)
    assert "customerID" not in df.columns


def test_load_and_clean_drops_zero_tenure(tmp_path):
    """Customers with tenure 0 (no TotalCharges yet) must be dropped."""
    path = _make_sample_csv(tmp_path)
    df = load_and_clean(path)
    # .all() is True only if every remaining row has a non-zero tenure.
    assert (df["tenure"] != 0).all()


def test_load_and_clean_total_charges_numeric(tmp_path):
    """TotalCharges must end up as a numeric column with no missing values."""
    path = _make_sample_csv(tmp_path)
    df = load_and_clean(path)
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    assert not df["TotalCharges"].isna().any()


def test_encode_categoricals_no_objects(tmp_path):
    """After encoding, no text (object or string) columns may remain.

    scikit-learn models need numeric input, so any text column left over would
    make training fail.
    """
    path = _make_sample_csv(tmp_path)
    df = load_and_clean(path)
    df = encode_categoricals(df)
    # select_dtypes keeps only the text columns; the result should have none.
    assert df.select_dtypes(include=["object", "string"]).empty
