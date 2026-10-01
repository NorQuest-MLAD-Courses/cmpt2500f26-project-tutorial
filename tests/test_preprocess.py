"""Minimal tests for the preprocessing module."""

import pandas as pd
import numpy as np
import sys, os

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import load_and_clean, encode_categoricals


def _make_sample_csv(tmp_path):
    """Create a tiny CSV that mimics the raw data format."""
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
    path = tmp_path / "sample.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    return str(path)


def test_load_and_clean_drops_customer_id(tmp_path):
    path = _make_sample_csv(tmp_path)
    df = load_and_clean(path)
    assert "customerID" not in df.columns


def test_load_and_clean_drops_zero_tenure(tmp_path):
    path = _make_sample_csv(tmp_path)
    df = load_and_clean(path)
    assert (df["tenure"] != 0).all()


def test_load_and_clean_total_charges_numeric(tmp_path):
    path = _make_sample_csv(tmp_path)
    df = load_and_clean(path)
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    assert not df["TotalCharges"].isna().any()


def test_encode_categoricals_no_objects(tmp_path):
    path = _make_sample_csv(tmp_path)
    df = load_and_clean(path)
    df = encode_categoricals(df)
    assert df.select_dtypes(include="object").empty
