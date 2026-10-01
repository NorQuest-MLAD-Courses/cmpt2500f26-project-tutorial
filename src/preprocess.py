"""
preprocess.py — Load raw data, clean it, encode categoricals, and save processed data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_and_clean(path):
    """Load CSV and perform basic cleaning."""
    df = pd.read_csv(path)

    # Drop the customer ID column — it is not a feature
    df = df.drop(columns=["customerID"])

    # TotalCharges is stored as string in the raw data; convert to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Rows with tenure == 0 have missing TotalCharges; drop them
    df = df[df["tenure"] != 0].copy()

    # Fill any remaining NaN in TotalCharges with the column mean
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

    # SeniorCitizen is 0/1 in raw data — convert to string for uniform encoding
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    return df


def encode_categoricals(df):
    """Label-encode every object column."""
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])
    return df


def main():
    raw_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    out_path = "data/processed/churn_processed.csv"

    print(f"Loading raw data from {raw_path} ...")
    df = load_and_clean(raw_path)

    print("Encoding categorical features ...")
    df = encode_categoricals(df)

    print(f"Saving processed data to {out_path} ...")
    df.to_csv(out_path, index=False)

    print(f"Done. {len(df)} rows written.")


if __name__ == "__main__":
    main()
