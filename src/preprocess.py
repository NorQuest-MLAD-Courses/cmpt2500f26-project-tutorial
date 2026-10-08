"""
preprocess.py — Load raw data, clean it, encode categoricals, and save processed data.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_and_clean(path):
    """Load CSV and perform basic cleaning."""
    df = pd.read_csv(path)

    df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = df[df["tenure"] != 0].copy()

    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    return df


def encode_categoricals(df):
    """Label-encode every object column."""
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess raw churn data.")
    parser.add_argument("--input", default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
                        help="Path to raw CSV")
    parser.add_argument("--output", default="data/processed/churn_processed.csv",
                        help="Path to save processed CSV")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading raw data from {args.input} ...")
    df = load_and_clean(args.input)

    print("Encoding categorical features ...")
    df = encode_categoricals(df)

    print(f"Saving processed data to {args.output} ...")
    df.to_csv(args.output, index=False)

    print(f"Done. {len(df)} rows written.")


if __name__ == "__main__":
    main()
