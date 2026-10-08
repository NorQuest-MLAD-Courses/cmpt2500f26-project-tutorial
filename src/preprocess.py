"""
preprocess.py — Load raw data, clean it, encode categoricals, and save processed data.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from utils.config import load_config


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
    parser.add_argument("--config", default="config/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--input", default=None, help="Override raw data path")
    parser.add_argument("--output", default=None, help="Override output path")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    raw_path = args.input or cfg["paths"]["raw_data"]
    out_path = args.output or cfg["paths"]["processed_data"]

    print(f"Loading raw data from {raw_path} ...")
    df = load_and_clean(raw_path)

    print("Encoding categorical features ...")
    df = encode_categoricals(df)

    print(f"Saving processed data to {out_path} ...")
    df.to_csv(out_path, index=False)

    print(f"Done. {len(df)} rows written.")


if __name__ == "__main__":
    main()
