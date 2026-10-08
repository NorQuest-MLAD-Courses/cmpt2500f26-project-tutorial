"""
predict.py — Load a saved pipeline and print predictions for new data.
"""

import argparse
import pickle
import pandas as pd

from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Predict churn using a saved pipeline.")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--data", default=None, help="Override input data path")
    parser.add_argument("--model", default=None, help="Override model path")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_path = args.model or cfg["paths"]["model"]
    data_path = args.data or cfg["paths"]["processed_data"]

    print(f"Loading pipeline from {model_path} ...")
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    print(f"Loading data from {data_path} ...")
    df = pd.read_csv(data_path)

    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    yhat = pipeline.predict(df)

    print("Predictions:")
    for i, pred in enumerate(yhat):
        print(f"  Row {i}: {'Churn' if pred == 1 else 'No Churn'}")

    print(f"\nTotal: {len(yhat)} predictions ({sum(yhat)} churn, {len(yhat) - sum(yhat)} no churn)")


if __name__ == "__main__":
    main()
