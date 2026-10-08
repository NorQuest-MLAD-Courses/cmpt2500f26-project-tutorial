"""
predict.py — Load a saved pipeline and print predictions for new data.
"""

import argparse
import pickle
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Predict churn using a saved pipeline.")
    parser.add_argument("--data", default="data/processed/churn_processed.csv",
                        help="Path to input CSV (features only, or with Churn column)")
    parser.add_argument("--model", default="models/model.pkl",
                        help="Path to saved pipeline")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading pipeline from {args.model} ...")
    with open(args.model, "rb") as f:
        pipeline = pickle.load(f)

    print(f"Loading data from {args.data} ...")
    df = pd.read_csv(args.data)

    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    yhat = pipeline.predict(df)

    print("Predictions:")
    for i, pred in enumerate(yhat):
        print(f"  Row {i}: {'Churn' if pred == 1 else 'No Churn'}")

    print(f"\nTotal: {len(yhat)} predictions ({sum(yhat)} churn, {len(yhat) - sum(yhat)} no churn)")


if __name__ == "__main__":
    main()
