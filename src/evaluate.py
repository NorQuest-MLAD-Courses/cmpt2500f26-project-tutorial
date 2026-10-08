"""
evaluate.py — Load a saved pipeline and evaluate it on the test split.
"""

import argparse
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved churn pipeline.")
    parser.add_argument("--data", default="data/processed/churn_processed.csv",
                        help="Path to processed CSV")
    parser.add_argument("--model", default="models/model.pkl",
                        help="Path to saved pipeline")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=40)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading data from {args.data} ...")
    df = pd.read_csv(args.data)
    X = df.drop(columns=["Churn"])
    y = df["Churn"].values

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    print(f"Loading pipeline from {args.model} ...")
    with open(args.model, "rb") as f:
        pipeline = pickle.load(f)

    yhat = pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, yhat):.4f}")
    print()
    print(classification_report(y_test, yhat))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, yhat))


if __name__ == "__main__":
    main()
