"""
evaluate.py — Load a saved pipeline and evaluate it on the test split.
"""

import argparse
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved churn pipeline.")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--data", default=None, help="Override processed data path")
    parser.add_argument("--model", default=None, help="Override model path")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    data_path = args.data or cfg["paths"]["processed_data"]
    model_path = args.model or cfg["paths"]["model"]
    test_size = cfg["training"]["test_size"]
    random_state = cfg["training"]["random_state"]

    print(f"Loading data from {data_path} ...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"].values

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Loading pipeline from {model_path} ...")
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    yhat = pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, yhat):.4f}")
    print()
    print(classification_report(y_test, yhat))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, yhat))


if __name__ == "__main__":
    main()
