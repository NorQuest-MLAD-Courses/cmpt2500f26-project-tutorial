"""
train.py — Load processed data, split, train a pipeline, and save it to disk.
"""

import argparse
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def build_pipeline(num_cols, random_state):
    """Construct a pipeline: scale numerics, pass-through the rest, then classify."""
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), num_cols)],
        remainder="passthrough",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(random_state=random_state)),
    ])


def parse_args():
    parser = argparse.ArgumentParser(description="Train a churn-prediction pipeline.")
    parser.add_argument("--data", default="data/processed/churn_processed.csv",
                        help="Path to processed CSV")
    parser.add_argument("--model-out", default="models/model.pkl",
                        help="Path to save the trained pipeline")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=40)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading data from {args.data} ...")
    df = pd.read_csv(args.data)

    X = df.drop(columns=["Churn"])
    y = df["Churn"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    print("Training pipeline ...")
    pipeline = build_pipeline(NUM_COLS, args.random_state)
    pipeline.fit(X_train, y_train)

    yhat = pipeline.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, yhat):.4f}")
    print(classification_report(y_test, yhat))

    with open(args.model_out, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Pipeline saved to {args.model_out}")


if __name__ == "__main__":
    main()
