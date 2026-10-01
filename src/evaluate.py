"""
evaluate.py — Load a saved pipeline and evaluate it on the test split.
"""

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


PROCESSED_DATA = "data/processed/churn_processed.csv"
MODEL_PATH = "models/model.pkl"
TEST_SIZE = 0.30
RANDOM_STATE = 40


def main():
    # Load data and reproduce the same split
    print(f"Loading data from {PROCESSED_DATA} ...")
    df = pd.read_csv(PROCESSED_DATA)
    X = df.drop(columns=["Churn"])
    y = df["Churn"].values

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Load pipeline (scaling is built in)
    print(f"Loading pipeline from {MODEL_PATH} ...")
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)

    # Predict and report
    yhat = pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, yhat):.4f}")
    print()
    print(classification_report(y_test, yhat))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, yhat))


if __name__ == "__main__":
    main()
