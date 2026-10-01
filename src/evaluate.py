"""
evaluate.py — Load a saved model and evaluate it on the test split.
"""

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


PROCESSED_DATA = "data/processed/churn_processed.csv"
MODEL_PATH = "models/model.pkl"
TEST_SIZE = 0.30
RANDOM_STATE = 40
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def main():
    # Load data and reproduce the same split
    print(f"Loading data from {PROCESSED_DATA} ...")
    df = pd.read_csv(PROCESSED_DATA)
    X = df.drop(columns=["Churn"])
    y = df["Churn"].values

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Load saved model and scaler
    print(f"Loading model from {MODEL_PATH} ...")
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    scaler = artifact["scaler"]

    # Scale test features
    X_test[NUM_COLS] = scaler.transform(X_test[NUM_COLS])

    # Predict and report
    yhat = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, yhat):.4f}")
    print()
    print(classification_report(y_test, yhat))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, yhat))


if __name__ == "__main__":
    main()
