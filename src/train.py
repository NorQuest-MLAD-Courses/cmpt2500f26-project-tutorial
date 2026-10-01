"""
train.py — Load processed data, split, scale, train a model, and save it to disk.
"""

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


# --- Settings (hard-coded for now) ---
PROCESSED_DATA = "data/processed/churn_processed.csv"
MODEL_PATH = "models/model.pkl"
TEST_SIZE = 0.30
RANDOM_STATE = 40
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def main():
    # Load processed data
    print(f"Loading data from {PROCESSED_DATA} ...")
    df = pd.read_csv(PROCESSED_DATA)

    # Separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"].values

    # Train / test split (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Scale numerical columns
    scaler = StandardScaler()
    X_train[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
    X_test[NUM_COLS] = scaler.transform(X_test[NUM_COLS])

    # Train model
    print("Training GradientBoostingClassifier ...")
    model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # Quick evaluation on test set
    yhat = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, yhat):.4f}")
    print(classification_report(y_test, yhat))

    # Save model and scaler together
    artifact = {"model": model, "scaler": scaler}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
