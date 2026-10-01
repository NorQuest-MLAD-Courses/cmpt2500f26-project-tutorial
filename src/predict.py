"""
predict.py — Load a saved model and print predictions for new data.
"""

import pickle
import pandas as pd


MODEL_PATH = "models/model.pkl"
DATA_PATH = "data/processed/churn_processed.csv"
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def main():
    # Load model and scaler
    print(f"Loading model from {MODEL_PATH} ...")
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    scaler = artifact["scaler"]

    # Load data (in practice this would be new, unseen data)
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    # Drop target if present
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    # Scale numerical columns
    df[NUM_COLS] = scaler.transform(df[NUM_COLS])

    # Predict
    yhat = model.predict(df)

    # Print predictions to stdout
    print("Predictions:")
    for i, pred in enumerate(yhat):
        print(f"  Row {i}: {'Churn' if pred == 1 else 'No Churn'}")

    print(f"\nTotal: {len(yhat)} predictions ({sum(yhat)} churn, {len(yhat) - sum(yhat)} no churn)")


if __name__ == "__main__":
    main()
