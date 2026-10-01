"""
predict.py — Load a saved pipeline and print predictions for new data.
"""

import pickle
import pandas as pd


MODEL_PATH = "models/model.pkl"
DATA_PATH = "data/processed/churn_processed.csv"


def main():
    # Load pipeline
    print(f"Loading pipeline from {MODEL_PATH} ...")
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)

    # Load data (in practice this would be new, unseen data)
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    # Predict — the pipeline handles scaling internally
    yhat = pipeline.predict(df)

    # Print predictions to stdout
    print("Predictions:")
    for i, pred in enumerate(yhat):
        print(f"  Row {i}: {'Churn' if pred == 1 else 'No Churn'}")

    print(f"\nTotal: {len(yhat)} predictions ({sum(yhat)} churn, {len(yhat) - sum(yhat)} no churn)")


if __name__ == "__main__":
    main()
