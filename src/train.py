"""
train.py — Load processed data, split, train a pipeline, and save it to disk.

The pipeline bundles scaling and the classifier into a single object,
so predict.py and evaluate.py no longer need to manage the scaler separately.
"""

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


# --- Settings ---
PROCESSED_DATA = "data/processed/churn_processed.csv"
MODEL_PATH = "models/model.pkl"
TEST_SIZE = 0.30
RANDOM_STATE = 40
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def build_pipeline(num_cols):
    """Construct a pipeline: scale numerics, pass-through the rest, then classify."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
        ],
        remainder="passthrough",  # leave non-numeric columns untouched
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ])


def main():
    # Load processed data
    print(f"Loading data from {PROCESSED_DATA} ...")
    df = pd.read_csv(PROCESSED_DATA)

    X = df.drop(columns=["Churn"])
    y = df["Churn"].values

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Build and train pipeline
    print("Training pipeline ...")
    pipeline = build_pipeline(NUM_COLS)
    pipeline.fit(X_train, y_train)

    # Quick evaluation
    yhat = pipeline.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, yhat):.4f}")
    print(classification_report(y_test, yhat))

    # Save the entire pipeline
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Pipeline saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
