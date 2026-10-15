"""
predict.py — Load a saved pipeline and print predictions for new data.

This stage shows how the trained model is used once it exists: load it from
disk, give it customer rows, and get a churn / no churn label for each one.
It runs on its own, without train.py having to run in the same session,
because the pipeline is read from the file that train.py saved.

How to run (from the project root, after make preprocess and make train):
    make predict
    python src/predict.py
    python src/predict.py --config config/default.yaml --data my_customers.csv

Settings come from config/default.yaml (paths.model, paths.processed_data).
--model and --data override them for a single run.

Input:  the pickled pipeline and a CSV of encoded customer rows
Output: one prediction per row plus a summary count, printed
"""

import argparse
import pickle
import pandas as pd

# Our YAML loader from src/utils/config.py (importable because src/ is the
# script's folder and so is on Python's import path).
from utils.config import load_config


def parse_args():
    """Read command-line options.

    Returns:
        An argparse.Namespace with .config, .data and .model.
    """
    # --data and --model default to None, meaning "not given"; main() then
    # takes the path from the config file. --data lets you score any CSV
    # with the same columns.
    parser = argparse.ArgumentParser(description="Predict churn using a saved pipeline.")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--data", default=None, help="Override input data path")
    parser.add_argument("--model", default=None, help="Override model path")
    return parser.parse_args()


def main():
    """Load the saved pipeline, predict for every row and print the results."""
    args = parse_args()
    cfg = load_config(args.config)

    # A flag wins if it was given (`a or b` returns b when a is None);
    # otherwise use the config value.
    model_path = args.model or cfg["paths"]["model"]
    data_path = args.data or cfg["paths"]["processed_data"]

    # Load pipeline.
    # The pickle holds the whole scikit-learn Pipeline (scaler + classifier).
    # Only unpickle files you trust: loading a pickle can run arbitrary code.
    print(f"Loading pipeline from {model_path} ...")
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    # Load data (in practice this would be new, unseen data).
    # For the demo we reuse the processed training file, which has the same
    # encoded columns the model expects.
    print(f"Loading data from {data_path} ...")
    df = pd.read_csv(data_path)

    # Drop target if present.
    # The model expects only feature columns. Real new customers would not
    # have a Churn value, but our demo file does.
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    # Predict. The pipeline handles scaling internally: it applies the
    # scaler fitted on the training data, then the classifier. We pass raw
    # (unscaled) features and cannot forget the scaling step.
    yhat = pipeline.predict(df)

    # Print predictions to stdout.
    # Each label is 1 (churn) or 0 (no churn), so sum(yhat) counts churners.
    print("Predictions:")
    for i, pred in enumerate(yhat):
        print(f"  Row {i}: {'Churn' if pred == 1 else 'No Churn'}")

    print(f"\nTotal: {len(yhat)} predictions ({sum(yhat)} churn, {len(yhat) - sum(yhat)} no churn)")


# Run main() only when executed as a script, not when imported.
if __name__ == "__main__":
    main()
