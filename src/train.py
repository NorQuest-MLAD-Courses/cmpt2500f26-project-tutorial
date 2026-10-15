"""
train.py — Load processed data, split, train a pipeline, and save it to disk.

The pipeline bundles scaling and the classifier into a single object,
so predict.py and evaluate.py no longer need to manage the scaler separately.

This is the second stage of the pipeline. It reads the numeric table written
by preprocess.py, trains a scikit-learn Pipeline and saves it to a file, so
that evaluate.py and predict.py can load it without retraining.

How to run (from the project root, after make preprocess):
    make train
    python src/train.py
    python src/train.py --config config/experiment.yaml
    python src/train.py --test-size 0.2 --random-state 42 --model-out models/m2.pkl

Settings come from config/default.yaml (or the file given with --config):
paths, the numeric feature list, the train/test split and the model
hyperparameters. Command-line flags override single values for one run.

Input:  the processed CSV (paths.processed_data, or --data)
Output: the whole pickled Pipeline (paths.model, or --model-out),
        plus test accuracy and a classification report, printed
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

# Our YAML loader from src/utils/config.py (importable because src/ is the
# script's folder and so is on Python's import path).
from utils.config import load_config


def build_pipeline(num_cols, model_params):
    """Construct a pipeline: scale numerics, pass-through the rest, then classify.

    Args:
        num_cols: names of the numeric columns to standardise.
        model_params: dict of classifier hyperparameters, from the config's
            model.params section.

    Returns:
        An unfitted scikit-learn Pipeline (preprocessor, then classifier).
    """
    # A ColumnTransformer applies different steps to different columns.
    # Each transformer is (name, transformer, columns): here only the numeric
    # columns go through StandardScaler. Without remainder="passthrough" the
    # default is remainder="drop", and every other column would be silently
    # thrown away. Note that the output puts the scaled columns first and the
    # passed-through columns after them.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
        ],
        remainder="passthrough",  # leave non-numeric columns untouched
    )
    # A Pipeline chains the steps under names. fit() fits each step in order
    # and predict() runs new data through the same steps, so the scaling at
    # prediction time always matches the scaling used in training.
    return Pipeline([
        ("preprocessor", preprocessor),
        # **model_params unpacks the dict into keyword arguments, so
        # {"random_state": 40} becomes GradientBoostingClassifier(random_state=40).
        # Any hyperparameter added to the config reaches the model this way.
        ("classifier", GradientBoostingClassifier(**model_params)),
    ])


def parse_args():
    """Read command-line options.

    Returns:
        An argparse.Namespace with .config, .data, .model_out, .test_size and
        .random_state.
    """
    # Every override flag defaults to None, meaning "not given on the command
    # line"; main() then takes the value from the config file.
    # type=float/int converts the text typed on the command line into a
    # number. argparse turns "--model-out" into the attribute args.model_out.
    # evaluate.py accepts the same two split flags. If you train with, say,
    # --test-size 0.2, evaluate with --test-size 0.2 as well, so it scores the
    # model on the rows it never saw.
    parser = argparse.ArgumentParser(description="Train a churn-prediction pipeline.")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--data", default=None, help="Override processed data path")
    parser.add_argument("--model-out", default=None, help="Override model output path")
    parser.add_argument("--test-size", type=float, default=None,
                        help="Override the test-set fraction from the config")
    parser.add_argument("--random-state", type=int, default=None,
                        help="Override the random seed from the config")
    return parser.parse_args()


def main():
    """Train the pipeline on the training split, report test scores and save it."""
    args = parse_args()
    cfg = load_config(args.config)

    # ---- Resolve settings: command-line flag > config file ----
    # For paths, `a or b` is enough: it returns b when a is None.
    # For numbers we test `is not None` instead, because `or` treats 0 (and
    # 0.0) as false: "--random-state 0" would be ignored and the config value
    # used. `is not None` only falls back when the flag was not given at all.
    data_path = args.data or cfg["paths"]["processed_data"]
    model_path = args.model_out or cfg["paths"]["model"]
    num_cols = cfg["features"]["numerical"]
    test_size = args.test_size if args.test_size is not None else cfg["training"]["test_size"]
    random_state = args.random_state if args.random_state is not None else cfg["training"]["random_state"]
    model_params = cfg["model"]["params"]

    # Load processed data
    print(f"Loading data from {data_path} ...")
    df = pd.read_csv(data_path)

    # Separate features and target.
    # X holds the input columns; y holds the label (Churn: 0 = no, 1 = yes).
    X = df.drop(columns=["Churn"])
    y = df["Churn"].values

    # Train / test split (stratified to preserve class balance).
    # The test fraction and the seed come from the config (or the flags).
    # The same seed gives the same split every run.
    # stratify=y keeps the share of churners the same in both parts; without
    # it, one part could by chance get noticeably more churners than the other.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Build and train pipeline.
    # fit() runs the steps in order on the training data: the scaler learns
    # the training mean and standard deviation and scales X_train, then the
    # classifier is fitted on the scaled result.
    print("Training pipeline ...")
    pipeline = build_pipeline(num_cols, model_params)
    pipeline.fit(X_train, y_train)

    # Quick evaluation on test set (yhat = predicted labels).
    # predict() scales X_test with the training statistics stored inside the
    # pipeline, then classifies. The test set is never used for fitting.
    yhat = pipeline.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, yhat):.4f}")
    print(classification_report(y_test, yhat))

    # Save the entire pipeline.
    # One object now holds both the fitted scaler and the model, so there is
    # no separate scaler to keep in step. pickle serialises Python objects to
    # bytes; "wb" opens the file for writing bytes.
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Pipeline saved to {model_path}")


# Run main() only when executed as a script, not when imported.
if __name__ == "__main__":
    main()
