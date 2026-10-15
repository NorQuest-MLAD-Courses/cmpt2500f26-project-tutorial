"""
evaluate.py — Load a saved pipeline and evaluate it on the test split.

This stage checks how well the trained model does on data it never saw during
training. It does not get the test set from train.py directly. Instead it
reads the same processed CSV and repeats the same split, which gives exactly
the same test rows because the split settings and random seed match.

How to run (from the project root, after make preprocess and make train):
    make evaluate
    python src/evaluate.py
    python src/evaluate.py --config config/default.yaml --model models/other.pkl

Settings come from config/default.yaml: paths.processed_data, paths.model,
training.test_size and training.random_state. --data and --model override
the two paths for a single run, and --test-size and --random-state override
the split settings (use the same values you trained with).

Input:  the processed CSV and the pickled pipeline (models/model.pkl by default)
Output: accuracy, a classification report and a confusion matrix, printed
"""

import argparse
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Our YAML loader from src/utils/config.py (importable because src/ is the
# script's folder and so is on Python's import path).
from utils.config import load_config


def parse_args():
    """Read command-line options.

    Returns:
        An argparse.Namespace with .config, .data, .model, .test_size and
        .random_state.
    """
    # Every option defaults to None, meaning "not given on the command line";
    # main() then takes the value from the config file. The split flags match
    # the ones in train.py: if you trained with --test-size 0.2, evaluate with
    # --test-size 0.2 too, or the test rows will not be the ones the model
    # never saw.
    parser = argparse.ArgumentParser(description="Evaluate a saved churn pipeline.")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--data", default=None, help="Override processed data path")
    parser.add_argument("--model", default=None, help="Override model path")
    parser.add_argument("--test-size", type=float, default=None,
                        help="Override the test-set fraction from the config")
    parser.add_argument("--random-state", type=int, default=None,
                        help="Override the random seed from the config")
    return parser.parse_args()


def main():
    """Rebuild the test split, load the saved pipeline and print its metrics."""
    args = parse_args()
    cfg = load_config(args.config)

    # A flag wins if it was given; otherwise use the config value.
    data_path = args.data or cfg["paths"]["processed_data"]
    model_path = args.model or cfg["paths"]["model"]
    # Read from the same config section as train.py, so both scripts split
    # the data identically. If they differed, the "test" rows here would
    # include rows the model was trained on and the scores would look better
    # than they really are. `is not None` (rather than `or`) is used for the
    # numeric flags, because a valid value such as 0 would otherwise be ignored.
    test_size = args.test_size if args.test_size is not None else cfg["training"]["test_size"]
    random_state = args.random_state if args.random_state is not None else cfg["training"]["random_state"]

    # Load data and reproduce the same split
    print(f"Loading data from {data_path} ...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"].values

    # We only need the test half, so the training parts are thrown away
    # with "_". stratify=y keeps the churn/no-churn ratio the same in both
    # halves, as in train.py.
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Load pipeline (scaling is built in).
    # The pickle now holds the whole scikit-learn Pipeline: the scaler and the
    # classifier together. pipeline.predict() scales the raw test features
    # with the training statistics first, so no separate scaler step is
    # needed here. Only unpickle files you trust: loading a pickle can run
    # arbitrary code.
    print(f"Loading pipeline from {model_path} ...")
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    # Predict and report
    yhat = pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, yhat):.4f}")
    print()
    # Precision, recall and F1 per class. Churners are the minority class, so
    # these show whether the model actually finds them.
    print(classification_report(y_test, yhat))
    # Rows are the true class (y_test), columns the predicted class (yhat):
    # [[true negatives, false positives], [false negatives, true positives]].
    print("Confusion matrix:")
    print(confusion_matrix(y_test, yhat))


# Run main() only when executed as a script, not when imported.
if __name__ == "__main__":
    main()
