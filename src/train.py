"""
train.py — Load processed data, split, train a pipeline, save it, and log to MLflow.

The pipeline bundles scaling and the classifier into a single object,
so predict.py and evaluate.py do not need to manage the scaler separately.

This is the second stage of the pipeline. It reads the numeric table written
by preprocess.py, trains a scikit-learn Pipeline and saves it to a file, so
that evaluate.py and predict.py can load it without retraining. Each run is
also recorded in MLflow: its parameters, its test metrics and the model
itself, so runs can be compared later in the MLflow UI (make mlflow-ui).

How to run (from the project root, after make preprocess):
    make train
    python src/train.py
    python src/train.py --config config/experiment.yaml
    python src/train.py --test-size 0.2 --random-state 42 --model-out models/m2.pkl
    python src/train.py --experiment-name feature-test

Settings come from config/default.yaml (or the file given with --config):
paths, the numeric feature list, the train/test split, the model
hyperparameters and the MLflow tracking URI and experiment name.
Command-line flags override single values for one run.

Input:  the processed CSV (paths.processed_data, or --data)
Output: the whole pickled Pipeline (paths.model, or --model-out),
        an MLflow run (parameters, metrics and the logged model) in the
        tracking store (sqlite:///mlflow.db by default),
        plus test accuracy, F1 and a classification report, printed
"""

import argparse
import pickle
import pandas as pd
import mlflow
# skops is the safe serialisation format MLflow 3 uses for scikit-learn
# models; we use two of its helpers when logging the model below.
from skops.io import dumps, get_untrusted_types
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

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
        An argparse.Namespace with .config, .data, .model_out, .test_size,
        .random_state and .experiment_name.
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
    parser.add_argument("--experiment-name", default=None,
                        help="Override the MLflow experiment name from the config")
    return parser.parse_args()


def main():
    """Train the pipeline, report test scores, save it and log the run to MLflow."""
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
    # --experiment-name lets you keep a batch of runs apart from the main
    # history without editing the config.
    experiment_name = args.experiment_name or cfg["mlflow"]["experiment_name"]

    # --- MLflow setup ---
    # The tracking URI says where MLflow stores run data. sqlite:///mlflow.db
    # is a local database file in the project root; model files go under
    # mlruns/. set_experiment() selects the experiment (a named group of
    # runs) and creates it if it does not exist yet. Both calls must come
    # before start_run(), or the run would go to the default location.
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(experiment_name)

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

    # Everything inside this "with" block is recorded as one MLflow run.
    # The run is closed automatically when the block ends, even on an error.
    with mlflow.start_run():
        # Log parameters: the settings we chose before training. They show
        # up as columns in the MLflow UI, so runs can be filtered and sorted.
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("model", cfg["model"]["name"])
        # One parameter per hyperparameter, prefixed "model_" so they do not
        # clash with the names above (for example model_random_state).
        for k, v in model_params.items():
            mlflow.log_param(f"model_{k}", v)

        # Train.
        # fit() runs the steps in order on the training data: the scaler
        # learns the training mean and standard deviation and scales X_train,
        # then the classifier is fitted on the scaled result.
        print("Training pipeline ...")
        pipeline = build_pipeline(num_cols, model_params)
        pipeline.fit(X_train, y_train)

        # Evaluate on the test set (yhat = predicted labels).
        # F1 balances precision and recall for the churn class. It is more
        # informative than accuracy here, because churners are the minority
        # and a model that rarely predicts churn can still score high
        # accuracy.
        yhat = pipeline.predict(X_test)
        acc = accuracy_score(y_test, yhat)
        f1 = f1_score(y_test, yhat)
        print(f"Test accuracy: {acc:.4f}")
        print(f"Test F1:       {f1:.4f}")
        print(classification_report(y_test, yhat))

        # Log metrics: the results we measured after training.
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Log the model. MLflow saves scikit-learn models in the safe "skops"
        # format, which only loads object types it has been told to trust.
        # We trained this pipeline ourselves, so we trust every type in it.
        # dumps() serialises the pipeline in memory, and get_untrusted_types()
        # lists the types skops would refuse (for example the internal Tree
        # object inside gradient boosting). Without skops_trusted_types the
        # log_model call fails with an "untrusted types" error.
        # name="model" is the model's name inside the run, so it can be loaded
        # later as "runs:/<run-id>/model". (MLflow 3 uses name=; the older
        # positional artifact_path still works but prints a warning.)
        trusted_types = get_untrusted_types(data=dumps(pipeline))
        mlflow.sklearn.log_model(
            pipeline, name="model", skops_trusted_types=trusted_types
        )

        # Also save locally.
        # evaluate.py and predict.py still load models/model.pkl with pickle,
        # so we keep writing it next to the MLflow copy. Unlike skops, pickle
        # loads any object type, so only open pickle files you trust.
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)
        print(f"Pipeline saved to {model_path}")
        print(f"MLflow run logged to {mlflow.get_tracking_uri()} "
              f"(experiment: {experiment_name})")


# Run main() only when executed as a script, not when imported.
if __name__ == "__main__":
    main()
