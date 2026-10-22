"""
tune.py — Hyperparameter search across model families.
Produces deployment-ready pipelines that handle full preprocessing internally.

The pipelines from train.py expect data that preprocess.py has already
label-encoded, and that encoder was never saved. An API that receives raw
customer data (strings such as "Female", "One year", "DSL") could not use
them. The pipelines built here put an OrdinalEncoder for the categorical
columns inside the pipeline, next to the StandardScaler for the numeric ones,
so each saved file accepts cleaned but not encoded data and does all the
transformation itself.

The script tries every combination in a small grid for three model families,
logs each one as an MLflow run, ranks them by F1 score and saves the best two
as the models for API versions v1 and v2.

How to run (from the project root; takes a few minutes):
    make tune
    python src/tune.py
    python src/tune.py --experiment-name tuning-wider-grid

Input:  the raw CSV (paths.raw_data in config/default.yaml), cleaned with
        load_and_clean() from preprocess.py
Output: models/model_v1.pkl (best F1) and models/model_v2.pkl (second best),
        plus one MLflow run per candidate in the "churn-tuning" experiment
"""

import argparse
import pickle
# product() yields every combination of several lists, which turns a grid of
# hyperparameter values into the list of candidates to try.
from itertools import product
import mlflow
from skops.io import dumps, get_untrusted_types
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from utils.config import load_config
# Reuse the cleaning step from preprocess.py so both scripts clean the data in
# the same way. Importing it does not run preprocess.main(), thanks to the
# `if __name__ == "__main__"` guard in that file.
from preprocess import load_and_clean


# --- Column definitions ---
# Every input column is listed in exactly one of these two lists. The order
# matters: the ColumnTransformer outputs the numeric columns first, then the
# categorical ones, in the order given here.
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

CAT_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def build_deployment_pipeline(classifier, num_cols, cat_cols):
    """Build a pipeline that accepts cleaned (not encoded) data.

    Args:
        classifier: an unfitted scikit-learn classifier.
        num_cols: numeric columns to standardise.
        cat_cols: text columns to encode as integers.

    Returns:
        An unfitted Pipeline: ColumnTransformer (scale + encode), then the
        classifier.
    """
    # OrdinalEncoder maps each category to an integer, and it is fitted and
    # saved as part of the pipeline, so prediction uses exactly the mapping
    # learned in training. handle_unknown="use_encoded_value" with
    # unknown_value=-1 means a category never seen in training (for example a
    # new payment method sent to the API) becomes -1 instead of raising an
    # error. No remainder is set, so the default "drop" applies: any column
    # not listed in num_cols or cat_cols is discarded, which also protects the
    # model from unexpected extra columns.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1), cat_cols),
        ]
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


# --- Candidate models and grids ---
# For each model family: the class to build and a grid of hyperparameter
# values. Every combination in a grid is tried, so the number of runs is the
# product of the list lengths: 2*2*2 = 8 for gradient boosting, 3*3 = 9 for
# the random forest and 4*1 = 4 for logistic regression, 21 in total.
# max_depth=None lets random forest trees grow until their leaves are pure.
# C is the inverse regularisation strength (smaller C = simpler model), and
# max_iter=1000 gives the solver enough steps to converge.
CANDIDATES = {
    "GradientBoosting": {
        "class": GradientBoostingClassifier,
        "grid": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
        },
    },
    "RandomForest": {
        "class": RandomForestClassifier,
        "grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
        },
    },
    "LogisticRegression": {
        "class": LogisticRegression,
        "grid": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "max_iter": [1000],
        },
    },
}


def parse_args():
    """Read command-line options.

    Returns:
        An argparse.Namespace with .config and .experiment_name.
    """
    # The search logs to its own "churn-tuning" experiment by default, so its
    # many runs stay separate from the regular train.py runs.
    parser = argparse.ArgumentParser(description="Hyperparameter search.")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--experiment-name", default="churn-tuning",
                        help="MLflow experiment to log the search runs to")
    return parser.parse_args()


def main():
    """Train and log every candidate, then save the two best pipelines."""
    args = parse_args()
    cfg = load_config(args.config)

    # Same tracking store as train.py; set_experiment() creates the
    # experiment on first use. Both calls must come before start_run().
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(args.experiment_name)

    # Load raw data and apply cleaning only (no encoding).
    # We do not read the processed CSV: its columns are already label-encoded,
    # and the pipeline's own OrdinalEncoder needs the original strings.
    raw_path = cfg["paths"]["raw_data"]
    df = load_and_clean(raw_path)

    # Churn is still the text "Yes"/"No" here. Comparing with "Yes" gives
    # True/False, and astype(int) turns that into 1/0.
    X = df.drop(columns=["Churn"])
    y = (df["Churn"] == "Yes").astype(int).values

    # Same test size and seed as train.py, read from the config, so the
    # scores are measured on a comparable hold-out set. stratify=y keeps the
    # share of churners equal in both parts.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["training"]["test_size"],
        random_state=cfg["training"]["random_state"],
        stratify=y,
    )

    # One entry per candidate, used afterwards to rank them.
    results = []

    # ---- Grid search: try every combination for every model family ----
    for model_name, spec in CANDIDATES.items():
        print(f"\n--- {model_name} ---")
        grid = spec["grid"]

        # Generate all combinations.
        # product(*grid.values()) yields tuples such as (100, 3, 0.05); zip
        # pairs them with the parameter names to make a dict like
        # {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05}.
        keys = list(grid.keys())
        for combo in product(*grid.values()):
            params = dict(zip(keys, combo))
            # **params passes the dict as keyword arguments. The fixed
            # random_state makes each candidate's result repeatable.
            classifier = spec["class"](**params, random_state=cfg["training"]["random_state"])
            pipeline = build_deployment_pipeline(classifier, NUM_COLS, CAT_COLS)

            # One MLflow run per candidate. run_name makes each row in the
            # MLflow UI readable, for example "RandomForest_{'n_estimators': 100, ...}".
            with mlflow.start_run(run_name=f"{model_name}_{params}"):
                # model_family lets you filter or group runs in the UI.
                mlflow.log_param("model_family", model_name)
                for k, v in params.items():
                    mlflow.log_param(k, v)

                # Fit on the training split, score on the held-out test split.
                pipeline.fit(X_train, y_train)
                yhat = pipeline.predict(X_test)

                acc = accuracy_score(y_test, yhat)
                f1 = f1_score(y_test, yhat)

                # Log the scores and the model, as in train.py. skops needs to
                # be told which types to trust, and the list differs between
                # model families (logistic regression needs none, the tree
                # ensembles need the tree type), so we compute it per model.
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_score", f1)
                trusted_types = get_untrusted_types(data=dumps(pipeline))
                mlflow.sklearn.log_model(
                    pipeline, name="model", skops_trusted_types=trusted_types
                )

                # Keep the fitted pipeline in memory so the best ones can be
                # saved after the loop without retraining.
                print(f"  {params} → acc={acc:.4f}  f1={f1:.4f}")
                results.append({
                    "model_name": model_name,
                    "params": params,
                    "accuracy": acc,
                    "f1_score": f1,
                    "pipeline": pipeline,
                })

    # Rank by F1 and save top two.
    # F1 is used instead of accuracy because churners are the minority class:
    # F1 rewards models that actually find them.
    results.sort(key=lambda r: r["f1_score"], reverse=True)

    # The best model becomes v1 and the runner-up v2: two distinct models,
    # so the API can offer two versions. They are pickled for the API to load.
    for rank, label in [(0, "v1"), (1, "v2")]:
        best = results[rank]
        out_path = f"models/model_{label}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(best["pipeline"], f)
        print(f"\nSaved {label}: {best['model_name']} "
              f"(f1={best['f1_score']:.4f}) → {out_path}")


# Run main() only when executed as a script, not when imported.
if __name__ == "__main__":
    main()
