"""
drift.py — Generate a data drift report comparing reference and current data.
Uses Evidently to detect changes in feature distributions that might
indicate the model is seeing data it was not trained on.

Data drift means the data arriving in production no longer looks like the
data the model was trained on (for example, customers now pay more per month
than before). The API keeps answering with status 200, so infrastructure
monitoring does not notice; the predictions may still be getting worse.
This script compares two datasets column by column:

* reference data: the cleaned training data (the raw CSV named in the config,
  passed through load_and_clean);
* current data: a CSV of recent production inputs given with --current or,
  if none is given, a copy of the reference data with drift added on purpose
  by simulate_drift(), so the report has something to find.

How to run (from the project root):

    python src/drift.py                                # simulated drift
    python src/drift.py --current data/new_batch.csv   # real current data
    make drift                                         # same as the first line

Output: an HTML report (default reports/drift_report.html) to open in a
browser. It shows, for each feature, whether its distribution has changed
and summary statistics for both datasets.
"""

import argparse
import os
import pandas as pd
import numpy as np
# Evidently is a model-monitoring library. A Report is a list of metrics or
# *presets* (ready-made groups of metrics) that is run on a pair of datasets.
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

# Reuse the project's own config loader and cleaning step, so the reference
# data is prepared in exactly the same way as for training.
from utils.config import load_config
from preprocess import load_and_clean


def parse_args():
    """Read command-line options: config file, optional current data, output path."""
    parser = argparse.ArgumentParser(description="Generate a data drift report.")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--current", default=None,
                        help="Path to current (production) data CSV. "
                             "If not provided, the script simulates drift "
                             "by perturbing the reference data.")
    parser.add_argument("--output", default="reports/drift_report.html",
                        help="Path to save the HTML report.")
    return parser.parse_args()


def simulate_drift(df, random_state=42):
    """
    Create a 'drifted' copy of the data by perturbing some features.
    This simulates what happens when production data diverges from
    the training distribution.
    """
    # A seeded random generator makes the simulated drift the same on every
    # run. df.copy() leaves the reference DataFrame unchanged.
    rng = np.random.RandomState(random_state)
    drifted = df.copy()

    # Shift numerical features
    # MonthlyCharges: 30% higher on average, plus random noise (normal
    # distribution, mean 0, standard deviation 10) for each row.
    drifted["MonthlyCharges"] = drifted["MonthlyCharges"] * 1.3 + rng.normal(0, 10, len(drifted))
    # tenure: 30% shorter, kept at 1 month or more and stored as whole months.
    drifted["tenure"] = (drifted["tenure"] * 0.7).clip(lower=1).astype(int)

    # Change categorical distribution: more month-to-month contracts
    # rng.random gives one number in [0, 1) per row; about 40% of them are
    # below 0.4, so about 40% of rows are switched to "Month-to-month".
    mask = rng.random(len(drifted)) < 0.4
    drifted.loc[mask, "Contract"] = "Month-to-month"

    return drifted


def main():
    """Load both datasets, run the Evidently report and save it as HTML."""
    args = parse_args()
    cfg = load_config(args.config)

    # ---- Reference data: the cleaned training data ----
    raw_path = cfg["paths"]["raw_data"]
    print(f"Loading reference data from {raw_path} ...")
    reference = load_and_clean(raw_path)

    # Drift is about the model's inputs, so the target column is removed.
    if "Churn" in reference.columns:
        reference = reference.drop(columns=["Churn"])

    # ---- Current data: real production data, or simulated drift ----
    if args.current:
        print(f"Loading current data from {args.current} ...")
        current = pd.read_csv(args.current)
    else:
        print("No current data provided — simulating drift ...")
        current = simulate_drift(reference)

    if "Churn" in current.columns:
        current = current.drop(columns=["Churn"])

    # ---- Build and run the report ----
    # DataDriftPreset: for every column, a statistical test checks whether
    # the current values come from the same distribution as the reference
    # values. Evidently picks the test from the column type (numerical or
    # categorical) and the number of rows, marks each column as drifted or
    # not, and reports the share of drifted columns.
    # DataSummaryPreset: descriptive statistics for each column in both
    # datasets (counts, missing values, mean, min/max, most common values),
    # which help explain what changed.
    print("Generating drift report ...")
    report = Report(metrics=[
        DataDriftPreset(),
        DataSummaryPreset(),
    ])

    # report.run does the calculations and returns a snapshot (the results);
    # the Report object itself only describes what to calculate.
    snapshot = report.run(reference_data=reference, current_data=current)

    # Create the output folder (e.g. reports/) if needed, then write the
    # results as a self-contained HTML page.
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    snapshot.save_html(args.output)
    print(f"Report saved to {args.output}")
    print("Open it in your browser to explore the results.")


if __name__ == "__main__":
    main()
