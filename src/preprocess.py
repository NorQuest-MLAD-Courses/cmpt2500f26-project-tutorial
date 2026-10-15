"""
preprocess.py — Load raw data, clean it, encode categoricals, and save processed data.

This is the first stage of the pipeline. It turns the raw Telco CSV into a
fully numeric table that train.py can read. Each script in src/ talks to the
next one through files on disk, so this one ends by writing a CSV.

How to run (from the project root):
    make preprocess
    python src/preprocess.py
    python src/preprocess.py --config config/default.yaml --input other.csv

Settings come from config/default.yaml (paths.raw_data, paths.processed_data).
The --input and --output flags override the config for a single run.

Input:  the raw Telco CSV (paths.raw_data, or --input)
Output: the processed CSV (paths.processed_data, or --output)
"""

import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Our own helper in src/utils/config.py. This import works because Python puts
# the script's folder (src/) on the import path when you run src/preprocess.py.
from utils.config import load_config


def load_and_clean(path):
    """Load CSV and perform basic cleaning.

    Args:
        path: path to the raw Telco churn CSV file.

    Returns:
        A cleaned pandas DataFrame. Categorical columns are still text.
    """
    df = pd.read_csv(path)

    # Drop the customer ID column. It is a unique label per row, not a
    # feature, and a model could memorise it instead of learning patterns.
    df = df.drop(columns=["customerID"])

    # TotalCharges is stored as string in the raw data; convert to numeric.
    # errors="coerce" turns blank strings into NaN instead of raising an error.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Rows with tenure == 0 have missing TotalCharges; drop them.
    # .copy() gives us an independent DataFrame, which avoids pandas'
    # SettingWithCopyWarning when we modify columns below.
    df = df[df["tenure"] != 0].copy()

    # Fill any remaining NaN in TotalCharges with the column mean.
    # Unlike the first pass, the result is assigned back, so it takes effect.
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

    # SeniorCitizen is 0/1 in raw data; convert to "No"/"Yes" so it is encoded
    # the same way as the other yes/no text columns.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    return df


def encode_categoricals(df):
    """Label-encode every object column.

    Args:
        df: the cleaned DataFrame from load_and_clean().

    Returns:
        The same DataFrame with every text column replaced by integer codes.
    """
    le = LabelEncoder()
    # select_dtypes picks the text columns. Text is dtype "object" in pandas 2
    # but the new "string" dtype in pandas 3, so we list both to work with
    # either version. The one encoder is refitted on each column in turn;
    # categories are sorted alphabetically and numbered from 0. This also
    # encodes the target: Churn becomes No -> 0, Yes -> 1.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = le.fit_transform(df[col])
    return df


def parse_args():
    """Read command-line options.

    Returns:
        An argparse.Namespace with .config, .input and .output.
    """
    # argparse turns "--input path" on the command line into args.input, and
    # "python src/preprocess.py --help" prints the help text below.
    # --input and --output default to None, which means "not given": the
    # value then comes from the config file instead.
    parser = argparse.ArgumentParser(description="Preprocess raw churn data.")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--input", default=None, help="Override raw data path")
    parser.add_argument("--output", default=None, help="Override output path")
    return parser.parse_args()


def main():
    """Run the preprocessing stage: load, clean, encode, save."""
    # Read the command-line flags, then load the YAML config as a dict.
    args = parse_args()
    cfg = load_config(args.config)

    # Precedence: command-line flag first, then the config file.
    # `a or b` returns b when a is None, so a flag wins only if it was given.
    raw_path = args.input or cfg["paths"]["raw_data"]
    out_path = args.output or cfg["paths"]["processed_data"]

    print(f"Loading raw data from {raw_path} ...")
    df = load_and_clean(raw_path)

    print("Encoding categorical features ...")
    df = encode_categoricals(df)

    # Write the processed table for train.py. index=False stops pandas from
    # adding the row index as an extra, meaningless column.
    print(f"Saving processed data to {out_path} ...")
    df.to_csv(out_path, index=False)

    print(f"Done. {len(df)} rows written.")


# Run main() only when the file is executed as a script. When another module
# or a test does `import preprocess`, nothing runs and the functions above can
# be called on their own.
if __name__ == "__main__":
    main()
