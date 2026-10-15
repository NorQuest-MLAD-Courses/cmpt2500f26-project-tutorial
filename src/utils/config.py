"""
config.py — Load a YAML configuration file and return it as a dictionary.

All project settings (data paths, feature lists, train/test split, model
hyperparameters) live in config/default.yaml. Every script in src/ calls
load_config() once at start-up and reads its settings from the returned dict,
so changing a setting means editing one YAML file instead of the code.

How to use (this module is imported, not run on its own):
    from utils.config import load_config
    cfg = load_config("config/default.yaml")
    cfg["paths"]["raw_data"]   # -> "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

Input:  a YAML file
Output: a nested Python dict with the same structure as the YAML
"""

import yaml


def load_config(path="config/default.yaml"):
    """Read a YAML file and return its contents as a nested dict.

    Args:
        path: path to the YAML file, relative to the folder you run from
            (normally the project root).

    Returns:
        The parsed YAML as a dict (nested dicts and lists).
    """
    # The "with" block closes the file automatically, even if parsing fails.
    # safe_load only builds plain types (dict, list, str, int, float, bool).
    # yaml.load without a safe loader can construct arbitrary Python objects,
    # which is a security risk if the file comes from someone else.
    with open(path, "r") as f:
        return yaml.safe_load(f)
