"""
config.py — Load a YAML configuration file and return it as a dictionary.
"""

import yaml


def load_config(path="config/default.yaml"):
    """Read a YAML file and return its contents as a nested dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
