"""Tests for the config loader.

These tests cover ``load_config`` in ``src/utils/config.py``, the helper that
reads a YAML file into a Python dictionary. Run them from the project root
with::

    pytest

The first test writes its own small YAML file, so it does not depend on the
project's real configuration. The second test does read the shipped
``config/default.yaml``, because its job is to catch a missing section in that
file before a script fails on it. It uses a relative path, so pytest must be
started from the project root.
"""

import sys, os

# The module under test lives in src/, which is not on Python's import path
# by default. Prepending it to sys.path lets us write `from utils.config import ...`.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.config import load_config


def test_load_config_returns_dict(tmp_path):
    """load_config should parse a YAML file into a nested dict."""
    # tmp_path is a built-in pytest fixture: a fresh, empty temporary
    # directory created for this test only. We write a two-line YAML file
    # there instead of touching the real config folder.
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("paths:\n  raw_data: data/raw/test.csv\n")
    cfg = load_config(str(cfg_file))
    assert isinstance(cfg, dict)
    # Nested YAML keys become nested dicts: paths -> raw_data.
    assert cfg["paths"]["raw_data"] == "data/raw/test.csv"


def test_default_config_has_required_keys():
    """The shipped default.yaml must contain the keys every script expects."""
    cfg = load_config("config/default.yaml")
    # One check per top-level section the scripts read.
    assert "paths" in cfg
    assert "features" in cfg
    assert "training" in cfg
    assert "model" in cfg
