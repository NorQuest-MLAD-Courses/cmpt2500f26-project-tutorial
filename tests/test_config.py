"""Tests for the config loader."""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.config import load_config


def test_load_config_returns_dict(tmp_path):
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("paths:\n  raw_data: data/raw/test.csv\n")
    cfg = load_config(str(cfg_file))
    assert isinstance(cfg, dict)
    assert cfg["paths"]["raw_data"] == "data/raw/test.csv"


def test_default_config_has_required_keys():
    """The shipped default.yaml must contain the keys every script expects."""
    cfg = load_config("config/default.yaml")
    assert "paths" in cfg
    assert "features" in cfg
    assert "training" in cfg
    assert "model" in cfg
