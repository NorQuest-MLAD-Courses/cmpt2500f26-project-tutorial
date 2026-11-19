"""Tests for the training monitor in src/utils/monitoring.py.

``TrainingMonitor`` keeps training metrics (epochs, scores, feature
importances) as Prometheus metrics. These tests record a few values and read
them back from prometheus_client's registry. They never call ``start()``, so
no metrics server is started and no network port is needed.

Run it from the project root with::

    pytest
"""

import pytest
from prometheus_client import REGISTRY
import sys, os

# The module under test lives in src/, which is not on Python's import path
# by default. Prepending it to sys.path lets us write `from utils import ...`.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.monitoring import TrainingMonitor


# Every Prometheus metric is registered once per process, under its name, so
# a second TrainingMonitor in the same test run would fail with "Duplicated
# timeseries". scope="module" makes pytest create this fixture only once for
# the whole file and share it between the tests below.
@pytest.fixture(scope="module")
def monitor():
    return TrainingMonitor(port=8002)


def test_record_epoch_updates_metrics(monitor):
    """record_epoch counts the epoch and stores the scores it is given."""
    # get_sample_value reads a metric's current value by its name. The
    # counter is read before and after, because the fixture is shared and
    # another test may already have counted epochs.
    before = REGISTRY.get_sample_value("training_epochs_completed_total")
    monitor.record_epoch(train_acc=0.9, val_acc=0.8, val_f1=0.6)
    after = REGISTRY.get_sample_value("training_epochs_completed_total")
    assert after == before + 1
    assert REGISTRY.get_sample_value("validation_accuracy") == 0.8
    assert REGISTRY.get_sample_value("validation_f1_score") == 0.6


def test_record_feature_importance_keeps_top_n(monitor):
    """Only the top_n most important features are published."""
    importances = {"tenure": 0.5, "Contract": 0.3, "gender": 0.01}
    monitor.record_feature_importance(importances, top_n=2)
    # Labelled metrics are read with a dict of label values.
    assert REGISTRY.get_sample_value("feature_importance", {"feature_name": "tenure"}) == 0.5
    assert REGISTRY.get_sample_value("feature_importance", {"feature_name": "Contract"}) == 0.3
    # gender is third, so it was left out. A missing sample reads as None.
    assert REGISTRY.get_sample_value("feature_importance", {"feature_name": "gender"}) is None
