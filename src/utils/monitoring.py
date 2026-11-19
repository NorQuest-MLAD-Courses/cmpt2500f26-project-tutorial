"""
monitoring.py — Prometheus metrics for training processes.
Starts a lightweight HTTP server to expose training metrics
so Prometheus can scrape them independently of the Flask API.

Why a separate server: training (for example `src/tune.py`) runs as its own
short-lived process, not inside the Flask API, so the API's /metrics endpoint
cannot report on it. TrainingMonitor starts a small metrics web server inside
the training process itself (port 8002 by default). Prometheus scrapes that
port while training runs, in the same way it scrapes the API.

Usage, from a training script:

    from utils.monitoring import TrainingMonitor

    monitor = TrainingMonitor(port=8002)
    monitor.start()                      # metrics now at http://<host>:8002/metrics
    ...
    monitor.record_epoch(train_acc=..., val_acc=..., train_f1=..., val_f1=...)
    monitor.set_duration(seconds_since_start)
    monitor.record_feature_importance({"tenure": 0.4, ...})
    ...
    monitor.stop()

Outputs: Prometheus metrics only (training progress, model scores, feature
importances, CPU and memory use). Nothing is written to disk.
"""

import os
import time
import threading
import psutil
from prometheus_client import start_http_server, Counter, Gauge


class TrainingMonitor:
    """Base training monitor with common metrics."""

    def __init__(self, port=8002):
        """Create the training metrics; nothing is served until start() is called.

        Each metric is registered in prometheus_client's global registry when
        it is created, so create only one TrainingMonitor per process.
        """
        self.port = port

        # Training progress
        # Counter: only goes up. One increment per completed epoch or, for a
        # hyperparameter search, per evaluated candidate model.
        self.epochs_completed = Counter(
            "training_epochs_completed_total",
            "Total training epochs completed",
        )
        # Gauge: a value that is set directly and can go up or down.
        self.training_duration = Gauge(
            "training_duration_seconds",
            "Time elapsed since training started",
        )

        # Model performance
        # Gauges hold the latest score reported, so a dashboard shows how the
        # scores change as training goes on. Comparing training and
        # validation scores helps spot overfitting (training much higher).
        self.train_accuracy = Gauge("training_accuracy", "Training set accuracy")
        self.val_accuracy = Gauge("validation_accuracy", "Validation set accuracy")
        self.train_f1 = Gauge("training_f1_score", "Training set F1 score")
        self.val_f1 = Gauge("validation_f1_score", "Validation set F1 score")

        # Feature importance (top features)
        # One gauge with a `feature_name` label: each feature becomes its own
        # time series, e.g. feature_importance{feature_name="tenure"} 0.42.
        self.feature_importance = Gauge(
            "feature_importance",
            "Feature importance value",
            ["feature_name"],
        )

        # System resources
        # CPU and memory used by this training process, updated in the
        # background (see _monitor_resources). rss ("resident set size") is
        # the physical memory the process is using, in bytes.
        self.cpu_usage = Gauge("training_cpu_usage_percent", "CPU usage during training")
        self.memory_usage = Gauge("training_memory_usage_bytes", "Memory usage during training")
        self.memory_percent = Gauge("training_memory_usage_percent", "Memory usage percentage")

        # The background thread and the flag used to tell it to stop.
        # threading.Event is a thread-safe on/off flag: stop() sets it and the
        # loop in _monitor_resources checks it.
        self._resource_thread = None
        self._stop_event = threading.Event()

    def start(self):
        """Start the metrics HTTP server and resource monitoring thread."""
        # start_http_server runs a small web server in a background thread
        # and returns at once, so training continues while the server answers
        # Prometheus at http://<host>:<port>/metrics. It serves every metric
        # in the global registry, including the ones created in __init__.
        start_http_server(self.port)
        print(f"Training metrics server started on port {self.port}")

        # Start a second background thread that samples CPU and memory.
        # daemon=True means the thread does not keep the program alive: when
        # the training script finishes, Python exits without waiting for it.
        self._stop_event.clear()
        self._resource_thread = threading.Thread(
            target=self._monitor_resources, daemon=True
        )
        self._resource_thread.start()

    def stop(self):
        """Stop the resource monitoring thread."""
        # The loop notices the flag on its next check (within 5 seconds).
        # The metrics HTTP server keeps running until the process exits.
        self._stop_event.set()

    def _monitor_resources(self):
        """Update CPU and memory metrics every 5 seconds."""
        # psutil.Process gives access to resource usage of one process; here
        # the current one. (The leading underscore marks this method as
        # internal: it runs in the background thread started by start().)
        process = psutil.Process(os.getpid())
        while not self._stop_event.is_set():
            # cpu_percent() reports usage since the previous call, so the
            # first reading is 0.0 and later readings cover the last 5 seconds.
            self.cpu_usage.set(process.cpu_percent())
            mem = process.memory_info()
            self.memory_usage.set(mem.rss)
            self.memory_percent.set(process.memory_percent())
            time.sleep(5)

    def record_epoch(self, train_acc=None, val_acc=None, train_f1=None, val_f1=None):
        """Record metrics for a completed epoch or evaluation round."""
        self.epochs_completed.inc()
        # Scores are optional; only the ones passed in are updated, and the
        # others keep their previous values.
        if train_acc is not None:
            self.train_accuracy.set(train_acc)
        if val_acc is not None:
            self.val_accuracy.set(val_acc)
        if train_f1 is not None:
            self.train_f1.set(train_f1)
        if val_f1 is not None:
            self.val_f1.set(val_f1)

    def record_feature_importance(self, importance_dict, top_n=5):
        """Record top-N feature importances."""
        # Sort (name, importance) pairs from largest to smallest importance
        # and keep the first top_n. Limiting the number of features keeps the
        # number of time series (one per label value) small.
        sorted_features = sorted(
            importance_dict.items(), key=lambda x: x[1], reverse=True
        )[:top_n]
        # .labels(feature_name=...) selects the time series for one feature.
        # A feature that drops out of the top N keeps its last value.
        for name, value in sorted_features:
            self.feature_importance.labels(feature_name=name).set(value)

    def set_duration(self, seconds):
        """Set the elapsed training duration."""
        self.training_duration.set(seconds)
