"""Tests for MLflow tracking abstraction."""
from pathlib import Path
import pytest
from netsentry.training.tracking import MLflowTracker


def test_mlflow_tracker_disabled():
    tracker = MLflowTracker(experiment_name="test-exp", enabled=False)
    with tracker.start_run() as run_id:
        assert run_id is None
        tracker.log_params({"a": 1})
        tracker.log_metrics({"metric": 0.95})


def test_mlflow_tracker_enabled(tmp_path: Path):
    db_path = tmp_path / "mlruns.db"
    uri = f"sqlite:///{db_path.as_posix()}"
    tracker = MLflowTracker(experiment_name="test-exp-active", tracking_uri=uri, enabled=True)

    with tracker.start_run(run_name="unit-test-run") as run_id:
        assert run_id is not None
        tracker.log_params({"lr": 0.1, "model": "xgboost"})
        tracker.log_metrics({"roc_auc": 0.98})
