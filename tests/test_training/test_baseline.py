"""Tests for BaselineOrchestrator and baseline discovery (`netsentry.training.baseline`)."""
import json
from pathlib import Path
import numpy as np
import pytest

from netsentry.training.baseline import (
    BaselineOrchestrator,
    BaselineOrchestrationResult,
    discover_baseline_configs,
)
from netsentry.training.tracking import MLflowTracker


@pytest.fixture
def mock_dataset():
    np.random.seed(42)
    X_tr = np.random.randn(80, 4)
    y_tr = np.random.choice([0, 1], size=80)
    X_val = np.random.randn(20, 4)
    y_val = np.random.choice([0, 1], size=20)
    return X_tr, y_tr, X_val, y_val


def test_discovers_baseline_configs_and_excludes_tuned(tmp_path: Path):
    cfg_dir = tmp_path / "models"
    cfg_dir.mkdir()

    (cfg_dir / "xgboost.yaml").write_text("name: xgboost\n")
    (cfg_dir / "lightgbm.yaml").write_text("name: lightgbm\n")
    (cfg_dir / "logistic_regression.yaml").write_text("name: logistic_regression\n")
    (cfg_dir / "xgboost_tuned.yaml").write_text("name: xgboost_tuned\n")
    (cfg_dir / "readme.txt").write_text("not a yaml\n")

    discovered = discover_baseline_configs(cfg_dir)
    names = [f.name for f in discovered]

    assert "xgboost.yaml" in names
    assert "lightgbm.yaml" in names
    assert "logistic_regression.yaml" in names
    assert "xgboost_tuned.yaml" not in names
    assert "readme.txt" not in names
    assert len(discovered) == 3


def test_trains_all_models_and_persists_winner(mock_dataset, tmp_path: Path):
    X_tr, y_tr, X_val, y_val = mock_dataset

    # Create 2 fast baseline configs
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "dummy_a.yaml").write_text(
        """
name: dummy_a
implementation:
  class_path: sklearn.dummy.DummyClassifier
preprocessing:
  scaling: none
parameters:
  strategy: most_frequent
"""
    )
    (cfg_dir / "dummy_b.yaml").write_text(
        """
name: dummy_b
implementation:
  class_path: sklearn.dummy.DummyClassifier
preprocessing:
  scaling: none
parameters:
  strategy: uniform
  random_state: 42
"""
    )

    db_path = tmp_path / "baseline_orch.db"
    uri = f"sqlite:///{db_path.as_posix()}"
    tracker = MLflowTracker(experiment_name="test-orch", tracking_uri=uri, enabled=True)

    artifacts_dir = tmp_path / "artifacts"
    orchestrator = BaselineOrchestrator(
        config_dir=cfg_dir,
        primary_metric="recall",
        tracker=tracker,
        artifacts_output_dir=artifacts_dir,
    )

    res = orchestrator.run(X_train=X_tr, y_train=y_tr, X_val=X_val, y_val=y_val)

    assert isinstance(res, BaselineOrchestrationResult)
    assert res.winner in {"dummy_a", "dummy_b"}
    assert len(res.models) == 2
    assert all(m.status == "SUCCESS" for m in res.models)

    # Check persisted JSON artifact
    artifact_file = artifacts_dir / "latest.json"
    assert artifact_file.exists()
    with open(artifact_file, "r") as f:
        data = json.load(f)
    assert data["winner"] == res.winner
    assert data["metric"] == "recall"
    assert len(data["models"]) == 2


def test_failed_model_does_not_stop_pipeline(mock_dataset, tmp_path: Path):
    X_tr, y_tr, X_val, y_val = mock_dataset

    cfg_dir = tmp_path / "configs_with_failure"
    cfg_dir.mkdir()
    # Valid model
    (cfg_dir / "good.yaml").write_text(
        """
name: good_model
implementation:
  class_path: sklearn.dummy.DummyClassifier
preprocessing:
  scaling: none
parameters:
  strategy: most_frequent
"""
    )
    # Invalid class_path model
    (cfg_dir / "broken.yaml").write_text(
        """
name: broken_model
implementation:
  class_path: non_existent.module.BrokenClass
preprocessing:
  scaling: none
"""
    )

    db_path = tmp_path / "failure_test.db"
    uri = f"sqlite:///{db_path.as_posix()}"
    tracker = MLflowTracker(experiment_name="test-fail", tracking_uri=uri, enabled=True)

    orchestrator = BaselineOrchestrator(
        config_dir=cfg_dir,
        primary_metric="recall",
        tracker=tracker,
        artifacts_output_dir=tmp_path / "art",
    )

    res = orchestrator.run(X_train=X_tr, y_train=y_tr, X_val=X_val, y_val=y_val)

    assert res.winner == "good_model"
    assert len(res.models) == 2
    good_eval = next(m for m in res.models if m.name == "good_model")
    broken_eval = next(m for m in res.models if m.name == "broken")
    assert good_eval.status == "SUCCESS"
    assert broken_eval.status == "FAILED"


def test_all_models_failed_fails_job(mock_dataset, tmp_path: Path):
    X_tr, y_tr, X_val, y_val = mock_dataset

    cfg_dir = tmp_path / "configs_all_bad"
    cfg_dir.mkdir()
    (cfg_dir / "broken.yaml").write_text(
        """
name: broken
implementation:
  class_path: non_existent.module.BrokenClass
preprocessing:
  scaling: none
"""
    )

    orchestrator = BaselineOrchestrator(
        config_dir=cfg_dir,
        artifacts_output_dir=tmp_path / "art",
    )

    with pytest.raises(RuntimeError, match="All baseline model runs failed"):
        orchestrator.run(X_train=X_tr, y_train=y_tr, X_val=X_val, y_val=y_val)
