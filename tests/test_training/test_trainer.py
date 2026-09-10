"""Tests for model-agnostic Trainer."""
from pathlib import Path
import numpy as np
import pytest

from netsentry.models.config import load_model_config
from netsentry.models.factory import create_model
from netsentry.training.config import TrainingConfig, TrackingConfig
from netsentry.training.tracking import MLflowTracker
from netsentry.training.trainer import Trainer, TrainingResult


@pytest.fixture
def mock_dataset():
    np.random.seed(42)
    X_tr = np.random.randn(80, 8)
    y_tr = np.random.choice([0, 1], size=80)
    X_val = np.random.randn(20, 8)
    y_val = np.random.choice([0, 1], size=20)
    return X_tr, y_tr, X_val, y_val


def test_trainer_trains_logistic_regression(mock_dataset, tmp_path: Path):
    X_tr, y_tr, X_val, y_val = mock_dataset
    model_cfg = load_model_config("configs/models/logistic_regression.yaml")
    model = create_model(model_cfg)

    db_path = tmp_path / "mlruns_lr.db"
    uri = f"sqlite:///{db_path.as_posix()}"
    tracker = MLflowTracker(experiment_name="test-training", tracking_uri=uri, enabled=True)
    trainer = Trainer(model=model, model_config=model_cfg, tracker=tracker)

    result = trainer.train(X_tr, y_tr, X_val, y_val)

    assert isinstance(result, TrainingResult)
    assert result.run_id is not None
    assert "accuracy" in result.metrics
    assert "roc_auc" in result.metrics
    assert result.training_metadata["model_name"] == "logistic_regression"
    assert result.training_metadata["training_rows"] == 80


def test_trainer_trains_xgboost(mock_dataset, tmp_path: Path):
    X_tr, y_tr, X_val, y_val = mock_dataset
    model_cfg = load_model_config("configs/models/xgboost.yaml")
    model = create_model(model_cfg)

    db_path = tmp_path / "mlruns_xgb.db"
    uri = f"sqlite:///{db_path.as_posix()}"
    tracker = MLflowTracker(experiment_name="test-training", tracking_uri=uri, enabled=True)
    trainer = Trainer(model=model, model_config=model_cfg, tracker=tracker)

    result = trainer.train(X_tr, y_tr, X_val, y_val)

    assert isinstance(result, TrainingResult)
    assert result.run_id is not None
    assert "roc_auc" in result.metrics
    assert result.training_metadata["model_name"] == "xgboost"
