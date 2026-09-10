"""Tests for baseline comparison, Tuner study execution, and fresh retraining."""
from pathlib import Path
import numpy as np
import pytest
import yaml

from netsentry.tuning.result import BaselineComparisonReport, TuningResult
from netsentry.tuning.tuner import Tuner, compare_baselines


@pytest.fixture
def mock_dataset():
    np.random.seed(42)
    X_tr = np.random.randn(80, 6)
    y_tr = np.random.choice([0, 1], size=80)
    X_val = np.random.randn(20, 6)
    y_val = np.random.choice([0, 1], size=20)
    return X_tr, y_tr, X_val, y_val


def test_compare_baselines(mock_dataset):
    X_tr, y_tr, X_val, y_val = mock_dataset
    models = ["configs/models/logistic_regression.yaml", "configs/models/xgboost.yaml"]

    report = compare_baselines(
        model_configs=models,
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_val,
        y_val=y_val,
        selection_metric="roc_auc",
        tracker=None,
    )

    assert isinstance(report, BaselineComparisonReport)
    assert len(report.results) == 2
    assert report.best_baseline.selected is True
    # Verify descending sort
    assert report.results[0].selection_score >= report.results[1].selection_score


def test_tuner_executes_and_emits_tuned_config(mock_dataset, tmp_path: Path):
    X_tr, y_tr, X_val, y_val = mock_dataset

    # Create temporary mini tuning config to keep test fast (3 trials)
    mini_tuning_file = tmp_path / "tune_fast.yaml"
    mini_tuning_file.write_text(
        """
model_config: configs/models/logistic_regression.yaml
tuning:
  n_trials: 3
  direction: maximize
  metric: roc_auc
search_space:
  C:
    type: float
    low: 0.1
    high: 2.0
"""
    )

    out_models_dir = tmp_path / "models"
    tuner = Tuner(tuning_config=mini_tuning_file, tracker=None)

    tuning_res = tuner.tune(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_val,
        y_val=y_val,
        retrain_on_train_val=True,
        output_dir=str(out_models_dir),
    )

    assert isinstance(tuning_res, TuningResult)
    assert len(tuning_res.study.trials) == 3
    assert "C" in tuning_res.best_params
    assert Path(tuning_res.tuned_model_config_path).exists()

    # Verify original baseline YAML was NOT overwritten
    orig_lr_text = Path("configs/models/logistic_regression.yaml").read_text()
    assert "name: logistic_regression\n" in orig_lr_text

    # Verify tuned model YAML contains updated parameters
    with open(tuning_res.tuned_model_config_path, "r") as f:
        tuned_cfg = yaml.safe_load(f)
    assert tuned_cfg["name"] == "logistic_regression_tuned"
    assert "C" in tuned_cfg["parameters"]

    # Verify final retrained model is ready for inference
    preds = tuning_res.final_model.predict(X_val)
    assert preds.shape == (20,)
