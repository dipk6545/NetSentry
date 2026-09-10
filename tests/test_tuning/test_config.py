"""Tests for TuningConfig parser and search space validator."""
from pathlib import Path
import pytest
from netsentry.tuning.config import load_tuning_config, TuningConfigValidationError


def test_load_valid_tuning_yaml(tmp_path: Path):
    cfg_file = tmp_path / "tune_xgb.yaml"
    cfg_file.write_text(
        """
model_config: configs/models/xgboost.yaml
tuning:
  n_trials: 25
  direction: maximize
  metric: roc_auc
search_space:
  n_estimators:
    type: int
    low: 50
    high: 150
  learning_rate:
    type: float
    low: 0.01
    high: 0.3
    log: true
  booster:
    type: categorical
    choices: [gbtree, dart]
"""
    )
    cfg = load_tuning_config(cfg_file)
    assert cfg.model_config_path == "configs/models/xgboost.yaml"
    assert cfg.tuning.n_trials == 25
    assert cfg.tuning.direction == "maximize"
    assert "n_estimators" in cfg.search_space
    assert cfg.search_space["n_estimators"].param_type == "int"
    assert cfg.search_space["booster"].choices == ["gbtree", "dart"]


def test_invalid_param_bounds_raises(tmp_path: Path):
    cfg_file = tmp_path / "invalid_bounds.yaml"
    cfg_file.write_text(
        """
model_config: configs/models/xgboost.yaml
tuning:
  n_trials: 10
search_space:
  n_estimators:
    type: int
    low: 100
    high: 50
"""
    )
    with pytest.raises(TuningConfigValidationError, match="low .* cannot exceed high"):
        load_tuning_config(cfg_file)
