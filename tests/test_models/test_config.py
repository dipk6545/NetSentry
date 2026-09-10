"""Tests for ModelConfig and YAML configuration loader."""
import pytest
from pathlib import Path
from netsentry.models.config import load_model_config, ModelConfigValidationError


def test_load_valid_logistic_regression_yaml(tmp_path: Path):
    cfg_file = tmp_path / "logreg.yaml"
    cfg_file.write_text(
        """
name: logistic_regression
implementation:
  class_path: sklearn.linear_model.LogisticRegression
preprocessing:
  scaling: standard
parameters:
  C: 0.5
  max_iter: 500
"""
    )
    cfg = load_model_config(cfg_file)
    assert cfg.name == "logistic_regression"
    assert cfg.implementation.class_path == "sklearn.linear_model.LogisticRegression"
    assert cfg.preprocessing.scaling == "standard"
    assert cfg.parameters["C"] == 0.5
    assert cfg.parameters["max_iter"] == 500


def test_load_valid_xgboost_yaml(tmp_path: Path):
    cfg_file = tmp_path / "xgb.yaml"
    cfg_file.write_text(
        """
name: xgboost
implementation:
  class_path: xgboost.XGBClassifier
preprocessing:
  scaling: none
parameters:
  n_estimators: 50
"""
    )
    cfg = load_model_config(cfg_file)
    assert cfg.name == "xgboost"
    assert cfg.preprocessing.scaling == "none"
    assert cfg.parameters["n_estimators"] == 50


def test_invalid_scaling_raises_error(tmp_path: Path):
    cfg_file = tmp_path / "invalid_scaling.yaml"
    cfg_file.write_text(
        """
name: test_model
implementation:
  class_path: sklearn.tree.DecisionTreeClassifier
preprocessing:
  scaling: invalid_scaler_name
"""
    )
    with pytest.raises(ModelConfigValidationError, match="Invalid scaling strategy"):
        load_model_config(cfg_file)


def test_missing_mandatory_fields_raises_error():
    with pytest.raises(ModelConfigValidationError, match="Missing required field 'name'"):
        load_model_config({"implementation": {"class_path": "sklearn.linear_model.Ridge"}})

    with pytest.raises(ModelConfigValidationError, match="Missing required mapping 'implementation'"):
        load_model_config({"name": "test_model"})
