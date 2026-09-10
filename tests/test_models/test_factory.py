"""Tests for Model Factory and dynamic instantiation."""
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from netsentry.models.config import load_model_config
from netsentry.models.factory import create_model


def test_create_logistic_regression_pipeline():
    config = load_model_config("configs/models/logistic_regression.yaml")
    model = create_model(config)

    assert isinstance(model, Pipeline)
    assert "preprocessor" in model.named_steps
    assert isinstance(model.named_steps["preprocessor"], StandardScaler)
    assert "estimator" in model.named_steps
    assert isinstance(model.named_steps["estimator"], LogisticRegression)


def test_create_xgboost_pipeline():
    config = load_model_config("configs/models/xgboost.yaml")
    model = create_model(config)

    assert isinstance(model, Pipeline)
    assert "preprocessor" not in model.named_steps
    assert "estimator" in model.named_steps
    assert isinstance(model.named_steps["estimator"], XGBClassifier)


def test_create_model_from_custom_class_path():
    cfg = load_model_config({
        "name": "dummy_classifier",
        "implementation": {"class_path": "sklearn.dummy.DummyClassifier"},
        "preprocessing": {"scaling": "none"},
        "parameters": {"strategy": "most_frequent"}
    })
    model = create_model(cfg)
    assert isinstance(model, Pipeline)
    assert model.named_steps["estimator"].strategy == "most_frequent"
