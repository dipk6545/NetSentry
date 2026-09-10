"""Tests for Optuna objective sampling and execution."""
import numpy as np
import optuna
import pytest

from netsentry.models.config import load_model_config
from netsentry.tuning.config import load_tuning_config
from netsentry.tuning.objective import GenericObjective, sample_parameters


def test_sample_parameters():
    tuning_cfg = load_tuning_config("configs/tuning/logistic_regression.yaml")
    study = optuna.create_study()
    trial = study.ask()

    sampled = sample_parameters(trial, tuning_cfg.search_space)
    assert "C" in sampled
    assert "max_iter" in sampled
    assert 0.01 <= sampled["C"] <= 10.0
    assert 200 <= sampled["max_iter"] <= 1000


def test_generic_objective_trial_evaluation():
    np.random.seed(42)
    X_tr = np.random.randn(60, 6)
    y_tr = np.random.choice([0, 1], size=60)
    X_val = np.random.randn(20, 6)
    y_val = np.random.choice([0, 1], size=20)

    model_cfg = load_model_config("configs/models/logistic_regression.yaml")
    tuning_cfg = load_tuning_config("configs/tuning/logistic_regression.yaml")

    objective = GenericObjective(
        base_model_config=model_cfg,
        tuning_config=tuning_cfg,
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_val,
        y_val=y_val,
        tracker=None,  # offline evaluation
    )

    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    score = objective(trial)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
