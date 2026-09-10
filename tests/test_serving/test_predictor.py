"""Tests for Predictor service."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from netsentry.serving.model_loader import ChampionModel
from netsentry.serving.predictor import Predictor


def test_predictor_uses_loaded_threshold():
    model = LogisticRegression()
    # Fit simple 2D line
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    y = np.array([0, 1, 1])
    model.fit(X, y)

    # 1. High threshold 0.99
    champ_high = ChampionModel(model=model, version="1", threshold=0.99, model_name="NetSentry")
    pred_high = Predictor(champion_model=champ_high)
    p_high, prob_high = pred_high.predict({"f1": 1.0, "f2": 1.0})
    # With threshold 0.99, borderline sample predicted 0
    assert p_high == 0 or prob_high >= 0.99

    # 2. Low threshold 0.01
    champ_low = ChampionModel(model=model, version="1", threshold=0.01, model_name="NetSentry")
    pred_low = Predictor(champion_model=champ_low)
    p_low, prob_low = pred_low.predict({"f1": 1.0, "f2": 1.0})
    # With threshold 0.01, sample with prob > 0.01 predicted 1
    assert p_low == 1
