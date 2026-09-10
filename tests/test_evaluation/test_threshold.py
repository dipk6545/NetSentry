"""Tests for validation threshold selection and application."""
import numpy as np
import pytest
from netsentry.evaluation.config import ThresholdConfig
from netsentry.evaluation.threshold import find_optimal_threshold, apply_threshold


def test_find_optimal_threshold_maximizes_f1():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Skewed probabilities: attacks have probabilities around 0.35
    probs = np.array([0.05, 0.1, 0.15, 0.2, 0.35, 0.40, 0.45, 0.50])

    res = find_optimal_threshold(y_true, probs, config=ThresholdConfig(objective="f1"))
    assert res.selected_threshold < 0.50  # Must choose lower threshold to capture attacks
    assert res.best_score > 0.80


def test_apply_threshold():
    probs = np.array([0.1, 0.45, 0.6, 0.8])
    preds = apply_threshold(probs, threshold=0.5)
    assert np.array_equal(preds, [0, 0, 1, 1])

    preds_lower = apply_threshold(probs, threshold=0.4)
    assert np.array_equal(preds_lower, [0, 1, 1, 1])
