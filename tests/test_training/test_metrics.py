"""Tests for metrics computation."""
import numpy as np
import pytest
from netsentry.training.metrics import calculate_metrics


def test_calculate_metrics_without_probabilities():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]

    metrics = calculate_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" not in metrics
    assert metrics["accuracy"] == 0.8


def test_calculate_metrics_with_probabilities():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 1, 0]
    y_prob = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.3, 0.7],
        [0.85, 0.15],
    ])

    metrics = calculate_metrics(y_true, y_pred, y_prob)
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] == 1.0
    assert metrics["f1"] == 1.0
