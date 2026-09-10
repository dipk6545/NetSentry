"""Tests for evaluation metrics and confusion rates."""
import numpy as np
import pytest
from netsentry.evaluation.metrics import compute_binary_metrics


def test_compute_binary_metrics():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 0, 1, 1]
    y_prob = [0.1, 0.7, 0.4, 0.9, 0.85]

    m = compute_binary_metrics(y_true, y_pred, y_prob)

    assert "accuracy" in m
    assert "precision" in m
    assert "recall" in m
    assert "f1" in m
    assert "roc_auc" in m
    assert "pr_auc" in m
    assert "false_positive_rate" in m
    assert "false_negative_rate" in m

    # 1 FP out of 2 true negatives = 0.5 FPR
    assert m["false_positive_rate"] == 0.5
    # 1 FN out of 3 true positives = 1/3 FNR
    assert pytest.approx(m["false_negative_rate"], 0.01) == 0.3333
