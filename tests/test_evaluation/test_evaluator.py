"""Tests for end-to-end Evaluator."""
from pathlib import Path
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from netsentry.evaluation.config import EvaluationConfig, QualityGatesConfig
from netsentry.evaluation.evaluator import Evaluator, EvaluationResult


def test_evaluator_end_to_end():
    np.random.seed(42)
    # Generate synthetic training/val/test data
    X_val = np.random.randn(50, 4)
    y_val = np.random.choice([0, 1], size=50)

    X_test = np.random.randn(50, 4)
    y_test = np.random.choice([0, 1], size=50)
    labels_test = ["BENIGN" if y == 0 else "PortScan" for y in y_test]

    # Pre-fitted pipeline
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
    pipeline.fit(X_val, y_val)

    # Tolerant quality gates for testing
    cfg = EvaluationConfig(
        quality_gates=QualityGatesConfig(min_recall=0.1, min_f1=0.1, min_roc_auc=0.1, max_latency_ms=100.0)
    )

    evaluator = Evaluator(config=cfg, tracker=None)
    result = evaluator.evaluate(
        model=pipeline,
        model_name="test_logreg",
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        labels_test=labels_test,
    )

    assert isinstance(result, EvaluationResult)
    assert 0.01 <= result.selected_threshold <= 0.99
    assert "accuracy" in result.selected_threshold_metrics
    assert "BENIGN" in result.slice_metrics
    assert "PortScan" in result.slice_metrics
    assert result.latency.samples_evaluated == 30
    assert result.quality_gate.passed is True
