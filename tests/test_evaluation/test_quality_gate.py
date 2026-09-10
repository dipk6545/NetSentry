"""Tests for quality gate evaluation."""
import pytest
from netsentry.evaluation.config import QualityGatesConfig
from netsentry.evaluation.quality_gate import evaluate_quality_gates


def test_quality_gate_passes():
    metrics = {"recall": 0.96, "f1": 0.92, "roc_auc": 0.97}
    config = QualityGatesConfig(min_recall=0.95, min_f1=0.90, min_roc_auc=0.95, max_latency_ms=10.0)

    decision = evaluate_quality_gates(metrics, latency_ms=4.5, config=config)
    assert decision.passed is True
    assert len(decision.failure_reasons) == 0


def test_quality_gate_fails():
    metrics = {"recall": 0.91, "f1": 0.88, "roc_auc": 0.92}
    config = QualityGatesConfig(min_recall=0.95, min_f1=0.90, min_roc_auc=0.95, max_latency_ms=10.0)

    decision = evaluate_quality_gates(metrics, latency_ms=15.0, config=config)
    assert decision.passed is False
    assert len(decision.failure_reasons) == 4  # All 4 failed
