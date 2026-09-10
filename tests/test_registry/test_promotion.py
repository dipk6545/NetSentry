"""Tests for Promotion Decision logic."""
import pytest
from netsentry.registry.comparator import ModelComparisonMetrics
from netsentry.registry.config import PromotionConfig, PromotionRule
from netsentry.registry.promotion import evaluate_promotion, PromotionDecision


@pytest.fixture
def mock_promotion_config():
    return PromotionConfig(
        require_quality_gates=True,
        require_challenger_better=True,
        rules={
            "recall": PromotionRule(minimum=0.90, improvement_required=True),
            "precision": PromotionRule(minimum=0.85),
            "latency_p95_ms": PromotionRule(maximum=20.0),
        },
    )


def test_first_release_promotion(mock_promotion_config):
    comp = ModelComparisonMetrics(
        champion_metrics={},
        challenger_metrics={"recall": 0.94, "precision": 0.88, "latency_p95_ms": 5.0},
        deltas={},
        primary_metric="recall",
    )
    decision = evaluate_promotion(comp, champion_version=None, challenger_version="1", config=mock_promotion_config)
    assert decision.passed is True
    assert decision.champion_version is None
    assert decision.challenger_version == "1"


def test_challenger_promoted_when_better(mock_promotion_config):
    comp = ModelComparisonMetrics(
        champion_metrics={"recall": 0.91, "precision": 0.86, "latency_p95_ms": 6.0},
        challenger_metrics={"recall": 0.95, "precision": 0.89, "latency_p95_ms": 7.0},
        deltas={"recall": 0.04, "precision": 0.03, "latency_p95_ms": 1.0},
        primary_metric="recall",
    )
    decision = evaluate_promotion(comp, champion_version="1", challenger_version="2", config=mock_promotion_config)
    assert decision.passed is True
    assert len(decision.rule_failures) == 0


def test_challenger_rejected_when_lower_recall(mock_promotion_config):
    comp = ModelComparisonMetrics(
        champion_metrics={"recall": 0.94, "precision": 0.86, "latency_p95_ms": 6.0},
        challenger_metrics={"recall": 0.92, "precision": 0.95, "latency_p95_ms": 7.0},
        deltas={"recall": -0.02, "precision": 0.09, "latency_p95_ms": 1.0},
        primary_metric="recall",
    )
    decision = evaluate_promotion(comp, champion_version="1", challenger_version="2", config=mock_promotion_config)
    assert decision.passed is False
    assert any("primary metric 'recall'" in f for f in decision.rule_failures)
