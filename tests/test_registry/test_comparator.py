"""Tests for ChampionChallengerComparator."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from netsentry.registry.comparator import ChampionChallengerComparator, ModelComparisonMetrics


def test_champion_challenger_comparator():
    np.random.seed(42)
    X_val = np.random.randn(40, 4)
    y_val = np.random.choice([0, 1], size=40)
    X_test = np.random.randn(40, 4)
    y_test = np.random.choice([0, 1], size=40)
    labels_test = ["BENIGN" if y == 0 else "DDoS" for y in y_test]

    # Two models with different performance
    champ = LogisticRegression(C=0.01)
    champ.fit(X_val, y_val)

    chall = LogisticRegression(C=10.0)
    chall.fit(X_val, y_val)

    comparator = ChampionChallengerComparator()
    comp_metrics: ModelComparisonMetrics = comparator.compare(
        champion_model=champ,
        challenger_model=chall,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        labels_test=labels_test,
    )

    assert isinstance(comp_metrics, ModelComparisonMetrics)
    assert "recall" in comp_metrics.champion_metrics
    assert "recall" in comp_metrics.challenger_metrics
    assert "recall" in comp_metrics.deltas
    assert "latency_p95_ms" in comp_metrics.deltas
