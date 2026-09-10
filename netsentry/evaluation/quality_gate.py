"""
Quality Gate Auditor (`netsentry.evaluation.quality_gate`).
----------------------------------------------------------
Validates test evaluation performance against declarative threshold gates.
Produces an audit pass/fail decision with breakdown.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from netsentry.evaluation.config import QualityGatesConfig


@dataclass(frozen=True)
class GateCheck:
    """Individual quality rule check."""
    metric_name: str
    required_value: float
    actual_value: float
    passed: bool
    description: str


@dataclass(frozen=True)
class QualityGateDecision:
    """Overall quality gate pass/fail outcome."""
    passed: bool
    checks: List[GateCheck]
    failure_reasons: List[str]


def evaluate_quality_gates(
    metrics: Dict[str, float],
    latency_ms: float,
    config: QualityGatesConfig,
) -> QualityGateDecision:
    """
    Evaluates whether candidate model satisfies all quality gates.
    """
    checks: List[GateCheck] = []
    failures: List[str] = []

    # Recall check
    if config.min_recall is not None:
        rec = metrics.get("recall", 0.0)
        passed = rec >= config.min_recall
        checks.append(GateCheck("recall", config.min_recall, rec, passed, "Minimum test recall"))
        if not passed:
            failures.append(f"Recall {rec:.4f} below threshold {config.min_recall}")

    # F1 check
    if config.min_f1 is not None:
        f1 = metrics.get("f1", 0.0)
        passed = f1 >= config.min_f1
        checks.append(GateCheck("f1", config.min_f1, f1, passed, "Minimum test F1 score"))
        if not passed:
            failures.append(f"F1 {f1:.4f} below threshold {config.min_f1}")

    # ROC-AUC check
    if config.min_roc_auc is not None:
        roc = metrics.get("roc_auc", 0.0)
        passed = roc >= config.min_roc_auc
        checks.append(GateCheck("roc_auc", config.min_roc_auc, roc, passed, "Minimum test ROC-AUC"))
        if not passed:
            failures.append(f"ROC-AUC {roc:.4f} below threshold {config.min_roc_auc}")

    # Latency check
    if config.max_latency_ms is not None:
        passed = latency_ms <= config.max_latency_ms
        checks.append(GateCheck("latency_ms", config.max_latency_ms, latency_ms, passed, "Maximum inference latency (p95 ms)"))
        if not passed:
            failures.append(f"Latency {latency_ms:.2f}ms exceeds maximum {config.max_latency_ms}ms")

    overall_passed = len(failures) == 0
    return QualityGateDecision(
        passed=overall_passed,
        checks=checks,
        failure_reasons=failures,
    )
