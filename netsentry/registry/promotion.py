"""
Promotion Decision Engine (`netsentry.registry.promotion`).
------------------------------------------------------------
Applies declarative promotion rules to determine if a Challenger should
displace the incumbent Champion.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from netsentry.registry.comparator import ModelComparisonMetrics
from netsentry.registry.config import PromotionConfig


@dataclass(frozen=True)
class PromotionDecision:
    """Outcome of promotion evaluation."""
    passed: bool
    reason: str
    champion_version: Optional[str]
    challenger_version: str
    metric_comparison: ModelComparisonMetrics
    rule_failures: List[str]


def evaluate_promotion(
    comparison: ModelComparisonMetrics,
    champion_version: Optional[str],
    challenger_version: str,
    config: PromotionConfig,
) -> PromotionDecision:
    """
    Evaluates whether Challenger qualifies for promotion over Champion.
    Handles both First Release (champion is None) and Challenger vs Champion.
    """
    rule_failures: List[str] = []
    chall_m = comparison.challenger_metrics
    deltas = comparison.deltas

    # Case 1: First Production Release (No incumbent champion)
    if champion_version is None:
        # Check static minimum constraints only
        for metric_name, rule in config.rules.items():
            val = chall_m.get(metric_name)
            if val is not None:
                if rule.minimum is not None and val < rule.minimum:
                    rule_failures.append(
                        f"First release constraint failed: {metric_name}={val:.4f} < minimum={rule.minimum}"
                    )
                if rule.maximum is not None and val > rule.maximum:
                    rule_failures.append(
                        f"First release constraint failed: {metric_name}={val:.4f} > maximum={rule.maximum}"
                    )

        passed = len(rule_failures) == 0
        reason = "First production release promoted to @champion." if passed else "First release failed constraints."
        return PromotionDecision(
            passed=passed,
            reason=reason,
            champion_version=None,
            challenger_version=challenger_version,
            metric_comparison=comparison,
            rule_failures=rule_failures,
        )

    # Case 2: Challenger vs Champion Evaluation
    # 1. Primary metric improvement requirement
    if config.require_challenger_better:
        primary = comparison.primary_metric
        primary_delta = deltas.get(primary, 0.0)
        if primary_delta <= 0.0:
            rule_failures.append(
                f"Challenger did not improve primary metric '{primary}': delta={primary_delta:+.4f}"
            )

    # 2. Evaluate all configured promotion rules and operational boundaries
    for metric_name, rule in config.rules.items():
        val = chall_m.get(metric_name)
        delta = deltas.get(metric_name, 0.0)

        if val is not None:
            if rule.minimum is not None and val < rule.minimum:
                rule_failures.append(
                    f"Operational minimum violated for {metric_name}: {val:.4f} < {rule.minimum}"
                )
            if rule.maximum is not None and val > rule.maximum:
                rule_failures.append(
                    f"Operational maximum exceeded for {metric_name}: {val:.4f} > {rule.maximum}"
                )
            if rule.improvement_required and delta <= 0.0:
                rule_failures.append(
                    f"Improvement required for {metric_name} but delta is {delta:+.4f}"
                )

    passed = len(rule_failures) == 0
    if passed:
        reason = f"Challenger v{challenger_version} outperformed Champion v{champion_version} across all operational criteria."
    else:
        reason = f"Challenger v{challenger_version} rejected: {'; '.join(rule_failures)}"

    return PromotionDecision(
        passed=passed,
        reason=reason,
        champion_version=champion_version,
        challenger_version=challenger_version,
        metric_comparison=comparison,
        rule_failures=rule_failures,
    )
