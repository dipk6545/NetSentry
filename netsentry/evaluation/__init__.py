"""NetSentry Evaluation Layer."""
from netsentry.evaluation.config import EvaluationConfig, ThresholdConfig, QualityGatesConfig, load_evaluation_config
from netsentry.evaluation.metrics import compute_binary_metrics
from netsentry.evaluation.threshold import find_optimal_threshold, apply_threshold, ThresholdSearchResult
from netsentry.evaluation.slices import compute_slice_metrics, AttackSliceMetric
from netsentry.evaluation.latency import measure_inference_latency, LatencyProfile
from netsentry.evaluation.quality_gate import evaluate_quality_gates, QualityGateDecision, GateCheck
from netsentry.evaluation.evaluator import Evaluator, EvaluationResult

__all__ = [
    "EvaluationConfig",
    "ThresholdConfig",
    "QualityGatesConfig",
    "load_evaluation_config",
    "compute_binary_metrics",
    "find_optimal_threshold",
    "apply_threshold",
    "ThresholdSearchResult",
    "compute_slice_metrics",
    "AttackSliceMetric",
    "measure_inference_latency",
    "LatencyProfile",
    "evaluate_quality_gates",
    "QualityGateDecision",
    "GateCheck",
    "Evaluator",
    "EvaluationResult",
]
