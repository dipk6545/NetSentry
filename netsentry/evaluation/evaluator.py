"""
Final Model Evaluator (`netsentry.evaluation.evaluator`).
--------------------------------------------------------
Orchestrates:
1. Validation decision threshold optimization & freezing.
2. Evaluation on untouched TEST set.
3. Default vs selected threshold metrics.
4. Attack family slices.
5. Inference latency profiling.
6. Quality gate decision audit.
7. MLflow evaluation artifact logging.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import json
import numpy as np
import polars as pl

from netsentry.evaluation.config import EvaluationConfig, load_evaluation_config
from netsentry.evaluation.latency import LatencyProfile, measure_inference_latency
from netsentry.evaluation.metrics import compute_binary_metrics
from netsentry.evaluation.quality_gate import QualityGateDecision, evaluate_quality_gates
from netsentry.evaluation.slices import AttackSliceMetric, compute_slice_metrics
from netsentry.evaluation.threshold import apply_threshold, find_optimal_threshold
from netsentry.models.interface import has_predict_proba
from netsentry.training.tracking import MLflowTracker


@dataclass(frozen=True)
class EvaluationResult:
    """Complete evaluation report for a candidate model."""
    model_name: str
    default_metrics: Dict[str, float]
    selected_threshold: float
    selected_threshold_metrics: Dict[str, float]
    slice_metrics: Dict[str, AttackSliceMetric]
    latency: LatencyProfile
    quality_gate: QualityGateDecision


class Evaluator:
    """Coordinates validation threshold search and untouched TEST set evaluation."""

    def __init__(
        self,
        config: Optional[Union[str, Path, EvaluationConfig]] = None,
        tracker: Optional[MLflowTracker] = None,
    ):
        if config is None:
            self.config = EvaluationConfig()
        elif isinstance(config, (str, Path)):
            self.config = load_evaluation_config(config)
        else:
            self.config = config

        self.tracker = tracker

    def evaluate(
        self,
        model: Any,
        model_name: str,
        X_val: Union[np.ndarray, pl.DataFrame],
        y_val: Union[np.ndarray, pl.Series],
        X_test: Union[np.ndarray, pl.DataFrame],
        y_test: Union[np.ndarray, pl.Series],
        labels_test: Union[np.ndarray, pl.Series, list],
    ) -> EvaluationResult:
        """
        Executes complete evaluation protocol:
        - Scans X_val for threshold tau*
        - Applies tau* strictly to untouched X_test
        - Audits slice metrics, latency, and quality gates
        """
        X_v = X_val.to_numpy() if isinstance(X_val, pl.DataFrame) else np.asarray(X_val)
        y_v = y_val.to_numpy() if isinstance(y_val, pl.Series) else np.asarray(y_val, dtype=int)
        X_t = X_test.to_numpy() if isinstance(X_test, pl.DataFrame) else np.asarray(X_test)
        y_t = y_test.to_numpy() if isinstance(y_test, pl.Series) else np.asarray(y_test, dtype=int)

        supports_proba = has_predict_proba(model)

        # 1. Validation Threshold Search
        selected_th = 0.50
        if supports_proba and self.config.threshold.enabled:
            val_probs = model.predict_proba(X_v)
            th_search = find_optimal_threshold(
                y_val_true=y_v,
                y_val_prob=val_probs,
                config=self.config.threshold,
            )
            selected_th = th_search.selected_threshold

        # 2. Test Set Evaluation (untouched X_test)
        test_probs = model.predict_proba(X_t) if supports_proba else None
        default_test_preds = model.predict(X_t)
        selected_test_preds = (
            apply_threshold(test_probs, threshold=selected_th)
            if test_probs is not None
            else default_test_preds
        )

        # 3. Compute Metrics
        default_metrics = compute_binary_metrics(y_true=y_t, y_pred=default_test_preds, y_prob=test_probs)
        selected_metrics = compute_binary_metrics(y_true=y_t, y_pred=selected_test_preds, y_prob=test_probs)

        # 4. Attack Slice Evaluation
        slice_metrics = compute_slice_metrics(
            y_true=y_t,
            y_pred=selected_test_preds,
            labels=labels_test,
        )

        # 5. Inference Latency
        latency = measure_inference_latency(model, X_t, iterations=30)

        # 6. Quality Gate
        qg_decision = evaluate_quality_gates(
            metrics=selected_metrics,
            latency_ms=latency.p95_ms,
            config=self.config.quality_gates,
        )

        # 7. Track in MLflow if active
        if self.tracker and self.tracker.enabled:
            self.tracker.log_params({
                "eval_model_name": model_name,
                "selected_threshold": selected_th,
                "threshold_objective": self.config.threshold.objective,
                "quality_gate_passed": qg_decision.passed,
            })
            # Log selected test metrics
            eval_metrics_to_log = {f"test_{k}": v for k, v in selected_metrics.items() if isinstance(v, (int, float))}
            eval_metrics_to_log["test_latency_p95_ms"] = latency.p95_ms
            self.tracker.log_metrics(eval_metrics_to_log)

        return EvaluationResult(
            model_name=model_name,
            default_metrics=default_metrics,
            selected_threshold=selected_th,
            selected_threshold_metrics=selected_metrics,
            slice_metrics=slice_metrics,
            latency=latency,
            quality_gate=qg_decision,
        )
