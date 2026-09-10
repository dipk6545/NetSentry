"""
Champion-Challenger Comparator (`netsentry.registry.comparator`).
-----------------------------------------------------------------
Evaluates both Champion and Challenger models side-by-side using the EXACT SAME
evaluation dataset, frozen threshold protocol, and latency benchmarks.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
import polars as pl

from netsentry.evaluation.config import EvaluationConfig
from netsentry.evaluation.evaluator import Evaluator, EvaluationResult
from netsentry.registry.config import ComparisonConfig


@dataclass(frozen=True)
class ModelComparisonMetrics:
    """Side-by-side operational comparison metrics between two models."""
    champion_metrics: Dict[str, float]
    challenger_metrics: Dict[str, float]
    deltas: Dict[str, float]  # challenger - champion
    primary_metric: str


class ChampionChallengerComparator:
    """Compares Champion vs Challenger models on a unified test split."""

    def __init__(
        self,
        eval_config: Optional[EvaluationConfig] = None,
        comp_config: Optional[ComparisonConfig] = None,
    ):
        self.eval_config = eval_config or EvaluationConfig()
        self.comp_config = comp_config or ComparisonConfig()
        self.evaluator = Evaluator(config=self.eval_config)

    def compare(
        self,
        champion_model: Any,
        challenger_model: Any,
        X_val: Union[np.ndarray, pl.DataFrame],
        y_val: Union[np.ndarray, pl.Series],
        X_test: Union[np.ndarray, pl.DataFrame],
        y_test: Union[np.ndarray, pl.Series],
        labels_test: Union[np.ndarray, pl.Series, list],
    ) -> ModelComparisonMetrics:
        """
        Executes uniform evaluation across Champion and Challenger.
        """
        # Evaluate Champion
        champ_res: EvaluationResult = self.evaluator.evaluate(
            model=champion_model,
            model_name="champion",
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            labels_test=labels_test,
        )

        # Evaluate Challenger
        chall_res: EvaluationResult = self.evaluator.evaluate(
            model=challenger_model,
            model_name="challenger",
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            labels_test=labels_test,
        )

        # Merge core performance metrics with latency
        champ_m = {**champ_res.selected_threshold_metrics, "latency_p95_ms": champ_res.latency.p95_ms}
        chall_m = {**chall_res.selected_threshold_metrics, "latency_p95_ms": chall_res.latency.p95_ms}

        # Calculate deltas for all tracked metrics
        deltas: Dict[str, float] = {}
        all_metrics = set(champ_m.keys()).union(chall_m.keys())
        for k in all_metrics:
            c_val = champ_m.get(k, 0.0)
            ch_val = chall_m.get(k, 0.0)
            deltas[k] = round(float(ch_val - c_val), 4)

        return ModelComparisonMetrics(
            champion_metrics=champ_m,
            challenger_metrics=chall_m,
            deltas=deltas,
            primary_metric=self.comp_config.primary_metric,
        )
