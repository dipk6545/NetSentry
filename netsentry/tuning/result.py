"""
Tuning Results and Baseline Selection Contract (`netsentry.tuning.result`).
--------------------------------------------------------------------------
Data structures for comparing baseline models and holding tuning outputs.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import optuna


@dataclass(frozen=True)
class BaselineModelResult:
    """Individual baseline evaluation result."""
    model_name: str
    run_id: Optional[str]
    validation_metrics: Dict[str, float]
    selection_metric: str
    selection_score: float
    selected: bool = False


@dataclass(frozen=True)
class BaselineComparisonReport:
    """Structured report ranking all candidate baselines by validation performance."""
    results: List[BaselineModelResult]
    best_baseline: BaselineModelResult
    selection_metric: str


@dataclass(frozen=True)
class TuningResult:
    """Encapsulates Optuna study results, best parameters, and the retrained final candidate."""
    study: optuna.Study
    best_params: Dict[str, Any]
    best_score: float
    tuned_model_config_path: str
    final_model: Any
    retrained_on_train_val: bool = True
