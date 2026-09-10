"""
Evaluation Configuration Schema (`netsentry.evaluation.config`).
--------------------------------------------------------------
Defines settings for final test evaluation, validation threshold search,
and quality gates.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


class EvaluationConfigValidationError(ValueError):
    """Raised when evaluation configuration is invalid."""
    pass


@dataclass(frozen=True)
class ThresholdConfig:
    """Settings for validation-set decision threshold search."""
    enabled: bool = True
    objective: str = "f1"  # "f1", "recall", "precision"
    min_precision: Optional[float] = None  # optional constraint when optimizing recall

    def __post_init__(self):
        valid_objs = {"f1", "recall", "precision"}
        if self.objective not in valid_objs:
            raise EvaluationConfigValidationError(
                f"Invalid threshold objective '{self.objective}'. Allowed: {valid_objs}"
            )


@dataclass(frozen=True)
class QualityGatesConfig:
    """Production quality gate thresholds."""
    min_recall: Optional[float] = 0.90
    min_f1: Optional[float] = 0.85
    min_roc_auc: Optional[float] = 0.90
    max_latency_ms: Optional[float] = 20.0


@dataclass(frozen=True)
class EvaluationConfig:
    """Root evaluation configuration."""
    primary_metric: str = "f1"
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    quality_gates: QualityGatesConfig = field(default_factory=QualityGatesConfig)


def load_evaluation_config(config_source: Union[str, Path, Dict[str, Any]]) -> EvaluationConfig:
    """Loads and validates an EvaluationConfig from YAML file or dictionary."""
    if isinstance(config_source, (str, Path)):
        path = Path(config_source)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation config file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)
    elif isinstance(config_source, dict):
        raw_cfg = config_source
    else:
        raise TypeError(f"Expected file path or dict, got {type(config_source)}")

    if not isinstance(raw_cfg, dict):
        raise EvaluationConfigValidationError("Evaluation config must be a mapping/dict.")

    eval_raw = raw_cfg.get("evaluation", {})
    primary_metric = str(eval_raw.get("primary_metric", "f1")).lower()

    th_raw = raw_cfg.get("threshold", {})
    threshold_cfg = ThresholdConfig(
        enabled=bool(th_raw.get("enabled", True)),
        objective=str(th_raw.get("objective", "f1")).lower(),
        min_precision=float(th_raw["min_precision"]) if "min_precision" in th_raw else None,
    )

    qg_raw = raw_cfg.get("quality_gates", {})
    recall_gate = qg_raw.get("recall", {}).get("min")
    f1_gate = qg_raw.get("f1", {}).get("min")
    roc_gate = qg_raw.get("roc_auc", {}).get("min")
    lat_gate = qg_raw.get("latency_ms", {}).get("max")

    qg_cfg = QualityGatesConfig(
        min_recall=float(recall_gate) if recall_gate is not None else None,
        min_f1=float(f1_gate) if f1_gate is not None else None,
        min_roc_auc=float(roc_gate) if roc_gate is not None else None,
        max_latency_ms=float(lat_gate) if lat_gate is not None else None,
    )

    return EvaluationConfig(
        primary_metric=primary_metric,
        threshold=threshold_cfg,
        quality_gates=qg_cfg,
    )
