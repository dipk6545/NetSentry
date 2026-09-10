"""
Training Configuration Schema (`netsentry.training.config`).
-----------------------------------------------------------
Parses and validates training session and MLflow tracking settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


class TrainingConfigValidationError(ValueError):
    """Raised when training configuration fails validation."""
    pass


@dataclass(frozen=True)
class TrackingConfig:
    """Settings for experiment tracking with MLflow."""
    enabled: bool = True
    tracking_uri: Optional[str] = "sqlite:///mlruns.db"


@dataclass(frozen=True)
class EvaluationConfig:
    """Primary metrics and evaluation settings."""
    primary_metric: str = "roc_auc"

    def __post_init__(self):
        valid_metrics = {"roc_auc", "f1", "precision", "recall", "accuracy"}
        if self.primary_metric.lower() not in valid_metrics:
            raise TrainingConfigValidationError(
                f"Invalid primary_metric '{self.primary_metric}'. Allowed: {valid_metrics}"
            )


@dataclass(frozen=True)
class TrainingConfig:
    """Root training configuration."""
    experiment_name: str = "netsentry-model-training"
    random_seed: int = 42
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self):
        if not self.experiment_name or not isinstance(self.experiment_name, str):
            raise TrainingConfigValidationError("experiment_name must be a non-empty string.")


def load_training_config(config_source: Union[str, Path, Dict[str, Any]]) -> TrainingConfig:
    """Loads and validates a TrainingConfig from a YAML file path or dictionary."""
    if isinstance(config_source, (str, Path)):
        path = Path(config_source)
        if not path.exists():
            raise FileNotFoundError(f"Training config file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)
    elif isinstance(config_source, dict):
        raw_cfg = config_source
    else:
        raise TypeError(f"Expected file path or dict, got {type(config_source)}")

    if not isinstance(raw_cfg, dict):
        raise TrainingConfigValidationError("Training config root must be a dictionary.")

    exp_name = str(raw_cfg.get("experiment_name", "netsentry-model-training"))
    seed = int(raw_cfg.get("random_seed", 42))

    tracking_raw = raw_cfg.get("tracking", {})
    tracking_cfg = TrackingConfig(
        enabled=bool(tracking_raw.get("enabled", True)),
        tracking_uri=tracking_raw.get("tracking_uri", "file:./mlruns"),
    )

    eval_raw = raw_cfg.get("evaluation", {})
    eval_cfg = EvaluationConfig(
        primary_metric=str(eval_raw.get("primary_metric", "roc_auc")).lower()
    )

    return TrainingConfig(
        experiment_name=exp_name,
        random_seed=seed,
        tracking=tracking_cfg,
        evaluation=eval_cfg,
    )
