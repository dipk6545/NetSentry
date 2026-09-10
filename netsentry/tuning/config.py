"""
Tuning Configuration Schema (`netsentry.tuning.config`).
--------------------------------------------------------
Defines dataclasses for parsing and validating hyperparameter search spaces,
tuning targets, and trial parameters from YAML.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


class TuningConfigValidationError(ValueError):
    """Raised when tuning configuration validation fails."""
    pass


@dataclass(frozen=True)
class ParameterSearchSpace:
    """Defines the distribution and bounds for an individual hyperparameter."""
    name: str
    param_type: str  # "int", "float", "categorical"
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None
    log: bool = False
    choices: Optional[List[Any]] = None

    def __post_init__(self):
        valid_types = {"int", "float", "categorical"}
        if self.param_type not in valid_types:
            raise TuningConfigValidationError(
                f"Parameter '{self.name}' has invalid type '{self.param_type}'. Allowed: {valid_types}"
            )
        if self.param_type in {"int", "float"}:
            if self.low is None or self.high is None:
                raise TuningConfigValidationError(
                    f"Parameter '{self.name}' of type '{self.param_type}' must specify 'low' and 'high'."
                )
            if self.low > self.high:
                raise TuningConfigValidationError(
                    f"Parameter '{self.name}': low ({self.low}) cannot exceed high ({self.high})."
                )
        elif self.param_type == "categorical":
            if not self.choices or not isinstance(self.choices, list):
                raise TuningConfigValidationError(
                    f"Categorical parameter '{self.name}' must have a non-empty list of 'choices'."
                )


@dataclass(frozen=True)
class TuningSettings:
    """Optuna study settings."""
    n_trials: int = 30
    direction: str = "maximize"  # "maximize" or "minimize"
    metric: str = "roc_auc"
    timeout_seconds: Optional[int] = None

    def __post_init__(self):
        if self.direction not in {"maximize", "minimize"}:
            raise TuningConfigValidationError(
                f"Invalid direction '{self.direction}'. Must be 'maximize' or 'minimize'."
            )
        if self.n_trials <= 0:
            raise TuningConfigValidationError("n_trials must be positive.")


@dataclass(frozen=True)
class TuningConfig:
    """Complete tuning configuration linking model config to search spaces."""
    model_config_path: str
    tuning: TuningSettings
    search_space: Dict[str, ParameterSearchSpace]

    def __post_init__(self):
        if not self.search_space:
            raise TuningConfigValidationError("Search space cannot be empty.")


def load_tuning_config(config_source: Union[str, Path, Dict[str, Any]]) -> TuningConfig:
    """Loads and validates a TuningConfig from YAML file or dictionary."""
    if isinstance(config_source, (str, Path)):
        path = Path(config_source)
        if not path.exists():
            raise FileNotFoundError(f"Tuning config file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)
    elif isinstance(config_source, dict):
        raw_cfg = config_source
    else:
        raise TypeError(f"Expected file path or dict, got {type(config_source)}")

    if not isinstance(raw_cfg, dict):
        raise TuningConfigValidationError("Tuning config must be a mapping/dict.")

    if "model_config" not in raw_cfg:
        raise TuningConfigValidationError("Missing required field 'model_config'.")
    model_cfg_path = str(raw_cfg["model_config"])

    tuning_raw = raw_cfg.get("tuning", {})
    tuning_settings = TuningSettings(
        n_trials=int(tuning_raw.get("n_trials", 30)),
        direction=str(tuning_raw.get("direction", "maximize")).lower(),
        metric=str(tuning_raw.get("metric", "roc_auc")).lower(),
        timeout_seconds=tuning_raw.get("timeout_seconds"),
    )

    space_raw = raw_cfg.get("search_space", {})
    if not isinstance(space_raw, dict) or not space_raw:
        raise TuningConfigValidationError("Missing or invalid 'search_space' mapping.")

    search_space = {}
    for p_name, p_spec in space_raw.items():
        if not isinstance(p_spec, dict):
            raise TuningConfigValidationError(f"Search space for '{p_name}' must be a mapping.")
        p_type = str(p_spec.get("type", "float")).lower()
        search_space[p_name] = ParameterSearchSpace(
            name=p_name,
            param_type=p_type,
            low=float(p_spec["low"]) if "low" in p_spec else None,
            high=float(p_spec["high"]) if "high" in p_spec else None,
            step=float(p_spec["step"]) if "step" in p_spec else None,
            log=bool(p_spec.get("log", False)),
            choices=p_spec.get("choices"),
        )

    return TuningConfig(
        model_config_path=model_cfg_path,
        tuning=tuning_settings,
        search_space=search_space,
    )
