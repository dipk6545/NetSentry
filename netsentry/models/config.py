"""
Model Configuration Schema (`netsentry.models.config`).
-------------------------------------------------------
Validates and parses declarative YAML model configs into typed dataclasses.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


class ModelConfigValidationError(ValueError):
    """Raised when model configuration validation fails."""
    pass


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configures preprocessing steps (e.g., scaling) applied before the estimator."""
    scaling: str = "none"  # "none", "standard", "minmax", "robust"

    def __post_init__(self):
        valid_scalings = {"none", "standard", "minmax", "robust"}
        if self.scaling.lower() not in valid_scalings:
            raise ModelConfigValidationError(
                f"Invalid scaling strategy '{self.scaling}'. Allowed: {valid_scalings}"
            )


@dataclass(frozen=True)
class ImplementationConfig:
    """Configures dynamic class loading path for the estimator."""
    class_path: str

    def __post_init__(self):
        if not self.class_path or "." not in self.class_path:
            raise ModelConfigValidationError(
                f"Invalid class_path '{self.class_path}'. Expected full module path (e.g. 'sklearn.linear_model.LogisticRegression')."
            )


@dataclass(frozen=True)
class ModelConfig:
    """Validated model definition containing implementation, preprocessing, and parameters."""
    name: str
    implementation: ImplementationConfig
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ModelConfigValidationError("Model config 'name' must be a non-empty string.")


def load_model_config(config_source: Union[str, Path, Dict[str, Any]]) -> ModelConfig:
    """
    Loads and validates a ModelConfig from a YAML file path or dictionary.
    """
    if isinstance(config_source, (str, Path)):
        path = Path(config_source)
        if not path.exists():
            raise FileNotFoundError(f"Model config file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)
    elif isinstance(config_source, dict):
        raw_cfg = config_source
    else:
        raise TypeError(f"Expected file path or dict, got {type(config_source)}")

    if not isinstance(raw_cfg, dict):
        raise ModelConfigValidationError("Model config root must be a dictionary/mapping.")

    if "name" not in raw_cfg:
        raise ModelConfigValidationError("Missing required field 'name' in model configuration.")
    if "implementation" not in raw_cfg or not isinstance(raw_cfg["implementation"], dict):
        raise ModelConfigValidationError("Missing required mapping 'implementation' in model configuration.")
    if "class_path" not in raw_cfg["implementation"]:
        raise ModelConfigValidationError("Missing required field 'class_path' in 'implementation'.")

    # Parse implementation
    impl = ImplementationConfig(class_path=str(raw_cfg["implementation"]["class_path"]))

    # Parse preprocessing
    prep_raw = raw_cfg.get("preprocessing", {})
    if not isinstance(prep_raw, dict):
        raise ModelConfigValidationError("'preprocessing' must be a mapping if provided.")
    prep = PreprocessingConfig(scaling=str(prep_raw.get("scaling", "none")).lower())

    # Parse parameters
    params = raw_cfg.get("parameters", {})
    if not isinstance(params, dict):
        raise ModelConfigValidationError("'parameters' must be a mapping of hyperparameter kwargs.")

    return ModelConfig(
        name=str(raw_cfg["name"]),
        implementation=impl,
        preprocessing=prep,
        parameters=params,
    )
