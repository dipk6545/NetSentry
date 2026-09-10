"""
Model Preprocessing Module (`netsentry.models.preprocessing`).
--------------------------------------------------------------
Builds sklearn preprocessing transformers based on ModelConfig.
Ensures feature scaling and transforms are isolated from domain feature engineering.
"""

from typing import Optional
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from netsentry.models.config import PreprocessingConfig, ModelConfigValidationError


def build_preprocessor(config: PreprocessingConfig) -> Optional[TransformerMixin]:
    """
    Constructs a scikit-learn transformer from PreprocessingConfig.
    Returns None if scaling is 'none'.
    """
    strategy = config.scaling.lower()

    if strategy == "none":
        return None
    elif strategy == "standard":
        return StandardScaler()
    elif strategy == "minmax":
        return MinMaxScaler()
    elif strategy == "robust":
        return RobustScaler()
    else:
        raise ModelConfigValidationError(f"Unsupported scaling strategy: '{strategy}'")
