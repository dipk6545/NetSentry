"""
Model Factory Module (`netsentry.models.factory`).
--------------------------------------------------
Instantiates models and composes them with appropriate preprocessing pipelines
strictly according to declarative ModelConfig without hardcoded algorithm branches.
"""

import importlib
from typing import Any
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from netsentry.models.config import ModelConfig, load_model_config
from netsentry.models.preprocessing import build_preprocessor


def _resolve_class(class_path: str) -> Any:
    """
    Dynamically imports a class from its fully qualified dotted path.
    Example: 'sklearn.linear_model.LogisticRegression' -> LogisticRegression class
    """
    if "." not in class_path:
        raise ValueError(f"Invalid class path: '{class_path}'. Expected 'module.submodule.ClassName'")

    module_name, class_name = class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_name}' for model: {e}") from e

    try:
        cls = getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(f"Module '{module_name}' has no class named '{class_name}': {e}") from e

    return cls


def create_model(config: ModelConfig) -> Any:
    """
    Constructs an estimator composed with optional preprocessing as a unified object.
    
    If scaling/preprocessing is specified:
        returns Pipeline([('preprocessor', transformer), ('estimator', estimator)])
    Else:
        returns Pipeline([('estimator', estimator)]) or estimator directly.
        (Using Pipeline uniformly guarantees consistent fit/predict delegation).
    """
    # 1. Resolve and instantiate the estimator with parameters
    estimator_cls = _resolve_class(config.implementation.class_path)
    estimator = estimator_cls(**config.parameters)

    # 2. Build preprocessing transformer (if any)
    transformer = build_preprocessor(config.preprocessing)

    # 3. Compose pipeline
    steps = []
    if transformer is not None:
        steps.append(("preprocessor", transformer))
    steps.append(("estimator", estimator))

    return Pipeline(steps=steps)
