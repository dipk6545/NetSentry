"""
Model Interface and Protocol (`netsentry.models.interface`).
------------------------------------------------------------
Defines the standard contract for estimators used across NetSentry training,
evaluation, and inference microservices.
"""

from typing import Protocol, Any, Optional, runtime_checkable
import numpy as np


@runtime_checkable
class ModelInterface(Protocol):
    """
    Standard contract expected by the training and evaluation pipelines.
    Guarantees fit and predict.
    """

    def fit(self, X: Any, y: Any) -> "ModelInterface":
        """Fits the preprocessor and estimator on training data."""
        ...

    def predict(self, X: Any) -> np.ndarray:
        """Predicts binary classification labels."""
        ...


@runtime_checkable
class ProbabilisticModelInterface(ModelInterface, Protocol):
    """Capability protocol for models providing calibrated class probabilities."""

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predicts class probabilities."""
        ...


def has_predict_proba(model: Any) -> bool:
    """Utility to safely check if a model or pipeline supports predict_proba."""
    return hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba"))
