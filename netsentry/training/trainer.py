"""
Model-Agnostic Trainer (`netsentry.training.trainer`).
------------------------------------------------------
Coordinates fitting, validation evaluation, metric calculation,
and MLflow tracking for any model conforming to ModelInterface.
"""

from dataclasses import dataclass
import time
from typing import Any, Dict, Optional, Union
import numpy as np
import polars as pl

from netsentry.models.config import ModelConfig
from netsentry.models.interface import has_predict_proba
from netsentry.training.config import TrainingConfig
from netsentry.training.metrics import calculate_metrics
from netsentry.training.tracking import MLflowTracker


@dataclass(frozen=True)
class TrainingResult:
    """Structured artifact summarizing model training execution and metrics."""
    model: Any
    metrics: Dict[str, float]
    training_metadata: Dict[str, Any]
    run_id: Optional[str] = None


class Trainer:
    """
    Orchestrates the training lifecycle for an estimator.
    Completely decoupled from specific ML algorithms.
    """

    def __init__(
        self,
        model: Any,
        model_config: ModelConfig,
        training_config: Optional[TrainingConfig] = None,
        tracker: Optional[MLflowTracker] = None,
    ):
        self.model = model
        self.model_config = model_config
        self.training_config = training_config or TrainingConfig()
        self.tracker = tracker or MLflowTracker(
            experiment_name=self.training_config.experiment_name,
            tracking_uri=self.training_config.tracking.tracking_uri,
            enabled=self.training_config.tracking.enabled,
        )

    def train(
        self,
        X_train: Union[np.ndarray, pl.DataFrame],
        y_train: Union[np.ndarray, pl.Series],
        X_val: Union[np.ndarray, pl.DataFrame],
        y_val: Union[np.ndarray, pl.Series],
        run_name: Optional[str] = None,
    ) -> TrainingResult:
        """
        Fits the model pipeline, computes validation metrics, logs with tracker,
        and returns a TrainingResult.
        """
        # Convert polars to numpy if needed
        X_tr = X_train.to_numpy() if isinstance(X_train, pl.DataFrame) else np.asarray(X_train)
        y_tr = y_train.to_numpy() if isinstance(y_train, pl.Series) else np.asarray(y_train)
        X_v = X_val.to_numpy() if isinstance(X_val, pl.DataFrame) else np.asarray(X_val)
        y_v = y_val.to_numpy() if isinstance(y_val, pl.Series) else np.asarray(y_val)

        run_display_name = run_name or f"{self.model_config.name}"

        with self.tracker.start_run(run_name=run_display_name) as run_id:
            # 1. Fit the complete model object
            start_time = time.perf_counter()
            self.model.fit(X_tr, y_tr)
            fit_duration = time.perf_counter() - start_time

            # 2. Validation predictions
            val_preds = self.model.predict(X_v)
            val_probs = None
            if has_predict_proba(self.model):
                val_probs = self.model.predict_proba(X_v)

            # 3. Calculate validation metrics
            metrics = calculate_metrics(y_true=y_v, y_pred=val_preds, y_prob=val_probs)

            # 4. Metadata
            metadata: Dict[str, Any] = {
                "model_name": self.model_config.name,
                "class_path": self.model_config.implementation.class_path,
                "scaling": self.model_config.preprocessing.scaling,
                "training_rows": int(len(X_tr)),
                "validation_rows": int(len(X_v)),
                "feature_count": int(X_tr.shape[1]),
                "training_duration_seconds": round(fit_duration, 4),
            }

            # 5. MLflow logging
            if self.tracker.enabled:
                # Log model parameters + training config metadata
                log_params = {
                    **metadata,
                    **{f"param_{k}": v for k, v in self.model_config.parameters.items()},
                    "random_seed": self.training_config.random_seed,
                }
                self.tracker.log_params(log_params)
                self.tracker.log_metrics(metrics)
                self.tracker.log_model(self.model, artifact_path="model")

            return TrainingResult(
                model=self.model,
                metrics=metrics,
                training_metadata=metadata,
                run_id=run_id,
            )
