"""
MLflow Tracking Abstraction (`netsentry.training.tracking`).
-----------------------------------------------------------
Manages experiment initialization, run lifecycle, metric/parameter logging,
and model artifact recording without coupling training logic to MLflow.
"""

from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional
import mlflow
import mlflow.sklearn


class MLflowTracker:
    """Encapsulates MLflow experiment tracking operations."""

    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = "file:./mlruns", enabled: bool = True):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.enabled = enabled

        if self.enabled and self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)

    @contextmanager
    def start_run(self, run_name: Optional[str] = None) -> Generator[Optional[str], None, None]:
        """Context manager to scope an MLflow run safely."""
        if not self.enabled:
            yield None
            return

        with mlflow.start_run(run_name=run_name) as run:
            yield run.info.run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        """Logs configuration parameters and dataset metadata."""
        if not self.enabled:
            return

        # Flatten nested structures or complex objects into string/primitive representations
        flat_params = {}
        for k, v in params.items():
            if isinstance(v, (dict, list)):
                flat_params[k] = str(v)
            else:
                flat_params[k] = v

        mlflow.log_params(flat_params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Logs evaluation metrics."""
        if not self.enabled:
            return
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        """Logs the complete model object (preprocessor + estimator)."""
        if not self.enabled:
            return
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
