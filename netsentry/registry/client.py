"""
MLflow Model Registry Client Abstraction (`netsentry.registry.client`).
----------------------------------------------------------------------
Wraps the MLflow Tracking and Model Registry APIs to provide typed operations
for model registration, version query, alias management, and artifact retrieval.
"""

from typing import Any, Dict, List, Optional
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion


class RegistryClient:
    """High-level abstraction over the MLflow Model Registry."""

    def __init__(self, tracking_uri: Optional[str] = None):
        self.tracking_uri = tracking_uri
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def register_model_version(
        self,
        name: str,
        source: str,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> ModelVersion:
        """Registers a new model version from an MLflow artifact URI."""
        # Ensure registered model entity exists
        try:
            self.client.create_registered_model(name=name, description=description)
        except Exception:
            # Model already exists
            pass

        mv = self.client.create_model_version(
            name=name,
            source=source,
            run_id=run_id,
            tags=tags,
            description=description,
        )
        return mv

    def get_latest_versions(self, name: str) -> List[ModelVersion]:
        """Returns all registered versions for a model."""
        return self.client.search_model_versions(f"name = '{name}'")

    def get_model_version(self, name: str, version: str) -> ModelVersion:
        """Retrieves a specific version record."""
        return self.client.get_model_version(name=name, version=version)

    def get_model_version_by_alias(self, name: str, alias: str) -> Optional[ModelVersion]:
        """Retrieves model version associated with a lifecycle alias (e.g. @champion)."""
        try:
            return self.client.get_model_version_by_alias(name=name, alias=alias)
        except Exception:
            return None

    def set_model_alias(self, name: str, alias: str, version: str) -> None:
        """Sets or updates a model alias pointing to the specified version."""
        self.client.set_registered_model_alias(name=name, alias=alias, version=version)

    def delete_model_alias(self, name: str, alias: str) -> None:
        """Deletes a model alias if present."""
        try:
            self.client.delete_registered_model_alias(name=name, alias=alias)
        except Exception:
            pass

    def load_model_by_alias(self, name: str, alias: str) -> Any:
        """Loads the Python model pipeline artifact by alias (e.g. models:/NetSentry@champion)."""
        model_uri = f"models:/{name}@{alias}"
        return mlflow.sklearn.load_model(model_uri)

    def load_model_by_version(self, name: str, version: str) -> Any:
        """Loads the Python model pipeline artifact by explicit version number."""
        model_uri = f"models:/{name}/{version}"
        return mlflow.sklearn.load_model(model_uri)
