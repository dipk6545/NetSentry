"""
Model Registrar (`netsentry.registry.registration`).
----------------------------------------------------
Registers evaluated model artifacts from MLflow runs with rich audit metadata
and Git commit lineage into the Model Registry.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import subprocess
from typing import Any, Dict, Optional
from mlflow.entities.model_registry import ModelVersion

from netsentry.registry.client import RegistryClient


def get_git_commit_sha() -> str:
    """Retrieves current Git commit SHA, or 'unknown' if not in git."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class RegistrationResult:
    """Details of a newly registered model version."""
    model_name: str
    version: str
    run_id: str
    source_artifact_uri: str
    tags: Dict[str, str]
    created_at_utc: str


class ModelRegistrar:
    """Registers trained artifacts into the MLflow Model Registry."""

    def __init__(self, client: RegistryClient):
        self.client = client

    def register(
        self,
        model_name: str,
        run_id: str,
        artifact_path: str = "model",
        metadata: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> RegistrationResult:
        """
        Creates a registered model version linked to the MLflow run artifact.
        """
        source_uri = f"runs:/{run_id}/{artifact_path}"
        now_str = datetime.now(timezone.utc).isoformat()
        git_sha = get_git_commit_sha()

        tags: Dict[str, str] = {
            "mlflow.run_id": str(run_id),
            "git.commit_sha": git_sha,
            "registered_at_utc": now_str,
        }

        if metadata:
            for k, v in metadata.items():
                tags[str(k)] = str(v)

        desc = description or f"Registered from MLflow Run {run_id} at {now_str}"

        mv: ModelVersion = self.client.register_model_version(
            name=model_name,
            source=source_uri,
            run_id=run_id,
            tags=tags,
            description=desc,
        )

        return RegistrationResult(
            model_name=model_name,
            version=str(mv.version),
            run_id=run_id,
            source_artifact_uri=source_uri,
            tags=tags,
            created_at_utc=now_str,
        )
