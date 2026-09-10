"""Tests for ModelRegistrar and RegistryClient."""
from pathlib import Path
import pytest
from sklearn.dummy import DummyClassifier

from netsentry.registry.client import RegistryClient
from netsentry.registry.registration import ModelRegistrar
from netsentry.training.tracking import MLflowTracker


def test_model_registration(tmp_path: Path):
    db_path = tmp_path / "registry_test.db"
    uri = f"sqlite:///{db_path.as_posix()}"
    tracker = MLflowTracker(experiment_name="test-reg-exp", tracking_uri=uri, enabled=True)

    # 1. Log an initial model artifact in an MLflow run
    model = DummyClassifier(strategy="constant", constant=1)
    model.fit([[0], [1]], [0, 1])

    with tracker.start_run(run_name="dummy_run") as run_id:
        tracker.log_model(model, artifact_path="model")

    # 2. Register via ModelRegistrar
    client = RegistryClient(tracking_uri=uri)
    registrar = ModelRegistrar(client=client)

    result = registrar.register(
        model_name="NetSentryTest",
        run_id=run_id,
        metadata={"data_version": "v1.0", "threshold": 0.35},
    )

    assert result.model_name == "NetSentryTest"
    assert result.version == "1"
    assert result.tags["data_version"] == "v1.0"
    assert "git.commit_sha" in result.tags
