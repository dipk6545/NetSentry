"""Tests for versioned decision threshold persistence and retrieval."""
from pathlib import Path
import pytest
from sklearn.dummy import DummyClassifier

from netsentry.registry.aliases import AliasManager
from netsentry.registry.client import RegistryClient
from netsentry.registry.registration import ModelRegistrar
from netsentry.training.tracking import MLflowTracker


def test_register_and_retrieve_decision_threshold(tmp_path: Path):
    db_path = tmp_path / "threshold_test.db"
    uri = f"sqlite:///{db_path.as_posix()}"
    tracker = MLflowTracker(experiment_name="test-threshold", tracking_uri=uri, enabled=True)

    model = DummyClassifier(strategy="constant", constant=1)
    model.fit([[0], [1]], [0, 1])

    with tracker.start_run(run_name="run_th") as run_id:
        tracker.log_model(model, artifact_path="model")

    client = RegistryClient(tracking_uri=uri)
    registrar = ModelRegistrar(client=client)

    # Register model with threshold 0.37
    res = registrar.register(
        model_name="NetSentryThresholdModel",
        run_id=run_id,
        optimal_threshold=0.37,
    )
    assert res.version == "1"

    # Retrieve model threshold by version
    th = client.get_model_threshold("NetSentryThresholdModel", "1")
    assert th == 0.37


def test_get_alias_threshold(tmp_path: Path):
    db_path = tmp_path / "threshold_alias_test.db"
    uri = f"sqlite:///{db_path.as_posix()}"
    tracker = MLflowTracker(experiment_name="test-threshold-alias", tracking_uri=uri, enabled=True)

    model = DummyClassifier(strategy="constant", constant=1)
    model.fit([[0], [1]], [0, 1])

    with tracker.start_run(run_name="run_champ") as run_id:
        tracker.log_model(model, artifact_path="model")

    client = RegistryClient(tracking_uri=uri)
    registrar = ModelRegistrar(client=client)
    res = registrar.register(
        model_name="NetSentryChampTh",
        run_id=run_id,
        optimal_threshold=0.37,
    )

    alias_mgr = AliasManager(client=client, model_name="NetSentryChampTh")
    alias_mgr.assign_champion(res.version)

    # Resolve via alias
    th = client.get_alias_threshold("NetSentryChampTh", "champion")
    assert th == 0.37


def test_version_isolation_for_thresholds(tmp_path: Path):
    db_path = tmp_path / "threshold_iso_test.db"
    uri = f"sqlite:///{db_path.as_posix()}"
    tracker = MLflowTracker(experiment_name="test-threshold-iso", tracking_uri=uri, enabled=True)

    model = DummyClassifier(strategy="constant", constant=1)
    model.fit([[0], [1]], [0, 1])

    with tracker.start_run(run_name="run_v1") as run1_id:
        tracker.log_model(model, artifact_path="model")
    with tracker.start_run(run_name="run_v2") as run2_id:
        tracker.log_model(model, artifact_path="model")

    client = RegistryClient(tracking_uri=uri)
    registrar = ModelRegistrar(client=client)

    res1 = registrar.register("NetSentryIso", run_id=run1_id, optimal_threshold=0.37)
    res2 = registrar.register("NetSentryIso", run_id=run2_id, optimal_threshold=0.42)

    assert client.get_model_threshold("NetSentryIso", res1.version) == 0.37
    assert client.get_model_threshold("NetSentryIso", res2.version) == 0.42


def test_missing_threshold_raises_clear_error(tmp_path: Path):
    db_path = tmp_path / "threshold_missing_test.db"
    uri = f"sqlite:///{db_path.as_posix()}"
    tracker = MLflowTracker(experiment_name="test-threshold-missing", tracking_uri=uri, enabled=True)

    model = DummyClassifier(strategy="constant", constant=1)
    model.fit([[0], [1]], [0, 1])

    with tracker.start_run(run_name="run_noth") as run_id:
        tracker.log_model(model, artifact_path="model")

    client = RegistryClient(tracking_uri=uri)
    registrar = ModelRegistrar(client=client)

    # Register without threshold
    res = registrar.register("NetSentryNoTh", run_id=run_id)

    with pytest.raises(ValueError, match="does not have a 'decision_threshold' tag"):
        client.get_model_threshold("NetSentryNoTh", res.version)

    with pytest.raises(ValueError, match="No model version found for alias 'champion'"):
        client.get_alias_threshold("NetSentryNoTh", "champion")
