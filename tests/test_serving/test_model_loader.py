"""Tests for ModelLoader loading champion model and matching threshold."""
from pathlib import Path
import pytest
from sklearn.linear_model import LogisticRegression

from netsentry.registry.aliases import AliasManager
from netsentry.registry.client import RegistryClient
from netsentry.registry.registration import ModelRegistrar
from netsentry.serving.config import ServingConfig
from netsentry.serving.model_loader import ModelLoader, ChampionModel
from netsentry.training.tracking import MLflowTracker


@pytest.fixture
def mock_registry_with_versions(tmp_path: Path):
    db_path = tmp_path / "serve_test.db"
    uri = f"sqlite:///{db_path.as_posix()}"
    tracker = MLflowTracker(experiment_name="test-serving", tracking_uri=uri, enabled=True)
    client = RegistryClient(tracking_uri=uri)
    registrar = ModelRegistrar(client=client)
    alias_mgr = AliasManager(client=client, model_name="NetSentry")

    # Train and register v1 (threshold 0.37)
    m1 = LogisticRegression(C=0.1)
    m1.fit([[0, 0], [1, 1]], [0, 1])
    with tracker.start_run(run_name="run_v1") as r1_id:
        tracker.log_model(m1, artifact_path="model")
    res1 = registrar.register("NetSentry", run_id=r1_id, optimal_threshold=0.37)

    # Train and register v2 (threshold 0.42)
    m2 = LogisticRegression(C=1.0)
    m2.fit([[0, 0], [1, 1]], [0, 1])
    with tracker.start_run(run_name="run_v2") as r2_id:
        tracker.log_model(m2, artifact_path="model")
    res2 = registrar.register("NetSentry", run_id=r2_id, optimal_threshold=0.42)

    # Set @champion -> v1
    alias_mgr.assign_champion(res1.version)

    return uri, client, alias_mgr, res1.version, res2.version


def test_champion_model_and_threshold_loaded(mock_registry_with_versions):
    uri, client, alias_mgr, v1, v2 = mock_registry_with_versions
    cfg = ServingConfig(model_name="NetSentry", champion_alias="champion", tracking_uri=uri)
    loader = ModelLoader(config=cfg, client=client)

    champ: ChampionModel = loader.load()
    assert champ.version == v1
    assert champ.threshold == 0.37
    assert champ.model is not None


def test_champion_change_loads_new_version_and_new_threshold(mock_registry_with_versions):
    uri, client, alias_mgr, v1, v2 = mock_registry_with_versions
    cfg = ServingConfig(model_name="NetSentry", champion_alias="champion", tracking_uri=uri)
    loader = ModelLoader(config=cfg, client=client)

    # Initially v1
    c1 = loader.load()
    assert c1.version == v1
    assert c1.threshold == 0.37

    # Update @champion -> v2
    alias_mgr.assign_champion(v2)

    # Reload
    c2 = loader.load()
    assert c2.version == v2
    assert c2.threshold == 0.42  # Matching threshold for v2, NOT 0.37!


def test_missing_champion_fails(tmp_path: Path):
    db_path = tmp_path / "empty_registry.db"
    uri = f"sqlite:///{db_path.as_posix()}"
    cfg = ServingConfig(model_name="NetSentryEmpty", champion_alias="champion", tracking_uri=uri)
    loader = ModelLoader(config=cfg)

    with pytest.raises(RuntimeError, match="No model version found for alias '@champion'"):
        loader.load()
