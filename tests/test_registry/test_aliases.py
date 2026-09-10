"""Tests for AliasManager."""
from pathlib import Path
import pytest
from sklearn.dummy import DummyClassifier

from netsentry.registry.aliases import AliasManager
from netsentry.registry.client import RegistryClient
from netsentry.registry.registration import ModelRegistrar
from netsentry.training.tracking import MLflowTracker


def test_alias_management(tmp_path: Path):
    db_path = tmp_path / "alias_test.db"
    uri = f"sqlite:///{db_path.as_posix()}"
    tracker = MLflowTracker(experiment_name="test-alias-exp", tracking_uri=uri, enabled=True)

    model = DummyClassifier(strategy="constant", constant=1)
    model.fit([[0], [1]], [0, 1])

    with tracker.start_run(run_name="run1") as run1_id:
        tracker.log_model(model, artifact_path="model")
    with tracker.start_run(run_name="run2") as run2_id:
        tracker.log_model(model, artifact_path="model")

    client = RegistryClient(tracking_uri=uri)
    registrar = ModelRegistrar(client=client)
    res1 = registrar.register("NetSentryAliasTest", run_id=run1_id)
    res2 = registrar.register("NetSentryAliasTest", run_id=run2_id)

    alias_mgr = AliasManager(client=client, model_name="NetSentryAliasTest")

    # Initially no aliases
    assert alias_mgr.get_champion_version() is None
    assert alias_mgr.get_challenger_version() is None

    # Assign champion to v1
    alias_mgr.assign_champion("1")
    assert alias_mgr.get_champion_version() == "1"

    # Assign challenger to v2
    alias_mgr.assign_challenger("2")
    assert alias_mgr.get_challenger_version() == "2"

    # Promote challenger to champion
    alias_mgr.promote_challenger_to_champion("2")
    assert alias_mgr.get_champion_version() == "2"
    assert alias_mgr.get_challenger_version() is None  # Challenger cleared
