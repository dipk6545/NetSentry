"""Tests for end-to-end ModelLifecycleManager."""
from pathlib import Path
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from netsentry.registry.client import RegistryClient
from netsentry.registry.config import RegistryConfig, PromotionConfig, PromotionRule
from netsentry.registry.lifecycle import ModelLifecycleManager, LifecyclePromotionResult
from netsentry.training.tracking import MLflowTracker


def test_lifecycle_first_release_and_challenger_promotion(tmp_path: Path):
    db_path = tmp_path / "lifecycle_test.db"
    uri = f"sqlite:///{db_path.as_posix()}"
    tracker = MLflowTracker(experiment_name="test-lifecycle", tracking_uri=uri, enabled=True)
    client = RegistryClient(tracking_uri=uri)

    np.random.seed(42)
    X_val = np.random.randn(40, 4)
    y_val = np.random.choice([0, 1], size=40)
    X_test = np.random.randn(40, 4)
    y_test = np.random.choice([0, 1], size=40)
    labels_test = ["BENIGN" if y == 0 else "Bot" for y in y_test]

    # Models
    model_v1 = LogisticRegression(C=0.01)
    model_v1.fit(X_val, y_val)
    with tracker.start_run(run_name="run_v1") as run1_id:
        tracker.log_model(model_v1, artifact_path="model")

    # Liberal test config
    cfg = RegistryConfig(
        model_name="NetSentryLifecycleTest",
        promotion=PromotionConfig(
            require_quality_gates=False,
            require_challenger_better=True,
            rules={"recall": PromotionRule(minimum=0.01, improvement_required=False)},
        ),
    )

    lifecycle = ModelLifecycleManager(config=cfg, registry_client=client, tracker=tracker)

    # 1. First release promotion -> becomes Champion v1
    res1 = lifecycle.promote_candidate(
        candidate_model=model_v1,
        run_id=run1_id,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        labels_test=labels_test,
    )

    assert isinstance(res1, LifecyclePromotionResult)
    assert res1.registered_version == "1"
    assert res1.is_new_champion is True
    assert res1.champion_version_after == "1"
    assert lifecycle.alias_manager.get_champion_version() == "1"
