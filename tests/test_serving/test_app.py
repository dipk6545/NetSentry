"""Tests for FastAPI Serving Endpoints."""
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from netsentry.registry.aliases import AliasManager
from netsentry.registry.client import RegistryClient
from netsentry.registry.registration import ModelRegistrar
from netsentry.serving.app import create_app
from netsentry.serving.config import ServingConfig
from netsentry.serving.model_loader import ModelLoader
from netsentry.training.tracking import MLflowTracker


@pytest.fixture
def test_client_and_registry(tmp_path: Path):
    db_path = tmp_path / "app_test.db"
    uri = f"sqlite:///{db_path.as_posix()}"
    tracker = MLflowTracker(experiment_name="app-test", tracking_uri=uri, enabled=True)
    client = RegistryClient(tracking_uri=uri)
    registrar = ModelRegistrar(client=client)
    alias_mgr = AliasManager(client=client, model_name="NetSentry")

    m = LogisticRegression()
    m.fit([[0, 0], [1, 1]], [0, 1])

    with tracker.start_run(run_name="app_run") as run_id:
        tracker.log_model(m, artifact_path="model")

    res = registrar.register("NetSentry", run_id=run_id, optimal_threshold=0.35)
    alias_mgr.assign_champion(res.version)

    cfg = ServingConfig(model_name="NetSentry", champion_alias="champion", tracking_uri=uri)
    loader = ModelLoader(config=cfg, client=client)
    app = create_app(config=cfg, model_loader=loader)

    # Lifespan trigger using TestClient context
    with TestClient(app) as test_client:
        yield test_client, alias_mgr, registrar, tracker, m


def test_root_endpoint(test_client_and_registry):
    client, _, _, _, _ = test_client_and_registry
    resp = client.get("/")
    assert resp.status_code == 200
    assert "message" in resp.json()


def test_health_endpoint(test_client_and_registry):
    client, _, _, _, _ = test_client_and_registry
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_endpoint(test_client_and_registry):
    client, _, _, _, _ = test_client_and_registry
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "READY"
    assert data["model_version"] == "1"
    assert data["decision_threshold"] == 0.35


def test_predict_single_flow(test_client_and_registry):
    client, _, _, _, _ = test_client_and_registry
    payload = {"features": {"f1": 1.0, "f2": 1.0}}
    resp = client.post("/v1/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert data["prediction"] in (0, 1)
    assert "probability" in data
    assert data["model_version"] == "1"


def test_hot_reload_endpoint(test_client_and_registry):
    client, alias_mgr, registrar, tracker, model = test_client_and_registry

    # Register v2 with threshold 0.48
    with tracker.start_run(run_name="v2_run") as r2_id:
        tracker.log_model(model, artifact_path="model")
    res2 = registrar.register("NetSentry", run_id=r2_id, optimal_threshold=0.48)

    # Promote to champion
    alias_mgr.assign_champion(res2.version)

    # Call hot reload
    resp = client.post("/v1/model/reload")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert data["model_version"] == "2"
    assert data["decision_threshold"] == 0.48

    # Ready probe reflects new state
    ready_resp = client.get("/ready")
    assert ready_resp.json()["model_version"] == "2"
    assert ready_resp.json()["decision_threshold"] == 0.48
