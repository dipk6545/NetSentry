"""
Health & Readiness Endpoints (`netsentry.serving.health`).
---------------------------------------------------------
Provides liveness and readiness probes inspecting ChampionModel status.
"""

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health(request: Request):
    """Basic service liveness check."""
    return {"status": "ok"}


@router.get("/ready")
async def readiness(request: Request):
    """
    Readiness probe: validates that a champion model, version,
    and decision threshold are successfully loaded in memory.
    """
    loader = getattr(request.app.state, "model_loader", None)
    if loader is None or not hasattr(loader, "_champion_model") or loader._champion_model is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "NOT_READY", "reason": "Champion model not loaded."},
        )

    champ = loader.champion_model
    return {
        "status": "READY",
        "model_name": champ.model_name,
        "model_version": champ.version,
        "decision_threshold": champ.threshold,
    }
