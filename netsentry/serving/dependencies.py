"""
Serving Dependencies (`netsentry.serving.dependencies`).
--------------------------------------------------------
FastAPI dependency injection for Predictor and ModelLoader.
"""

from fastapi import HTTPException, Request, status
from netsentry.serving.predictor import Predictor


def get_predictor(request: Request) -> Predictor:
    """Dependency injecting Predictor initialized with the in-memory ChampionModel."""
    loader = getattr(request.app.state, "model_loader", None)
    if loader is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model loader is uninitialized.",
        )
    try:
        champ = loader.champion_model
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    return Predictor(champion_model=champ)
