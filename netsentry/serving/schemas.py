"""
Serving Request and Response Schemas (`netsentry.serving.schemas`).
-------------------------------------------------------------------
Preserves the exact request/response contract of NetSentry API.
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input payload with network flow features."""
    features: Dict[str, float] = Field(..., description="Dictionary mapping feature names to numerical values.")


class PredictionResponse(BaseModel):
    """Binary threat detection outcome and classification metadata."""
    prediction: int = Field(..., description="0 for BENIGN, 1 for ATTACK.")
    probability: Optional[float] = Field(None, description="Calibrated threat probability of ATTACK class.")
    model_version: Optional[str] = Field(None, description="Active champion model version producing inference.")
