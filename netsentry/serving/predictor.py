"""
Serving Predictor Service (`netsentry.serving.predictor`).
---------------------------------------------------------
Applies champion model inference using the champion model's frozen threshold.
Supports DataFrame and Dictionary feature inputs.
"""

from typing import Any, Dict, Tuple, Union
import numpy as np
import polars as pl

from netsentry.models.interface import has_predict_proba
from netsentry.serving.model_loader import ChampionModel


class Predictor:
    """Coordinates prediction execution using the loaded ChampionModel."""

    def __init__(self, champion_model: ChampionModel):
        self.champion = champion_model

    def predict(self, features: Union[Dict[str, float], pl.DataFrame, np.ndarray]) -> Tuple[int, float]:
        """
        Executes binary threat prediction.
        Returns (prediction, probability).
        """
        if isinstance(features, dict):
            # Model pipeline expects 2D structure
            X = pl.DataFrame([features]).to_numpy()
        elif isinstance(features, pl.DataFrame):
            X = features.to_numpy()
        else:
            X = np.asarray(features)
            if X.ndim == 1:
                X = X.reshape(1, -1)

        model = self.champion.model
        threshold = self.champion.threshold

        if has_predict_proba(model):
            probs = model.predict_proba(X)
            attack_prob = float(probs[0, 1]) if probs.ndim == 2 and probs.shape[1] == 2 else float(probs[0])
            prediction = int(attack_prob >= threshold)
        else:
            preds = model.predict(X)
            prediction = int(preds[0])
            attack_prob = float(prediction)

        return prediction, round(attack_prob, 4)
