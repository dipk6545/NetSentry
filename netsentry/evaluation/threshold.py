"""
Validation Decision Threshold Optimizer (`netsentry.evaluation.threshold`).
----------------------------------------------------------------------------
Finds the optimal decision threshold tau* on VALIDATION probabilities
according to the configured business objective (e.g., maximize F1 or recall).
Ensures the test set remains completely untouched.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from netsentry.evaluation.config import ThresholdConfig


@dataclass(frozen=True)
class ThresholdSearchResult:
    """Outcome of threshold optimization on validation data."""
    selected_threshold: float
    objective: str
    best_score: float
    scores_by_threshold: Dict[float, float]


def find_optimal_threshold(
    y_val_true: Union[np.ndarray, list],
    y_val_prob: Union[np.ndarray, list],
    config: Optional[ThresholdConfig] = None,
    num_steps: int = 100,
) -> ThresholdSearchResult:
    """
    Scans decision thresholds in (0.01, 0.99) on validation data to optimize
    the configured objective: 'f1', 'recall', or 'precision'.
    """
    cfg = config or ThresholdConfig()
    y_t = np.asarray(y_val_true, dtype=int)
    probs = np.asarray(y_val_prob)
    if probs.ndim == 2 and probs.shape[1] == 2:
        probs = probs[:, 1]

    thresholds = np.linspace(0.01, 0.99, num_steps)
    best_th = 0.50
    best_score = -1.0
    history: Dict[float, float] = {}

    for th in thresholds:
        th_val = round(float(th), 4)
        preds = (probs >= th).astype(int)

        prec = precision_score(y_t, preds, zero_division=0)
        rec = recall_score(y_t, preds, zero_division=0)
        f1 = f1_score(y_t, preds, zero_division=0)

        # Evaluate objective
        if cfg.objective == "f1":
            score = f1
        elif cfg.objective == "recall":
            if cfg.min_precision is not None and prec < cfg.min_precision:
                score = 0.0  # Constraint penalty
            else:
                score = rec
        elif cfg.objective == "precision":
            score = prec
        else:
            score = f1

        history[th_val] = float(score)

        if score > best_score:
            best_score = score
            best_th = th_val

    return ThresholdSearchResult(
        selected_threshold=best_th,
        objective=cfg.objective,
        best_score=float(best_score),
        scores_by_threshold=history,
    )


def apply_threshold(probs: Union[np.ndarray, list], threshold: float = 0.5) -> np.ndarray:
    """Applies a frozen decision threshold to probabilities to produce binary labels."""
    pr = np.asarray(probs)
    if pr.ndim == 2 and pr.shape[1] == 2:
        pr = pr[:, 1]
    return (pr >= threshold).astype(int)
