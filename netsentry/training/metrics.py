"""
Metrics Calculation Module (`netsentry.training.metrics`).
---------------------------------------------------------
Computes model-agnostic evaluation metrics for binary classification.
Supports capability-aware metrics calculation when probabilities are present.
"""

from typing import Dict, Optional, Union
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def calculate_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_prob: Optional[Union[np.ndarray, list]] = None,
) -> Dict[str, float]:
    """
    Computes classification performance metrics.
    
    Mandatory metrics (from y_pred):
        - accuracy
        - precision
        - recall
        - f1
        
    Capability-dependent metrics (if y_prob provided):
        - roc_auc
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_t, y_p)),
        "precision": float(precision_score(y_t, y_p, zero_division=0)),
        "recall": float(recall_score(y_t, y_p, zero_division=0)),
        "f1": float(f1_score(y_t, y_p, zero_division=0)),
    }

    if y_prob is not None:
        y_pr = np.asarray(y_prob)
        # Handle binary probabilities shape (N, 2) or (N,)
        if y_pr.ndim == 2 and y_pr.shape[1] == 2:
            y_pr = y_pr[:, 1]

        try:
            metrics["roc_auc"] = float(roc_auc_score(y_t, y_pr))
        except ValueError:
            # e.g., single-class present in validation slice
            metrics["roc_auc"] = 0.0

    return metrics
