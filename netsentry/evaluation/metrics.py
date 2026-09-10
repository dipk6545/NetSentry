"""
Evaluation Metrics & Confusion Matrix (`netsentry.evaluation.metrics`).
-----------------------------------------------------------------------
Computes ranking and threshold-specific metrics including security rates.
"""

from typing import Any, Dict, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def compute_binary_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_prob: Optional[Union[np.ndarray, list]] = None,
) -> Dict[str, float]:
    """
    Computes standard classification metrics:
    - accuracy, precision, recall, f1
    - roc_auc, pr_auc (if probabilities provided)
    - false_positive_rate, false_negative_rate
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    metrics = {
        "accuracy": float(accuracy_score(y_t, y_p)),
        "precision": float(precision_score(y_t, y_p, zero_division=0)),
        "recall": float(recall_score(y_t, y_p, zero_division=0)),
        "f1": float(f1_score(y_t, y_p, zero_division=0)),
    }

    # Confusion matrix rates
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)

    metrics["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    metrics["false_negative_rate"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    if y_prob is not None:
        y_pr = np.asarray(y_prob)
        if y_pr.ndim == 2 and y_pr.shape[1] == 2:
            y_pr = y_pr[:, 1]
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_t, y_pr))
            metrics["pr_auc"] = float(average_precision_score(y_t, y_pr))
        except ValueError:
            metrics["roc_auc"] = 0.0
            metrics["pr_auc"] = 0.0

    return metrics
