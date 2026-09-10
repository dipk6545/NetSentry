"""
Attack-Family Slice Evaluation (`netsentry.evaluation.slices`).
--------------------------------------------------------------
Audits fine-grained model recall, precision, and error rates broken down
by attack family label (e.g. DDoS, PortScan, Botnet, Web Attack).
"""

from dataclasses import dataclass
from typing import Dict, List, Union
import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


@dataclass(frozen=True)
class AttackSliceMetric:
    """Performance metrics for an individual attack family slice."""
    label_name: str
    sample_count: int
    true_attacks: int
    true_benign: int
    predicted_attacks: int
    recall: float
    precision: float
    f1: float
    accuracy: float


def compute_slice_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    labels: Union[np.ndarray, list, pl.Series],
) -> Dict[str, AttackSliceMetric]:
    """
    Computes classification performance broken down across every attack family.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    labs = np.asarray(labels, dtype=str)

    unique_labels = sorted(list(set(labs)))
    slice_report: Dict[str, AttackSliceMetric] = {}

    for label in unique_labels:
        mask = (labs == label)
        slice_y_t = y_t[mask]
        slice_y_p = y_p[mask]
        count = int(np.sum(mask))

        true_atk = int(np.sum(slice_y_t == 1))
        true_ben = int(np.sum(slice_y_t == 0))
        pred_atk = int(np.sum(slice_y_p == 1))

        # Precision & Recall for this slice
        prec = float(precision_score(slice_y_t, slice_y_p, zero_division=0))
        rec = float(recall_score(slice_y_t, slice_y_p, zero_division=0))
        f1 = float(f1_score(slice_y_t, slice_y_p, zero_division=0))
        acc = float(accuracy_score(slice_y_t, slice_y_p))

        slice_report[label] = AttackSliceMetric(
            label_name=label,
            sample_count=count,
            true_attacks=true_atk,
            true_benign=true_ben,
            predicted_attacks=pred_atk,
            recall=rec,
            precision=prec,
            f1=f1,
            accuracy=acc,
        )

    return slice_report
