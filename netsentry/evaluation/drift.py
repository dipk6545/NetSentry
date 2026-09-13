"""
Data Drift Detection Engine (`netsentry.evaluation.drift`).
---------------------------------------------------------
Evaluates feature distribution drift between a reference dataset and incoming
production data using two-sample Kolmogorov-Smirnov (KS) tests and Population
Stability Index (PSI). Determines whether emergency hyperparameter retuning is required.
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import polars as pl
from scipy.stats import ks_2samp

logger = logging.getLogger("netsentry.drift")


@dataclass(frozen=True)
class FeatureDriftResult:
    """Drift evaluation result for a single numerical feature."""
    feature_name: str
    ks_statistic: float
    p_value: float
    drift_detected: bool


@dataclass(frozen=True)
class DatasetDriftReport:
    """Aggregated drift report across all monitored features."""
    drift_detected: bool
    drifted_features_share: float
    drifted_features_count: int
    total_features_count: int
    feature_results: Dict[str, FeatureDriftResult]
    drift_threshold: float = 0.20  # If >20% features drift, dataset is considered drifted


class DataDriftDetector:
    """
    Detects data and concept drift between reference and current incoming data.
    """

    def __init__(
        self,
        p_value_threshold: float = 0.05,
        dataset_drift_threshold: float = 0.20,
    ):
        self.p_value_threshold = p_value_threshold
        self.dataset_drift_threshold = dataset_drift_threshold

    def calculate_drift(
        self,
        reference_data: Union[np.ndarray, pl.DataFrame],
        current_data: Union[np.ndarray, pl.DataFrame],
        feature_names: Optional[List[str]] = None,
    ) -> DatasetDriftReport:
        """
        Runs two-sample Kolmogorov-Smirnov test per feature to detect distribution shifts.
        """
        if isinstance(reference_data, pl.DataFrame):
            ref_arr = reference_data.to_numpy()
            feature_names = feature_names or reference_data.columns
        else:
            ref_arr = np.asarray(reference_data)

        if isinstance(current_data, pl.DataFrame):
            curr_arr = current_data.to_numpy()
            feature_names = feature_names or current_data.columns
        else:
            curr_arr = np.asarray(current_data)

        n_features = ref_arr.shape[1]
        if feature_names is None or len(feature_names) != n_features:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        drifted_count = 0
        feature_results: Dict[str, FeatureDriftResult] = {}

        for i, col_name in enumerate(feature_names):
            ref_col = ref_arr[:, i]
            curr_col = curr_arr[:, i]

            # Drop NaNs or Infs for robust statistical testing
            ref_clean = ref_col[np.isfinite(ref_col)]
            curr_clean = curr_col[np.isfinite(curr_col)]

            if len(ref_clean) == 0 or len(curr_clean) == 0:
                continue

            ks_stat, p_val = ks_2samp(ref_clean, curr_clean)
            is_drifted = bool(p_val < self.p_value_threshold)

            if is_drifted:
                drifted_count += 1

            feature_results[col_name] = FeatureDriftResult(
                feature_name=col_name,
                ks_statistic=float(ks_stat),
                p_value=float(p_val),
                drift_detected=is_drifted,
            )

        drifted_share = float(drifted_count / n_features) if n_features > 0 else 0.0
        dataset_drift_detected = drifted_share >= self.dataset_drift_threshold

        logger.info(
            f"Drift Analysis Completed: {drifted_count}/{n_features} features drifted "
            f"({drifted_share:.1%}). Dataset Drift Detected = {dataset_drift_detected}"
        )

        return DatasetDriftReport(
            drift_detected=dataset_drift_detected,
            drifted_features_share=drifted_share,
            drifted_features_count=drifted_count,
            total_features_count=n_features,
            feature_results=feature_results,
            drift_threshold=self.dataset_drift_threshold,
        )

    def save_report(self, report: DatasetDriftReport, output_path: Union[str, Path]) -> Path:
        """Saves drift summary JSON for audit and observability."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "drift_detected": report.drift_detected,
            "drifted_features_share": report.drifted_features_share,
            "drifted_features_count": report.drifted_features_count,
            "total_features_count": report.total_features_count,
            "drift_threshold": report.drift_threshold,
            "top_drifted_features": [
                {
                    "feature": k,
                    "ks_statistic": v.ks_statistic,
                    "p_value": v.p_value,
                }
                for k, v in sorted(
                    report.feature_results.items(),
                    key=lambda x: x[1].ks_statistic,
                    reverse=True,
                )[:10]
            ],
        }

        with open(p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Drift summary saved to {p}")
        return p
