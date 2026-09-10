"""
NetSentry Dataset Splitting Contract (`netsentry.features.split`)
----------------------------------------------------------------
Implements dual dataset splitting contracts:
1. Primary Benchmark (Stratified 70/15/15): Evaluates normal supervised detection
   preserving exact class proportions across Train (70%), Val (15%), and Test (15%).
2. Secondary Robustness Benchmark (Day-based / Temporal Split): Evaluates
   unseen-attack-family generalization (e.g. testing model on unseen threat families).
"""

from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class SplitDataset:
    """Standard container for dataset splits adhering to Scikit-Learn / XGBoost contract."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    labels_test: List[str]  # Raw string attack labels for slice auditing


def extract_feature_matrix(
    df: pl.DataFrame,
    target_col: str = "is_attack",
    label_col: str = "Label"
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Separates feature matrix X from target vector y and metadata label strings."""
    feature_cols = [c for c in df.columns if c not in (target_col, label_col)]
    X = df.select(feature_cols).to_numpy().astype(np.float32)
    y = df[target_col].to_numpy().astype(np.int32)
    labels = df[label_col].to_list()
    return X, y, feature_cols, labels


def get_stratified_splits(
    df: pl.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> SplitDataset:
    """
    Primary Benchmark Split Contract:
    Performs 3-way stratified partition preserving binary target proportions.
    - 70% Train: Model parameter optimization
    - 15% Validation: Threshold tuning and hyperparameter search
    - 15% Unseen Test: Final unbiased security benchmark
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    X, y, feature_names, labels = extract_feature_matrix(df)
    labels_arr = np.array(labels)
    n_total = len(y)

    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))

    # First split: Separate Test set
    X_temp, X_test, y_temp, y_test, _, labels_test = train_test_split(
        X, y, labels_arr,
        test_size=n_test,
        stratify=y,
        random_state=seed
    )

    # Second split: Separate Train and Validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=n_val,
        stratify=y_temp,
        random_state=seed
    )

    return SplitDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        labels_test=labels_test.tolist()
    )


def get_temporal_splits(
    df: pl.DataFrame,
    test_attacks: Optional[List[str]] = None
) -> SplitDataset:
    """
    Secondary Robustness Experiment (Generalization to Unseen Attack Families):
    Separates designated attack families completely from the training set,
    placing them exclusively into the Test set to measure zero-day generalization.
    Default unseen attack test families: ['Bot', 'PortScan', 'DDoS']
    """
    holdout_attacks = test_attacks or ["Bot", "PortScan", "DDoS"]

    # Filter holdout attack families strictly to the test set
    test_mask = df["Label"].is_in(holdout_attacks)
    test_df = df.filter(test_mask)
    train_val_df = df.filter(~test_mask)

    # Stratified split on the remaining seen data
    X_tv, y_tv, feature_names, _ = extract_feature_matrix(train_val_df)
    X_test, y_test, _, labels_test = extract_feature_matrix(test_df)

    # Split train_val into 80% train and 20% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=0.20,
        stratify=y_tv,
        random_state=42
    )

    return SplitDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        labels_test=labels_test
    )