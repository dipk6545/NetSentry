"""
Automated unit tests for Dataset Splitting Contracts (`netsentry/features/split.py`).
"""

import pytest
import polars as pl
import numpy as np
from netsentry.features.split import get_stratified_splits, get_temporal_splits, SplitDataset


@pytest.fixture
def clean_mock_dataset():
    # 100 rows: 80 benign, 20 attacks across different families
    n_benign = 80
    n_attacks = 20

    labels = ["BENIGN"] * n_benign + ["DDoS"] * 10 + ["PortScan"] * 6 + ["Bot"] * 4
    targets = [0] * n_benign + [1] * n_attacks

    data = {
        f"feat_{i}": np.random.randn(100).tolist()
        for i in range(5)
    }
    data["is_attack"] = targets
    data["Label"] = labels
    return pl.DataFrame(data)


def test_stratified_splits_ratios(clean_mock_dataset):
    splits = get_stratified_splits(clean_mock_dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

    assert isinstance(splits, SplitDataset)
    # Total samples: 100 -> 70 Train, 15 Val, 15 Test
    assert splits.X_train.shape[0] == 70
    assert splits.X_val.shape[0] == 15
    assert splits.X_test.shape[0] == 15

    # Features: 5 numeric columns
    assert len(splits.feature_names) == 5
    assert "is_attack" not in splits.feature_names
    assert "Label" not in splits.feature_names

    # Test label tracking exists for slicing
    assert len(splits.labels_test) == 15


def test_stratified_splits_class_balance(clean_mock_dataset):
    splits = get_stratified_splits(clean_mock_dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

    # 20% attacks overall (80% benign)
    train_attack_rate = splits.y_train.mean()
    val_attack_rate = splits.y_val.mean()
    test_attack_rate = splits.y_test.mean()

    assert np.isclose(train_attack_rate, 0.20, atol=0.03)
    assert np.isclose(val_attack_rate, 0.20, atol=0.03)
    assert np.isclose(test_attack_rate, 0.20, atol=0.03)


def test_temporal_unseen_attack_splits(clean_mock_dataset):
    # Holdout DDoS completely from training
    splits = get_temporal_splits(clean_mock_dataset, test_attacks=["DDoS"])

    # Test set must contain all DDoS samples
    assert all(lbl == "DDoS" for lbl in splits.labels_test)
    assert splits.X_test.shape[0] == 10
    assert splits.y_test.sum() == 10  # All 10 are attacks

    # Train and validation must NEVER have seen DDoS
    # All samples in train and val must be non-DDoS
    assert clean_mock_dataset.filter(pl.col("Label") != "DDoS").height == 90
    assert splits.X_train.shape[0] + splits.X_val.shape[0] == 90