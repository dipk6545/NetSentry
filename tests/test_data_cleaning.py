"""
Automated unit tests for the Cleaning module (`netsentry/data/cleaning.py`).
"""

import pytest
import polars as pl
import numpy as np
from netsentry.data.cleaning import clean_network_flows


@pytest.fixture
def dirty_sample_df():
    return pl.DataFrame({
        # Metadata to be removed
        "Flow_ID": ["1", "2", "3", "4", "4"],
        "Source_IP": ["192.168.1.1"] * 5,
        "Source_Port": [80] * 5,
        "Destination_IP": ["10.0.0.1"] * 5,
        "Timestamp": ["2017-07-07 01:00:00"] * 5,
        # Constant zero-variance column to be removed
        "Bwd_PSH_Flags": [0, 0, 0, 0, 0],
        # Duplicate feature to be removed
        "Subflow_Fwd_Packets": [10, 20, 30, 40, 40],
        # Valid feature with inf values
        "Flow_Duration": [100.0, float("inf"), 300.0, 400.0, 400.0],
        "Total_Fwd_Packets": [5.0, 10.0, 15.0, 20.0, 20.0],
        # Labels
        "Label": ["BENIGN", "DDoS", "PortScan", "BENIGN", "BENIGN"]
    })


def test_clean_network_flows_prunes_metadata(dirty_sample_df):
    clean_df = clean_network_flows(dirty_sample_df)
    cols = clean_df.columns
    assert "Flow_ID" not in cols
    assert "Source_IP" not in cols
    assert "Source_Port" not in cols
    assert "Destination_IP" not in cols
    assert "Timestamp" not in cols


def test_clean_network_flows_prunes_constants(dirty_sample_df):
    clean_df = clean_network_flows(dirty_sample_df)
    cols = clean_df.columns
    assert "Bwd_PSH_Flags" not in cols
    assert "Subflow_Fwd_Packets" not in cols


def test_clean_network_flows_handles_inf(dirty_sample_df):
    clean_df = clean_network_flows(dirty_sample_df)
    # The row with inf in Flow_Duration must be dropped
    assert clean_df.filter(pl.col("Flow_Duration").is_infinite()).height == 0


def test_clean_network_flows_deduplicates(dirty_sample_df):
    clean_df = clean_network_flows(dirty_sample_df)
    # Row 4 was repeated twice; deduplication should keep only one
    assert clean_df.height == 3


def test_clean_network_flows_creates_binary_target(dirty_sample_df):
    clean_df = clean_network_flows(dirty_sample_df)
    assert "is_attack" in clean_df.columns
    assert set(clean_df["is_attack"].unique().to_list()) == {0, 1}

    # Verify mapping: BENIGN -> 0, PortScan -> 1
    benign_targets = clean_df.filter(pl.col("Label") == "BENIGN")["is_attack"].to_list()
    assert all(t == 0 for t in benign_targets)

    attack_targets = clean_df.filter(pl.col("Label") == "PortScan")["is_attack"].to_list()
    assert all(t == 1 for t in attack_targets)