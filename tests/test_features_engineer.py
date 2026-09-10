"""
Unit tests for NetSentry Feature Engineering (`netsentry.features.engineer`).
"""

import polars as pl
import pytest
from netsentry.features.engineer import (
    compute_directional_ratios,
    compute_flag_interactions,
    compute_temporal_rate_features,
    engineer_network_features,
)


@pytest.fixture
def mock_flow_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Total_Fwd_Packets": [10.0, 0.0, 50.0],
            "Total_Backward_Packets": [5.0, 0.0, 25.0],
            "Total_Length_of_Fwd_Packets": [1000.0, 0.0, 5000.0],
            "Total_Length_of_Bwd_Packets": [500.0, 0.0, 2500.0],
            "SYN_Flag_Count": [1.0, 0.0, 1.0],
            "ACK_Flag_Count": [0.0, 1.0, 1.0],
            "RST_Flag_Count": [2.0, 0.0, 0.0],
            "Flow_Duration": [1000.0, 0.0, 50000.0],
        }
    )


def test_compute_directional_ratios(mock_flow_df: pl.DataFrame):
    res = compute_directional_ratios(mock_flow_df.lazy()).collect()
    assert "Fwd_to_Bwd_Packet_Ratio" in res.columns
    assert "Fwd_to_Bwd_Byte_Ratio" in res.columns
    assert "Avg_Fwd_Packet_Payload" in res.columns

    # Row 0: 10 / (5 + 1) = 1.66666...
    assert pytest.approx(res["Fwd_to_Bwd_Packet_Ratio"][0], 0.01) == 1.6667
    # Row 0: 1000 / (500 + 1) = 1.996...
    assert pytest.approx(res["Fwd_to_Bwd_Byte_Ratio"][0], 0.01) == 1.996


def test_compute_flag_interactions(mock_flow_df: pl.DataFrame):
    res = compute_flag_interactions(mock_flow_df.lazy()).collect()
    assert "SYN_no_ACK_Indicator" in res.columns
    assert "RST_Density" in res.columns

    # Row 0: SYN=1, ACK=0 -> 1 * (1 - 0) = 1.0
    assert res["SYN_no_ACK_Indicator"][0] == 1.0
    # Row 1: SYN=0, ACK=1 -> 0 * (1 - 1) = 0.0
    assert res["SYN_no_ACK_Indicator"][1] == 0.0
    # Row 2: SYN=1, ACK=1 -> 1 * (1 - 1) = 0.0
    assert res["SYN_no_ACK_Indicator"][2] == 0.0


def test_compute_temporal_rate_features(mock_flow_df: pl.DataFrame):
    res = compute_temporal_rate_features(mock_flow_df.lazy()).collect()
    assert "Log_Flow_Duration" in res.columns
    assert "Flow_Packet_Density" in res.columns

    # Check non-negative log transformation
    assert res["Log_Flow_Duration"][1] == 0.0  # log(0 + 1) == 0


def test_engineer_network_features_pipeline(mock_flow_df: pl.DataFrame):
    enriched = engineer_network_features(mock_flow_df)
    expected_cols = [
        "Fwd_to_Bwd_Packet_Ratio",
        "Fwd_to_Bwd_Byte_Ratio",
        "Avg_Fwd_Packet_Payload",
        "SYN_no_ACK_Indicator",
        "RST_Density",
        "Log_Flow_Duration",
        "Flow_Packet_Density",
    ]
    for col in expected_cols:
        assert col in enriched.columns

    # Verify no nulls or NaNs generated
    for col in expected_cols:
        assert enriched[col].is_null().sum() == 0
        assert enriched[col].is_nan().sum() == 0
