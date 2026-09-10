"""
End-to-End integration test for the Data Pipeline (`netsentry.data.pipeline`).
"""

import pytest
import polars as pl
from pathlib import Path
from netsentry.data.pipeline import run_data_pipeline


def test_data_pipeline_e2e(tmp_path):
    # Create mock raw CSV file with dirty data
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    mock_csv = raw_dir / "Monday-WorkingHours.pcap_ISCX.csv"

    mock_df = pl.DataFrame({
        "Flow ID": ["1", "2", "3", "3"],
        " Source IP": ["192.168.1.1"] * 4,
        " Source Port": [80] * 4,
        " Destination IP": ["10.0.0.1"] * 4,
        " Timestamp": ["2017-07-07 01:00:00"] * 4,
        " Flow Duration": [100.0, 200.0, 300.0, 300.0],
        " Total Fwd Packets": [5.0, 10.0, 15.0, 15.0],
        " Total Backward Packets": [2.0, 4.0, 6.0, 6.0],
        " Total Length of Fwd Packets": [100.0, 200.0, 300.0, 300.0],
        " Total Length of Bwd Packets": [50.0, 100.0, 150.0, 150.0],
        " Flow Bytes/s": [1.5, 2.5, 3.5, 3.5],
        " Flow Packets/s": [0.5, 1.0, 1.5, 1.5],
        " Bwd PSH Flags": [0, 0, 0, 0],
        " Subflow Fwd Packets": [5, 10, 15, 15],
        " Label": ["BENIGN", "PortScan", "BENIGN", "BENIGN"]
    })
    mock_df.write_csv(mock_csv)

    output_parquet = tmp_path / "processed" / "clean_flows.parquet"

    # 1. Run pipeline
    result = run_data_pipeline(
        raw_source=raw_dir,
        output_path=output_parquet,
        sample_validation_size=100
    )

    assert result["status"] == "success"
    assert output_parquet.exists()

    # 2. Verify cleaned dataset contents
    df_out = pl.read_parquet(output_parquet)
    assert df_out.height == 3  # Deduplication removed the duplicate 4th row
    assert "is_attack" in df_out.columns
    assert "Flow_ID" not in df_out.columns
    assert "Bwd_PSH_Flags" not in df_out.columns

    # 3. Test caching behavior
    cached_result = run_data_pipeline(
        raw_source=raw_dir,
        output_path=output_parquet
    )
    assert cached_result["status"] == "cached"
    assert cached_result["total_records"] == 3