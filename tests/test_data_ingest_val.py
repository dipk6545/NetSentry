"""
Automated unit tests for Ingestion and Validation modules (`netsentry/data`).
"""

import pytest
import polars as pl
from pathlib import Path
from netsentry.data.ingestion import ingest_raw_flows, discover_raw_files, normalize_headers
from netsentry.data.validation import validate_raw_schema, DataValidationError, MANDATORY_FLOW_COLUMNS


@pytest.fixture
def sample_valid_df():
    data = {col: [10.0, 20.0] for col in MANDATORY_FLOW_COLUMNS if col != "Label"}
    data["Label"] = ["BENIGN", "DDoS"]
    data["Extra_Feature"] = [1.0, 2.0]
    return pl.DataFrame(data)


def test_discover_raw_files():
    files = discover_raw_files("data/raw")
    assert len(files) == 8
    assert all(f.exists() for f in files)


def test_normalize_headers():
    dirty_df = pl.LazyFrame({
        " Flow ID ": ["1"],
        " Total/Fwd Packets ": [10],
        " Label ": ["BENIGN"]
    })
    normalized = normalize_headers(dirty_df)
    cols = normalized.collect_schema().names()
    assert cols == ["Flow_ID", "Total_Fwd_Packets", "Label"]


def test_validation_success(sample_valid_df):
    stats = validate_raw_schema(sample_valid_df)
    assert stats["sample_rows"] == 2
    assert "BENIGN" in stats["unique_labels"]
    assert "DDoS" in stats["unique_labels"]


def test_validation_missing_label(sample_valid_df):
    invalid = sample_valid_df.drop("Label")
    with pytest.raises(DataValidationError, match="Target column 'Label' missing"):
        validate_raw_schema(invalid)


def test_validation_missing_mandatory_col(sample_valid_df):
    invalid = sample_valid_df.drop("Flow_Duration")
    with pytest.raises(DataValidationError, match="Missing mandatory flow columns"):
        validate_raw_schema(invalid)


def test_validation_unauthorized_label(sample_valid_df):
    invalid = sample_valid_df.with_columns(pl.Series("Label", ["UNKNOWN_TROJAN", "BENIGN"]))
    with pytest.raises(DataValidationError, match="Detected unauthorized/unknown attack labels"):
        validate_raw_schema(invalid)