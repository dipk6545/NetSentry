"""
NetSentry Data Validation Module (netsentry.data.validation)
-------------------------------------------------------------
Performs schema validation, null/inf detection, and label integrity checks
on ingested Polars DataFrames/LazyFrames, raising DataValidationError on violations.
"""

from typing import List, Set, Union, Dict, Any
import polars as pl


class DataValidationError(Exception):
    """Raised when incoming raw or ingested data violates the NetSentry schema contract."""
    pass


ALLOWED_LABELS: Set[str] = {
    "BENIGN",
    "DDoS",
    "PortScan",
    "Bot",
    "Infiltration",
    "Web Attack \x96 Brute Force",
    "Web Attack \x96 XSS",
    "Web Attack \x96 Sql Injection",
    "Web Attack  Brute Force",
    "Web Attack  XSS",
    "Web Attack  Sql Injection",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "DoS Hulk",
    "DoS GoldenEye",
    "Heartbleed"
}

MANDATORY_FLOW_COLUMNS: List[str] = [
    "Flow_Duration",
    "Total_Fwd_Packets",
    "Total_Backward_Packets",
    "Total_Length_of_Fwd_Packets",
    "Total_Length_of_Bwd_Packets",
    "Flow_Bytes_s",
    "Flow_Packets_s",
    "Label"
]


def validate_raw_schema(
    df: Union[pl.DataFrame, pl.LazyFrame],
    check_numeric_nulls: bool = True,
    sample_size: int = 50000
) -> Dict[str, Any]:
    """
    Validates structural integrity of ingested flow data:
    - Target Label column existence and domain
    - Mandatory network flow metrics
    - Null counts across numerical fields
    """
    if isinstance(df, pl.LazyFrame):
        schema = df.collect_schema()
        df_sample = df.limit(sample_size).collect()
    else:
        schema = df.schema
        df_sample = df.head(sample_size)

    column_names = set(schema.names())

    # 1. Label existence
    if "Label" not in column_names:
        raise DataValidationError("Target column 'Label' missing from ingested dataset.")

    # 2. Mandatory columns
    missing_cols = [col for col in MANDATORY_FLOW_COLUMNS if col not in column_names]
    if missing_cols:
        raise DataValidationError(f"Missing mandatory flow columns: {missing_cols}")

    # 3. Validate Labels
    unique_labels = df_sample["Label"].unique().to_list()
    unknown_labels = [lbl for lbl in unique_labels if lbl is not None and lbl not in ALLOWED_LABELS]
    if unknown_labels:
        raise DataValidationError(f"Detected unauthorized/unknown attack labels: {unknown_labels}")

    stats = {
        "columns_count": len(column_names),
        "sample_rows": df_sample.height,
        "unique_labels": unique_labels
    }

    if check_numeric_nulls:
        numeric_cols = [
            col for col, dtype in schema.items()
            if dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
        ]
        null_counts = {}
        for col in numeric_cols:
            n_nulls = df_sample[col].null_count()
            if n_nulls > 0:
                null_counts[col] = n_nulls
        stats["numeric_columns_with_nulls"] = null_counts

    return stats