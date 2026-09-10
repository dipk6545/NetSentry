"""
NetSentry Data Cleaning Module (`netsentry.data.cleaning`)
---------------------------------------------------------
Performs deterministic data preparation on validated network flows:
1. Strips non-behavioral metadata (IPs, ports, timestamps, flow IDs).
2. Drops zero-variance constant columns.
3. Drops duplicate rows.
4. Coerces inf / -inf values and removes corrupted records.
5. Generates the binary classification target `is_attack`.

Notice: Feature engineering (scaling, encodings, ratios) is strictly decoupled
and deferred to the downstream feature engineering phase.
"""

from typing import List, Union
import polars as pl


# 1. Non-behavioral metadata columns causing target leakage or memorization
METADATA_COLUMNS: List[str] = [
    "Flow_ID",
    "Source_IP",
    "Source_Port",
    "Destination_IP",
    "Timestamp"
]

# 2. Known constant zero-variance columns in CIC-IDS2017
ZERO_VARIANCE_COLUMNS: List[str] = [
    "Bwd_PSH_Flags",
    "Bwd_URG_Flags",
    "Fwd_Avg_Bytes_Bulk",
    "Fwd_Avg_Packets_Bulk",
    "Fwd_Avg_Bulk_Rate",
    "Bwd_Avg_Bytes_Bulk",
    "Bwd_Avg_Packets_Bulk",
    "Bwd_Avg_Bulk_Rate"
]

# 3. Known redundant duplicate columns
DUPLICATE_FEATURE_COLUMNS: List[str] = [
    "Subflow_Fwd_Packets",
    "Subflow_Bwd_Packets",
    "Subflow_Fwd_Bytes",
    "Subflow_Bwd_Bytes",
    "Avg_Fwd_Segment_Size",
    "Avg_Bwd_Segment_Size",
    "Fwd_Header_Length_1"
]


def drop_metadata_columns(df: pl.LazyFrame) -> pl.LazyFrame:
    """Removes non-behavioral identifier columns."""
    existing_cols = df.collect_schema().names()
    to_drop = [c for c in METADATA_COLUMNS if c in existing_cols]
    return df.drop(to_drop) if to_drop else df


def drop_constant_columns(df: pl.LazyFrame) -> pl.LazyFrame:
    """Removes constant zero-variance and duplicate feature columns."""
    existing_cols = df.collect_schema().names()
    to_drop = [c for c in ZERO_VARIANCE_COLUMNS + DUPLICATE_FEATURE_COLUMNS if c in existing_cols]
    return df.drop(to_drop) if to_drop else df


def sanitize_numeric_values(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Coerces inf and -inf values to null across all floating-point columns,
    then drops records containing nulls in core numerical attributes.
    """
    schema = df.collect_schema()
    float_cols = [c for c, dtype in schema.items() if dtype in (pl.Float32, pl.Float64)]

    # Replace inf and -inf with null
    replace_exprs = [
        pl.when(pl.col(c).is_infinite())
        .then(None)
        .otherwise(pl.col(c))
        .alias(c)
        for c in float_cols
    ]

    if replace_exprs:
        df = df.with_columns(replace_exprs)

    # Drop records containing any nulls across numerical columns
    return df.drop_nulls()


def add_binary_target(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Creates binary target `is_attack`:
    - 0 for BENIGN
    - 1 for all 14 cyberattack families
    Preserves raw string `Label` for downstream slice evaluations.
    """
    return df.with_columns(
        pl.when(pl.col("Label").str.to_uppercase() == "BENIGN")
        .then(0)
        .otherwise(1)
        .cast(pl.UInt8)
        .alias("is_attack")
    )


def clean_network_flows(df: Union[pl.DataFrame, pl.LazyFrame]) -> pl.DataFrame:
    """
    Executes the full deterministic cleaning sequence on validated network flows:
    Metadata Pruning -> Constant Pruning -> Duplicate Row Removal -> Sanitize Infs -> Binary Target.
    """
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    # Step 1: Remove metadata
    lf = drop_metadata_columns(lf)

    # Step 2: Remove constant & duplicate features
    lf = drop_constant_columns(lf)

    # Step 3: Sanitize inf / nulls
    lf = sanitize_numeric_values(lf)

    # Step 4: Add is_attack binary target
    lf = add_binary_target(lf)

    # Collect and remove exact duplicate rows in memory
    clean_df = lf.collect()
    clean_df = clean_df.unique()

    return clean_df