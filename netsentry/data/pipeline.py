"""
NetSentry Data Pipeline Orchestrator (`netsentry.data.pipeline`)
---------------------------------------------------------------
Orchestrates the offline data preparation lifecycle:
Raw CSVs -> Ingestion -> Schema Validation -> Deterministic Cleaning -> Processed Parquet.
"""

import time
import argparse
from pathlib import Path
from typing import Dict, Any, Union
import polars as pl

from netsentry.data.ingestion import ingest_raw_flows, discover_raw_files
from netsentry.data.validation import validate_raw_schema
from netsentry.data.cleaning import clean_network_flows


def run_data_pipeline(
    raw_source: Union[str, Path] = "data/raw",
    output_path: Union[str, Path] = "data/processed/clean_network_flows.parquet",
    sample_validation_size: int = 50000,
    force_recompute: bool = False
) -> Dict[str, Any]:
    """
    Coordinates data ingestion, validation, and cleaning:
    1. Checks if processed dataset already exists (caching).
    2. Lazily scans raw network flows across all partition files.
    3. Executes non-blocking schema and categorical validation checks.
    4. Applies deterministic cleaning, deduplication, and target mapping.
    5. Saves the final production dataset to Snappy-compressed Parquet.
    """
    out_file = Path(output_path)
    if out_file.exists() and not force_recompute:
        print(f"📦 Found cached processed dataset at: {out_file.resolve()}")
        df_existing = pl.read_parquet(out_file)
        return {
            "status": "cached",
            "output_path": str(out_file.resolve()),
            "total_records": df_existing.height,
            "total_features": len(df_existing.columns),
            "attacks": int(df_existing["is_attack"].sum()),
            "benign": int((df_existing["is_attack"] == 0).sum())
        }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    # 1. Ingestion
    print(f"\n🚀 [Stage 1/4] Scanning raw network flows from: {raw_source}")
    lf_raw = ingest_raw_flows(source=raw_source)

    # 2. Validation
    print(f"🔍 [Stage 2/4] Validating schema and attack categories (sample={sample_validation_size:,})...")
    val_stats = validate_raw_schema(lf_raw, sample_size=sample_validation_size)
    print(f"   Validation passed ({val_stats['columns_count']} columns, {len(val_stats['unique_labels'])} unique labels in sample)")

    # 3. Cleaning & Target Generation
    print("🧹 [Stage 3/4] Executing deterministic cleaning and deduplication...")
    clean_df = clean_network_flows(lf_raw)

    # 4. Storage
    print(f"💾 [Stage 4/4] Writing compressed Parquet to: {out_file.resolve()}...")
    clean_df.write_parquet(out_file, compression="snappy")

    duration = time.perf_counter() - start_time
    attacks = int(clean_df["is_attack"].sum())
    benign = int((clean_df["is_attack"] == 0).sum())

    print(f"\n✅ Data Pipeline Completed in {duration:.2f} seconds!")
    print(f"   - Total Clean Flows: {clean_df.height:,}")
    print(f"   - Features:          {len(clean_df.columns)} (including 'is_attack' and 'Label')")
    print(f"   - Benign Flows:      {benign:,} ({(benign / clean_df.height) * 100:.2f}%)")
    print(f"   - Malicious Attacks: {attacks:,} ({(attacks / clean_df.height) * 100:.2f}%)")

    return {
        "status": "success",
        "output_path": str(out_file.resolve()),
        "duration_sec": round(duration, 2),
        "total_records": clean_df.height,
        "total_features": len(clean_df.columns),
        "attacks": attacks,
        "benign": benign
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NetSentry Data Preparation Pipeline")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw CSV directory")
    parser.add_argument("--output", type=str, default="data/processed/clean_network_flows.parquet", help="Path to output Parquet")
    parser.add_argument("--force", action="store_true", help="Force recomputation even if output exists")
    args = parser.parse_args()

    run_data_pipeline(
        raw_source=args.raw_dir,
        output_path=args.output,
        force_recompute=args.force
    )