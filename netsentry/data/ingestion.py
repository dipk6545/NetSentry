"""
NetSentry Data Ingestion Module (`netsentry.data.ingestion`)
-----------------------------------------------------------
Discovers and ingests raw network flow CSV files into Polars LazyFrames
with normalized column headers (stripping extraneous whitespaces).
"""

from pathlib import Path
from typing import List, Union
import polars as pl


def discover_raw_files(raw_dir: Union[str, Path], pattern: str = "*.pcap_ISCX.csv") -> List[Path]:
    """Finds all matching raw CSV files in the target directory."""
    path = Path(raw_dir)
    if not path.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {path.resolve()}")

    files = sorted(list(path.glob(pattern)))
    if not files:
        # Fallback to any CSV if specific pattern yields no results
        files = sorted(list(path.glob("*.csv")))

    if not files:
        raise FileNotFoundError(f"No raw CSV files found in {path.resolve()}")

    return files


def normalize_headers(df: pl.LazyFrame) -> pl.LazyFrame:
    """Strips leading, trailing, and redundant whitespaces from column names."""
    rename_mapping = {col: col.strip().replace(" ", "_").replace("/", "_") for col in df.collect_schema().names()}
    return df.rename(rename_mapping)


def ingest_raw_flows(
    source: Union[str, Path, List[Union[str, Path]]] = "data/raw",
    pattern: str = "*.pcap_ISCX.csv"
) -> pl.LazyFrame:
    """
    Lazily ingests raw network flow CSVs, normalizing headers across all partitions.
    """
    if isinstance(source, (str, Path)):
        src_path = Path(source)
        if src_path.is_file():
            files = [src_path]
        else:
            files = discover_raw_files(src_path, pattern=pattern)
    else:
        files = [Path(p) for p in source]

    lazy_frames = []
    for file_path in files:
        # Scan raw CSV lazily with relaxed string fallback for mixed types and utf8-lossy encoding
        lf = pl.scan_csv(
            file_path,
            infer_schema_length=10000,
            ignore_errors=True,
            encoding="utf8-lossy",
            null_values=["", "NA", "null", "NaN", "Infinity", "-Infinity", "inf", "-inf"]
        )
        # Normalize column names
        lf = normalize_headers(lf)
        lazy_frames.append(lf)

    if len(lazy_frames) == 1:
        return lazy_frames[0]

    return pl.concat(lazy_frames, how="diagonal_relaxed")

