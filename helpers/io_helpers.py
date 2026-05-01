"""
Shared I/O utilities for the EVAnalysis pipeline.
"""

from pathlib import Path

import polars as pl


def save_parquet(df: pl.DataFrame, path: Path, tag: str = "") -> None:
    """
    Writes df to path as a Parquet file, creating parent directories if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    prefix = f"{tag} " if tag else ""
    print(f"  {prefix}Saved {path.name} ({len(df)} rows)")


def infer_snapshot_interval_ms(station_snapshots_path: Path) -> int:
    """
    Infers the snapshot bucket
      interval from station_snapshots.parquet.

    Returns the minimum difference between consecutive unique simtime_ms values,
    which equals the interval at which the simulator emitted snapshots.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not station_snapshots_path.exists():
        raise FileNotFoundError(
            f"station_snapshots.parquet not found at {station_snapshots_path}. "
            "Run analyse_station() before any module that needs the interval."
        )

    interval: int = (
        pl.read_parquet(station_snapshots_path)
        .select("simtime_ms")
        .unique()
        .sort("simtime_ms")
        .with_columns(pl.col("simtime_ms").diff().alias("diff"))
        ["diff"]
        .drop_nulls()
        .min()
    ) or 1

    return interval