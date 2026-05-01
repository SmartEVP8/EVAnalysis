"""
Module for analysing EV wait-time-in-queue metrics.

Reads raw WaitTimeInQueueMetric data and writes:
  - waittime_snapshots.parquet : per-EV enriched log snapped to the station snapshot time grid
  - waittime_percentiles.parquet : per-time-slot wait-time percentiles and EV counts
"""

from pathlib import Path

import polars as pl

from helpers.loader import add_time_columns, MS_PER_DAY
from helpers.constants import OUTPUT_ROOT, PERCENTILES
from helpers.io_helpers import save_parquet, infer_snapshot_interval_ms


def analyse_wait_time(
    parquet_path: Path,
    run_id: str,
    output_root: Path = OUTPUT_ROOT,
) -> None:
    """
    Analyses wait-time-in-queue data for a single simulation run.

    StartChargingTime is used as the reference clock for temporal grouping.
    Each record is snapped to the nearest station-snapshot interval so that
    downstream joins with station data are aligned on the same time grid.
    """
    print(f"\n[WaitTime] Analysing {parquet_path.name}...")

    df = pl.read_parquet(parquet_path)

    if "WaitTimeInQueue" not in df.columns:
        raise ValueError(
            f"Expected column 'WaitTimeInQueue' not found. "
            f"Available columns: {df.columns}"
        )

    station_snapshots_path = output_root / run_id / "analysis" / "station_snapshots.parquet"
    snapshot_interval_ms = infer_snapshot_interval_ms(station_snapshots_path)

    df = add_time_columns(df, "StartChargingTime")

    snapshot_df = (
        df
        .with_columns([
            (pl.col("WaitTimeInQueue") / 60_000).alias("wait_minutes"),
            (
                (pl.col("simtime_ms") // snapshot_interval_ms) * snapshot_interval_ms
            ).cast(pl.Int64).alias("simtime_ms"),
        ])
        .with_columns([
            (
                (pl.col("simtime_ms") // 1_000 // 3_600).cast(pl.Utf8).str.zfill(2)
                + pl.lit(":")
                + ((pl.col("simtime_ms") // 1_000 % 3_600) // 60).cast(pl.Utf8).str.zfill(2)
            ).alias("time_label"),
        ])
        .select([
            "EVId", "StationId",
            "ArrivalAtStationTime", "StartChargingTime",
            "WaitTimeInQueue", "wait_minutes",
            "day", "weekday_name", "simtime_ms", "time_label",
        ])
        .sort(["day", "simtime_ms", "EVId"])
    )

    out_analysis = output_root / run_id / "analysis"
    save_parquet(snapshot_df, out_analysis / "waittime_snapshots.parquet", "[WaitTime]")

    percentile_df = (
        snapshot_df
        .group_by(["weekday_name", "simtime_ms", "time_label"])
        .agg(
            [pl.col("wait_minutes").quantile(q).alias(f"wait_p{int(q * 100)}")
             for q in PERCENTILES]
            + [pl.col("EVId").count().alias("ev_count")]
        )
        .sort(["weekday_name", "simtime_ms"])
    )

    out_percentiles = output_root / run_id / "percentiles" / "waittime"
    save_parquet(percentile_df, out_percentiles / "waittime_percentiles.parquet", "[WaitTime]")