"""
Module for analyzing EV wait time in queue metrics.
"""

from pathlib import Path
import polars as pl
from init.loader import WEEKDAY_NAMES, MS_PER_DAY, SIMULATION_START_DOW

OUTPUT_ROOT = Path("runs")

PERCENTILES = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def analyse_wait_time(parquet_path: Path, run_id: str, output_root: Path = OUTPUT_ROOT) -> None:
    print(f"\n[WaitTime] Analysing {parquet_path.name}...")

    df = pl.read_parquet(parquet_path)

    if "WaitTimeInQueue" not in df.columns:
        raise ValueError(f"Expected column 'WaitTimeInQueue' not found. "
                         f"Available: {df.columns}")

    snapshot_df = (
        df.with_columns([
            (pl.col("WaitTimeInQueue") / 60_000).alias("wait_minutes"),
            (pl.col("StartChargingTime") // MS_PER_DAY).cast(pl.Int32).alias("day"),
            (pl.col("StartChargingTime") % MS_PER_DAY).cast(pl.Int64).alias("simtime_ms"),
        ])
        .with_columns([
            ((pl.col("day") + SIMULATION_START_DOW) % 7)
              .map_elements(lambda x: WEEKDAY_NAMES[x], return_dtype=pl.Utf8)
              .alias("weekday_name"),
        ])
        .with_columns([
            (
                (pl.col("simtime_ms") // 1000 // 3600).cast(pl.Utf8).str.zfill(2)
                + pl.lit(":")
                + ((pl.col("simtime_ms") // 1000 % 3600) // 60).cast(pl.Utf8).str.zfill(2)
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
    out_analysis.mkdir(parents=True, exist_ok=True)

    snapshot_df.write_parquet(out_analysis / "waittime_snapshots.parquet")
    print(f"  Saved waittime_snapshots.parquet ({len(snapshot_df)} rows)")

    percentile_df = (
        snapshot_df
        .group_by(["weekday_name", "simtime_ms", "time_label"])
        .agg(
            [pl.col("wait_minutes").quantile(q).alias(f"wait_p{int(q*100)}")
             for q in PERCENTILES]
            + [pl.col("EVId").count().alias("ev_count")]
        )
        .sort(["weekday_name", "simtime_ms"])
    )

    out_percentiles = output_root / run_id / "percentiles" / "waittime"
    out_percentiles.mkdir(parents=True, exist_ok=True)

    percentile_df.write_parquet(out_percentiles / "waittime_percentiles.parquet")
    print(f"  Saved waittime_percentiles.parquet ({len(percentile_df)} rows)")