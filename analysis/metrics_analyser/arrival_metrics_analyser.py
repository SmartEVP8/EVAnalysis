"""
Module for analyzing EV arrival and deadline metrics.

Processes ArrivalAtDestinationMetric snapshot data to compute:
  - The percentage of EVs that missed their deadline per time slot.
  - The distribution of path deviation (in km) for late and on-time arrivals.

Output is saved as a single Parquet file for downstream dashboard use.

EVs that drove directly to their destination (DriveDirectlyToDestination=true) are
excluded from missed-deadline and path-deviation statistics, because they never
interacted with the charging network and their path deviation is meaningless in
that context.
"""

from pathlib import Path
import polars as pl
import numpy as np
from .type_schemas import ARRIVE_AT_DESTINATION_SCHEMA, validate_schema
from init.loader import add_arrival_day_columns_to_parquet

OUTPUT_ROOT = Path("runs")


def load_snapshot_time_buckets(run_id: str, output_root: Path) -> pl.Series:
    """
    Returns the sorted unique simtime_ms values from station_snapshots.parquet.
    """
    station_snapshots_path = output_root / run_id / "analysis" / "station_snapshots.parquet"
    if not station_snapshots_path.exists():
        raise FileNotFoundError(
            f"station_snapshots.parquet not found at {station_snapshots_path}. "
            "Run analyse_station() before analyse_arrival()."
        )
    buckets = (
        pl.read_parquet(station_snapshots_path)
        .select("simtime_ms")
        .unique()
        .sort("simtime_ms")
        ["simtime_ms"]
    )
    return buckets


def snap_to_nearest_bucket(df: pl.DataFrame, buckets: pl.Series) -> pl.DataFrame:
    buckets_arrivals = buckets.to_numpy()

    arrival_milliseconds = df["simtime_ms"].to_numpy()

    right_index = np.searchsorted(buckets_arrivals, arrival_milliseconds)
    left_index  = np.clip(right_index - 1, 0, len(buckets_arrivals) - 1)
    right_index = np.clip(right_index, 0, len(buckets_arrivals) - 1)

    left_dist  = np.abs(arrival_milliseconds - buckets_arrivals[left_index])
    right_dist = np.abs(arrival_milliseconds - buckets_arrivals[right_index])

    nearest_index = np.where(left_dist <= right_dist, left_index, right_index)
    nearest_ticks = buckets_arrivals[nearest_index]

    return df.with_columns([
        pl.Series("simtime_ms", nearest_ticks).cast(pl.Int64),
        (
            ((pl.Series("simtime_ms", nearest_ticks) // 1000 // 3600).cast(pl.Utf8).str.zfill(2))
            + pl.lit(":")
            + (((pl.Series("simtime_ms", nearest_ticks) // 1000 % 3600) // 60).cast(pl.Utf8).str.zfill(2))
        ).alias("time_label"),
    ])

def analyse_arrival(parquet_path: Path, run_id: str, output_root: Path = OUTPUT_ROOT) -> None:
    """
    Analyses EV arrival deadline compliance and path deviation for a simulation run.

    Reads raw arrival snapshot data, enriches it with temporal metadata, snaps
    each EV's arrival time to the nearest station snapshot tick, then aggregates
    per time slot to produce:
      - missed_deadline_pct : share of EVs that missed their deadline (0–100)
      - path_deviation_minutes*  : percentile distribution of route deviation in minutes
      - delta_arrival_* : percentile distribution of arrival time delta in minutes
      - ev_wait_time : percentile distribution of wait time in queues for evs in minutes
    """
    print(f"\n[Arrival] Analysing {parquet_path.name}...")

    df = add_arrival_day_columns_to_parquet(parquet_path)

    validate_schema(df, ARRIVE_AT_DESTINATION_SCHEMA, "ArrivalAtDestinationMetric")

    snapshot_df = df.with_columns([
        (pl.col("PathDeviation") / 1000 / 60).alias("path_deviation_minutes"),
        (pl.col("DeltaArrivalTime") / 1000 / 60).alias("delta_arrival_minutes"),
        pl.col("MissedDeadline").cast(pl.Boolean).alias("missed_deadline"),
        pl.col("DriveDirectlyToDestination").cast(pl.Boolean).alias("drive_directly"),
    ]).select([
        "day", "weekday_name", "simtime_ms", "time_label",
        "ExpectedArrivalTime", "ActualArrivalTime",
        "path_deviation_minutes", "delta_arrival_minutes", "missed_deadline",
        "drive_directly",
    ]).sort(["day", "simtime_ms"])

    out_analysis = output_root / run_id / "analysis"
    out_analysis.mkdir(parents=True, exist_ok=True)

    snapshot_df.write_parquet(out_analysis / "arrival_snapshots.parquet")
    print(f"  Saved arrival_snapshots.parquet ({len(snapshot_df)} rows)")

    time_buckets = load_snapshot_time_buckets(run_id, output_root)
    snapshot_df = snap_to_nearest_bucket(snapshot_df, time_buckets)

    # Exclude direct-drive EVs from charging-related metrics
    snapshot_df = snapshot_df.filter(pl.col("drive_directly") == False)

    percentiles = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

    agg_df = (
        snapshot_df
        .group_by(["weekday_name", "simtime_ms", "time_label"])
        .agg(
            [
                (pl.col("missed_deadline").sum() / pl.col("missed_deadline").count() * 100)
                    .alias("missed_deadline_pct"),

                pl.col("missed_deadline").sum().alias("missed_deadline_count"),
                pl.col("missed_deadline").count().alias("total_arrivals"),
            ]
            + [pl.col("path_deviation_minutes").quantile(q).alias(f"path_deviation_minutes_p{int(q * 100)}")
               for q in percentiles]
            + [pl.col("delta_arrival_minutes").quantile(q).alias(f"delta_arrival_minutes_p{int(q * 100)}")
               for q in percentiles]
        )
        .sort(["weekday_name", "simtime_ms"])
    )

    out_percentiles = output_root / run_id / "percentiles" / "arrival"
    out_percentiles.mkdir(parents=True, exist_ok=True)

    agg_df.write_parquet(out_percentiles / "arrival_percentiles.parquet")
    print(f"  Saved arrival_percentiles.parquet ({len(agg_df)} rows)")