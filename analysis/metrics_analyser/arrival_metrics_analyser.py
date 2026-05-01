"""
Module for analysing EV arrival and deadline metrics.

Processes ArrivalAtDestinationMetric snapshot data and writes:
  - arrival_snapshots.parquet : enriched per-row log with deviation / delta buckets
  - arrival_buckets.parquet   : per-time-slot aggregates of deadline misses and
                                path-deviation / arrival-delta bucket counts

EVs with DriveDirectlyToDestination=True are excluded from all charging-related
statistics because they never interacted
with the charging network.
"""

from pathlib import Path

import numpy as np
import polars as pl

from helpers.loader import add_arrival_day_columns_to_parquet
from helpers.type_schemas import ARRIVE_AT_DESTINATION_SCHEMA, validate_schema
from helpers.constants import OUTPUT_ROOT
from helpers.io_helpers import save_parquet


DEVIATION_BUCKET_BREAKS: list[int]  = [5, 10, 15, 30, 60]
DEVIATION_BUCKET_LABELS: list[str]  = ["0-5", "5-10", "10-15", "15-30", "30-60", "60+"]


def load_snapshot_time_buckets(run_id: str, output_root: Path) -> pl.Series:
    """
    Returns the sorted unique simtime_ms values from station_snapshots.parquet.

    Arrival events are snapped to these buckets so that all downstream
    aggregations share a consistent time grid.
    """
    station_snapshots_path = output_root / run_id / "analysis" / "station_snapshots.parquet"
    if not station_snapshots_path.exists():
        raise FileNotFoundError(
            f"station_snapshots.parquet not found at {station_snapshots_path}. "
            "Run analyse_station() before analyse_arrival()."
        )
    return (
        pl.read_parquet(station_snapshots_path)
        .select("simtime_ms")
        .unique()
        .sort("simtime_ms")
        ["simtime_ms"]
    )


def snap_to_nearest_bucket(df: pl.DataFrame, buckets: pl.Series) -> pl.DataFrame:
    """
    Replaces each row's simtime_ms with the nearest value in buckets and
    updates time_label accordingly.
    """
    bucket_array: np.ndarray  = buckets.to_numpy()
    arrival_array: np.ndarray = df["simtime_ms"].to_numpy()

    right_idx = np.searchsorted(bucket_array, arrival_array)
    left_idx  = np.clip(right_idx - 1, 0, len(bucket_array) - 1)
    right_idx = np.clip(right_idx, 0, len(bucket_array) - 1)

    left_dist  = np.abs(arrival_array - bucket_array[left_idx])
    right_dist = np.abs(arrival_array - bucket_array[right_idx])
    snapped: np.ndarray = bucket_array[
        np.where(left_dist <= right_dist, left_idx, right_idx)
    ]

    snapped_series = pl.Series("simtime_ms", snapped).cast(pl.Int64)
    hours = (snapped_series // 1_000 // 3_600).cast(pl.Utf8).str.zfill(2)
    minutes = ((snapped_series // 1_000 % 3_600) // 60).cast(pl.Utf8).str.zfill(2)
    time_labels = hours + pl.Series([":" for _ in range(len(snapped))]) + minutes

    return df.with_columns([
        snapped_series,
        time_labels.alias("time_label"),
    ])


def build_bucket_agg_exprs(col: str, prefix: str) -> list[pl.Expr]:
    """Returns one count expression per deviation label for col."""
    return [
        (pl.col(col) == label).sum().alias(f"{prefix}_{label}")
        for label in DEVIATION_BUCKET_LABELS
    ]


def analyse_arrival(
    parquet_path: Path,
    run_id: str,
    output_root: Path = OUTPUT_ROOT,
) -> None:
    """
    Analyses EV arrival deadline compliance and path deviation for a simulation run.

    Per time slot, the output contains:
      - missed_deadline_pct     : share of EVs (excluding direct-drive) that missed (0–100)
      - missed_deadline_count   : raw count of misses
      - total_arrivals          : total non-direct-drive EVs in this time slot
      - path_deviation_<bucket> : count of EVs in each deviation bucket
      - delta_arrival_<bucket>  : count of EVs in each arrival-delta bucket
    """
    print(f"\n[Arrival] Analysing {parquet_path.name}...")

    df = add_arrival_day_columns_to_parquet(parquet_path)
    validate_schema(df, ARRIVE_AT_DESTINATION_SCHEMA, "ArrivalAtDestinationMetric")

    snapshot_df = (
        df
        .with_columns([
            (pl.col("PathDeviation") / 1_000 / 60).alias("path_deviation_minutes"),
            (
                (pl.col("ActualArrivalTime") - pl.col("ExpectedArrivalTime"))
                .clip(lower_bound=0) / 1_000 / 60
            ).alias("delta_arrival_minutes"),
            pl.col("MissedDeadline").cast(pl.Boolean).alias("missed_deadline"),
            pl.col("DriveDirectlyToDestination").cast(pl.Boolean).alias("drive_directly"),
        ])
        .with_columns([
            pl.col("path_deviation_minutes")
              .cut(breaks=DEVIATION_BUCKET_BREAKS, labels=DEVIATION_BUCKET_LABELS)
              .alias("path_deviation_bucket"),
            pl.col("delta_arrival_minutes")
              .cut(breaks=DEVIATION_BUCKET_BREAKS, labels=DEVIATION_BUCKET_LABELS)
              .alias("delta_arrival_bucket"),
        ])
        .select([
            "day", "weekday_name", "simtime_ms", "time_label",
            "ExpectedArrivalTime", "ActualArrivalTime",
            "path_deviation_minutes", "delta_arrival_minutes",
            "path_deviation_bucket", "delta_arrival_bucket",
            "missed_deadline", "drive_directly",
        ])
        .sort(["day", "simtime_ms"])
    )

    out_analysis = output_root / run_id / "analysis"
    save_parquet(snapshot_df, out_analysis / "arrival_snapshots.parquet", "[Arrival]")

    # Snap arrival events onto the station snapshot time grid before aggregating.
    time_buckets = load_snapshot_time_buckets(run_id, output_root)
    agg_df_snapped = snap_to_nearest_bucket(snapshot_df, time_buckets)

    # Exclude direct-drive EVs: they never used the charging network.
    agg_input = agg_df_snapped.filter(pl.col("drive_directly") == False)

    agg_df = (
        agg_input
        .group_by(["weekday_name", "simtime_ms", "time_label"])
        .agg(
            [
                (pl.col("missed_deadline").sum() / pl.col("missed_deadline").count() * 100)
                  .alias("missed_deadline_pct"),
                pl.col("missed_deadline").sum().alias("missed_deadline_count"),
                pl.col("missed_deadline").count().alias("total_arrivals"),
            ]
            + build_bucket_agg_exprs("path_deviation_bucket", "path_deviation")
            + build_bucket_agg_exprs("delta_arrival_bucket",  "delta_arrival")
        )
        .sort(["weekday_name", "simtime_ms"])
    )

    buckets_output = output_root / run_id / "buckets" / "arrival"
    save_parquet(agg_df, buckets_output / "arrival_buckets.parquet", "[Arrival]")