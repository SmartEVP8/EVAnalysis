"""
Module for analysing aggregate station-level metrics.

Reads raw StationSnapshotMetric data, validates the schema, derives KPIs,
and writes:
  - station_snapshots.parquet  : enriched per-row snapshot log
  - station_percentiles.parquet: network-wide percentiles grouped by weekday + time slot

Station utilization is computed as a capacity-weighted mean across all chargers
belonging to a station at each snapshot bucket:
  sum(Utilization * MaxKWh) / sum(MaxKWh)

This weights each charger's utilization by its capacity, so a 250 kWh charger
contributes proportionally more than a 100 kWh charger. Sourced from
charger_snapshots.parquet produced by analyse_charger().
"""

from pathlib import Path

import polars as pl

from helpers.loader import add_day_columns_to_parquet
from helpers.type_schemas import STATION_SCHEMA, validate_schema
from helpers.constants import OUTPUT_ROOT, PERCENTILES
from helpers.io_helpers import save_parquet


def _load_charger_utilization(run_id: str, output_root: Path) -> pl.DataFrame:
    """
    Computes per-station, per-bucket utilization from charger_snapshots.parquet.

    Utilization = sum(Utilization * MaxKWh) / sum(MaxKWh) 
    a capacity-weighted mean across all chargers at a station for each snapshot bucket. Buckets where
    sum(MaxKWh) == 0 produce null.
    """
    charger_snapshots_path = output_root / run_id / "analysis" / "charger_snapshots.parquet"
    if not charger_snapshots_path.exists():
        raise FileNotFoundError(
            f"charger_snapshots.parquet not found at {charger_snapshots_path}. "
            "Run analyse_charger() before analyse_station()."
        )

    return (
        pl.read_parquet(charger_snapshots_path)
        .with_columns([
            (pl.col("Utilization") * pl.col("MaxKWh")).alias("weighted_utilization"),
        ])
        .group_by(["StationId", "simtime_ms"])
        .agg([
            pl.col("weighted_utilization").sum().alias("total_weighted_utilization"),
            pl.col("MaxKWh").sum().alias("total_max_kwh"),
        ])
        .with_columns([
            pl.when(pl.col("total_max_kwh") > 0)
              .then(pl.col("total_weighted_utilization") / pl.col("total_max_kwh"))
              .otherwise(None)
              .alias("utilization")
        ])
        .select(["StationId", "simtime_ms", "utilization"])
    )


def analyse_station(
    parquet_path: Path,
    run_id: str,
    output_root: Path = OUTPUT_ROOT,
) -> None:
    """
    Analyses station snapshot data for a single simulation run.

    Derives utilization, cancellation
    rate, and expected wait time; validates the schema; and exports a snapshot
    log plus aggregated percentiles.
    """
    print(f"\n[Station] Analysing {parquet_path.name}...")

    df = add_day_columns_to_parquet(parquet_path)
    validate_schema(df, STATION_SCHEMA, "StationSnapshotMetric")

    charger_utilization = _load_charger_utilization(run_id, output_root)

    snapshot_df = (
        df
        .with_columns([
            (pl.col("ExpectedWaitTimeMiliseconds") / 60_000).cast(pl.Int64)
              .alias("expected_wait_minutes"),

            pl.when(pl.col("Reservations") > 0)
              .then(pl.col("Cancellations") / pl.col("Reservations"))
              .otherwise(None)
              .alias("cancellation_rate"),
        ])
        .join(charger_utilization, on=["StationId", "simtime_ms"], how="left")
        .select([
            "StationId", "day", "weekday_name", "simtime_ms", "time_label",
            "utilization", "Price", "Reservations", "Cancellations",
            "cancellation_rate", "TotalChargers",
            "ExpectedWaitTimeMiliseconds", "expected_wait_minutes",
        ])
        .sort(["StationId", "day", "simtime_ms"])
    )

    out_analysis = output_root / run_id / "analysis"
    save_parquet(snapshot_df, out_analysis / "station_snapshots.parquet", "[Station]")

    percentile_df = (
        snapshot_df
        .group_by(["weekday_name", "simtime_ms", "time_label"])
        .agg(
            [pl.col("utilization").quantile(q).alias(f"utilization_p{int(q * 100)}")
             for q in PERCENTILES]
            + [pl.col("Price").quantile(q).alias(f"price_p{int(q * 100)}")
               for q in PERCENTILES]
            + [pl.col("expected_wait_minutes").quantile(q).alias(f"wait_time_p{int(q * 100)}")
               for q in PERCENTILES]
        )
        .sort(["weekday_name", "simtime_ms"])
    )

    out_station_percentiles = output_root / run_id / "percentiles" / "station"
    save_parquet(percentile_df, out_station_percentiles / "station_percentiles.parquet", "[Station]")