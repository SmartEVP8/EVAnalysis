"""
Module for analyzing aggregate station-level metrics.
Calculates high-level KPIs such as station utilization, cancellation rates, 
and price distributions across the charging network.
"""

from pathlib import Path
import polars as pl
from init.loader import add_day_columns_to_parquet
from .type_schemas import STATION_SCHEMA, validate_schema

OUTPUT_ROOT = Path("runs")

PERCENTILES = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def analyse_station(parquet_path: Path, run_id: str, output_root: Path = OUTPUT_ROOT) -> None:
    """
    Performs aggregate analysis on station metrics for a specific simulation run.
    
    This function processes station snapshot data to derive key performance 
    indicators (KPIs), handles division-by-zero guards, and exports logs
    and statistical summaries.
    """
    print(f"\n[Station] Analysing {parquet_path.name}...")

    # Load data and add time-based columns
    df = add_day_columns_to_parquet(parquet_path)
    
    validate_schema(df, STATION_SCHEMA, "StationSnapshotMetric")

    out_analysis = output_root / run_id / "analysis"
    out_analysis.mkdir(parents=True, exist_ok=True)


    snapshot_df = (
        df.with_columns([
            pl.when(pl.col("TotalMaxKWh") > 0)
              .then(pl.col("TotalDeliveredKWh") / pl.col("TotalMaxKWh"))
              .otherwise(None)
              .alias("utilization"),
            
            (pl.col("ExpectedWaitTimeMiliseconds") / 60_000).alias("expected_wait_minutes"),

            pl.when(pl.col("Reservations") > 0)
              .then(pl.col("Cancellations") / pl.col("Reservations"))
              .otherwise(None)
              .alias("cancellation_rate"),
        ])
        .select([
            "StationId", "day", "weekday_name", "simtime_ms", 
            "time_label", "utilization", "Price", "Reservations", 
            "Cancellations", "cancellation_rate", "TotalChargers", 
            "ExpectedWaitTimeMiliseconds", "expected_wait_minutes",
        ])
    )

    snapshot_df = snapshot_df.sort(
        ["StationId", "day", "simtime_ms"]
    )
    snapshot_df.write_parquet(out_analysis / "station_snapshots.parquet")

    print(f"  Saved station_snapshots.parquet ({len(snapshot_df)} rows)")

    out_percentiles = output_root / run_id / "percentiles" / "station"
    out_percentiles.mkdir(parents=True, exist_ok=True)

    # Aggregate percentiles across all stations to see network-wide trends
    percentile_df = (
        snapshot_df
        .group_by(["weekday_name", "simtime_ms", "time_label"])
        .agg(
            [pl.col("utilization").quantile(q).alias(f"utilization_p{int(q*100)}")
             for q in PERCENTILES]
            +
            [pl.col("Price").quantile(q).alias(f"price_p{int(q*100)}")
             for q in PERCENTILES]
             +
            [pl.col("expected_wait_minutes").quantile(q).alias(f"wait_time_p{int(q*100)}")
             for q in PERCENTILES]
        )
        .sort(["weekday_name", "simtime_ms"])
    )

    out_path = out_percentiles / "station_percentiles.parquet"
    percentile_df.write_parquet(out_path)

    print(f"  Saved station_percentiles.parquet ({len(percentile_df)} rows)")
