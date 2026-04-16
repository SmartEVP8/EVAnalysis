"""
Module for analyzing electric vehicle charger metrics.
Provides functionality to process snapshot data, validate schemas, 
and generate statistical analysis for utilization and queue'ing.
"""

from pathlib import Path
import polars as pl
from init.loader import add_day_columns_to_parquet
from .type_schemas import CHARGER_SCHEMA, validate_schema

OUTPUT_ROOT = Path("runs")

PERCENTILES = [0.25, 0.50, 0.75, 0.90, 0.95]


def analyse_charger(parquet_path: Path, run_id: str) -> None:
    """
    Performs analysis on charger snapshot data for a specific simulation run.
    
    This function reads raw parquet data, adds temporal metadata (days of simulation run (e.g., 0, 1, 2)),
    weekdays (e.g., Monday, Tuesday), and time labels (e.g., "08:00-09:00"), validates the schema integrity,
    and exports both a sorted snapshot log and aggregated global percentiles for charger utilization and queue sizes.

    Args:
        parquet_path (Path): The file path to the input parquet file containing charger metrics.
        run_id (str): The unique identifier for the current simulation run, used for output organization.

    Raises:
        SchemaValidationError: If the input dataframe does not match CHARGER_SCHEMA.
        FileNotFoundError: If the parquet_path does not exist.
    """
    print(f"\n[Charger] Analysing {parquet_path.name}...")

    # Load data and add time-based columns
    df = add_day_columns_to_parquet(parquet_path)
    
    validate_schema(df, CHARGER_SCHEMA, "ChargerSnapshotMetric")

    out_analysis = OUTPUT_ROOT / run_id / "analysis"
    out_analysis.mkdir(parents=True, exist_ok=True)

    snapshot_df = df.select([
        "StationId", "ChargerId",
        "day", "weekday_idx", "weekday_name",
        "time_of_day", "time_label",
        "Utilization", "QueueSize",
        "DeliveredKW", "TargetEVDemandKW",
    ]).sort(["StationId", "ChargerId", "day", "time_of_day"])

    snapshot_df.write_parquet(out_analysis / "charger_snapshots.parquet")

    print(f"  Saved charger_snapshots.parquet ({len(snapshot_df)} rows)")

    out_percentiles = OUTPUT_ROOT / run_id / "percentiles" / "charger"
    out_percentiles.mkdir(parents=True, exist_ok=True)

    # Calculate global percentiles grouped by time of day and weekday
    percentile_df = (
        snapshot_df
        .group_by(["weekday_name", "time_of_day", "time_label"])
        .agg(
            [pl.col("Utilization").quantile(q).alias(f"utilization_p{int(q*100)}")
             for q in PERCENTILES]
            +
            [pl.col("QueueSize").quantile(q).alias(f"queue_size_p{int(q*100)}")
             for q in PERCENTILES]
        )
        .sort(["weekday_name", "time_of_day"])
    )

    out_path = out_percentiles / "charger_percentiles_global.parquet"
    percentile_df.write_parquet(out_path)

    print(f"  Saved charger_percentiles_global.parquet ({len(percentile_df)} rows)")