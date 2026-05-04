"""
Module for analysing EV charger metrics.

Reads raw ChargerSnapshotMetric data, validates the schema, and writes:
  - charger_snapshots.parquet  : sorted per-row snapshot log
  - charger_percentiles.parquet: utilization percentiles grouped by weekday + time slot
"""

from pathlib import Path

import polars as pl

from helpers.loader import add_day_columns_to_parquet
from helpers.type_schemas import CHARGER_SCHEMA, validate_schema
from helpers.constants import OUTPUT_ROOT, PERCENTILES
from helpers.io_helpers import save_parquet


def analyse_charger(
    parquet_path: Path,
    run_id: str,
    output_root: Path = OUTPUT_ROOT,
) -> None:
    """
    Analyses charger snapshot data for a single simulation run.

    Reads raw Parquet data, enriches it with temporal metadata, validates the
    schema, and exports a snapshot log plus aggregated utilization percentiles.

    Args:
        parquet_path: Path to ChargerSnapshotMetric.parquet.
        run_id:       Simulation run identifier (e.g. 'Run_001').
        output_root:  Root directory under which all run output is written.
    """
    print(f"\n[Charger] Analysing {parquet_path.name}...")

    df = add_day_columns_to_parquet(parquet_path)
    validate_schema(df, CHARGER_SCHEMA, "ChargerSnapshotMetric")

    snapshot_df = (
        df
        .select([
            "StationId", "ChargerId",
            "day", "weekday_name",
            "simtime_ms", "time_label",
            "Utilization", "MaxKWh",
        ])
        .sort(["StationId", "ChargerId", "day", "simtime_ms"])
    )

    out_analysis = output_root / run_id / "analysis"
    save_parquet(snapshot_df, out_analysis / "charger_snapshots.parquet", "[Charger]")

    percentile_df = (
        snapshot_df
        .group_by(["weekday_name", "simtime_ms", "time_label"])
        .agg([
            pl.col("Utilization").quantile(q).alias(f"utilization_p{int(q * 100)}")
            for q in PERCENTILES
        ])
        .sort(["weekday_name", "simtime_ms"])
    )

    out_percentiles = output_root / run_id / "percentiles" / "charger"
    save_parquet(percentile_df, out_percentiles / "charger_percentiles.parquet", "[Charger]")