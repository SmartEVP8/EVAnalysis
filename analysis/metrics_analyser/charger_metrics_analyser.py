"""
Charger-level metrics:
  - Utilization per snapshot (read directly from parquet)
  - P25 | P50 | P75 | P90 | P95 of utilization — per day
  - Queue size per snapshot
  - P25 | P50 | P75 | P90 | P95 of queue size — per day

Results are written to:
    runs/{run_id}/analysis/charger_snapshots.parquet
    runs/{run_id}/percentiles/charger/charger_percentiles_{weekday}_{day}.parquet
"""

from pathlib import Path
import polars as pl
from init.loader import add_day_columns_to_parquet
from .type_schemas import CHARGER_SCHEMA, validate_schema

OUTPUT_ROOT = Path("runs")

PERCENTILES = [0.25, 0.50, 0.75, 0.90, 0.95]


def analyse_charger(parquet_path: Path, run_id: str) -> None:
    df = add_day_columns_to_parquet(parquet_path)
    validate_schema(df, CHARGER_SCHEMA, "ChargerSnapshotMetric")

    snapshot_df = df.select([
        "StationId", "ChargerId", "day", "weekday_idx", "weekday_name",
        "time_of_day", "time_label", "Utilization", "QueueSize",
        "DeliveredKW", "TargetEVDemandKW",
    ])

    # Snapshots as one file
    out_analysis = OUTPUT_ROOT / run_id / "analysis"
    out_analysis.mkdir(parents=True, exist_ok=True)
    snapshot_df.sort(["StationId", "ChargerId", "day", "time_of_day"]).write_parquet(out_analysis / "charger_snapshots.parquet")

    # Percentile files per day
    out_percentiles = OUTPUT_ROOT / run_id / "percentiles" / "charger"
    out_percentiles.mkdir(parents=True, exist_ok=True)

    for (day, weekday_name), group_df in snapshot_df.group_by(["day", "weekday_name"]):
        weekday_name_lower = weekday_name.lower()
        percentile_df = (
            group_df.group_by(["StationId", "ChargerId"])
            .agg(
                [pl.col("Utilization").quantile(q).alias(f"utilization_p{int(q*100)}")
                 for q in PERCENTILES]
                +
                [pl.col("QueueSize").quantile(q).alias(f"queue_size_p{int(q*100)}")
                 for q in PERCENTILES]
            )
            .sort(["StationId", "ChargerId"])
        )

        filename = f"charger_percentiles_{weekday_name_lower}_{day}.parquet"
        percentile_df.write_parquet(out_percentiles / filename)