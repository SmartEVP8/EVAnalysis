"""
Station-level metrics:
  - Utilization: TotalDeliveredKWh / TotalMaxKWh
  - P25 | P50 | P75 | P90 | P95 of station utilization
  - Queue size: TotalQueueSize / TotalChargers
  - P25 | P50 | P75 | P90 | P95 of queue size
  - Cancellation rate: Cancellations / Reservations
  - Total reservations
  - Price

Results are written to:
    runs/{run_id}/analysis/station_snapshots.parquet
    runs/{run_id}/analysis/station_percentiles.parquet
"""

from pathlib import Path
import polars as pl
from init.loader import add_day_columns_to_parquet
from .type_schemas import STATION_SCHEMA, validate_schema

OUTPUT_ROOT = Path("runs")

def analyse_station(parquet_path: Path, run_id: str) -> None:
    print(f"\n[Station] Analysing {parquet_path.name}...")
    df = add_day_columns_to_parquet(parquet_path)
    validate_schema(df, STATION_SCHEMA, "StationSnapshotMetric")

    snapshot_df = (
        df.with_columns([
            (pl.col("TotalDeliveredKWh") / pl.col("TotalMaxKWh")).alias("utilization"),
            (pl.col("TotalQueueSize") / pl.col("TotalChargers")).alias("queue_size_per_charger"),
            pl.when(pl.col("Reservations") > 0)
                .then(pl.col("Cancellations") / pl.col("Reservations"))
                .otherwise(0.0).alias("cancellation_rate"),
        ])
        .select([
            "StationId", "day", "weekday_idx", "weekday_name", "time_of_day",
            "time_label", "utilization", "queue_size_per_charger", "TotalQueueSize",
            "cancellation_rate", "Reservations", "Cancellations", "Price", "TotalChargers",
        ])
    )

    percentile_df = (
        snapshot_df.group_by("StationId")
        .agg([
            pl.col("utilization").quantile(q).alias(f"utilization_p{int(q*100)}") 
            for q in [0.25, 0.50, 0.75, 0.90, 0.95]
        ] + [
            pl.col("queue_size_per_charger").quantile(q).alias(f"queue_size_p{int(q*100)}") 
            for q in [0.25, 0.50, 0.75, 0.90, 0.95]
        ])
    )

    out_dir = OUTPUT_ROOT / run_id / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    snapshot_df.sort(["StationId", "day", "time_of_day"]).write_parquet(out_dir / "station_snapshots.parquet")
    percentile_df.sort("StationId").write_parquet(out_dir / "station_percentiles.parquet")