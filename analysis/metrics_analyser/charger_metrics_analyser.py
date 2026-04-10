"""
Charger-level metrics:
  - Utilization per snapshot (read directly from parquet)
  - P25 | P50 | P75 | P90 | P95 of utilization across all snapshots
  - Queue size per snapshot
  - P25 | P50 | P75 | P90 | P95 of queue size across all snapshots

Results are written to:
    runs/{run_id}/analysis/charger_snapshots.parquet
    runs/{run_id}/analysis/charger_percentiles.parquet
"""


from pathlib import Path
import polars as pl
from init.loader import add_day_columns_to_parquet
from .type_schemas import CHARGER_SCHEMA, validate_schema

OUTPUT_ROOT = Path("runs")

def analyse_charger(parquet_path: Path, run_id: str) -> None:
    print(f"\n[Charger] Analysing {parquet_path.name}...")
    df = add_day_columns_to_parquet(parquet_path)
    validate_schema(df, CHARGER_SCHEMA, "ChargerSnapshotMetric")

    snapshot_df = df.select([
        "StationId", "ChargerId", "day", "weekday_idx", "weekday_name",
        "time_of_day", "time_label", "Utilization", "QueueSize",
        "DeliveredKW", "TargetEVDemandKW",
    ])

    percentile_df = (
        df.group_by(["StationId", "ChargerId"])
        .agg([
            pl.col("Utilization").quantile(q).alias(f"utilization_p{int(q*100)}") 
            for q in [0.25, 0.50, 0.75, 0.90, 0.95]
        ] + [
            pl.col("QueueSize").quantile(q).alias(f"queue_size_p{int(q*100)}") 
            for q in [0.25, 0.50, 0.75, 0.90, 0.95]
        ])
    )

    out_dir = OUTPUT_ROOT / run_id / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    snapshot_df.sort(["StationId", "ChargerId", "day", "time_of_day"]).write_parquet(out_dir / "charger_snapshots.parquet")
    percentile_df.sort(["StationId", "ChargerId"]).write_parquet(out_dir / "charger_percentiles.parquet")