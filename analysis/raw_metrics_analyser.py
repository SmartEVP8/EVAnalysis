"""
analyser.py
-----------
Computes QoS metrics from parquet files.

Charger-level metrics:
  - Utilization per snapshot (read directly from parquet)
  - P25 | P50 | P75 | P90 | P95 of utilization across all snapshots
  - Queue size per snapshot
  - P25 | P50 | P75 | P90 | P95 of queue size across all snapshots

Station-level metrics:
  - Utilization: sum(DeliveredKW) / sum(MaxKWh) across all chargers at the station
  - P25 | P50 | P75 | P90 | P95 of station utilization
  - Queue size: sum(charger queue sizes) / total chargers at the station
  - P25 | P50 | P75 | P90 | P95 of queue size
  - Cancellation rate: Cancellations / Reservations
  - Total reservations
  - Price

Results are written to:
    runs/{run_id}/analysis/charger_analysis.parquet
    runs/{run_id}/analysis/station_analysis.parquet
"""

from pathlib import Path

import polars as pl

from init.loader import load_and_enrich

OUTPUT_ROOT = Path("runs")

def analyse_charger(parquet_path: Path, run_id: str) -> None:
    print(f"\n[Charger] Analysing {parquet_path.name}...")
    df = load_and_enrich(parquet_path)

    # These are already computed per row by the simulation, we just keep them.
    snapshot_df = df.select([
        "StationId",
        "ChargerId",
        "day",
        "weekday_idx",
        "weekday_name",
        "time_of_day",
        "time_label",
        "Utilization",
        "QueueSize",
        "DeliveredKW",
        "TargetEVDemandKW",
    ])

    percentile_df = (
        df.group_by(["StationId", "ChargerId"])
        .agg([
            pl.col("Utilization").quantile(0.25).alias("utilization_p25"),
            pl.col("Utilization").quantile(0.50).alias("utilization_p50"),
            pl.col("Utilization").quantile(0.75).alias("utilization_p75"),
            pl.col("Utilization").quantile(0.90).alias("utilization_p90"),
            pl.col("Utilization").quantile(0.95).alias("utilization_p95"),

            pl.col("QueueSize").quantile(0.25).alias("queue_size_p25"),
            pl.col("QueueSize").quantile(0.50).alias("queue_size_p50"),
            pl.col("QueueSize").quantile(0.75).alias("queue_size_p75"),
            pl.col("QueueSize").quantile(0.90).alias("queue_size_p90"),
            pl.col("QueueSize").quantile(0.95).alias("queue_size_p95"),
        ])
    )

    out_dir = OUTPUT_ROOT / run_id / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = out_dir / "charger_snapshots.parquet"
    percentile_path = out_dir / "charger_percentiles.parquet"

    snapshot_df.sort(["StationId", "ChargerId", "day", "time_of_day"]).write_parquet(snapshot_path)
    percentile_df.sort(["StationId", "ChargerId"]).write_parquet(percentile_path)

    print(f"  Saved {snapshot_path}  ({len(snapshot_df)} rows)")
    print(f"  Saved {percentile_path}  ({len(percentile_df)} rows)")



def analyse_station(parquet_path: Path, run_id: str) -> None:
    print(f"\n[Station] Analysing {parquet_path.name}...")
    df = load_and_enrich(parquet_path)

    snapshot_df = (
        df.with_columns([
            (pl.col("TotalDeliveredKWh") / pl.col("TotalMaxKWh"))
                .alias("utilization"),

            (pl.col("TotalQueueSize") / pl.col("TotalChargers"))
                .alias("queue_size_per_charger"),

            pl.when(pl.col("Reservations") > 0)
                .then(pl.col("Cancellations") / pl.col("Reservations"))
                .otherwise(0.0)
                .alias("cancellation_rate"),
        ])
        .select([
            "StationId",
            "day",
            "weekday_idx",
            "weekday_name",
            "time_of_day",
            "time_label",
            "utilization",
            "queue_size_per_charger",
            "TotalQueueSize",
            "cancellation_rate",
            "Reservations",
            "Cancellations",
            "Price",
            "TotalChargers",
        ])
    )

    percentile_df = (
        snapshot_df.group_by("StationId")
        .agg([
            pl.col("utilization").quantile(0.25).alias("utilization_p25"),
            pl.col("utilization").quantile(0.50).alias("utilization_p50"),
            pl.col("utilization").quantile(0.75).alias("utilization_p75"),
            pl.col("utilization").quantile(0.90).alias("utilization_p90"),
            pl.col("utilization").quantile(0.95).alias("utilization_p95"),

            pl.col("queue_size_per_charger").quantile(0.25).alias("queue_size_p25"),
            pl.col("queue_size_per_charger").quantile(0.50).alias("queue_size_p50"),
            pl.col("queue_size_per_charger").quantile(0.75).alias("queue_size_p75"),
            pl.col("queue_size_per_charger").quantile(0.90).alias("queue_size_p90"),
            pl.col("queue_size_per_charger").quantile(0.95).alias("queue_size_p95"),
        ])
    )

    out_dir = OUTPUT_ROOT / run_id / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = out_dir / "station_snapshots.parquet"
    percentile_path = out_dir / "station_percentiles.parquet"

    snapshot_df.sort(["StationId", "day", "time_of_day"]).write_parquet(snapshot_path)
    percentile_df.sort("StationId").write_parquet(percentile_path)

    print(f"  Saved → {snapshot_path}  ({len(snapshot_df)} rows)")
    print(f"  Saved → {percentile_path}  ({len(percentile_df)} rows)")


# When the EV metric parquet is added, implement this function and call it from main.py alongside the others.
#
# def analyse_ev(parquet_path: Path, run_id: str) -> None:
#   something something at some point in the future