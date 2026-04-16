"""
Module for analyzing EV arrival and deadline metrics.

Processes ArrivalAtDestinationMetric snapshot data to compute:
  - The percentage of EVs that missed their deadline per time slot.
  - The distribution of path deviation (in km) for late and on-time arrivals.

Output is saved as a single Parquet file for downstream dashboard use.
"""

from pathlib import Path
import polars as pl
from .type_schemas import ARRIVE_AT_DESTINATION_SCHEMA, validate_schema
from init.loader import add_arrival_day_columns_to_parquet

OUTPUT_ROOT = Path("runs")


def analyse_arrival(parquet_path: Path, run_id: str) -> None:
    """
    Analyses EV arrival deadline compliance and path deviation for a simulation run.

    Reads raw arrival snapshot data, enriches it with temporal metadata, then
    aggregates per time slot to produce:
      - missed_deadline_pct : share of EVs that missed their deadline (0–100)
      - path_deviation_minutes*  : percentile distribution of route deviation in minutes
      - delta_arrival_*      : percentile distribution of arrival time delta in milliseconds
    """
    print(f"\n[Arrival] Analysing {parquet_path.name}...")

    df = add_arrival_day_columns_to_parquet(parquet_path)

    validate_schema(df, ARRIVE_AT_DESTINATION_SCHEMA, "ArrivalAtDestinationMetric")

    snapshot_df = df.with_columns([
        (pl.col("PathDeviation") / 1000 / 60).alias("path_deviation_minutes"),

        (pl.col("DeltaArrivalTime")).alias("delta_arrival_ms"),

        pl.col("MissedDeadline").cast(pl.Boolean).alias("missed_deadline"),
    ]).select([
        "day", "weekday_name", "simtime_ms", "time_label",
        "ExpectedArrivalTime", "ActualArrivalTime",
        "path_deviation_minutes", "delta_arrival_ms", "missed_deadline",
    ]).sort(["day", "simtime_ms"])

    out_analysis = OUTPUT_ROOT / run_id / "analysis"
    out_analysis.mkdir(parents=True, exist_ok=True)

    snapshot_df.write_parquet(out_analysis / "arrival_snapshots.parquet")
    print(f"  Saved arrival_snapshots.parquet ({len(snapshot_df)} rows)")

    # Aggregate per (weekday, time slot)
    percentiles = [0.25, 0.50, 0.75, 0.90, 0.95]

    agg_df = (
        snapshot_df
        .group_by(["weekday_name", "simtime_ms", "time_label"])
        .agg(
            [
                # Percentage of EVs that missed their deadline in this slot
                (pl.col("missed_deadline").sum() / pl.col("missed_deadline").count() * 100)
                    .alias("missed_deadline_pct"),

                pl.col("missed_deadline").sum().alias("missed_deadline_count"),
                pl.col("missed_deadline").count().alias("total_arrivals"),
            ]
            + [pl.col("path_deviation_minutes").quantile(q).alias(f"path_deviation_minutes_p{int(q * 100)}")
               for q in percentiles]
            + [pl.col("delta_arrival_ms").quantile(q).alias(f"delta_arrival_s_p{int(q * 100)}")
               for q in percentiles]
        )
        .sort(["weekday_name", "simtime_ms"])
    )

    out_percentiles = OUTPUT_ROOT / run_id / "percentiles" / "arrival"
    out_percentiles.mkdir(parents=True, exist_ok=True)

    agg_df.write_parquet(out_percentiles / "arrival_percentiles.parquet")
    print(f"  Saved arrival_percentiles.parquet ({len(agg_df)} rows)")