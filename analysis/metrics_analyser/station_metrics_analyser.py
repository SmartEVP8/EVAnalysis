from pathlib import Path
import polars as pl
from init.loader import add_day_columns_to_parquet
from .type_schemas import STATION_SCHEMA, validate_schema

OUTPUT_ROOT = Path("runs")

PERCENTILES = [0.25, 0.50, 0.75, 0.90, 0.95]


def analyse_station(parquet_path: Path, run_id: str) -> None:
    print(f"\n[Station] Analysing {parquet_path.name}...")

    df = add_day_columns_to_parquet(parquet_path)
    validate_schema(df, STATION_SCHEMA, "StationSnapshotMetric")

    snapshot_df = (
        df.with_columns([
            pl.when(pl.col("TotalMaxKWh") > 0)
              .then(pl.col("TotalDeliveredKWh") / pl.col("TotalMaxKWh"))
              .otherwise(None)
              .alias("utilization"),

            pl.col("TotalQueueSize").alias("total_queue_size"),

            pl.when(pl.col("Reservations") > 0)
              .then(pl.col("Cancellations") / pl.col("Reservations"))
              .otherwise(None)
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
            "total_queue_size",
            "Price",

            "Reservations",
            "Cancellations",
            "cancellation_rate",
            "TotalChargers",
        ])
    )

    out_analysis = OUTPUT_ROOT / run_id / "analysis"
    out_analysis.mkdir(parents=True, exist_ok=True)

    snapshot_df = snapshot_df.sort(
        ["StationId", "day", "time_of_day"]
    )

    snapshot_df.write_parquet(out_analysis / "station_snapshots.parquet")

    print(f"  Saved station_snapshots.parquet ({len(snapshot_df)} rows)")


    out_percentiles = OUTPUT_ROOT / run_id / "percentiles" / "station"
    out_percentiles.mkdir(parents=True, exist_ok=True)

    clean_df = snapshot_df.drop_nulls(["utilization", "total_queue_size", "Price"])

    percentile_df = (
        clean_df
        .group_by(["weekday_name", "time_of_day", "time_label"])
        .agg(
            [pl.col("utilization").quantile(q).alias(f"utilization_p{int(q*100)}")
             for q in PERCENTILES]
            +
            [pl.col("total_queue_size").quantile(q).alias(f"queue_size_p{int(q*100)}")
             for q in PERCENTILES]
            +
            [pl.col("Price").quantile(q).alias(f"price_p{int(q*100)}")
             for q in PERCENTILES]
        )
        .sort(["weekday_name", "time_of_day"])
    )

    out_path = out_percentiles / "station_percentiles_global.parquet"
    percentile_df.write_parquet(out_path)

    print(f"  Saved station_percentiles_global.parquet ({len(percentile_df)} rows)")