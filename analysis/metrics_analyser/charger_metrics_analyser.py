from pathlib import Path
import polars as pl
from init.loader import add_day_columns_to_parquet
from .type_schemas import CHARGER_SCHEMA, validate_schema

OUTPUT_ROOT = Path("runs")

PERCENTILES = [0.25, 0.50, 0.75, 0.90, 0.95]


def analyse_charger(parquet_path: Path, run_id: str) -> None:
    print(f"\n[Charger] Analysing {parquet_path.name}...")

    df = add_day_columns_to_parquet(parquet_path)
    validate_schema(df, CHARGER_SCHEMA, "ChargerSnapshotMetric")

    snapshot_df = df.select([
        "StationId", "ChargerId",
        "day", "weekday_idx", "weekday_name",
        "time_of_day", "time_label",
        "Utilization", "QueueSize",
        "DeliveredKW", "TargetEVDemandKW",
    ])

    out_analysis = OUTPUT_ROOT / run_id / "analysis"
    out_analysis.mkdir(parents=True, exist_ok=True)

    snapshot_df = snapshot_df.sort(
        ["StationId", "ChargerId", "day", "time_of_day"]
    )

    snapshot_df.write_parquet(out_analysis / "charger_snapshots.parquet")

    print(f"  Saved charger_snapshots.parquet ({len(snapshot_df)} rows)")

    out_percentiles = OUTPUT_ROOT / run_id / "percentiles" / "charger"
    out_percentiles.mkdir(parents=True, exist_ok=True)

    clean_df = snapshot_df.drop_nulls(["Utilization", "QueueSize"])

    percentile_df = (
        clean_df
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