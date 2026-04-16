"""
Identifies statistical anomalies in station and charger performance using
the Interquartile Range (IQR) method.
A reading is flagged when it falls more than
INTERQUARTILE_RANGE_MULTIPLIER multiplied by IQR above the 75th percentile
or below the 25th percentile of its peer group.
"""

from pathlib import Path
import polars as pl

OUTPUT_ROOT = Path("runs")

INTERQUARTILE_RANGE_MULTIPLIER = 3

STATION_METRICS = ["utilization", "total_queue_size"]

CHARGER_METRICS = ["Utilization", "QueueSize"]


def detect_outliers(
    df: pl.DataFrame,
    id_cols: list[str],
    metric_cols: list[str],
    label: str,
) -> pl.DataFrame:
    """
    Flags rows whose metric values fall outside the IQR-based Tukey fences
    for their (weekday, time-of-day) peer group.
    """
    all_flags: list[pl.DataFrame] = []
    group_cols = ["weekday_name", "time_of_day"]

    for metric in metric_cols:
        if metric not in df.columns:
            continue

        # IQR bounds: computed across all entities at each (weekday, time) slot
        ranges_df = (
            df.group_by(group_cols)
            .agg([
                pl.col(metric).cast(pl.Float32).quantile(0.25).alias("p25"),
                pl.col(metric).cast(pl.Float32).quantile(0.75).alias("p75"),
            ])
            .with_columns(
                (pl.col("p75") - pl.col("p25")).alias("interquartile_range")
            )
            .with_columns([
                (pl.col("p75") + INTERQUARTILE_RANGE_MULTIPLIER * pl.col("interquartile_range")).alias("upper"),
                (pl.col("p25") - INTERQUARTILE_RANGE_MULTIPLIER * pl.col("interquartile_range")).alias("lower"),
            ])
        )

        outliers = (
            df.with_columns(pl.col(metric).cast(pl.Float32))
            .join(ranges_df, on=group_cols)
            .filter(
                (pl.col(metric) > pl.col("upper")) | (pl.col(metric) < pl.col("lower"))
            )
            .with_columns([
                pl.col(metric).alias("value"),
                pl.lit(metric).alias("metric"),
                pl.lit(label).alias("label"),
                pl.when(pl.col(metric) > pl.col("upper"))
                  .then(pl.lit("HIGH"))
                  .otherwise(pl.lit("LOW"))
                  .alias("flag"),
            ])
            .select(id_cols + group_cols + ["time_label", "metric", "label", "flag", "value", "p25", "p75", "upper", "lower"])
        )

        if not outliers.is_empty():
            all_flags.append(outliers)

    if not all_flags:
        return pl.DataFrame()

    return pl.concat(all_flags)


def process_outliers(run_id: str) -> None:
    """
    Loads snapshot files for a simulation run, detects outliers, and writes
    the results to Parquet.
    """
    run_root = OUTPUT_ROOT / run_id
    analysis_dir = run_root / "analysis"
    outliers_dir = run_root / "outliers"
    outliers_dir.mkdir(exist_ok=True)

    station_path = analysis_dir / "station_snapshots.parquet"
    if station_path.exists():
        df = pl.read_parquet(station_path)
        outliers = detect_outliers(df, ["StationId"], STATION_METRICS, "Station")
        if not outliers.is_empty():
            outliers.write_parquet(outliers_dir / "station_outliers.parquet")
            print(f"Found {len(outliers)} station anomalies.")
        else:
            print("No station outliers found.")

    charger_path = analysis_dir / "charger_snapshots.parquet"
    if charger_path.exists():
        df = pl.read_parquet(charger_path)
        outliers = detect_outliers(df, ["StationId", "ChargerId"], CHARGER_METRICS, "Charger")
        if not outliers.is_empty():
            outliers.write_parquet(outliers_dir / "charger_outliers.parquet")
            print(f"Found {len(outliers)} charger anomalies.")
        else:
            print("No charger outliers found.")