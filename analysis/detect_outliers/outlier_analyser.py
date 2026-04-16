"""
Identifies statistical anomalies in station and charger performance using 
the Interquartile Range (IQR) method.
"""

from pathlib import Path
import polars as pl

OUTPUT_ROOT = Path("runs")

INTERQUARTILE_RANGE_MULTIPLIER = 1.5

STATION_METRICS = ["utilization_p50", "utilization_p90", "queue_size_p50", "queue_size_p90"]
CHARGER_METRICS = ["utilization_p50", "utilization_p90", "queue_size_p50", "queue_size_p90"]

def detect_outliers_global(
    df: pl.DataFrame,
    id_cols: list[str],
    metric_cols: list[str],
    label: str
) -> pl.DataFrame:
    """
    Analyzes and flags rows that deviate significantly from their peers at the same timestamp.
    """
    all_flags = []

    for metric in metric_cols:
        if metric not in df.columns:
            continue

        ranges_df = df.group_by("day").agg([
            pl.col(metric).quantile(0.25).alias("p25"),
            pl.col(metric).quantile(0.75).alias("p75"),
        ]).with_columns(
            (pl.col("p75") - pl.col("p25")).alias("interquartile_range")
        ).with_columns([
            (pl.col("p75") + INTERQUARTILE_RANGE_MULTIPLIER * pl.col("interquartile_range")).alias("upper"),
            (pl.col("p25") - INTERQUARTILE_RANGE_MULTIPLIER * pl.col("interquartile_range")).alias("lower"),
        ])

        outliers = df.join(ranges_df, on="day").filter(
            (pl.col(metric) > pl.col("upper")) | (pl.col(metric) < pl.col("lower"))
        ).with_columns([
            pl.col(metric).alias("value"),
            pl.lit(metric).alias("metric"),
            pl.when(pl.col(metric) > pl.col("upper")).then(pl.lit("HIGH")).otherwise(pl.lit("LOW")).alias("flag")
        ]).select(id_cols + ["day", "metric", "flag", "value", "p25", "p75", "upper", "lower"])

        if not outliers.is_empty():
            all_flags.append(outliers)

    if not all_flags:
        return pl.DataFrame()

    return pl.concat(all_flags)

def process_outliers(run_dir: Path, run_id: str):
    """
    Loads global percentile files and saves detected outliers.
    """
    analysis_dir = run_dir / "analysis"
    outliers_dir = OUTPUT_ROOT / run_id / "outliers"
    outliers_dir.mkdir(exist_ok=True)

    station_p_path = analysis_dir / "station_percentiles_global.parquet"
    if station_p_path.exists():
        df = pl.read_parquet(station_p_path)
        outliers = detect_outliers_global(df, ["StationId"], STATION_METRICS, "Station")
        if not outliers.is_empty():
            outliers.write_parquet(outliers_dir / "station_outliers.parquet")
            print(f"  Found {len(outliers)} station anomalies.")
        else:
            print("No station outliers were found")

    charger_p_path = analysis_dir / "charger_percentiles_global.parquet"
    if charger_p_path.exists():
        df = pl.read_parquet(charger_p_path)
        outliers = detect_outliers_global(df, ["StationId", "ChargerId"], CHARGER_METRICS, "Charger")
        if not outliers.is_empty():
            outliers.write_parquet(outliers_dir / "charger_outliers.parquet")
            print(f"  Found {len(outliers)} charger anomalies.")
        else:
            print("No charger outliers were found")