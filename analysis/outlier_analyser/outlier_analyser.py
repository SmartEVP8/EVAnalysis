"""
Identifies statistical outliers by comparing each entity's percentile values
against the mean and standard deviation of all entities for the same weekday.

An outlier is defined as a value that falls outside the fences computed using the
IQR method:
  upper = P75 + IQR_MULTIPLIER * (P75 - P25)
  lower = P25 - IQR_MULTIPLIER * (P75 - P25)

Reads from:
    runs/{run_id}/percentiles/station/station_percentiles_{weekday}_{day}.parquet
    runs/{run_id}/percentiles/charger/charger_percentiles_{weekday}_{day}.parquet

Writes to:
    runs/{run_id}/outliers/station_outliers.parquet
    runs/{run_id}/outliers/charger_outliers.parquet
"""

from pathlib import Path

import polars as pl

OUTPUT_ROOT = Path("runs")

IQR_MULTIPLIER = 1.5

STATION_METRICS = ["utilization_p50", "utilization_p90", "utilization_p95",
                   "queue_size_p50", "queue_size_p90", "queue_size_p95"]

CHARGER_METRICS = ["utilization_p50", "utilization_p90", "utilization_p95",
                   "queue_size_p50", "queue_size_p90", "queue_size_p95"]


def _detect_outliers(
    df: pl.DataFrame,
    id_cols: list[str],
    metric_cols: list[str],
    day: int,
    dayOfWeek: str,
) -> pl.DataFrame:

    flagged_rows = []
 
    for metric in metric_cols:
        if metric not in df.columns:
            continue
 
        series = df[metric].cast(pl.Float64)
        p25 = series.quantile(0.25)
        p75 = series.quantile(0.75)
        iqr = p75 - p25
 
        # If IQR is zero, all stations have the same value — no outliers.
        if not iqr or iqr == 0:
            continue
 
        upper = p75 + IQR_MULTIPLIER * iqr
        lower = p25 - IQR_MULTIPLIER * iqr
 
        base = df.select(id_cols).with_columns(series.alias("value"))
 
        high = (
            base.filter(pl.col("value") > upper)
            .with_columns([
                pl.lit(day).alias("day"),
                pl.lit(dayOfWeek).alias("weekday"),
                pl.lit(metric).alias("metric"),
                pl.lit("HIGH").alias("flag"),
                pl.lit(p25).alias("p25"),
                pl.lit(p75).alias("p75"),
                pl.lit(iqr).alias("IQR"),
                pl.lit(upper).alias("upper_fence"),
                pl.lit(lower).alias("lower_fence"),
            ])
        )
 
        low = (
            base.filter(pl.col("value") < lower)
            .with_columns([
                pl.lit(day).alias("day"),
                pl.lit(dayOfWeek).alias("weekday"),
                pl.lit(metric).alias("metric"),
                pl.lit("LOW").alias("flag"),
                pl.lit(p25).alias("p25"),
                pl.lit(p75).alias("p75"),
                pl.lit(iqr).alias("IQR"),
                pl.lit(upper).alias("upper_fence"),
                pl.lit(lower).alias("lower_fence"),
            ])
        )
 
        flagged_rows.extend([high, low])
 
    if not flagged_rows:
        return pl.DataFrame()
 
    return pl.concat(flagged_rows)


def _load_percentile_files(directory: Path) -> list[tuple[int, str, pl.DataFrame]]:
    """
    Load all parquet files in a directory.
    Filename convention: {prefix}_{weekday}_{day}.parquet
    Returns a list of (day, weekday, DataFrame) tuples.
    """
    result = []
    for path in sorted(directory.glob("*.parquet")):
        parts = path.stem.split("_")
        day = int(parts[-1])
        dayOfWeek = parts[-2]
        result.append((day, dayOfWeek, pl.read_parquet(path)))
    return result


def detect_station_outliers(run_id: str) -> None:
    source_dir = OUTPUT_ROOT / run_id / "percentiles" / "station"

    if not source_dir.exists():
        raise Exception("Station percentile folder is missing, cannot detect station outliers.")

    files = _load_percentile_files(source_dir)
    if not files:
        raise Exception("No station percentile files found, cannot detect station outliers.")

    all_outliers = []
    for day, dayOfWeek, df in files:
        outliers = _detect_outliers(df, ["StationId"], STATION_METRICS, day, dayOfWeek)
        if not outliers.is_empty():
            all_outliers.append(outliers)

    if not all_outliers:
        print("No station outliers detected.")
        return

    result = pl.concat(all_outliers).sort(["day", "metric", "flag", "StationId"])

    out_dir = OUTPUT_ROOT / run_id / "outliers"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "station_outliers.parquet"
    result.write_parquet(out_path)
    print(f"Found {len(result)} station anomalies -> {out_path}")


def detect_charger_outliers(run_id: str) -> None:
    source_dir = OUTPUT_ROOT / run_id / "percentiles" / "charger"

    if not source_dir.exists():
        raise Exception("Charger percentile folder is missing, cannot detect charger outliers.")

    files = _load_percentile_files(source_dir)
    if not files:
        raise Exception("No charger percentile files found, cannot detect charger outliers.")

    all_outliers = []
    for day, dayOfWeek, df in files:
        outliers = _detect_outliers(df, ["StationId", "ChargerId"], CHARGER_METRICS, day, dayOfWeek)
        if not outliers.is_empty():
            all_outliers.append(outliers)

    if not all_outliers:
        print("No charger outliers detected.")
        return

    result = pl.concat(all_outliers).sort(["day", "metric", "flag", "StationId", "ChargerId"])

    out_dir = OUTPUT_ROOT / run_id / "outliers"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "charger_outliers.parquet"
    result.write_parquet(out_path)
    print(f"Found {len(result)} charger anomalies -> {out_path}")