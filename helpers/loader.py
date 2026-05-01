"""
Utilities for reading raw simulation Parquet files and enriching them with
human-readable temporal metadata.
"""

from pathlib import Path
import polars as pl

SIMULATION_START_DOW: int = 0

WEEKDAY_NAMES: list[str] = [
    "Sunday", "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday",
]

MS_PER_DAY: int = 86_400_000


def weekday_name_expr(day_col: str) -> pl.Expr:
    return (
        (pl.col(day_col) + SIMULATION_START_DOW) % 7
    ).map_elements(lambda x: WEEKDAY_NAMES[x], return_dtype=pl.Utf8)


def time_label_expr(simtime_ms_col: str) -> pl.Expr:
    ms = pl.col(simtime_ms_col)
    hours = (ms // 1_000 // 3_600).cast(pl.Utf8).str.zfill(2)
    minutes = ((ms // 1_000 % 3_600) // 60).cast(pl.Utf8).str.zfill(2)
    return hours + pl.lit(":") + minutes


def add_time_columns(
    df: pl.DataFrame,
    sim_time_col: str,
    *,
    day_alias: str = "day",
    simtime_alias: str = "simtime_ms",
    weekday_alias: str = "weekday_name",
    time_label_alias: str = "time_label",
) -> pl.DataFrame:
    """
    Enriches a DataFrame with human-readable temporal columns derived from a
    raw simulation-millisecond column.

    day          : zero-based simulation day (0 = first Sunday)
    simtime_ms   : milliseconds since midnight on that day
    weekday_name : e.g. 'Monday'
    time_label   : e.g. '14:30'
    """
    
    df = df.with_columns([
        (pl.col(sim_time_col) // MS_PER_DAY).cast(pl.Int32).alias(day_alias),
        (pl.col(sim_time_col) % MS_PER_DAY).cast(pl.Int64).alias(simtime_alias),
    ])

    df = df.with_columns([
        weekday_name_expr(day_alias).alias(weekday_alias),
    ])

    df = df.with_columns([
        time_label_expr(simtime_alias).alias(time_label_alias),
    ])

    return df


def add_day_columns_to_parquet(parquet_path: Path) -> pl.DataFrame:
    """
    Reads a raw snapshot Parquet file and adds temporal metadata derived from SimTime.
    """
    df = pl.read_parquet(parquet_path)
    return add_time_columns(df, "SimTime")


def add_arrival_day_columns_to_parquet(parquet_path: Path) -> pl.DataFrame:
    """
    Reads an arrival Parquet file and adds temporal metadata derived from ActualArrivalTime.

    Column names match the generic defaults so downstream analysers
    can use the same field names regardless of source.
    """
    df = pl.read_parquet(parquet_path)
    return add_time_columns(df, "ActualArrivalTime")


def infer_run_id(parquet_path: Path) -> str:
    """
    Infers the simulation run ID from the folder that contains a data file.
    """
    return parquet_path.parent.name


def unique_days(df: pl.DataFrame) -> list[int]:
    return sorted(df["day"].unique().to_list())


def unique_stations(df: pl.DataFrame) -> list[int]:
    return sorted(df["StationId"].unique().to_list())


def unique_chargers(df: pl.DataFrame, station_id: int) -> list[int]:
    return sorted(
        df.filter(pl.col("StationId") == station_id)["ChargerId"].unique().to_list()
    )


def filter_day(df: pl.DataFrame, day: int) -> pl.DataFrame:
    return df.filter(pl.col("day") == day)


def filter_station(df: pl.DataFrame, station_id: int) -> pl.DataFrame:
    return df.filter(pl.col("StationId") == station_id)