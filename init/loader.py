"""
Provides utilities for reading raw simulation data and enriching it with 
human-readable time and date information.
"""

from pathlib import Path
import polars as pl

# Reference point: The simulation clock starts at midnight on a Sunday.
SIMULATION_START_DOW = 0

WEEKDAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday",
                 "Thursday", "Friday", "Saturday"]

# Milliseconds in a day
MS_PER_DAY = 86_400_000


def infer_run_id(parquet_path: Path) -> str:
    """
    Identifies the simulation run ID based on the folder structure.
    
    Args:
        parquet_path (Path): The path to a data file.
    Returns:
        str: The name of the parent directory (e.g., 'Run_001').
    """
    return parquet_path.parent.name


def add_day_columns_to_parquet(parquet_path: Path) -> pl.DataFrame:
    """
    Reads a raw parquet file and adds calendar metadata.

    Raw simulation data only tracks milliseconds. This function adds:
    - day: Which day of the simulation we are on (0, 1, 2...).
    - time_of_day: How many seconds have passed since midnight.
    - weekday_idx: The day of the week (0-6).
    - weekday_name: The actual name (e.g., 'Monday').
    - time_label: A formatted string for visualisation (e.g., '14:30').

    Args:
        parquet_path (Path): Path to the raw .parquet file.
    Returns:
        pl.DataFrame: A table enriched with temporal columns.
    """
    df = pl.read_parquet(parquet_path)

    df = df.with_columns([
        (pl.col("SimTime") // MS_PER_DAY).alias("day").cast(pl.Int32),
        (pl.col("SimTime") % MS_PER_DAY // 1000).alias("time_of_day").cast(pl.Int32),
    ])

    df = df.with_columns([
        ((pl.col("day") + SIMULATION_START_DOW) % 7)
        .alias("weekday_idx")
        .cast(pl.Int32),
    ])

    mapping = pl.DataFrame({
        "weekday_idx": list(range(7)),
        "weekday_name": WEEKDAY_NAMES,
    }).with_columns(pl.col("weekday_idx").cast(pl.Int32))

    df = df.join(mapping, on="weekday_idx", how="left")

    df = df.with_columns([
        (
            (pl.col("time_of_day") // 3600).cast(pl.Utf8).str.zfill(2)
            + pl.lit(":")
            + ((pl.col("time_of_day") % 3600) // 60).cast(pl.Utf8).str.zfill(2)
        ).alias("time_label")
    ])

    return df


def unique_days(df: pl.DataFrame) -> list[int]:
    """Returns a sorted list of every day index represented in the dataset."""
    return sorted(df["day"].unique().to_list())


def unique_stations(df: pl.DataFrame) -> list[int]:
    """Returns a sorted list of every StationId present in the data."""
    return sorted(df["StationId"].unique().to_list())


def unique_chargers(df: pl.DataFrame, station_id: int) -> list[int]:
    """Returns a sorted list of chargers belonging to a specific station."""
    return sorted(
        df.filter(pl.col("StationId") == station_id)["ChargerId"].unique().to_list()
    )


def filter_day(df: pl.DataFrame, day: int) -> pl.DataFrame:
    """Helper to slice the data for a specific 24-hour period."""
    return df.filter(pl.col("day") == day)


def filter_station(df: pl.DataFrame, station_id: int) -> pl.DataFrame:
    """Helper to slice the data for a specific station."""
    return df.filter(pl.col("StationId") == station_id)