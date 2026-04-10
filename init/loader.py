"""
loader.py
---------
Reads parquet files and enriches them with calendar columns derived from SimTime.
"""

from pathlib import Path

import polars as pl

# Day 0 of the simulation is a Sunday.
# Python's calendar uses Mon=0, so we use our own mapping: Sun=0 ... Sat=6.
SIMULATION_START_DOW = 0

WEEKDAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday",
                 "Thursday", "Friday", "Saturday"]

SECONDS_PER_DAY = 86_400


def infer_run_id(parquet_path: Path) -> str:
    """
    Derive the run ID from the parquet file's path.
    """
    return parquet_path.parent.name


def load_and_enrich(parquet_path: Path) -> pl.DataFrame:
    """
    Read a parquet file and attach derived columns.

    day          : simulation day index (0-based)
    time_of_day  : seconds since midnight on that day
    weekday_idx  : 0=Sun, 1=Mon, … 6=Sat
    weekday_name : e.g. "Sunday"
    time_label   : "HH:MM" for display on chart x-axis
    """
    df = pl.read_parquet(parquet_path)

    df = df.with_columns([
        (pl.col("SimTime") // SECONDS_PER_DAY).alias("day").cast(pl.Int32),
        (pl.col("SimTime") % SECONDS_PER_DAY).alias("time_of_day").cast(pl.Int32),
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
    """Return sorted list of all day indices present in the data."""
    return sorted(df["day"].unique().to_list())


def unique_stations(df: pl.DataFrame) -> list[int]:
    """Return sorted list of all station IDs present in the data."""
    return sorted(df["StationId"].unique().to_list())


def unique_chargers(df: pl.DataFrame, station_id: int) -> list[int]:
    """Return sorted list of charger IDs for a given station."""
    return sorted(
        df.filter(pl.col("StationId") == station_id)["ChargerId"].unique().to_list()
    )


def filter_day(df: pl.DataFrame, day: int) -> pl.DataFrame:
    return df.filter(pl.col("day") == day)


def filter_station(df: pl.DataFrame, station_id: int) -> pl.DataFrame:
    return df.filter(pl.col("StationId") == station_id)