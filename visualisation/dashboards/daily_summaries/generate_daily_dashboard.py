"""
EVAnalysis – daily summary dashboard generator.
Produces one PNG per simulated day, saved to:
    runs/{run_id}/daily_summaries/{weekday_name}_{day_number}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from tqdm import tqdm

from .daily_summary_renderer import render_daily_summary

def generate_daily_summaries(
    run_id: str,
    station_snapshot_df: pl.DataFrame,
    arrival_snapshot_df: pl.DataFrame | None,
    out_dir: Path,
) -> None:
    """
    Generates one daily summary PNG per simulated day.
    """
    days = (
        station_snapshot_df
        .select(["day", "weekday_name"])
        .unique()
        .sort("day")
        .iter_rows()
    )
    day_list = list(days)

    print(f"Generating {len(day_list)} daily summary dashboards...")

    out_dir.mkdir(parents=True, exist_ok=True)

    for day, weekday in tqdm(day_list, desc="Daily summaries"):

        station_day = station_snapshot_df.filter(pl.col("day") == day)

        arrival_day = arrival_snapshot_df.filter(pl.col("day") == day)

        render_daily_summary(
            run_id = run_id,
            day = day,
            weekday = weekday,
            station_day_df = station_day,
            arrival_day_df = arrival_day,
            out_dir = out_dir,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-day summary dashboards for an EVAnalysis run."
    )
    parser.add_argument("--run-dir", required=True,
                        help="Path to the run directory, e.g. runs/my_run_01")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    station_path = run_dir / "analysis" / "station_snapshots.parquet"
    arrival_path = run_dir / "analysis" / "arrival_snapshots.parquet"

    if not station_path.exists():
        raise FileNotFoundError(f"station_snapshots.parquet not found at {station_path}")

    generate_daily_summaries(
        run_id = run_dir.name,
        station_snapshot_df = pl.read_parquet(station_path),
        arrival_snapshot_df = pl.read_parquet(arrival_path) if arrival_path.exists() else None,
        out_dir = run_dir / "daily_summaries",
    )

if __name__ == "__main__":
    main()