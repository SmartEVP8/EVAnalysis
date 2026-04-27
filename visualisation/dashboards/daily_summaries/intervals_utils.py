"""
Interval bucketing utilities for the daily summary dashboard.

Derives intervals from station snapshot data and provides
a general-purpose function for aggregating any DataFrame into those buckets.
All math lives here; renderers stay free of bucketing logic.
"""
from __future__ import annotations

import polars as pl


def build_intervals(station_day_df: pl.DataFrame) -> pl.DataFrame:
    """
    Derives the set of time buckets from station snapshot data.

    The interval size is inferred from the two smallest consecutive snapshot
    timestamps, so it automatically adapts if the simulation config changes
    (e.g. 15-min vs 30-min snapshots).
    """
    sorted_times = (
        station_day_df
        .select("simtime_ms")
        .unique()
        .sort("simtime_ms")
        ["simtime_ms"]
        .to_list()
    )

    if len(sorted_times) < 2:
        times = sorted_times[0] if sorted_times else 0
        return pl.DataFrame({"interval_ms": [times], "time_label": [_ms_to_label(times)]})

    return pl.DataFrame({
        "interval_ms": sorted_times,
        "time_label": [_ms_to_label(t) for t in sorted_times],
    })


def bucket_into_intervals(
    df: pl.DataFrame,
    intervals: pl.DataFrame,
    simtime_column: str,
    aggregate_expressions: list[pl.Expr],
) -> pl.DataFrame:
    """
    Floors each row's simtime into the nearest bucket, then aggregates.

    Missing buckets are filled with zeros so the result has as many rows as the intervals.
    """
    interval_ms = int(intervals["interval_ms"][1] - intervals["interval_ms"][0]) if len(intervals) > 1 else 1

    bucketed = df.with_columns(
        (pl.col(simtime_column) // interval_ms * interval_ms).alias("interval_ms")
    )

    aggregated = (
        bucketed
        .group_by("interval_ms")
        .agg(aggregate_expressions)
    )

    result = (
        intervals
        .join(aggregated, on="interval_ms", how="left")
        .sort("interval_ms")
    )

    column_names = [expr.meta.output_name() for expr in aggregate_expressions]
    result = result.with_columns([
        pl.col(name).fill_null(0) for name in column_names
    ])

    return result


def _ms_to_label(simtime_ms: int) -> str:
    total_minutes = simtime_ms // 60_000
    hours = (total_minutes // 60) % 24
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"