"""
EVAnalysis – daily summary renderer.

Handles all data aggregation and figure composition for a single simulated day.
Called by generate_daily_summaries.py; not intended to be run directly.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np

from .daily_summary_charts import (
    BG, TEXT,
    UTIL_COLORS, QUEUE_COLORS, DEVIATION_COLORS, KPI_COLORS,
    draw_kpi_card, draw_layered_bar_chart,
)
from .intervals_utils import build_intervals, bucket_into_intervals


def filter_arrival_evs(arrival_df: pl.DataFrame) -> pl.DataFrame:
    """
    Returns only rows for EVs that did NOT drive directly to their destination,
    i.e. those that went through the charging network.
    Falls back to the full DataFrame if the column is absent (graceful degradation).
    """
    if "drive_directly" in arrival_df.columns:
        return arrival_df.filter(pl.col("drive_directly") == False)
    return arrival_df


def render_daily_summary(
    run_id: str,
    day: int,
    weekday: str,
    station_day_df: pl.DataFrame,
    arrival_day_df: pl.DataFrame | None,
    out_dir: Path,
) -> None:
    """
    Renders and saves the daily summary dashboard for a single simulation day.

    run_id          : simulation run identifier (used in the title)
    day             : integer day number within the simulation
    weekday         : human-readable weekday name (e.g. "Monday")
    station_day_df  : station_snapshots rows filtered to this day
    arrival_day_df  : arrival_snapshots rows filtered to this day (may be None/empty).
                      Expected to contain `drive_directly` and `path_deviation_minutes`.
    out_dir         : directory where the PNG will be saved
    """

    avg_utilization = station_day_df["utilization"].mean()
    avg_queue = station_day_df["total_queue_size"].mean()
    total_reservations = int(station_day_df["Reservations"].sum())
    total_cancellations = int(station_day_df["Cancellations"].sum())
    day_cancel_rate = (total_cancellations / total_reservations * 100 if total_reservations > 0 else None)

    missed_pct: float | None = None
    missed_subtitle: str | None = None
    arrival_df: pl.DataFrame | None = None

    if arrival_day_df is not None and not arrival_day_df.is_empty():
        arrival_df = filter_arrival_evs(arrival_day_df)

        if not arrival_df.is_empty() and "missed_deadline" in arrival_df.columns:
            n_total = len(arrival_df)
            n_missed = int(arrival_df["missed_deadline"].sum())
            missed_pct = n_missed / n_total * 100 if n_total > 0 else 0.0
            missed_subtitle = f"{n_missed:,} of {n_total:,} arrivals"

    intervals = build_intervals(station_day_df)
    labels = intervals["time_label"].to_list()

    station_aggregation = bucket_into_intervals(
        df=station_day_df,
        intervals=intervals,
        simtime_column="simtime_ms",
        aggregate_expressions=[
            pl.col("utilization").quantile(0.25).alias("util_p25"),
            pl.col("utilization").quantile(0.50).alias("util_p50"),
            pl.col("utilization").quantile(0.75).alias("util_p75"),
            pl.col("utilization").max().alias("util_max"),

            pl.col("total_queue_size").quantile(0.25).alias("queue_p25"),
            pl.col("total_queue_size").quantile(0.50).alias("queue_p50"),
            pl.col("total_queue_size").quantile(0.75).alias("queue_p75"),
            pl.col("total_queue_size").max().alias("queue_max"),
        ],
    )

    util_p25 = station_aggregation["util_p25"].to_numpy()
    util_p50 = station_aggregation["util_p50"].to_numpy()
    util_p75 = station_aggregation["util_p75"].to_numpy()
    util_max = station_aggregation["util_max"].to_numpy()

    queue_p25 = station_aggregation["queue_p25"].to_numpy()
    queue_p50 = station_aggregation["queue_p50"].to_numpy()
    queue_p75 = station_aggregation["queue_p75"].to_numpy()
    queue_max = station_aggregation["queue_max"].to_numpy()

    dev_p25 = dev_p50 = dev_p75 = dev_max = np.zeros(len(intervals))

    if arrival_df is not None and not arrival_df.is_empty() and "path_deviation_minutes" in arrival_df.columns:
        deviation_aggregation = bucket_into_intervals(
            df=arrival_df,
            intervals=intervals,
            simtime_column="simtime_ms",
            aggregate_expressions=[
                pl.col("path_deviation_minutes").quantile(0.25).alias("dev_p25"),
                pl.col("path_deviation_minutes").quantile(0.50).alias("dev_p50"),
                pl.col("path_deviation_minutes").quantile(0.75).alias("dev_p75"),
                pl.col("path_deviation_minutes").max().alias("dev_max"),
            ],
        )
        dev_p25 = deviation_aggregation["dev_p25"].to_numpy()
        dev_p50 = deviation_aggregation["dev_p50"].to_numpy()
        dev_p75 = deviation_aggregation["dev_p75"].to_numpy()
        dev_max = deviation_aggregation["dev_max"].to_numpy()

    figure = plt.figure(figsize=(22, 22), facecolor=BG)

    grid = gridspec.GridSpec(
        5, 1,
        figure=figure,
        height_ratios=[0.30, 1.40, 2.2, 2.2, 2.2],
        hspace=0.55,
        left=0.05, right=0.97, top=0.97, bottom=0.03,
    )

    ax_title = figure.add_subplot(grid[0])
    ax_title.axis("off")
    ax_title.set_facecolor(BG)
    ax_title.text(
        0.5, 0.5,
        f"SmartEV  |  Run: {run_id}  |  Daily Summary  |  Day {day} – {weekday}",
        horizontalalignment="center", verticalalignment="center",
        fontsize=26, fontweight="bold", color=TEXT,
        transform=ax_title.transAxes,
    )

    kpi_grid = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=grid[1], wspace=0.1, hspace=0.3)
    kpi_data = [
        ("Avg Utilization", f"{avg_utilization:.2%}" if avg_utilization is not None else "N/A", None),
        ("Avg Queue Size", f"{avg_queue:.2f}" if avg_queue is not None else "N/A", None),
        ("Missed Deadlines", f"{missed_pct:.1f}%" if missed_pct is not None else "N/A", missed_subtitle),
        ("Total Reservations", f"{total_reservations:,}", None),
        ("Total Cancellations", f"{total_cancellations:,}", None),
        ("Cancellation Rate", f"{day_cancel_rate:.1f}%" if day_cancel_rate is not None else "N/A", None),
    ]
    for i, (label, value, subtitle) in enumerate(kpi_data):
        row = i // 3
        col = i % 3
        draw_kpi_card(figure.add_subplot(kpi_grid[row, col]), label, value, KPI_COLORS[i], subtitle)

    percentiles_formatter = mticker.FuncFormatter(lambda v, _: f"{v:.0%}")

    # Utilization
    draw_layered_bar_chart(
        figure.add_subplot(grid[2]),
        interval_labels=labels,
        p25=util_p25, p50=util_p50, p75=util_p75, p_max=util_max,
        colors=UTIL_COLORS,
        title="Station Utilization per Snapshot Interval",
        ylabel="Utilization",
        y_formatter=percentiles_formatter,
        if_empty_message="No utilization data for this day",
    )

    # Queue size
    draw_layered_bar_chart(
        figure.add_subplot(grid[3]),
        interval_labels=labels,
        p25=queue_p25, p50=queue_p50, p75=queue_p75, p_max=queue_max,
        colors=QUEUE_COLORS,
        title="Station Queue Size per Snapshot Interval",
        ylabel="Queue size (EVs)",
        if_empty_message="No queue data for this day",
    )

    # Path deviation
    draw_layered_bar_chart(
        figure.add_subplot(grid[4]),
        interval_labels=labels,
        p25=dev_p25, p50=dev_p50, p75=dev_p75, p_max=dev_max,
        colors=DEVIATION_COLORS,
        title="Path Deviation per Arrival Interval",
        ylabel="Path deviation (minutes)",
        if_empty_message="No path deviation data for this day",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_dir / f"{weekday}_{day}.png", dpi=150, facecolor=BG)
    plt.close(figure)