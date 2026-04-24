"""
EVAnalysis – daily summary dashboard generator.

Produces one PNG per simulated day, saved to:
    runs/{run_id}/daily_summaries/{weekday_name}_{day_number}.png

Layout
------
  Row 0 – Title bar
  Row 1 – 6 KPI cards:
               Avg Utilization | Avg Queue Size | Missed Deadlines %
               Total Reservations | Total Cancellations | Cancellation Rate
  Row 2 – Layered bar chart: Utilization per snapshot interval  (p25/p50/p75/max)
  Row 3 – Layered bar chart: Queue Size per snapshot interval   (p25/p50/p75/max)
  Row 4 – Layered bar chart: Path Deviation (km) per arrival interval (p25/p50/p75/max)
           Only EVs with drive_directly=false are included.

Data sources (read from runs/{run_id}/analysis/)
-------------------------------------------------
  station_snapshots.parquet  – one row per (StationId, day, simtime_ms)
  arrival_snapshots.parquet  – one row per EV arrival event.
                               Must contain the `drive_directly` boolean column
                               produced by analyse_arrival.py.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
from tqdm import tqdm

# ── Colour palette (matches existing dashboards) ──────────────────────────────
BG       = "#0f1117"
PANEL_BG = "#1a1d27"
ACCENT   = "#4fc3f7"   # blue   – utilization / KPI 1
ACCENT2  = "#81c784"   # green  – queue / KPI 2
ACCENT3  = "#e57373"   # red    – deadlines / KPI 3
ACCENT4  = "#ffb74d"   # amber  – reservations KPI 4
ACCENT5  = "#ce93d8"   # purple – cancellations KPI 5
ACCENT6  = "#f06292"   # pink   – cancellation rate KPI 6
TEXT     = "#e8eaf6"
SUBTEXT  = "#9fa8da"
BORDER   = "#2a2d3e"

# Layered bar colour ramps  (p25 base → p50 → p75 → max top)
UTIL_COLORS      = ["#1b5e20", "#66bb6a", "#fdd835", "#e53935"]  # green  ramp
QUEUE_COLORS     = ["#0d47a1", "#42a5f5", "#fdd835", "#e53935"]  # blue   ramp
DEVIATION_COLORS = ["#4a148c", "#ab47bc", "#fdd835", "#e53935"]  # purple ramp

KPI_COLORS = [ACCENT, ACCENT2, ACCENT3, ACCENT4, ACCENT5, ACCENT6]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _filter_charging_evs(arrival_df: pl.DataFrame) -> pl.DataFrame:
    """
    Returns only rows for EVs that did NOT drive directly to their destination,
    i.e. those that went through the charging network.
    Falls back to the full DataFrame if the column is absent (graceful degradation).
    """
    if "drive_directly" in arrival_df.columns:
        return arrival_df.filter(pl.col("drive_directly") == False)  # noqa: E712
    return arrival_df


def draw_kpi_card(
    ax: plt.Axes,
    label: str,
    value: str,
    color: str,
    subtitle: str | None = None,
) -> None:
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(1.5)

    value_y = 0.38 if subtitle is None else 0.44
    label_y = 0.72 if subtitle is None else 0.78

    ax.text(0.5, value_y, value,
            ha="center", va="center", transform=ax.transAxes,
            fontsize=26, fontweight="bold", color=color, fontfamily="monospace")
    ax.text(0.5, label_y, label,
            ha="center", va="center", transform=ax.transAxes,
            fontsize=11, color=color, alpha=0.65)

    if subtitle is not None:
        ax.text(0.5, 0.18, subtitle,
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color=color, alpha=0.50, fontfamily="monospace")

    ax.set_xticks([])
    ax.set_yticks([])


def draw_layered_bar_chart(
    ax: plt.Axes,
    interval_labels: list[str],
    p25: np.ndarray,
    p50: np.ndarray,
    p75: np.ndarray,
    p_max: np.ndarray,
    colors: list[str],
    title: str,
    ylabel: str,
    y_formatter=None,
    empty_message: str = "No data available",
) -> None:
    """
    Draws a stacked (layered) bar chart where each segment is the *incremental*
    band between percentile thresholds, so the total bar height equals the max value.

    Bands (bottom → top):
        colors[0]  0      → p25
        colors[1]  p25    → p50
        colors[2]  p50    → p75
        colors[3]  p75    → max
    """
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)
    ax.set_title(title, color=SUBTEXT, fontsize=13, pad=6)

    if len(interval_labels) == 0 or p_max.sum() == 0:
        ax.text(0.5, 0.5, empty_message,
                ha="center", va="center", transform=ax.transAxes,
                fontsize=13, color=ACCENT3, style="italic")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    x     = np.arange(len(interval_labels))
    width = 0.65

    h0 = p25
    h1 = np.maximum(p50 - p25, 0)
    h2 = np.maximum(p75 - p50, 0)
    h3 = np.maximum(p_max - p75, 0)

    ax.bar(x, h0,                  color=colors[0], width=width)
    ax.bar(x, h1, bottom=h0,       color=colors[1], width=width)
    ax.bar(x, h2, bottom=h0+h1,    color=colors[2], width=width)
    ax.bar(x, h3, bottom=h0+h1+h2, color=colors[3], width=width)

    ax.set_xticks(x)
    ax.set_xticklabels(interval_labels, rotation=45, ha="right",
                       fontsize=8, color=SUBTEXT)
    ax.tick_params(axis="y", colors=SUBTEXT, labelsize=9)
    ax.set_ylabel(ylabel, color=SUBTEXT, fontsize=10)
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(bottom=0)

    if y_formatter is not None:
        ax.yaxis.set_major_formatter(y_formatter)

    legend_labels = ["≤ p25", "p25 → p50", "p50 → p75", "p75 → max"]
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, legend_labels)]
    ax.legend(handles=patches, loc="upper left", fontsize=8,
              facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT)

    ax.grid(axis="y", color=BORDER, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)


# ── Per-day rendering ─────────────────────────────────────────────────────────

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

    Parameters
    ----------
    run_id          : simulation run identifier (used in the title)
    day             : integer day number within the simulation
    weekday         : human-readable weekday name (e.g. "Monday")
    station_day_df  : station_snapshots rows filtered to this day
    arrival_day_df  : arrival_snapshots rows filtered to this day (may be None/empty).
                      Expected to contain `drive_directly` and `path_deviation_km`.
    out_dir         : directory where the PNG will be saved
    """

    # ── KPI computations ──────────────────────────────────────────────────────

    avg_utilization     = station_day_df["utilization"].mean()
    avg_queue           = station_day_df["total_queue_size"].mean()
    total_reservations  = int(station_day_df["Reservations"].sum())
    total_cancellations = int(station_day_df["Cancellations"].sum())
    day_cancel_rate     = (total_cancellations / total_reservations * 100
                           if total_reservations > 0 else None)

    # Missed deadlines – charging EVs only
    missed_pct: float | None     = None
    missed_subtitle: str | None  = None
    charging_day_df: pl.DataFrame | None = None

    if arrival_day_df is not None and not arrival_day_df.is_empty():
        charging_day_df = _filter_charging_evs(arrival_day_df)

        if not charging_day_df.is_empty() and "missed_deadline" in charging_day_df.columns:
            n_total         = len(charging_day_df)
            n_missed        = int(charging_day_df["missed_deadline"].sum())
            missed_pct      = n_missed / n_total * 100 if n_total > 0 else 0.0
            missed_subtitle = f"{n_missed:,} of {n_total:,} arrivals"

    # ── Per-interval station percentiles ─────────────────────────────────────

    interval_agg = (
        station_day_df
        .group_by(["simtime_ms", "time_label"])
        .agg([
            pl.col("utilization").quantile(0.25).alias("util_p25"),
            pl.col("utilization").quantile(0.50).alias("util_p50"),
            pl.col("utilization").quantile(0.75).alias("util_p75"),
            pl.col("utilization").max().alias("util_max"),

            pl.col("total_queue_size").quantile(0.25).alias("queue_p25"),
            pl.col("total_queue_size").quantile(0.50).alias("queue_p50"),
            pl.col("total_queue_size").quantile(0.75).alias("queue_p75"),
            pl.col("total_queue_size").max().alias("queue_max"),
        ])
        .sort("simtime_ms")
    )

    labels    = interval_agg["time_label"].to_list()
    util_p25  = interval_agg["util_p25"].fill_null(0).to_numpy()
    util_p50  = interval_agg["util_p50"].fill_null(0).to_numpy()
    util_p75  = interval_agg["util_p75"].fill_null(0).to_numpy()
    util_max  = interval_agg["util_max"].fill_null(0).to_numpy()
    queue_p25 = interval_agg["queue_p25"].fill_null(0).to_numpy()
    queue_p50 = interval_agg["queue_p50"].fill_null(0).to_numpy()
    queue_p75 = interval_agg["queue_p75"].fill_null(0).to_numpy()
    queue_max = interval_agg["queue_max"].fill_null(0).to_numpy()

    # ── Per-interval path deviation percentiles (charging EVs only) ───────────

    dev_labels: list[str] = []
    dev_p25 = dev_p50 = dev_p75 = dev_max = np.zeros(0)

    if charging_day_df is not None and not charging_day_df.is_empty() \
            and "path_deviation_km" in charging_day_df.columns:
        dev_agg = (
            charging_day_df
            .group_by(["simtime_ms", "time_label"])
            .agg([
                pl.col("path_deviation_km").quantile(0.25).alias("dev_p25"),
                pl.col("path_deviation_km").quantile(0.50).alias("dev_p50"),
                pl.col("path_deviation_km").quantile(0.75).alias("dev_p75"),
                pl.col("path_deviation_km").max().alias("dev_max"),
            ])
            .sort("simtime_ms")
        )
        dev_labels = dev_agg["time_label"].to_list()
        dev_p25    = dev_agg["dev_p25"].fill_null(0).to_numpy()
        dev_p50    = dev_agg["dev_p50"].fill_null(0).to_numpy()
        dev_p75    = dev_agg["dev_p75"].fill_null(0).to_numpy()
        dev_max    = dev_agg["dev_max"].fill_null(0).to_numpy()

    # ── Figure layout ─────────────────────────────────────────────────────────

    figure = plt.figure(figsize=(22, 22), facecolor=BG)

    grid = gridspec.GridSpec(
        5, 1,
        figure=figure,
        height_ratios=[0.30, 1.40, 2.2, 2.2, 2.2], 
        hspace=0.55,
        left=0.05, right=0.97, top=0.97, bottom=0.03,
    )

    # Title
    ax_title = figure.add_subplot(grid[0])
    ax_title.axis("off")
    ax_title.set_facecolor(BG)
    ax_title.text(
        0.5, 0.5,
        f"SmartEV  |  Run: {run_id}  |  Daily Summary  |  Day {day} – {weekday}",
        ha="center", va="center",
        fontsize=26, fontweight="bold", color=TEXT,
        transform=ax_title.transAxes,
    )

    # KPI row – 6 cards
    kpi_grid = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=grid[1], wspace=0.1, hspace=0.3)
    kpi_data = [
        ("Avg Utilization",     f"{avg_utilization:.2%}"   if avg_utilization is not None else "N/A", None),
        ("Avg Queue Size",      f"{avg_queue:.2f}"          if avg_queue       is not None else "N/A", None),
        ("Missed Deadlines",    f"{missed_pct:.1f}%"        if missed_pct      is not None else "N/A", missed_subtitle),
        ("Total Reservations",  f"{total_reservations:,}",  None),
        ("Total Cancellations", f"{total_cancellations:,}", None),
        ("Cancellation Rate",   f"{day_cancel_rate:.1f}%"   if day_cancel_rate is not None else "N/A", None),
    ]
    for i, (label, value, subtitle) in enumerate(kpi_data):
        ax_kpi = figure.add_subplot(kpi_grid[i // 3, i % 3])
        draw_kpi_card(ax_kpi, label, value, KPI_COLORS[i], subtitle)

    pct_fmt = mticker.FuncFormatter(lambda v, _: f"{v:.0%}")

    # Utilization bar chart
    draw_layered_bar_chart(
        figure.add_subplot(grid[2]),
        interval_labels=labels,
        p25=util_p25, p50=util_p50, p75=util_p75, p_max=util_max,
        colors=UTIL_COLORS,
        title="Station Utilization per Snapshot Interval",
        ylabel="Utilization",
        y_formatter=pct_fmt,
    )

    # Queue size bar chart
    draw_layered_bar_chart(
        figure.add_subplot(grid[3]),
        interval_labels=labels,
        p25=queue_p25, p50=queue_p50, p75=queue_p75, p_max=queue_max,
        colors=QUEUE_COLORS,
        title="Station Queue Size per Snapshot Interval",
        ylabel="Queue size (EVs)",
    )

    # Path deviation bar chart – charging EVs only
    draw_layered_bar_chart(
        figure.add_subplot(grid[4]),
        interval_labels=dev_labels,
        p25=dev_p25, p50=dev_p50, p75=dev_p75, p_max=dev_max,
        colors=DEVIATION_COLORS,
        title="Path Deviation per Arrival Interval  (charging EVs only)",
        ylabel="Path deviation (km)",
        empty_message="No path deviation data for this day",
    )

    # ── Save ──────────────────────────────────────────────────────────────────

    out_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_dir / f"{weekday}_{day}.png", dpi=150, facecolor=BG)
    plt.close(figure)


# ── Multiprocessing worker ────────────────────────────────────────────────────

_w_run_id     = None
_w_station_df = None
_w_arrival_df = None
_w_out_dir    = None


def _init_worker(run_id, station_df, arrival_df, out_dir):
    global _w_run_id, _w_station_df, _w_arrival_df, _w_out_dir
    _w_run_id     = run_id
    _w_station_df = station_df
    _w_arrival_df = arrival_df
    _w_out_dir    = out_dir


def _render_day_task(args: tuple[int, str]) -> None:
    day, weekday = args

    out_path = _w_out_dir / f"{weekday}_{day}.png"
    if out_path.exists():
        return

    station_day = _w_station_df.filter(pl.col("day") == day)

    arrival_day: pl.DataFrame | None = None
    if _w_arrival_df is not None and not _w_arrival_df.is_empty():
        arrival_day = _w_arrival_df.filter(pl.col("day") == day)

    render_daily_summary(
        run_id         = _w_run_id,
        day            = day,
        weekday        = weekday,
        station_day_df = station_day,
        arrival_day_df = arrival_day,
        out_dir        = _w_out_dir,
    )


# ── Public entry point ────────────────────────────────────────────────────────

def generate_daily_summaries(
    run_id: str,
    station_snapshot_df: pl.DataFrame,
    arrival_snapshot_df: pl.DataFrame | None,
    out_dir: Path,
) -> None:
    """
    Generates one daily summary PNG per simulated day.

    Parameters
    ----------
    run_id              : simulation run identifier
    station_snapshot_df : full station_snapshots.parquet DataFrame
    arrival_snapshot_df : full arrival_snapshots.parquet DataFrame (or None).
                          Must include `drive_directly` and `path_deviation_km`
                          columns written by analyse_arrival.py.
    out_dir             : destination directory (runs/{run_id}/daily_summaries)
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

    with mp.Pool(
        processes=None,
        initializer=_init_worker,
        initargs=(run_id, station_snapshot_df, arrival_snapshot_df, out_dir),
    ) as pool:
        for _ in tqdm(
            pool.imap_unordered(_render_day_task, day_list),
            total=len(day_list),
            desc="Daily summaries",
        ):
            pass


# ── CLI ───────────────────────────────────────────────────────────────────────

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
        run_id              = run_dir.name,
        station_snapshot_df = pl.read_parquet(station_path),
        arrival_snapshot_df = pl.read_parquet(arrival_path) if arrival_path.exists() else None,
        out_dir             = run_dir / "daily_summaries",
    )


if __name__ == "__main__":
    main()