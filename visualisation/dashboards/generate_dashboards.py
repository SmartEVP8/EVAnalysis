"""
EVAnalysis – dashboard generator.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from tqdm import tqdm

import visualisation.dashboards.charts.arrival_delay_diagram as arrivals_distribution
import visualisation.dashboards.charts.cancellation_rate_diagram as cancellation_distribution
import visualisation.dashboards.charts.outlier_diagram as outliers_distribution
import visualisation.dashboards.charts.price_diagram as price_diagram
import visualisation.dashboards.charts.station_utilization_diagram as utilization_distribution
import visualisation.dashboards.charts.station_queue_diagram as queue_size_distribution


BG = "#0f1117"
PANEL_BG = "#1a1d27"
ACCENT = "#4fc3f7"
ACCENT2 = "#81c784"
ACCENT3 = "#e57373"
TEXT = "#e8eaf6"
SUBTEXT = "#9fa8da"
BORDER = "#2a2d3e"

KPI_COLORS = [ACCENT, ACCENT2, ACCENT3]


# Helpers
def seconds_to_hhmm(ms: int) -> str:
    seconds = ms // 1000
    hours = (seconds // 3600) % 24
    minutes = (seconds % 3600) // 60
    return f"{hours:02d}:{minutes:02d}"


def load_image_as_array(path: Path):
    if not path.exists():
        return None
    return mpimg.imread(str(path))


# KPI stands for Key Performance Indicators.
def draw_kpi_card(axes: plt.Axes, label: str, value: str, color: str, subtitle: str | None = None) -> None:
    """
    Draws a KPI card.

    If `subtitle` is provided it is rendered as a small secondary line below the main value.
    This is (only) used for the Missed Deadlines percentage card.
    """
    axes.set_facecolor(PANEL_BG)
    for border in axes.spines.values():
        border.set_edgecolor(color)
        border.set_linewidth(1.5)

    value_y = 0.38 if subtitle is None else 0.44
    label_y = 0.72 if subtitle is None else 0.78

    axes.text(0.5, value_y,
              value,
              horizontalalignment="center", verticalalignment="center",
              transform=axes.transAxes,
              fontsize=30, fontweight="bold", color=color, fontfamily="monospace")

    axes.text(0.5, label_y,
              label,
              horizontalalignment="center", verticalalignment="center",
              transform=axes.transAxes,
              fontsize=13, color=color, alpha=0.65)

    if subtitle is not None:
        axes.text(0.5, 0.20,
                  subtitle,
                  horizontalalignment="center", verticalalignment="center",
                  transform=axes.transAxes,
                  fontsize=11, color=color, alpha=0.50, fontfamily="monospace")

    axes.set_xticks([])
    axes.set_yticks([])


def draw_heatmap_panel(axes: plt.Axes, image_array, title: str) -> None:
    axes.set_facecolor(PANEL_BG)
    axes.set_title(title, color=SUBTEXT, fontsize=14, pad=6)
    axes.set_xticks([])
    axes.set_yticks([])
    for border in axes.spines.values():
        border.set_edgecolor(BORDER)

    if image_array is not None:
        axes.imshow(image_array, aspect="auto")
    else:
        axes.text(0.5, 0.5, "Heatmap not found",
                horizontalalignment="center", verticalalignment="center", transform=axes.transAxes,
                fontsize=13, color=ACCENT3, style="italic")


def render_dashboard(
    run_id: str,
    current_station_df: pl.DataFrame,
    station_snapshot_df: pl.DataFrame,
    arrival_snapshot_df: pl.DataFrame,
    outlier_analysis_df: pl.DataFrame,
    missed_deadlines_percent: float | None,
    total_arrivals: int | None,
    heatmap_directory: Path,
    out_dir: Path,
    simtime_ms: int,
    index: int,
) -> None:

    average_utilization = current_station_df["utilization"].mean()
    average_queue = current_station_df["total_queue_size"].mean()

    day_of_sim = current_station_df["day"][0]
    weekday = current_station_df["weekday_name"][0]
    time_str = seconds_to_hhmm(simtime_ms)
    title = f"SmartEV  |  Run: {run_id}  |  Day {day_of_sim}, {weekday}, {time_str}"

    figure = plt.figure(figsize=(22, 20), facecolor=BG)

    figure_grid = gridspec.GridSpec(
        5, 1,
        figure=figure,
        height_ratios=[0.35, 0.85, 2.5, 2.5, 3.0],
        hspace=0.55,
        left=0.03, right=0.97, top=0.97, bottom=0.03,
    )

    axes_title = figure.add_subplot(figure_grid[0])
    axes_title.axis("off")
    axes_title.text(0.5, 0.5, title,
                  horizontalalignment="center", verticalalignment="center",
                  fontsize=28, fontweight="bold", color=TEXT)

    # Row 1
    kpi_grid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=figure_grid[1], wspace=0.04)

    if missed_deadlines_percent is not None:
        missed_value = f"{missed_deadlines_percent:.1f}%"
        missed_subtitle = f"of {total_arrivals:,} arrivals" if total_arrivals is not None else None
    else:
        missed_value = "N/A"
        missed_subtitle = None

    kpi_data = [
        ("Avg Utilization",  f"{average_utilization:.2%}", None),
        ("Avg Queue Size",   f"{average_queue:.2f}", None),
        ("Missed Deadlines", missed_value, missed_subtitle),
    ]
    for i, (label, value, subtitle) in enumerate(kpi_data):
        draw_kpi_card(figure.add_subplot(kpi_grid[i]), label, value, KPI_COLORS[i], subtitle)

    # Row 2
    distribution_grid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=figure_grid[2], wspace=0.08)
    utilization_distribution.render(figure.add_subplot(distribution_grid[0]), station_snapshot_df, simtime_ms)
    queue_size_distribution.render(figure.add_subplot(distribution_grid[1]), station_snapshot_df, simtime_ms)
    price_diagram.render(figure.add_subplot(distribution_grid[2]), station_snapshot_df, simtime_ms)

    # Row 3
    distribution_grid_2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=figure_grid[3], wspace=0.08)
    arrivals_distribution.render(figure.add_subplot(distribution_grid_2[0]), arrival_snapshot_df, simtime_ms)
    cancellation_distribution.render(  figure.add_subplot(distribution_grid_2[1]), station_snapshot_df, simtime_ms)
    outliers_distribution.render(figure.add_subplot(distribution_grid_2[2]), outlier_analysis_df, simtime_ms)

    # Row 4
    heatmap_grid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=figure_grid[4], wspace=0.04)
    heatmap_names  = ["utilization", "queue_size", "cancellation_rate"]
    heatmap_titles = ["UTILIZATION", "QUEUE SIZE", "CANCELLATION"]
    for i, (heatmap_name, heatmap_title) in enumerate(zip(heatmap_names, heatmap_titles)):
        path = heatmap_directory / heatmap_name / f"{heatmap_name}_{index - 1}.png"
        draw_heatmap_panel(figure.add_subplot(heatmap_grid[i]), load_image_as_array(path), heatmap_title)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dashboard_{index}.png"
    figure.savefig(out_path, dpi=150, facecolor=BG)
    plt.close(figure)

"""
Each worker loads the shared DataFrames once, and each task only receives the per-frame
arguments (simtime_ms, index, a few scalars and paths).
"""
_worker_run_id = None
_worker_station_snapshot_df = None
_worker_arrival_snapshot_df = None
_worker_outlier_analysis_df = None
_worker_missed_deadlines_pct = None
_worker_total_arrivals = None
_worker_heatmap_dir = None
_worker_out_dir = None


def init_dashboard_worker(
    run_id: str,
    station_snapshot_df: pl.DataFrame,
    arrival_snapshot_df: pl.DataFrame,
    outlier_analysis_df: pl.DataFrame,
    missed_deadlines_percent: float | None,
    total_arrivals: int | None,
    heatmap_dir: Path,
    out_dir: Path,
) -> None:
    """Runs once per worker process"""
    global _worker_run_id, _worker_station_snapshot_df, _worker_arrival_snapshot_df
    global _worker_outlier_analysis_df, _worker_missed_deadlines_pct
    global _worker_total_arrivals, _worker_heatmap_dir, _worker_out_dir

    _worker_run_id = run_id
    _worker_station_snapshot_df = station_snapshot_df
    _worker_arrival_snapshot_df = arrival_snapshot_df
    _worker_outlier_analysis_df = outlier_analysis_df
    _worker_missed_deadlines_pct = missed_deadlines_percent
    _worker_total_arrivals = total_arrivals
    _worker_heatmap_dir = heatmap_dir
    _worker_out_dir = out_dir


@dataclass
class DashboardTask:
    simtime_ms: int
    index: int
    current_station_df: pl.DataFrame


def render_dashboard_task(task: DashboardTask) -> None:
    """Renders one dashboard frame using worker-local shared state."""
    out_path = _worker_out_dir / f"dashboard_{task.index}.png"
    if out_path.exists():
        return

    render_dashboard(
        run_id = _worker_run_id,
        current_station_df = task.current_station_df,
        station_snapshot_df = _worker_station_snapshot_df,
        arrival_snapshot_df = _worker_arrival_snapshot_df,
        outlier_analysis_df = _worker_outlier_analysis_df,
        missed_deadlines_percent = _worker_missed_deadlines_pct,
        total_arrivals = _worker_total_arrivals,
        heatmap_directory = _worker_heatmap_dir,
        out_dir = _worker_out_dir,
        simtime_ms = task.simtime_ms,
        index = task.index,
    )


def generate_dashboards(
    run_id: str,
    station_snapshot_df: pl.DataFrame,
    arrival_snapshot_df: pl.DataFrame,
    outlier_analysis_df: pl.DataFrame,
    heatmap_dir: Path,
    out_dir: Path,
) -> None:
    """
    Orchestrates parallel dashboard generation for a full simulation run.

    Computes KPIs and pre-groups snapshots by timestamp in the main
    process, then distributes per-frame rendering tasks across a worker pool.
    The shared DataFrames are sent to each worker once via the pool
    initialiser rather than being serialized with every task.

    Pool size defaults to None, which lets Python use all available CPU cores.
    Frames that already exist on disk are skipped.
    """
    missed_deadline_pct: float | None = None
    total_arrivals: int | None = None
    if not arrival_snapshot_df.is_empty() and "missed_deadline" in arrival_snapshot_df.columns:
        missed_deadline_pct = arrival_snapshot_df["missed_deadline"].mean() * 100
        total_arrivals = len(arrival_snapshot_df)

    station_by_time: dict[int, pl.DataFrame] = {
        int(key[0]): dataframe
        for key, dataframe in station_snapshot_df.group_by("simtime_ms")
    }

    times = station_snapshot_df["simtime_ms"].unique().sort()
    tasks = [
        DashboardTask(
            simtime_ms = int(timestamp),
            index = index,
            current_station_df = station_by_time[int(timestamp)],
        )
        for index, timestamp in enumerate(times, start=1)
    ]

    total = len(tasks)
    print(f"Generating {total} dashboards...")

    with mp.Pool(
        processes=None,
        initializer=init_dashboard_worker,
        initargs=(
            run_id,
            station_snapshot_df,
            arrival_snapshot_df,
            outlier_analysis_df,
            missed_deadline_pct,
            total_arrivals,
            heatmap_dir,
            out_dir,
        ),
    ) as pool:
        for _ in tqdm(
            pool.imap_unordered(render_dashboard_task, tasks),
            total=total,
            desc="Dashboards",
        ):
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    station_path = run_dir / "analysis" / "station_snapshots.parquet"
    arrival_path = run_dir / "analysis" / "arrival_snapshots.parquet"
    outlier_path = run_dir / "outliers"  / "station_outliers.parquet"

    if not station_path.exists():
        raise FileNotFoundError(station_path)

    generate_dashboards(
        run_id              = run_dir.name,
        station_snapshot_df = pl.read_parquet(station_path),
        arrival_snapshot_df = pl.read_parquet(arrival_path) if arrival_path.exists() else pl.DataFrame(),
        outlier_analysis_df = pl.read_parquet(outlier_path) if outlier_path.exists() else pl.DataFrame(),
        heatmap_dir         = run_dir / "heatmaps",
        out_dir             = run_dir / "dashboards",
    )


if __name__ == "__main__":
    main()