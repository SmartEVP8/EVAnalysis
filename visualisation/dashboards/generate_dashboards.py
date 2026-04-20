"""
EVAnalysis – dashboard generator.
"""

import argparse
from pathlib import Path

import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

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
    This is used for the Missed Deadlines percentage card.
    """
    axes.set_facecolor(PANEL_BG)
    for border in axes.spines.values():
        border.set_edgecolor(color)
        border.set_linewidth(1.5)

    # Shift the main value up slightly when a subtitle is present so both lines
    # fit inside the card without overlapping the label at the top.
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
    current_station_df: pl.DataFrame, # Pre-sliced to the current simtime_ms from render_dashboards
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

    # Row 1 — KPI cards
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

    # Row 4 — heatmap panels
    heatmap_grid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=figure_grid[4], wspace=0.04)
    heatmap_names  = ["utilization", "queue_size", "cancellation_rate"]
    heatmap_titles = ["UTILIZATION", "QUEUE SIZE", "CANCELLATION"]
    for i, (name, hm_title) in enumerate(zip(heatmap_names, heatmap_titles)):
        path = heatmap_directory / name / f"{name}_{index - 1}.png"
        draw_heatmap_panel(figure.add_subplot(heatmap_grid[i]), load_image_as_array(path), hm_title)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dashboard_{index}.png"
    figure.savefig(out_path, dpi=150, facecolor=BG)
    plt.close(figure)
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_id  = run_dir.name

    station_path = run_dir / "analysis" / "station_snapshots.parquet"
    arrival_path = run_dir / "analysis" / "arrival_snapshots.parquet"
    outlier_path = run_dir / "outliers"  / "station_outliers.parquet"

    if not station_path.exists():
        raise FileNotFoundError(station_path)

    station_snapshot_df = pl.read_parquet(station_path)
    arrival_snapshot_df = pl.read_parquet(arrival_path) if arrival_path.exists() else pl.DataFrame()
    outlier_analysis_df = pl.read_parquet(outlier_path) if outlier_path.exists() else pl.DataFrame()

    missed_deadline_pct: float | None = None
    total_arrivals: int | None = None
    if not arrival_snapshot_df.is_empty() and "missed_deadline" in arrival_snapshot_df.columns:
        missed_deadline_pct = arrival_snapshot_df["missed_deadline"].mean() * 100
        total_arrivals = len(arrival_snapshot_df)

    station_by_time: dict[int, pl.DataFrame] = {
        int(simtime_ms): df
        for simtime_ms, df in station_snapshot_df.group_by("simtime_ms")
    }

    heatmap_dir = run_dir / "heatmaps"
    out_dir     = run_dir / "dashboards"

    times = station_snapshot_df["simtime_ms"].unique().sort()
    print(f"Generating {len(times)} dashboards...")

    for index, timestamp in enumerate(times, start=1):
        simtime_ms = int(timestamp)
        render_dashboard(
            run_id = run_id,
            current_station_df = station_by_time[simtime_ms],
            station_snapshot_df = station_snapshot_df,
            arrival_snapshot_df = arrival_snapshot_df,
            outlier_analysis_df = outlier_analysis_df,
            missed_deadlines_percent = missed_deadline_pct,
            total_arrivals = total_arrivals,
            heatmap_directory = heatmap_dir,
            out_dir = out_dir,
            simtime_ms = simtime_ms,
            index = index,
        )


if __name__ == "__main__":
    main()