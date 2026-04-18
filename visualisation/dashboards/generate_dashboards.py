"""
EVAnalysis – per-snapshot dashboard generator.
"""

import argparse
import io
from pathlib import Path

import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import numpy as np

import charts.arrival_delay_histogram as arrival_hist
import charts.cancellation_rate_bar as cancel_bar
import charts.outlier_diagram as outlier_plot
import charts.price_distribution as price_dist
import charts.station_utilization_bar as util_dist
import charts.station_queue_bar as queue_dist


# ── Colour palette ────────────────────────────────────────────────────────────
BG        = "#0f1117"
PANEL_BG  = "#1a1d27"
ACCENT    = "#4fc3f7"
ACCENT2   = "#81c784"
ACCENT3   = "#e57373"
TEXT      = "#e8eaf6"
SUBTEXT   = "#9fa8da"
BORDER    = "#2a2d3e"

STAT_COLORS = [ACCENT, ACCENT2, ACCENT3]


# ── Helpers ───────────────────────────────────────────────────────────────────

def seconds_to_hhmm(ms: int) -> str:
    seconds = ms // 1000
    h = (seconds // 3600) % 24
    m = (seconds % 3600) // 60
    return f"{h:02d}:{m:02d}"


def load_image_as_array(path: Path):
    if not path.exists():
        return None
    return mpimg.imread(str(path))


def draw_sub_figure(ax, fig_to_draw):
    """Renders a matplotlib figure onto a specific axis using the modern buffer API."""
    fig_to_draw.canvas.draw()
    
    rgba_buffer = fig_to_draw.canvas.buffer_rgba()
    
    img = np.asarray(rgba_buffer)
    
    ax.imshow(img)
    ax.set_axis_off()
    
    plt.close(fig_to_draw)


def draw_stat_box(ax, label: str, value: str, color: str):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(1.5)

    ax.text(0.5, 0.62, value,
            ha="center", va="center", transform=ax.transAxes,
            fontsize=26, fontweight="bold", color=color, fontfamily="monospace")
    ax.text(0.5, 0.28, label,
            ha="center", va="center", transform=ax.transAxes,
            fontsize=16, color=SUBTEXT, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_heatmap_panel(ax, img_array, title: str):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=SUBTEXT, fontsize=24, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

    if img_array is not None:
        ax.imshow(img_array, aspect="auto")
    else:
        ax.text(0.5, 0.5, "Heatmap not found",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=16, color=ACCENT3, style="italic")


def draw_placeholder(ax, label: str, idx: int):
    colors = [ACCENT, ACCENT2, ACCENT3, "#ce93d8", "#ffb74d", "#4db6ac"]
    color = colors[idx % len(colors)]

    ax.set_facecolor(PANEL_BG)

    rng = np.random.default_rng(idx * 7)
    x = np.arange(8)
    y = rng.uniform(0.2, 0.85, 8)
    ax.bar(x, y, color=color, alpha=0.25, width=0.7)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(0.5, 0.5, label,
            ha="center", va="center", transform=ax.transAxes,
            fontsize=16, color=color, style="italic", alpha=0.8)


# ── Main dashboard renderer ───────────────────────────────────────────────────

def render_dashboard(
    run_id: str,
    snapshot_df: pl.DataFrame,
    arrival_df: pl.DataFrame,
    outlier_df: pl.DataFrame,
    missed_pct: float | None,
    heatmap_dir: Path,
    out_dir: Path,
    simtime_ms: int,
    index: int,
):
    avg_queue = snapshot_df["total_queue_size"].mean()
    avg_util  = snapshot_df["utilization"].mean()

    day  = snapshot_df["day"][0]
    wday = snapshot_df["weekday_name"][0]
    time_str = seconds_to_hhmm(simtime_ms)

    title = f"Simulation Run: {run_id}  |  Day {day}, {wday}, {time_str}"

    fig = plt.figure(figsize=(22, 18), facecolor=BG)

    outer = gridspec.GridSpec(
        5, 1,
        figure=fig,
        height_ratios=[0.5, 1.5, 4, 3, 3],
        hspace=0.45,
        left=0.03, right=0.97, top=0.97, bottom=0.03,
    )

    # Title
    ax_title = fig.add_subplot(outer[0])
    ax_title.axis("off")
    ax_title.text(0.5, 0.5, title,
                  ha="center", va="center",
                  fontsize=32, fontweight="bold", color=TEXT)

    # KPIs
    stat_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.04)
    kpi_data = [
        ("Average Utilization", f"{avg_util:.2%}"),
        ("Average Queue Size", f"{avg_queue:.2f}"),
        ("Missed Deadlines", f"{missed_pct:.1f}%" if missed_pct else "N/A"),
    ]

    for i, (label, val) in enumerate(kpi_data):
        draw_stat_box(fig.add_subplot(stat_gs[i]), label, val, STAT_COLORS[i])

    # Heatmaps
    hm_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.04)
    names = ["utilization", "queue_size", "cancellation_rate"]
    titles = ["UTILIZATION", "QUEUE SIZE", "CANCELLATION RATE"]

    for i, name in enumerate(names):
        path = heatmap_dir / name / f"{name}_{index - 1}.png"
        img = load_image_as_array(path)

        draw_heatmap_panel(
            fig.add_subplot(hm_gs[i]),
            img,
            titles[i]
        )

    # ── Grid Mapping for the 6 charts ─────────────────────────────────────────
    # We will fill outer[3] and outer[4] (each has 3 columns)
    
    row3_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[3], wspace=0.1)
    row4_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[4], wspace=0.1)
    
    # Define which function goes to which slot
    # Logic: (Module, DataFrame, GridSpecSlot)
    chart_tasks = [
        (arrival_hist, arrival_df, row3_gs[0]),
        (cancel_bar,   snapshot_df, row3_gs[1]),
        (outlier_plot, outlier_df,  row3_gs[2]),
        (price_dist,   snapshot_df, row4_gs[0]),
        (queue_dist,    snapshot_df, row4_gs[1]),
        (util_dist,     snapshot_df, row4_gs[2]),
    ]

    for module, df_source, slot in chart_tasks:
        # Generate the figure from the sub-module
        sub_fig = module.render(df_source, simtime_ms)
        # Draw it into our dashboard
        draw_sub_figure(fig.add_subplot(slot), sub_fig)

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dashboard_{index}.png"
    fig.savefig(out_path, dpi=150, facecolor=BG)
    plt.close(fig)

    print(f"Saved -> {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_id = run_dir.name

    station_path = run_dir / "analysis" / "station_snapshots.parquet"
    arrival_path = run_dir / "analysis" / "arrival_snapshots.parquet"
    outlier_path = run_dir / "outliers" / "station_outliers.parquet"


    if not station_path.exists():
        raise FileNotFoundError(station_path)

    snap_df = pl.read_parquet(station_path)
    arr_df = pl.read_parquet(arrival_path) if arrival_path.exists() else pl.DataFrame()
    out_df = pl.read_parquet(outlier_path) if outlier_path.exists() else pl.DataFrame()

    # Missed deadlines
    missed_pct = None
    if not arr_df.is_empty() and "missed_deadline" in arr_df.columns:
        missed_pct = arr_df["missed_deadline"].mean() * 100

    heatmap_dir = run_dir / "heatmaps"
    
    out_dir = run_dir / "dashboards"

    times = snap_df["simtime_ms"].unique().sort()

    print(f"Generating {len(times)} dashboards...")

    for i, t in enumerate(times, start=1):
        # We filter the snapshot for the general stats, 
        # but the sub-charts usually filter themselves inside their own render logic
        current_snap = snap_df.filter(pl.col("simtime_ms") == t)
        
        render_dashboard(
            run_id=run_id,
            snapshot_df=current_snap,
            arrival_df=arr_df,
            outlier_df=out_df,
            missed_pct=missed_pct,
            heatmap_dir=heatmap_dir,
            out_dir=out_dir,
            simtime_ms=int(t),
            index=i
        )


if __name__ == "__main__":
    main()