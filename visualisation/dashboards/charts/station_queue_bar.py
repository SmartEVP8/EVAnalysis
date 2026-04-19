"""
Queue size distribution as a step-line across all stations at a single snapshot moment.
"""
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

PANEL_BG = "#1a1d27"
ACCENT = "#81c784"
TEXT     = "#e8eaf6"
SUBTEXT  = "#9fa8da"
BORDER   = "#2a2d3e"

QUEUE_COL = "total_queue_size"


def render(axes: plt.Axes, station_snapshots: pl.DataFrame, simtime_ms: int) -> None:
    axes.set_facecolor(PANEL_BG)
    for border in axes.spines.values():
        border.set_edgecolor(BORDER)

    dataframe = station_snapshots.filter(pl.col("simtime_ms") == simtime_ms)

    if dataframe.is_empty() or QUEUE_COL not in dataframe.columns:
        missing = f"Column '{QUEUE_COL}' not found" if not dataframe.is_empty() else "No data"
        axes.text(0.5, 0.5, missing, horizontalalignment="center", verticalalignment="center",
                transform=axes.transAxes, color=SUBTEXT, fontsize=11, style="italic")
        axes.set_xticks([])
        axes.set_yticks([])
        return

    queue_size_values = np.clip(dataframe[QUEUE_COL].drop_nulls().to_numpy(), 0, 20)
    counts, edges = np.histogram(queue_size_values, bins=20, range=(0, 20))
    centres = (edges[:-1] + edges[1:]) / 2

    axes.plot(centres, counts, color=ACCENT, linewidth=1.4, zorder=3)
    axes.fill_between(centres, counts, alpha=0.15, color=ACCENT, zorder=2)

    axes.set_title("Queue Size Distribution", color=TEXT, fontsize=12, pad=6)
    axes.set_xlabel("Queue Size",  color=SUBTEXT, fontsize=9)
    axes.set_ylabel("Number of Stations",  color=SUBTEXT, fontsize=9)
    axes.set_xlim(0, 20)
    axes.tick_params(colors=SUBTEXT, labelsize=8)
    axes.grid(axis="y", color=BORDER, linewidth=0.5, zorder=1)