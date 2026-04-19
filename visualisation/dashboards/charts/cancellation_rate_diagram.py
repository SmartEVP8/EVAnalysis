"""
Cancellation rate distribution as a step-line across all stations at a single snapshot moment.
"""
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

PANEL_BG = "#1a1d27"
ACCENT  = "#81c784"
TEXT     = "#e8eaf6"
SUBTEXT  = "#9fa8da"
BORDER   = "#2a2d3e"


def render(axes: plt.Axes, station_snapshots: pl.DataFrame, simtime_ms: int) -> None:
    axes.set_facecolor(PANEL_BG)
    for border in axes.spines.values():
        border.set_edgecolor(BORDER)

    dataframe = (
        station_snapshots
        .filter(pl.col("simtime_ms") == simtime_ms)
        .filter(pl.col("Reservations") > 0)
    )

    if dataframe.is_empty() or "cancellation_rate" not in dataframe.columns:
        axes.text(0.5, 0.5, "No reservations at this snapshot", horizontalalignment="center", verticalalignment="center",
                transform=axes.transAxes, color=SUBTEXT, fontsize=11, style="italic")
        axes.set_xticks([])
        axes.set_yticks([])
        return
    
    total_reservations  = int(dataframe["Reservations"].sum())
    total_cancellations = int(dataframe["Cancellations"].sum())

    axes.text(0.98, 0.95,
              f"Reservations: {total_reservations}\nCancellations: {total_cancellations}",
              horizontalalignment="right", verticalalignment="top",
              transform=axes.transAxes,
              color=SUBTEXT, fontsize=8, fontfamily="monospace")

    cancellation_rate = dataframe["cancellation_rate"].to_numpy()
    bins  = np.linspace(0.0, 1.0, 11)

    counts, edges = np.histogram(cancellation_rate, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2

    axes.plot(centres, counts, color=ACCENT, linewidth=1.4, zorder=3)
    axes.fill_between(centres, counts, alpha=0.15, color=ACCENT, zorder=2)

    axes.set_title("Cancellation Rate Distribution", color=TEXT, fontsize=12, pad=6)
    axes.set_xlabel("Cancellation rate", color=SUBTEXT, fontsize=9)
    axes.set_ylabel("Number of Stations", color=SUBTEXT, fontsize=9)
    axes.set_xlim(0.0, 1.0)
    axes.set_xticks(np.linspace(0.0, 1.0, 6))
    axes.tick_params(colors=SUBTEXT, labelsize=8)
    axes.grid(axis="y", color=BORDER, linewidth=0.5, zorder=1)