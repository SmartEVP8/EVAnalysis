"""
Price distribution as a step-line across all stations at a single snapshot moment.
"""
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

PANEL_BG = "#1a1d27"
ACCENT  = "#e57373"
TEXT     = "#e8eaf6"
SUBTEXT  = "#9fa8da"
BORDER   = "#2a2d3e"


def render(
    axes: plt.Axes,
    station_snapshots: pl.DataFrame,
    simtime_ms: int,
) -> None:
    axes.set_facecolor(PANEL_BG)
    for border in axes.spines.values():
        border.set_edgecolor(BORDER)

    dataframe = station_snapshots.filter(pl.col("simtime_ms") == simtime_ms)

    if dataframe.is_empty() or "Price" not in dataframe.columns:
        axes.text(0.5, 0.5, "No price data", horizontalalignment="center", verticalalignment="center",
                transform=axes.transAxes, color=SUBTEXT, fontsize=11, style="italic")
        axes.set_xticks([])
        axes.set_yticks([])
        return

    prices = dataframe["Price"].to_numpy()
    
    if len(prices) == 0:
        axes.text(0.5, 0.5, "No price data", horizontalalignment="center", verticalalignment="center",
                transform=axes.transAxes, color=SUBTEXT, fontsize=11, style="italic")
        axes.set_xticks([])
        axes.set_yticks([])
        return

    n_bins = min(30, max(5, len(prices) // 3))
    counts, edges = np.histogram(prices, bins=n_bins)
    centres = (edges[:-1] + edges[1:]) / 2

    axes.plot(centres, counts, color=ACCENT, linewidth=1.4, zorder=3)
    axes.fill_between(centres, counts, alpha=0.15, color=ACCENT, zorder=2)

    axes.set_title("Price Distribution", color=TEXT, fontsize=12, pad=6)
    axes.set_xlabel("Price (DKK/kWh)", color=SUBTEXT, fontsize=9)
    axes.set_ylabel("Number of Stations", color=SUBTEXT, fontsize=9)
    axes.tick_params(colors=SUBTEXT, labelsize=8)
    axes.grid(axis="y", color=BORDER, linewidth=0.5, zorder=1)