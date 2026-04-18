"""
Histogram of electricity prices across stations at a single snapshot moment.
"""
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

BG       = "#0f1117"
PANEL_BG = "#1a1d27"
ACCENT   = "#4fc3f7"
ACCENT2  = "#81c784"
TEXT     = "#e8eaf6"
SUBTEXT  = "#9fa8da"
BORDER   = "#2a2d3e"


def render(
    station_snapshots: pl.DataFrame,
    simtime_ms: int,
    figsize: tuple[float, float] = (7, 3.5),
) -> plt.Figure:
    """
    Render a histogram of Price across all stations at the given simtime_ms,
    with vertical lines for mean and median.

    Parameters
    ----------
    station_snapshots:
        Full station_snapshots.parquet loaded as a Polars DataFrame.
    simtime_ms:
        The simulation timestamp to filter to.
    figsize:
        Figure dimensions in inches.
    """
    df = station_snapshots.filter(pl.col("simtime_ms") == simtime_ms)

    if df.is_empty():
        fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
        ax.set_facecolor(PANEL_BG)
        ax.text(0.5, 0.5, "No data for this snapshot",
                ha="center", va="center", transform=ax.transAxes,
                color=SUBTEXT, fontsize=12, style="italic")
        ax.set_axis_off()
        return fig

    prices = df["Price"].drop_nulls().to_numpy()

    if len(prices) == 0:
        fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
        ax.set_facecolor(PANEL_BG)
        ax.text(0.5, 0.5, "No price data available",
                ha="center", va="center", transform=ax.transAxes,
                color=SUBTEXT, fontsize=12, style="italic")
        ax.set_axis_off()
        return fig

    mean_price   = float(np.mean(prices))
    median_price = float(np.median(prices))

    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(PANEL_BG)

    n_bins = min(20, max(5, len(prices) // 3))
    ax.hist(prices, bins=n_bins, color=ACCENT, alpha=0.75, edgecolor=BORDER, zorder=2)

    ax.axvline(mean_price,   color=ACCENT2, linewidth=1.2, linestyle="--",
               label=f"Mean: {mean_price:.2f}")
    ax.axvline(median_price, color="#ce93d8", linewidth=1.2, linestyle=":",
               label=f"Median: {median_price:.2f}")

    ax.set_xlabel("Price (DKK/kWh)", color=SUBTEXT, fontsize=9)
    ax.set_ylabel("Station count", color=SUBTEXT, fontsize=9)
    ax.set_title("Price Distribution Across Stations", color=TEXT, fontsize=11, pad=6)
    ax.tick_params(colors=SUBTEXT, labelsize=8)

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(axis="y", color=BORDER, linewidth=0.5, zorder=1)

    ax.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=SUBTEXT, edgecolor=BORDER)

    fig.tight_layout()
    return fig