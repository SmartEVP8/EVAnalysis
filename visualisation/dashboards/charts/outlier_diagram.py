"""
Outlier deviation distribution as step-lines for HIGH and LOW outliers.
"""
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

PANEL_BG = "#1a1d27"
ACCENT  = "#b54707"
ACCENT2   = "#e57373"
TEXT     = "#e8eaf6"
SUBTEXT  = "#9fa8da"
BORDER   = "#2a2d3e"


def render(axes: plt.Axes, station_outliers: pl.DataFrame, simtime_ms: int) -> None:
    axes.set_facecolor(PANEL_BG)
    for border in axes.spines.values():
        border.set_edgecolor(BORDER)

    dataframe = (
        station_outliers
        .filter(pl.col("simtime_ms") == simtime_ms)
        .with_columns(
            pl.when(pl.col("flag") == "HIGH")
              .then(pl.col("value") - pl.col("upper"))
              .otherwise(pl.col("lower") - pl.col("value"))
              .alias("deviation")
        )
    )

    if dataframe.is_empty():
        axes.text(0.5, 0.5, "No outliers at this snapshot", horizontalalignment="center", verticalalignment="center",
                transform=axes.transAxes, color=ACCENT2, fontsize=11, style="italic")
        axes.set_xticks([])
        axes.set_yticks([])
        return

    deviations_high = dataframe.filter(pl.col("flag") == "HIGH")["deviation"].to_numpy()
    deviations_low  = dataframe.filter(pl.col("flag") == "LOW")["deviation"].to_numpy()

    deviations_all = np.concatenate([deviations_high, deviations_low])
    deviation_largest  = float(deviations_all.max()) if len(deviations_all) > 0 else 1.0
    n_bins   = min(20, max(5, len(deviations_all) // 2))

    def step_line(data, color, label):
        if len(data) == 0:
            return
        counts, edges = np.histogram(data, bins=n_bins, range=(0, deviation_largest))
        centres = (edges[:-1] + edges[1:]) / 2
        axes.plot(centres, counts, color=color, linewidth=1.4, zorder=3, label=label)
        axes.fill_between(centres, counts, alpha=0.15, color=color, zorder=2)

    step_line(deviations_high, ACCENT, f"HIGH ({len(deviations_high)})")
    step_line(deviations_low,  ACCENT2,  f"LOW ({len(deviations_low)})")

    axes.set_title("Outlier Deviation Distribution", color=TEXT, fontsize=12, pad=6)
    axes.set_xlabel("Deviation beyond IQR fence", color=SUBTEXT, fontsize=9)
    axes.set_ylabel("Number of Stations", color=SUBTEXT, fontsize=9)
    axes.tick_params(colors=SUBTEXT, labelsize=8)
    axes.grid(axis="y", color=BORDER, linewidth=0.5, zorder=1)
    axes.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=SUBTEXT, edgecolor=BORDER)