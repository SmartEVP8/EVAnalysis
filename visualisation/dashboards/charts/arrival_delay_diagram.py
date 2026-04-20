"""
Arrival delay distribution as step-lines for early and late arrivals.
"""
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

PANEL_BG = "#1a1d27"
ACCENT   = "#4fc3f7"
TEXT     = "#e8eaf6"
SUBTEXT  = "#9fa8da"
BORDER   = "#2a2d3e"


def render(axes: plt.Axes, arrival_snapshots: pl.DataFrame, simtime_ms: int) -> None:
    axes.set_facecolor(PANEL_BG)
    for border in axes.spines.values():
        border.set_edgecolor(BORDER)

    dataframe = (
        arrival_snapshots
        .filter(pl.col("simtime_ms") <= simtime_ms)
        .filter(pl.col("delta_arrival_minutes") != 0)
        .drop_nulls(subset=["delta_arrival_minutes"])
    )

    if dataframe.is_empty():
        axes.text(0.5, 0.5, "No arrival data yet", horizontalalignment="center", verticalalignment="center",
                transform=axes.transAxes, color=SUBTEXT, fontsize=11, style="italic")
        axes.set_xticks([])
        axes.set_yticks([])
        return

    arrivals_deltas_minutes  = dataframe["delta_arrival_minutes"].to_numpy()
    early_arrivals   = arrivals_deltas_minutes[arrivals_deltas_minutes <= 0]
    late_arrivals    = arrivals_deltas_minutes[arrivals_deltas_minutes > 0]

    bin_count = max(min(15, max(3, len(arrivals_deltas_minutes) // 4)), 3)

    def step_line(data, bins, color, label):
        if len(data) == 0:
            return
        counts, edges = np.histogram(data, bins=bins)
        centres = (edges[:-1] + edges[1:]) / 2
        axes.plot(centres, counts, color=color, linewidth=1.4, zorder=3, label=label)
        axes.fill_between(centres, counts, alpha=0.15, color=color, zorder=2)

    step_line(early_arrivals, bin_count, ACCENT, f"Early ({len(early_arrivals)})")
    step_line(late_arrivals,  bin_count, ACCENT, f"Late ({len(late_arrivals)})")

    axes.axvline(0, color="#e8eaf6", linewidth=1.0, linestyle="--", alpha=0.4, zorder=4)

    axes.set_title("Delay on Arrival Distribution", color=TEXT, fontsize=12, pad=6)
    axes.set_xlabel("Delta minutes (negative means early)", color=SUBTEXT, fontsize=9)
    axes.set_ylabel("EV count", color=SUBTEXT, fontsize=9)
    axes.tick_params(colors=SUBTEXT, labelsize=8)
    axes.grid(axis="y", color=BORDER, linewidth=0.5, zorder=1)
    axes.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=SUBTEXT, edgecolor=BORDER)