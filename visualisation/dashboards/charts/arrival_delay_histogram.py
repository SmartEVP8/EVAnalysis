"""
Histogram of EV arrival delays (delta_arrival_minutes) for all arrivals
up to the current snapshot time.
"""
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

BG       = "#0f1117"
PANEL_BG = "#1a1d27"
ACCENT   = "#4fc3f7"
ACCENT2  = "#81c784"
ACCENT3  = "#e57373"
TEXT     = "#e8eaf6"
SUBTEXT  = "#9fa8da"
BORDER   = "#2a2d3e"


def render(
    arrival_snapshots: pl.DataFrame,
    simtime_ms: int,
    figsize: tuple[float, float] = (7, 3.5),
) -> plt.Figure:
    """
    Render a histogram of delta_arrival_minutes for all EVs that arrived
    up to (and including) simtime_ms. Negative = arrived early, positive = late.

    Parameters
    ----------
    arrival_snapshots:
        Full arrival_snapshots.parquet loaded as a Polars DataFrame.
    simtime_ms:
        The simulation timestamp — only arrivals up to this point are shown.
    figsize:
        Figure dimensions in inches.
    """
    df = (
        arrival_snapshots
        .filter(pl.col("simtime_ms") <= simtime_ms)
        .filter(pl.col("delta_arrival_minutes") != 0)  # exclude no-deviation rows
        .drop_nulls(subset=["delta_arrival_minutes"])
    )

    if df.is_empty():
        fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
        ax.set_facecolor(PANEL_BG)
        ax.text(0.5, 0.5, "No arrival data yet",
                ha="center", va="center", transform=ax.transAxes,
                color=SUBTEXT, fontsize=12, style="italic")
        ax.set_axis_off()
        return fig

    deltas = df["delta_arrival_minutes"].to_numpy()

    n_early = int((deltas < 0).sum())
    n_late  = int((deltas > 0).sum())

    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(PANEL_BG)

    n_bins = min(30, max(5, len(deltas) // 2))

    # Split histogram: early arrivals in green, late in red
    ax.hist(deltas[deltas <= 0], bins=n_bins // 2, color=ACCENT2, alpha=0.75,
            edgecolor=BORDER, zorder=2, label=f"Early ({n_early})")
    ax.hist(deltas[deltas > 0],  bins=n_bins // 2, color=ACCENT3, alpha=0.75,
            edgecolor=BORDER, zorder=2, label=f"Late ({n_late})")

    ax.axvline(0, color=ACCENT, linewidth=1.0, linestyle="--", alpha=0.7)

    mean_delta = float(np.mean(deltas))
    ax.axvline(mean_delta, color="#ffb74d", linewidth=1.0, linestyle=":",
               label=f"Mean: {mean_delta:+.1f} min")

    ax.set_xlabel("Arrival Δ (minutes, negative = early)", color=SUBTEXT, fontsize=9)
    ax.set_ylabel("EV count", color=SUBTEXT, fontsize=9)
    ax.set_title("Arrival Delay Distribution", color=TEXT, fontsize=11, pad=6)
    ax.tick_params(colors=SUBTEXT, labelsize=8)

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(axis="y", color=BORDER, linewidth=0.5, zorder=1)

    ax.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=SUBTEXT, edgecolor=BORDER)

    fig.tight_layout()
    return fig