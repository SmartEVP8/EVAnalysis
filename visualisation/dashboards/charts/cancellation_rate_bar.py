"""
Bar chart showing cancellation rate per station at a single snapshot moment,
with raw reservation and cancellation counts annotated.
"""
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

BG       = "#0f1117"
PANEL_BG = "#1a1d27"
ACCENT   = "#4fc3f7"
ACCENT3  = "#e57373"
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
    Render a horizontal bar chart of cancellation_rate per station at simtime_ms.
    Only stations with at least one reservation are shown (avoids 0/0 noise).
    Bars are annotated with raw cancellation / reservation counts.

    Parameters
    ----------
    station_snapshots:
        Full station_snapshots.parquet loaded as a Polars DataFrame.
    simtime_ms:
        The simulation timestamp to filter to.
    figsize:
        Figure dimensions in inches.
    """
    df = (
        station_snapshots
        .filter(pl.col("simtime_ms") == simtime_ms)
        .filter(pl.col("Reservations") > 0)   # only stations with bookings
        .sort("cancellation_rate", descending=True)
    )

    if df.is_empty():
        fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
        ax.set_facecolor(PANEL_BG)
        ax.text(0.5, 0.5, "No reservations at this snapshot",
                ha="center", va="center", transform=ax.transAxes,
                color=SUBTEXT, fontsize=12, style="italic")
        ax.set_axis_off()
        return fig

    station_ids   = [str(s) for s in df["StationId"].to_list()]
    rates         = df["cancellation_rate"].to_list()
    cancellations = df["Cancellations"].to_list()
    reservations  = df["Reservations"].to_list()

    colors = [ACCENT3 if r > 0.5 else ACCENT for r in rates]

    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(PANEL_BG)

    y = np.arange(len(station_ids))
    ax.barh(y, rates, color=colors, height=0.65, zorder=2)

    # Annotate each bar with "X / Y" (cancellations / reservations)
    for i, (rate, c, r) in enumerate(zip(rates, cancellations, reservations)):
        ax.text(
            rate + 0.01, i,
            f"{c}/{r}",
            va="center", color=SUBTEXT, fontsize=6.5,
        )

    # Reference line at 0.5
    ax.axvline(0.5, color=ACCENT3, linewidth=0.8, linestyle="--", alpha=0.5, zorder=3)

    ax.set_xlim(0, 1.15)
    ax.set_yticks(y)
    ax.set_yticklabels(station_ids, color=SUBTEXT, fontsize=7)
    ax.set_xlabel("Cancellation rate", color=SUBTEXT, fontsize=9)
    ax.set_title("Cancellation Rate per Station", color=TEXT, fontsize=11, pad=6)
    ax.tick_params(axis="x", colors=SUBTEXT, labelsize=8)

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(axis="x", color=BORDER, linewidth=0.5, zorder=1)

    fig.tight_layout()
    return fig