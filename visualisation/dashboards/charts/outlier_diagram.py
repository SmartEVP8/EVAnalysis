"""
Dot plot showing stations flagged as outliers at a single snapshot moment,
with their observed value plotted against the IQR bounds.
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
    station_outliers: pl.DataFrame,
    simtime_ms: int,
    figsize: tuple[float, float] = (7, 3.5),
) -> plt.Figure:
    """
    Render a dot plot of outlier stations at the given simtime_ms.
    Each point shows the station's observed value; horizontal bars show
    the IQR [p25, p75] and whisker [lower, upper] bounds for context.

    Parameters
    ----------
    station_outliers:
        Full station_outliers.parquet loaded as a Polars DataFrame.
    simtime_ms:
        The simulation timestamp to filter to.
    figsize:
        Figure dimensions in inches.
    """
    df = (
        station_outliers
        .filter(pl.col("simtime_ms") == simtime_ms)
        .sort("value", descending=True)
    )

    if df.is_empty():
        fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
        ax.set_facecolor(PANEL_BG)
        ax.text(0.5, 0.5, "No outliers at this snapshot ✓",
                ha="center", va="center", transform=ax.transAxes,
                color=ACCENT2, fontsize=12, style="italic")
        ax.set_axis_off()
        return fig

    labels = [f"S{row['StationId']} ({row['metric']})" for row in df.iter_rows(named=True)]
    values = df["value"].to_list()
    uppers = df["upper"].to_list()
    p75s   = df["p75"].to_list()
    flags  = df["flag"].to_list()

    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(PANEL_BG)

    y = np.arange(len(labels))

    # Draw the upper whisker as a faint reference line
    for i, (u, p) in enumerate(zip(uppers, p75s)):
        ax.plot([p, u], [y[i], y[i]], color=BORDER, linewidth=2.0, solid_capstyle="round", zorder=1)

    # Draw observed value dot
    dot_colors = [ACCENT3 if f == "HIGH" else ACCENT for f in flags]
    ax.scatter(values, y, color=dot_colors, s=40, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=SUBTEXT, fontsize=7)
    ax.set_xlabel("Metric value", color=SUBTEXT, fontsize=9)
    ax.set_title("Station Outliers at This Snapshot", color=TEXT, fontsize=11, pad=6)
    ax.tick_params(axis="x", colors=SUBTEXT, labelsize=8)

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(axis="x", color=BORDER, linewidth=0.5, zorder=0)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ACCENT3,
               markersize=6, label="HIGH outlier"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ACCENT,
               markersize=6, label="LOW outlier"),
        Line2D([0], [0], color=BORDER, linewidth=2, label="IQR → upper bound"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, facecolor=PANEL_BG,
              labelcolor=SUBTEXT, edgecolor=BORDER)

    fig.tight_layout()
    return fig