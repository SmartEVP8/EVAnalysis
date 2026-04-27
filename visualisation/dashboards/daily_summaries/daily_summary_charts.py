"""
Shared colour constants and matplotlib drawing primitives for the daily summary dashboard.
"""
from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


BG = "#0f1117"
PANEL_BG = "#1a1d27"
ACCENT = "#4fc3f7"   # blue   – utilization / KPI 1
ACCENT2 = "#81c784"   # green  – queue / KPI 2
ACCENT3 = "#e57373"   # red    – deadlines / KPI 3
ACCENT4 = "#ce93d8"   # purple – cancellations KPI 4
ACCENT5 = "#ffb74d"   # amber  – reservations KPI 5
ACCENT6 = "#f06292"   # pink   – cancellation rate KPI 6
TEXT = "#e8eaf6"
SUBTEXT = "#9fa8da"
BORDER = "#2a2d3e"


UTIL_COLORS      = ["#1b5e20", "#66bb6a", "#fdd835", "#e53935"] # green ramp
QUEUE_COLORS     = ["#0d47a1", "#42a5f5", "#fdd835", "#e53935"] # blue ramp
DEVIATION_COLORS = ["#4a148c", "#ab47bc", "#fdd835", "#e53935"] # purple ramp

KPI_COLORS = [ACCENT, ACCENT2, ACCENT3, ACCENT4, ACCENT5, ACCENT6]


def draw_kpi_card(
    axes: plt.Axes,
    label: str,
    value: str,
    color: str,
    subtitle: str | None = None,
) -> None:
    """Renders a single KPI card into the given axes."""
    axes.set_facecolor(PANEL_BG)
    for spine in axes.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(1.5)

    value_y = 0.38 if subtitle is None else 0.44
    label_y = 0.72 if subtitle is None else 0.78

    axes.text(0.5, value_y, value,
            horizontalalignment="center", verticalalignment="center", transform=axes.transAxes,
            fontsize=26, fontweight="bold", color=color, fontfamily="monospace")
    axes.text(0.5, label_y, label,
            horizontalalignment="center", verticalalignment="center", transform=axes.transAxes,
            fontsize=11, color=color, alpha=0.65)

    if subtitle is not None:
        axes.text(0.5, 0.18, subtitle,
                horizontalalignment="center", verticalalignment="center", transform=axes.transAxes,
                fontsize=10, color=color, alpha=0.50, fontfamily="monospace")

    axes.set_xticks([])
    axes.set_yticks([])


def draw_layered_bar_chart(
    axes: plt.Axes,
    interval_labels: list[str],
    p25: np.ndarray,
    p50: np.ndarray,
    p75: np.ndarray,
    p_max: np.ndarray,
    colors: list[str],
    title: str,
    ylabel: str,
    y_formatter=None,
    if_empty_message: str = "No data available",
) -> None:
    """
    Draws a stacked (layered) bar chart where each segment is the incremental
    band between percentile thresholds, so the total bar height equals the max value.

    Bands (bottom -> top):
        colors[0] | from 0 to p25
        colors[1] | from p25 to p50
        colors[2] | from p50 to p75
        colors[3] | from p75 to max

    If there is no data, an italic empty-state message is displayed instead.
    """
    axes.set_facecolor(PANEL_BG)
    for border in axes.spines.values():
        border.set_edgecolor(BORDER)
        border.set_linewidth(0.8)
    axes.set_title(title, color=SUBTEXT, fontsize=13, pad=6)

    if len(interval_labels) == 0 or p_max.sum() == 0:
        axes.text(0.5, 0.5, if_empty_message,
                horizontalalignment="center", verticalalignment="center", transform=axes.transAxes,
                fontsize=13, color=ACCENT3, style="italic")
        axes.set_xticks([])
        axes.set_yticks([])
        return

    x = np.arange(len(interval_labels))
    width = 0.65

    h0 = p25
    h1 = np.maximum(p50 - p25, 0)
    h2 = np.maximum(p75 - p50, 0)
    h3 = np.maximum(p_max - p75, 0)

    axes.bar(x, h0, color=colors[0], width=width, zorder=1)
    axes.bar(x, h1, bottom=h0, color=colors[1], width=width, zorder=2)
    axes.bar(x, h2, bottom=h0+h1, color=colors[2], width=width, zorder=3)
    axes.bar(x, h3, bottom=h0+h1+h2, color=colors[3], width=width, zorder=4)

    axes.set_xticks(x)
    axes.set_xticklabels(interval_labels, rotation=45, horizontalalignment="right", fontsize=8, color=SUBTEXT)
    axes.tick_params(axis="y", colors=SUBTEXT, labelsize=9)
    axes.set_ylabel(ylabel, color=SUBTEXT, fontsize=10)
    axes.set_xlim(-0.5, len(x) - 0.5)
    axes.set_ylim(bottom=0)

    if y_formatter is not None:
        axes.yaxis.set_major_formatter(y_formatter)

    legend_labels = ["≤ p25", "p25 -> p50", "p50 -> p75", "p75 -> max"]
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, legend_labels)]
    axes.legend(handles=patches, loc="upper left", fontsize=8,
              facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT)

    axes.grid(axis="y", color=BORDER, linewidth=0.5, alpha=0.6)
    axes.set_axisbelow(True)