"""
Module: dashboard
Renders a simulation score dashboard as a PNG image using matplotlib.

Called at the end of run_scoring() in save_scores.py. Reads the already-computed
score dicts and produces scoring/dashboard.png alongside the JSON files.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


BACKGROUND = "#0E1117"
SURFACE = "#161B24"
CARD_BACKGROUND = "#1C232F"
BORDER = "#252D3A"
TEXT_PRIMARY_COLOR = "#E8ECF0"
TEXT_SECONDARY_COLOR = "#7A8899"
TEXT_TERTIARY_COLOR = "#4A5568"

def score_color(score_0_to_100: float) -> str:
    if score_0_to_100 >= 75:
        return "#1D9E75"
    if score_0_to_100 >= 40:
        return "#EF9F27"
    return "#E24B4A"

def score_label(score_0_to_100: float) -> str:
    if score_0_to_100 >= 75:
        return "GOOD"
    if score_0_to_100 >= 40:
        return "MODERATE"
    return "POOR"



def draw_ring(axes: plt.Axes, score: float) -> None:
    """Draws the circular score ring in the given axes."""
    axes.set_aspect("equal")
    axes.axis("off")

    full = 2 * math.pi
    filled = full * (score / 100)
    gap = 0.04

    background_arc = np.linspace(filled + gap / 2, full - gap / 2 + filled, 300)
    score_arc  = np.linspace(gap / 2, filled - gap / 2, max(2, int(300 * score / 100)))

    ring_outer, ring_inner = 1.0, 0.72

    def arc_patch(arc_values, color, alpha=1.0):
        x_outer = ring_outer * np.cos(arc_values)
        y_outer = ring_outer * np.sin(arc_values)

        x_inner = ring_inner * np.cos(arc_values[::-1])
        y_inner = ring_inner * np.sin(arc_values[::-1])

        vertices = np.column_stack([
            np.concatenate([x_outer, x_inner]),
            np.concatenate([y_outer, y_inner]),
        ])

        patch = plt.Polygon(
            vertices,
            closed=True,
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            zorder=3
        )

        axes.add_patch(patch)

    # full circle
    if len(background_arc) > 1:
        arc_patch(background_arc, BORDER)

    # filled part of the circle
    if len(score_arc) > 1:
        color = score_color(score)
        arc_patch(score_arc, color)

        for angle in [score_arc[0], score_arc[-1]]:
            center_x = ((ring_outer + ring_inner) / 2) * math.cos(angle)
            center_y = ((ring_outer + ring_inner) / 2) * math.sin(angle)
            cap_radius = (ring_outer - ring_inner) / 2
            circle = plt.Circle((center_x, center_y), cap_radius, color=color, zorder=4)
            axes.add_patch(circle)

    # centre score text
    axes.text(0, 0.08, f"{score:.1f}",
            horizontalalignment="center", verticalalignment="center", fontsize=28, fontweight="bold",
            color=TEXT_PRIMARY_COLOR, zorder=5)
    axes.text(0, -0.22, "out of 100",
            horizontalalignment="center", verticalalignment="center", fontsize=7,
            color=TEXT_SECONDARY_COLOR, zorder=5)

    color = score_color(score)
    label = score_label(score)
    axes.text(0, -0.48, label,
            horizontalalignment="center", verticalalignment="center", fontsize=6.5, fontweight="bold",
            color=color, zorder=5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color + "22",
                      edgecolor=color + "55", linewidth=0.5))

    axes.set_xlim(-1.25, 1.25)
    axes.set_ylim(-1.25, 1.25)


def draw_metric_row(axes: plt.Axes, metrics: list[dict], title: str) -> None:
    axes.axis("off")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)

    # section title
    axes.text(0, 1.0, title, fontsize=7, fontweight="bold",
            color=TEXT_TERTIARY_COLOR, verticalalignment="top",
            transform=axes.transAxes)

    metrics_length = len(metrics)
    if metrics_length == 0:
        return

    card_width = 1.0 / metrics_length
    gap = 0.015
    padding = 0.025

    for i, metric in enumerate(metrics):
        score = metric["score"]
        color = score_color(score)
        label = score_label(score)

        left_edge = i * card_width + gap / 2
        right_edge = left_edge + card_width - gap
        bottom_edge, top_edge = 0.0, 0.88

        # Card background
        rectangle = FancyBboxPatch((left_edge, bottom_edge), right_edge - left_edge, top_edge - bottom_edge,
                              boxstyle="round,pad=0.008",
                              facecolor=CARD_BACKGROUND, edgecolor=BORDER,
                              linewidth=0.5, zorder=2,
                              transform=axes.transAxes, clip_on=False)
        axes.add_patch(rectangle)

        center_x = (left_edge + right_edge) / 2

        # Metric name
        axes.text(center_x, top_edge - padding, f'{metric["name"]} score',
            horizontalalignment="center", verticalalignment="top", fontsize=9,
            color=TEXT_PRIMARY_COLOR, transform=axes.transAxes, zorder=3)

        # score value
        axes.text(center_x, (bottom_edge + top_edge) / 2 + 0.05, f"{score:.1f}",
            horizontalalignment="right", verticalalignment="center", fontsize=14, fontweight="bold",
            color=score_color(score),
            transform=axes.transAxes, zorder=3)

        # "/ 100"
        axes.text(center_x, (bottom_edge + top_edge) / 2 + 0.05, " / 100",
            horizontalalignment="left", verticalalignment="center", fontsize=14, fontweight="bold",
            color=TEXT_PRIMARY_COLOR,
            transform=axes.transAxes, zorder=3)
        
        # Denominator
        axes.text(center_x, (bottom_edge + top_edge) / 2 - 0.10, "",
            horizontalalignment="center", verticalalignment="center", fontsize=6,
            color=TEXT_TERTIARY_COLOR, transform=axes.transAxes, zorder=3)

        bar_y = bottom_edge + 0.10
        bar_height = 0.055
        bar_left_edge = left_edge + padding
        bar_right_edge = right_edge - padding
        bar_width_total = bar_right_edge - bar_left_edge

        track = FancyBboxPatch((bar_left_edge, bar_y), bar_width_total, bar_height,
                               boxstyle="round,pad=0.002",
                               facecolor=BORDER, edgecolor="none",
                               zorder=3, transform=axes.transAxes, clip_on=False)
        axes.add_patch(track)

        fill_width = bar_width_total * min(score / 100, 1.0)
        if fill_width > 0.002:
            fill = FancyBboxPatch((bar_left_edge, bar_y), fill_width, bar_height,
                                  boxstyle="round,pad=0.002",
                                  facecolor=color, edgecolor="none",
                                  zorder=4, transform=axes.transAxes, clip_on=False)
            axes.add_patch(fill)

        axes.text(center_x, bottom_edge + 0.30, label,
                horizontalalignment="center", verticalalignment="center", fontsize=5.5, fontweight="bold",
                color=color, transform=axes.transAxes, zorder=3,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=color + "22",
                          edgecolor=color + "55", linewidth=0.4))



def generate_dashboard(
    run_id: str,
    overall: float,
    ev_scores: dict,
    station_scores: dict,
    output_path: Path,
) -> None:
    """
    Renders the score dashboard and saves it as a PNG.

    run_id : simulation run UUID string
    overall : float in [0, 1] — multiplied by 100 internally
    ev_scores : the ev_scores dict from run_scoring()
    station_scores : the station_scores dict from run_scoring()
    output_path : full path to write the PNG (e.g. scoring_dir / "dashboard.png")
    """
    overall_display = overall

    ev_metrics_names = ev_scores.get("per_metric", {})
    ev_metrics_display_names = {
        "path_deviation_minutes": "Path deviation",
        "delta_arrival_minutes": "Delta arrival",
        "missed_deadline": "Missed deadline",
    }
    ev_metrics = [
        {"name": ev_metrics_display_names.get(metric_key, metric_key), "score": round(metric_value["metric_score"] * 100, 1)}
        for metric_key, metric_value in ev_metrics_names.items()
    ]

    station_metrics_names = station_scores.get("per_metric", {})
    station_metrics_display_names = {
        "utilization": "Utilization",
        "queue_size":  "Queue size",
    }
    station_metrics = [
        {"name": station_metrics_display_names.get(metric_key, metric_key), "score": round(metric_value["metric_score"] * 100, 1)}
        for metric_key, metric_value in station_metrics_names.items()
    ]

    fig = plt.figure(figsize=(10, 6.2), facecolor=BACKGROUND)

    gridspec = fig.add_gridspec(
        3, 2,
        left=0.03, right=0.97,
        top=0.90, bottom=0.05,
        wspace=0.06,
        hspace=0.55,
        width_ratios=[1, 2],
        height_ratios=[0.12, 1, 1],
    )

    axes_header = fig.add_subplot(gridspec[0, :])
    axes_header.axis("off")
    
    axes_header.text(0.0, 1.0, "Simulation score",
                   fontsize=16, fontweight="bold", color=TEXT_PRIMARY_COLOR,
                   verticalalignment="top", transform=axes_header.transAxes)
    
    axes_header.text(0.0, 0.0, f"RUN: {run_id}",
                   fontsize=13, color=TEXT_PRIMARY_COLOR,
                   verticalalignment="top", transform=axes_header.transAxes,
                   fontfamily="monospace")

    line = plt.Line2D([0, 1], [0.1, 0.1], transform=axes_header.transAxes,
                      color=BORDER, linewidth=0.5)
    axes_header.add_line(line)

    axes_ring = fig.add_subplot(gridspec[1:, 0], facecolor=BACKGROUND)
    axes_ring.set_facecolor(BACKGROUND)
    draw_ring(axes_ring, overall_display)

    axes_ev = fig.add_subplot(gridspec[1, 1], facecolor=BACKGROUND)
    axes_ev.set_facecolor(BACKGROUND)
    draw_metric_row(axes_ev, ev_metrics, "EV METRICS")

    ax_station = fig.add_subplot(gridspec[2, 1], facecolor=BACKGROUND)
    ax_station.set_facecolor(BACKGROUND)
    draw_metric_row(ax_station, station_metrics, "STATION METRICS")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight",
                facecolor=BACKGROUND, edgecolor="none")
    plt.close(fig)
    print(f"  Scoring Dashboard PNG -> {output_path}")