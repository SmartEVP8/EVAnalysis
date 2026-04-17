"""
Converts interpolated grid data into formatted PNG images with consistent 
styling, color scales, and geographic boundaries.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from visualisation.heatmaps.inverse_distance_weighting import interpolate_grid
from .denmark import DenmarkGrid, build_land_mask, load_denmark_boundary
from .heatmaps_loader import HeatmapDataset, SnapshotFrame

METRICS: list[tuple[str, str]] = [
    ("queue_size", "total_queue_size"),
    ("utilization", "utilization"),
    ("cancellation_rate", "cancellation_rate"),
]

# Aesthetic settings for the heatmaps (colors, labels, and value ranges)
METRIC_CONFIG: dict[str, dict] = {
    "queue_size": {
        "cmap": "magma",
        "colorbar_label": "Queue size",
        "vmin": 0.0,
        "vmax": 10.0,
    },
    "utilization": {
        "cmap": "magma",
        "colorbar_label": "Utilization",
        "vmin": 0.0,
        "vmax": 1.0,
    },
    "cancellation_rate": {
        "cmap": "magma",
        "colorbar_label": "Cancellation rate",
        "vmin": 0.0,
        "vmax": 1.0,
    },
}

_WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

_METRIC_DISPLAY_NAMES: dict[str, str] = {
    "queue_size": "Queue Size",
    "utilization": "Utilization",
    "cancellation_rate": "Cancellation Rate",
}

# The dark-mode background color (matches the aesthetics of modern dashboards)
BG = "#0b0f14"


def format_title(metric_name: str, snap: SnapshotFrame) -> str:
    row = snap.data.row(0, named=True)
    metric_display = _METRIC_DISPLAY_NAMES.get(metric_name, metric_name.replace("_", " ").title())
    return f"{row['weekday_name']}, Day {row['day']} of simulation, {row['time_label']}, {metric_display}"


def render_all(
    dataset: HeatmapDataset,
    output_dir: Path,
    resolution_km: float = 5.0,
    use_land_mask: bool = True,
    dpi: int = 150,
) -> None:
    """
    This function serves as the main loop for generating images at each time step in the
    simulation by first setting up the geographic grid and land mask, then iterating
    over each metric and each moment in time, interpolating station data into a smooth heatmap.
    Then it applies the land mask to prevent values from extending into the ocean, and
    finally saving the resulting image as a PNG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = DenmarkGrid.default(resolution_km=resolution_km)
    land_mask = build_land_mask(grid) if use_land_mask else None
    dk_boundary = load_denmark_boundary()

    # Bounding box
    extent = [grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max]

    for metric_name, col_name in METRICS:
        metric_dir = output_dir / metric_name
        metric_dir.mkdir(parents=True, exist_ok=True)

        config = METRIC_CONFIG[metric_name]

        # Use tqdm to show a progress bar in the terminal while rendering
        for i, snap in enumerate(
            tqdm(dataset.snapshots, desc=f"Rendering {metric_name}")
        ):
            out_path = metric_dir / f"{metric_name}_{i}.png"

            try:
                lats, lons, values = snap.metric_arrays(col_name)
            except Exception:
                continue

            if len(values) == 0:
                continue

            raster = interpolate_grid(
                lats, lons, values,
                grid.lat_grid,
                grid.lon_grid,
            )

            if land_mask is not None:
                raster[~land_mask] = np.nan

            raster = np.nan_to_num(raster, nan=0.0)

            fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG)
            ax.set_facecolor(BG)

            # Draw the heatmap
            im = ax.imshow(
                raster,
                extent=extent,
                origin="lower",
                cmap=config["cmap"],
                vmin=config["vmin"],
                vmax=config["vmax"],
                interpolation="sinc",
            )

            dk_boundary.boundary.plot(
                ax=ax,
                linewidth=0.8,
                color="#b5b5b5",
                zorder=3,
            )

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.10)
            cax.set_facecolor(BG)

            cb = fig.colorbar(im, cax=cax)
            cb.set_label(config["colorbar_label"], color="white", fontsize=9)
            cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
            cb.outline.set_edgecolor("#444444")

            title = format_title(metric_name, snap)
            ax.text(
                0.5,
                0.985,
                title,
                transform=ax.transAxes,
                color="white",
                fontsize=8.5,
                va="top",
                ha="center",
                alpha=0.85,
            )

            ax.set_axis_off()

            fig.savefig(
                out_path,
                dpi=dpi,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )

            plt.close(fig)