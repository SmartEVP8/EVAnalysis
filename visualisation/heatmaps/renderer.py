"""
Converts interpolated grid data into formatted PNG images with consistent 
styling, color scales, and geographic boundaries.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from visualisation.heatmaps.inverse_distance_weighting import IDWInterpolator
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

_METRIC_DISPLAY_NAMES: dict[str, str] = {
    "queue_size": "Queue Size",
    "utilization": "Utilization",
    "cancellation_rate": "Cancellation Rate",
}

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

    extent = [grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max]

    # Figure geometry computed once.
    lat_range = grid.lat_max - grid.lat_min
    lon_range = grid.lon_max - grid.lon_min
    lat_correction = np.cos(np.radians(56.0))
    fig_w = 8.0
    fig_h = fig_w * (lat_range / lon_range) / lat_correction

    # Preallocate the zero raster used when a snapshot has no data.
    zero_raster = np.zeros_like(grid.lat_grid)

    for metric_name, col_name in METRICS:
        metric_dir = output_dir / metric_name
        metric_dir.mkdir(parents=True, exist_ok=True)

        config = METRIC_CONFIG[metric_name]

        interpolator: IDWInterpolator | None = None

        # Use tqdm to show a progress bar in the terminal while rendering
        for i, snap in enumerate(
            tqdm(dataset.snapshots, desc=f"Rendering {metric_name}")
        ):
            out_path = metric_dir / f"{metric_name}_{i}.png"

            # Skip frames that have already been rendered.
            if out_path.exists():
                continue

            try:
                lats, lons, values = snap.metric_arrays(col_name)
                has_data = len(values) > 0
            except Exception:
                has_data = False

            if has_data:
                if interpolator is None or interpolator._n_stations != len(lats):
                    interpolator = IDWInterpolator(
                        lats, lons,
                        grid.lat_grid, grid.lon_grid,
                    )

                raster = interpolator.interpolate(values)

                if land_mask is not None:
                    raster[~land_mask] = np.nan
                raster = np.nan_to_num(raster, nan=0.0)
            else:
                raster = zero_raster.copy()

            fig, axes = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
            axes.set_facecolor(BG)
            axes.margins(0)
            axes.set_xlim(grid.lon_min, grid.lon_max)
            axes.set_ylim(grid.lat_min, grid.lat_max)

            # Draw the heatmap
            im = axes.imshow(
                raster,
                extent=extent,
                origin="lower",
                cmap=config["cmap"],
                vmin=config["vmin"],
                vmax=config["vmax"],
                interpolation="sinc",
                aspect="auto",
            )

            dk_boundary.boundary.plot(
                ax=axes,
                linewidth=0.8,
                color="#b5b5b5",
                zorder=3,
            )

            colorbar_axes = fig.add_axes([0.92, 0.1, 0.05, 0.78])
            colorbar_axes.set_facecolor(BG)

            colorbar = fig.colorbar(im, cax=colorbar_axes)
            colorbar.set_label(config["colorbar_label"], color="white", fontsize=9)
            colorbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
            colorbar.outline.set_edgecolor("#444444")

            title = format_title(metric_name, snap)
            axes.text(
                0.5,
                0.985,
                title,
                transform=axes.transAxes,
                color="white",
                fontsize=14,
                va="top",
                ha="center",
                alpha=0.85,
            )

            axes.set_axis_off()

            fig.savefig(
                out_path,
                dpi=dpi,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )

            plt.close(fig)