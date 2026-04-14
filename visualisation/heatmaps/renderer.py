import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from visualisation.heatmaps.idw import interpolate_grid

from .denmark import DenmarkGrid, build_land_mask, load_denmark_boundary
from .heatmaps_loader import HeatmapDataset

logger = logging.getLogger(__name__)

METRICS: list[tuple[str, str]] = [
    ("queue_size", "total_queue_size"),
    ("utilization", "utilization"),
    ("cancellation_rate", "cancellation_rate"),
]

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

BG = "#0b0f14"

def decode_snapshot(snapshot_id: int) -> tuple[int, str]:
    """
    snapshot_id = day * 1_000_000 + time_of_day_ms
    Returns (day, "HH:MM")
    """
    day = snapshot_id // 1_000_000
    time_of_day = snapshot_id % 1_000_000

    total_seconds = time_of_day
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)

    return day, f"{hours:02d}:{minutes:02d}"


def format_title(metric_name: str, snapshot_id: int) -> str:
    day, time_str = decode_snapshot(snapshot_id)
    weekday = _WEEKDAYS[day % 7]
    metric_display = _METRIC_DISPLAY_NAMES.get(
        metric_name,
        metric_name.replace("_", " ").title()
    )
    return f"{weekday}, Day {day} of simulation, {time_str}, {metric_display}"


def render_all(
    dataset: HeatmapDataset,
    output_dir: Path,
    resolution_km: float = 5.0,
    use_land_mask: bool = True,
    dpi: int = 150,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = DenmarkGrid.default(resolution_km=resolution_km)
    land_mask = build_land_mask(grid) if use_land_mask else None

    logger.info("Loading Denmark boundary (10m resolution)...")
    dk_boundary = load_denmark_boundary()

    extent = [grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max]

    for metric_name, col_name in METRICS:
        metric_dir = output_dir / metric_name
        metric_dir.mkdir(parents=True, exist_ok=True)

        cfg = METRIC_CONFIG[metric_name]

        for i, snap in enumerate(
            tqdm(dataset.snapshots, desc=f"Rendering {metric_name}")
        ):
            out_path = metric_dir / f"{metric_name}_{i:04d}.png"

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
                power=2.0
            )

            if land_mask is not None:
                raster[~land_mask] = np.nan

            raster = np.nan_to_num(raster, nan=0.0)

            fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG)
            ax.set_facecolor(BG)

            im = ax.imshow(
                raster,
                extent=extent,
                origin="lower",
                cmap=cfg["cmap"],
                vmin=cfg["vmin"],
                vmax=cfg["vmax"],
                interpolation="bilinear",
            )

            dk_boundary.boundary.plot(
                ax=ax,
                linewidth=0.8,
                color="#c8c8c8",
                zorder=3,
            )

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.10)
            cax.set_facecolor(BG)

            cb = fig.colorbar(im, cax=cax)
            cb.set_label(cfg["colorbar_label"], color="white", fontsize=9)
            cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
            cb.outline.set_edgecolor("#444444")

            title = format_title(metric_name, snap.snapshot_id)
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

    logger.info("Done.")