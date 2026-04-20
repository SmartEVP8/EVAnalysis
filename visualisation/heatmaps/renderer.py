"""
Converts interpolated grid data into formatted PNG images with consistent 
styling, color scales, and geographic boundaries.
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
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


"""
Each worker process calls _init_worker() once at startup. That function
builds the expensive shared objects (grid, land mask, boundary, figure
geometry) and stores them in module-level globals. Subsequent calls to
_render_frame() in that same process reuse them without rebuilding.
"""

_worker_grid = None
_worker_land_mask = None
_worker_boundary = None
_worker_extent = None
_worker_fig_w = None
_worker_fig_h = None
_worker_zero_raster = None
_worker_interpolators: dict[str, IDWInterpolator] = {}


def _init_worker(resolution_km: float, use_land_mask: bool) -> None:
    """
    Runs once per worker process at pool startup.
    Builds the grid, land mask, boundary, and figure geometry.
    """
    global _worker_grid, _worker_land_mask, _worker_boundary
    global _worker_extent, _worker_fig_w, _worker_fig_h, _worker_zero_raster

    _worker_grid = DenmarkGrid.default(resolution_km=resolution_km)
    _worker_land_mask = build_land_mask(_worker_grid) if use_land_mask else None
    _worker_boundary = load_denmark_boundary()
    _worker_extent = [
        _worker_grid.lon_min, _worker_grid.lon_max,
        _worker_grid.lat_min, _worker_grid.lat_max,
    ]

    lat_range = _worker_grid.lat_max - _worker_grid.lat_min
    lon_range = _worker_grid.lon_max - _worker_grid.lon_min
    _worker_fig_w = 8.0
    _worker_fig_h = _worker_fig_w * (lat_range / lon_range) / np.cos(np.radians(56.0))
    _worker_zero_raster = np.zeros_like(_worker_grid.lat_grid)

@dataclass
class _FrameTask:
    """Everything a worker needs to render one heatmap frame."""
    metric_name: str
    col_name:    str
    snapshot:    SnapshotFrame
    frame_index: int
    out_path:    Path
    dpi:         int


def _render_frame(task: _FrameTask) -> None:
    """
    Renders and saves a single heatmap frame. Runs inside a worker process.
    Grid, land mask, and boundary come from worker-local globals set by
    _init_worker, so they are never re-sent over the process boundary.
    The IDWInterpolator is cached per-worker per-metric: built on first use,
    reused for every subsequent frame that lands on the same worker.
    """
    if task.out_path.exists():
        return

    grid        = _worker_grid
    land_mask   = _worker_land_mask
    dk_boundary    = _worker_boundary
    extent      = _worker_extent
    zero_raster = _worker_zero_raster
    config      = METRIC_CONFIG[task.metric_name]

    try:
        lats, lons, values = task.snapshot.metric_arrays(task.col_name)
        has_data = len(values) > 0
    except Exception:
        has_data = False

    if has_data:
        interpolator = _worker_interpolators.get(task.metric_name)
        if interpolator is None or interpolator._n_stations != len(lats):
            interpolator = IDWInterpolator(lats, lons, grid.lat_grid, grid.lon_grid)
            _worker_interpolators[task.metric_name] = interpolator

        raster = interpolator.interpolate(values)
        if land_mask is not None:
            raster[~land_mask] = np.nan
        raster = np.nan_to_num(raster, nan=0.0)
    else:
        raster = zero_raster.copy()

    fig, axes = plt.subplots(figsize=(_worker_fig_w, _worker_fig_h), facecolor=BG)
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

    dk_boundary.boundary.plot(ax=axes, linewidth=0.8, color="#b5b5b5", zorder=3)

    colorbar_axes = fig.add_axes([0.92, 0.1, 0.05, 0.78])
    colorbar_axes.set_facecolor(BG)
    colorbar = fig.colorbar(im, cax=colorbar_axes)
    colorbar.set_label(config["colorbar_label"], color="white", fontsize=9)
    colorbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    colorbar.outline.set_edgecolor("#444444")

    row = task.snapshot.data.row(0, named=True)
    metric_display = _METRIC_DISPLAY_NAMES.get(
        task.metric_name, task.metric_name.replace("_", " ").title()
    )
    title = (
        f"{row['weekday_name']}, Day {row['day']} of simulation, "
        f"{row['time_label']}, {metric_display}"
    )
    axes.text(
        0.5, 0.985, title,
        transform=axes.transAxes,
        color="white", fontsize=14, verticalalignment="top", horizontalalignment="center", alpha=0.85,
    )

    axes.set_axis_off()
    fig.savefig(
        task.out_path,
        dpi=task.dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def render_all(
    dataset: HeatmapDataset,
    output_dir: Path,
    resolution_km: float = 5.0,
    use_land_mask: bool = True,
    dpi: int = 150,
) -> None:
    """
    Renders all heatmap frames in parallel using a multiprocessing pool.

    Each worker process initialises the grid, land mask, boundary, and figure
    geometry once via _init_worker, then handles many frames without rebuilding
    any of that shared state. The IDWInterpolator is also cached per-worker.

    Frames that already exist on disk are skipped. Pool size defaults to None,
    which lets Python use all available CPU cores automatically.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[_FrameTask] = []
    for metric_name, col_name in METRICS:
        metric_dir = output_dir / metric_name
        metric_dir.mkdir(parents=True, exist_ok=True)
        for i, snap in enumerate(dataset.snapshots):
            tasks.append(_FrameTask(
                metric_name = metric_name,
                col_name    = col_name,
                snapshot        = snap,
                frame_index = i,
                out_path    = metric_dir / f"{metric_name}_{i}.png",
                dpi         = dpi,
            ))

    total = len(tasks)
    print(f"Rendering {total} heatmap frames across all metrics...")

    with mp.Pool(
        processes=None,
        initializer=_init_worker,
        initargs=(resolution_km, use_land_mask),
    ) as pool:

        for _ in tqdm(
            pool.imap_unordered(_render_frame, tasks),
            total=total,
            desc="Heatmaps",
        ):
            pass