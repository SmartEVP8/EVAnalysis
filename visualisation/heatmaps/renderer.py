import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from .denmark import DenmarkGrid, build_land_mask, load_denmark_boundary
from .idw import interpolate_grid
from .loader import HeatmapDataset

logger = logging.getLogger(__name__)

METRICS: list[tuple[str, str]] = [
    ("queue_size", "queue_size_per_charger"),
    ("utilization", "utilization"),
]

METRIC_CONFIG: dict[str, dict] = {
    "queue_size": {
        "cmap": "magma",
        "vmin": 0.0,
        "vmax": None,
        "colorbar_label": "Avg queue size (vehicles)",
    },
    "utilization": {
        "cmap": "magma",
        "vmin": 0.0,
        "vmax": 1.0,
        "colorbar_label": "Utilization (0 – 1)",
    },
}

_WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

_METRIC_DISPLAY_NAMES: dict[str, str] = {
    "queue_size": "Queue Size",
    "utilization": "Utilization",
}

BG = "#0b0f14"


def decode_snapshot(snapshot_id: int) -> tuple[int, str]:
    """
    Decode a snapshot id into (simulation_day, time_string).

    snapshot_id = day * 1_000_000 + time_of_day_ms
    Returns the day index and a "HH:MM" string.
    """
    day = snapshot_id // 1_000_000
    time_of_day = snapshot_id % 1_000_000

    total_seconds = time_of_day / 1000
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)

    return day, f"{hours:02d}:{minutes:02d}"


def format_title(metric_name: str, snapshot_id: int) -> str:
    """
    Build a human-readable title string.

    Format: "{Day of week}, Day {X} of simulation, {HH:MM}, {Metric Name}"
    Day 0 = Monday, cycling through the week indefinitely.
    """
    day, time_str = decode_snapshot(snapshot_id)
    weekday = _WEEKDAYS[day % 7]
    metric_display = _METRIC_DISPLAY_NAMES.get(metric_name, metric_name.replace("_", " ").title())
    return f"{weekday}, Day {day} of simulation, {time_str}, {metric_display}"


def render_all(
    dataset: HeatmapDataset,
    output_dir: Path,
    resolution_km: float = 5.0,
    idw_power: float = 2.0,
    use_land_mask: bool = True,
    dpi: int = 150,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = DenmarkGrid.default(resolution_km=resolution_km)
    land_mask = build_land_mask(grid) if use_land_mask else None

    logger.info("Loading Denmark boundary (10m resolution)...")
    dk_boundary = load_denmark_boundary()

    _compute_global_vmaxes(dataset)

    extent = [grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max]

    for metric_name, col_name in METRICS:
        metric_dir = output_dir / metric_name
        metric_dir.mkdir(parents=True, exist_ok=True)
        cfg = METRIC_CONFIG[metric_name]

        for i, snap in enumerate(tqdm(dataset.snapshots, desc=f"Rendering {metric_name}")):
            out_path = metric_dir / f"{metric_name}_{i:04d}.png"

            try:
                lats, lons, values = snap.metric_arrays(col_name)
            except Exception:
                continue
            if len(values) == 0:
                continue

            raster = interpolate_grid(
                grid.lat_grid, grid.lon_grid,
                lats, lons, values,
                power=idw_power,
            )

            if land_mask is not None:
                raster[~land_mask] = np.nan

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
                ax=ax, linewidth=0.8, color="#c8c8c8", zorder=3
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
                0.5, 0.985,
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


def _compute_global_vmaxes(dataset: HeatmapDataset) -> None:
    for metric_name, col_name in METRICS:
        cfg = METRIC_CONFIG[metric_name]
        if cfg["vmax"] is not None:
            continue
        global_max = 0.0
        for snap in dataset.snapshots:
            try:
                _, _, values = snap.metric_arrays(col_name)
                if len(values):
                    global_max = max(global_max, float(values.max()))
            except Exception:
                continue
        cfg["vmax"] = global_max if global_max > 0 else 1.0