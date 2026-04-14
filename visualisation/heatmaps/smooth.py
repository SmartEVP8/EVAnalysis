import numpy as np
from scipy.ndimage import gaussian_filter


def gaussian_splat(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Clean heatmap generator:
    - converts station points into smooth gaussian influence field
    """

    H, W = grid_lats.shape
    heat = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    lat_min, lat_max = grid_lats[-1, 0], grid_lats[0, 0]
    lon_min, lon_max = grid_lons[0, 0], grid_lons[0, -1]

    xs = (lons - lon_min) / (lon_max - lon_min) * (W - 1)
    ys = (lat_max - lats) / (lat_max - lat_min) * (H - 1)

    for x, y, v in zip(xs, ys, values):
        if not np.isfinite(v):
            continue

        ix = int(x)
        iy = int(y)

        if ix < 0 or ix >= W or iy < 0 or iy >= H:
            continue

        heat[iy, ix] += v
        weight[iy, ix] += 1.0

    heat = gaussian_filter(heat, sigma=sigma)
    weight = gaussian_filter(weight, sigma=sigma)

    return np.divide(heat, weight, out=np.zeros_like(heat), where=weight > 0)