import numpy as np
from scipy.spatial import KDTree

def interpolate_grid(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    k: int = 1,
    max_dist_km: float = 30.0,
) -> np.ndarray:

    def to_cartesian(lat, lon):
        R = 6371.0
        lat_rad, lon_rad = np.radians(lat), np.radians(lon)
        x = R * np.cos(lat_rad) * np.cos(lon_rad)
        y = R * np.cos(lat_rad) * np.sin(lon_rad)
        z = R * np.sin(lat_rad)
        return np.stack([x, y, z], axis=-1)

    station_coords = to_cartesian(lats, lons)
    grid_coords = to_cartesian(grid_lats.flatten(), grid_lons.flatten())

    tree = KDTree(station_coords)
    dists, indices = tree.query(grid_coords, k=k, distance_upper_bound=max_dist_km)

    padded_values = np.append(values, 0.0)
    
    raster_flat = padded_values[indices]

    if k == 1:
        raster_flat[dists > max_dist_km] = 0.0

    return raster_flat.reshape(grid_lats.shape)