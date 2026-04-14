import numpy as np
from scipy.spatial import cKDTree

EARTH_RADIUS_KM = 6371.0

def lonlat_to_cartesian(lat, lon):
    """Converts lat/lon to 3D Cartesian coordinates for Euclidean distance."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = EARTH_RADIUS_KM * np.cos(lat_rad) * np.cos(lon_rad)
    y = EARTH_RADIUS_KM * np.cos(lat_rad) * np.sin(lon_rad)
    z = EARTH_RADIUS_KM * np.sin(lat_rad)
    return np.stack([x, y, z], axis=-1)

def interpolate_grid(
    pixel_lats: np.ndarray,
    pixel_lons: np.ndarray,
    station_lats: np.ndarray,
    station_lons: np.ndarray,
    station_values: np.ndarray,
    power: float = 2.0,
    k: int = 12,
    max_dist_km: float = 500.0,
) -> np.ndarray:
    H, W = pixel_lats.shape
    N = len(station_lats)

    if N == 0:
        return np.full((H, W), np.nan)

    station_points = lonlat_to_cartesian(station_lats, station_lons)
    pixel_points = lonlat_to_cartesian(pixel_lats.ravel(), pixel_lons.ravel())

    tree = cKDTree(station_points)

    distances, indices = tree.query(pixel_points, k=k, distance_upper_bound=max_dist_km)

    weights = 1.0 / np.maximum(distances, 1e-9)**power
    weights[np.isinf(distances)] = 0

    neighbor_values = station_values[indices]

    sum_weights = np.sum(weights, axis=1)
    weighted_values = np.sum(weights * neighbor_values, axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        result_flat = weighted_values / sum_weights
    
    return result_flat.reshape(H, W)