"""
Provides spatial interpolation utilities to convert scattered station data 
into a continuous grid (raster) for mapping.
"""

import numpy as np
from scipy.spatial import KDTree


def _to_cartesian(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Converts Latitude/Longitude into 3D XYZ coordinates (Cartesian).
    """
    R = 6371.0  # Earth's approximate radius in kilometers
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    x = R * np.cos(lat_rad) * np.cos(lon_rad)
    y = R * np.cos(lat_rad) * np.sin(lon_rad)
    z = R * np.sin(lat_rad)
    return np.stack([x, y, z], axis=-1)


class IDWInterpolator:
    """
    A pre-built interpolator for a fixed set of station positions and a fixed grid.

    Building the KDTree and querying every grid cell for
    its k nearest stations are done once in __init__. After that, interpolate()
    is just an array index + reshape.
    """

    def __init__(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
        k: int = 5,
        max_dist_km: float = 30.0,
    ) -> None:
        self._grid_shape = grid_lats.shape
        self._k = k
        self._n_stations = len(lats)

        station_coords = _to_cartesian(lats, lons)
        grid_coords = _to_cartesian(grid_lats.flatten(), grid_lons.flatten())

        tree = KDTree(station_coords)

        # Query once - indices and distances are reused for every snapshot.
        self._dists, self._indices = tree.query(
            grid_coords, k=k, distance_upper_bound=max_dist_km
        )

    def interpolate(self, values: np.ndarray) -> np.ndarray:
        """
        Map a new set of per-station values onto the cached grid.
        """
        # Ensure 'values' matches the length the KDTree was built for.
        # If the calling loop is correct, this is a safety net.
        n_input = len(values)
        
        # We use self._n_stations + 1 to accommodate the KDTree 
        # 'upper_bound' index (which points to the end of the array)
        padded = np.zeros(self._n_stations + 1)
        
        # Fill only up to what we have or what the tree expects
        fill_limit = min(n_input, self._n_stations)
        padded[:fill_limit] = values[:fill_limit]
        
        neighbor_values = padded[self._indices]

        if self._k == 1:
            raster_flat = neighbor_values
        else:
            # Simple average of the k-nearest neighbors
            raster_flat = np.mean(neighbor_values, axis=1)

        return raster_flat.reshape(self._grid_shape)

def interpolate_grid(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    k: int = 5,
    max_dist_km: float = 30.0,
) -> np.ndarray:

    interpolator = IDWInterpolator(lats, lons, grid_lats, grid_lons, k, max_dist_km)
    return interpolator.interpolate(values)