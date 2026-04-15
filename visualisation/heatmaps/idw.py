"""
Provides spatial interpolation utilities to convert scattered station data 
into a continuous grid (raster) for mapping.
"""

import numpy as np
from scipy.spatial import KDTree

def interpolate_grid(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    k: int = 5,
    max_dist_km: float = 30.0,
) -> np.ndarray:
    """
    Spreads charging station data across a geographic grid using Nearest Neighbor logic.

    Because stations are just points on a map, we need to decide what the values 
    are for the "empty spaces" between them. This function finds the closest 
    station(s) for every single square on our grid.

    Args:
        lats (np.ndarray): Latitudes of the actual charging stations.
        lons (np.ndarray): Longitudes of the actual charging stations.
        values (np.ndarray): The metric we are mapping (e.g., Utilization or Queue Size).
        grid_lats (np.ndarray): The latitude values for every cell in our target grid.
        grid_lons (np.ndarray): The longitude values for every cell in our target grid.
        k (int): How many nearby stations to consider (default is 1, the single closest).
        max_dist_km (float): The maximum distance to look for a station. If a grid 
            square is further away than this, it's marked as empty (0.0).

    Returns:
        np.ndarray: A 2D grid of interpolated values ready for plotting.
    """

    def to_cartesian(lat, lon):
        """
        Converts Latitude/Longitude into 3D XYZ coordinates (Cartesian).
        """
        R = 6371.0  # Earth's approximate radius in kilometers
        lat_rad, lon_rad = np.radians(lat), np.radians(lon)
        x = R * np.cos(lat_rad) * np.cos(lon_rad)
        y = R * np.cos(lat_rad) * np.sin(lon_rad)
        z = R * np.sin(lat_rad)
        return np.stack([x, y, z], axis=-1)

    #Transform stations and target grid into 3D space
    station_coords = to_cartesian(lats, lons)
    grid_coords = to_cartesian(grid_lats.flatten(), grid_lons.flatten())

    tree = KDTree(station_coords)
    indices = tree.query(grid_coords, k=k, distance_upper_bound=max_dist_km)

    # Map the station values to the grid. We add a '0.0' at the end of our values.
    # The KDTree returns an index pointing to this extra zero if no station is found within max_dist_km.
    padded_values = np.append(values, 0.0)
    raster_flat = padded_values[indices]

    # Reshape the flat list of results back into the 2D shape of our map
    return raster_flat.reshape(grid_lats.shape)