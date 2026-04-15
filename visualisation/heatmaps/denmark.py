"""
Provides geospatial grid definitions and land-masking utilities for Denmark.
Handles the conversion between geographic coordinates (Lat/Lon) and 2D grid arrays.
"""

from dataclasses import dataclass
import numpy as np
import rasterio.features
import geopandas as gpd

"""
Geographic bounding box for Denmark found at: https://da.wikipedia.org/wiki/Fil:La2-demis-denmark.png

Importantly: These bounds include Bornholm, despite us not having it in the simulation.
This is done because NaturalEarth draws it anyways, even if LON_MAX is lowered to 13.20,
so we would have it regardless.
"""
LAT_MIN = 54.30
LAT_MAX = 57.80
LON_MIN =  7.90
LON_MAX = 15.20


@dataclass(frozen=True)
class DenmarkGrid:
    """
    Represents a rectangular geographic grid over Denmark.
    
    Calculates the necessary dimensions for a 2D array based on 
    a desired kilometer resolution, taking into account the Earth's curvature 
    at Denmark's latitude.

    Attributes:
        lat_min (float): Southern boundary latitude.
        lat_max (float): Northern boundary latitude.
        lon_min (float): Western boundary longitude.
        lon_max (float): Eastern boundary longitude.
        height (int): Number of rows in the resulting grid (Y-axis).
        width (int): Number of columns in the resulting grid (X-axis).
    """
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    height: int
    width: int

    @property
    def lat_grid(self) -> np.ndarray:
        """
        Generates a 2D array where each cell contains its corresponding latitude.
        Used for spatial broadcasting and calculations.
        """
        lats = np.linspace(self.lat_max, self.lat_min, self.height)
        return np.tile(lats[:, None], (1, self.width))

    @property
    def lon_grid(self) -> np.ndarray:
        """
        Generates a 2D array where each cell contains its corresponding longitude.
        Used for spatial broadcasting and calculations.
        """
        lons = np.linspace(self.lon_min, self.lon_max, self.width)
        return np.tile(lons[None, :], (self.height, 1))

    @classmethod
    def default(cls, resolution_km: float = 5.0) -> "DenmarkGrid":
        """
        Factory method to create a grid based on a kilometer resolution.

        Calculates height and width by approximating degrees to kilometers.
        Importantly: Longitudinal distance is scaled by cos(56 degrees) to account for 
        convergence towards the poles.

        Args:
            resolution_km (float): The size of each grid cell side in kilometers.
        """
        km_per_lat_deg = 111.0
        # Scale longitude by the cosine of the average latitude of Denmark (~56.0N)
        km_per_lon_deg = 111.0 * np.cos(np.radians(56.0))

        height = int((LAT_MAX - LAT_MIN) * km_per_lat_deg / resolution_km)
        width  = int((LON_MAX - LON_MIN) * km_per_lon_deg / resolution_km)

        return cls(
            lat_min=LAT_MIN, lat_max=LAT_MAX,
            lon_min=LON_MIN, lon_max=LON_MAX,
            height=height,   width=width,
        )


def load_denmark_boundary() -> gpd.GeoDataFrame:
    """
    Fetches official Danish border geometries from Natural Earth data.

    Returns:
        gpd.GeoDataFrame: A GeoPandas dataframe containing the Denmark polygon in WGS84.
    """
    url = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
    world = gpd.read_file(url)
    return world[world["ADMIN"] == "Denmark"].to_crs("EPSG:4326")


def build_land_mask(grid: DenmarkGrid) -> np.ndarray:
    """
    Creates a boolean mask where True represents land and False represents sea.
    This acts as a high-level wrapper for the rasterization process, providing 
    a fallback mechanism (all-land mask) if the external boundary data fails to load.

    Args:
        grid (DenmarkGrid): The grid dimensions to mask.

    Returns:
        np.ndarray: A boolean 2D array of shape (height, width).
    """
    try:
        return _land_mask_raster(grid)
    except Exception as e:
        import warnings
        warnings.warn(f"Land mask failed ({e}), falling back to all-land mask.")
        # Fallback: Treat the entire bounding box as land
        return np.ones((grid.height, grid.width), dtype=bool)


def _land_mask_raster(grid: DenmarkGrid) -> np.ndarray:
    """
    Internal logic to rasterize vector geometry into the grid.

    Uses rasterio to mark the Danish polygon onto a NumPy array based 
    on the grid's affine transform.
    """
    dk = load_denmark_boundary()
    dk_union = dk.geometry.union_all()

    # Define the mapping from geographic coordinates to pixel indices
    transform = rasterio.transform.from_bounds(
        grid.lon_min, grid.lat_min,
        grid.lon_max, grid.lat_max,
        grid.width, grid.height
    )
    
    # Use the map of Denmark to mark which grid squares are land (1) and which are sea (0).
    mask = rasterio.features.rasterize(
        [(dk_union, 1)],
        out_shape=(grid.height, grid.width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )

    # Flip the array because for whatever reason map is upside down
    return np.flipud(mask.astype(bool))