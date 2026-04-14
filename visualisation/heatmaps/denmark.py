"""
Denmark geographic constants and land-mask utilities.

The bounding box covers mainland Denmark plus the main islands.
Bornholm (~15°E) is excluded by the eastern bound – add it later if needed.

Coordinate order throughout this module: (lat, lon).
"""

from dataclasses import dataclass

import numpy as np
import rasterio.features
import geopandas as gpd

LAT_MIN = 54.50
LAT_MAX = 57.80
LON_MIN =  8.00
LON_MAX = 13.00


@dataclass(frozen=True)
class DenmarkGrid:
    """A regular lat/lon pixel grid covering Denmark."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    height: int
    width: int

    @property
    def lat_grid(self) -> np.ndarray:
        """(height, width) array of centre-pixel latitudes, high→low."""
        lats = np.linspace(self.lat_max, self.lat_min, self.height)
        return np.tile(lats[:, None], (1, self.width))

    @property
    def lon_grid(self) -> np.ndarray:
        """(height, width) array of centre-pixel longitudes, left→right."""
        lons = np.linspace(self.lon_min, self.lon_max, self.width)
        return np.tile(lons[None, :], (self.height, 1))

    @classmethod
    def default(cls, resolution_km: float = 5.0) -> "DenmarkGrid":
        """
        Create a grid whose pixel size approximates *resolution_km*.

        1° lat ≈ 111 km everywhere.
        1° lon ≈ 111 km × cos(lat) ≈ 63 km at Denmark's mean latitude (~56°).
        """
        km_per_lat_deg = 111.0
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
    Load a high-resolution Denmark boundary from Natural Earth 10m data.

    10m (1:10,000,000) is ~11x more detailed than the 110m dataset,
    giving proper coastline fidelity for the peninsula and islands.
    """
    url = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
    world = gpd.read_file(url)
    return world[world["ADMIN"] == "Denmark"].to_crs("EPSG:4326")


def build_land_mask(grid: DenmarkGrid) -> np.ndarray:
    """
    Fast rasterized land mask (NO shapely per-pixel operations).
    """
    try:
        return _land_mask_raster(grid)
    except Exception as e:
        import warnings
        warnings.warn(f"Land mask failed ({e}), falling back to all-land mask.")
        return np.ones((grid.height, grid.width), dtype=bool)


def _land_mask_raster(grid: DenmarkGrid) -> np.ndarray:
    dk = load_denmark_boundary()
    dk_union = dk.geometry.union_all()

    transform = rasterio.transform.from_bounds(
        grid.lon_min, grid.lat_min,
        grid.lon_max, grid.lat_max,
        grid.width, grid.height
    )

    mask = rasterio.features.rasterize(
        [(dk_union, 1)],
        out_shape=(grid.height, grid.width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )

    return np.flipud(mask.astype(bool))