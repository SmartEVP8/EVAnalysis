"""
Handles the loading and structuring of simulation data for geographic visualization.
It merges time-series snapshots with physical station coordinates to create a 
sequence of frames suitable for heatmap rendering.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import polars as pl


class Snap:
    """Standardized column names found in the processed snapshot files."""
    STATION_ID = "StationId"
    UTILIZATION = "utilization"
    QUEUE = "total_queue_size"
    CANCELLATION = "cancellation_rate"
    DAY = "day"
    TIME = "simtime_ms"


class Station:
    """Standardized column names found in the station location metadata."""
    STATION_ID = "StationId"
    LAT = "Latitude"
    LON = "Longitude"


@dataclass
class SnapshotFrame:
    """
    Represents a single 'slice' of time in the simulation.
    
    Contains all the charging station metrics (utilization, etc.) paired with 
    their physical GPS coordinates for that specific moment.
    """
    snapshot_id: int
    data: pl.DataFrame

    def metric_arrays(self, metric_col: str):
        """
        Extracts the data into a format that plotting libraries can understand.
        
        Returns:
            A tuple of (Latitudes, Longitudes, Values) as NumPy arrays.
        """
        df = self.data.drop_nulls(subset=[metric_col])
        return (
            df[Station.LAT].to_numpy(),
            df[Station.LON].to_numpy(),
            df[metric_col].to_numpy(),
        )


@dataclass
class HeatmapDataset:
    """
    A collection of all SnapshotFrames for a simulation run.
    """
    snapshots: list[SnapshotFrame] = field(default_factory=list)

    def __len__(self):
        return len(self.snapshots)


def load_heatmap_data(
    snapshots_path: Path,
    stations_path: Path,
) -> HeatmapDataset:
    """
    Reads the analysis files and prepares the data for the heatmap renderer.
    Then generates a unique snapshot_id by combining the Day and Time fields, 
    and finally merges the two datasets so that each data point is associated with a geographic location.
    Args:
        snapshots_path (Path): Path to the 'station_snapshots.parquet' file.
        stations_path (Path): Path to the 'stations_locations.parquet' file.

    Returns:
        HeatmapDataset: An organized collection of time-stamped location data.

    Raises:
        ValueError: If the StationIds in the two files don't match.
    """
    snapshots_df = pl.read_parquet(snapshots_path)
    stations_df = pl.read_parquet(stations_path)

    snapshots_df = snapshots_df.with_columns(
        (
            pl.col(Snap.DAY).cast(pl.Int64) * 1_000_000
            + pl.col(Snap.TIME).cast(pl.Int64)
        ).alias("snapshot_id")
    )

    snapshots_df = snapshots_df.select([
        "snapshot_id",
        Snap.STATION_ID,
        Snap.UTILIZATION,
        Snap.QUEUE,
        Snap.CANCELLATION,
    ])

    stations_df = stations_df.select([
        Station.STATION_ID,
        Station.LAT,
        Station.LON,
    ])

    joined = snapshots_df.join(stations_df, on=Snap.STATION_ID, how="inner")

    if joined.is_empty():
        raise ValueError(
            "Join failed: StationId mismatch between snapshots and stations file. "
            "Check if the station metadata matches the simulation run."
        )

    dataset = HeatmapDataset()

    for (sid,), frame in joined.group_by("snapshot_id", maintain_order=True):
        dataset.snapshots.append(SnapshotFrame(sid, frame))

    return dataset