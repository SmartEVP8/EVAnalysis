from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import polars as pl


# ----------------------------
# Expected raw schemas
# ----------------------------
class Snap:
    STATION_ID = "StationId"
    UTILIZATION = "utilization"
    QUEUE = "total_queue_size"
    CANCELLATION = "cancellation_rate"
    DAY = "day"
    TIME = "time_of_day"


class Station:
    STATION_ID = "StationId"
    LAT = "Latitude"
    LON = "Longitude"


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class SnapshotFrame:
    snapshot_id: int
    data: pl.DataFrame

    def metric_arrays(self, metric_col: str):
        df = self.data.drop_nulls(subset=[metric_col])
        return (
            df[Station.LAT].to_numpy(),
            df[Station.LON].to_numpy(),
            df[metric_col].to_numpy(),
        )


@dataclass
class HeatmapDataset:
    snapshots: list[SnapshotFrame] = field(default_factory=list)

    def __len__(self):
        return len(self.snapshots)


# ----------------------------
# Loader
# ----------------------------
def load_heatmap_data(
    snapshots_path: Path,
    stations_path: Path,
) -> HeatmapDataset:

    snapshots_df = pl.read_parquet(snapshots_path)
    stations_df = pl.read_parquet(stations_path)

    # Build stable snapshot id
    snapshots_df = snapshots_df.with_columns(
        (pl.col(Snap.DAY).cast(pl.Int64) * 1_000_000
         + pl.col(Snap.TIME).cast(pl.Int64)).alias("snapshot_id")
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
            "Join failed: StationId mismatch between snapshots and stations file"
        )

    dataset = HeatmapDataset()

    snapshot_ids = (
        joined.select("snapshot_id")
        .unique()
        .sort("snapshot_id")
        .to_series()
        .to_list()
    )

    for sid in snapshot_ids:
        frame = joined.filter(pl.col("snapshot_id") == sid)
        dataset.snapshots.append(SnapshotFrame(sid, frame))

    return dataset