import polars as pl


def utilization(chargers: pl.DataFrame) -> pl.DataFrame:
    """
    Per-charger utilization aggregated across all snapshots.

    For each charger, computes:
      - MeanUtilization: average of snapshot utilization values
      - P50, P75, P90: percentiles of snapshot utilization values
      - SnapshotCount: number of snapshots observed
    """
    return (
        chargers
        .group_by("ChargerId", "StationId", "IsDual")
        .agg(
            pl.col("Utilization").mean().alias("MeanUtilization"),
            pl.col("Utilization").quantile(0.50).alias("P50"),
            pl.col("Utilization").quantile(0.75).alias("P75"),
            pl.col("Utilization").quantile(0.90).alias("P90"),
            pl.col("Utilization").count().alias("SnapshotCount"),
        )
        .sort("ChargerId")
    )


def queue_size(chargers: pl.DataFrame) -> pl.DataFrame:
    """
    Per-charger queue size aggregated across all snapshots.

    For each charger, computes:
      - P50, P75, P90: percentiles of snapshot queue sizes
      - MaxQueueSize: worst-case queue observed
    """
    return (
        chargers
        .group_by("ChargerId", "StationId", "IsDual")
        .agg(
            pl.col("QueueSize").quantile(0.50).alias("P50"),
            pl.col("QueueSize").quantile(0.75).alias("P75"),
            pl.col("QueueSize").quantile(0.90).alias("P90"),
            pl.col("QueueSize").max().alias("MaxQueueSize"),
        )
        .sort("ChargerId")
    )


def summary(chargers: pl.DataFrame) -> pl.DataFrame:
    """
    Combined per-charger summary: utilization and queue size.
    """
    util = utilization(chargers).rename({
        "P50": "Util_P50",
        "P75": "Util_P75",
        "P90": "Util_P90",
    })
    queue = queue_size(chargers).rename({
        "P50": "Queue_P50",
        "P75": "Queue_P75",
        "P90": "Queue_P90",
    }).drop("StationId", "IsDual")

    return (
        util
        .join(queue, on="ChargerId")
        .sort("ChargerId")
    )