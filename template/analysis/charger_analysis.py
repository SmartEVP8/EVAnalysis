import polars as pl


def utilization_percentiles(chargers: pl.DataFrame) -> pl.DataFrame:
    return (
        chargers
        .group_by("ChargerId", "StationId", "IsDual")
        .agg(
            pl.col("Utilization").mean().alias("Mean"),
            pl.col("Utilization").count().alias("SnapshotCount"),
        )
        .sort("ChargerId")
    )


def queue_size_percentiles(chargers: pl.DataFrame) -> pl.DataFrame:
    return (
        chargers
        .group_by("ChargerId", "StationId", "IsDual")
        .agg(
            pl.col("QueueSize").mean().alias("Mean"),
            pl.col("QueueSize").max().alias("Max"),
        )
        .sort("ChargerId")
    )


def activity_rate(chargers: pl.DataFrame) -> pl.DataFrame:
    """
    Fraction of snapshots where the charger delivered any energy (Utilization > 0).
    This is our proxy for charger activity since durationSeconds is not available.

    ActivityRate = snapshots_with_delivery / total_snapshots
    """
    return (
        chargers
        .group_by("ChargerId", "StationId", "IsDual")
        .agg(
            (pl.col("Utilization") > 0).sum().alias("ActiveSnapshots"),
            pl.col("Utilization").count().alias("TotalSnapshots"),
        )
        .with_columns(
            (pl.col("ActiveSnapshots") / pl.col("TotalSnapshots")).alias("ActivityRate")
        )
        .sort("ChargerId")
    )


def summary(chargers: pl.DataFrame) -> pl.DataFrame:
    """
    Combined per-charger summary: utilization percentiles, queue percentiles, activity rate.
    Useful for a single overview table.
    """
    util = utilization_percentiles(chargers).rename({
        "P50": "Util_P50", "P75": "Util_P75", "P90": "Util_P90", "Mean": "Util_Mean",
    }).drop("SnapshotCount")

    queue = queue_size_percentiles(chargers).rename({
        "P50": "Queue_P50", "P75": "Queue_P75", "P90": "Queue_P90",
        "Mean": "Queue_Mean", "Max": "Queue_Max",
    })

    act = activity_rate(chargers).select("ChargerId", "ActiveSnapshots", "TotalSnapshots", "ActivityRate")

    return (
        util
        .join(queue.drop("StationId", "IsDual"), on="ChargerId")
        .join(act, on="ChargerId")
        .sort("ChargerId")
    )