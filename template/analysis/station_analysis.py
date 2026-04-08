import polars as pl


def utilization_percentiles(chargers: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregates charger-level utilization up to station level by computing the
    mean utilization across all chargers at the station for each snapshot,
    then computing percentiles of those per-snapshot means.
    """
    return (
        chargers
        .group_by("StationId", "SimTime")
        .agg(pl.col("Utilization").mean().alias("MeanChargerUtilization"))
        .group_by("StationId")
        .agg(
            pl.col("MeanChargerUtilization").quantile(0.50).alias("P50"),
            pl.col("MeanChargerUtilization").quantile(0.75).alias("P75"),
            pl.col("MeanChargerUtilization").quantile(0.90).alias("P90"),
            pl.col("MeanChargerUtilization").mean().alias("Mean"),
        )
        .sort("StationId")
    )


def queue_size_percentiles(stations: pl.DataFrame) -> pl.DataFrame:
    """
    Uses TotalQueueSize from StationSnapshotMetric, which is already the
    sum of all charger queue sizes at that station (computed by SmartEV).
    """
    return (
        stations
        .group_by("StationId")
        .agg(
            pl.col("TotalQueueSize").quantile(0.50).alias("P50"),
            pl.col("TotalQueueSize").quantile(0.75).alias("P75"),
            pl.col("TotalQueueSize").quantile(0.90).alias("P90"),
            pl.col("TotalQueueSize").mean().alias("Mean"),
            pl.col("TotalQueueSize").max().alias("Max"),
        )
        .sort("StationId")
    )


def active_chargers_per_snapshot(chargers: pl.DataFrame) -> pl.DataFrame:
    """
    Per station per snapshot: how many chargers were active (Utilization > 0)
    and what fraction of total chargers that represents.
    """
    per_snapshot = (
        chargers
        .group_by("StationId", "SimTime")
        .agg(
            (pl.col("Utilization") > 0).sum().alias("ActiveChargers"),
            pl.col("ChargerId").count().alias("TotalChargers"),
        )
        .with_columns(
            (pl.col("ActiveChargers") / pl.col("TotalChargers")).alias("ActiveFraction")
        )
    )

    return (
        per_snapshot
        .group_by("StationId")
        .agg(
            pl.col("ActiveChargers").quantile(0.50).alias("ActiveCount_P50"),
            pl.col("ActiveChargers").quantile(0.75).alias("ActiveCount_P75"),
            pl.col("ActiveChargers").quantile(0.90).alias("ActiveCount_P90"),
            pl.col("ActiveFraction").quantile(0.50).alias("ActiveFraction_P50"),
            pl.col("ActiveFraction").quantile(0.75).alias("ActiveFraction_P75"),
            pl.col("ActiveFraction").quantile(0.90).alias("ActiveFraction_P90"),
            pl.col("TotalChargers").first().alias("TotalChargers"),
        )
        .sort("StationId")
    )


def reservation_stats(stations: pl.DataFrame) -> pl.DataFrame:
    """
    Total reservations, total cancellations, and cancellation rate per station
    across the full simulation.

    CancellationRate = totalCancelled / totalRequested.
    """
    return (
        stations
        .group_by("StationId")
        .agg(
            pl.col("Reservations").sum().alias("TotalReservations"),
            pl.col("Cancellations").sum().alias("TotalCancellations"),
        )
        .with_columns(
            pl.when(pl.col("TotalReservations") > 0)
            .then(pl.col("TotalCancellations") / pl.col("TotalReservations"))
            .otherwise(0.0)
            .alias("CancellationRate")
        )
        .sort("StationId")
    )


def price_stats(stations: pl.DataFrame) -> pl.DataFrame:
    """
    Mean, min, and max energy price per station across all snapshots.
    Price varies over sim time as energy prices fluctuate.
    """
    return (
        stations
        .group_by("StationId")
        .agg(
            pl.col("Price").mean().alias("MeanPrice"),
            pl.col("Price").min().alias("MinPrice"),
            pl.col("Price").max().alias("MaxPrice"),
        )
        .sort("StationId")
    )


def summary(stations: pl.DataFrame, chargers: pl.DataFrame) -> pl.DataFrame:
    """
    Combined per-station summary across all metrics.
    Joins charger-derived metrics with station-level metrics.
    """
    util = utilization_percentiles(chargers).rename({
        "P50": "Util_P50", "P75": "Util_P75", "P90": "Util_P90", "Mean": "Util_Mean",
    })
    queue = queue_size_percentiles(stations).rename({
        "P50": "Queue_P50", "P75": "Queue_P75", "P90": "Queue_P90",
        "Mean": "Queue_Mean", "Max": "Queue_Max",
    })
    activity = active_chargers_per_snapshot(chargers)
    reservations = reservation_stats(stations)
    price = price_stats(stations)

    return (
        util
        .join(queue, on="StationId")
        .join(activity, on="StationId")
        .join(reservations, on="StationId")
        .join(price, on="StationId")
        .sort("StationId")
    )