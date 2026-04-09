import polars as pl


def utilization(chargers: pl.DataFrame) -> pl.DataFrame:
    """
    Per-station utilization aggregated across all snapshots and chargers.

    For each snapshot, computes the mean utilization across all chargers at
    that station. Then aggregates those per-snapshot means into:
      - MeanUtilization: mean of per-snapshot means
      - P50, P75, P90: percentiles of per-snapshot means
    """
    return (
        chargers
        .group_by("StationId", "SimTime")
        .agg(pl.col("Utilization").mean().alias("SnapshotMeanUtilization"))
        .group_by("StationId")
        .agg(
            pl.col("SnapshotMeanUtilization").mean().alias("MeanUtilization"),
            pl.col("SnapshotMeanUtilization").quantile(0.50).alias("P50"),
            pl.col("SnapshotMeanUtilization").quantile(0.75).alias("P75"),
            pl.col("SnapshotMeanUtilization").quantile(0.90).alias("P90"),
        )
        .sort("StationId")
    )


def queue_size(stations: pl.DataFrame) -> pl.DataFrame:
    """
    Per-station average queue size across all snapshots.

    TotalQueueSize is already the sum of all charger queue sizes at the
    station per snapshot (computed by C#). This divides by TotalChargers
    to get the mean queue size per charger at that station, then averages
    across snapshots.
    """
    return (
        stations
        .with_columns(
            (pl.col("TotalQueueSize") / pl.col("TotalChargers")).alias("MeanChargerQueueSize")
        )
        .group_by("StationId")
        .agg(
            pl.col("MeanChargerQueueSize").mean().alias("MeanQueueSize"),
            pl.col("TotalQueueSize").mean().alias("MeanTotalQueueSize"),
            pl.col("TotalQueueSize").max().alias("MaxTotalQueueSize"),
        )
        .sort("StationId")
    )


def reservation_stats(stations: pl.DataFrame) -> pl.DataFrame:
    """
    Per-station reservation and cancellation totals across the full simulation.

    CancellationRate = TotalCancellations / TotalReservations.
    Stations with zero reservations get a rate of 0.
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


def charging_revenue(stations: pl.DataFrame) -> pl.DataFrame:
    """
    Per-station total charging revenue across the full simulation.

    For each snapshot: revenue = TotalDeliveredKWh x Price (DKK/kWh).
    Summed across all snapshots per station.
    """
    return (
        stations
        .with_columns(
            (pl.col("TotalDeliveredKWh") * pl.col("Price")).alias("SnapshotRevenue")
        )
        .group_by("StationId")
        .agg(
            pl.col("SnapshotRevenue").sum().alias("TotalRevenueDKK"),
            pl.col("TotalDeliveredKWh").sum().alias("TotalDeliveredKWh"),
        )
        .sort("StationId")
    )


def summary(stations: pl.DataFrame, chargers: pl.DataFrame) -> pl.DataFrame:
    """
    Combined per-station summary: utilization, queue size, reservations, revenue.
    """
    util = utilization(chargers).rename({
        "P50": "Util_P50",
        "P75": "Util_P75",
        "P90": "Util_P90",
    })
    queue = queue_size(stations)
    reservations = reservation_stats(stations)
    revenue = charging_revenue(stations)

    return (
        util
        .join(queue, on="StationId")
        .join(reservations, on="StationId")
        .join(revenue, on="StationId")
        .sort("StationId")
    )