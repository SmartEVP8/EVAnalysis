"""
Module: station_scorer
Computes penalty scores for charging station behaviour across a simulation run.

Lower scores are better. Scores are unbounded above zero.

Penalty logic:
  - Utilization:  (1 - mean) + (1 - p25)   → penalises idle capacity
  - Queue Size:   mean + p90                → penalises congestion (typical + tail)
  - Wait Time:    mean + p90                → penalises time spent waiting for a charger
                  (zero-wait rows included; milliseconds converted to minutes)
"""

import polars as pl

_MS_TO_MINUTES = 1 / 60_000


def score_stations(
    station_snapshots: pl.DataFrame,
    station_percentiles: pl.DataFrame,
    wait_time_metrics: pl.DataFrame,
) -> dict:
    """
    Computes station-level penalty scores from snapshot, percentile, and wait time data.

    Args:
        station_snapshots:   DataFrame from station_snapshots.parquet
        station_percentiles: DataFrame from station_percentiles.parquet
        wait_time_metrics:   DataFrame from WaitTimeInQueueMetric.parquet

    Returns:
        A dict with keys: utilization_score, queue_size_score, wait_time_score
    """

    # --- Utilization ---
    # Lower utilization = more idle capacity = higher penalty
    utilization_mean = station_snapshots["utilization"].mean()
    utilization_p25  = station_percentiles["utilization_p25"].mean()

    utilization_score = (1.0 - utilization_mean) + (1.0 - utilization_p25)

    # --- Queue Size ---
    # Higher queue = more congestion = higher penalty
    queue_mean = station_snapshots["total_queue_size"].mean()
    queue_p90  = station_percentiles["queue_size_p90"].mean()

    queue_size_score = queue_mean + queue_p90

    # --- Wait Time ---
    # Convert milliseconds → minutes, then penalise typical + tail behaviour
    # Zero-wait rows (EV walked straight onto a charger) are included
    wait_minutes = wait_time_metrics["WaitTimeInQueue"] * _MS_TO_MINUTES

    wait_mean = wait_minutes.mean()
    wait_p90  = wait_minutes.quantile(0.90, interpolation="nearest")

    wait_time_score = wait_mean + wait_p90

    return {
        "utilization_score": round(utilization_score, 6),
        "queue_size_score":  round(queue_size_score, 6),
        "wait_time_score":   round(wait_time_score, 6),
    }