"""
Module: ev_scorer
Computes penalty scores for EV behaviour across a simulation run.

Lower scores are better. Scores are unbounded above zero.

Only EVs that did NOT drive directly to their destination are considered
(drive_directly == false), since direct-drive EVs never interact with
the charging infrastructure and skew the metrics.

Penalty logic:
  - Path Deviation:    mean + p90  of path_deviation_minutes
  - Wait Time:         mean + p90  of delta_arrival_minutes
  - Missed Deadline:   mean(missed_deadline) → gives a rate, e.g. 0.12 = 12%
"""

import polars as pl


def score_evs(
    arrival_snapshots: pl.DataFrame,
    arrival_percentiles: pl.DataFrame,
) -> dict:
    """
    Computes EV-level penalty scores from arrival snapshot and percentile data.

    Args:
        arrival_snapshots:   DataFrame from arrival_snapshots.parquet
        arrival_percentiles: DataFrame from arrival_percentiles.parquet

    Returns:
        A dict with keys: path_deviation_score, wait_time_score, missed_deadline_score
    """

    # Filter to only EVs that went through the charging infrastructure
    routed = arrival_snapshots.filter(pl.col("drive_directly") == False)

    if routed.is_empty():
        return {
            "path_deviation_score":  None,
            "wait_time_score":       None,
            "missed_deadline_score": None,
        }

    # --- Path Deviation ---
    # Higher deviation = more detour = higher penalty
    path_dev_mean = routed["path_deviation_minutes"].mean()
    path_dev_p90  = arrival_percentiles["path_deviation_minutes_p90"].mean()

    path_deviation_score = path_dev_mean + path_dev_p90

    # --- Wait Time (delta arrival) ---
    # Positive delta = arrived later than expected = higher penalty
    wait_mean = routed["delta_arrival_minutes"].mean()
    wait_p90  = arrival_percentiles["delta_arrival_minutes_p90"].mean()

    wait_time_score = wait_mean + wait_p90

    # --- Missed Deadline ---
    # Cast bool → int (True=1, False=0), then take mean to get rate
    missed_deadline_score = routed["missed_deadline"].cast(pl.Int32).mean()

    return {
        "path_deviation_score":  round(path_deviation_score, 6),
        "wait_time_score":       round(wait_time_score, 6),
        "missed_deadline_score": round(missed_deadline_score, 6),
    }