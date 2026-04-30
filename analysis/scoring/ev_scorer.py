"""
ev_scorer.py
Scores EV-level simulation metrics per snapshot tick:
  - path_deviation   (time-bucket weighted, per tick)
  - delta_arrival    (time-bucket weighted, per tick)
  - ev_wait_time     (Gaussian decay averaged over EVs at that tick)
  - missed_deadline  (proportion of non-direct arrivals that missed, per tick)

The run-wide aggregate for each metric is the mean of its per-tick scores.
The weighted_aggregate is the weighted mean of the four metric aggregates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import polars as pl

# list of (buckets, weights)
PATH_DEVIATION_BUCKETS: list[tuple[float, int]] = [
    (5, 1),
    (10, 1),
    (15, 1),
    (30, 1),
    (60, 2),
    (float("inf"), 3),
]
PATH_DEVIATION_BUCKET_LABELS = ["5", "10", "15", "30", "60", "60+"]

# list of (buckets, weights)
DELTA_ARRIVAL_BUCKETS: list[tuple[float, int]] = [
    (0, 2),
    (5, 1),
    (10, 1),
    (15, 1),
    (30, 1),
    (60, 1),
    (float("inf"), 1),
]
DELTA_ARRIVAL_BUCKET_LABELS = ["0", "5", "10", "15", "30", "60", "60+"]

METRIC_WEIGHTS: dict[str, int] = {
    "path_deviation": 1,
    "delta_arrival": 1,
    "ev_wait_time": 3,
    "missed_deadline": 2,
}

# This is the log function
def wait_score(x: float) -> float:
    return math.exp(-((x / 45) ** 2))


def count_buckets(values: list[float | None], buckets: list[tuple[float, int]]) -> list[int]:
    counts = [0] * len(buckets)
    for value in values:
        if value is None:
            continue
        previous = float("-inf")
        for i, (upper, _) in enumerate(buckets):
            if upper == 0:
                if value == 0.0:
                    counts[i] += 1
                    break
                previous = 0.0
            elif previous < value <= upper:
                counts[i] += 1
                break
            else:
                previous = upper
    return counts


def bucket_score(entries: list[int], buckets: list[tuple[float, int]]) -> float:
    """
    score = sum(entries[i] * weight[i]) / (total_entries * sum(weights))

    Returns 0.0 if there are no entries in this tick (no arrivals → no score).
    Lower score = more entries in heavy buckets = worse outcome.
    """
    weights       = [w for _, w in buckets]
    total_entries = sum(entries)
    if total_entries == 0:
        return 0.0
    weighted_sum = sum(e * w for e, w in zip(entries, weights))
    return weighted_sum / (total_entries * sum(weights))


def wait_time_score(wait_minutes: list[float]) -> float:
    if not wait_minutes:
        return 0.0
    return sum(wait_score(x) for x in wait_minutes) / len(wait_minutes)


def missed_deadline_score(
    total: int, direct: int, missed: int
) -> tuple[float, float]:
    """Returns (result, score). Score = 1 - result."""
    ev_not_directly = total - direct
    if ev_not_directly <= 0:
        return 0.0, 1.0
    result = missed / ev_not_directly
    return result, 1.0 - result



@dataclass
class EVScores:
    per_snapshot: pl.DataFrame

    path_deviation_aggregate: float
    delta_arrival_aggregate: float
    ev_wait_time_aggregate: float
    missed_deadline_aggregate: float
    weighted_aggregate: float

    def to_dict(self) -> dict:
        return {
            "per_metric": {
                "path_deviation_minutes": {
                    "higher_is_better": False,
                    "bucket_labels":  PATH_DEVIATION_BUCKET_LABELS,
                    "bucket_weights": [w for _, w in PATH_DEVIATION_BUCKETS],
                    "aggregate_score": round(self.path_deviation_aggregate, 6),
                },
                "delta_arrival_minutes": {
                    "higher_is_better": False,
                    "bucket_labels":  DELTA_ARRIVAL_BUCKET_LABELS,
                    "bucket_weights": [w for _, w in DELTA_ARRIVAL_BUCKETS],
                    "aggregate_score": round(self.delta_arrival_aggregate, 6),
                },
                "ev_wait_time": {
                    "higher_is_better": False,
                    "aggregate_score": round(self.ev_wait_time_aggregate, 6),
                },
                "missed_deadline": {
                    "higher_is_better": False,
                    "aggregate_score": round(self.missed_deadline_aggregate, 6),
                },
            },
            "metric_weights":     METRIC_WEIGHTS,
            "weighted_aggregate": round(self.weighted_aggregate, 6),
        }


def _score_tick(
    arrival_tick: pl.DataFrame,
    wait_tick: pl.DataFrame,
) -> dict:
    path_deviation_entries = count_buckets(
        arrival_tick["path_deviation_minutes"].to_list(),
        PATH_DEVIATION_BUCKETS,
    )
    delta_arrival_entries = count_buckets(
        arrival_tick["delta_arrival_minutes"].to_list(),
        DELTA_ARRIVAL_BUCKETS,
    )

    path_deviation_score = bucket_score(path_deviation_entries, PATH_DEVIATION_BUCKETS)
    delta_arrival_score = bucket_score(delta_arrival_entries, DELTA_ARRIVAL_BUCKETS)

    wait_time_score = wait_time_score(wait_tick["wait_time_minutes"].to_list())

    total = len(arrival_tick)
    direct = int(arrival_tick["drive_directly"].sum())
    missed = int(arrival_tick["missed_deadline"].sum())
    proportion, missed_deadline_score = missed_deadline_score(total, direct, missed)

    return {
        "path_deviation_entries": path_deviation_entries,
        "delta_arrival_entries":  delta_arrival_entries,
        "path_deviation_score":   path_deviation_score,
        "delta_arrival_score":    delta_arrival_score,
        "ev_wait_time_score":     wait_time_score,
        "missed_deadline_score":  missed_deadline_score,
        "missed_proportion":      proportion,
        "total_arrivals":         total,
        "direct_drive_arrivals":  direct,
        "missed_deadlines":       missed,
    }


def compute_ev_scores(run_id: str, output_root: Path) -> EVScores:
    """
    Reads analysis parquets and returns EVScores with per-tick detail and
    run-wide aggregates.
    """
    base = output_root / run_id / "analysis"

    arrivals  = pl.read_parquet(base / "arrival_snapshots.parquet")
    wait_time_df   = pl.read_parquet(base / "wait_time_snapshots.parquet")

    snapshots = (
        arrivals.select("simtime_ms").unique().sort("simtime_ms")["simtime_ms"].to_list()
    )

    rows = []
    for snapshot in snapshots:
        arrival_tick = arrivals.filter(pl.col("simtime_ms") == snapshot)
        wait_tick = wait_time_df.filter(pl.col("simtime_ms") == snapshot)
        row = _score_tick(arrival_tick, wait_tick)
        row["simtime_ms"] = snapshot
        rows.append(row)

    per_snapshot = pl.DataFrame(rows).select([
        "simtime_ms",
        "path_deviation_score",
        "delta_arrival_score",
        "ev_wait_time_score",
        "missed_deadline_score",
        "missed_proportion",
        "total_arrivals",
        "direct_drive_arrivals",
        "missed_deadlines",
        "path_deviation_entries",
        "delta_arrival_entries",
    ])

    pathdeviation_aggregation = per_snapshot["path_deviation_score"].mean()
    delta_arrival_aggregation = per_snapshot["delta_arrival_score"].mean()
    wait_time_aggregation = per_snapshot["ev_wait_time_score"].mean()
    missed_deadline_aggregation = per_snapshot["missed_deadline_score"].mean()

    weighted_aggregate = (
        METRIC_WEIGHTS["path_deviation"]  * pathdeviation_aggregation
        + METRIC_WEIGHTS["delta_arrival"] * delta_arrival_aggregation
        + METRIC_WEIGHTS["ev_wait_time"]  * wait_time_aggregation
        + METRIC_WEIGHTS["missed_deadline"] * missed_deadline_aggregation
    ) / sum(METRIC_WEIGHTS.values())

    return EVScores(
        per_snapshot=per_snapshot,
        path_deviation_aggregate=pathdeviation_aggregation,
        delta_arrival_aggregate=delta_arrival_aggregation,
        ev_wait_time_aggregate=wait_time_aggregation,
        missed_deadline_aggregate=missed_deadline_aggregation,
        weighted_aggregate=weighted_aggregate,
    )