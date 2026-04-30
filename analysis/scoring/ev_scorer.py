"""
ev_scorer.py
Scores EV-level simulation metrics:
  - path_deviation   (time-bucket weighted)
  - delta_arrival    (time-bucket weighted)
  - ev_wait_time     (Gaussian decay, per-EV then averaged)
  - missed_deadline  (proportion of non-direct arrivals that missed)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import polars as pl


PATH_DEVIATION_BUCKET_LABELS = ["5", "10", "15", "30", "60", "60+"]
PATH_DEVIATION_WEIGHTS = [1, 1, 1, 1, 2, 3]

DELTA_ARRIVAL_BUCKET_LABELS = ["0", "5", "10", "15", "30", "60", "60+"]
DELTA_ARRIVAL_WEIGHTS = [2, 1, 1, 1, 1, 1, 1]


def wait_score(x: float) -> float:
    """Gaussian decay: 1.0 at x=0, approaches 0 for large wait times (minutes)."""
    return math.exp(-((x / 45) ** 2))


@dataclass
class EVScores:
    path_deviation_entries: list[int]
    delta_arrival_entries: list[int]

    path_deviation_score: float = field(init=False)
    delta_arrival_score: float = field(init=False)
    ev_wait_time_score: float = field(init=False)
    missed_deadline_score: float = field(init=False)

    weighted_aggregate: float = field(init=False)

    total_arrivals: int = 0
    missed_proportion: float = 0.0

    METRIC_WEIGHTS: dict[str, int] = field(
        default_factory=lambda: {
            "path_deviation": 1,
            "delta_arrival": 1,
            "ev_wait_time": 3,
            "missed_deadline": 2,
        },
        repr=False,
    )

    def to_dictionary(self) -> dict:
        return {
            "per_metric": {
                "path_deviation_minutes": {
                    "higher_is_better": False,
                    "bucket_labels": PATH_DEVIATION_BUCKET_LABELS,
                    "bucket_weights": PATH_DEVIATION_WEIGHTS,
                    "bucket_entries": self.path_deviation_entries,
                    "metric_score": round(self.path_deviation_score, 6),
                },
                "delta_arrival_minutes": {
                    "higher_is_better": False,
                    "bucket_labels": DELTA_ARRIVAL_BUCKET_LABELS,
                    "bucket_weights": DELTA_ARRIVAL_WEIGHTS,
                    "bucket_entries": self.delta_arrival_entries,
                    "metric_score": round(self.delta_arrival_score, 6),
                },
                "ev_wait_time": {
                    "higher_is_better": False,
                    "metric_score": round(self.ev_wait_time_score, 6),
                },
                "missed_deadline": {
                    "higher_is_better": False,
                    "total_arrivals": self.total_arrivals,
                    "missed_proportion": round(self.missed_proportion, 6),
                    "metric_score": round(self.missed_deadline_score, 6),
                },
            },
            "metric_weights": self.METRIC_WEIGHTS,
            "weighted_aggregate": round(self.weighted_aggregate, 6),
        }



def bucket_score(entries: list[int], weights: list[int]) -> float:
    if len(entries) != len(weights):
        raise ValueError(
            f"entries length {len(entries)} != weights length {len(weights)}"
        )

    total_entries = sum(entries)
    if total_entries == 0:
        return 0.0

    weight_sum = sum(weights)
    weighted_sum = sum(percentile_entry * weight for percentile_entry, weight in zip(entries, weights))

    return weighted_sum / (total_entries * weight_sum)


def score_path_deviation(entries: list[int]) -> float:
    return bucket_score(entries, PATH_DEVIATION_WEIGHTS)


def score_delta_arrival(entries: list[int]) -> float:
    return bucket_score(entries, DELTA_ARRIVAL_WEIGHTS)


def score_ev_wait_time(wait_times: pl.Series | list[float]) -> float:
    if isinstance(wait_times, pl.Series):
        values = wait_times.to_list()
    else:
        values = list(wait_times)

    if not values:
        return 0.0

    total_score = sum(wait_score(i) for i in values)
    return total_score / len(values)


def score_missed_deadline(
    total_arrivals: int,
    direct_drive_arrivals: int,
    missed_deadlines: int,
) -> tuple[float, float]:
    evs_not_direct = total_arrivals - direct_drive_arrivals
    if evs_not_direct <= 0:
        return (0.0, 1.0)

    proportion = missed_deadlines / evs_not_direct
    return (proportion, 1.0 - proportion)


def compute_ev_scores(
    *,
    path_deviation_entries: list[int],
    delta_arrival_entries: list[int],
    ev_wait_time_series: pl.Series | list[float],
    total_arrivals: int,
    direct_drive_arrivals: int,
    missed_deadlines: int,
) -> EVScores:
    scores = EVScores(
        path_deviation_entries=path_deviation_entries,
        delta_arrival_entries=delta_arrival_entries,
    )

    scores.path_deviation_score = score_path_deviation(path_deviation_entries)
    scores.delta_arrival_score = score_delta_arrival(delta_arrival_entries)
    scores.ev_wait_time_score = score_ev_wait_time(ev_wait_time_series)

    proportion, deadline_score = score_missed_deadline(
        total_arrivals, direct_drive_arrivals, missed_deadlines
    )
    scores.missed_deadline_score = deadline_score
    scores.missed_proportion = proportion
    scores.total_arrivals = total_arrivals

    weights = scores.METRIC_WEIGHTS
    scores.weighted_aggregate = (
        weights["path_deviation"] * scores.path_deviation_score
        + weights["delta_arrival"] * scores.delta_arrival_score
        + weights["ev_wait_time"] * scores.ev_wait_time_score
        + weights["missed_deadline"] * scores.missed_deadline_score
    ) / sum(weights.values())

    return scores