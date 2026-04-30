"""
station_scorer.py
Scores station-level simulation metrics:
  - utilization (percentile-relative, higher is better)
  - wait_time (Gaussian decay on mean wait per station, lower is better)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import polars as pl


PERCENTILES = ["p25", "p50", "p75", "p90", "p95", "p99"]

METRIC_WEIGHTS: dict[str, int] = {
    "utilization": 1,
    "wait_time": 3,
}


def wait_score(x: float) -> float:
    return math.exp(-((x / 45) ** 2))


@dataclass
class StationScores:
    utilization_score: float
    wait_time_score: float
    number_of_stations: int

    utilization_percentile_scores: dict[str, float] = field(default_factory=dict)
    utilization_percentile_maxes: dict[str, float] = field(default_factory=dict)

    weighted_aggregate: float = field(init=False)

    def __post_init__(self):
        self.weighted_aggregate = (
            METRIC_WEIGHTS["utilization"] * self.utilization_score
            + METRIC_WEIGHTS["wait_time"] * self.wait_time_score
        ) / sum(METRIC_WEIGHTS.values())

    def to_dictionary(self) -> dict:
        return {
            "per_metric": {
                "utilization": {
                    "higher_is_better": True,
                    "percentile_maxes": {
                        key: round(value, 6)
                        for key, value in self.utilization_percentile_maxes.items()
                    },
                    "percentile_scores": {
                        key: round(value, 6)
                        for key, value in self.utilization_percentile_scores.items()
                    },
                    "metric_score": round(self.utilization_score, 6),
                },
                "wait_time": {
                    "higher_is_better": False,
                    "metric_score": round(self.wait_time_score, 6),
                },
            },
            "n_stations": self.number_of_stations,
            "metric_weights": METRIC_WEIGHTS,
            "weighted_aggregate": round(self.weighted_aggregate, 6),
        }



def score_utilization(station_percentiles: pl.DataFrame) -> tuple[float, dict, dict]:
    agg_exprs = [pl.col(p).max().alias(p) for p in PERCENTILES]
    agg_exprs.append(pl.col("max").max().alias("max"))

    per_station = station_percentiles.group_by("station_id").agg(agg_exprs)

    score_columns = PERCENTILES + ["max"]
    percentile_scores: dict[str, float] = {}
    percentile_maxes: dict[str, float] = {}

    for col in score_columns:
        global_max = per_station[col].max()
        percentile_maxes[col] = float(global_max) if global_max is not None else 0.0

        if global_max is None or global_max == 0.0:
            percentile_scores[col] = 0.0
            continue

        station_scores = (per_station[col] / global_max).to_list()
        percentile_scores[col] = sum(station_scores) / len(station_scores)

    metric_score = sum(percentile_scores.values()) / len(score_columns)
    return metric_score, percentile_scores, percentile_maxes



def score_wait_time(station_wait_times: pl.DataFrame) -> float:
    mean_per_station = (
        station_wait_times
        .group_by("station_id")
        .agg(pl.col("wait_time_minutes").mean().alias("mean_wait"))
        ["mean_wait"]
        .to_list()
    )

    if not mean_per_station:
        return 1.0

    station_scores = [wait_score(t) for t in mean_per_station]
    return sum(station_scores) / len(station_scores)


def compute_station_scores(
    *,
    station_percentiles: pl.DataFrame,
    station_wait_times: pl.DataFrame,
) -> StationScores:
    utilization_score, pct_scores, pct_maxes = score_utilization(station_percentiles)
    wait_time_score = score_wait_time(station_wait_times)

    n_stations = station_percentiles["station_id"].n_unique()

    return StationScores(
        utilization_score=utilization_score,
        wait_time_score=wait_time_score,
        number_of_stations=n_stations,
        utilization_percentile_scores=pct_scores,
        utilization_percentile_maxes=pct_maxes,
    )