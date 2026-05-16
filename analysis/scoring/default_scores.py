from __future__ import annotations

import math
from dataclasses import dataclass, field

PATH_DEVIATION_BUCKETS: list[tuple[float, int]] = [
    (5,            0),
    (10,           0),
    (15,           2),
    (30,           6),
    (60,           12),
    (float("inf"), 15),
]

DELTA_ARRIVAL_BUCKETS: list[tuple[float, int]] = [
    (0,            0),
    (5,            1),
    (10,           2),
    (15,           3),
    (30,           6),
    (60,           10),
    (float("inf"), 15),
]

WAIT_TIME_BUCKETS: list[tuple[str, int]] = [
    ("p25",  1),
    ("p50",  3),
    ("p75",  5),
    ("p90",  6),
    ("p95",  10),
    ("p99",  30),
]

EXPECTED_WAIT_TIME_BUCKETS: list[tuple[str, int]] = [
    ("p25",  1),
    ("p50",  3),
    ("p75",  5),
    ("p90",  6),
    ("p95",  10),
    ("p99",  50),
]

PATH_DEVIATION_BUCKET_LABELS: list[str] = ["5", "10", "15", "30", "60", "60+"]
DELTA_ARRIVAL_BUCKET_LABELS: list[str] = ["0", "5", "10", "15", "30", "60", "60+"]
PERCENTILE_NAMES: list[str] = ["p25", "p50", "p75", "p90", "p95", "p99"]

WAIT_WEIGHTS: float = float(sum(w for _, w in WAIT_TIME_BUCKETS))
TOTAL_WAIT_WEIGHT: float = float(sum(w for _, w in EXPECTED_WAIT_TIME_BUCKETS))

EV_METRIC_WEIGHTS: dict[str, int] = {
    "path_deviation": 1,
    "delta_arrival": 2,
    "ev_wait_time": 3,
    "missed_deadline": 1,
}

STATION_METRIC_WEIGHTS: dict[str, int] = {
    "utilization": 1,
    "expected_wait_time": 3,
}

GROUP_WEIGHTS: dict[str, int] = {
    "ev": 1,
    "station": 1,
}

@dataclass(frozen=True)
class ScoringConfig:
    path_deviation_buckets: list[tuple[float, int]] = field(default_factory=lambda: PATH_DEVIATION_BUCKETS)
    delta_arrival_buckets: list[tuple[float, int]] = field(default_factory=lambda: DELTA_ARRIVAL_BUCKETS)
    wait_time_buckets: list[tuple[str, int]] = field(default_factory=lambda: WAIT_TIME_BUCKETS)
    expected_wait_time_buckets: list[tuple[str, int]] = field(default_factory=lambda: EXPECTED_WAIT_TIME_BUCKETS)
    ev_metric_weights: dict[str, int] = field(default_factory=lambda: EV_METRIC_WEIGHTS)
    station_metric_weights: dict[str, int] = field(default_factory=lambda: STATION_METRIC_WEIGHTS)
    group_weights: dict[str, int] = field(default_factory=lambda: GROUP_WEIGHTS)

DEFAULT_SCORING_CONFIG = ScoringConfig()


def bucket_labels(buckets: list[tuple[float, int]]) -> list[str]:
    labels: list[str] = []
    previous_upper = 0.0

    for upper_bound, _ in buckets:
        if math.isinf(upper_bound):
            labels.append(f"{int(previous_upper)}+")
        else:
            labels.append(str(int(upper_bound)) if upper_bound.is_integer() else str(upper_bound))
        previous_upper = upper_bound

    return labels

