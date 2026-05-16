from __future__ import annotations

"""Scoring configuration defaults.

Edit this file to adjust penalty buckets, metric weights, and other scoring
parameters. Changes here will affect all rescoring runs and new simulations
that use the scoring pipeline.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Path Deviation Penalty Buckets (distance in minutes or units)
# ─────────────────────────────────────────────────────────────────────────────
PATH_DEVIATION_BUCKETS: list[tuple[float, int]] = [
    (5, 0),
    (10, 0),
    (15, 2),
    (30, 6),
    (60, 12),
    (float("inf"), 15),
]

PATH_DEVIATION_BUCKET_LABELS: list[str] = ["5", "10", "15", "30", "60", "60+"]

# ─────────────────────────────────────────────────────────────────────────────
# Delta Arrival Penalty Buckets (minutes late)
# ─────────────────────────────────────────────────────────────────────────────
DELTA_ARRIVAL_BUCKETS: list[tuple[float, int]] = [
    (0, 0),
    (5, 1),
    (10, 2),
    (15, 3),
    (30, 6),
    (60, 10),
    (float("inf"), 15),
]

DELTA_ARRIVAL_BUCKET_LABELS: list[str] = ["0", "5", "10", "15", "30", "60", "60+"]

# ─────────────────────────────────────────────────────────────────────────────
# EV Wait Time Percentile Weights
# ─────────────────────────────────────────────────────────────────────────────
WAIT_TIME_BUCKETS: list[tuple[str, int]] = [
    ("p25", 1),
    ("p50", 3),
    ("p75", 5),
    ("p90", 6),
    ("p95", 10),
    ("p99", 30),
]

# ─────────────────────────────────────────────────────────────────────────────
# EV Metric Weights (how much each metric contributes to EV score)
# ─────────────────────────────────────────────────────────────────────────────
EV_METRIC_WEIGHTS: dict[str, int] = {
    "path_deviation": 0,
    "delta_arrival": 0,
    "ev_wait_time": 0,
    "missed_deadline": 1,
}

# ─────────────────────────────────────────────────────────────────────────────
# Station Expected Wait Time Percentile Weights
# ─────────────────────────────────────────────────────────────────────────────
EXPECTED_WAIT_TIME_BUCKETS: list[tuple[str, int]] = [
    ("p25", 1),
    ("p50", 3),
    ("p75", 5),
    ("p90", 6),
    ("p95", 10),
    ("p99", 50),
]

# ─────────────────────────────────────────────────────────────────────────────
# Station Metric Weights (how much each metric contributes to station score)
# ─────────────────────────────────────────────────────────────────────────────
STATION_METRIC_WEIGHTS: dict[str, int] = {
    "utilization": 1,
    "expected_wait_time": 3,
}

# ─────────────────────────────────────────────────────────────────────────────
# Group Weights (how much EV and station scores contribute to overall score)
# ─────────────────────────────────────────────────────────────────────────────
GROUP_WEIGHTS: dict[str, int] = {
    "ev": 1,
    "station": 1,
}

# ─────────────────────────────────────────────────────────────────────────────
# Scoring Decay Parameters
# ─────────────────────────────────────────────────────────────────────────────
# EV wait time decay: exp(-(x/WAIT_DECAY_MINUTES)²)
# At 45 minutes: score ≈ 0.37,  at 90 minutes: score ≈ 0.02
EV_WAIT_DECAY_MINUTES: float = 45.0

# Station expected wait time decay (same formula as EV)
STATION_WAIT_DECAY_MINUTES: float = 45.0

# ─────────────────────────────────────────────────────────────────────────────
# Warmup Period
# ─────────────────────────────────────────────────────────────────────────────
# Milliseconds to exclude from scoring after simulation start.
# Allows the system to settle before metrics are counted.
WARMUP_MS: int = (3 * 60 * 60 * 1000) - 1200000  # 3 hours minus 20 minutes to exclude the initial ramp-up and ramp-down periods