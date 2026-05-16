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
    (30, 8),
    (60, 15),
    (float("inf"), 30),
]

PATH_DEVIATION_BUCKET_LABELS: list[str] = ["5", "10", "15", "30", "60", "60+"]

# ─────────────────────────────────────────────────────────────────────────────
# Delta Arrival Penalty Buckets (minutes late)
# ─────────────────────────────────────────────────────────────────────────────
DELTA_ARRIVAL_BUCKETS: list[tuple[float, int]] = [
    (0, 0),
    (5, 2),
    (10, 10),
    (15, 25),
    (30, 50),
    (60, 100),
    (float("inf"), 250),
]

DELTA_ARRIVAL_BUCKET_LABELS: list[str] = ["0", "5", "10", "15", "30", "60", "60+"]

# ─────────────────────────────────────────────────────────────────────────────
# EV Wait Time Percentile Weights
# ─────────────────────────────────────────────────────────────────────────────
WAIT_TIME_BUCKETS: list[tuple[str, int]] = [
    ("p25", 2),
    ("p50", 5),
    ("p75", 10),
    ("p90", 20),
    ("p95", 50),
    ("p99", 100),
]

# ─────────────────────────────────────────────────────────────────────────────
# EV Metric Weights (how much each metric contributes to EV score)
# ─────────────────────────────────────────────────────────────────────────────
EV_METRIC_WEIGHTS: dict[str, int] = {
    "path_deviation": 2,
    "delta_arrival": 2,
    "ev_wait_time": 3,
    "missed_deadline": 1,
}

# ─────────────────────────────────────────────────────────────────────────────
# Station Expected Wait Time Percentile Weights
# ─────────────────────────────────────────────────────────────────────────────
EXPECTED_WAIT_TIME_BUCKETS: list[tuple[str, int]] = [
    ("p25", 2),
    ("p50", 5),
    ("p75", 10),
    ("p90", 20),
    ("p95", 50),
    ("p99", 100),
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