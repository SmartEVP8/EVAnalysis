from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from analysis.scoring.scoring_config import ScoringConfig
from helpers.constants import PERCENTILES

import polars as pl

PERCENTILE_NAMES: list[str] = ["p25", "p50", "p75", "p90", "p95", "p99"]

def expected_wait_score(wait_time_column: str, wait_decay_minutes: float) -> pl.Expr:
    return (-((pl.col(wait_time_column) / wait_decay_minutes) ** 2)).exp()

def utilization_score(utilization_column: str) -> pl.Expr:
    return pl.col(utilization_column).clip(0.0, 1.0)

def compute_station_scores(run_id: str, output_root: Path, config: ScoringConfig) -> StationScores:
    snapshots_path = output_root / run_id / "percentiles" / "station" / "station_percentiles.parquet"
    snapshots = pl.read_parquet(snapshots_path)

    per_bucket = snapshots.with_columns([
        *[utilization_score(f"utilization_{name}").alias(f"utilization_score_{name}") for name in PERCENTILE_NAMES]
    ]).with_columns([
        pl.mean_horizontal([f"utilization_score_{name}" for name in PERCENTILE_NAMES])
          .alias("utilization_score")
    ]).with_columns([
        *[expected_wait_score(f"wait_time_{name}", config.wait_decay_minutes).alias(f"wait_score_{name}") for name in PERCENTILE_NAMES]
    ]).with_columns([
        pl.mean_horizontal([f"wait_score_{name}" for name in PERCENTILE_NAMES])
          .alias("expected_wait_score")
    ])

    return StationScores(
        per_bucket=per_bucket,
    )

@dataclass
class StationScores:
    per_bucket: pl.DataFrame
