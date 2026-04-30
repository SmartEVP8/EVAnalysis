"""
station_scorer.py
-----------------
Scores station-level simulation metrics per snapshot tick:
  - utilization            (per-tick max-normalised, averaged across stations)
  - expected_wait_time     (Gaussian decay on per-station expected wait, averaged)

Normalisation for utilization is per-tick: each station's utilization at tick T
is divided by the maximum utilization observed across all stations at tick T.

The run-wide aggregate for each metric is the mean of its per-tick scores.
The weighted_aggregate is the weighted mean of the two metric aggregates.

Entry point:
    scores = compute_station_scores(run_id, output_root)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import polars as pl


PERCENTILES = ["p25", "p50", "p75", "p90", "p95", "p99"]

METRIC_WEIGHTS: dict[str, int] = {
    "utilization":         1,
    "expected_wait_time":  3,
}


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------

def _wait_score(x: float) -> float:
    return math.exp(-((x / 45) ** 2))


def _score_utilization_tick(tick_df: pl.DataFrame) -> float:
    """
    Score utilization for a single tick.

    tick_df must have a 'utilization' column (one row per station).
    Each station's utilization is divided by the tick's max utilization,
    then all station scores are averaged.

    Returns 0.0 if all stations have zero utilization at this tick.
    """
    values = tick_df["utilization"].drop_nulls().to_list()
    if not values:
        return 0.0

    tick_max = max(values)
    if tick_max == 0.0:
        return 0.0

    return sum(v / tick_max for v in values) / len(values)


def _score_expected_wait_tick(tick_df: pl.DataFrame) -> float:
    """
    Score expected wait time for a single tick.

    tick_df must have a 'station_expected_wait_time' column (one row per station,
    value in minutes). Applies Gaussian decay to each station's expected wait,
    then averages across stations.

    Returns 0.0 if no data at this tick.

    NOTE: 'station_expected_wait_time' column is not yet present in
    station_snapshots.parquet — it will be added to StationSnapshotMetric soon.
    """
    values = tick_df["station_expected_wait_time"].drop_nulls().to_list()
    if not values:
        return 0.0
    return sum(_wait_score(x) for x in values) / len(values)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class StationScores:
    # Per-tick detail — one row per simtime_ms
    per_tick: pl.DataFrame

    # Run-wide aggregates (mean across ticks)
    utilization_aggregate: float
    expected_wait_time_aggregate: float
    weighted_aggregate: float

    number_of_stations: int

    def to_dict(self) -> dict:
        return {
            "per_metric": {
                "utilization": {
                    "higher_is_better": True,
                    "aggregate_score":  round(self.utilization_aggregate, 6),
                },
                "expected_wait_time": {
                    "higher_is_better": False,
                    "aggregate_score":  round(self.expected_wait_time_aggregate, 6),
                },
            },
            "number_of_stations": self.number_of_stations,
            "metric_weights":     METRIC_WEIGHTS,
            "weighted_aggregate": round(self.weighted_aggregate, 6),
        }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_station_scores(run_id: str, output_root: Path) -> StationScores:
    """
    Reads station_snapshots.parquet and returns StationScores with per-tick
    detail and run-wide aggregates.

    Parquet consumed
    analysis/station_snapshots.parquet
        StationId, simtime_ms, utilization, station_expected_wait_time (minutes)

    NOTE: station_expected_wait_time will be null for all rows until the column
    is added to StationSnapshotMetric. The scorer handles this currently by
    returning 0.0 for ticks where the column is entirely null.
    """
    path = output_root / run_id / "analysis" / "station_snapshots.parquet"
    snapshots = pl.read_parquet(path)

    # Add placeholder column if not yet present in the parquet
    if "station_expected_wait_time" not in snapshots.columns:
        snapshots = snapshots.with_columns(
            pl.lit(None).cast(pl.Float64).alias("station_expected_wait_time")
        )

    ticks = (
        snapshots.select("simtime_ms").unique().sort("simtime_ms")["simtime_ms"].to_list()
    )
    n_stations = snapshots["StationId"].n_unique()

    rows = []
    for tick in ticks:
        tick_df = snapshots.filter(pl.col("simtime_ms") == tick)

        util_score = _score_utilization_tick(tick_df)
        wait_score = _score_expected_wait_tick(tick_df)

        rows.append({
            "simtime_ms":            tick,
            "utilization_score":     util_score,
            "expected_wait_score":   wait_score,
        })

    per_tick = pl.DataFrame(rows)

    util_agg = per_tick["utilization_score"].mean()
    wait_agg = per_tick["expected_wait_score"].mean()

    weighted_aggregate = (
        METRIC_WEIGHTS["utilization"] * util_agg
        + METRIC_WEIGHTS["expected_wait_time"] * wait_agg
    ) / sum(METRIC_WEIGHTS.values())

    return StationScores(
        per_tick=per_tick,
        utilization_aggregate=util_agg,
        expected_wait_time_aggregate=wait_agg,
        weighted_aggregate=weighted_aggregate,
        number_of_stations=n_stations,
    )