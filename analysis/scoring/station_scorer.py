"""
Module: station_scorer
Scores charging station performance across utilization and queue_size metrics.

Each metric is aggregated across time (default: max), then each percentile
column is scored relative to the fleet-wide maximum observed value.

Scoring convention:
- utilization : higher is better -> score = value / max(value)
- queue_size : lower  is better -> score = 1 - value / max(value)

The final station score is the mean of all individual percentile scores.
"""

from __future__ import annotations
import polars as pl

PERCENTILES = ["p25", "p50", "p75", "p90", "p95", "p99"]

# True means higher score is better, False means lower score is better. Is used to either get the direct score, or do 1-score
STATION_METRICS: dict[str, bool] = {
    "utilization": True,
    "queue_size": False,
}


def score_stations(
    station_snapshots: pl.DataFrame,
    time_aggregation: str = "max",  # "max" or "mean"
) -> dict:
    agg_fn = pl.Expr.max if time_aggregation == "max" else pl.Expr.mean
    aggregated: dict[str, float] = (
        station_snapshots
        .select([agg_fn(pl.col(c)) for c in station_snapshots.columns if c not in ("weekday_name", "simtime_ms", "time_label")])
        .row(0, named=True)
    )

    metric_results: dict[str, dict] = {}
    all_scores: list[float] = []

    for metric, higher_is_better in STATION_METRICS.items():
        pct_values: dict[str, float] = {}
        for pct in PERCENTILES:
            col = f"{metric}_{pct}"
            if col in aggregated:
                pct_values[pct] = float(aggregated[col])

        if not pct_values:
            continue

        ref_max = max(pct_values.values())

        percentile_scores: dict[str, float] = {}
        for pct, value in pct_values.items():
            if ref_max == 0.0:
                score = 1.0 if not higher_is_better else 0.0
            else:
                ratio = value / ref_max
                score = ratio if higher_is_better else 1.0 - ratio
            percentile_scores[pct] = round(score, 6)

        metric_score = sum(percentile_scores.values()) / len(percentile_scores)
        metric_results[metric] = {
            "higher_is_better":  higher_is_better,
            "aggregated_values": {k: round(v, 6) for k, v in pct_values.items()},
            "percentile_scores": percentile_scores,
            "metric_score":      round(metric_score, 6),
        }
        all_scores.extend(percentile_scores.values())

    aggregate = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return {
        "time_aggregation": time_aggregation,
        "per_metric": metric_results,
        "aggregate": round(aggregate, 6),
    }