"""
Module: station_scorer
Scores charging station performance across utilization and queue_size metrics.

Each metric is aggregated across time (default: max), then each percentile
column is scored relative to the fleet-wide maximum observed value.

Scoring convention:
- utilization : higher is better  -> score = v / max(v)
- queue_size : lower  is better  -> score = 1 - v / max(v)

The final station score is the mean of all individual percentile scores.
"""

from __future__ import annotations
import polars as pl

# Percentile groups
PERCENTILES = ["p25", "p50", "p75", "p90", "p95", "p99"]

# True means that higher score is better, False means lower score is better. Is used to figure out if we do 1-score or not.
STATION_METRICS: dict[str, bool] = {
    "utilization": True,
    "queue_size":  False,
}


def score_stations(
    station_snapshots: pl.DataFrame,
    time_aggregation: str,
) -> dict:
    aggregation = pl.Expr.max if time_aggregation == "max" else pl.Expr.mean
    aggregated: dict[str, float] = (
        station_snapshots
        .select([aggregation(pl.col(column)) for column in station_snapshots.columns if column not in ("weekday_name", "simtime_ms", "time_label")])
        .row(0, named=True)
    )

    metric_results: dict[str, dict] = {}
    all_scores: list[float] = []

    for metric, higher_is_better in STATION_METRICS.items():
        # Collect the aggregated value for each percentile
        percent_values: dict[str, float] = {}
        for percent in PERCENTILES:
            column = f"{metric}_{percent}"
            if column in aggregated:
                percent_values[percent] = float(aggregated[column])

        if not percent_values:
            continue

        maximum_reference = max(percent_values.values())

        percentile_scores: dict[str, float] = {}
        for percent, value in percent_values.items():
            if maximum_reference == 0.0:
                score = 1.0 if not higher_is_better else 0.0
            else:
                ratio = value / maximum_reference
                score = ratio if higher_is_better else 1.0 - ratio
            percentile_scores[percent] = round(score, 6)

        metric_score = sum(percentile_scores.values()) / len(percentile_scores)
        metric_results[metric] = {
            "higher_is_better": higher_is_better,
            "aggregated_values": {percentile_group: round(value, 6) for percentile_group, value in percent_values.items()},
            "percentile_scores": percentile_scores,
            "metric_score": round(metric_score, 6),
        }
        all_scores.extend(percentile_scores.values())

    aggregate = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return {
        "time_aggregation": time_aggregation,
        "per_metric": metric_results,
        "aggregate": round(aggregate, 6),
    }