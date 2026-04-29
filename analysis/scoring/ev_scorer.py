"""
Module: ev_scorer
Scores EV behaviour across path deviation, delta arrival time, and missed deadlines.

Scoring convention:
- path_deviation : lower is better  -> score = 1 - grouping / max(grouping) [already minutes]
- delta_arrival : lower is better  -> score = 1 - grouping / max(grouping) [already minutes]
- missed_deadline : lower is better  -> score = 1 - mean(missed_deadline_pct / 100)

Negative path_deviation values (EV arrived faster than direct route) are clamped
to 0 before scoring — they represent a bonus that we treat as perfect.
"""

from __future__ import annotations
import polars as pl

# Percentile groups
PERCENTILES = ["p25", "p50", "p75", "p90", "p95", "p99"]

# True means that higher score is better, False means lower score is better. Is used to figure out if we do 1-score or not.
EV_PERCENTILE_METRICS: dict[str, bool] = {
    "path_deviation_minutes": False,
    "delta_arrival_minutes":  False,
}


def score_evs(
    ev_percentiles: pl.DataFrame,
    time_aggregation: str = "max", # "max" or "mean"
) -> dict:
    non_data_columns = {"weekday_name", "simtime_ms", "time_label",
                     "missed_deadline_pct", "missed_deadline_count", "total_arrivals"}
    data_columns = [column for column in ev_percentiles.columns if column not in non_data_columns]

    aggregation = pl.Expr.max if time_aggregation == "max" else pl.Expr.mean
    aggregated: dict[str, float] = (
        ev_percentiles
        .select([aggregation(pl.col(column)) for column in data_columns])
        .row(0, named=True)
    )

    metric_results: dict[str, dict] = {}
    all_scores: list[float] = []

    for metric, higher_is_better in EV_PERCENTILE_METRICS.items():
        percent_values: dict[str, float] = {}
        for percent in PERCENTILES:
            column = f"{metric}_{percent}"
            if column in aggregated:
                raw_value = float(aggregated[column])
                percent_values[percent] = max(0.0, raw_value)

        if not percent_values:
            continue

        maximum_reference = max(percent_values.values())

        percentile_scores: dict[str, float] = {}
        for percent, value in percent_values.items():
            if maximum_reference == 0.0:
                score = 1.0
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

    if "missed_deadline_pct" in ev_percentiles.columns and "total_arrivals" in ev_percentiles.columns:
        weighted_score = (
            ev_percentiles
            .filter(pl.col("total_arrivals") > 0)
            .select([
                (pl.col("missed_deadline_pct") / 100.0 * pl.col("total_arrivals")).alias("weighted_missed"),
                pl.col("total_arrivals"),
            ])
        )
        total_arrivals = weighted_score["total_arrivals"].sum()
        total_missed = weighted_score["weighted_missed"].sum()

        proportion = float(total_missed / total_arrivals) if total_arrivals > 0 else 0.0
        missed_deadline_score = round(1.0 - proportion, 6)

        metric_results["missed_deadline"] = {
            "higher_is_better": False,
            "total_arrivals": int(total_arrivals),
            "missed_proportion": round(proportion, 6),
            "metric_score": missed_deadline_score,
        }
        all_scores.append(missed_deadline_score)

    aggregate = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return {
        "time_aggregation": time_aggregation,
        "per_metric": metric_results,
        "aggregate": round(aggregate, 6),
    }