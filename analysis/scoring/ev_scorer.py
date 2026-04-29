"""
Module: ev_scorer
Scores EV behaviour across path deviation, delta arrival time, and missed deadlines.

Scoring convention:
- path_deviation : lower is better -> score = 1 - v / max(v)
- delta_arrival : lower is better -> score = 1 - v / max(v)
- missed_deadline : lower is better -> score = 1 - mean(missed_deadline_pct / 100)

Negative path_deviation values (EV arrived faster than direct route) are clamped
to 0 before scoring.
"""

from __future__ import annotations
import polars as pl

PERCENTILES = ["p25", "p50", "p75", "p90", "p95"]

# True means higher score is better, False means lower score is better. Is used to either get the direct score, or do 1-score
EV_PERCENTILE_METRICS: dict[str, bool] = {
    "path_deviation_minutes": False,
    "delta_arrival_minutes":  False,
}


def score_evs(
    ev_percentiles: pl.DataFrame,
    time_aggregation: str = "max",
) -> dict:
    non_data_cols = {"weekday_name", "simtime_ms", "time_label",
                     "missed_deadline_pct", "missed_deadline_count", "total_arrivals"}
    data_cols = [c for c in ev_percentiles.columns if c not in non_data_cols]

    agg_fn = pl.Expr.max if time_aggregation == "max" else pl.Expr.mean
    aggregated: dict[str, float] = (
        ev_percentiles
        .select([agg_fn(pl.col(c)) for c in data_cols])
        .row(0, named=True)
    )

    metric_results: dict[str, dict] = {}
    all_scores: list[float] = []

    for metric, higher_is_better in EV_PERCENTILE_METRICS.items():
        pct_values: dict[str, float] = {}
        for pct in PERCENTILES:
            col = f"{metric}_{pct}"
            if col in aggregated:
                raw = float(aggregated[col])
                pct_values[pct] = max(0.0, raw)

        if not pct_values:
            continue

        ref_max = max(pct_values.values())

        percentile_scores: dict[str, float] = {}
        for pct, value in pct_values.items():
            if ref_max == 0.0:
                score = 1.0
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

    if "missed_deadline_pct" in ev_percentiles.columns and "total_arrivals" in ev_percentiles.columns:
        weighted = (
            ev_percentiles
            .filter(pl.col("total_arrivals") > 0)
            .select([
                (pl.col("missed_deadline_pct") / 100.0 * pl.col("total_arrivals")).alias("weighted_missed"),
                pl.col("total_arrivals"),
            ])
        )
        total_arrivals = weighted["total_arrivals"].sum()
        total_missed   = weighted["weighted_missed"].sum()

        proportion = float(total_missed / total_arrivals) if total_arrivals > 0 else 0.0
        md_score   = round(1.0 - proportion, 6)

        metric_results["missed_deadline"] = {
            "higher_is_better": False,
            "total_arrivals":   int(total_arrivals),
            "missed_proportion": round(proportion, 6),
            "metric_score":     md_score,
        }
        all_scores.append(md_score)

    aggregate = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return {
        "time_aggregation": time_aggregation,
        "per_metric":       metric_results,
        "aggregate":        round(aggregate, 6),
    }