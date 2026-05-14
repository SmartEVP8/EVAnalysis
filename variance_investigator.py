"""
variance_investigator.py

Scores every run found under OUTPUT_ROOT/runs/ against a set of scoring
configurations, then prints a variance report so you can see whether the
scoring system meaningfully differentiates between simulation runs.

Usage
-----
    python variance_investigator.py

Each "scoring configuration" is a dict that can override any combination of:
    - EV_METRIC_WEIGHTS        (dict[str, int])
    - STATION_METRIC_WEIGHTS   (dict[str, int])
    - GROUP_WEIGHTS            (dict[str, int])

Decay constants (EV_WAIT_DECAY_MINUTES / ST_WAIT_DECAY_MINUTES) are never
overridden — they stay at whatever value is set in the scorer modules.

Add or remove entries in SCORING_CONFIGS to explore the space.
"""

from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

# ── locate the project root (the directory that contains this file) ──────────
ROOT = Path(__file__).parent
OUTPUT_ROOT = ROOT / "runs"   # adjust if your runs live elsewhere

# ── import the modules we're going to monkeypatch ───────────────────────────
import analysis.scoring.ev_scorer as ev_mod
import analysis.scoring.station_scorer as st_mod
import analysis.scoring.simulation_scorer as sim_mod

from helpers.variance_configs import SCORING_CONFIGS

# ────────────────────────────────────────────────────────────────────────────
# Monkeypatching helpers
# ────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def apply_config(cfg: dict[str, Any]):
    """
    Temporarily overrides module-level weight constants in the three scorer
    modules, runs the body of the `with` block, then restores the originals.

    Decay constants are intentionally never touched here.
    """
    # save originals
    orig = {
        "ev_EV_METRIC_WEIGHTS":      copy.deepcopy(ev_mod.EV_METRIC_WEIGHTS),
        "st_STATION_METRIC_WEIGHTS": copy.deepcopy(st_mod.STATION_METRIC_WEIGHTS),
        "sim_GROUP_WEIGHTS":         copy.deepcopy(sim_mod.GROUP_WEIGHTS),
        "sim_TOTAL_GROUP_WEIGHT":    sim_mod.TOTAL_GROUP_WEIGHT,
    }

    try:
        if "EV_METRIC_WEIGHTS" in cfg:
            ev_mod.EV_METRIC_WEIGHTS.clear()
            ev_mod.EV_METRIC_WEIGHTS.update(cfg["EV_METRIC_WEIGHTS"])

        if "STATION_METRIC_WEIGHTS" in cfg:
            st_mod.STATION_METRIC_WEIGHTS.clear()
            st_mod.STATION_METRIC_WEIGHTS.update(cfg["STATION_METRIC_WEIGHTS"])

        if "GROUP_WEIGHTS" in cfg:
            sim_mod.GROUP_WEIGHTS.clear()
            sim_mod.GROUP_WEIGHTS.update(cfg["GROUP_WEIGHTS"])
            sim_mod.TOTAL_GROUP_WEIGHT = sum(sim_mod.GROUP_WEIGHTS.values())

        yield

    finally:
        ev_mod.EV_METRIC_WEIGHTS.clear()
        ev_mod.EV_METRIC_WEIGHTS.update(orig["ev_EV_METRIC_WEIGHTS"])

        st_mod.STATION_METRIC_WEIGHTS.clear()
        st_mod.STATION_METRIC_WEIGHTS.update(orig["st_STATION_METRIC_WEIGHTS"])

        sim_mod.GROUP_WEIGHTS.clear()
        sim_mod.GROUP_WEIGHTS.update(orig["sim_GROUP_WEIGHTS"])
        sim_mod.TOTAL_GROUP_WEIGHT = orig["sim_TOTAL_GROUP_WEIGHT"]


# ────────────────────────────────────────────────────────────────────────────
# Output paths
# ────────────────────────────────────────────────────────────────────────────

INVESTIGATIONS_ROOT = ROOT / "investigations"


def investigation_dir(run_id: str, config_name: str) -> Path:
    return INVESTIGATIONS_ROOT / run_id / config_name


# ────────────────────────────────────────────────────────────────────────────
# Data collection
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    run_id: str
    config_name: str
    score: sim_mod.SimulationScore | None = None
    error: str | None = None


def score_run(run_id: str, cfg: dict[str, Any]) -> RunResult:
    """
    Scores a single run under a single config.

    Returns a RunResult holding the full SimulationScore object so callers
    can write whatever outputs they need from it.
    """
    try:
        with apply_config(cfg):
            ev_scores      = sim_mod.compute_ev_scores(run_id, OUTPUT_ROOT)
            station_scores = sim_mod.compute_station_scores(run_id, OUTPUT_ROOT)
            score = sim_mod.SimulationScore(
                run_id=run_id,
                source_path=str(OUTPUT_ROOT / run_id),
                ev_scores=ev_scores,
                station_scores=station_scores,
            )
        return RunResult(run_id=run_id, config_name=cfg["name"], score=score)
    except Exception as exc:  # noqa: BLE001
        print(f"  [!] {run_id} / {cfg['name']}: {exc}")
        return RunResult(run_id=run_id, config_name=cfg["name"], error=str(exc))


# ────────────────────────────────────────────────────────────────────────────
# Per-run output writing
# ────────────────────────────────────────────────────────────────────────────

def write_run_outputs(result: RunResult) -> None:
    """
    Writes simulation_score.json and simulation_score.parquet for one
    (run_id, config) pair under investigations/{run_id}/{config_name}/.
    """
    if result.score is None:
        return

    out_dir = investigation_dir(result.run_id, result.config_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    result.score.write_json(out_dir / "simulation_score.json")
    result.score.write_parquet(out_dir / "simulation_score.parquet")


# ────────────────────────────────────────────────────────────────────────────
# Comparison parquet
# ────────────────────────────────────────────────────────────────────────────

METRIC_COLS = [
    "overall_aggregate",
    "ev_weighted_aggregate",
    "station_weighted_aggregate",
    "path_deviation_aggregate",
    "delta_arrival_aggregate",
    "ev_wait_time_aggregate",
    "missed_deadline_aggregate",
    "utilization_aggregate",
    "expected_wait_aggregate",
]


def result_to_flat_row(result: RunResult, cfg: dict[str, Any]) -> dict:
    """
    Flattens a RunResult into a single dict row for the comparison parquet.
    Mirrors the structure of simulation_score.json so nothing is lost.
    """
    ev_weights  = cfg.get("EV_METRIC_WEIGHTS", {})
    st_weights  = cfg.get("STATION_METRIC_WEIGHTS", {})
    grp_weights = cfg.get("GROUP_WEIGHTS", {})

    row: dict = {
        "run_id": result.run_id,
        "config": result.config_name,
        "error":  result.error or "",
        # Config parameters (handy for grouping/filtering in analysis tools)
        "cfg_ev_weight_path_deviation":  ev_weights.get("path_deviation"),
        "cfg_ev_weight_delta_arrival":   ev_weights.get("delta_arrival"),
        "cfg_ev_weight_ev_wait_time":    ev_weights.get("ev_wait_time"),
        "cfg_ev_weight_missed_deadline": ev_weights.get("missed_deadline"),
        "cfg_st_weight_utilization":     st_weights.get("utilization"),
        "cfg_st_weight_expected_wait":   st_weights.get("expected_wait_time"),
        "cfg_group_weight_ev":           grp_weights.get("ev"),
        "cfg_group_weight_station":      grp_weights.get("station"),
    }

    if result.score is None:
        for col in METRIC_COLS:
            row[col] = None
        return row

    s = result.score
    row.update({
        "overall_aggregate":          s.overall_aggregate,
        "ev_weighted_aggregate":      s.ev_weighted_aggregate,
        "path_deviation_aggregate":   s.path_deviation_aggregate,
        "delta_arrival_aggregate":    s.delta_arrival_aggregate,
        "ev_wait_time_aggregate":     s.ev_wait_time_aggregate,
        "missed_deadline_aggregate":  s.missed_deadline_aggregate,
        "station_weighted_aggregate": s.station_weighted_aggregate,
        "utilization_aggregate":      s.utilization_aggregate,
        "expected_wait_aggregate":    s.expected_wait_aggregate,
    })
    return row


def build_comparison_df(results: list[RunResult], configs: list[dict[str, Any]]) -> pl.DataFrame:
    """One row per (run_id, config) with all aggregates and config params."""
    cfg_by_name = {c["name"]: c for c in configs}
    return pl.DataFrame([
        result_to_flat_row(r, cfg_by_name[r.config_name])
        for r in results
    ])


def write_comparison_parquet(df: pl.DataFrame) -> Path:
    out_path = INVESTIGATIONS_ROOT / "comparison.parquet"
    INVESTIGATIONS_ROOT.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    print(f"[Investigator] Wrote {out_path}")
    return out_path


# ────────────────────────────────────────────────────────────────────────────
# Variance parquet
# ────────────────────────────────────────────────────────────────────────────

def build_variance_df(df: pl.DataFrame) -> pl.DataFrame:
    """
    Produces a tidy table with one row per (run_id, config) containing:
        run_id            – the simulation run
        config            – the scoring configuration name
        overall_aggregate – the run's overall score under that config
        delta_from_baseline – difference vs the same run's baseline score
                              (positive = scored higher than baseline)

    Rows where either the config score or the baseline score is null are
    kept but will have a null delta.
    """
    baseline = (
        df.filter(pl.col("config") == "baseline")
        .select(["run_id", pl.col("overall_aggregate").alias("baseline_aggregate")])
    )

    return (
        df.select(["run_id", "config", "overall_aggregate"])
        .join(baseline, on="run_id", how="left")
        .with_columns(
            (pl.col("overall_aggregate") - pl.col("baseline_aggregate"))
            .alias("delta_from_baseline")
        )
        .sort(["run_id", "config"])
    )


def write_variance_parquet(df: pl.DataFrame) -> Path:
    out_path = INVESTIGATIONS_ROOT / "variance.parquet"
    INVESTIGATIONS_ROOT.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    print(f"[Investigator] Wrote {out_path}")
    return out_path


# ────────────────────────────────────────────────────────────────────────────
# Console report
# ────────────────────────────────────────────────────────────────────────────

def print_wide_table(df: pl.DataFrame, float_cols: list[str]) -> None:
    formatted = df.with_columns([pl.col(c).round(4) for c in float_cols if c in df.columns])
    print(formatted)


def print_variance_report(df: pl.DataFrame) -> None:
    """
    For each scoring config show cross-run variance on overall_aggregate,
    a full per-metric breakdown, and a ranking-stability table.
    """
    configs = df["config"].unique().sort().to_list()
    runs    = df["run_id"].unique().sort().to_list()

    sep = "─" * 80
    print("\n" + sep)
    print("  VARIANCE INVESTIGATION REPORT")
    print(sep)
    print(f"  Runs found : {', '.join(runs)}")
    print(f"  Configs    : {len(configs)}")
    print(sep)

    # ── cross-run variance per config ────────────────────────────────────
    print("\n── Per-Config Cross-Run Variance (overall_aggregate) ──────────────────────\n")
    summary_rows = []
    for cfg_name in configs:
        scores = df.filter(pl.col("config") == cfg_name)["overall_aggregate"].drop_nulls()
        lo  = scores.min() or 0.0
        hi  = scores.max() or 0.0
        summary_rows.append({
            "config": cfg_name,
            "min":    round(lo, 4),
            "max":    round(hi, 4),
            "range":  round(hi - lo, 4),
            "mean":   round(scores.mean() or 0.0, 4),
            "std":    round(scores.std()  or 0.0, 4),
        })

    print(
        pl.DataFrame(summary_rows).sort("range", descending=True)
    )

    # ── detailed scores per config ────────────────────────────────────────
    display_cols = ["run_id"] + METRIC_COLS
    print("\n── Detailed Scores Per Config ──────────────────────────────────────────────")
    for cfg_name in configs:
        sub = df.filter(pl.col("config") == cfg_name).select(
            [c for c in display_cols if c in df.columns]
        )
        print(f"\n  Config: {cfg_name}")
        print_wide_table(sub, METRIC_COLS)

    # ── ranking stability ─────────────────────────────────────────────────
    print("\n── Ranking Stability Across Configs ────────────────────────────────────────")
    print("  (rank 1 = best overall_aggregate)\n")

    rank_rows: dict[str, dict[str, int]] = {r: {} for r in runs}
    for cfg_name in configs:
        ordered = (
            df.filter(pl.col("config") == cfg_name)
            .sort("overall_aggregate", descending=True)["run_id"]
            .to_list()
        )
        for rank, run_id in enumerate(ordered, start=1):
            rank_rows[run_id][cfg_name] = rank

    print(
        pl.DataFrame([{"run_id": rid, **ranks} for rid, ranks in rank_rows.items()])
        .sort("run_id")
    )

    print("\n" + sep)
    print("TIP: configs with high 'range' drive the most score separation.")
    print("TIP: stable rankings across configs → weights are not the bottleneck.")
    print(sep + "\n")

# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    run_dirs = sorted(p for p in OUTPUT_ROOT.iterdir() if p.is_dir())

    if not run_dirs:
        print(f"No run directories found under {OUTPUT_ROOT}. Exiting.")
        return

    run_ids = [p.name for p in run_dirs]
    print(f"Found {len(run_ids)} run(s): {', '.join(run_ids)}")
    print(f"Scoring against {len(SCORING_CONFIGS)} config(s)...\n")

    results: list[RunResult] = []
    for run_id in run_ids:
        for cfg in SCORING_CONFIGS:
            print(f"  Scoring {run_id} / '{cfg['name']}'...")
            result = score_run(run_id, cfg)
            results.append(result)
            write_run_outputs(result)

    comparison_df = build_comparison_df(results, SCORING_CONFIGS)
    write_comparison_parquet(comparison_df)

    variance_df = build_variance_df(comparison_df)
    write_variance_parquet(variance_df)

    print_variance_report(comparison_df)


if __name__ == "__main__":
    main()