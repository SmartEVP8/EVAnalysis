import argparse
import sys
import time
import tomllib
from pathlib import Path

import polars as pl

from analysis.scoring.simulation_scorer import compute_simulation_score

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.toml"

def load_config() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score existing simulation runs with varying penalty configurations.")
    parser.add_argument(
        "--run-id", 
        type=str, 
        help="Specific run ID to score. If omitted, scores all runs in the data directory."
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=PROJECT_ROOT / "runs" / "variance_tracking", 
        help="Directory to save the variance analysis results."
    )
    return parser.parse_args()

def get_target_runs(perkuet_root: Path, target_id: str | None) -> list[Path]:
    if target_id:
        run_dir = perkuet_root / target_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run '{target_id}' not found in {perkuet_root}")
        return [run_dir]

    runs = [p for p in perkuet_root.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No simulation runs found in {perkuet_root}")
    
    return runs

def main() -> None:
    args = parse_args()
    config_toml = load_config()
    
    perkuet_root = Path(config_toml["paths"]["perkuet_dir"])
    if not perkuet_root.is_absolute():
        perkuet_root = (PROJECT_ROOT / perkuet_root).resolve()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        target_runs = get_target_runs(perkuet_root, args.run_id)
    except Exception as e:
        print(f"Error resolving runs: {e}")
        sys.exit(1)

    configs = {
        "baseline": ScoringConfig(),
        "heavy_wait_penalty": ScoringConfig(wait_decay_minutes=15.0),
        "lenient_deviation": ScoringConfig(
            path_deviation_buckets=[(15, 0), (30, 2), (60, 5), (float("inf"), 10)]
        ),
        "strict_deadline": ScoringConfig(
            ev_metric_weights={
                "path_deviation": 1.0,
                "delta_arrival": 1.0,
                "ev_wait_time": 3.0,
                "missed_deadline": 10.0,
            }
        )
    }

    results = []
    total_runs = len(target_runs)
    
    print(f"Starting variance scoring on {total_runs} runs across {len(configs)} configurations.")

    for i, run_path in enumerate(target_runs, 1):
        run_id = run_path.name
        print(f"\nProcessing [{i}/{total_runs}]: {run_id}")
        
        if not (run_path / "analysis").exists() or not (run_path / "percentiles").exists():
            print(f"  Skipping {run_id} - missing analysis or percentiles directories.")
            continue

        for config_name, scoring_config in configs.items():
            start_time = time.perf_counter()
            config_output_dir = output_dir / config_name / run_id
            
            try:
                score = compute_simulation_score(
                    run_id=run_id,
                    source_path=str(run_path),
                    output_root=perkuet_root,
                    output_path=config_output_dir / "simulation_score.parquet",
                    config=scoring_config
                )
                
                elapsed = time.perf_counter() - start_time
                print(f"  [{config_name}] scored in {elapsed:.2f}s - Overall: {score.overall_aggregate:.4f}")
                
                results.append({
                    "run_id": run_id,
                    "config_name": config_name,
                    "overall_score": score.overall_aggregate,
                    "ev_wait_score": score.ev_wait_time_aggregate,
                    "path_deviation_score": score.path_deviation_aggregate,
                    "missed_deadline_score": score.missed_deadline_aggregate,
                    "utilization_score": score.utilization_aggregate,
                    "expected_wait_score": score.expected_wait_aggregate
                })
            except Exception as e:
                print(f"  [{config_name}] Failed to score {run_id}: {e}")

    if not results:
        print("\nNo results generated. Exiting.")
        sys.exit(1)

    results_df = pl.DataFrame(results)
    summary_path = output_dir / "variance_summary.csv"
    results_df.write_csv(summary_path)
    
    print(f"\nVariance scoring complete. Summary written to {summary_path}")

if __name__ == "__main__":
    main()
