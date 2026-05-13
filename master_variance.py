import sys
from pathlib import Path
import polars as pl

from analysis.scoring.scoring_config import ScoringConfig
from analysis.scoring.simulation_scorer import compute_simulation_score, SimulationScore

# --- KILL THE FILE SPAM ---
# This dynamically overrides the write methods so the scorer does all the math 
# in memory but writes ZERO individual parquet/json files to your disk.
SimulationScore.write_parquet = lambda self, path: None
SimulationScore.write_json = lambda self, path: None

PROJECT_ROOT = Path(__file__).resolve().parent

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run master_variance.py <path_to_grid_session>")
        sys.exit(1)

    target_root = Path(sys.argv[1]).resolve()
    if not target_root.is_dir():
        print(f"Directory not found: {target_root}")
        sys.exit(1)

    # Grab runs that actually have data
    runs = [
        p for p in target_root.iterdir() 
        if p.is_dir() and (p / "analysis").exists() and (p / "percentiles").exists()
    ]

    if not runs:
        print(f"No processed simulation runs found in {target_root}")
        sys.exit(1)

    # Define the penalty configurations to test
    configs = {
        "heavy_wait": ScoringConfig(wait_decay_minutes=15.0),
        "lenient_dev": ScoringConfig(path_deviation_buckets=[(15, 0), (30, 2), (60, 5), (float("inf"), 10)]),
        "strict_dead": ScoringConfig(
            ev_metric_weights={
                "path_deviation": 1.0,
                "delta_arrival": 1.0,
                "ev_wait_time": 3.0,
                "missed_deadline": 10.0,
            }
        )
    }
    
    baseline_config = ScoringConfig()

    results = []
    total_runs = len(runs)
    print(f"Computing Master Variance Table for {total_runs} runs...")

    for i, run_path in enumerate(runs, 1):
        run_id = run_path.name
        print(f"  [{i}/{total_runs}] Scoring {run_id}...")
        
        row_data = {"run_id": run_id}
        
        # 1. Compute Baseline
        try:
            baseline = compute_simulation_score(run_id, str(run_path), target_root, None, baseline_config)
            row_data["baseline_overall"] = round(baseline.overall_aggregate, 6)
            row_data["baseline_wait"] = round(baseline.ev_wait_time_aggregate, 6)
            row_data["baseline_deviation"] = round(baseline.path_deviation_aggregate, 6)
        except Exception as e:
            print(f"    Failed baseline for {run_id}: {e}")
            continue # If baseline fails, skip the run

        # 2. Compute Penalties and Deltas
        for config_name, config_obj in configs.items():
            try:
                score = compute_simulation_score(run_id, str(run_path), target_root, None, config_obj)
                
                # Overall Score & Delta
                row_data[f"{config_name}_overall"] = round(score.overall_aggregate, 6)
                row_data[f"{config_name}_delta_overall"] = round(score.overall_aggregate - baseline.overall_aggregate, 6)
                
                # Wait Score & Delta
                row_data[f"{config_name}_wait"] = round(score.ev_wait_time_aggregate, 6)
                row_data[f"{config_name}_delta_wait"] = round(score.ev_wait_time_aggregate - baseline.ev_wait_time_aggregate, 6)
                
                # Deviation Score & Delta
                row_data[f"{config_name}_deviation"] = round(score.path_deviation_aggregate, 6)
                row_data[f"{config_name}_delta_deviation"] = round(score.path_deviation_aggregate - baseline.path_deviation_aggregate, 6)
                
            except Exception as e:
                print(f"    Failed {config_name} for {run_id}: {e}")

        results.append(row_data)

    if not results:
        print("No results generated.")
        sys.exit(1)

    # Create the massive flat DataFrame
    df = pl.DataFrame(results)
    
    # Calculate maximum variance spread for sorting
    penalty_cols = [f"{c}_overall" for c in configs.keys()]
    df = df.with_columns([
        (pl.max_horizontal(penalty_cols + ["baseline_overall"]) - 
         pl.min_horizontal(penalty_cols + ["baseline_overall"])).alias("max_variance_spread")
    ]).sort("max_variance_spread", descending=True)

    # Save to exactly ONE file
    output_path = target_root / f"{target_root.name}_MASTER_VARIANCE.csv"
    df.write_csv(output_path)
    
    print("\n" + "="*60)
    print(f" DONE! One big result file generated: {output_path.name}")
    print("="*60)
    print("Columns include your Baseline, Penalty Scores, and the +/- Delta for side-by-side comparison.")

if __name__ == "__main__":
    main()
