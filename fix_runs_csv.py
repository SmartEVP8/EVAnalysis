import csv
import shutil
from pathlib import Path

from grid_search import (
    load_perkuet_root,
    validate_metrics_parquet,
    run_analysis,
    run_scoring,
    RESULT_FIELDNAMES
)


def clean_analysis_outputs(run_dir: Path) -> None:
    """Remove analysis output directories to ensure fresh re-analysis."""
    output_dirs = ["analysis", "buckets", "outliers", "percentiles"]
    for dirname in output_dirs:
        dir_path = run_dir / dirname
        if dir_path.exists():
            print(f"  Cleaning {dirname}/...")
            shutil.rmtree(dir_path)


def repair_csv(csv_path_str: str):
    csv_path = Path(csv_path_str)
    if not csv_path.exists():
        print(f"Error: Could not find CSV at {csv_path}")
        return

    perkuet_root = load_perkuet_root()

    # 1. Get all run folders sorted by creation time
    all_run_dirs = [p for p in perkuet_root.iterdir() if p.is_dir()]
    all_run_dirs.sort(key=lambda p: p.stat().st_mtime)

    # 2. Read the existing CSV into memory
    print(f"Reading {csv_path.name}...")
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    fixed_count = 0

    # 3. Iterate through rows and fix the broken ones
    for row in rows:
        if row["status"] == "error" or not row["run_id"]:
            iteration = int(row["iteration"])
            
            run_dir = all_run_dirs[iteration - 1]
            
            print(f"\nFixing Iteration {iteration} using folder {run_dir.name}...")
            
            try:
                # Clean up previous analysis outputs and re-run the entire analysis
                print(f"  Cleaning previous analysis outputs...")
                clean_analysis_outputs(run_dir)
                
                print(f"  Validating parquet metrics...")
                validate_metrics_parquet(run_dir)
                
                # Use perkuet_root to process the update in-place rather than making a recovery folder
                print(f"  Running analysis pipeline...")
                run_analysis(run_dir, perkuet_root)
                
                print(f"  Computing scores...")
                scores = run_scoring(run_dir.name, perkuet_root)
                
                # Update the row data in memory
                row["run_id"] = run_dir.name
                row["status"] = "ok"
                row["error"] = ""
                
                # Update all the zeroes with real scores
                row["missed_deadline_aggregate"] = f"{scores['missed_deadline_aggregate']:.6f}"
                row["ev_wait_time_aggregate"] = f"{scores['ev_wait_time_aggregate']:.6f}"
                row["utilization_aggregate"] = f"{scores['utilization_aggregate']:.6f}"
                row["expected_wait_time_aggregate"] = f"{scores['expected_wait_time_aggregate']:.6f}"
                row["ev_aggregate"] = f"{scores['ev_aggregate']:.6f}"
                row["station_aggregate"] = f"{scores['station_aggregate']:.6f}"
                row["overall_score"] = f"{scores['overall_score']:.6f}"
                
                fixed_count += 1
                print(f"Successfully fixed row {iteration}!")
                
            except Exception as e:
                print(f"Failed to fix row {iteration}: {e}")

    # 4. Save everything back to the file
    if fixed_count > 0:
        print(f"\nWriting {fixed_count} fixed rows back to CSV...")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        print("CSV restored!")
    else:
        print("\nNo rows needed fixing.")

if __name__ == "__main__":
    # UPDATE THIS PATH to where the broken CSV is located
    TARGET_CSV = "" 
    
    repair_csv(TARGET_CSV)