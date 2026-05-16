from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import analysis.scoring.simulation_scorer as simulation_scorer

from seeded_runs import RESULT_FIELDNAMES

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-run scoring for every run in a session directory.")
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Directory containing the original run folders and results CSV.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to <source_dir>/score.csv.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def find_results_csv(source_dir: Path) -> Path:
    for name in ("grid_search_results.csv", "seeded_runs_results.csv"):
        candidate = source_dir / name
        if candidate.is_file():
            return candidate

    csv_files = sorted(path for path in source_dir.glob("*.csv") if path.is_file() and path.name != "score.csv")
    if len(csv_files) == 1:
        return csv_files[0]
    if not csv_files:
        raise FileNotFoundError(f"Could not find a source results CSV in {source_dir}")
    raise RuntimeError(f"Found multiple CSV files in {source_dir}; pass an explicit source directory with one results CSV.")


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def score_row(row: dict[str, str], *, source_root: Path) -> dict[str, Any]:
    scored_row: dict[str, Any] = {field: row.get(field, "") for field in RESULT_FIELDNAMES}
    run_id = (row.get("run_id") or "").strip()

    if not run_id:
        scored_row["status"] = row.get("status", "error") or "error"
        scored_row["error"] = row.get("error", "Missing run_id") or "Missing run_id"
        return scored_row

    started_at = time.perf_counter()
    score = simulation_scorer.compute_simulation_score(
        run_id=run_id,
        source_path=str(source_root),
        output_root=source_root,
    )
    score_seconds = time.perf_counter() - started_at

    scored_row.update(
        {
            "status": "ok",
            "error": "",
            "score_seconds": f"{score_seconds:.4f}",
            "missed_deadline_aggregate": f"{score.missed_deadline_aggregate:.6f}",
            "ev_wait_time_aggregate": f"{score.ev_wait_time_aggregate:.6f}",
            "utilization_aggregate": f"{score.utilization_aggregate:.6f}",
            "expected_wait_time_aggregate": f"{score.expected_wait_aggregate:.6f}",
            "pathdev_aggregate": f"{score.path_deviation_aggregate:.6f}",
            "deltarrival_aggregate": f"{score.delta_arrival_aggregate:.6f}",
            "ev_aggregate": f"{score.ev_weighted_aggregate:.6f}",
            "station_aggregate": f"{score.station_weighted_aggregate:.6f}",
            "overall_score": f"{score.overall_aggregate:.6f}",
        }
    )
    return scored_row


def write_rows(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source_dir = resolve_path(args.source_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {source_dir}")

    source_csv = find_results_csv(source_dir)
    output_csv = resolve_path(args.output_file) if args.output_file else source_dir / "score.csv"
    rows = load_rows(source_csv)

    print(f"Reading runs from: {source_dir}")
    print(f"Source CSV       : {source_csv}")
    print(f"Writing score CSV : {output_csv}")

    scored_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        run_id = row.get("run_id", "")
        print(f"[{index}/{len(rows)}] {run_id or '<missing run_id>'}")
        try:
            scored_rows.append(score_row(row, source_root=source_dir))
        except Exception as exc:  # noqa: BLE001
            failed_row: dict[str, Any] = {field: row.get(field, "") for field in RESULT_FIELDNAMES}
            failed_row["status"] = "error"
            failed_row["error"] = str(exc)
            scored_rows.append(failed_row)
            print(f"  failed: {exc}")

    write_rows(output_csv, scored_rows)
    print("Done.")


if __name__ == "__main__":
    main()
