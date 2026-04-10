"""
main.py
-------
Entry point for the SmartEV analysis pipeline.

Reads the parquet root directory from config.toml.

Usage
-----
    python main.py              # process the most recently modified run
    python main.py <uuid>       # process a specific run by its UUID
"""

import sys
import tomllib
from pathlib import Path

from analysis.raw_metrics_analyser import analyse_station, analyse_charger

CONFIG_PATH = Path(__file__).parent / "config.toml"

STATION_FILENAME = "StationSnapshotMetric.parquet"
CHARGER_FILENAME = "ChargerSnapshotMetric.parquet"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        print(f"Error: config.toml not found at {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def resolve_run(perkuet_root: Path, uuid: str | None) -> Path:
    """
    Return the run directory to process.
    If uuid is given, verify it exists and return it.
    Otherwise, return the most recently modified run folder.
    """
    if uuid:
        run_dir = perkuet_root / uuid
        if not run_dir.is_dir():
            print(f"Error: run '{uuid}' not found in {perkuet_root}")
            sys.exit(1)
        return run_dir

    runs = [p for p in perkuet_root.iterdir() if p.is_dir()]
    if not runs:
        print(f"Error: no run folders found in '{perkuet_root}'.")
        sys.exit(1)

    return max(runs, key=lambda p: p.stat().st_mtime)


def main() -> None:
    config = load_config()
    perkuet_root = Path(config["paths"]["perkuet_dir"])

    if not perkuet_root.exists():
        print(f"Error: directory '{perkuet_root}' does not exist.")
        sys.exit(1)

    uuid = sys.argv[1] if len(sys.argv) > 1 else None
    run_dir = resolve_run(perkuet_root, uuid)
    run_id = run_dir.name

    print(f"Run ID : {run_id}")
    print(f"Source : {run_dir}")

    station_path = run_dir / STATION_FILENAME
    charger_path = run_dir / CHARGER_FILENAME

    if station_path.exists():
        analyse_station(station_path, run_id)
    else:
        print(f"[warn] {STATION_FILENAME} not found, skipping station analysis.")

    if charger_path.exists():
        analyse_charger(charger_path, run_id)
    else:
        print(f"[warn] {CHARGER_FILENAME} not found, skipping charger analysis.")

    print("\nAll done.")


if __name__ == "__main__":
    main()