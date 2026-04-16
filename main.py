"""
This script manages configuration loading, handles command-line arguments, 
and resolves which simulation run should be processed.
"""

import sys
import tomllib
from pathlib import Path

from pipeline.run_pipeline import PipelineRunner

CONFIG_PATH = Path(__file__).parent / "config.toml"


def load_config() -> dict:
    """
    Reads the project's settings from the config.toml file.
    
    This typically includes system paths, such as where the raw simulation 
    parquet files are stored.
    """
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def resolve_run(perkuet_root: Path, uuid: str | None) -> Path:
    """
    Determines which simulation directoryg to process.

    If a specific ID (UUID) is provided, it looks for that folder and executes that run.
    If no ID is provided, it automatically selects the most recently modified directory.

    Args:
        perkuet_root (Path): The base directory where all simulation runs are stored.
        uuid (str | None): An optional specific run identifier.

    Returns:
        Path: The absolute path to the chosen simulation run directory.

    Raises:
        FileNotFoundError: If a specific UUID is requested but doesn't exist,
        or if no runs are found in the directory.
    """
    if uuid:
        run_dir = perkuet_root / uuid
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run '{uuid}' not found in {perkuet_root}")
        return run_dir

    runs = [p for p in perkuet_root.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No simulation runs found in {perkuet_root}")
        
    return max(runs, key=lambda p: p.stat().st_mtime)


def main():
    config = load_config()
    perkuet_root = Path(config["paths"]["perkuet_dir"])

    uuid = sys.argv[1] if len(sys.argv) > 1 else None
    
    try:
        run_dir = resolve_run(perkuet_root, uuid)
        pipeline = PipelineRunner(run_dir)
        pipeline.run_all()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()