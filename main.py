import logging
import sys
import tomllib
import polars as pl
from pathlib import Path

from template.models import station_snapshot, charger_snapshot
from template.analysis import charger_analysis, station_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).parent / "config.toml"


def load_config() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def latest_run(perkuet_dir: Path) -> Path:
    runs = [p for p in perkuet_dir.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No run folders found in {perkuet_dir}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def main() -> None:
    toml = load_config()
    base = CONFIG_PATH.parent
    perkuet_dir = (base / toml["paths"]["perkuet_dir"]).resolve()

    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run(perkuet_dir)
    logger.info("Loading run: %s", run_dir.name)

    stations = station_snapshot.load(run_dir / "StationSnapshotMetric.parquet")
    chargers = charger_snapshot.load(run_dir / "ChargerSnapshotMetric.parquet")

    # Analysis
    charger_summary = charger_analysis.summary(chargers)
    station_summary = station_analysis.summary(stations, chargers)
    print(charger_summary)
    print(station_summary)


if __name__ == "__main__":
    main()