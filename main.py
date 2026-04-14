import sys
import tomllib
from pathlib import Path

from pipeline.run_pipeline import PipelineRunner


CONFIG_PATH = Path(__file__).parent / "config.toml"


def load_config() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def resolve_run(perkuet_root: Path, uuid: str | None) -> Path:
    if uuid:
        run_dir = perkuet_root / uuid
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run '{uuid}' not found")
        return run_dir

    runs = [p for p in perkuet_root.iterdir() if p.is_dir()]
    return max(runs, key=lambda p: p.stat().st_mtime)


def main():
    config = load_config()
    perkuet_root = Path(config["paths"]["perkuet_dir"])

    uuid = sys.argv[1] if len(sys.argv) > 1 else None
    run_dir = resolve_run(perkuet_root, uuid)

    pipeline = PipelineRunner(run_dir)
    pipeline.run_all()


if __name__ == "__main__":
    main()