# SmartEVAnalysis

## Setup

Install uv: https://docs.astral.sh/uv/getting-started/installation/

Then:
```bash
git clone <repo-url>
cd SmartEVAnalysis
uv sync
```

That's it — uv will install the correct Python version and all dependencies automatically.

## Running

```bash
uv run jupyter lab   # open analysis notebooks
uv run main.py       # run the main script
```

## Adding dependencies

```bash
uv add <package>     # add a new dependency
```

Make sure to commit the updated `pyproject.toml` and `uv.lock` after adding deps.
