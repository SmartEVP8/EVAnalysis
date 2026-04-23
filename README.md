# SmartEVAnalysis

## Setup

Install uv: https://docs.astral.sh/uv/getting-started/installation/ , or through pacman or brew

Then:
```bash
cd SmartEVAnalysis
uv sync
```

That's it — uv will install the correct Python version and all dependencies automatically.

## Running

```bash
uv run jupyter lab   # open analysis notebooks
uv run main.py       # run the main script
```

## Random Search For Cost Weights

Use this to sample random cost-weight combinations, run the headless simulation, and then run the normal analysis pipeline per generated run.

```bash
uv run random_grid_search.py \
	--s random
	--iterations 20 \
```
## Grid Search For Cost Weights

Use this to sample Grid cost-weight combinations, run the headless simulation, and then run the normal analysis pipeline per generated run.

```bash
uv run random_grid_search.py \
	--s grid
```

Results are appended to `runs/random_search_results.csv`.

## Adding dependencies

```bash
uv add <package>     # add a new dependency
```

Make sure to commit the updated `pyproject.toml` and `uv.lock` after adding deps.
