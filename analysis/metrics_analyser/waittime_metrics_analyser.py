"""
Module for analyzing EV wait time in queue metrics.

Processes WaitTimeInQueueMetric data to compute percentile distributions of
how long EVs waited before charging began.
"""

from pathlib import Path
import polars as pl

OUTPUT_ROOT = Path("runs")

PERCENTILES = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def analyse_wait_time(parquet_path: Path, run_id: str, output_root: Path = OUTPUT_ROOT) -> None:
    """
    Computes percentile distributions of EV queue wait time for a simulation run.

    Reads raw WaitTimeInQueueMetric data and produces a single-row Parquet file
    containing run-wide percentiles of WaitTimeInQueue (in milliseconds).
    """
    print(f"\n[WaitTime] Analysing {parquet_path.name}...")

    df = pl.read_parquet(parquet_path)

    waittime_column = "WaitTimeInQueue"
    if waittime_column not in df.columns:
        raise ValueError(f"Expected column '{waittime_column}' not found in {parquet_path.name}. "
                         f"Available columns: {df.columns}")

    percentile_row = {
        f"ev_½_p{int(q * 100)}": float(df[waittime_column].quantile(q)) / 1000 / 60
        for q in PERCENTILES
    }

    percentile_df = pl.DataFrame([percentile_row])

    out_percentiles = output_root / run_id / "percentiles" / "waittime"
    out_percentiles.mkdir(parents=True, exist_ok=True)

    out_path = out_percentiles / "wait_time_percentiles.parquet"
    percentile_df.write_parquet(out_path)

    print(f"  Saved waittime_snapshots.parquet ({len(percentile_df)} rows)")