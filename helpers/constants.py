"""
Shared constants for the EVAnalysis pipeline.
"""

from pathlib import Path

OUTPUT_ROOT: Path = Path("runs")

PERCENTILES: list[float] = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]