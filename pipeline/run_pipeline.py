"""
Defines the main execution pipeline for the EVAnalysis project.

Coordinates the flow from raw Parquet metrics to statistical analysis,
outlier detection, spatial heatmaps, interval/daily dashboards, and scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from helpers.constants import OUTPUT_ROOT
from analysis.metrics_analyser.station_metrics_analyser import analyse_station
from analysis.metrics_analyser.charger_metrics_analyser import analyse_charger
from analysis.metrics_analyser.arrival_metrics_analyser import analyse_arrival
from analysis.metrics_analyser.waittime_metrics_analyser import analyse_wait_time
from analysis.detect_outliers.outlier_analyser import process_outliers
from analysis.scoring.simulation_scorer import compute_simulation_score
from visualisation.heatmaps.heatmaps_loader import load_heatmap_data
from visualisation.heatmaps.renderer import render_all
from visualisation.dashboards.generate_dashboards import generate_dashboards
from visualisation.dashboards.daily_summaries.generate_daily_dashboard import generate_daily_summaries


@dataclass(frozen=True)
class RunPaths:
    """All file paths associated with a single simulation run."""

    run_dir: Path

    # Raw simulation inputs
    station_metrics: Path
    charger_metrics: Path
    arrival_metrics: Path
    wait_time_metrics: Path

    # Derived analysis outputs
    analysis_dir: Path
    station_snapshots: Path
    arrival_snapshots: Path

    # Static reference data
    stations_locations: Path

    # Outlier detection outputs
    outlier_dir: Path
    station_outliers: Path

    # Visualisation outputs
    heatmap_dir: Path
    dashboard_dir: Path

    # Aggregation outputs used by scoring
    station_percentiles: Path
    arrival_buckets: Path

    @classmethod
    def from_run_dir(cls, run_dir: Path, output_root: Path = OUTPUT_ROOT) -> RunPaths:
        analysis_dir = output_root / run_dir.name / "analysis"
        outlier_dir = output_root / run_dir.name / "outliers"

        return cls(
            run_dir=run_dir,

            station_metrics=run_dir / "StationSnapshotMetric.parquet",
            charger_metrics=run_dir / "ChargerSnapshotMetric.parquet",
            arrival_metrics=run_dir / "ArrivalAtDestinationMetric.parquet",
            wait_time_metrics=run_dir / "WaitTimeInQueueMetric.parquet",

            analysis_dir=analysis_dir,
            station_snapshots=analysis_dir / "station_snapshots.parquet",
            arrival_snapshots=analysis_dir / "arrival_snapshots.parquet",

            stations_locations=Path("data/stations_locations.parquet"),

            outlier_dir=outlier_dir,
            station_outliers=outlier_dir / "station_outliers.parquet",

            heatmap_dir=output_root / run_dir.name / "heatmaps",
            dashboard_dir=output_root / run_dir.name / "dashboards",

            station_percentiles=(
                output_root / run_dir.name / "percentiles" / "station"
                / "station_percentiles.parquet"
            ),
            arrival_buckets=(
                output_root / run_dir.name / "buckets" / "arrival"
                / "arrival_buckets.parquet"
            ),
        )


class PipelineRunner:
    """Orchestrates the data processing and visualisation pipeline for a simulation run."""

    def __init__(self, run_dir: Path, output_root: Path = OUTPUT_ROOT) -> None:
        self.run_id = run_dir.name
        self.output_root = output_root
        self.paths = RunPaths.from_run_dir(run_dir, output_root)


    def _assert_file_exists(self, path: Path, description: str) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Pipeline error: {description} not found at {path}"
            )

    def _read_parquet_if_exists(self, path: Path) -> pl.DataFrame:
        return pl.read_parquet(path) if path.exists() else pl.DataFrame()


    def run_analysis(self) -> None:
        paths = self.paths

        self._assert_file_exists(paths.charger_metrics, "Charger metrics")
        analyse_charger(paths.charger_metrics, self.run_id, self.output_root)

        self._assert_file_exists(paths.station_metrics, "Station metrics")
        analyse_station(paths.station_metrics, self.run_id, self.output_root)

        self._assert_file_exists(paths.arrival_metrics, "Arrival metrics")
        analyse_arrival(paths.arrival_metrics, self.run_id, self.output_root)

        self._assert_file_exists(paths.wait_time_metrics, "Wait time metrics")
        analyse_wait_time(paths.wait_time_metrics, self.run_id, self.output_root)

    def run_outlier_detection(self) -> None:
        process_outliers(self.run_id, self.output_root)

    def run_heatmaps(self) -> None:
        paths = self.paths
        self._assert_file_exists(paths.station_snapshots, "Station snapshots")

        dataset = load_heatmap_data(
            snapshots_path=paths.station_snapshots,
            stations_path=paths.stations_locations,
        )
        render_all(
            dataset,
            output_dir=paths.heatmap_dir,
            resolution_km=5.0,
            use_land_mask=True,
            dpi=150,
        )

    def run_dashboards(self) -> None:
        paths = self.paths
        self._assert_file_exists(paths.station_snapshots, "Station snapshots")

        station_snapshot_df = pl.read_parquet(paths.station_snapshots)
        arrival_snapshot_df = self._read_parquet_if_exists(paths.arrival_snapshots)
        outlier_analysis_df = self._read_parquet_if_exists(paths.station_outliers)

        generate_dashboards(
            run_id=self.run_id,
            station_snapshot_df=station_snapshot_df,
            arrival_snapshot_df=arrival_snapshot_df,
            outlier_analysis_df=outlier_analysis_df,
            heatmap_dir=paths.heatmap_dir,
            out_dir=paths.dashboard_dir / "intervals",
        )

        generate_daily_summaries(
            run_id=self.run_id,
            station_snapshot_df=station_snapshot_df,
            arrival_snapshot_df=arrival_snapshot_df,
            out_dir=paths.dashboard_dir / "daily",
        )

    def run_scoring(self) -> None:
        paths = self.paths
        self._assert_file_exists(paths.station_percentiles, "Station percentiles")
        self._assert_file_exists(paths.arrival_buckets, "Arrival buckets")

        compute_simulation_score(
            run_id=self.run_id,
            source_path=str(paths.run_dir),
            output_root=self.output_root,
        )

    def run_all(self) -> None:
        print(f"Run ID: {self.run_id}")
        print(f"Source: {self.paths.run_dir}")

        self.run_analysis()
        self.run_outlier_detection()
        #self.run_heatmaps()
        #self.run_dashboards()
        self.run_scoring()