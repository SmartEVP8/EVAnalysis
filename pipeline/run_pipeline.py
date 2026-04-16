"""
Module: run_pipeline
Defines the main execution pipeline for the EVAnalysis project.
Coordinates the flow from raw parquet metrics to statistical analysis and final heatmaps.
"""

from dataclasses import dataclass
from pathlib import Path

from analysis.metrics_analyser.station_metrics_analyser import analyse_station
from analysis.metrics_analyser.charger_metrics_analyser import analyse_charger
from analysis.metrics_analyser.arrival_metrics_analyser import analyse_arrival
from analysis.detect_outliers.outlier_analyser import process_outliers
from visualisation.heatmaps.heatmaps_loader import load_heatmap_data
from visualisation.heatmaps.renderer import render_all


@dataclass(frozen=True)
class RunPaths:
    """
    All file paths associated with a single simulation run.
    """
    run_dir: Path
    station_metrics: Path
    charger_metrics: Path
    arrival_metrics: Path

    analysis_dir: Path
    station_snapshots: Path

    stations_locations: Path

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> "RunPaths":
        analysis_dir = Path("runs") / run_dir.name / "analysis"
        return cls(
            run_dir=run_dir,
            station_metrics=run_dir / "StationSnapshotMetric.parquet",
            charger_metrics=run_dir / "ChargerSnapshotMetric.parquet",
            arrival_metrics=run_dir / "ArrivalAtDestinationMetric.parquet",
            station_snapshots=analysis_dir / "station_snapshots.parquet",
            stations_locations=Path("data/stations_locations.parquet"),
            analysis_dir=analysis_dir,
        )


class PipelineRunner:
    """
    Orchestrates the data processing and visualisation pipeline for a simulation run.
    """

    def __init__(self, run_dir: Path):
        """
        Args:
            run_dir: Directory containing the raw simulation outputs
                     (Perkuet/{run_id} inside the SmartEV repo).
        """
        self.run_id = run_dir.name
        self.paths = RunPaths.from_run_dir(run_dir)


    def file_exists(self, path: Path, description: str) -> bool:
        if not path.exists():
            raise FileNotFoundError(f"{description} not found at {path}")
            
        return True


    def run_analysis(self) -> None:
        """Runs metric analysis for stations, chargers, and EV arrivals."""
        p = self.paths

        if self.file_exists(p.station_metrics, "Station metrics"):
            analyse_station(p.station_metrics, self.run_id)

        if self.file_exists(p.charger_metrics, "Charger metrics"):
            analyse_charger(p.charger_metrics, self.run_id)

        if self.file_exists(p.arrival_metrics, "Arrival metrics"):
            analyse_arrival(p.arrival_metrics, self.run_id)


    def run_outlier_detection(self) -> None:
        """Flags statistical outliers in the processed snapshot data."""
        process_outliers(self.run_id)


    def run_heatmaps(self) -> None:
        """
        Renders spatial heatmaps from the analysed station snapshots.
        """
        p = self.paths

        self.file_exists(p.station_snapshots, "Station snapshots")

        dataset = load_heatmap_data(
            snapshots_path=p.station_snapshots,
            stations_path=p.stations_locations,
        )

        render_all(
            dataset,
            output_dir=Path("runs") / self.run_id / "heatmaps",
            resolution_km=5.0,
            use_land_mask=True,
            dpi=150,
        )


    def run_all(self) -> None:
        print(f"Run ID: {self.run_id}")
        print(f"Source: {self.paths.run_dir}")

        self.run_analysis()
        self.run_outlier_detection()
        self.run_heatmaps()