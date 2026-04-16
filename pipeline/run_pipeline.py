"""
Module: run_pipeline
Defines the main execution pipeline for the EVAnalysis project.
Coordinates the flow from raw parquet metrics to statistical analysis and final heatmaps.
"""

from pathlib import Path

from analysis.metrics_analyser.station_metrics_analyser import analyse_station
from analysis.metrics_analyser.charger_metrics_analyser import analyse_charger
from analysis.detect_outliers.outlier_analyser import process_outliers
from visualisation.heatmaps.heatmaps_loader import load_heatmap_data
from visualisation.heatmaps.renderer import render_all


class PipelineRunner:
    """
    Orchestrates the data processing and visualization pipeline for a specific simulation run.
    
    This class manages file paths, triggers metric analysis for stations and chargers,
    and handles the generation of spatial heatmaps based on the processed results.

    Attributes:
        run_dir (Path): The directory containing the raw simulation outputs (Perkuet/{run_id} in SmartEV).
        run_id (str): The unique name of the run (derived from the directory name).
        station_metrics (Path): Path to the raw station-level metric parquet.
        charger_metrics (Path): Path to the raw charger-level metric parquet.
        analysis_dir (Path): The target directory for processed analysis files.
        station_analysis (Path): Path to the cleaned station snapshot file.
        stations_locations (Path): Path to the static station geographic coordinates.
    """

    def __init__(self, run_dir: Path):
        """
        Initializes the PipelineRunner with necessary file paths.

        Args:
            run_dir (Path): The directory where the simulation raw data is stored (Perkuet/{run_id} in SmartEV).
        """
        self.run_dir = run_dir
        self.run_id = run_dir.name

        # Define input file paths
        self.station_metrics = run_dir / "StationSnapshotMetric.parquet"
        self.charger_metrics = run_dir / "ChargerSnapshotMetric.parquet"

        # Define output and reference paths
        self.analysis_dir = Path("runs") / self.run_id / "analysis"
        self.station_analysis = self.analysis_dir / "station_snapshots.parquet"
        self.stations_locations = Path("data/stations_locations.parquet")

    def run_analysis(self):
        """
        Executes the statistical analysis for both stations and chargers.
        """
        if self.station_metrics.exists():
            analyse_station(self.station_metrics, self.run_id)
        else:
            print(f"Missing station raw parquet at {self.station_metrics}")

        if self.charger_metrics.exists():
            analyse_charger(self.charger_metrics, self.run_id)
        else:
            print(f"Missing charger raw parquet at {self.charger_metrics}")

    def run_heatmaps(self):
        """
        Generates spatial visualizations based on the analyzed station data.

        This method loads the processed snapshots, joins them with geographic 
        location data, and renders heatmap images.

        Raises:
            FileNotFoundError: If no station analysis has been performed, or if
            station analysis has been saved incorrectly.
        """
        if not self.station_analysis.exists():
            raise FileNotFoundError(
                f"Missing analysis output: {self.station_analysis}. "
                "Ensure run_analysis() is called before run_heatmaps()."
            )

        # Merge snapshot data with geographic coordinates
        dataset = load_heatmap_data(
            snapshots_path=self.station_analysis,
            stations_path=self.stations_locations,
        )

        # Trigger the rendering engine with specific spatial parameters
        render_all(
            dataset,
            output_dir=Path("runs") / self.run_id / "heatmaps",
            resolution_km=5.0,
            use_land_mask=True,
            dpi=150,
        )

    def run_all(self):
        """
        A convenience method to execute the full end-to-end pipeline.
        
        Prints the Run ID and source directory for logging purposes before 
        starting the analysis and visualization phases.
        """
        print(f"Run ID: {self.run_id}")
        print(f"Source: {self.run_dir}")

        self.run_analysis()
        process_outliers(self.run_id)
        # self.run_heatmaps()