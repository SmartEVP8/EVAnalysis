from pathlib import Path

from analysis.metrics_analyser.station_metrics_analyser import analyse_station
from analysis.metrics_analyser.charger_metrics_analyser import analyse_charger
from visualisation.heatmaps.heatmaps_loader import load_heatmap_data
from visualisation.heatmaps.renderer import render_all


class PipelineRunner:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_id = run_dir.name

        self.station_metrics = run_dir / "StationSnapshotMetric.parquet"
        self.charger_metrics = run_dir / "ChargerSnapshotMetric.parquet"

        self.analysis_dir = Path("runs") / self.run_id / "analysis"
        self.station_analysis = self.analysis_dir / "station_snapshots.parquet"

        self.stations_locations = Path("data/stations_locations.parquet")

    def run_analysis(self):
        if self.station_metrics.exists():
            analyse_station(self.station_metrics, self.run_id)
        else:
            print("[warn] missing station raw parquet")

        if self.charger_metrics.exists():
            analyse_charger(self.charger_metrics, self.run_id)
        else:
            print("[warn] missing charger raw parquet")

    def run_heatmaps(self):
        if not self.station_analysis.exists():
            raise FileNotFoundError(
                f"Missing analysis output: {self.station_analysis}"
            )

        dataset = load_heatmap_data(
            snapshots_path=self.station_analysis,
            stations_path=self.stations_locations,
        )

        render_all(
            dataset,
            output_dir=Path("runs") / self.run_id / "heatmaps",
            resolution_km=5.0,
            use_land_mask=True,
            dpi=150,
        )

    def run_all(self):
        print(f"Run ID: {self.run_id}")
        print(f"Source: {self.run_dir}")

        self.run_analysis()
        self.run_heatmaps()