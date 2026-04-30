"""
simulation_scorer.py
Aggregates EV and station scores into a single simulation score and writes
simulation_score.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from ev_scorer import EVScores
from station_scorer import StationScores



GROUP_WEIGHTS: dict[str, int] = {
    "ev": 1,
    "station": 1,
}


class SimulationScore:
    def __init__(
        self,
        run_id: str,
        source_path: str,
        ev_scores: EVScores,
        station_scores: StationScores,
    ):
        self.run_id = run_id
        self.source_path = source_path
        self.ev_scores = ev_scores
        self.station_scores = station_scores
        self.overall_aggregate = self.compute_overall()

    def compute_overall(self) -> float:
        """
        Weighted average of the EV and station weighted aggregates.
        """
        total_weight = sum(GROUP_WEIGHTS.values())
        return (
            GROUP_WEIGHTS["ev"] * self.ev_scores.weighted_aggregate
            + GROUP_WEIGHTS["station"] * self.station_scores.weighted_aggregate
        ) / total_weight

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "simulation_config": {
                "source": self.source_path,
            },
            "ev_scores": self.ev_scores.to_dictionary(),
            "station_scores": self.station_scores.to_dictionary(),
            "group_weights": GROUP_WEIGHTS,
            "overall_aggregate": round(self.overall_aggregate, 6),
        }

    def write_json(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[simulation_scorer] Wrote {output_path}")



def compute_simulation_score(
    *,
    run_id: str,
    source_path: str,
    ev_scores: EVScores,
    station_scores: StationScores,
    output_path: str | Path = "simulation_score.json",
) -> SimulationScore:
    """
    Combine EV and station scores into a SimulationScore, write JSON, and return.
    """
    result = SimulationScore(
        run_id=run_id,
        source_path=source_path,
        ev_scores=ev_scores,
        station_scores=station_scores,
    )
    result.write_json(output_path)
    return result