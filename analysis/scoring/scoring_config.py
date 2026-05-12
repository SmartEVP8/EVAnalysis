from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ScoringConfig:
    group_weights: dict[str, float] = field(default_factory=lambda: {"ev": 1.0, "station": 1.0})
    ev_metric_weights: dict[str, float] = field(default_factory=lambda: {
        "path_deviation": 1.0,
        "delta_arrival": 1.0,
        "ev_wait_time": 3.0,
        "missed_deadline": 2.0,
    })
    station_metric_weights: dict[str, float] = field(default_factory=lambda: {
        "utilization": 1.0,
        "expected_wait_time": 3.0,
    })
    path_deviation_buckets: list[tuple[float, int]] = field(default_factory=lambda: [
        (5, 0), (10, 0), (15, 2), (30, 6), (60, 12), (float("inf"), 15)
    ])
    delta_arrival_buckets: list[tuple[float, int]] = field(default_factory=lambda: [
        (0, 0), (5, 1), (10, 2), (15, 3), (30, 6), (60, 10), (float("inf"), 15)
    ])
    wait_decay_minutes: float = 45.0
    warmup_ms: int = (3 * 60 * 60 * 1000) - 1200000
    sim_epoch: datetime = field(default_factory=lambda: datetime(2024, 1, 1, 0, 0, 0))

    def _generate_labels(self, buckets: list[tuple[float, int]]) -> list[str]:
        labels = []
        for i, (upper_bound, _) in enumerate(buckets):
            if upper_bound == float("inf"):
                previous_bound = buckets[i-1][0] if i > 0 else 0
                labels.append(f"{int(previous_bound)}+")
            else:
                labels.append(str(int(upper_bound)))
        return labels

    @property
    def path_deviation_labels(self) -> list[str]:
        return self._generate_labels(self.path_deviation_buckets)

    @property
    def delta_arrival_labels(self) -> list[str]:
        return self._generate_labels(self.delta_arrival_buckets)

    @property
    def total_group_weight(self) -> float:
        return sum(self.group_weights.values())

    @property
    def ev_total_weight(self) -> float:
        return sum(self.ev_metric_weights.values())

    @property
    def station_total_weight(self) -> float:
        return sum(self.station_metric_weights.values())
