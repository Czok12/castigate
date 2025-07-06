# castigatio_backend/app/services/learning_service.py
from collections import deque
from typing import Deque

import numpy as np

from app.models.performance import PerformanceMetrics


class LearningService:
    """Service für adaptives Lernen und Systemoptimierung."""

    def __init__(self) -> None:
        self.metrics_history: Deque[PerformanceMetrics] = deque(maxlen=100)
        self.adaptive_parameters = {
            "retrieval_k_multiplier": 4,  # Startwert
            "rerank_confidence_threshold": 0.5,
        }

    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Zeichnet Metriken auf und löst Lernzyklen aus."""
        self.metrics_history.append(metrics)

        # Periodische Anpassung (z.B. alle 10 Anfragen)
        if len(self.metrics_history) % 10 == 0:
            self._run_adaptation_cycle()

    def _run_adaptation_cycle(self) -> None:
        """Passt Systemparameter basierend auf den letzten Metriken an."""
        if len(self.metrics_history) < 10:
            return

        recent_metrics = list(self.metrics_history)[-10:]
        avg_confidence = float(np.mean([m.confidence_score for m in recent_metrics]))
        avg_response_time = float(
            np.mean([m.total_duration_ms for m in recent_metrics])
        )

        # Beispielhafte Lernregel:
        # Wenn Konfidenz niedrig, aber Zeit vorhanden, hole mehr Kandidaten für das Reranking.
        if avg_confidence < 0.75 and avg_response_time < 5000:  # < 5 Sekunden
            new_multiplier = min(
                self.adaptive_parameters["retrieval_k_multiplier"] + 1, 6
            )
            if new_multiplier != self.adaptive_parameters["retrieval_k_multiplier"]:
                self.adaptive_parameters["retrieval_k_multiplier"] = new_multiplier
                print(
                    f"INFO (LearningService): Konfidenz niedrig. Erhöhe Retrieval-Multiplier auf {new_multiplier}."
                )

        # Wenn Konfidenz hoch, aber Zeit knapp, hole weniger Kandidaten.
        elif avg_confidence > 0.85 and avg_response_time > 8000:  # > 8 Sekunden
            new_multiplier = max(
                self.adaptive_parameters["retrieval_k_multiplier"] - 1, 2
            )
            if new_multiplier != self.adaptive_parameters["retrieval_k_multiplier"]:
                self.adaptive_parameters["retrieval_k_multiplier"] = new_multiplier
                print(
                    f"INFO (LearningService): Antwortzeit hoch. Reduziere Retrieval-Multiplier auf {new_multiplier}."
                )


learning_service = LearningService()
