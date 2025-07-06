from app.models.performance import PerformanceMetrics
from app.services.learning_service import LearningService


def test_learning_service_adaptation():
    service = LearningService()
    # Fülle die History mit 10 Einträgen mit niedriger Konfidenz und schneller Antwort
    for _ in range(10):
        metrics = PerformanceMetrics(
            total_duration_ms=2000.0,
            retrieval_duration_ms=500.0,
            enhancement_duration_ms=200.0,
            rerank_duration_ms=200.0,
            generation_duration_ms=500.0,
            quality_duration_ms=200.0,
            confidence_score=0.6,
            cache_hit=False,
        )
        service.record_metrics(metrics)
    # Nach 10 Einträgen sollte der retrieval_k_multiplier erhöht werden
    assert service.adaptive_parameters["retrieval_k_multiplier"] > 4
