from app.models.performance import PerformanceMetrics, SystemReport


def test_performance_metrics_model():
    metrics = PerformanceMetrics(
        total_duration_ms=1000.0,
        retrieval_duration_ms=200.0,
        enhancement_duration_ms=100.0,
        rerank_duration_ms=100.0,
        generation_duration_ms=500.0,
        quality_duration_ms=100.0,
        confidence_score=0.8,
        cache_hit=True,
    )
    assert metrics.total_duration_ms == 1000.0
    assert metrics.cache_hit is True


def test_system_report_model():
    report = SystemReport(
        total_queries=10,
        cache_hit_rate=0.5,
        avg_response_time_ms=1200.0,
        avg_confidence=0.7,
        adaptive_parameters={"retrieval_k_multiplier": 4},
    )
    assert report.total_queries == 10
    assert report.adaptive_parameters["retrieval_k_multiplier"] == 4
