# castigatio_backend/app/api/system.py
from fastapi import APIRouter

from app.models.performance import SystemReport
from app.services.cache_service import cache_service
from app.services.learning_service import learning_service

router = APIRouter()


@router.get("/system/report", response_model=SystemReport, tags=["System"])
async def get_system_report() -> SystemReport:
    """Gibt einen umfassenden Bericht über die Systemleistung und Parameter zurück."""
    stats = cache_service.query_cache.stats()
    total_queries = stats[0] + stats[1]
    hit_rate = stats[0] / total_queries if total_queries > 0 else 0.0

    avg_response = 0.0
    avg_confidence = 0.0
    if learning_service.metrics_history:
        avg_response = sum(
            m.total_duration_ms for m in learning_service.metrics_history
        ) / len(learning_service.metrics_history)
        avg_confidence = sum(
            m.confidence_score for m in learning_service.metrics_history
        ) / len(learning_service.metrics_history)

    return SystemReport(
        total_queries=total_queries,
        cache_hit_rate=round(hit_rate, 2),
        avg_response_time_ms=round(avg_response, 2),
        avg_confidence=round(avg_confidence, 2),
        adaptive_parameters=learning_service.adaptive_parameters,
    )
