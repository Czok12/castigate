from typing import Any, Dict

from pydantic import BaseModel


class PerformanceMetrics(BaseModel):
    """Metriken für eine einzelne RAG-Anfrage."""

    total_duration_ms: float
    retrieval_duration_ms: float
    enhancement_duration_ms: float
    rerank_duration_ms: float
    generation_duration_ms: float
    quality_duration_ms: float
    confidence_score: float
    cache_hit: bool


class SystemReport(BaseModel):
    """Umfassender Systembericht."""

    total_queries: int
    cache_hit_rate: float
    avg_response_time_ms: float
    avg_confidence: float
    adaptive_parameters: Dict[str, Any]
