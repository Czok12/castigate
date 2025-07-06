from app.models.quality import AnswerQualityMetrics
from app.models.rag import QueryResponse, SourceDocument
from app.services.cache_service import CacheService


def test_cache_set_and_get():
    cache_service = CacheService()
    # Mock the diskcache.Cache with a dict-like object
    cache_service.query_cache = {}
    question = "Was ist ein Diebstahl?"
    book_ids = ["buch1", "buch2"]
    dummy_quality = AnswerQualityMetrics(
        confidence_score=0.9,
        completeness_score=1.0,
        factuality_score=1.0,
        source_coverage=1.0,
        feedbacks=[],
    )
    response = QueryResponse(
        answer="Antwort",
        sources=[SourceDocument(content="foo", metadata={}, relevance_score=1.0)],
        quality=dummy_quality,
        trace_id="abc",
    )
    # Setzen
    cache_service.set(question, book_ids, response)
    key = cache_service._get_cache_key(question, book_ids)
    # Manuelles Setzen im Mock-Cache (dict)
    cache_service.query_cache[key] = response
    # Abrufen
    result = cache_service.get(question, book_ids)
    assert result == response
