from app.models.query import EnhancedQuery
from app.services.query_enhancer_service import query_enhancer_service


class TestQueryEnhancerService:
    def test_enhance_query_basic(self):
        query = "Wie ist die Haftung bei Vertragsschluss gemäß § 433 BGB?"
        result = query_enhancer_service.enhance_query(query)
        assert isinstance(result, EnhancedQuery)
        assert result.original_query == query
        assert "haftung" in result.keywords
        assert "vertragsschluss" in result.keywords or "vertrag" in result.keywords
        assert "433" in result.entities.get("paragraph", [])
        assert "BGB" in result.entities.get("law", [])

    def test_enhance_query_synonyms(self):
        query = "Kündigung eines Vertrags"
        result = query_enhancer_service.enhance_query(query)
        # Synonyme sollten enthalten sein
        assert (
            "beendigung" in result.enhanced_query
            or "vereinbarung" in result.enhanced_query
        )

    def test_enhance_query_keywords_length(self):
        query = "abc de fg hij klm"
        result = query_enhancer_service.enhance_query(query)
        # Nur Wörter mit mehr als 3 Zeichen
        assert all(len(word) > 3 for word in result.keywords)
