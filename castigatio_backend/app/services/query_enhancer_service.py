import re
from typing import Dict, List

from app.models.query import EnhancedQuery


class QueryEnhancerService:
    """Service zur semantischen Analyse und Erweiterung von Anfragen."""

    def __init__(self):
        # Einfache juristische Synonyme und Keywords
        self.synonyms = {
            "haftung": ["verantwortlichkeit"],
            "vertrag": ["vereinbarung", "kontrakt"],
            "kündigung": ["beendigung"],
            "schadenersatz": ["schadensersatz", "wiedergutmachung"],
        }
        self.legal_patterns = {
            "paragraph": r"§\s*(\d+[a-z]?)",
            "law": r"\b(BGB|StGB|GG|HGB|ZPO)\b",
        }

    def _extract_keywords(self, query: str) -> List[str]:
        """Extrahiert simple Keywords (Wörter länger als 3 Zeichen)."""
        return [word for word in re.findall(r"\b\w+\b", query.lower()) if len(word) > 3]

    def _expand_synonyms(self, words: List[str]) -> List[str]:
        """Erweitert eine Wortliste um vordefinierte Synonyme."""
        expanded = set(words)
        for word in words:
            if word in self.synonyms:
                expanded.update(self.synonyms[word])
        return list(expanded)

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extrahiert juristische Entitäten mittels Regex."""
        entities = {}
        for entity_type, pattern in self.legal_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                entities[entity_type] = list(set(matches))
        return entities

    def enhance_query(self, query: str) -> EnhancedQuery:
        """Führt eine vollständige Analyse und Erweiterung durch."""
        original_query = query
        keywords = self._extract_keywords(original_query)
        expanded_keywords = self._expand_synonyms(keywords)

        # Erstelle eine erweiterte Query für die semantische Suche
        enhanced_query = " ".join(expanded_keywords)

        entities = self._extract_entities(original_query)

        return EnhancedQuery(
            original_query=original_query,
            enhanced_query=enhanced_query,
            keywords=keywords,  # Die originalen Keywords für das Reranking
            entities=entities,
        )


query_enhancer_service = QueryEnhancerService()
