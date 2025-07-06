# castigatio_backend/app/services/quality_service.py
"""
Service zur Bewertung der Qualität von generierten RAG-Antworten.
"""

import re
from typing import List

import numpy as np

from app.models.rag import AnswerQualityMetrics, SourceDocument


class QualityService:
    """Service zur Bewertung der Qualität von generierten RAG-Antworten."""

    def _calculate_source_reliability(self, sources: List[SourceDocument]) -> float:
        """Bewertet die Zuverlässigkeit der Quellen basierend auf ihren Relevanz-Scores."""
        if not sources:
            return 0.0
        scores = [s.relevance_score for s in sources if s.relevance_score is not None]
        if not scores:
            return 0.3  # Default, wenn keine Scores vorhanden sind
        avg_score = float(np.mean(scores))
        consistency_bonus = 1.0 - float(np.std(scores))
        return min((avg_score + consistency_bonus) / 2, 1.0)

    def _calculate_answer_relevance(self, question: str, answer: str) -> float:
        """Prüft, ob die Antwort die Schlüsselbegriffe der Frage enthält."""
        question_words = set(re.findall(r"\b\w+\b", question.lower()))
        answer_words = set(re.findall(r"\b\w+\b", answer.lower()))
        if not question_words:
            return 0.5
        common_words = question_words.intersection(answer_words)
        return len(common_words) / len(question_words)

    def _calculate_citation_quality(self, answer: str) -> float:
        """Bewertet die Qualität der Zitationen in der Antwort."""
        paragraph_mentions = len(re.findall(r"§\s*\d+", answer))
        article_mentions = len(re.findall(r"Art\.\s*\d+", answer))
        return min((paragraph_mentions + article_mentions) / 5.0, 1.0)

    def assess_answer_quality(
        self, query: str, answer: str, sources: List[SourceDocument]
    ) -> AnswerQualityMetrics:
        """Führt alle Qualitätsprüfungen durch und gibt ein Metrik-Objekt zurück."""
        source_reliability = self._calculate_source_reliability(sources)
        answer_relevance = self._calculate_answer_relevance(query, answer)
        citation_quality = self._calculate_citation_quality(answer)
        confidence_score = (
            source_reliability * 0.5 + answer_relevance * 0.3 + citation_quality * 0.2
        )
        return AnswerQualityMetrics(
            confidence_score=round(confidence_score, 4),
            source_reliability=round(source_reliability, 4),
            answer_relevance=round(answer_relevance, 4),
            citation_quality=round(citation_quality, 4),
        )


quality_service = QualityService()
