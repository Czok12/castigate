"""
Unit-Tests für den QualityService.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
from models.rag import SourceDocument
from services.quality_service import quality_service


def test_assess_answer_quality_typical():
    q = "Was ist § 433 BGB?"
    a = "§ 433 BGB regelt den Kaufvertrag."
    docs = [SourceDocument(content="a", metadata={}, relevance_score=0.9)]
    metrics = quality_service.assess_answer_quality(q, a, docs)
    assert 0.0 <= metrics.confidence_score <= 1.0
    assert 0.0 <= metrics.source_reliability <= 1.0
    assert 0.0 <= metrics.answer_relevance <= 1.0
    assert 0.0 <= metrics.citation_quality <= 1.0


def test_assess_answer_quality_no_sources():
    q = "Was ist § 433 BGB?"
    a = "§ 433 BGB regelt den Kaufvertrag."
    docs: list[SourceDocument] = []
    metrics = quality_service.assess_answer_quality(q, a, docs)
    assert metrics.source_reliability == 0.0
    assert 0.0 <= metrics.confidence_score <= 1.0


def test_assess_answer_quality_no_relevance():
    q = "Was ist § 433 BGB?"
    a = "Unrelated answer."
    docs = [SourceDocument(content="a", metadata={}, relevance_score=0.5)]
    metrics = quality_service.assess_answer_quality(q, a, docs)
    assert metrics.answer_relevance < 0.5
