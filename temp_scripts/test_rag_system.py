#!/usr/bin/env python3
"""
🧪 TEST SUITE FÜR JURISTISCHE WISSENSDATENBANK
=============================================

Umfassende Tests für alle Kernkomponenten
"""
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Füge das Projekt-Verzeichnis zum Python-Pfad hinzu
project_dir = Path(__file__).parent
sys.path.append(str(project_dir))


class TestBasicRAGSystem:
    """Tests für das grundlegende RAG-System"""

    def test_retriever_loading(self):
        """Test, ob der Retriever korrekt geladen wird"""
        try:
            from app import load_retriever

            # Mock für FAISS-Datenbank
            with patch("app.FAISS.load_local") as mock_load:
                mock_load.return_value = Mock()
                retriever = load_retriever()
                assert retriever is not None
        except ImportError:
            pytest.skip("app.py nicht verfügbar")

    def test_llm_loading(self):
        """Test, ob das LLM korrekt geladen wird"""
        try:
            from app import load_llm

            with patch("app.OllamaLLM") as mock_llm:
                mock_llm.return_value = Mock()
                llm = load_llm()
                assert llm is not None
        except ImportError:
            pytest.skip("app.py nicht verfügbar")


class TestIntelligentCaching:
    """Tests für das intelligente Caching-System"""

    @pytest.fixture
    def temp_cache_db(self):
        """Temporäre Cache-Datenbank für Tests"""
        temp_dir = tempfile.mkdtemp()
        cache_path = Path(temp_dir) / "test_cache.db"
        yield str(cache_path)
        shutil.rmtree(temp_dir)

    def test_cache_initialization(self, temp_cache_db):
        """Test Cache-Initialisierung"""
        try:
            from intelligent_caching_system import HierarchicalCacheManager

            cache = HierarchicalCacheManager(temp_cache_db)
            assert cache is not None
            assert Path(temp_cache_db).exists()
        except ImportError:
            pytest.skip("intelligent_caching_system nicht verfügbar")

    def test_cache_operations(self, temp_cache_db):
        """Test grundlegende Cache-Operationen"""
        try:
            from intelligent_caching_system import HierarchicalCacheManager

            cache = HierarchicalCacheManager(temp_cache_db)

            # Test Cache-Miss
            result = cache.get_cached_answer("Test Query")
            assert result is None

            # Test Cache-Hit nach Store
            test_data = {
                "answer": "Test Answer",
                "confidence": 0.8,
                "sources": ["source1"],
                "processing_time": 1.0,
            }

            cache.store_answer("Test Query", test_data, mode="balanced")
            cached_result = cache.get_cached_answer("Test Query", mode="balanced")

            assert cached_result is not None
            assert cached_result["answer"] == "Test Answer"

        except ImportError:
            pytest.skip("intelligent_caching_system nicht verfügbar")


class TestQueryEnhancement:
    """Tests für die Query-Enhancement-Engine"""

    def test_synonym_expansion(self):
        """Test Synonym-Erweiterung"""
        try:
            from semantic_query_enhancer import JuristischeSynonymErweiterung

            expander = JuristischeSynonymErweiterung()

            synonyms, terms = expander.expand_query("Vertrag")
            assert "Vereinbarung" in synonyms or "Kontrakt" in synonyms

        except ImportError:
            pytest.skip("semantic_query_enhancer nicht verfügbar")

    def test_intent_classification(self):
        """Test Intent-Klassifizierung"""
        try:
            from semantic_query_enhancer import QueryIntentClassifier

            classifier = QueryIntentClassifier()

            intent, confidence = classifier.classify("Was ist ein Kaufvertrag?")
            assert intent in ["definition", "voraussetzungen", "rechtslage"]
            assert 0 <= confidence <= 1

        except ImportError:
            pytest.skip("semantic_query_enhancer nicht verfügbar")


class TestAdvancedRetrieval:
    """Tests für die erweiterte Retrieval-Engine"""

    def test_query_processor_initialization(self):
        """Test Query-Processor Initialisierung"""
        try:
            from advanced_retrieval_engine import QueryProcessor

            processor = QueryProcessor()
            assert processor is not None

        except ImportError:
            pytest.skip("advanced_retrieval_engine nicht verfügbar")

    def test_legal_entity_extraction(self):
        """Test juristische Entity-Extraktion"""
        try:
            from advanced_retrieval_engine import QueryProcessor

            processor = QueryProcessor()

            test_query = "§ 433 BGB und Art. 3 GG"
            entities = processor.extract_legal_entities(test_query)

            assert "paragraphs" in entities
            assert "articles" in entities

        except ImportError:
            pytest.skip("advanced_retrieval_engine nicht verfügbar")


class TestPerformanceMonitoring:
    """Tests für Performance-Monitoring"""

    def test_metrics_collection(self):
        """Test Metriken-Sammlung"""
        try:
            from performance_monitoring import PerformanceMetrics, PerformanceMonitor

            monitor = PerformanceMonitor(window_size=10)

            # Test Metrik-Recording
            test_metrics = PerformanceMetrics(
                timestamp=1234567890.0,
                response_time=1.5,
                retrieval_time=0.8,
                generation_time=0.7,
                cache_hit_rate=0.85,
                confidence_score=0.9,
                query_length=50,
                result_count=4,
            )

            monitor.record_metrics(test_metrics)
            summary = monitor.get_current_performance_summary()

            assert summary is not None
            assert "avg_response_time" in summary

        except ImportError:
            pytest.skip("performance_monitoring nicht verfügbar")


class TestJuristischeValidierung:
    """Tests für juristische Validierung"""

    def test_legal_citation_validation(self):
        """Test Validierung juristischer Zitate"""
        try:
            from validation import Juristische_Validierung

            validator = Juristische_Validierung()

            test_text = "Nach § 433 BGB und Art. 3 GG gilt folgendes..."
            result = validator.validate_legal_citation(test_text)

            assert result["has_legal_citations"] is True
            assert result["citation_count"] >= 2

        except ImportError:
            pytest.skip("validation nicht verfügbar")

    def test_answer_quality_check(self):
        """Test Antwort-Qualitätsprüfung"""
        try:
            from validation import Juristische_Validierung

            validator = Juristische_Validierung()

            good_answer = "Nach § 433 BGB ist ein Kaufvertrag ein Vertrag..."
            quality = validator.check_answer_quality(good_answer)

            assert "quality_score" in quality
            assert 0 <= quality["quality_score"] <= 1

        except ImportError:
            pytest.skip("validation nicht verfügbar")


class TestIntegration:
    """Integrationstests für das Gesamtsystem"""

    @patch("streamlit.cache_resource")
    def test_unified_app_imports(self, mock_cache):
        """Test, ob unified_jura_app importiert werden kann"""
        try:
            mock_cache.side_effect = lambda func: func  # Bypass caching decorator
            import unified_jura_app

            assert hasattr(unified_jura_app, "main")

        except ImportError as e:
            pytest.fail(f"unified_jura_app kann nicht importiert werden: {e}")

    def test_requirements_availability(self):
        """Test, ob alle wichtigen Requirements verfügbar sind"""
        required_packages = [
            "streamlit",
            "langchain",
            "sentence_transformers",
            "faiss",  # faiss-cpu
            "fitz",  # PyMuPDF
            "pandas",
            "numpy",
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            pytest.skip(f"Fehlende Packages: {missing_packages}")


# Performance-Tests
class TestPerformance:
    """Performance-Tests für kritische Komponenten"""

    def test_query_enhancement_speed(self):
        """Test Geschwindigkeit der Query-Enhancement"""
        try:
            import time

            from semantic_query_enhancer import SemanticQueryEnhancer

            enhancer = SemanticQueryEnhancer()
            test_query = "Was ist ein Kaufvertrag?"

            start_time = time.time()
            result = enhancer.enhance_query(test_query)
            end_time = time.time()

            processing_time = end_time - start_time
            assert processing_time < 1.0  # Sollte unter 1 Sekunde dauern
            assert result is not None

        except ImportError:
            pytest.skip("semantic_query_enhancer nicht verfügbar")


if __name__ == "__main__":
    # Tests ausführen
    pytest.main([__file__, "-v", "--tb=short"])
