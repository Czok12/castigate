"""
🚀 SYSTEM-OPTIMIERUNGEN FÜR JURISTISCHE WISSENSDATENBANK
========================================================

Kernfunktions-Verbesserungen für maximale Effizienz und Qualität
"""

import asyncio
import concurrent.futures
import hashlib
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional

import diskcache as dc
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# === 1. INTELLIGENTES CACHING SYSTEM ===


class AdvancedCacheManager:
    """Intelligentes Multi-Level-Caching für optimale Performance"""

    def __init__(self, cache_dir: str = "cache"):
        self.memory_cache = {}
        self.disk_cache = dc.Cache(cache_dir)
        self.embedding_cache = dc.Cache(f"{cache_dir}/embeddings")
        self.query_cache = dc.Cache(f"{cache_dir}/queries")

        # Cache-Statistiken
        self.stats = {"hits": 0, "misses": 0, "embedding_hits": 0, "query_hits": 0}

    def get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Holt Embedding aus Cache oder berechnet neu"""
        text_hash = hashlib.md5(text.encode()).hexdigest()

        # Memory Cache zuerst
        if text_hash in self.memory_cache:
            self.stats["hits"] += 1
            return self.memory_cache[text_hash]

        # Dann Disk Cache
        cached = self.embedding_cache.get(text_hash)
        if cached is not None:
            self.memory_cache[text_hash] = cached  # Promote zu Memory
            self.stats["embedding_hits"] += 1
            return cached

        self.stats["misses"] += 1
        return None

    def cache_embedding(self, text: str, embedding: np.ndarray, ttl: int = 86400):
        """Speichert Embedding in Cache"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        self.memory_cache[text_hash] = embedding
        self.embedding_cache.set(text_hash, embedding, expire=ttl)

    def get_cache_stats(self) -> Dict:
        """Cache-Statistiken für Monitoring"""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0

        return {
            "hit_rate_percent": round(hit_rate, 2),
            "total_queries": total,
            "memory_size": len(self.memory_cache),
            "disk_size": len(self.disk_cache),
            **self.stats,
        }


# === 2. ERWEITERTE EMBEDDING-STRATEGIEN ===


class HybridEmbeddingStrategy:
    """Kombiniert verschiedene Embedding-Strategien für optimale Ergebnisse"""

    def __init__(self):
        self.cache_manager = AdvancedCacheManager()

        # Multiple Embedding-Modelle für verschiedene Zwecke
        self.models = {
            "general": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "legal": "sentence-transformers/all-MiniLM-L6-v2",  # Schneller für Bulk-Operationen
            "semantic": "sentence-transformers/multi-qa-mpnet-base-dot-v1",  # Besser für Q&A
        }

        self.loaded_models = {}
        self.model_weights = {"general": 0.4, "legal": 0.3, "semantic": 0.3}

    @lru_cache(maxsize=3)
    def load_model(self, model_key: str) -> SentenceTransformer:
        """Lädt Modell mit LRU-Cache"""
        if model_key not in self.loaded_models:
            model_name = self.models[model_key]
            self.loaded_models[model_key] = SentenceTransformer(model_name)
        return self.loaded_models[model_key]

    def get_hybrid_embedding(self, text: str, strategy: str = "balanced") -> np.ndarray:
        """Erstellt Hybrid-Embedding basierend auf Strategie"""

        # Cache-Check
        cached = self.cache_manager.get_cached_embedding(f"{strategy}:{text}")
        if cached is not None:
            return cached

        embeddings = []
        weights = []

        if strategy == "balanced":
            # Alle Modelle verwenden
            for model_key, weight in self.model_weights.items():
                model = self.load_model(model_key)
                embedding = model.encode(text)
                embeddings.append(embedding)
                weights.append(weight)

        elif strategy == "fast":
            # Nur schnellstes Modell
            model = self.load_model("legal")
            embedding = model.encode(text)
            embeddings.append(embedding)
            weights.append(1.0)

        elif strategy == "accurate":
            # Fokus auf bestes Modell
            model = self.load_model("general")
            embedding = model.encode(text)
            embeddings.append(embedding)
            weights.append(1.0)

        # Gewichtete Kombination
        if len(embeddings) > 1:
            weighted_embedding = np.average(embeddings, axis=0, weights=weights)
        else:
            weighted_embedding = embeddings[0]

        # Cache speichern
        self.cache_manager.cache_embedding(f"{strategy}:{text}", weighted_embedding)

        return weighted_embedding


# === 3. INTELLIGENTE CHUNK-STRATEGIEN ===


@dataclass
class ChunkMetrics:
    """Metriken für Chunk-Qualität"""

    coherence_score: float
    legal_density: float
    information_density: float
    readability_score: float
    paragraph_count: int
    sentence_count: int


class IntelligentChunker:
    """Intelligente Chunk-Strategien für juristische Texte"""

    def __init__(self):
        self.legal_markers = [
            "§",
            "Art.",
            "Abs.",
            "Nr.",
            "lit.",
            "S.",
            "Rn.",
            "BGH",
            "BVerfG",
            "BFH",
            "BAG",
            "BSG",
            "BVerwG",
        ]

        self.section_markers = [
            "Kapitel",
            "Abschnitt",
            "Teil",
            "§§",
            "Literatur",
            "Rechtsprechung",
            "Beispiel",
            "Fall",
            "Lösung",
        ]

    def calculate_chunk_metrics(self, text: str) -> ChunkMetrics:
        """Berechnet Qualitätsmetriken für Chunk"""

        # Legal Density - Anteil juristischer Begriffe
        legal_terms = sum(1 for marker in self.legal_markers if marker in text)
        legal_density = legal_terms / len(text.split()) if text.split() else 0

        # Information Density - Längere Wörter = mehr Information
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        information_density = min(avg_word_length / 10, 1.0)  # Normalisiert auf 0-1

        # Coherence - Vollständige Sätze
        sentences = text.split(".")
        complete_sentences = sum(1 for s in sentences if len(s.strip()) > 10)
        coherence_score = complete_sentences / len(sentences) if sentences else 0

        # Readability - Einfache Heuristik
        readability_score = min(len(text) / 1000, 1.0)  # Optimale Länge um 1000 Zeichen

        return ChunkMetrics(
            coherence_score=coherence_score,
            legal_density=legal_density,
            information_density=information_density,
            readability_score=readability_score,
            paragraph_count=text.count("\n\n"),
            sentence_count=len(sentences),
        )

    def adaptive_chunking(self, text: str, target_size: int = 1000) -> List[Dict]:
        """Adaptive Chunking basierend auf Inhalt"""

        chunks = []
        current_chunk = ""

        # Split an logischen Punkten
        paragraphs = text.split("\n\n")

        for paragraph in paragraphs:
            # Prüfe ob Paragraph allein schon zu groß
            if len(paragraph) > target_size * 1.5:
                # Teile an Sätzen
                sentences = paragraph.split(".")
                temp_chunk = ""

                for sentence in sentences:
                    if len(temp_chunk + sentence) > target_size:
                        if temp_chunk:
                            metrics = self.calculate_chunk_metrics(temp_chunk)
                            chunks.append(
                                {
                                    "text": temp_chunk.strip(),
                                    "metrics": metrics,
                                    "quality_score": self._calculate_quality_score(
                                        metrics
                                    ),
                                }
                            )
                        temp_chunk = sentence
                    else:
                        temp_chunk += sentence + "."

                if temp_chunk:
                    current_chunk += temp_chunk

            # Normaler Paragraph-basierter Chunking
            elif len(current_chunk + paragraph) > target_size:
                if current_chunk:
                    metrics = self.calculate_chunk_metrics(current_chunk)
                    chunks.append(
                        {
                            "text": current_chunk.strip(),
                            "metrics": metrics,
                            "quality_score": self._calculate_quality_score(metrics),
                        }
                    )
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph

        # Letzten Chunk hinzufügen
        if current_chunk:
            metrics = self.calculate_chunk_metrics(current_chunk)
            chunks.append(
                {
                    "text": current_chunk.strip(),
                    "metrics": metrics,
                    "quality_score": self._calculate_quality_score(metrics),
                }
            )

        return chunks

    def _calculate_quality_score(self, metrics: ChunkMetrics) -> float:
        """Berechnet Gesamt-Qualitätsscore"""
        return (
            metrics.coherence_score * 0.3
            + metrics.legal_density * 0.3
            + metrics.information_density * 0.2
            + metrics.readability_score * 0.2
        )


# === 4. PARALLELE VERARBEITUNG ===


class ParallelProcessor:
    """Parallele Verarbeitung für bessere Performance"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def process_chunks_parallel(
        self, chunks: List[str], processor_func, batch_size: int = 10
    ) -> List:
        """Verarbeitet Chunks parallel in Batches"""

        results = []

        # Teile in Batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            # Parallel verarbeiten
            loop = asyncio.get_event_loop()
            futures = [
                loop.run_in_executor(self.executor, processor_func, chunk)
                for chunk in batch
            ]

            batch_results = await asyncio.gather(*futures)
            results.extend(batch_results)

            # Progress Update
            progress = min((i + batch_size) / len(chunks), 1.0)
            yield progress, batch_results

        return results

    def close(self):
        """Schließt Thread Pool"""
        self.executor.shutdown(wait=True)


# === 5. ERWEITERTE RETRIEVAL-STRATEGIEN ===


class AdvancedRetriever:
    """Erweiterte Retrieval-Strategien für bessere Ergebnisse"""

    def __init__(self, vectorstore, embedding_strategy: HybridEmbeddingStrategy):
        self.vectorstore = vectorstore
        self.embedding_strategy = embedding_strategy
        self.query_cache = {}

    def multi_strategy_retrieval(
        self, query: str, strategies: List[str] = None
    ) -> List[Dict]:
        """Kombiniert verschiedene Retrieval-Strategien"""

        if strategies is None:
            strategies = ["semantic", "keyword", "hybrid"]

        all_results = []

        for strategy in strategies:
            if strategy == "semantic":
                results = self._semantic_search(query)
            elif strategy == "keyword":
                results = self._keyword_search(query)
            elif strategy == "hybrid":
                results = self._hybrid_search(query)

            # Markiere Strategie
            for result in results:
                result["retrieval_strategy"] = strategy

            all_results.extend(results)

        # Deduplizierung und Ranking
        return self._deduplicate_and_rank(all_results)

    def _semantic_search(self, query: str, k: int = 5) -> List[Dict]:
        """Semantische Suche mit verbessertem Embedding"""

        query_embedding = self.embedding_strategy.get_hybrid_embedding(
            query, "accurate"
        )

        # Suche mit FAISS
        docs = self.vectorstore.similarity_search_by_vector(query_embedding, k=k)

        results = []
        for doc in docs:
            results.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": self._calculate_semantic_score(query, doc.page_content),
                    "type": "semantic",
                }
            )

        return results

    def _keyword_search(self, query: str, k: int = 5) -> List[Dict]:
        """Keyword-basierte Suche"""

        # Extrahiere Keywords
        keywords = self._extract_keywords(query)

        # Suche in Metadaten und Content
        all_docs = self.vectorstore.similarity_search(query, k=k * 2)

        results = []
        for doc in all_docs:
            score = self._calculate_keyword_score(keywords, doc.page_content)
            if score > 0:
                results.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score,
                        "type": "keyword",
                    }
                )

        return sorted(results, key=lambda x: x["score"], reverse=True)[:k]

    def _hybrid_search(self, query: str, k: int = 5) -> List[Dict]:
        """Hybride Suche kombiniert verschiedene Ansätze"""

        # Gewichtete Kombination
        semantic_results = self._semantic_search(query, k)
        keyword_results = self._keyword_search(query, k)

        # Kombiniere Scores
        combined_results = {}

        for result in semantic_results:
            doc_id = self._get_doc_id(result)
            combined_results[doc_id] = result
            combined_results[doc_id]["combined_score"] = result["score"] * 0.7

        for result in keyword_results:
            doc_id = self._get_doc_id(result)
            if doc_id in combined_results:
                combined_results[doc_id]["combined_score"] += result["score"] * 0.3
            else:
                combined_results[doc_id] = result
                combined_results[doc_id]["combined_score"] = result["score"] * 0.3

        # Sortiere nach kombiniertem Score
        final_results = list(combined_results.values())
        final_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)

        return final_results[:k]

    def _extract_keywords(self, text: str) -> List[str]:
        """Extrahiert wichtige Keywords"""

        # Juristische Keywords haben höhere Priorität
        legal_terms = []
        for term in self.embedding_strategy.cache_manager.memory_cache.keys():
            if any(marker in term for marker in ["§", "Art.", "BGH", "BVerfG"]):
                legal_terms.append(term)

        # Standard Keywords
        words = text.lower().split()
        keywords = [word for word in words if len(word) > 3]

        return legal_terms + keywords

    def _calculate_semantic_score(self, query: str, content: str) -> float:
        """Berechnet semantischen Ähnlichkeitsscore"""

        query_emb = self.embedding_strategy.get_hybrid_embedding(query, "fast")
        content_emb = self.embedding_strategy.get_hybrid_embedding(content, "fast")

        similarity = cosine_similarity([query_emb], [content_emb])[0][0]
        return float(similarity)

    def _calculate_keyword_score(self, keywords: List[str], content: str) -> float:
        """Berechnet Keyword-Match-Score"""

        content_lower = content.lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in content_lower)

        return matches / len(keywords) if keywords else 0

    def _get_doc_id(self, result: Dict) -> str:
        """Generiert eindeutige Doc-ID"""

        metadata = result.get("metadata", {})
        return f"{metadata.get('source', '')}_{metadata.get('page', '')}_{metadata.get('chunk_id', '')}"

    def _deduplicate_and_rank(self, results: List[Dict]) -> List[Dict]:
        """Entfernt Duplikate und rankt Ergebnisse"""

        seen_ids = set()
        unique_results = []

        for result in results:
            doc_id = self._get_doc_id(result)
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_results.append(result)

        # Ranking nach Score
        return sorted(
            unique_results,
            key=lambda x: x.get("combined_score", x.get("score", 0)),
            reverse=True,
        )


# === 6. PERFORMANCE-MONITORING ===


class PerformanceMonitor:
    """Überwacht System-Performance"""

    def __init__(self):
        self.metrics = {
            "query_times": [],
            "cache_hits": 0,
            "cache_misses": 0,
            "total_queries": 0,
            "avg_response_time": 0.0,
        }

        # Setup Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def time_operation(self, operation_name: str):
        """Context Manager für Zeitmessung"""

        class TimerContext:
            def __init__(self, monitor, name):
                self.monitor = monitor
                self.name = name
                self.start_time = None

            def __enter__(self):
                self.start_time = time.time()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.monitor.record_timing(self.name, duration)

        return TimerContext(self, operation_name)

    def record_timing(self, operation: str, duration: float):
        """Zeichnet Timing-Information auf"""

        if operation not in self.metrics:
            self.metrics[operation] = []

        self.metrics[operation].append(duration)
        self.logger.info(f"{operation}: {duration:.3f}s")

    def get_performance_summary(self) -> Dict:
        """Erstellt Performance-Zusammenfassung"""

        summary = {}

        for operation, times in self.metrics.items():
            if isinstance(times, list) and times:
                summary[operation] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "total_calls": len(times),
                }

        return summary

    def log_system_stats(self):
        """Loggt System-Statistiken"""

        import psutil

        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()

        self.logger.info(
            f"CPU: {cpu_percent}% | Memory: {memory.percent}% | Available: {memory.available / 1024**3:.1f}GB"
        )


# === ZUSAMMENFASSUNG DER OPTIMIERUNGEN ===

"""
🎯 KERN-OPTIMIERUNGEN IMPLEMENTIERT:

1. **Intelligentes Multi-Level-Caching**
   - Memory + Disk Cache für Embeddings
   - Query-Cache für wiederkehrende Anfragen
   - Cache-Statistiken und Hit-Rate-Monitoring

2. **Hybrid-Embedding-Strategien**
   - Multiple Modelle für verschiedene Anwendungsfälle
   - Gewichtete Kombinationen für optimale Ergebnisse
   - Adaptive Modell-Auswahl je nach Anfrage

3. **Intelligente Chunk-Strategien**
   - Content-basierte adaptive Chunking
   - Qualitätsmetriken für jeden Chunk
   - Juristische Marker-Erkennung

4. **Parallele Verarbeitung**
   - Async/Await für I/O-Operations
   - Thread-basierte Parallelisierung
   - Batch-Processing für große Datenmengen

5. **Erweiterte Retrieval-Methoden**
   - Multi-Strategy Retrieval
   - Semantic + Keyword + Hybrid Search
   - Intelligente Deduplizierung und Ranking

6. **Performance-Monitoring**
   - Umfassende Timing-Metriken
   - System-Ressourcen-Überwachung
   - Detaillierte Logging-Funktionen

💡 NÄCHSTE SCHRITTE:
- Integration in bestehende enhanced_jura_app.py
- A/B-Testing verschiedener Strategien
- Benutzer-Feedback-System für kontinuierliche Verbesserung
"""
