"""
🧠 INTELLIGENTES HIERARCHISCHES CACHING SYSTEM
==============================================

Multi-Level-Caching mit adaptiver TTL, Semantic Similarity und Performance-Optimierung
"""

import hashlib
import json
import pickle
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class CacheEntry:
    """Cache-Eintrag mit Metadaten"""

    query: str
    query_hash: str
    query_embedding: Any
    answer: str
    sources: List[Dict]
    confidence_score: float
    retrieval_method: str
    timestamp: float
    access_count: int
    last_accessed: float
    performance_metrics: Dict
    ttl: float  # Time To Live in seconds
    cache_level: str  # "hot", "warm", "cold"


class SemanticSimilarityMatcher:
    """Semantic Similarity für Cache-Lookups"""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.sentence_transformer = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )

    def get_query_embedding(self, query: str):
        """Erstellt Query-Embedding"""
        return self.sentence_transformer.encode(query)

    def find_similar_queries(
        self, query_embedding, cached_entries: List[CacheEntry], top_k: int = 3
    ) -> List[Tuple[CacheEntry, float]]:
        """Findet semantisch ähnliche Queries im Cache"""

        similarities = []

        for entry in cached_entries:
            similarity = self.cosine_similarity(query_embedding, entry.query_embedding)
            if similarity >= self.similarity_threshold:
                similarities.append((entry, similarity))

        # Sortiert nach Similarity-Score
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Berechnet Cosine Similarity"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class AdaptiveTTLCalculator:
    """Adaptive TTL-Berechnung basierend auf Performance und Nutzung"""

    def __init__(self):
        self.base_ttl = 3600  # 1 Stunde
        self.max_ttl = 86400  # 24 Stunden
        self.min_ttl = 300  # 5 Minuten

        # Gewichtungsfaktoren
        self.confidence_weight = 0.3
        self.access_weight = 0.25
        self.performance_weight = 0.25
        self.recency_weight = 0.2

    def calculate_ttl(self, entry: CacheEntry) -> float:
        """Berechnet adaptive TTL basierend auf Entry-Eigenschaften"""

        # Confidence-Faktor (höhere Confidence = längere TTL)
        confidence_factor = min(entry.confidence_score / 0.9, 1.0)

        # Access-Faktor (häufiger Zugriff = längere TTL)
        access_factor = min(entry.access_count / 10, 1.0)

        # Performance-Faktor (bessere Performance = längere TTL)
        response_time = entry.performance_metrics.get("response_time", 2.0)
        performance_factor = max(0.1, min(2.0 / response_time, 1.0))

        # Recency-Faktor (neuere Einträge = längere TTL)
        age_hours = (time.time() - entry.timestamp) / 3600
        recency_factor = max(0.1, min(24 / max(age_hours, 1), 1.0))

        # Gewichtete Kombination
        total_factor = (
            confidence_factor * self.confidence_weight
            + access_factor * self.access_weight
            + performance_factor * self.performance_weight
            + recency_factor * self.recency_weight
        )

        # TTL berechnen
        calculated_ttl = self.base_ttl * total_factor
        return max(self.min_ttl, min(calculated_ttl, self.max_ttl))


class HierarchicalCacheManager:
    """Hierarchisches Cache-Management mit SQLite Backend"""

    def __init__(self, cache_db_path: str = "intelligent_cache.db"):
        self.cache_db_path = cache_db_path
        self.semantic_matcher = SemanticSimilarityMatcher()
        self.ttl_calculator = AdaptiveTTLCalculator()

        # Cache-Level Konfiguration
        self.cache_levels = {
            "hot": {"max_entries": 50, "ttl_multiplier": 1.5},
            "warm": {"max_entries": 200, "ttl_multiplier": 1.0},
            "cold": {"max_entries": 1000, "ttl_multiplier": 0.5},
        }

        # Statistiken
        self.cache_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "semantic_hits": 0,
            "level_stats": defaultdict(lambda: {"hits": 0, "entries": 0}),
        }

        self._initialize_database()

    def _initialize_database(self):
        """Initialisiert SQLite-Datenbank für Cache"""

        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT UNIQUE,
                query TEXT,
                query_embedding BLOB,
                answer TEXT,
                sources TEXT,
                confidence_score REAL,
                retrieval_method TEXT,
                timestamp REAL,
                access_count INTEGER,
                last_accessed REAL,
                performance_metrics TEXT,
                ttl REAL,
                cache_level TEXT,
                expires_at REAL
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_query_hash ON cache_entries(query_hash)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cache_level ON cache_entries(cache_level)
        """
        )

        conn.commit()
        conn.close()

    def get_query_hash(self, query: str, mode: str = "balanced") -> str:
        """Erstellt Hash für Query + Modus"""
        combined = f"{query}:{mode}"
        return hashlib.md5(combined.encode()).hexdigest()

    def get_cached_answer(
        self, query: str, mode: str = "balanced", use_semantic_matching: bool = True
    ) -> Optional[Dict]:
        """Sucht nach gecachter Antwort"""

        self.cache_stats["total_requests"] += 1

        # 1. Exakter Cache-Lookup
        query_hash = self.get_query_hash(query, mode)
        exact_match = self._get_exact_match(query_hash)

        if exact_match:
            self.cache_stats["cache_hits"] += 1
            self.cache_stats["level_stats"][exact_match["cache_level"]]["hits"] += 1
            return exact_match

        # 2. Semantic Similarity Lookup
        if use_semantic_matching:
            semantic_match = self._get_semantic_match(query)
            if semantic_match:
                self.cache_stats["semantic_hits"] += 1
                return semantic_match

        # 3. Cache Miss
        self.cache_stats["cache_misses"] += 1
        return None

    def _get_exact_match(self, query_hash: str) -> Optional[Dict]:
        """Sucht nach exaktem Cache-Match"""

        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM cache_entries 
            WHERE query_hash = ? AND expires_at > ?
        """,
            (query_hash, time.time()),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            # Access-Count aktualisieren
            self._update_access_count(query_hash)
            return self._row_to_dict(row)

        return None

    def _get_semantic_match(self, query: str) -> Optional[Dict]:
        """Sucht nach semantisch ähnlicher gecachter Antwort"""

        query_embedding = self.semantic_matcher.get_query_embedding(query)

        # Alle aktiven Cache-Einträge laden
        active_entries = self._get_active_entries()

        if not active_entries:
            return None

        # Semantic Similarity suchen
        similar_entries = self.semantic_matcher.find_similar_queries(
            query_embedding, active_entries, top_k=1
        )

        if similar_entries:
            best_match, similarity = similar_entries[0]

            # Access-Count aktualisieren
            self._update_access_count(best_match.query_hash)

            return {
                "query": best_match.query,
                "answer": best_match.answer,
                "sources": best_match.sources,
                "confidence_score": best_match.confidence_score,
                "retrieval_method": f"semantic_cache_{similarity:.3f}",
                "cache_level": best_match.cache_level,
                "semantic_similarity": similarity,
            }

        return None

    def cache_answer(
        self,
        query: str,
        answer: str,
        sources: List[Dict],
        confidence_score: float,
        retrieval_method: str,
        performance_metrics: Dict,
        mode: str = "balanced",
    ) -> Dict:
        """Cached eine neue Antwort"""

        query_hash = self.get_query_hash(query, mode)
        query_embedding = self.semantic_matcher.get_query_embedding(query)

        # Cache-Entry erstellen
        entry = CacheEntry(
            query=query,
            query_hash=query_hash,
            query_embedding=query_embedding,
            answer=answer,
            sources=sources,
            confidence_score=confidence_score,
            retrieval_method=retrieval_method,
            timestamp=time.time(),
            access_count=1,
            last_accessed=time.time(),
            performance_metrics=performance_metrics,
            ttl=0,  # Wird berechnet
            cache_level="",  # Wird bestimmt
        )

        # TTL berechnen
        entry.ttl = self.ttl_calculator.calculate_ttl(entry)

        # Cache-Level bestimmen
        entry.cache_level = self._determine_cache_level(entry)

        # In Datenbank speichern
        self._save_to_database(entry)

        # Cache-Wartung
        self._maintain_cache()

        return {
            "cached": True,
            "cache_level": entry.cache_level,
            "ttl": entry.ttl,
            "expires_at": entry.timestamp + entry.ttl,
        }

    def _determine_cache_level(self, entry: CacheEntry) -> str:
        """Bestimmt Cache-Level basierend auf Qualität"""

        # Hot Cache: Hohe Confidence + gute Performance
        if (
            entry.confidence_score >= 0.8
            and entry.performance_metrics.get("response_time", 10) <= 2.0
        ):
            return "hot"

        # Warm Cache: Mittlere Qualität
        elif entry.confidence_score >= 0.6:
            return "warm"

        # Cold Cache: Niedrige Qualität
        else:
            return "cold"

    def _save_to_database(self, entry: CacheEntry):
        """Speichert Entry in SQLite-Datenbank"""

        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO cache_entries (
                query_hash, query, query_embedding, answer, sources,
                confidence_score, retrieval_method, timestamp, access_count,
                last_accessed, performance_metrics, ttl, cache_level, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                entry.query_hash,
                entry.query,
                pickle.dumps(entry.query_embedding),
                entry.answer,
                json.dumps(entry.sources),
                entry.confidence_score,
                entry.retrieval_method,
                entry.timestamp,
                entry.access_count,
                entry.last_accessed,
                json.dumps(entry.performance_metrics),
                entry.ttl,
                entry.cache_level,
                entry.timestamp + entry.ttl,
            ),
        )

        conn.commit()
        conn.close()

    def _get_active_entries(self) -> List[CacheEntry]:
        """Lädt alle aktiven Cache-Einträge"""

        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM cache_entries 
            WHERE expires_at > ?
            ORDER BY last_accessed DESC
            LIMIT 100
        """,
            (time.time(),),
        )

        rows = cursor.fetchall()
        conn.close()

        entries = []
        for row in rows:
            entry = CacheEntry(
                query=row[2],
                query_hash=row[1],
                query_embedding=pickle.loads(row[3]),
                answer=row[4],
                sources=json.loads(row[5]),
                confidence_score=row[6],
                retrieval_method=row[7],
                timestamp=row[8],
                access_count=row[9],
                last_accessed=row[10],
                performance_metrics=json.loads(row[11]),
                ttl=row[12],
                cache_level=row[13],
            )
            entries.append(entry)

        return entries

    def _update_access_count(self, query_hash: str):
        """Aktualisiert Access-Count für Cache-Entry"""

        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE cache_entries 
            SET access_count = access_count + 1, last_accessed = ?
            WHERE query_hash = ?
        """,
            (time.time(), query_hash),
        )

        conn.commit()
        conn.close()

    def _maintain_cache(self):
        """Cache-Wartung: Alte Einträge löschen, Level-Limits einhalten"""

        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()

        # 1. Abgelaufene Einträge löschen
        cursor.execute("DELETE FROM cache_entries WHERE expires_at < ?", (time.time(),))

        # 2. Cache-Level Limits einhalten
        for level, config in self.cache_levels.items():
            cursor.execute(
                """
                DELETE FROM cache_entries 
                WHERE cache_level = ? AND id NOT IN (
                    SELECT id FROM cache_entries 
                    WHERE cache_level = ? 
                    ORDER BY last_accessed DESC 
                    LIMIT ?
                )
            """,
                (level, level, config["max_entries"]),
            )

        conn.commit()
        conn.close()

    def _row_to_dict(self, row) -> Dict:
        """Konvertiert DB-Row zu Dictionary"""

        return {
            "query": row[2],
            "answer": row[4],
            "sources": json.loads(row[5]),
            "confidence_score": row[6],
            "retrieval_method": row[7],
            "cache_level": row[13],
            "access_count": row[9],
            "last_accessed": row[10],
        }

    def get_cache_statistics(self) -> Dict:
        """Liefert detaillierte Cache-Statistiken"""

        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()

        # Gesamtanzahl Einträge
        cursor.execute(
            "SELECT COUNT(*) FROM cache_entries WHERE expires_at > ?", (time.time(),)
        )
        total_entries = cursor.fetchone()[0]

        # Einträge pro Level
        cursor.execute(
            """
            SELECT cache_level, COUNT(*) 
            FROM cache_entries 
            WHERE expires_at > ? 
            GROUP BY cache_level
        """,
            (time.time(),),
        )
        level_counts = dict(cursor.fetchall())

        # Durchschnittliche Confidence
        cursor.execute(
            """
            SELECT AVG(confidence_score) 
            FROM cache_entries 
            WHERE expires_at > ?
        """,
            (time.time(),),
        )
        avg_confidence = cursor.fetchone()[0] or 0

        conn.close()

        # Hit-Rate berechnen
        hit_rate = (
            self.cache_stats["cache_hits"] / max(self.cache_stats["total_requests"], 1)
        ) * 100

        semantic_hit_rate = (
            self.cache_stats["semantic_hits"]
            / max(self.cache_stats["total_requests"], 1)
        ) * 100

        return {
            "total_entries": total_entries,
            "level_distribution": level_counts,
            "average_confidence": round(avg_confidence, 3),
            "hit_rate": round(hit_rate, 2),
            "semantic_hit_rate": round(semantic_hit_rate, 2),
            "total_requests": self.cache_stats["total_requests"],
            "cache_hits": self.cache_stats["cache_hits"],
            "cache_misses": self.cache_stats["cache_misses"],
            "semantic_hits": self.cache_stats["semantic_hits"],
        }

    def clear_cache(self, level: Optional[str] = None):
        """Löscht Cache-Einträge"""

        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()

        if level:
            cursor.execute("DELETE FROM cache_entries WHERE cache_level = ?", (level,))
        else:
            cursor.execute("DELETE FROM cache_entries")

        conn.commit()
        conn.close()

        # Statistiken zurücksetzen
        self.cache_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "semantic_hits": 0,
            "level_stats": defaultdict(lambda: {"hits": 0, "entries": 0}),
        }
