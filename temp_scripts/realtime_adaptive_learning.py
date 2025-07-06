"""
🧠 REAL-TIME ADAPTIVE LEARNING ENGINE
====================================

Kontinuierliches Lernen in Echtzeit aus jeder Nutzerinteraktion mit sofortiger Systemanpassung
"""

import json
import sqlite3
import statistics
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class InteractionEvent:
    """Einzelne Nutzerinteraktion"""

    timestamp: float
    query: str
    enhanced_query: str
    intent: str
    response_time: float
    confidence_score: float
    retrieval_method: str
    sources_count: int
    user_satisfaction: Optional[float] = None  # Implicit durch Verhalten
    session_id: str = "default"
    interaction_type: str = "query"  # query, refinement, follow_up
    context_length: int = 0
    cache_hit: bool = False


@dataclass
class LearningInsight:
    """Gelerntes Pattern"""

    pattern_id: str
    pattern_type: str  # query_optimization, retrieval_tuning, caching_strategy
    confidence: float
    impact_score: float
    usage_count: int
    success_rate: float
    last_applied: float
    parameters: Dict[str, Any]


class RealTimeLearningEngine:
    """Haupt-Engine für Real-Time Learning"""

    def __init__(self, db_path: str = "realtime_learning.db"):
        self.db_path = db_path
        self.interaction_buffer = deque(maxlen=1000)
        self.learning_insights = {}
        self.active_experiments = {}

        # Real-Time Processing
        self.processing_queue = deque()
        self.is_processing = True
        self.processing_thread = None
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Learning Konfiguration
        self.learning_config = {
            "min_interactions_for_pattern": 5,
            "confidence_threshold": 0.7,
            "adaptation_rate": 0.1,
            "experiment_duration": 3600,  # 1 Stunde
            "max_concurrent_experiments": 3,
        }

        # Performance-Tracking für verschiedene Strategien
        self.strategy_performance = defaultdict(
            lambda: {
                "total_usage": 0,
                "success_count": 0,
                "avg_response_time": 0.0,
                "avg_confidence": 0.0,
                "user_satisfaction": 0.0,
            }
        )

        # Adaptive Parameter
        self.adaptive_parameters = {
            "retrieval_k": 5,
            "cache_ttl_multiplier": 1.0,
            "confidence_threshold": 0.6,
            "semantic_similarity_threshold": 0.85,
            "query_expansion_aggressiveness": 0.5,
        }

        self._initialize_database()
        self._start_processing()

    def _initialize_database(self):
        """Initialisiert SQLite-Datenbank für persistentes Lernen"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                query TEXT,
                enhanced_query TEXT,
                intent TEXT,
                response_time REAL,
                confidence_score REAL,
                retrieval_method TEXT,
                sources_count INTEGER,
                user_satisfaction REAL,
                session_id TEXT,
                interaction_type TEXT,
                context_length INTEGER,
                cache_hit BOOLEAN
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_insights (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT,
                confidence REAL,
                impact_score REAL,
                usage_count INTEGER,
                success_rate REAL,
                last_applied REAL,
                parameters TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS adaptive_parameters (
                parameter_name TEXT PRIMARY KEY,
                current_value REAL,
                last_updated REAL,
                update_reason TEXT
            )
        """
        )

        conn.commit()
        conn.close()

        # Existierende Parameter laden
        self._load_adaptive_parameters()

    def record_interaction(
        self,
        query: str,
        enhanced_query: str,
        intent: str,
        response_time: float,
        confidence_score: float,
        retrieval_method: str,
        sources_count: int,
        session_id: str = "default",
        cache_hit: bool = False,
        user_satisfaction: Optional[float] = None,
    ) -> Dict:
        """Zeichnet Nutzerinteraktion auf und startet sofortiges Lernen"""

        interaction = InteractionEvent(
            timestamp=time.time(),
            query=query,
            enhanced_query=enhanced_query,
            intent=intent,
            response_time=response_time,
            confidence_score=confidence_score,
            retrieval_method=retrieval_method,
            sources_count=sources_count,
            user_satisfaction=user_satisfaction,
            session_id=session_id,
            cache_hit=cache_hit,
            context_length=len(query),
        )

        # Zu Buffer und Processing-Queue hinzufügen
        self.interaction_buffer.append(interaction)
        self.processing_queue.append(interaction)

        # Implizite User Satisfaction ableiten
        if user_satisfaction is None:
            interaction.user_satisfaction = self._infer_user_satisfaction(interaction)

        # Sofortiges Lernen triggern
        learning_results = self._trigger_immediate_learning(interaction)

        # In Datenbank speichern
        self._save_interaction(interaction)

        return {
            "interaction_recorded": True,
            "immediate_learning_triggered": len(learning_results) > 0,
            "learning_insights": learning_results,
            "adaptive_parameters_updated": self._check_parameter_updates(),
            "inferred_satisfaction": interaction.user_satisfaction,
        }

    def _infer_user_satisfaction(self, interaction: InteractionEvent) -> float:
        """Leitet implizite User Satisfaction aus Interaktions-Daten ab"""

        satisfaction_score = 0.5  # Neutral start

        # Confidence-basierte Bewertung
        satisfaction_score += (interaction.confidence_score - 0.5) * 0.3

        # Response-Time-basierte Bewertung (schneller = besser)
        if interaction.response_time < 1.0:
            satisfaction_score += 0.2
        elif interaction.response_time < 2.0:
            satisfaction_score += 0.1
        elif interaction.response_time > 5.0:
            satisfaction_score -= 0.2

        # Cache-Hit-Bonus
        if interaction.cache_hit:
            satisfaction_score += 0.1

        # Sources-Count-Bewertung
        if 3 <= interaction.sources_count <= 6:
            satisfaction_score += 0.1
        elif interaction.sources_count < 2:
            satisfaction_score -= 0.1

        # Intent-spezifische Adjustments
        if interaction.intent in ["definition", "voraussetzungen"]:
            # Diese Intents profitieren von hoher Confidence
            satisfaction_score += (interaction.confidence_score - 0.7) * 0.2

        return max(0.1, min(1.0, satisfaction_score))

    def _trigger_immediate_learning(self, interaction: InteractionEvent) -> List[Dict]:
        """Triggert sofortiges Lernen aus der Interaktion"""

        learning_results = []

        # 1. Query-Pattern-Learning
        query_pattern = self._analyze_query_pattern(interaction)
        if query_pattern:
            learning_results.append(query_pattern)

        # 2. Performance-Pattern-Learning
        performance_pattern = self._analyze_performance_pattern(interaction)
        if performance_pattern:
            learning_results.append(performance_pattern)

        # 3. Retrieval-Strategy-Learning
        retrieval_pattern = self._analyze_retrieval_strategy(interaction)
        if retrieval_pattern:
            learning_results.append(retrieval_pattern)

        # 4. Cache-Strategy-Learning
        cache_pattern = self._analyze_cache_strategy(interaction)
        if cache_pattern:
            learning_results.append(cache_pattern)

        return learning_results

    def _analyze_query_pattern(self, interaction: InteractionEvent) -> Optional[Dict]:
        """Analysiert Query-Patterns für Optimierung"""

        # Sammle ähnliche Queries aus Buffer
        similar_interactions = [
            i
            for i in self.interaction_buffer
            if i.intent == interaction.intent
            and abs(len(i.query) - len(interaction.query)) < 20
        ]

        if (
            len(similar_interactions)
            >= self.learning_config["min_interactions_for_pattern"]
        ):
            # Berechne durchschnittliche Performance für diesen Query-Typ
            avg_satisfaction = statistics.mean(
                [i.user_satisfaction for i in similar_interactions]
            )
            avg_response_time = statistics.mean(
                [i.response_time for i in similar_interactions]
            )
            avg_confidence = statistics.mean(
                [i.confidence_score for i in similar_interactions]
            )

            # Lerne optimale Query-Enhancement-Parameter
            if avg_satisfaction > 0.7:  # Gutes Pattern
                pattern_id = f"query_opt_{interaction.intent}_{hash(interaction.query[:20]) % 1000}"

                optimal_params = {
                    "intent": interaction.intent,
                    "optimal_query_length": len(interaction.enhanced_query),
                    "success_indicators": {
                        "avg_satisfaction": avg_satisfaction,
                        "avg_response_time": avg_response_time,
                        "avg_confidence": avg_confidence,
                    },
                }

                self._store_learning_insight(
                    pattern_id=pattern_id,
                    pattern_type="query_optimization",
                    confidence=avg_satisfaction,
                    impact_score=avg_satisfaction * 0.8
                    + (1 - avg_response_time / 5) * 0.2,
                    parameters=optimal_params,
                )

                return {
                    "type": "query_pattern",
                    "pattern_id": pattern_id,
                    "optimization": optimal_params,
                }

        return None

    def _analyze_performance_pattern(
        self, interaction: InteractionEvent
    ) -> Optional[Dict]:
        """Analysiert Performance-Patterns für Systemoptimierung"""

        method_interactions = [
            i
            for i in self.interaction_buffer
            if i.retrieval_method == interaction.retrieval_method
        ]

        if len(method_interactions) >= 10:  # Mindestens 10 Samples
            avg_performance = statistics.mean(
                [i.user_satisfaction for i in method_interactions]
            )

            # Update Strategy Performance Tracking
            self.strategy_performance[interaction.retrieval_method].update(
                {
                    "total_usage": len(method_interactions),
                    "success_count": sum(
                        1 for i in method_interactions if i.user_satisfaction > 0.7
                    ),
                    "avg_response_time": statistics.mean(
                        [i.response_time for i in method_interactions]
                    ),
                    "avg_confidence": statistics.mean(
                        [i.confidence_score for i in method_interactions]
                    ),
                    "user_satisfaction": avg_performance,
                }
            )

            # Adaptive Parameter-Anpassung
            if avg_performance > 0.8:  # Sehr gute Performance
                self._adapt_parameters_for_method(
                    interaction.retrieval_method, "increase"
                )
            elif avg_performance < 0.5:  # Schlechte Performance
                self._adapt_parameters_for_method(
                    interaction.retrieval_method, "decrease"
                )

            return {
                "type": "performance_pattern",
                "method": interaction.retrieval_method,
                "performance_score": avg_performance,
                "recommended_action": (
                    "increase"
                    if avg_performance > 0.8
                    else "decrease" if avg_performance < 0.5 else "maintain"
                ),
            }

        return None

    def _analyze_retrieval_strategy(
        self, interaction: InteractionEvent
    ) -> Optional[Dict]:
        """Analysiert Retrieval-Strategien für Optimierung"""

        # Analyse der optimalen Anzahl Sources für diesen Intent
        intent_interactions = [
            i
            for i in self.interaction_buffer
            if i.intent == interaction.intent and i.user_satisfaction is not None
        ]

        if len(intent_interactions) >= 5:
            # Gruppiere nach Sources-Count
            sources_performance = defaultdict(list)
            for i in intent_interactions:
                sources_performance[i.sources_count].append(i.user_satisfaction)

            # Finde optimale Anzahl Sources
            best_sources_count = max(
                sources_performance.keys(),
                key=lambda k: statistics.mean(sources_performance[k]),
            )

            if best_sources_count != self.adaptive_parameters["retrieval_k"]:
                # Vorschlag für Parameter-Anpassung
                pattern_id = f"retrieval_opt_{interaction.intent}"

                self._store_learning_insight(
                    pattern_id=pattern_id,
                    pattern_type="retrieval_tuning",
                    confidence=0.8,
                    impact_score=0.7,
                    parameters={
                        "intent": interaction.intent,
                        "optimal_sources_count": best_sources_count,
                        "current_sources_count": self.adaptive_parameters[
                            "retrieval_k"
                        ],
                    },
                )

                return {
                    "type": "retrieval_strategy",
                    "pattern_id": pattern_id,
                    "recommendation": f"Für Intent '{interaction.intent}' optimal: {best_sources_count} Sources",
                }

        return None

    def _analyze_cache_strategy(self, interaction: InteractionEvent) -> Optional[Dict]:
        """Analysiert Cache-Strategien für Optimierung"""

        cache_interactions = [i for i in self.interaction_buffer if i.cache_hit]
        no_cache_interactions = [i for i in self.interaction_buffer if not i.cache_hit]

        if len(cache_interactions) >= 5 and len(no_cache_interactions) >= 5:
            cache_satisfaction = statistics.mean(
                [i.user_satisfaction for i in cache_interactions]
            )
            no_cache_satisfaction = statistics.mean(
                [i.user_satisfaction for i in no_cache_interactions]
            )

            if (
                cache_satisfaction > no_cache_satisfaction + 0.1
            ):  # Cache ist signifikant besser
                # Erhöhe Cache-Agressivität
                new_ttl_multiplier = (
                    self.adaptive_parameters["cache_ttl_multiplier"] * 1.1
                )
                self._update_adaptive_parameter(
                    "cache_ttl_multiplier",
                    new_ttl_multiplier,
                    "Cache performance is superior",
                )

                return {
                    "type": "cache_strategy",
                    "action": "increase_caching",
                    "reason": f"Cache satisfaction ({cache_satisfaction:.2f}) > No-cache ({no_cache_satisfaction:.2f})",
                }

        return None

    def _adapt_parameters_for_method(self, method: str, action: str):
        """Adaptiert Parameter basierend auf Methoden-Performance"""

        if "multi_stage" in method and action == "increase":
            # Erhöhe Retrieval-K für Multi-Stage
            new_k = min(10, self.adaptive_parameters["retrieval_k"] + 1)
            self._update_adaptive_parameter(
                "retrieval_k", new_k, "Multi-stage method performing well"
            )

        elif "cache" in method and action == "increase":
            # Erhöhe Cache-Aggressivität
            new_threshold = max(
                0.7, self.adaptive_parameters["semantic_similarity_threshold"] - 0.05
            )
            self._update_adaptive_parameter(
                "semantic_similarity_threshold",
                new_threshold,
                "Cache method performing well",
            )

    def _store_learning_insight(
        self,
        pattern_id: str,
        pattern_type: str,
        confidence: float,
        impact_score: float,
        parameters: Dict,
    ):
        """Speichert Learning Insight in Datenbank"""

        insight = LearningInsight(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            confidence=confidence,
            impact_score=impact_score,
            usage_count=0,
            success_rate=0.0,
            last_applied=time.time(),
            parameters=parameters,
        )

        self.learning_insights[pattern_id] = insight

        # In DB speichern
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO learning_insights 
            (pattern_id, pattern_type, confidence, impact_score, usage_count, 
             success_rate, last_applied, parameters)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                pattern_id,
                pattern_type,
                confidence,
                impact_score,
                insight.usage_count,
                insight.success_rate,
                insight.last_applied,
                json.dumps(parameters),
            ),
        )

        conn.commit()
        conn.close()

    def _update_adaptive_parameter(
        self, param_name: str, new_value: float, reason: str
    ):
        """Aktualisiert adaptive Parameter"""

        old_value = self.adaptive_parameters.get(param_name)
        self.adaptive_parameters[param_name] = new_value

        # In DB speichern
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO adaptive_parameters 
            (parameter_name, current_value, last_updated, update_reason)
            VALUES (?, ?, ?, ?)
        """,
            (param_name, new_value, time.time(), reason),
        )

        conn.commit()
        conn.close()

        print(
            f"🔧 Parameter updated: {param_name} = {new_value} (was {old_value}) - {reason}"
        )

    def _load_adaptive_parameters(self):
        """Lädt adaptive Parameter aus Datenbank"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT parameter_name, current_value FROM adaptive_parameters"
            )
            rows = cursor.fetchall()

            for param_name, value in rows:
                self.adaptive_parameters[param_name] = value

            conn.close()
        except:
            pass  # Erste Ausführung, keine Parameter vorhanden

    def _start_processing(self):
        """Startet Background-Processing für kontinuierliches Lernen"""

        def processing_loop():
            while self.is_processing:
                try:
                    if self.processing_queue:
                        interaction = self.processing_queue.popleft()
                        # Weitere Background-Analyse hier
                        self._deep_pattern_analysis(interaction)

                    time.sleep(0.1)  # Kurze Pause
                except Exception as e:
                    print(f"Processing error: {e}")
                    time.sleep(1)

        self.processing_thread = threading.Thread(target=processing_loop, daemon=True)
        self.processing_thread.start()

    def _deep_pattern_analysis(self, interaction: InteractionEvent):
        """Tiefere Pattern-Analyse im Background"""

        # Hier können zeitaufwändige Analysen durchgeführt werden
        # z.B. Session-basierte Patterns, Long-term Trends, etc.
        pass

    def get_adaptive_parameters(self) -> Dict:
        """Liefert aktuelle adaptive Parameter"""
        return self.adaptive_parameters.copy()

    def get_learning_insights(self) -> Dict:
        """Liefert aktuelle Learning Insights"""

        insights_summary = {}

        for pattern_id, insight in self.learning_insights.items():
            insights_summary[pattern_id] = {
                "type": insight.pattern_type,
                "confidence": insight.confidence,
                "impact_score": insight.impact_score,
                "usage_count": insight.usage_count,
                "parameters": insight.parameters,
            }

        return insights_summary

    def get_realtime_statistics(self) -> Dict:
        """Liefert Real-Time-Lernstatistiken"""

        recent_interactions = [
            i for i in self.interaction_buffer if time.time() - i.timestamp < 3600
        ]

        if not recent_interactions:
            return {"status": "no_recent_data"}

        return {
            "recent_interactions_count": len(recent_interactions),
            "avg_satisfaction": statistics.mean(
                [i.user_satisfaction for i in recent_interactions]
            ),
            "avg_response_time": statistics.mean(
                [i.response_time for i in recent_interactions]
            ),
            "avg_confidence": statistics.mean(
                [i.confidence_score for i in recent_interactions]
            ),
            "cache_hit_rate": sum(1 for i in recent_interactions if i.cache_hit)
            / len(recent_interactions)
            * 100,
            "most_common_intent": max(
                set([i.intent for i in recent_interactions]),
                key=[i.intent for i in recent_interactions].count,
            ),
            "learning_insights_count": len(self.learning_insights),
            "active_experiments": len(self.active_experiments),
            "adaptive_parameters": self.adaptive_parameters,
            "strategy_performance": dict(self.strategy_performance),
        }

    def _check_parameter_updates(self) -> bool:
        """Prüft ob Parameter in letzter Zeit aktualisiert wurden"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(*) FROM adaptive_parameters 
            WHERE last_updated > ?
        """,
            (time.time() - 300,),
        )  # Letzte 5 Minuten

        count = cursor.fetchone()[0]
        conn.close()

        return count > 0

    def _save_interaction(self, interaction: InteractionEvent):
        """Speichert Interaktion in Datenbank"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO interactions 
            (timestamp, query, enhanced_query, intent, response_time, confidence_score,
             retrieval_method, sources_count, user_satisfaction, session_id, 
             interaction_type, context_length, cache_hit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                interaction.timestamp,
                interaction.query,
                interaction.enhanced_query,
                interaction.intent,
                interaction.response_time,
                interaction.confidence_score,
                interaction.retrieval_method,
                interaction.sources_count,
                interaction.user_satisfaction,
                interaction.session_id,
                interaction.interaction_type,
                interaction.context_length,
                interaction.cache_hit,
            ),
        )

        conn.commit()
        conn.close()

    def stop_processing(self):
        """Stoppt Background-Processing"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)

        self.executor.shutdown(wait=True)
