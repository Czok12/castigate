"""
🧠 ADAPTIVE LEARNING SYSTEM
===========================

Selbstlernendes System das aus User-Feedback kontinuierlich optimiert
"""

import pickle
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class UserFeedback:
    """User-Feedback für eine Antwort"""

    query: str
    answer: str
    sources_used: List[Dict]
    user_rating: float  # 1-5 Scale
    feedback_type: str  # "rating", "correction", "preference"
    feedback_text: Optional[str]
    timestamp: float
    session_id: str
    query_intent: Optional[str]  # "definition", "process", "comparison", etc.


@dataclass
class LearningPattern:
    """Gelerntes Muster aus User-Feedback"""

    pattern_type: str
    pattern_data: Dict
    confidence: float
    usage_count: int
    success_rate: float
    last_updated: float


class QueryIntentClassifier:
    """Klassifiziert Query-Intent für bessere Optimierung"""

    def __init__(self):
        self.intent_patterns = {
            "definition": [
                "was ist",
                "was bedeutet",
                "definition",
                "begriff",
                "versteht man unter",
                "bezeichnet",
            ],
            "voraussetzungen": [
                "voraussetzung",
                "bedingung",
                "erforderlich",
                "notwendig",
                "muss",
                "brauch",
            ],
            "rechtsfolgen": [
                "folge",
                "konsequenz",
                "bewirkt",
                "führt zu",
                "ergebnis",
                "wirkung",
            ],
            "vergleich": [
                "unterschied",
                "vergleich",
                "vs",
                "gegenüber",
                "abgrenzung",
                "differenz",
            ],
            "prozess": ["wie", "ablauf", "verfahren", "vorgehen", "schritt", "prozess"],
            "rechtsprechung": [
                "urteil",
                "entscheidung",
                "rechtsprechung",
                "gericht",
                "bgh",
                "bverfg",
            ],
        }

    def classify_intent(self, query: str) -> Tuple[str, float]:
        """Klassifiziert Query-Intent mit Confidence"""

        query_lower = query.lower()
        intent_scores = {}

        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                intent_scores[intent] = score / len(patterns)

        if not intent_scores:
            return "general", 0.3

        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = intent_scores[best_intent]

        return best_intent, confidence


class AdaptiveLearningEngine:
    """Haupt-Learning-Engine für kontinuierliche Verbesserung"""

    def __init__(self, model_path: str = "adaptive_model.pkl"):
        self.model_path = model_path
        self.feedback_history: deque[UserFeedback] = deque(maxlen=1000)
        self.learned_patterns: Dict[str, LearningPattern] = {}
        self.intent_classifier = QueryIntentClassifier()

        # Performance-Tracking für verschiedene Strategien
        self.strategy_performance = defaultdict(
            lambda: {
                "total_queries": 0,
                "avg_rating": 0.0,
                "success_rate": 0.0,
                "response_times": deque(maxlen=50),
            }
        )

        # Query-Optimierungen basierend auf Feedback
        self.query_optimizations = {
            "synonyms": defaultdict(list),
            "expansions": defaultdict(list),
            "intent_adjustments": defaultdict(dict),
        }

        # Lade existierendes Modell
        self._load_model()

    def record_feedback(
        self,
        query: str,
        answer: str,
        sources: List[Dict],
        rating: float,
        feedback_text: Optional[str] = None,
        session_id: str = "default",
        response_time: float = 0.0,
    ) -> Dict:
        """Zeichnet User-Feedback auf und lernt daraus"""

        # Intent klassifizieren
        intent, intent_confidence = self.intent_classifier.classify_intent(query)

        # Feedback erstellen
        feedback = UserFeedback(
            query=query,
            answer=answer,
            sources_used=sources,
            user_rating=rating,
            feedback_type="rating",
            feedback_text=feedback_text,
            timestamp=time.time(),
            session_id=session_id,
            query_intent=intent,
        )

        self.feedback_history.append(feedback)

        # Sofortiges Learning
        learning_results = self._process_feedback(feedback, response_time)

        # Modell speichern (periodisch)
        if len(self.feedback_history) % 10 == 0:
            self._save_model()

        return {
            "feedback_processed": True,
            "intent_detected": intent,
            "intent_confidence": intent_confidence,
            "learning_results": learning_results,
            "total_feedback_count": len(self.feedback_history),
        }

    def _process_feedback(self, feedback: UserFeedback, response_time: float) -> Dict:
        """Verarbeitet Feedback und aktualisiert Lernmuster"""

        results = {}

        # 1. Query-Pattern Learning
        if feedback.user_rating >= 4.0:  # Gutes Feedback
            self._learn_successful_patterns(feedback)
            results["positive_patterns"] = "learned"
        elif feedback.user_rating <= 2.0:  # Schlechtes Feedback
            self._learn_failure_patterns(feedback)
            results["failure_patterns"] = "analyzed"

        # 2. Intent-spezifische Optimierungen
        intent_optimizations = self._optimize_for_intent(feedback)
        results["intent_optimizations"] = intent_optimizations

        # 3. Source-Quality Learning
        source_quality = self._analyze_source_quality(feedback)
        results["source_analysis"] = source_quality

        # 4. Performance-Tracking
        self._update_performance_metrics(feedback, response_time)
        results["performance_updated"] = True

        return results

    def _learn_successful_patterns(self, feedback: UserFeedback):
        """Lernt aus erfolgreichem Feedback"""

        query_words = set(feedback.query.lower().split())
        intent = feedback.query_intent

        # Erfolgreiche Query-Patterns
        pattern_key = f"successful_{intent}"

        if pattern_key in self.learned_patterns:
            pattern = self.learned_patterns[pattern_key]
            pattern.usage_count += 1
            pattern.success_rate = (
                pattern.success_rate * (pattern.usage_count - 1) + 1.0
            ) / pattern.usage_count
            pattern.last_updated = time.time()
        else:
            self.learned_patterns[pattern_key] = LearningPattern(
                pattern_type="successful_query",
                pattern_data={
                    "intent": intent,
                    "common_words": list(query_words),
                    "avg_rating": feedback.user_rating,
                },
                confidence=0.7,
                usage_count=1,
                success_rate=1.0,
                last_updated=time.time(),
            )

        # Query-Expansions lernen
        for word in query_words:
            if word not in self.query_optimizations["expansions"]:
                self.query_optimizations["expansions"][word] = []

            # Füge verwandte Begriffe aus erfolgreichen Sources hinzu
            for source in feedback.sources_used:
                source_words = set(source.get("content", "").lower().split())
                related_words = source_words - query_words
                for related in list(related_words)[:3]:  # Top 3
                    if (
                        len(related) > 3
                        and related not in self.query_optimizations["expansions"][word]
                    ):
                        self.query_optimizations["expansions"][word].append(related)

    def _learn_failure_patterns(self, feedback: UserFeedback):
        """Lernt aus gescheitertem Feedback"""

        query_words = set(feedback.query.lower().split())
        intent = feedback.query_intent

        # Problematische Patterns identifizieren
        pattern_key = f"problematic_{intent}"

        if pattern_key in self.learned_patterns:
            pattern = self.learned_patterns[pattern_key]
            pattern.usage_count += 1
            current_fail_rate = 1.0 - pattern.success_rate
            new_fail_rate = (
                current_fail_rate * (pattern.usage_count - 1) + 1.0
            ) / pattern.usage_count
            pattern.success_rate = 1.0 - new_fail_rate
            pattern.last_updated = time.time()
        else:
            self.learned_patterns[pattern_key] = LearningPattern(
                pattern_type="problematic_query",
                pattern_data={
                    "intent": intent,
                    "problematic_words": list(query_words),
                    "avg_rating": feedback.user_rating,
                    "common_issues": [],
                },
                confidence=0.8,
                usage_count=1,
                success_rate=0.0,
                last_updated=time.time(),
            )

        # Feedback-Text analysieren
        if feedback.feedback_text:
            self._analyze_feedback_text(feedback.feedback_text, intent)

    def _optimize_for_intent(self, feedback: UserFeedback) -> Dict:
        """Optimiert Strategien für spezifische Intents"""

        intent = feedback.query_intent
        rating = feedback.user_rating

        if intent not in self.query_optimizations["intent_adjustments"]:
            self.query_optimizations["intent_adjustments"][intent] = {
                "preferred_chunk_size": 1000,
                "preferred_k": 4,
                "weight_adjustments": {},
                "template_preference": "standard",
            }

        adjustments = self.query_optimizations["intent_adjustments"][intent]

        # Adaptiere Parameter basierend auf Feedback
        if rating >= 4.0:  # Gutes Feedback
            # Verstärke aktuelle Einstellungen
            if "good_responses" not in adjustments:
                adjustments["good_responses"] = 0
            adjustments["good_responses"] += 1

        elif rating <= 2.0:  # Schlechtes Feedback
            # Experimentiere mit anderen Einstellungen
            if "bad_responses" not in adjustments:
                adjustments["bad_responses"] = 0
            adjustments["bad_responses"] += 1

            # Vorschläge für Verbesserungen
            if intent == "definition":
                adjustments["template_preference"] = "detailed"
            elif intent == "prozess":
                adjustments["preferred_k"] = 6  # Mehr Kontext für Prozesse
            elif intent == "vergleich":
                adjustments["template_preference"] = "comprehensive"

        return adjustments

    def _analyze_source_quality(self, feedback: UserFeedback) -> Dict:
        """Analysiert Qualität der verwendeten Quellen"""

        source_analysis = {
            "total_sources": len(feedback.sources_used),
            "quality_indicators": [],
        }

        for source in feedback.sources_used:
            # Page-Nummer als Qualitäts-Indikator
            page = source.get("metadata", {}).get("page", 999)
            if isinstance(page, int):
                if page < 50:  # Frühe Seiten oft wichtiger
                    source_analysis["quality_indicators"].append("early_page")
                elif page > 500:
                    source_analysis["quality_indicators"].append("late_page")

            # Content-Länge als Indikator
            content_length = len(source.get("content", ""))
            if content_length > 1500:
                source_analysis["quality_indicators"].append("comprehensive_content")
            elif content_length < 300:
                source_analysis["quality_indicators"].append("brief_content")

        return source_analysis

    def _analyze_feedback_text(self, feedback_text: str, intent: str):
        """Analysiert freien Feedback-Text für Patterns"""

        feedback_lower = feedback_text.lower()

        # Häufige Feedback-Patterns
        if "zu kurz" in feedback_lower or "mehr detail" in feedback_lower:
            if intent not in self.query_optimizations["intent_adjustments"]:
                self.query_optimizations["intent_adjustments"][intent] = {}
            self.query_optimizations["intent_adjustments"][intent][
                "needs_more_detail"
            ] = True

        if "zu lang" in feedback_lower or "zu ausführlich" in feedback_lower:
            if intent not in self.query_optimizations["intent_adjustments"]:
                self.query_optimizations["intent_adjustments"][intent] = {}
            self.query_optimizations["intent_adjustments"][intent][
                "needs_less_detail"
            ] = True

        if "unverständlich" in feedback_lower or "kompliziert" in feedback_lower:
            if intent not in self.query_optimizations["intent_adjustments"]:
                self.query_optimizations["intent_adjustments"][intent] = {}
            self.query_optimizations["intent_adjustments"][intent][
                "needs_simplification"
            ] = True

    def _update_performance_metrics(self, feedback: UserFeedback, response_time: float):
        """Aktualisiert Performance-Metriken"""

        intent = feedback.query_intent
        metrics = self.strategy_performance[intent]

        # Zähler aktualisieren
        metrics["total_queries"] += 1

        # Rolling Average für Rating
        current_avg = metrics["avg_rating"]
        total = metrics["total_queries"]
        metrics["avg_rating"] = (
            current_avg * (total - 1) + feedback.user_rating
        ) / total

        # Success Rate (Rating >= 3.5 = Success)
        success = 1.0 if feedback.user_rating >= 3.5 else 0.0
        current_success = metrics["success_rate"]
        metrics["success_rate"] = (current_success * (total - 1) + success) / total

        # Response Times
        if response_time > 0:
            metrics["response_times"].append(response_time)

    def get_optimization_suggestions(self, query: str) -> Dict:
        """Gibt Optimierungsvorschläge für eine Query basierend auf gelernten Patterns"""

        intent, confidence = self.intent_classifier.classify_intent(query)
        suggestions = {
            "detected_intent": intent,
            "intent_confidence": confidence,
            "query_expansions": [],
            "parameter_suggestions": {},
            "template_suggestion": "standard",
        }

        # Query-Expansions basierend auf gelernten Patterns
        query_words = query.lower().split()
        for word in query_words:
            if word in self.query_optimizations["expansions"]:
                expansions = self.query_optimizations["expansions"][word][:2]  # Top 2
                suggestions["query_expansions"].extend(expansions)

        # Intent-spezifische Optimierungen
        if intent in self.query_optimizations["intent_adjustments"]:
            adjustments = self.query_optimizations["intent_adjustments"][intent]

            suggestions["parameter_suggestions"] = {
                "k": adjustments.get("preferred_k", 4),
                "chunk_size": adjustments.get("preferred_chunk_size", 1000),
                "template": adjustments.get("template_preference", "standard"),
            }

            suggestions["template_suggestion"] = adjustments.get(
                "template_preference", "standard"
            )

        # Erfolgs-Wahrscheinlichkeit basierend auf gelernten Patterns
        success_pattern_key = f"successful_{intent}"
        if success_pattern_key in self.learned_patterns:
            pattern = self.learned_patterns[success_pattern_key]
            suggestions["expected_success_rate"] = pattern.success_rate
        else:
            suggestions["expected_success_rate"] = 0.7  # Default

        return suggestions

    def get_learning_report(self) -> Dict:
        """Gibt umfassenden Lernbericht zurück"""

        return {
            "total_feedback": len(self.feedback_history),
            "learned_patterns": len(self.learned_patterns),
            "intent_performance": dict(self.strategy_performance),
            "query_optimizations": {
                "expansion_rules": len(self.query_optimizations["expansions"]),
                "intent_adjustments": len(
                    self.query_optimizations["intent_adjustments"]
                ),
            },
            "recent_feedback_summary": self._get_recent_feedback_summary(),
            "top_performing_intents": self._get_top_performing_intents(),
            "improvement_suggestions": self._generate_system_improvements(),
        }

    def _get_recent_feedback_summary(self) -> Dict:
        """Zusammenfassung des letzten Feedbacks"""

        if not self.feedback_history:
            return {"no_feedback": True}

        recent = list(self.feedback_history)[-10:]  # Letzte 10

        return {
            "avg_rating": np.mean([f.user_rating for f in recent]),
            "most_common_intent": max(
                set([f.query_intent for f in recent]),
                key=[f.query_intent for f in recent].count,
            ),
            "feedback_count": len(recent),
        }

    def _get_top_performing_intents(self) -> List[Tuple[str, float]]:
        """Top-performende Intents basierend auf Success Rate"""

        performance_list = [
            (intent, metrics["success_rate"])
            for intent, metrics in self.strategy_performance.items()
            if metrics["total_queries"] >= 3  # Mindestens 3 Queries
        ]

        return sorted(performance_list, key=lambda x: x[1], reverse=True)[:5]

    def _generate_system_improvements(self) -> List[str]:
        """Generiert Verbesserungsvorschläge für das System"""

        improvements = []

        # Analysiere Performance pro Intent
        for intent, metrics in self.strategy_performance.items():
            if metrics["total_queries"] >= 5:
                if metrics["success_rate"] < 0.6:
                    improvements.append(
                        f"Intent '{intent}' benötigt Optimierung (Success Rate: {metrics['success_rate']:.1%})"
                    )

                if (
                    metrics["response_times"]
                    and np.mean(metrics["response_times"]) > 5.0
                ):
                    improvements.append(
                        f"Intent '{intent}' hat hohe Antwortzeiten (Ø {np.mean(metrics['response_times']):.1f}s)"
                    )

        # Analysiere Lernmuster
        problematic_patterns = [
            pattern
            for pattern in self.learned_patterns.values()
            if pattern.pattern_type == "problematic_query" and pattern.usage_count >= 3
        ]

        if problematic_patterns:
            improvements.append(
                f"{len(problematic_patterns)} problematische Query-Patterns identifiziert"
            )

        return improvements

    def _save_model(self):
        """Speichert das Lernmodell"""

        try:
            model_data = {
                "feedback_history": list(self.feedback_history),
                "learned_patterns": self.learned_patterns,
                "strategy_performance": dict(self.strategy_performance),
                "query_optimizations": self.query_optimizations,
                "timestamp": time.time(),
            }

            with open(self.model_path, "wb") as f:
                pickle.dump(model_data, f)

        except Exception as e:
            print(f"Fehler beim Speichern des Lernmodells: {e}")

    def _load_model(self):
        """Lädt existierendes Lernmodell"""

        try:
            with open(self.model_path, "rb") as f:
                model_data = pickle.load(f)

            # Lade Daten
            if "feedback_history" in model_data:
                self.feedback_history.extend(
                    [
                        UserFeedback(**fb) if isinstance(fb, dict) else fb
                        for fb in model_data["feedback_history"]
                    ]
                )

            if "learned_patterns" in model_data:
                self.learned_patterns = model_data["learned_patterns"]

            if "strategy_performance" in model_data:
                self.strategy_performance.update(model_data["strategy_performance"])

            if "query_optimizations" in model_data:
                self.query_optimizations.update(model_data["query_optimizations"])

        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            # Keine existierende Datei oder beschädigt - mit leeren Daten starten
            pass
        except Exception as e:
            print(f"Fehler beim Laden des Lernmodells: {e}")


class FeedbackCollector:
    """Sammelt User-Feedback auf verschiedene Weise"""

    def __init__(self, learning_engine: AdaptiveLearningEngine):
        self.learning_engine = learning_engine

    def collect_rating_feedback(
        self,
        query: str,
        answer: str,
        sources: List[Dict],
        rating: float,
        session_id: str = "default",
    ) -> Dict:
        """Sammelt Bewertungs-Feedback"""

        return self.learning_engine.record_feedback(
            query=query,
            answer=answer,
            sources=sources,
            rating=rating,
            session_id=session_id,
        )

    def collect_text_feedback(
        self,
        query: str,
        answer: str,
        sources: List[Dict],
        feedback_text: str,
        implied_rating: float = 3.0,
        session_id: str = "default",
    ) -> Dict:
        """Sammelt Text-Feedback und leitet Rating ab"""

        # Einfache Sentiment-Analyse für implied rating
        feedback_lower = feedback_text.lower()

        positive_words = [
            "gut",
            "perfekt",
            "hilfreich",
            "klar",
            "verständlich",
            "danke",
        ]
        negative_words = [
            "schlecht",
            "falsch",
            "unverständlich",
            "unvollständig",
            "fehler",
        ]

        positive_count = sum(1 for word in positive_words if word in feedback_lower)
        negative_count = sum(1 for word in negative_words if word in feedback_lower)

        if positive_count > negative_count:
            implied_rating = min(5.0, 3.0 + positive_count * 0.5)
        elif negative_count > positive_count:
            implied_rating = max(1.0, 3.0 - negative_count * 0.5)

        return self.learning_engine.record_feedback(
            query=query,
            answer=answer,
            sources=sources,
            rating=implied_rating,
            feedback_text=feedback_text,
            session_id=session_id,
        )

    def collect_implicit_feedback(
        self,
        query: str,
        answer: str,
        sources: List[Dict],
        user_actions: Dict,
        session_id: str = "default",
    ) -> Dict:
        """Sammelt implizites Feedback basierend auf User-Aktionen"""

        # Implizite Rating-Ableitung
        rating = 3.0  # Baseline

        # Zeit, die der User mit der Antwort verbracht hat
        if "reading_time" in user_actions:
            reading_time = user_actions["reading_time"]
            if reading_time > 30:  # Über 30 Sekunden gelesen
                rating += 0.5
            elif reading_time < 5:  # Unter 5 Sekunden gelesen
                rating -= 0.5

        # Hat der User die Quellen angesehen?
        if user_actions.get("viewed_sources", False):
            rating += 0.3

        # Hat der User eine Folge-Frage gestellt?
        if user_actions.get("follow_up_question", False):
            rating += 0.2  # Zeigt Interesse

        # Hat der User die Antwort kopiert?
        if user_actions.get("copied_answer", False):
            rating += 0.7  # Starkes positives Signal

        rating = max(1.0, min(5.0, rating))  # Begrenzen auf 1-5

        return self.learning_engine.record_feedback(
            query=query,
            answer=answer,
            sources=sources,
            rating=rating,
            feedback_text=f"Implicit feedback: {user_actions}",
            session_id=session_id,
        )
