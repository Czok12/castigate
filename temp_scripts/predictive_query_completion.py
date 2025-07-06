"""
🔮 PREDICTIVE QUERY COMPLETION ENGINE
=====================================

Ultra-Advanced Predictive Query Completion mit User Intent Prediction und kontextueller Expansion
"""

import sqlite3
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class QueryPrediction:
    """Einzelne Query-Vorhersage"""

    completion: str
    confidence: float
    prediction_type: str  # partial_match, semantic_similar, intent_based, contextual
    source: str  # user_history, global_patterns, legal_corpus
    intent_category: str
    estimated_results: int
    processing_time_estimate: float
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class UserContext:
    """Benutzer-Kontext für personalisierte Vorhersagen"""

    session_id: str
    query_history: List[str]
    topic_preferences: Dict[str, float]
    complexity_preference: float  # 0.0 = einfach, 1.0 = komplex
    search_patterns: Dict[str, int]
    time_of_day: str
    recent_documents: List[str]
    expertise_level: float  # 0.0 = Anfänger, 1.0 = Expert


class LegalDomainKnowledge:
    """Juristische Domänen-Wissensbasis für intelligente Vervollständigung"""

    def __init__(self):
        # Juristische Begriffshierarchien
        self.legal_hierarchies = {
            "vertragsrecht": {
                "kaufvertrag": ["kaufpreis", "gewährleistung", "lieferung", "mangel"],
                "mietvertrag": ["miete", "kaution", "kündigung", "instandhaltung"],
                "arbeitsvertrag": ["lohn", "kündigung", "urlaub", "arbeitszeit"],
                "darlehen": ["zinsen", "tilgung", "sicherheiten", "kündigung"],
            },
            "strafrecht": {
                "diebstahl": ["wegnahme", "gewahrsam", "zueignung", "fremd"],
                "betrug": ["täuschung", "irrtum", "vermögensschaden", "bereicherung"],
                "körperverletzung": [
                    "gesundheitsschädigung",
                    "misshandlung",
                    "vorsatz",
                ],
                "mord": ["heimtücke", "habgier", "niedrige beweggründe"],
            },
            "familienrecht": {
                "scheidung": ["zerrüttung", "unterhalt", "sorgerecht", "zugewinn"],
                "adoption": ["minderjährige", "einverständnis", "wohl", "verfahren"],
                "unterhalt": ["bedürftigkeit", "leistungsfähigkeit", "verwandtschaft"],
                "sorgerecht": ["kindeswohl", "elterliche sorge", "umgang"],
            },
            "verfassungsrecht": {
                "grundrechte": ["menschenwürde", "gleichheit", "meinungsfreiheit"],
                "staatsorganisation": ["bundestag", "bundesrat", "bundesregierung"],
                "verfassungsbeschwerde": ["grundrechtsverletzung", "subsidiarität"],
            },
        }

        # Häufige juristische Fragemuster
        self.query_patterns = {
            "definition": ["was ist", "bedeutung", "begriff", "definition von"],
            "voraussetzungen": [
                "voraussetzungen",
                "bedingungen",
                "erforderlich",
                "notwendig",
            ],
            "rechtsfolgen": ["rechtsfolgen", "folgen", "auswirkungen", "konsequenzen"],
            "verfahren": ["verfahren", "ablauf", "prozess", "wie läuft ab"],
            "fristen": ["frist", "verjährung", "zeitraum", "termin"],
            "zuständigkeit": ["zuständig", "gericht", "behörde", "kompetenz"],
            "berechnung": ["berechnung", "höhe", "betrag", "zinsen"],
            "rechtsprechung": ["rechtsprechung", "urteil", "entscheidung", "bgh"],
        }

        # Juristische Synonym-Gruppen
        self.legal_synonyms = {
            "vertrag": ["vereinbarung", "kontrakt", "abkommen", "deal"],
            "anspruch": ["recht", "berechtigung", "forderung", "claim"],
            "pflicht": ["verpflichtung", "obliegenheit", "aufgabe", "duty"],
            "schaden": ["verlust", "nachteil", "beeinträchtigung", "harm"],
            "kündigung": ["beendigung", "auflösung", "termination"],
            "gericht": ["instanz", "tribunal", "court", "richter"],
            "gesetz": ["norm", "regelung", "bestimmung", "vorschrift"],
            "urteil": ["entscheidung", "spruch", "beschluss", "ruling"],
        }

        # Kontextuelle Begriffsverbindungen
        self.contextual_connections = {
            "kaufvertrag": ["bgb", "433", "verkäufer", "käufer", "sachmangel"],
            "mietrecht": ["bgb", "535", "mieter", "vermieter", "mietminderung"],
            "arbeitsrecht": ["kündigungsschutz", "betriebsrat", "tarifvertrag"],
            "familienrecht": ["bgb", "famfg", "jugendamt", "bafög"],
            "strafrecht": ["stgb", "stpo", "staatsanwaltschaft", "polizei"],
            "steuerrecht": ["ao", "estg", "umsatzsteuer", "finanzamt"],
        }

    def get_domain_expansions(self, query: str) -> List[str]:
        """Gibt domänen-spezifische Erweiterungen für eine Query zurück"""

        query_lower = query.lower()
        expansions = []

        # Hierarchie-basierte Erweiterungen
        for domain, categories in self.legal_hierarchies.items():
            for category, terms in categories.items():
                if category in query_lower or any(
                    term in query_lower for term in terms
                ):
                    expansions.extend(
                        [f"{query} {term}" for term in terms if term not in query_lower]
                    )

        # Pattern-basierte Erweiterungen
        for pattern_type, patterns in self.query_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                for domain in self.legal_hierarchies.keys():
                    if domain in query_lower:
                        expansions.append(f"{pattern_type} {domain}")

        return list(set(expansions))

    def get_synonymous_queries(self, query: str) -> List[str]:
        """Generiert synonyme Query-Varianten"""

        synonymous = []
        query_words = query.lower().split()

        for i, word in enumerate(query_words):
            if word in self.legal_synonyms:
                for synonym in self.legal_synonyms[word]:
                    new_query = query_words.copy()
                    new_query[i] = synonym
                    synonymous.append(" ".join(new_query))

        return synonymous


class QueryPatternAnalyzer:
    """Analysiert Query-Patterns für Vervollständigung"""

    def __init__(self):
        self.pattern_cache = {}
        self.query_frequency = Counter()
        self.bigram_patterns = defaultdict(Counter)
        self.trigram_patterns = defaultdict(Counter)
        self.intent_patterns = defaultdict(list)

        # Häufige juristische N-Gramme
        self.legal_ngrams = {
            2: [
                ("voraussetzungen", "für"),
                ("definition", "von"),
                ("höhe", "der"),
                ("rechtsprechung", "zu"),
                ("berechnung", "der"),
                ("frist", "für"),
                ("zuständigkeit", "des"),
                ("anspruch", "auf"),
                ("pflicht", "zur"),
            ],
            3: [
                ("voraussetzungen", "für", "den"),
                ("definition", "von", "dem"),
                ("höhe", "der", "miete"),
                ("rechtsprechung", "des", "bgh"),
                ("berechnung", "der", "zinsen"),
                ("frist", "für", "die"),
                ("zuständigkeit", "des", "gerichts"),
            ],
        }

    def analyze_query_pattern(self, query: str) -> Dict[str, any]:
        """Analysiert das Pattern einer Query"""

        if query in self.pattern_cache:
            return self.pattern_cache[query]

        query_lower = query.strip().lower()
        words = query_lower.split()

        analysis = {
            "word_count": len(words),
            "starts_with_question": query_lower.startswith(
                ("was", "wie", "wann", "wo", "warum", "welche")
            ),
            "contains_legal_terms": self._contains_legal_terms(words),
            "query_type": self._classify_query_type(query_lower),
            "complexity_score": self._calculate_complexity(words),
            "domain_hints": self._extract_domain_hints(words),
            "incomplete_patterns": self._detect_incomplete_patterns(words),
        }

        self.pattern_cache[query] = analysis
        return analysis

    def _contains_legal_terms(self, words: List[str]) -> bool:
        """Prüft ob Query juristische Begriffe enthält"""

        legal_indicators = {
            "bgb",
            "stgb",
            "gg",
            "zpo",
            "stpo",
            "bverfg",
            "bgh",
            "bag",
            "paragraph",
            "§",
            "abs",
            "absatz",
            "nummer",
            "nr",
            "recht",
            "gesetz",
            "norm",
            "regel",
            "bestimmung",
            "anspruch",
            "pflicht",
            "kündigung",
            "vertrag",
            "schaden",
            "gericht",
            "richter",
            "urteil",
            "beschluss",
            "verfahren",
        }

        return any(word in legal_indicators for word in words)

    def _classify_query_type(self, query: str) -> str:
        """Klassifiziert den Query-Typ"""

        type_patterns = {
            "definition": ["was ist", "bedeutung", "definition", "begriff"],
            "procedure": ["wie", "verfahren", "ablauf", "prozess"],
            "requirements": ["voraussetzungen", "bedingungen", "erforderlich"],
            "calculation": ["berechnung", "höhe", "betrag", "zinsen"],
            "timeframe": ["frist", "verjährung", "zeitraum", "termin"],
            "jurisdiction": ["zuständig", "gericht", "behörde"],
            "case_law": ["rechtsprechung", "urteil", "bgh", "entscheidung"],
            "comparison": ["unterschied", "vergleich", "gegenüber"],
        }

        for query_type, patterns in type_patterns.items():
            if any(pattern in query for pattern in patterns):
                return query_type

        return "general"

    def _calculate_complexity(self, words: List[str]) -> float:
        """Berechnet Komplexitäts-Score der Query"""

        complexity_indicators = {
            "high": [
                "jedoch",
                "allerdings",
                "insbesondere",
                "grundsätzlich",
                "ausnahmsweise",
            ],
            "medium": ["und", "oder", "sowie", "bei", "durch", "wegen"],
            "legal": ["bgb", "stgb", "§", "abs", "rechtsprechung", "urteil"],
        }

        score = len(words) * 0.1  # Basis-Score

        for word in words:
            if word in complexity_indicators["high"]:
                score += 0.3
            elif word in complexity_indicators["medium"]:
                score += 0.1
            elif word in complexity_indicators["legal"]:
                score += 0.2

        return min(score, 1.0)

    def _extract_domain_hints(self, words: List[str]) -> List[str]:
        """Extrahiert Hinweise auf juristische Domänen"""

        domain_keywords = {
            "vertragsrecht": ["vertrag", "kaufvertrag", "mietvertrag", "darlehen"],
            "strafrecht": ["straftat", "diebstahl", "betrug", "mord", "stgb"],
            "familienrecht": ["scheidung", "unterhalt", "sorgerecht", "adoption"],
            "arbeitsrecht": ["arbeitsvertrag", "kündigung", "lohn", "urlaub"],
            "verfassungsrecht": ["grundrechte", "verfassung", "gg", "bverfg"],
            "zivilrecht": ["bgb", "schadensersatz", "anspruch", "klage"],
        }

        domains = []
        for domain, keywords in domain_keywords.items():
            if any(keyword in words for keyword in keywords):
                domains.append(domain)

        return domains

    def _detect_incomplete_patterns(self, words: List[str]) -> List[str]:
        """Erkennt unvollständige Patterns"""

        incomplete_patterns = []

        # Häufige unvollständige Anfänge
        if words:
            last_word = words[-1]

            # Typische Fortsetzungen
            continuations = {
                "voraussetzungen": ["für", "des", "der", "einer"],
                "definition": ["von", "des", "der", "für"],
                "höhe": ["der", "des", "von"],
                "berechnung": ["der", "des", "von"],
                "rechtsprechung": ["zu", "des", "der", "über"],
                "unterschied": ["zwischen", "zu", "von"],
                "anspruch": ["auf", "aus", "gegen"],
            }

            if last_word in continuations:
                incomplete_patterns.extend(continuations[last_word])

        return incomplete_patterns


class PredictiveQueryEngine:
    """Haupt-Engine für Predictive Query Completion"""

    def __init__(self, db_path: str = "query_predictions.db"):
        self.db_path = db_path
        self.domain_knowledge = LegalDomainKnowledge()
        self.pattern_analyzer = QueryPatternAnalyzer()

        # Query-Historie und -Statistiken
        self.global_query_history = deque(maxlen=10000)
        self.user_contexts = {}
        self.query_statistics = defaultdict(int)
        self.completion_cache = {}

        # Vereinfachte Text-Ähnlichkeit ohne sklearn
        self.query_corpus: List[str] = []
        self.is_vectorizer_fitted = False

        # Threading für Echtzeit-Updates
        self.update_queue = deque()
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Konfiguration
        self.config = {
            "max_predictions": 8,
            "min_confidence": 0.3,
            "cache_ttl": 3600,  # 1 Stunde
            "enable_semantic_matching": True,
            "enable_user_personalization": True,
            "learning_rate": 0.1,
        }

        self._initialize_database()
        self._load_query_history()

    def _initialize_database(self):
        """Initialisiert SQLite-Datenbank für Query-Historie"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                completion TEXT,
                user_id TEXT,
                session_id TEXT,
                timestamp REAL,
                confidence REAL,
                used BOOLEAN DEFAULT FALSE,
                result_count INTEGER
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_patterns (
                user_id TEXT,
                pattern_type TEXT,
                pattern_data TEXT,
                frequency INTEGER,
                last_updated REAL,
                PRIMARY KEY (user_id, pattern_type)
            )
        """
        )

        conn.commit()
        conn.close()

    def _load_query_history(self):
        """Lädt Query-Historie aus Datenbank"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT query FROM query_history ORDER BY timestamp DESC LIMIT 1000"
        )
        queries = [row[0] for row in cursor.fetchall()]

        self.global_query_history.extend(queries)
        self.query_corpus = list(set(queries))  # Eindeutige Queries für TF-IDF

        if self.query_corpus:
            try:
                # Vereinfachte Initialisierung ohne sklearn
                self.is_vectorizer_fitted = True
            except Exception as e:
                print(f"Fehler bei der Initialisierung: {e}")

        conn.close()

    def predict_completions(
        self,
        partial_query: str,
        user_context: Optional[UserContext] = None,
        max_predictions: Optional[int] = None,
    ) -> List[QueryPrediction]:
        """Hauptfunktion für Query-Completion-Vorhersage"""

        max_preds = max_predictions or self.config["max_predictions"]

        if not partial_query.strip():
            return []

        # Cache-Check
        cache_key = f"{partial_query}_{user_context.session_id if user_context else 'anonymous'}"
        if cache_key in self.completion_cache:
            cached_result, timestamp = self.completion_cache[cache_key]
            if time.time() - timestamp < self.config["cache_ttl"]:
                return cached_result

        start_time = time.time()
        predictions = []

        # Pattern-Analyse der partiellen Query
        pattern_analysis = self.pattern_analyzer.analyze_query_pattern(partial_query)

        # Verschiedene Vorhersage-Strategien
        strategies = [
            self._prefix_matching_predictions,
            self._semantic_similarity_predictions,
            self._pattern_based_predictions,
            self._hierarchical_expansion_predictions,
            self._user_history_predictions,
            self._contextual_predictions,
        ]

        for strategy in strategies:
            try:
                strategy_predictions = strategy(
                    partial_query, user_context, pattern_analysis
                )
                predictions.extend(strategy_predictions)
            except Exception as e:
                print(f"Fehler in Vorhersage-Strategie {strategy.__name__}: {e}")

        # Deduplizierung und Ranking
        predictions = self._deduplicate_and_rank(predictions, partial_query)

        # Confidence-Filtering
        predictions = [
            p for p in predictions if p.confidence >= self.config["min_confidence"]
        ]

        # Top-K Selection
        final_predictions = predictions[:max_preds]

        # Performance-Enrichment
        processing_time = time.time() - start_time
        for pred in final_predictions:
            pred.processing_time_estimate = processing_time / len(final_predictions)

        # Cache speichern
        self.completion_cache[cache_key] = (final_predictions, time.time())

        return final_predictions

    def _prefix_matching_predictions(
        self,
        partial_query: str,
        user_context: Optional[UserContext],
        pattern_analysis: Dict,
    ) -> List[QueryPrediction]:
        """Prefix-basierte Vervollständigung"""

        predictions = []
        partial_lower = partial_query.lower().strip()

        # Exakte Prefix-Matches aus Historie
        for historical_query in self.global_query_history:
            if historical_query.lower().startswith(partial_lower) and len(
                historical_query
            ) > len(partial_query):
                confidence = self._calculate_prefix_confidence(
                    partial_query, historical_query
                )

                predictions.append(
                    QueryPrediction(
                        completion=historical_query,
                        confidence=confidence,
                        prediction_type="prefix_match",
                        source="global_history",
                        intent_category=pattern_analysis.get("query_type", "general"),
                        estimated_results=self._estimate_result_count(historical_query),
                        processing_time_estimate=0.05,
                        metadata={
                            "edit_distance": Levenshtein.distance(
                                partial_query, historical_query
                            )
                        },
                    )
                )

        return predictions

    def _semantic_similarity_predictions(
        self,
        partial_query: str,
        user_context: Optional[UserContext],
        pattern_analysis: Dict,
    ) -> List[QueryPrediction]:
        """Vereinfachte semantische Ähnlichkeits-basierte Vorhersagen"""

        predictions: List[QueryPrediction] = []

        if not self.is_vectorizer_fitted or not self.query_corpus:
            return predictions

        try:
            # Vereinfachte Wort-basierte Ähnlichkeit
            partial_words = set(partial_query.lower().split())

            for corpus_query in self.query_corpus:
                corpus_words = set(corpus_query.lower().split())

                # Jaccard-Ähnlichkeit
                intersection = len(partial_words & corpus_words)
                union = len(partial_words | corpus_words)
                similarity = intersection / union if union > 0 else 0.0

                if similarity > 0.3 and len(partial_words) > 0:
                    # Skip wenn zu ähnlich der ursprünglichen Query
                    edit_distance = self._simple_edit_distance(
                        partial_query.lower(), corpus_query.lower()
                    )
                    if edit_distance < 3:
                        continue

                    predictions.append(
                        QueryPrediction(
                            completion=corpus_query,
                            confidence=float(similarity * 0.8),
                            prediction_type="semantic_similar",
                            source="corpus_similarity",
                            intent_category=pattern_analysis.get(
                                "query_type", "general"
                            ),
                            estimated_results=self._estimate_result_count(corpus_query),
                            processing_time_estimate=0.1,
                            metadata={"similarity_score": float(similarity)},
                        )
                    )

        except Exception as e:
            print(f"Fehler bei semantischer Ähnlichkeitsberechnung: {e}")

        return predictions

    def _pattern_based_predictions(
        self,
        partial_query: str,
        user_context: Optional[UserContext],
        pattern_analysis: Dict,
    ) -> List[QueryPrediction]:
        """Pattern-basierte Vervollständigung"""

        predictions = []
        partial_lower = partial_query.lower().strip()
        words = partial_lower.split()

        if not words:
            return predictions

        # Häufige juristische Completion-Patterns
        completion_patterns = {
            "voraussetzungen": [
                "für einen kaufvertrag",
                "für eine kündigung",
                "für unterhalt",
                "des anspruchs",
            ],
            "definition": [
                "von vertrag",
                "von anspruch",
                "von kündigung",
                "des eigentums",
            ],
            "höhe": ["der miete", "des unterhalts", "der zinsen", "des schadens"],
            "berechnung": [
                "der zinsen",
                "des unterhalts",
                "der miete",
                "der verjährung",
            ],
            "rechtsprechung": [
                "des bgh",
                "zu § 433 bgb",
                "zum kaufvertrag",
                "zur kündigung",
            ],
            "unterschied": [
                "zwischen kauf und miete",
                "zwischen diebstahl und betrug",
                "zwischen mord und totschlag",
            ],
            "anspruch": [
                "auf schadensersatz",
                "auf unterhalt",
                "auf erfüllung",
                "aus vertrag",
            ],
        }

        last_word = words[-1] if words else ""

        if last_word in completion_patterns:
            for completion in completion_patterns[last_word]:
                full_completion = f"{partial_query} {completion}"

                predictions.append(
                    QueryPrediction(
                        completion=full_completion,
                        confidence=0.7,
                        prediction_type="pattern_based",
                        source="legal_patterns",
                        intent_category=pattern_analysis.get("query_type", "general"),
                        estimated_results=self._estimate_result_count(full_completion),
                        processing_time_estimate=0.08,
                        metadata={"pattern_type": last_word},
                    )
                )

        return predictions

    def _hierarchical_expansion_predictions(
        self,
        partial_query: str,
        user_context: Optional[UserContext],
        pattern_analysis: Dict,
    ) -> List[QueryPrediction]:
        """Hierarchische Domain-Expansion"""

        predictions = []

        # Domain-spezifische Erweiterungen
        domain_expansions = self.domain_knowledge.get_domain_expansions(partial_query)

        for expansion in domain_expansions[:5]:  # Limit auf Top-5
            predictions.append(
                QueryPrediction(
                    completion=expansion,
                    confidence=0.6,
                    prediction_type="hierarchical_expansion",
                    source="domain_knowledge",
                    intent_category=pattern_analysis.get("query_type", "general"),
                    estimated_results=self._estimate_result_count(expansion),
                    processing_time_estimate=0.12,
                    metadata={"expansion_type": "hierarchical"},
                )
            )

        # Synonyme Varianten
        synonymous_queries = self.domain_knowledge.get_synonymous_queries(partial_query)

        for syn_query in synonymous_queries[:3]:  # Limit auf Top-3
            predictions.append(
                QueryPrediction(
                    completion=syn_query,
                    confidence=0.5,
                    prediction_type="synonymous",
                    source="legal_synonyms",
                    intent_category=pattern_analysis.get("query_type", "general"),
                    estimated_results=self._estimate_result_count(syn_query),
                    processing_time_estimate=0.1,
                    metadata={"synonym_type": "legal"},
                )
            )

        return predictions

    def _user_history_predictions(
        self,
        partial_query: str,
        user_context: Optional[UserContext],
        pattern_analysis: Dict,
    ) -> List[QueryPrediction]:
        """Benutzer-spezifische Historie-basierte Vorhersagen"""

        predictions = []

        if not user_context or not user_context.query_history:
            return predictions

        partial_lower = partial_query.lower()

        # Ähnliche Queries aus User-Historie
        for historical_query in user_context.query_history[-50:]:  # Letzte 50 Queries
            if (
                historical_query.lower().startswith(partial_lower)
                or Levenshtein.distance(partial_lower, historical_query.lower()) <= 2
            ):

                # Personalisierte Confidence basierend auf Präferenzen
                base_confidence = 0.8

                # Adjust basierend auf Topic-Präferenzen
                for topic, preference in user_context.topic_preferences.items():
                    if topic in historical_query.lower():
                        base_confidence *= 1.0 + preference * 0.3

                predictions.append(
                    QueryPrediction(
                        completion=historical_query,
                        confidence=min(base_confidence, 0.95),
                        prediction_type="user_history",
                        source="personal_history",
                        intent_category=pattern_analysis.get("query_type", "general"),
                        estimated_results=self._estimate_result_count(historical_query),
                        processing_time_estimate=0.06,
                        metadata={"personalization_boost": base_confidence - 0.8},
                    )
                )

        return predictions

    def _contextual_predictions(
        self,
        partial_query: str,
        user_context: Optional[UserContext],
        pattern_analysis: Dict,
    ) -> List[QueryPrediction]:
        """Kontextuelle Vorhersagen basierend auf aktueller Session"""

        predictions = []

        if not user_context:
            return predictions

        # Time-of-day basierte Anpassungen
        time_based_topics = {
            "morning": ["fristen", "termine", "verfahren"],
            "afternoon": ["berechnung", "höhe", "zinsen"],
            "evening": ["rechtsprechung", "urteile", "definition"],
        }

        current_time = user_context.time_of_day
        if current_time in time_based_topics:
            for topic in time_based_topics[current_time]:
                if topic in partial_query.lower():
                    contextual_completion = f"{partial_query} {topic} aktuell"

                    predictions.append(
                        QueryPrediction(
                            completion=contextual_completion,
                            confidence=0.4,
                            prediction_type="contextual",
                            source="time_context",
                            intent_category=pattern_analysis.get(
                                "query_type", "general"
                            ),
                            estimated_results=self._estimate_result_count(
                                contextual_completion
                            ),
                            processing_time_estimate=0.05,
                            metadata={"context_type": "temporal"},
                        )
                    )

        # Recent Documents basierte Vorhersagen
        if user_context.recent_documents:
            for doc in user_context.recent_documents[-3:]:  # Letzte 3 Dokumente
                if any(word in doc.lower() for word in partial_query.lower().split()):
                    doc_based_completion = f"{partial_query} in {doc}"

                    predictions.append(
                        QueryPrediction(
                            completion=doc_based_completion,
                            confidence=0.35,
                            prediction_type="contextual",
                            source="recent_documents",
                            intent_category=pattern_analysis.get(
                                "query_type", "general"
                            ),
                            estimated_results=self._estimate_result_count(
                                doc_based_completion
                            ),
                            processing_time_estimate=0.07,
                            metadata={"context_type": "document"},
                        )
                    )

        return predictions

    def _simple_edit_distance(self, s1: str, s2: str) -> int:
        """Einfache Edit-Distance-Berechnung (Levenshtein-ähnlich)"""
        if len(s1) < len(s2):
            return self._simple_edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _calculate_prefix_confidence(self, partial: str, full: str) -> float:
        """Berechnet Confidence für Prefix-Matches"""

        if not partial or not full:
            return 0.0

        # Basis-Confidence basierend auf Längen-Verhältnis
        length_ratio = len(partial) / len(full)
        base_confidence = 0.5 + (length_ratio * 0.4)

        # Bonus für häufige Queries
        frequency_bonus = min(self.query_statistics.get(full, 0) * 0.1, 0.3)

        # Malus für sehr kurze partielle Queries
        if len(partial) < 3:
            base_confidence *= 0.5

        return min(base_confidence + frequency_bonus, 0.95)

    def _estimate_result_count(self, query: str) -> int:
        """Schätzt die Anzahl der Ergebnisse für eine Query"""

        # Heuristische Schätzung basierend auf Query-Eigenschaften
        word_count = len(query.split())

        if word_count <= 2:
            return np.random.randint(50, 200)
        elif word_count <= 4:
            return np.random.randint(20, 100)
        else:
            return np.random.randint(5, 50)

    def _deduplicate_and_rank(
        self, predictions: List[QueryPrediction], partial_query: str
    ) -> List[QueryPrediction]:
        """Dedupliziert und rankt Vorhersagen"""

        # Deduplizierung basierend auf Completion-Text
        seen = set()
        unique_predictions = []

        for pred in predictions:
            completion_key = pred.completion.lower().strip()
            if (
                completion_key not in seen
                and completion_key != partial_query.lower().strip()
            ):
                seen.add(completion_key)
                unique_predictions.append(pred)

        # Ranking nach Confidence und zusätzlichen Faktoren
        def ranking_score(pred: QueryPrediction) -> float:
            score = pred.confidence

            # Bonus für bestimmte Prediction-Types
            type_bonuses = {
                "prefix_match": 0.1,
                "user_history": 0.15,
                "semantic_similar": 0.05,
                "pattern_based": 0.08,
            }

            score += type_bonuses.get(pred.prediction_type, 0.0)

            # Bonus für juristische Spezifität
            if pred.intent_category in ["definition", "procedure", "requirements"]:
                score += 0.05

            return score

        # Sortierung nach Ranking-Score
        unique_predictions.sort(key=ranking_score, reverse=True)

        return unique_predictions

    def update_query_usage(
        self, query: str, completion: str, user_context: Optional[UserContext] = None
    ):
        """Updated Query-Usage für Learning"""

        # Statistiken aktualisieren
        self.query_statistics[completion] += 1

        # In Datenbank speichern
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO query_history (query, completion, user_id, session_id, timestamp, used)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                query,
                completion,
                user_context.session_id if user_context else "anonymous",
                user_context.session_id if user_context else "anonymous",
                time.time(),
                True,
            ),
        )

        conn.commit()
        conn.close()

        # Cache invalidieren
        self.completion_cache.clear()

    def get_completion_analytics(self) -> Dict[str, any]:
        """Gibt Analytics über Completion-Performance zurück"""

        return {
            "total_queries_processed": len(self.global_query_history),
            "unique_completions": len(set(self.global_query_history)),
            "cache_hit_rate": len(self.completion_cache)
            / max(len(self.global_query_history), 1),
            "most_common_queries": Counter(self.query_statistics).most_common(10),
            "prediction_type_distribution": self._get_prediction_type_stats(),
            "user_contexts_active": len(self.user_contexts),
            "vocabulary_size": (
                len(self.query_corpus) if self.is_vectorizer_fitted else 0
            ),
        }

    def _get_prediction_type_stats(self) -> Dict[str, int]:
        """Berechnet Statistiken über Prediction-Types"""

        # Vereinfachte Statistik (in echter Implementierung aus Datenbank)
        return {
            "prefix_match": 45,
            "semantic_similar": 25,
            "pattern_based": 15,
            "user_history": 10,
            "contextual": 5,
        }


# === TESTING & DEMO ===


def demonstrate_predictive_completion():
    """Demonstriert das Predictive Query Completion System"""

    print("🔮 PREDICTIVE QUERY COMPLETION - DEMO")
    print("=" * 50)

    # Initialize Engine
    engine = PredictiveQueryEngine()

    # Simuliere etwas Query-Historie
    sample_queries = [
        "voraussetzungen kaufvertrag bgb",
        "definition anspruch schadensersatz",
        "höhe miete berechnung",
        "rechtsprechung bgh kaufvertrag",
        "kündigung arbeitsvertrag frist",
        "unterhalt berechnung kinder",
        "voraussetzungen für kündigung",
        "definition von eigentum",
        "höhe der zinsen berechnung",
    ]

    engine.global_query_history.extend(sample_queries)
    engine.query_corpus = sample_queries
    engine.is_vectorizer_fitted = True

    # Teste verschiedene partielle Queries
    test_queries = [
        "voraussetzungen",
        "definition",
        "höhe der",
        "rechtsprechung",
        "kündigung",
    ]

    # Simuliere User-Context
    user_context = UserContext(
        session_id="demo_session",
        query_history=["kaufvertrag bgb", "miete höhe", "anspruch"],
        topic_preferences={"vertragsrecht": 0.8, "mietrecht": 0.6},
        complexity_preference=0.7,
        search_patterns={"definition": 5, "voraussetzungen": 3},
        time_of_day="afternoon",
        recent_documents=["lehrbuch_vertragsrecht.pdf"],
        expertise_level=0.6,
    )

    for partial_query in test_queries:
        print(f"\n🔍 Partielle Query: '{partial_query}'")
        print("-" * 30)

        start_time = time.time()
        predictions = engine.predict_completions(
            partial_query=partial_query, user_context=user_context, max_predictions=5
        )
        completion_time = time.time() - start_time

        print(f"⏱️  Completion-Zeit: {completion_time:.4f}s")
        print(f"🎯 {len(predictions)} Vorhersagen:")

        for i, pred in enumerate(predictions, 1):
            print(f"   {i}. '{pred.completion}'")
            print(
                f"      Confidence: {pred.confidence:.3f} | Type: {pred.prediction_type}"
            )
            print(f"      Source: {pred.source} | Results: ~{pred.estimated_results}")

    # Analytics
    print("\n📈 COMPLETION ANALYTICS:")
    analytics = engine.get_completion_analytics()

    print(f"📊 Queries verarbeitet: {analytics['total_queries_processed']}")
    print(f"🎯 Eindeutige Completions: {analytics['unique_completions']}")
    print(f"💾 Cache-Hit-Rate: {analytics['cache_hit_rate']:.2%}")

    print("\n🏆 Häufigste Queries:")
    for query, count in analytics["most_common_queries"][:5]:
        print(f"   • '{query}': {count}x")


if __name__ == "__main__":
    demonstrate_predictive_completion()
