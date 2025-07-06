"""
🧠 SEMANTIC QUERY ENHANCEMENT ENGINE
===================================

Intelligente Query-Verbesserung durch Expansion, Intent-Detection und Kontext-Anreicherung
"""

import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class QueryEnhancement:
    """Query-Enhancement Resultat"""

    original_query: str
    enhanced_query: str
    intent: str
    intent_confidence: float
    expansions: List[str]
    synonyms: List[str]
    legal_entities: List[str]
    enhancement_score: float
    processing_time: float


class JuristischeSynonymErweiterung:
    """Erweitert Queries um juristische Synonyme und Fachbegriffe"""

    def __init__(self):
        # Juristische Synonymgruppen
        self.synonym_groups = {
            # Grundlegende Rechtsbegriffe
            "vertrag": ["vereinbarung", "kontrakt", "abkommen", "übereinkunft"],
            "eigentum": ["besitz", "eigentumsrecht", "vermögensrecht"],
            "haftung": ["verantwortlichkeit", "haftpflicht", "schuld", "verschulden"],
            "anspruch": ["forderung", "berechtigung", "recht", "anwartschaft"],
            "verjährung": ["verwirkung", "erlöschen", "zeitablauf"],
            # Vertragsrecht
            "kaufvertrag": ["kauf", "erwerbsgeschäft", "veräußerung"],
            "miete": ["mietvertrag", "mietverhältnis", "pacht"],
            "werkvertrag": ["werklohn", "auftragsarbeit", "werkleistung"],
            "dienstvertrag": [
                "dienstleistung",
                "arbeitsvertrag",
                "dienstverpflichtung",
            ],
            # Sachenrecht
            "pfandrecht": ["sicherheit", "besicherung", "pfand"],
            "hypothek": ["grundschuld", "grundpfandrecht", "reallast"],
            "erbbaurecht": ["baurecht", "erbbau"],
            # Schuldrecht
            "schadensersatz": [
                "schadenersatz",
                "ersatz",
                "wiedergutmachung",
                "entschädigung",
            ],
            "unmöglichkeit": ["leistungshindernis", "erfüllungshindernis"],
            "verzug": ["verspätung", "säumnis", "pflichtverletzung"],
            # Familienrecht
            "ehe": ["eheschließung", "heirat", "eheliche gemeinschaft"],
            "scheidung": ["ehescheidung", "eheauflösung", "trennung"],
            "unterhalt": ["unterhaltspflicht", "alimentierung", "versorgung"],
            # Gesellschaftsrecht
            "gesellschaft": [
                "personengesellschaft",
                "kapitalgesellschaft",
                "unternehmen",
            ],
            "gmbh": ["gesellschaft mit beschränkter haftung", "kapitalgesellschaft"],
            "ag": ["aktiengesellschaft", "börsenunternehmen"],
            # Strafrecht
            "vorsatz": ["absicht", "wille", "intention"],
            "fahrlässigkeit": ["nachlässigkeit", "sorgfaltspflichtverletzung"],
            "schuld": ["verschulden", "verantwortlichkeit", "vorwerfbarkeit"],
        }

        # Umgekehrte Zuordnung erstellen
        self.word_to_synonyms = {}
        for main_word, synonyms in self.synonym_groups.items():
            self.word_to_synonyms[main_word] = synonyms
            for synonym in synonyms:
                self.word_to_synonyms[synonym] = [main_word] + [
                    s for s in synonyms if s != synonym
                ]

    def expand_query(
        self, query: str, max_expansions: int = 3
    ) -> Tuple[List[str], List[str]]:
        """Erweitert Query um Synonyme"""

        query_lower = query.lower()
        words = re.findall(r"\b\w+\b", query_lower)

        expansions = []
        found_synonyms = []

        for word in words:
            if word in self.word_to_synonyms:
                synonyms = self.word_to_synonyms[word][:max_expansions]
                found_synonyms.extend(synonyms)

                # Query-Varianten erstellen
                for synonym in synonyms:
                    expanded_query = re.sub(
                        r"\b" + re.escape(word) + r"\b",
                        synonym,
                        query_lower,
                        flags=re.IGNORECASE,
                    )
                    if expanded_query != query_lower:
                        expansions.append(expanded_query)

        return expansions[:max_expansions], found_synonyms


class JuristischeEntityExtraktion:
    """Extrahiert juristische Entitäten aus Queries"""

    def __init__(self):
        # Patterns für juristische Entitäten
        self.entity_patterns = {
            "paragraph": r"§\s*(\d+[a-z]?(?:\s+Abs\.?\s+\d+)?(?:\s+S\.?\s+\d+)?)",
            "article": r"Art\.?\s*(\d+[a-z]?(?:\s+Abs\.?\s+\d+)?(?:\s+S\.?\s+\d+)?)",
            "bgb": r"§\s*(\d+)\s+BGB",
            "stgb": r"§\s*(\d+)\s+StGB",
            "gg": r"Art\.?\s*(\d+)\s+GG",
            "zpo": r"§\s*(\d+)\s+ZPO",
            "court_decision": r"(BGH|BVerfG|BFH|BVerwG)\s+(?:Urt\.|Beschl\.?)\s+v\.\s+(\d{1,2}\.\d{1,2}\.\d{4})",
            "az": r"(\d+\s+[A-Z]+\s+\d+/\d+)",  # Aktenzeichen
            "date": r"(\d{1,2}\.\d{1,2}\.\d{4})",
            "euro": r"(\d+(?:\.\d{3})*(?:,\d{2})?\s*€)",
            "prozent": r"(\d+(?:,\d+)?\s*%)",
        }

    def extract_entities(self, query: str) -> List[Dict]:
        """Extrahiert juristische Entitäten"""

        entities = []

        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, query, re.IGNORECASE)

            for match in matches:
                entity = {
                    "type": entity_type,
                    "text": match.group(0),
                    "value": match.group(1) if match.groups() else match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                }
                entities.append(entity)

        return entities


class QueryIntentClassifier:
    """Klassifiziert die Absicht hinter einer Query"""

    def __init__(self):
        # Intent-Patterns mit Gewichtungen
        self.intent_patterns = {
            "definition": {
                "patterns": [
                    r"was ist",
                    r"was bedeutet",
                    r"definition von",
                    r"begriff",
                    r"versteht man unter",
                    r"bezeichnet",
                    r"definiert als",
                ],
                "weight": 1.0,
            },
            "voraussetzungen": {
                "patterns": [
                    r"voraussetzung",
                    r"bedingung",
                    r"erforderlich",
                    r"notwendig",
                    r"muss",
                    r"brauch",
                    r"setzt voraus",
                    r"erfordert",
                ],
                "weight": 0.9,
            },
            "rechtsfolgen": {
                "patterns": [
                    r"folge",
                    r"konsequenz",
                    r"bewirkt",
                    r"führt zu",
                    r"ergebnis",
                    r"wirkung",
                    r"auswirkung",
                    r"hat zur folge",
                ],
                "weight": 0.9,
            },
            "vergleich": {
                "patterns": [
                    r"unterschied",
                    r"vergleich",
                    r"vs",
                    r"gegenüber",
                    r"abgrenzung",
                    r"differenz",
                    r"im gegensatz",
                    r"unterscheidet sich",
                ],
                "weight": 0.8,
            },
            "prozess": {
                "patterns": [
                    r"wie",
                    r"ablauf",
                    r"verfahren",
                    r"vorgehen",
                    r"schritt",
                    r"prozess",
                    r"durchführung",
                    r"vorgehensweise",
                ],
                "weight": 0.8,
            },
            "rechtsprechung": {
                "patterns": [
                    r"urteil",
                    r"entscheidung",
                    r"rechtsprechung",
                    r"gericht",
                    r"bgh",
                    r"bverfg",
                    r"richter",
                    r"gerichtshof",
                ],
                "weight": 0.7,
            },
            "praxis": {
                "patterns": [
                    r"beispiel",
                    r"fall",
                    r"praxis",
                    r"anwendung",
                    r"konkret",
                    r"praktisch",
                    r"in der praxis",
                ],
                "weight": 0.6,
            },
            "berechnung": {
                "patterns": [
                    r"berechnung",
                    r"höhe",
                    r"betrag",
                    r"summe",
                    r"kosten",
                    r"wert",
                    r"€",
                    r"euro",
                    r"prozent",
                ],
                "weight": 0.8,
            },
        }

    def classify(self, query: str) -> Tuple[str, float]:
        """Klassifiziert Query-Intent"""

        query_lower = query.lower()
        intent_scores = {}

        for intent, config in self.intent_patterns.items():
            score = 0
            pattern_matches = 0

            for pattern in config["patterns"]:
                matches = len(re.findall(pattern, query_lower))
                if matches > 0:
                    pattern_matches += 1
                    score += matches * config["weight"]

            if pattern_matches > 0:
                # Normalisierung basierend auf Anzahl Pattern
                normalized_score = score / len(config["patterns"])
                intent_scores[intent] = normalized_score

        if not intent_scores:
            return "general", 0.3

        # Bester Intent
        if intent_scores:
            best_intent = max(intent_scores.keys(), key=lambda k: intent_scores[k])
            confidence = min(intent_scores[best_intent], 1.0)
        else:
            best_intent = "general"
            confidence = 0.3

        return best_intent, confidence


class ContextualQueryExpander:
    """Erweitert Queries basierend auf Kontext und Intent"""

    def __init__(self):
        # Intent-spezifische Expansion-Templates
        self.expansion_templates = {
            "definition": [
                "{query} bedeutung",
                "{query} begriff erklärung",
                "was versteht man unter {query}",
                "{query} definition rechtlich",
            ],
            "voraussetzungen": [
                "{query} bedingungen",
                "wann {query}",
                "{query} erforderlich",
                "{query} voraussetzung erfüllt",
            ],
            "rechtsfolgen": [
                "{query} auswirkungen",
                "{query} konsequenzen",
                "folgen von {query}",
                "{query} rechtliche wirkung",
            ],
            "vergleich": [
                "{query} unterschiede",
                "{query} abgrenzung",
                "{query} vs",
                "{query} vergleich anderen",
            ],
            "prozess": [
                "{query} ablauf",
                "wie funktioniert {query}",
                "{query} verfahren schritt",
                "{query} durchführung",
            ],
            "rechtsprechung": [
                "{query} urteile",
                "{query} rechtsprechung",
                "{query} gerichtsentscheidungen",
                "{query} bgh",
            ],
            "praxis": [
                "{query} beispiele",
                "{query} praxis anwendung",
                "{query} praktische fälle",
                "{query} konkret",
            ],
            "berechnung": [
                "{query} höhe berechnung",
                "{query} kosten",
                "{query} betrag ermittlung",
                "wie hoch {query}",
            ],
        }

    def expand_by_intent(
        self, query: str, intent: str, max_expansions: int = 2
    ) -> List[str]:
        """Erweitert Query basierend auf Intent"""

        if intent not in self.expansion_templates:
            return []

        # Kernbegriff aus Query extrahieren
        core_term = self._extract_core_term(query)

        expansions = []
        templates = self.expansion_templates[intent][:max_expansions]

        for template in templates:
            expanded = template.format(query=core_term)
            if expanded.lower() != query.lower():
                expansions.append(expanded)

        return expansions

    def _extract_core_term(self, query: str) -> str:
        """Extrahiert Kernbegriff aus Query"""

        # Stopwords entfernen
        stopwords = {
            "was",
            "ist",
            "sind",
            "der",
            "die",
            "das",
            "ein",
            "eine",
            "und",
            "oder",
            "bei",
            "für",
            "von",
            "zu",
            "in",
            "an",
            "auf",
            "mit",
            "nach",
            "vor",
            "wie",
            "wenn",
            "wann",
            "wo",
            "warum",
            "welche",
            "welcher",
            "welches",
        }

        words = re.findall(r"\b\w+\b", query.lower())
        core_words = [w for w in words if w not in stopwords and len(w) > 3]

        if core_words:
            return " ".join(core_words[:3])  # Max 3 Kernwörter
        else:
            return query.strip()


class SemanticQueryEnhancer:
    """Haupt-Engine für semantische Query-Enhancement"""

    def __init__(self):
        self.synonym_expander = JuristischeSynonymErweiterung()
        self.entity_extractor = JuristischeEntityExtraktion()
        self.intent_classifier = QueryIntentClassifier()
        self.contextual_expander = ContextualQueryExpander()

        # Statistiken
        self.enhancement_stats = {
            "total_queries": 0,
            "enhanced_queries": 0,
            "intent_distribution": Counter(),
            "avg_enhancement_score": 0.0,
            "processing_times": [],
        }

    def enhance_query(
        self,
        query: str,
        enable_synonyms: bool = True,
        enable_intent_expansion: bool = True,
        enable_entity_extraction: bool = True,
    ) -> QueryEnhancement:
        """Hauptmethode für Query-Enhancement"""

        start_time = time.time()
        self.enhancement_stats["total_queries"] += 1

        # 1. Intent-Klassifizierung
        intent, intent_confidence = self.intent_classifier.classify(query)
        self.enhancement_stats["intent_distribution"][intent] += 1

        # 2. Entity-Extraktion
        entities = []
        if enable_entity_extraction:
            entities = self.entity_extractor.extract_entities(query)

        # 3. Synonym-Expansion
        expansions = []
        synonyms = []
        if enable_synonyms:
            expansion_result = self.synonym_expander.expand_query(query)
            expansions = expansion_result[0]
            synonyms = expansion_result[1]

        # 4. Intent-basierte Expansion
        if enable_intent_expansion:
            intent_expansions = self.contextual_expander.expand_by_intent(query, intent)
            expansions.extend(intent_expansions)

        # 5. Enhanced Query erstellen
        enhanced_query = self._create_enhanced_query(
            query, expansions, entities, intent
        )

        # 6. Enhancement-Score berechnen
        enhancement_score = self._calculate_enhancement_score(
            query, enhanced_query, intent_confidence, len(synonyms), len(entities)
        )

        processing_time = time.time() - start_time
        self.enhancement_stats["processing_times"].append(processing_time)

        if enhanced_query != query:
            self.enhancement_stats["enhanced_queries"] += 1

        # Enhancement-Objekt erstellen
        enhancement = QueryEnhancement(
            original_query=query,
            enhanced_query=enhanced_query,
            intent=intent,
            intent_confidence=intent_confidence,
            expansions=expansions[:5],  # Top 5 Expansionen
            synonyms=synonyms[:3],  # Top 3 Synonyme
            legal_entities=[e["text"] for e in entities],
            enhancement_score=enhancement_score,
            processing_time=processing_time,
        )

        return enhancement

    def _create_enhanced_query(
        self,
        original_query: str,
        expansions: List[str],
        entities: List[Dict],
        intent: str,
    ) -> str:
        """Erstellt optimierte Enhanced Query"""

        enhanced_parts = [original_query]

        # Intent-spezifische Erweiterungen
        if intent in ["definition", "voraussetzungen", "rechtsfolgen"]:
            # Für diese Intents sind Synonyme besonders wertvoll
            if expansions:
                enhanced_parts.append(expansions[0])

        elif intent == "rechtsprechung":
            # Für Rechtsprechung: Gerichtsbezug verstärken
            enhanced_parts.append(f"{original_query} gericht urteil")

        elif intent == "praxis":
            # Für Praxis: Beispiele und Anwendung
            enhanced_parts.append(f"{original_query} beispiel anwendung")

        # Entity-Enhancement
        entity_terms = []
        for entity in entities:
            if entity["type"] in ["paragraph", "article", "bgb", "stgb", "gg"]:
                entity_terms.append(entity["text"])

        if entity_terms:
            enhanced_parts.extend(entity_terms)

        # Finale Enhanced Query (begrenzt auf reasonable Länge)
        enhanced_query = " ".join(enhanced_parts[:3])  # Max 3 Teile

        return enhanced_query

    def _calculate_enhancement_score(
        self,
        original: str,
        enhanced: str,
        intent_confidence: float,
        synonym_count: int,
        entity_count: int,
    ) -> float:
        """Berechnet Enhancement-Score"""

        # Basis-Score von Intent-Confidence
        score = intent_confidence * 0.4

        # Synonym-Bonus
        synonym_bonus = min(synonym_count * 0.1, 0.3)
        score += synonym_bonus

        # Entity-Bonus
        entity_bonus = min(entity_count * 0.15, 0.3)
        score += entity_bonus

        # Längen-Verbesserung (mehr Kontext)
        if len(enhanced) > len(original):
            length_bonus = min(
                (len(enhanced) - len(original)) / len(original) * 0.2, 0.2
            )
            score += length_bonus

        return min(score, 1.0)

    def get_query_suggestions(self, query: str, top_k: int = 3) -> List[str]:
        """Liefert Query-Verbesserungsvorschläge"""

        enhancement = self.enhance_query(query)

        suggestions = []

        # 1. Enhanced Query
        if enhancement.enhanced_query != query:
            suggestions.append(enhancement.enhanced_query)

        # 2. Synonym-Varianten
        for expansion in enhancement.expansions[:2]:
            if expansion not in suggestions:
                suggestions.append(expansion)

        # 3. Intent-spezifische Vorschläge
        intent_suggestions = self._get_intent_specific_suggestions(
            query, enhancement.intent
        )
        for suggestion in intent_suggestions:
            if suggestion not in suggestions and len(suggestions) < top_k:
                suggestions.append(suggestion)

        return suggestions[:top_k]

    def _get_intent_specific_suggestions(self, query: str, intent: str) -> List[str]:
        """Liefert Intent-spezifische Verbesserungsvorschläge"""

        suggestions = []

        if intent == "definition":
            suggestions.extend(
                [
                    f"Was bedeutet {query} rechtlich?",
                    f"Definition von {query} im Recht",
                    f"{query} Begriff Erklärung",
                ]
            )

        elif intent == "voraussetzungen":
            suggestions.extend(
                [
                    f"Welche Bedingungen für {query}?",
                    f"Wann ist {query} erfüllt?",
                    f"{query} Voraussetzungen prüfen",
                ]
            )

        elif intent == "rechtsfolgen":
            suggestions.extend(
                [
                    f"Welche Folgen hat {query}?",
                    f"{query} rechtliche Konsequenzen",
                    f"Auswirkungen von {query}",
                ]
            )

        return suggestions

    def analyze_query_quality(self, query: str) -> Dict:
        """Analysiert Query-Qualität und gibt Verbesserungsvorschläge"""

        enhancement = self.enhance_query(query)

        quality_metrics = {
            "length_score": min(len(query) / 50, 1.0),  # Optimal: ~50 Zeichen
            "specificity_score": len(enhancement.legal_entities) * 0.3,
            "intent_clarity": enhancement.intent_confidence,
            "enhancement_potential": enhancement.enhancement_score,
        }

        overall_quality = sum(quality_metrics.values()) / len(quality_metrics)

        # Verbesserungsvorschläge
        improvements = []

        if quality_metrics["length_score"] < 0.3:
            improvements.append("Query zu kurz - fügen Sie mehr Details hinzu")

        if quality_metrics["specificity_score"] < 0.2:
            improvements.append("Erwähnen Sie spezifische Paragraphen oder Gesetze")

        if quality_metrics["intent_clarity"] < 0.5:
            improvements.append("Machen Sie Ihre Frage präziser")

        if not enhancement.legal_entities:
            improvements.append("Verwenden Sie juristische Fachbegriffe")

        return {
            "overall_quality": overall_quality,
            "metrics": quality_metrics,
            "improvements": improvements,
            "suggestions": self.get_query_suggestions(query),
            "detected_intent": enhancement.intent,
            "confidence": enhancement.intent_confidence,
        }

    def get_enhancement_statistics(self) -> Dict:
        """Liefert Enhancement-Statistiken"""

        if self.enhancement_stats["processing_times"]:
            avg_processing_time = sum(self.enhancement_stats["processing_times"]) / len(
                self.enhancement_stats["processing_times"]
            )
        else:
            avg_processing_time = 0

        enhancement_rate = (
            self.enhancement_stats["enhanced_queries"]
            / max(self.enhancement_stats["total_queries"], 1)
        ) * 100

        return {
            "total_queries_processed": self.enhancement_stats["total_queries"],
            "queries_enhanced": self.enhancement_stats["enhanced_queries"],
            "enhancement_rate": round(enhancement_rate, 2),
            "avg_processing_time": round(avg_processing_time * 1000, 2),  # ms
            "intent_distribution": dict(self.enhancement_stats["intent_distribution"]),
            "most_common_intent": (
                self.enhancement_stats["intent_distribution"].most_common(1)[0]
                if self.enhancement_stats["intent_distribution"]
                else "none"
            ),
        }
