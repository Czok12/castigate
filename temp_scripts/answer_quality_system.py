"""
🎯 INTELLIGENTE ANTWORT-QUALITÄTSBEWERTUNG
==========================================

Confidence-Scoring, Fact-Checking und strukturierte Antwort-Templates
"""

import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import numpy as np


class ConfidenceLevel(Enum):
    """Confidence-Level für Antworten"""

    VERY_HIGH = "sehr_hoch"
    HIGH = "hoch"
    MEDIUM = "mittel"
    LOW = "niedrig"
    VERY_LOW = "sehr_niedrig"


@dataclass
class AnswerQualityMetrics:
    """Metriken für Antwortqualität"""

    confidence_score: float
    confidence_level: ConfidenceLevel
    source_reliability: float
    legal_accuracy: float
    completeness_score: float
    clarity_score: float
    citation_quality: float
    factual_consistency: float


class LegalFactChecker:
    """Fact-Checking für juristische Inhalte"""

    def __init__(self):
        # Bekannte juristische Fakten und Patterns
        self.legal_facts_patterns = {
            "paragraph_references": r"§\s*(\d+[a-z]?)\s*(?:Abs\.\s*(\d+))?\s*([A-Z]+)",
            "article_references": r"Art\.\s*(\d+[a-z]?)\s*(?:Abs\.\s*(\d+))?\s*([A-Z]+)",
            "court_decisions": r"(BGH|BVerfG|BFH|BAG|BSG|BVerwG).*?(\d{1,2}\.\d{1,2}\.\d{4})",
            "legal_principles": [
                "Treu und Glauben",
                "Verhältnismäßigkeitsprinzip",
                "Bestimmtheitsgrundsatz",
                "Rechtssicherheit",
                "Gewaltenteilung",
            ],
        }

        # Juristische Schlüsselbegriffe für Konsistenz-Check
        self.key_legal_terms = {
            "vertragsrecht": ["angebot", "annahme", "willenserklärung", "vertrag"],
            "schadenersatz": [
                "schaden",
                "kausalität",
                "verschulden",
                "rechtswidrigkeit",
            ],
            "eigentum": ["besitz", "eigentümer", "herausgabeanspruch", "vindikation"],
            "deliktsrecht": ["unerlaubte handlung", "schädiger", "geschädigter"],
        }

    def check_legal_consistency(self, answer: str, sources: List[Dict]) -> Dict:
        """Prüft juristische Konsistenz der Antwort"""

        consistency_score = 0.0
        issues = []

        # 1. Paragraph/Artikel-Referenzen prüfen
        answer_refs = self._extract_legal_references(answer)
        source_refs = []

        for source in sources:
            source_refs.extend(
                self._extract_legal_references(source.get("content", ""))
            )

        # Konsistenz der Referenzen
        ref_consistency = self._check_reference_consistency(answer_refs, source_refs)
        consistency_score += ref_consistency * 0.3

        if ref_consistency < 0.7:
            issues.append(
                "Inkonsistente Gesetzesreferenzen zwischen Antwort und Quellen"
            )

        # 2. Juristische Terminologie prüfen
        term_consistency = self._check_terminology_consistency(answer, sources)
        consistency_score += term_consistency * 0.3

        # 3. Logische Strukturierung prüfen
        structure_score = self._check_legal_structure(answer)
        consistency_score += structure_score * 0.4

        if structure_score < 0.5:
            issues.append("Antwort folgt nicht der juristischen Argumentationsstruktur")

        return {
            "consistency_score": min(consistency_score, 1.0),
            "reference_consistency": ref_consistency,
            "terminology_consistency": term_consistency,
            "structure_score": structure_score,
            "issues": issues,
        }

    def _extract_legal_references(self, text: str) -> List[Dict]:
        """Extrahiert juristische Referenzen aus Text"""

        references = []

        # Paragraph-Referenzen
        paragraph_matches = re.finditer(
            self.legal_facts_patterns["paragraph_references"], text, re.IGNORECASE
        )

        for match in paragraph_matches:
            references.append(
                {
                    "type": "paragraph",
                    "law": match.group(3) if len(match.groups()) >= 3 else "unknown",
                    "section": match.group(1),
                    "subsection": match.group(2) if match.group(2) else None,
                }
            )

        # Artikel-Referenzen
        article_matches = re.finditer(
            self.legal_facts_patterns["article_references"], text, re.IGNORECASE
        )

        for match in article_matches:
            references.append(
                {
                    "type": "article",
                    "law": match.group(3) if len(match.groups()) >= 3 else "unknown",
                    "section": match.group(1),
                    "subsection": match.group(2) if match.group(2) else None,
                }
            )

        return references

    def _check_reference_consistency(
        self, answer_refs: List[Dict], source_refs: List[Dict]
    ) -> float:
        """Prüft Konsistenz der Gesetzesreferenzen"""

        if not answer_refs and not source_refs:
            return 1.0  # Beide haben keine Referenzen = konsistent

        if not answer_refs or not source_refs:
            return 0.5  # Eine Seite hat Referenzen, andere nicht = teilweise konsistent

        # Vergleiche Referenzen
        consistent_refs = 0

        for answer_ref in answer_refs:
            for source_ref in source_refs:
                if (
                    answer_ref["type"] == source_ref["type"]
                    and answer_ref["law"] == source_ref["law"]
                    and answer_ref["section"] == source_ref["section"]
                ):
                    consistent_refs += 1
                    break

        return consistent_refs / len(answer_refs) if answer_refs else 0.0

    def _check_terminology_consistency(self, answer: str, sources: List[Dict]) -> float:
        """Prüft Konsistenz der juristischen Terminologie"""

        answer_lower = answer.lower()
        source_texts = " ".join([s.get("content", "") for s in sources]).lower()

        consistency_scores = []

        for domain, terms in self.key_legal_terms.items():
            answer_has_domain = any(term in answer_lower for term in terms)
            source_has_domain = any(term in source_texts for term in terms)

            if answer_has_domain == source_has_domain:
                consistency_scores.append(1.0)
            elif answer_has_domain and not source_has_domain:
                consistency_scores.append(0.3)  # Antwort führt neue Begriffe ein
            else:
                consistency_scores.append(0.7)  # Antwort lässt relevante Begriffe weg

        return float(np.mean(consistency_scores)) if consistency_scores else 1.0

    def _check_legal_structure(self, answer: str) -> float:
        """Prüft juristische Argumentationsstruktur"""

        structure_indicators = {
            "definition": ["bedeutet", "ist", "versteht man", "definiert"],
            "requirements": ["voraussetzung", "erforderlich", "notwendig", "bedarf"],
            "legal_consequence": ["folgt", "ergibt sich", "führt zu", "bewirkt"],
            "exception": ["jedoch", "aber", "ausnahme", "es sei denn"],
            "conclusion": ["daher", "somit", "folglich", "mithin"],
        }

        answer_lower = answer.lower()
        structure_score = 0.0

        for structure_type, indicators in structure_indicators.items():
            has_structure = any(indicator in answer_lower for indicator in indicators)
            if has_structure:
                structure_score += 0.2  # Jede Struktur-Komponente gibt 20%

        return min(structure_score, 1.0)


class ConfidenceCalculator:
    """Berechnet Confidence-Scores für Antworten"""

    def __init__(self):
        self.fact_checker = LegalFactChecker()

    def calculate_confidence(
        self,
        answer: str,
        sources: List[Dict],
        retrieval_scores: List[float],
        query: str,
    ) -> AnswerQualityMetrics:
        """Berechnet umfassende Confidence-Metriken"""

        # 1. Source Reliability basierend auf Retrieval-Scores
        source_reliability = self._calculate_source_reliability(retrieval_scores)

        # 2. Legal Accuracy durch Fact-Checking
        fact_check_result = self.fact_checker.check_legal_consistency(answer, sources)
        legal_accuracy = fact_check_result["consistency_score"]

        # 3. Completeness basierend auf Query-Coverage
        completeness_score = self._calculate_completeness(answer, query, sources)

        # 4. Clarity basierend auf Struktur und Verständlichkeit
        clarity_score = self._calculate_clarity(answer)

        # 5. Citation Quality
        citation_quality = self._calculate_citation_quality(answer, sources)

        # 6. Factual Consistency zwischen Antwort und Quellen
        factual_consistency = self._calculate_factual_consistency(answer, sources)

        # Gesamt-Confidence berechnen
        confidence_score = (
            source_reliability * 0.2
            + legal_accuracy * 0.25
            + completeness_score * 0.2
            + clarity_score * 0.15
            + citation_quality * 0.1
            + factual_consistency * 0.1
        )

        # Confidence-Level bestimmen
        confidence_level = self._determine_confidence_level(confidence_score)

        return AnswerQualityMetrics(
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            source_reliability=source_reliability,
            legal_accuracy=legal_accuracy,
            completeness_score=completeness_score,
            clarity_score=clarity_score,
            citation_quality=citation_quality,
            factual_consistency=factual_consistency,
        )

    def _calculate_source_reliability(self, retrieval_scores: List[float]) -> float:
        """Berechnet Zuverlässigkeit der Quellen"""

        if not retrieval_scores:
            return 0.0

        # Hohe Scores = hohe Zuverlässigkeit
        avg_score = np.mean(retrieval_scores)

        # Score-Verteilung berücksichtigen
        score_std = float(np.std(retrieval_scores))
        consistency_bonus = max(
            0.0, (0.2 - score_std) * 2
        )  # Bonus für konsistente Scores

        return min(float(avg_score) + consistency_bonus, 1.0)

    def _calculate_completeness(
        self, answer: str, query: str, sources: List[Dict]
    ) -> float:
        """Berechnet Vollständigkeit der Antwort"""

        query_terms = set(query.lower().split())
        answer_terms = set(answer.lower().split())

        # Basis-Coverage: Wie viele Query-Terms sind in der Antwort?
        basic_coverage = len(query_terms.intersection(answer_terms)) / len(query_terms)

        # Erweiterte Coverage: Nutzt die Antwort zusätzliche relevante Informationen aus Quellen?
        source_terms = set()
        for source in sources:
            source_terms.update(source.get("content", "").lower().split())

        relevant_source_terms = source_terms - query_terms - answer_terms
        extended_coverage = min(len(relevant_source_terms) / 50, 0.3)  # Max 30% Bonus

        # Juristische Vollständigkeit: Werden Voraussetzungen, Rechtsfolgen etc. behandelt?
        legal_completeness = self._assess_legal_completeness(answer)

        return basic_coverage * 0.5 + extended_coverage + legal_completeness * 0.2

    def _assess_legal_completeness(self, answer: str) -> float:
        """Bewertet juristische Vollständigkeit"""

        legal_components = {
            "definition": ["bedeutet", "ist", "versteht man unter"],
            "requirements": ["voraussetzung", "erforderlich", "muss"],
            "legal_basis": ["§", "art.", "gesetz", "norm"],
            "consequences": ["folge", "bewirkt", "führt zu"],
            "exceptions": ["ausnahme", "jedoch", "es sei denn"],
        }

        answer_lower = answer.lower()
        components_found = 0

        for component, indicators in legal_components.items():
            if any(indicator in answer_lower for indicator in indicators):
                components_found += 1

        return components_found / len(legal_components)

    def _calculate_clarity(self, answer: str) -> float:
        """Berechnet Klarheit und Verständlichkeit"""

        # Satzlänge (optimal: 15-25 Wörter pro Satz)
        sentences = [s.strip() for s in answer.split(".") if s.strip()]
        if not sentences:
            return 0.0

        avg_sentence_length = float(np.mean([len(s.split()) for s in sentences]))
        sentence_score = max(0.0, 1 - abs(avg_sentence_length - 20) / 20)

        # Struktur (Absätze, Aufzählungen)
        structure_score = min(answer.count("\n") / 3, 1.0)

        # Juristische Klarheit (keine Widersprüche)
        contradiction_penalty = self._detect_contradictions(answer)

        clarity = (
            sentence_score * 0.4
            + structure_score * 0.3
            + (1 - contradiction_penalty) * 0.3
        )

        return max(0, min(clarity, 1.0))

    def _detect_contradictions(self, answer: str) -> float:
        """Erkennt Widersprüche in der Antwort"""

        contradiction_patterns = [
            (r"ist\s+erforderlich", r"ist\s+nicht\s+erforderlich"),
            (r"ist\s+zulässig", r"ist\s+unzulässig"),
            (r"besteht\s+ein\s+anspruch", r"besteht\s+kein\s+anspruch"),
            (r"ist\s+möglich", r"ist\s+nicht\s+möglich"),
        ]

        answer_lower = answer.lower()
        contradictions = 0

        for positive_pattern, negative_pattern in contradiction_patterns:
            if re.search(positive_pattern, answer_lower) and re.search(
                negative_pattern, answer_lower
            ):
                contradictions += 1

        return min(contradictions * 0.2, 1.0)  # Jeder Widerspruch = 20% Penalty

    def _calculate_citation_quality(self, answer: str, sources: List[Dict]) -> float:
        """Bewertet Qualität der Zitationen"""

        # Gesetzesreferenzen in der Antwort
        legal_refs = len(re.findall(r"§\s*\d+|Art\.\s*\d+", answer))

        # Quellenverweise (falls implementiert)
        source_refs = answer.count("[Quelle") + answer.count("(vgl.")

        # Bonuspunkte für präzise Referenzen
        precise_refs = len(re.findall(r"§\s*\d+\s+Abs\.\s*\d+", answer))

        citation_score = (
            min(legal_refs / 3, 0.4)  # Bis zu 40% für Gesetzesreferenzen
            + min(source_refs / len(sources), 0.3)  # Bis zu 30% für Quellenverweise
            + min(precise_refs / 2, 0.3)  # Bis zu 30% für präzise Referenzen
        )

        return min(citation_score, 1.0)

    def _calculate_factual_consistency(self, answer: str, sources: List[Dict]) -> float:
        """Berechnet faktische Konsistenz zwischen Antwort und Quellen"""

        # Extrahiere Key-Facts aus Antwort
        answer_facts = self._extract_key_facts(answer)

        # Extrahiere Key-Facts aus Quellen
        source_facts = []
        for source in sources:
            source_facts.extend(self._extract_key_facts(source.get("content", "")))

        if not answer_facts:
            return 1.0  # Keine Facts zu überprüfen

        # Prüfe Konsistenz
        consistent_facts = 0
        for answer_fact in answer_facts:
            for source_fact in source_facts:
                if self._facts_are_consistent(answer_fact, source_fact):
                    consistent_facts += 1
                    break

        return consistent_facts / len(answer_facts)

    def _extract_key_facts(self, text: str) -> List[str]:
        """Extrahiert Schlüssel-Facts aus Text"""

        # Einfache Heuristik: Sätze mit juristischen Schlüsselwörtern
        key_indicators = [
            "ist erforderlich",
            "ist zulässig",
            "besteht ein anspruch",
            "ist verpflichtet",
            "hat das recht",
            "darf nicht",
            "muss",
        ]

        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
        key_facts = []

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in key_indicators):
                key_facts.append(sentence)

        return key_facts

    def _facts_are_consistent(self, fact1: str, fact2: str) -> bool:
        """Prüft ob zwei Facts konsistent sind"""

        # Einfache Konsistenz-Prüfung basierend auf gemeinsamen Begriffen
        fact1_words = set(fact1.lower().split())
        fact2_words = set(fact2.lower().split())

        common_words = fact1_words.intersection(fact2_words)
        union_words = fact1_words.union(fact2_words)

        # Jaccard-Ähnlichkeit
        similarity = len(common_words) / len(union_words) if union_words else 0

        return similarity > 0.3  # 30% Ähnlichkeit = konsistent

    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Bestimmt Confidence-Level basierend auf Score"""

        if confidence_score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.75:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


class StructuredAnswerGenerator:
    """Generiert strukturierte Antworten mit Templates"""

    def __init__(self):
        self.confidence_calculator = ConfidenceCalculator()

        self.answer_templates = {
            "legal_analysis": self._create_legal_analysis_template(),
            "quick_answer": self._create_quick_answer_template(),
            "comprehensive": self._create_comprehensive_template(),
            "case_analysis": self._create_case_analysis_template(),
        }

    def generate_structured_answer(
        self,
        raw_answer: str,
        sources: List[Dict],
        retrieval_scores: List[float],
        query: str,
        template_type: str = "legal_analysis",
    ) -> Dict:
        """Generiert strukturierte Antwort mit Confidence-Bewertung"""

        # Confidence-Metriken berechnen
        quality_metrics = self.confidence_calculator.calculate_confidence(
            raw_answer, sources, retrieval_scores, query
        )

        # Template auswählen und anwenden
        template = self.answer_templates.get(
            template_type, self.answer_templates["legal_analysis"]
        )
        structured_answer = self._apply_template(
            template, raw_answer, sources, quality_metrics
        )

        # Zusätzliche Metadaten
        result = {
            "structured_answer": structured_answer,
            "quality_metrics": quality_metrics,
            "confidence_summary": self._create_confidence_summary(quality_metrics),
            "improvement_suggestions": self._generate_improvement_suggestions(
                quality_metrics
            ),
            "template_used": template_type,
            "timestamp": time.time(),
        }

        return result

    def _create_legal_analysis_template(self) -> Dict:
        """Template für juristische Analyse"""

        return {
            "structure": [
                "kurze_antwort",
                "rechtliche_grundlagen",
                "voraussetzungen",
                "rechtsfolgen",
                "besonderheiten",
                "quellen",
            ],
            "confidence_indicator": True,
            "source_integration": True,
        }

    def _create_quick_answer_template(self) -> Dict:
        """Template für schnelle Antworten"""

        return {
            "structure": ["direkte_antwort", "wichtigste_norm", "kernpunkt"],
            "confidence_indicator": True,
            "source_integration": False,
        }

    def _create_comprehensive_template(self) -> Dict:
        """Template für umfassende Antworten"""

        return {
            "structure": [
                "zusammenfassung",
                "rechtliche_grundlagen",
                "detaillierte_analyse",
                "voraussetzungen",
                "rechtsfolgen",
                "ausnahmen",
                "praxishinweise",
                "rechtsprechung",
                "literatur",
                "quellen",
            ],
            "confidence_indicator": True,
            "source_integration": True,
        }

    def _create_case_analysis_template(self) -> Dict:
        """Template für Fallanalyse"""

        return {
            "structure": [
                "sachverhalt",
                "rechtliche_probleme",
                "anspruchsgrundlagen",
                "pruefungsschema",
                "subsumtion",
                "ergebnis",
            ],
            "confidence_indicator": True,
            "source_integration": True,
        }

    def _apply_template(
        self,
        template: Dict,
        raw_answer: str,
        sources: List[Dict],
        quality_metrics: AnswerQualityMetrics,
    ) -> str:
        """Wendet Template auf Roh-Antwort an"""

        structured_parts = []

        # Confidence-Indikator
        if template.get("confidence_indicator"):
            confidence_indicator = self._format_confidence_indicator(quality_metrics)
            structured_parts.append(confidence_indicator)

        # Template-Struktur durchgehen
        for section in template["structure"]:
            section_content = self._extract_section_content(
                raw_answer, section, sources
            )
            if section_content:
                section_title = self._format_section_title(section)
                structured_parts.append(f"**{section_title}**\n{section_content}")

        return "\n\n".join(structured_parts)

    def _format_confidence_indicator(
        self, quality_metrics: AnswerQualityMetrics
    ) -> str:
        """Formatiert Confidence-Indikator"""

        confidence_emoji = {
            ConfidenceLevel.VERY_HIGH: "🟢",
            ConfidenceLevel.HIGH: "🔵",
            ConfidenceLevel.MEDIUM: "🟡",
            ConfidenceLevel.LOW: "🟠",
            ConfidenceLevel.VERY_LOW: "🔴",
        }

        emoji = confidence_emoji.get(quality_metrics.confidence_level, "⚪")

        return (
            f"{emoji} **Vertrauenswürdigkeit: {quality_metrics.confidence_level.value.title()}** "
            f"({quality_metrics.confidence_score:.1%})"
        )

    def _extract_section_content(
        self, raw_answer: str, section: str, sources: List[Dict]
    ) -> str:
        """Extrahiert Content für spezifische Sektion"""

        # Vereinfachte Sektion-Extraktion basierend auf Keywords
        section_keywords = {
            "kurze_antwort": ["antwort", "zusammenfassung"],
            "rechtliche_grundlagen": ["§", "art.", "gesetz", "norm"],
            "voraussetzungen": ["voraussetzung", "erforderlich", "bedingung"],
            "rechtsfolgen": ["folge", "wirkung", "konsequenz"],
            "besonderheiten": ["jedoch", "besonders", "ausnahme"],
            "quellen": ["quelle", "literatur", "rechtsprechung"],
        }

        if section == "quellen":
            return self._format_sources(sources)

        keywords = section_keywords.get(section, [])
        if not keywords:
            return raw_answer  # Fallback: gesamte Antwort

        # Finde relevante Sätze
        sentences = raw_answer.split(".")
        relevant_sentences = []

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                relevant_sentences.append(sentence.strip())

        return ". ".join(relevant_sentences) + "." if relevant_sentences else ""

    def _format_section_title(self, section: str) -> str:
        """Formatiert Sektions-Titel"""

        title_map = {
            "kurze_antwort": "Kurze Antwort",
            "rechtliche_grundlagen": "Rechtliche Grundlagen",
            "voraussetzungen": "Voraussetzungen",
            "rechtsfolgen": "Rechtsfolgen",
            "besonderheiten": "Besonderheiten",
            "quellen": "Quellen",
            "detaillierte_analyse": "Detaillierte Analyse",
            "praxishinweise": "Praxishinweise",
        }

        return title_map.get(section, section.replace("_", " ").title())

    def _format_sources(self, sources: List[Dict]) -> str:
        """Formatiert Quellen-Sektion"""

        if not sources:
            return "Keine Quellen verfügbar."

        formatted_sources = []
        for i, source in enumerate(sources[:3], 1):  # Max 3 Quellen
            metadata = source.get("metadata", {})
            source_name = metadata.get("source", "Unbekannte Quelle")
            page = metadata.get("page", "N/A")

            formatted_sources.append(f"{i}. {source_name}, S. {page}")

        return "\n".join(formatted_sources)

    def _create_confidence_summary(self, quality_metrics: AnswerQualityMetrics) -> Dict:
        """Erstellt Confidence-Zusammenfassung"""

        return {
            "overall_confidence": f"{quality_metrics.confidence_score:.1%}",
            "level": quality_metrics.confidence_level.value,
            "key_strengths": self._identify_strengths(quality_metrics),
            "key_weaknesses": self._identify_weaknesses(quality_metrics),
        }

    def _identify_strengths(self, metrics: AnswerQualityMetrics) -> List[str]:
        """Identifiziert Stärken der Antwort"""

        strengths = []

        if metrics.source_reliability > 0.8:
            strengths.append("Hohe Quellenqualität")
        if metrics.legal_accuracy > 0.8:
            strengths.append("Hohe juristische Genauigkeit")
        if metrics.clarity_score > 0.8:
            strengths.append("Klare und verständliche Darstellung")
        if metrics.citation_quality > 0.7:
            strengths.append("Gute Zitation von Rechtsnormen")

        return strengths

    def _identify_weaknesses(self, metrics: AnswerQualityMetrics) -> List[str]:
        """Identifiziert Schwächen der Antwort"""

        weaknesses = []

        if metrics.source_reliability < 0.5:
            weaknesses.append("Niedrige Quellenqualität")
        if metrics.legal_accuracy < 0.5:
            weaknesses.append("Mögliche juristische Ungenauigkeiten")
        if metrics.completeness_score < 0.5:
            weaknesses.append("Unvollständige Behandlung des Themas")
        if metrics.clarity_score < 0.5:
            weaknesses.append("Unklare oder schwer verständliche Formulierung")

        return weaknesses

    def _generate_improvement_suggestions(
        self, metrics: AnswerQualityMetrics
    ) -> List[str]:
        """Generiert Verbesserungsvorschläge"""

        suggestions = []

        if metrics.legal_accuracy < 0.7:
            suggestions.append("Überprüfung der juristischen Genauigkeit empfohlen")
        if metrics.completeness_score < 0.6:
            suggestions.append("Erwägung zusätzlicher Aspekte oder Voraussetzungen")
        if metrics.citation_quality < 0.6:
            suggestions.append("Verbesserung der Gesetzeszitierung")
        if metrics.clarity_score < 0.6:
            suggestions.append("Strukturierung und Vereinfachung der Sprache")

        return suggestions
