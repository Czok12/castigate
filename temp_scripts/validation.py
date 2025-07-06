# Validierungsmodul für juristische RAG-Anwendung

import re
from typing import Any, Dict


class Juristische_Validierung:
    """Validiert juristische Inhalte für bessere Qualität"""

    def __init__(self):
        # Häufige juristische Begriffe und Patterns
        self.paragraph_pattern = (
            r"§\s*\d+(?:\s*Abs\.\s*\d+)?(?:\s*S\.\s*\d+)?\s*[A-Z]{2,4}"
        )
        self.artikel_pattern = r"Art\.\s*\d+(?:\s*Abs\.\s*\d+)?\s*[A-Z]+"

    def validate_legal_citation(self, text: str) -> Dict[str, Any]:
        """Überprüft, ob juristische Zitate korrekt formatiert sind"""
        paragraphs = re.findall(self.paragraph_pattern, text)
        artikel = re.findall(self.artikel_pattern, text)

        return {
            "paragraphs_found": paragraphs,
            "artikel_found": artikel,
            "has_legal_citations": len(paragraphs) > 0 or len(artikel) > 0,
            "citation_count": len(paragraphs) + len(artikel),
        }

    def check_answer_quality(self, answer: str) -> Dict[str, Any]:
        """Bewertet die Qualität einer juristischen Antwort"""
        validation = self.validate_legal_citation(answer)

        # Prüfe auf unzulässige Phrasen
        problematic_phrases = [
            "ich bin mir nicht sicher",
            "möglicherweise",
            "könnte sein",
            "wahrscheinlich",
        ]

        has_uncertainty = any(
            phrase in answer.lower() for phrase in problematic_phrases
        )

        # Prüfe auf Struktur
        has_structure = any(marker in answer for marker in ["1.", "2.", "**", "###"])

        return {
            "legal_citations": validation,
            "contains_uncertainty": has_uncertainty,
            "well_structured": has_structure,
            "answer_length": len(answer),
            "quality_score": self._calculate_quality_score(
                validation, has_uncertainty, has_structure
            ),
        }

    def _calculate_quality_score(
        self, citations: Dict, uncertainty: bool, structure: bool
    ) -> float:
        """Berechnet einen Qualitätsscore (0-1)"""
        score = 0.5  # Basisscore

        if citations["has_legal_citations"]:
            score += 0.3

        if not uncertainty:
            score += 0.1

        if structure:
            score += 0.1

        return min(score, 1.0)


# Verwendungsbeispiel:
# validator = Juristische_Validierung()
# quality = validator.check_answer_quality(question, answer, context)
# if quality['quality_score'] < 0.7:
#     print("⚠️ Antwort könnte überarbeitet werden")
