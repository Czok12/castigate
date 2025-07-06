#!/usr/bin/env python3
"""
Juristische Quellenvalidator und Zitatsgenerator
Validiert Zitate und generiert korrekte Quellenangaben für verschiedene Zitationsstile
"""

import re
from typing import Dict


class JuristischeQuellenValidator:
    """Validiert und korrigiert juristische Quellenangaben"""

    def __init__(self):
        self.german_law_codes = {
            "BGB": "Bürgerliches Gesetzbuch",
            "StGB": "Strafgesetzbuch",
            "GG": "Grundgesetz",
            "ZPO": "Zivilprozessordnung",
            "StPO": "Strafprozessordnung",
            "HGB": "Handelsgesetzbuch",
            "AO": "Abgabenordnung",
            "SGB": "Sozialgesetzbuch",
            "VwGO": "Verwaltungsgerichtsordnung",
            "BVerfGG": "Bundesverfassungsgerichtsgesetz",
        }

        self.court_abbreviations = {
            "BGH": "Bundesgerichtshof",
            "BVerfG": "Bundesverfassungsgericht",
            "BAG": "Bundesarbeitsgericht",
            "BFH": "Bundesfinanzhof",
            "BSG": "Bundessozialgericht",
            "BVerwG": "Bundesverwaltungsgericht",
            "OLG": "Oberlandesgericht",
            "LG": "Landgericht",
            "AG": "Amtsgericht",
        }

    def validate_paragraph_reference(self, ref: str) -> Dict:
        """Validiert Paragraphen-Referenzen"""
        # Pattern für § 123 Abs. 2 S. 1 BGB
        pattern = r"§\s*(\d+[a-z]?)\s*(?:Abs\.\s*(\d+))?\s*(?:S\.\s*(\d+))?\s*([A-Z]+)"
        match = re.match(pattern, ref.strip(), re.IGNORECASE)

        if not match:
            return {
                "valid": False,
                "error": "Ungültiges Paragraphen-Format",
                "suggestion": "Format: § 123 Abs. 2 S. 1 BGB",
            }

        paragraph, absatz, satz, gesetzbuch = match.groups()

        # Validiere Gesetzbuch
        if gesetzbuch.upper() not in self.german_law_codes:
            return {
                "valid": False,
                "error": f"Unbekanntes Gesetzbuch: {gesetzbuch}",
                "suggestion": f'Mögliche Gesetzbücher: {", ".join(self.german_law_codes.keys())}',
            }

        return {
            "valid": True,
            "formatted": self._format_paragraph_reference(
                paragraph, absatz, satz, gesetzbuch
            ),
            "components": {
                "paragraph": paragraph,
                "absatz": absatz,
                "satz": satz,
                "gesetzbuch": gesetzbuch.upper(),
                "full_name": self.german_law_codes[gesetzbuch.upper()],
            },
        }

    def _format_paragraph_reference(
        self, paragraph: str, absatz: str, satz: str, gesetzbuch: str
    ) -> str:
        """Formatiert Paragraphen-Referenz korrekt"""
        ref = f"§ {paragraph}"
        if absatz:
            ref += f" Abs. {absatz}"
        if satz:
            ref += f" S. {satz}"
        ref += f" {gesetzbuch.upper()}"
        return ref

    def validate_court_decision(self, ref: str) -> Dict:
        """Validiert Gerichtsentscheidungs-Referenzen"""
        # Pattern für BGH, Urt. v. 12.03.2020 - I ZR 123/19
        pattern = r"([A-Z]+)[,\s]+(?:Urt\.|Beschl\.)\s*v\.\s*(\d{1,2}\.\d{1,2}\.\d{4})\s*-\s*(.+)"
        match = re.match(pattern, ref.strip(), re.IGNORECASE)

        if not match:
            return {
                "valid": False,
                "error": "Ungültiges Gerichtsentscheidung-Format",
                "suggestion": "Format: BGH, Urt. v. 12.03.2020 - I ZR 123/19",
            }

        court, date, aktenzeichen = match.groups()

        if court.upper() not in self.court_abbreviations:
            return {
                "valid": False,
                "error": f"Unbekanntes Gericht: {court}",
                "suggestion": f'Mögliche Gerichte: {", ".join(self.court_abbreviations.keys())}',
            }

        return {
            "valid": True,
            "formatted": f"{court.upper()}, Urt. v. {date} - {aktenzeichen}",
            "components": {
                "court": court.upper(),
                "court_full": self.court_abbreviations[court.upper()],
                "date": date,
                "aktenzeichen": aktenzeichen,
            },
        }


class ZitationsstilGenerator:
    """Generiert Zitate in verschiedenen wissenschaftlichen Stilen"""

    def __init__(self):
        self.validator = JuristischeQuellenValidator()

    def generate_book_citation(
        self,
        author: str,
        title: str,
        edition: str = "",
        year: str = "",
        publisher: str = "",
        page: str = "",
        style: str = "Juristische",
    ) -> str:
        """Generiert Buchzitation"""

        if style == "Juristische":
            # Juristische Zitierweise: Autor, Titel, Auflage Jahr, Seite
            citation = author
            if title:
                citation += f", {title}"
            if edition and year:
                citation += f", {edition} {year}"
            elif year:
                citation += f", {year}"
            if page:
                citation += f", S. {page}"
            return citation

        elif style == "APA":
            # APA: Autor (Jahr). Titel (Auflage). Verlag.
            citation = f"{author} ({year}). {title}"
            if edition:
                citation += f" ({edition})"
            if publisher:
                citation += f". {publisher}"
            citation += "."
            return citation

        elif style == "Harvard":
            # Harvard: Autor Jahr, Seite
            citation = f"{author} {year}"
            if page:
                citation += f", S. {page}"
            return citation

        elif style == "Chicago":
            # Chicago: Autor. Titel. Auflage. Ort: Verlag, Jahr.
            citation = f"{author}. {title}."
            if edition:
                citation += f" {edition}."
            if publisher and year:
                citation += f" {publisher}, {year}."
            return citation

        else:
            return f"{author}, {title}, {year}, S. {page}"

    def generate_article_citation(
        self,
        author: str,
        title: str,
        journal: str,
        year: str = "",
        volume: str = "",
        page: str = "",
        style: str = "Juristische",
    ) -> str:
        """Generiert Artikel-Zitation"""

        if style == "Juristische":
            citation = f"{author}, {title}, {journal}"
            if year:
                citation += f" {year}"
            if page:
                citation += f", S. {page}"
            return citation

        elif style == "APA":
            citation = f"{author} ({year}). {title}. {journal}"
            if volume:
                citation += f", {volume}"
            if page:
                citation += f", {page}"
            return citation + "."

        else:
            return f"{author}, {title}, {journal} ({year}), S. {page}"

    def generate_legal_citation(self, ref_type: str, reference: str) -> str:
        """Generiert juristische Zitation"""

        if ref_type == "paragraph":
            validation = self.validator.validate_paragraph_reference(reference)
            if validation["valid"]:
                return validation["formatted"]
            else:
                return reference

        elif ref_type == "court_decision":
            validation = self.validator.validate_court_decision(reference)
            if validation["valid"]:
                return validation["formatted"]
            else:
                return reference

        return reference


def main():
    """Beispiel-Nutzung des Validators und Generators"""
    validator = JuristischeQuellenValidator()
    generator = ZitationsstilGenerator()

    print("=== Juristische Quellenvalidierung ===\n")

    # Test Paragraphen-Referenzen
    test_paragraphs = [
        "§ 433 BGB",
        "§ 433 Abs. 1 BGB",
        "§ 433 Abs. 1 S. 1 BGB",
        "§ 123 XYZ",  # Ungültig
        "433 BGB",  # Ungültig
    ]

    print("Paragraphen-Validierung:")
    for ref in test_paragraphs:
        result = validator.validate_paragraph_reference(ref)
        print(f"  {ref}: {'✓' if result['valid'] else '✗'}")
        if result["valid"]:
            print(f"    → {result['formatted']}")
        else:
            print(f"    → Fehler: {result['error']}")
            print(f"    → Vorschlag: {result['suggestion']}")
        print()

    # Test Gerichtsentscheidungen
    test_decisions = [
        "BGH, Urt. v. 12.03.2020 - I ZR 123/19",
        "BVerfG, Beschl. v. 01.01.2021 - 1 BvR 456/20",
        "XYZ, Urt. v. 12.03.2020 - I ZR 123/19",  # Ungültig
    ]

    print("Gerichtsentscheidungs-Validierung:")
    for ref in test_decisions:
        result = validator.validate_court_decision(ref)
        print(f"  {ref}: {'✓' if result['valid'] else '✗'}")
        if result["valid"]:
            print(f"    → {result['formatted']}")
        else:
            print(f"    → Fehler: {result['error']}")
        print()

    # Test Zitationsgenerierung
    print("=== Zitationsgenerierung ===\n")

    styles = ["Juristische", "APA", "Harvard", "Chicago"]

    print("Buchzitation in verschiedenen Stilen:")
    for style in styles:
        citation = generator.generate_book_citation(
            author="Medicus/Petersen",
            title="Grundwissen zum Bürgerlichen Recht",
            edition="11. Aufl.",
            year="2019",
            publisher="C.H.Beck",
            page="123",
            style=style,
        )
        print(f"  {style}: {citation}")

    print("\nArtikel-Zitation:")
    for style in styles:
        citation = generator.generate_article_citation(
            author="Mustermann",
            title="Die Bedeutung der GoA im modernen Recht",
            journal="JuS",
            year="2023",
            volume="3",
            page="234-240",
            style=style,
        )
        print(f"  {style}: {citation}")


if __name__ == "__main__":
    main()
