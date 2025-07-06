#!/usr/bin/env python3
"""
Spezielle Analyse für gescannte PDFs
"""

import os

import fitz
import pytesseract
from PIL import Image, ImageEnhance


def analyze_scanned_pdf(pdf_path: str):
    """Analysiert ein gescanntes PDF und testet verschiedene OCR-Strategien."""
    print("🔍 Analyse für gescanntes PDF")
    print("=" * 50)

    try:
        doc = fitz.open(pdf_path)

        # Teste verschiedene Seiten
        test_pages = [10, 20, 50, 100, 200]  # Verschiedene Seiten testen

        for page_num in test_pages:
            if page_num < len(doc):
                print(f"\n📄 Teste Seite {page_num + 1}:")

                page = doc.load_page(page_num)

                # Prüfe ob Seite Text hat
                pdf_text = page.get_text().strip()
                if pdf_text:
                    print(f"   📝 PDF hat Text ({len(pdf_text)} Zeichen)")
                    first_line = pdf_text.split("\n")[0][:80]
                    print(f"   📝 Erste Zeile: '{first_line}...'")

                    # Umlaute im PDF-Text
                    umlauts = [c for c in pdf_text if c in "äöüÄÖÜß"]
                    if umlauts:
                        print(f"   🔤 Umlaute im PDF: {set(umlauts)}")
                else:
                    print("   📝 Keine direkten Text-Daten")

                # Erstelle hochauflösenden Screenshot
                for zoom in [1.0, 2.0, 3.0]:
                    print(f"   🔍 Teste mit {zoom}x Zoom...")

                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)

                    screenshot_path = f"test_page_{page_num + 1}_zoom_{zoom}.png"
                    pix.save(screenshot_path)

                    # OCR auf Screenshot
                    success = test_advanced_ocr(screenshot_path)

                    # Aufräumen
                    if os.path.exists(screenshot_path):
                        os.remove(screenshot_path)

                    if success:
                        break  # Stoppe bei erfolgreichem OCR

        doc.close()

    except Exception as e:
        print(f"❌ Fehler: {e}")


def test_advanced_ocr(image_path: str) -> bool:
    """Testet erweiterte OCR-Strategien."""
    try:
        image = Image.open(image_path)

        print(f"      📊 Bild: {image.size}, {image.mode}")

        # Verschiedene OCR-Konfigurationen
        configs = [
            ("Standard Deutsch", "deu", "--oem 3 --psm 6"),
            ("Auto-Erkennung", "deu", "--oem 3 --psm 3"),
            ("Einzelner Block", "deu", "--oem 3 --psm 8"),
            ("Deutsch + Englisch", "deu+eng", "--oem 3 --psm 6"),
            ("Ohne Einschränkungen", "deu", ""),
        ]

        for config_name, lang, config in configs:
            try:
                # Verschiedene Bildvorbereitungen
                test_images = [
                    ("Original", image),
                    ("Graustufen", image.convert("L")),
                    (
                        "Kontrast +50%",
                        ImageEnhance.Contrast(image.convert("L")).enhance(1.5),
                    ),
                    (
                        "Kontrast +100%",
                        ImageEnhance.Contrast(image.convert("L")).enhance(2.0),
                    ),
                ]

                for prep_name, prep_image in test_images:
                    text = pytesseract.image_to_string(
                        prep_image, lang=lang, config=config
                    ).strip()

                    if text and len(text) > 20:  # Nur bei substantiellem Text
                        print(
                            f"      ✅ {config_name} ({prep_name}): {len(text)} Zeichen"
                        )

                        # Erste Zeilen zeigen
                        lines = [
                            line.strip() for line in text.split("\n") if line.strip()
                        ][:3]
                        for i, line in enumerate(lines):
                            print(
                                f"         {i+1}: {line[:80]}{'...' if len(line) > 80 else ''}"
                            )

                        # Umlaute prüfen
                        umlauts = [c for c in text if c in "äöüÄÖÜß"]
                        if umlauts:
                            unique_umlauts = set(umlauts)
                            print(
                                f"      🔤 UMLAUTE GEFUNDEN: {', '.join(unique_umlauts)} (total: {len(umlauts)})"
                            )
                        else:
                            print(f"      ❌ Keine Umlaute in {len(text)} Zeichen")

                            # Analyse möglicher Ersetzungen
                            check_common_ocr_errors(text)

                        return True  # Erfolg!

            except Exception as e:
                print(f"      ❌ {config_name} Fehler: {e}")
                continue

        return False  # Kein Erfolg

    except Exception as e:
        print(f"      ❌ Bild-Fehler: {e}")
        return False


def check_common_ocr_errors(text: str):
    """Prüft auf häufige OCR-Fehler bei deutschen Umlauten."""
    errors_found = []

    # Häufige OCR-Fehler
    error_patterns = {
        "ä": ["ae", "a", "â", "à"],
        "ö": ["oe", "o", "ô", "ò"],
        "ü": ["ue", "u", "û", "ù"],
        "ß": ["ss", "B", "b", "8"],
        "Ä": ["Ae", "A", "Â", "À"],
        "Ö": ["Oe", "O", "Ô", "Ò"],
        "Ü": ["Ue", "U", "Û", "Ù"],
    }

    text_lower = text.lower()

    for correct, wrong_list in error_patterns.items():
        for wrong in wrong_list:
            if wrong.lower() in text_lower:
                # Prüfe Kontext - ist es wirklich ein Fehler?
                if wrong.lower() == "ss" and "klasse" in text_lower:
                    continue  # 'ss' in 'Klasse' ist korrekt
                if wrong.lower() in ["a", "o", "u"] and len(wrong) == 1:
                    continue  # Einzelne Vokale sind oft korrekt

                errors_found.append(f"{wrong} → {correct}")

    if errors_found:
        print(f"      ⚠️ Mögliche OCR-Fehler: {', '.join(errors_found[:5])}")

    # Spezielle deutsche Wörter prüfen
    german_words_with_umlauts = [
        ("naemlich", "nämlich"),
        ("ueber", "über"),
        ("fuer", "für"),
        ("waehrend", "während"),
        ("moegliche", "mögliche"),
        ("hoeren", "hören"),
        ("koennen", "können"),
        ("muessen", "müssen"),
    ]

    for wrong, correct in german_words_with_umlauts:
        if wrong in text_lower:
            errors_found.append(f"'{wrong}' → '{correct}'")

    if errors_found:
        print(f"      📝 Deutsche Wort-Fehler: {', '.join(errors_found[:3])}")


def main():
    """Hauptfunktion."""
    print("🔍 Erweiterte Analyse für gescannte PDFs")
    print("=" * 60)

    pdf_path = "dein_lehrbuch.pdf"

    if not os.path.exists(pdf_path):
        print(f"❌ PDF nicht gefunden: {pdf_path}")
        return

    analyze_scanned_pdf(pdf_path)

    print("\n✅ Erweiterte Analyse abgeschlossen!")


if __name__ == "__main__":
    main()
