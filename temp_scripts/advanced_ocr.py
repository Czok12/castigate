"""
Erweiterte OCR-Funktionen für bessere Umlaut-Erkennung
Speziell optimiert für deutsche juristische Texte
"""

import os
from typing import Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter


class AdvancedOCR:
    """Erweiterte OCR-Klasse mit verschiedenen Optimierungsstrategien."""

    def __init__(self):
        self.tesseract_configs = {
            "standard": r"--oem 3 --psm 6",
            "german_optimized": r"--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzäöüÄÖÜß0123456789 .,;:!?()-/§",
            "legal_text": r"--oem 3 --psm 4 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzäöüÄÖÜß0123456789 .,;:!?()-/§°",
            "single_block": r"--oem 3 --psm 8",
            "single_word": r"--oem 3 --psm 7",
        }

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Bildvorverarbeitung für bessere OCR-Qualität."""
        # Zu Graustufen konvertieren
        if image.mode != "L":
            image = image.convert("L")

        # Kontrast erhöhen
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)

        # Schärfe erhöhen
        image = image.filter(ImageFilter.SHARPEN)

        return image

    def preprocess_image_opencv(self, image_path: str) -> Optional[np.ndarray]:
        """Erweiterte Bildvorverarbeitung mit OpenCV."""
        try:
            # Bild laden
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None

            # Rauschen entfernen
            img = cv2.medianBlur(img, 3)

            # Adaptive Schwellwertbildung für besseren Kontrast
            img = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Morphologische Operationen zur Textoptimierung
            kernel = np.ones((1, 1), np.uint8)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

            return img
        except Exception as e:
            print(f"OpenCV Vorverarbeitung fehlgeschlagen: {e}")
            return None

    def extract_text_advanced(self, image_path: str) -> Tuple[str, str]:
        """
        Extrahiert Text mit verschiedenen Strategien.

        Returns:
            Tuple[str, str]: (extracted_text, method_used)
        """
        best_text = ""
        best_method = "none"
        best_confidence = 0

        # Strategie 1: Standard PIL mit verbesserter Konfiguration
        try:
            image = Image.open(image_path)
            processed_image = self.preprocess_image(image)

            for config_name, config in self.tesseract_configs.items():
                try:
                    text = pytesseract.image_to_string(
                        processed_image, lang="deu", config=config
                    ).strip()

                    if text and len(text) > len(best_text):
                        # Einfache Qualitätsbewertung basierend auf deutschen Wörtern
                        german_indicators = [
                            "der",
                            "die",
                            "das",
                            "und",
                            "oder",
                            "mit",
                            "von",
                            "zu",
                            "in",
                            "an",
                            "auf",
                            "für",
                        ]
                        umlaut_count = (
                            text.count("ä")
                            + text.count("ö")
                            + text.count("ü")
                            + text.count("Ä")
                            + text.count("Ö")
                            + text.count("Ü")
                            + text.count("ß")
                        )
                        german_word_count = sum(
                            1 for word in german_indicators if word in text.lower()
                        )

                        confidence = (
                            len(text) + (umlaut_count * 5) + (german_word_count * 3)
                        )

                        if confidence > best_confidence:
                            best_text = text
                            best_method = f"PIL_{config_name}"
                            best_confidence = confidence

                except Exception as e:
                    print(f"Tesseract-Konfiguration {config_name} fehlgeschlagen: {e}")
                    continue

        except Exception as e:
            print(f"PIL-Verarbeitung fehlgeschlagen: {e}")

        # Strategie 2: OpenCV Vorverarbeitung
        try:
            processed_cv = self.preprocess_image_opencv(image_path)
            if processed_cv is not None:
                # Konvertiere zurück zu PIL für Tesseract
                pil_image = Image.fromarray(processed_cv)

                text = pytesseract.image_to_string(
                    pil_image,
                    lang="deu",
                    config=self.tesseract_configs["german_optimized"],
                ).strip()

                if text and len(text) > len(best_text):
                    umlaut_count = (
                        text.count("ä")
                        + text.count("ö")
                        + text.count("ü")
                        + text.count("Ä")
                        + text.count("Ö")
                        + text.count("Ü")
                        + text.count("ß")
                    )
                    confidence = len(text) + (umlaut_count * 5)

                    if confidence > best_confidence:
                        best_text = text
                        best_method = "OpenCV_preprocessed"
                        best_confidence = confidence

        except Exception as e:
            print(f"OpenCV-Verarbeitung fehlgeschlagen: {e}")

        # Strategie 3: Fallback ohne Einschränkungen
        if not best_text:
            try:
                image = Image.open(image_path)
                text = pytesseract.image_to_string(image, lang="deu").strip()
                if text:
                    best_text = text
                    best_method = "fallback_standard"
            except Exception as e:
                print(f"Fallback OCR fehlgeschlagen: {e}")

        return best_text, best_method

    def test_umlaut_recognition(self, test_image_path: str = None) -> None:
        """Test-Funktion für Umlaut-Erkennung."""
        if test_image_path and os.path.exists(test_image_path):
            print(f"🧪 Teste OCR mit: {test_image_path}")
            text, method = self.extract_text_advanced(test_image_path)
            print(f"📝 Erkannter Text ({method}):")
            print(f"'{text}'")

            # Umlaut-Analyse
            umlauts = ["ä", "ö", "ü", "Ä", "Ö", "Ü", "ß"]
            found_umlauts = [u for u in umlauts if u in text]
            if found_umlauts:
                print(f"✅ Gefundene Umlaute: {', '.join(found_umlauts)}")
            else:
                print("❌ Keine Umlaute erkannt")
        else:
            print("❌ Kein Test-Bild verfügbar")

    def diagnose_ocr_setup(self) -> None:
        """Diagnostiziert die OCR-Einrichtung."""
        print("🔍 OCR-Diagnose:")

        # Tesseract-Version
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract Version: {version}")
        except Exception as e:
            print(f"❌ Tesseract-Version konnte nicht ermittelt werden: {e}")

        # Verfügbare Sprachen
        try:
            langs = pytesseract.get_languages(config="")
            if "deu" in langs:
                print("✅ Deutsche Sprachdaten verfügbar")
            else:
                print("❌ Deutsche Sprachdaten nicht gefunden")
                print("   Installation: brew install tesseract-lang")
        except Exception as e:
            print(f"❌ Sprachen konnten nicht ermittelt werden: {e}")

        # Test mit Beispieltext
        print("\n🧪 Teste Umlaut-Erkennung mit Beispielbild...")
        try:
            # Erstelle Test-Bild mit Umlauten
            from PIL import ImageDraw, ImageFont

            test_img = Image.new("RGB", (400, 100), color="white")
            draw = ImageDraw.Draw(test_img)

            test_text = "Übung: Prüfung der Strafbarkeit gemäß §§ 223, 224"

            try:
                # Versuche eine bessere Schrift zu verwenden
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            draw.text((10, 30), test_text, fill="black", font=font)

            # Speichere temporär
            temp_path = "temp_umlaut_test.png"
            test_img.save(temp_path)

            # Teste OCR
            text, method = self.extract_text_advanced(temp_path)
            print(f"📝 Original: {test_text}")
            print(f"📝 Erkannt ({method}): {text}")

            # Bereinige
            if os.path.exists(temp_path):
                os.remove(temp_path)

        except Exception as e:
            print(f"❌ Test-Bild konnte nicht erstellt werden: {e}")


if __name__ == "__main__":
    ocr = AdvancedOCR()
    ocr.diagnose_ocr_setup()
