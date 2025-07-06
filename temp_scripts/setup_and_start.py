#!/usr/bin/env python3
"""
🏛️ Juristische Wissensdatenbank - Setup und Start
Automatisches Setup und Start des kompletten Systems
"""

import json
import os
import subprocess
import sys
from pathlib import Path


class JuraSystemSetup:
    """Setup-Manager für das juristische System"""

    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.requirements_file = self.base_dir / "requirements_erweitert.txt"

    def check_system_requirements(self):
        """Prüft System-Voraussetzungen"""
        print("🔍 Prüfe System-Voraussetzungen...")

        # Python-Version prüfen
        python_version = sys.version_info
        if python_version < (3, 8):
            print("❌ Python 3.8+ erforderlich")
            return False
        print(f"✅ Python {python_version.major}.{python_version.minor}")

        # Ollama prüfen
        try:
            result = subprocess.run(
                ["ollama", "--version"], capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                print("✅ Ollama gefunden")
            else:
                print(
                    "⚠️  Ollama nicht gefunden - bitte installieren: https://ollama.ai"
                )
        except FileNotFoundError:
            print("⚠️  Ollama nicht gefunden - bitte installieren: https://ollama.ai")

        return True

    def install_dependencies(self):
        """Installiert Python-Abhängigkeiten"""
        print("📦 Installiere Abhängigkeiten...")

        if not self.requirements_file.exists():
            print(f"❌ {self.requirements_file} nicht gefunden")
            return False

        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(self.requirements_file),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            print("✅ Abhängigkeiten erfolgreich installiert")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Fehler bei Installation: {e}")
            return False

    def setup_ollama_model(self):
        """Lädt das Ollama-Modell"""
        print("🤖 Setup Ollama-Modell...")

        try:
            # Prüfe ob Modell bereits vorhanden
            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, check=False
            )
            if "llama3.2" in result.stdout:
                print("✅ Llama3.2 bereits vorhanden")
                return True

            # Lade Modell
            print("⬇️  Lade Llama3.2... (das kann einige Minuten dauern)")
            result = subprocess.run(["ollama", "pull", "llama3.2"], check=True)
            print("✅ Llama3.2 erfolgreich geladen")
            return True

        except (FileNotFoundError, subprocess.CalledProcessError):
            print("⚠️  Ollama-Setup fehlgeschlagen")
            return False

    def check_databases(self):
        """Prüft Datenbank-Status"""
        print("🗃️  Prüfe Datenbanken...")

        # FAISS-Datenbank
        faiss_path = self.base_dir / "faiss_db"
        if faiss_path.exists():
            print("✅ FAISS-Datenbank gefunden")
        else:
            print("⚠️  FAISS-Datenbank nicht gefunden")
            print("   → Führen Sie 'python ingest.py' aus")

        # Bibliotheks-Datenbank
        bib_path = self.base_dir / "juristische_bibliothek.db"
        if bib_path.exists():
            print("✅ Bibliotheks-Datenbank gefunden")
        else:
            print("ℹ️  Bibliotheks-Datenbank wird beim ersten Start erstellt")

        # Metadaten
        meta_path = self.base_dir / "chunk_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                chunks = metadata.get("total_chunks", 0)
                print(f"📊 {chunks:,} Text-Chunks verfügbar")

        return True

    def create_startup_script(self):
        """Erstellt Startup-Skript"""
        startup_content = """#!/bin/bash
# Juristische Wissensdatenbank - Startup Script

echo "🏛️  Starte Juristische Wissensdatenbank..."

# Prüfe virtuelle Umgebung
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtuelle Umgebung aktiv: $VIRTUAL_ENV"
else
    echo "⚠️  Keine virtuelle Umgebung erkannt"
fi

# Prüfe Ollama
if command -v ollama >/dev/null 2>&1; then
    echo "✅ Ollama verfügbar"
    # Starte Ollama-Server falls nicht läuft
    if ! pgrep -f "ollama serve" > /dev/null; then
        echo "🚀 Starte Ollama-Server..."
        ollama serve &
        sleep 2
    fi
else
    echo "❌ Ollama nicht gefunden"
    exit 1
fi

# Starte Hauptanwendung
echo "🚀 Starte Hauptanwendung..."
streamlit run jura_hauptapp.py --server.port 8501 --server.address localhost
"""

        startup_path = self.base_dir / "start_jura_system.sh"
        with open(startup_path, "w", encoding="utf-8") as f:
            f.write(startup_content)

        # Mache ausführbar
        os.chmod(startup_path, 0o755)
        print(f"✅ Startup-Skript erstellt: {startup_path}")

        return str(startup_path)

    def run_setup(self):
        """Führt komplettes Setup durch"""
        print("=" * 60)
        print("🏛️  JURISTISCHE WISSENSDATENBANK - SYSTEM SETUP")
        print("=" * 60)

        # System-Checks
        if not self.check_system_requirements():
            return False

        print("\n" + "-" * 40)

        # Abhängigkeiten
        if not self.install_dependencies():
            print("⚠️  Setup kann trotz Fehlern fortgesetzt werden")

        print("\n" + "-" * 40)

        # Ollama-Setup
        self.setup_ollama_model()

        print("\n" + "-" * 40)

        # Datenbanken prüfen
        self.check_databases()

        print("\n" + "-" * 40)

        # Startup-Skript
        startup_script = self.create_startup_script()

        print("\n" + "=" * 60)
        print("✅ SETUP ABGESCHLOSSEN!")
        print("=" * 60)

        print(
            f"""
🚀 SYSTEM STARTEN:

   Methode 1 - Startup-Skript:
   chmod +x {startup_script}
   ./{startup_script}

   Methode 2 - Direkt:
   streamlit run jura_hauptapp.py

   Methode 3 - Mit spezifischem Port:
   streamlit run jura_hauptapp.py --server.port 8502

📚 ERSTE SCHRITTE:

   1. Legen Sie PDF-Lehrbücher in ./originale/
   2. Führen Sie aus: python ingest.py
   3. Starten Sie das System (siehe oben)
   4. Öffnen Sie http://localhost:8501

🆘 SUPPORT:

   - README.md für detaillierte Anleitung
   - System-Setup Tab in der Anwendung
   - GitHub Issues für Probleme

⚠️  WICHTIG:

   - Stellen Sie sicher, dass Ollama läuft
   - Erste Ingest kann einige Minuten dauern
   - Prüfen Sie alle Antworten auf Korrektheit
        """
        )

        return True


def main():
    """Hauptfunktion"""
    if len(sys.argv) > 1 and sys.argv[1] == "--start-only":
        # Nur starten, kein Setup
        print("🚀 Starte Juristische Wissensdatenbank...")
        try:
            subprocess.run(
                [sys.executable, "-m", "streamlit", "run", "jura_hauptapp.py"],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ Fehler beim Start: {e}")
        return

    # Vollständiges Setup
    setup = JuraSystemSetup()
    success = setup.run_setup()

    if success:
        # Frage nach direktem Start
        while True:
            start_now = input("\n🚀 System jetzt starten? (j/n): ").lower().strip()
            if start_now in ["j", "ja", "y", "yes"]:
                print("Starte System...")
                try:
                    subprocess.run(
                        [sys.executable, "-m", "streamlit", "run", "jura_hauptapp.py"],
                        check=False,
                    )
                except KeyboardInterrupt:
                    print("\n👋 System beendet")
                break
            elif start_now in ["n", "nein", "no"]:
                print("👋 Setup abgeschlossen. Starten Sie später mit:")
                print("   streamlit run jura_hauptapp.py")
                break
            else:
                print("Bitte antworten Sie mit 'j' oder 'n'")


if __name__ == "__main__":
    main()
