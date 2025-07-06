#!/usr/bin/env python3
"""
Juristische Wissensdatenbank - Hauptstartskript
Bietet eine einheitliche Oberfläche für alle Funktionen des Systems
"""

import os
import sys
from pathlib import Path

import streamlit as st

# Füge das aktuelle Verzeichnis zum Python-Pfad hinzu
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))


def main():
    st.set_page_config(
        page_title="🏛️ Juristische Wissensdatenbank",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Haupttitel
    st.title("🏛️ Juristische Wissensdatenbank & Zitationsengine")
    st.markdown(
        """
    ---
    **Willkommen zu Ihrem spezialisierten System für juristische Lehrbücher!**
    
    Dieses System bietet zwei Hauptfunktionen:
    1. **📚 Wissensdatenbank**: Durchsuchen Sie Ihre Lehrbücher mit natürlichen Fragen
    2. **📝 Zitationsengine**: Automatische Generierung korrekter Quellenangaben
    """
    )

    # Seitenauswahl
    with st.sidebar:
        st.header("🧭 Navigation")

        page = st.selectbox(
            "Wählen Sie eine Funktion:",
            [
                "🏠 Übersicht",
                "🧠 Erweiterte Wissensdatenbank",
                "📝 Zitationsengine",
                "🔍 Erweiterte Quellensuche",
                "📚 Bibliotheksverwaltung",
                "⚙️ System-Setup",
                "📖 Anleitung",
            ],
        )

        st.markdown("---")

        # Systeminformationen
        st.markdown("### 📊 System-Status")

        # Prüfe Datenbank-Status
        db_exists = os.path.exists("faiss_db")
        st.metric("Datenbank", "✅ Aktiv" if db_exists else "❌ Nicht gefunden")

        if os.path.exists("chunk_metadata.json"):
            import json

            with open("chunk_metadata.json", "r", encoding="utf-8") as f:
                metadata = json.load(f)
                st.metric("Chunks", f"{metadata.get('total_chunks', 0):,}")

        # Prüfe Bibliotheks-DB
        bib_exists = os.path.exists("juristische_bibliothek.db")
        st.metric("Bibliothek-DB", "✅ Aktiv" if bib_exists else "❌ Nicht gefunden")

    # Seiteninhalt basierend auf Auswahl
    if page == "🏠 Übersicht":
        show_overview()
    elif page == "🧠 Erweiterte Wissensdatenbank":
        show_knowledge_base()
    elif page == "📝 Zitationsengine":
        show_citation_engine()
    elif page == "🔍 Erweiterte Quellensuche":
        show_advanced_search()
    elif page == "📚 Bibliotheksverwaltung":
        show_library_management()
    elif page == "⚙️ System-Setup":
        show_system_setup()
    elif page == "📖 Anleitung":
        show_instructions()


def show_overview():
    """Zeigt die Übersichtsseite"""
    st.header("🏠 System-Übersicht")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 Wissensdatenbank")
        st.markdown(
            """
        - **Natürliche Fragen stellen**
        - **Strukturierte Antworten** mit Rechtsgrundlagen
        - **Automatische Quellenangaben**
        - **Kontextuelle Suche** in Lehrbüchern
        """
        )

        if st.button("🚀 Zur Wissensdatenbank", key="to_kb"):
            st.experimental_set_query_params(page="knowledge")

    with col2:
        st.subheader("📝 Zitationsengine")
        st.markdown(
            """
        - **Automatische Zitatvorschläge**
        - **Verschiedene Zitationsstile**
        - **Rechtsnormen-Erkennung**
        - **Quellenvalidierung**
        """
        )

        if st.button("🚀 Zur Zitationsengine", key="to_cite"):
            st.experimental_set_query_params(page="citation")

    # Statistiken
    st.markdown("---")
    st.subheader("📊 Aktuelle Statistiken")

    if os.path.exists("juristische_bibliothek.db"):
        try:
            from juristische_bibliothek_verwaltung import (
                JuristischeBibliothekVerwaltung,
            )

            bibliothek = JuristischeBibliothekVerwaltung()
            stats = bibliothek.statistiken_abrufen()

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("📚 Lehrbücher", stats.get("gesamtzahl_buecher", 0))

            with col2:
                st.metric("🧩 Text-Chunks", stats.get("gesamtzahl_chunks", 0))

            with col3:
                rechtsbereiche = len(stats.get("nach_rechtsgebiet", {}))
                st.metric("⚖️ Rechtsbereiche", rechtsbereiche)

            with col4:
                st.metric("🔄 Status", "Bereit")

        except ImportError:
            st.warning("Bibliotheksverwaltung nicht verfügbar")
    else:
        st.info("Initialisieren Sie zuerst die Bibliotheksdatenbank")


def show_knowledge_base():
    """Lädt die erweiterte Wissensdatenbank"""
    try:
        exec(open("enhanced_jura_app.py").read())
    except FileNotFoundError:
        st.error("❌ Enhanced Jura App nicht gefunden!")
        st.info(
            "Stellen Sie sicher, dass 'enhanced_jura_app.py' im aktuellen Verzeichnis vorhanden ist."
        )


def show_citation_engine():
    """Zeigt die Zitationsengine"""
    st.header("📝 Zitationsengine")
    st.info("Diese Funktion ist in der erweiterten Wissensdatenbank integriert.")

    if st.button("🔗 Zur integrierten Zitationsengine"):
        st.experimental_set_query_params(page="knowledge")


def show_advanced_search():
    """Zeigt erweiterte Suchfunktionen"""
    st.header("🔍 Erweiterte Quellensuche")
    st.info(
        "Diese Funktion ist in der erweiterten Wissensdatenbank als separater Tab verfügbar."
    )

    if st.button("🔗 Zur erweiterten Suche"):
        st.experimental_set_query_params(page="knowledge")


def show_library_management():
    """Zeigt die Bibliotheksverwaltung"""
    st.header("📚 Bibliotheksverwaltung")

    try:
        from juristische_bibliothek_verwaltung import (
            JuristischeBibliothekVerwaltung,
            LehrbuchMetadata,
        )

        bibliothek = JuristischeBibliothekVerwaltung()

        tab1, tab2, tab3 = st.tabs(
            ["📖 Alle Bücher", "➕ Neues Buch", "📊 Statistiken"]
        )

        with tab1:
            st.subheader("📖 Vorhandene Lehrbücher")
            alle_buecher = bibliothek.alle_lehrbuecher()

            if alle_buecher:
                for buch in alle_buecher:
                    with st.expander(f"📚 {buch['autor']}: {buch['titel']}"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write(f"**Autor:** {buch['autor']}")
                            st.write(f"**Titel:** {buch['titel']}")
                            st.write(f"**Auflage:** {buch['auflage']}")
                            st.write(f"**Jahr:** {buch['jahr']}")

                        with col2:
                            st.write(f"**Verlag:** {buch['verlag']}")
                            st.write(f"**Rechtsgebiet:** {buch['rechtsgebiet']}")
                            st.write(f"**Seitenzahl:** {buch['seitenzahl']}")
                            st.write(f"**Chunks:** {buch['chunk_anzahl']}")
            else:
                st.info("Noch keine Lehrbücher in der Datenbank vorhanden.")

        with tab2:
            st.subheader("➕ Neues Lehrbuch hinzufügen")

            with st.form("neues_buch"):
                col1, col2 = st.columns(2)

                with col1:
                    autor = st.text_input(
                        "Autor *", placeholder="z.B. Medicus/Petersen"
                    )
                    titel = st.text_input(
                        "Titel *", placeholder="z.B. Grundwissen zum Bürgerlichen Recht"
                    )
                    auflage = st.text_input("Auflage", placeholder="z.B. 11. Aufl.")
                    jahr = st.text_input("Jahr *", placeholder="z.B. 2019")

                with col2:
                    verlag = st.text_input("Verlag", placeholder="z.B. C.H.Beck")
                    isbn = st.text_input("ISBN", placeholder="z.B. 978-3-406-73593-9")
                    rechtsbereiche = bibliothek.rechtsbereiche_abrufen()
                    rechtsgebiet = st.selectbox(
                        "Rechtsgebiet *",
                        [bereich["name"] for bereich in rechtsbereiche],
                    )
                    seitenzahl = st.number_input("Seitenzahl", min_value=1, value=300)

                datei_pfad = st.text_input(
                    "Dateipfad (optional)", placeholder="/pfad/zur/pdf/datei.pdf"
                )

                submit = st.form_submit_button("📚 Lehrbuch hinzufügen")

                if submit:
                    if autor and titel and jahr and rechtsgebiet:
                        buch_id = bibliothek.generiere_lehrbuch_id(autor, titel, jahr)

                        neues_buch = LehrbuchMetadata(
                            id=buch_id,
                            titel=titel,
                            autor=autor,
                            auflage=auflage,
                            jahr=jahr,
                            verlag=verlag,
                            isbn=isbn,
                            rechtsgebiet=rechtsgebiet,
                            seitenzahl=seitenzahl,
                            datei_pfad=datei_pfad,
                        )

                        if datei_pfad and os.path.exists(datei_pfad):
                            neues_buch.datei_hash = bibliothek.datei_hash_berechnen(
                                datei_pfad
                            )

                        if bibliothek.lehrbuch_hinzufuegen(neues_buch):
                            st.success("✅ Lehrbuch erfolgreich hinzugefügt!")
                            st.rerun()
                        else:
                            st.error(
                                "❌ Fehler beim Hinzufügen (möglicherweise bereits vorhanden)"
                            )
                    else:
                        st.error("❌ Bitte füllen Sie alle Pflichtfelder (*) aus")

        with tab3:
            st.subheader("📊 Bibliotheks-Statistiken")
            stats = bibliothek.statistiken_abrufen()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("📚 Gesamtzahl Bücher", stats["gesamtzahl_buecher"])

            with col2:
                st.metric("🧩 Gesamtzahl Chunks", stats["gesamtzahl_chunks"])

            with col3:
                bereiche_count = len(stats["nach_rechtsgebiet"])
                st.metric("⚖️ Abgedeckte Rechtsbereiche", bereiche_count)

            if stats["nach_rechtsgebiet"]:
                st.subheader("📊 Verteilung nach Rechtsgebieten")

                import pandas as pd

                df = pd.DataFrame(
                    list(stats["nach_rechtsgebiet"].items()),
                    columns=["Rechtsgebiet", "Anzahl Bücher"],
                )
                st.bar_chart(df.set_index("Rechtsgebiet"))

            # Export-Option
            st.subheader("💾 Daten-Export")
            if st.button("📤 Bibliothek als JSON exportieren"):
                if bibliothek.export_zu_json("bibliothek_export.json"):
                    st.success("✅ Export erfolgreich nach 'bibliothek_export.json'")
                else:
                    st.error("❌ Export fehlgeschlagen")

    except ImportError:
        st.error("❌ Bibliotheksverwaltung nicht verfügbar!")
        st.info(
            "Stellen Sie sicher, dass 'juristische_bibliothek_verwaltung.py' vorhanden ist."
        )


def show_system_setup():
    """Zeigt System-Setup Optionen"""
    st.header("⚙️ System-Setup")

    tab1, tab2, tab3 = st.tabs(["🗃️ Datenbank", "📦 Abhängigkeiten", "🔧 Konfiguration"])

    with tab1:
        st.subheader("🗃️ Datenbank-Setup")

        # FAISS Datenbank Status
        faiss_exists = os.path.exists("faiss_db")
        st.write(
            f"**FAISS Vektordatenbank:** {'✅ Vorhanden' if faiss_exists else '❌ Nicht gefunden'}"
        )

        if not faiss_exists:
            st.warning(
                "Die FAISS-Datenbank wurde nicht gefunden. Führen Sie zuerst das Ingest-Skript aus."
            )
            st.code("python ingest.py", language="bash")

        # Bibliotheks-Datenbank Status
        bib_db_exists = os.path.exists("juristische_bibliothek.db")
        st.write(
            f"**Bibliotheks-Datenbank:** {'✅ Vorhanden' if bib_db_exists else '❌ Nicht gefunden'}"
        )

        if not bib_db_exists:
            if st.button("🔧 Bibliotheks-Datenbank initialisieren"):
                try:
                    from juristische_bibliothek_verwaltung import (
                        JuristischeBibliothekVerwaltung,
                    )

                    JuristischeBibliothekVerwaltung()
                    st.success("✅ Bibliotheks-Datenbank erfolgreich initialisiert!")
                    st.rerun()
                except ImportError:
                    st.error("❌ Bibliotheksverwaltung nicht verfügbar")

    with tab2:
        st.subheader("📦 Abhängigkeiten prüfen")

        required_packages = [
            "streamlit",
            "langchain",
            "langchain-community",
            "langchain-core",
            "langchain-ollama",
            "faiss-cpu",
            "sentence-transformers",
            "PyMuPDF",
        ]

        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                st.write(f"✅ {package}")
            except ImportError:
                st.write(f"❌ {package} - Nicht installiert")

        st.subheader("📋 Installation")
        st.code("pip install -r requirements.txt", language="bash")

    with tab3:
        st.subheader("🔧 System-Konfiguration")

        st.markdown(
            """
        **Aktuelle Konfiguration:**
        - **Embedding-Modell:** sentence-transformers/paraphrase-multilingual-mpnet-base-v2
        - **LLM:** Ollama Llama3.2
        - **Vektor-DB:** FAISS
        - **Device:** MPS (Apple Silicon optimiert)
        """
        )

        st.info(
            "📌 Konfigurationsänderungen können in den jeweiligen Python-Dateien vorgenommen werden."
        )


def show_instructions():
    """Zeigt Bedienungsanleitung"""
    st.header("📖 Bedienungsanleitung")

    st.markdown(
        """
    ## 🚀 Erste Schritte
    
    ### 1. System vorbereiten
    - Installieren Sie alle Abhängigkeiten: `pip install -r requirements.txt`
    - Stellen Sie sicher, dass Ollama läuft und das Llama3.2-Modell verfügbar ist
    
    ### 2. Lehrbuch hinzufügen
    - Legen Sie Ihre PDF-Lehrbücher in das Projektverzeichnis
    - Führen Sie das Ingest-Skript aus: `python ingest.py`
    - Alternativ verwenden Sie die Bibliotheksverwaltung in dieser App
    
    ### 3. System nutzen
    
    #### 🧠 Wissensdatenbank
    - Stellen Sie natürliche Fragen zu Ihren Lehrbüchern
    - Das System findet relevante Textstellen und beantwortet strukturiert
    - Beispiel: "Was sind die Voraussetzungen der GoA?"
    
    #### 📝 Zitationsengine
    - Fügen Sie Ihren Text ein (z.B. aus einer Hausarbeit)
    - Das System findet passende Quellen und schlägt Zitate vor
    - Wählen Sie den gewünschten Zitationsstil (Juristische, APA, Harvard, etc.)
    
    #### 🔍 Erweiterte Suche
    - Semantische Suche nach Konzepten
    - Paragraphen-spezifische Suche
    - Stichwort-basierte Suche
    
    ## 🎯 Tipps für beste Ergebnisse
    
    ### Fragen stellen
    - Seien Sie spezifisch: "Erläutere die Voraussetzungen des § 433 BGB"
    - Nutzen Sie juristische Fachbegriffe
    - Fragen Sie nach Struktur: "Gliedere die Prüfung der..."
    
    ### Zitationen
    - Fügen Sie längere Textpassagen ein für bessere Treffer
    - Nutzen Sie fachspezifische Begriffe
    - Prüfen Sie die Relevanz der Vorschläge
    
    ## ⚠️ Wichtige Hinweise
    
    - **Qualitätskontrolle:** Prüfen Sie alle generierten Antworten und Zitate
    - **Aktualität:** Stellen Sie sicher, dass Ihre Lehrbücher aktuell sind
    - **Vollständigkeit:** Das System kann nur Informationen aus den eingefügten Büchern nutzen
    
    ## 🔧 Fehlerbehebung
    
    ### Häufige Probleme
    - **"Datenbank nicht gefunden":** Führen Sie `python ingest.py` aus
    - **Ollama-Fehler:** Stellen Sie sicher, dass Ollama läuft (`ollama serve`)
    - **Langsame Antworten:** Reduzieren Sie die Kontext-Größe in den Einstellungen
    
    ### Performance-Optimierung
    - Verwenden Sie SSDs für bessere Ladezeiten
    - Mehr RAM beschleunigt die Embeddings-Verarbeitung
    - Apple Silicon Macs nutzen automatisch MPS-Beschleunigung
    
    ## 📞 Support
    
    Bei Problemen oder Fragen:
    1. Prüfen Sie die Konsolen-Ausgaben auf Fehlermeldungen
    2. Überprüfen Sie die System-Setup Seite
    3. Stellen Sie sicher, dass alle Abhängigkeiten installiert sind
    """
    )


if __name__ == "__main__":
    main()
