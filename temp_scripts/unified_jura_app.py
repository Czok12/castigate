#!/usr/bin/env python3
"""
🏛️ UNIFIED JURISTISCHE WISSENSDATENBANK
======================================

Konsolidierte Hauptanwendung mit allen Optimierungsmodulen
Vereint alle Features in einer benutzerfreundlichen Oberfläche
"""
import sys
from pathlib import Path

import streamlit as st

# Füge das aktuelle Verzeichnis zum Python-Pfad hinzu
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Importiere verfügbare Module
AVAILABLE_MODULES = {}

try:
    from optimized_jura_app import OptimizedJuraRAG, load_optimized_system

    AVAILABLE_MODULES["optimized"] = True
except ImportError:
    AVAILABLE_MODULES["optimized"] = False

try:
    from ultra_optimized_jura_app import UltraOptimizedJuraRAG, load_ultra_system

    AVAILABLE_MODULES["ultra"] = True
except ImportError:
    AVAILABLE_MODULES["ultra"] = False

try:
    from ultimate_juristic_ai import UltimateJuristicAI

    AVAILABLE_MODULES["ultimate"] = True
except ImportError:
    AVAILABLE_MODULES["ultimate"] = False

try:
    from enhanced_jura_app import EnhancedJuraKnowledgeBase, JuraCitationEngine

    AVAILABLE_MODULES["enhanced"] = True
except ImportError:
    AVAILABLE_MODULES["enhanced"] = False

# Fallback zu Basis-App
try:
    from app import load_llm, load_retriever

    AVAILABLE_MODULES["basic"] = True
except ImportError:
    AVAILABLE_MODULES["basic"] = False


def main():
    """Unified Hauptanwendung"""
    st.set_page_config(
        page_title="🏛️ Juristische Wissensdatenbank - Unified",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header
    st.title("🏛️ Juristische Wissensdatenbank - Unified System")

    # System-Status anzeigen
    with st.sidebar:
        st.header("📊 System-Status")

        module_status = {
            "🚀 Ultra-Optimiert": AVAILABLE_MODULES.get("ultra", False),
            "⚡ Optimiert": AVAILABLE_MODULES.get("optimized", False),
            "🌟 Ultimate AI": AVAILABLE_MODULES.get("ultimate", False),
            "📚 Enhanced": AVAILABLE_MODULES.get("enhanced", False),
            "📖 Basis": AVAILABLE_MODULES.get("basic", False),
        }

        for module_name, is_available in module_status.items():
            if is_available:
                st.success(f"{module_name} ✅")
            else:
                st.error(f"{module_name} ❌")

        st.markdown("---")

        # System-Modus auswählen
        available_modes = []
        if AVAILABLE_MODULES.get("ultra"):
            available_modes.append("🚀 Ultra-Optimiert")
        if AVAILABLE_MODULES.get("ultimate"):
            available_modes.append("🌟 Ultimate AI")
        if AVAILABLE_MODULES.get("optimized"):
            available_modes.append("⚡ Optimiert")
        if AVAILABLE_MODULES.get("enhanced"):
            available_modes.append("📚 Enhanced")
        if AVAILABLE_MODULES.get("basic"):
            available_modes.append("📖 Basis")

        if not available_modes:
            st.error("❌ Keine Module verfügbar!")
            st.stop()

        selected_mode = st.selectbox(
            "🎯 System-Modus wählen:",
            available_modes,
            help="Wählen Sie den gewünschten System-Modus basierend auf verfügbaren Modulen",
        )

    # Haupt-Content basierend auf gewähltem Modus
    if selected_mode == "🚀 Ultra-Optimiert" and AVAILABLE_MODULES.get("ultra"):
        run_ultra_mode()
    elif selected_mode == "🌟 Ultimate AI" and AVAILABLE_MODULES.get("ultimate"):
        run_ultimate_mode()
    elif selected_mode == "⚡ Optimiert" and AVAILABLE_MODULES.get("optimized"):
        run_optimized_mode()
    elif selected_mode == "📚 Enhanced" and AVAILABLE_MODULES.get("enhanced"):
        run_enhanced_mode()
    elif selected_mode == "📖 Basis" and AVAILABLE_MODULES.get("basic"):
        run_basic_mode()
    else:
        st.error(f"Modus '{selected_mode}' ist nicht verfügbar!")


def run_ultra_mode():
    """Ultra-Optimierter Modus"""
    st.header("🚀 Ultra-Optimiertes RAG-System")

    try:
        from ultra_optimized_jura_app import main_ultra

        main_ultra()
    except Exception as e:
        st.error(f"Fehler beim Laden des Ultra-Modus: {e}")
        st.info("Fallback zu einem anderen verfügbaren Modus.")


def run_ultimate_mode():
    """Ultimate AI Modus"""
    st.header("🌟 Ultimate Juristische KI")

    try:
        from ultimate_juristic_ai import main

        main()
    except Exception as e:
        st.error(f"Fehler beim Laden des Ultimate-Modus: {e}")


def run_optimized_mode():
    """Optimierter Modus"""
    st.header("⚡ Performance-Optimiertes System")

    try:
        from optimized_jura_app import main_optimized

        main_optimized()
    except Exception as e:
        st.error(f"Fehler beim Laden des Optimized-Modus: {e}")


def run_enhanced_mode():
    """Enhanced Modus"""
    st.header("📚 Enhanced Jura-System")

    try:
        from enhanced_jura_app import main

        main()
    except Exception as e:
        st.error(f"Fehler beim Laden des Enhanced-Modus: {e}")


def run_basic_mode():
    """Basis-Modus"""
    st.header("📖 Basis RAG-System")

    try:
        # Implementiere Basis-Funktionalität
        st.info("📖 Basis-Modus wird geladen...")

        # Import basic components
        retriever = load_retriever()
        if retriever is None:
            st.error(
                "❌ Vektordatenbank nicht gefunden. Bitte führen Sie zuerst ingest.py aus."
            )
            return

        llm = load_llm()

        # Basic chat interface
        st.subheader("💬 Frage stellen")
        user_question = st.text_area(
            "Ihre Frage:",
            placeholder="Stellen Sie hier Ihre juristische Frage...",
            height=100,
        )

        if st.button("🔍 Antwort suchen", type="primary"):
            if user_question:
                with st.spinner("🔍 Suche nach relevanten Informationen..."):
                    # Basic RAG pipeline
                    docs = retriever.invoke(user_question)
                    context = "\n\n".join([doc.page_content for doc in docs])

                    from langchain_core.output_parsers import StrOutputParser
                    from langchain_core.prompts import PromptTemplate

                    prompt_template = PromptTemplate(
                        input_variables=["context", "question"],
                        template="""
Du bist ein erfahrener Rechtsexperte. Beantworte die folgende Frage basierend auf dem gegebenen Kontext aus einem Rechtslehrbuch.

Kontext:
{context}

Frage: {question}

Antwort:
""",
                    )

                    chain = prompt_template | llm | StrOutputParser()
                    answer = chain.invoke(
                        {"context": context, "question": user_question}
                    )

                    st.subheader("📝 Antwort:")
                    st.write(answer)

                    # Zeige Quellen
                    st.subheader("📚 Verwendete Quellen:")
                    for i, doc in enumerate(docs, 1):
                        with st.expander(f"Quelle {i}"):
                            st.write(doc.page_content)
                            st.caption(f"Metadaten: {doc.metadata}")
            else:
                st.warning("⚠️ Bitte geben Sie eine Frage ein.")

    except Exception as e:
        st.error(f"Fehler beim Laden des Basis-Modus: {e}")


if __name__ == "__main__":
    main()
