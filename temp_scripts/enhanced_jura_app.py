import json
import os
import re
from datetime import datetime
from typing import Dict, List

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama.llms import OllamaLLM

# --- Konfiguration ---
DB_PATH = "faiss_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CITATION_STYLES = {
    "DIN": "DIN ISO 690",
    "APA": "APA Style",
    "Harvard": "Harvard Style",
    "Chicago": "Chicago Manual",
    "Juristische": "Juristische Zitierweise",
}


class JuraCitationEngine:
    """Erweiterte Zitationsengine für juristische Texte"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def extract_legal_references(self, text: str) -> List[Dict[str, str]]:
        """Extrahiert juristische Referenzen aus Text"""
        patterns = {
            "paragraph": r"§\s*(\d+[a-z]?)\s*(?:Abs\.\s*(\d+))?\s*(?:S\.\s*(\d+))?\s*([A-Z]+)",
            "article": r"Art\.\s*(\d+[a-z]?)\s*(?:Abs\.\s*(\d+))?\s*([A-Z]+)",
            "bgh_decision": r"BGH.*?(\d{1,2}\.\d{1,2}\.\d{4})",
            "bverfg_decision": r"BVerfG.*?(\d{1,2}\.\d{1,2}\.\d{4})",
            "page_reference": r"S\.\s*(\d+(?:-\d+)?)",
        }

        references = []
        for ref_type, pattern in patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                references.append(
                    {
                        "type": ref_type,
                        "match": match.group(0),
                        "groups": match.groups(),
                    }
                )

        return references

    def generate_citation(self, doc_metadata: Dict, style: str = "Juristische") -> str:
        """Generiert korrekte Zitation basierend auf Stil"""
        source = doc_metadata.get("source", "Unbekannte Quelle")
        page = doc_metadata.get("page", "")

        # Versuche Autor und Titel aus Quelle zu extrahieren
        author, title = self._parse_source_info(source)

        if style == "Juristische":
            # Juristische Zitierweise: Autor, Titel, Auflage, Jahr, Seite
            citation = f"{author}, {title}"
            if page:
                citation += f", S. {page}"
            return citation

        elif style == "APA":
            # APA: Autor (Jahr). Titel. Verlag.
            year = self._extract_year_from_source(source)
            return f"{author} ({year}). {title}."

        elif style == "Harvard":
            # Harvard: Autor Jahr, Seite
            year = self._extract_year_from_source(source)
            citation = f"{author} {year}"
            if page:
                citation += f", S. {page}"
            return citation

        else:
            return f"{source}, S. {page}" if page else source

    def _parse_source_info(self, source: str) -> tuple:
        """Parst Quelleninformation um Autor und Titel zu extrahieren"""
        # Vereinfachte Logik - kann erweitert werden
        parts = source.split(".")
        if len(parts) >= 2:
            return parts[0].strip(), parts[1].strip()
        return source, ""

    def _extract_year_from_source(self, source: str) -> str:
        """Extrahiert Jahr aus Quellenangabe"""
        year_match = re.search(r"(\d{4})", source)
        return year_match.group(1) if year_match else "o.J."

    def suggest_citations_for_text(
        self, user_text: str, num_suggestions: int = 5
    ) -> List[Dict]:
        """Schlägt Zitationen für einen Benutzertext vor"""
        # Hole relevante Dokumente
        retrieved_docs = self.retriever.invoke(user_text)[:num_suggestions]

        suggestions = []
        for i, doc in enumerate(retrieved_docs):
            # Berechne Relevanz-Score (vereinfacht)
            relevance = self._calculate_relevance(user_text, doc.page_content)

            suggestion = {
                "rank": i + 1,
                "content": (
                    doc.page_content[:300] + "..."
                    if len(doc.page_content) > 300
                    else doc.page_content
                ),
                "metadata": doc.metadata,
                "relevance_score": relevance,
                "citations": {
                    style: self.generate_citation(doc.metadata, style)
                    for style in CITATION_STYLES.keys()
                },
                "legal_references": self.extract_legal_references(doc.page_content),
            }
            suggestions.append(suggestion)

        return suggestions

    def _calculate_relevance(self, query: str, content: str) -> float:
        """Berechnet Relevanz-Score zwischen Query und Content"""
        # Vereinfachter Ansatz - in Realität würde man cosine similarity verwenden
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words:
            return 0.0

        intersection = query_words.intersection(content_words)
        return len(intersection) / len(query_words)


class EnhancedJuraKnowledgeBase:
    """Erweiterte Wissensdatenbank mit verbesserter Antwortlogik"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.citation_engine = JuraCitationEngine(retriever, llm)

    def answer_legal_question(self, question: str, context_size: int = 4) -> Dict:
        """Beantwortet juristische Fragen mit erweiterten Kontext"""

        # Erweiterte Prompt-Vorlage
        enhanced_prompt = """
        Du bist ein hochqualifizierter Rechtsexperte und juristischer Tutor. 
        
        AUFGABE: Beantworte die Frage präzise, strukturiert und wissenschaftlich fundiert.
        
        VORGEHEN:
        1. Analysiere die Frage und identifiziere die Rechtsbereiche
        2. Nutze AUSSCHLIESSLICH die Informationen aus dem bereitgestellten Kontext
        3. Strukturiere deine Antwort logisch und verständlich
        4. Zitiere relevante Paragraphen, wenn im Kontext erwähnt
        5. Wenn der Kontext nicht ausreicht, sage das explizit
        
        ANTWORTSTRUKTUR:
        1. **Kurze Antwort:** [Direkte Antwort in 1-2 Sätzen]
        2. **Ausführliche Erläuterung:** [Detaillierte Analyse mit Struktur]
        3. **Rechtliche Grundlagen:** [Relevante Paragraphen/Artikel aus dem Kontext]
        4. **Praxishinweise:** [Wenn aus dem Kontext ableitbar]
        
        WICHTIG: Erfinde keine Rechtsnormen oder Fakten. Stütze dich ausschließlich auf den Kontext.
        
        Kontext aus Lehrbuch:
        {context}
        
        Frage:
        {question}
        
        Strukturierte Antwort:
        """

        prompt = PromptTemplate(
            template=enhanced_prompt, input_variables=["context", "question"]
        )

        # RAG-Kette mit erweiterten Kontext
        retriever_with_size = self.retriever
        retriever_with_size.search_kwargs = {"k": context_size}

        rag_chain = (
            {"context": retriever_with_size, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Generiere Antwort
        answer = rag_chain.invoke(question)

        # Hole verwendete Dokumente für Zitationen
        retrieved_docs = retriever_with_size.invoke(question)

        return {
            "answer": answer,
            "sources": retrieved_docs,
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "context_size": context_size,
        }


# --- Streamlit App ---
@st.cache_resource
def load_retriever():
    """Lädt den Retriever"""
    if not os.path.exists(DB_PATH):
        return None

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "mps"},
    )
    vectorstore = FAISS.load_local(
        DB_PATH, embedding_model, allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})


@st.cache_resource
def load_llm():
    """Lädt das Sprachmodell"""
    return OllamaLLM(model="llama3.2")


def main():
    st.set_page_config(
        page_title="🏛️ Juristische Wissensdatenbank & Zitationsengine",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🏛️ Juristische Wissensdatenbank & Zitationsengine")
    st.markdown("---")

    # Lade Komponenten
    retriever = load_retriever()
    if retriever is None:
        st.error(f"Fehler: Die Vektordatenbank unter '{DB_PATH}' wurde nicht gefunden.")
        st.warning("Bitte führe zuerst das Ingest-Skript aus.")
        st.stop()

    llm = load_llm()

    # Initialisiere Systeme
    knowledge_base = EnhancedJuraKnowledgeBase(retriever, llm)
    citation_engine = JuraCitationEngine(retriever, llm)

    # Sidebar für Einstellungen
    with st.sidebar:
        st.header("⚙️ Einstellungen")
        context_size = st.slider(
            "Kontext-Größe", 2, 10, 4, help="Anzahl der verwendeten Dokument-Chunks"
        )

        citation_style = st.selectbox(
            "Zitationsstil",
            options=list(CITATION_STYLES.keys()),
            format_func=lambda x: CITATION_STYLES[x],
        )

        st.markdown("---")
        st.markdown("### 📊 Datenbankinfo")
        if os.path.exists("chunk_metadata.json"):
            with open("chunk_metadata.json", "r", encoding="utf-8") as f:
                metadata = json.load(f)
                st.metric("Gesamt Chunks", metadata.get("total_chunks", "N/A"))
                st.metric(
                    "Durchschn. Chunk-Größe",
                    f"{metadata.get('average_chunk_size', 'N/A')} Zeichen",
                )

    # Haupttabs
    tab1, tab2, tab3 = st.tabs(
        [
            "🧠 Erweiterte Wissensdatenbank",
            "📝 Zitationsengine",
            "🔍 Erweiterte Quellensuche",
        ]
    )

    # Tab 1: Erweiterte Wissensdatenbank
    with tab1:
        st.header("🧠 Stelle komplexe juristische Fragen")
        st.markdown(
            """
        Diese erweiterte Wissensdatenbank beantwortet komplexe juristische Fragen strukturiert und wissenschaftlich fundiert.
        
        **Beispiel-Fragen:**
        - "Was sind die Voraussetzungen der Geschäftsführung ohne Auftrag?"
        - "Erläutere den Unterschied zwischen Anfechtung und Nichtigkeit von Rechtsgeschäften"
        - "Welche Haftungsrisiken bestehen bei Vereinsvorständen?"
        """
        )

        question = st.text_area(
            "Ihre juristische Frage:",
            placeholder="Stellen Sie hier Ihre detaillierte Frage zum Rechtsstoff...",
            height=100,
            key="enhanced_question",
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            ask_button = st.button("🔍 Frage beantworten", type="primary")
        with col2:
            st.markdown(f"*Kontext: {context_size} Chunks*")

        if ask_button and question:
            with st.spinner("Analysiere Rechtsfrage und suche relevante Quellen..."):
                result = knowledge_base.answer_legal_question(question, context_size)

                # Antwort anzeigen
                st.success("✅ Antwort generiert")
                st.markdown("### 📖 Antwort:")
                st.markdown(result["answer"])

                # Quellen anzeigen
                with st.expander("📚 Verwendete Quellen und Zitationen"):
                    for i, doc in enumerate(result["sources"]):
                        st.markdown(f"**Quelle {i+1}:**")
                        citation = citation_engine.generate_citation(
                            doc.metadata, citation_style
                        )
                        st.code(citation, language="text")
                        st.info(
                            doc.page_content[:500] + "..."
                            if len(doc.page_content) > 500
                            else doc.page_content
                        )
                        st.markdown("---")

    # Tab 2: Zitationsengine
    with tab2:
        st.header("📝 Automatische Zitationsvorschläge")
        st.markdown(
            """
        Fügen Sie Ihren Text ein und erhalten Sie automatisch passende Zitationsvorschläge mit korrekten Quellenangaben.
        
        **So funktioniert es:**
        1. Text eingeben (z.B. aus Ihrer Hausarbeit)
        2. System findet passende Stellen im Lehrbuch
        3. Generiert korrekte Zitationen in verschiedenen Stilen
        """
        )

        user_text = st.text_area(
            "Ihr Text für Zitationsvorschläge:",
            placeholder="Fügen Sie hier den Text ein, für den Sie Quellen und Zitate benötigen...",
            height=150,
            key="citation_text",
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            num_suggestions = st.slider("Anzahl Vorschläge", 1, 10, 5)
        with col2:
            suggest_button = st.button("🎯 Zitate vorschlagen", type="primary")

        if suggest_button and user_text:
            with st.spinner("Suche passende Zitate und generiere Quellenangaben..."):
                suggestions = citation_engine.suggest_citations_for_text(
                    user_text, num_suggestions
                )

                st.success(f"✅ {len(suggestions)} Zitationsvorschläge gefunden")

                for suggestion in suggestions:
                    with st.expander(
                        f"📌 Vorschlag {suggestion['rank']} (Relevanz: {suggestion['relevance_score']:.2%})"
                    ):

                        # Zitationen in verschiedenen Stilen
                        st.markdown("**🎓 Zitationen:**")
                        selected_citation = suggestion["citations"].get(
                            citation_style, "N/A"
                        )
                        st.code(selected_citation, language="text")

                        # Andere Stile als Details
                        with st.expander("Andere Zitationsstile"):
                            for style, citation in suggestion["citations"].items():
                                if style != citation_style:
                                    st.markdown(f"**{CITATION_STYLES[style]}:**")
                                    st.code(citation, language="text")

                        # Rechtliche Referenzen
                        if suggestion["legal_references"]:
                            st.markdown("**⚖️ Erkannte Rechtsnormen:**")
                            for ref in suggestion["legal_references"]:
                                st.badge(ref["match"])

                        # Textinhalt
                        st.markdown("**📄 Textinhalt:**")
                        st.info(suggestion["content"])

                        # Metadaten
                        metadata = suggestion["metadata"]
                        st.caption(
                            f"Quelle: {metadata.get('source', 'N/A')} | Seite: {metadata.get('page', 'N/A')}"
                        )

    # Tab 3: Erweiterte Quellensuche
    with tab3:
        st.header("🔍 Erweiterte Quellensuche")
        st.markdown(
            """
        Durchsuchen Sie die juristische Datenbank mit verschiedenen Suchstrategien.
        """
        )

        search_type = st.radio(
            "Suchmodus:", ["Semantische Suche", "Paragraphen-Suche", "Stichwort-Suche"]
        )

        if search_type == "Semantische Suche":
            search_query = st.text_input(
                "Semantische Suche:", placeholder="z.B. Haftung des Geschäftsführers"
            )
        elif search_type == "Paragraphen-Suche":
            search_query = st.text_input(
                "Paragraphen-Suche:", placeholder="z.B. § 433 BGB, Art. 3 GG"
            )
        else:
            search_query = st.text_input(
                "Stichwort-Suche:",
                placeholder="z.B. Anfechtung, Nichtigkeit, Verjährung",
            )

        if search_query:
            with st.spinner("Durchsuche Datenbank..."):
                results = retriever.invoke(search_query)

                st.success(f"✅ {len(results)} Ergebnisse gefunden")

                for i, doc in enumerate(results):
                    with st.expander(f"📖 Ergebnis {i+1}"):
                        # Zitation
                        citation = citation_engine.generate_citation(
                            doc.metadata, citation_style
                        )
                        st.markdown("**Zitation:**")
                        st.code(citation, language="text")

                        # Inhalt
                        st.markdown("**Inhalt:**")
                        st.info(doc.page_content)

                        # Rechtliche Referenzen
                        legal_refs = citation_engine.extract_legal_references(
                            doc.page_content
                        )
                        if legal_refs:
                            st.markdown("**Rechtsnormen:**")
                            for ref in legal_refs:
                                st.badge(ref["match"])


if __name__ == "__main__":
    main()
