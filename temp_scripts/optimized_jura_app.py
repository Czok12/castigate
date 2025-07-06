"""
🚀 PERFORMANCE-OPTIMIERTE JURISTISCHE WISSENSDATENBANK
====================================================

Praktische Implementierung der wichtigsten Optimierungen
Direkt einsatzbereit für Integration in das bestehende System
"""

import hashlib
import json
import os
import time
from typing import Dict, List, Optional

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama.llms import OllamaLLM

# === KONFIGURATION ===
DB_PATH = "faiss_db"
CACHE_DIR = "performance_cache"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Performance-Einstellungen
PERFORMANCE_CONFIG = {
    "enable_caching": True,
    "cache_ttl": 86400,  # 24 Stunden
    "max_memory_cache": 1000,
    "batch_size": 10,
    "retrieval_strategy": "hybrid",  # 'fast', 'balanced', 'accurate'
    "chunk_quality_threshold": 0.6,
}

# === INTELLIGENTES CACHING ===


class SmartCache:
    """Intelligentes Caching-System mit Memory + Disk Storage"""

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.memory_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0, "total_requests": 0}

        # Cache-Datei Pfade
        self.query_cache_file = os.path.join(cache_dir, "query_cache.json")
        self.embedding_cache_file = os.path.join(cache_dir, "embedding_cache.json")

        # Lade bestehende Caches
        self._load_disk_cache()

    def _load_disk_cache(self):
        """Lädt Cache von Disk"""
        try:
            if os.path.exists(self.query_cache_file):
                with open(self.query_cache_file, "r", encoding="utf-8") as f:
                    self.query_disk_cache = json.load(f)
            else:
                self.query_disk_cache = {}
        except (json.JSONDecodeError, FileNotFoundError):
            self.query_disk_cache = {}

    def _save_disk_cache(self):
        """Speichert Cache auf Disk"""
        try:
            with open(self.query_cache_file, "w", encoding="utf-8") as f:
                json.dump(self.query_disk_cache, f, ensure_ascii=False, indent=2)
        except (IOError, OSError) as e:
            st.warning(f"Cache konnte nicht gespeichert werden: {e}")

    def get_query_result(self, query: str, retrieval_params: Dict) -> Optional[Dict]:
        """Holt Query-Ergebnis aus Cache"""
        cache_key = self._generate_cache_key(query, retrieval_params)

        self.cache_stats["total_requests"] += 1

        # Memory Cache zuerst
        if cache_key in self.memory_cache:
            self.cache_stats["hits"] += 1
            return self.memory_cache[cache_key]

        # Dann Disk Cache
        if cache_key in self.query_disk_cache:
            result = self.query_disk_cache[cache_key]
            # Promote zu Memory Cache
            if len(self.memory_cache) < PERFORMANCE_CONFIG["max_memory_cache"]:
                self.memory_cache[cache_key] = result
            self.cache_stats["hits"] += 1
            return result

        self.cache_stats["misses"] += 1
        return None

    def cache_query_result(self, query: str, retrieval_params: Dict, result: Dict):
        """Speichert Query-Ergebnis im Cache"""
        cache_key = self._generate_cache_key(query, retrieval_params)

        # Memory Cache
        if len(self.memory_cache) < PERFORMANCE_CONFIG["max_memory_cache"]:
            self.memory_cache[cache_key] = result

        # Disk Cache
        self.query_disk_cache[cache_key] = result

        # Periodisch speichern
        if len(self.query_disk_cache) % 10 == 0:
            self._save_disk_cache()

    def _generate_cache_key(self, query: str, params: Dict) -> str:
        """Generiert eindeutigen Cache-Key"""
        combined = f"{query}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(combined.encode()).hexdigest()

    def get_cache_stats(self) -> Dict:
        """Gibt Cache-Statistiken zurück"""
        hit_rate = (
            self.cache_stats["hits"] / max(self.cache_stats["total_requests"], 1)
        ) * 100

        return {
            "hit_rate": round(hit_rate, 1),
            "memory_cache_size": len(self.memory_cache),
            "disk_cache_size": len(self.query_disk_cache),
            **self.cache_stats,
        }


# === ERWEITERTE RETRIEVAL-STRATEGIEN ===


class EnhancedRetriever:
    """Optimierte Retrieval-Engine mit verschiedenen Strategien"""

    def __init__(self, vectorstore, embedding_model):
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model
        self.cache = SmartCache()

        # Juristische Keywords für bessere Suche
        self.legal_keywords = [
            "§",
            "Art.",
            "Abs.",
            "Nr.",
            "S.",
            "Rn.",
            "BGH",
            "BVerfG",
            "BFH",
            "BAG",
            "BSG",
            "BVerwG",
            "OLG",
            "LG",
            "AG",
            "VG",
            "OVG",
            "BGB",
            "StGB",
            "GG",
            "ZPO",
            "StPO",
            "HGB",
        ]

    def smart_retrieve(
        self, query: str, strategy: Optional[str] = None, k: int = 4
    ) -> List[Dict]:
        """Intelligente Retrieval mit verschiedenen Strategien"""

        if strategy is None:
            strategy = PERFORMANCE_CONFIG["retrieval_strategy"]

        # Cache-Check
        retrieval_params = {"strategy": strategy, "k": k}
        cached_result = self.cache.get_query_result(query, retrieval_params)

        if cached_result and PERFORMANCE_CONFIG["enable_caching"]:
            return cached_result["documents"]

        # Zeitmessung
        start_time = time.time()

        # Retrieval basierend auf Strategie
        if strategy == "fast":
            documents = self._fast_retrieval(query, k)
        elif strategy == "accurate":
            documents = self._accurate_retrieval(query, k)
        else:  # balanced/hybrid
            documents = self._hybrid_retrieval(query, k)

        # Performance-Metriken
        retrieval_time = time.time() - start_time

        # Ergebnis strukturieren
        result = {
            "documents": documents,
            "retrieval_time": retrieval_time,
            "strategy_used": strategy,
            "cache_hit": False,
        }

        # Cache speichern
        if PERFORMANCE_CONFIG["enable_caching"]:
            self.cache.cache_query_result(query, retrieval_params, result)

        return documents

    def _fast_retrieval(self, query: str, k: int) -> List[Dict]:
        """Schnelle Retrieval-Strategie"""

        # Direkte Similarity Search ohne Preprocessing
        docs = self.vectorstore.similarity_search(query, k=k)

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "strategy": "fast",
                "relevance_score": self._calculate_simple_score(
                    query, doc.page_content
                ),
            }
            for doc in docs
        ]

    def _accurate_retrieval(self, query: str, k: int) -> List[Dict]:
        """Präzise Retrieval-Strategie mit erweiterten Metriken"""

        # Erweiterte Suche mit mehr Kandidaten
        candidate_docs = self.vectorstore.similarity_search(query, k=k * 2)

        # Ranking mit erweiterten Metriken
        scored_docs = []
        for doc in candidate_docs:
            score = self._calculate_comprehensive_score(query, doc)
            scored_docs.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "strategy": "accurate",
                    "relevance_score": score,
                    "legal_density": self._calculate_legal_density(doc.page_content),
                    "content_quality": self._calculate_content_quality(
                        doc.page_content
                    ),
                }
            )

        # Sortiere nach Score und nimm Top-k
        scored_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored_docs[:k]

    def _hybrid_retrieval(self, query: str, k: int) -> List[Dict]:
        """Hybride Strategie kombiniert verschiedene Ansätze"""

        # Semantic Search
        semantic_docs = self.vectorstore.similarity_search(query, k=k)

        # Keyword-basierte Suche für juristische Begriffe
        legal_terms = [
            term for term in self.legal_keywords if term.lower() in query.lower()
        ]

        if legal_terms:
            # Erweiterte Suche nach juristischen Begriffen
            legal_query = " ".join(legal_terms)
            legal_docs = self.vectorstore.similarity_search(legal_query, k=k // 2)
        else:
            legal_docs = []

        # Kombiniere und dedupliziere
        all_docs = semantic_docs + legal_docs
        unique_docs = self._deduplicate_documents(all_docs)

        # Score und sortiere
        scored_docs = []
        for doc in unique_docs:
            base_score = self._calculate_simple_score(query, doc.page_content)
            legal_bonus = self._calculate_legal_relevance_bonus(query, doc.page_content)

            total_score = base_score + legal_bonus

            scored_docs.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "strategy": "hybrid",
                    "relevance_score": total_score,
                    "legal_relevance": legal_bonus > 0,
                }
            )

        scored_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored_docs[:k]

    def _calculate_simple_score(self, query: str, content: str) -> float:
        """Einfache Relevanz-Berechnung"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words:
            return 0.0

        intersection = query_words.intersection(content_words)
        return len(intersection) / len(query_words)

    def _calculate_comprehensive_score(self, query: str, doc) -> float:
        """Umfassende Score-Berechnung"""
        content = doc.page_content

        # Basis-Relevanz
        relevance = self._calculate_simple_score(query, content)

        # Juristische Relevanz
        legal_score = self._calculate_legal_density(content)

        # Content-Qualität
        quality_score = self._calculate_content_quality(content)

        # Metadaten-Bonus (z.B. bestimmte Seiten bevorzugen)
        metadata_bonus = 0
        if "page" in doc.metadata:
            # Niedrigere Seitenzahlen könnten wichtiger sein (Grundlagen)
            page_num = doc.metadata.get("page", 999)
            if isinstance(page_num, int) and page_num < 100:
                metadata_bonus = 0.1

        # Gewichtete Kombination
        total_score = (
            relevance * 0.5 + legal_score * 0.3 + quality_score * 0.2 + metadata_bonus
        )

        return total_score

    def _calculate_legal_density(self, text: str) -> float:
        """Berechnet Dichte juristischer Begriffe"""
        words = text.split()
        if not words:
            return 0.0

        legal_terms = sum(
            1
            for word in words
            if any(keyword in word for keyword in self.legal_keywords)
        )
        return min(legal_terms / len(words) * 10, 1.0)  # Normalisiert auf 0-1

    def _calculate_content_quality(self, text: str) -> float:
        """Bewertet Content-Qualität"""

        # Länge (optimal um 800-1200 Zeichen)
        length_score = 1.0 - abs(len(text) - 1000) / 1000
        length_score = max(0, min(1, length_score))

        # Vollständigkeit (vollständige Sätze)
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
        sentence_score = min(len(sentences) / 5, 1.0)  # Optimal: ~5 Sätze

        # Struktur (Absätze)
        paragraph_score = min(text.count("\n") / 3, 1.0)

        return (length_score + sentence_score + paragraph_score) / 3

    def _calculate_legal_relevance_bonus(self, query: str, content: str) -> float:
        """Berechnet Bonus für juristische Relevanz"""

        query_legal_terms = [
            term for term in self.legal_keywords if term.lower() in query.lower()
        ]

        if not query_legal_terms:
            return 0.0

        content_matches = sum(1 for term in query_legal_terms if term in content)
        return content_matches / len(query_legal_terms) * 0.2  # Max 20% Bonus

    def _deduplicate_documents(self, docs: List) -> List:
        """Entfernt Duplikate basierend auf Content"""
        seen_content = set()
        unique_docs = []

        for doc in docs:
            # Verwende ersten Teil des Contents als Identifier
            content_id = doc.page_content[:100]
            if content_id not in seen_content:
                seen_content.add(content_id)
                unique_docs.append(doc)

        return unique_docs


# === OPTIMIERTE RAG-KLASSE ===


class OptimizedJuraRAG:
    """Optimierte RAG-Implementierung für juristische Texte"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.performance_stats = {
            "total_queries": 0,
            "avg_response_time": 0.0,
            "cache_hit_rate": 0.0,
        }

        # Optimierte Prompts
        self.prompts = {
            "standard": self._create_standard_prompt(),
            "detailed": self._create_detailed_prompt(),
            "quick": self._create_quick_prompt(),
        }

    def answer_question(
        self, question: str, mode: str = "standard", context_size: int = 4
    ) -> Dict:
        """Beantwortet Frage mit optimierter Pipeline"""

        start_time = time.time()

        # Retrieval-Strategie basierend auf Modus
        strategy_map = {"quick": "fast", "standard": "balanced", "detailed": "accurate"}

        strategy = strategy_map.get(mode, "balanced")

        # Optimierte Retrieval
        retrieved_docs = self.retriever.smart_retrieve(
            question, strategy=strategy, k=context_size
        )

        # Context zusammenstellen
        context = self._build_context(retrieved_docs)

        # Prompt auswählen und ausführen
        prompt = self.prompts[mode]

        # RAG-Kette
        rag_chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Antwort generieren
        answer = rag_chain.invoke(question)

        # Performance-Metriken aktualisieren
        response_time = time.time() - start_time
        self._update_performance_stats(response_time)

        # Ergebnis zusammenstellen
        result = {
            "answer": answer,
            "sources": retrieved_docs,
            "response_time": response_time,
            "mode": mode,
            "strategy": strategy,
            "context_size": len(retrieved_docs),
            "cache_stats": self.retriever.cache.get_cache_stats(),
        }

        return result

    def _create_standard_prompt(self) -> PromptTemplate:
        """Standard-Prompt für ausgewogene Antworten"""
        template = """Du bist ein juristischer Experte. Beantworte die Frage präzise basierend auf dem Kontext.

Kontext:
{context}

Frage: {question}

Antwort (strukturiert mit Rechtsnormen):"""

        return PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

    def _create_detailed_prompt(self) -> PromptTemplate:
        """Detaillierter Prompt für umfassende Antworten"""
        template = """Du bist ein hochqualifizierter Rechtsexperte. Gib eine umfassende, strukturierte Antwort.

ANTWORTSTRUKTUR:
1. **Kurze Antwort**: [Direkter Bezug zur Frage]
2. **Rechtliche Grundlagen**: [Relevante Paragraphen/Artikel]
3. **Detaillierte Erläuterung**: [Ausführliche Analyse]
4. **Praxishinweise**: [Wenn aus Kontext ableitbar]

Kontext aus Lehrbuch:
{context}

Frage: {question}

Strukturierte Expertenantwort:"""

        return PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

    def _create_quick_prompt(self) -> PromptTemplate:
        """Schneller Prompt für kurze Antworten"""
        template = """Kurze, präzise Antwort basierend auf dem Kontext:

{context}

Frage: {question}

Kurze Antwort:"""

        return PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

    def _build_context(self, retrieved_docs: List[Dict]) -> str:
        """Baut optimierten Context aus Dokumenten"""

        context_parts = []

        for i, doc in enumerate(retrieved_docs):
            # Metadaten hinzufügen
            metadata = doc.get("metadata", {})
            source_info = f"[Quelle {i+1}: {metadata.get('source', 'N/A')}, S. {metadata.get('page', 'N/A')}]"

            # Content mit Qualitäts-Indicator
            content = doc["content"]
            if doc.get("legal_relevance"):
                content = f"🏛️ {content}"  # Marker für juristische Relevanz

            context_parts.append(f"{source_info}\n{content}")

        return "\n\n---\n\n".join(context_parts)

    def _update_performance_stats(self, response_time: float):
        """Aktualisiert Performance-Statistiken"""
        self.performance_stats["total_queries"] += 1

        # Rolling Average für Response Time
        current_avg = self.performance_stats["avg_response_time"]
        total_queries = self.performance_stats["total_queries"]

        new_avg = (current_avg * (total_queries - 1) + response_time) / total_queries
        self.performance_stats["avg_response_time"] = new_avg

    def get_performance_report(self) -> Dict:
        """Gibt detaillierten Performance-Report zurück"""
        cache_stats = self.retriever.cache.get_cache_stats()

        return {
            "query_performance": self.performance_stats,
            "cache_performance": cache_stats,
            "system_health": {
                "avg_response_time_status": (
                    "good"
                    if self.performance_stats["avg_response_time"] < 5.0
                    else "slow"
                ),
                "cache_efficiency_status": (
                    "good" if cache_stats["hit_rate"] > 30 else "low"
                ),
            },
        }


# === STREAMLIT INTEGRATION ===


@st.cache_resource
def load_optimized_system():
    """Lädt das optimierte System mit Caching"""

    if not os.path.exists(DB_PATH):
        return None, None, None

    # Embedding Model
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "mps"},
    )

    # Vectorstore
    vectorstore = FAISS.load_local(
        DB_PATH, embedding_model, allow_dangerous_deserialization=True
    )

    # Enhanced Retriever
    retriever = EnhancedRetriever(vectorstore, embedding_model)

    # LLM
    llm = OllamaLLM(model="llama3.2")

    # Optimized RAG
    rag = OptimizedJuraRAG(retriever, llm)

    return retriever, llm, rag


def create_performance_dashboard():
    """Erstellt Performance-Dashboard für Streamlit"""

    st.sidebar.markdown("### ⚡ Performance-Optimierungen")

    # Konfiguration
    strategy = st.sidebar.selectbox(
        "Retrieval-Strategie",
        ["balanced", "fast", "accurate"],
        help="Fast=Geschwindigkeit, Accurate=Präzision, Balanced=Ausgewogen",
    )

    mode = st.sidebar.selectbox(
        "Antwort-Modus",
        ["standard", "quick", "detailed"],
        help="Quick=Kurz, Standard=Ausgewogen, Detailed=Umfassend",
    )

    context_size = st.sidebar.slider(
        "Kontext-Größe", 2, 8, 4, help="Anzahl der verwendeten Dokument-Chunks"
    )

    enable_cache = st.sidebar.checkbox(
        "Caching aktivieren",
        value=PERFORMANCE_CONFIG["enable_caching"],
        help="Beschleunigt wiederholte Anfragen",
    )

    # Update globale Konfiguration
    PERFORMANCE_CONFIG["retrieval_strategy"] = strategy
    PERFORMANCE_CONFIG["enable_caching"] = enable_cache

    return {
        "strategy": strategy,
        "mode": mode,
        "context_size": context_size,
        "enable_cache": enable_cache,
    }


def display_performance_metrics(rag_system: OptimizedJuraRAG):
    """Zeigt Performance-Metriken an"""

    if rag_system:
        performance_report = rag_system.get_performance_report()

        with st.sidebar.expander("📊 Performance-Metriken"):

            # Query Performance
            query_perf = performance_report["query_performance"]
            st.metric(
                "Durchschn. Antwortzeit", f"{query_perf['avg_response_time']:.2f}s"
            )
            st.metric("Gesamt-Anfragen", query_perf["total_queries"])

            # Cache Performance
            cache_perf = performance_report["cache_performance"]
            st.metric("Cache Hit-Rate", f"{cache_perf['hit_rate']:.1f}%")
            st.metric("Memory Cache", cache_perf["memory_cache_size"])

            # System Health
            health = performance_report["system_health"]
            st.write("**System-Status:**")
            st.write(f"⚡ Antwortzeit: {health['avg_response_time_status']}")
            st.write(f"💾 Cache-Effizienz: {health['cache_efficiency_status']}")


# === MAIN OPTIMIZED APP ===


def main_optimized():
    """Hauptfunktion der optimierten App"""

    st.title("🚀 Optimierte Juristische Wissensdatenbank")

    # Lade optimiertes System
    _, _, rag = load_optimized_system()

    if not rag:
        st.error("❌ Optimiertes System konnte nicht geladen werden!")
        return

    # Performance-Dashboard
    config = create_performance_dashboard()

    # Performance-Metriken anzeigen
    display_performance_metrics(rag)

    # Hauptinterface
    st.markdown("### 💬 Optimierte Fragenbearbeitung")

    question = st.text_area(
        "Ihre juristische Frage:",
        placeholder="z.B. Was sind die Voraussetzungen der GoA?",
        height=100,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("🔍 Optimiert beantworten", type="primary"):
            if question:
                with st.spinner(f"Verarbeite mit {config['mode']}-Modus..."):

                    result = rag.answer_question(
                        question=question,
                        mode=config["mode"],
                        context_size=config["context_size"],
                    )

                    # Antwort anzeigen
                    st.success(
                        f"✅ Antwort in {result['response_time']:.2f}s generiert"
                    )
                    st.markdown("### 📖 Antwort:")
                    st.markdown(result["answer"])

                    # Performance-Info
                    with st.expander("⚡ Performance-Details"):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Antwortzeit", f"{result['response_time']:.2f}s")
                        with col2:
                            st.metric("Strategie", result["strategy"])
                        with col3:
                            cache_hit_rate = result["cache_stats"]["hit_rate"]
                            st.metric("Cache Hit-Rate", f"{cache_hit_rate}%")

                        st.json(result["cache_stats"])

                    # Quellen
                    with st.expander("📚 Verwendete Quellen"):
                        for i, source in enumerate(result["sources"]):
                            st.markdown(f"**Quelle {i+1}:**")

                            # Metadaten
                            metadata = source.get("metadata", {})
                            st.write(
                                f"📄 {metadata.get('source', 'N/A')}, Seite {metadata.get('page', 'N/A')}"
                            )

                            # Relevanz-Info
                            if "relevance_score" in source:
                                st.progress(source["relevance_score"])
                                st.caption(f"Relevanz: {source['relevance_score']:.2%}")

                            # Content
                            st.info(
                                source["content"][:300] + "..."
                                if len(source["content"]) > 300
                                else source["content"]
                            )
                            st.markdown("---")

    with col2:
        st.markdown("**Konfiguration:**")
        st.write(f"🎯 Modus: {config['mode']}")
        st.write(f"⚡ Strategie: {config['strategy']}")
        st.write(f"📊 Kontext: {config['context_size']} Chunks")
        st.write(f"💾 Cache: {'✅' if config['enable_cache'] else '❌'}")


if __name__ == "__main__":
    main_optimized()
