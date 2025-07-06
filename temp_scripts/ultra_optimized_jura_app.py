"""
🚀 ULTRA-OPTIMIERTE JURISTISCHE WISSENSDATENBANK
================================================

Integration aller Optimierungsmodule für maximale Performance und Qualität
"""

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

# Import unserer neuen Optimierungsmodule
try:
    from advanced_retrieval_engine import MultiStageRetriever
    from answer_quality_system import ConfidenceCalculator, StructuredAnswerGenerator
    from performance_monitoring import (
        AutoOptimizer,
        PerformanceMetrics,
        PerformanceMonitor,
        PerformanceReporter,
    )

    OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    st.warning(f"Einige Optimierungsmodule nicht verfügbar: {e}")
    OPTIMIZATIONS_AVAILABLE = False

# === KONFIGURATION ===
DB_PATH = "faiss_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

ULTRA_CONFIG = {
    "enable_multi_stage_retrieval": True,
    "enable_confidence_scoring": True,
    "enable_performance_monitoring": True,
    "enable_auto_optimization": True,
    "retrieval_stages": ["semantic", "keyword", "legal_entity", "rerank"],
    "answer_template": "legal_analysis",
    "monitoring_window": 50,
    "optimization_interval": 10,  # Nach 10 Queries
}


class UltraOptimizedJuraRAG:
    """Ultra-optimierte RAG-Engine mit allen verfügbaren Optimierungen"""

    def __init__(self):
        self.vectorstore = None
        self.embedding_model = None
        self.llm = None

        # Optimierungskomponenten
        self.multi_stage_retriever = None
        self.confidence_calculator = None
        self.answer_generator = None
        self.performance_monitor = None
        self.auto_optimizer = None
        self.performance_reporter = None

        # Statistiken
        self.query_count = 0
        self.total_response_time = 0.0

        # Initialisierung
        self._initialize_system()

    def _initialize_system(self):
        """Initialisiert das ultra-optimierte System"""

        try:
            # Basis-Komponenten laden
            if not os.path.exists(DB_PATH):
                st.error("❌ Datenbank nicht gefunden. Führen Sie erst das Ingest aus.")
                return False

            # Embedding Model
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "mps"},
                encode_kwargs={"normalize_embeddings": True},
            )

            # Vectorstore
            self.vectorstore = FAISS.load_local(
                DB_PATH, self.embedding_model, allow_dangerous_deserialization=True
            )

            # LLM
            self.llm = OllamaLLM(model="llama3.2")

            # Optimierungskomponenten initialisieren
            if OPTIMIZATIONS_AVAILABLE:
                self._initialize_optimizations()

            st.success("✅ Ultra-optimiertes System erfolgreich initialisiert!")
            return True

        except Exception as e:
            st.error(f"❌ Fehler bei der Initialisierung: {e}")
            return False

    def _initialize_optimizations(self):
        """Initialisiert alle Optimierungskomponenten"""

        # Multi-Stage Retriever
        if ULTRA_CONFIG["enable_multi_stage_retrieval"]:
            self.multi_stage_retriever = MultiStageRetriever(
                self.vectorstore, self.embedding_model
            )

            # TF-IDF vorbereiten
            try:
                all_docs = self.vectorstore.similarity_search("", k=100)
                doc_texts = [doc.page_content for doc in all_docs]
                self.multi_stage_retriever.fit_tfidf(doc_texts)
            except Exception as e:
                st.warning(f"TF-IDF Initialisierung fehlgeschlagen: {e}")

        # Confidence Calculator & Answer Generator
        if ULTRA_CONFIG["enable_confidence_scoring"]:
            self.confidence_calculator = ConfidenceCalculator()
            self.answer_generator = StructuredAnswerGenerator()

        # Performance Monitoring
        if ULTRA_CONFIG["enable_performance_monitoring"]:
            self.performance_monitor = PerformanceMonitor(
                window_size=ULTRA_CONFIG["monitoring_window"]
            )

            if ULTRA_CONFIG["enable_auto_optimization"]:
                self.auto_optimizer = AutoOptimizer(self.performance_monitor)
                self.performance_reporter = PerformanceReporter(
                    self.performance_monitor, self.auto_optimizer
                )

    def ultra_answer_question(
        self,
        question: str,
        mode: str = "balanced",
        use_confidence_scoring: bool = True,
        use_structured_output: bool = True,
    ) -> Dict:
        """Ultra-optimierte Fragebeantwortung mit allen Features"""

        start_time = time.time()
        self.query_count += 1

        try:
            # Phase 1: Erweiterte Retrieval
            if (
                self.multi_stage_retriever
                and ULTRA_CONFIG["enable_multi_stage_retrieval"]
            ):
                retrieved_docs = self.multi_stage_retriever.multi_stage_retrieve(
                    question,
                    k=self._get_optimal_k(mode),
                    stages=ULTRA_CONFIG["retrieval_stages"],
                )
                retrieval_time = time.time() - start_time
                retrieval_method = "multi_stage"
            else:
                # Fallback: Standard Retrieval
                docs = self.vectorstore.similarity_search(question, k=4)
                retrieved_docs = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "retrieval_method": "standard",
                    }
                    for doc in docs
                ]
                retrieval_time = time.time() - start_time
                retrieval_method = "standard"

            # Phase 2: Context Building
            context = self._build_enhanced_context(retrieved_docs)

            # Phase 3: LLM Generation
            llm_start_time = time.time()
            prompt = self._select_optimal_prompt(mode)

            rag_chain = (
                {"context": lambda x: context, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )

            raw_answer = rag_chain.invoke(question)
            llm_generation_time = time.time() - llm_start_time

            # Phase 4: Antwort-Optimierung
            if (
                use_structured_output
                and self.answer_generator
                and ULTRA_CONFIG["enable_confidence_scoring"]
            ):

                # Confidence-Scoring und strukturierte Antwort
                retrieval_scores = [
                    doc.get("relevance_score", 0.5) for doc in retrieved_docs
                ]

                structured_result = self.answer_generator.generate_structured_answer(
                    raw_answer,
                    retrieved_docs,
                    retrieval_scores,
                    question,
                    template_type=ULTRA_CONFIG["answer_template"],
                )

                final_answer = structured_result["structured_answer"]
                quality_metrics = structured_result["quality_metrics"]
                confidence_summary = structured_result["confidence_summary"]

            else:
                # Standard-Antwort
                final_answer = raw_answer
                quality_metrics = None
                confidence_summary = {"overall_confidence": "nicht verfügbar"}

            # Phase 5: Performance-Tracking
            total_response_time = time.time() - start_time
            self.total_response_time += total_response_time

            if (
                self.performance_monitor
                and ULTRA_CONFIG["enable_performance_monitoring"]
            ):
                self._record_performance_metrics(
                    question,
                    total_response_time,
                    retrieval_time,
                    llm_generation_time,
                    retrieved_docs,
                    quality_metrics,
                )

            # Phase 6: Auto-Optimierung (periodisch)
            if (
                self.auto_optimizer
                and ULTRA_CONFIG["enable_auto_optimization"]
                and self.query_count % ULTRA_CONFIG["optimization_interval"] == 0
            ):

                optimization_result = self.auto_optimizer.run_optimization_cycle()
                if optimization_result["status"] == "optimizations_applied":
                    st.info(
                        f"🔧 Automatische Optimierungen angewendet: {len(optimization_result['optimizations'])}"
                    )

            # Ergebnis zusammenstellen
            result = {
                "answer": final_answer,
                "sources": retrieved_docs,
                "performance": {
                    "total_time": total_response_time,
                    "retrieval_time": retrieval_time,
                    "llm_time": llm_generation_time,
                    "retrieval_method": retrieval_method,
                },
                "quality": {
                    "confidence_summary": confidence_summary,
                    "num_sources": len(retrieved_docs),
                    "quality_metrics": (
                        asdict(quality_metrics) if quality_metrics else None
                    ),
                },
                "metadata": {
                    "query_count": self.query_count,
                    "mode": mode,
                    "optimizations_enabled": OPTIMIZATIONS_AVAILABLE,
                },
            }

            return result

        except Exception as e:
            st.error(f"❌ Fehler bei der Fragebeantwortung: {e}")
            return {
                "error": str(e),
                "answer": "Entschuldigung, es ist ein Fehler aufgetreten.",
                "sources": [],
                "performance": {"total_time": time.time() - start_time},
            }

    def _get_optimal_k(self, mode: str) -> int:
        """Bestimmt optimales k basierend auf Modus"""

        k_map = {"quick": 3, "balanced": 4, "comprehensive": 6, "expert": 8}

        return k_map.get(mode, 4)

    def _select_optimal_prompt(self, mode: str) -> PromptTemplate:
        """Wählt optimalen Prompt basierend auf Modus"""

        if mode == "quick":
            template = """Kurze, präzise Antwort auf die juristische Frage basierend auf dem Kontext:

{context}

Frage: {question}

Kurze Antwort:"""

        elif mode == "comprehensive":
            template = """Du bist ein hochqualifizierter Rechtsexperte. Gib eine umfassende, strukturierte Antwort.

**ANTWORTSTRUKTUR:**
1. **Kurze Antwort**: [Direkter Bezug zur Frage]
2. **Rechtliche Grundlagen**: [Relevante Paragraphen/Artikel]
3. **Detaillierte Erläuterung**: [Ausführliche Analyse]
4. **Praxishinweise**: [Wenn aus Kontext ableitbar]

Kontext aus Lehrbuch:
{context}

Frage: {question}

Strukturierte Expertenantwort:"""

        elif mode == "expert":
            template = """Du bist ein führender Rechtsexperte. Gib eine tiefgreifende, wissenschaftliche Analyse.

**WISSENSCHAFTLICHE ANALYSE:**
1. **Problemstellung**: [Rechtliche Einordnung]
2. **Normative Grundlagen**: [Gesetzliche Basis mit Fundstellen]
3. **Systematische Einordnung**: [Rechtsgebiet und Zusammenhänge]
4. **Dogmatische Analyse**: [Theoretische Durchdringung]
5. **Praktische Konsequenzen**: [Anwendung und Rechtsfolgen]
6. **Kritische Würdigung**: [Probleme und offene Fragen]

Kontext aus juristischer Literatur:
{context}

Rechtsfrage: {question}

Wissenschaftliche Analyse:"""

        else:  # balanced
            template = """Du bist ein juristischer Experte. Beantworte die Frage präzise und strukturiert basierend auf dem Kontext.

Kontext:
{context}

Frage: {question}

Antwort (strukturiert mit Rechtsnormen):"""

        return PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

    def _build_enhanced_context(self, retrieved_docs: List[Dict]) -> str:
        """Baut erweiterten Context mit Metadaten"""

        context_parts = []

        for i, doc in enumerate(retrieved_docs):
            # Metadaten extrahieren
            metadata = doc.get("metadata", {})
            source_info = f"[Quelle {i+1}: {metadata.get('source', 'N/A')}, S. {metadata.get('page', 'N/A')}]"

            # Relevanz-Indikator
            relevance_score = doc.get("relevance_score", doc.get("final_score", 0))
            if relevance_score > 0.8:
                relevance_indicator = "🟢 Sehr relevant"
            elif relevance_score > 0.6:
                relevance_indicator = "🔵 Relevant"
            elif relevance_score > 0.4:
                relevance_indicator = "🟡 Teilweise relevant"
            else:
                relevance_indicator = "🟠 Zusatzinfo"

            # Content mit Indicators
            content = doc["content"]

            # Juristische Relevanz markieren
            if (
                doc.get("legal_relevance")
                or doc.get("retrieval_method") == "legal_entity"
            ):
                content = f"⚖️ {content}"

            context_part = f"{source_info}\n{relevance_indicator}\n{content}"
            context_parts.append(context_part)

        return "\n\n---\n\n".join(context_parts)

    def _record_performance_metrics(
        self,
        question: str,
        total_time: float,
        retrieval_time: float,
        llm_time: float,
        retrieved_docs: List[Dict],
        quality_metrics,
    ):
        """Zeichnet Performance-Metriken auf"""

        # Cache-Hit-Rate berechnen (vereinfacht)
        cache_hit_rate = 0.5  # Placeholder - würde aus tatsächlichem Cache kommen

        # Qualitäts-Score aus retrieved docs
        quality_scores = [doc.get("relevance_score", 0.5) for doc in retrieved_docs]
        avg_quality = (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
        )

        # Confidence aus Quality-Metrics
        confidence = quality_metrics.confidence_score if quality_metrics else 0.7

        metrics = PerformanceMetrics(
            timestamp=time.time(),
            query_processing_time=total_time - retrieval_time - llm_time,
            retrieval_time=retrieval_time,
            llm_generation_time=llm_time,
            total_response_time=total_time,
            cache_hit_rate=cache_hit_rate,
            memory_usage_mb=100.0,  # Placeholder
            retrieval_quality_score=avg_quality,
            answer_confidence=confidence,
            user_satisfaction=0.8,  # Placeholder
            tokens_processed=len(question.split()) * 10,  # Grober Schätzwert
            documents_retrieved=len(retrieved_docs),
        )

        self.performance_monitor.record_metrics(metrics)

    def get_system_status(self) -> Dict:
        """Gibt aktuellen System-Status zurück"""

        status = {
            "initialized": self.vectorstore is not None,
            "optimizations_available": OPTIMIZATIONS_AVAILABLE,
            "query_count": self.query_count,
            "avg_response_time": (
                self.total_response_time / self.query_count
                if self.query_count > 0
                else 0
            ),
            "enabled_features": {
                "multi_stage_retrieval": self.multi_stage_retriever is not None,
                "confidence_scoring": self.confidence_calculator is not None,
                "performance_monitoring": self.performance_monitor is not None,
                "auto_optimization": self.auto_optimizer is not None,
            },
        }

        # Performance-Details hinzufügen
        if self.performance_monitor:
            performance_summary = (
                self.performance_monitor.get_current_performance_summary()
            )
            status["performance"] = performance_summary

            active_alerts = self.performance_monitor.get_active_alerts()
            status["active_alerts"] = len(active_alerts)
            status["system_health"] = (
                "healthy" if len(active_alerts) == 0 else "needs_attention"
            )

        return status

    def get_comprehensive_report(self) -> Optional[Dict]:
        """Generiert umfassenden System-Report"""

        if not self.performance_reporter:
            return None

        return self.performance_reporter.generate_comprehensive_report()


# === STREAMLIT ULTRA-INTERFACE ===


@st.cache_resource
def load_ultra_system():
    """Lädt das ultra-optimierte System mit Caching"""
    system = UltraOptimizedJuraRAG()
    return system


def create_ultra_sidebar():
    """Erstellt erweiterte Sidebar mit allen Optionen"""

    st.sidebar.markdown("### 🚀 Ultra-Optimierungen")

    # Modus-Auswahl
    mode = st.sidebar.selectbox(
        "Antwort-Modus",
        ["quick", "balanced", "comprehensive", "expert"],
        index=1,
        help="Quick=Schnell, Balanced=Ausgewogen, Comprehensive=Umfassend, Expert=Wissenschaftlich",
    )

    # Feature-Toggles
    st.sidebar.markdown("### 🎛️ Feature-Kontrolle")

    use_multi_stage = st.sidebar.checkbox(
        "Multi-Stage-Retrieval",
        value=ULTRA_CONFIG["enable_multi_stage_retrieval"],
        help="Verwendet erweiterte Retrieval-Strategien für bessere Ergebnisse",
    )

    use_confidence = st.sidebar.checkbox(
        "Confidence-Scoring",
        value=ULTRA_CONFIG["enable_confidence_scoring"],
        help="Bewertet Vertrauenswürdigkeit der Antworten",
    )

    use_structured = st.sidebar.checkbox(
        "Strukturierte Ausgabe", value=True, help="Formatiert Antworten mit Templates"
    )

    use_monitoring = st.sidebar.checkbox(
        "Performance-Monitoring",
        value=ULTRA_CONFIG["enable_performance_monitoring"],
        help="Überwacht System-Performance in Echtzeit",
    )

    return {
        "mode": mode,
        "use_multi_stage": use_multi_stage,
        "use_confidence": use_confidence,
        "use_structured": use_structured,
        "use_monitoring": use_monitoring,
    }


def display_ultra_metrics(system: UltraOptimizedJuraRAG):
    """Zeigt erweiterte System-Metriken an"""

    status = system.get_system_status()

    with st.sidebar.expander("📊 System-Status"):

        # Basis-Status
        st.metric("Anfragen verarbeitet", status["query_count"])
        st.metric("Ø Antwortzeit", f"{status['avg_response_time']:.2f}s")

        # Feature-Status
        st.write("**Aktivierte Features:**")
        for feature, enabled in status["enabled_features"].items():
            icon = "✅" if enabled else "❌"
            st.write(f"{icon} {feature.replace('_', ' ').title()}")

        # Performance-Details
        if "performance" in status:
            perf = status["performance"]
            st.write("**Performance:**")
            st.write(f"🎯 Status: {perf.get('status', 'unknown')}")

            if "current_metrics" in perf:
                metrics = perf["current_metrics"]
                st.write(f"⚡ Antwortzeit: {metrics.get('avg_response_time', 0):.2f}s")
                st.write(f"💾 Cache-Rate: {metrics.get('avg_cache_hit_rate', 0):.1%}")
                st.write(f"🎯 Qualität: {metrics.get('avg_quality_score', 0):.1%}")

        # Alerts
        if status.get("active_alerts", 0) > 0:
            st.warning(f"⚠️ {status['active_alerts']} aktive Alerts")


def display_ultra_results(result: Dict):
    """Zeigt ultra-optimierte Ergebnisse an"""

    if "error" in result:
        st.error(f"❌ Fehler: {result['error']}")
        return

    # Hauptantwort
    st.markdown("### 📖 Antwort:")
    st.markdown(result["answer"])

    # Performance-Metriken
    perf = result.get("performance", {})
    quality = result.get("quality", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Antwortzeit", f"{perf.get('total_time', 0):.2f}s")
    with col2:
        st.metric("Retrieval", f"{perf.get('retrieval_time', 0):.2f}s")
    with col3:
        st.metric("LLM-Zeit", f"{perf.get('llm_time', 0):.2f}s")
    with col4:
        confidence = quality.get("confidence_summary", {}).get(
            "overall_confidence", "N/A"
        )
        st.metric("Vertrauen", confidence)

    # Erweiterte Informationen
    with st.expander("🔍 Detaillierte Analyse"):

        # Quality-Metriken
        if quality.get("quality_metrics"):
            st.markdown("#### Qualitäts-Metriken")
            metrics = quality["quality_metrics"]

            col1, col2 = st.columns(2)
            with col1:
                st.write(
                    f"**Confidence-Level:** {metrics.get('confidence_level', 'N/A')}"
                )
                st.write(
                    f"**Quellenqualität:** {metrics.get('source_reliability', 0):.1%}"
                )
                st.write(
                    f"**Vollständigkeit:** {metrics.get('completeness_score', 0):.1%}"
                )

            with col2:
                st.write(
                    f"**Juristische Genauigkeit:** {metrics.get('legal_accuracy', 0):.1%}"
                )
                st.write(f"**Klarheit:** {metrics.get('clarity_score', 0):.1%}")
                st.write(
                    f"**Zitationsqualität:** {metrics.get('citation_quality', 0):.1%}"
                )

        # Performance-Details
        st.markdown("#### Performance-Details")
        st.write(f"**Retrieval-Methode:** {perf.get('retrieval_method', 'N/A')}")
        st.write(f"**Anzahl Quellen:** {quality.get('num_sources', 0)}")

        metadata = result.get("metadata", {})
        st.write(f"**Query #{metadata.get('query_count', 0)}**")
        st.write(
            f"**Optimierungen aktiv:** {'✅' if metadata.get('optimizations_enabled') else '❌'}"
        )

    # Quellen
    sources = result.get("sources", [])
    if sources:
        with st.expander("📚 Verwendete Quellen"):
            for i, source in enumerate(sources[:5], 1):  # Max 5 Quellen anzeigen

                st.markdown(f"**Quelle {i}:**")

                # Metadaten
                metadata = source.get("metadata", {})
                st.write(
                    f"📄 {metadata.get('source', 'Unbekannt')}, Seite {metadata.get('page', 'N/A')}"
                )

                # Retrieval-Info
                method = source.get("retrieval_method", "standard")
                relevance = source.get("relevance_score", source.get("final_score", 0))

                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"🔍 Methode: {method}")
                with col2:
                    if relevance:
                        st.write(f"🎯 Relevanz: {relevance:.1%}")

                # Content-Preview
                content = source.get("content", "")
                preview = content[:200] + "..." if len(content) > 200 else content
                st.info(preview)

                st.markdown("---")


def main_ultra():
    """Hauptfunktion der Ultra-App"""

    st.title("🚀 Ultra-Optimierte Juristische Wissensdatenbank")
    st.markdown("*Maximale Performance durch intelligente Optimierungen*")

    # System laden
    system = load_ultra_system()

    if not system.get_system_status()["initialized"]:
        st.error("❌ System konnte nicht initialisiert werden!")
        st.info("💡 Stellen Sie sicher, dass die Datenbank existiert (faiss_db/)")
        return

    # Sidebar-Konfiguration
    config = create_ultra_sidebar()

    # System-Metriken anzeigen
    display_ultra_metrics(system)

    # Hauptinterface
    st.markdown("### 💬 Ultra-Intelligente Fragenbearbeitung")

    question = st.text_area(
        "Ihre juristische Frage:",
        placeholder="z.B. Welche Voraussetzungen müssen für eine Geschäftsführung ohne Auftrag erfüllt sein?",
        height=120,
        help="Stellen Sie präzise juristische Fragen für die besten Ergebnisse",
    )

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button("🚀 Ultra-Antwort generieren", type="primary"):
            if question:
                with st.spinner(
                    f"Verarbeite mit {config['mode']}-Modus und Ultra-Optimierungen..."
                ):

                    result = system.ultra_answer_question(
                        question=question,
                        mode=config["mode"],
                        use_confidence_scoring=config["use_confidence"],
                        use_structured_output=config["use_structured"],
                    )

                    display_ultra_results(result)
            else:
                st.warning("⚠️ Bitte geben Sie eine Frage ein.")

    with col2:
        if st.button("📊 System-Report"):
            report = system.get_comprehensive_report()
            if report:
                st.json(report)
            else:
                st.info("Performance-Monitoring nicht verfügbar")

    with col3:
        if st.button("🔧 System-Status"):
            status = system.get_system_status()
            st.json(status)

    # Footer mit aktueller Konfiguration
    st.markdown("---")
    st.markdown("### ⚙️ Aktuelle Konfiguration")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(f"**Modus:** {config['mode']}")
        st.write(f"**Multi-Stage:** {'✅' if config['use_multi_stage'] else '❌'}")

    with col2:
        st.write(f"**Confidence:** {'✅' if config['use_confidence'] else '❌'}")
        st.write(f"**Strukturiert:** {'✅' if config['use_structured'] else '❌'}")

    with col3:
        st.write(f"**Monitoring:** {'✅' if config['use_monitoring'] else '❌'}")
        st.write(f"**Optimierungen:** {'✅' if OPTIMIZATIONS_AVAILABLE else '❌'}")


if __name__ == "__main__":
    # Import-Fix für dataclasses.asdict
    try:
        from dataclasses import asdict
    except ImportError:

        def asdict(instance):
            return instance.__dict__

    main_ultra()
