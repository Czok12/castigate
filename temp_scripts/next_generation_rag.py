"""
🚀 NEXT-GENERATION RAG SYSTEM
=============================

Integration aller neuen Optimierungsmodule für maximale Performance
"""

import os
import time
from typing import Dict, List

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama.llms import OllamaLLM

# Import der neuen Optimierungsmodule
try:
    from intelligent_caching_system import HierarchicalCacheManager
    from semantic_query_enhancer import SemanticQueryEnhancer

    NEXT_GEN_MODULES_AVAILABLE = True
except ImportError as e:
    st.warning(f"Next-Generation Module nicht verfügbar: {e}")
    NEXT_GEN_MODULES_AVAILABLE = False

# Import der bestehenden Optimierungen
try:
    from advanced_retrieval_engine import MultiStageRetriever
    from answer_quality_system import ConfidenceCalculator, StructuredAnswerGenerator
    from performance_monitoring import (
        AutoOptimizer,
        PerformanceMonitor,
        PerformanceReporter,
    )

    EXISTING_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    st.warning(f"Bestehende Optimierungen nicht verfügbar: {e}")
    EXISTING_OPTIMIZATIONS_AVAILABLE = False

# === KONFIGURATION ===
DB_PATH = "faiss_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

NEXT_GEN_CONFIG = {
    # Caching
    "enable_intelligent_caching": True,
    "cache_semantic_similarity": True,
    "cache_ttl_adaptive": True,
    # Query Enhancement
    "enable_query_enhancement": True,
    "enable_synonym_expansion": True,
    "enable_intent_detection": True,
    "enable_contextual_expansion": True,
    # Advanced Retrieval
    "enable_multi_stage_retrieval": True,
    "retrieval_stages": ["semantic", "keyword", "legal_entity", "rerank"],
    # Quality & Performance
    "enable_confidence_scoring": True,
    "enable_performance_monitoring": True,
    "enable_auto_optimization": True,
    # Antwort-Optimierung
    "answer_template": "legal_analysis",
    "enable_structured_output": True,
    # System-Optimierung
    "monitoring_window": 50,
    "optimization_interval": 10,
    "cache_max_entries": 1000,
}


class NextGenerationRAG:
    """Next-Generation RAG-Engine mit allen verfügbaren Optimierungen"""

    def __init__(self):
        self.vectorstore = None
        self.embedding_model = None
        self.llm = None

        # Next-Generation Module
        self.cache_manager = None
        self.query_enhancer = None

        # Bestehende Optimierungskomponenten
        self.multi_stage_retriever = None
        self.confidence_calculator = None
        self.answer_generator = None
        self.performance_monitor = None
        self.auto_optimizer = None
        self.performance_reporter = None

        # Statistiken
        self.query_count = 0
        self.total_response_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0

        # Initialisierung
        self._initialize_system()

    def _initialize_system(self):
        """Initialisiert das Next-Generation System"""

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

            # Next-Generation Module initialisieren
            if NEXT_GEN_MODULES_AVAILABLE:
                self._initialize_next_gen_modules()

            # Bestehende Optimierungen initialisieren
            if EXISTING_OPTIMIZATIONS_AVAILABLE:
                self._initialize_existing_optimizations()

            st.success("✅ Next-Generation System erfolgreich initialisiert!")
            return True

        except Exception as e:
            st.error(f"❌ Fehler bei der Initialisierung: {e}")
            return False

    def _initialize_next_gen_modules(self):
        """Initialisiert Next-Generation Module"""

        # Intelligentes Cache-System
        if NEXT_GEN_CONFIG["enable_intelligent_caching"]:
            self.cache_manager = HierarchicalCacheManager()
            st.info("🧠 Intelligentes Cache-System aktiviert")

        # Semantic Query Enhancement
        if NEXT_GEN_CONFIG["enable_query_enhancement"]:
            self.query_enhancer = SemanticQueryEnhancer()
            st.info("🔍 Semantic Query Enhancement aktiviert")

    def _initialize_existing_optimizations(self):
        """Initialisiert bestehende Optimierungskomponenten"""

        # Multi-Stage Retriever
        if NEXT_GEN_CONFIG["enable_multi_stage_retrieval"]:
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
        if NEXT_GEN_CONFIG["enable_confidence_scoring"]:
            self.confidence_calculator = ConfidenceCalculator()
            self.answer_generator = StructuredAnswerGenerator()

        # Performance Monitoring
        if NEXT_GEN_CONFIG["enable_performance_monitoring"]:
            self.performance_monitor = PerformanceMonitor(
                window_size=NEXT_GEN_CONFIG["monitoring_window"]
            )

            if NEXT_GEN_CONFIG["enable_auto_optimization"]:
                self.auto_optimizer = AutoOptimizer(self.performance_monitor)
                self.performance_reporter = PerformanceReporter(
                    self.performance_monitor, self.auto_optimizer
                )

    def next_gen_answer_question(
        self,
        question: str,
        mode: str = "balanced",
        use_caching: bool = True,
        use_query_enhancement: bool = True,
        use_confidence_scoring: bool = True,
        use_structured_output: bool = True,
    ) -> Dict:
        """Next-Generation Fragebeantwortung mit allen Features"""

        start_time = time.time()
        self.query_count += 1

        try:
            # Phase 1: Query Enhancement
            enhanced_query_data = None
            final_query = question

            if use_query_enhancement and self.query_enhancer:
                enhanced_query_data = self.query_enhancer.enhance_query(
                    question,
                    enable_synonyms=NEXT_GEN_CONFIG["enable_synonym_expansion"],
                    enable_intent_expansion=NEXT_GEN_CONFIG[
                        "enable_contextual_expansion"
                    ],
                    enable_entity_extraction=True,
                )
                final_query = enhanced_query_data.enhanced_query
                query_enhancement_time = enhanced_query_data.processing_time
            else:
                query_enhancement_time = 0

            # Phase 2: Cache-Lookup
            cache_result = None
            if use_caching and self.cache_manager:
                cache_result = self.cache_manager.get_cached_answer(
                    final_query,
                    mode,
                    use_semantic_matching=NEXT_GEN_CONFIG["cache_semantic_similarity"],
                )

                if cache_result:
                    self.cache_hits += 1
                    total_time = time.time() - start_time

                    return {
                        "answer": cache_result["answer"],
                        "sources": cache_result["sources"],
                        "confidence_score": cache_result.get("confidence_score", 0.8),
                        "retrieval_method": f"cache_{cache_result.get('cache_level', 'unknown')}",
                        "processing_time": total_time,
                        "query_enhancement": (
                            enhanced_query_data.__dict__
                            if enhanced_query_data
                            else None
                        ),
                        "cache_hit": True,
                        "cache_level": cache_result.get("cache_level"),
                        "semantic_similarity": cache_result.get("semantic_similarity"),
                        "performance_metrics": {
                            "query_enhancement_time": query_enhancement_time,
                            "cache_lookup_time": time.time()
                            - start_time
                            - query_enhancement_time,
                            "total_time": total_time,
                        },
                    }
                else:
                    self.cache_misses += 1

            # Phase 3: Erweiterte Retrieval
            retrieval_start_time = time.time()

            if (
                self.multi_stage_retriever
                and NEXT_GEN_CONFIG["enable_multi_stage_retrieval"]
            ):
                retrieved_docs = self.multi_stage_retriever.multi_stage_retrieve(
                    final_query,
                    k=self._get_optimal_k(mode),
                    stages=NEXT_GEN_CONFIG["retrieval_stages"],
                )
                retrieval_method = "next_gen_multi_stage"
            else:
                # Fallback: Standard Retrieval
                docs = self.vectorstore.similarity_search(final_query, k=4)
                retrieved_docs = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "retrieval_method": "standard",
                    }
                    for doc in docs
                ]
                retrieval_method = "standard_fallback"

            retrieval_time = time.time() - retrieval_start_time

            # Phase 4: Context Building mit Enhancement
            context = self._build_next_gen_context(retrieved_docs, enhanced_query_data)

            # Phase 5: LLM Generation
            llm_start_time = time.time()
            prompt = self._select_optimal_prompt(mode)

            rag_chain = (
                {"context": lambda x: context, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )

            raw_answer = rag_chain.invoke(final_query)
            llm_generation_time = time.time() - llm_start_time

            # Phase 6: Antwort-Optimierung & Confidence-Scoring
            confidence_score = 0.7  # Default
            structured_answer = raw_answer

            if use_confidence_scoring and self.confidence_calculator:
                confidence_result = self.confidence_calculator.calculate_confidence(
                    final_query, raw_answer, retrieved_docs, "balanced"
                )
                confidence_score = confidence_result["overall_confidence"]

                if use_structured_output and self.answer_generator:
                    structured_result = (
                        self.answer_generator.generate_structured_answer(
                            final_query,
                            raw_answer,
                            retrieved_docs,
                            NEXT_GEN_CONFIG["answer_template"],
                        )
                    )
                    structured_answer = structured_result["structured_answer"]

            total_time = time.time() - start_time

            # Performance-Metriken zusammenstellen
            performance_metrics = {
                "query_enhancement_time": query_enhancement_time,
                "retrieval_time": retrieval_time,
                "llm_generation_time": llm_generation_time,
                "total_time": total_time,
                "cache_lookup_time": (
                    0
                    if cache_result
                    else time.time()
                    - start_time
                    - query_enhancement_time
                    - retrieval_time
                    - llm_generation_time
                ),
            }

            # Phase 7: Caching der Antwort
            if use_caching and self.cache_manager:
                cache_info = self.cache_manager.cache_answer(
                    question=final_query,
                    answer=structured_answer,
                    sources=retrieved_docs,
                    confidence_score=confidence_score,
                    retrieval_method=retrieval_method,
                    performance_metrics=performance_metrics,
                    mode=mode,
                )
            else:
                cache_info = None

            # Performance-Monitoring
            if self.performance_monitor:
                self.performance_monitor.record_query(
                    final_query, total_time, confidence_score, len(retrieved_docs)
                )

            # Finale Antwort zusammenstellen
            result = {
                "answer": structured_answer,
                "sources": retrieved_docs,
                "confidence_score": confidence_score,
                "retrieval_method": retrieval_method,
                "processing_time": total_time,
                "query_enhancement": (
                    enhanced_query_data.__dict__ if enhanced_query_data else None
                ),
                "cache_hit": False,
                "cache_info": cache_info,
                "performance_metrics": performance_metrics,
            }

            return result

        except Exception as e:
            st.error(f"Fehler bei der Fragebeantwortung: {e}")
            return {
                "answer": f"Entschuldigung, es ist ein Fehler aufgetreten: {e}",
                "sources": [],
                "confidence_score": 0.0,
                "retrieval_method": "error",
                "processing_time": time.time() - start_time,
                "error": str(e),
            }

    def _get_optimal_k(self, mode: str) -> int:
        """Bestimmt optimale Anzahl Retrieval-Ergebnisse"""

        mode_mapping = {"quick": 3, "balanced": 5, "comprehensive": 8, "expert": 10}

        return mode_mapping.get(mode, 5)

    def _build_next_gen_context(
        self, retrieved_docs: List[Dict], enhanced_query_data
    ) -> str:
        """Erstellt erweiterten Kontext mit Query-Enhancement-Informationen"""

        contexts = []

        for i, doc in enumerate(retrieved_docs, 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            retrieval_method = doc.get("retrieval_method", "unknown")
            score = doc.get("score", 0.0)

            # Enhanced Context mit mehr Metadaten
            context_header = (
                f"📄 **Quelle {i}** (Score: {score:.3f}, Methode: {retrieval_method})"
            )

            if metadata:
                if "page" in metadata:
                    context_header += f" - Seite {metadata['page']}"
                if "source" in metadata:
                    context_header += f" - {metadata['source']}"

            context_block = f"{context_header}\n{content}\n"
            contexts.append(context_block)

        # Query-Enhancement-Informationen hinzufügen
        enhancement_context = ""
        if enhanced_query_data:
            enhancement_info = []
            if enhanced_query_data.intent != "general":
                enhancement_info.append(f"Intent: {enhanced_query_data.intent}")
            if enhanced_query_data.legal_entities:
                enhancement_info.append(
                    f"Juristische Entitäten: {', '.join(enhanced_query_data.legal_entities)}"
                )
            if enhanced_query_data.synonyms:
                enhancement_info.append(
                    f"Erkannte Synonyme: {', '.join(enhanced_query_data.synonyms[:3])}"
                )

            if enhancement_info:
                enhancement_context = (
                    f"\n🧠 **Query-Enhancement**: {' | '.join(enhancement_info)}\n\n"
                )

        return enhancement_context + "\n---\n".join(contexts)

    def _select_optimal_prompt(self, mode: str) -> PromptTemplate:
        """Wählt optimales Prompt basierend auf Modus"""

        prompts = {
            "quick": PromptTemplate(
                input_variables=["context", "question"],
                template="""Basierend auf dem folgenden Kontext, beantworte die Frage präzise und knapp:

Kontext:
{context}

Frage: {question}

Kurze Antwort:""",
            ),
            "balanced": PromptTemplate(
                input_variables=["context", "question"],
                template="""Als Experte für deutsches Recht, analysiere den folgenden Kontext und beantworte die Frage strukturiert:

Kontext:
{context}

Frage: {question}

Bitte gib eine strukturierte Antwort mit folgenden Elementen:
1. Kurze Einordnung
2. Rechtliche Grundlagen
3. Praktische Anwendung
4. Relevante Paragraphen/Urteile

Antwort:""",
            ),
            "comprehensive": PromptTemplate(
                input_variables=["context", "question"],
                template="""Als erfahrener Jurist, erstelle eine umfassende Analyse basierend auf dem Kontext:

Kontext:
{context}

Frage: {question}

Erstelle eine detaillierte juristische Analyse mit:

## Rechtslage
- Gesetzliche Grundlagen
- Relevante Normen und Paragraphen

## Rechtsprechung
- Wichtige Urteile und Entscheidungen
- Entwicklung der Rechtsprechung

## Praktische Anwendung
- Anwendbarkeit im konkreten Fall
- Praxishinweise und Besonderheiten

## Zusammenfassung
- Kernaussagen und Handlungsempfehlungen

Antwort:""",
            ),
            "expert": PromptTemplate(
                input_variables=["context", "question"],
                template="""Als Rechtsexperte mit Spezialisierung auf deutsches Recht, führe eine tiefgreifende Analyse durch:

Kontext:
{context}

Frage: {question}

Führe eine Expertenanalyse durch mit:

## 1. Rechtsdogmatische Einordnung
- Rechtsgebiet und systematische Stellung
- Begriffsdefinitionen und Abgrenzungen

## 2. Normative Grundlagen
- Einschlägige Gesetze und Verordnungen
- Verfassungsrechtlicher Hintergrund

## 3. Rechtsprechungsanalyse
- Höchstrichterliche Rechtsprechung
- Entwicklungslinien und Trends
- Streitpunkte und offene Fragen

## 4. Literaturmeinungen
- Herrschende Lehre
- Mindermeinungen und Kritik

## 5. Praktische Relevanz
- Anwendungsbereiche
- Verfahrensrechtliche Aspekte
- Gestaltungshinweise

## 6. Fazit und Ausblick
- Zusammenfassende Bewertung
- Entwicklungstendenzen

Antwort:""",
            ),
        }

        return prompts.get(mode, prompts["balanced"])

    def get_next_gen_statistics(self) -> Dict:
        """Liefert umfassende Next-Generation-Statistiken"""

        stats = {
            "query_processing": {
                "total_queries": self.query_count,
                "avg_response_time": self.total_response_time
                / max(self.query_count, 1),
                "total_processing_time": self.total_response_time,
            },
            "caching": {
                "cache_enabled": self.cache_manager is not None,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": (
                    self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
                )
                * 100,
                "cache_statistics": (
                    self.cache_manager.get_cache_statistics()
                    if self.cache_manager
                    else None
                ),
            },
            "query_enhancement": {
                "enhancement_enabled": self.query_enhancer is not None,
                "enhancement_statistics": (
                    self.query_enhancer.get_enhancement_statistics()
                    if self.query_enhancer
                    else None
                ),
            },
            "system_status": {
                "next_gen_modules_available": NEXT_GEN_MODULES_AVAILABLE,
                "existing_optimizations_available": EXISTING_OPTIMIZATIONS_AVAILABLE,
                "active_features": self._get_active_features(),
            },
        }

        # Performance-Monitor-Statistiken hinzufügen
        if self.performance_monitor:
            stats["performance_monitoring"] = (
                self.performance_monitor.get_current_metrics()
            )

        return stats

    def _get_active_features(self) -> List[str]:
        """Liefert Liste aktiver Features"""

        active_features = []

        if self.cache_manager:
            active_features.append("Intelligent Caching")
        if self.query_enhancer:
            active_features.append("Semantic Query Enhancement")
        if self.multi_stage_retriever:
            active_features.append("Multi-Stage Retrieval")
        if self.confidence_calculator:
            active_features.append("Confidence Scoring")
        if self.answer_generator:
            active_features.append("Structured Answers")
        if self.performance_monitor:
            active_features.append("Performance Monitoring")
        if self.auto_optimizer:
            active_features.append("Auto Optimization")

        return active_features

    def clear_cache(self):
        """Leert den Cache"""
        if self.cache_manager:
            self.cache_manager.clear_cache()
            st.success("🗑️ Cache geleert")
        else:
            st.warning("Kein Cache-System aktiv")

    def get_query_suggestions(self, query: str) -> List[str]:
        """Liefert Query-Verbesserungsvorschläge"""
        if self.query_enhancer:
            return self.query_enhancer.get_query_suggestions(query)
        else:
            return []

    def analyze_query_quality(self, query: str) -> Dict:
        """Analysiert Query-Qualität"""
        if self.query_enhancer:
            return self.query_enhancer.analyze_query_quality(query)
        else:
            return {"status": "query_enhancement_not_available"}


# === STREAMLIT UI ===


def main_next_generation():
    """Haupt-Streamlit-App für Next-Generation RAG"""

    st.set_page_config(
        page_title="🚀 Next-Generation Juristische Wissensdatenbank",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🚀 Next-Generation Juristische Wissensdatenbank")
    st.markdown("**Intelligente Fragebeantwortung mit modernsten KI-Optimierungen**")

    # System initialisieren
    if "next_gen_system" not in st.session_state:
        with st.spinner("Initialisiere Next-Generation System..."):
            st.session_state.next_gen_system = NextGenerationRAG()

    system = st.session_state.next_gen_system

    # Sidebar-Konfiguration
    st.sidebar.header("🎛️ Next-Gen Konfiguration")

    # Erweiterte Konfiguration
    config = {
        "mode": st.sidebar.selectbox(
            "🎯 Antwort-Modus",
            ["quick", "balanced", "comprehensive", "expert"],
            index=1,
            help="Quick: Schnelle Antworten | Balanced: Ausgewogen | Comprehensive: Detailliert | Expert: Umfassende Analyse",
        ),
        "use_caching": st.sidebar.checkbox(
            "🧠 Intelligentes Caching",
            value=True,
            help="Verwendet hierarchisches Caching mit Semantic Similarity",
        ),
        "use_query_enhancement": st.sidebar.checkbox(
            "🔍 Query Enhancement",
            value=True,
            help="Erweitert Queries um Synonyme und juristische Fachbegriffe",
        ),
        "use_multi_stage": st.sidebar.checkbox(
            "🎯 Multi-Stage Retrieval",
            value=True,
            help="Mehrstufiges intelligentes Retrieval-System",
        ),
        "use_confidence": st.sidebar.checkbox(
            "📊 Confidence Scoring",
            value=True,
            help="Bewertet Antwortqualität und Verlässlichkeit",
        ),
        "use_structured": st.sidebar.checkbox(
            "📝 Strukturierte Antworten",
            value=True,
            help="Formatiert Antworten nach juristischen Templates",
        ),
        "use_monitoring": st.sidebar.checkbox(
            "📈 Performance Monitoring",
            value=True,
            help="Überwacht System-Performance in Echtzeit",
        ),
    }

    # Haupt-Interface
    st.markdown("### 💬 Stellen Sie Ihre juristische Frage")

    question = st.text_area(
        "Frage:",
        placeholder="z.B. Was sind die Voraussetzungen für Geschäftsführung ohne Auftrag nach § 677 BGB?",
        height=120,
        help="Nutzen Sie spezifische juristische Begriffe und Paragraphen für beste Ergebnisse",
    )

    # Query-Qualitätsanalyse
    if question and len(question.strip()) > 5:
        if st.button("🔍 Query analysieren"):
            quality_analysis = system.analyze_query_quality(question)
            if "overall_quality" in quality_analysis:
                st.subheader("🎯 Query-Qualitätsanalyse")

                col1, col2 = st.columns(2)

                with col1:
                    quality_score = quality_analysis["overall_quality"]
                    st.metric(
                        "Gesamt-Qualität",
                        f"{quality_score:.2f}",
                        delta=(
                            f"{quality_score-0.5:.2f}" if quality_score > 0.5 else None
                        ),
                    )

                    st.write(
                        "**Erkannter Intent:**",
                        quality_analysis.get("detected_intent", "Unbekannt"),
                    )
                    st.write(
                        "**Confidence:**",
                        f"{quality_analysis.get('confidence', 0):.2f}",
                    )

                with col2:
                    if quality_analysis.get("suggestions"):
                        st.write("**Verbesserungsvorschläge:**")
                        for suggestion in quality_analysis["suggestions"]:
                            st.write(f"• {suggestion}")

                if quality_analysis.get("improvements"):
                    st.write("**Optimierungstipps:**")
                    for improvement in quality_analysis["improvements"]:
                        st.info(improvement)

    # Haupt-Buttons
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

    with col1:
        if st.button(
            "🚀 Next-Gen Antwort generieren", type="primary", disabled=not question
        ):
            if question:
                with st.spinner(
                    f"🔄 Verarbeite mit {config['mode']}-Modus und Next-Gen Optimierungen..."
                ):

                    result = system.next_gen_answer_question(
                        question=question,
                        mode=config["mode"],
                        use_caching=config["use_caching"],
                        use_query_enhancement=config["use_query_enhancement"],
                        use_confidence_scoring=config["use_confidence"],
                        use_structured_output=config["use_structured"],
                    )

                    # Ergebnisse anzeigen
                    display_next_gen_results(result)

    with col2:
        if st.button("📊 System-Status"):
            display_system_status(system)

    with col3:
        if st.button("📈 Statistiken"):
            display_statistics(system)

    with col4:
        if st.button("🗑️ Cache leeren"):
            system.clear_cache()

    # Footer
    st.markdown("---")
    st.markdown(
        """
    ### 🚀 Next-Generation Features
    
    **Intelligentes Caching:** Hierarchisches Caching mit Semantic Similarity und adaptiver TTL
    
    **Query Enhancement:** Automatische Erweiterung um Synonyme, Intent-Detection und juristische Entitäten
    
    **Multi-Stage Retrieval:** Kombination aus Semantic Search, Keyword-Matching und Entity-Recognition
    
    **Performance Monitoring:** Echtzeit-Überwachung und automatische Optimierung
    """
    )


def display_next_gen_results(result: Dict):
    """Zeigt Next-Generation Ergebnisse an"""

    # Haupt-Antwort
    st.subheader("💬 Antwort")
    st.write(result["answer"])

    # Performance-Metriken prominent anzeigen
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        processing_time = result.get("processing_time", 0)
        st.metric("⏱️ Verarbeitungszeit", f"{processing_time:.2f}s")

    with col2:
        confidence = result.get("confidence_score", 0)
        confidence_label = (
            "Sehr Hoch"
            if confidence > 0.8
            else (
                "Hoch"
                if confidence > 0.6
                else "Mittel" if confidence > 0.4 else "Niedrig"
            )
        )
        st.metric("🎯 Confidence", confidence_label, f"{confidence:.2f}")

    with col3:
        cache_status = "✅ Cache Hit" if result.get("cache_hit") else "❌ Cache Miss"
        st.metric("🧠 Cache", cache_status)

    with col4:
        method = result.get("retrieval_method", "unknown")
        st.metric("🔍 Methode", method.replace("_", " ").title())

    # Erweiterte Metriken in Expander
    with st.expander("📊 Detaillierte Performance-Metriken"):
        performance = result.get("performance_metrics", {})

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Timing-Breakdown:**")
            st.write(
                f"• Query Enhancement: {performance.get('query_enhancement_time', 0):.3f}s"
            )
            st.write(f"• Retrieval: {performance.get('retrieval_time', 0):.3f}s")
            st.write(
                f"• LLM Generation: {performance.get('llm_generation_time', 0):.3f}s"
            )
            st.write(f"• Cache Lookup: {performance.get('cache_lookup_time', 0):.3f}s")

        with col2:
            if result.get("cache_info"):
                cache_info = result["cache_info"]
                st.write("**Cache-Informationen:**")
                st.write(f"• Level: {cache_info.get('cache_level', 'N/A')}")
                st.write(f"• TTL: {cache_info.get('ttl', 0):.0f}s")

    # Query Enhancement Details
    if result.get("query_enhancement"):
        with st.expander("🔍 Query Enhancement Details"):
            enhancement = result["query_enhancement"]

            col1, col2 = st.columns(2)

            with col1:
                st.write(
                    f"**Original Query:** {enhancement.get('original_query', 'N/A')}"
                )
                st.write(
                    f"**Enhanced Query:** {enhancement.get('enhanced_query', 'N/A')}"
                )
                st.write(
                    f"**Intent:** {enhancement.get('intent', 'N/A')} ({enhancement.get('intent_confidence', 0):.2f})"
                )

            with col2:
                if enhancement.get("legal_entities"):
                    st.write("**Juristische Entitäten:**")
                    for entity in enhancement["legal_entities"]:
                        st.write(f"• {entity}")

                if enhancement.get("synonyms"):
                    st.write("**Gefundene Synonyme:**")
                    for synonym in enhancement["synonyms"]:
                        st.write(f"• {synonym}")

    # Quellen
    with st.expander("📚 Verwendete Quellen"):
        sources = result.get("sources", [])

        for i, source in enumerate(sources, 1):
            st.write(f"**Quelle {i}:**")

            # Metadaten
            metadata = source.get("metadata", {})
            if metadata:
                meta_info = []
                if "page" in metadata:
                    meta_info.append(f"Seite {metadata['page']}")
                if "source" in metadata:
                    meta_info.append(metadata["source"])
                if meta_info:
                    st.caption(" | ".join(meta_info))

            # Content (gekürzt)
            content = source.get("content", "")
            if len(content) > 500:
                st.write(content[:500] + "...")
            else:
                st.write(content)

            st.write("---")


def display_system_status(system):
    """Zeigt System-Status"""

    st.subheader("🔧 System-Status")

    features = system._get_active_features()

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Aktive Features:**")
        for feature in features:
            st.write(f"✅ {feature}")

    with col2:
        st.write("**Module-Verfügbarkeit:**")
        st.write(f"{'✅' if NEXT_GEN_MODULES_AVAILABLE else '❌'} Next-Gen Module")
        st.write(
            f"{'✅' if EXISTING_OPTIMIZATIONS_AVAILABLE else '❌'} Bestehende Optimierungen"
        )


def display_statistics(system):
    """Zeigt detaillierte Statistiken"""

    st.subheader("📈 System-Statistiken")

    stats = system.get_next_gen_statistics()

    # Query Processing
    st.write("**Query Processing:**")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Gesamt Queries", stats["query_processing"]["total_queries"])
    with col2:
        st.metric(
            "Ø Antwortzeit", f"{stats['query_processing']['avg_response_time']:.2f}s"
        )
    with col3:
        st.metric(
            "Gesamt-Zeit", f"{stats['query_processing']['total_processing_time']:.1f}s"
        )

    # Caching
    if stats["caching"]["cache_enabled"]:
        st.write("**Caching:**")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Cache Hits", stats["caching"]["cache_hits"])
        with col2:
            st.metric("Cache Misses", stats["caching"]["cache_misses"])
        with col3:
            st.metric("Hit Rate", f"{stats['caching']['cache_hit_rate']:.1f}%")

    # Query Enhancement
    if stats["query_enhancement"]["enhancement_enabled"]:
        st.write("**Query Enhancement:**")
        enhancement_stats = stats["query_enhancement"]["enhancement_statistics"]
        if enhancement_stats:
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Enhanced Queries", enhancement_stats.get("queries_enhanced", 0)
                )
                st.metric(
                    "Enhancement Rate",
                    f"{enhancement_stats.get('enhancement_rate', 0):.1f}%",
                )

            with col2:
                st.write("**Intent Distribution:**")
                intent_dist = enhancement_stats.get("intent_distribution", {})
                for intent, count in intent_dist.items():
                    st.write(f"• {intent}: {count}")


if __name__ == "__main__":
    main_next_generation()
