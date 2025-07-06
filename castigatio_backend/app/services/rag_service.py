"""RAG-Service für Castigatio: Retrieval, Re-Ranking und Antwortgenerierung auf Basis von Vektor-Datenbanken.

Unterstützt Explainable AI durch XAI-Logging.
Strikte Quellenbindung und Transparenz.
"""

import os
from datetime import datetime
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM

from app.core.config import DEVICE, EMBEDDING_MODEL, FAISS_DB_PATH, LLM_MODEL
from app.models.performance import PerformanceMetrics
from app.models.rag import QueryRequest, QueryResponse, SourceDocument

# Neue Services für Caching, Learning und Performance
from app.services.cache_service import cache_service
from app.services.learning_service import learning_service
from app.services.quality_service import quality_service
from app.services.query_enhancer_service import query_enhancer_service
from app.services.xai_service import xai_service


# Die MergedRetriever Klasse bleibt unverändert...
class MergedRetriever(Runnable):
    def __init__(
        self,
        book_ids: Optional[list[str]],
        embedding_model,
        context_size: int,
        k_multiplier: int,
    ):
        self.embedding_model = embedding_model
        self.book_ids = self._get_available_book_ids(book_ids)
        self.context_size = context_size
        self.k_multiplier = k_multiplier
        self.vectorstores = self._load_vectorstores()
        if not self.vectorstores:
            raise RuntimeError(
                "Keine gültigen Vektor-Datenbanken gefunden. Bitte den Ingestion-Prozess durchführen."
            )

    def _get_available_book_ids(self, requested_ids: Optional[list[str]]) -> list[str]:
        if not FAISS_DB_PATH.exists():
            return []
        all_dbs = {db_dir.name for db_dir in FAISS_DB_PATH.iterdir() if db_dir.is_dir()}
        if requested_ids:
            valid_ids = [book_id for book_id in requested_ids if book_id in all_dbs]
            if not valid_ids:
                import logging

                logging.warning(
                    "Keine der angefragten Buch-IDs %s wurde gefunden. Verfügbar sind: %s",
                    requested_ids,
                    all_dbs,
                )
            return valid_ids
        return list(all_dbs)

    def _load_vectorstores(self) -> list[FAISS]:
        stores = []
        for book_id in self.book_ids:
            db_path = str(FAISS_DB_PATH / book_id)
            if os.path.exists(db_path):
                try:
                    stores.append(
                        FAISS.load_local(
                            db_path,
                            self.embedding_model,
                            allow_dangerous_deserialization=True,
                        )
                    )
                except Exception as e:
                    import logging

                    logging.warning(
                        "Konnte Vektor-DB für Buch '%s' nicht laden: %s",
                        book_id,
                        e,
                    )
        return stores

    def invoke(self, input: str, config=None, **kwargs) -> list[Document]:
        if not self.vectorstores:
            return []
        k = self.context_size * self.k_multiplier
        all_results_with_scores = []
        for store in self.vectorstores:
            try:
                results = store.similarity_search_with_relevance_scores(input, k=k)
                all_results_with_scores.extend(results)
            except Exception as e:
                import logging

                logging.error("Fehler bei der Suche in einem Vektor-Store: %s", e)
        all_results_with_scores.sort(key=lambda x: x[1], reverse=True)
        # Duplikate entfernen und Score anreichern
        unique_docs = {}
        for doc, score in all_results_with_scores:
            if doc.page_content not in unique_docs:
                doc.metadata["semantic_score"] = round(score, 4)
                unique_docs[doc.page_content] = (doc, score)
        final_docs = []
        for doc, _ in unique_docs.values():
            final_docs.append(doc)
        return final_docs


class RAGService:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": DEVICE},
            encode_kwargs={"normalize_embeddings": False},
        )
        self.llm = OllamaLLM(model=LLM_MODEL)
        self.prompt = self._initialize_prompt()
        import logging

        logging.info("RAG-Service initialisiert.")

    def _initialize_prompt(self):
        prompt_template = """
        Du bist ein hochqualifizierter juristischer Tutor namens 'Castigatio'. Deine Aufgabe ist es, die Frage des Studenten präzise, klar und didaktisch aufzubereiten.
        Deine Antwort muss sich strikt und ausschließlich auf die Informationen aus dem bereitgestellten 'Kontext' (Auszüge aus einem Lehrbuch/Gesetz) stützen. Erfinde keine Fakten oder Paragraphen.

        ANWEISUNGEN FÜR DIE ANTWORT:
        1. Beginne mit einer klaren, direkten Antwort auf die Frage.
        2. Strukturiere deine Antwort logisch in Abschnitten oder Aufzählungspunkten. Nutze Markdown für die Formatierung.
        3. Wenn im Kontext Paragraphen (§) oder Artikel (Art.) erwähnt werden, die für die Antwort relevant sind, nenne diese explizit (z.B. "gemäß § 433 BGB...").
        4. Wenn die Informationen im Kontext nicht ausreichen, um die Frage zu beantworten, antworte ausschließlich mit: "Die zur Beantwortung dieser Frage benötigten Informationen sind in den vorliegenden Textauszügen nicht enthalten." Gib keine allgemeinen Ratschläge.

        KONTEXT:
        {context}

        FRAGE:
        {question}

        DEINE STRUKTURIERTE ANTWORT:
        """
        return PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

    def _rerank_and_fuse(self, docs: list[Document], enhanced_query) -> list[Document]:
        for doc in docs:
            semantic_score = doc.metadata.get("semantic_score", 0.0)
            content_lower = doc.page_content.lower()
            keyword_matches = sum(
                1
                for keyword in getattr(enhanced_query, "keywords", [])
                if keyword in content_lower
            )
            keyword_score = (
                keyword_matches / (len(getattr(enhanced_query, "keywords", [])) + 1e-6)
                if getattr(enhanced_query, "keywords", None)
                else 0.0
            )
            final_score = (semantic_score * 0.7) + (keyword_score * 0.3)
            doc.metadata["relevance_score"] = round(final_score, 4)
        docs.sort(key=lambda x: x.metadata.get("relevance_score", 0.0), reverse=True)
        return docs

    def _format_docs(self, docs: list[Document]) -> str:
        return "\n\n---\n\n".join(
            f"Quelle: {doc.metadata.get('source_file', 'Unbekannt')}, Seite: {doc.metadata.get('page', 'Unbekannt')}\nScore: {doc.metadata.get('relevance_score', 0.0):.4f}\n\n{doc.page_content}"
            for doc in docs
        )

    def ask_question(self, request: QueryRequest) -> QueryResponse:
        total_start_time = datetime.now()

        # Schritt 0: Cache-Prüfung
        cached_response = cache_service.get(request.question, request.book_ids)
        if cached_response:
            import logging

            logging.info("Cache-Treffer!")
            return cached_response

        if not FAISS_DB_PATH.exists() or not any(FAISS_DB_PATH.iterdir()):
            raise RuntimeError(
                "Keine Vektor-Datenbanken gefunden. Führen Sie zuerst den Ingestion-Prozess durch."
            )

        trace_id = xai_service.create_trace(request)

        enhancer_start_time = datetime.now()
        enhanced_query = query_enhancer_service.enhance_query(request.question)
        enhancer_end_time = datetime.now()
        xai_service.log_step(
            trace_id,
            "1. Query Enhancement",
            {"original_query": request.question},
            enhanced_query.model_dump(),
            enhancer_start_time,
            enhancer_end_time,
        )

        retrieval_start_time = datetime.now()
        k_multiplier = learning_service.adaptive_parameters["retrieval_k_multiplier"]
        retriever = MergedRetriever(
            request.book_ids, self.embedding_model, request.context_size, k_multiplier
        )
        candidate_docs = retriever.invoke(enhanced_query.enhanced_query)
        retrieval_end_time = datetime.now()
        xai_service.log_step(
            trace_id,
            "2. Candidate Retrieval",
            {"enhanced_query": enhanced_query.enhanced_query},
            {"retrieved_docs_count": len(candidate_docs)},
            retrieval_start_time,
            retrieval_end_time,
        )

        rerank_start_time = datetime.now()
        final_docs = self._rerank_and_fuse(candidate_docs, enhanced_query)[
            : request.context_size
        ]
        final_sources = [
            SourceDocument(
                content=doc.page_content,
                metadata=doc.metadata,
                relevance_score=doc.metadata.get("relevance_score"),
            )
            for doc in final_docs
        ]
        rerank_end_time = datetime.now()
        xai_service.log_step(
            trace_id,
            "3. Reranking & Fusion",
            {"candidate_count": len(candidate_docs)},
            {"final_docs": [s.model_dump() for s in final_sources]},
            rerank_start_time,
            rerank_end_time,
        )

        context_str = self._format_docs(final_docs)
        chain = self.prompt | self.llm | StrOutputParser()
        generation_start_time = datetime.now()
        answer_text = chain.invoke(
            {"context": context_str, "question": request.question}
        )
        generation_end_time = datetime.now()
        xai_service.log_step(
            trace_id,
            "4. Answer Generation",
            {"context_length": len(context_str)},
            {"final_answer": answer_text},
            generation_start_time,
            generation_end_time,
        )

        quality_start_time = datetime.now()
        quality_metrics = quality_service.assess_answer_quality(
            request.question, answer_text, final_sources
        )
        quality_end_time = datetime.now()
        xai_service.log_step(
            trace_id,
            "5. Quality Assessment",
            {},
            quality_metrics.model_dump(),
            quality_start_time,
            quality_end_time,
        )

        total_end_time = datetime.now()

        # Performance-Daten sammeln und loggen
        metrics = PerformanceMetrics(
            total_duration_ms=(total_end_time - total_start_time).total_seconds()
            * 1000,
            enhancement_duration_ms=(
                enhancer_end_time - enhancer_start_time
            ).total_seconds()
            * 1000,
            retrieval_duration_ms=(
                retrieval_end_time - retrieval_start_time
            ).total_seconds()
            * 1000,
            rerank_duration_ms=(rerank_end_time - rerank_start_time).total_seconds()
            * 1000,
            generation_duration_ms=(
                generation_end_time - generation_start_time
            ).total_seconds()
            * 1000,
            quality_duration_ms=(quality_end_time - quality_start_time).total_seconds()
            * 1000,
            confidence_score=quality_metrics.confidence_score,
            cache_hit=False,
        )
        learning_service.record_metrics(metrics)

        xai_service.finalize_trace(trace_id, answer_text)

        response = QueryResponse(
            answer=answer_text,
            sources=final_sources,
            quality=quality_metrics,
            trace_id=trace_id,
        )

        # Antwort im Cache speichern
        cache_service.set(request.question, request.book_ids, response)

        return response


rag_service = RAGService()
