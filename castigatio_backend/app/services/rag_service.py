import os
import uuid
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM

from app.core.config import DEVICE, EMBEDDING_MODEL, FAISS_DB_PATH, LLM_MODEL
from app.models.rag import QueryRequest, QueryResponse, SourceDocument


class MergedRetriever(Runnable):
    """Ein benutzerdefinierter Retriever, der mehrere FAISS-Indizes zusammenführt."""

    def __init__(
        self, book_ids: Optional[List[str]], embedding_model, context_size: int
    ):
        self.embedding_model = embedding_model
        self.book_ids = self._get_available_book_ids(book_ids)
        self.context_size = context_size
        self.vectorstores = self._load_vectorstores()
        if not self.vectorstores:
            raise RuntimeError(
                "Keine gültigen Vektor-Datenbanken gefunden. Bitte den Ingestion-Prozess durchführen."
            )

    def _get_available_book_ids(self, requested_ids: Optional[List[str]]) -> List[str]:
        """Gibt eine Liste gültiger, existierender Buch-IDs zurück."""
        if not FAISS_DB_PATH.exists():
            return []

        all_dbs = {db_dir.name for db_dir in FAISS_DB_PATH.iterdir() if db_dir.is_dir()}

        if requested_ids:
            # Filtere nach angefragten und existierenden IDs
            valid_ids = [book_id for book_id in requested_ids if book_id in all_dbs]
            if not valid_ids:
                print(
                    f"WARNUNG: Keine der angefragten Buch-IDs {requested_ids} wurde gefunden. Verfügbar sind: {all_dbs}"
                )
            return valid_ids
        # Wenn keine IDs angefragt wurden, nutze alle verfügbaren
        return list(all_dbs)

    def _load_vectorstores(self) -> List[FAISS]:
        """Lädt die FAISS-Indizes für die angegebenen Buch-IDs."""
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
                    print(
                        f"WARNUNG: Konnte Vektor-DB für Buch '{book_id}' nicht laden: {e}"
                    )
        return stores

    def invoke(self, input, config=None, **kwargs) -> List[Document]:
        """Führt die Suche in allen geladenen Vektor-DBs durch und fusioniert die Ergebnisse."""
        if not self.vectorstores:
            return []

        all_results_with_scores = []
        for store in self.vectorstores:
            try:
                results = store.similarity_search_with_relevance_scores(
                    input, k=self.context_size
                )
                all_results_with_scores.extend(results)
            except Exception as e:
                print(f"Fehler bei der Suche in einem Vektor-Store: {e}")

        # Sortiere alle Ergebnisse nach Score (höher ist besser bei Relevanz-Score)
        all_results_with_scores.sort(key=lambda x: x[1], reverse=True)

        # Gebe die besten Dokumente über alle Bücher hinweg zurück
        top_docs = []
        for doc, score in all_results_with_scores[: self.context_size]:
            doc.metadata["relevance_score"] = round(score, 4)
            top_docs.append(doc)

        return top_docs

    def _format_docs(self, docs: List[Document]) -> str:
        """Formatiert die Dokumente für den LLM-Kontext."""
        return "\n\n---\n\n".join(
            f"Quelle: {doc.metadata.get('source_file', 'Unbekannt')}, Seite: {doc.metadata.get('page', 'Unbekannt')}\n\n{doc.page_content}"
            for doc in docs
        )


class RAGService:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": DEVICE},
            encode_kwargs={
                "normalize_embeddings": False
            },  # Cosine-Similarity funktioniert besser ohne Normalisierung
        )
        self.llm = OllamaLLM(model=LLM_MODEL)
        print("INFO: RAG-Service initialisiert.")

    def _format_docs(self, docs: List[Document]) -> str:
        """Formatiert die Dokumente für den LLM-Kontext."""
        return "\n\n---\n\n".join(
            f"Quelle: {doc.metadata.get('source_file', 'Unbekannt')}, Seite: {doc.metadata.get('page', 'Unbekannt')}\n\n{doc.page_content}"
            for doc in docs
        )

    def get_rag_chain(
        self, book_ids: Optional[List[str]], context_size: int
    ) -> Runnable:
        """Erstellt und gibt die vollständige RAG-Kette zurück."""
        retriever = MergedRetriever(
            book_ids=book_ids,
            embedding_model=self.embedding_model,
            context_size=context_size,
        )

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
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        return (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def ask_question(self, request: QueryRequest) -> QueryResponse:
        """Verarbeitet eine Anfrage, generiert eine Antwort und gibt die Quellen zurück."""
        if not FAISS_DB_PATH.exists() or not any(FAISS_DB_PATH.iterdir()):
            raise RuntimeError(
                "Keine Vektor-Datenbanken gefunden. Bitte führen Sie zuerst den Ingestion-Prozess für mindestens ein Buch durch."
            )

        # Eigene RAG-Kette, um Quellen separat zu behandeln
        retriever = MergedRetriever(
            request.book_ids, self.embedding_model, request.context_size
        )
        retrieved_docs = retriever.invoke(request.question)
        context_str = self._format_docs(retrieved_docs)

        # Prompt manuell erstellen
        prompt_template = """
        Du bist ein hochqualifizierter juristischer Tutor namens 'Castigatio'. Deine Aufgabe ist es, die Frage des Studenten präzise, klar und didaktisch aufzubereiten.
        Deine Antwort muss sich strikt und ausschließlich auf die Informationen aus dem bereitgestellten 'Kontext' (Auszüge aus einem Lehrbuch/Gesetz) stützen. Erfinde keine Fakten oder Paragraphen.

        ANWEISUNGEN FÜR DIE ANTWORT:
        1. Beginne mit einer klaren, direkten Antwort auf die Frage.
        2. Strukturiere deine Antwort logisch in Abschnitten oder Aufzählungspunkten. Nutze Markdown für die Formatierung.
        3. Wenn im Kontext Paragraphen (§) oder Artikel (Art.) erwähnt werden, die für die Antwort relevant sind, nenne diese explizit in deiner Antwort (z.B. "gemäß § 433 BGB...").
        4. Wenn die Informationen im Kontext nicht ausreichen, um die Frage zu beantworten, antworte ausschließlich mit: "Die zur Beantwortung dieser Frage benötigten Informationen sind in den vorliegenden Textauszügen nicht enthalten." Gib keine allgemeinen Ratschläge.

        KONTEXT:
        {context}

        FRAGE:
        {question}

        DEINE STRUKTURIERTE ANTWORT:
        """
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        chain = prompt | self.llm | StrOutputParser()
        answer_text = chain.invoke(
            {"context": context_str, "question": request.question}
        )

        sources = [
            SourceDocument(
                content=doc.page_content,
                metadata=doc.metadata,
                relevance_score=doc.metadata.get("relevance_score"),
            )
            for doc in retrieved_docs
        ]
        trace_id = f"trace-{uuid.uuid4()}"

        return QueryResponse(answer=answer_text, sources=sources, trace_id=trace_id)

    def _get_sources(
        self, query: str, book_ids: Optional[List[str]], context_size: int
    ) -> List[SourceDocument]:
        """Holt die Quellen separat, um sie mit Relevanz-Scores zurückzugeben."""
        retriever = MergedRetriever(
            book_ids=book_ids,
            embedding_model=self.embedding_model,
            context_size=context_size,
        )
        docs = retriever.invoke(query)
        return [
            SourceDocument(
                content=doc.page_content,
                metadata=doc.metadata,
                relevance_score=doc.metadata.get("relevance_score"),
            )
            for doc in docs
        ]


rag_service = RAGService()
