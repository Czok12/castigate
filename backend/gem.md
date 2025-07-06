Einverstanden. Jetzt, da wir eine solide Basis für die Verwaltung und das Ingestieren unserer Bücher haben, implementieren wir das Herzstück von Castigatio: die RAG-Pipeline. Dieser Service wird eine Nutzerfrage entgegennehmen, relevante Informationen aus unserer Vektordatenbank abrufen und mithilfe des LLMs eine fundierte Antwort generieren.

Wir nutzen die bewährte Logik aus Ihren alten `app.py` und `enhanced_jura_app.py`, um eine robuste und erweiterbare RAG-Kette zu bauen.

---

### Prompt 4: Kern-RAG-Pipeline implementieren

**Ziel:** Implementierung der zentralen Frage-Antwort-Funktionalität. Ein neuer `/query`-Endpunkt soll eine Nutzerfrage verarbeiten und eine auf den indexierten Dokumenten basierende, vom LLM generierte Antwort zurückgeben.

**Kontext:** Wir haben einen `LibraryService` und einen `IngestionService`. Nun erstellen wir den `RAGService`, der die FAISS-Indizes nutzt, um Fragen zu beantworten.

**Aufgaben:**

1. **RAG-Modelle definieren:** Erstelle Pydantic-Modelle für die Anfrage und die Antwort der RAG-API.
2. **`RAGService` erstellen:** Baue einen Service, der die FAISS-Indizes lädt, das LLM initialisiert und die Logik für die Beantwortung von Fragen kapselt. Der Service muss in der Lage sein, mit mehreren FAISS-Indizes (einen pro Buch) umzugehen.
3. **`MergedRetriever` implementieren:** Erstelle eine benutzerdefinierte Retriever-Klasse, die mehrere FAISS-Datenbanken gleichzeitig abfragen und die Ergebnisse zusammenführen kann.
4. **API-Endpunkt für Queries:** Implementiere den `POST /query`-Endpunkt, der den `RAGService` aufruft.
5. **`main.py` aktualisieren**, um den neuen Endpunkt einzubinden.

**Anweisung:** Erstelle und aktualisiere die folgenden Dateien mit dem angegebenen Inhalt.

**1. Datei: `castigatio_backend/app/models/rag.py`**

Diese Modelle definieren die Datenstruktur für die Kommunikation mit dem RAG-Endpunkt.

```python
# castigatio_backend/app/models/rag.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    """Anfragemodell für eine RAG-Query."""
    question: str = Field(..., min_length=10, description="Die juristische Frage des Nutzers.")
    book_ids: Optional[List[str]] = Field(None, description="Optionale Liste von Buch-IDs, die durchsucht werden sollen. Wenn leer, werden alle durchsucht.")
    context_size: int = Field(4, gt=0, le=10, description="Anzahl der zu holenden Dokument-Chunks.")
    mode: str = Field("balanced", description="Der Antwortmodus (z.B. 'quick', 'balanced', 'detailed').")

class SourceDocument(BaseModel):
    """Modell für ein Quelldokument, das zur Antwortbeigetragen hat."""
    content: str
    metadata: Dict[str, Any]
    relevance_score: Optional[float] = None

class QueryResponse(BaseModel):
    """Antwortmodell einer RAG-Query."""
    answer: str
    sources: List[SourceDocument]
    trace_id: str = Field(..., description="Eine eindeutige ID zur Nachverfolgung dieser Anfrage.")
```

**2. Datei: `castigatio_backend/app/services/rag_service.py`**

Dies ist der neue, zentrale Service. Er enthält die Logik zum Laden der Vektor-DBs und zur Erstellung der RAG-Kette. Besonders wichtig ist der `MergedRetriever`.

```python
# castigatio_backend/app/services/rag_service.py
import os
import uuid
import itertools
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.documents import Document

from app.core.config import FAISS_DB_PATH, EMBEDDING_MODEL, LLM_MODEL, DEVICE
from app.models.rag import QueryRequest, QueryResponse, SourceDocument

class MergedRetriever(Runnable):
    """Ein benutzerdefinierter Retriever, der mehrere FAISS-Indizes zusammenführt."""
    def __init__(self, book_ids: Optional[List[str]], embedding_model, context_size: int):
        self.embedding_model = embedding_model
        self.book_ids = self._get_available_book_ids(book_ids)
        self.context_size = context_size
        self.vectorstores = self._load_vectorstores()

    def _get_available_book_ids(self, requested_ids: Optional[List[str]]) -> List[str]:
        """Gibt eine Liste gültiger, existierender Buch-IDs zurück."""
        if not FAISS_DB_PATH.exists():
            return []

        all_dbs = {db_dir.name for db_dir in FAISS_DB_PATH.iterdir() if db_dir.is_dir()}

        if requested_ids:
            # Filtere nach angefragten und existierenden IDs
            return [book_id for book_id in requested_ids if book_id in all_dbs]
        # Wenn keine IDs angefragt wurden, nutze alle verfügbaren
        return list(all_dbs)

    def _load_vectorstores(self) -> List[FAISS]:
        """Lädt die FAISS-Indizes für die angegebenen Buch-IDs."""
        stores = []
        for book_id in self.book_ids:
            db_path = str(FAISS_DB_PATH / book_id)
            if os.path.exists(db_path):
                try:
                    stores.append(FAISS.load_local(db_path, self.embedding_model, allow_dangerous_deserialization=True))
                except Exception as e:
                    print(f"WARNUNG: Konnte Vektor-DB für Buch '{book_id}' nicht laden: {e}")
        return stores

    def invoke(self, input: str) -> List[Document]:
        """Führt die Suche in allen geladenen Vektor-DBs durch und fusioniert die Ergebnisse."""
        if not self.vectorstores:
            return []

        # Führe die Suche in allen Stores parallel durch (vereinfacht)
        all_results_with_scores = []
        for store in self.vectorstores:
            results = store.similarity_search_with_score(input, k=self.context_size)
            all_results_with_scores.extend(results)

        # Sortiere alle Ergebnisse nach Score (niedriger ist besser bei L2-Distanz)
        all_results_with_scores.sort(key=lambda x: x[1])

        # Gebe die besten Ergebnisse über alle Bücher hinweg zurück
        # Konvertiere Tupel (Document, score) zu Document
        top_docs = [doc for doc, score in all_results_with_scores[:self.context_size]]
        return top_docs

    def format_docs(self, docs: List[Document]) -> str:
        """Formatiert die Dokumente für den LLM-Kontext."""
        return "\n\n---\n\n".join(
            f"Quelle: {doc.metadata.get('source_file', 'Unbekannt')}, Seite: {doc.metadata.get('page', 'Unbekannt')}\n\n{doc.page_content}"
            for doc in docs
        )

class RAGService:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': DEVICE},
        )
        self.llm = OllamaLLM(model=LLM_MODEL)
        print("INFO: RAG-Service initialisiert.")

    def get_rag_chain(self, book_ids: Optional[List[str]], context_size: int) -> Runnable:
        """Erstellt und gibt die vollständige RAG-Kette zurück."""
        retriever = MergedRetriever(
            book_ids=book_ids,
            embedding_model=self.embedding_model,
            context_size=context_size
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
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        return (
            {"context": retriever | retriever.format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def ask_question(self, request: QueryRequest) -> QueryResponse:
        """Verarbeitet eine Anfrage, generiert eine Antwort und gibt die Quellen zurück."""
        if not FAISS_DB_PATH.exists() or not any(FAISS_DB_PATH.iterdir()):
            raise RuntimeError("Keine Vektor-Datenbanken gefunden. Bitte führen Sie zuerst den Ingestion-Prozess für mindestens ein Buch durch.")

        rag_chain = self.get_rag_chain(request.book_ids, request.context_size)
        answer_text = rag_chain.invoke(request.question)

        # Erneuter Abruf der Dokumente, um die genauen Quellen zu erhalten
        retriever = MergedRetriever(
            book_ids=request.book_ids,
            embedding_model=self.embedding_model,
            context_size=request.context_size
        )
        retrieved_docs = retriever.invoke(request.question)
        sources = [SourceDocument(content=doc.page_content, metadata=doc.metadata) for doc in retrieved_docs]

        trace_id = f"trace-{uuid.uuid4()}"

        return QueryResponse(answer=answer_text, sources=sources, trace_id=trace_id)

rag_service = RAGService()
```

**3. Datei: `castigatio_backend/app/api/rag.py`**

Der Endpunkt, der Anfragen an den `RAGService` weiterleitet.

```python
# castigatio_backend/app/api/rag.py
from fastapi import APIRouter, HTTPException, status
from app.models.rag import QueryRequest, QueryResponse
from app.services.rag_service import rag_service

router = APIRouter()

@router.post("/query", response_model=QueryResponse, summary="Frage an die Wissensdatenbank stellen", tags=["RAG"])
async def process_query(request: QueryRequest):
    """
    Verarbeitet eine juristische Frage und gibt eine generierte Antwort
    sowie die verwendeten Quellen zurück.

    - **question**: Die Frage, die gestellt werden soll.
    - **book_ids**: Eine optionale Liste von Buch-IDs. Wenn nicht angegeben, wird in allen Büchern gesucht.
    - **context_size**: Die Anzahl der relevantesten Text-Chunks, die für die Antwort verwendet werden sollen.
    """
    try:
        response = rag_service.ask_question(request)
        return response
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        # Loggen Sie den Fehler für die Fehlersuche
        print(f"ERROR during query processing: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Ein interner Fehler ist aufgetreten: {e}")
```

**4. Datei: `castigatio_backend/app/main.py` (Anpassen)**

Wir binden den neuen RAG-Router ein.

```python
# castigatio_backend/app/main.py
from fastapi import FastAPI
from app.api import status, library, ingestion, rag # <-- Importiere rag

app = FastAPI(
    title="🏛️ Castigatio - Juristische Wissensdatenbank",
    description="Das Backend für die juristische RAG-Anwendung mit erweiterten KI-Funktionen.",
    version="0.4.0",
)

# Binde die API-Router ein
app.include_router(status.router, prefix="/api/v1")
app.include_router(library.router, prefix="/api/v1")
app.include_router(ingestion.router, prefix="/api/v1")
app.include_router(rag.router, prefix="/api/v1") # <-- Binde den RAG-Router ein

@app.get("/", summary="Root-Endpunkt", tags=["System"])
async def read_root():
    """Ein einfacher Willkommens-Endpunkt."""
    return {"message": "Willkommen beim Backend von Castigatio!"}
```

**Überprüfung:**

1. Stellen Sie sicher, dass alle neuen Abhängigkeiten installiert sind: `pip install -r castigatio_backend/requirements.txt`.
2. Vergewissern Sie sich, dass Sie mindestens ein Buch über den Ingestion-Endpunkt verarbeitet haben und ein entsprechender Ordner in `castigatio_backend/data/faiss_index/` existiert.
3. Starten Sie den Server neu: `uvicorn app.main:app --reload`.
4. Öffnen Sie die API-Dokumentation unter **http://localhost:8000/docs**.
5. Finden Sie den neuen Endpunkt `POST /api/v1/query` im Abschnitt "RAG".
6. Testen Sie ihn, indem Sie auf "Try it out" klicken und eine Anfrage stellen. Sie können die `book_ids` leer lassen, um alle Bücher zu durchsuchen, oder spezifische IDs angeben.
   ```json
   {
     "question": "Was sind die Voraussetzungen einer wirksamen Anfechtung?",
     "book_ids": [],
     "context_size": 4
   }
   ```
7. Sie sollten eine strukturierte Antwort und eine Liste der Quellen zurückerhalten.

Damit ist die Kernfunktionalität Ihrer Anwendung implementiert! Als Nächstes können wir sie mit fortgeschrittenen Funktionen wie der Zitations-Engine oder dem XAI-Dashboard weiter verbessern.
