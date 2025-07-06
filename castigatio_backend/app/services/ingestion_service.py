import os
import time

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import DEVICE, EMBEDDING_MODEL, FAISS_DB_PATH, PDF_LIBRARY_PATH
from app.models.ingestion import IngestionResponse
from app.services.library_service import library_service


class IngestionService:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "§", "Art.", ". ", " ", ""],
        )
        FAISS_DB_PATH.mkdir(exist_ok=True)

    def _load_and_chunk_pdf(self, file_path: str, book_id: str) -> list:
        """Lädt PDF, extrahiert Text und zerlegt ihn in Chunks mit Metadaten."""
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            raise FileNotFoundError(
                f"Konnte PDF nicht öffnen: {file_path}. Fehler: {e}"
            )

        chunks_with_metadata = []
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text("text")
            if not page_text.strip():
                continue

            page_chunks = self.text_splitter.split_text(page_text)
            for i, chunk_text in enumerate(page_chunks):
                metadata = {
                    "book_id": book_id,
                    "source_file": os.path.basename(file_path),
                    "page": page_num,
                    "chunk_num": i,
                }
                chunks_with_metadata.append({"text": chunk_text, "metadata": metadata})

        doc.close()
        return chunks_with_metadata

    def process_book(self, book_id: str) -> IngestionResponse:
        """Verarbeitet ein einzelnes Lehrbuch und fügt es dem Vektor-Store hinzu."""
        start_time = time.time()

        book = library_service.get_book_by_id(book_id)
        if not book:
            raise ValueError(
                f"Buch mit ID '{book_id}' nicht in der Bibliothek gefunden."
            )
        if not book.dateiname:
            raise ValueError(f"Für Buch '{book_id}' ist kein Dateiname hinterlegt.")

        pdf_path = PDF_LIBRARY_PATH / book.dateiname
        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF-Datei '{book.dateiname}' nicht im Verzeichnis '{PDF_LIBRARY_PATH}' gefunden."
            )

        # Schritt 1: PDF laden und in Chunks zerlegen
        chunks_with_metadata = self._load_and_chunk_pdf(str(pdf_path), book_id)
        if not chunks_with_metadata:
            raise ValueError("Kein Text konnte aus der PDF extrahiert werden.")

        texts = [chunk["text"] for chunk in chunks_with_metadata]
        metadatas = [chunk["metadata"] for chunk in chunks_with_metadata]

        # Schritt 2: Vektor-Datenbank erstellen oder aktualisieren
        db_path = str(FAISS_DB_PATH / book_id)
        if os.path.exists(db_path):
            # Lade existierende DB und füge Dokumente hinzu
            vectorstore = FAISS.load_local(
                db_path, self.embedding_model, allow_dangerous_deserialization=True
            )
            vectorstore.add_texts(texts=texts, metadatas=metadatas)
        else:
            # Erstelle neue DB
            vectorstore = FAISS.from_texts(
                texts=texts, embedding=self.embedding_model, metadatas=metadatas
            )

        # Speichere die DB
        vectorstore.save_local(db_path)

        # Schritt 3: Metadaten in der Bibliotheks-DB aktualisieren
        library_service.update_book_metadata(
            book_id, {"chunk_anzahl": len(chunks_with_metadata)}
        )

        processing_time = time.time() - start_time
        return IngestionResponse(
            book_id=book_id,
            chunk_count=len(chunks_with_metadata),
            processing_time_seconds=round(processing_time, 2),
            message=f"Buch '{book.titel}' erfolgreich verarbeitet und indexiert.",
        )


ingestion_service = IngestionService()
