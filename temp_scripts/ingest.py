import os

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- Konfiguration ---
PDF_PATH = "dein_lehrbuch.pdf"  # <-- ÄNDERN: Gib hier den Pfad zu deiner PDF an
DB_PATH = "faiss_db"


def load_and_chunk_pdf(file_path):
    print("Lade PDF...")
    doc = fitz.open(file_path)

    # Sammle Text mit Metadaten (Seitenzahlen)
    pages_content = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()
        if page_text.strip():  # Nur nicht-leere Seiten
            pages_content.append(
                {
                    "text": page_text,
                    "page": page_num + 1,  # Seitenzahlen beginnen bei 1
                    "source": os.path.basename(file_path),
                }
            )
    doc.close()

    print(
        f"PDF geladen. {len(pages_content)} Seiten gefunden. Zerlege Text in Chunks..."
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Größe der Textstücke in Zeichen
        chunk_overlap=150,  # Überlappung zwischen den Stücken
        length_function=len,
        # Juraspezifische Trenner für bessere Chunks
        separators=["\n\n", "\n", "§", "Art.", ". ", " "],
    )

    # Erstelle Chunks mit Metadaten
    chunks_with_metadata = []
    for page_content in pages_content:
        page_chunks = text_splitter.split_text(page_content["text"])
        for i, chunk in enumerate(page_chunks):
            chunks_with_metadata.append(
                {
                    "text": chunk,
                    "metadata": {
                        "source": page_content["source"],
                        "page": page_content["page"],
                        "chunk_id": f"page_{page_content['page']}_chunk_{i+1}",
                    },
                }
            )

    print(f"{len(chunks_with_metadata)} Chunks mit Metadaten erstellt.")
    return chunks_with_metadata


def create_and_save_vectorstore(chunks_with_metadata):
    print("Erstelle Embeddings und Vektordatenbank...")
    # Wir benutzen ein gutes deutsches Embedding-Modell
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={"device": "cpu"},  # 'cuda' wenn du eine NVIDIA GPU hast
    )

    # Extrahiere Texte und Metadaten
    texts = [chunk["text"] for chunk in chunks_with_metadata]
    metadatas = [chunk["metadata"] for chunk in chunks_with_metadata]

    # Dieser Schritt kann bei 800 Seiten einige Minuten dauern!
    vectorstore = FAISS.from_texts(
        texts=texts, embedding=embedding_model, metadatas=metadatas
    )

    # Speichere die Datenbank lokal ab
    vectorstore.save_local(DB_PATH)
    print(f"Vektordatenbank wurde in '{DB_PATH}' gespeichert.")


if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(
            f"Fehler: Die Datei '{PDF_PATH}' wurde nicht gefunden. Bitte passe den Pfad an."
        )
    else:
        # Nur ausführen, wenn die Datenbank noch nicht existiert
        if not os.path.exists(DB_PATH):
            chunks_with_metadata = load_and_chunk_pdf(PDF_PATH)
            create_and_save_vectorstore(chunks_with_metadata)
        else:
            print(
                f"Datenbank '{DB_PATH}' existiert bereits. Überspringe die Erstellung."
            )
