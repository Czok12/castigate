# castigatio_backend/app/main.py
from fastapi import FastAPI

from app.api import citation, ingestion, library, rag, status  # <-- Importiere citation

app = FastAPI(
    title="🏛️ Castigatio - Juristische Wissensdatenbank",
    description="Das Backend für die juristische RAG-Anwendung mit erweiterten KI-Funktionen.",
    version="0.5.0",  # <-- Version erhöht
)

# Binde die API-Router ein
app.include_router(status.router, prefix="/api/v1")
app.include_router(library.router, prefix="/api/v1", tags=["Bibliothek"])
app.include_router(ingestion.router, prefix="/api/v1", tags=["Ingestion"])
app.include_router(rag.router, prefix="/api/v1", tags=["RAG"])
app.include_router(
    citation.router, prefix="/api/v1", tags=["Zitation"]
)  # <-- Binde den Zitations-Router ein


@app.get("/", summary="Root-Endpunkt", tags=["System"])
async def read_root():
    """Ein einfacher Willkommens-Endpunkt."""
    return {"message": "Willkommen beim Backend von Castigatio!"}
