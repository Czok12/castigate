from fastapi import FastAPI

from app.api import ingestion, library, rag, status

app = FastAPI(
    title="🏛️ Castigatio - Juristische Wissensdatenbank",
    description="Das Backend für die juristische RAG-Anwendung mit erweiterten KI-Funktionen.",
    version="0.4.0",
)


# Binde die API-Router ein

app.include_router(status.router, prefix="/api/v1")
app.include_router(library.router, prefix="/api/v1")
app.include_router(ingestion.router, prefix="/api/v1")
app.include_router(rag.router, prefix="/api/v1")


@app.get("/", summary="Root-Endpunkt", tags=["System"])
async def read_root():
    """
    Ein einfacher Willkommens-Endpunkt.
    """
    return {"message": "Willkommen beim Backend von Castigatio!"}
