from fastapi import FastAPI

from app.api import status

app = FastAPI(
    title="🏛️ Juristische Wissensdatenbank - Backend",
    description="Das Backend für die juristische RAG-Anwendung mit erweiterten KI-Funktionen.",
    version="0.1.0",
)

app.include_router(status.router, prefix="/api/v1")

@app.get("/", summary="Root-Endpunkt", tags=["System"])
async def read_root():
    """Ein einfacher Willkommens-Endpunkt."""
    return {"message": "Willkommen beim Backend der Juristischen Wissensdatenbank!"}
