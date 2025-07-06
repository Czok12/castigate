from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Anfragemodell für eine RAG-Query."""

    question: str = Field(
        ..., min_length=10, description="Die juristische Frage des Nutzers."
    )
    book_ids: Optional[List[str]] = Field(
        None,
        description="Optionale Liste von Buch-IDs, die durchsucht werden sollen. Wenn leer, werden alle durchsucht.",
    )
    context_size: int = Field(
        4, gt=0, le=10, description="Anzahl der zu holenden Dokument-Chunks."
    )
    mode: str = Field(
        "balanced",
        description="Der Antwortmodus (z.B. 'quick', 'balanced', 'detailed').",
    )


class SourceDocument(BaseModel):
    """Modell für ein Quelldokument, das zur Antwortbeigetragen hat."""

    content: str
    metadata: Dict[str, Any]
    relevance_score: Optional[float] = None


class QueryResponse(BaseModel):
    """Antwortmodell einer RAG-Query."""

    answer: str
    sources: List[SourceDocument]
    trace_id: str = Field(
        ..., description="Eine eindeutige ID zur Nachverfolgung dieser Anfrage."
    )
