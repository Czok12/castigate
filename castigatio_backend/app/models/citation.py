# castigatio_backend/app/models/citation.py
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models.rag import SourceDocument


class CitationSuggestRequest(BaseModel):
    """Anfragemodell, um Zitationsvorschläge für einen Text zu erhalten."""

    text: str = Field(
        ...,
        min_length=20,
        description="Der Text, für den Quellen gefunden werden sollen.",
    )
    book_ids: Optional[List[str]] = Field(
        None, description="Optionale Liste von Buch-IDs zur Einschränkung der Suche."
    )
    num_suggestions: int = Field(
        5, gt=0, le=10, description="Anzahl der gewünschten Vorschläge."
    )


class CitationSuggestion(BaseModel):
    """Modell für einen einzelnen Zitationsvorschlag."""

    rank: int
    relevance_score: float
    source_document: SourceDocument
    generated_citations: Dict[str, str] = Field(
        ...,
        description="Ein Dictionary, das Zitationsstile auf die formatierte Zitation abbildet.",
    )


class CitationSuggestResponse(BaseModel):
    """Antwortmodell mit einer Liste von Zitationsvorschlägen."""

    suggestions: List[CitationSuggestion]


class CitationValidateRequest(BaseModel):
    """Anfragemodell zur Validierung einer einzelnen Zitation."""

    citation_string: str = Field(..., description="Die zu validierende Zitation.")


class CitationValidateResponse(BaseModel):
    """Antwortmodell für die Validierung einer Zitation."""

    is_valid: bool
    citation_type: Optional[str] = None
    formatted_string: Optional[str] = None
    components: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
