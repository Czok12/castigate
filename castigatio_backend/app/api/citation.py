"""API-Endpunkte für Zitationsvorschläge und -validierung (castigatio_backend)."""

from fastapi import APIRouter, HTTPException, status

from app.models.citation import (
    CitationSuggestRequest,
    CitationSuggestResponse,
    CitationValidateRequest,
    CitationValidateResponse,
)
from app.services.citation_service import citation_service

router = APIRouter()


@router.post(
    "/cite/suggest",
    response_model=CitationSuggestResponse,
    summary="Zitationsvorschläge für Text erhalten",
    tags=["Zitation"],
)
async def suggest_citations(request: CitationSuggestRequest):
    """Nimmt einen Text entgegen und findet relevante Passagen in der Wissensdatenbank.
    Gibt eine Liste mit Zitationsvorschlägen in verschiedenen Formaten zurück.
    """
    try:
        return citation_service.suggest_citations_for_text(request)
    except Exception as e:
        print(f"ERROR during citation suggestion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e


@router.post(
    "/cite/validate",
    response_model=CitationValidateResponse,
    summary="Eine juristische Zitation validieren",
    tags=["Zitation"],
)
async def validate_citation(request: CitationValidateRequest):
    """Überprüft, ob eine gegebene Zeichenkette einem gängigen juristischen Zitationsformat entspricht."""
    return citation_service.validate_citation_string(request)
