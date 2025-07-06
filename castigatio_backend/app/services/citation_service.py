# castigatio_backend/app/services/citation_service.py
import re
from typing import Any, Dict

from app.models.citation import (
    CitationSuggestion,
    CitationSuggestRequest,
    CitationSuggestResponse,
    CitationValidateRequest,
    CitationValidateResponse,
)
from app.models.rag import SourceDocument
from app.services.library_service import library_service
from app.services.rag_service import MergedRetriever, rag_service

CITATION_STYLES = ["Juristisch", "APA", "Chicago"]


class CitationService:
    def __init__(self):
        # Wir nutzen das bereits initialisierte Embedding-Modell vom RAGService
        self.embedding_model = rag_service.embedding_model

    def suggest_citations_for_text(
        self, request: CitationSuggestRequest
    ) -> CitationSuggestResponse:
        """Findet relevante Quellen für einen Text und generiert Zitationen."""
        retriever = MergedRetriever(
            book_ids=request.book_ids,
            embedding_model=self.embedding_model,
            context_size=request.num_suggestions,
        )
        retrieved_docs = retriever.invoke(request.text)

        suggestions = []
        for i, doc in enumerate(retrieved_docs, 1):
            source_doc = SourceDocument(
                content=doc.page_content,
                metadata=doc.metadata,
                relevance_score=doc.metadata.get("relevance_score"),
            )

            generated_citations = {
                style: self._generate_citation(doc.metadata, style)
                for style in CITATION_STYLES
            }

            suggestions.append(
                CitationSuggestion(
                    rank=i + 1,
                    relevance_score=doc.metadata.get("relevance_score", 0.0),
                    source_document=source_doc,
                    generated_citations=generated_citations,
                )
            )

        return CitationSuggestResponse(suggestions=suggestions)

    def _generate_citation(self, metadata: Dict[str, Any], style: str) -> str:
        """Generiert eine formatierte Zitation basierend auf den Metadaten."""
        book_id = metadata.get("book_id")
        page = metadata.get("page")

        book = library_service.get_document_by_id(book_id) if book_id else None

        if not book:
            source_file = metadata.get("source_file", "Unbekannte Quelle")
            return f"{source_file}, S. {page}" if page else source_file

        autor = book.autor
        titel = book.titel
        jahr = book.jahr

        if style == "Juristisch":
            citation = f"{autor}, {titel}"
            if page:
                citation += f", S. {page}"
            return citation
        elif style == "APA":
            return f"{autor} ({jahr}). *{titel}*. (S. {page})."
        elif style == "Chicago":
            return f"{autor}, “{titel},” {jahr}, {page}."
        else:
            return f"{autor}, {titel}, S. {page}"

    def validate_citation_string(
        self, request: CitationValidateRequest
    ) -> CitationValidateResponse:
        """Validiert eine gegebene juristische Zitation."""
        patterns = {
            "paragraph": r"§\s*(\d+[a-z]?)\s*(?:Abs\.?\s*(\d+))?\s*(?:S\.\s*(\d+))?\s*([A-Z]{2,5})",
            "article": r"Art\.?\s*(\d+[a-z]?)\s*(?:Abs\.?\s*(\d+))?\s*([A-Z]{2,5})",
            "court_decision": r"([A-Z]+)\s*,\s*(?:Urt\.|Beschl\.)\s*v\.\s*(\d{1,2}\.\d{1,2}\.\d{4})\s*-\s*(.+)",
        }

        ref = request.citation_string.strip()

        for ref_type, pattern in patterns.items():
            match = re.fullmatch(pattern, ref, re.IGNORECASE)
            if match:
                components = {
                    f"group_{i + 1}": g for i, g in enumerate(match.groups()) if g
                }
                return CitationValidateResponse(
                    is_valid=True,
                    citation_type=ref_type,
                    formatted_string=match.group(0),
                    components=components,
                )

        return CitationValidateResponse(
            is_valid=False,
            error_message="Format nicht erkannt. Gültige Formate: '§ 433 Abs. 1 S. 1 BGB' oder 'BGH, Urt. v. 01.01.2023 - 1 StR 123/22'.",
        )


citation_service = CitationService()
