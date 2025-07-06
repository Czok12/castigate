from fastapi import APIRouter, HTTPException, status

from app.models.rag import QueryRequest, QueryResponse
from app.services.rag_service import rag_service

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Frage an die Wissensdatenbank stellen",
    tags=["RAG"],
)
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
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
        )
    except Exception as e:
        # Loggen Sie den Fehler für die Fehlersuche
        print(f"ERROR during query processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ein interner Fehler ist aufgetreten: {e}",
        )
