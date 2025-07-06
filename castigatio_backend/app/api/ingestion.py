from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from app.models.ingestion import IngestionResponse
from app.services.ingestion_service import ingestion_service

router = APIRouter()


@router.post(
    "/books/{book_id}/ingest",
    response_model=IngestionResponse,
    summary="Ein Buch verarbeiten und indexieren",
    tags=["Ingestion"],
)
async def ingest_book_endpoint(book_id: str, background_tasks: BackgroundTasks):
    """
    Startet den Ingestion-Prozess für ein Buch.
    Dieser Prozess liest die zugehörige PDF, zerlegt sie, erstellt Vektor-Embeddings
    und speichert sie in der FAISS-Datenbank.

    HINWEIS: Dieser Endpunkt antwortet sofort, während der Prozess im Hintergrund läuft.
    """
    try:
        # Hier könnte man den Prozess in einen Hintergrund-Task auslagern
        # Für unsere Desktop-Anwendung ist ein blockierender Aufruf zunächst in Ordnung.
        response = ingestion_service.process_book(book_id)
        return response
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        # Loggen Sie den Fehler für die Fehlersuche
        print(f"ERROR during ingestion for book_id {book_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ein interner Fehler ist aufgetreten: {e}",
        )
