from pydantic import BaseModel


class IngestionResponse(BaseModel):
    """Antwortmodell für den Ingestion-Prozess."""

    book_id: str
    chunk_count: int
    processing_time_seconds: float
    message: str
