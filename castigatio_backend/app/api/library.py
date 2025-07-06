from typing import List

from fastapi import APIRouter, HTTPException, status

from app.models.library import BookMetadata, BookMetadataCreate
from app.services.library_service import library_service

router = APIRouter()


@router.get(
    "/books",
    response_model=List[BookMetadata],
    summary="Alle Bücher abrufen",
    tags=["Bibliothek"],
)
async def get_all_books():
    """Ruft eine Liste aller Lehrbücher in der Bibliothek ab, sortiert nach Autor und Jahr."""
    return library_service.get_all_books()


@router.post(
    "/books",
    response_model=BookMetadata,
    status_code=status.HTTP_201_CREATED,
    summary="Neues Buch hinzufügen",
    tags=["Bibliothek"],
)
async def add_new_book(book_data: BookMetadataCreate):
    """Fügt ein neues Lehrbuch zur Bibliothek hinzu."""
    try:
        return library_service.add_book(book_data)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.get(
    "/books/{book_id}",
    response_model=BookMetadata,
    summary="Ein Buch abrufen",
    tags=["Bibliothek"],
)
async def get_book(book_id: str):
    """Ruft die Metadaten für ein spezifisches Buch anhand seiner ID ab."""
    book = library_service.get_book_by_id(book_id)
    if not book:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Buch nicht gefunden."
        )
    return book


@router.delete(
    "/books/{book_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Ein Buch löschen",
    tags=["Bibliothek"],
)
async def delete_book(book_id: str):
    """Löscht ein Buch und die zugehörigen Vektor-Daten (letzteres wird später implementiert)."""
    if not library_service.delete_book(book_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Buch nicht gefunden."
        )
    return None  # Bei 204 No Content wird kein Body zurückgegeben
