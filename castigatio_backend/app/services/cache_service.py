# castigatio_backend/app/services/cache_service.py
import hashlib
from typing import List, Optional

import diskcache as dc

from app.core.config import DATA_DIR
from app.models.rag import QueryResponse


class CacheService:
    """Service für semantisches Caching von RAG-Anfragen."""

    def __init__(self) -> None:
        cache_path = DATA_DIR / "cache"
        cache_path.mkdir(exist_ok=True)
        self.query_cache = dc.Cache(str(cache_path / "queries"))

    def _get_cache_key(self, question: str, book_ids: Optional[List[str]]) -> str:
        """Erstellt einen eindeutigen Schlüssel für eine Anfrage."""
        sorted_ids = sorted(book_ids) if book_ids else []
        key_string = f"{question}-{','.join(sorted_ids)}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(
        self, question: str, book_ids: Optional[List[str]]
    ) -> Optional[QueryResponse]:
        key = self._get_cache_key(question, book_ids)
        return self.query_cache.get(key)

    def set(
        self, question: str, book_ids: Optional[List[str]], response: QueryResponse
    ) -> None:
        key = self._get_cache_key(question, book_ids)
        # Cache für 24 Stunden speichern
        self.query_cache.set(key, response, expire=86400)


cache_service = CacheService()
