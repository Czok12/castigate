from typing import Dict, List

from pydantic import BaseModel


class EnhancedQuery(BaseModel):
    """Strukturierte Daten einer erweiterten Anfrage."""

    original_query: str
    enhanced_query: str
    keywords: List[str]
    entities: Dict[str, List[str]]
