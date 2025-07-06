# castigatio_backend/app/models/xai.py
from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel

from app.models.rag import QueryRequest


class DecisionStep(BaseModel):
    """Modell für einen einzelnen Schritt im RAG-Entscheidungsprozess."""

    step_name: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    start_time: datetime
    end_time: datetime
    duration_ms: float


class ExplanationTrace(BaseModel):
    """
    Modell für den vollständigen, nachvollziehbaren Ablauf (Trace)
    einer RAG-Anfrage.
    """

    trace_id: str
    request: QueryRequest
    final_answer: str
    steps: List[DecisionStep]
    total_duration_ms: float
