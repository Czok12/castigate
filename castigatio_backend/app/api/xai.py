# castigatio_backend/app/api/xai.py
from fastapi import APIRouter, HTTPException, status

from app.models.xai import ExplanationTrace
from app.services.xai_service import xai_service

router = APIRouter()


@router.get(
    "/xai/traces/{trace_id}",
    response_model=ExplanationTrace,
    summary="RAG-Prozess-Trace abrufen",
    tags=["XAI"],
)
async def get_explanation_trace(trace_id: str):
    """
    Ruft die detaillierten, protokollierten Schritte für eine spezifische
    RAG-Anfrage ab. Die `trace_id` wird von einer `/query`-Anfrage zurückgegeben.
    """
    trace = xai_service.get_trace_by_id(trace_id)
    if not trace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Trace nicht gefunden."
        )
    return trace
