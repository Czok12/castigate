from datetime import datetime

from fastapi import APIRouter

from app.models.status import StatusResponse

router = APIRouter()


@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Systemstatus abrufen",
    tags=["System"],
)
async def get_status():
    """
    Gibt den aktuellen Status des Backend-Servers zurück.
    Nützlich als Health-Check für das Frontend.
    """
    return StatusResponse(
        status="ok",
        message="Castigatio Backend ist betriebsbereit.",
        timestamp=datetime.now(),
    )
