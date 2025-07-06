from datetime import datetime

from pydantic import BaseModel


class StatusResponse(BaseModel):
    """
    Standard-Antwortmodell für den Systemstatus-Endpunkt.
    """

    status: str
    message: str
    timestamp: datetime
