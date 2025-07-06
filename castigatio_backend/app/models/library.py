import re
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class BookMetadataBase(BaseModel):
    """Grundlegende Metadaten für ein Lehrbuch."""

    titel: str = Field(..., min_length=3, description="Der Titel des Lehrbuchs.")
    autor: str = Field(
        ..., min_length=3, description="Der/die Autor(en) des Lehrbuchs."
    )
    auflage: Optional[str] = Field(None, description="Die Auflage, z.B. '11. Aufl.'")
    jahr: int = Field(..., gt=1800, lt=2100, description="Das Erscheinungsjahr.")
    verlag: Optional[str] = Field(None, description="Der Verlag des Buches.")
    isbn: Optional[str] = Field(None, description="Die ISBN des Buches.")
    rechtsgebiet: str = Field(..., description="Haupt-Rechtsgebiet, z.B. 'Zivilrecht'.")
    dateiname: Optional[str] = Field(
        None, description="Der Dateiname der PDF-Datei, z.B. 'Medicus_BGB_AT.pdf'."
    )


class BookMetadataCreate(BookMetadataBase):
    """Modell zum Hinzufügen eines neuen Buches."""

    pass


class BookMetadata(BookMetadataBase):
    """Vollständiges Modell eines Buches, wie es von der API zurückgegeben wird."""

    id: str = Field(
        ..., description="Eine eindeutige, aus Autor/Titel/Jahr generierte ID."
    )
    hinzugefuegt_am: datetime = Field(..., description="Zeitstempel der Erstellung.")
    aktualisiert_am: datetime = Field(
        ..., description="Zeitstempel der letzten Aktualisierung."
    )
    datei_hash: Optional[str] = Field(
        None, description="SHA256-Hash der zugehörigen Datei."
    )
    chunk_anzahl: int = Field(
        0, description="Anzahl der Text-Chunks, in die das Buch zerlegt wurde."
    )

    class Config:
        from_attributes = True  # Erlaubt die Erstellung des Modells aus ORM-Objekten

    @field_validator("id")
    @classmethod
    def valid_id(cls, v: str) -> str:
        """Stellt sicher, dass die ID keine ungültigen Zeichen enthält."""
        if not re.match(r"^[a-z0-9_]+$", v):
            raise ValueError(
                "ID darf nur Kleinbuchstaben, Zahlen und Unterstriche enthalten."
            )
        return v
