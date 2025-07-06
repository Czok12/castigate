Perfekt. Das Grundgerüst steht. Jetzt füllen wir es mit der ersten Kernfunktionalität: der Bibliotheksverwaltung. Dies ist ein entscheidender Schritt, da alle nachfolgenden Funktionen (Ingestion, RAG) auf den hier verwalteten Metadaten aufbauen.

Wir werden die Logik der alten `juristische_bibliothek_verwaltung.py` in eine saubere Service-Klasse überführen und die entsprechenden API-Endpunkte bereitstellen.

---

### Prompt 2: Bibliotheks-Service integrieren

**Ziel:** Das Backend um eine solide Verwaltung für die juristische Bibliothek erweitern, inklusive Metadaten in einer SQLite-Datenbank. Das System soll wissen, welche Bücher existieren.

**Kontext:** Wir haben ein funktionierendes API-Grundgerüst. Jetzt integrieren wir die Logik zur Verwaltung der Lehrbücher.

**Aufgaben:**

1. **Abhängigkeiten hinzufügen:** `python-dateutil` für eine robustere Datumsverarbeitung.
2. **API-Modelle definieren:** Pydantic-Modelle für das Erstellen und Anzeigen von Büchern.
3. **Service-Schicht erstellen:** Kapsle die gesamte Datenbanklogik im `LibraryService`. Dieser Service wird für die Interaktion mit der `castigatio_bibliothek.db` zuständig sein.
4. **API-Endpunkte implementieren:** Erstelle Endpunkte zum Abrufen, Hinzufügen und Löschen von Büchern.
5. **Service in der `main.py` einbinden.**

**Anweisung:** Erstelle und aktualisiere die folgenden Dateien mit dem angegebenen Inhalt.

**1. Datei: `castigatio_backend/requirements.txt` (Anpassen)**

Füge `python-dateutil` hinzu. Diese Bibliothek hilft beim zuverlässigen Parsen von Datum-Strings.

```python
# castigatio_backend/requirements.txt
fastapi
uvicorn[standard]
pydantic
python-dotenv
python-dateutil
```

**2. Datei: `castigatio_backend/app/models/library.py`**

Diese Datei definiert die Datenstrukturen für unsere API. Wir trennen zwischen dem, was der Nutzer zum Erstellen (`Create`) sendet, und dem, was das System zurückgibt (`BookMetadata`).

```python
# castigatio_backend/app/models/library.py
import re
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

class BookMetadataBase(BaseModel):
    """Grundlegende Metadaten für ein Lehrbuch."""
    titel: str = Field(..., min_length=3, description="Der Titel des Lehrbuchs.")
    autor: str = Field(..., min_length=3, description="Der/die Autor(en) des Lehrbuchs.")
    auflage: Optional[str] = Field(None, description="Die Auflage, z.B. '11. Aufl.'")
    jahr: int = Field(..., gt=1800, lt=2100, description="Das Erscheinungsjahr.")
    verlag: Optional[str] = Field(None, description="Der Verlag des Buches.")
    isbn: Optional[str] = Field(None, description="Die ISBN des Buches.")
    rechtsgebiet: str = Field(..., description="Haupt-Rechtsgebiet, z.B. 'Zivilrecht'.")
    dateiname: Optional[str] = Field(None, description="Der Dateiname der PDF-Datei, z.B. 'Medicus_BGB_AT.pdf'.")

class BookMetadataCreate(BookMetadataBase):
    """Modell zum Hinzufügen eines neuen Buches."""
    pass

class BookMetadata(BookMetadataBase):
    """Vollständiges Modell eines Buches, wie es von der API zurückgegeben wird."""
    id: str = Field(..., description="Eine eindeutige, aus Autor/Titel/Jahr generierte ID.")
    hinzugefuegt_am: datetime = Field(..., description="Zeitstempel der Erstellung.")
    aktualisiert_am: datetime = Field(..., description="Zeitstempel der letzten Aktualisierung.")
    datei_hash: Optional[str] = Field(None, description="SHA256-Hash der zugehörigen Datei.")
    chunk_anzahl: int = Field(0, description="Anzahl der Text-Chunks, in die das Buch zerlegt wurde.")

    class Config:
        from_attributes = True # Erlaubt die Erstellung des Modells aus ORM-Objekten

    @field_validator('id')
    @classmethod
    def valid_id(cls, v: str) -> str:
        """Stellt sicher, dass die ID keine ungültigen Zeichen enthält."""
        if not re.match(r'^[a-z0-9_]+$', v):
            raise ValueError('ID darf nur Kleinbuchstaben, Zahlen und Unterstriche enthalten.')
        return v
```

**3. Datei: `castigatio_backend/app/services/library_service.py`**

Dies ist die Kernlogik für die Bibliotheksverwaltung.

```python
# castigatio_backend/app/services/library_service.py
import sqlite3
import hashlib
import re
from datetime import datetime
from typing import List, Dict, Optional, Any
from dateutil.parser import isoparse

from app.core.config import LIBRARY_DB_PATH
from app.models.library import BookMetadata, BookMetadataCreate

class LibraryService:
    def __init__(self, db_path: str = str(LIBRARY_DB_PATH)):
        self.db_path = db_path
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self):
        """Erstellt die Datenbanktabelle, falls sie nicht existiert."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS lehrbuecher (
                    id TEXT PRIMARY KEY,
                    titel TEXT NOT NULL,
                    autor TEXT NOT NULL,
                    auflage TEXT,
                    jahr INTEGER NOT NULL,
                    verlag TEXT,
                    isbn TEXT,
                    rechtsgebiet TEXT,
                    dateiname TEXT,
                    hinzugefuegt_am TEXT NOT NULL,
                    aktualisiert_am TEXT NOT NULL,
                    datei_hash TEXT,
                    chunk_anzahl INTEGER DEFAULT 0
                )
            """)
            conn.commit()

    def _generate_book_id(self, autor: str, titel: str, jahr: int) -> str:
        """Generiert eine saubere, eindeutige ID aus Buchdaten."""
        # Nimmt die ersten 10 Zeichen von Autor und Titel
        autor_part = autor.split('/')[0].split(',')[0].strip()
        titel_part = titel.strip()

        combined = f"{autor_part} {titel_part} {jahr}".lower()
        # Ersetzt alles, was kein Buchstabe oder eine Zahl ist, durch einen Unterstrich
        sanitized = re.sub(r'[^a-z0-9]+', '_', combined)
        # Entfernt führende/nachfolgende Unterstriche und reduziert mehrere zu einem
        clean_id = re.sub(r'_+', '_', sanitized).strip('_')
        return clean_id[:60] # Kürzt auf max. 60 Zeichen

    def _map_row_to_model(self, row: sqlite3.Row) -> BookMetadata:
        """Konvertiert eine Datenbankzeile in ein Pydantic-Modell."""
        return BookMetadata(
            id=row['id'],
            titel=row['titel'],
            autor=row['autor'],
            auflage=row['auflage'],
            jahr=row['jahr'],
            verlag=row['verlag'],
            isbn=row['isbn'],
            rechtsgebiet=row['rechtsgebiet'],
            dateiname=row['dateiname'],
            hinzugefuegt_am=isoparse(row['hinzugefuegt_am']),
            aktualisiert_am=isoparse(row['aktualisiert_am']),
            datei_hash=row['datei_hash'],
            chunk_anzahl=row['chunk_anzahl']
        )

    def get_all_books(self) -> List[BookMetadata]:
        """Ruft alle Bücher aus der Datenbank ab."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM lehrbuecher ORDER BY autor, jahr DESC")
            rows = cursor.fetchall()
            return [self._map_row_to_model(row) for row in rows]

    def get_book_by_id(self, book_id: str) -> Optional[BookMetadata]:
        """Ruft ein einzelnes Buch anhand seiner ID ab."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM lehrbuecher WHERE id = ?", (book_id,))
            row = cursor.fetchone()
            return self._map_row_to_model(row) if row else None

    def add_book(self, book_data: BookMetadataCreate) -> BookMetadata:
        """Fügt ein neues Buch zur Datenbank hinzu."""
        book_id = self._generate_book_id(book_data.autor, book_data.titel, book_data.jahr)
        if self.get_book_by_id(book_id):
            raise ValueError(f"Ein Buch mit der ID '{book_id}' existiert bereits.")

        now_iso = datetime.now().isoformat()

        book_dict = book_data.model_dump()
        book_dict['id'] = book_id
        book_dict['hinzugefuegt_am'] = now_iso
        book_dict['aktualisiert_am'] = now_iso
        book_dict['chunk_anzahl'] = 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO lehrbuecher (id, titel, autor, auflage, jahr, verlag, isbn, rechtsgebiet, dateiname, hinzugefuegt_am, aktualisiert_am, chunk_anzahl)
                VALUES (:id, :titel, :autor, :auflage, :jahr, :verlag, :isbn, :rechtsgebiet, :dateiname, :hinzugefuegt_am, :aktualisiert_am, :chunk_anzahl)
            """, book_dict)
            conn.commit()

        # Lese das gerade eingefügte Buch, um das vollständige Modell zurückzugeben
        return self.get_book_by_id(book_id)

    def delete_book(self, book_id: str) -> bool:
        """Löscht ein Buch aus der Datenbank."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM lehrbuecher WHERE id = ?", (book_id,))
            conn.commit()
            return cursor.rowcount > 0

# Singleton-Instanz des Services
library_service = LibraryService()
```

**4. Datei: `castigatio_backend/app/api/library.py`**

Dies ist die "Übersetzungsschicht" zwischen HTTP-Anfragen und unserem `LibraryService`.

```python
# castigatio_backend/app/api/library.py
from typing import List
from fastapi import APIRouter, HTTPException, status
from app.models.library import BookMetadata, BookMetadataCreate
from app.services.library_service import library_service

router = APIRouter()

@router.get("/books", response_model=List[BookMetadata], summary="Alle Bücher abrufen", tags=["Bibliothek"])
async def get_all_books():
    """Ruft eine Liste aller Lehrbücher in der Bibliothek ab, sortiert nach Autor und Jahr."""
    return library_service.get_all_books()

@router.post("/books", response_model=BookMetadata, status_code=status.HTTP_201_CREATED, summary="Neues Buch hinzufügen", tags=["Bibliothek"])
async def add_new_book(book_data: BookMetadataCreate):
    """Fügt ein neues Lehrbuch zur Bibliothek hinzu."""
    try:
        return library_service.add_book(book_data)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))

@router.get("/books/{book_id}", response_model=BookMetadata, summary="Ein Buch abrufen", tags=["Bibliothek"])
async def get_book(book_id: str):
    """Ruft die Metadaten für ein spezifisches Buch anhand seiner ID ab."""
    book = library_service.get_book_by_id(book_id)
    if not book:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Buch nicht gefunden.")
    return book

@router.delete("/books/{book_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Ein Buch löschen", tags=["Bibliothek"])
async def delete_book(book_id: str):
    """Löscht ein Buch und die zugehörigen Vektor-Daten (letzteres wird später implementiert)."""
    if not library_service.delete_book(book_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Buch nicht gefunden.")
    return None # Bei 204 No Content wird kein Body zurückgegeben
```

**5. Datei: `castigatio_backend/app/main.py` (Anpassen)**

Wir binden den neuen Bibliotheks-Router in unsere Hauptanwendung ein.

```python
# castigatio_backend/app/main.py
from fastapi import FastAPI
from app.api import status, library # <-- Importiere den neuen Router

app = FastAPI(
    title="🏛️ Castigatio - Juristische Wissensdatenbank",
    description="Das Backend für die juristische RAG-Anwendung mit erweiterten KI-Funktionen.",
    version="0.2.0",
)

# Binde die API-Router ein
app.include_router(status.router, prefix="/api/v1")
app.include_router(library.router, prefix="/api/v1") # <-- Binde den Bibliotheks-Router ein

@app.get("/", summary="Root-Endpunkt", tags=["System"])
async def read_root():
    """Ein einfacher Willkommens-Endpunkt."""
    return {"message": "Willkommen beim Backend von Castigatio!"}
```

**Überprüfung:**

1. Führen Sie `pip install -r castigatio_backend/requirements.txt` aus, falls Sie es noch nicht getan haben.
2. Starten Sie den Server neu: `uvicorn app.main:app --reload`.
3. Öffnen Sie die API-Dokumentation unter **http://localhost:8000/docs**.
4. Testen Sie die neuen Endpunkte unter dem Tag "Bibliothek":
   - `POST /books` um ein neues Buch zu erstellen.
   - `GET /books` um zu sehen, ob es hinzugefügt wurde.
   - `GET /books/{book_id}` um das spezifische Buch abzurufen.
   - `DELETE /books/{book_id}` um es wieder zu löschen.

Wenn diese Schritte funktionieren, haben wir erfolgreich eine robuste Basis für die Verwaltung unserer juristischen Texte geschaffen. Wir sind bereit für den nächsten Schritt: das Ingestion-System.
