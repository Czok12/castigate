import re
import sqlite3
from datetime import datetime
from typing import List, Optional

from dateutil.parser import isoparse

from app.core.config import LIBRARY_DB_PATH
from app.models.library import BookMetadata, BookMetadataCreate


class LibraryService:
    def update_book_metadata(self, book_id: str, updates: dict) -> bool:
        """Aktualisiert spezifische Felder für ein Buch."""
        if not updates:
            return False

        updates["aktualisiert_am"] = datetime.now().isoformat()
        set_clause = ", ".join([f"{key} = ?" for key in updates.keys()])
        values = list(updates.values()) + [book_id]

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE lehrbuecher SET {set_clause} WHERE id = ?", values)
            conn.commit()
            return cursor.rowcount > 0

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
        autor_part = autor.split("/")[0].split(",")[0].strip()
        titel_part = titel.strip()

        combined = f"{autor_part} {titel_part} {jahr}".lower()
        # Ersetzt alles, was kein Buchstabe oder eine Zahl ist, durch einen Unterstrich
        sanitized = re.sub(r"[^a-z0-9]+", "_", combined)
        # Entfernt führende/nachfolgende Unterstriche und reduziert mehrere zu einem
        clean_id = re.sub(r"_+", "_", sanitized).strip("_")
        return clean_id[:60]  # Kürzt auf max. 60 Zeichen

    def _map_row_to_model(self, row: sqlite3.Row) -> BookMetadata:
        """Konvertiert eine Datenbankzeile in ein Pydantic-Modell."""
        return BookMetadata(
            id=row["id"],
            titel=row["titel"],
            autor=row["autor"],
            auflage=row["auflage"],
            jahr=row["jahr"],
            verlag=row["verlag"],
            isbn=row["isbn"],
            rechtsgebiet=row["rechtsgebiet"],
            dateiname=row["dateiname"],
            hinzugefuegt_am=isoparse(row["hinzugefuegt_am"]),
            aktualisiert_am=isoparse(row["aktualisiert_am"]),
            datei_hash=row["datei_hash"],
            chunk_anzahl=row["chunk_anzahl"],
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
        book_id = self._generate_book_id(
            book_data.autor, book_data.titel, book_data.jahr
        )
        if self.get_book_by_id(book_id):
            raise ValueError(f"Ein Buch mit der ID '{book_id}' existiert bereits.")

        now_iso = datetime.now().isoformat()

        book_dict = book_data.model_dump()
        book_dict["id"] = book_id
        book_dict["hinzugefuegt_am"] = now_iso
        book_dict["aktualisiert_am"] = now_iso
        book_dict["chunk_anzahl"] = 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO lehrbuecher (id, titel, autor, auflage, jahr, verlag, isbn, rechtsgebiet, dateiname, hinzugefuegt_am, aktualisiert_am, chunk_anzahl)
                VALUES (:id, :titel, :autor, :auflage, :jahr, :verlag, :isbn, :rechtsgebiet, :dateiname, :hinzugefuegt_am, :aktualisiert_am, :chunk_anzahl)
            """,
                book_dict,
            )
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
