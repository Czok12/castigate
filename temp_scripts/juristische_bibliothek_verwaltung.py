#!/usr/bin/env python3
"""
Juristische Lehrbuch-Datenbank Verwaltungstool
Verwaltet Lehrbücher, Metadaten und bietet erweiterte Suchfunktionen
"""

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List


@dataclass
class LehrbuchMetadata:
    """Metadaten für ein juristisches Lehrbuch"""

    id: str
    titel: str
    autor: str
    auflage: str
    jahr: str
    verlag: str
    isbn: str
    rechtsgebiet: str
    seitenzahl: int
    sprache: str = "Deutsch"
    hinzugefuegt: str = ""
    aktualisiert: str = ""
    datei_pfad: str = ""
    datei_hash: str = ""
    chunk_anzahl: int = 0

    def __post_init__(self):
        if not self.hinzugefuegt:
            self.hinzugefuegt = datetime.now().isoformat()
        if not self.aktualisiert:
            self.aktualisiert = self.hinzugefuegt


class JuristischeBibliothekVerwaltung:
    """Verwaltung der juristischen Lehrbuch-Datenbank"""

    def __init__(self, db_pfad: str = "juristische_bibliothek.db"):
        self.db_pfad = db_pfad
        self.init_database()

    def init_database(self):
        """Initialisiert die SQLite-Datenbank"""
        conn = sqlite3.connect(self.db_pfad)
        cursor = conn.cursor()

        # Lehrbücher-Tabelle
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS lehrbuecher (
                id TEXT PRIMARY KEY,
                titel TEXT NOT NULL,
                autor TEXT NOT NULL,
                auflage TEXT,
                jahr TEXT,
                verlag TEXT,
                isbn TEXT,
                rechtsgebiet TEXT,
                seitenzahl INTEGER,
                sprache TEXT DEFAULT 'Deutsch',
                hinzugefuegt TEXT,
                aktualisiert TEXT,
                datei_pfad TEXT,
                datei_hash TEXT,
                chunk_anzahl INTEGER DEFAULT 0
            )
        """
        )

        # Rechtsbereiche-Tabelle
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS rechtsbereiche (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                beschreibung TEXT,
                uebergeordnet TEXT
            )
        """
        )

        # Standard-Rechtsbereiche einfügen
        standard_bereiche = [
            ("Zivilrecht", "Bürgerliches Recht, Vertragsrecht, Deliktsrecht", None),
            ("Strafrecht", "Allgemeines und Besonderes Strafrecht", None),
            ("Öffentliches Recht", "Verfassungsrecht, Verwaltungsrecht", None),
            ("Handelsrecht", "Gesellschaftsrecht, Kapitalmarktrecht", "Zivilrecht"),
            ("Arbeitsrecht", "Individual- und Kollektivarbeitsrecht", "Zivilrecht"),
            ("Familienrecht", "Ehe, Scheidung, Unterhaltsrecht", "Zivilrecht"),
            ("Erbrecht", "Gesetzliche und gewillkürte Erbfolge", "Zivilrecht"),
            ("Sachenrecht", "Eigentum, Besitz, Grundstücksrecht", "Zivilrecht"),
            ("Schuldrecht", "Allgemeines und Besonderes Schuldrecht", "Zivilrecht"),
            (
                "Verfassungsrecht",
                "Grundrechte, Staatsorganisation",
                "Öffentliches Recht",
            ),
            (
                "Verwaltungsrecht",
                "Allgemeines und Besonderes Verwaltungsrecht",
                "Öffentliches Recht",
            ),
            (
                "Steuerrecht",
                "Allgemeines und Besonderes Steuerrecht",
                "Öffentliches Recht",
            ),
            ("Europarecht", "EU-Recht, Europäische Grundrechte", "Öffentliches Recht"),
        ]

        for bereich in standard_bereiche:
            cursor.execute(
                """
                INSERT OR IGNORE INTO rechtsbereiche (name, beschreibung, uebergeordnet)
                VALUES (?, ?, ?)
            """,
                bereich,
            )

        conn.commit()
        conn.close()

    def lehrbuch_hinzufuegen(self, lehrbuch: LehrbuchMetadata) -> bool:
        """Fügt ein neues Lehrbuch zur Datenbank hinzu"""
        conn = sqlite3.connect(self.db_pfad)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO lehrbuecher VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    lehrbuch.id,
                    lehrbuch.titel,
                    lehrbuch.autor,
                    lehrbuch.auflage,
                    lehrbuch.jahr,
                    lehrbuch.verlag,
                    lehrbuch.isbn,
                    lehrbuch.rechtsgebiet,
                    lehrbuch.seitenzahl,
                    lehrbuch.sprache,
                    lehrbuch.hinzugefuegt,
                    lehrbuch.aktualisiert,
                    lehrbuch.datei_pfad,
                    lehrbuch.datei_hash,
                    lehrbuch.chunk_anzahl,
                ),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def lehrbuch_aktualisieren(self, lehrbuch_id: str, updates: Dict) -> bool:
        """Aktualisiert ein Lehrbuch"""
        conn = sqlite3.connect(self.db_pfad)
        cursor = conn.cursor()

        # Füge aktualisiert-Zeitstempel hinzu
        updates["aktualisiert"] = datetime.now().isoformat()

        set_clause = ", ".join([f"{key} = ?" for key in updates.keys()])
        values = list(updates.values()) + [lehrbuch_id]

        cursor.execute(
            f"""
            UPDATE lehrbuecher 
            SET {set_clause}
            WHERE id = ?
        """,
            values,
        )

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return success

    def lehrbuch_loeschen(self, lehrbuch_id: str) -> bool:
        """Löscht ein Lehrbuch aus der Datenbank"""
        conn = sqlite3.connect(self.db_pfad)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM lehrbuecher WHERE id = ?", (lehrbuch_id,))
        success = cursor.rowcount > 0

        conn.commit()
        conn.close()

        return success

    def lehrbuch_suchen(self, suchkriterien: Dict) -> List[Dict]:
        """Sucht Lehrbücher nach verschiedenen Kriterien"""
        conn = sqlite3.connect(self.db_pfad)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        where_clauses = []
        values = []

        for field, value in suchkriterien.items():
            if value:
                if field in ["titel", "autor", "verlag", "rechtsgebiet"]:
                    where_clauses.append(f"{field} LIKE ?")
                    values.append(f"%{value}%")
                else:
                    where_clauses.append(f"{field} = ?")
                    values.append(value)

        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        cursor.execute(
            f"""
            SELECT * FROM lehrbuecher 
            WHERE {where_clause}
            ORDER BY autor, titel
        """,
            values,
        )

        result = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return result

    def alle_lehrbuecher(self) -> List[Dict]:
        """Gibt alle Lehrbücher zurück"""
        return self.lehrbuch_suchen({})

    def rechtsbereiche_abrufen(self) -> List[Dict]:
        """Gibt alle verfügbaren Rechtsbereiche zurück"""
        conn = sqlite3.connect(self.db_pfad)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM rechtsbereiche ORDER BY name")
        result = [dict(row) for row in cursor.fetchall()]

        conn.close()
        return result

    def statistiken_abrufen(self) -> Dict:
        """Gibt Statistiken über die Bibliothek zurück"""
        conn = sqlite3.connect(self.db_pfad)
        cursor = conn.cursor()

        stats = {}

        # Gesamtzahl Bücher
        cursor.execute("SELECT COUNT(*) FROM lehrbuecher")
        stats["gesamtzahl_buecher"] = cursor.fetchone()[0]

        # Bücher pro Rechtsgebiet
        cursor.execute(
            """
            SELECT rechtsgebiet, COUNT(*) as anzahl 
            FROM lehrbuecher 
            GROUP BY rechtsgebiet 
            ORDER BY anzahl DESC
        """
        )
        stats["nach_rechtsgebiet"] = dict(cursor.fetchall())

        # Neueste Bücher
        cursor.execute(
            """
            SELECT titel, autor, hinzugefuegt 
            FROM lehrbuecher 
            ORDER BY hinzugefuegt DESC 
            LIMIT 5
        """
        )
        stats["neueste_buecher"] = cursor.fetchall()

        # Gesamtzahl Chunks
        cursor.execute("SELECT SUM(chunk_anzahl) FROM lehrbuecher")
        stats["gesamtzahl_chunks"] = cursor.fetchone()[0] or 0

        conn.close()
        return stats

    def datei_hash_berechnen(self, datei_pfad: str) -> str:
        """Berechnet SHA-256 Hash einer Datei"""
        if not os.path.exists(datei_pfad):
            return ""

        sha256_hash = hashlib.sha256()
        with open(datei_pfad, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def generiere_lehrbuch_id(self, autor: str, titel: str, jahr: str) -> str:
        """Generiert eine eindeutige ID für ein Lehrbuch"""
        basis = f"{autor}_{titel}_{jahr}".lower()
        # Entferne Sonderzeichen und ersetze Leerzeichen
        basis = "".join(c for c in basis if c.isalnum() or c == "_")
        return basis.replace(" ", "_")[:50]

    def export_zu_json(self, datei_pfad: str) -> bool:
        """Exportiert die Bibliothek als JSON"""
        try:
            data = {
                "export_datum": datetime.now().isoformat(),
                "lehrbuecher": self.alle_lehrbuecher(),
                "rechtsbereiche": self.rechtsbereiche_abrufen(),
                "statistiken": self.statistiken_abrufen(),
            }

            with open(datei_pfad, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return True
        except (IOError, OSError):
            return False


def main():
    """Beispiel-Nutzung der Bibliotheksverwaltung"""
    bibliothek = JuristischeBibliothekVerwaltung()

    print("=== Juristische Bibliotheksverwaltung ===\n")

    # Beispiel-Lehrbuch hinzufügen
    beispiel_buch = LehrbuchMetadata(
        id="medicus_petersen_bgr_2019",
        titel="Grundwissen zum Bürgerlichen Recht",
        autor="Medicus/Petersen",
        auflage="11. Aufl.",
        jahr="2019",
        verlag="C.H.Beck",
        isbn="978-3-406-73593-9",
        rechtsgebiet="Zivilrecht",
        seitenzahl=450,
        datei_pfad="/path/to/medicus_bgr.pdf",
    )

    # Hash der Datei berechnen (wenn Datei existiert)
    if os.path.exists(beispiel_buch.datei_pfad):
        beispiel_buch.datei_hash = bibliothek.datei_hash_berechnen(
            beispiel_buch.datei_pfad
        )

    # Buch hinzufügen
    if bibliothek.lehrbuch_hinzufuegen(beispiel_buch):
        print("✓ Beispiel-Lehrbuch erfolgreich hinzugefügt")
    else:
        print("! Beispiel-Lehrbuch bereits vorhanden")

    # Statistiken anzeigen
    print("\n=== Bibliotheks-Statistiken ===")
    stats = bibliothek.statistiken_abrufen()
    print(f"Gesamtzahl Bücher: {stats['gesamtzahl_buecher']}")
    print(f"Gesamtzahl Chunks: {stats['gesamtzahl_chunks']}")

    print("\nBücher nach Rechtsgebiet:")
    for gebiet, anzahl in stats["nach_rechtsgebiet"].items():
        print(f"  {gebiet}: {anzahl}")

    # Alle Bücher anzeigen
    print("\n=== Alle Lehrbücher ===")
    alle_buecher = bibliothek.alle_lehrbuecher()
    for buch in alle_buecher:
        print(f"- {buch['autor']}: {buch['titel']} ({buch['jahr']})")

    # Rechtsbereiche anzeigen
    print("\n=== Verfügbare Rechtsbereiche ===")
    bereiche = bibliothek.rechtsbereiche_abrufen()
    for bereich in bereiche:
        print(f"- {bereich['name']}: {bereich['beschreibung']}")

    # Export
    export_pfad = "bibliothek_export.json"
    if bibliothek.export_zu_json(export_pfad):
        print(f"\n✓ Bibliothek erfolgreich nach {export_pfad} exportiert")


if __name__ == "__main__":
    main()
