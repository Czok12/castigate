1. **Einzelnutzer-Fokus:** Die gesamte Architektur ist für eine private Einzelnutzer-Anwendung mit optionalem NAS-Zugriff optimiert.
2. **Transparenz über Magie:** Implementiere Funktionen so, dass der Nutzer den Weg von der Frage zur Antwort nachvollziehen kann (Explainable AI). Vermeide undurchsichtige "Blackbox"-Logik.
3. **Strikte Quellenbindung:** Das System darf **niemals** Informationen erfinden. Jede Aussage in einer Antwort muss auf die bereitgestellten Quellen zurückführbar sein. Baue Sicherheitsmechanismen gegen Halluzinationen ein.

### **Python Backend (`/backend`)**

5. **Strikte Typisierung mit Mypy:** Jede Funktion, Variable und jeder Rückgabewert muss mit Type-Hints versehen sein. Dein Code muss Mypy-konform sein (`--strict` mode).
6. **Code-Formatierung mit Black:** Halte dich ausnahmslos an das `black`-Format (Zeilenlänge 88).
7. **Linting mit Ruff:** Generiere Code, der den in `pyproject.toml` definierten `ruff`-Regeln entspricht. Korrigiere aktiv Linting-Fehler.
8. **Modulare Struktur beachten:** Platziere Code in der korrekten Datei gemäß der Projektstruktur (`api/`, `core/`, `utils/`). Die `core`-Logik darf keine Kenntnis von der `api`-Schicht haben.
9. **Zentrale Konfiguration nutzen:** Hartkodierte Pfade oder Werte sind verboten. Importiere alle Konfigurationen aus `dango_ki/utils/config.py`.
10. **Tests sind obligatorisch:** Für jede neue Funktion in `dango_ki/core/` oder `dango_ki/utils/` musst du einen entsprechenden Unit-Test im `tests/`-Verzeichnis erstellen. Nutze `pytest` und mocke externe Abhängigkeiten.

### **TypeScript Frontend (`/frontend`)**

11. **Strikte Typisierung:** Definiere `interface` oder `type` für alle Props, Zustände und API-Antwort-Objekte. Die `any`-Typisierung ist nur in Ausnahmefällen mit Begründung erlaubt.
12. **Komponenten-Modularität:** Erstelle kleine, wiederverwendbare React-Komponenten im `/components`-Verzeichnis, die jeweils nur eine Aufgabe erfüllen.
13. **API-Kommunikation kapseln:** Alle `fetch`-Aufrufe an das Backend müssen in dedizierten Funktionen innerhalb von `/src/api/` gekapselt werden.

### **Allgemeine Anweisungen**

14. **Sicherheit priorisieren:** Da die Anwendung lokale Dateien liest und Prozesse startet, achte besonders auf die Absicherung von Dateipfaden und externen Prozessaufrufen.
15. **Dokumentation als Pflicht:** Erstelle für jede neue öffentliche Funktion oder Klasse eine prägnante Google-Style Docstring (Python) oder TSDoc-Kommentare (TypeScript).
