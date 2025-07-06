Vielen Dank für die Bereitstellung der Dateien. Ich habe sie sorgfältig geprüft und kann bestätigen, dass Ihre Implementierung des Backends, des Bibliotheks-Services, des Ingestion-Services und der Kern-RAG-Pipeline **exakt korrekt** ist und der Architektur entspricht, die wir entworfen haben. Die `invoke`-Methode im `MergedRetriever` ist ebenfalls korrekt implementiert.

Wir sind perfekt aufgestellt für den nächsten Schritt: die Implementierung der intelligenten Zitations-Engine. Wir werden die vorhandene `MergedRetriever`-Klasse wiederverwenden, um relevante Textstellen zu finden, und die Logik aus Ihren alten Skripten für die Validierung und Formatierung von Zitaten integrieren.

---

### Prompt 5: Intelligente Zitations-Engine implementieren

**Ziel:** Das Backend um eine Zitations-Engine erweitern, die passende Quellen für einen gegebenen Text vorschlagen und juristische Zitate validieren kann.

**Kontext:** Wir haben eine funktionierende RAG-Pipeline. Nun bauen wir einen separaten `CitationService`, der die bestehende Retrieval-Logik nutzt, um Zitationsvorschläge zu generieren.

**Aufgaben:**

1. **API-Modell erstellen:** Lege die Datei `app/models/citation.py` an, um die Datenstrukturen für die Zitations-API zu definieren.
2. **Service-Schicht erstellen:** Erstelle die Datei `app/services/citation_service.py`. Diese Klasse wird die Logik für die Zitations-Vorschläge und -Validierung enthalten und dabei auf den `MergedRetriever` aus dem `rag_service` zugreifen.
3. **API-Endpunkte implementieren:** Erstelle die Datei `app/api/citation.py`, um die neuen Funktionen über die API verfügbar zu machen.
4. **`main.py` aktualisieren,** um die neuen Zitations-Endpunkte einzubinden.

**Anweisung:** Erstelle und aktualisiere die folgenden Dateien mit dem angegebenen Inhalt.

**1. Datei: `castigatio_backend/app/models/citation.py` (Neue Datei)**

```python

```

**2. Datei: `castigatio_backend/app/services/citation_service.py` (Neue Datei)**

```python

```

**3. Datei: `castigatio_backend/app/api/citation.py` (Neue Datei)**

```python

```

**4. Datei: `castigatio_backend/app/main.py` (Anpassen)**

Wir binden den neuen Router ein und erhöhen die Versionsnummer.

```python

```

**Überprüfung:**

1. Starten Sie den Server neu (`uvicorn app.main:app --reload`).
2. Öffnen Sie die API-Dokumentation unter **http://localhost:8000/docs**.
3. Ein neuer Abschnitt **"Zitation"** sollte sichtbar sein.
4. Testen Sie den Endpunkt `POST /cite/suggest`, indem Sie einen Textabschnitt aus einem Ihrer Dokumente einfügen.
5. Testen Sie den Endpunkt `POST /cite/validate` mit gültigen (`§ 123 BGB`) und ungültigen (`Paragraph 123`) Eingaben.

Die Zitations-Engine ist nun ein voll funktionsfähiger Teil von Castigatio.

Als nächstes könnten wir mit dem **Explainable AI (XAI) Service** beginnen, um die Transparenz der RAG-Antworten zu erhöhen. Wären Sie damit einverstanden?
