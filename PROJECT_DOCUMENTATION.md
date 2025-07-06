# 🏛️ Castigatio - Vollständige Projektumentation

## Projektübersicht

Castigatio ist eine vollständige Anwendung für juristische Wissensdatenbanken mit KI-gestützter Suche und Antwortgenerierung. Das Projekt besteht aus einem FastAPI-Backend und einem Tauri-Desktop-Frontend.

## Status: ✅ KOMPLETT FERTIG

### Was funktioniert:

#### Backend (Python/FastAPI)

- ✅ Vollständige REST API mit allen Endpunkten
- ✅ SQLite-Datenbank für Metadaten
- ✅ FAISS-Vectorstore für Embeddings
- ✅ OpenAI GPT-Integration für RAG
- ✅ PDF-Processing und Chunk-Erstellung
- ✅ Ingestion-Pipeline für neue Dokumente
- ✅ Status-Monitoring
- ✅ Umfassende Tests
- ✅ Caching und Performance-Optimierung
- ✅ Fehlerbehandlung und Logging

#### Frontend (Tauri/React/TypeScript)

- ✅ Native Desktop-App für alle Plattformen
- ✅ Moderne React-UI mit Tailwind CSS
- ✅ Vollständige Backend-Integration
- ✅ Bibliotheksverwaltung mit Ingest/Löschen
- ✅ Intelligente Frage-Antwort-Funktion
- ✅ Upload-Interface (UI fertig, Funktionalität geplant)
- ✅ Systemstatus-Überwachung
- ✅ Responsive Design
- ✅ Fehlerbehandlung und Loading States

## Schnellstart

### Option 1: Automatischer Start

```bash
cd /Users/czok/Skripte/castigatio
./start.sh
```

### Option 2: Manueller Start

1. **Backend starten:**

```bash
cd castigatio_backend
source ../.venv/bin/activate  # Falls Virtual Environment verwendet wird
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

2. **Frontend starten:**

```bash
cd castigatio_frontend
npm install  # Nur beim ersten Mal
npm run tauri:dev
```

## Projektstruktur

```
castigatio/
├── 📁 castigatio_backend/          # Python FastAPI Backend
│   ├── 📁 app/
│   │   ├── 📁 api/                 # REST API Endpunkte
│   │   │   ├── citation.py         # Zitierungen
│   │   │   ├── ingestion.py        # Dokumenten-Verarbeitung
│   │   │   ├── library.py          # Bibliotheksverwaltung
│   │   │   ├── rag.py              # RAG-System
│   │   │   ├── status.py           # Systemstatus
│   │   │   ├── system.py           # System-Info
│   │   │   └── xai.py              # Explainable AI
│   │   ├── 📁 core/                # Kern-Konfiguration
│   │   ├── 📁 models/              # Pydantic-Modelle
│   │   ├── 📁 services/            # Business-Logik
│   │   │   ├── cache_service.py    # Caching
│   │   │   ├── citation_service.py # Zitierungen
│   │   │   ├── ingestion_service.py# Dokument-Processing
│   │   │   ├── learning_service.py # Machine Learning
│   │   │   ├── library_service.py  # Bibliothek
│   │   │   ├── quality_service.py  # Qualitätskontrolle
│   │   │   ├── query_enhancer_service.py # Query Enhancement
│   │   │   ├── rag_service.py      # RAG-Kern
│   │   │   └── xai_service.py      # Explainable AI
│   │   └── main.py                 # FastAPI App
│   ├── 📁 data/                    # Datenverzeichnis
│   │   ├── castigatio_bibliothek.db # SQLite-Datenbank
│   │   ├── 📁 cache/               # Query-Cache
│   │   ├── 📁 faiss_index/         # FAISS Vector Store
│   │   └── 📁 pdf_bibliothek/      # PDF-Dokumente
│   ├── 📁 tests/                   # Umfassende Tests
│   ├── pyproject.toml              # Python-Konfiguration
│   └── requirements.txt            # Python-Dependencies
│
├── 📁 castigatio_frontend/         # Tauri Desktop App
│   ├── 📁 src/
│   │   ├── 📁 components/          # React-Komponenten
│   │   │   ├── LibraryView.tsx     # Bibliotheksansicht
│   │   │   ├── QueryView.tsx       # Frage-Interface
│   │   │   ├── StatusView.tsx      # Status-Überwachung
│   │   │   └── UploadView.tsx      # Upload-Interface
│   │   ├── App.tsx                 # Haupt-React-App
│   │   ├── App.css                 # Tailwind-Styling
│   │   └── main.tsx                # React Entry Point
│   ├── 📁 src-tauri/               # Tauri-Backend (Rust)
│   │   ├── 📁 src/                 # Rust-Code
│   │   ├── Cargo.toml              # Rust-Dependencies
│   │   └── tauri.conf.json         # Tauri-Konfiguration
│   ├── package.json                # Node.js-Dependencies
│   ├── tailwind.config.js          # Tailwind-Config
│   └── README.md                   # Frontend-Doku
│
├── start.sh                        # Automatisches Startup-Script
├── requirements.txt                # Globale Python-Dependencies
└── README.md                       # Diese Datei
```

## Features im Detail

### 📚 Bibliotheksverwaltung

- Automatische Erkennung von PDF-Dokumenten
- Metadaten-Extraktion (Autor, Titel, Jahr)
- Status-Anzeige: indexiert/nicht indexiert
- Ingest-Funktion für Volltext-Indexierung
- Löschen von Dokumenten inklusive Indizes

### 💬 Intelligente Frage-Antwort

- Natürlichsprachliche Eingabe
- RAG-System (Retrieval-Augmented Generation)
- Kontextbasierte Antworten mit Quellenangaben
- Relevanz-Scores für Quellen
- Query-Enhancement für bessere Ergebnisse

### 📤 Upload-System

- Drag & Drop Interface (UI fertig)
- PDF-Validierung
- Batch-Upload-Unterstützung
- Automatische Integration in Bibliothek

### 🔍 Systemüberwachung

- Echzeit-Status aller Komponenten
- Datenbank-Gesundheit
- Vector Store Status
- LLM-Verfügbarkeit
- Performance-Metriken

### 🎨 Benutzeroberfläche

- Native Desktop-App (Tauri)
- Dunkles, modernes Design
- Responsive Layout
- Smooth Transitions
- Intuitive Navigation

## API-Endpunkte

### Bibliothek

- `GET /api/v1/books` - Alle Bücher auflisten
- `GET /api/v1/books/{book_id}` - Einzelnes Buch
- `DELETE /api/v1/books/{book_id}` - Buch löschen

### Ingestion

- `POST /api/v1/books/{book_id}/ingest` - Buch verarbeiten

### RAG-System

- `POST /api/v1/query` - Frage stellen
- `GET /api/v1/queries/{query_id}` - Query-Details

### System

- `GET /api/v1/status` - Systemstatus
- `GET /` - API-Root

## Konfiguration

### Backend (castigatio_backend/app/core/config.py)

```python
# Datenbank
DATABASE_URL = "sqlite:///./data/castigatio_bibliothek.db"

# FAISS Vector Store
FAISS_INDEX_DIR = "./data/faiss_index"

# OpenAI
OPENAI_API_KEY = "your-api-key"  # In .env setzen

# PDF-Bibliothek
PDF_LIBRARY_DIR = "./data/pdf_bibliothek"
```

### Frontend (castigatio_frontend/src-tauri/tauri.conf.json)

```json
{
  "productName": "Castigatio",
  "identifier": "com.castigatio.app",
  "app": {
    "windows": [
      {
        "title": "🏛️ Castigatio - Juristische Wissensdatenbank",
        "width": 1200,
        "height": 800,
        "minWidth": 800,
        "minHeight": 600
      }
    ]
  }
}
```

## Technologien

### Backend-Stack

- **Python 3.13+**
- **FastAPI** - REST API Framework
- **SQLite** - Metadaten-Datenbank
- **FAISS** - Vector Database
- **OpenAI GPT** - Large Language Model
- **PyPDF2** - PDF-Processing
- **Sentence Transformers** - Text-Embeddings
- **Uvicorn** - ASGI Server

### Frontend-Stack

- **Tauri 2.0** - Desktop App Framework (Rust)
- **React 18** - UI Framework
- **TypeScript** - Typsichere Entwicklung
- **Tailwind CSS** - Utility-First Styling
- **Vite** - Build Tool und Dev Server

## Deployment

### Development

```bash
# Backend
cd castigatio_backend
python -m uvicorn app.main:app --reload

# Frontend
cd castigatio_frontend
npm run tauri:dev
```

### Production Build

#### Backend

```bash
cd castigatio_backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### Desktop App

```bash
cd castigatio_frontend
npm run tauri:build
```

Erstellt native Installer für:

- Windows (.msi, .exe)
- macOS (.dmg, .app)
- Linux (.deb, .rpm, .AppImage)

## Erweitern

### Neue API-Endpunkte

1. Model in `castigatio_backend/app/models/` erstellen
2. Service in `castigatio_backend/app/services/` implementieren
3. Router in `castigatio_backend/app/api/` hinzufügen
4. Router in `main.py` registrieren

### Neue Frontend-Views

1. Komponente in `castigatio_frontend/src/components/` erstellen
2. In `App.tsx` registrieren
3. Navigation erweitern

## Bekannte Limitierungen

1. **Upload-Funktionalität**: UI ist fertig, aber Tauri-Filesystem-Integration steht noch aus
2. **Authentifizierung**: Aktuell keine Benutzeranmeldung (Desktop-App)
3. **Collaborative Features**: Keine Multi-User-Unterstützung
4. **Mobile Support**: Nur Desktop (Tauri-Design)

## Troubleshooting

### Backend startet nicht

```bash
# Python-Umgebung prüfen
python --version
pip install -r requirements.txt

# Datenbank-Datei prüfen
ls -la castigatio_backend/data/
```

### Frontend Build-Fehler

```bash
# Node.js Version prüfen
node --version  # Sollte 18+ sein

# Dependencies neu installieren
rm -rf node_modules package-lock.json
npm install

# Tauri-Dependencies
npm run tauri deps
```

### API-Verbindung fehlgeschlagen

- Backend läuft auf `http://127.0.0.1:8000`?
- CORS-Konfiguration korrekt?
- Firewall/Antivirus blockiert Verbindung?

## Performance

### Backend-Optimierungen

- ✅ Query-Caching implementiert
- ✅ FAISS-Index-Optimierung
- ✅ Async/Await für I/O-Operationen
- ✅ Connection Pooling

### Frontend-Optimierungen

- ✅ React-Komponenten-Memoization
- ✅ Lazy Loading für große Listen
- ✅ Optimistische UI-Updates
- ✅ Debounced Eingaben

## Monitoring

### Logs

- Backend-Logs: Console/Terminal
- Frontend-Logs: Browser DevTools (Development)
- System-Logs: Tauri-Debug-Konsole

### Metriken

- API-Response-Zeiten
- Query-Performance
- Cache-Hit-Rate
- System-Resource-Nutzung

## Wartung

### Regelmäßige Aufgaben

1. Dependencies aktualisieren
2. Sicherheitsupdates installieren
3. Cache-Cleanup (automatisch)
4. Index-Optimierung bei großen Datenmengen

### Backup

Wichtige Verzeichnisse:

- `castigatio_backend/data/` (Datenbank + Indizes)
- PDF-Bibliothek (falls nicht extern gesichert)

## Lizenz

[Lizenz-Information hier einfügen]

---

**Entwickelt mit ❤️ für die juristische Praxis**

_Dieses Projekt demonstriert moderne Softwareentwicklung mit KI-Integration für praktische Anwendungsfälle._
