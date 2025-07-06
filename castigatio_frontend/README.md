# # Castigatio Frontend

Eine moderne Desktop-Anwendung für juristische Wissensdatenbanken, entwickelt mit Tauri, React und TypeScript.

## Features

- 📚 **Bibliotheksverwaltung**: Verwalten Sie Ihre juristischen PDF-Dokumente
- 💬 **Intelligente Suche**: Stellen Sie Fragen und erhalten Sie KI-gestützte Antworten
- 📤 **Upload-System**: Einfaches Hinzufügen neuer PDFs zur Bibliothek
- 🔍 **Systemstatus**: Überwachen Sie die Gesundheit des Backends
- 🎨 **Modernes UI**: Dunkles Theme mit Tailwind CSS

## Technologien

- **Framework**: Tauri 2.0 (Rust + Web)
- **Frontend**: React 18 + TypeScript
- **Styling**: Tailwind CSS
- **Build Tool**: Vite
- **Desktop**: Native Desktop-App für Windows, macOS, Linux

## Installation

### Voraussetzungen

- Node.js (Version 18+)
- Rust (für Tauri)
- npm oder yarn

### Setup

1. Dependencies installieren:

```bash
cd castigatio_frontend
npm install
```

2. Tauri dependencies installieren:

```bash
npm run tauri -- deps
```

### Entwicklung

1. Backend starten (in einem separaten Terminal):

```bash
cd ../castigatio_backend
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

2. Frontend im Entwicklungsmodus starten:

```bash
npm run tauri dev
```

Das öffnet automatisch ein Desktop-Fenster mit der Anwendung.

### Build

Für eine Production-Build:

```bash
npm run tauri build
```

Die erstellte Desktop-App finden Sie im `src-tauri/target/release/bundle/` Verzeichnis.auri + React + Typescript

This template should help get you started developing with Tauri, React and Typescript in Vite.

## Recommended IDE Setup

- [VS Code](https://code.visualstudio.com/) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)
