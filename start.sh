#!/bin/bash

# Castigatio Startup Script
# Startet Backend und Frontend zusammen

echo "🏛️ Castigatio - Juristische Wissensdatenbank"
echo "=============================================="

# Prüfen ob wir im richtigen Verzeichnis sind
if [ ! -d "castigatio_backend" ] || [ ! -d "castigatio_frontend" ]; then
    echo "❌ Fehler: Dieses Script muss im Hauptverzeichnis des Projekts ausgeführt werden."
    exit 1
fi

# Backend starten (im Hintergrund)
echo "🚀 Starte Backend..."
cd castigatio_backend

# Python-Umgebung aktivieren falls vorhanden
if [ -f "../.venv/bin/activate" ]; then
    source ../.venv/bin/activate
    echo "✅ Virtual Environment aktiviert"
fi

# Backend im Hintergrund starten
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
echo "✅ Backend gestartet (PID: $BACKEND_PID)"

# Kurz warten damit Backend Zeit hat zu starten
sleep 3

# Frontend starten
echo "🚀 Starte Frontend..."
cd ../castigatio_frontend

# Dependencies installieren falls node_modules fehlt
if [ ! -d "node_modules" ]; then
    echo "📦 Installiere Frontend-Dependencies..."
    npm install
fi

# Frontend starten
echo "✅ Starte Tauri App..."
npm run tauri:dev

# Cleanup: Backend stoppen wenn Frontend beendet wird
echo "🛑 Stoppe Backend..."
kill $BACKEND_PID 2>/dev/null
echo "✅ Castigatio beendet"
