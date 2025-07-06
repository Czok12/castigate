import os
from pathlib import Path

from dotenv import load_dotenv

# Lade Umgebungsvariablen aus .env-Datei (optional)
load_dotenv()

# Basispfad des Projekts (castigatio_backend)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Zentraler Datenpfad (z.B. auf einem NAS oder lokal)
# Kann über eine .env-Variable überschrieben werden
# Beispiel für .env: DATA_STORAGE_PATH="/mnt/nas/jura_data"
DATA_STORAGE_PATH = os.getenv("DATA_STORAGE_PATH", str(BASE_DIR / "data"))
DATA_DIR = Path(DATA_STORAGE_PATH)

# Sicherstellen, dass das Datenverzeichnis existiert
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Datenbank- und Bibliotheks-Pfade
LIBRARY_DB_PATH = DATA_DIR / "castigatio_bibliothek.db"
FAISS_DB_PATH = DATA_DIR / "faiss_index"
PDF_LIBRARY_PATH = DATA_DIR / "pdf_bibliothek"

# Sicherstellen, dass der PDF-Ordner existiert
PDF_LIBRARY_PATH.mkdir(exist_ok=True)

# Modell-Konfigurationen
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL = "llama3.2"

# Geräteeinstellung für ML-Modelle ('cpu', 'cuda', 'mps')
DEVICE = os.getenv("DEVICE", "cpu")
