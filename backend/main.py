from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.status import router as status_router

app = FastAPI(
    title="Juristische Wissensdatenbank API",
    description="Backend für die juristische Wissensdatenbank mit RAG-System",
    version="2.0.0",
)

# CORS-Middleware für Tauri-Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "tauri://localhost",
        "http://localhost:3000",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API-Router registrieren
app.include_router(health_router, prefix="/api/v1", tags=["health"])
app.include_router(status_router, prefix="/api/v1", tags=["status"])


@app.get("/")
def read_root():
    return {
        "message": "Juristische Wissensdatenbank API",
        "version": "2.0.0",
        "docs": "/docs",
    }
