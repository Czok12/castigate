from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Gesundheitscheck für das Backend"""
    return {
        "status": "healthy",
        "message": "Backend läuft erfolgreich"
    }


@router.get("/status")
async def system_status():
    """Detaillierter Systemstatus"""
    import os
    
    # Prüfe wichtige Systemkomponenten
    status = {
        "database": {
            "faiss_db": os.path.exists("faiss_db"),
            "library_db": os.path.exists("juristische_bibliothek.db")
        },
        "services": {
            "knowledge_base": True,  # TODO: Implementiere echte Checks
            "citation_engine": True,
            "library_management": True
        }
    }
    
    return status
