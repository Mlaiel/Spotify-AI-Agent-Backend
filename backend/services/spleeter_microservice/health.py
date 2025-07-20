"""
Endpoint de santé pour le microservice Spleeter.
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/health", tags=["Monitoring"])
def health_check():
    return {"status": "ok"}
