"""
Utilitaires avancés pour le microservice Spleeter (validation, sécurité, monitoring, gestion fichiers).
"""

import os
import tempfile
import logging
from fastapi import UploadFile, HTTPException
from starlette.status import HTTP_400_BAD_REQUEST

from .config import settings

def validate_audio_file(file: UploadFile):
    ext = file.filename.split(".")[-1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        logging.warning(f"Extension non supportée: {ext}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Extension non supportée: {ext}")
    
    # Taille max
    file.file.seek(0, os.SEEK_END)
    size_mb = file.file.tell() / (1024 * 1024)
    file.file.seek(0)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        logging.warning(f"Fichier trop volumineux: {size_mb:.2f} MB")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,)
            detail=f"Fichier trop volumineux: {size_mb:.2f} MB (max {settings.MAX_FILE_SIZE_MB} MB)"
        )
    # Placeholder pour scan antivirus (ex: ClamAV)
    # if not scan_antivirus(file):
    #     raise HTTPException(status_code=400, detail="Fichier infecté")
    return True

def save_temp_file(file: UploadFile) -> str:
    suffix = os.path.splitext(file.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read()
        logging.info(f"Fichier temporaire créé: {tmp.name}")
        return tmp.name

def cleanup_temp_file(path: str):
    try:
        os.remove(path)
        logging.info(f"Fichier temporaire supprimé: {path}")
    except Exception as e:
        logging.error(f"Erreur suppression fichier temporaire: {e}")
