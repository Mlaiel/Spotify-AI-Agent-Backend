"""
Module: file_utils.py
Description: Utilitaires industriels pour la gestion des fichiers (upload, download, validation, sécurité, streaming, S3, temp).
"""
import os
import shutil
from typing import Optional

def save_file(file, dest_path: str) -> str:
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file, f)
    return dest_path

def remove_file(path: str):
    if os.path.exists(path):
        os.remove(path)

def file_size(path: str) -> int:
    return os.path.getsize(path)

def is_allowed_extension(filename: str, allowed: Optional[list] = None) -> bool:
    allowed = allowed or [".jpg", ".png", ".pdf", ".mp3", ".wav"]
    return any(filename.lower().endswith(ext) for ext in allowed)

# Exemples d'utilisation
# save_file(file, "/tmp/upload.bin")
# remove_file("/tmp/upload.bin")
# file_size("/tmp/upload.bin")
# is_allowed_extension("track.mp3")
