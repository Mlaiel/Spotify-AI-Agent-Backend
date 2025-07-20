"""
Module: id_utils.py
Description: Génération industrielle d'identifiants uniques (UUID, nanoid, shortid), pour microservices, tracking, sécurité.
"""
import uuid
import secrets
import base64
from typing import Optional

def generate_uuid() -> str:
    return str(uuid.uuid4())

def generate_short_id(length: int = 12) -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(length)).decode("utf-8").rstrip("=")[:length]

# Exemples d'utilisation
# generate_uuid()
# generate_short_id(10)
