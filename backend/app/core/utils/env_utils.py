"""
Module: env_utils.py
Description: Utilitaires industriels pour la gestion des variables d'environnement, chargement .env, validation, fallback, secrets.
"""
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

def get_env(key: str, default: Optional[str] = None, required: bool = False) -> str:
    value = os.getenv(key, default)
    if required and value is None:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return value

# Exemples d'utilisation
# get_env("DATABASE_URL", required=True)
# get_env("DEBUG", default="false")
