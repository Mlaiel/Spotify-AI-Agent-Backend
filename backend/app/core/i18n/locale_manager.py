"""
Module: locale_manager.py
Description: Gestion industrielle des locales, détection automatique, fallback, chargement dynamique des fichiers de traduction, support multi-tenant et multi-langue.
"""
import os
import json
from typing import Dict, Any, Optional

LOCALES_PATH = os.path.join(os.path.dirname(__file__), "locales")
DEFAULT_LOCALE = "en"

class LocaleManager:
    _cache: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def load_locale(cls, locale: str, domain: str = "messages") -> Dict[str, Any]:
        """
        Charge dynamiquement un fichier de traduction (messages, errors, system, etc) pour une locale donnée.
        """
        key = f"{locale}:{domain}"
        if key in cls._cache:
            return cls._cache[key]
        path = os.path.join(LOCALES_PATH, locale, f"{domain}.json")
        if not os.path.exists(path):
            # Fallback sur la locale par défaut
            path = os.path.join(LOCALES_PATH, DEFAULT_LOCALE, f"{domain}.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cls._cache[key] = data
        return data

    @classmethod
    def get_message(cls, key: str, locale: str = DEFAULT_LOCALE, domain: str = "messages") -> Optional[str]:
        """
        Récupère une chaîne traduite à partir d'une clé, d'une locale et d'un domaine.
        """
        data = cls.load_locale(locale, domain)
        return data.get(key)

    @classmethod
    def available_locales(cls) -> list:
        """
        Retourne la liste des locales disponibles (dossiers présents dans locales/).
        """
        return [d for d in os.listdir(LOCALES_PATH) if os.path.isdir(os.path.join(LOCALES_PATH, d)) and not d.startswith(".")]

# Exemples d'utilisation
# print(LocaleManager.get_message("welcome", locale="fr")
# print(LocaleManager.available_locales()
