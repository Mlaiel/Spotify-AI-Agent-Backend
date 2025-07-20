"""
Module: locale_utils.py
Description: Fonctions utilitaires pour la gestion avancée des locales, détection automatique, fallback, et validation des langues supportées.
"""
import locale
from babel import Locale
from typing import List, Optional

SUPPORTED_LOCALES = [
    "en", "fr", "de", "es", "it", "pt", "nl", "ru", "ja", "ko", "zh_CN", "zh_TW", "ar", "hi", # ...
]

class LocaleUtils:
    @staticmethod
    def get_best_locale(accept_language: str, supported_locales: Optional[List[str]] = None) -> str:
        """
        Détecte la meilleure locale à partir de l'en-tête HTTP Accept-Language.
        """
        import babel
        supported = supported_locales or SUPPORTED_LOCALES
        return babel.Locale.negotiate(accept_language, supported) or supported[0]

    @staticmethod
    def is_supported(locale_code: str) -> bool:
        """
        Vérifie si une locale est supportée.
        """
        return locale_code in SUPPORTED_LOCALES

    @staticmethod
    def get_display_name(locale_code: str, display_locale: str = "en") -> str:
        """
        Retourne le nom affiché d'une locale dans une autre langue (ex: 'French' pour 'fr').
        """
        try:
            return Locale.parse(locale_code).get_display_name(display_locale)
        except Exception:
            return locale_code

    @staticmethod
    def get_all_supported_locales() -> List[str]:
        return SUPPORTED_LOCALES

# Exemples d'utilisation
# print(LocaleUtils.get_best_locale("fr-FR,fr;q=0.9,en;q=0.8")
# print(LocaleUtils.get_display_name("de", "fr")
