"""
Module: text_processor.py
Description: Outils avancés de traitement de texte multilingue pour la normalisation, le nettoyage, la détection de langue et la préparation de contenu IA/analytics.
"""
import re
import unicodedata
from langdetect import detect
from typing import Optional

class TextProcessor:
    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalise le texte Unicode (NFKC), retire les espaces superflus et standardise la casse.
        """
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def clean(text: str) -> str:
        """
        Nettoie le texte (suppression HTML, caractères spéciaux, etc).
        """
        text = re.sub(r"<[^>]+>", "", text)  # Remove HTML tags
        text = re.sub(r"[^\w\s.,;:!?'-]", "", text)
        return text

    @staticmethod
    def detect_language(text: str) -> Optional[str]:
        """
        Détecte la langue d'un texte (utile pour IA, analytics, etc).
        """
        try:
            return detect(text)
        except Exception:
            return None

    @staticmethod
    def tokenize(text: str, lang: str = "en") -> list:
        """
        Tokenisation simple, extensible selon la langue.
        """
        return re.findall(r"\w+", text)

# Exemples d'utilisation
# print(TextProcessor.normalize("  Héllo   World!  ")
# print(TextProcessor.detect_language("Bonjour le monde")
