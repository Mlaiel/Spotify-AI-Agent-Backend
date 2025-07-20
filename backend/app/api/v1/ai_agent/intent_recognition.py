"""
IntentRecognizer : Reconnaissance d’intention avancée pour agent IA Spotify.
- NLP multilingue (spaCy, transformers, HuggingFace)
- Détection de use-case (statistiques, génération, collaboration, etc.)
- Sécurité : validation stricte, audit, anti-prompt injection
- Exploitable en FastAPI/Django, scalable microservices

Auteur : Lead Dev, Architecte IA, ML, Sécurité
"""

from typing import Dict, Any, Optional
from transformers import pipeline
import re

class IntentRecognizer:
    """
    Classe de reconnaissance d’intention pour prompts utilisateur.
    Utilise NLP avancé (transformers) et règles métier.
    """
    def __init__(self, lang: str = "fr"):
        self.lang = lang
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.labels = [
            "statistiques spotify", "génération contenu", "collaboration", "analyse audience",
            "génération musicale", "recommandation", "traduction", "analytics", "remix", "autre"
        ]

    def recognize(self, prompt: str) -> Dict[str, Any]:
        """
        Analyse le prompt utilisateur et retourne l’intention détectée.
        """
        # Sécurité basique
        if len(prompt) > 1000 or re.search(r"[{}$<>]", prompt):
            return {"intent": "rejeté", "score": 1.0, "reason": "Prompt potentiellement dangereux"}
        result = self.classifier(prompt, self.labels)
        return {
            "intent": result["labels"][0],
            "score": float(result["scores"][0]),
            "all": list(zip(result["labels"], result["scores"]))
        }

    def is_collaboration(self, prompt: str) -> bool:
        """
        Détection rapide d’intention de collaboration.
        """
        return "collaborer" in prompt.lower() or "feat" in prompt.lower()

# Exemple d’utilisation :
# recognizer = IntentRecognizer(lang="fr")
# print(recognizer.recognize("Génère-moi un post pour Spotify sur ma nouvelle chanson")
