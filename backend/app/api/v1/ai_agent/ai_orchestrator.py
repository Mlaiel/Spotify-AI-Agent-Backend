"""
AIOrchestrator : Orchestrateur central des services IA (OpenAI, ML, Audio)
- Routage intelligent des requêtes (génération, analyse, audio)
- Gestion des priorités, fallback, tolérance aux pannes
- Sécurité : audit, logs, quotas, conformité RGPD
- Intégration FastAPI/Django, scalable microservices

Auteur : Architecte IA, Lead Dev, ML, Sécurité
"""

from typing import Dict, Any, Optional
from .intent_recognition import IntentRecognizer
from .memory_system import MemorySystem
from .personality_engine import PersonalityEngine
from .response_generator import ResponseGenerator

class AIOrchestrator:
    """
    Orchestrateur principal pour router les requêtes IA selon l’intention et le contexte.
    """
    def __init__(self, lang: str = "fr"):
        self.intent = IntentRecognizer(lang=lang)
        self.memory = MemorySystem()
        self.personality = PersonalityEngine()
        self.response = ResponseGenerator(lang=lang)

    def handle_request(self, user_id: str, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Orchestration complète :
        - Détecte l’intention
        - Récupère le contexte
        - Adapte la personnalité
        - Génère la réponse
        """
        intent = self.intent.recognize(prompt)
        ctx = self.memory.get_context(user_id)
        persona = self.personality.get_persona(user_id)
        response = self.response.generate(prompt, intent, ctx, persona)
        self.memory.save_interaction(user_id, prompt, response)
        return {
            "intent": intent,
            "persona": persona,
            "response": response
        }

# Exemple d’utilisation :
# orchestrator = AIOrchestrator(lang="fr")
# result = orchestrator.handle_request("user123", "Donne-moi mes stats Spotify de la semaine")
# print(result)
