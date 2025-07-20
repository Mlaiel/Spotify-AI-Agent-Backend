"""
ConversationHandler : Gestionnaire de conversation avancé
- Multi-turn, suivi de contexte, relances intelligentes
- Sécurité : validation, audit, RGPD
- Intégration FastAPI/Django, scalable

Auteur : Lead Dev, Backend, Sécurité
"""

from typing import Dict, Any, Optional
from .memory_system import MemorySystem

class ConversationHandler:
    """
    Gère le flux conversationnel, le suivi du contexte et la logique de relance.
    """
    def __init__(self):
        self.memory = MemorySystem()

    def process_message(self, user_id: str, message: str) -> Dict[str, Any]:
        """
        Traite un message utilisateur, met à jour le contexte et propose la prochaine action.
        """
        ctx = self.memory.get_context(user_id)
        # Logique de relance intelligente (exemple simplifié)
        if len(ctx["history"]) > 0 and ctx["history"][-1]["prompt"] == message:
            followup = "Souhaitez-vous approfondir ce sujet ou passer à autre chose ?"
        else:
            followup = None
        self.memory.save_interaction(user_id, message, followup)
        return {
            "context": ctx,
            "followup": followup
        }

# Exemple d’utilisation :
# handler = ConversationHandler()
# print(handler.process_message("user123", "Montre-moi mes stats Spotify")
