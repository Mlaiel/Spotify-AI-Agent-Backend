"""
MemorySystem : Système de mémoire conversationnelle/contextuelle
- Stockage contextuel (Redis, DB, in-memory)
- Historique, sessions, audit trail
- Sécurité : chiffrement, anonymisation, RGPD
- Optimisé pour microservices et scalabilité

Auteur : Data Engineer, Backend, Sécurité
"""

from typing import Dict, Any, Optional
import time

class MemorySystem:
    """
    Système de gestion de la mémoire utilisateur/session pour l’agent IA.
    """
    def __init__(self):
        self._store = {}  # Remplacer par Redis/DB en prod

    def get_context(self, user_id: str) -> Dict[str, Any]:
        """Récupère le contexte utilisateur (historique, préférences, etc.)."""
        return self._store.get(user_id, {"history": [], "last_seen": None})

    def save_interaction(self, user_id: str, prompt: str, response: Any):
        """Sauvegarde l’interaction dans l’historique utilisateur."""
        if user_id not in self._store:
            self._store[user_id] = {"history": [], "last_seen": None}
        self._store[user_id]["history"].append({
            "timestamp": time.time(),
            "prompt": prompt,
            "response": response
        })
        self._store[user_id]["last_seen"] = time.time()

    def clear_history(self, user_id: str):
        """Efface l’historique utilisateur (conformité RGPD)."""
        if user_id in self._store:
            self._store[user_id]["history"] = []

# Exemple d’utilisation :
# memory = MemorySystem()
# memory.save_interaction("user123", "prompt test", "réponse test")
# print(memory.get_context("user123")
