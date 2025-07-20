import requests
import logging
from typing import Dict, Any

logger = logging.getLogger("AIModeration")

class AIModerationService:
    """
    Service d'appel à une API IA externe (Hugging Face, modèle custom) pour la modération de messages WebSocket.
    """
    def __init__(self, api_url: str, api_key: str = None):
        self.api_url = api_url
        self.api_key = api_key

    def moderate(self, message: str, user_id: str) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {"inputs": message, "user_id": user_id}
        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=2)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Modération IA: {result}")
            return result
        except Exception as e:
            logger.error(f"Erreur modération IA: {e}")
            return {"allowed": True, "reason": "Erreur IA, fallback permissif"}

# Exemple d'intégration :
# ai_service = AIModerationService(api_url="https://api-inference.huggingface.co/models/xxx")
# verdict = ai_service.moderate("message", "user_id")
# if not verdict["allowed"]:
#     ... refuser le message ...
