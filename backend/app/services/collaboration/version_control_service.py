import logging
from typing import Dict, List, Any

logger = logging.getLogger("version_control_service")

class VersionControlService:
    """
    Service de gestion de versions collaborative (historique, rollback, audit, hooks, sécurité).
    Permet de suivre, restaurer, auditer les modifications sur les objets collaboratifs (workspaces, playlists, IA, etc.).
    """
    def __init__(self):
        self.history: Dict[str, List[Dict[str, Any]]] = {}  # resource_id -> list of versions
        self.hooks = []
    def save_version(self, resource_id: str, data: Any, user_id: str):
        version = {"data": data, "user_id": user_id}
        self.history.setdefault(resource_id, []).append(version)
        logger.info(f"Version sauvegardée pour {resource_id} par {user_id}")
        self.audit(resource_id, user_id, "save")
        for hook in self.hooks:
            hook(resource_id, data, user_id)
    def get_history(self, resource_id: str) -> List[Dict[str, Any]]:
        return self.history.get(resource_id, [])
    def rollback(self, resource_id: str, version_index: int):
        if resource_id in self.history and 0 <= version_index < len(self.history[resource_id]):
            logger.info(f"Rollback {resource_id} à la version {version_index}")
            return self.history[resource_id][version_index]["data"]
        logger.warning(f"Rollback impossible pour {resource_id}")
        return None
    def register_hook(self, hook):
        self.hooks.append(hook)
        logger.info(f"VersionControl hook enregistré: {hook}")
    def audit(self, resource_id: str, user_id: str, action: str):
        logger.info(f"[AUDIT] {action} sur {resource_id} par {user_id}")
