import logging
from typing import Dict, List

logger = logging.getLogger("permission_service")

class PermissionService:
    """
    Service de gestion des permissions avancé (RBAC, audit, sécurité, hooks, logique métier IA/Spotify).
    Gère les droits d’accès, les rôles, la granularité, l’audit et la conformité.
    """
    def __init__(self):
        self.roles: Dict[str, List[str]] = {
            "admin": ["read", "write", "delete", "invite", "manage"],
            "editor": ["read", "write", "invite"],
            "viewer": ["read"],
        }
        self.user_roles: Dict[str, str] = {}  # user_id -> role
        self.hooks = []
    def assign_role(self, user_id: str, role: str):
        self.user_roles[user_id] = role
        logger.info(f"Rôle {role} assigné à {user_id}")
    def check_permission(self, user_id: str, resource_id: str, action: str) -> bool:
        role = self.user_roles.get(user_id, "viewer")
        allowed = action in self.roles.get(role, [])
        logger.info(f"Permission check: {user_id} ({role}) sur {resource_id} pour {action}: {allowed}")
        self.audit(user_id, resource_id, action, allowed)
        for hook in self.hooks:
            hook(user_id, resource_id, action, allowed)
        return allowed
    def register_hook(self, hook):
        self.hooks.append(hook)
        logger.info(f"Permission hook enregistré: {hook}")
    def audit(self, user_id: str, resource_id: str, action: str, allowed: bool):
        logger.info(f"[AUDIT] {user_id} tente {action} sur {resource_id}: {'OK' if allowed else 'REFUS'}")
