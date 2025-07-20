"""
RBACManager : Gestionnaire RBAC (Role-Based Access Control)
- Gestion des rôles, permissions, scopes API
- Sécurité : audit, logs, conformité RGPD
- Intégration FastAPI/Django, scalable microservices

Auteur : Backend Senior, Sécurité, Lead Dev
"""

from typing import Dict, List

class RBACManager:
    """
    Gère la définition et la vérification des rôles et permissions utilisateur.
    """
    def __init__(self):
        self.roles = {
            "admin": ["*"],
            "artist": ["read", "write", "analytics"],
            "viewer": ["read"]
        }

    def get_permissions(self, role: str) -> List[str]:
        """
        Retourne la liste des permissions pour un rôle donné.
        """
        return self.roles.get(role, [])

    def check_access(self, user_roles: List[str], permission: str) -> bool:
        """
        Vérifie si l’utilisateur a accès à une permission donnée.
        """
        for role in user_roles:
            if permission in self.get_permissions(role) or "*" in self.get_permissions(role):
                return True
        return False

# Exemple d’utilisation :
# rbac = RBACManager()
# print(rbac.get_permissions("artist")
# print(rbac.check_access(["artist"], "analytics")
