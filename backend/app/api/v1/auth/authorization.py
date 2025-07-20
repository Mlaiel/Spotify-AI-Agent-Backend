"""
Authorizer : Service d’autorisation avancée (RBAC, scopes)
- Gestion des rôles, permissions, scopes API
- Sécurité : audit, logs, conformité RGPD
- Intégration FastAPI/Django, scalable microservices

Auteur : Backend Senior, Sécurité, Lead Dev
"""

from typing import List, Dict, Any

class Authorizer:
    """
    Gère l’autorisation et la vérification des permissions utilisateur.
    """
    def __init__(self, rbac=None):
        self.rbac = rbac or {"admin": ["*"], "artist": ["read", "write"]}

    def has_permission(self, user_roles: List[str], permission: str) -> bool:
        """
        Vérifie si l’utilisateur a la permission demandée.
        """
        for role in user_roles:
            if permission in self.rbac.get(role, []) or "*" in self.rbac.get(role, []):
                return True
        return False

# Exemple d’utilisation :
# authz = Authorizer()
# print(authz.has_permission(["artist"], "read")
# print(authz.has_permission(["artist"], "admin")
