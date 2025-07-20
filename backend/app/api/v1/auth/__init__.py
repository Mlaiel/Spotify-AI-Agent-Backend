"""
Module Authentification & Sécurité pour Spotify AI Agent

Ce package expose tous les services d’authentification, autorisation et sécurité avancée :
- Authentification OAuth2 (Spotify, Auth0, Firebase)
- Gestion JWT, sessions, RBAC
- Middleware sécurité, audit, conformité RGPD
- Intégration scalable (FastAPI, Django, microservices)

Auteur : Lead Dev, Architecte IA, Backend Senior, Sécurité
"""

from .authentication import Authenticator
from .authorization import Authorizer
from .oauth2_handlers import OAuth2Handler
from .jwt_manager import JWTManager
from .session_manager import SessionManager
from .rbac_manager import RBACManager
from .security_middleware import SecurityMiddleware

__all__ = [
    "Authenticator",
    "Authorizer",
    "OAuth2Handler",
    "JWTManager",
    "SessionManager",
    "RBACManager",
    "SecurityMiddleware",
]
