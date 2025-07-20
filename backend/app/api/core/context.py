"""
🎵 Spotify AI Agent - API Request Context Management
===================================================

Système de gestion de contexte des requêtes API avec patterns enterprise,
traçabilité complète, et injection de dépendances.

Architecture:
- Contexte de requête thread-safe
- Correlation IDs pour traçabilité
- User context et session management
- Performance tracking par requête
- Error context pour debugging
- Injection de dépendances

Développé par Fahed Mlaiel - Enterprise Context Management Expert
"""

import uuid
import time
from datetime import datetime
from typing import Any, Dict, Optional, List, Union
from contextvars import ContextVar
from dataclasses import dataclass, field, asdict
from enum import Enum

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class RequestPhase(str, Enum):
    """Phases d'une requête API"""
    RECEIVED = "received"
    AUTHENTICATED = "authenticated"
    AUTHORIZED = "authorized" 
    VALIDATED = "validated"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class UserContext:
    """Contexte utilisateur dans la requête"""
    user_id: Optional[str] = None
    username: Optional[str] = None
    email: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    spotify_id: Optional[str] = None
    subscription_type: Optional[str] = None
    is_authenticated: bool = False
    is_premium: bool = False
    auth_method: Optional[str] = None  # jwt, api_key, oauth
    session_id: Optional[str] = None
    
    def has_role(self, role: str) -> bool:
        """Vérifie si l'utilisateur a un rôle"""
        return role in self.roles
    
    def has_permission(self, permission: str) -> bool:
        """Vérifie si l'utilisateur a une permission"""
        return permission in self.permissions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return asdict(self)


@dataclass
class PerformanceContext:
    """Contexte de performance de la requête"""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    memory_start: Optional[int] = None
    memory_peak: Optional[int] = None
    memory_end: Optional[int] = None
    cpu_time: Optional[float] = None
    db_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    external_calls: int = 0
    
    def finish(self):
        """Marque la fin de la requête et calcule les métriques"""
        self.end_time = time.time()
        if self.start_time:
            self.duration_ms = (self.end_time - self.start_time) * 1000
    
    def add_db_query(self):
        """Incrémente le compteur de requêtes DB"""
        self.db_queries += 1
    
    def add_cache_hit(self):
        """Incrémente le compteur de cache hits"""
        self.cache_hits += 1
    
    def add_cache_miss(self):
        """Incrémente le compteur de cache misses"""
        self.cache_misses += 1
    
    def add_external_call(self):
        """Incrémente le compteur d'appels externes"""
        self.external_calls += 1


@dataclass
class ErrorContext:
    """Contexte d'erreur pour debugging"""
    error_id: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_details: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    user_message: Optional[str] = None
    retry_count: int = 0
    is_retryable: bool = False
    
    def set_error(self, error: Exception, user_message: Optional[str] = None):
        """Configure l'erreur dans le contexte"""
        self.error_id = str(uuid.uuid4())
        self.error_type = type(error).__name__
        self.error_message = str(error)
        self.user_message = user_message
        import traceback
        self.stack_trace = traceback.format_exc()


@dataclass
class RequestContext:
    """Contexte complet d'une requête API"""
    
    # Identification
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    
    # Timing
    timestamp: datetime = field(default_factory=datetime.utcnow)
    phase: RequestPhase = RequestPhase.RECEIVED
    
    # Request Details
    method: Optional[str] = None
    path: Optional[str] = None
    query_params: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    
    # User Context
    user: UserContext = field(default_factory=UserContext)
    
    # Performance
    performance: PerformanceContext = field(default_factory=PerformanceContext)
    
    # Error Handling
    error: Optional[ErrorContext] = None
    
    # Custom Data
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def set_phase(self, phase: RequestPhase):
        """Change la phase de la requête"""
        self.phase = phase
    
    def set_user(self, user: UserContext):
        """Configure le contexte utilisateur"""
        self.user = user
    
    def set_error(self, error: Exception, user_message: Optional[str] = None):
        """Configure une erreur"""
        self.error = ErrorContext()
        self.error.set_error(error, user_message)
        self.phase = RequestPhase.ERROR
    
    def add_metadata(self, key: str, value: Any):
        """Ajoute des métadonnées"""
        self.metadata[key] = value
    
    def add_tag(self, tag: str):
        """Ajoute un tag"""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour logging"""
        return {
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "phase": self.phase,
            "method": self.method,
            "path": self.path,
            "user_id": self.user.user_id if self.user else None,
            "duration_ms": self.performance.duration_ms,
            "db_queries": self.performance.db_queries,
            "cache_hits": self.performance.cache_hits,
            "cache_misses": self.performance.cache_misses,
            "metadata": self.metadata,
            "tags": self.tags,
            "error_id": self.error.error_id if self.error else None
        }


@dataclass  
class APIContext:
    """Contexte global de l'API"""
    app_name: str = "Spotify AI Agent"
    app_version: str = "2.0.0"
    environment: str = "development"
    deployment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    startup_time: datetime = field(default_factory=datetime.utcnow)
    
    # Métriques globales
    total_requests: int = 0
    active_requests: int = 0
    total_errors: int = 0
    
    def increment_requests(self):
        """Incrémente le compteur de requêtes"""
        self.total_requests += 1
        self.active_requests += 1
    
    def decrement_active_requests(self):
        """Décrémente les requêtes actives"""
        self.active_requests = max(0, self.active_requests - 1)
    
    def increment_errors(self):
        """Incrémente le compteur d'erreurs"""
        self.total_errors += 1


# =============================================================================
# CONTEXT STORAGE AVEC CONTEXTVARS
# =============================================================================

# Variables de contexte thread-safe
_request_context: ContextVar[Optional[RequestContext]] = ContextVar(
    'request_context', 
    default=None
)

_api_context: ContextVar[Optional[APIContext]] = ContextVar(
    'api_context',
    default=None
)


def get_request_context() -> Optional[RequestContext]:
    """Retourne le contexte de requête actuel"""
    return _request_context.get()


def set_request_context(context: RequestContext) -> None:
    """Définit le contexte de requête"""
    _request_context.set(context)


def get_api_context() -> Optional[APIContext]:
    """Retourne le contexte API actuel"""
    return _api_context.get()


def set_api_context(context: APIContext) -> None:
    """Définit le contexte API"""
    _api_context.set(context)


def clear_request_context() -> None:
    """Efface le contexte de requête"""
    _request_context.set(None)


# =============================================================================
# MIDDLEWARE DE CONTEXTE
# =============================================================================

class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware pour gérer le contexte des requêtes"""
    
    def __init__(self, app, api_context: Optional[APIContext] = None):
        super().__init__(app)
        self.api_context = api_context or APIContext()
        set_api_context(self.api_context)
    
    async def dispatch(self, request: Request, call_next):
        """Traite la requête avec contexte"""
        
        # Créer le contexte de requête
        context = RequestContext()
        
        # Extraire les informations de la requête
        context.method = request.method
        context.path = str(request.url.path)
        context.query_params = dict(request.query_params)
        context.headers = dict(request.headers)
        context.user_agent = request.headers.get("user-agent")
        
        # IP Address avec support proxy
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            context.ip_address = forwarded_for.split(",")[0].strip()
        else:
            context.ip_address = getattr(request.client, "host", None)
        
        # Correlation ID depuis header ou généré
        context.correlation_id = request.headers.get(
            "x-correlation-id", 
            context.correlation_id
        )
        
        # Session ID
        context.session_id = request.headers.get("x-session-id")
        
        # Définir le contexte
        set_request_context(context)
        
        # Incrémenter les métriques
        self.api_context.increment_requests()
        
        try:
            # Marquer comme en traitement
            context.set_phase(RequestPhase.PROCESSING)
            
            # Traiter la requête
            response = await call_next(request)
            
            # Marquer comme complétée
            context.set_phase(RequestPhase.COMPLETED)
            
            # Ajouter les headers de réponse
            response.headers["x-request-id"] = context.request_id
            response.headers["x-correlation-id"] = context.correlation_id
            
            return response
            
        except Exception as e:
            # Gérer l'erreur
            context.set_error(e)
            self.api_context.increment_errors()
            raise
            
        finally:
            # Finaliser les métriques de performance
            context.performance.finish()
            self.api_context.decrement_active_requests()
            
            # Nettoyer le contexte
            clear_request_context()


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def create_user_context(
    user_id: str,
    username: str = None,
    email: str = None,
    roles: List[str] = None,
    **kwargs
) -> UserContext:
    """Crée un contexte utilisateur"""
    return UserContext(
        user_id=user_id,
        username=username,
        email=email,
        roles=roles or [],
        is_authenticated=True,
        **kwargs
    )


def get_current_user() -> Optional[UserContext]:
    """Retourne l'utilisateur actuel depuis le contexte"""
    context = get_request_context()
    return context.user if context else None


def get_request_id() -> Optional[str]:
    """Retourne l'ID de la requête actuelle"""
    context = get_request_context()
    return context.request_id if context else None


def get_correlation_id() -> Optional[str]:
    """Retourne l'ID de corrélation"""
    context = get_request_context()
    return context.correlation_id if context else None


def add_request_metadata(key: str, value: Any) -> None:
    """Ajoute des métadonnées à la requête actuelle"""
    context = get_request_context()
    if context:
        context.add_metadata(key, value)


def add_request_tag(tag: str) -> None:
    """Ajoute un tag à la requête actuelle"""
    context = get_request_context()
    if context:
        context.add_tag(tag)


# Instance middleware pour export
request_context_middleware = RequestContextMiddleware


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "RequestPhase",
    "UserContext", 
    "PerformanceContext",
    "ErrorContext",
    "RequestContext",
    "APIContext",
    "RequestContextMiddleware",
    "get_request_context",
    "set_request_context",
    "get_api_context",
    "set_api_context",
    "clear_request_context",
    "create_user_context",
    "get_current_user",
    "get_request_id",
    "get_correlation_id",
    "add_request_metadata",
    "add_request_tag",
    "request_context_middleware"
]
