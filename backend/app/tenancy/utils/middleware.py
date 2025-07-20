"""
🌐 Tenant Middleware - Middleware Multi-Tenant
==============================================

Middleware FastAPI pour la gestion automatique du contexte multi-tenant.
Détection et injection du tenant dans les requêtes.

Author: Backend Senior Developer - Fahed Mlaiel
Version: 1.0.0
"""

import logging
from typing import Optional, Dict, Any, Callable
from contextvars import ContextVar
from fastapi import Request, Response, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
import time

from app.tenancy.managers.tenant_manager import tenant_manager
from app.tenancy.models.tenant import Tenant

logger = logging.getLogger(__name__)

# Variables de contexte pour le tenant actuel
current_tenant_context: ContextVar[Optional[str]] = ContextVar('current_tenant', default=None)
current_tenant_data: ContextVar[Optional[Tenant]] = ContextVar('current_tenant_data', default=None)


class TenantMiddleware(BaseHTTPMiddleware):
    """
    Middleware de gestion multi-tenant.
    
    Responsabilités:
    - Détection du tenant depuis la requête
    - Validation de l'existence et du statut du tenant
    - Injection du contexte tenant
    - Métriques de performance par tenant
    """

    def __init__(
        self,
        app,
        tenant_header: str = "X-Tenant-ID",
        tenant_subdomain: bool = True,
        tenant_path: bool = False,
        require_tenant: bool = True,
        excluded_paths: list = None
    ):
        super().__init__(app)
        self.tenant_header = tenant_header
        self.tenant_subdomain = tenant_subdomain
        self.tenant_path = tenant_path
        self.require_tenant = require_tenant
        self.excluded_paths = excluded_paths or [
            "/health",
            "/metrics", 
            "/docs",
            "/openapi.json",
            "/favicon.ico"
        ]
        
        # Cache des tenants pour performance
        self._tenant_cache: Dict[str, Tenant] = {}
        self._cache_ttl = 300  # 5 minutes

    async def dispatch(
        self, 
        request: Request, 
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Traitement principal du middleware"""
        
        start_time = time.time()
        tenant_id = None
        tenant_data = None
        
        try:
            # Vérification si le chemin est exclu
            if self._is_path_excluded(request.url.path):
                return await call_next(request)

            # Détection du tenant
            tenant_id = await self._detect_tenant(request)
            
            if not tenant_id and self.require_tenant:
                raise HTTPException(
                    status_code=400,
                    detail="Tenant ID requis"
                )

            # Récupération des données du tenant
            if tenant_id:
                tenant_data = await self._get_tenant_data(tenant_id)
                
                if not tenant_data:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Tenant '{tenant_id}' non trouvé"
                    )
                
                if not tenant_data.is_active:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Tenant '{tenant_id}' inactif"
                    )

            # Injection du contexte
            current_tenant_context.set(tenant_id)
            current_tenant_data.set(tenant_data)
            
            # Ajout des headers de contexte à la requête
            if tenant_id:
                request.state.tenant_id = tenant_id
                request.state.tenant_data = tenant_data

            # Traitement de la requête
            response = await call_next(request)
            
            # Ajout des headers de réponse
            if tenant_id:
                response.headers["X-Tenant-ID"] = tenant_id
                response.headers["X-Tenant-Status"] = tenant_data.status if tenant_data else "unknown"

            # Métriques
            processing_time = time.time() - start_time
            await self._record_tenant_metrics(tenant_id, request, response, processing_time)

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Erreur middleware tenant: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Erreur interne du serveur"
            )
        finally:
            # Nettoyage du contexte
            current_tenant_context.set(None)
            current_tenant_data.set(None)

    def _is_path_excluded(self, path: str) -> bool:
        """Vérifier si le chemin est exclu du middleware"""
        return any(path.startswith(excluded) for excluded in self.excluded_paths)

    async def _detect_tenant(self, request: Request) -> Optional[str]:
        """
        Détecter l'ID du tenant depuis la requête.
        
        Ordre de priorité:
        1. Header X-Tenant-ID
        2. Sous-domaine 
        3. Paramètre de chemin
        4. Query parameter
        """
        
        # 1. Header personnalisé
        tenant_id = request.headers.get(self.tenant_header)
        if tenant_id:
            return tenant_id

        # 2. Sous-domaine
        if self.tenant_subdomain:
            host = request.headers.get("host", "")
            if "." in host:
                subdomain = host.split(".")[0]
                # Exclure les sous-domaines système
                if subdomain not in ["www", "api", "admin", "app"]:
                    return subdomain

        # 3. Paramètre de chemin (/tenant/{tenant_id}/...)
        if self.tenant_path:
            path_parts = request.url.path.strip("/").split("/")
            if len(path_parts) >= 2 and path_parts[0] == "tenant":
                return path_parts[1]

        # 4. Query parameter
        tenant_id = request.query_params.get("tenant_id")
        if tenant_id:
            return tenant_id

        return None

    async def _get_tenant_data(self, tenant_id: str) -> Optional[Tenant]:
        """Récupérer les données du tenant avec cache"""
        
        # Vérification du cache
        if tenant_id in self._tenant_cache:
            return self._tenant_cache[tenant_id]

        try:
            # Récupération depuis le gestionnaire
            tenant_data = await tenant_manager.get_tenant(tenant_id)
            
            # Mise en cache
            if tenant_data:
                self._tenant_cache[tenant_id] = tenant_data

            return tenant_data

        except Exception as e:
            logger.error(f"Erreur récupération tenant {tenant_id}: {str(e)}")
            return None

    async def _record_tenant_metrics(
        self,
        tenant_id: Optional[str],
        request: Request,
        response: Response,
        processing_time: float
    ):
        """Enregistrer les métriques de performance par tenant"""
        if not tenant_id:
            return

        try:
            # En production, utiliser un système de métriques plus robuste
            metrics = {
                "tenant_id": tenant_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "processing_time": processing_time,
                "timestamp": time.time()
            }
            
            # Log pour développement
            logger.debug(f"Tenant metrics: {metrics}")
            
            # TODO: Intégrer avec TenantPerformanceManager
            
        except Exception as e:
            logger.error(f"Erreur enregistrement métriques: {str(e)}")


class TenantContextManager:
    """Gestionnaire de contexte pour les tests et l'usage manuel"""
    
    def __init__(self, tenant_id: str, tenant_data: Optional[Tenant] = None):
        self.tenant_id = tenant_id
        self.tenant_data = tenant_data
        self._old_tenant_id = None
        self._old_tenant_data = None

    async def __aenter__(self):
        # Sauvegarde du contexte actuel
        self._old_tenant_id = current_tenant_context.get()
        self._old_tenant_data = current_tenant_data.get()
        
        # Définition du nouveau contexte
        current_tenant_context.set(self.tenant_id)
        current_tenant_data.set(self.tenant_data)
        
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Restauration du contexte précédent
        current_tenant_context.set(self._old_tenant_id)
        current_tenant_data.set(self._old_tenant_data)


# Fonctions utilitaires

def get_current_tenant_id() -> Optional[str]:
    """Obtenir l'ID du tenant actuel"""
    return current_tenant_context.get()


def get_current_tenant() -> Optional[Tenant]:
    """Obtenir les données du tenant actuel"""
    return current_tenant_data.get()


def require_tenant_context() -> str:
    """Obtenir l'ID du tenant en exigeant qu'il existe"""
    tenant_id = current_tenant_context.get()
    if not tenant_id:
        raise HTTPException(
            status_code=400,
            detail="Contexte tenant requis"
        )
    return tenant_id


async def tenant_context(tenant_id: str, tenant_data: Optional[Tenant] = None):
    """
    Context manager pour définir manuellement le contexte tenant.
    Utile pour les tests et les tâches en arrière-plan.
    """
    return TenantContextManager(tenant_id, tenant_data)


# Décorateur pour fonctions nécessitant un contexte tenant

def with_tenant_context(func: Callable):
    """
    Décorateur pour s'assurer qu'une fonction a un contexte tenant.
    """
    async def wrapper(*args, **kwargs):
        tenant_id = get_current_tenant_id()
        if not tenant_id:
            raise HTTPException(
                status_code=400,
                detail="Contexte tenant requis pour cette opération"
            )
        return await func(*args, **kwargs)
    
    return wrapper


# Configuration par défaut du middleware
DEFAULT_MIDDLEWARE_CONFIG = {
    "tenant_header": "X-Tenant-ID",
    "tenant_subdomain": True,
    "tenant_path": False,
    "require_tenant": True,
    "excluded_paths": [
        "/health",
        "/metrics",
        "/docs",
        "/openapi.json",
        "/favicon.ico",
        "/static/",
        "/admin/",
        "/auth/login",
        "/auth/register"
    ]
}
