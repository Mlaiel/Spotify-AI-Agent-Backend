"""
🎯 Tenant Context Management - Gestion Centralisée du Contexte Tenant
====================================================================

Système ultra-avancé de gestion des contextes multi-tenant avec sécurité 
enterprise-grade et performance optimisée.

Author: Lead Dev + Architecte IA - Fahed Mlaiel
"""

import uuid
import asyncio
import threading
from typing import Optional, Dict, Any, List, Set
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor

from ...core.config import settings
from ...core.exceptions import TenantError
from ...utils.cache import CacheManager
from ...security.encryption import EncryptionManager
from ..exceptions import TenantNotFoundError, SecurityViolationError


class TenantType(Enum):
    """Types de tenants supportés"""
    SPOTIFY_ARTIST = "spotify_artist"
    RECORD_LABEL = "record_label" 
    MUSIC_PRODUCER = "music_producer"
    DISTRIBUTOR = "distributor"
    ENTERPRISE = "enterprise"
    PLATFORM = "platform"


class IsolationLevel(Enum):
    """Niveaux d'isolation des données"""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"


@dataclass
class TenantMetadata:
    """Métadonnées avancées du tenant"""
    region: str
    timezone: str
    locale: str
    features: Set[str] = field(default_factory=set)
    tier: str = "standard"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data_residency: str = "EU"
    encryption_key_version: int = 1
    compliance_flags: Dict[str, bool] = field(default_factory=dict)


@dataclass
class SecurityContext:
    """Contexte de sécurité pour le tenant"""
    access_level: str
    permissions: Set[str] = field(default_factory=set)
    roles: Set[str] = field(default_factory=set)
    ip_whitelist: List[str] = field(default_factory=list)
    mfa_required: bool = False
    audit_required: bool = True
    encryption_required: bool = True


class TenantContext:
    """
    Contexte tenant ultra-avancé avec sécurité enterprise-grade
    
    Features:
    - Thread-safe context management
    - Automatic security validation
    - Real-time monitoring integration
    - Cache optimization
    - Audit logging
    """
    
    def __init__(
        self,
        tenant_id: str,
        tenant_type: TenantType,
        metadata: Optional[TenantMetadata] = None,
        security: Optional[SecurityContext] = None,
        isolation_level: IsolationLevel = IsolationLevel.STRICT
    ):
        self.tenant_id = tenant_id
        self.tenant_type = tenant_type
        self.metadata = metadata or TenantMetadata(
            region="EU", 
            timezone="UTC", 
            locale="en-US"
        )
        self.security = security or SecurityContext(access_level="standard")
        self.isolation_level = isolation_level
        
        # Context tracking
        self.context_id = str(uuid.uuid4())
        self.created_at = datetime.now(timezone.utc)
        self.access_count = 0
        self.last_access = self.created_at
        
        # Performance tracking
        self._query_cache = {}
        self._connection_pool = None
        
        # Security tracking
        self._access_log = []
        self._security_events = []
        
        # Logger
        self.logger = logging.getLogger(f"tenant.{tenant_id}")
        
        # Initialize context
        self._initialize_context()
    
    def _initialize_context(self):
        """Initialise le contexte tenant avec validation sécurisée"""
        try:
            # Validate tenant existence
            if not self._validate_tenant():
                raise TenantNotFoundError(f"Tenant {self.tenant_id} not found")
            
            # Security validation
            if not self._validate_security():
                raise SecurityViolationError(f"Security validation failed for {self.tenant_id}")
            
            # Initialize encryption
            if self.security.encryption_required:
                self._init_encryption()
            
            # Log context creation
            self.logger.info(f"Tenant context initialized: {self.context_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tenant context: {e}")
            raise
    
    def _validate_tenant(self) -> bool:
        """Valide l'existence et l'état du tenant"""
        # Implementation would check database/cache
        return True
    
    def _validate_security(self) -> bool:
        """Valide les permissions et contraintes de sécurité"""
        # Implementation would check security policies
        return True
    
    def _init_encryption(self):
        """Initialise le système de chiffrement pour ce tenant"""
        self._encryption_manager = EncryptionManager(
            tenant_id=self.tenant_id,
            key_version=self.metadata.encryption_key_version
        )
    
    def update_activity(self):
        """Met à jour l'activité du tenant"""
        self.access_count += 1
        self.last_access = datetime.now(timezone.utc)
        self.metadata.last_activity = self.last_access
    
    def log_access(self, resource: str, action: str, metadata: Optional[Dict] = None):
        """Log d'accès pour audit"""
        access_record = {
            "timestamp": datetime.now(timezone.utc),
            "resource": resource,
            "action": action,
            "metadata": metadata or {},
            "context_id": self.context_id
        }
        self._access_log.append(access_record)
        
        if self.security.audit_required:
            self.logger.info(f"Access logged: {resource}/{action}")
    
    def get_database_name(self) -> str:
        """Retourne le nom de la base de données pour ce tenant"""
        if self.isolation_level == IsolationLevel.NONE:
            return "shared_db"
        elif self.isolation_level == IsolationLevel.BASIC:
            return f"tenant_{self.tenant_type.value}"
        else:
            return f"tenant_{self.tenant_id}"
    
    def get_schema_name(self) -> str:
        """Retourne le nom du schéma pour ce tenant"""
        if self.isolation_level in [IsolationLevel.NONE, IsolationLevel.BASIC]:
            return "public"
        else:
            return f"tenant_{self.tenant_id}"
    
    def get_cache_prefix(self) -> str:
        """Retourne le préfixe de cache pour ce tenant"""
        return f"tenant:{self.tenant_id}:"
    
    def to_dict(self) -> Dict[str, Any]:
        """Sérialise le contexte tenant"""
        return {
            "tenant_id": self.tenant_id,
            "tenant_type": self.tenant_type.value,
            "context_id": self.context_id,
            "isolation_level": self.isolation_level.value,
            "created_at": self.created_at.isoformat(),
            "access_count": self.access_count,
            "metadata": {
                "region": self.metadata.region,
                "timezone": self.metadata.timezone,
                "locale": self.metadata.locale,
                "tier": self.metadata.tier
            }
        }


class TenantContextManager:
    """
    Gestionnaire de contextes tenant ultra-avancé
    
    Features:
    - Thread-safe context switching
    - Automatic context cleanup
    - Performance optimization
    - Memory management
    - Concurrent access support
    """
    
    def __init__(self):
        self._contexts: Dict[str, TenantContext] = {}
        self._thread_local = threading.local()
        self._async_contexts: Dict[str, TenantContext] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=10)
        
        # Cache and monitoring
        self._cache_manager = CacheManager()
        self.logger = logging.getLogger("tenant.context_manager")
        
        # Performance tracking
        self._access_stats = {}
        self._context_lifecycle = {}
    
    def create_context(
        self,
        tenant_id: str,
        tenant_type: TenantType,
        **kwargs
    ) -> TenantContext:
        """Crée un nouveau contexte tenant"""
        with self._lock:
            if tenant_id in self._contexts:
                self.logger.warning(f"Context for tenant {tenant_id} already exists")
                return self._contexts[tenant_id]
            
            context = TenantContext(
                tenant_id=tenant_id,
                tenant_type=tenant_type,
                **kwargs
            )
            
            self._contexts[tenant_id] = context
            self._context_lifecycle[tenant_id] = {
                "created": datetime.now(timezone.utc),
                "access_count": 0
            }
            
            self.logger.info(f"Created context for tenant: {tenant_id}")
            return context
    
    def get_context(self, tenant_id: str) -> Optional[TenantContext]:
        """Récupère le contexte tenant"""
        with self._lock:
            context = self._contexts.get(tenant_id)
            if context:
                context.update_activity()
                self._context_lifecycle[tenant_id]["access_count"] += 1
            return context
    
    def set_current_context(self, context: TenantContext):
        """Définit le contexte tenant actuel (thread-safe)"""
        self._thread_local.current_context = context
        context.update_activity()
        self.logger.debug(f"Set current context: {context.tenant_id}")
    
    def get_current_context(self) -> Optional[TenantContext]:
        """Récupère le contexte tenant actuel"""
        return getattr(self._thread_local, 'current_context', None)
    
    def clear_current_context(self):
        """Efface le contexte tenant actuel"""
        if hasattr(self._thread_local, 'current_context'):
            delattr(self._thread_local, 'current_context')
    
    @contextmanager
    def context(self, tenant_id: str):
        """Context manager synchrone pour le contexte tenant"""
        context = self.get_context(tenant_id)
        if not context:
            raise TenantNotFoundError(f"Context not found for tenant: {tenant_id}")
        
        previous_context = self.get_current_context()
        try:
            self.set_current_context(context)
            yield context
        finally:
            if previous_context:
                self.set_current_context(previous_context)
            else:
                self.clear_current_context()
    
    @asynccontextmanager
    async def async_context(self, tenant_id: str):
        """Context manager asynchrone pour le contexte tenant"""
        context = self.get_context(tenant_id)
        if not context:
            raise TenantNotFoundError(f"Context not found for tenant: {tenant_id}")
        
        # Store in asyncio task context
        task = asyncio.current_task()
        task_id = id(task) if task else "no_task"
        
        previous_context = self._async_contexts.get(task_id)
        try:
            self._async_contexts[task_id] = context
            yield context
        finally:
            if previous_context:
                self._async_contexts[task_id] = previous_context
            else:
                self._async_contexts.pop(task_id, None)
    
    def get_current_async_context(self) -> Optional[TenantContext]:
        """Récupère le contexte tenant asynchrone actuel"""
        task = asyncio.current_task()
        task_id = id(task) if task else "no_task"
        return self._async_contexts.get(task_id)
    
    def cleanup_context(self, tenant_id: str):
        """Nettoie un contexte tenant"""
        with self._lock:
            context = self._contexts.pop(tenant_id, None)
            if context:
                self.logger.info(f"Cleaned up context for tenant: {tenant_id}")
                # Cleanup resources
                context._query_cache.clear()
                self._context_lifecycle.pop(tenant_id, None)
    
    def cleanup_all(self):
        """Nettoie tous les contextes"""
        with self._lock:
            tenant_ids = list(self._contexts.keys())
            for tenant_id in tenant_ids:
                self.cleanup_context(tenant_id)
            self._async_contexts.clear()
            self.logger.info("Cleaned up all tenant contexts")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du gestionnaire de contextes"""
        with self._lock:
            return {
                "active_contexts": len(self._contexts),
                "async_contexts": len(self._async_contexts),
                "lifecycle_stats": dict(self._context_lifecycle),
                "memory_usage": self._get_memory_usage()
            }
    
    def _get_memory_usage(self) -> Dict[str, int]:
        """Calcule l'utilisation mémoire"""
        import sys
        total_size = 0
        for context in self._contexts.values():
            total_size += sys.getsizeof(context)
        
        return {
            "total_contexts_size": total_size,
            "average_context_size": total_size // max(len(self._contexts), 1)
        }


# Instance globale du gestionnaire de contextes
tenant_context_manager = TenantContextManager()


def get_current_tenant() -> Optional[TenantContext]:
    """Fonction utilitaire pour récupérer le tenant actuel"""
    # Try async context first
    try:
        async_context = tenant_context_manager.get_current_async_context()
        if async_context:
            return async_context
    except RuntimeError:
        # Not in async context
        pass
    
    # Fall back to thread local context
    return tenant_context_manager.get_current_context()


def require_tenant() -> TenantContext:
    """Fonction utilitaire qui requiert un tenant actuel"""
    context = get_current_tenant()
    if not context:
        raise TenantNotFoundError("No tenant context found")
    return context
