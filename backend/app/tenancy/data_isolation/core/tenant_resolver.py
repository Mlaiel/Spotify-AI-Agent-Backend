"""
🔍 Tenant Resolver - Résolution Intelligente des Tenants
=======================================================

Système ultra-avancé de résolution et reconnaissance automatique des tenants
avec support multi-sources et cache intelligent.

Author: Lead Dev + Architecte IA - Fahed Mlaiel
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
import urllib.parse
from abc import ABC, abstractmethod

from .tenant_context import TenantContext, TenantType, TenantMetadata, SecurityContext
from ..exceptions import TenantNotFoundError, SecurityViolationError
from ...utils.cache import CacheManager
from ...core.database import DatabaseManager


class ResolutionStrategy(Enum):
    """Stratégies de résolution des tenants"""
    HEADER = "header"           # Résolution par header HTTP
    SUBDOMAIN = "subdomain"     # Résolution par sous-domaine
    PATH = "path"              # Résolution par chemin URL
    TOKEN = "token"            # Résolution par token JWT
    API_KEY = "api_key"        # Résolution par clé API
    DATABASE = "database"       # Résolution par base de données
    HYBRID = "hybrid"          # Combinaison de plusieurs stratégies


class ResolutionSource(Enum):
    """Sources de résolution"""
    HTTP_REQUEST = "http_request"
    WEBSOCKET = "websocket"
    MESSAGE_QUEUE = "message_queue"
    BACKGROUND_TASK = "background_task"
    DIRECT = "direct"


@dataclass
class ResolutionConfig:
    """Configuration de la résolution des tenants"""
    primary_strategy: ResolutionStrategy = ResolutionStrategy.HEADER
    fallback_strategies: List[ResolutionStrategy] = field(default_factory=list)
    
    # Header-based resolution
    tenant_header_name: str = "X-Tenant-ID"
    tenant_type_header_name: str = "X-Tenant-Type"
    
    # Subdomain-based resolution
    subdomain_pattern: str = r"^([a-zA-Z0-9-]+)\."
    domain_mapping: Dict[str, str] = field(default_factory=dict)
    
    # Path-based resolution
    path_pattern: str = r"^/tenant/([a-zA-Z0-9-]+)/"
    path_prefix: str = "/tenant/"
    
    # Token-based resolution
    jwt_secret: Optional[str] = None
    jwt_algorithms: List[str] = field(default_factory=lambda: ["HS256"])
    
    # API Key resolution
    api_key_header: str = "X-API-Key"
    api_key_query_param: str = "api_key"
    
    # Caching
    cache_enabled: bool = True
    cache_ttl: int = 3600
    negative_cache_ttl: int = 300  # Cache for failed resolutions
    
    # Security
    validate_tenant_access: bool = True
    require_active_tenant: bool = True
    whitelist_enabled: bool = False
    blacklist_enabled: bool = True
    
    # Performance
    resolution_timeout: int = 5
    max_resolution_attempts: int = 3


@dataclass
class ResolutionResult:
    """Résultat de la résolution d'un tenant"""
    tenant_id: str
    tenant_type: TenantType
    strategy_used: ResolutionStrategy
    source: ResolutionSource
    metadata: Optional[TenantMetadata] = None
    security: Optional[SecurityContext] = None
    resolution_time: float = 0.0
    from_cache: bool = False
    confidence: float = 1.0  # Niveau de confiance (0.0 à 1.0)


class TenantResolver(ABC):
    """Interface abstraite pour les résolveurs de tenant"""
    
    @abstractmethod
    async def resolve(self, source_data: Dict[str, Any]) -> Optional[ResolutionResult]:
        """Résout un tenant à partir des données source"""
        pass
    
    @abstractmethod
    def can_handle(self, strategy: ResolutionStrategy) -> bool:
        """Vérifie si ce résolveur peut gérer la stratégie"""
        pass


class HeaderTenantResolver(TenantResolver):
    """Résolveur basé sur les headers HTTP"""
    
    def __init__(self, config: ResolutionConfig):
        self.config = config
        self.logger = logging.getLogger("resolver.header")
    
    def can_handle(self, strategy: ResolutionStrategy) -> bool:
        return strategy == ResolutionStrategy.HEADER
    
    async def resolve(self, source_data: Dict[str, Any]) -> Optional[ResolutionResult]:
        headers = source_data.get("headers", {})
        
        # Get tenant ID from header
        tenant_id = headers.get(self.config.tenant_header_name)
        if not tenant_id:
            return None
        
        # Get tenant type from header or default
        tenant_type_str = headers.get(self.config.tenant_type_header_name, "spotify_artist")
        try:
            tenant_type = TenantType(tenant_type_str)
        except ValueError:
            tenant_type = TenantType.SPOTIFY_ARTIST
        
        return ResolutionResult(
            tenant_id=tenant_id,
            tenant_type=tenant_type,
            strategy_used=ResolutionStrategy.HEADER,
            source=ResolutionSource.HTTP_REQUEST,
            confidence=1.0
        )


class SubdomainTenantResolver(TenantResolver):
    """Résolveur basé sur les sous-domaines"""
    
    def __init__(self, config: ResolutionConfig):
        self.config = config
        self.pattern = re.compile(config.subdomain_pattern)
        self.logger = logging.getLogger("resolver.subdomain")
    
    def can_handle(self, strategy: ResolutionStrategy) -> bool:
        return strategy == ResolutionStrategy.SUBDOMAIN
    
    async def resolve(self, source_data: Dict[str, Any]) -> Optional[ResolutionResult]:
        host = source_data.get("host")
        if not host:
            return None
        
        match = self.pattern.match(host)
        if not match:
            return None
        
        tenant_id = match.group(1)
        
        # Map subdomain to tenant type if configured
        tenant_type = TenantType.SPOTIFY_ARTIST
        if tenant_id in self.config.domain_mapping:
            tenant_type_str = self.config.domain_mapping[tenant_id]
            try:
                tenant_type = TenantType(tenant_type_str)
            except ValueError:
                pass
        
        return ResolutionResult(
            tenant_id=tenant_id,
            tenant_type=tenant_type,
            strategy_used=ResolutionStrategy.SUBDOMAIN,
            source=ResolutionSource.HTTP_REQUEST,
            confidence=0.9
        )


class PathTenantResolver(TenantResolver):
    """Résolveur basé sur le chemin URL"""
    
    def __init__(self, config: ResolutionConfig):
        self.config = config
        self.pattern = re.compile(config.path_pattern)
        self.logger = logging.getLogger("resolver.path")
    
    def can_handle(self, strategy: ResolutionStrategy) -> bool:
        return strategy == ResolutionStrategy.PATH
    
    async def resolve(self, source_data: Dict[str, Any]) -> Optional[ResolutionResult]:
        path = source_data.get("path")
        if not path:
            return None
        
        match = self.pattern.match(path)
        if not match:
            return None
        
        tenant_id = match.group(1)
        
        return ResolutionResult(
            tenant_id=tenant_id,
            tenant_type=TenantType.SPOTIFY_ARTIST,
            strategy_used=ResolutionStrategy.PATH,
            source=ResolutionSource.HTTP_REQUEST,
            confidence=0.8
        )


class TokenTenantResolver(TenantResolver):
    """Résolveur basé sur les tokens JWT"""
    
    def __init__(self, config: ResolutionConfig):
        self.config = config
        self.logger = logging.getLogger("resolver.token")
        
        try:
            import jwt
            self.jwt = jwt
        except ImportError:
            self.jwt = None
            self.logger.warning("PyJWT not installed, token resolution disabled")
    
    def can_handle(self, strategy: ResolutionStrategy) -> bool:
        return strategy == ResolutionStrategy.TOKEN and self.jwt is not None
    
    async def resolve(self, source_data: Dict[str, Any]) -> Optional[ResolutionResult]:
        if not self.jwt or not self.config.jwt_secret:
            return None
        
        # Get token from Authorization header
        auth_header = source_data.get("headers", {}).get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header[7:]  # Remove "Bearer " prefix
        
        try:
            payload = self.jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=self.config.jwt_algorithms
            )
            
            tenant_id = payload.get("tenant_id")
            if not tenant_id:
                return None
            
            tenant_type_str = payload.get("tenant_type", "spotify_artist")
            try:
                tenant_type = TenantType(tenant_type_str)
            except ValueError:
                tenant_type = TenantType.SPOTIFY_ARTIST
            
            # Extract security context from token
            security = SecurityContext(
                access_level=payload.get("access_level", "standard"),
                permissions=set(payload.get("permissions", [])),
                roles=set(payload.get("roles", []))
            )
            
            return ResolutionResult(
                tenant_id=tenant_id,
                tenant_type=tenant_type,
                strategy_used=ResolutionStrategy.TOKEN,
                source=ResolutionSource.HTTP_REQUEST,
                security=security,
                confidence=1.0
            )
            
        except self.jwt.InvalidTokenError:
            self.logger.warning("Invalid JWT token provided")
            return None


class DatabaseTenantResolver(TenantResolver):
    """Résolveur basé sur la base de données"""
    
    def __init__(self, config: ResolutionConfig):
        self.config = config
        self.logger = logging.getLogger("resolver.database")
        self.db_manager = DatabaseManager()
    
    def can_handle(self, strategy: ResolutionStrategy) -> bool:
        return strategy == ResolutionStrategy.DATABASE
    
    async def resolve(self, source_data: Dict[str, Any]) -> Optional[ResolutionResult]:
        # This would query the database for tenant information
        tenant_id = source_data.get("tenant_id")
        if not tenant_id:
            return None
        
        try:
            # Mock database query - replace with actual implementation
            tenant_data = await self._query_tenant_data(tenant_id)
            if not tenant_data:
                return None
            
            tenant_type = TenantType(tenant_data.get("type", "spotify_artist"))
            
            metadata = TenantMetadata(
                region=tenant_data.get("region", "EU"),
                timezone=tenant_data.get("timezone", "UTC"),
                locale=tenant_data.get("locale", "en-US"),
                tier=tenant_data.get("tier", "standard")
            )
            
            return ResolutionResult(
                tenant_id=tenant_id,
                tenant_type=tenant_type,
                strategy_used=ResolutionStrategy.DATABASE,
                source=ResolutionSource.DIRECT,
                metadata=metadata,
                confidence=1.0
            )
            
        except Exception as e:
            self.logger.error(f"Database resolution failed for tenant {tenant_id}: {e}")
            return None
    
    async def _query_tenant_data(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Query tenant data from database"""
        # Implementation would use actual database connection
        # This is a mock implementation
        return {
            "id": tenant_id,
            "type": "spotify_artist",
            "region": "EU",
            "timezone": "UTC",
            "locale": "en-US",
            "tier": "premium"
        }


class SmartTenantResolver:
    """
    Résolveur intelligent de tenants avec support multi-stratégies
    
    Features:
    - Support de multiples stratégies de résolution
    - Cache intelligent avec TTL configurable
    - Fallback automatique entre stratégies
    - Validation de sécurité
    - Métriques et monitoring
    - Performance optimisée
    """
    
    def __init__(self, config: ResolutionConfig):
        self.config = config
        self.logger = logging.getLogger("tenant.resolver")
        
        # Initialize resolvers
        self.resolvers: Dict[ResolutionStrategy, TenantResolver] = {
            ResolutionStrategy.HEADER: HeaderTenantResolver(config),
            ResolutionStrategy.SUBDOMAIN: SubdomainTenantResolver(config),
            ResolutionStrategy.PATH: PathTenantResolver(config),
            ResolutionStrategy.TOKEN: TokenTenantResolver(config),
            ResolutionStrategy.DATABASE: DatabaseTenantResolver(config)
        }
        
        # Cache manager
        self.cache_manager = CacheManager() if config.cache_enabled else None
        
        # Statistics
        self.resolution_stats = {
            "total_resolutions": 0,
            "cache_hits": 0,
            "strategy_usage": {strategy.value: 0 for strategy in ResolutionStrategy},
            "average_resolution_time": 0.0
        }
        
        # Security lists
        self.tenant_whitelist: set = set()
        self.tenant_blacklist: set = set()
    
    async def resolve_tenant(
        self, 
        source_data: Dict[str, Any],
        source: ResolutionSource = ResolutionSource.HTTP_REQUEST
    ) -> Optional[TenantContext]:
        """
        Résout un tenant à partir des données source
        
        Args:
            source_data: Données source (headers, host, path, etc.)
            source: Source de la demande de résolution
            
        Returns:
            Contexte tenant ou None si non trouvé
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(source_data, source)
            if self.cache_manager:
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    self.resolution_stats["cache_hits"] += 1
                    return await self._create_tenant_context(cached_result, from_cache=True)
            
            # Try primary strategy first
            result = await self._resolve_with_strategy(
                self.config.primary_strategy,
                source_data,
                source
            )
            
            # Try fallback strategies if primary failed
            if not result:
                for strategy in self.config.fallback_strategies:
                    result = await self._resolve_with_strategy(strategy, source_data, source)
                    if result:
                        break
            
            if not result:
                return None
            
            # Validate resolved tenant
            if not await self._validate_tenant(result):
                return None
            
            # Cache the result
            if self.cache_manager:
                await self.cache_manager.set(
                    cache_key,
                    result,
                    ttl=self.config.cache_ttl
                )
            
            # Update statistics
            resolution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            result.resolution_time = resolution_time
            self._update_stats(result.strategy_used, resolution_time)
            
            # Create tenant context
            return await self._create_tenant_context(result)
            
        except Exception as e:
            self.logger.error(f"Tenant resolution failed: {e}")
            return None
    
    async def _resolve_with_strategy(
        self,
        strategy: ResolutionStrategy,
        source_data: Dict[str, Any],
        source: ResolutionSource
    ) -> Optional[ResolutionResult]:
        """Résout avec une stratégie spécifique"""
        resolver = self.resolvers.get(strategy)
        if not resolver or not resolver.can_handle(strategy):
            return None
        
        try:
            result = await asyncio.wait_for(
                resolver.resolve(source_data),
                timeout=self.config.resolution_timeout
            )
            
            if result:
                result.source = source
                self.logger.debug(f"Tenant resolved using {strategy.value}: {result.tenant_id}")
            
            return result
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Tenant resolution timeout with strategy {strategy.value}")
            return None
        except Exception as e:
            self.logger.error(f"Resolution error with strategy {strategy.value}: {e}")
            return None
    
    async def _validate_tenant(self, result: ResolutionResult) -> bool:
        """Valide un tenant résolu"""
        if not self.config.validate_tenant_access:
            return True
        
        # Check whitelist
        if self.config.whitelist_enabled and self.tenant_whitelist:
            if result.tenant_id not in self.tenant_whitelist:
                self.logger.warning(f"Tenant {result.tenant_id} not in whitelist")
                return False
        
        # Check blacklist
        if self.config.blacklist_enabled and result.tenant_id in self.tenant_blacklist:
            self.logger.warning(f"Tenant {result.tenant_id} is blacklisted")
            return False
        
        # Check if tenant is active (if required)
        if self.config.require_active_tenant:
            # This would check database for tenant status
            # For now, assume all tenants are active
            pass
        
        return True
    
    async def _create_tenant_context(
        self, 
        result: ResolutionResult,
        from_cache: bool = False
    ) -> TenantContext:
        """Crée un contexte tenant à partir du résultat"""
        result.from_cache = from_cache
        
        context = TenantContext(
            tenant_id=result.tenant_id,
            tenant_type=result.tenant_type,
            metadata=result.metadata,
            security=result.security
        )
        
        self.logger.debug(f"Created tenant context: {context.tenant_id}")
        return context
    
    def _generate_cache_key(
        self, 
        source_data: Dict[str, Any],
        source: ResolutionSource
    ) -> str:
        """Génère une clé de cache pour les données source"""
        import hashlib
        
        # Create a deterministic hash of the source data
        data_str = str(sorted(source_data.items())) + source.value
        return f"tenant_resolution:{hashlib.md5(data_str.encode()).hexdigest()}"
    
    def _update_stats(self, strategy: ResolutionStrategy, resolution_time: float):
        """Met à jour les statistiques de résolution"""
        self.resolution_stats["total_resolutions"] += 1
        self.resolution_stats["strategy_usage"][strategy.value] += 1
        
        # Update average resolution time
        total = self.resolution_stats["total_resolutions"]
        current_avg = self.resolution_stats["average_resolution_time"]
        new_avg = ((current_avg * (total - 1)) + resolution_time) / total
        self.resolution_stats["average_resolution_time"] = new_avg
    
    def add_to_whitelist(self, tenant_id: str):
        """Ajoute un tenant à la whitelist"""
        self.tenant_whitelist.add(tenant_id)
        self.logger.info(f"Added tenant {tenant_id} to whitelist")
    
    def add_to_blacklist(self, tenant_id: str):
        """Ajoute un tenant à la blacklist"""
        self.tenant_blacklist.add(tenant_id)
        self.logger.warning(f"Added tenant {tenant_id} to blacklist")
    
    def remove_from_whitelist(self, tenant_id: str):
        """Supprime un tenant de la whitelist"""
        self.tenant_whitelist.discard(tenant_id)
        self.logger.info(f"Removed tenant {tenant_id} from whitelist")
    
    def remove_from_blacklist(self, tenant_id: str):
        """Supprime un tenant de la blacklist"""
        self.tenant_blacklist.discard(tenant_id)
        self.logger.info(f"Removed tenant {tenant_id} from blacklist")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de résolution"""
        return {
            "resolution_stats": dict(self.resolution_stats),
            "cache_enabled": self.config.cache_enabled,
            "primary_strategy": self.config.primary_strategy.value,
            "fallback_strategies": [s.value for s in self.config.fallback_strategies],
            "whitelist_size": len(self.tenant_whitelist),
            "blacklist_size": len(self.tenant_blacklist)
        }
    
    async def clear_cache(self):
        """Vide le cache de résolution"""
        if self.cache_manager:
            await self.cache_manager.clear_prefix("tenant_resolution:")
            self.logger.info("Tenant resolution cache cleared")


# Factory function
def create_tenant_resolver(
    primary_strategy: ResolutionStrategy = ResolutionStrategy.HEADER,
    **kwargs
) -> SmartTenantResolver:
    """Factory pour créer un résolveur de tenant"""
    config = ResolutionConfig(
        primary_strategy=primary_strategy,
        **kwargs
    )
    return SmartTenantResolver(config)


# Global resolver instance
_global_resolver: Optional[SmartTenantResolver] = None


def get_tenant_resolver() -> SmartTenantResolver:
    """Retourne l'instance globale du résolveur"""
    global _global_resolver
    if _global_resolver is None:
        _global_resolver = create_tenant_resolver()
    return _global_resolver


def set_tenant_resolver(resolver: SmartTenantResolver):
    """Définit l'instance globale du résolveur"""
    global _global_resolver
    _global_resolver = resolver
