#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gestionnaire de Contexte Multi-Tenant avec Isolation Stricte

Ce module fournit un système de gestion de contexte tenant avec:
- Isolation stricte des données par tenant
- Configuration hiérarchique par tenant
- Validation et contrôle d'accès RBAC
- Cache sécurisé des contextes
- Audit trail complet
- Support de la personnalisation par tenant
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import structlog
from prometheus_client import Counter, Histogram, Gauge
import hashlib
from pathlib import Path

logger = structlog.get_logger(__name__)

# Métriques Prometheus
TENANT_CONTEXT_OPERATIONS = Counter(
    'slack_tenant_context_operations_total',
    'Total tenant context operations',
    ['operation', 'tenant', 'result']
)

TENANT_CONTEXT_DURATION = Histogram(
    'slack_tenant_context_duration_seconds',
    'Tenant context operation duration',
    ['operation']
)

ACTIVE_TENANTS = Gauge(
    'slack_tenant_contexts_active',
    'Number of active tenant contexts'
)

class IsolationLevel(Enum):
    """Niveaux d'isolation des données."""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"

class PermissionLevel(Enum):
    """Niveaux de permissions."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    SYSTEM = "system"

@dataclass
class TenantPermissions:
    """Permissions spécifiques à un tenant."""
    level: PermissionLevel
    resources: Set[str] = field(default_factory=set)
    operations: Set[str] = field(default_factory=set)
    restrictions: Dict[str, Any] = field(default_factory=dict)
    
    def can_access(self, resource: str, operation: str) -> bool:
        """Vérifie si l'accès est autorisé."""
        if self.level == PermissionLevel.SYSTEM:
            return True
        
        if resource in self.resources and operation in self.operations:
            return True
        
        return False

@dataclass
class TenantConfig:
    """Configuration complète d'un tenant."""
    id: str
    name: str
    
    # Configuration Slack
    slack_config: Dict[str, Any] = field(default_factory=dict)
    
    # Localisation
    locale: str = "fr_FR"
    timezone: str = "UTC"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Alertes
    alert_config: Dict[str, Any] = field(default_factory=dict)
    escalation_policy: Optional[str] = None
    
    # Sécurité
    permissions: TenantPermissions = field(default_factory=lambda: TenantPermissions(PermissionLevel.READ))
    encryption_key: Optional[str] = None
    
    # Personnalisation
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    branding: Dict[str, Any] = field(default_factory=dict)
    
    # Métadonnées
    enabled: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()

@dataclass
class TenantContext:
    """Contexte d'exécution pour un tenant."""
    tenant: TenantConfig
    request_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

class TenantContextManager:
    """
    Gestionnaire de contexte multi-tenant avec isolation stricte.
    """
    
    def __init__(self, config: Dict[str, Any], cache_manager=None):
        self.config = config
        self.cache_manager = cache_manager
        self.logger = logger.bind(component="tenant_context_manager")
        
        # Configuration
        self.isolation_level = IsolationLevel(config.get("isolation_level", "strict"))
        self.cache_ttl = config.get("cache_ttl", 600)
        self.max_tenants = config.get("max_tenants", 1000)
        self.audit_enabled = config.get("audit_enabled", True)
        self.encryption_enabled = config.get("encryption_enabled", True)
        
        # Stockage interne
        self._tenant_configs: Dict[str, TenantConfig] = {}
        self._active_contexts: Dict[str, TenantContext] = {}
        self._tenant_permissions: Dict[str, TenantPermissions] = {}
        
        # Cache des configurations
        self._config_cache: Dict[str, TenantConfig] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Audit trail
        self._audit_log: List[Dict[str, Any]] = []
        
        # Lock pour thread-safety
        self._lock = asyncio.Lock()
        
        # Initialisation
        asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Initialise le gestionnaire de contexte."""
        try:
            await self._load_tenant_configs()
            await self._setup_default_permissions()
            
            self.logger.info(
                "Gestionnaire de contexte tenant initialisé",
                tenants_count=len(self._tenant_configs),
                isolation_level=self.isolation_level.value
            )
            
            ACTIVE_TENANTS.set(len(self._tenant_configs))
            
        except Exception as e:
            self.logger.error("Erreur lors de l'initialisation", error=str(e))
            raise
    
    async def _load_tenant_configs(self):
        """Charge les configurations des tenants."""
        # Charger depuis les fichiers de configuration
        configs_dir = Path(__file__).parent / "configs" / "tenants"
        
        if configs_dir.exists():
            for config_file in configs_dir.glob("*.json"):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    
                    tenant_config = self._parse_tenant_config(config_data)
                    self._tenant_configs[tenant_config.id] = tenant_config
                    
                    self.logger.debug(
                        "Configuration tenant chargée",
                        tenant_id=tenant_config.id,
                        file=str(config_file)
                    )
                    
                except Exception as e:
                    self.logger.error(
                        "Erreur chargement config tenant",
                        file=str(config_file),
                        error=str(e)
                    )
        
        # Créer des configurations par défaut si aucune n'existe
        if not self._tenant_configs:
            await self._create_default_tenant_configs()
    
    def _parse_tenant_config(self, data: Dict[str, Any]) -> TenantConfig:
        """Parse les données de configuration en objet TenantConfig."""
        permissions_data = data.get("permissions", {})
        permissions = TenantPermissions(
            level=PermissionLevel(permissions_data.get("level", "read")),
            resources=set(permissions_data.get("resources", [])),
            operations=set(permissions_data.get("operations", [])),
            restrictions=permissions_data.get("restrictions", {})
        )
        
        return TenantConfig(
            id=data["id"],
            name=data["name"],
            slack_config=data.get("slack_config", {}),
            locale=data.get("locale", "fr_FR"),
            timezone=data.get("timezone", "UTC"),
            date_format=data.get("date_format", "%Y-%m-%d %H:%M:%S"),
            alert_config=data.get("alert_config", {}),
            escalation_policy=data.get("escalation_policy"),
            permissions=permissions,
            encryption_key=data.get("encryption_key"),
            custom_fields=data.get("custom_fields", {}),
            branding=data.get("branding", {}),
            enabled=data.get("enabled", True),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else None
        )
    
    async def _create_default_tenant_configs(self):
        """Crée des configurations de tenant par défaut."""
        default_tenants = [
            {
                "id": "spotify_free",
                "name": "Spotify Free",
                "slack_config": {
                    "channel": "#alerts-free",
                    "bot_name": "Spotify AI Free"
                },
                "locale": "fr_FR",
                "alert_config": {
                    "max_alerts_per_hour": 10,
                    "severity_filter": ["warning", "critical"]
                },
                "permissions": {
                    "level": "read",
                    "resources": ["alerts", "notifications"],
                    "operations": ["view"]
                }
            },
            {
                "id": "spotify_premium",
                "name": "Spotify Premium", 
                "slack_config": {
                    "channel": "#alerts-premium",
                    "bot_name": "Spotify AI Premium"
                },
                "locale": "fr_FR",
                "alert_config": {
                    "max_alerts_per_hour": 50,
                    "severity_filter": ["info", "warning", "critical"]
                },
                "permissions": {
                    "level": "write",
                    "resources": ["alerts", "notifications", "templates"],
                    "operations": ["view", "create", "update"]
                }
            }
        ]
        
        for tenant_data in default_tenants:
            tenant_config = self._parse_tenant_config(tenant_data)
            self._tenant_configs[tenant_config.id] = tenant_config
    
    async def _setup_default_permissions(self):
        """Configure les permissions par défaut."""
        for tenant_id, config in self._tenant_configs.items():
            self._tenant_permissions[tenant_id] = config.permissions
    
    async def get_tenant_context(
        self,
        tenant_id: str,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs
    ) -> Optional[TenantContext]:
        """
        Récupère le contexte d'un tenant.
        
        Args:
            tenant_id: ID du tenant
            request_id: ID de la requête
            user_id: ID de l'utilisateur
            **kwargs: Contexte additionnel
            
        Returns:
            Contexte tenant ou None si non trouvé
        """
        start_time = datetime.utcnow()
        
        try:
            # Vérifier le cache d'abord
            cached_config = await self._get_cached_config(tenant_id)
            if cached_config is None:
                # Charger depuis le stockage
                tenant_config = self._tenant_configs.get(tenant_id)
                if tenant_config is None:
                    self.logger.warning(
                        "Tenant non trouvé",
                        tenant_id=tenant_id,
                        user_id=user_id
                    )
                    return None
                
                # Mettre en cache
                await self._cache_config(tenant_id, tenant_config)
                cached_config = tenant_config
            
            # Vérifier si le tenant est activé
            if not cached_config.enabled:
                self.logger.warning(
                    "Tenant désactivé",
                    tenant_id=tenant_id
                )
                return None
            
            # Créer le contexte
            context = TenantContext(
                tenant=cached_config,
                request_id=request_id or self._generate_request_id(),
                user_id=user_id,
                additional_context=kwargs
            )
            
            # Stocker le contexte actif
            async with self._lock:
                self._active_contexts[context.request_id] = context
                
                # Nettoyer les anciens contextes
                await self._cleanup_old_contexts()
            
            # Mettre à jour la dernière activité
            cached_config.last_activity = datetime.utcnow()
            
            # Audit
            if self.audit_enabled:
                await self._audit_context_access(tenant_id, context)
            
            # Métriques
            TENANT_CONTEXT_OPERATIONS.labels(
                operation="get",
                tenant=tenant_id,
                result="success"
            ).inc()
            
            return context
            
        except Exception as e:
            self.logger.error(
                "Erreur récupération contexte tenant",
                tenant_id=tenant_id,
                error=str(e)
            )
            
            TENANT_CONTEXT_OPERATIONS.labels(
                operation="get",
                tenant=tenant_id,
                result="error"
            ).inc()
            
            return None
        
        finally:
            duration = (datetime.utcnow() - start_time).total_seconds()
            TENANT_CONTEXT_DURATION.labels(operation="get").observe(duration)
    
    async def _get_cached_config(self, tenant_id: str) -> Optional[TenantConfig]:
        """Récupère une configuration depuis le cache."""
        # Cache local d'abord
        if tenant_id in self._config_cache:
            cache_time = self._cache_timestamps.get(tenant_id)
            if cache_time and datetime.utcnow() - cache_time < timedelta(seconds=self.cache_ttl):
                return self._config_cache[tenant_id]
        
        # Cache Redis si disponible
        if self.cache_manager:
            try:
                cached_data = await self.cache_manager.get(f"tenant_config:{tenant_id}")
                if cached_data:
                    return self._parse_tenant_config(cached_data)
            except Exception as e:
                self.logger.warning("Erreur cache Redis", error=str(e))
        
        return None
    
    async def _cache_config(self, tenant_id: str, config: TenantConfig):
        """Met une configuration en cache."""
        # Cache local
        self._config_cache[tenant_id] = config
        self._cache_timestamps[tenant_id] = datetime.utcnow()
        
        # Cache Redis
        if self.cache_manager:
            try:
                config_data = {
                    "id": config.id,
                    "name": config.name,
                    "slack_config": config.slack_config,
                    "locale": config.locale,
                    "timezone": config.timezone,
                    "date_format": config.date_format,
                    "alert_config": config.alert_config,
                    "escalation_policy": config.escalation_policy,
                    "permissions": {
                        "level": config.permissions.level.value,
                        "resources": list(config.permissions.resources),
                        "operations": list(config.permissions.operations),
                        "restrictions": config.permissions.restrictions
                    },
                    "custom_fields": config.custom_fields,
                    "branding": config.branding,
                    "enabled": config.enabled,
                    "created_at": config.created_at.isoformat() if config.created_at else None,
                    "updated_at": config.updated_at.isoformat() if config.updated_at else None
                }
                
                await self.cache_manager.set(
                    f"tenant_config:{tenant_id}",
                    config_data,
                    ttl=self.cache_ttl
                )
            except Exception as e:
                self.logger.warning("Erreur mise en cache Redis", error=str(e))
    
    def _generate_request_id(self) -> str:
        """Génère un ID de requête unique."""
        import uuid
        return str(uuid.uuid4())
    
    async def _cleanup_old_contexts(self):
        """Nettoie les anciens contextes actifs."""
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        old_contexts = [
            request_id for request_id, context in self._active_contexts.items()
            if context.created_at < cutoff_time
        ]
        
        for request_id in old_contexts:
            del self._active_contexts[request_id]
    
    async def _audit_context_access(self, tenant_id: str, context: TenantContext):
        """Enregistre l'accès au contexte dans l'audit trail."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "context_access",
            "tenant_id": tenant_id,
            "request_id": context.request_id,
            "user_id": context.user_id,
            "ip_address": context.ip_address,
            "user_agent": context.user_agent
        }
        
        self._audit_log.append(audit_entry)
        
        # Limiter la taille du log
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-5000:]
    
    async def set_tenant_context(
        self,
        tenant_id: str,
        config_updates: Dict[str, Any]
    ):
        """
        Met à jour la configuration d'un tenant.
        
        Args:
            tenant_id: ID du tenant
            config_updates: Mises à jour de configuration
        """
        start_time = datetime.utcnow()
        
        try:
            async with self._lock:
                # Récupérer la configuration existante
                existing_config = self._tenant_configs.get(tenant_id)
                if existing_config is None:
                    raise ValueError(f"Tenant {tenant_id} non trouvé")
                
                # Appliquer les mises à jour
                updated_config = self._apply_config_updates(existing_config, config_updates)
                
                # Valider la configuration
                await self._validate_tenant_config(updated_config)
                
                # Sauvegarder
                self._tenant_configs[tenant_id] = updated_config
                
                # Invalider le cache
                if tenant_id in self._config_cache:
                    del self._config_cache[tenant_id]
                    del self._cache_timestamps[tenant_id]
                
                if self.cache_manager:
                    await self.cache_manager.delete(f"tenant_config:{tenant_id}")
            
            # Audit
            if self.audit_enabled:
                await self._audit_config_update(tenant_id, config_updates)
            
            # Métriques
            TENANT_CONTEXT_OPERATIONS.labels(
                operation="update",
                tenant=tenant_id,
                result="success"
            ).inc()
            
            self.logger.info(
                "Configuration tenant mise à jour",
                tenant_id=tenant_id,
                updates=list(config_updates.keys())
            )
            
        except Exception as e:
            self.logger.error(
                "Erreur mise à jour contexte tenant",
                tenant_id=tenant_id,
                error=str(e)
            )
            
            TENANT_CONTEXT_OPERATIONS.labels(
                operation="update",
                tenant=tenant_id,
                result="error"
            ).inc()
            
            raise
        
        finally:
            duration = (datetime.utcnow() - start_time).total_seconds()
            TENANT_CONTEXT_DURATION.labels(operation="update").observe(duration)
    
    def _apply_config_updates(
        self,
        config: TenantConfig,
        updates: Dict[str, Any]
    ) -> TenantConfig:
        """Applique les mises à jour à une configuration."""
        # Créer une copie de la configuration
        import copy
        updated_config = copy.deepcopy(config)
        
        # Appliquer les mises à jour
        for key, value in updates.items():
            if hasattr(updated_config, key):
                if key == "permissions" and isinstance(value, dict):
                    # Traitement spécial pour les permissions
                    if "level" in value:
                        updated_config.permissions.level = PermissionLevel(value["level"])
                    if "resources" in value:
                        updated_config.permissions.resources = set(value["resources"])
                    if "operations" in value:
                        updated_config.permissions.operations = set(value["operations"])
                    if "restrictions" in value:
                        updated_config.permissions.restrictions.update(value["restrictions"])
                else:
                    setattr(updated_config, key, value)
        
        # Mettre à jour le timestamp
        updated_config.updated_at = datetime.utcnow()
        
        return updated_config
    
    async def _validate_tenant_config(self, config: TenantConfig):
        """Valide une configuration de tenant."""
        # Validation de base
        if not config.id or not config.name:
            raise ValueError("ID et nom du tenant obligatoires")
        
        # Validation de la locale
        if config.locale not in ["fr_FR", "en_US", "de_DE", "es_ES"]:
            raise ValueError(f"Locale non supportée: {config.locale}")
        
        # Validation des permissions selon le niveau d'isolation
        if self.isolation_level in [IsolationLevel.STRICT, IsolationLevel.PARANOID]:
            if not config.permissions.resources:
                raise ValueError("Ressources obligatoires en mode strict")
        
        # Validation personnalisée
        await self._custom_validation(config)
    
    async def _custom_validation(self, config: TenantConfig):
        """Validation personnalisée selon la logique métier."""
        # Vérifier les limits selon le type de tenant
        if "free" in config.id.lower():
            max_alerts = config.alert_config.get("max_alerts_per_hour", 0)
            if max_alerts > 20:
                raise ValueError("Limite d'alertes dépassée pour tenant free")
        
        # Vérifier la configuration Slack
        slack_config = config.slack_config
        if slack_config.get("channel") and not slack_config["channel"].startswith("#"):
            raise ValueError("Canal Slack doit commencer par #")
    
    async def _audit_config_update(self, tenant_id: str, updates: Dict[str, Any]):
        """Enregistre la mise à jour de configuration."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "config_update",
            "tenant_id": tenant_id,
            "updates": updates,
            "isolation_level": self.isolation_level.value
        }
        
        self._audit_log.append(audit_entry)
    
    async def check_permissions(
        self,
        tenant_id: str,
        resource: str,
        operation: str,
        context: Optional[TenantContext] = None
    ) -> bool:
        """
        Vérifie les permissions d'accès pour un tenant.
        
        Args:
            tenant_id: ID du tenant
            resource: Ressource demandée
            operation: Opération demandée
            context: Contexte additionnel
            
        Returns:
            True si autorisé, False sinon
        """
        try:
            permissions = self._tenant_permissions.get(tenant_id)
            if permissions is None:
                return False
            
            # Vérification de base
            if not permissions.can_access(resource, operation):
                return False
            
            # Vérifications additionnelles selon l'isolation
            if self.isolation_level == IsolationLevel.PARANOID:
                # En mode paranoid, vérifier le contexte utilisateur
                if context and not context.user_id:
                    return False
                
                # Vérifier les restrictions temporelles
                restrictions = permissions.restrictions
                if "time_restrictions" in restrictions:
                    current_hour = datetime.utcnow().hour
                    allowed_hours = restrictions["time_restrictions"].get("allowed_hours", [])
                    if allowed_hours and current_hour not in allowed_hours:
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Erreur vérification permissions",
                tenant_id=tenant_id,
                resource=resource,
                operation=operation,
                error=str(e)
            )
            return False
    
    async def list_tenants(
        self,
        include_disabled: bool = False,
        filter_permissions: Optional[PermissionLevel] = None
    ) -> List[Dict[str, Any]]:
        """Liste tous les tenants avec leurs métadonnées."""
        tenants_info = []
        
        for tenant_id, config in self._tenant_configs.items():
            if not include_disabled and not config.enabled:
                continue
            
            if filter_permissions and config.permissions.level != filter_permissions:
                continue
            
            tenant_info = {
                "id": config.id,
                "name": config.name,
                "enabled": config.enabled,
                "locale": config.locale,
                "timezone": config.timezone,
                "permissions_level": config.permissions.level.value,
                "last_activity": config.last_activity.isoformat() if config.last_activity else None,
                "created_at": config.created_at.isoformat() if config.created_at else None,
                "updated_at": config.updated_at.isoformat() if config.updated_at else None
            }
            
            # Ajouter les informations de contexte actif
            active_contexts = [
                ctx for ctx in self._active_contexts.values()
                if ctx.tenant.id == tenant_id
            ]
            tenant_info["active_contexts"] = len(active_contexts)
            
            tenants_info.append(tenant_info)
        
        return tenants_info
    
    async def get_audit_trail(
        self,
        tenant_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Récupère l'audit trail."""
        if tenant_id:
            # Filtrer par tenant
            filtered_logs = [
                entry for entry in self._audit_log
                if entry.get("tenant_id") == tenant_id
            ]
        else:
            filtered_logs = self._audit_log
        
        # Retourner les plus récents
        return sorted(filtered_logs, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé du gestionnaire de contexte."""
        try:
            return {
                "status": "healthy",
                "isolation_level": self.isolation_level.value,
                "tenants_count": len(self._tenant_configs),
                "active_contexts": len(self._active_contexts),
                "cache_enabled": self.cache_manager is not None,
                "audit_enabled": self.audit_enabled,
                "last_check": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat()
            }

# Factory function
def create_tenant_context_manager(config: Dict[str, Any], cache_manager=None) -> TenantContextManager:
    """Crée une instance du gestionnaire de contexte tenant."""
    return TenantContextManager(config, cache_manager)
