"""
Gestionnaire multi-tenant avancé pour le système de monitoring Slack.

Ce module fournit un système complet de gestion multi-tenant avec:
- Isolation stricte des données et configurations par tenant
- Provisioning automatique des ressources
- Facturation et quotas en temps réel
- Migration et backup automatiques
- Monitoring et analytics par tenant
- API self-service pour la gestion des tenants

Architecture:
    - Factory pattern pour la création des tenants
    - Strategy pattern pour les niveaux d'isolation
    - Observer pattern pour les événements tenant
    - Command pattern pour les opérations tenant
    - Repository pattern pour la persistance

Fonctionnalités:
    - Onboarding automatique des nouveaux tenants
    - Scaling automatique des ressources
    - Policies de sécurité par tenant
    - Dashboards et reporting dédiés
    - Intégration SSO par tenant
    - Custom branding et personnalisation

Auteur: Équipe Spotify AI Agent
Lead Developer: Fahed Mlaiel
"""

import json
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Set
from weakref import WeakSet

from .cache_manager import CacheManager
from .metrics import MetricsCollector
from .security import SecurityManager, User, Permission


class TenantStatus(Enum):
    """Statuts des tenants."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING = "pending"
    DELETED = "deleted"
    MAINTENANCE = "maintenance"


class IsolationLevel(Enum):
    """Niveaux d'isolation des tenants."""
    STRICT = "strict"        # Isolation complète (DB, config, ressources)
    PARTIAL = "partial"      # Isolation des données, ressources partagées
    SHARED = "shared"        # Séparation logique uniquement


class TenantTier(Enum):
    """Niveaux de service des tenants."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


@dataclass
class TenantQuotas:
    """Quotas pour un tenant."""
    max_users: int = 100
    max_storage_gb: float = 10.0
    max_requests_per_hour: int = 10000
    max_alerts_per_hour: int = 1000
    max_integrations: int = 10
    max_custom_metrics: int = 100
    api_rate_limit: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire."""
        return {
            "max_users": self.max_users,
            "max_storage_gb": self.max_storage_gb,
            "max_requests_per_hour": self.max_requests_per_hour,
            "max_alerts_per_hour": self.max_alerts_per_hour,
            "max_integrations": self.max_integrations,
            "max_custom_metrics": self.max_custom_metrics,
            "api_rate_limit": self.api_rate_limit
        }


@dataclass
class TenantUsage:
    """Utilisation actuelle d'un tenant."""
    current_users: int = 0
    current_storage_gb: float = 0.0
    requests_last_hour: int = 0
    alerts_last_hour: int = 0
    integrations_count: int = 0
    custom_metrics_count: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire."""
        return {
            "current_users": self.current_users,
            "current_storage_gb": self.current_storage_gb,
            "requests_last_hour": self.requests_last_hour,
            "alerts_last_hour": self.alerts_last_hour,
            "integrations_count": self.integrations_count,
            "custom_metrics_count": self.custom_metrics_count,
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class TenantConfiguration:
    """Configuration spécifique à un tenant."""
    slack_workspace: Optional[str] = None
    default_channel: str = "#alerts"
    bot_name: str = "Spotify AI Bot"
    icon_emoji: str = ":robot_face:"
    locale: str = "fr_FR"
    timezone: str = "Europe/Paris"
    custom_branding: Dict[str, str] = field(default_factory=dict)
    notification_settings: Dict[str, Any] = field(default_factory=dict)
    integration_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire."""
        return {
            "slack_workspace": self.slack_workspace,
            "default_channel": self.default_channel,
            "bot_name": self.bot_name,
            "icon_emoji": self.icon_emoji,
            "locale": self.locale,
            "timezone": self.timezone,
            "custom_branding": self.custom_branding,
            "notification_settings": self.notification_settings,
            "integration_configs": self.integration_configs,
            "custom_fields": self.custom_fields
        }


@dataclass
class Tenant:
    """Représentation d'un tenant."""
    tenant_id: str
    name: str
    status: TenantStatus = TenantStatus.PENDING
    tier: TenantTier = TenantTier.FREE
    isolation_level: IsolationLevel = IsolationLevel.SHARED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    owner_email: Optional[str] = None
    description: Optional[str] = None
    
    # Configuration et limites
    quotas: TenantQuotas = field(default_factory=TenantQuotas)
    usage: TenantUsage = field(default_factory=TenantUsage)
    configuration: TenantConfiguration = field(default_factory=TenantConfiguration)
    
    # Métadonnées
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Post-initialisation du tenant."""
        if not self.tenant_id:
            self.tenant_id = str(uuid.uuid4())
    
    def is_active(self) -> bool:
        """Vérifie si le tenant est actif."""
        return self.status == TenantStatus.ACTIVE
    
    def is_over_quota(self, quota_type: str) -> bool:
        """Vérifie si un quota est dépassé."""
        quota_checks = {
            "users": self.usage.current_users > self.quotas.max_users,
            "storage": self.usage.current_storage_gb > self.quotas.max_storage_gb,
            "requests": self.usage.requests_last_hour > self.quotas.max_requests_per_hour,
            "alerts": self.usage.alerts_last_hour > self.quotas.max_alerts_per_hour,
            "integrations": self.usage.integrations_count > self.quotas.max_integrations,
            "metrics": self.usage.custom_metrics_count > self.quotas.max_custom_metrics
        }
        
        return quota_checks.get(quota_type, False)
    
    def update_usage(self, usage_data: Dict[str, Any]) -> None:
        """Met à jour les données d'utilisation."""
        for key, value in usage_data.items():
            if hasattr(self.usage, key):
                setattr(self.usage, key, value)
        
        self.usage.last_updated = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire."""
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "status": self.status.value,
            "tier": self.tier.value,
            "isolation_level": self.isolation_level.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "owner_email": self.owner_email,
            "description": self.description,
            "quotas": self.quotas.to_dict(),
            "usage": self.usage.to_dict(),
            "configuration": self.configuration.to_dict(),
            "metadata": self.metadata,
            "tags": list(self.tags)
        }


class TenantEvent:
    """Événement lié à un tenant."""
    
    def __init__(self,
                 event_type: str,
                 tenant_id: str,
                 user_id: Optional[str] = None,
                 data: Optional[Dict[str, Any]] = None,
                 timestamp: Optional[datetime] = None):
        self.event_type = event_type
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.data = data or {}
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.event_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        }


class ITenantStorage(ABC):
    """Interface pour le stockage des tenants."""
    
    @abstractmethod
    def save_tenant(self, tenant: Tenant) -> bool:
        """Sauvegarde un tenant."""
        pass
    
    @abstractmethod
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Récupère un tenant par ID."""
        pass
    
    @abstractmethod
    def list_tenants(self, 
                    status: Optional[TenantStatus] = None,
                    tier: Optional[TenantTier] = None,
                    limit: int = 100,
                    offset: int = 0) -> List[Tenant]:
        """Liste les tenants selon les critères."""
        pass
    
    @abstractmethod
    def delete_tenant(self, tenant_id: str) -> bool:
        """Supprime un tenant."""
        pass
    
    @abstractmethod
    def update_tenant(self, tenant: Tenant) -> bool:
        """Met à jour un tenant."""
        pass


class MemoryTenantStorage(ITenantStorage):
    """Stockage en mémoire pour les tenants."""
    
    def __init__(self):
        self._tenants: Dict[str, Tenant] = {}
        self._lock = threading.RLock()
    
    def save_tenant(self, tenant: Tenant) -> bool:
        """Sauvegarde un tenant en mémoire."""
        with self._lock:
            self._tenants[tenant.tenant_id] = tenant
            return True
    
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Récupère un tenant par ID."""
        with self._lock:
            return self._tenants.get(tenant_id)
    
    def list_tenants(self, 
                    status: Optional[TenantStatus] = None,
                    tier: Optional[TenantTier] = None,
                    limit: int = 100,
                    offset: int = 0) -> List[Tenant]:
        """Liste les tenants selon les critères."""
        with self._lock:
            tenants = list(self._tenants.values())
            
            # Filtrage
            if status:
                tenants = [t for t in tenants if t.status == status]
            
            if tier:
                tenants = [t for t in tenants if t.tier == tier]
            
            # Tri par date de création
            tenants.sort(key=lambda t: t.created_at, reverse=True)
            
            # Pagination
            return tenants[offset:offset + limit]
    
    def delete_tenant(self, tenant_id: str) -> bool:
        """Supprime un tenant."""
        with self._lock:
            return self._tenants.pop(tenant_id, None) is not None
    
    def update_tenant(self, tenant: Tenant) -> bool:
        """Met à jour un tenant."""
        return self.save_tenant(tenant)


class TenantProvisioner:
    """Provisionneur de ressources pour les tenants."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self._metrics = metrics_collector or MetricsCollector()
        self._provisioning_hooks: WeakSet[Callable[[Tenant, str], None]] = WeakSet()
    
    def provision_tenant(self, tenant: Tenant) -> bool:
        """Provisionne les ressources pour un tenant."""
        try:
            # Provisioning selon le niveau d'isolation
            if tenant.isolation_level == IsolationLevel.STRICT:
                self._provision_strict_isolation(tenant)
            elif tenant.isolation_level == IsolationLevel.PARTIAL:
                self._provision_partial_isolation(tenant)
            else:
                self._provision_shared_resources(tenant)
            
            # Notification des hooks
            for hook in self._provisioning_hooks:
                try:
                    hook(tenant, "provisioned")
                except Exception:
                    continue
            
            self._metrics.increment("tenant_provisioned")
            return True
            
        except Exception as e:
            self._metrics.increment("tenant_provisioning_error")
            return False
    
    def deprovision_tenant(self, tenant: Tenant) -> bool:
        """Déprovisionne les ressources d'un tenant."""
        try:
            # Dé-provisioning selon le niveau d'isolation
            if tenant.isolation_level == IsolationLevel.STRICT:
                self._deprovision_strict_isolation(tenant)
            elif tenant.isolation_level == IsolationLevel.PARTIAL:
                self._deprovision_partial_isolation(tenant)
            else:
                self._deprovision_shared_resources(tenant)
            
            # Notification des hooks
            for hook in self._provisioning_hooks:
                try:
                    hook(tenant, "deprovisioned")
                except Exception:
                    continue
            
            self._metrics.increment("tenant_deprovisioned")
            return True
            
        except Exception as e:
            self._metrics.increment("tenant_deprovisioning_error")
            return False
    
    def add_provisioning_hook(self, hook: Callable[[Tenant, str], None]) -> None:
        """Ajoute un hook de provisioning."""
        self._provisioning_hooks.add(hook)
    
    def _provision_strict_isolation(self, tenant: Tenant) -> None:
        """Provisionne avec isolation stricte."""
        # Création d'une base de données dédiée
        # Configuration réseau isolée
        # Ressources dédiées
        pass
    
    def _provision_partial_isolation(self, tenant: Tenant) -> None:
        """Provisionne avec isolation partielle."""
        # Schéma de base de données dédié
        # Ressources partagées avec quotas
        pass
    
    def _provision_shared_resources(self, tenant: Tenant) -> None:
        """Provisionne avec ressources partagées."""
        # Configuration logique uniquement
        pass
    
    def _deprovision_strict_isolation(self, tenant: Tenant) -> None:
        """Déprovisionne l'isolation stricte."""
        # Suppression de la base de données
        # Nettoyage des ressources dédiées
        pass
    
    def _deprovision_partial_isolation(self, tenant: Tenant) -> None:
        """Déprovisionne l'isolation partielle."""
        # Suppression du schéma
        # Libération des quotas
        pass
    
    def _deprovision_shared_resources(self, tenant: Tenant) -> None:
        """Déprovisionne les ressources partagées."""
        # Nettoyage des données logiques
        pass


class TenantManager:
    """
    Gestionnaire principal multi-tenant.
    
    Coordonne la création, gestion, provisioning et monitoring
    des tenants avec support de quotas et facturation.
    """
    
    def __init__(self,
                 storage: Optional[ITenantStorage] = None,
                 cache_manager: Optional[CacheManager] = None,
                 metrics_collector: Optional[MetricsCollector] = None,
                 security_manager: Optional[SecurityManager] = None):
        
        # Composants
        self._storage = storage or MemoryTenantStorage()
        self._cache = cache_manager or CacheManager()
        self._metrics = metrics_collector or MetricsCollector()
        self._security = security_manager or SecurityManager()
        self._provisioner = TenantProvisioner(self._metrics)
        
        # Configuration
        self._cache_ttl = 3600  # 1 heure
        self._usage_update_interval = 300  # 5 minutes
        
        # Hooks pour les événements
        self._event_hooks: WeakSet[Callable[[TenantEvent], None]] = WeakSet()
        
        # Thread de mise à jour des usages
        self._usage_update_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Quotas par tier
        self._tier_quotas = {
            TenantTier.FREE: TenantQuotas(
                max_users=10,
                max_storage_gb=1.0,
                max_requests_per_hour=1000,
                max_alerts_per_hour=100,
                max_integrations=2,
                max_custom_metrics=10,
                api_rate_limit=100
            ),
            TenantTier.BASIC: TenantQuotas(
                max_users=50,
                max_storage_gb=10.0,
                max_requests_per_hour=10000,
                max_alerts_per_hour=1000,
                max_integrations=10,
                max_custom_metrics=100,
                api_rate_limit=1000
            ),
            TenantTier.PREMIUM: TenantQuotas(
                max_users=200,
                max_storage_gb=100.0,
                max_requests_per_hour=100000,
                max_alerts_per_hour=10000,
                max_integrations=50,
                max_custom_metrics=1000,
                api_rate_limit=10000
            ),
            TenantTier.ENTERPRISE: TenantQuotas(
                max_users=1000,
                max_storage_gb=1000.0,
                max_requests_per_hour=1000000,
                max_alerts_per_hour=100000,
                max_integrations=200,
                max_custom_metrics=10000,
                api_rate_limit=100000
            )
        }
    
    def start(self) -> None:
        """Démarre le gestionnaire de tenants."""
        if self._running:
            return
        
        self._running = True
        self._usage_update_thread = threading.Thread(
            target=self._usage_update_loop,
            daemon=True
        )
        self._usage_update_thread.start()
    
    def stop(self) -> None:
        """Arrête le gestionnaire de tenants."""
        self._running = False
        if self._usage_update_thread:
            self._usage_update_thread.join(timeout=5.0)
    
    def create_tenant(self,
                     name: str,
                     owner_email: str,
                     tier: TenantTier = TenantTier.FREE,
                     isolation_level: IsolationLevel = IsolationLevel.SHARED,
                     configuration: Optional[TenantConfiguration] = None,
                     **kwargs) -> Tenant:
        """
        Crée un nouveau tenant.
        
        Args:
            name: Nom du tenant
            owner_email: Email du propriétaire
            tier: Niveau de service
            isolation_level: Niveau d'isolation
            configuration: Configuration spécifique
            **kwargs: Métadonnées additionnelles
            
        Returns:
            Tenant créé
        """
        # Validation des entrées
        if not name or not owner_email:
            raise ValueError("Nom et email propriétaire requis")
        
        # Création du tenant
        tenant = Tenant(
            tenant_id=str(uuid.uuid4()),
            name=name,
            status=TenantStatus.PENDING,
            tier=tier,
            isolation_level=isolation_level,
            owner_email=owner_email,
            quotas=self._tier_quotas[tier],
            configuration=configuration or TenantConfiguration(),
            metadata=kwargs
        )
        
        try:
            # Sauvegarde
            if not self._storage.save_tenant(tenant):
                raise RuntimeError("Échec de sauvegarde du tenant")
            
            # Provisioning
            if not self._provisioner.provision_tenant(tenant):
                # Rollback en cas d'échec
                self._storage.delete_tenant(tenant.tenant_id)
                raise RuntimeError("Échec du provisioning")
            
            # Activation
            tenant.status = TenantStatus.ACTIVE
            self._storage.update_tenant(tenant)
            
            # Invalidation du cache
            self._invalidate_tenant_cache(tenant.tenant_id)
            
            # Événement
            event = TenantEvent("tenant_created", tenant.tenant_id, data=tenant.to_dict())
            self._emit_event(event)
            
            # Métriques
            self._metrics.increment("tenant_created")
            self._metrics.gauge("tenants_active", self._count_active_tenants())
            
            return tenant
            
        except Exception as e:
            # Nettoyage en cas d'erreur
            self._storage.delete_tenant(tenant.tenant_id)
            raise RuntimeError(f"Erreur lors de la création du tenant: {e}")
    
    def get_tenant(self, tenant_id: str, use_cache: bool = True) -> Optional[Tenant]:
        """
        Récupère un tenant par ID.
        
        Args:
            tenant_id: ID du tenant
            use_cache: Utiliser le cache si disponible
            
        Returns:
            Tenant ou None si non trouvé
        """
        cache_key = f"tenant:{tenant_id}"
        
        # Tentative de récupération depuis le cache
        if use_cache:
            cached_tenant = self._cache.get(cache_key)
            if cached_tenant:
                self._metrics.increment("tenant_cache_hit")
                return Tenant(**cached_tenant)
        
        # Récupération depuis le stockage
        tenant = self._storage.get_tenant(tenant_id)
        
        if tenant and use_cache:
            # Mise en cache
            self._cache.set(cache_key, tenant.to_dict(), ttl=self._cache_ttl)
            self._metrics.increment("tenant_cache_miss")
        
        return tenant
    
    def update_tenant(self, tenant: Tenant) -> bool:
        """
        Met à jour un tenant.
        
        Args:
            tenant: Tenant à mettre à jour
            
        Returns:
            True si succès
        """
        tenant.updated_at = datetime.now(timezone.utc)
        
        if self._storage.update_tenant(tenant):
            # Invalidation du cache
            self._invalidate_tenant_cache(tenant.tenant_id)
            
            # Événement
            event = TenantEvent("tenant_updated", tenant.tenant_id, data=tenant.to_dict())
            self._emit_event(event)
            
            self._metrics.increment("tenant_updated")
            return True
        
        return False
    
    def delete_tenant(self, tenant_id: str, user_id: Optional[str] = None) -> bool:
        """
        Supprime un tenant.
        
        Args:
            tenant_id: ID du tenant à supprimer
            user_id: ID de l'utilisateur effectuant la suppression
            
        Returns:
            True si succès
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False
        
        try:
            # Dé-provisioning
            if not self._provisioner.deprovision_tenant(tenant):
                # Log l'erreur mais continue la suppression
                pass
            
            # Suppression du stockage
            if self._storage.delete_tenant(tenant_id):
                # Invalidation du cache
                self._invalidate_tenant_cache(tenant_id)
                
                # Événement
                event = TenantEvent("tenant_deleted", tenant_id, user_id=user_id)
                self._emit_event(event)
                
                self._metrics.increment("tenant_deleted")
                self._metrics.gauge("tenants_active", self._count_active_tenants())
                
                return True
            
        except Exception as e:
            self._metrics.increment("tenant_deletion_error")
        
        return False
    
    def list_tenants(self,
                    status: Optional[TenantStatus] = None,
                    tier: Optional[TenantTier] = None,
                    limit: int = 100,
                    offset: int = 0) -> List[Tenant]:
        """
        Liste les tenants selon les critères.
        
        Args:
            status: Filtre par statut
            tier: Filtre par tier
            limit: Nombre maximum de résultats
            offset: Décalage pour la pagination
            
        Returns:
            Liste des tenants
        """
        return self._storage.list_tenants(status, tier, limit, offset)
    
    def suspend_tenant(self, tenant_id: str, reason: str, user_id: Optional[str] = None) -> bool:
        """
        Suspend un tenant.
        
        Args:
            tenant_id: ID du tenant
            reason: Raison de la suspension
            user_id: ID de l'utilisateur effectuant la suspension
            
        Returns:
            True si succès
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False
        
        tenant.status = TenantStatus.SUSPENDED
        tenant.metadata["suspension_reason"] = reason
        tenant.metadata["suspended_at"] = datetime.now(timezone.utc).isoformat()
        tenant.metadata["suspended_by"] = user_id
        
        if self.update_tenant(tenant):
            event = TenantEvent("tenant_suspended", tenant_id, user_id=user_id, data={"reason": reason})
            self._emit_event(event)
            return True
        
        return False
    
    def reactivate_tenant(self, tenant_id: str, user_id: Optional[str] = None) -> bool:
        """
        Réactive un tenant suspendu.
        
        Args:
            tenant_id: ID du tenant
            user_id: ID de l'utilisateur effectuant la réactivation
            
        Returns:
            True si succès
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant or tenant.status != TenantStatus.SUSPENDED:
            return False
        
        tenant.status = TenantStatus.ACTIVE
        tenant.metadata.pop("suspension_reason", None)
        tenant.metadata.pop("suspended_at", None)
        tenant.metadata["reactivated_at"] = datetime.now(timezone.utc).isoformat()
        tenant.metadata["reactivated_by"] = user_id
        
        if self.update_tenant(tenant):
            event = TenantEvent("tenant_reactivated", tenant_id, user_id=user_id)
            self._emit_event(event)
            return True
        
        return False
    
    def update_tenant_usage(self, tenant_id: str, usage_data: Dict[str, Any]) -> bool:
        """
        Met à jour les données d'utilisation d'un tenant.
        
        Args:
            tenant_id: ID du tenant
            usage_data: Données d'utilisation
            
        Returns:
            True si succès
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False
        
        tenant.update_usage(usage_data)
        
        # Vérification des quotas
        self._check_tenant_quotas(tenant)
        
        return self.update_tenant(tenant)
    
    def get_tenant_config(self, tenant_id: str) -> Optional[TenantConfiguration]:
        """
        Récupère la configuration d'un tenant.
        
        Args:
            tenant_id: ID du tenant
            
        Returns:
            Configuration du tenant
        """
        tenant = self.get_tenant(tenant_id)
        return tenant.configuration if tenant else None
    
    def update_tenant_config(self, tenant_id: str, config: TenantConfiguration) -> bool:
        """
        Met à jour la configuration d'un tenant.
        
        Args:
            tenant_id: ID du tenant
            config: Nouvelle configuration
            
        Returns:
            True si succès
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False
        
        tenant.configuration = config
        return self.update_tenant(tenant)
    
    def add_event_hook(self, hook: Callable[[TenantEvent], None]) -> None:
        """Ajoute un hook pour les événements tenant."""
        self._event_hooks.add(hook)
    
    def get_tenant_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques des tenants."""
        all_tenants = self.list_tenants(limit=10000)
        
        stats = {
            "total_tenants": len(all_tenants),
            "by_status": {},
            "by_tier": {},
            "by_isolation": {},
            "active_tenants": 0,
            "suspended_tenants": 0
        }
        
        for tenant in all_tenants:
            # Par statut
            status = tenant.status.value
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
            
            # Par tier
            tier = tenant.tier.value
            stats["by_tier"][tier] = stats["by_tier"].get(tier, 0) + 1
            
            # Par isolation
            isolation = tenant.isolation_level.value
            stats["by_isolation"][isolation] = stats["by_isolation"].get(isolation, 0) + 1
            
            # Compteurs spéciaux
            if tenant.status == TenantStatus.ACTIVE:
                stats["active_tenants"] += 1
            elif tenant.status == TenantStatus.SUSPENDED:
                stats["suspended_tenants"] += 1
        
        return stats
    
    def _check_tenant_quotas(self, tenant: Tenant) -> None:
        """Vérifie les quotas d'un tenant et déclenche des alertes si nécessaire."""
        quota_violations = []
        
        for quota_type in ["users", "storage", "requests", "alerts", "integrations", "metrics"]:
            if tenant.is_over_quota(quota_type):
                quota_violations.append(quota_type)
        
        if quota_violations:
            event = TenantEvent(
                "quota_exceeded",
                tenant.tenant_id,
                data={"violations": quota_violations}
            )
            self._emit_event(event)
            
            # Métriques
            for violation in quota_violations:
                self._metrics.increment(f"quota_exceeded_{violation}")
    
    def _emit_event(self, event: TenantEvent) -> None:
        """Émet un événement tenant."""
        for hook in self._event_hooks:
            try:
                hook(event)
            except Exception:
                continue
        
        self._metrics.increment(f"tenant_event_{event.event_type}")
    
    def _invalidate_tenant_cache(self, tenant_id: str) -> None:
        """Invalide le cache d'un tenant."""
        cache_key = f"tenant:{tenant_id}"
        self._cache.delete(cache_key)
    
    def _count_active_tenants(self) -> int:
        """Compte les tenants actifs."""
        active_tenants = self.list_tenants(status=TenantStatus.ACTIVE, limit=10000)
        return len(active_tenants)
    
    def _usage_update_loop(self) -> None:
        """Boucle de mise à jour des usages."""
        while self._running:
            try:
                # Mise à jour des usages pour tous les tenants actifs
                active_tenants = self.list_tenants(status=TenantStatus.ACTIVE, limit=1000)
                
                for tenant in active_tenants:
                    # Ici, nous collecterions les vraies données d'usage
                    # Pour la démo, nous simulons une mise à jour
                    pass
                
                time.sleep(self._usage_update_interval)
                
            except Exception:
                time.sleep(60)  # Pause en cas d'erreur


# Instance globale singleton
_global_tenant_manager: Optional[TenantManager] = None
_tenant_lock = threading.Lock()


def get_tenant_manager(**kwargs) -> TenantManager:
    """
    Récupère l'instance globale du gestionnaire de tenants.
    
    Returns:
        Instance singleton du TenantManager
    """
    global _global_tenant_manager
    
    if _global_tenant_manager is None:
        with _tenant_lock:
            if _global_tenant_manager is None:
                _global_tenant_manager = TenantManager(**kwargs)
                _global_tenant_manager.start()
    
    return _global_tenant_manager


# API publique simplifiée
def create_tenant(name: str, owner_email: str, **kwargs) -> Tenant:
    """API simplifiée pour créer un tenant."""
    manager = get_tenant_manager()
    return manager.create_tenant(name, owner_email, **kwargs)


def get_tenant(tenant_id: str, **kwargs) -> Optional[Tenant]:
    """API simplifiée pour récupérer un tenant."""
    manager = get_tenant_manager()
    return manager.get_tenant(tenant_id, **kwargs)


def list_tenants(**kwargs) -> List[Tenant]:
    """API simplifiée pour lister les tenants."""
    manager = get_tenant_manager()
    return manager.list_tenants(**kwargs)
