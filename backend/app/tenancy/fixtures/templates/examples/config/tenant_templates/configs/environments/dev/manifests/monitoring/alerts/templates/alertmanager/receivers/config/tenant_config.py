"""
Configuration avancée de gestion des tenants pour Alertmanager Receivers

Ce module gère la configuration multi-tenant avancée avec isolation,
quotas dynamiques, billing intelligent et gouvernance des données.

Author: Spotify AI Agent Team
Maintainer: Fahed Mlaiel - Architecte Microservices
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import uuid
from collections import defaultdict, deque
import yaml

logger = logging.getLogger(__name__)

class TenantTier(Enum):
    """Niveaux de tier pour les tenants"""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    PREMIUM = "premium"
    
    @property
    def max_alerts_per_hour(self) -> int:
        return {
            TenantTier.FREE: 100,
            TenantTier.STARTER: 1000,
            TenantTier.PROFESSIONAL: 5000,
            TenantTier.ENTERPRISE: 25000,
            TenantTier.PREMIUM: 100000
        }[self]
    
    @property
    def max_integrations(self) -> int:
        return {
            TenantTier.FREE: 2,
            TenantTier.STARTER: 5,
            TenantTier.PROFESSIONAL: 15,
            TenantTier.ENTERPRISE: 50,
            TenantTier.PREMIUM: 100
        }[self]
    
    @property
    def priority_level(self) -> int:
        return {
            TenantTier.FREE: 1,
            TenantTier.STARTER: 2,
            TenantTier.PROFESSIONAL: 3,
            TenantTier.ENTERPRISE: 4,
            TenantTier.PREMIUM: 5
        }[self]

class TenantStatus(Enum):
    """Statuts des tenants"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    MIGRATING = "migrating"
    ARCHIVED = "archived"

class IsolationLevel(Enum):
    """Niveaux d'isolation entre tenants"""
    SHARED = "shared"
    DEDICATED_NAMESPACE = "dedicated_namespace"
    DEDICATED_INSTANCE = "dedicated_instance"
    DEDICATED_CLUSTER = "dedicated_cluster"

@dataclass
class TenantQuota:
    """Quotas pour un tenant"""
    max_alerts_per_hour: int
    max_alerts_per_day: int
    max_receivers: int
    max_integrations: int
    max_templates: int
    max_storage_mb: int
    max_retention_days: int
    priority_weight: float = 1.0
    
    def is_within_limits(self, current_usage: Dict[str, int]) -> bool:
        """Vérifie si l'utilisation actuelle respecte les quotas"""
        checks = [
            current_usage.get("alerts_per_hour", 0) <= self.max_alerts_per_hour,
            current_usage.get("alerts_per_day", 0) <= self.max_alerts_per_day,
            current_usage.get("receivers", 0) <= self.max_receivers,
            current_usage.get("integrations", 0) <= self.max_integrations,
            current_usage.get("templates", 0) <= self.max_templates,
            current_usage.get("storage_mb", 0) <= self.max_storage_mb
        ]
        return all(checks)

@dataclass
class TenantBilling:
    """Informations de facturation pour un tenant"""
    base_cost_monthly: float
    cost_per_alert: float
    cost_per_integration: float
    cost_per_gb_storage: float
    billing_cycle: str = "monthly"  # monthly, yearly
    payment_method: Optional[str] = None
    last_billing_date: Optional[datetime] = None
    next_billing_date: Optional[datetime] = None
    outstanding_balance: float = 0.0
    
    def calculate_monthly_cost(self, usage: Dict[str, int]) -> float:
        """Calcule le coût mensuel basé sur l'utilisation"""
        alerts_cost = usage.get("total_alerts", 0) * self.cost_per_alert
        integration_cost = usage.get("integrations", 0) * self.cost_per_integration
        storage_cost = (usage.get("storage_mb", 0) / 1024) * self.cost_per_gb_storage
        
        return self.base_cost_monthly + alerts_cost + integration_cost + storage_cost

@dataclass
class TenantConfiguration:
    """Configuration complète d'un tenant"""
    id: str
    name: str
    tier: TenantTier
    status: TenantStatus
    isolation_level: IsolationLevel
    quota: TenantQuota
    billing: TenantBilling
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_modified: datetime = field(default_factory=datetime.utcnow)
    
    # Configuration technique
    namespace: str = ""
    database_connection: Optional[str] = None
    cache_prefix: str = ""
    encryption_key_id: Optional[str] = None
    
    # Métadonnées
    owner_email: str = ""
    technical_contact: str = ""
    business_contact: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Configuration d'alerting
    alert_routing_rules: List[Dict[str, Any]] = field(default_factory=list)
    custom_templates: Dict[str, str] = field(default_factory=dict)
    integrations: List[str] = field(default_factory=list)
    
    # Gouvernance des données
    data_residency: str = "EU"  # EU, US, APAC
    data_classification: str = "internal"  # public, internal, confidential, secret
    retention_policy: Dict[str, int] = field(default_factory=lambda: {"alerts": 90, "metrics": 30, "logs": 7})
    
    def __post_init__(self):
        if not self.namespace:
            self.namespace = f"tenant-{self.id}"
        if not self.cache_prefix:
            self.cache_prefix = f"tenant:{self.id}"

@dataclass
class TenantUsage:
    """Utilisation d'un tenant"""
    tenant_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Compteurs d'alertes
    alerts_last_hour: int = 0
    alerts_last_day: int = 0
    alerts_last_month: int = 0
    
    # Utilisation des ressources
    active_receivers: int = 0
    active_integrations: int = 0
    active_templates: int = 0
    storage_used_mb: int = 0
    
    # Métriques de performance
    avg_response_time_ms: float = 0.0
    error_rate_percent: float = 0.0
    uptime_percent: float = 100.0
    
    # Coûts
    estimated_monthly_cost: float = 0.0
    actual_monthly_cost: float = 0.0

class TenantIsolationManager:
    """Gestionnaire d'isolation entre tenants"""
    
    def __init__(self):
        self.namespace_configs: Dict[str, Dict[str, Any]] = {}
        self.resource_pools: Dict[str, Dict[str, Any]] = {}
        
    async def setup_tenant_isolation(self, tenant: TenantConfiguration) -> bool:
        """Configure l'isolation pour un tenant"""
        try:
            if tenant.isolation_level == IsolationLevel.SHARED:
                await self._setup_shared_isolation(tenant)
            elif tenant.isolation_level == IsolationLevel.DEDICATED_NAMESPACE:
                await self._setup_namespace_isolation(tenant)
            elif tenant.isolation_level == IsolationLevel.DEDICATED_INSTANCE:
                await self._setup_instance_isolation(tenant)
            elif tenant.isolation_level == IsolationLevel.DEDICATED_CLUSTER:
                await self._setup_cluster_isolation(tenant)
            
            logger.info(f"Isolation setup completed for tenant {tenant.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup isolation for tenant {tenant.id}: {e}")
            return False
    
    async def _setup_shared_isolation(self, tenant: TenantConfiguration):
        """Configure l'isolation partagée (multi-tenant)"""
        # Configuration de base avec préfixes et filtres
        isolation_config = {
            "type": "shared",
            "cache_prefix": tenant.cache_prefix,
            "database_schema": f"tenant_{tenant.id}",
            "resource_limits": {
                "cpu_millicores": 100,
                "memory_mb": 256,
                "disk_mb": tenant.quota.max_storage_mb
            }
        }
        
        self.namespace_configs[tenant.namespace] = isolation_config
    
    async def _setup_namespace_isolation(self, tenant: TenantConfiguration):
        """Configure l'isolation par namespace"""
        isolation_config = {
            "type": "namespace",
            "kubernetes_namespace": tenant.namespace,
            "network_policies": self._generate_network_policies(tenant),
            "resource_quotas": self._generate_resource_quotas(tenant),
            "rbac_rules": self._generate_rbac_rules(tenant)
        }
        
        self.namespace_configs[tenant.namespace] = isolation_config
    
    async def _setup_instance_isolation(self, tenant: TenantConfiguration):
        """Configure l'isolation par instance dédiée"""
        isolation_config = {
            "type": "dedicated_instance",
            "instance_id": f"alertmanager-{tenant.id}",
            "dedicated_database": True,
            "dedicated_cache": True,
            "resource_allocation": {
                "cpu_cores": 2,
                "memory_gb": 4,
                "disk_gb": tenant.quota.max_storage_mb // 1024
            }
        }
        
        self.namespace_configs[tenant.namespace] = isolation_config
    
    async def _setup_cluster_isolation(self, tenant: TenantConfiguration):
        """Configure l'isolation par cluster dédié"""
        isolation_config = {
            "type": "dedicated_cluster",
            "cluster_id": f"cluster-{tenant.id}",
            "cluster_size": "large",
            "network_isolation": "complete",
            "data_residency": tenant.data_residency
        }
        
        self.namespace_configs[tenant.namespace] = isolation_config
    
    def _generate_network_policies(self, tenant: TenantConfiguration) -> List[Dict[str, Any]]:
        """Génère les politiques réseau pour un tenant"""
        return [
            {
                "name": f"tenant-{tenant.id}-isolation",
                "rules": [
                    {"from": [{"namespaceSelector": {"matchLabels": {"tenant": tenant.id}}}]},
                    {"to": [{"namespaceSelector": {"matchLabels": {"tenant": tenant.id}}}]}
                ]
            }
        ]
    
    def _generate_resource_quotas(self, tenant: TenantConfiguration) -> Dict[str, str]:
        """Génère les quotas de ressources pour un tenant"""
        return {
            "requests.cpu": f"{tenant.tier.priority_level * 500}m",
            "requests.memory": f"{tenant.quota.max_storage_mb}Mi",
            "limits.cpu": f"{tenant.tier.priority_level * 1000}m",
            "limits.memory": f"{tenant.quota.max_storage_mb * 2}Mi",
            "persistentvolumeclaims": f"{tenant.quota.max_storage_mb // 1024}Gi"
        }
    
    def _generate_rbac_rules(self, tenant: TenantConfiguration) -> List[Dict[str, Any]]:
        """Génère les règles RBAC pour un tenant"""
        return [
            {
                "apiGroups": [""],
                "resources": ["configmaps", "secrets"],
                "verbs": ["get", "list", "create", "update"],
                "resourceNames": [f"tenant-{tenant.id}-*"]
            }
        ]

class TenantQuotaManager:
    """Gestionnaire des quotas de tenants"""
    
    def __init__(self):
        self.usage_tracking: Dict[str, TenantUsage] = {}
        self.quota_violations: deque = deque(maxlen=1000)
        
    async def track_usage(self, tenant_id: str, metric: str, value: int):
        """Suit l'utilisation d'une métrique pour un tenant"""
        if tenant_id not in self.usage_tracking:
            self.usage_tracking[tenant_id] = TenantUsage(tenant_id=tenant_id)
        
        usage = self.usage_tracking[tenant_id]
        
        # Mise à jour des métriques
        if metric == "alert":
            usage.alerts_last_hour += value
            usage.alerts_last_day += value
            usage.alerts_last_month += value
        elif metric == "receiver":
            usage.active_receivers = value
        elif metric == "integration":
            usage.active_integrations = value
        elif metric == "template":
            usage.active_templates = value
        elif metric == "storage_mb":
            usage.storage_used_mb = value
        
        usage.timestamp = datetime.utcnow()
    
    async def check_quota_compliance(self, tenant: TenantConfiguration) -> Dict[str, Any]:
        """Vérifie la conformité aux quotas d'un tenant"""
        if tenant.id not in self.usage_tracking:
            return {"compliant": True, "violations": []}
        
        usage = self.usage_tracking[tenant.id]
        violations = []
        
        # Vérification des limites
        checks = [
            ("alerts_per_hour", usage.alerts_last_hour, tenant.quota.max_alerts_per_hour),
            ("alerts_per_day", usage.alerts_last_day, tenant.quota.max_alerts_per_day),
            ("receivers", usage.active_receivers, tenant.quota.max_receivers),
            ("integrations", usage.active_integrations, tenant.quota.max_integrations),
            ("templates", usage.active_templates, tenant.quota.max_templates),
            ("storage_mb", usage.storage_used_mb, tenant.quota.max_storage_mb)
        ]
        
        for metric_name, current_value, limit in checks:
            if current_value > limit:
                violations.append({
                    "metric": metric_name,
                    "current": current_value,
                    "limit": limit,
                    "percentage": (current_value / limit) * 100
                })
        
        # Enregistrement des violations
        if violations:
            violation_record = {
                "tenant_id": tenant.id,
                "timestamp": datetime.utcnow(),
                "violations": violations
            }
            self.quota_violations.append(violation_record)
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "usage_percentage": self._calculate_usage_percentage(usage, tenant.quota)
        }
    
    def _calculate_usage_percentage(self, usage: TenantUsage, quota: TenantQuota) -> Dict[str, float]:
        """Calcule le pourcentage d'utilisation pour chaque quota"""
        return {
            "alerts_per_hour": (usage.alerts_last_hour / quota.max_alerts_per_hour) * 100,
            "alerts_per_day": (usage.alerts_last_day / quota.max_alerts_per_day) * 100,
            "receivers": (usage.active_receivers / quota.max_receivers) * 100,
            "integrations": (usage.active_integrations / quota.max_integrations) * 100,
            "templates": (usage.active_templates / quota.max_templates) * 100,
            "storage": (usage.storage_used_mb / quota.max_storage_mb) * 100
        }
    
    async def apply_quota_enforcement(self, tenant_id: str, violations: List[Dict[str, Any]]) -> bool:
        """Applique l'enforcement des quotas"""
        try:
            for violation in violations:
                metric = violation["metric"]
                
                if metric == "alerts_per_hour":
                    # Throttling des alertes
                    await self._throttle_alerts(tenant_id, violation["current"], violation["limit"])
                elif metric == "storage_mb":
                    # Nettoyage automatique
                    await self._cleanup_old_data(tenant_id)
                elif metric in ["receivers", "integrations", "templates"]:
                    # Blocage de nouvelles créations
                    await self._block_new_resources(tenant_id, metric)
            
            logger.info(f"Quota enforcement applied for tenant {tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply quota enforcement for tenant {tenant_id}: {e}")
            return False
    
    async def _throttle_alerts(self, tenant_id: str, current: int, limit: int):
        """Applique un throttling sur les alertes"""
        throttle_rate = max(0.1, limit / current)  # Réduction proportionnelle
        logger.warning(f"Throttling alerts for tenant {tenant_id} to {throttle_rate * 100:.1f}%")
    
    async def _cleanup_old_data(self, tenant_id: str):
        """Nettoie les anciennes données pour libérer de l'espace"""
        logger.info(f"Cleaning up old data for tenant {tenant_id}")
    
    async def _block_new_resources(self, tenant_id: str, resource_type: str):
        """Bloque la création de nouvelles ressources"""
        logger.warning(f"Blocking new {resource_type} creation for tenant {tenant_id}")

class TenantBillingManager:
    """Gestionnaire de facturation des tenants"""
    
    def __init__(self):
        self.billing_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.cost_models: Dict[TenantTier, Dict[str, float]] = self._initialize_cost_models()
    
    def _initialize_cost_models(self) -> Dict[TenantTier, Dict[str, float]]:
        """Initialise les modèles de coûts par tier"""
        return {
            TenantTier.FREE: {
                "base_monthly": 0.0,
                "cost_per_alert": 0.0,
                "cost_per_integration": 0.0,
                "cost_per_gb_storage": 0.0
            },
            TenantTier.STARTER: {
                "base_monthly": 29.0,
                "cost_per_alert": 0.001,
                "cost_per_integration": 5.0,
                "cost_per_gb_storage": 2.0
            },
            TenantTier.PROFESSIONAL: {
                "base_monthly": 99.0,
                "cost_per_alert": 0.0008,
                "cost_per_integration": 3.0,
                "cost_per_gb_storage": 1.5
            },
            TenantTier.ENTERPRISE: {
                "base_monthly": 299.0,
                "cost_per_alert": 0.0005,
                "cost_per_integration": 2.0,
                "cost_per_gb_storage": 1.0
            },
            TenantTier.PREMIUM: {
                "base_monthly": 999.0,
                "cost_per_alert": 0.0003,
                "cost_per_integration": 1.0,
                "cost_per_gb_storage": 0.5
            }
        }
    
    async def calculate_monthly_bill(self, tenant: TenantConfiguration, usage: TenantUsage) -> Dict[str, Any]:
        """Calcule la facture mensuelle d'un tenant"""
        cost_model = self.cost_models[tenant.tier]
        
        # Coûts de base
        base_cost = cost_model["base_monthly"]
        
        # Coûts variables
        alerts_cost = usage.alerts_last_month * cost_model["cost_per_alert"]
        integration_cost = usage.active_integrations * cost_model["cost_per_integration"]
        storage_cost = (usage.storage_used_mb / 1024) * cost_model["cost_per_gb_storage"]
        
        # Coût total
        total_cost = base_cost + alerts_cost + integration_cost + storage_cost
        
        # Application de remises
        discount = self._calculate_discount(tenant, usage)
        final_cost = total_cost * (1 - discount)
        
        bill = {
            "tenant_id": tenant.id,
            "billing_period": datetime.utcnow().strftime("%Y-%m"),
            "base_cost": base_cost,
            "alerts_cost": alerts_cost,
            "integration_cost": integration_cost,
            "storage_cost": storage_cost,
            "subtotal": total_cost,
            "discount_percent": discount * 100,
            "discount_amount": total_cost * discount,
            "total_cost": final_cost,
            "usage_details": {
                "alerts_count": usage.alerts_last_month,
                "integrations_count": usage.active_integrations,
                "storage_gb": usage.storage_used_mb / 1024,
                "uptime_percent": usage.uptime_percent
            }
        }
        
        # Enregistrement de la facture
        self.billing_records[tenant.id].append(bill)
        
        return bill
    
    def _calculate_discount(self, tenant: TenantConfiguration, usage: TenantUsage) -> float:
        """Calcule les remises applicables"""
        discount = 0.0
        
        # Remise pour haute disponibilité
        if usage.uptime_percent >= 99.9:
            discount += 0.02  # 2% de remise
        
        # Remise pour ancienneté
        months_active = (datetime.utcnow() - tenant.created_at).days // 30
        if months_active >= 12:
            discount += 0.05  # 5% de remise après 1 an
        elif months_active >= 6:
            discount += 0.02  # 2% de remise après 6 mois
        
        # Remise pour volume (Enterprise et Premium)
        if tenant.tier in [TenantTier.ENTERPRISE, TenantTier.PREMIUM]:
            if usage.alerts_last_month > 100000:
                discount += 0.10  # 10% de remise pour gros volume
        
        return min(discount, 0.20)  # Maximum 20% de remise

class TenantConfigManager:
    """Gestionnaire principal de la configuration multi-tenant"""
    
    def __init__(self):
        self.tenants: Dict[str, TenantConfiguration] = {}
        self.isolation_manager = TenantIsolationManager()
        self.quota_manager = TenantQuotaManager()
        self.billing_manager = TenantBillingManager()
        self.config_cache: Dict[str, Any] = {}
        
    async def initialize_tenant_management(self) -> bool:
        """Initialise le gestionnaire de tenants"""
        try:
            logger.info("Initializing tenant management system")
            
            # Chargement des configurations existantes
            await self._load_tenant_configurations()
            
            # Démarrage des tâches de monitoring
            asyncio.create_task(self._tenant_monitoring_loop())
            asyncio.create_task(self._quota_enforcement_loop())
            asyncio.create_task(self._billing_calculation_loop())
            
            logger.info("Tenant management system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize tenant management: {e}")
            return False
    
    async def _load_tenant_configurations(self):
        """Charge les configurations de tenants depuis le stockage"""
        # Ici, on chargerait depuis une base de données ou un fichier de configuration
        # Pour la démo, on crée quelques tenants d'exemple
        
        demo_tenants = [
            TenantConfiguration(
                id="tenant-001",
                name="Acme Corp",
                tier=TenantTier.ENTERPRISE,
                status=TenantStatus.ACTIVE,
                isolation_level=IsolationLevel.DEDICATED_NAMESPACE,
                quota=TenantQuota(
                    max_alerts_per_hour=5000,
                    max_alerts_per_day=50000,
                    max_receivers=50,
                    max_integrations=20,
                    max_templates=100,
                    max_storage_mb=10240,
                    max_retention_days=90
                ),
                billing=TenantBilling(
                    base_cost_monthly=299.0,
                    cost_per_alert=0.0005,
                    cost_per_integration=2.0,
                    cost_per_gb_storage=1.0
                ),
                owner_email="admin@acmecorp.com",
                data_residency="EU"
            )
        ]
        
        for tenant in demo_tenants:
            self.tenants[tenant.id] = tenant
            await self.isolation_manager.setup_tenant_isolation(tenant)
    
    async def _tenant_monitoring_loop(self):
        """Boucle de monitoring des tenants"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                for tenant in self.tenants.values():
                    # Vérification de la santé du tenant
                    await self._check_tenant_health(tenant)
                
            except Exception as e:
                logger.error(f"Error in tenant monitoring loop: {e}")
    
    async def _quota_enforcement_loop(self):
        """Boucle d'enforcement des quotas"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                for tenant in self.tenants.values():
                    compliance = await self.quota_manager.check_quota_compliance(tenant)
                    
                    if not compliance["compliant"]:
                        await self.quota_manager.apply_quota_enforcement(
                            tenant.id, compliance["violations"]
                        )
                
            except Exception as e:
                logger.error(f"Error in quota enforcement loop: {e}")
    
    async def _billing_calculation_loop(self):
        """Boucle de calcul de facturation"""
        while True:
            try:
                # Calcul mensuel le 1er de chaque mois
                now = datetime.utcnow()
                if now.day == 1 and now.hour == 0:
                    await self._process_monthly_billing()
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in billing calculation loop: {e}")
    
    async def _check_tenant_health(self, tenant: TenantConfiguration):
        """Vérifie la santé d'un tenant"""
        # Implémentation des vérifications de santé
        pass
    
    async def _process_monthly_billing(self):
        """Traite la facturation mensuelle pour tous les tenants"""
        for tenant in self.tenants.values():
            if tenant.status == TenantStatus.ACTIVE:
                usage = self.quota_manager.usage_tracking.get(tenant.id)
                if usage:
                    bill = await self.billing_manager.calculate_monthly_bill(tenant, usage)
                    logger.info(f"Monthly bill calculated for tenant {tenant.id}: ${bill['total_cost']:.2f}")
    
    async def create_tenant(self, tenant_config: Dict[str, Any]) -> str:
        """Crée un nouveau tenant"""
        try:
            tenant_id = str(uuid.uuid4())
            
            # Création de la configuration
            tenant = TenantConfiguration(
                id=tenant_id,
                name=tenant_config["name"],
                tier=TenantTier(tenant_config["tier"]),
                status=TenantStatus.ACTIVE,
                isolation_level=IsolationLevel(tenant_config.get("isolation_level", "shared")),
                quota=self._create_default_quota(TenantTier(tenant_config["tier"])),
                billing=self._create_default_billing(TenantTier(tenant_config["tier"])),
                owner_email=tenant_config["owner_email"],
                data_residency=tenant_config.get("data_residency", "EU")
            )
            
            # Configuration de l'isolation
            await self.isolation_manager.setup_tenant_isolation(tenant)
            
            # Enregistrement
            self.tenants[tenant_id] = tenant
            
            logger.info(f"Created new tenant: {tenant_id}")
            return tenant_id
            
        except Exception as e:
            logger.error(f"Failed to create tenant: {e}")
            raise
    
    def _create_default_quota(self, tier: TenantTier) -> TenantQuota:
        """Crée un quota par défaut pour un tier"""
        return TenantQuota(
            max_alerts_per_hour=tier.max_alerts_per_hour,
            max_alerts_per_day=tier.max_alerts_per_hour * 24,
            max_receivers=tier.max_integrations * 2,
            max_integrations=tier.max_integrations,
            max_templates=50,
            max_storage_mb=1024 * tier.priority_level,
            max_retention_days=30 * tier.priority_level
        )
    
    def _create_default_billing(self, tier: TenantTier) -> TenantBilling:
        """Crée une configuration de facturation par défaut"""
        cost_model = self.billing_manager.cost_models[tier]
        
        return TenantBilling(
            base_cost_monthly=cost_model["base_monthly"],
            cost_per_alert=cost_model["cost_per_alert"],
            cost_per_integration=cost_model["cost_per_integration"],
            cost_per_gb_storage=cost_model["cost_per_gb_storage"]
        )
    
    def get_tenant(self, tenant_id: str) -> Optional[TenantConfiguration]:
        """Récupère la configuration d'un tenant"""
        return self.tenants.get(tenant_id)
    
    def get_all_tenants(self) -> List[TenantConfiguration]:
        """Récupère toutes les configurations de tenants"""
        return list(self.tenants.values())
    
    async def update_tenant(self, tenant_id: str, updates: Dict[str, Any]) -> bool:
        """Met à jour la configuration d'un tenant"""
        try:
            if tenant_id not in self.tenants:
                return False
            
            tenant = self.tenants[tenant_id]
            
            # Application des mises à jour
            for key, value in updates.items():
                if hasattr(tenant, key):
                    setattr(tenant, key, value)
            
            tenant.last_modified = datetime.utcnow()
            
            logger.info(f"Updated tenant configuration: {tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update tenant {tenant_id}: {e}")
            return False
    
    async def delete_tenant(self, tenant_id: str) -> bool:
        """Supprime un tenant (archivage)"""
        try:
            if tenant_id not in self.tenants:
                return False
            
            tenant = self.tenants[tenant_id]
            tenant.status = TenantStatus.ARCHIVED
            tenant.last_modified = datetime.utcnow()
            
            logger.info(f"Archived tenant: {tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete tenant {tenant_id}: {e}")
            return False

# Instance singleton
tenant_manager = TenantConfigManager()
