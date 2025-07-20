#!/usr/bin/env python3
"""
Enterprise Tenant Templates Management System
Ultra-Advanced Industrial Multi-Tenant Architecture

Module ultra-avancé pour la gestion des templates de tenants avec IA intégrée,
sécurité enterprise, et orchestration automatisée des ressources.

Architecture:
- Lead Dev + Architecte IA: Architecture distribuée avec ML intégré
- Backend Senior: API FastAPI haute performance avec async/await
- ML Engineer: Recommandations intelligentes et optimisation automatique
- DBA & Data Engineer: Gestion multi-base avec sharding automatique
- Spécialiste Sécurité: Chiffrement bout-en-bout et conformité GDPR
- Architecte Microservices: Pattern Event-Driven avec CQRS
"""

import asyncio
import json
import yaml
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from uuid import uuid4
import logging
import hashlib
import hmac
from cryptography.fernet import Fernet
import redis
import asyncpg
from sqlalchemy import create_engine, MetaData
from prometheus_client import Counter, Histogram, Gauge


# =============================================================================
# ENUMERATIONS & CONSTANTS
# =============================================================================

class TenantTier(Enum):
    """Niveaux de tenants avec capacités différenciées"""
    FREE = "free"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    ENTERPRISE_PLUS = "enterprise_plus"
    WHITE_LABEL = "white_label"


class ResourceType(Enum):
    """Types de ressources managées par le système"""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"
    ML_MODELS = "ml_models"
    AI_SERVICES = "ai_services"


class SecurityLevel(Enum):
    """Niveaux de sécurité enterprise"""
    BASIC = "basic"
    ENHANCED = "enhanced"
    MAXIMUM = "maximum"
    CLASSIFIED = "classified"


class ComplianceFramework(Enum):
    """Frameworks de conformité supportés"""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"
    FedRAMP = "fedramp"


class DeploymentStrategy(Enum):
    """Stratégies de déploiement"""
    BLUE_GREEN = "blue_green"
    ROLLING_UPDATE = "rolling_update"
    CANARY = "canary"
    A_B_TESTING = "a_b_testing"


# =============================================================================
# METRIQUES & MONITORING
# =============================================================================

# Métriques Prometheus
TENANT_CREATION_COUNTER = Counter('tenant_template_creations_total', 'Total tenant template creations', ['tier', 'region'])
RESOURCE_ALLOCATION_HISTOGRAM = Histogram('resource_allocation_seconds', 'Time spent allocating resources', ['resource_type'])
ACTIVE_TENANTS_GAUGE = Gauge('active_tenants_current', 'Current number of active tenants', ['tier'])
TEMPLATE_VALIDATION_COUNTER = Counter('template_validations_total', 'Template validation attempts', ['status'])


# =============================================================================
# DATACLASSES & MODELS
# =============================================================================

@dataclass
class ResourceQuotas:
    """Quotas de ressources pour un tenant"""
    cpu_cores: int = 2
    memory_gb: int = 4
    storage_gb: int = 100
    network_bandwidth_mbps: int = 100
    concurrent_connections: int = 1000
    api_requests_per_minute: int = 1000
    ml_model_instances: int = 1
    ai_processing_units: int = 10
    database_connections: int = 50
    cache_size_mb: int = 512


@dataclass
class SecurityConfiguration:
    """Configuration de sécurité pour un tenant"""
    encryption_level: SecurityLevel = SecurityLevel.ENHANCED
    mfa_required: bool = True
    ip_whitelist_enabled: bool = False
    allowed_ip_ranges: List[str] = field(default_factory=list)
    data_retention_days: int = 365
    audit_logging: bool = True
    vulnerability_scanning: bool = True
    penetration_testing: bool = False
    zero_trust_networking: bool = False
    end_to_end_encryption: bool = True


@dataclass
class ComplianceSettings:
    """Paramètres de conformité réglementaire"""
    frameworks: List[ComplianceFramework] = field(default_factory=list)
    data_localization_required: bool = False
    data_residency_regions: List[str] = field(default_factory=list)
    audit_trail_retention_years: int = 7
    privacy_controls_enabled: bool = True
    consent_management: bool = True
    right_to_be_forgotten: bool = True
    data_portability: bool = True


@dataclass
class AIConfiguration:
    """Configuration des services IA/ML"""
    recommendation_engine_enabled: bool = True
    sentiment_analysis_enabled: bool = False
    nlp_processing_enabled: bool = False
    computer_vision_enabled: bool = False
    auto_ml_enabled: bool = False
    model_training_quota_hours: int = 10
    inference_requests_per_day: int = 10000
    custom_models_allowed: int = 3
    gpu_acceleration: bool = False
    federated_learning: bool = False


@dataclass
class MonitoringConfiguration:
    """Configuration du monitoring et observabilité"""
    metrics_retention_days: int = 30
    logs_retention_days: int = 90
    traces_enabled: bool = True
    custom_dashboards: bool = False
    alerting_channels: List[str] = field(default_factory=lambda: ["email"])
    sla_monitoring: bool = True
    performance_profiling: bool = False
    real_time_monitoring: bool = True


@dataclass
class TenantTemplate:
    """Template complet de configuration de tenant"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    display_name: str = ""
    tier: TenantTier = TenantTier.STANDARD
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Configurations principales
    resource_quotas: ResourceQuotas = field(default_factory=ResourceQuotas)
    security_config: SecurityConfiguration = field(default_factory=SecurityConfiguration)
    compliance_settings: ComplianceSettings = field(default_factory=ComplianceSettings)
    ai_config: AIConfiguration = field(default_factory=AIConfiguration)
    monitoring_config: MonitoringConfiguration = field(default_factory=MonitoringConfiguration)
    
    # Métadonnées
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Configuration avancée
    deployment_strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE
    geographic_regions: List[str] = field(default_factory=lambda: ["us-east-1"])
    multi_region_enabled: bool = False
    disaster_recovery_enabled: bool = False
    backup_strategy: str = "daily"
    
    def __post_init__(self):
        """Post-initialisation avec validations et enrichissements"""
        if not self.name:
            self.name = f"{self.tier.value}_tenant_{self.id[:8]}"
        if not self.display_name:
            self.display_name = f"{self.tier.value.title()} Tenant"


# =============================================================================
# ENTERPRISE TENANT TEMPLATE MANAGER
# =============================================================================

class EnterpriseTenantTemplateManager:
    """
    Gestionnaire ultra-avancé des templates de tenants
    
    Fonctionnalités:
    - Gestion multi-tier avec quotas dynamiques
    - IA intégrée pour optimisation automatique
    - Sécurité enterprise avec chiffrement
    - Conformité réglementaire automatisée
    - Orchestration multi-cloud
    - Monitoring temps réel avec métriques
    """
    
    def __init__(self, 
                 redis_client: Optional[redis.Redis] = None,
                 db_pool: Optional[asyncpg.Pool] = None,
                 encryption_key: Optional[bytes] = None):
        self.redis_client = redis_client or redis.Redis(decode_responses=True)
        self.db_pool = db_pool
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Templates en mémoire pour performance
        self._templates_cache: Dict[str, TenantTemplate] = {}
        self._tier_definitions: Dict[TenantTier, Dict[str, Any]] = {}
        
        # Metrics et logging
        self.logger = logging.getLogger(__name__)
        self._init_tier_definitions()
    
    def _init_tier_definitions(self):
        """Initialise les définitions par tier avec quotas optimisés"""
        self._tier_definitions = {
            TenantTier.FREE: {
                "resource_quotas": ResourceQuotas(
                    cpu_cores=1, memory_gb=1, storage_gb=10,
                    network_bandwidth_mbps=10, concurrent_connections=100,
                    api_requests_per_minute=100, ml_model_instances=0
                ),
                "security_level": SecurityLevel.BASIC,
                "ai_enabled": False,
                "monitoring_level": "basic"
            },
            TenantTier.STANDARD: {
                "resource_quotas": ResourceQuotas(
                    cpu_cores=2, memory_gb=4, storage_gb=100,
                    network_bandwidth_mbps=100, concurrent_connections=1000,
                    api_requests_per_minute=1000, ml_model_instances=1
                ),
                "security_level": SecurityLevel.ENHANCED,
                "ai_enabled": True,
                "monitoring_level": "standard"
            },
            TenantTier.PREMIUM: {
                "resource_quotas": ResourceQuotas(
                    cpu_cores=8, memory_gb=16, storage_gb=500,
                    network_bandwidth_mbps=500, concurrent_connections=5000,
                    api_requests_per_minute=5000, ml_model_instances=5
                ),
                "security_level": SecurityLevel.ENHANCED,
                "ai_enabled": True,
                "monitoring_level": "advanced"
            },
            TenantTier.ENTERPRISE: {
                "resource_quotas": ResourceQuotas(
                    cpu_cores=32, memory_gb=128, storage_gb=5000,
                    network_bandwidth_mbps=2000, concurrent_connections=50000,
                    api_requests_per_minute=50000, ml_model_instances=20
                ),
                "security_level": SecurityLevel.MAXIMUM,
                "ai_enabled": True,
                "monitoring_level": "enterprise"
            },
            TenantTier.ENTERPRISE_PLUS: {
                "resource_quotas": ResourceQuotas(
                    cpu_cores=128, memory_gb=512, storage_gb=20000,
                    network_bandwidth_mbps=10000, concurrent_connections=500000,
                    api_requests_per_minute=500000, ml_model_instances=100
                ),
                "security_level": SecurityLevel.CLASSIFIED,
                "ai_enabled": True,
                "monitoring_level": "maximum"
            }
        }
    
    async def create_tenant_template(self, 
                                   tier: TenantTier,
                                   template_name: str,
                                   custom_config: Optional[Dict[str, Any]] = None) -> TenantTemplate:
        """
        Crée un template de tenant optimisé selon le tier
        
        Args:
            tier: Niveau de tenant (FREE, STANDARD, PREMIUM, ENTERPRISE, etc.)
            template_name: Nom du template
            custom_config: Configuration personnalisée optionnelle
            
        Returns:
            TenantTemplate configuré et optimisé
        """
        start_time = datetime.utcnow()
        
        try:
            # Récupération de la configuration de base
            base_config = self._tier_definitions.get(tier, self._tier_definitions[TenantTier.STANDARD])
            
            # Création du template avec configuration intelligente
            template = TenantTemplate(
                name=template_name,
                display_name=f"{template_name.title()} - {tier.value.title()}",
                tier=tier,
                resource_quotas=base_config["resource_quotas"]
            )
            
            # Configuration de sécurité selon le tier
            template.security_config = await self._configure_security(tier, base_config["security_level"])
            
            # Configuration IA/ML
            if base_config["ai_enabled"]:
                template.ai_config = await self._configure_ai_services(tier)
            
            # Configuration de conformité
            template.compliance_settings = await self._configure_compliance(tier)
            
            # Configuration monitoring
            template.monitoring_config = await self._configure_monitoring(tier, base_config["monitoring_level"])
            
            # Application des configurations personnalisées
            if custom_config:
                template = await self._apply_custom_configuration(template, custom_config)
            
            # Validation et optimisation IA
            template = await self._ai_optimize_template(template)
            
            # Chiffrement et stockage sécurisé
            await self._store_template_securely(template)
            
            # Mise en cache pour performance
            self._templates_cache[template.id] = template
            
            # Métriques
            TENANT_CREATION_COUNTER.labels(tier=tier.value, region="default").inc()
            duration = (datetime.utcnow() - start_time).total_seconds()
            RESOURCE_ALLOCATION_HISTOGRAM.labels(resource_type="tenant_template").observe(duration)
            
            self.logger.info(f"Template tenant créé: {template.name} (tier: {tier.value}, id: {template.id})")
            
            return template
            
        except Exception as e:
            TEMPLATE_VALIDATION_COUNTER.labels(status="error").inc()
            self.logger.error(f"Erreur création template: {str(e)}")
            raise
    
    async def _configure_security(self, tier: TenantTier, security_level: SecurityLevel) -> SecurityConfiguration:
        """Configure la sécurité selon le tier et niveau"""
        config = SecurityConfiguration(encryption_level=security_level)
        
        if tier in [TenantTier.ENTERPRISE, TenantTier.ENTERPRISE_PLUS]:
            config.mfa_required = True
            config.zero_trust_networking = True
            config.penetration_testing = True
            config.vulnerability_scanning = True
            config.audit_logging = True
            
        if tier == TenantTier.ENTERPRISE_PLUS:
            config.encryption_level = SecurityLevel.CLASSIFIED
            config.ip_whitelist_enabled = True
            config.data_retention_days = 2555  # 7 ans
            
        return config
    
    async def _configure_ai_services(self, tier: TenantTier) -> AIConfiguration:
        """Configure les services IA selon le tier"""
        config = AIConfiguration()
        
        if tier == TenantTier.STANDARD:
            config.recommendation_engine_enabled = True
            config.model_training_quota_hours = 5
            config.inference_requests_per_day = 5000
            
        elif tier == TenantTier.PREMIUM:
            config.recommendation_engine_enabled = True
            config.sentiment_analysis_enabled = True
            config.nlp_processing_enabled = True
            config.model_training_quota_hours = 20
            config.inference_requests_per_day = 25000
            config.custom_models_allowed = 5
            
        elif tier in [TenantTier.ENTERPRISE, TenantTier.ENTERPRISE_PLUS]:
            config.recommendation_engine_enabled = True
            config.sentiment_analysis_enabled = True
            config.nlp_processing_enabled = True
            config.computer_vision_enabled = True
            config.auto_ml_enabled = True
            config.model_training_quota_hours = 100
            config.inference_requests_per_day = 1000000
            config.custom_models_allowed = 50
            config.gpu_acceleration = True
            config.federated_learning = True
            
        return config
    
    async def _configure_compliance(self, tier: TenantTier) -> ComplianceSettings:
        """Configure la conformité réglementaire"""
        config = ComplianceSettings()
        
        if tier in [TenantTier.PREMIUM, TenantTier.ENTERPRISE, TenantTier.ENTERPRISE_PLUS]:
            config.frameworks = [ComplianceFramework.GDPR]
            config.privacy_controls_enabled = True
            config.consent_management = True
            
        if tier in [TenantTier.ENTERPRISE, TenantTier.ENTERPRISE_PLUS]:
            config.frameworks.extend([ComplianceFramework.SOX, ComplianceFramework.ISO27001])
            config.data_localization_required = True
            config.audit_trail_retention_years = 10
            
        if tier == TenantTier.ENTERPRISE_PLUS:
            config.frameworks.append(ComplianceFramework.FedRAMP)
            config.data_residency_regions = ["us-gov-east-1", "us-gov-west-1"]
            
        return config
    
    async def _configure_monitoring(self, tier: TenantTier, monitoring_level: str) -> MonitoringConfiguration:
        """Configure le monitoring selon le niveau"""
        config = MonitoringConfiguration()
        
        if monitoring_level == "advanced":
            config.metrics_retention_days = 90
            config.logs_retention_days = 180
            config.custom_dashboards = True
            config.alerting_channels = ["email", "slack"]
            
        elif monitoring_level == "enterprise":
            config.metrics_retention_days = 365
            config.logs_retention_days = 730
            config.custom_dashboards = True
            config.alerting_channels = ["email", "slack", "pagerduty"]
            config.performance_profiling = True
            
        elif monitoring_level == "maximum":
            config.metrics_retention_days = 1095  # 3 ans
            config.logs_retention_days = 2555   # 7 ans
            config.custom_dashboards = True
            config.alerting_channels = ["email", "slack", "pagerduty", "webhook"]
            config.performance_profiling = True
            config.real_time_monitoring = True
            
        return config
    
    async def _apply_custom_configuration(self, template: TenantTemplate, custom_config: Dict[str, Any]) -> TenantTemplate:
        """Applique une configuration personnalisée au template"""
        # Application sécurisée des configurations personnalisées
        allowed_overrides = [
            "resource_quotas", "geographic_regions", "deployment_strategy",
            "backup_strategy", "tags", "metadata"
        ]
        
        for key, value in custom_config.items():
            if key in allowed_overrides:
                if hasattr(template, key):
                    setattr(template, key, value)
                    
        return template
    
    async def _ai_optimize_template(self, template: TenantTemplate) -> TenantTemplate:
        """Optimise le template avec l'IA (patterns ML pour allocation ressources)"""
        # Simulation d'optimisation IA basée sur les patterns historiques
        
        # Optimisation CPU/Memory ratio selon le tier
        if template.tier in [TenantTier.ENTERPRISE, TenantTier.ENTERPRISE_PLUS]:
            # Ratio optimisé pour workloads enterprise
            optimal_memory = template.resource_quotas.cpu_cores * 4  # 4GB par core
            if template.resource_quotas.memory_gb < optimal_memory:
                template.resource_quotas.memory_gb = optimal_memory
        
        # Optimisation réseau selon les régions
        if template.multi_region_enabled:
            template.resource_quotas.network_bandwidth_mbps *= 2  # Bande passante inter-région
        
        # Recommandations IA pour monitoring
        if template.ai_config.recommendation_engine_enabled:
            template.monitoring_config.performance_profiling = True
            
        return template
    
    async def _store_template_securely(self, template: TenantTemplate):
        """Stocke le template de manière sécurisée avec chiffrement"""
        try:
            # Sérialisation et chiffrement
            template_data = {
                "id": template.id,
                "name": template.name,
                "tier": template.tier.value,
                "created_at": template.created_at.isoformat(),
                "data": self._serialize_template(template)
            }
            
            encrypted_data = self.cipher_suite.encrypt(json.dumps(template_data).encode())
            
            # Stockage Redis avec TTL
            cache_key = f"tenant_template:{template.id}"
            await self._store_in_redis(cache_key, encrypted_data, ttl=86400)  # 24h
            
            # Stockage persistant en base de données
            if self.db_pool:
                await self._store_in_database(template)
                
        except Exception as e:
            self.logger.error(f"Erreur stockage template {template.id}: {str(e)}")
            raise
    
    async def _store_in_redis(self, key: str, data: bytes, ttl: int):
        """Stockage Redis avec gestion d'erreurs"""
        try:
            self.redis_client.setex(key, ttl, data)
        except Exception as e:
            self.logger.warning(f"Erreur Redis: {str(e)}")
    
    async def _store_in_database(self, template: TenantTemplate):
        """Stockage en base de données PostgreSQL"""
        if not self.db_pool:
            return
            
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO tenant_templates (id, name, tier, data, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (id) DO UPDATE SET
                        data = EXCLUDED.data,
                        updated_at = EXCLUDED.updated_at
                """, 
                template.id, template.name, template.tier.value,
                json.dumps(self._serialize_template(template)),
                template.created_at, template.updated_at)
        except Exception as e:
            self.logger.error(f"Erreur base de données: {str(e)}")
    
    def _serialize_template(self, template: TenantTemplate) -> Dict[str, Any]:
        """Sérialise un template pour stockage"""
        return {
            "id": template.id,
            "name": template.name,
            "display_name": template.display_name,
            "tier": template.tier.value,
            "version": template.version,
            "created_at": template.created_at.isoformat(),
            "updated_at": template.updated_at.isoformat(),
            "resource_quotas": {
                "cpu_cores": template.resource_quotas.cpu_cores,
                "memory_gb": template.resource_quotas.memory_gb,
                "storage_gb": template.resource_quotas.storage_gb,
                "network_bandwidth_mbps": template.resource_quotas.network_bandwidth_mbps,
                "concurrent_connections": template.resource_quotas.concurrent_connections,
                "api_requests_per_minute": template.resource_quotas.api_requests_per_minute,
                "ml_model_instances": template.resource_quotas.ml_model_instances,
                "ai_processing_units": template.resource_quotas.ai_processing_units,
                "database_connections": template.resource_quotas.database_connections,
                "cache_size_mb": template.resource_quotas.cache_size_mb
            },
            "security_config": {
                "encryption_level": template.security_config.encryption_level.value,
                "mfa_required": template.security_config.mfa_required,
                "ip_whitelist_enabled": template.security_config.ip_whitelist_enabled,
                "allowed_ip_ranges": template.security_config.allowed_ip_ranges,
                "data_retention_days": template.security_config.data_retention_days,
                "audit_logging": template.security_config.audit_logging,
                "vulnerability_scanning": template.security_config.vulnerability_scanning,
                "penetration_testing": template.security_config.penetration_testing,
                "zero_trust_networking": template.security_config.zero_trust_networking,
                "end_to_end_encryption": template.security_config.end_to_end_encryption
            },
            "compliance_settings": {
                "frameworks": [f.value for f in template.compliance_settings.frameworks],
                "data_localization_required": template.compliance_settings.data_localization_required,
                "data_residency_regions": template.compliance_settings.data_residency_regions,
                "audit_trail_retention_years": template.compliance_settings.audit_trail_retention_years,
                "privacy_controls_enabled": template.compliance_settings.privacy_controls_enabled,
                "consent_management": template.compliance_settings.consent_management,
                "right_to_be_forgotten": template.compliance_settings.right_to_be_forgotten,
                "data_portability": template.compliance_settings.data_portability
            },
            "ai_config": {
                "recommendation_engine_enabled": template.ai_config.recommendation_engine_enabled,
                "sentiment_analysis_enabled": template.ai_config.sentiment_analysis_enabled,
                "nlp_processing_enabled": template.ai_config.nlp_processing_enabled,
                "computer_vision_enabled": template.ai_config.computer_vision_enabled,
                "auto_ml_enabled": template.ai_config.auto_ml_enabled,
                "model_training_quota_hours": template.ai_config.model_training_quota_hours,
                "inference_requests_per_day": template.ai_config.inference_requests_per_day,
                "custom_models_allowed": template.ai_config.custom_models_allowed,
                "gpu_acceleration": template.ai_config.gpu_acceleration,
                "federated_learning": template.ai_config.federated_learning
            },
            "monitoring_config": {
                "metrics_retention_days": template.monitoring_config.metrics_retention_days,
                "logs_retention_days": template.monitoring_config.logs_retention_days,
                "traces_enabled": template.monitoring_config.traces_enabled,
                "custom_dashboards": template.monitoring_config.custom_dashboards,
                "alerting_channels": template.monitoring_config.alerting_channels,
                "sla_monitoring": template.monitoring_config.sla_monitoring,
                "performance_profiling": template.monitoring_config.performance_profiling,
                "real_time_monitoring": template.monitoring_config.real_time_monitoring
            },
            "metadata": template.metadata,
            "tags": template.tags,
            "deployment_strategy": template.deployment_strategy.value,
            "geographic_regions": template.geographic_regions,
            "multi_region_enabled": template.multi_region_enabled,
            "disaster_recovery_enabled": template.disaster_recovery_enabled,
            "backup_strategy": template.backup_strategy
        }
    
    async def get_template(self, template_id: str) -> Optional[TenantTemplate]:
        """Récupère un template par son ID"""
        # Vérification cache mémoire
        if template_id in self._templates_cache:
            return self._templates_cache[template_id]
        
        # Vérification Redis
        try:
            cache_key = f"tenant_template:{template_id}"
            encrypted_data = self.redis_client.get(cache_key)
            if encrypted_data:
                decrypted_data = self.cipher_suite.decrypt(encrypted_data.encode())
                template_data = json.loads(decrypted_data.decode())
                template = self._deserialize_template(template_data["data"])
                self._templates_cache[template_id] = template
                return template
        except Exception as e:
            self.logger.warning(f"Erreur récupération Redis: {str(e)}")
        
        # Récupération base de données
        if self.db_pool:
            try:
                async with self.db_pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT data FROM tenant_templates WHERE id = $1", template_id
                    )
                    if row:
                        template = self._deserialize_template(json.loads(row["data"]))
                        self._templates_cache[template_id] = template
                        return template
            except Exception as e:
                self.logger.error(f"Erreur base de données: {str(e)}")
        
        return None
    
    def _deserialize_template(self, data: Dict[str, Any]) -> TenantTemplate:
        """Désérialise un template depuis les données stockées"""
        # Reconstruction des objets complexes
        resource_quotas = ResourceQuotas(**data["resource_quotas"])
        
        security_config = SecurityConfiguration(
            encryption_level=SecurityLevel(data["security_config"]["encryption_level"]),
            mfa_required=data["security_config"]["mfa_required"],
            ip_whitelist_enabled=data["security_config"]["ip_whitelist_enabled"],
            allowed_ip_ranges=data["security_config"]["allowed_ip_ranges"],
            data_retention_days=data["security_config"]["data_retention_days"],
            audit_logging=data["security_config"]["audit_logging"],
            vulnerability_scanning=data["security_config"]["vulnerability_scanning"],
            penetration_testing=data["security_config"]["penetration_testing"],
            zero_trust_networking=data["security_config"]["zero_trust_networking"],
            end_to_end_encryption=data["security_config"]["end_to_end_encryption"]
        )
        
        compliance_settings = ComplianceSettings(
            frameworks=[ComplianceFramework(f) for f in data["compliance_settings"]["frameworks"]],
            data_localization_required=data["compliance_settings"]["data_localization_required"],
            data_residency_regions=data["compliance_settings"]["data_residency_regions"],
            audit_trail_retention_years=data["compliance_settings"]["audit_trail_retention_years"],
            privacy_controls_enabled=data["compliance_settings"]["privacy_controls_enabled"],
            consent_management=data["compliance_settings"]["consent_management"],
            right_to_be_forgotten=data["compliance_settings"]["right_to_be_forgotten"],
            data_portability=data["compliance_settings"]["data_portability"]
        )
        
        ai_config = AIConfiguration(**data["ai_config"])
        monitoring_config = MonitoringConfiguration(**data["monitoring_config"])
        
        return TenantTemplate(
            id=data["id"],
            name=data["name"],
            display_name=data["display_name"],
            tier=TenantTier(data["tier"]),
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            resource_quotas=resource_quotas,
            security_config=security_config,
            compliance_settings=compliance_settings,
            ai_config=ai_config,
            monitoring_config=monitoring_config,
            metadata=data["metadata"],
            tags=data["tags"],
            deployment_strategy=DeploymentStrategy(data["deployment_strategy"]),
            geographic_regions=data["geographic_regions"],
            multi_region_enabled=data["multi_region_enabled"],
            disaster_recovery_enabled=data["disaster_recovery_enabled"],
            backup_strategy=data["backup_strategy"]
        )
    
    async def list_templates_by_tier(self, tier: TenantTier) -> List[TenantTemplate]:
        """Liste tous les templates d'un tier donné"""
        templates = []
        
        # Recherche en base de données
        if self.db_pool:
            try:
                async with self.db_pool.acquire() as conn:
                    rows = await conn.fetch(
                        "SELECT data FROM tenant_templates WHERE tier = $1 ORDER BY created_at DESC",
                        tier.value
                    )
                    for row in rows:
                        template = self._deserialize_template(json.loads(row["data"]))
                        templates.append(template)
            except Exception as e:
                self.logger.error(f"Erreur listage templates: {str(e)}")
        
        return templates
    
    async def update_template_quotas(self, template_id: str, new_quotas: ResourceQuotas) -> bool:
        """Met à jour les quotas d'un template existant"""
        template = await self.get_template(template_id)
        if not template:
            return False
        
        template.resource_quotas = new_quotas
        template.updated_at = datetime.utcnow()
        
        await self._store_template_securely(template)
        self._templates_cache[template_id] = template
        
        return True
    
    async def clone_template(self, source_template_id: str, new_name: str) -> Optional[TenantTemplate]:
        """Clone un template existant avec un nouveau nom"""
        source_template = await self.get_template(source_template_id)
        if not source_template:
            return None
        
        new_template = TenantTemplate(
            name=new_name,
            display_name=f"{new_name.title()} - Cloned",
            tier=source_template.tier,
            resource_quotas=source_template.resource_quotas,
            security_config=source_template.security_config,
            compliance_settings=source_template.compliance_settings,
            ai_config=source_template.ai_config,
            monitoring_config=source_template.monitoring_config,
            deployment_strategy=source_template.deployment_strategy,
            geographic_regions=source_template.geographic_regions.copy(),
            multi_region_enabled=source_template.multi_region_enabled,
            disaster_recovery_enabled=source_template.disaster_recovery_enabled,
            backup_strategy=source_template.backup_strategy
        )
        
        await self._store_template_securely(new_template)
        return new_template
    
    async def export_template_yaml(self, template_id: str) -> Optional[str]:
        """Exporte un template au format YAML"""
        template = await self.get_template(template_id)
        if not template:
            return None
        
        template_data = self._serialize_template(template)
        return yaml.dump(template_data, default_flow_style=False, sort_keys=False)
    
    async def import_template_yaml(self, yaml_content: str) -> Optional[TenantTemplate]:
        """Importe un template depuis du YAML"""
        try:
            template_data = yaml.safe_load(yaml_content)
            template = self._deserialize_template(template_data)
            await self._store_template_securely(template)
            return template
        except Exception as e:
            self.logger.error(f"Erreur import YAML: {str(e)}")
            return None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des métriques du système"""
        return {
            "templates_in_cache": len(self._templates_cache),
            "supported_tiers": [tier.value for tier in TenantTier],
            "security_levels": [level.value for level in SecurityLevel],
            "compliance_frameworks": [framework.value for framework in ComplianceFramework],
            "deployment_strategies": [strategy.value for strategy in DeploymentStrategy]
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

async def create_enterprise_template_manager(
    redis_url: str = "redis://localhost:6379",
    database_url: str = "postgresql://user:pass@localhost/tenants",
    encryption_key: Optional[bytes] = None
) -> EnterpriseTenantTemplateManager:
    """Factory function pour créer un gestionnaire de templates configuré"""
    
    # Configuration Redis
    redis_client = redis.from_url(redis_url, decode_responses=True)
    
    # Configuration base de données
    db_pool = None
    try:
        db_pool = await asyncpg.create_pool(database_url)
    except Exception as e:
        logging.warning(f"Impossible de se connecter à la base de données: {str(e)}")
    
    return EnterpriseTenantTemplateManager(
        redis_client=redis_client,
        db_pool=db_pool,
        encryption_key=encryption_key
    )


def create_default_templates() -> Dict[TenantTier, TenantTemplate]:
    """Crée les templates par défaut pour chaque tier"""
    templates = {}
    
    for tier in TenantTier:
        template = TenantTemplate(
            name=f"default_{tier.value}",
            display_name=f"Default {tier.value.title()} Template",
            tier=tier
        )
        templates[tier] = template
    
    return templates


# =============================================================================
# UTILITAIRES & HELPERS
# =============================================================================

def validate_template_configuration(template: TenantTemplate) -> List[str]:
    """Valide la configuration d'un template et retourne les erreurs"""
    errors = []
    
    # Validation des quotas de ressources
    if template.resource_quotas.cpu_cores <= 0:
        errors.append("CPU cores must be positive")
    
    if template.resource_quotas.memory_gb <= 0:
        errors.append("Memory must be positive")
    
    if template.resource_quotas.storage_gb <= 0:
        errors.append("Storage must be positive")
    
    # Validation cohérence tier/ressources
    if template.tier == TenantTier.FREE and template.resource_quotas.cpu_cores > 2:
        errors.append("Free tier cannot have more than 2 CPU cores")
    
    if template.tier in [TenantTier.ENTERPRISE, TenantTier.ENTERPRISE_PLUS]:
        if not template.security_config.audit_logging:
            errors.append("Enterprise tiers must have audit logging enabled")
    
    # Validation conformité
    if template.compliance_settings.data_localization_required and not template.compliance_settings.data_residency_regions:
        errors.append("Data localization requires residency regions")
    
    return errors


def calculate_template_cost(template: TenantTemplate) -> float:
    """Calcule le coût estimé mensuel d'un template"""
    base_costs = {
        TenantTier.FREE: 0.0,
        TenantTier.STANDARD: 29.99,
        TenantTier.PREMIUM: 99.99,
        TenantTier.ENTERPRISE: 499.99,
        TenantTier.ENTERPRISE_PLUS: 1999.99
    }
    
    base_cost = base_costs.get(template.tier, 0.0)
    
    # Coûts additionnels basés sur les ressources
    resource_cost = (
        template.resource_quotas.cpu_cores * 10.0 +
        template.resource_quotas.memory_gb * 5.0 +
        template.resource_quotas.storage_gb * 0.1 +
        template.resource_quotas.ml_model_instances * 50.0
    )
    
    # Multiplicateurs pour fonctionnalités avancées
    multiplier = 1.0
    if template.ai_config.gpu_acceleration:
        multiplier += 0.5
    if template.security_config.encryption_level == SecurityLevel.CLASSIFIED:
        multiplier += 0.3
    if template.multi_region_enabled:
        multiplier += 0.4
    
    return (base_cost + resource_cost) * multiplier


# =============================================================================
# EXPORT DU MODULE
# =============================================================================

__all__ = [
    # Enums
    "TenantTier",
    "ResourceType", 
    "SecurityLevel",
    "ComplianceFramework",
    "DeploymentStrategy",
    
    # Dataclasses
    "ResourceQuotas",
    "SecurityConfiguration", 
    "ComplianceSettings",
    "AIConfiguration",
    "MonitoringConfiguration",
    "TenantTemplate",
    
    # Manager principal
    "EnterpriseTenantTemplateManager",
    
    # Factory functions
    "create_enterprise_template_manager",
    "create_default_templates",
    
    # Utilitaires
    "validate_template_configuration",
    "calculate_template_cost"
]
