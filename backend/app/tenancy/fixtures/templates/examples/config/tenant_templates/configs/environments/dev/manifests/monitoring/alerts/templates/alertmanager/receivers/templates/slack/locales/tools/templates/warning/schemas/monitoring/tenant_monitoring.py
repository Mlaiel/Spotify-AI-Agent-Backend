"""
Advanced Tenant Monitoring - Industrial Multi-Tenant Isolation System
====================================================================

Ce module fournit une architecture de monitoring multi-tenant ultra-avancée
avec isolation complète, sécurité renforcée et analytics per-tenant.

Features:
- Complete tenant isolation and data segregation
- Per-tenant SLA monitoring and reporting
- Tenant-specific alerting and escalation
- Resource usage tracking and billing
- Compliance monitoring per tenant
- ML-based tenant behavior analysis
- Auto-scaling per tenant workloads
"""

from typing import Dict, List, Optional, Union, Any, Set
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timedelta
import json
from .metric_schemas import MetricSchema, MetricCategory, MetricType
from .alert_schemas import AlertRule, AlertPriority
from .dashboard_schemas import Dashboard, DashboardType


class TenantTier(str, Enum):
    """Niveaux de service par tenant"""
    ENTERPRISE = "enterprise"        # Service premium avec SLA strict
    PROFESSIONAL = "professional"    # Service standard
    STARTER = "starter"              # Service basique
    TRIAL = "trial"                  # Version d'évaluation
    CUSTOM = "custom"                # Configuration personnalisée


class TenantStatus(str, Enum):
    """Statuts de tenant"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    MIGRATING = "migrating"
    MAINTENANCE = "maintenance"
    DECOMMISSIONED = "decommissioned"


class IsolationLevel(str, Enum):
    """Niveaux d'isolation"""
    STRICT = "strict"                # Isolation complète (DB, réseau, compute)
    STANDARD = "standard"            # Isolation logique standard
    SHARED = "shared"                # Ressources partagées avec étiquetage
    HYBRID = "hybrid"                # Isolation hybride selon données


class DataRetentionPolicy(BaseModel):
    """Politique de rétention des données par tenant"""
    metrics_retention: str = Field("30d", description="Rétention métriques")
    logs_retention: str = Field("7d", description="Rétention logs")
    traces_retention: str = Field("3d", description="Rétention traces")
    events_retention: str = Field("90d", description="Rétention événements")
    compliance_retention: str = Field("7y", description="Rétention conformité")
    
    # Configuration compression
    enable_compression: bool = Field(True, description="Compression activée")
    compression_after: str = Field("1d", description="Compression après")
    
    # Archivage
    archive_enabled: bool = Field(True, description="Archivage activé")
    archive_after: str = Field("30d", description="Archivage après")
    archive_storage: str = Field("s3", description="Stockage archive")


class TenantQuota(BaseModel):
    """Quotas et limites par tenant"""
    
    # Métriques
    max_metrics: int = Field(10000, description="Nombre max de métriques")
    max_cardinality: int = Field(1000000, description="Cardinalité max")
    ingestion_rate: int = Field(10000, description="Taux ingestion/sec")
    
    # Queries
    max_query_rate: int = Field(100, description="Requêtes/sec max")
    max_query_duration: str = Field("5m", description="Durée max requête")
    max_query_complexity: int = Field(1000, description="Complexité max")
    
    # Alertes
    max_alert_rules: int = Field(500, description="Règles alerte max")
    max_notifications_per_hour: int = Field(1000, description="Notifications/h max")
    
    # Tableaux de bord
    max_dashboards: int = Field(100, description="Dashboards max")
    max_widgets_per_dashboard: int = Field(50, description="Widgets/dashboard max")
    
    # Stockage
    max_storage_gb: float = Field(100.0, description="Stockage max (GB)")
    
    # Ressources compute
    max_cpu_cores: float = Field(4.0, description="CPU cores max")
    max_memory_gb: float = Field(8.0, description="Mémoire max (GB)")
    
    @validator('max_storage_gb', 'max_cpu_cores', 'max_memory_gb')
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError('Value must be positive')
        return v


class TenantSLA(BaseModel):
    """Service Level Agreement par tenant"""
    
    # Availability
    availability_target: float = Field(99.9, description="Disponibilité cible (%)")
    
    # Performance
    response_time_p95: float = Field(500.0, description="Temps réponse P95 (ms)")
    response_time_p99: float = Field(1000.0, description="Temps réponse P99 (ms)")
    
    # Throughput
    min_throughput: int = Field(1000, description="Débit minimum (req/s)")
    
    # Error rates
    max_error_rate: float = Field(0.1, description="Taux erreur max (%)")
    
    # Data consistency
    data_consistency_target: float = Field(99.99, description="Cohérence données (%)")
    
    # Recovery time
    rto: str = Field("4h", description="Recovery Time Objective")
    rpo: str = Field("1h", description="Recovery Point Objective")
    
    # Support
    support_response_time: str = Field("1h", description="Temps réponse support")
    
    @validator('availability_target', 'data_consistency_target')
    def validate_percentage(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Percentage must be between 0 and 100')
        return v
    
    @validator('max_error_rate')
    def validate_error_rate(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Error rate must be between 0 and 100')
        return v


class TenantBilling(BaseModel):
    """Configuration de facturation par tenant"""
    
    # Modèle de facturation
    billing_model: str = Field("usage", description="Modèle (fixed, usage, hybrid)")
    
    # Coûts fixes
    base_cost_monthly: float = Field(0.0, description="Coût de base mensuel")
    
    # Coûts par usage
    cost_per_metric_ingested: float = Field(0.001, description="Coût/métrique ingérée")
    cost_per_query: float = Field(0.01, description="Coût/requête")
    cost_per_gb_storage: float = Field(0.1, description="Coût/GB stockage")
    cost_per_alert: float = Field(0.05, description="Coût/alerte")
    
    # Limites et overages
    overage_multiplier: float = Field(2.0, description="Multiplicateur dépassement")
    billing_currency: str = Field("USD", description="Devise")
    
    # Facturation temps réel
    real_time_billing: bool = Field(True, description="Facturation temps réel")
    billing_precision: int = Field(4, description="Précision décimales")


class TenantSecurityConfig(BaseModel):
    """Configuration sécurité par tenant"""
    
    # Chiffrement
    encryption_at_rest: bool = Field(True, description="Chiffrement au repos")
    encryption_in_transit: bool = Field(True, description="Chiffrement en transit")
    encryption_key_rotation: str = Field("90d", description="Rotation clés")
    
    # Authentification
    mfa_required: bool = Field(True, description="MFA obligatoire")
    session_timeout: str = Field("8h", description="Timeout session")
    max_concurrent_sessions: int = Field(10, description="Sessions simultanées max")
    
    # Autorisation
    rbac_enabled: bool = Field(True, description="RBAC activé")
    attribute_based_access: bool = Field(True, description="Contrôle d'accès par attributs")
    
    # Audit
    audit_logging: bool = Field(True, description="Logs d'audit")
    data_lineage_tracking: bool = Field(True, description="Traçabilité données")
    
    # Conformité
    compliance_frameworks: List[str] = Field(
        default_factory=lambda: ["GDPR", "SOC2"],
        description="Frameworks de conformité"
    )
    
    # Monitoring sécurité
    security_monitoring: bool = Field(True, description="Monitoring sécurité")
    threat_detection: bool = Field(True, description="Détection menaces")
    
    # PII/PHI
    pii_detection: bool = Field(True, description="Détection PII")
    data_anonymization: bool = Field(True, description="Anonymisation données")


class TenantMLConfig(BaseModel):
    """Configuration ML/IA par tenant"""
    
    # Modèles ML
    enable_anomaly_detection: bool = Field(True, description="Détection anomalies")
    enable_predictive_analytics: bool = Field(True, description="Analytics prédictifs")
    enable_auto_scaling: bool = Field(True, description="Auto-scaling ML")
    
    # Configuration modèles
    model_training_schedule: str = Field("daily", description="Planning entraînement")
    model_retention: str = Field("30d", description="Rétention modèles")
    
    # Ressources ML
    ml_compute_quota: float = Field(2.0, description="Quota compute ML (cores)")
    ml_memory_quota: float = Field(4.0, description="Quota mémoire ML (GB)")
    
    # Privacy ML
    differential_privacy: bool = Field(False, description="Confidentialité différentielle")
    federated_learning: bool = Field(False, description="Apprentissage fédéré")


class TenantConfig(BaseModel):
    """Configuration complète d'un tenant"""
    
    # Identifiants
    tenant_id: str = Field(..., description="ID unique du tenant")
    tenant_name: str = Field(..., description="Nom du tenant")
    display_name: str = Field(..., description="Nom d'affichage")
    
    # Métadonnées
    description: str = Field("", description="Description")
    tags: List[str] = Field(default_factory=list, description="Tags")
    
    # Configuration de service
    tier: TenantTier = Field(..., description="Niveau de service")
    status: TenantStatus = Field(TenantStatus.ACTIVE, description="Statut")
    isolation_level: IsolationLevel = Field(IsolationLevel.STANDARD, description="Niveau isolation")
    
    # Limites et quotas
    quotas: TenantQuota = Field(default_factory=TenantQuota, description="Quotas")
    
    # SLA et performance
    sla: TenantSLA = Field(default_factory=TenantSLA, description="SLA")
    
    # Rétention des données
    data_retention: DataRetentionPolicy = Field(
        default_factory=DataRetentionPolicy, description="Politique rétention"
    )
    
    # Facturation
    billing: TenantBilling = Field(default_factory=TenantBilling, description="Configuration facturation")
    
    # Sécurité
    security: TenantSecurityConfig = Field(
        default_factory=TenantSecurityConfig, description="Configuration sécurité"
    )
    
    # ML/IA
    ml_config: TenantMLConfig = Field(default_factory=TenantMLConfig, description="Configuration ML")
    
    # Régions et déploiement
    regions: List[str] = Field(default_factory=list, description="Régions de déploiement")
    primary_region: str = Field("us-east-1", description="Région primaire")
    
    # Contacts
    admin_contacts: List[str] = Field(default_factory=list, description="Contacts administrateurs")
    technical_contacts: List[str] = Field(default_factory=list, description="Contacts techniques")
    billing_contacts: List[str] = Field(default_factory=list, description="Contacts facturation")
    
    # Métadonnées techniques
    namespace: str = Field(..., description="Namespace Kubernetes")
    database_schema: str = Field(..., description="Schéma base de données")
    
    # Audit
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field("", description="Créé par")
    
    @validator('tenant_id')
    def validate_tenant_id(cls, v):
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Tenant ID must be alphanumeric with hyphens/underscores')
        return v.lower()
    
    class Config:
        schema_extra = {
            "example": {
                "tenant_id": "spotify-premium-corp",
                "tenant_name": "Spotify Premium Corporate",
                "tier": "enterprise",
                "isolation_level": "strict",
                "namespace": "tenant-spotify-premium-corp",
                "database_schema": "tenant_spotify_premium_corp"
            }
        }


class TenantMetrics(BaseModel):
    """Métriques spécifiques à un tenant"""
    
    tenant_id: str = Field(..., description="ID du tenant")
    
    # Utilisation des ressources
    current_cpu_usage: float = Field(0.0, description="Utilisation CPU actuelle")
    current_memory_usage: float = Field(0.0, description="Utilisation mémoire actuelle")
    current_storage_usage: float = Field(0.0, description="Utilisation stockage actuelle")
    
    # Métriques de performance
    avg_response_time: float = Field(0.0, description="Temps réponse moyen")
    request_rate: float = Field(0.0, description="Taux de requêtes")
    error_rate: float = Field(0.0, description="Taux d'erreur")
    
    # Métriques de conformité SLA
    availability_percentage: float = Field(0.0, description="Pourcentage disponibilité")
    sla_compliance_score: float = Field(0.0, description="Score conformité SLA")
    
    # Métriques de coût
    current_monthly_cost: float = Field(0.0, description="Coût mensuel actuel")
    projected_monthly_cost: float = Field(0.0, description="Coût mensuel projeté")
    
    # Métriques d'utilisation
    active_users: int = Field(0, description="Utilisateurs actifs")
    api_calls_per_hour: int = Field(0, description="Appels API/heure")
    data_ingested_gb: float = Field(0.0, description="Données ingérées (GB)")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TenantAlert(BaseModel):
    """Alerte spécifique à un tenant"""
    
    tenant_id: str = Field(..., description="ID du tenant")
    alert_name: str = Field(..., description="Nom de l'alerte")
    severity: str = Field(..., description="Sévérité")
    description: str = Field(..., description="Description")
    
    # Détails
    metric_name: str = Field(..., description="Nom métrique")
    current_value: float = Field(..., description="Valeur actuelle")
    threshold_value: float = Field(..., description="Valeur seuil")
    
    # État
    status: str = Field("firing", description="Statut alerte")
    started_at: datetime = Field(..., description="Début alerte")
    acknowledged_at: Optional[datetime] = Field(None, description="Acquittement")
    resolved_at: Optional[datetime] = Field(None, description="Résolution")
    
    # Contexte
    labels: Dict[str, str] = Field(default_factory=dict, description="Labels")
    annotations: Dict[str, str] = Field(default_factory=dict, description="Annotations")


class TenantMonitoringService(BaseModel):
    """Service de monitoring multi-tenant"""
    
    # Configuration globale
    service_name: str = Field("tenant-monitoring", description="Nom du service")
    version: str = Field("1.0.0", description="Version")
    
    # Tenants
    tenants: Dict[str, TenantConfig] = Field(default_factory=dict, description="Configuration tenants")
    
    # Templates et modèles
    tier_templates: Dict[TenantTier, TenantConfig] = Field(
        default_factory=dict, description="Templates par tier"
    )
    
    # Métriques globales
    global_metrics_enabled: bool = Field(True, description="Métriques globales activées")
    cross_tenant_analytics: bool = Field(False, description="Analytics cross-tenant")
    
    # Sécurité globale
    tenant_isolation_validation: bool = Field(True, description="Validation isolation")
    data_leak_detection: bool = Field(True, description="Détection fuite données")
    
    def add_tenant(self, tenant_config: TenantConfig) -> None:
        """Ajouter un nouveau tenant"""
        if tenant_config.tenant_id in self.tenants:
            raise ValueError(f"Tenant {tenant_config.tenant_id} already exists")
        
        # Validation des quotas selon le tier
        self._validate_tenant_quotas(tenant_config)
        
        self.tenants[tenant_config.tenant_id] = tenant_config
    
    def get_tenant(self, tenant_id: str) -> Optional[TenantConfig]:
        """Récupérer configuration d'un tenant"""
        return self.tenants.get(tenant_id)
    
    def update_tenant(self, tenant_id: str, updates: Dict[str, Any]) -> bool:
        """Mettre à jour configuration tenant"""
        if tenant_id not in self.tenants:
            return False
        
        tenant = self.tenants[tenant_id]
        for key, value in updates.items():
            if hasattr(tenant, key):
                setattr(tenant, key, value)
        
        tenant.updated_at = datetime.utcnow()
        return True
    
    def remove_tenant(self, tenant_id: str) -> bool:
        """Supprimer un tenant"""
        if tenant_id in self.tenants:
            del self.tenants[tenant_id]
            return True
        return False
    
    def get_tenants_by_tier(self, tier: TenantTier) -> List[TenantConfig]:
        """Récupérer tenants par tier"""
        return [t for t in self.tenants.values() if t.tier == tier]
    
    def get_active_tenants(self) -> List[TenantConfig]:
        """Récupérer tenants actifs"""
        return [t for t in self.tenants.values() if t.status == TenantStatus.ACTIVE]
    
    def _validate_tenant_quotas(self, tenant_config: TenantConfig) -> None:
        """Valider les quotas selon le tier"""
        tier_limits = {
            TenantTier.TRIAL: {"max_metrics": 100, "max_storage_gb": 1.0},
            TenantTier.STARTER: {"max_metrics": 1000, "max_storage_gb": 10.0},
            TenantTier.PROFESSIONAL: {"max_metrics": 5000, "max_storage_gb": 50.0},
            TenantTier.ENTERPRISE: {"max_metrics": 50000, "max_storage_gb": 500.0}
        }
        
        if tenant_config.tier in tier_limits:
            limits = tier_limits[tenant_config.tier]
            if tenant_config.quotas.max_metrics > limits["max_metrics"]:
                raise ValueError(f"Metrics quota exceeds tier limit")
            if tenant_config.quotas.max_storage_gb > limits["max_storage_gb"]:
                raise ValueError(f"Storage quota exceeds tier limit")


# Configurations prédéfinies par tier
def create_enterprise_tenant_template() -> TenantConfig:
    """Créer template tenant enterprise"""
    return TenantConfig(
        tenant_id="template-enterprise",
        tenant_name="Enterprise Template",
        tier=TenantTier.ENTERPRISE,
        isolation_level=IsolationLevel.STRICT,
        quotas=TenantQuota(
            max_metrics=50000,
            max_cardinality=10000000,
            ingestion_rate=50000,
            max_storage_gb=500.0,
            max_cpu_cores=16.0,
            max_memory_gb=32.0
        ),
        sla=TenantSLA(
            availability_target=99.95,
            response_time_p95=200.0,
            response_time_p99=500.0,
            max_error_rate=0.01,
            rto="1h",
            rpo="15m"
        ),
        security=TenantSecurityConfig(
            mfa_required=True,
            encryption_at_rest=True,
            encryption_in_transit=True,
            compliance_frameworks=["GDPR", "SOC2", "HIPAA", "PCI-DSS"]
        ),
        namespace="tenant-enterprise-template",
        database_schema="tenant_enterprise_template"
    )


def create_professional_tenant_template() -> TenantConfig:
    """Créer template tenant professional"""
    return TenantConfig(
        tenant_id="template-professional",
        tenant_name="Professional Template", 
        tier=TenantTier.PROFESSIONAL,
        isolation_level=IsolationLevel.STANDARD,
        quotas=TenantQuota(
            max_metrics=5000,
            max_storage_gb=50.0,
            max_cpu_cores=4.0,
            max_memory_gb=8.0
        ),
        sla=TenantSLA(
            availability_target=99.9,
            response_time_p95=500.0,
            response_time_p99=1000.0
        ),
        namespace="tenant-professional-template",
        database_schema="tenant_professional_template"
    )


def create_default_tenant_monitoring_service() -> TenantMonitoringService:
    """Créer service de monitoring avec templates par défaut"""
    service = TenantMonitoringService()
    
    # Ajouter templates par tier
    service.tier_templates[TenantTier.ENTERPRISE] = create_enterprise_tenant_template()
    service.tier_templates[TenantTier.PROFESSIONAL] = create_professional_tenant_template()
    
    return service


# Export des classes principales
__all__ = [
    "TenantTier",
    "TenantStatus",
    "IsolationLevel",
    "DataRetentionPolicy",
    "TenantQuota",
    "TenantSLA",
    "TenantBilling",
    "TenantSecurityConfig",
    "TenantMLConfig",
    "TenantConfig",
    "TenantMetrics",
    "TenantAlert",
    "TenantMonitoringService",
    "create_enterprise_tenant_template",
    "create_professional_tenant_template",
    "create_default_tenant_monitoring_service"
]
