"""
Tenant Management Schemas - Ultra-Advanced Edition
================================================

Schémas ultra-avancés pour la gestion multi-tenant avec isolation complète,
analytics par tenant, facturation, conformité et monitoring avancé.

Features:
- Isolation multi-tenant complète
- Métriques et analytics par tenant
- Facturation et usage tracking
- Conformité et audit
- Provisioning automatisé
- SLA et gouvernance
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Literal, Set, Tuple
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import UUID4, PositiveInt, NonNegativeFloat, PositiveFloat, EmailStr
import json


class TenantTier(str, Enum):
    """Niveaux de service tenant."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class TenantStatus(str, Enum):
    """Statuts des tenants."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PROVISIONING = "provisioning"
    DEPROVISIONING = "deprovisioning"
    MIGRATING = "migrating"
    MAINTENANCE = "maintenance"
    ARCHIVED = "archived"


class BillingModel(str, Enum):
    """Modèles de facturation."""
    FIXED = "fixed"
    PER_USER = "per_user"
    USAGE_BASED = "usage_based"
    HYBRID = "hybrid"
    FREEMIUM = "freemium"
    ENTERPRISE = "enterprise"


class ComplianceFramework(str, Enum):
    """Frameworks de conformité."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"
    SOC2 = "soc2"


class TenantResourceLimits(BaseModel):
    """Limites de ressources par tenant."""
    
    # Limites utilisateurs
    max_users: Optional[PositiveInt] = Field(None, description="Nombre max d'utilisateurs")
    max_admin_users: Optional[PositiveInt] = Field(None, description="Nombre max d'admins")
    max_concurrent_sessions: Optional[PositiveInt] = Field(None, description="Sessions concurrentes max")
    
    # Limites données
    max_storage_gb: Optional[PositiveFloat] = Field(None, description="Stockage max en GB")
    max_data_transfer_gb_month: Optional[PositiveFloat] = Field(None, description="Transfert max par mois")
    max_database_size_gb: Optional[PositiveFloat] = Field(None, description="Taille DB max")
    
    # Limites API
    max_api_calls_per_minute: Optional[PositiveInt] = Field(None, description="Appels API max/min")
    max_api_calls_per_day: Optional[PositiveInt] = Field(None, description="Appels API max/jour")
    max_webhook_endpoints: Optional[PositiveInt] = Field(None, description="Webhooks max")
    
    # Limites compute
    max_cpu_cores: Optional[PositiveInt] = Field(None, description="Cœurs CPU max")
    max_memory_gb: Optional[PositiveFloat] = Field(None, description="Mémoire max en GB")
    max_processing_time_minutes: Optional[PositiveInt] = Field(None, description="Temps traitement max")
    
    # Limites ML/AI
    max_ml_models: Optional[PositiveInt] = Field(None, description="Modèles ML max")
    max_training_hours_month: Optional[PositiveFloat] = Field(None, description="Heures training max/mois")
    max_predictions_per_day: Optional[PositiveInt] = Field(None, description="Prédictions max/jour")
    
    # Limites fonctionnelles
    max_projects: Optional[PositiveInt] = Field(None, description="Projets max")
    max_integrations: Optional[PositiveInt] = Field(None, description="Intégrations max")
    max_custom_fields: Optional[PositiveInt] = Field(None, description="Champs custom max")
    
    # Rétention des données
    data_retention_days: Optional[PositiveInt] = Field(None, description="Rétention données en jours")
    backup_retention_days: Optional[PositiveInt] = Field(None, description="Rétention backups en jours")
    audit_log_retention_days: Optional[PositiveInt] = Field(None, description="Rétention logs audit")


class TenantFeatures(BaseModel):
    """Features disponibles par tenant."""
    
    # Features de base
    basic_analytics: bool = Field(default=True, description="Analytics de base")
    advanced_analytics: bool = Field(default=False, description="Analytics avancé")
    real_time_monitoring: bool = Field(default=False, description="Monitoring temps réel")
    
    # Features ML/AI
    ml_models: bool = Field(default=False, description="Modèles ML")
    auto_ml: bool = Field(default=False, description="AutoML")
    custom_algorithms: bool = Field(default=False, description="Algorithmes custom")
    
    # Features intégration
    api_access: bool = Field(default=True, description="Accès API")
    webhook_support: bool = Field(default=False, description="Support webhooks")
    third_party_integrations: bool = Field(default=False, description="Intégrations tierces")
    
    # Features collaboration
    team_management: bool = Field(default=False, description="Gestion équipe")
    role_based_access: bool = Field(default=False, description="Contrôle d'accès basé rôles")
    audit_logging: bool = Field(default=False, description="Logs d'audit")
    
    # Features support
    email_support: bool = Field(default=True, description="Support email")
    phone_support: bool = Field(default=False, description="Support téléphone")
    dedicated_support: bool = Field(default=False, description="Support dédié")
    sla_guarantee: bool = Field(default=False, description="Garantie SLA")
    
    # Features sécurité
    sso_integration: bool = Field(default=False, description="Intégration SSO")
    two_factor_auth: bool = Field(default=False, description="Authentification 2FA")
    ip_whitelisting: bool = Field(default=False, description="Whitelist IP")
    custom_security_policies: bool = Field(default=False, description="Politiques sécurité custom")
    
    # Features personnalisation
    custom_branding: bool = Field(default=False, description="Branding personnalisé")
    custom_domains: bool = Field(default=False, description="Domaines personnalisés")
    white_labeling: bool = Field(default=False, description="White labeling")


class TenantConfiguration(BaseModel):
    """Configuration complète d'un tenant."""
    
    tenant_id: UUID4 = Field(..., description="ID unique du tenant")
    name: str = Field(..., min_length=1, max_length=200, description="Nom du tenant")
    slug: str = Field(..., min_length=1, max_length=100, description="Slug unique")
    description: Optional[str] = Field(None, max_length=1000, description="Description")
    
    # Informations organisation
    organization_name: str = Field(..., description="Nom de l'organisation")
    industry: Optional[str] = Field(None, description="Secteur d'activité")
    company_size: Optional[str] = Field(None, description="Taille de l'entreprise")
    country: str = Field(..., description="Pays")
    timezone: str = Field(default="UTC", description="Fuseau horaire")
    
    # Contact principal
    primary_contact_email: EmailStr = Field(..., description="Email contact principal")
    primary_contact_name: str = Field(..., description="Nom contact principal")
    billing_email: EmailStr = Field(..., description="Email facturation")
    technical_contact_email: Optional[EmailStr] = Field(None, description="Email contact technique")
    
    # Configuration de service
    tier: TenantTier = Field(..., description="Niveau de service")
    status: TenantStatus = Field(default=TenantStatus.PROVISIONING, description="Statut")
    billing_model: BillingModel = Field(..., description="Modèle de facturation")
    
    # Dates importantes
    created_at: datetime = Field(default_factory=datetime.utcnow)
    activated_at: Optional[datetime] = Field(None, description="Date d'activation")
    trial_ends_at: Optional[datetime] = Field(None, description="Fin de période d'essai")
    contract_start_date: Optional[datetime] = Field(None, description="Début contrat")
    contract_end_date: Optional[datetime] = Field(None, description="Fin contrat")
    
    # Limites et features
    resource_limits: TenantResourceLimits = Field(default_factory=TenantResourceLimits)
    features: TenantFeatures = Field(default_factory=TenantFeatures)
    
    # Configuration technique
    database_schema: str = Field(..., description="Schéma de base de données")
    storage_bucket: str = Field(..., description="Bucket de stockage")
    cdn_domain: Optional[str] = Field(None, description="Domaine CDN")
    custom_domain: Optional[str] = Field(None, description="Domaine personnalisé")
    
    # Sécurité
    encryption_key_id: str = Field(..., description="ID clé de chiffrement")
    compliance_frameworks: List[ComplianceFramework] = Field(default_factory=list)
    data_residency_region: str = Field(..., description="Région résidence données")
    
    # Métadonnées
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags personnalisés")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées custom")
    
    @validator('slug')
    def validate_slug(cls, v):
        import re
        if not re.match(r'^[a-z0-9-]+$', v):
            raise ValueError("Slug must contain only lowercase letters, numbers, and hyphens")
        return v
    
    @validator('trial_ends_at')
    def validate_trial_period(cls, v, values):
        if v is not None and 'created_at' in values:
            if v <= values['created_at']:
                raise ValueError("Trial end date must be after creation date")
        return v


class TenantUsageMetrics(BaseModel):
    """Métriques d'usage détaillées par tenant."""
    
    tenant_id: UUID4 = Field(..., description="ID du tenant")
    measurement_period_start: datetime = Field(..., description="Début période mesure")
    measurement_period_end: datetime = Field(..., description="Fin période mesure")
    
    # Métriques utilisateurs
    total_users: int = Field(default=0, ge=0, description="Total utilisateurs")
    active_users: int = Field(default=0, ge=0, description="Utilisateurs actifs")
    monthly_active_users: int = Field(default=0, ge=0, description="Utilisateurs actifs mensuels")
    new_users: int = Field(default=0, ge=0, description="Nouveaux utilisateurs")
    
    # Métriques d'engagement
    total_sessions: int = Field(default=0, ge=0, description="Total sessions")
    avg_session_duration_minutes: NonNegativeFloat = Field(default=0.0, description="Durée session moyenne")
    total_page_views: int = Field(default=0, ge=0, description="Total vues pages")
    feature_usage: Dict[str, int] = Field(default_factory=dict, description="Usage par feature")
    
    # Métriques API
    total_api_calls: int = Field(default=0, ge=0, description="Total appels API")
    successful_api_calls: int = Field(default=0, ge=0, description="Appels API réussis")
    failed_api_calls: int = Field(default=0, ge=0, description="Appels API échoués")
    avg_api_response_time_ms: NonNegativeFloat = Field(default=0.0, description="Temps réponse API moyen")
    
    # Métriques données
    storage_used_gb: NonNegativeFloat = Field(default=0.0, description="Stockage utilisé en GB")
    data_transfer_gb: NonNegativeFloat = Field(default=0.0, description="Transfert données en GB")
    database_queries: int = Field(default=0, ge=0, description="Requêtes base de données")
    
    # Métriques ML/AI
    ml_predictions: int = Field(default=0, ge=0, description="Prédictions ML")
    training_hours: NonNegativeFloat = Field(default=0.0, description="Heures d'entraînement")
    model_deployments: int = Field(default=0, ge=0, description="Déploiements modèles")
    
    # Métriques compute
    cpu_hours: NonNegativeFloat = Field(default=0.0, description="Heures CPU")
    memory_gb_hours: NonNegativeFloat = Field(default=0.0, description="GB-heures mémoire")
    processing_jobs: int = Field(default=0, ge=0, description="Jobs de traitement")
    
    # Métriques business
    revenue_generated: Optional[Decimal] = Field(None, description="Revenus générés")
    transactions_processed: int = Field(default=0, ge=0, description="Transactions traitées")
    conversion_events: int = Field(default=0, ge=0, description="Événements conversion")
    
    # Métriques support
    support_tickets: int = Field(default=0, ge=0, description="Tickets support")
    avg_resolution_time_hours: Optional[NonNegativeFloat] = Field(None, description="Temps résolution moyen")
    satisfaction_score: Optional[float] = Field(None, ge=1.0, le=5.0, description="Score satisfaction")
    
    # Calculs dérivés
    @property
    def api_success_rate(self) -> float:
        """Calcule le taux de succès des API."""
        total = self.total_api_calls
        if total == 0:
            return 0.0
        return self.successful_api_calls / total
    
    @property
    def user_engagement_rate(self) -> float:
        """Calcule le taux d'engagement utilisateur."""
        if self.total_users == 0:
            return 0.0
        return self.active_users / self.total_users


class TenantBilling(BaseModel):
    """Informations de facturation détaillées par tenant."""
    
    tenant_id: UUID4 = Field(..., description="ID du tenant")
    billing_period_start: datetime = Field(..., description="Début période facturation")
    billing_period_end: datetime = Field(..., description="Fin période facturation")
    
    # Informations base
    currency: str = Field(default="USD", description="Devise")
    billing_model: BillingModel = Field(..., description="Modèle de facturation")
    
    # Coûts de base
    base_subscription_cost: Decimal = Field(default=0, ge=0, description="Coût abonnement de base")
    user_based_cost: Decimal = Field(default=0, ge=0, description="Coût basé utilisateurs")
    
    # Coûts d'usage
    api_usage_cost: Decimal = Field(default=0, ge=0, description="Coût usage API")
    storage_cost: Decimal = Field(default=0, ge=0, description="Coût stockage")
    compute_cost: Decimal = Field(default=0, ge=0, description="Coût compute")
    data_transfer_cost: Decimal = Field(default=0, ge=0, description="Coût transfert données")
    ml_training_cost: Decimal = Field(default=0, ge=0, description="Coût entraînement ML")
    
    # Coûts support
    support_cost: Decimal = Field(default=0, ge=0, description="Coût support")
    professional_services_cost: Decimal = Field(default=0, ge=0, description="Coût services pro")
    
    # Réductions et crédits
    discounts: List[Dict[str, Any]] = Field(default_factory=list, description="Réductions appliquées")
    credits_applied: Decimal = Field(default=0, ge=0, description="Crédits appliqués")
    promotional_credits: Decimal = Field(default=0, ge=0, description="Crédits promotionnels")
    
    # Taxes
    tax_rate_percent: NonNegativeFloat = Field(default=0.0, le=100.0, description="Taux de taxe")
    tax_amount: Decimal = Field(default=0, ge=0, description="Montant taxes")
    
    # Totaux
    subtotal: Decimal = Field(default=0, ge=0, description="Sous-total")
    total_amount: Decimal = Field(default=0, ge=0, description="Montant total")
    amount_due: Decimal = Field(default=0, ge=0, description="Montant dû")
    
    # Détails de paiement
    payment_method: Optional[str] = Field(None, description="Méthode de paiement")
    payment_status: str = Field(default="pending", description="Statut paiement")
    payment_date: Optional[datetime] = Field(None, description="Date paiement")
    invoice_number: Optional[str] = Field(None, description="Numéro facture")
    
    # Métriques d'usage facturables
    billable_users: int = Field(default=0, ge=0, description="Utilisateurs facturables")
    billable_api_calls: int = Field(default=0, ge=0, description="Appels API facturables")
    billable_storage_gb: NonNegativeFloat = Field(default=0.0, description="Stockage facturable GB")
    billable_compute_hours: NonNegativeFloat = Field(default=0.0, description="Heures compute facturables")
    
    @property
    def effective_rate_per_user(self) -> Decimal:
        """Calcule le coût effectif par utilisateur."""
        if self.billable_users == 0:
            return Decimal('0')
        return self.total_amount / self.billable_users


class TenantAnalytics(BaseModel):
    """Analytics avancées par tenant avec insights business."""
    
    tenant_id: UUID4 = Field(..., description="ID du tenant")
    analysis_date: datetime = Field(default_factory=datetime.utcnow)
    analysis_period_days: PositiveInt = Field(default=30, description="Période d'analyse en jours")
    
    # Métriques de croissance
    user_growth_rate: Optional[float] = Field(None, description="Taux de croissance utilisateurs")
    revenue_growth_rate: Optional[float] = Field(None, description="Taux de croissance revenus")
    usage_growth_rate: Optional[float] = Field(None, description="Taux de croissance usage")
    
    # Métriques de rétention
    user_retention_rate_7d: Optional[float] = Field(None, ge=0.0, le=1.0, description="Rétention 7 jours")
    user_retention_rate_30d: Optional[float] = Field(None, ge=0.0, le=1.0, description="Rétention 30 jours")
    user_retention_rate_90d: Optional[float] = Field(None, ge=0.0, le=1.0, description="Rétention 90 jours")
    
    # Métriques d'engagement
    daily_active_users_avg: NonNegativeFloat = Field(default=0.0, description="DAU moyen")
    session_frequency: NonNegativeFloat = Field(default=0.0, description="Fréquence sessions")
    feature_adoption_rates: Dict[str, float] = Field(default_factory=dict, description="Taux adoption features")
    
    # Métriques de performance
    system_availability_percent: NonNegativeFloat = Field(default=100.0, le=100.0, description="Disponibilité système")
    avg_response_time_ms: NonNegativeFloat = Field(default=0.0, description="Temps réponse moyen")
    error_rate_percent: NonNegativeFloat = Field(default=0.0, le=100.0, description="Taux d'erreur")
    
    # Métriques business
    customer_lifetime_value: Optional[Decimal] = Field(None, description="Valeur vie client")
    monthly_recurring_revenue: Optional[Decimal] = Field(None, description="MRR")
    churn_risk_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Score risque churn")
    
    # Insights et recommandations
    insights: List[Dict[str, Any]] = Field(default_factory=list, description="Insights automatiques")
    recommendations: List[Dict[str, Any]] = Field(default_factory=list, description="Recommandations")
    alerts: List[Dict[str, Any]] = Field(default_factory=list, description="Alertes business")
    
    # Comparaisons
    industry_benchmarks: Dict[str, float] = Field(default_factory=dict, description="Benchmarks secteur")
    peer_comparisons: Dict[str, float] = Field(default_factory=dict, description="Comparaisons pairs")
    
    # Prédictions
    predicted_growth_30d: Optional[float] = Field(None, description="Croissance prédite 30j")
    predicted_churn_probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Probabilité churn")
    recommended_tier: Optional[TenantTier] = Field(None, description="Tier recommandé")


# Export des classes principales
__all__ = [
    "TenantTier",
    "TenantStatus",
    "BillingModel",
    "ComplianceFramework",
    "TenantResourceLimits",
    "TenantFeatures",
    "TenantConfiguration",
    "TenantUsageMetrics",
    "TenantBilling",
    "TenantAnalytics"
]
