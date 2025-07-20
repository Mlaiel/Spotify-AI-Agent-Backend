# =============================================================================
# Multi-Tenant Support - Architecture Enterprise
# =============================================================================
# 
# Module de support multi-tenant complet avec isolation des données,
# gestion des quotas, facturation, et administration centralisée.
# Architecture enterprise avec sécurité avancée et scalabilité.
#
# Auteur: Microservices Architect + Security Specialist
# Direction Technique: Fahed Mlaiel
# Version: 2.0.0 Enterprise
# =============================================================================

"""
Multi-Tenant Support Enterprise

Ce module fournit un système complet de multi-tenancy avec:

Fonctionnalités Principales:
- Isolation complète des données par tenant
- Gestion avancée des quotas et limitations
- Facturation automatisée par usage
- Administration centralisée multi-tenant
- Sécurité renforcée avec chiffrement par tenant

Composants:
- TenantManager: Gestion centralisée des tenants
- QuotaManager: Gestion des quotas et limitations
- BillingManager: Système de facturation automatisé
- SecurityManager: Sécurité et isolation des données
- AdminManager: Administration multi-tenant

Modèles d'Isolation:
- Shared Database, Shared Schema (avec tenant_id)
- Shared Database, Separate Schema
- Separate Database per Tenant
- Hybrid avec données partagées et privées
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import hashlib
import json
from decimal import Decimal
import re

# Imports pour sécurité et chiffrement
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import secrets

# Imports du Core Engine
from . import (
    core_registry, TenantContext, TenantTier, EngineStatus,
    DEFAULT_WORKFLOW_TIMEOUT
)

logger = logging.getLogger(__name__)

# ===========================
# Configuration & Constants
# ===========================

# Modèles d'isolation
ISOLATION_MODELS = {
    "shared_db_shared_schema": "Shared Database, Shared Schema",
    "shared_db_separate_schema": "Shared Database, Separate Schema", 
    "separate_database": "Separate Database per Tenant",
    "hybrid": "Hybrid Model"
}

# Configuration par défaut
DEFAULT_TENANT_CONFIG = {
    "isolation_model": "shared_db_shared_schema",
    "data_retention_days": 365,
    "backup_enabled": True,
    "monitoring_enabled": True,
    "analytics_enabled": True,
    "custom_branding": False,
    "api_rate_limit": 1000,  # requêtes par heure
    "storage_limit_gb": 10,
    "user_limit": 25,
    "incident_limit_per_hour": 500,
    "workflow_limit": 50
}

# Tarification par tier
TIER_PRICING = {
    TenantTier.BASIC: {
        "monthly_base": Decimal("29.99"),
        "per_incident": Decimal("0.01"),
        "per_user": Decimal("5.00"),
        "per_gb_storage": Decimal("2.00"),
        "per_workflow": Decimal("0.50")
    },
    TenantTier.STANDARD: {
        "monthly_base": Decimal("99.99"),
        "per_incident": Decimal("0.005"),
        "per_user": Decimal("10.00"),
        "per_gb_storage": Decimal("1.50"),
        "per_workflow": Decimal("0.30")
    },
    TenantTier.PREMIUM: {
        "monthly_base": Decimal("299.99"),
        "per_incident": Decimal("0.002"),
        "per_user": Decimal("15.00"),
        "per_gb_storage": Decimal("1.00"),
        "per_workflow": Decimal("0.20")
    },
    TenantTier.ENTERPRISE: {
        "monthly_base": Decimal("999.99"),
        "per_incident": Decimal("0.001"),
        "per_user": Decimal("20.00"),
        "per_gb_storage": Decimal("0.50"),
        "per_workflow": Decimal("0.10"),
        "custom_pricing": True
    }
}

# ===========================
# Enums & Types
# ===========================

class IsolationModel(Enum):
    """Modèles d'isolation des données"""
    SHARED_DB_SHARED_SCHEMA = "shared_db_shared_schema"
    SHARED_DB_SEPARATE_SCHEMA = "shared_db_separate_schema"
    SEPARATE_DATABASE = "separate_database"
    HYBRID = "hybrid"

class TenantStatus(Enum):
    """Statuts de tenant"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    PENDING = "pending"
    CANCELLED = "cancelled"
    MIGRATING = "migrating"

class QuotaType(Enum):
    """Types de quotas"""
    INCIDENTS_PER_HOUR = "incidents_per_hour"
    STORAGE_GB = "storage_gb"
    USERS = "users"
    WORKFLOWS = "workflows"
    API_REQUESTS = "api_requests"
    DATA_RETENTION_DAYS = "data_retention_days"

class BillingCycle(Enum):
    """Cycles de facturation"""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    USAGE_BASED = "usage_based"

class UsageType(Enum):
    """Types d'usage pour la facturation"""
    INCIDENT_PROCESSED = "incident_processed"
    STORAGE_USED = "storage_used"
    USER_ACTIVE = "user_active"
    WORKFLOW_EXECUTED = "workflow_executed"
    API_REQUEST = "api_request"

# ===========================
# Modèles de Données
# ===========================

@dataclass
class TenantConfiguration:
    """Configuration complète d'un tenant"""
    tenant_id: str
    isolation_model: IsolationModel = IsolationModel.SHARED_DB_SHARED_SCHEMA
    database_config: Dict[str, str] = field(default_factory=dict)
    schema_name: Optional[str] = None
    encryption_key: Optional[str] = None
    
    # Configuration fonctionnelle
    data_retention_days: int = 365
    backup_enabled: bool = True
    monitoring_enabled: bool = True
    analytics_enabled: bool = True
    custom_branding: bool = False
    
    # Limites et quotas
    quotas: Dict[QuotaType, int] = field(default_factory=dict)
    
    # Configuration de sécurité
    ip_whitelist: List[str] = field(default_factory=list)
    require_mfa: bool = False
    session_timeout_minutes: int = 480  # 8 heures
    password_policy: Dict[str, Any] = field(default_factory=dict)
    
    # Intégrations
    integrations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.quotas:
            self.quotas = self._get_default_quotas()
        if not self.encryption_key:
            self.encryption_key = self._generate_encryption_key()
        if not self.password_policy:
            self.password_policy = self._get_default_password_policy()
    
    def _get_default_quotas(self) -> Dict[QuotaType, int]:
        """Quotas par défaut"""
        return {
            QuotaType.INCIDENTS_PER_HOUR: 500,
            QuotaType.STORAGE_GB: 10,
            QuotaType.USERS: 25,
            QuotaType.WORKFLOWS: 50,
            QuotaType.API_REQUESTS: 1000,
            QuotaType.DATA_RETENTION_DAYS: 365
        }
    
    def _generate_encryption_key(self) -> str:
        """Génère une clé de chiffrement unique pour le tenant"""
        key = Fernet.generate_key()
        return base64.urlsafe_b64encode(key).decode()
    
    def _get_default_password_policy(self) -> Dict[str, Any]:
        """Politique de mot de passe par défaut"""
        return {
            "min_length": 8,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_numbers": True,
            "require_special_chars": True,
            "max_age_days": 90,
            "prevent_reuse": 5
        }

@dataclass
class TenantUsage:
    """Données d'usage d'un tenant"""
    tenant_id: str
    period_start: datetime
    period_end: datetime
    
    # Métriques d'usage
    incidents_processed: int = 0
    storage_used_gb: float = 0.0
    active_users: int = 0
    workflows_executed: int = 0
    api_requests: int = 0
    
    # Coûts calculés
    calculated_cost: Decimal = Decimal("0.00")
    
    # Détails par type
    usage_details: Dict[UsageType, List[Dict[str, Any]]] = field(default_factory=dict)
    
    def add_usage(self, usage_type: UsageType, amount: Union[int, float], 
                  metadata: Dict[str, Any] = None):
        """Ajoute des données d'usage"""
        if usage_type not in self.usage_details:
            self.usage_details[usage_type] = []
        
        entry = {
            "amount": amount,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        self.usage_details[usage_type].append(entry)
        
        # Mise à jour des compteurs principaux
        if usage_type == UsageType.INCIDENT_PROCESSED:
            self.incidents_processed += int(amount)
        elif usage_type == UsageType.STORAGE_USED:
            self.storage_used_gb = max(self.storage_used_gb, float(amount))
        elif usage_type == UsageType.USER_ACTIVE:
            self.active_users = max(self.active_users, int(amount))
        elif usage_type == UsageType.WORKFLOW_EXECUTED:
            self.workflows_executed += int(amount)
        elif usage_type == UsageType.API_REQUEST:
            self.api_requests += int(amount)

@dataclass
class TenantBilling:
    """Information de facturation d'un tenant"""
    tenant_id: str
    billing_cycle: BillingCycle = BillingCycle.MONTHLY
    tier: TenantTier = TenantTier.BASIC
    
    # Informations de paiement
    billing_contact: Dict[str, str] = field(default_factory=dict)
    payment_method: Dict[str, str] = field(default_factory=dict)
    
    # Facturation
    next_billing_date: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(days=30))
    last_invoice_date: Optional[datetime] = None
    current_balance: Decimal = Decimal("0.00")
    credit_limit: Decimal = Decimal("1000.00")
    
    # Historique
    invoices: List[Dict[str, Any]] = field(default_factory=list)
    payment_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def is_payment_overdue(self) -> bool:
        """Vérifie si le paiement est en retard"""
        return datetime.utcnow() > self.next_billing_date and self.current_balance > 0

@dataclass
class QuotaViolation:
    """Violation de quota"""
    tenant_id: str
    quota_type: QuotaType
    current_usage: Union[int, float]
    quota_limit: Union[int, float]
    violation_percentage: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    
    def __post_init__(self):
        if self.quota_limit > 0:
            self.violation_percentage = (self.current_usage / self.quota_limit) * 100

# ===========================
# Gestionnaire de Quotas
# ===========================

class QuotaManager:
    """Gestionnaire des quotas et limitations"""
    
    def __init__(self):
        self.current_usage: Dict[str, Dict[QuotaType, Union[int, float]]] = {}
        self.violations: List[QuotaViolation] = []
        self.quota_callbacks: Dict[QuotaType, List[Callable]] = {}
        
        logger.info("Gestionnaire de quotas initialisé")
    
    def set_tenant_quota(self, tenant_id: str, quota_type: QuotaType, limit: Union[int, float]):
        """Définit un quota pour un tenant"""
        tenant_context = core_registry.get_tenant(tenant_id)
        if tenant_context:
            tenant_context.quotas[quota_type.value] = limit
            logger.info(f"Quota {quota_type.value} défini à {limit} pour le tenant {tenant_id}")
    
    def get_tenant_quota(self, tenant_id: str, quota_type: QuotaType) -> Union[int, float]:
        """Récupère le quota d'un tenant"""
        tenant_context = core_registry.get_tenant(tenant_id)
        if tenant_context and quota_type.value in tenant_context.quotas:
            return tenant_context.quotas[quota_type.value]
        return DEFAULT_TENANT_CONFIG.get(quota_type.value, 0)
    
    def get_current_usage(self, tenant_id: str, quota_type: QuotaType) -> Union[int, float]:
        """Récupère l'usage actuel d'un tenant"""
        if tenant_id not in self.current_usage:
            self.current_usage[tenant_id] = {}
        return self.current_usage[tenant_id].get(quota_type, 0)
    
    def increment_usage(self, tenant_id: str, quota_type: QuotaType, 
                       amount: Union[int, float] = 1) -> bool:
        """Incrémente l'usage et vérifie les quotas"""
        if tenant_id not in self.current_usage:
            self.current_usage[tenant_id] = {}
        
        current = self.current_usage[tenant_id].get(quota_type, 0)
        new_usage = current + amount
        quota_limit = self.get_tenant_quota(tenant_id, quota_type)
        
        # Vérification du quota (-1 = illimité)
        if quota_limit != -1 and new_usage > quota_limit:
            # Violation de quota
            violation = QuotaViolation(
                tenant_id=tenant_id,
                quota_type=quota_type,
                current_usage=new_usage,
                quota_limit=quota_limit
            )
            self.violations.append(violation)
            
            # Appel des callbacks de violation
            self._notify_quota_violation(violation)
            
            logger.warning(f"Quota violation pour {tenant_id}: {quota_type.value} "
                         f"({new_usage}/{quota_limit})")
            return False
        
        # Mise à jour de l'usage
        self.current_usage[tenant_id][quota_type] = new_usage
        
        # Vérification des seuils d'alerte (80% et 95%)
        if quota_limit != -1:
            usage_percentage = (new_usage / quota_limit) * 100
            if usage_percentage >= 95:
                self._notify_quota_warning(tenant_id, quota_type, usage_percentage, "critical")
            elif usage_percentage >= 80:
                self._notify_quota_warning(tenant_id, quota_type, usage_percentage, "warning")
        
        return True
    
    def check_quota_before_action(self, tenant_id: str, quota_type: QuotaType, 
                                 amount: Union[int, float] = 1) -> bool:
        """Vérifie si une action respecterait les quotas sans l'exécuter"""
        current = self.get_current_usage(tenant_id, quota_type)
        quota_limit = self.get_tenant_quota(tenant_id, quota_type)
        
        if quota_limit == -1:  # Illimité
            return True
        
        return (current + amount) <= quota_limit
    
    def reset_usage(self, tenant_id: str, quota_type: Optional[QuotaType] = None):
        """Remet à zéro l'usage (pour cycles de facturation)"""
        if tenant_id not in self.current_usage:
            return
        
        if quota_type:
            self.current_usage[tenant_id][quota_type] = 0
        else:
            self.current_usage[tenant_id] = {}
        
        logger.info(f"Usage remis à zéro pour {tenant_id}" +
                   (f" ({quota_type.value})" if quota_type else ""))
    
    def get_quota_status(self, tenant_id: str) -> Dict[str, Any]:
        """Récupère le statut complet des quotas d'un tenant"""
        status = {
            "tenant_id": tenant_id,
            "quotas": {},
            "violations": [],
            "warnings": []
        }
        
        for quota_type in QuotaType:
            current = self.get_current_usage(tenant_id, quota_type)
            limit = self.get_tenant_quota(tenant_id, quota_type)
            
            quota_info = {
                "current": current,
                "limit": limit,
                "percentage": (current / limit * 100) if limit > 0 else 0,
                "unlimited": limit == -1
            }
            
            status["quotas"][quota_type.value] = quota_info
            
            # Classification des alertes
            if limit != -1 and current > limit:
                status["violations"].append(quota_type.value)
            elif limit != -1 and (current / limit) >= 0.8:
                status["warnings"].append(quota_type.value)
        
        return status
    
    def register_quota_callback(self, quota_type: QuotaType, callback: Callable):
        """Enregistre un callback pour les violations de quota"""
        if quota_type not in self.quota_callbacks:
            self.quota_callbacks[quota_type] = []
        self.quota_callbacks[quota_type].append(callback)
    
    def _notify_quota_violation(self, violation: QuotaViolation):
        """Notifie les violations de quota"""
        callbacks = self.quota_callbacks.get(violation.quota_type, [])
        for callback in callbacks:
            try:
                asyncio.create_task(callback(violation))
            except Exception as e:
                logger.error(f"Erreur lors du callback de violation de quota: {e}")
    
    def _notify_quota_warning(self, tenant_id: str, quota_type: QuotaType, 
                            percentage: float, level: str):
        """Notifie les avertissements de quota"""
        logger.info(f"Quota warning ({level}) pour {tenant_id}: "
                   f"{quota_type.value} à {percentage:.1f}%")

# ===========================
# Gestionnaire de Facturation
# ===========================

class BillingManager:
    """Gestionnaire de facturation automatisée"""
    
    def __init__(self):
        self.tenant_billing: Dict[str, TenantBilling] = {}
        self.usage_tracking: Dict[str, TenantUsage] = {}
        
        logger.info("Gestionnaire de facturation initialisé")
    
    def setup_tenant_billing(self, tenant_id: str, tier: TenantTier, 
                           billing_cycle: BillingCycle = BillingCycle.MONTHLY) -> TenantBilling:
        """Configure la facturation pour un tenant"""
        billing = TenantBilling(
            tenant_id=tenant_id,
            tier=tier,
            billing_cycle=billing_cycle
        )
        
        self.tenant_billing[tenant_id] = billing
        
        # Initialise le tracking d'usage
        self.usage_tracking[tenant_id] = TenantUsage(
            tenant_id=tenant_id,
            period_start=datetime.utcnow(),
            period_end=self._calculate_period_end(billing_cycle)
        )
        
        logger.info(f"Facturation configurée pour {tenant_id} (tier: {tier.value})")
        return billing
    
    def _calculate_period_end(self, billing_cycle: BillingCycle) -> datetime:
        """Calcule la fin de période selon le cycle"""
        now = datetime.utcnow()
        
        if billing_cycle == BillingCycle.MONTHLY:
            return now + timedelta(days=30)
        elif billing_cycle == BillingCycle.QUARTERLY:
            return now + timedelta(days=90)
        elif billing_cycle == BillingCycle.YEARLY:
            return now + timedelta(days=365)
        else:  # USAGE_BASED
            return now + timedelta(days=1)  # Facturation quotidienne
    
    def track_usage(self, tenant_id: str, usage_type: UsageType, 
                   amount: Union[int, float], metadata: Dict[str, Any] = None):
        """Enregistre l'usage pour facturation"""
        if tenant_id not in self.usage_tracking:
            # Auto-initialisation si pas encore configuré
            self.usage_tracking[tenant_id] = TenantUsage(
                tenant_id=tenant_id,
                period_start=datetime.utcnow(),
                period_end=datetime.utcnow() + timedelta(days=30)
            )
        
        usage = self.usage_tracking[tenant_id]
        usage.add_usage(usage_type, amount, metadata)
        
        logger.debug(f"Usage tracké pour {tenant_id}: {usage_type.value} +{amount}")
    
    def calculate_invoice(self, tenant_id: str) -> Dict[str, Any]:
        """Calcule la facture pour un tenant"""
        billing = self.tenant_billing.get(tenant_id)
        usage = self.usage_tracking.get(tenant_id)
        
        if not billing or not usage:
            logger.error(f"Configuration de facturation manquante pour {tenant_id}")
            return {}
        
        tier_pricing = TIER_PRICING[billing.tier]
        invoice = {
            "tenant_id": tenant_id,
            "period_start": usage.period_start.isoformat(),
            "period_end": usage.period_end.isoformat(),
            "tier": billing.tier.value,
            "billing_cycle": billing.billing_cycle.value,
            "line_items": [],
            "subtotal": Decimal("0.00"),
            "taxes": Decimal("0.00"),
            "total": Decimal("0.00"),
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Coût de base mensuel
        base_cost = tier_pricing["monthly_base"]
        if billing.billing_cycle == BillingCycle.QUARTERLY:
            base_cost *= 3 * Decimal("0.95")  # 5% de réduction
        elif billing.billing_cycle == BillingCycle.YEARLY:
            base_cost *= 12 * Decimal("0.90")  # 10% de réduction
        
        invoice["line_items"].append({
            "description": f"Plan {billing.tier.value} - Base",
            "quantity": 1,
            "unit_price": float(base_cost),
            "total": float(base_cost)
        })
        invoice["subtotal"] += base_cost
        
        # Coûts d'usage
        if usage.incidents_processed > 0:
            incident_cost = Decimal(str(usage.incidents_processed)) * tier_pricing["per_incident"]
            invoice["line_items"].append({
                "description": "Incidents traités",
                "quantity": usage.incidents_processed,
                "unit_price": float(tier_pricing["per_incident"]),
                "total": float(incident_cost)
            })
            invoice["subtotal"] += incident_cost
        
        if usage.storage_used_gb > 0:
            storage_cost = Decimal(str(usage.storage_used_gb)) * tier_pricing["per_gb_storage"]
            invoice["line_items"].append({
                "description": "Stockage utilisé (GB)",
                "quantity": float(usage.storage_used_gb),
                "unit_price": float(tier_pricing["per_gb_storage"]),
                "total": float(storage_cost)
            })
            invoice["subtotal"] += storage_cost
        
        if usage.active_users > 0:
            user_cost = Decimal(str(usage.active_users)) * tier_pricing["per_user"]
            invoice["line_items"].append({
                "description": "Utilisateurs actifs",
                "quantity": usage.active_users,
                "unit_price": float(tier_pricing["per_user"]),
                "total": float(user_cost)
            })
            invoice["subtotal"] += user_cost
        
        if usage.workflows_executed > 0:
            workflow_cost = Decimal(str(usage.workflows_executed)) * tier_pricing["per_workflow"]
            invoice["line_items"].append({
                "description": "Workflows exécutés",
                "quantity": usage.workflows_executed,
                "unit_price": float(tier_pricing["per_workflow"]),
                "total": float(workflow_cost)
            })
            invoice["subtotal"] += workflow_cost
        
        # Calcul des taxes (exemple: 20% TVA)
        tax_rate = Decimal("0.20")
        invoice["taxes"] = invoice["subtotal"] * tax_rate
        invoice["total"] = invoice["subtotal"] + invoice["taxes"]
        
        # Sauvegarde de l'usage calculé
        usage.calculated_cost = invoice["total"]
        
        return invoice
    
    def generate_invoice(self, tenant_id: str) -> Dict[str, Any]:
        """Génère et enregistre une facture"""
        invoice = self.calculate_invoice(tenant_id)
        
        if not invoice:
            return {}
        
        # Génération d'un ID de facture
        invoice_id = f"INV-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        invoice["invoice_id"] = invoice_id
        
        # Mise à jour de la facturation
        billing = self.tenant_billing[tenant_id]
        billing.invoices.append(invoice)
        billing.current_balance += Decimal(str(invoice["total"]))
        billing.last_invoice_date = datetime.utcnow()
        
        # Calcul de la prochaine date de facturation
        if billing.billing_cycle == BillingCycle.MONTHLY:
            billing.next_billing_date = datetime.utcnow() + timedelta(days=30)
        elif billing.billing_cycle == BillingCycle.QUARTERLY:
            billing.next_billing_date = datetime.utcnow() + timedelta(days=90)
        elif billing.billing_cycle == BillingCycle.YEARLY:
            billing.next_billing_date = datetime.utcnow() + timedelta(days=365)
        
        logger.info(f"Facture générée pour {tenant_id}: {invoice_id} - {invoice['total']}€")
        
        # Reset de la période d'usage
        self._reset_usage_period(tenant_id)
        
        return invoice
    
    def _reset_usage_period(self, tenant_id: str):
        """Remet à zéro la période d'usage après facturation"""
        billing = self.tenant_billing[tenant_id]
        
        self.usage_tracking[tenant_id] = TenantUsage(
            tenant_id=tenant_id,
            period_start=datetime.utcnow(),
            period_end=self._calculate_period_end(billing.billing_cycle)
        )
    
    def process_payment(self, tenant_id: str, amount: Decimal, 
                       payment_method: str = "card") -> Dict[str, Any]:
        """Traite un paiement (simulation)"""
        billing = self.tenant_billing.get(tenant_id)
        if not billing:
            return {"success": False, "error": "Configuration de facturation non trouvée"}
        
        # Simulation du traitement de paiement
        payment_id = f"PAY-{uuid.uuid4().hex[:12].upper()}"
        
        payment_record = {
            "payment_id": payment_id,
            "amount": float(amount),
            "method": payment_method,
            "status": "completed",
            "processed_at": datetime.utcnow().isoformat()
        }
        
        # Mise à jour du solde
        billing.current_balance -= amount
        billing.payment_history.append(payment_record)
        
        logger.info(f"Paiement traité pour {tenant_id}: {amount}€ (ID: {payment_id})")
        
        return {
            "success": True,
            "payment_id": payment_id,
            "new_balance": float(billing.current_balance)
        }
    
    def get_billing_summary(self, tenant_id: str) -> Dict[str, Any]:
        """Récupère un résumé de facturation"""
        billing = self.tenant_billing.get(tenant_id)
        usage = self.usage_tracking.get(tenant_id)
        
        if not billing:
            return {}
        
        return {
            "tenant_id": tenant_id,
            "tier": billing.tier.value,
            "billing_cycle": billing.billing_cycle.value,
            "current_balance": float(billing.current_balance),
            "next_billing_date": billing.next_billing_date.isoformat(),
            "payment_overdue": billing.is_payment_overdue(),
            "current_usage": {
                "incidents_processed": usage.incidents_processed if usage else 0,
                "storage_used_gb": usage.storage_used_gb if usage else 0,
                "active_users": usage.active_users if usage else 0,
                "workflows_executed": usage.workflows_executed if usage else 0
            },
            "total_invoices": len(billing.invoices),
            "total_payments": len(billing.payment_history)
        }

# ===========================
# Gestionnaire de Sécurité Multi-Tenant
# ===========================

class TenantSecurityManager:
    """Gestionnaire de sécurité pour l'environnement multi-tenant"""
    
    def __init__(self):
        self.tenant_encryption_keys: Dict[str, Fernet] = {}
        self.access_logs: List[Dict[str, Any]] = []
        
        logger.info("Gestionnaire de sécurité multi-tenant initialisé")
    
    def setup_tenant_encryption(self, tenant_id: str, encryption_key: Optional[str] = None) -> str:
        """Configure le chiffrement pour un tenant"""
        if not encryption_key:
            encryption_key = Fernet.generate_key().decode()
        
        fernet = Fernet(encryption_key.encode())
        self.tenant_encryption_keys[tenant_id] = fernet
        
        logger.info(f"Chiffrement configuré pour le tenant {tenant_id}")
        return encryption_key
    
    def encrypt_tenant_data(self, tenant_id: str, data: str) -> str:
        """Chiffre des données pour un tenant spécifique"""
        if tenant_id not in self.tenant_encryption_keys:
            raise ValueError(f"Clé de chiffrement non configurée pour le tenant {tenant_id}")
        
        fernet = self.tenant_encryption_keys[tenant_id]
        encrypted_data = fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt_tenant_data(self, tenant_id: str, encrypted_data: str) -> str:
        """Déchiffre des données pour un tenant spécifique"""
        if tenant_id not in self.tenant_encryption_keys:
            raise ValueError(f"Clé de chiffrement non configurée pour le tenant {tenant_id}")
        
        fernet = self.tenant_encryption_keys[tenant_id]
        decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted_data = fernet.decrypt(decoded_data)
        return decrypted_data.decode()
    
    def validate_tenant_access(self, tenant_id: str, user_id: str, 
                             resource: str, action: str) -> bool:
        """Valide l'accès d'un utilisateur à une ressource"""
        tenant_context = core_registry.get_tenant(tenant_id)
        if not tenant_context or not tenant_context.is_active:
            self._log_access_violation(tenant_id, user_id, resource, action, "inactive_tenant")
            return False
        
        # Vérification des permissions
        required_permission = f"{resource}.{action}"
        if required_permission not in tenant_context.permissions:
            self._log_access_violation(tenant_id, user_id, resource, action, "insufficient_permissions")
            return False
        
        # Log d'accès autorisé
        self._log_access(tenant_id, user_id, resource, action, True)
        return True
    
    def _log_access(self, tenant_id: str, user_id: str, resource: str, 
                   action: str, granted: bool):
        """Enregistre un accès dans les logs"""
        log_entry = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "granted": granted,
            "timestamp": datetime.utcnow().isoformat(),
            "ip_address": "127.0.0.1"  # À récupérer du contexte de la requête
        }
        
        self.access_logs.append(log_entry)
        
        # Limitation de la taille des logs en mémoire
        if len(self.access_logs) > 10000:
            self.access_logs = self.access_logs[-5000:]
    
    def _log_access_violation(self, tenant_id: str, user_id: str, resource: str, 
                            action: str, reason: str):
        """Enregistre une violation d'accès"""
        self._log_access(tenant_id, user_id, resource, action, False)
        logger.warning(f"Violation d'accès: {tenant_id}/{user_id} -> {resource}.{action} ({reason})")
    
    def get_tenant_access_logs(self, tenant_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Récupère les logs d'accès d'un tenant"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            log for log in self.access_logs
            if (log["tenant_id"] == tenant_id and 
                datetime.fromisoformat(log["timestamp"]) > cutoff_time)
        ]

# ===========================
# Gestionnaire Principal Multi-Tenant
# ===========================

class MultiTenantManager:
    """Gestionnaire principal du système multi-tenant"""
    
    def __init__(self):
        self.quota_manager = QuotaManager()
        self.billing_manager = BillingManager()
        self.security_manager = TenantSecurityManager()
        self.tenant_configurations: Dict[str, TenantConfiguration] = {}
        
        # Setup des callbacks de quota
        self._setup_quota_callbacks()
        
        logger.info("Gestionnaire multi-tenant principal initialisé")
    
    def _setup_quota_callbacks(self):
        """Configure les callbacks de quota"""
        async def handle_quota_violation(violation: QuotaViolation):
            logger.warning(f"Quota violation callback: {violation.tenant_id} - {violation.quota_type.value}")
            
            # Notifier l'administrateur du tenant
            # Dans un vrai système, ceci enverrait une notification
            
            # Suspendre le tenant si violation critique
            if violation.violation_percentage > 150:  # 50% au-dessus de la limite
                await self.suspend_tenant(violation.tenant_id, f"Quota violation: {violation.quota_type.value}")
        
        for quota_type in QuotaType:
            self.quota_manager.register_quota_callback(quota_type, handle_quota_violation)
    
    async def create_tenant(self, tenant_data: Dict[str, Any]) -> TenantContext:
        """Crée un nouveau tenant"""
        tenant_id = tenant_data.get("tenant_id", str(uuid.uuid4()))
        tenant_name = tenant_data.get("name", f"Tenant {tenant_id}")
        tier = TenantTier(tenant_data.get("tier", "basic"))
        
        # Création du contexte tenant
        tenant_context = TenantContext(
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            tier=tier
        )
        
        # Configuration détaillée
        config = TenantConfiguration(
            tenant_id=tenant_id,
            isolation_model=IsolationModel(tenant_data.get("isolation_model", "shared_db_shared_schema"))
        )
        
        # Configuration spécifique au tier
        if tier == TenantTier.ENTERPRISE:
            config.custom_branding = True
            config.require_mfa = True
            config.data_retention_days = 2555  # 7 ans
        
        self.tenant_configurations[tenant_id] = config
        
        # Enregistrement dans le registre central
        core_registry.register_tenant(tenant_context)
        
        # Configuration de la sécurité
        encryption_key = self.security_manager.setup_tenant_encryption(tenant_id)
        config.encryption_key = encryption_key
        
        # Configuration de la facturation
        billing_cycle = BillingCycle(tenant_data.get("billing_cycle", "monthly"))
        self.billing_manager.setup_tenant_billing(tenant_id, tier, billing_cycle)
        
        # Configuration des quotas selon le tier
        self._setup_tenant_quotas(tenant_id, tier)
        
        logger.info(f"Tenant créé: {tenant_name} ({tenant_id}) - Tier: {tier.value}")
        
        return tenant_context
    
    def _setup_tenant_quotas(self, tenant_id: str, tier: TenantTier):
        """Configure les quotas selon le tier"""
        quotas = {
            TenantTier.BASIC: {
                QuotaType.INCIDENTS_PER_HOUR: 100,
                QuotaType.STORAGE_GB: 1,
                QuotaType.USERS: 5,
                QuotaType.WORKFLOWS: 10,
                QuotaType.API_REQUESTS: 1000
            },
            TenantTier.STANDARD: {
                QuotaType.INCIDENTS_PER_HOUR: 500,
                QuotaType.STORAGE_GB: 10,
                QuotaType.USERS: 25,
                QuotaType.WORKFLOWS: 50,
                QuotaType.API_REQUESTS: 5000
            },
            TenantTier.PREMIUM: {
                QuotaType.INCIDENTS_PER_HOUR: 2000,
                QuotaType.STORAGE_GB: 100,
                QuotaType.USERS: 100,
                QuotaType.WORKFLOWS: 200,
                QuotaType.API_REQUESTS: 20000
            },
            TenantTier.ENTERPRISE: {
                QuotaType.INCIDENTS_PER_HOUR: -1,  # Illimité
                QuotaType.STORAGE_GB: -1,
                QuotaType.USERS: -1,
                QuotaType.WORKFLOWS: -1,
                QuotaType.API_REQUESTS: -1
            }
        }
        
        tier_quotas = quotas.get(tier, quotas[TenantTier.BASIC])
        for quota_type, limit in tier_quotas.items():
            self.quota_manager.set_tenant_quota(tenant_id, quota_type, limit)
    
    async def update_tenant(self, tenant_id: str, updates: Dict[str, Any]) -> bool:
        """Met à jour un tenant"""
        tenant_context = core_registry.get_tenant(tenant_id)
        if not tenant_context:
            return False
        
        # Mise à jour du contexte
        for key, value in updates.items():
            if key == "tier" and isinstance(value, str):
                new_tier = TenantTier(value)
                tenant_context.tier = new_tier
                self._setup_tenant_quotas(tenant_id, new_tier)
            elif hasattr(tenant_context, key):
                setattr(tenant_context, key, value)
        
        # Mise à jour de la configuration
        config = self.tenant_configurations.get(tenant_id)
        if config:
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        tenant_context.last_activity = datetime.utcnow()
        logger.info(f"Tenant {tenant_id} mis à jour")
        return True
    
    async def suspend_tenant(self, tenant_id: str, reason: str = "") -> bool:
        """Suspend un tenant"""
        tenant_context = core_registry.get_tenant(tenant_id)
        if not tenant_context:
            return False
        
        tenant_context.is_active = False
        logger.warning(f"Tenant {tenant_id} suspendu: {reason}")
        
        # Notification (dans un vrai système)
        return True
    
    async def activate_tenant(self, tenant_id: str) -> bool:
        """Active un tenant"""
        tenant_context = core_registry.get_tenant(tenant_id)
        if not tenant_context:
            return False
        
        tenant_context.is_active = True
        tenant_context.last_activity = datetime.utcnow()
        logger.info(f"Tenant {tenant_id} activé")
        return True
    
    def get_tenant_info(self, tenant_id: str) -> Dict[str, Any]:
        """Récupère les informations complètes d'un tenant"""
        tenant_context = core_registry.get_tenant(tenant_id)
        config = self.tenant_configurations.get(tenant_id)
        
        if not tenant_context:
            return {}
        
        return {
            "tenant_id": tenant_id,
            "name": tenant_context.tenant_name,
            "tier": tenant_context.tier.value,
            "status": "active" if tenant_context.is_active else "inactive",
            "created_at": tenant_context.created_at.isoformat(),
            "last_activity": tenant_context.last_activity.isoformat(),
            "quotas": self.quota_manager.get_quota_status(tenant_id),
            "billing": self.billing_manager.get_billing_summary(tenant_id),
            "configuration": {
                "isolation_model": config.isolation_model.value if config else "unknown",
                "data_retention_days": config.data_retention_days if config else 365,
                "monitoring_enabled": config.monitoring_enabled if config else True
            }
        }
    
    def list_tenants(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Liste tous les tenants"""
        tenants = []
        
        for tenant_context in core_registry.list_tenants():
            if not active_only or tenant_context.is_active:
                tenants.append(self.get_tenant_info(tenant_context.tenant_id))
        
        return tenants
    
    async def migrate_tenant(self, tenant_id: str, target_isolation_model: IsolationModel) -> bool:
        """Migre un tenant vers un nouveau modèle d'isolation"""
        config = self.tenant_configurations.get(tenant_id)
        if not config:
            return False
        
        logger.info(f"Migration du tenant {tenant_id} vers {target_isolation_model.value}")
        
        # Simulation de migration
        await asyncio.sleep(5)  # Simulation du temps de migration
        
        config.isolation_model = target_isolation_model
        logger.info(f"Migration du tenant {tenant_id} terminée")
        
        return True
    
    async def process_tenant_action(self, tenant_id: str, action: str, 
                                  quota_type: QuotaType, amount: Union[int, float] = 1) -> bool:
        """Traite une action tenant avec vérification de quota et facturation"""
        # Vérification du quota
        if not self.quota_manager.check_quota_before_action(tenant_id, quota_type, amount):
            logger.warning(f"Action refusée pour {tenant_id}: quota {quota_type.value} dépassé")
            return False
        
        # Incrémentation de l'usage
        if not self.quota_manager.increment_usage(tenant_id, quota_type, amount):
            logger.warning(f"Action échouée pour {tenant_id}: violation de quota")
            return False
        
        # Tracking pour facturation
        usage_mapping = {
            QuotaType.INCIDENTS_PER_HOUR: UsageType.INCIDENT_PROCESSED,
            QuotaType.API_REQUESTS: UsageType.API_REQUEST,
            QuotaType.WORKFLOWS: UsageType.WORKFLOW_EXECUTED,
            QuotaType.STORAGE_GB: UsageType.STORAGE_USED,
            QuotaType.USERS: UsageType.USER_ACTIVE
        }
        
        if quota_type in usage_mapping:
            self.billing_manager.track_usage(tenant_id, usage_mapping[quota_type], amount)
        
        return True

# ===========================
# Exports
# ===========================

__all__ = [
    "MultiTenantManager",
    "QuotaManager",
    "BillingManager", 
    "TenantSecurityManager",
    "TenantConfiguration",
    "TenantUsage",
    "TenantBilling",
    "QuotaViolation",
    "IsolationModel",
    "TenantStatus",
    "QuotaType",
    "BillingCycle",
    "UsageType"
]

logger.info("Module Multi-Tenant Support chargé")
