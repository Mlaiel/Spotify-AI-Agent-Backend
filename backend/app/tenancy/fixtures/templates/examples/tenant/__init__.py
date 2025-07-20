# -*- coding: utf-8 -*-
"""
Enterprise Tenant Management System
===================================

Module de gestion avancée des tenants multi-niveaux pour Spotify AI Agent.
Système industrialisé de provisioning, configuration et gestion du cycle de vie des tenants.

Architecture:
- Multi-tier tenant system (Free, Professional, Enterprise, Custom)
- Schema-based isolation avec encryption
- Auto-scaling et resource management
- Compliance et audit intégrés
- ML-powered tenant analytics
- Advanced security et threat detection

Auteur: Équipe Platform Engineering
Version: 2024.2.0
"""

import json
import uuid
import datetime
import logging
from typing import Dict, List, Optional, Any, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import asyncio
from jinja2 import Environment, FileSystemLoader, Template
import yaml
from pydantic import BaseModel, validator, Field

# Configuration du logging
logger = logging.getLogger(__name__)

# ==============================================================================
# ENUMS ET TYPES DE BASE
# ==============================================================================

class TenantTier(str, Enum):
    """Niveaux de service tenant avec capacités différentiées."""
    FREE = "free"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"

class TenantStatus(str, Enum):
    """États du cycle de vie tenant."""
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"
    MIGRATING = "migrating"
    MAINTENANCE = "maintenance"

class IsolationLevel(str, Enum):
    """Niveaux d'isolation des données tenant."""
    SHARED = "shared"          # Tables partagées avec tenant_id
    SCHEMA = "schema"          # Schémas dédiés par tenant
    DATABASE = "database"     # Bases de données dédiées
    CLUSTER = "cluster"       # Clusters dédiés

class ComplianceFramework(str, Enum):
    """Frameworks de conformité supportés."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"

# ==============================================================================
# MODÈLES DE DONNÉES AVANCÉS
# ==============================================================================

@dataclass
class TenantLimits:
    """Limites et quotas par tenant avec scaling dynamique."""
    max_users: int = 10
    storage_gb: float = 5.0
    ai_sessions_per_month: int = 100
    api_rate_limit_per_hour: int = 1000
    concurrent_sessions: int = 10
    custom_integrations: int = 5
    data_retention_days: int = 365
    backup_retention_days: int = 90
    
    # Limites avancées
    ml_training_jobs_per_month: int = 10
    real_time_predictions_per_day: int = 10000
    custom_models: int = 2
    webhook_endpoints: int = 10
    scheduled_jobs: int = 50
    
    # Scaling automatique
    auto_scale_enabled: bool = False
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    max_scale_factor: float = 2.0

@dataclass
class SecurityPolicy:
    """Politique de sécurité tenant avec threat detection."""
    password_policy: Dict[str, Any] = field(default_factory=lambda: {
        "min_length": 12,
        "require_special_chars": True,
        "require_numbers": True,
        "require_uppercase": True,
        "require_lowercase": True,
        "max_age_days": 90,
        "history_count": 12,
        "lockout_attempts": 5,
        "lockout_duration_minutes": 30
    })
    
    session_config: Dict[str, Any] = field(default_factory=lambda: {
        "timeout_minutes": 480,
        "absolute_timeout_hours": 24,
        "concurrent_sessions_limit": 5,
        "idle_timeout_minutes": 60,
        "remember_me_days": 30
    })
    
    mfa_config: Dict[str, Any] = field(default_factory=lambda: {
        "required": True,
        "methods": ["totp", "sms", "email", "webauthn"],
        "backup_codes": 10,
        "grace_period_days": 7
    })
    
    encryption: Dict[str, Any] = field(default_factory=lambda: {
        "algorithm": "AES-256-GCM",
        "key_rotation_days": 90,
        "at_rest": True,
        "in_transit": True,
        "field_level": True
    })
    
    threat_detection: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "ml_anomaly_detection": True,
        "geo_blocking": False,
        "suspicious_activity_threshold": 0.7,
        "auto_suspend_on_threat": True,
        "notification_channels": ["email", "slack", "webhook"]
    })

@dataclass  
class AIConfiguration:
    """Configuration IA avancée avec ML personnalisé."""
    model_access: Dict[str, bool] = field(default_factory=lambda: {
        "gpt-3.5-turbo": True,
        "gpt-4": False,
        "claude-3": False,
        "gemini-pro": False,
        "custom_models": False,
        "fine_tuned_models": False
    })
    
    rate_limits: Dict[str, int] = field(default_factory=lambda: {
        "requests_per_minute": 50,
        "tokens_per_day": 100000,
        "concurrent_requests": 5,
        "batch_size_limit": 100
    })
    
    features: Dict[str, bool] = field(default_factory=lambda: {
        "context_memory": True,
        "custom_prompts": True,
        "conversation_export": True,
        "ai_analytics": True,
        "model_fine_tuning": False,
        "embeddings_generation": True,
        "semantic_search": True,
        "auto_categorization": True
    })
    
    safety_settings: Dict[str, Any] = field(default_factory=lambda: {
        "content_filter": True,
        "profanity_filter": True,
        "personal_info_detection": True,
        "bias_detection": True,
        "hallucination_detection": True,
        "max_session_duration": 3600,
        "safety_threshold": 0.8
    })
    
    ml_pipeline: Dict[str, Any] = field(default_factory=lambda: {
        "auto_ml_enabled": False,
        "model_monitoring": True,
        "drift_detection": True,
        "a_b_testing": False,
        "model_versioning": True,
        "performance_tracking": True
    })

# ==============================================================================
# GESTIONNAIRE PRINCIPAL DES TENANTS
# ==============================================================================

class TenantManager:
    """
    Gestionnaire principal pour la gestion avancée des tenants.
    
    Fonctionnalités:
    - Provisioning automatisé multi-tier
    - Isolation et sécurisation
    - Monitoring et analytics
    - Compliance et audit
    - Auto-scaling et optimisation
    """
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else Path(__file__).parent
        self.template_env = Environment(
            loader=FileSystemLoader(str(self.base_path)),
            enable_async=True
        )
        self._load_tier_configurations()
        
    def _load_tier_configurations(self) -> None:
        """Charge les configurations par tier depuis les fichiers."""
        self.tier_configs = {}
        
        for tier in TenantTier:
            config_file = self.base_path / f"{tier.value}_init.json"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.tier_configs[tier] = json.load(f)
    
    async def create_tenant(
        self,
        tenant_id: str,
        tier: TenantTier,
        tenant_name: str = None,
        custom_config: Dict[str, Any] = None,
        compliance_frameworks: List[ComplianceFramework] = None
    ) -> Dict[str, Any]:
        """
        Crée un nouveau tenant avec configuration complète.
        
        Args:
            tenant_id: Identifiant unique du tenant
            tier: Niveau de service
            tenant_name: Nom display du tenant
            custom_config: Configuration personnalisée
            compliance_frameworks: Frameworks de conformité requis
            
        Returns:
            Configuration complète du tenant créé
        """
        logger.info(f"Création tenant {tenant_id} niveau {tier.value}")
        
        # Configuration de base par tier
        base_config = self.tier_configs.get(tier, {})
        
        # Génération configuration complète
        tenant_config = await self._generate_tenant_config(
            tenant_id=tenant_id,
            tier=tier,
            tenant_name=tenant_name or tenant_id.title(),
            base_config=base_config,
            custom_config=custom_config or {},
            compliance_frameworks=compliance_frameworks or []
        )
        
        # Provisioning infrastructure
        await self._provision_tenant_infrastructure(tenant_config)
        
        # Configuration sécurité
        await self._setup_tenant_security(tenant_config)
        
        # Initialisation monitoring
        await self._setup_tenant_monitoring(tenant_config)
        
        # Audit trail
        await self._create_audit_entry(
            tenant_id=tenant_id,
            action="tenant_created",
            details={"tier": tier.value, "config": tenant_config}
        )
        
        logger.info(f"Tenant {tenant_id} créé avec succès")
        return tenant_config
    
    async def _generate_tenant_config(
        self,
        tenant_id: str,
        tier: TenantTier,
        tenant_name: str,
        base_config: Dict[str, Any],
        custom_config: Dict[str, Any],
        compliance_frameworks: List[ComplianceFramework]
    ) -> Dict[str, Any]:
        """Génère la configuration complète du tenant."""
        
        # Configuration des limites par tier
        limits_config = self._get_tier_limits(tier)
        
        # Configuration sécurité avancée
        security_config = self._get_security_config(tier, compliance_frameworks)
        
        # Configuration IA
        ai_config = self._get_ai_config(tier)
        
        # Merge configurations
        config = {
            "_metadata": {
                "template_type": "tenant_advanced",
                "template_version": "2024.2.0",
                "schema_version": "2024.2",
                "created_at": datetime.datetime.utcnow().isoformat(),
                "generator": "TenantManagerAdvanced",
                "tags": ["tenant", "enterprise", "multi-tier", "compliance"],
                "tier": tier.value,
                "compliance_frameworks": [f.value for f in compliance_frameworks]
            },
            
            "tenant_id": tenant_id,
            "tenant_name": tenant_name,
            "tier": tier.value,
            "status": TenantStatus.PENDING.value,
            
            "configuration": {
                "limits": limits_config,
                "security": security_config,
                "ai_configuration": ai_config,
                "features": self._get_features_config(tier),
                "integrations": self._get_integrations_config(tier),
                "compliance": self._get_compliance_config(compliance_frameworks)
            },
            
            "infrastructure": {
                "isolation_level": self._get_isolation_level(tier),
                "database": self._get_database_config(tenant_id, tier),
                "storage": self._get_storage_config(tenant_id, tier),
                "networking": self._get_networking_config(tenant_id, tier),
                "compute": self._get_compute_config(tier)
            },
            
            "monitoring": {
                "metrics": self._get_metrics_config(tier),
                "logging": self._get_logging_config(tier),
                "alerting": self._get_alerting_config(tier),
                "analytics": self._get_analytics_config(tier)
            },
            
            "billing": self._get_billing_config(tier),
            "lifecycle": self._get_lifecycle_config(tier),
            
            "created_at": datetime.datetime.utcnow().isoformat(),
            "updated_at": datetime.datetime.utcnow().isoformat(),
            "expires_at": None
        }
        
        # Merge avec custom config
        config = self._deep_merge(config, custom_config)
        
        return config
    
    def _get_tier_limits(self, tier: TenantTier) -> Dict[str, Any]:
        """Retourne les limites configurées par tier."""
        limits_map = {
            TenantTier.FREE: TenantLimits(
                max_users=5,
                storage_gb=1.0,
                ai_sessions_per_month=50,
                api_rate_limit_per_hour=100,
                concurrent_sessions=2
            ),
            TenantTier.PROFESSIONAL: TenantLimits(
                max_users=100,
                storage_gb=50.0,
                ai_sessions_per_month=1000,
                api_rate_limit_per_hour=1000,
                concurrent_sessions=10,
                auto_scale_enabled=True
            ),
            TenantTier.ENTERPRISE: TenantLimits(
                max_users=1000,
                storage_gb=500.0,
                ai_sessions_per_month=10000,
                api_rate_limit_per_hour=10000,
                concurrent_sessions=50,
                ml_training_jobs_per_month=100,
                custom_models=10,
                auto_scale_enabled=True,
                max_scale_factor=5.0
            ),
            TenantTier.CUSTOM: TenantLimits(
                max_users=999999,
                storage_gb=999999.0,
                ai_sessions_per_month=999999,
                api_rate_limit_per_hour=999999,
                concurrent_sessions=999,
                auto_scale_enabled=True,
                max_scale_factor=10.0
            )
        }
        
        limits = limits_map[tier]
        return {
            "max_users": limits.max_users,
            "storage_gb": limits.storage_gb,
            "ai_sessions_per_month": limits.ai_sessions_per_month,
            "api_rate_limit_per_hour": limits.api_rate_limit_per_hour,
            "concurrent_sessions": limits.concurrent_sessions,
            "custom_integrations": limits.custom_integrations,
            "ml_training_jobs_per_month": limits.ml_training_jobs_per_month,
            "real_time_predictions_per_day": limits.real_time_predictions_per_day,
            "custom_models": limits.custom_models,
            "auto_scaling": {
                "enabled": limits.auto_scale_enabled,
                "scale_up_threshold": limits.scale_up_threshold,
                "scale_down_threshold": limits.scale_down_threshold,
                "max_scale_factor": limits.max_scale_factor
            }
        }
    
    def _get_security_config(self, tier: TenantTier, compliance_frameworks: List[ComplianceFramework]) -> Dict[str, Any]:
        """Génère la configuration sécurité basée sur le tier et compliance."""
        base_security = SecurityPolicy()
        
        # Ajustements par tier
        if tier in [TenantTier.ENTERPRISE, TenantTier.CUSTOM]:
            base_security.password_policy["min_length"] = 16
            base_security.mfa_config["required"] = True
            base_security.threat_detection["ml_anomaly_detection"] = True
            base_security.encryption["field_level"] = True
            
        # Ajustements pour compliance
        if ComplianceFramework.HIPAA in compliance_frameworks:
            base_security.encryption["algorithm"] = "AES-256-GCM"
            base_security.password_policy["max_age_days"] = 60
            base_security.session_config["timeout_minutes"] = 240
            
        if ComplianceFramework.PCI_DSS in compliance_frameworks:
            base_security.password_policy["min_length"] = 16
            base_security.mfa_config["required"] = True
            base_security.threat_detection["auto_suspend_on_threat"] = True
        
        return {
            "password_policy": base_security.password_policy,
            "session_config": base_security.session_config,
            "mfa_config": base_security.mfa_config,
            "encryption": base_security.encryption,
            "threat_detection": base_security.threat_detection,
            "access_control": {
                "default_policy": "deny",
                "rbac_enabled": True,
                "abac_enabled": tier in [TenantTier.ENTERPRISE, TenantTier.CUSTOM],
                "audit_enabled": True,
                "session_recording": tier in [TenantTier.ENTERPRISE, TenantTier.CUSTOM]
            }
        }
    
    def _get_ai_config(self, tier: TenantTier) -> Dict[str, Any]:
        """Configuration IA par tier."""
        base_ai = AIConfiguration()
        
        if tier == TenantTier.FREE:
            base_ai.model_access = {"gpt-3.5-turbo": True}
            base_ai.rate_limits["requests_per_minute"] = 10
            base_ai.rate_limits["tokens_per_day"] = 10000
            
        elif tier == TenantTier.PROFESSIONAL:
            base_ai.model_access.update({"gpt-4": True, "embeddings_generation": True})
            base_ai.features["model_fine_tuning"] = False
            
        elif tier in [TenantTier.ENTERPRISE, TenantTier.CUSTOM]:
            base_ai.model_access.update({
                "gpt-4": True,
                "claude-3": True,
                "gemini-pro": True,
                "custom_models": True,
                "fine_tuned_models": True
            })
            base_ai.features.update({
                "model_fine_tuning": True,
                "a_b_testing": True,
                "auto_ml_enabled": True
            })
            base_ai.ml_pipeline["auto_ml_enabled"] = True
        
        return {
            "model_access": base_ai.model_access,
            "rate_limits": base_ai.rate_limits,
            "features": base_ai.features,
            "safety_settings": base_ai.safety_settings,
            "ml_pipeline": base_ai.ml_pipeline
        }
    
    def _get_features_config(self, tier: TenantTier) -> Dict[str, List[str]]:
        """Configuration des features par tier."""
        features_map = {
            TenantTier.FREE: {
                "enabled": [
                    "basic_ai",
                    "standard_collaboration",
                    "community_support",
                    "basic_analytics"
                ],
                "disabled": [
                    "advanced_ai",
                    "custom_integrations",
                    "priority_support",
                    "advanced_analytics",
                    "white_labeling",
                    "custom_branding"
                ]
            },
            TenantTier.PROFESSIONAL: {
                "enabled": [
                    "advanced_ai",
                    "full_collaboration",
                    "priority_support",
                    "analytics",
                    "custom_integrations",
                    "api_access",
                    "webhook_notifications",
                    "sso_integration"
                ],
                "disabled": [
                    "white_labeling",
                    "custom_branding",
                    "dedicated_instance",
                    "custom_models"
                ]
            },
            TenantTier.ENTERPRISE: {
                "enabled": [
                    "enterprise_ai",
                    "advanced_collaboration",
                    "dedicated_support",
                    "advanced_analytics",
                    "unlimited_integrations",
                    "full_api_access",
                    "webhook_notifications",
                    "sso_integration",
                    "white_labeling",
                    "custom_branding",
                    "audit_logs",
                    "compliance_reporting",
                    "custom_models",
                    "ml_pipeline"
                ],
                "disabled": []
            },
            TenantTier.CUSTOM: {
                "enabled": [
                    "all_features",
                    "custom_development",
                    "dedicated_infrastructure",
                    "24_7_support",
                    "custom_sla",
                    "on_premise_option"
                ],
                "disabled": []
            }
        }
        
        return features_map[tier]
    
    def _get_database_config(self, tenant_id: str, tier: TenantTier) -> Dict[str, Any]:
        """Configuration base de données par tier."""
        isolation_map = {
            TenantTier.FREE: IsolationLevel.SHARED,
            TenantTier.PROFESSIONAL: IsolationLevel.SCHEMA,
            TenantTier.ENTERPRISE: IsolationLevel.DATABASE,
            TenantTier.CUSTOM: IsolationLevel.CLUSTER
        }
        
        return {
            "isolation_level": isolation_map[tier].value,
            "schema_name": f"tenant_{tenant_id}" if isolation_map[tier] != IsolationLevel.SHARED else "public",
            "database_name": f"tenant_{tenant_id}_db" if isolation_map[tier] == IsolationLevel.DATABASE else "shared_db",
            "cluster_name": f"tenant_{tenant_id}_cluster" if isolation_map[tier] == IsolationLevel.CLUSTER else "shared_cluster",
            "encryption_at_rest": True,
            "backup_enabled": True,
            "backup_retention_days": 90 if tier in [TenantTier.ENTERPRISE, TenantTier.CUSTOM] else 30,
            "point_in_time_recovery": tier in [TenantTier.ENTERPRISE, TenantTier.CUSTOM],
            "replication": {
                "enabled": tier in [TenantTier.ENTERPRISE, TenantTier.CUSTOM],
                "type": "async",
                "replicas": 2 if tier == TenantTier.ENTERPRISE else 3 if tier == TenantTier.CUSTOM else 0
            },
            "performance": {
                "connection_pooling": True,
                "query_optimization": True,
                "index_tuning": tier in [TenantTier.ENTERPRISE, TenantTier.CUSTOM],
                "cache_enabled": True,
                "cache_size_mb": 1024 if tier == TenantTier.ENTERPRISE else 2048 if tier == TenantTier.CUSTOM else 256
            }
        }
    
    async def _provision_tenant_infrastructure(self, tenant_config: Dict[str, Any]) -> None:
        """Provisioning de l'infrastructure tenant."""
        tenant_id = tenant_config["tenant_id"]
        tier = TenantTier(tenant_config["tier"])
        
        logger.info(f"Provisioning infrastructure pour tenant {tenant_id}")
        
        # Provisioning base de données
        await self._provision_database(tenant_config)
        
        # Provisioning stockage
        await self._provision_storage(tenant_config)
        
        # Provisioning réseau
        await self._provision_networking(tenant_config)
        
        # Provisioning compute
        await self._provision_compute(tenant_config)
    
    async def _provision_database(self, tenant_config: Dict[str, Any]) -> None:
        """Provisioning base de données avec isolation."""
        db_config = tenant_config["infrastructure"]["database"]
        isolation_level = IsolationLevel(db_config["isolation_level"])
        
        if isolation_level == IsolationLevel.SCHEMA:
            # Création schéma dédié
            await self._create_tenant_schema(
                schema_name=db_config["schema_name"],
                tenant_config=tenant_config
            )
        elif isolation_level == IsolationLevel.DATABASE:
            # Création base dédiée
            await self._create_tenant_database(
                database_name=db_config["database_name"],
                tenant_config=tenant_config
            )
        elif isolation_level == IsolationLevel.CLUSTER:
            # Provisioning cluster dédié
            await self._create_tenant_cluster(
                cluster_name=db_config["cluster_name"],
                tenant_config=tenant_config
            )
    
    async def _create_tenant_schema(self, schema_name: str, tenant_config: Dict[str, Any]) -> None:
        """Crée un schéma dédié pour le tenant."""
        logger.info(f"Création schéma {schema_name}")
        # Implémentation création schéma PostgreSQL
        # CREATE SCHEMA IF NOT EXISTS {schema_name}
        # GRANT USAGE ON SCHEMA {schema_name} TO tenant_user_{tenant_id}
        pass
    
    async def _create_tenant_database(self, database_name: str, tenant_config: Dict[str, Any]) -> None:
        """Crée une base de données dédiée."""
        logger.info(f"Création base de données {database_name}")
        # Implémentation création base dédiée
        pass
    
    async def _create_tenant_cluster(self, cluster_name: str, tenant_config: Dict[str, Any]) -> None:
        """Provisioning cluster dédié."""
        logger.info(f"Provisioning cluster {cluster_name}")
        # Implémentation provisioning cluster
        pass
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge profond de dictionnaires."""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    async def _create_audit_entry(self, tenant_id: str, action: str, details: Dict[str, Any]) -> None:
        """Crée une entrée d'audit."""
        audit_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "tenant_id": tenant_id,
            "action": action,
            "details": details,
            "user_id": "system",
            "ip_address": "127.0.0.1",
            "user_agent": "TenantManager/2024.2.0"
        }
        
        logger.info(f"Audit: {action} pour tenant {tenant_id}")
        # Sauvegarde en base d'audit

# ==============================================================================
# FACTORY POUR TEMPLATES TENANT
# ==============================================================================

class TenantTemplateFactory:
    """Factory pour génération de templates tenant personnalisés."""
    
    def __init__(self, manager: TenantManager):
        self.manager = manager
    
    def create_custom_template(
        self,
        base_tier: TenantTier,
        customizations: Dict[str, Any],
        template_name: str
    ) -> Dict[str, Any]:
        """Crée un template personnalisé basé sur un tier existant."""
        
        base_config = self.manager.tier_configs.get(base_tier, {})
        custom_template = self.manager._deep_merge(base_config, customizations)
        
        # Métadonnées du template personnalisé
        custom_template["_metadata"] = {
            "template_type": "tenant_custom",
            "template_name": template_name,
            "base_tier": base_tier.value,
            "created_at": datetime.datetime.utcnow().isoformat(),
            "customizations": list(customizations.keys())
        }
        
        return custom_template
    
    def generate_migration_template(
        self,
        from_tier: TenantTier,
        to_tier: TenantTier,
        tenant_id: str
    ) -> Dict[str, Any]:
        """Génère un template de migration entre tiers."""
        
        return {
            "_metadata": {
                "template_type": "tenant_migration",
                "migration_id": str(uuid.uuid4()),
                "tenant_id": tenant_id,
                "from_tier": from_tier.value,
                "to_tier": to_tier.value,
                "created_at": datetime.datetime.utcnow().isoformat()
            },
            
            "migration_plan": {
                "phases": [
                    {
                        "phase": "preparation",
                        "steps": [
                            "backup_current_data",
                            "validate_target_tier",
                            "prepare_new_infrastructure"
                        ]
                    },
                    {
                        "phase": "migration",
                        "steps": [
                            "suspend_tenant_operations",
                            "migrate_data",
                            "update_configurations",
                            "validate_migration"
                        ]
                    },
                    {
                        "phase": "activation",
                        "steps": [
                            "activate_new_tier",
                            "resume_operations",
                            "notify_users",
                            "cleanup_old_resources"
                        ]
                    }
                ]
            },
            
            "rollback_plan": {
                "enabled": True,
                "backup_retention_hours": 72,
                "rollback_steps": [
                    "restore_from_backup",
                    "revert_configurations",
                    "reactivate_old_tier"
                ]
            }
        }

# ==============================================================================
# UTILITAIRES ET HELPERS
# ==============================================================================

def get_tenant_template_path(tier: TenantTier) -> Path:
    """Retourne le chemin du template pour un tier donné."""
    return Path(__file__).parent / f"{tier.value}_init.json"

def validate_tenant_config(config: Dict[str, Any]) -> List[str]:
    """Valide une configuration tenant et retourne les erreurs."""
    errors = []
    
    required_fields = ["tenant_id", "tier", "configuration"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Champ requis manquant: {field}")
    
    # Validation tier
    if "tier" in config:
        try:
            TenantTier(config["tier"])
        except ValueError:
            errors.append(f"Tier invalide: {config['tier']}")
    
    # Validation limites
    if "configuration" in config and "limits" in config["configuration"]:
        limits = config["configuration"]["limits"]
        if "max_users" in limits and limits["max_users"] <= 0:
            errors.append("max_users doit être positif")
    
    return errors

async def provision_tenant_async(
    tenant_id: str,
    tier: TenantTier,
    manager: TenantManager = None
) -> Dict[str, Any]:
    """Provisioning asynchrone d'un tenant."""
    if manager is None:
        manager = TenantManager()
    
    return await manager.create_tenant(tenant_id=tenant_id, tier=tier)

# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "TenantTier",
    "TenantStatus", 
    "IsolationLevel",
    "ComplianceFramework",
    "TenantLimits",
    "SecurityPolicy",
    "AIConfiguration",
    "TenantManager",
    "TenantTemplateFactory",
    "get_tenant_template_path",
    "validate_tenant_config",
    "provision_tenant_async"
]
