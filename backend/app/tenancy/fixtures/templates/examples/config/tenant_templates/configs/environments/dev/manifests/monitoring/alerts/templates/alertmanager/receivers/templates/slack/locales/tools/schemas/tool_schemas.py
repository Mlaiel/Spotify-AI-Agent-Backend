"""
Schémas Pydantic pour les outils de configuration et d'automatisation.

Ce module complète la suite de schémas avec les outils spécialisés,
les utilitaires de configuration et les helpers d'automatisation.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
from .validation_schemas import ToolConfigSchema, ParameterType


class ConfigurationScope(str, Enum):
    """Portée de configuration."""
    GLOBAL = "global"
    TENANT = "tenant"
    ENVIRONMENT = "environment"
    SERVICE = "service"
    COMPONENT = "component"


class ConfigurationFormat(str, Enum):
    """Formats de configuration supportés."""
    YAML = "yaml"
    JSON = "json"
    TOML = "toml"
    INI = "ini"
    XML = "xml"
    PROPERTIES = "properties"
    ENV = "env"


class TemplateEngine(str, Enum):
    """Moteurs de template supportés."""
    JINJA2 = "jinja2"
    MUSTACHE = "mustache"
    HANDLEBARS = "handlebars"
    GO_TEMPLATE = "go_template"
    LIQUID = "liquid"


class ConfigGeneratorSchema(BaseModel):
    """Générateur de configuration."""
    name: str = Field(..., description="Nom du générateur")
    description: str = Field(..., description="Description")
    scope: ConfigurationScope = Field(..., description="Portée de la configuration")
    output_format: ConfigurationFormat = Field(..., description="Format de sortie")
    template_engine: TemplateEngine = Field(TemplateEngine.JINJA2, description="Moteur de template")
    
    template_path: str = Field(..., description="Chemin du template")
    output_path: str = Field(..., description="Chemin de sortie")
    
    variables: Dict[str, Any] = Field(default_factory=dict, description="Variables du template")
    defaults: Dict[str, Any] = Field(default_factory=dict, description="Valeurs par défaut")
    required_variables: List[str] = Field(default_factory=list, description="Variables requises")
    
    validation_schema: Optional[str] = Field(None, description="Schéma de validation")
    post_generation_hooks: List[str] = Field(default_factory=list, description="Hooks post-génération")
    
    class Config:
        use_enum_values = True


class ConfigMergerSchema(BaseModel):
    """Outil de fusion de configurations."""
    name: str = Field(..., description="Nom du merger")
    strategy: str = Field("deep_merge", description="Stratégie de fusion")
    
    input_configs: List[str] = Field(..., description="Configurations d'entrée")
    output_config: str = Field(..., description="Configuration de sortie")
    
    merge_rules: Dict[str, str] = Field(
        default_factory=lambda: {
            "arrays": "append",  # append, replace, merge
            "objects": "merge",  # merge, replace
            "scalars": "override"  # override, keep_first
        },
        description="Règles de fusion"
    )
    
    conflict_resolution: str = Field("manual", description="Résolution de conflits")
    preserve_comments: bool = Field(True, description="Préserver les commentaires")
    validate_output: bool = Field(True, description="Valider la sortie")


class ConfigValidatorToolSchema(BaseModel):
    """Outil de validation de configuration."""
    name: str = Field(..., description="Nom du validateur")
    config_type: str = Field(..., description="Type de configuration")
    
    schema_files: List[str] = Field(..., description="Fichiers de schéma")
    config_files: List[str] = Field(..., description="Fichiers de configuration")
    
    validation_rules: List[str] = Field(default_factory=list, description="Règles de validation")
    custom_validators: List[str] = Field(default_factory=list, description="Validateurs personnalisés")
    
    strict_mode: bool = Field(False, description="Mode strict")
    fail_on_warning: bool = Field(False, description="Échouer sur avertissement")
    
    output_format: str = Field("json", description="Format de sortie du rapport")
    detailed_report: bool = Field(True, description="Rapport détaillé")


class SecretManagementSchema(BaseModel):
    """Gestion des secrets."""
    name: str = Field(..., description="Nom du gestionnaire de secrets")
    provider: str = Field(..., description="Fournisseur (vault, k8s, aws, etc.)")
    
    connection_config: Dict[str, Any] = Field(..., description="Configuration de connexion")
    
    encryption: Dict[str, str] = Field(
        default_factory=lambda: {
            "algorithm": "AES-256-GCM",
            "key_derivation": "PBKDF2",
            "iterations": 100000
        },
        description="Configuration de chiffrement"
    )
    
    access_policies: List[Dict[str, Any]] = Field(default_factory=list, description="Politiques d'accès")
    audit_logging: bool = Field(True, description="Journalisation d'audit")
    
    rotation_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "schedule": "0 2 * * 0",  # Hebdomadaire
            "retention_count": 5
        },
        description="Configuration de rotation"
    )


class BackupToolSchema(BaseModel):
    """Outil de sauvegarde."""
    name: str = Field(..., description="Nom de l'outil")
    backup_type: str = Field(..., description="Type de sauvegarde")
    
    source_paths: List[str] = Field(..., description="Chemins source")
    destination: str = Field(..., description="Destination de sauvegarde")
    
    schedule: str = Field(..., description="Planning de sauvegarde (cron)")
    retention_policy: Dict[str, str] = Field(
        default_factory=lambda: {
            "daily": "7d",
            "weekly": "4w",
            "monthly": "12m"
        },
        description="Politique de rétention"
    )
    
    compression: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "algorithm": "gzip",
            "level": 6
        },
        description="Configuration de compression"
    )
    
    encryption: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "algorithm": "AES-256",
            "key_rotation": True
        },
        description="Configuration de chiffrement"
    )
    
    verification: Dict[str, bool] = Field(
        default_factory=lambda: {
            "checksum_validation": True,
            "restore_test": False,
            "integrity_check": True
        },
        description="Configuration de vérification"
    )


class RestoreToolSchema(BaseModel):
    """Outil de restauration."""
    name: str = Field(..., description="Nom de l'outil")
    backup_source: str = Field(..., description="Source de sauvegarde")
    
    restore_target: str = Field(..., description="Cible de restauration")
    restore_point: Optional[datetime] = Field(None, description="Point de restauration")
    
    verification_steps: List[str] = Field(default_factory=list, description="Étapes de vérification")
    rollback_plan: Dict[str, Any] = Field(default_factory=dict, description="Plan de rollback")
    
    dry_run_enabled: bool = Field(True, description="Test à sec activé")
    incremental_restore: bool = Field(False, description="Restauration incrémentale")


class HealthCheckToolSchema(BaseModel):
    """Outil de health check."""
    name: str = Field(..., description="Nom de l'outil")
    check_type: str = Field(..., description="Type de vérification")
    
    targets: List[str] = Field(..., description="Cibles à vérifier")
    checks: List[Dict[str, Any]] = Field(..., description="Vérifications à effectuer")
    
    interval: str = Field("30s", description="Intervalle de vérification")
    timeout: str = Field("10s", description="Timeout de vérification")
    retries: int = Field(3, description="Nombre de tentatives")
    
    thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "response_time": 1000.0,  # ms
            "success_rate": 95.0,     # %
            "availability": 99.0      # %
        },
        description="Seuils de santé"
    )
    
    alerting: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "channels": ["slack", "email"],
            "escalation_delay": "5m"
        },
        description="Configuration d'alerte"
    )


class LoadTestingToolSchema(BaseModel):
    """Outil de test de charge."""
    name: str = Field(..., description="Nom de l'outil")
    test_type: str = Field(..., description="Type de test")
    
    target_url: str = Field(..., description="URL cible")
    test_scenarios: List[Dict[str, Any]] = Field(..., description="Scénarios de test")
    
    load_profile: Dict[str, Any] = Field(
        default_factory=lambda: {
            "users": 100,
            "ramp_up": "30s",
            "duration": "5m",
            "ramp_down": "30s"
        },
        description="Profil de charge"
    )
    
    performance_criteria: Dict[str, float] = Field(
        default_factory=lambda: {
            "max_response_time": 2000.0,
            "min_throughput": 100.0,
            "max_error_rate": 5.0
        },
        description="Critères de performance"
    )
    
    monitoring: Dict[str, bool] = Field(
        default_factory=lambda: {
            "real_time_metrics": True,
            "detailed_reports": True,
            "resource_monitoring": True
        },
        description="Configuration de monitoring"
    )


class LogAnalysisToolSchema(BaseModel):
    """Outil d'analyse de logs."""
    name: str = Field(..., description="Nom de l'outil")
    log_sources: List[str] = Field(..., description="Sources de logs")
    
    parsing_rules: List[Dict[str, str]] = Field(..., description="Règles de parsing")
    filters: List[Dict[str, Any]] = Field(default_factory=list, description="Filtres d'analyse")
    
    analysis_types: List[str] = Field(
        default_factory=lambda: ["error_detection", "pattern_analysis", "anomaly_detection"],
        description="Types d'analyse"
    )
    
    alerting_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Règles d'alerte")
    
    output_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "format": "json",
            "destination": "stdout",
            "include_raw_logs": False
        },
        description="Configuration de sortie"
    )


class MetricsCollectorSchema(BaseModel):
    """Collecteur de métriques."""
    name: str = Field(..., description="Nom du collecteur")
    collector_type: str = Field(..., description="Type de collecteur")
    
    targets: List[str] = Field(..., description="Cibles de collecte")
    metrics: List[str] = Field(..., description="Métriques à collecter")
    
    collection_interval: str = Field("30s", description="Intervalle de collecte")
    retention_period: str = Field("15d", description="Période de rétention")
    
    aggregation_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Règles d'agrégation")
    
    storage_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "backend": "prometheus",
            "compression": True,
            "sharding": False
        },
        description="Configuration de stockage"
    )


class AlertingToolSchema(BaseModel):
    """Outil d'alerting."""
    name: str = Field(..., description="Nom de l'outil")
    alert_manager: str = Field(..., description="Gestionnaire d'alertes")
    
    rules: List[str] = Field(..., description="Fichiers de règles")
    receivers: List[str] = Field(..., description="Destinataires")
    
    routing_config: Dict[str, Any] = Field(..., description="Configuration de routage")
    inhibition_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Règles d'inhibition")
    
    notification_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "group_wait": "10s",
            "group_interval": "10s",
            "repeat_interval": "1h"
        },
        description="Configuration de notification"
    )


class DashboardToolSchema(BaseModel):
    """Outil de dashboard."""
    name: str = Field(..., description="Nom de l'outil")
    dashboard_type: str = Field(..., description="Type de dashboard")
    
    data_sources: List[str] = Field(..., description="Sources de données")
    panels: List[Dict[str, Any]] = Field(..., description="Panneaux du dashboard")
    
    layout_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "rows": 4,
            "columns": 3,
            "responsive": True
        },
        description="Configuration de layout"
    )
    
    refresh_config: Dict[str, str] = Field(
        default_factory=lambda: {
            "interval": "30s",
            "auto_refresh": "true"
        },
        description="Configuration de rafraîchissement"
    )
    
    access_control: Dict[str, Any] = Field(
        default_factory=lambda: {
            "public": False,
            "allowed_users": [],
            "allowed_roles": []
        },
        description="Contrôle d'accès"
    )


class ReportingToolSchema(BaseModel):
    """Outil de reporting."""
    name: str = Field(..., description="Nom de l'outil")
    report_type: str = Field(..., description="Type de rapport")
    
    data_sources: List[str] = Field(..., description="Sources de données")
    queries: List[Dict[str, Any]] = Field(..., description="Requêtes de données")
    
    schedule: str = Field(..., description="Planning de génération")
    output_formats: List[str] = Field(
        default_factory=lambda: ["pdf", "html", "csv"],
        description="Formats de sortie"
    )
    
    template_config: Dict[str, str] = Field(
        default_factory=lambda: {
            "template_engine": "jinja2",
            "template_path": "/templates/reports/",
            "style_sheet": "default.css"
        },
        description="Configuration de template"
    )
    
    distribution: Dict[str, Any] = Field(
        default_factory=lambda: {
            "email_enabled": True,
            "recipients": [],
            "subject_template": "Weekly Report - {date}"
        },
        description="Configuration de distribution"
    )


class PerformanceOptimizerSchema(BaseModel):
    """Optimiseur de performance."""
    name: str = Field(..., description="Nom de l'optimiseur")
    optimization_type: str = Field(..., description="Type d'optimisation")
    
    targets: List[str] = Field(..., description="Cibles d'optimisation")
    metrics: List[str] = Field(..., description="Métriques à optimiser")
    
    algorithms: List[str] = Field(
        default_factory=lambda: ["auto_scaling", "resource_tuning", "cache_optimization"],
        description="Algorithmes d'optimisation"
    )
    
    constraints: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_cpu": "80%",
            "max_memory": "85%",
            "min_replicas": 2,
            "max_replicas": 10
        },
        description="Contraintes d'optimisation"
    )
    
    automation: Dict[str, bool] = Field(
        default_factory=lambda: {
            "auto_apply": False,
            "require_approval": True,
            "rollback_on_failure": True
        },
        description="Configuration d'automatisation"
    )


class MaintenanceToolSchema(BaseModel):
    """Outil de maintenance."""
    name: str = Field(..., description="Nom de l'outil")
    maintenance_type: str = Field(..., description="Type de maintenance")
    
    schedule: str = Field(..., description="Planning de maintenance")
    duration: str = Field(..., description="Durée estimée")
    
    tasks: List[Dict[str, Any]] = Field(..., description="Tâches de maintenance")
    dependencies: List[str] = Field(default_factory=list, description="Dépendances")
    
    notification_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "advance_notice": "24h",
            "reminder_notice": "1h",
            "channels": ["slack", "email"]
        },
        description="Configuration de notification"
    )
    
    rollback_plan: Dict[str, Any] = Field(default_factory=dict, description="Plan de rollback")
    verification_steps: List[str] = Field(default_factory=list, description="Étapes de vérification")


# Schéma composite pour tous les outils
class ToolSchemaRegistry(BaseModel):
    """Registre de tous les schémas d'outils."""
    config_generators: List[ConfigGeneratorSchema] = Field(default_factory=list)
    config_mergers: List[ConfigMergerSchema] = Field(default_factory=list)
    config_validators: List[ConfigValidatorToolSchema] = Field(default_factory=list)
    secret_managers: List[SecretManagementSchema] = Field(default_factory=list)
    backup_tools: List[BackupToolSchema] = Field(default_factory=list)
    restore_tools: List[RestoreToolSchema] = Field(default_factory=list)
    health_check_tools: List[HealthCheckToolSchema] = Field(default_factory=list)
    load_testing_tools: List[LoadTestingToolSchema] = Field(default_factory=list)
    log_analysis_tools: List[LogAnalysisToolSchema] = Field(default_factory=list)
    metrics_collectors: List[MetricsCollectorSchema] = Field(default_factory=list)
    alerting_tools: List[AlertingToolSchema] = Field(default_factory=list)
    dashboard_tools: List[DashboardToolSchema] = Field(default_factory=list)
    reporting_tools: List[ReportingToolSchema] = Field(default_factory=list)
    performance_optimizers: List[PerformanceOptimizerSchema] = Field(default_factory=list)
    maintenance_tools: List[MaintenanceToolSchema] = Field(default_factory=list)
