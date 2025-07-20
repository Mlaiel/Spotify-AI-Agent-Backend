"""
Schémas Pydantic avancés pour la validation et les outils.

Ce module définit tous les schémas pour la validation de données,
les outils d'automatisation et les utilitaires de configuration.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Callable
from pydantic import BaseModel, Field, validator
import re


class ValidationType(str, Enum):
    """Types de validation."""
    SCHEMA = "schema"
    DATA = "data"
    CONFIG = "config"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"
    BUSINESS = "business"


class ValidationSeverity(str, Enum):
    """Niveaux de sévérité de validation."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ToolCategory(str, Enum):
    """Catégories d'outils."""
    AUTOMATION = "automation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTENANCE = "maintenance"
    BACKUP = "backup"
    TESTING = "testing"


class ToolExecutionMode(str, Enum):
    """Modes d'exécution des outils."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    SCHEDULED = "scheduled"
    EVENT_DRIVEN = "event_driven"
    MANUAL = "manual"


class ParameterType(str, Enum):
    """Types de paramètres."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"
    FILE = "file"
    SECRET = "secret"


class ValidationErrorSchema(BaseModel):
    """Schéma pour les erreurs de validation."""
    field: str = Field(..., description="Champ en erreur")
    message: str = Field(..., description="Message d'erreur")
    severity: ValidationSeverity = Field(..., description="Sévérité de l'erreur")
    code: Optional[str] = Field(None, description="Code d'erreur")
    details: Dict[str, Any] = Field(default_factory=dict, description="Détails de l'erreur")
    suggestion: Optional[str] = Field(None, description="Suggestion de correction")
    documentation_url: Optional[str] = Field(None, description="URL de documentation")
    
    class Config:
        use_enum_values = True


class ValidationContextSchema(BaseModel):
    """Schéma pour le contexte de validation."""
    tenant_id: Optional[str] = Field(None, description="ID du tenant")
    environment: str = Field("dev", description="Environnement")
    validation_type: ValidationType = Field(..., description="Type de validation")
    strict_mode: bool = Field(False, description="Mode strict")
    ignore_warnings: bool = Field(False, description="Ignorer les avertissements")
    custom_rules: List[str] = Field(default_factory=list, description="Règles personnalisées")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées contextuelles")
    
    class Config:
        use_enum_values = True


class ValidationResultSchema(BaseModel):
    """Schéma pour les résultats de validation."""
    is_valid: bool = Field(..., description="Validation réussie")
    errors: List[ValidationErrorSchema] = Field(default_factory=list, description="Erreurs trouvées")
    warnings: List[ValidationErrorSchema] = Field(default_factory=list, description="Avertissements")
    info: List[ValidationErrorSchema] = Field(default_factory=list, description="Informations")
    summary: Dict[str, int] = Field(default_factory=dict, description="Résumé des résultats")
    validation_time: float = Field(..., description="Temps de validation en secondes")
    context: ValidationContextSchema = Field(..., description="Contexte de validation")
    recommendations: List[str] = Field(default_factory=list, description="Recommandations")
    
    @validator('summary')
    def calculate_summary(cls, v, values):
        """Calcule le résumé automatiquement."""
        errors = values.get('errors', [])
        warnings = values.get('warnings', [])
        info = values.get('info', [])
        
        return {
            'total_issues': len(errors) + len(warnings) + len(info),
            'errors': len(errors),
            'warnings': len(warnings),
            'info': len(info),
            'critical': len([e for e in errors if e.severity == ValidationSeverity.CRITICAL])
        }


class ValidationRuleSchema(BaseModel):
    """Schéma pour les règles de validation."""
    name: str = Field(..., description="Nom de la règle")
    description: str = Field(..., description="Description de la règle")
    type: ValidationType = Field(..., description="Type de validation")
    severity: ValidationSeverity = Field(ValidationSeverity.ERROR, description="Sévérité par défaut")
    enabled: bool = Field(True, description="Règle activée")
    pattern: Optional[str] = Field(None, description="Pattern de validation")
    expression: Optional[str] = Field(None, description="Expression de validation")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Paramètres de la règle")
    conditions: List[str] = Field(default_factory=list, description="Conditions d'application")
    tags: List[str] = Field(default_factory=list, description="Tags de classification")
    
    class Config:
        use_enum_values = True


class SchemaValidatorSchema(BaseModel):
    """Validateur de schémas Pydantic."""
    name: str = Field(..., description="Nom du validateur")
    schema_class: str = Field(..., description="Classe du schéma")
    rules: List[ValidationRuleSchema] = Field(..., description="Règles de validation")
    allow_extra: bool = Field(False, description="Autoriser les champs supplémentaires")
    validate_assignment: bool = Field(True, description="Valider les assignations")
    use_enum_values: bool = Field(True, description="Utiliser les valeurs d'enum")
    anystr_strip_whitespace: bool = Field(True, description="Supprimer les espaces")


class DataValidatorSchema(BaseModel):
    """Validateur de données."""
    name: str = Field(..., description="Nom du validateur")
    data_type: str = Field(..., description="Type de données")
    rules: List[ValidationRuleSchema] = Field(..., description="Règles de validation")
    transformations: List[str] = Field(default_factory=list, description="Transformations à appliquer")
    sanitization: Dict[str, bool] = Field(default_factory=dict, description="Règles de sanitisation")


class ConfigValidatorSchema(BaseModel):
    """Validateur de configuration."""
    name: str = Field(..., description="Nom du validateur")
    config_type: str = Field(..., description="Type de configuration")
    required_sections: List[str] = Field(..., description="Sections requises")
    optional_sections: List[str] = Field(default_factory=list, description="Sections optionnelles")
    validation_rules: List[ValidationRuleSchema] = Field(..., description="Règles de validation")
    cross_validation_rules: List[str] = Field(default_factory=list, description="Règles de validation croisée")


class SecurityValidatorSchema(BaseModel):
    """Validateur de sécurité."""
    name: str = Field(..., description="Nom du validateur")
    security_policies: List[str] = Field(..., description="Politiques de sécurité")
    vulnerability_checks: List[str] = Field(..., description="Vérifications de vulnérabilités")
    compliance_standards: List[str] = Field(..., description="Standards de conformité")
    encryption_requirements: Dict[str, str] = Field(default_factory=dict, description="Exigences de chiffrement")
    access_control_rules: List[str] = Field(default_factory=list, description="Règles de contrôle d'accès")


class ComplianceValidatorSchema(BaseModel):
    """Validateur de conformité."""
    name: str = Field(..., description="Nom du validateur")
    standards: List[str] = Field(..., description="Standards de conformité")
    requirements: List[Dict[str, Any]] = Field(..., description="Exigences de conformité")
    audit_trails: bool = Field(True, description="Pistes d'audit requises")
    documentation_required: List[str] = Field(default_factory=list, description="Documentation requise")
    reporting_frequency: str = Field("monthly", description="Fréquence de rapport")


class PerformanceValidatorSchema(BaseModel):
    """Validateur de performance."""
    name: str = Field(..., description="Nom du validateur")
    metrics: List[str] = Field(..., description="Métriques à valider")
    thresholds: Dict[str, float] = Field(..., description="Seuils de performance")
    load_testing: Dict[str, Any] = Field(default_factory=dict, description="Configuration de test de charge")
    benchmarks: List[Dict[str, Any]] = Field(default_factory=list, description="Benchmarks de référence")


class ToolParameterSchema(BaseModel):
    """Schéma pour les paramètres d'outils."""
    name: str = Field(..., description="Nom du paramètre")
    type: ParameterType = Field(..., description="Type du paramètre")
    description: str = Field(..., description="Description du paramètre")
    required: bool = Field(True, description="Paramètre requis")
    default_value: Optional[Any] = Field(None, description="Valeur par défaut")
    allowed_values: Optional[List[Any]] = Field(None, description="Valeurs autorisées")
    validation_pattern: Optional[str] = Field(None, description="Pattern de validation")
    min_value: Optional[Union[int, float]] = Field(None, description="Valeur minimale")
    max_value: Optional[Union[int, float]] = Field(None, description="Valeur maximale")
    sensitive: bool = Field(False, description="Paramètre sensible")
    
    class Config:
        use_enum_values = True


class ToolMetadataSchema(BaseModel):
    """Métadonnées d'un outil."""
    name: str = Field(..., description="Nom de l'outil")
    version: str = Field(..., description="Version de l'outil")
    description: str = Field(..., description="Description de l'outil")
    author: str = Field(..., description="Auteur de l'outil")
    category: ToolCategory = Field(..., description="Catégorie de l'outil")
    tags: List[str] = Field(default_factory=list, description="Tags de l'outil")
    documentation_url: Optional[str] = Field(None, description="URL de documentation")
    source_url: Optional[str] = Field(None, description="URL du code source")
    license: str = Field("MIT", description="Licence de l'outil")
    created_at: Optional[datetime] = Field(None, description="Date de création")
    updated_at: Optional[datetime] = Field(None, description="Date de mise à jour")
    
    class Config:
        use_enum_values = True


class ToolSecuritySchema(BaseModel):
    """Configuration de sécurité d'un outil."""
    requires_authentication: bool = Field(True, description="Authentification requise")
    required_permissions: List[str] = Field(default_factory=list, description="Permissions requises")
    allowed_environments: List[str] = Field(default_factory=list, description="Environnements autorisés")
    execution_context: str = Field("restricted", description="Contexte d'exécution")
    sandbox_enabled: bool = Field(True, description="Sandbox activé")
    network_access: bool = Field(False, description="Accès réseau autorisé")
    file_system_access: str = Field("read-only", description="Accès système de fichiers")
    resource_limits: Dict[str, str] = Field(
        default_factory=lambda: {
            "cpu": "100m",
            "memory": "128Mi",
            "disk": "1Gi",
            "network": "10Mbps"
        },
        description="Limites de ressources"
    )


class ToolLoggingSchema(BaseModel):
    """Configuration de logging d'un outil."""
    enabled: bool = Field(True, description="Logging activé")
    level: str = Field("INFO", description="Niveau de log")
    format: str = Field("json", description="Format des logs")
    destinations: List[str] = Field(default_factory=lambda: ["stdout"], description="Destinations")
    sensitive_data_masking: bool = Field(True, description="Masquage des données sensibles")
    audit_logging: bool = Field(True, description="Audit logging")
    retention_days: int = Field(30, description="Rétention en jours")


class ToolMonitoringSchema(BaseModel):
    """Configuration de monitoring d'un outil."""
    metrics_enabled: bool = Field(True, description="Métriques activées")
    health_check_enabled: bool = Field(True, description="Health check activé")
    performance_tracking: bool = Field(True, description="Suivi de performance")
    error_tracking: bool = Field(True, description="Suivi d'erreurs")
    alerting_enabled: bool = Field(True, description="Alertes activées")
    dashboards: List[str] = Field(default_factory=list, description="Dashboards disponibles")


class ToolResultSchema(BaseModel):
    """Résultat d'exécution d'un outil."""
    success: bool = Field(..., description="Exécution réussie")
    exit_code: int = Field(..., description="Code de sortie")
    output: str = Field("", description="Sortie standard")
    error_output: str = Field("", description="Sortie d'erreur")
    execution_time: float = Field(..., description="Temps d'exécution")
    resource_usage: Dict[str, Any] = Field(default_factory=dict, description="Utilisation des ressources")
    artifacts: List[str] = Field(default_factory=list, description="Artefacts générés")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées de résultat")


class ToolExecutionSchema(BaseModel):
    """Schéma d'exécution d'un outil."""
    tool_name: str = Field(..., description="Nom de l'outil")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Paramètres d'exécution")
    environment: str = Field("dev", description="Environnement d'exécution")
    execution_mode: ToolExecutionMode = Field(ToolExecutionMode.SYNCHRONOUS, description="Mode d'exécution")
    timeout: int = Field(300, description="Timeout en secondes")
    retry_count: int = Field(0, description="Nombre de tentatives")
    retry_delay: int = Field(1, description="Délai entre tentatives")
    dry_run: bool = Field(False, description="Exécution à sec")
    
    class Config:
        use_enum_values = True


class ToolConfigSchema(BaseModel):
    """Configuration complète d'un outil."""
    metadata: ToolMetadataSchema = Field(..., description="Métadonnées de l'outil")
    parameters: List[ToolParameterSchema] = Field(..., description="Paramètres de l'outil")
    security: ToolSecuritySchema = Field(..., description="Configuration de sécurité")
    logging: ToolLoggingSchema = Field(..., description="Configuration de logging")
    monitoring: ToolMonitoringSchema = Field(..., description="Configuration de monitoring")
    
    execution_config: Dict[str, Any] = Field(default_factory=dict, description="Configuration d'exécution")
    dependencies: List[str] = Field(default_factory=list, description="Dépendances de l'outil")
    pre_conditions: List[str] = Field(default_factory=list, description="Pré-conditions")
    post_conditions: List[str] = Field(default_factory=list, description="Post-conditions")
    
    enabled: bool = Field(True, description="Outil activé")
    maintenance_mode: bool = Field(False, description="Mode maintenance")


class AutomationToolSchema(BaseModel):
    """Outil d'automatisation."""
    config: ToolConfigSchema = Field(..., description="Configuration de base")
    triggers: List[Dict[str, Any]] = Field(..., description="Déclencheurs d'automatisation")
    workflow: List[Dict[str, Any]] = Field(..., description="Workflow d'exécution")
    scheduling: Dict[str, str] = Field(default_factory=dict, description="Configuration de planification")
    error_handling: Dict[str, Any] = Field(default_factory=dict, description="Gestion d'erreurs")


class DeploymentToolSchema(BaseModel):
    """Outil de déploiement."""
    config: ToolConfigSchema = Field(..., description="Configuration de base")
    deployment_strategy: str = Field(..., description="Stratégie de déploiement")
    environments: List[str] = Field(..., description="Environnements cibles")
    rollback_config: Dict[str, Any] = Field(default_factory=dict, description="Configuration de rollback")
    validation_steps: List[str] = Field(default_factory=list, description="Étapes de validation")


class MonitoringToolSchema(BaseModel):
    """Outil de monitoring."""
    config: ToolConfigSchema = Field(..., description="Configuration de base")
    metrics: List[str] = Field(..., description="Métriques surveillées")
    alert_rules: List[str] = Field(..., description="Règles d'alerte")
    dashboards: List[str] = Field(..., description="Dashboards associés")
    data_sources: List[str] = Field(..., description="Sources de données")


class SecurityToolSchema(BaseModel):
    """Outil de sécurité."""
    config: ToolConfigSchema = Field(..., description="Configuration de base")
    scan_types: List[str] = Field(..., description="Types de scan")
    vulnerability_database: str = Field(..., description="Base de données de vulnérabilités")
    compliance_checks: List[str] = Field(..., description="Vérifications de conformité")
    reporting_config: Dict[str, Any] = Field(default_factory=dict, description="Configuration de rapport")
