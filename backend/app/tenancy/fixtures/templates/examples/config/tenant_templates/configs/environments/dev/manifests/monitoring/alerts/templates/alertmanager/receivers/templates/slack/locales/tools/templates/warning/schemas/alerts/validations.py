"""
Validations avancées - Spotify AI Agent
Logique de validation métier et règles de cohérence des données
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Callable, Type
from uuid import UUID
from enum import Enum
import re
import ipaddress
import json
from dataclasses import dataclass

from pydantic import BaseModel, Field, validator, ValidationError
from pydantic.types import EmailStr

from . import (
    BaseSchema, AlertLevel, AlertStatus, WarningCategory, Priority, Environment,
    NotificationChannel, EscalationLevel, CorrelationMethod, WorkflowStatus,
    IncidentStatus, ModelFramework, SecurityLevel
)


class ValidationType(str, Enum):
    """Types de validation"""
    FIELD = "field"
    SCHEMA = "schema"
    BUSINESS = "business"
    SECURITY = "security"
    CONSISTENCY = "consistency"
    PERFORMANCE = "performance"


class ValidationSeverity(str, Enum):
    """Sévérité des erreurs de validation"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationContext(str, Enum):
    """Contexte de validation"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    BULK_OPERATION = "bulk_operation"
    IMPORT = "import"
    MIGRATION = "migration"


@dataclass
class ValidationResult:
    """Résultat de validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    info: List[str]
    metadata: Dict[str, Any]

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def add_error(self, message: str, field: Optional[str] = None):
        """Ajoute une erreur"""
        if field:
            message = f"{field}: {message}"
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str, field: Optional[str] = None):
        """Ajoute un avertissement"""
        if field:
            message = f"{field}: {message}"
        self.warnings.append(message)

    def add_info(self, message: str, field: Optional[str] = None):
        """Ajoute une information"""
        if field:
            message = f"{field}: {message}"
        self.info.append(message)


class BaseValidator:
    """Validateur de base"""
    
    def __init__(self, context: ValidationContext = ValidationContext.CREATE):
        self.context = context
    
    def validate(self, data: Any) -> ValidationResult:
        """Valide les données"""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            info=[],
            metadata={}
        )
        
        try:
            self._validate_impl(data, result)
        except Exception as e:
            result.add_error(f"Erreur de validation: {str(e)}")
        
        return result
    
    def _validate_impl(self, data: Any, result: ValidationResult):
        """Implémentation de la validation (à surcharger)"""
        pass


class AlertRuleValidator(BaseValidator):
    """Validateur pour les règles d'alerte"""
    
    def _validate_impl(self, rule: Any, result: ValidationResult):
        """Valide une règle d'alerte"""
        # Validation de la condition
        if hasattr(rule, 'condition'):
            self._validate_condition(rule.condition, result)
        
        # Validation des seuils
        if hasattr(rule, 'thresholds'):
            self._validate_thresholds(rule.thresholds, result)
        
        # Validation de la fenêtre temporelle
        if hasattr(rule, 'time_window_minutes'):
            self._validate_time_window(rule.time_window_minutes, result)
        
        # Validation des filtres
        if hasattr(rule, 'filters'):
            self._validate_filters(rule.filters, result)
        
        # Validation de la cohérence niveau/priorité
        if hasattr(rule, 'level') and hasattr(rule, 'priority'):
            self._validate_level_priority_consistency(rule.level, rule.priority, result)
    
    def _validate_condition(self, condition: str, result: ValidationResult):
        """Valide la condition d'alerte"""
        if not condition or not condition.strip():
            result.add_error("La condition ne peut pas être vide", "condition")
            return
        
        # Validation syntaxe PromQL/SQL basique
        if 'SELECT' in condition.upper():
            # Validation SQL basique
            if not re.search(r'\bFROM\b', condition, re.IGNORECASE):
                result.add_warning("Condition SQL sans clause FROM", "condition")
        elif '{' in condition and '}' in condition:
            # Validation PromQL basique
            if not re.search(r'[a-zA-Z_][a-zA-Z0-9_]*\{', condition):
                result.add_warning("Métrique PromQL non reconnue", "condition")
        
        # Validation des injections potentielles
        dangerous_patterns = [
            r';.*DROP\s+TABLE',
            r';.*DELETE\s+FROM',
            r';.*UPDATE\s+.*SET',
            r'<script[^>]*>',
            r'javascript:',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, condition, re.IGNORECASE):
                result.add_error("Pattern dangereux détecté dans la condition", "condition")
    
    def _validate_thresholds(self, thresholds: Dict[str, Any], result: ValidationResult):
        """Valide les seuils"""
        critical = thresholds.get('critical')
        warning = thresholds.get('warning')
        
        if critical is not None and warning is not None:
            if isinstance(critical, (int, float)) and isinstance(warning, (int, float)):
                if critical <= warning:
                    result.add_warning(
                        "Le seuil critique devrait être supérieur au seuil d'avertissement",
                        "thresholds"
                    )
        
        # Validation des valeurs négatives pour certaines métriques
        for threshold_type, value in thresholds.items():
            if isinstance(value, (int, float)) and value < 0:
                if threshold_type in ['cpu_usage', 'memory_usage', 'disk_usage']:
                    result.add_error(
                        f"Seuil {threshold_type} ne peut pas être négatif",
                        "thresholds"
                    )
    
    def _validate_time_window(self, time_window: int, result: ValidationResult):
        """Valide la fenêtre temporelle"""
        if time_window <= 0:
            result.add_error("La fenêtre temporelle doit être positive", "time_window_minutes")
        elif time_window > 10080:  # 7 jours
            result.add_warning("Fenêtre temporelle très longue (>7 jours)", "time_window_minutes")
        elif time_window < 1:
            result.add_warning("Fenêtre temporelle très courte (<1 minute)", "time_window_minutes")
    
    def _validate_filters(self, filters: Dict[str, Any], result: ValidationResult):
        """Valide les filtres"""
        for key, value in filters.items():
            if not key or not key.strip():
                result.add_error("Clé de filtre vide", "filters")
            
            if isinstance(value, str) and len(value) > 1000:
                result.add_warning(f"Valeur de filtre très longue pour {key}", "filters")
    
    def _validate_level_priority_consistency(self, level: AlertLevel, priority: Priority, result: ValidationResult):
        """Valide la cohérence entre niveau et priorité"""
        level_priority_mapping = {
            AlertLevel.CRITICAL: [Priority.P0, Priority.P1],
            AlertLevel.HIGH: [Priority.P1, Priority.P2],
            AlertLevel.MEDIUM: [Priority.P2, Priority.P3],
            AlertLevel.LOW: [Priority.P3, Priority.P4],
            AlertLevel.INFO: [Priority.P4, Priority.P5]
        }
        
        expected_priorities = level_priority_mapping.get(level, [])
        if expected_priorities and priority not in expected_priorities:
            result.add_warning(
                f"Incohérence niveau-priorité: {level} avec {priority}",
                "level_priority"
            )


class NotificationValidator(BaseValidator):
    """Validateur pour les notifications"""
    
    def _validate_impl(self, notification: Any, result: ValidationResult):
        """Valide une notification"""
        # Validation du canal
        if hasattr(notification, 'channel'):
            self._validate_channel(notification.channel, result)
        
        # Validation des destinataires
        if hasattr(notification, 'recipients'):
            self._validate_recipients(notification.recipients, result)
        
        # Validation du template
        if hasattr(notification, 'template_id'):
            self._validate_template(notification.template_id, result)
        
        # Validation de la limitation de débit
        if hasattr(notification, 'rate_limit'):
            self._validate_rate_limit(notification.rate_limit, result)
    
    def _validate_channel(self, channel: NotificationChannel, result: ValidationResult):
        """Valide le canal de notification"""
        if channel == NotificationChannel.EMAIL:
            # Validation spécifique email
            pass
        elif channel == NotificationChannel.SLACK:
            # Validation spécifique Slack
            pass
        elif channel == NotificationChannel.WEBHOOK:
            # Validation spécifique webhook
            pass
    
    def _validate_recipients(self, recipients: List[str], result: ValidationResult):
        """Valide les destinataires"""
        if not recipients:
            result.add_warning("Aucun destinataire spécifié", "recipients")
            return
        
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        
        for recipient in recipients:
            if '@' in recipient:  # Probablement un email
                if not email_pattern.match(recipient):
                    result.add_error(f"Format email invalide: {recipient}", "recipients")
            elif recipient.startswith('#'):  # Canal Slack
                if not re.match(r'^#[a-zA-Z0-9_-]+$', recipient):
                    result.add_error(f"Format canal Slack invalide: {recipient}", "recipients")
    
    def _validate_template(self, template_id: str, result: ValidationResult):
        """Valide le template"""
        if not template_id or not template_id.strip():
            result.add_error("Template ID requis", "template_id")
    
    def _validate_rate_limit(self, rate_limit: Dict[str, Any], result: ValidationResult):
        """Valide la limitation de débit"""
        max_per_minute = rate_limit.get('max_per_minute', 0)
        max_per_hour = rate_limit.get('max_per_hour', 0)
        
        if max_per_minute > 60:
            result.add_warning("Limite très élevée (>60/min)", "rate_limit")
        
        if max_per_hour > 0 and max_per_minute > 0:
            if max_per_minute * 60 > max_per_hour:
                result.add_error(
                    "Incohérence: limite/minute * 60 > limite/heure",
                    "rate_limit"
                )


class EscalationValidator(BaseValidator):
    """Validateur pour l'escalade"""
    
    def _validate_impl(self, escalation: Any, result: ValidationResult):
        """Valide une escalade"""
        if hasattr(escalation, 'levels'):
            self._validate_escalation_levels(escalation.levels, result)
        
        if hasattr(escalation, 'conditions'):
            self._validate_escalation_conditions(escalation.conditions, result)
    
    def _validate_escalation_levels(self, levels: List[Any], result: ValidationResult):
        """Valide les niveaux d'escalade"""
        if not levels:
            result.add_error("Au moins un niveau d'escalade requis", "levels")
            return
        
        # Vérifier l'ordre des délais
        previous_delay = 0
        for i, level in enumerate(levels):
            if hasattr(level, 'delay_minutes'):
                if level.delay_minutes <= previous_delay:
                    result.add_error(
                        f"Délai du niveau {i+1} doit être > niveau précédent",
                        "levels"
                    )
                previous_delay = level.delay_minutes
    
    def _validate_escalation_conditions(self, conditions: Dict[str, Any], result: ValidationResult):
        """Valide les conditions d'escalade"""
        if 'max_escalations' in conditions:
            max_esc = conditions['max_escalations']
            if max_esc < 1:
                result.add_error("Nombre max d'escalades doit être >= 1", "conditions")


class SecurityValidator(BaseValidator):
    """Validateur de sécurité"""
    
    def _validate_impl(self, data: Any, result: ValidationResult):
        """Valide la sécurité"""
        # Validation des permissions
        if hasattr(data, 'permissions'):
            self._validate_permissions(data.permissions, result)
        
        # Validation des données sensibles
        self._validate_sensitive_data(data, result)
        
        # Validation des accès IP
        if hasattr(data, 'ip_whitelist'):
            self._validate_ip_whitelist(data.ip_whitelist, result)
    
    def _validate_permissions(self, permissions: List[str], result: ValidationResult):
        """Valide les permissions"""
        valid_permissions = [
            'read', 'write', 'delete', 'admin',
            'alerts.read', 'alerts.write', 'alerts.delete',
            'notifications.send', 'escalation.manage'
        ]
        
        for perm in permissions:
            if perm not in valid_permissions:
                result.add_warning(f"Permission non reconnue: {perm}", "permissions")
    
    def _validate_sensitive_data(self, data: Any, result: ValidationResult):
        """Valide les données sensibles"""
        sensitive_patterns = [
            (r'password\s*[:=]\s*["\']?[^"\'\s]+', 'Mot de passe en clair détecté'),
            (r'api[_-]?key\s*[:=]\s*["\']?[^"\'\s]+', 'Clé API en clair détectée'),
            (r'secret\s*[:=]\s*["\']?[^"\'\s]+', 'Secret en clair détecté'),
            (r'token\s*[:=]\s*["\']?[^"\'\s]+', 'Token en clair détecté'),
        ]
        
        data_str = str(data).lower()
        for pattern, message in sensitive_patterns:
            if re.search(pattern, data_str, re.IGNORECASE):
                result.add_error(message, "sensitive_data")
    
    def _validate_ip_whitelist(self, ip_list: List[str], result: ValidationResult):
        """Valide la liste blanche d'IPs"""
        for ip in ip_list:
            try:
                ipaddress.ip_network(ip, strict=False)
            except ValueError:
                result.add_error(f"Format IP invalide: {ip}", "ip_whitelist")


class BusinessRuleValidator(BaseValidator):
    """Validateur de règles métier"""
    
    def __init__(self, context: ValidationContext = ValidationContext.CREATE,
                 tenant_config: Optional[Dict[str, Any]] = None):
        super().__init__(context)
        self.tenant_config = tenant_config or {}
    
    def _validate_impl(self, data: Any, result: ValidationResult):
        """Valide les règles métier"""
        # Validation des quotas tenant
        self._validate_tenant_quotas(data, result)
        
        # Validation des contraintes temporelles
        self._validate_time_constraints(data, result)
        
        # Validation des dépendances
        self._validate_dependencies(data, result)
    
    def _validate_tenant_quotas(self, data: Any, result: ValidationResult):
        """Valide les quotas tenant"""
        max_alerts = self.tenant_config.get('max_alerts_per_hour', 1000)
        max_notifications = self.tenant_config.get('max_notifications_per_day', 10000)
        
        if hasattr(data, 'alert_count') and data.alert_count > max_alerts:
            result.add_error(
                f"Quota d'alertes dépassé: {data.alert_count} > {max_alerts}",
                "quota"
            )
    
    def _validate_time_constraints(self, data: Any, result: ValidationResult):
        """Valide les contraintes temporelles"""
        now = datetime.now(timezone.utc)
        
        if hasattr(data, 'start_time') and hasattr(data, 'end_time'):
            if data.end_time <= data.start_time:
                result.add_error(
                    "Heure de fin doit être après heure de début",
                    "time_constraints"
                )
            
            if data.start_time > now + timedelta(days=365):
                result.add_warning(
                    "Heure de début très éloignée dans le futur",
                    "time_constraints"
                )
    
    def _validate_dependencies(self, data: Any, result: ValidationResult):
        """Valide les dépendances"""
        if hasattr(data, 'depends_on') and data.depends_on:
            for dep_id in data.depends_on:
                if not self._dependency_exists(dep_id):
                    result.add_error(
                        f"Dépendance introuvable: {dep_id}",
                        "dependencies"
                    )
    
    def _dependency_exists(self, dep_id: str) -> bool:
        """Vérifie si une dépendance existe"""
        # Ici, on ferait une vérification en base de données
        # Pour l'exemple, on retourne True
        return True


class CompositeValidator:
    """Validateur composite utilisant plusieurs validateurs"""
    
    def __init__(self, validators: List[BaseValidator]):
        self.validators = validators
    
    def validate(self, data: Any) -> ValidationResult:
        """Valide avec tous les validateurs"""
        composite_result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            info=[],
            metadata={}
        )
        
        for validator in self.validators:
            result = validator.validate(data)
            
            composite_result.errors.extend(result.errors)
            composite_result.warnings.extend(result.warnings)
            composite_result.info.extend(result.info)
            composite_result.metadata.update(result.metadata)
            
            if not result.is_valid:
                composite_result.is_valid = False
        
        return composite_result


class ValidationFactory:
    """Factory pour créer des validateurs"""
    
    @staticmethod
    def create_alert_validator(context: ValidationContext = ValidationContext.CREATE) -> CompositeValidator:
        """Crée un validateur pour les alertes"""
        return CompositeValidator([
            AlertRuleValidator(context),
            SecurityValidator(context),
            BusinessRuleValidator(context)
        ])
    
    @staticmethod
    def create_notification_validator(context: ValidationContext = ValidationContext.CREATE) -> CompositeValidator:
        """Crée un validateur pour les notifications"""
        return CompositeValidator([
            NotificationValidator(context),
            SecurityValidator(context),
            BusinessRuleValidator(context)
        ])
    
    @staticmethod
    def create_escalation_validator(context: ValidationContext = ValidationContext.CREATE) -> CompositeValidator:
        """Crée un validateur pour l'escalade"""
        return CompositeValidator([
            EscalationValidator(context),
            SecurityValidator(context),
            BusinessRuleValidator(context)
        ])


# Décorateur pour validation automatique
def validate_with(validator_type: str):
    """Décorateur pour validation automatique"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Logique de validation avant exécution
            if validator_type == 'alert':
                validator = ValidationFactory.create_alert_validator()
            elif validator_type == 'notification':
                validator = ValidationFactory.create_notification_validator()
            else:
                validator = ValidationFactory.create_escalation_validator()
            
            # Validation des arguments
            for arg in args:
                if isinstance(arg, BaseModel):
                    result = validator.validate(arg)
                    if not result.is_valid:
                        raise ValidationError(result.errors)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


__all__ = [
    'ValidationType', 'ValidationSeverity', 'ValidationContext', 'ValidationResult',
    'BaseValidator', 'AlertRuleValidator', 'NotificationValidator', 'EscalationValidator',
    'SecurityValidator', 'BusinessRuleValidator', 'CompositeValidator',
    'ValidationFactory', 'validate_with'
]
