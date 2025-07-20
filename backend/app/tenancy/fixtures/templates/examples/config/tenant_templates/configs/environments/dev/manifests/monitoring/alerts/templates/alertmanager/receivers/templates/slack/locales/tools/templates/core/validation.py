"""
Gestionnaire de validation avancé pour le système de tenancy
Auteur: Fahed Mlaiel - Lead Dev & Architecte IA
"""

import re
import json
import yaml
from typing import Dict, Any, List, Optional, Union, Callable, Type, get_type_hints
from datetime import datetime, date
from dataclasses import dataclass, field
from enum import Enum
import ipaddress
from pathlib import Path
import asyncio
from functools import wraps
import structlog

logger = structlog.get_logger(__name__)

class ValidationType(Enum):
    """Types de validation disponibles"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    EMAIL = "email"
    URL = "url"
    IP_ADDRESS = "ip_address"
    UUID = "uuid"
    DATE = "date"
    DATETIME = "datetime"
    JSON = "json"
    YAML = "yaml"
    REGEX = "regex"
    CUSTOM = "custom"

class ValidationSeverity(Enum):
    """Niveaux de sévérité des erreurs de validation"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationRule:
    """Règle de validation"""
    name: str
    validation_type: ValidationType
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable] = None
    error_message: Optional[str] = None
    severity: ValidationSeverity = ValidationSeverity.ERROR

@dataclass
class ValidationError:
    """Erreur de validation"""
    field: str
    rule: str
    message: str
    severity: ValidationSeverity
    value: Any = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ValidationResult:
    """Résultat de validation"""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    validated_data: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0

class BaseValidator:
    """Validateur de base"""
    
    def __init__(self):
        self.rules: Dict[str, ValidationRule] = {}
        self.custom_validators: Dict[str, Callable] = {}
        
    def add_rule(self, field: str, rule: ValidationRule) -> None:
        """Ajoute une règle de validation"""
        self.rules[field] = rule
        
    def add_custom_validator(self, name: str, validator: Callable) -> None:
        """Ajoute un validateur personnalisé"""
        self.custom_validators[name] = validator

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Valide les données selon les règles définies"""
        start_time = datetime.utcnow()
        errors = []
        warnings = []
        validated_data = {}
        
        try:
            for field, rule in self.rules.items():
                field_errors = self._validate_field(field, data.get(field), rule)
                
                for error in field_errors:
                    if error.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                        errors.append(error)
                    else:
                        warnings.append(error)
                
                # Si pas d'erreurs critiques, ajouter la valeur validée
                if not any(e.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                          for e in field_errors):
                    validated_data[field] = data.get(field)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                validated_data=validated_data,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error("Erreur lors de la validation", error=str(e))
            return ValidationResult(
                is_valid=False,
                errors=[ValidationError(
                    field="__global__",
                    rule="validation_exception",
                    message=f"Erreur de validation: {str(e)}",
                    severity=ValidationSeverity.CRITICAL
                )]
            )

    def _validate_field(self, field: str, value: Any, rule: ValidationRule) -> List[ValidationError]:
        """Valide un champ spécifique"""
        errors = []
        
        # Vérification de la présence si requis
        if rule.required and (value is None or value == ""):
            errors.append(ValidationError(
                field=field,
                rule="required",
                message=rule.error_message or f"Le champ {field} est requis",
                severity=rule.severity,
                value=value
            ))
            return errors
        
        # Si la valeur n'est pas requise et est vide, pas de validation
        if not rule.required and (value is None or value == ""):
            return errors
        
        # Validation selon le type
        type_errors = self._validate_type(field, value, rule)
        errors.extend(type_errors)
        
        # Validation des contraintes
        constraint_errors = self._validate_constraints(field, value, rule)
        errors.extend(constraint_errors)
        
        # Validation personnalisée
        if rule.custom_validator:
            custom_errors = self._validate_custom(field, value, rule)
            errors.extend(custom_errors)
        
        return errors

    def _validate_type(self, field: str, value: Any, rule: ValidationRule) -> List[ValidationError]:
        """Valide le type de la valeur"""
        errors = []
        
        try:
            if rule.validation_type == ValidationType.STRING:
                if not isinstance(value, str):
                    errors.append(self._create_type_error(field, value, "string", rule))
                    
            elif rule.validation_type == ValidationType.INTEGER:
                if not isinstance(value, int) or isinstance(value, bool):
                    errors.append(self._create_type_error(field, value, "integer", rule))
                    
            elif rule.validation_type == ValidationType.FLOAT:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    errors.append(self._create_type_error(field, value, "float", rule))
                    
            elif rule.validation_type == ValidationType.BOOLEAN:
                if not isinstance(value, bool):
                    errors.append(self._create_type_error(field, value, "boolean", rule))
                    
            elif rule.validation_type == ValidationType.EMAIL:
                if not self._is_valid_email(value):
                    errors.append(self._create_format_error(field, value, "email", rule))
                    
            elif rule.validation_type == ValidationType.URL:
                if not self._is_valid_url(value):
                    errors.append(self._create_format_error(field, value, "URL", rule))
                    
            elif rule.validation_type == ValidationType.IP_ADDRESS:
                if not self._is_valid_ip(value):
                    errors.append(self._create_format_error(field, value, "IP address", rule))
                    
            elif rule.validation_type == ValidationType.UUID:
                if not self._is_valid_uuid(value):
                    errors.append(self._create_format_error(field, value, "UUID", rule))
                    
            elif rule.validation_type == ValidationType.DATE:
                if not self._is_valid_date(value):
                    errors.append(self._create_format_error(field, value, "date", rule))
                    
            elif rule.validation_type == ValidationType.DATETIME:
                if not self._is_valid_datetime(value):
                    errors.append(self._create_format_error(field, value, "datetime", rule))
                    
            elif rule.validation_type == ValidationType.JSON:
                if not self._is_valid_json(value):
                    errors.append(self._create_format_error(field, value, "JSON", rule))
                    
            elif rule.validation_type == ValidationType.YAML:
                if not self._is_valid_yaml(value):
                    errors.append(self._create_format_error(field, value, "YAML", rule))
                    
            elif rule.validation_type == ValidationType.REGEX:
                if rule.pattern and not re.match(rule.pattern, str(value)):
                    errors.append(self._create_pattern_error(field, value, rule.pattern, rule))
                    
        except Exception as e:
            errors.append(ValidationError(
                field=field,
                rule="type_validation_error",
                message=f"Erreur lors de la validation du type: {str(e)}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
        
        return errors

    def _validate_constraints(self, field: str, value: Any, rule: ValidationRule) -> List[ValidationError]:
        """Valide les contraintes de valeur"""
        errors = []
        
        try:
            # Validation de la longueur pour les chaînes
            if isinstance(value, str):
                if rule.min_length is not None and len(value) < rule.min_length:
                    errors.append(ValidationError(
                        field=field,
                        rule="min_length",
                        message=f"La longueur minimale pour {field} est {rule.min_length}",
                        severity=rule.severity,
                        value=value
                    ))
                    
                if rule.max_length is not None and len(value) > rule.max_length:
                    errors.append(ValidationError(
                        field=field,
                        rule="max_length",
                        message=f"La longueur maximale pour {field} est {rule.max_length}",
                        severity=rule.severity,
                        value=value
                    ))
            
            # Validation des valeurs numériques
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if rule.min_value is not None and value < rule.min_value:
                    errors.append(ValidationError(
                        field=field,
                        rule="min_value",
                        message=f"La valeur minimale pour {field} est {rule.min_value}",
                        severity=rule.severity,
                        value=value
                    ))
                    
                if rule.max_value is not None and value > rule.max_value:
                    errors.append(ValidationError(
                        field=field,
                        rule="max_value",
                        message=f"La valeur maximale pour {field} est {rule.max_value}",
                        severity=rule.severity,
                        value=value
                    ))
            
            # Validation des valeurs autorisées
            if rule.allowed_values is not None and value not in rule.allowed_values:
                errors.append(ValidationError(
                    field=field,
                    rule="allowed_values",
                    message=f"La valeur {value} n'est pas autorisée pour {field}. Valeurs autorisées: {rule.allowed_values}",
                    severity=rule.severity,
                    value=value
                ))
                
        except Exception as e:
            errors.append(ValidationError(
                field=field,
                rule="constraint_validation_error",
                message=f"Erreur lors de la validation des contraintes: {str(e)}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
        
        return errors

    def _validate_custom(self, field: str, value: Any, rule: ValidationRule) -> List[ValidationError]:
        """Valide avec un validateur personnalisé"""
        errors = []
        
        try:
            if not rule.custom_validator(value):
                errors.append(ValidationError(
                    field=field,
                    rule="custom_validation",
                    message=rule.error_message or f"Validation personnalisée échouée pour {field}",
                    severity=rule.severity,
                    value=value
                ))
        except Exception as e:
            errors.append(ValidationError(
                field=field,
                rule="custom_validation_error",
                message=f"Erreur lors de la validation personnalisée: {str(e)}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
        
        return errors

    def _create_type_error(self, field: str, value: Any, expected_type: str, rule: ValidationRule) -> ValidationError:
        """Crée une erreur de type"""
        return ValidationError(
            field=field,
            rule="type_mismatch",
            message=rule.error_message or f"Le champ {field} doit être de type {expected_type}",
            severity=rule.severity,
            value=value
        )

    def _create_format_error(self, field: str, value: Any, format_name: str, rule: ValidationRule) -> ValidationError:
        """Crée une erreur de format"""
        return ValidationError(
            field=field,
            rule="format_invalid",
            message=rule.error_message or f"Le champ {field} doit être un {format_name} valide",
            severity=rule.severity,
            value=value
        )

    def _create_pattern_error(self, field: str, value: Any, pattern: str, rule: ValidationRule) -> ValidationError:
        """Crée une erreur de pattern"""
        return ValidationError(
            field=field,
            rule="pattern_mismatch",
            message=rule.error_message or f"Le champ {field} ne correspond pas au pattern {pattern}",
            severity=rule.severity,
            value=value
        )

    # Méthodes de validation de format
    def _is_valid_email(self, value: str) -> bool:
        """Valide un email"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return isinstance(value, str) and re.match(email_pattern, value) is not None

    def _is_valid_url(self, value: str) -> bool:
        """Valide une URL"""
        url_pattern = r'^https?:\/\/(?:[-\w.])+(?:\:[0-9]+)?(?:\/(?:[\w\/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
        return isinstance(value, str) and re.match(url_pattern, value) is not None

    def _is_valid_ip(self, value: str) -> bool:
        """Valide une adresse IP"""
        try:
            ipaddress.ip_address(value)
            return True
        except ValueError:
            return False

    def _is_valid_uuid(self, value: str) -> bool:
        """Valide un UUID"""
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
        return isinstance(value, str) and re.match(uuid_pattern, value.lower()) is not None

    def _is_valid_date(self, value: str) -> bool:
        """Valide une date"""
        try:
            datetime.strptime(value, '%Y-%m-%d')
            return True
        except (ValueError, TypeError):
            return False

    def _is_valid_datetime(self, value: str) -> bool:
        """Valide un datetime"""
        try:
            datetime.fromisoformat(value.replace('Z', '+00:00'))
            return True
        except (ValueError, TypeError):
            return False

    def _is_valid_json(self, value: str) -> bool:
        """Valide un JSON"""
        try:
            json.loads(value)
            return True
        except (ValueError, TypeError):
            return False

    def _is_valid_yaml(self, value: str) -> bool:
        """Valide un YAML"""
        try:
            yaml.safe_load(value)
            return True
        except (yaml.YAMLError, TypeError):
            return False

class TenantConfigValidator(BaseValidator):
    """Validateur spécialisé pour la configuration des tenants"""
    
    def __init__(self):
        super().__init__()
        self._setup_tenant_rules()
        
    def _setup_tenant_rules(self):
        """Configure les règles de validation pour les tenants"""
        
        # Informations de base du tenant
        self.add_rule("tenant_id", ValidationRule(
            name="tenant_id",
            validation_type=ValidationType.STRING,
            required=True,
            min_length=3,
            max_length=50,
            pattern=r'^[a-zA-Z0-9_-]+$',
            error_message="L'ID du tenant doit contenir uniquement des caractères alphanumériques, tirets et underscores"
        ))
        
        self.add_rule("name", ValidationRule(
            name="name",
            validation_type=ValidationType.STRING,
            required=True,
            min_length=2,
            max_length=100
        ))
        
        self.add_rule("email", ValidationRule(
            name="email",
            validation_type=ValidationType.EMAIL,
            required=True
        ))
        
        # Configuration des quotas
        self.add_rule("api_quota_per_hour", ValidationRule(
            name="api_quota_per_hour",
            validation_type=ValidationType.INTEGER,
            required=True,
            min_value=1,
            max_value=100000
        ))
        
        self.add_rule("storage_quota_gb", ValidationRule(
            name="storage_quota_gb",
            validation_type=ValidationType.FLOAT,
            required=True,
            min_value=0.1,
            max_value=1000.0
        ))
        
        # Configuration de sécurité
        self.add_rule("allowed_ips", ValidationRule(
            name="allowed_ips",
            validation_type=ValidationType.CUSTOM,
            required=False,
            custom_validator=self._validate_ip_list
        ))
        
        self.add_rule("encryption_enabled", ValidationRule(
            name="encryption_enabled",
            validation_type=ValidationType.BOOLEAN,
            required=True
        ))
        
        # Configuration des fonctionnalités
        self.add_rule("features", ValidationRule(
            name="features",
            validation_type=ValidationType.CUSTOM,
            required=True,
            custom_validator=self._validate_features_list
        ))

    def _validate_ip_list(self, value: List[str]) -> bool:
        """Valide une liste d'adresses IP"""
        if not isinstance(value, list):
            return False
        
        for ip in value:
            if not self._is_valid_ip(ip):
                return False
        return True

    def _validate_features_list(self, value: List[str]) -> bool:
        """Valide une liste de fonctionnalités"""
        if not isinstance(value, list):
            return False
        
        allowed_features = [
            "audio_processing", "playlist_generation", "recommendation_engine",
            "analytics", "social_sharing", "advanced_search", "custom_themes"
        ]
        
        for feature in value:
            if feature not in allowed_features:
                return False
        return True

class SchemaValidator:
    """Validateur de schémas JSON/YAML"""
    
    def __init__(self):
        self.schemas: Dict[str, Dict[str, Any]] = {}
        
    def register_schema(self, name: str, schema: Dict[str, Any]) -> None:
        """Enregistre un schéma de validation"""
        self.schemas[name] = schema
        
    def validate_against_schema(self, data: Dict[str, Any], schema_name: str) -> ValidationResult:
        """Valide des données contre un schéma"""
        if schema_name not in self.schemas:
            return ValidationResult(
                is_valid=False,
                errors=[ValidationError(
                    field="__schema__",
                    rule="schema_not_found",
                    message=f"Schéma {schema_name} non trouvé",
                    severity=ValidationSeverity.ERROR
                )]
            )
        
        schema = self.schemas[schema_name]
        return self._validate_recursive(data, schema)

    def _validate_recursive(self, data: Any, schema: Dict[str, Any], path: str = "") -> ValidationResult:
        """Validation récursive contre un schéma"""
        errors = []
        warnings = []
        
        # Implémentation simplifiée - en production, utiliser jsonschema
        if "type" in schema:
            if not self._check_type(data, schema["type"]):
                errors.append(ValidationError(
                    field=path or "__root__",
                    rule="type_mismatch",
                    message=f"Type attendu: {schema['type']}, reçu: {type(data).__name__}",
                    severity=ValidationSeverity.ERROR,
                    value=data
                ))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Vérifie le type d'une valeur"""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return True
        
        return isinstance(value, expected_python_type)

def validate_with_decorator(validator_class: Type[BaseValidator] = BaseValidator):
    """Décorateur pour la validation automatique"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            validator = validator_class()
            
            # Extraction des données à valider depuis les arguments
            if len(args) > 0 and isinstance(args[0], dict):
                data = args[0]
            elif 'data' in kwargs:
                data = kwargs['data']
            else:
                return await func(*args, **kwargs)
            
            # Validation
            result = validator.validate(data)
            
            if not result.is_valid:
                raise ValueError(f"Erreurs de validation: {[e.message for e in result.errors]}")
            
            # Remplacement des données par les données validées
            if len(args) > 0 and isinstance(args[0], dict):
                args = (result.validated_data,) + args[1:]
            elif 'data' in kwargs:
                kwargs['data'] = result.validated_data
            
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Version synchrone du wrapper
            validator = validator_class()
            
            if len(args) > 0 and isinstance(args[0], dict):
                data = args[0]
            elif 'data' in kwargs:
                data = kwargs['data']
            else:
                return func(*args, **kwargs)
            
            result = validator.validate(data)
            
            if not result.is_valid:
                raise ValueError(f"Erreurs de validation: {[e.message for e in result.errors]}")
            
            if len(args) > 0 and isinstance(args[0], dict):
                args = (result.validated_data,) + args[1:]
            elif 'data' in kwargs:
                kwargs['data'] = result.validated_data
            
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Instances globales des validateurs
tenant_validator = TenantConfigValidator()
schema_validator = SchemaValidator()

# Schémas prédéfinis
schema_validator.register_schema("tenant_config", {
    "type": "object",
    "required": ["tenant_id", "name", "email"],
    "properties": {
        "tenant_id": {"type": "string", "minLength": 3, "maxLength": 50},
        "name": {"type": "string", "minLength": 2, "maxLength": 100},
        "email": {"type": "string", "format": "email"},
        "api_quota_per_hour": {"type": "integer", "minimum": 1, "maximum": 100000},
        "storage_quota_gb": {"type": "number", "minimum": 0.1, "maximum": 1000.0}
    }
})
