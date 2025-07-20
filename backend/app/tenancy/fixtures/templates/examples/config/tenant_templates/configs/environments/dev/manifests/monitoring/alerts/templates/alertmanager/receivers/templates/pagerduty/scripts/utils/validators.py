#!/usr/bin/env python3
"""
Advanced Validators for PagerDuty Integration

Module de validation avancé pour les configurations, données, et intégrations PagerDuty.
Fournit des validateurs robustes avec support des schémas complexes,
validation de sécurité, et règles métier spécifiques.

Fonctionnalités:
- Validation de schémas JSON/YAML avancés
- Validation de sécurité (injection, XSS, etc.)
- Validation des formats PagerDuty
- Validation de règles métier
- Validation de performance
- Validation de conformité

Version: 1.0.0
Auteur: Spotify AI Agent Team
"""

import re
import json
import ipaddress
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from urllib.parse import urlparse
import yaml
from jsonschema import validate, ValidationError, Draft7Validator

class ValidationResult:
    """Résultat de validation avec détails"""
    
    def __init__(self, is_valid: bool = True, errors: List[str] = None, warnings: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def add_error(self, error: str):
        """Ajoute une erreur"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Ajoute un avertissement"""
        self.warnings.append(warning)
    
    def merge(self, other: 'ValidationResult'):
        """Fusionne avec un autre résultat"""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False

class PagerDutyValidator:
    """Validateur principal pour PagerDuty"""
    
    # Expressions régulières pour les formats PagerDuty
    API_KEY_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{20,}$')
    SERVICE_ID_PATTERN = re.compile(r'^P[A-Z0-9]{6}$')
    INTEGRATION_KEY_PATTERN = re.compile(r'^[a-f0-9]{32}$')
    INCIDENT_KEY_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
    USER_ID_PATTERN = re.compile(r'^P[A-Z0-9]{6}$')
    TEAM_ID_PATTERN = re.compile(r'^P[A-Z0-9]{6}$')
    ESCALATION_POLICY_ID_PATTERN = re.compile(r'^P[A-Z0-9]{6}$')
    
    # Schémas de validation
    PAGERDUTY_CONFIG_SCHEMA = {
        "type": "object",
        "properties": {
            "api_key": {
                "type": "string",
                "minLength": 20,
                "pattern": r"^[a-zA-Z0-9_-]+$"
            },
            "service_id": {
                "type": "string",
                "pattern": r"^P[A-Z0-9]{6}$"
            },
            "integration_key": {
                "type": "string",
                "pattern": r"^[a-f0-9]{32}$"
            },
            "routing_key": {
                "type": "string",
                "pattern": r"^[a-f0-9]{32}$"
            },
            "escalation_policy_id": {
                "type": "string",
                "pattern": r"^P[A-Z0-9]{6}$"
            },
            "notification_settings": {
                "type": "object",
                "properties": {
                    "retry_attempts": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10
                    },
                    "retry_delay": {
                        "type": "integer",
                        "minimum": 30,
                        "maximum": 3600
                    },
                    "timeout": {
                        "type": "integer",
                        "minimum": 5,
                        "maximum": 300
                    },
                    "batch_size": {
                        "type": "integer", 
                        "minimum": 1,
                        "maximum": 100
                    }
                },
                "additionalProperties": False
            },
            "alerting_rules": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1, "maxLength": 255},
                        "severity": {
                            "type": "string",
                            "enum": ["critical", "high", "medium", "low", "info"]
                        },
                        "conditions": {"type": "object"},
                        "actions": {"type": "array"},
                        "enabled": {"type": "boolean"},
                        "throttle_minutes": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 1440
                        }
                    },
                    "required": ["name", "severity", "conditions"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["api_key"],
        "additionalProperties": False
    }
    
    EVENT_SCHEMA = {
        "type": "object",
        "properties": {
            "routing_key": {
                "type": "string",
                "pattern": r"^[a-f0-9]{32}$"
            },
            "event_action": {
                "type": "string",
                "enum": ["trigger", "acknowledge", "resolve"]
            },
            "dedup_key": {
                "type": "string",
                "maxLength": 255
            },
            "payload": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 1024
                    },
                    "source": {
                        "type": "string",
                        "maxLength": 255
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "error", "warning", "info"]
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time"
                    },
                    "component": {
                        "type": "string",
                        "maxLength": 255
                    },
                    "group": {
                        "type": "string",
                        "maxLength": 255
                    },
                    "class": {
                        "type": "string", 
                        "maxLength": 255
                    },
                    "custom_details": {
                        "type": "object"
                    }
                },
                "required": ["summary", "source", "severity"],
                "additionalProperties": False
            },
            "images": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "src": {"type": "string", "format": "uri"},
                        "href": {"type": "string", "format": "uri"},
                        "alt": {"type": "string", "maxLength": 512}
                    },
                    "required": ["src"],
                    "additionalProperties": False
                }
            },
            "links": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "href": {"type": "string", "format": "uri"},
                        "text": {"type": "string", "maxLength": 300}
                    },
                    "required": ["href"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["routing_key", "event_action"],
        "additionalProperties": False
    }
    
    @classmethod
    def validate_api_key(cls, api_key: str) -> ValidationResult:
        """Valide une clé API PagerDuty"""
        result = ValidationResult()
        
        if not api_key:
            result.add_error("API key is required")
            return result
        
        if not isinstance(api_key, str):
            result.add_error("API key must be a string")
            return result
        
        if len(api_key) < 20:
            result.add_error("API key is too short (minimum 20 characters)")
        
        if not cls.API_KEY_PATTERN.match(api_key):
            result.add_error("API key contains invalid characters")
        
        # Vérification de sécurité basique
        if api_key.lower() in ['test', 'demo', 'sample', 'example']:
            result.add_warning("API key appears to be a test/demo key")
        
        return result
    
    @classmethod
    def validate_service_id(cls, service_id: str) -> ValidationResult:
        """Valide un ID de service PagerDuty"""
        result = ValidationResult()
        
        if not service_id:
            result.add_error("Service ID is required")
            return result
        
        if not isinstance(service_id, str):
            result.add_error("Service ID must be a string")
            return result
        
        if not cls.SERVICE_ID_PATTERN.match(service_id):
            result.add_error("Service ID format is invalid (expected: P followed by 6 alphanumeric characters)")
        
        return result
    
    @classmethod
    def validate_integration_key(cls, integration_key: str) -> ValidationResult:
        """Valide une clé d'intégration PagerDuty"""
        result = ValidationResult()
        
        if not integration_key:
            result.add_error("Integration key is required")
            return result
        
        if not isinstance(integration_key, str):
            result.add_error("Integration key must be a string")
            return result
        
        if not cls.INTEGRATION_KEY_PATTERN.match(integration_key):
            result.add_error("Integration key format is invalid (expected: 32 hexadecimal characters)")
        
        return result
    
    @classmethod
    def validate_event_payload(cls, event: Dict[str, Any]) -> ValidationResult:
        """Valide un payload d'événement PagerDuty"""
        result = ValidationResult()
        
        try:
            validate(instance=event, schema=cls.EVENT_SCHEMA)
        except ValidationError as e:
            result.add_error(f"Event schema validation failed: {e.message}")
            return result
        
        # Validations supplémentaires
        payload = event.get("payload", {})
        
        # Vérifier la longueur du summary
        summary = payload.get("summary", "")
        if len(summary) > 1024:
            result.add_warning("Summary is very long and may be truncated")
        
        # Vérifier les custom_details
        custom_details = payload.get("custom_details", {})
        if custom_details:
            if len(json.dumps(custom_details)) > 10240:  # 10KB
                result.add_warning("Custom details are very large and may impact performance")
        
        # Vérifier les URLs dans les liens et images
        for link in event.get("links", []):
            url_result = cls.validate_url(link.get("href", ""))
            if not url_result.is_valid:
                result.add_error(f"Invalid link URL: {url_result.errors[0]}")
        
        for image in event.get("images", []):
            url_result = cls.validate_url(image.get("src", ""))
            if not url_result.is_valid:
                result.add_error(f"Invalid image URL: {url_result.errors[0]}")
        
        return result
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> ValidationResult:
        """Valide une configuration PagerDuty complète"""
        result = ValidationResult()
        
        try:
            validate(instance=config, schema=cls.PAGERDUTY_CONFIG_SCHEMA)
        except ValidationError as e:
            result.add_error(f"Configuration schema validation failed: {e.message}")
            return result
        
        # Validations spécifiques
        api_key_result = cls.validate_api_key(config.get("api_key", ""))
        result.merge(api_key_result)
        
        if "service_id" in config:
            service_result = cls.validate_service_id(config["service_id"])
            result.merge(service_result)
        
        if "integration_key" in config:
            integration_result = cls.validate_integration_key(config["integration_key"])
            result.merge(integration_result)
        
        # Validation des règles d'alerte
        alerting_rules = config.get("alerting_rules", [])
        for i, rule in enumerate(alerting_rules):
            rule_result = cls.validate_alerting_rule(rule)
            if not rule_result.is_valid:
                for error in rule_result.errors:
                    result.add_error(f"Alerting rule {i}: {error}")
        
        return result
    
    @classmethod
    def validate_alerting_rule(cls, rule: Dict[str, Any]) -> ValidationResult:
        """Valide une règle d'alerte"""
        result = ValidationResult()
        
        # Vérifier les conditions
        conditions = rule.get("conditions", {})
        if not conditions:
            result.add_error("Alerting rule must have conditions")
        
        # Vérifier la logique des conditions
        if isinstance(conditions, dict):
            cls._validate_condition_logic(conditions, result)
        
        # Vérifier les actions
        actions = rule.get("actions", [])
        if not actions:
            result.add_warning("Alerting rule has no actions defined")
        
        # Vérifier le throttling
        throttle = rule.get("throttle_minutes", 0)
        if throttle > 0 and throttle < 5:
            result.add_warning("Throttle period is very short, may cause spam")
        
        return result
    
    @classmethod
    def _validate_condition_logic(cls, conditions: Dict[str, Any], result: ValidationResult):
        """Valide la logique des conditions"""
        # Vérifications basiques de la logique des conditions
        logical_operators = ["and", "or", "not"]
        comparison_operators = ["eq", "ne", "gt", "gte", "lt", "lte", "in", "contains"]
        
        for key, value in conditions.items():
            if key in logical_operators:
                if not isinstance(value, (list, dict)):
                    result.add_error(f"Logical operator '{key}' requires list or dict value")
            elif key in comparison_operators:
                # Validation spécifique aux opérateurs de comparaison
                pass
            else:
                # Peut être un nom de champ ou autre
                pass
    
    @classmethod
    def validate_url(cls, url: str) -> ValidationResult:
        """Valide une URL"""
        result = ValidationResult()
        
        if not url:
            result.add_error("URL is required")
            return result
        
        try:
            parsed = urlparse(url)
            
            if not parsed.scheme:
                result.add_error("URL must have a scheme (http/https)")
            elif parsed.scheme not in ['http', 'https']:
                result.add_error("URL scheme must be http or https")
            
            if not parsed.netloc:
                result.add_error("URL must have a valid domain")
            
            # Vérification de sécurité basique
            if parsed.netloc in ['localhost', '127.0.0.1', '0.0.0.0']:
                result.add_warning("URL points to localhost")
            
            # Vérifier les IP privées
            try:
                ip = ipaddress.ip_address(parsed.hostname)
                if ip.is_private:
                    result.add_warning("URL points to private IP address")
            except (ValueError, TypeError):
                # Pas une IP, c'est normal
                pass
                
        except Exception as e:
            result.add_error(f"Invalid URL format: {str(e)}")
        
        return result
    
    @classmethod
    def validate_email(cls, email: str) -> ValidationResult:
        """Valide une adresse email"""
        result = ValidationResult()
        
        if not email:
            result.add_error("Email is required")
            return result
        
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        
        if not email_pattern.match(email):
            result.add_error("Invalid email format")
        
        if len(email) > 254:
            result.add_error("Email is too long")
        
        # Vérifications de sécurité
        dangerous_patterns = [
            r'[<>"\']',  # Caractères potentiellement dangereux
            r'javascript:',  # JavaScript injection
            r'data:',  # Data URLs
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, email, re.IGNORECASE):
                result.add_error("Email contains potentially dangerous characters")
                break
        
        return result
    
    @classmethod
    def validate_phone_number(cls, phone: str) -> ValidationResult:
        """Valide un numéro de téléphone"""
        result = ValidationResult()
        
        if not phone:
            result.add_error("Phone number is required")
            return result
        
        # Nettoyer le numéro (supprimer espaces, tirets, parenthèses)
        clean_phone = re.sub(r'[\s\-\(\)\.]', '', phone)
        
        # Vérifier le format international basique
        if not re.match(r'^\+?[\d]{7,15}$', clean_phone):
            result.add_error("Invalid phone number format")
        
        if len(clean_phone) < 7:
            result.add_error("Phone number is too short")
        elif len(clean_phone) > 15:
            result.add_error("Phone number is too long")
        
        return result

class SecurityValidator:
    """Validateur de sécurité"""
    
    # Patterns de sécurité
    SQL_INJECTION_PATTERNS = [
        r"';\s*(DROP|DELETE|INSERT|UPDATE|CREATE|ALTER)",
        r"UNION\s+SELECT",
        r"1=1|1=0",
        r"OR\s+1=1",
        r"AND\s+1=1"
    ]
    
    XSS_PATTERNS = [
        r"<script",
        r"javascript:",
        r"onload=",
        r"onerror=",
        r"eval\s*\(",
        r"document\.cookie"
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        r";\s*(rm|del|format|shutdown)",
        r"\||\&\&|\|\|",
        r"`.*`",
        r"\$\(.*\)"
    ]
    
    @classmethod
    def check_sql_injection(cls, text: str) -> ValidationResult:
        """Vérifie les tentatives d'injection SQL"""
        result = ValidationResult()
        
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                result.add_error("Potential SQL injection detected")
                break
        
        return result
    
    @classmethod
    def check_xss(cls, text: str) -> ValidationResult:
        """Vérifie les tentatives XSS"""
        result = ValidationResult()
        
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                result.add_error("Potential XSS attack detected")
                break
        
        return result
    
    @classmethod
    def check_command_injection(cls, text: str) -> ValidationResult:
        """Vérifie les tentatives d'injection de commandes"""
        result = ValidationResult()
        
        for pattern in cls.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                result.add_error("Potential command injection detected")
                break
        
        return result
    
    @classmethod
    def validate_input_security(cls, text: str) -> ValidationResult:
        """Validation de sécurité complète pour les entrées"""
        result = ValidationResult()
        
        # Vérifier toutes les menaces
        sql_result = cls.check_sql_injection(text)
        xss_result = cls.check_xss(text)
        cmd_result = cls.check_command_injection(text)
        
        result.merge(sql_result)
        result.merge(xss_result) 
        result.merge(cmd_result)
        
        # Vérifications additionnelles
        if len(text) > 10000:
            result.add_warning("Input is very large, potential DoS risk")
        
        # Vérifier les caractères de contrôle
        if re.search(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', text):
            result.add_warning("Input contains control characters")
        
        return result

class PerformanceValidator:
    """Validateur de performance"""
    
    @classmethod
    def validate_config_performance(cls, config: Dict[str, Any]) -> ValidationResult:
        """Valide les aspects performance d'une configuration"""
        result = ValidationResult()
        
        notification_settings = config.get("notification_settings", {})
        
        # Vérifier les timeouts
        timeout = notification_settings.get("timeout", 30)
        if timeout > 120:
            result.add_warning("Timeout is very high, may impact user experience")
        elif timeout < 5:
            result.add_warning("Timeout is very low, may cause failures")
        
        # Vérifier les retry settings
        retry_attempts = notification_settings.get("retry_attempts", 3)
        retry_delay = notification_settings.get("retry_delay", 60)
        
        if retry_attempts > 5:
            result.add_warning("Too many retry attempts may cause delays")
        
        if retry_delay > 300:
            result.add_warning("Retry delay is very high")
        
        # Vérifier le batch size
        batch_size = notification_settings.get("batch_size", 10)
        if batch_size > 50:
            result.add_warning("Large batch size may impact API rate limits")
        
        # Vérifier le nombre de règles d'alerte
        alerting_rules = config.get("alerting_rules", [])
        if len(alerting_rules) > 100:
            result.add_warning("Many alerting rules may impact performance")
        
        return result

# Fonctions utilitaires
def validate_json_schema(data: Any, schema: Dict[str, Any]) -> ValidationResult:
    """Valide des données contre un schéma JSON"""
    result = ValidationResult()
    
    try:
        validate(instance=data, schema=schema)
    except ValidationError as e:
        result.add_error(f"Schema validation failed: {e.message}")
    
    return result

def validate_yaml_format(yaml_string: str) -> ValidationResult:
    """Valide le format YAML"""
    result = ValidationResult()
    
    try:
        yaml.safe_load(yaml_string)
    except yaml.YAMLError as e:
        result.add_error(f"Invalid YAML format: {str(e)}")
    
    return result

def validate_json_format(json_string: str) -> ValidationResult:
    """Valide le format JSON"""
    result = ValidationResult()
    
    try:
        json.loads(json_string)
    except json.JSONDecodeError as e:
        result.add_error(f"Invalid JSON format: {str(e)}")
    
    return result

# Export des classes et fonctions principales
__all__ = [
    "ValidationResult",
    "PagerDutyValidator", 
    "SecurityValidator",
    "PerformanceValidator",
    "validate_json_schema",
    "validate_yaml_format",
    "validate_json_format"
]
