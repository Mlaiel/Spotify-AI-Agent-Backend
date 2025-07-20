"""
Validateurs avancés pour la configuration Alertmanager Receivers

Ce module fournit une validation complète et robuste de toutes
les configurations, schemas et données d'entrée.

Author: Spotify AI Agent Team
Maintainer: Fahed Mlaiel - Spécialiste Sécurité Backend
"""

import logging
import re
import json
import yaml
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from pathlib import Path
import jsonschema
from urllib.parse import urlparse
import ipaddress

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Niveaux de validation disponibles"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"

class ValidationResult(Enum):
    """Résultats de validation"""
    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationIssue:
    """Représente un problème de validation"""
    level: ValidationResult
    message: str
    field: str
    value: Any = None
    suggestion: Optional[str] = None
    error_code: Optional[str] = None

@dataclass
class ValidationReport:
    """Rapport de validation complet"""
    is_valid: bool
    issues: List[ValidationIssue]
    warnings_count: int = 0
    errors_count: int = 0
    critical_count: int = 0
    
    def __post_init__(self):
        self.warnings_count = sum(1 for issue in self.issues if issue.level == ValidationResult.WARNING)
        self.errors_count = sum(1 for issue in self.issues if issue.level == ValidationResult.ERROR)
        self.critical_count = sum(1 for issue in self.issues if issue.level == ValidationResult.CRITICAL)
        self.is_valid = self.errors_count == 0 and self.critical_count == 0

class ConfigValidator:
    """Validateur principal de configuration"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.issues: List[ValidationIssue] = []
        
        # Patterns de validation
        self.email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        self.url_pattern = re.compile(
            r'^https?://(?:[-\w.])+(?::[0-9]+)?(?:/[^?\s]*)?(?:\?[^#\s]*)?(?:#[^\s]*)?$'
        )
        self.webhook_pattern = re.compile(
            r'^https://hooks\.slack\.com/services/[A-Z0-9]+/[A-Z0-9]+/[a-zA-Z0-9]+$'
        )
        
    def validate_receiver_config(self, config: Dict[str, Any]) -> ValidationReport:
        """Valide une configuration de receiver"""
        self.issues = []
        
        # Validation de la structure globale
        self._validate_global_structure(config)
        
        # Validation des tenants
        if "tenants" in config:
            self._validate_tenants_config(config["tenants"])
        
        # Validation de la configuration globale
        if "global" in config:
            self._validate_global_config(config["global"])
        
        return self._generate_report()
    
    def _validate_global_structure(self, config: Dict[str, Any]):
        """Valide la structure globale du fichier de configuration"""
        required_sections = ["global", "tenants"]
        
        for section in required_sections:
            if section not in config:
                self.issues.append(ValidationIssue(
                    level=ValidationResult.ERROR,
                    message=f"Section requise manquante: {section}",
                    field=section,
                    error_code="MISSING_REQUIRED_SECTION"
                ))
        
        # Validation des sections inattendues
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            allowed_sections = ["global", "tenants", "templates", "routes"]
            for section in config.keys():
                if section not in allowed_sections:
                    self.issues.append(ValidationIssue(
                        level=ValidationResult.WARNING,
                        message=f"Section non reconnue: {section}",
                        field=section,
                        suggestion="Vérifiez l'orthographe ou supprimez cette section"
                    ))
    
    def _validate_global_config(self, global_config: Dict[str, Any]):
        """Valide la configuration globale"""
        
        # Validation du timeout de résolution
        if "resolve_timeout" in global_config:
            timeout = global_config["resolve_timeout"]
            if not self._is_valid_duration(timeout):
                self.issues.append(ValidationIssue(
                    level=ValidationResult.ERROR,
                    message="Format de timeout invalide",
                    field="global.resolve_timeout",
                    value=timeout,
                    suggestion="Utilisez un format comme '5m', '30s', '1h'"
                ))
        
        # Validation SMTP
        if "smtp_smarthost" in global_config:
            smarthost = global_config["smtp_smarthost"]
            if not self._is_valid_smtp_host(smarthost):
                self.issues.append(ValidationIssue(
                    level=ValidationResult.ERROR,
                    message="Format de smarthost SMTP invalide",
                    field="global.smtp_smarthost",
                    value=smarthost,
                    suggestion="Format attendu: 'host:port'"
                ))
        
        # Validation email FROM
        if "smtp_from" in global_config:
            from_email = global_config["smtp_from"]
            if not self.email_pattern.match(from_email):
                self.issues.append(ValidationIssue(
                    level=ValidationResult.ERROR,
                    message="Adresse email FROM invalide",
                    field="global.smtp_from",
                    value=from_email
                ))
        
        # Validation des URLs d'API
        api_urls = ["slack_api_url", "pagerduty_url", "opsgenie_api_url"]
        for url_field in api_urls:
            if url_field in global_config:
                url = global_config[url_field]
                if not self._is_valid_url(url):
                    self.issues.append(ValidationIssue(
                        level=ValidationResult.ERROR,
                        message=f"URL invalide pour {url_field}",
                        field=f"global.{url_field}",
                        value=url
                    ))
    
    def _validate_tenants_config(self, tenants_config: Dict[str, Any]):
        """Valide la configuration des tenants"""
        
        if not tenants_config:
            self.issues.append(ValidationIssue(
                level=ValidationResult.ERROR,
                message="Aucun tenant configuré",
                field="tenants",
                error_code="NO_TENANTS_CONFIGURED"
            ))
            return
        
        for tenant_name, tenant_config in tenants_config.items():
            self._validate_single_tenant(tenant_name, tenant_config)
    
    def _validate_single_tenant(self, tenant_name: str, tenant_config: Dict[str, Any]):
        """Valide la configuration d'un tenant spécifique"""
        
        # Validation du nom de tenant
        if not self._is_valid_tenant_name(tenant_name):
            self.issues.append(ValidationIssue(
                level=ValidationResult.ERROR,
                message="Nom de tenant invalide",
                field=f"tenants.{tenant_name}",
                value=tenant_name,
                suggestion="Utilisez uniquement des lettres, chiffres et tirets"
            ))
        
        # Validation des métadonnées
        if "metadata" in tenant_config:
            self._validate_tenant_metadata(tenant_name, tenant_config["metadata"])
        
        # Validation des receivers
        if "receivers" in tenant_config:
            self._validate_tenant_receivers(tenant_name, tenant_config["receivers"])
        else:
            self.issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                message="Aucun receiver configuré pour ce tenant",
                field=f"tenants.{tenant_name}.receivers"
            ))
    
    def _validate_tenant_metadata(self, tenant_name: str, metadata: Dict[str, Any]):
        """Valide les métadonnées d'un tenant"""
        
        required_fields = ["name", "tier", "contact_team"]
        for field in required_fields:
            if field not in metadata:
                self.issues.append(ValidationIssue(
                    level=ValidationResult.WARNING,
                    message=f"Métadonnée manquante: {field}",
                    field=f"tenants.{tenant_name}.metadata.{field}"
                ))
        
        # Validation de l'email de contact
        if "contact_team" in metadata:
            contact = metadata["contact_team"]
            if "@" in contact and not self.email_pattern.match(contact):
                self.issues.append(ValidationIssue(
                    level=ValidationResult.ERROR,
                    message="Email de contact invalide",
                    field=f"tenants.{tenant_name}.metadata.contact_team",
                    value=contact
                ))
        
        # Validation du niveau SLA
        if "sla_level" in metadata:
            sla = metadata["sla_level"]
            if not self._is_valid_sla_level(sla):
                self.issues.append(ValidationIssue(
                    level=ValidationResult.WARNING,
                    message="Niveau SLA non standard",
                    field=f"tenants.{tenant_name}.metadata.sla_level",
                    value=sla,
                    suggestion="Utilisez un format comme '99.9%' ou '99.99%'"
                ))
    
    def _validate_tenant_receivers(self, tenant_name: str, receivers: List[Dict[str, Any]]):
        """Valide les receivers d'un tenant"""
        
        receiver_names = set()
        
        for i, receiver in enumerate(receivers):
            # Validation de la structure du receiver
            if "name" not in receiver:
                self.issues.append(ValidationIssue(
                    level=ValidationResult.ERROR,
                    message="Nom de receiver manquant",
                    field=f"tenants.{tenant_name}.receivers[{i}].name"
                ))
                continue
            
            receiver_name = receiver["name"]
            
            # Vérification des doublons
            if receiver_name in receiver_names:
                self.issues.append(ValidationIssue(
                    level=ValidationResult.ERROR,
                    message=f"Nom de receiver dupliqué: {receiver_name}",
                    field=f"tenants.{tenant_name}.receivers[{i}].name",
                    value=receiver_name
                ))
            receiver_names.add(receiver_name)
            
            # Validation du type de canal
            if "channel_type" not in receiver:
                self.issues.append(ValidationIssue(
                    level=ValidationResult.ERROR,
                    message="Type de canal manquant",
                    field=f"tenants.{tenant_name}.receivers[{i}].channel_type"
                ))
            else:
                channel_type = receiver["channel_type"]
                if not self._is_valid_channel_type(channel_type):
                    self.issues.append(ValidationIssue(
                        level=ValidationResult.ERROR,
                        message=f"Type de canal non supporté: {channel_type}",
                        field=f"tenants.{tenant_name}.receivers[{i}].channel_type",
                        value=channel_type,
                        suggestion="Types supportés: slack, email, pagerduty, webhook"
                    ))
            
            # Validation de la configuration spécifique
            if "config" in receiver:
                self._validate_receiver_config_section(
                    tenant_name, i, receiver["channel_type"], receiver["config"]
                )
    
    def _validate_receiver_config_section(
        self, 
        tenant_name: str, 
        receiver_index: int, 
        channel_type: str, 
        config: Dict[str, Any]
    ):
        """Valide la section config d'un receiver"""
        
        if channel_type == "slack":
            self._validate_slack_config(tenant_name, receiver_index, config)
        elif channel_type == "email":
            self._validate_email_config(tenant_name, receiver_index, config)
        elif channel_type == "pagerduty":
            self._validate_pagerduty_config(tenant_name, receiver_index, config)
        elif channel_type == "webhook":
            self._validate_webhook_config(tenant_name, receiver_index, config)
    
    def _validate_slack_config(self, tenant_name: str, receiver_index: int, config: Dict[str, Any]):
        """Valide une configuration Slack"""
        
        # Validation de l'URL du webhook
        if "webhook_url" in config:
            webhook_url = config["webhook_url"]
            if not self.webhook_pattern.match(webhook_url):
                self.issues.append(ValidationIssue(
                    level=ValidationResult.ERROR,
                    message="URL webhook Slack invalide",
                    field=f"tenants.{tenant_name}.receivers[{receiver_index}].config.webhook_url",
                    value=webhook_url
                ))
        
        # Validation du canal
        if "channel" in config:
            channel = config["channel"]
            if not channel.startswith("#") and not channel.startswith("@"):
                self.issues.append(ValidationIssue(
                    level=ValidationResult.WARNING,
                    message="Le canal Slack devrait commencer par # ou @",
                    field=f"tenants.{tenant_name}.receivers[{receiver_index}].config.channel",
                    value=channel
                ))
    
    def _validate_email_config(self, tenant_name: str, receiver_index: int, config: Dict[str, Any]):
        """Valide une configuration email"""
        
        # Validation des destinataires
        if "to" in config:
            recipients = config["to"]
            if isinstance(recipients, str):
                recipients = [recipients]
            
            for recipient in recipients:
                if not self.email_pattern.match(recipient):
                    self.issues.append(ValidationIssue(
                        level=ValidationResult.ERROR,
                        message=f"Adresse email invalide: {recipient}",
                        field=f"tenants.{tenant_name}.receivers[{receiver_index}].config.to",
                        value=recipient
                    ))
    
    def _validate_pagerduty_config(self, tenant_name: str, receiver_index: int, config: Dict[str, Any]):
        """Valide une configuration PagerDuty"""
        
        # Validation de la clé d'intégration
        if "integration_key" not in config:
            self.issues.append(ValidationIssue(
                level=ValidationResult.ERROR,
                message="Clé d'intégration PagerDuty manquante",
                field=f"tenants.{tenant_name}.receivers[{receiver_index}].config.integration_key"
            ))
        else:
            integration_key = config["integration_key"]
            if not self._is_valid_pagerduty_key(integration_key):
                self.issues.append(ValidationIssue(
                    level=ValidationResult.ERROR,
                    message="Format de clé d'intégration PagerDuty invalide",
                    field=f"tenants.{tenant_name}.receivers[{receiver_index}].config.integration_key",
                    value=integration_key
                ))
    
    def _validate_webhook_config(self, tenant_name: str, receiver_index: int, config: Dict[str, Any]):
        """Valide une configuration webhook"""
        
        # Validation de l'URL
        if "url" not in config:
            self.issues.append(ValidationIssue(
                level=ValidationResult.ERROR,
                message="URL webhook manquante",
                field=f"tenants.{tenant_name}.receivers[{receiver_index}].config.url"
            ))
        else:
            url = config["url"]
            if not self._is_valid_url(url):
                self.issues.append(ValidationIssue(
                    level=ValidationResult.ERROR,
                    message="URL webhook invalide",
                    field=f"tenants.{tenant_name}.receivers[{receiver_index}].config.url",
                    value=url
                ))
    
    # Méthodes de validation utilitaires
    def _is_valid_duration(self, duration: str) -> bool:
        """Valide un format de durée"""
        pattern = re.compile(r'^\d+[smhd]$')
        return bool(pattern.match(duration))
    
    def _is_valid_smtp_host(self, host: str) -> bool:
        """Valide un format de host SMTP"""
        if ":" not in host:
            return False
        
        hostname, port = host.rsplit(":", 1)
        try:
            port_num = int(port)
            return 1 <= port_num <= 65535 and len(hostname) > 0
        except ValueError:
            return False
    
    def _is_valid_url(self, url: str) -> bool:
        """Valide une URL"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _is_valid_tenant_name(self, name: str) -> bool:
        """Valide un nom de tenant"""
        pattern = re.compile(r'^[a-zA-Z0-9-_]+$')
        return bool(pattern.match(name)) and len(name) <= 50
    
    def _is_valid_sla_level(self, sla: str) -> bool:
        """Valide un niveau SLA"""
        pattern = re.compile(r'^\d{2,3}\.\d{1,2}%$')
        return bool(pattern.match(sla))
    
    def _is_valid_channel_type(self, channel_type: str) -> bool:
        """Valide un type de canal"""
        valid_types = ["slack", "email", "pagerduty", "webhook", "teams", "discord"]
        return channel_type in valid_types
    
    def _is_valid_pagerduty_key(self, key: str) -> bool:
        """Valide une clé PagerDuty"""
        # Les clés PagerDuty sont des chaînes alphanumériques de 32 caractères
        pattern = re.compile(r'^[a-zA-Z0-9]{32}$')
        return bool(pattern.match(key))
    
    def _generate_report(self) -> ValidationReport:
        """Génère le rapport de validation final"""
        return ValidationReport(
            is_valid=not any(issue.level in [ValidationResult.ERROR, ValidationResult.CRITICAL] 
                           for issue in self.issues),
            issues=self.issues
        )
    
    @staticmethod
    def validate_dependencies():
        """Valide que toutes les dépendances sont disponibles"""
        try:
            import jsonschema
            import yaml
            import prometheus_client
            logger.info("All required dependencies are available")
            return True
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            return False

class SchemaValidator:
    """Validateur de schémas JSON/YAML"""
    
    def __init__(self):
        self.schemas = self._load_schemas()
    
    def _load_schemas(self) -> Dict[str, Dict]:
        """Charge les schémas de validation"""
        
        # Schéma pour la configuration des receivers
        receiver_schema = {
            "type": "object",
            "properties": {
                "global": {
                    "type": "object",
                    "properties": {
                        "resolve_timeout": {"type": "string", "pattern": r"^\d+[smhd]$"},
                        "smtp_smarthost": {"type": "string"},
                        "smtp_from": {"type": "string", "format": "email"},
                        "smtp_require_tls": {"type": "boolean"}
                    },
                    "required": ["resolve_timeout"]
                },
                "tenants": {
                    "type": "object",
                    "patternProperties": {
                        "^[a-zA-Z0-9-_]+$": {
                            "type": "object",
                            "properties": {
                                "metadata": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "tier": {"type": "string"},
                                        "contact_team": {"type": "string"}
                                    },
                                    "required": ["name", "tier"]
                                },
                                "receivers": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "channel_type": {
                                                "type": "string",
                                                "enum": ["slack", "email", "pagerduty", "webhook"]
                                            },
                                            "enabled": {"type": "boolean"},
                                            "config": {"type": "object"}
                                        },
                                        "required": ["name", "channel_type"]
                                    }
                                }
                            },
                            "required": ["metadata", "receivers"]
                        }
                    }
                }
            },
            "required": ["global", "tenants"]
        }
        
        return {
            "receiver_config": receiver_schema
        }
    
    def validate_against_schema(self, data: Dict[str, Any], schema_name: str) -> ValidationReport:
        """Valide des données contre un schéma"""
        
        if schema_name not in self.schemas:
            return ValidationReport(
                is_valid=False,
                issues=[ValidationIssue(
                    level=ValidationResult.ERROR,
                    message=f"Schéma inconnu: {schema_name}",
                    field="schema"
                )]
            )
        
        schema = self.schemas[schema_name]
        issues = []
        
        try:
            jsonschema.validate(data, schema)
            is_valid = True
        except jsonschema.ValidationError as e:
            is_valid = False
            issues.append(ValidationIssue(
                level=ValidationResult.ERROR,
                message=str(e.message),
                field=".".join(str(path) for path in e.absolute_path),
                value=e.instance,
                error_code="SCHEMA_VALIDATION_ERROR"
            ))
        except jsonschema.SchemaError as e:
            is_valid = False
            issues.append(ValidationIssue(
                level=ValidationResult.CRITICAL,
                message=f"Erreur de schéma: {e.message}",
                field="schema",
                error_code="INVALID_SCHEMA"
            ))
        
        return ValidationReport(is_valid=is_valid, issues=issues)

# Instances par défaut
config_validator = ConfigValidator()
schema_validator = SchemaValidator()
