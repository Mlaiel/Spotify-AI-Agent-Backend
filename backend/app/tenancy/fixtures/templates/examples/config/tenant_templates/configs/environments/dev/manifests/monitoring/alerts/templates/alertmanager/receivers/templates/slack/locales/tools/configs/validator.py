"""
Validateur de configuration avancé pour le système de monitoring Slack.

Ce module fournit un système de validation multicouche avec:
- Validation syntaxique (format, types)
- Validation sémantique (cohérence, dépendances)
- Validation métier (règles business, contraintes)
- Validation de sécurité (valeurs sensibles, permissions)
- Génération de rapports détaillés avec suggestions de correction

Architecture:
    - Pattern Strategy pour différents types de validation
    - Builder pattern pour la construction des validateurs
    - Visitor pattern pour le parcours des configurations
    - Chain of Responsibility pour les règles de validation

Fonctionnalités:
    - Validation stricte et permissive
    - Support de schémas JSON Schema
    - Validation contextuelle par environnement
    - Règles personnalisées par tenant
    - Cache des résultats de validation
    - Métriques de performance intégrées

Auteur: Équipe Spotify AI Agent
Lead Developer: Fahed Mlaiel
"""

import re
import json
import ipaddress
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable, Pattern

import jsonschema
from jsonschema import Draft7Validator, FormatChecker

from .metrics import MetricsCollector


class ValidationLevel(Enum):
    """Niveaux de validation."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"


class ValidationCategory(Enum):
    """Catégories de validation."""
    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    BUSINESS = "business"
    SECURITY = "security"
    PERFORMANCE = "performance"


@dataclass
class ValidationIssue:
    """Issue de validation."""
    level: ValidationLevel
    category: ValidationCategory
    message: str
    path: str
    code: str
    suggestion: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    context: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        """Représentation textuelle de l'issue."""
        parts = [f"[{self.level.value.upper()}]"]
        if self.path:
            parts.append(f"({self.path})")
        parts.append(self.message)
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " ".join(parts)


@dataclass
class ValidationResult:
    """Résultat de validation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_time_ms: float = 0.0
    
    @property
    def errors(self) -> List[ValidationIssue]:
        """Issues de niveau erreur."""
        return [issue for issue in self.issues if issue.level == ValidationLevel.ERROR]
    
    @property
    def warnings(self) -> List[ValidationIssue]:
        """Issues de niveau avertissement."""
        return [issue for issue in self.issues if issue.level == ValidationLevel.WARNING]
    
    @property
    def has_errors(self) -> bool:
        """Indique s'il y a des erreurs."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Indique s'il y a des avertissements."""
        return len(self.warnings) > 0
    
    def add_issue(self, issue: ValidationIssue) -> None:
        """Ajoute une issue."""
        self.issues.append(issue)
        if issue.level == ValidationLevel.ERROR:
            self.is_valid = False
    
    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """Fusionne avec un autre résultat."""
        merged = ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            issues=self.issues + other.issues,
            metadata={**self.metadata, **other.metadata},
            validation_time_ms=self.validation_time_ms + other.validation_time_ms
        )
        return merged
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire."""
        return {
            "is_valid": self.is_valid,
            "has_errors": self.has_errors,
            "has_warnings": self.has_warnings,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "total_issues": len(self.issues),
            "issues": [
                {
                    "level": issue.level.value,
                    "category": issue.category.value,
                    "message": issue.message,
                    "path": issue.path,
                    "code": issue.code,
                    "suggestion": issue.suggestion,
                    "line_number": issue.line_number,
                    "column_number": issue.column_number,
                    "context": issue.context
                }
                for issue in self.issues
            ],
            "metadata": self.metadata,
            "validation_time_ms": self.validation_time_ms
        }


class IValidator(ABC):
    """Interface pour les validateurs."""
    
    @abstractmethod
    def validate(self, config: Dict[str, Any], context: Dict[str, Any] = None) -> ValidationResult:
        """Valide une configuration."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Retourne le nom du validateur."""
        pass


class SchemaValidator(IValidator):
    """Validateur basé sur JSON Schema."""
    
    def __init__(self, schema: Dict[str, Any], name: str = "schema"):
        self.schema = schema
        self.name = name
        self.validator = Draft7Validator(schema, format_checker=FormatChecker())
        self._metrics = MetricsCollector()
    
    def validate(self, config: Dict[str, Any], context: Dict[str, Any] = None) -> ValidationResult:
        """Valide contre le schéma JSON."""
        result = ValidationResult(is_valid=True)
        
        try:
            errors = sorted(self.validator.iter_errors(config), key=lambda e: e.path)
            
            for error in errors:
                path = ".".join(str(p) for p in error.absolute_path)
                
                issue = ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.SYNTAX,
                    message=error.message,
                    path=path,
                    code="SCHEMA_VIOLATION",
                    suggestion=self._generate_suggestion(error)
                )
                
                result.add_issue(issue)
            
            self._metrics.increment("schema_validation_completed")
            
        except Exception as e:
            issue = ValidationIssue(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.SYNTAX,
                message=f"Erreur de validation du schéma: {e}",
                path="",
                code="SCHEMA_ERROR"
            )
            result.add_issue(issue)
            self._metrics.increment("schema_validation_error")
        
        return result
    
    def get_name(self) -> str:
        return self.name
    
    def _generate_suggestion(self, error: jsonschema.ValidationError) -> Optional[str]:
        """Génère une suggestion de correction."""
        if error.validator == "required":
            missing_props = error.validator_value
            return f"Ajoutez les propriétés requises: {', '.join(missing_props)}"
        elif error.validator == "type":
            expected_type = error.validator_value
            return f"La valeur doit être de type {expected_type}"
        elif error.validator == "enum":
            valid_values = error.validator_value
            return f"Valeurs autorisées: {', '.join(map(str, valid_values))}"
        elif error.validator == "minimum":
            minimum = error.validator_value
            return f"La valeur doit être >= {minimum}"
        elif error.validator == "maximum":
            maximum = error.validator_value
            return f"La valeur doit être <= {maximum}"
        
        return None


class BusinessRuleValidator(IValidator):
    """Validateur de règles métier."""
    
    def __init__(self, name: str = "business"):
        self.name = name
        self._metrics = MetricsCollector()
        self._rules: List[Callable[[Dict[str, Any], Dict[str, Any]], ValidationResult]] = []
        
        # Enregistrement des règles par défaut
        self._register_default_rules()
    
    def validate(self, config: Dict[str, Any], context: Dict[str, Any] = None) -> ValidationResult:
        """Valide les règles métier."""
        result = ValidationResult(is_valid=True)
        context = context or {}
        
        for rule in self._rules:
            try:
                rule_result = rule(config, context)
                result = result.merge(rule_result)
                
            except Exception as e:
                issue = ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.BUSINESS,
                    message=f"Erreur lors de l'exécution d'une règle métier: {e}",
                    path="",
                    code="RULE_EXECUTION_ERROR"
                )
                result.add_issue(issue)
        
        self._metrics.increment("business_validation_completed")
        return result
    
    def get_name(self) -> str:
        return self.name
    
    def add_rule(self, rule: Callable[[Dict[str, Any], Dict[str, Any]], ValidationResult]) -> None:
        """Ajoute une règle de validation."""
        self._rules.append(rule)
    
    def _register_default_rules(self) -> None:
        """Enregistre les règles par défaut."""
        self.add_rule(self._validate_redis_config)
        self.add_rule(self._validate_slack_config)
        self.add_rule(self._validate_tenant_config)
        self.add_rule(self._validate_monitoring_config)
        self.add_rule(self._validate_security_config)
        self.add_rule(self._validate_performance_config)
    
    def _validate_redis_config(self, config: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Valide la configuration Redis."""
        result = ValidationResult(is_valid=True)
        
        redis_config = config.get("redis", {})
        if not redis_config:
            return result
        
        # Validation du host
        host = redis_config.get("host")
        if host:
            if not self._is_valid_hostname_or_ip(host):
                issue = ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.BUSINESS,
                    message=f"Host Redis invalide: {host}",
                    path="redis.host",
                    code="INVALID_REDIS_HOST",
                    suggestion="Utilisez un hostname valide ou une adresse IP"
                )
                result.add_issue(issue)
        
        # Validation du port
        port = redis_config.get("port")
        if port and not (1 <= port <= 65535):
            issue = ValidationIssue(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.BUSINESS,
                message=f"Port Redis invalide: {port}",
                path="redis.port",
                code="INVALID_REDIS_PORT",
                suggestion="Utilisez un port entre 1 et 65535"
            )
            result.add_issue(issue)
        
        # Validation des connexions
        max_conn = redis_config.get("max_connections", 0)
        min_conn = redis_config.get("min_connections", 0)
        
        if max_conn > 0 and min_conn > 0 and min_conn > max_conn:
            issue = ValidationIssue(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.BUSINESS,
                message="min_connections ne peut pas être supérieur à max_connections",
                path="redis.connections",
                code="INVALID_CONNECTION_CONFIG",
                suggestion="Ajustez les valeurs min_connections et max_connections"
            )
            result.add_issue(issue)
        
        return result
    
    def _validate_slack_config(self, config: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Valide la configuration Slack."""
        result = ValidationResult(is_valid=True)
        
        slack_config = config.get("slack", {})
        if not slack_config:
            return result
        
        # Validation des locales
        supported_locales = slack_config.get("supported_locales", [])
        default_locale = slack_config.get("default_locale")
        
        if default_locale and default_locale not in supported_locales:
            issue = ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.BUSINESS,
                message=f"Locale par défaut '{default_locale}' non dans la liste supportée",
                path="slack.default_locale",
                code="INVALID_DEFAULT_LOCALE",
                suggestion=f"Ajoutez '{default_locale}' à supported_locales ou changez default_locale"
            )
            result.add_issue(issue)
        
        # Validation des rate limits
        rate_limit = slack_config.get("rate_limit", {})
        if rate_limit:
            rpm = rate_limit.get("requests_per_minute", 0)
            burst = rate_limit.get("burst_limit", 0)
            
            if rpm > 0 and burst > rpm:
                issue = ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.BUSINESS,
                    message="burst_limit supérieur à requests_per_minute peut causer des problèmes",
                    path="slack.rate_limit",
                    code="SUSPICIOUS_RATE_LIMIT",
                    suggestion="Assurez-vous que burst_limit <= requests_per_minute"
                )
                result.add_issue(issue)
        
        return result
    
    def _validate_tenant_config(self, config: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Valide la configuration tenant."""
        result = ValidationResult(is_valid=True)
        
        tenant_config = config.get("tenant", {})
        if not tenant_config:
            return result
        
        # Validation de l'isolation level
        isolation = tenant_config.get("isolation_level")
        valid_levels = ["strict", "partial", "shared"]
        
        if isolation and isolation not in valid_levels:
            issue = ValidationIssue(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.BUSINESS,
                message=f"Niveau d'isolation invalide: {isolation}",
                path="tenant.isolation_level",
                code="INVALID_ISOLATION_LEVEL",
                suggestion=f"Utilisez une des valeurs: {', '.join(valid_levels)}"
            )
            result.add_issue(issue)
        
        return result
    
    def _validate_monitoring_config(self, config: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Valide la configuration monitoring."""
        result = ValidationResult(is_valid=True)
        
        monitoring_config = config.get("monitoring", {})
        if not monitoring_config:
            return result
        
        # Validation des seuils
        thresholds = monitoring_config.get("alert_thresholds", {})
        for metric_name, threshold in thresholds.items():
            if isinstance(threshold, (int, float)):
                if threshold < 0 or threshold > 100:
                    issue = ValidationIssue(
                        level=ValidationLevel.WARNING,
                        category=ValidationCategory.BUSINESS,
                        message=f"Seuil suspect pour {metric_name}: {threshold}%",
                        path=f"monitoring.alert_thresholds.{metric_name}",
                        code="SUSPICIOUS_THRESHOLD",
                        suggestion="Les seuils sont généralement entre 0 et 100%"
                    )
                    result.add_issue(issue)
        
        return result
    
    def _validate_security_config(self, config: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Valide la configuration sécurité."""
        result = ValidationResult(is_valid=True)
        
        # Vérifications de sécurité selon l'environnement
        environment = context.get("environment", "dev")
        
        if environment == "production":
            # En production, SSL doit être activé
            redis_config = config.get("redis", {})
            if not redis_config.get("ssl", False):
                issue = ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.SECURITY,
                    message="SSL doit être activé en production",
                    path="redis.ssl",
                    code="SSL_REQUIRED_PRODUCTION",
                    suggestion="Activez ssl: true pour l'environnement de production"
                )
                result.add_issue(issue)
        
        return result
    
    def _validate_performance_config(self, config: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """Valide la configuration performance."""
        result = ValidationResult(is_valid=True)
        
        # Validation des paramètres de performance
        performance_config = config.get("performance", {})
        if performance_config:
            concurrent_requests = performance_config.get("concurrent_requests", 0)
            if concurrent_requests > 1000:
                issue = ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.PERFORMANCE,
                    message=f"Nombre élevé de requêtes concurrentes: {concurrent_requests}",
                    path="performance.concurrent_requests",
                    code="HIGH_CONCURRENCY",
                    suggestion="Assurez-vous que votre infrastructure peut gérer cette charge"
                )
                result.add_issue(issue)
        
        return result
    
    def _is_valid_hostname_or_ip(self, value: str) -> bool:
        """Vérifie si une valeur est un hostname ou IP valide."""
        try:
            # Test IP
            ipaddress.ip_address(value)
            return True
        except ValueError:
            pass
        
        # Test hostname
        hostname_pattern = re.compile(
            r'^(?!-)[A-Z\d-]{1,63}(?<!-)(?:\.(?!-)[A-Z\d-]{1,63}(?<!-))*$', 
            re.IGNORECASE
        )
        return hostname_pattern.match(value) is not None


class SecurityValidator(IValidator):
    """Validateur de sécurité."""
    
    def __init__(self, name: str = "security"):
        self.name = name
        self._metrics = MetricsCollector()
        
        # Patterns de détection
        self._sensitive_patterns = [
            (re.compile(r'password', re.IGNORECASE), "PASSWORD_IN_CONFIG"),
            (re.compile(r'secret', re.IGNORECASE), "SECRET_IN_CONFIG"),
            (re.compile(r'token', re.IGNORECASE), "TOKEN_IN_CONFIG"),
            (re.compile(r'key(?!word)', re.IGNORECASE), "KEY_IN_CONFIG"),
        ]
        
        self._weak_password_patterns = [
            (re.compile(r'^.{1,7}$'), "PASSWORD_TOO_SHORT"),
            (re.compile(r'^[a-zA-Z]+$'), "PASSWORD_ONLY_LETTERS"),
            (re.compile(r'^[0-9]+$'), "PASSWORD_ONLY_NUMBERS"),
            (re.compile(r'^(password|123456|admin)$', re.IGNORECASE), "PASSWORD_TOO_COMMON"),
        ]
    
    def validate(self, config: Dict[str, Any], context: Dict[str, Any] = None) -> ValidationResult:
        """Valide la sécurité de la configuration."""
        result = ValidationResult(is_valid=True)
        context = context or {}
        
        # Détection de valeurs sensibles en dur
        self._check_hardcoded_secrets(config, result, "")
        
        # Validation des configurations de sécurité
        self._validate_encryption_config(config, result, context)
        self._validate_authentication_config(config, result, context)
        self._validate_network_security(config, result, context)
        
        self._metrics.increment("security_validation_completed")
        return result
    
    def get_name(self) -> str:
        return self.name
    
    def _check_hardcoded_secrets(self, obj: Any, result: ValidationResult, path: str) -> None:
        """Détecte les secrets codés en dur."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                
                # Vérification des clés sensibles
                for pattern, code in self._sensitive_patterns:
                    if pattern.search(key):
                        if isinstance(value, str) and not value.startswith("${"):
                            issue = ValidationIssue(
                                level=ValidationLevel.WARNING,
                                category=ValidationCategory.SECURITY,
                                message=f"Valeur sensible potentiellement codée en dur: {key}",
                                path=current_path,
                                code=code,
                                suggestion="Utilisez une variable d'environnement: ${VAR_NAME}"
                            )
                            result.add_issue(issue)
                
                self._check_hardcoded_secrets(value, result, current_path)
        
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                current_path = f"{path}[{i}]" if path else f"[{i}]"
                self._check_hardcoded_secrets(item, result, current_path)
    
    def _validate_encryption_config(self, config: Dict[str, Any], 
                                  result: ValidationResult, context: Dict[str, Any]) -> None:
        """Valide la configuration de chiffrement."""
        environment = context.get("environment", "dev")
        
        # En production, le chiffrement doit être activé
        if environment == "production":
            tenant_config = config.get("tenant", {})
            if not tenant_config.get("encryption_enabled", False):
                issue = ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.SECURITY,
                    message="Le chiffrement doit être activé en production",
                    path="tenant.encryption_enabled",
                    code="ENCRYPTION_REQUIRED_PRODUCTION",
                    suggestion="Activez encryption_enabled: true"
                )
                result.add_issue(issue)
    
    def _validate_authentication_config(self, config: Dict[str, Any], 
                                      result: ValidationResult, context: Dict[str, Any]) -> None:
        """Valide la configuration d'authentification."""
        security_config = config.get("security", {})
        if not security_config:
            return
        
        # Validation RBAC
        auth_config = security_config.get("auth", {})
        if auth_config.get("rbac_enabled", False):
            jwt_expiry = auth_config.get("jwt_expiry", 3600)
            if jwt_expiry > 86400:  # 24 heures
                issue = ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.SECURITY,
                    message=f"Durée de vie JWT élevée: {jwt_expiry}s",
                    path="security.auth.jwt_expiry",
                    code="LONG_JWT_EXPIRY",
                    suggestion="Considérez une durée plus courte pour la sécurité"
                )
                result.add_issue(issue)
    
    def _validate_network_security(self, config: Dict[str, Any], 
                                 result: ValidationResult, context: Dict[str, Any]) -> None:
        """Valide la sécurité réseau."""
        environment = context.get("environment", "dev")
        
        # Validation SSL/TLS
        if environment in ["staging", "production"]:
            redis_config = config.get("redis", {})
            if redis_config and not redis_config.get("ssl", False):
                issue = ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.SECURITY,
                    message=f"SSL requis en {environment}",
                    path="redis.ssl",
                    code="SSL_REQUIRED",
                    suggestion="Activez ssl: true"
                )
                result.add_issue(issue)


class PerformanceValidator(IValidator):
    """Validateur de performance."""
    
    def __init__(self, name: str = "performance"):
        self.name = name
        self._metrics = MetricsCollector()
    
    def validate(self, config: Dict[str, Any], context: Dict[str, Any] = None) -> ValidationResult:
        """Valide les aspects performance de la configuration."""
        result = ValidationResult(is_valid=True)
        context = context or {}
        
        self._validate_cache_config(config, result)
        self._validate_connection_pools(config, result)
        self._validate_timeouts(config, result)
        self._validate_resource_limits(config, result)
        
        self._metrics.increment("performance_validation_completed")
        return result
    
    def get_name(self) -> str:
        return self.name
    
    def _validate_cache_config(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Valide la configuration du cache."""
        cache_config = config.get("cache", {})
        if not cache_config:
            return
        
        # Validation des tailles de cache
        max_size = cache_config.get("max_cache_size", 0)
        l1_size = cache_config.get("l1_cache", {}).get("size", 0)
        
        if l1_size > max_size:
            issue = ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.PERFORMANCE,
                message="Taille du cache L1 supérieure au cache global",
                path="cache.l1_cache.size",
                code="INVALID_CACHE_SIZE_RATIO",
                suggestion="Ajustez les tailles des caches"
            )
            result.add_issue(issue)
    
    def _validate_connection_pools(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Valide les pools de connexions."""
        redis_config = config.get("redis", {})
        if redis_config:
            max_conn = redis_config.get("max_connections", 0)
            min_conn = redis_config.get("min_connections", 0)
            
            if max_conn > 0 and min_conn > 0:
                ratio = min_conn / max_conn
                if ratio < 0.1:  # Moins de 10%
                    issue = ValidationIssue(
                        level=ValidationLevel.INFO,
                        category=ValidationCategory.PERFORMANCE,
                        message="Ratio min/max connexions très faible",
                        path="redis.connections",
                        code="LOW_CONNECTION_RATIO",
                        suggestion="Considérez augmenter min_connections pour de meilleures performances"
                    )
                    result.add_issue(issue)
    
    def _validate_timeouts(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Valide les timeouts."""
        # Validation des timeouts Redis
        redis_config = config.get("redis", {})
        if redis_config:
            timeout = redis_config.get("timeout", 0)
            if timeout > 60:  # Plus d'une minute
                issue = ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.PERFORMANCE,
                    message=f"Timeout Redis élevé: {timeout}s",
                    path="redis.timeout",
                    code="HIGH_REDIS_TIMEOUT",
                    suggestion="Considérez un timeout plus court pour éviter les blocages"
                )
                result.add_issue(issue)
    
    def _validate_resource_limits(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Valide les limites de ressources."""
        performance_config = config.get("performance", {})
        if performance_config:
            concurrent_req = performance_config.get("concurrent_requests", 0)
            
            # Vérification des ressources système supposées
            if concurrent_req > 500:
                issue = ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.PERFORMANCE,
                    message=f"Nombre élevé de requêtes concurrentes: {concurrent_req}",
                    path="performance.concurrent_requests",
                    code="HIGH_CONCURRENCY_LIMIT",
                    suggestion="Vérifiez que votre infrastructure peut supporter cette charge"
                )
                result.add_issue(issue)


class ConfigValidator:
    """
    Validateur de configuration principal.
    
    Coordonne l'exécution de multiples validateurs spécialisés
    et produit un rapport de validation consolidé.
    """
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self._validators: List[IValidator] = []
        self._metrics = MetricsCollector()
        
        # Enregistrement des validateurs par défaut
        self._register_default_validators()
    
    def validate(self, config: Dict[str, Any], 
                context: Dict[str, Any] = None) -> ValidationResult:
        """
        Valide une configuration avec tous les validateurs enregistrés.
        
        Args:
            config: Configuration à valider
            context: Contexte de validation (environnement, tenant, etc.)
            
        Returns:
            Résultat consolidé de validation
        """
        import time
        
        start_time = time.time()
        context = context or {}
        
        # Résultat principal
        main_result = ValidationResult(is_valid=True)
        
        # Exécution de tous les validateurs
        for validator in self._validators:
            try:
                validator_result = validator.validate(config, context)
                main_result = main_result.merge(validator_result)
                
                self._metrics.increment(f"validator_{validator.get_name()}_executed")
                
            except Exception as e:
                issue = ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.SYNTAX,
                    message=f"Erreur du validateur {validator.get_name()}: {e}",
                    path="",
                    code="VALIDATOR_ERROR"
                )
                main_result.add_issue(issue)
                self._metrics.increment(f"validator_{validator.get_name()}_error")
        
        # Finalisation
        main_result.validation_time_ms = (time.time() - start_time) * 1000
        main_result.metadata["validator_count"] = len(self._validators)
        main_result.metadata["strict_mode"] = self.strict_mode
        
        # En mode strict, les warnings deviennent des erreurs
        if self.strict_mode:
            for issue in main_result.issues:
                if issue.level == ValidationLevel.WARNING:
                    issue.level = ValidationLevel.ERROR
                    main_result.is_valid = False
        
        self._metrics.histogram("config_validation_time_ms", main_result.validation_time_ms)
        self._metrics.increment("config_validation_completed")
        
        return main_result
    
    def add_validator(self, validator: IValidator) -> None:
        """Ajoute un validateur personnalisé."""
        self._validators.append(validator)
    
    def remove_validator(self, validator_name: str) -> bool:
        """Supprime un validateur par nom."""
        for i, validator in enumerate(self._validators):
            if validator.get_name() == validator_name:
                del self._validators[i]
                return True
        return False
    
    def _register_default_validators(self) -> None:
        """Enregistre les validateurs par défaut."""
        
        # Schéma de base pour la configuration Slack Tools
        base_schema = {
            "type": "object",
            "properties": {
                "redis": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                        "db": {"type": "integer", "minimum": 0},
                        "password": {"type": "string"},
                        "ssl": {"type": "boolean"},
                        "timeout": {"type": "integer", "minimum": 1},
                        "max_connections": {"type": "integer", "minimum": 1},
                        "min_connections": {"type": "integer", "minimum": 1}
                    },
                    "required": ["host", "port"]
                },
                "slack": {
                    "type": "object",
                    "properties": {
                        "default_locale": {"type": "string"},
                        "supported_locales": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "timeout": {"type": "integer", "minimum": 1},
                        "retry_attempts": {"type": "integer", "minimum": 1, "maximum": 10}
                    }
                },
                "cache": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "default_ttl": {"type": "integer", "minimum": 1},
                        "max_cache_size": {"type": "integer", "minimum": 1}
                    }
                },
                "monitoring": {
                    "type": "object",
                    "properties": {
                        "metrics_enabled": {"type": "boolean"},
                        "collection_interval": {"type": "integer", "minimum": 1}
                    }
                }
            },
            "required": ["redis", "slack"]
        }
        
        # Ajout des validateurs
        self.add_validator(SchemaValidator(base_schema, "base_schema"))
        self.add_validator(BusinessRuleValidator())
        self.add_validator(SecurityValidator())
        self.add_validator(PerformanceValidator())
    
    def get_validator_names(self) -> List[str]:
        """Retourne les noms de tous les validateurs enregistrés."""
        return [validator.get_name() for validator in self._validators]
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Statistiques de validation."""
        return {
            "validator_count": len(self._validators),
            "validator_names": self.get_validator_names(),
            "strict_mode": self.strict_mode,
            "metrics": self._metrics.get_all_metrics()
        }


# Fonctions utilitaires pour la validation
def validate_config_file(file_path: Union[str, Path], 
                        strict_mode: bool = False,
                        context: Dict[str, Any] = None) -> ValidationResult:
    """
    Valide un fichier de configuration.
    
    Args:
        file_path: Chemin vers le fichier
        strict_mode: Mode strict de validation
        context: Contexte de validation
        
    Returns:
        Résultat de validation
    """
    from .config_loader import load_config
    
    try:
        config = load_config(file_path)
        validator = ConfigValidator(strict_mode=strict_mode)
        return validator.validate(config, context)
        
    except Exception as e:
        result = ValidationResult(is_valid=False)
        issue = ValidationIssue(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.SYNTAX,
            message=f"Impossible de charger le fichier: {e}",
            path=str(file_path),
            code="FILE_LOAD_ERROR"
        )
        result.add_issue(issue)
        return result


def create_custom_validator(name: str, 
                           validation_func: Callable[[Dict[str, Any], Dict[str, Any]], ValidationResult]) -> IValidator:
    """
    Crée un validateur personnalisé.
    
    Args:
        name: Nom du validateur
        validation_func: Fonction de validation
        
    Returns:
        Instance de validateur personnalisé
    """
    
    class CustomValidator(IValidator):
        def __init__(self):
            self._name = name
            self._func = validation_func
        
        def validate(self, config: Dict[str, Any], context: Dict[str, Any] = None) -> ValidationResult:
            return self._func(config, context or {})
        
        def get_name(self) -> str:
            return self._name
    
    return CustomValidator()
