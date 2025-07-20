"""
Advanced Configuration Validator for Enterprise Tenant Environments

Ce module fournit une validation avancée des configurations d'environnement
avec support pour les schémas JSON Schema, validation de sécurité,
et vérification de conformité.

Développé par l'équipe d'experts dirigée par Fahed Mlaiel
"""

import json
import yaml
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import logging
import ipaddress
from urllib.parse import urlparse
import jsonschema
from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Niveaux de validation disponibles."""
    BASIC = "basic"
    ADVANCED = "advanced"
    STRICT = "strict"
    ENTERPRISE = "enterprise"


class SecurityLevel(str, Enum):
    """Niveaux de sécurité pour la validation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Résultat d'une validation de configuration."""
    
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    security_issues: List[str]
    performance_issues: List[str]
    compliance_issues: List[str]
    score: float = 0.0
    
    def __post_init__(self):
        """Calcule le score de validation."""
        total_issues = len(self.errors) + len(self.warnings) + len(self.security_issues)
        if total_issues == 0:
            self.score = 100.0
        else:
            # Pondération des problèmes
            error_weight = 10
            warning_weight = 3
            security_weight = 15
            
            penalty = (
                len(self.errors) * error_weight +
                len(self.warnings) * warning_weight +
                len(self.security_issues) * security_weight
            )
            self.score = max(0.0, 100.0 - penalty)
    
    def add_error(self, error: str) -> None:
        """Ajoute une erreur."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Ajoute un avertissement."""
        self.warnings.append(warning)
    
    def add_security_issue(self, issue: str) -> None:
        """Ajoute un problème de sécurité."""
        self.security_issues.append(issue)
        self.is_valid = False
    
    def add_performance_issue(self, issue: str) -> None:
        """Ajoute un problème de performance."""
        self.performance_issues.append(issue)
    
    def add_compliance_issue(self, issue: str) -> None:
        """Ajoute un problème de conformité."""
        self.compliance_issues.append(issue)


class ConfigValidator:
    """Validateur avancé de configurations d'environnement."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.ADVANCED):
        """Initialise le validateur."""
        self.validation_level = validation_level
        self.security_patterns = self._load_security_patterns()
        self.performance_thresholds = self._load_performance_thresholds()
        self.schema_cache = {}
    
    def _load_security_patterns(self) -> Dict[str, re.Pattern]:
        """Charge les patterns de sécurité."""
        return {
            "hardcoded_secret": re.compile(r'(password|secret|key|token).*=.*["\'][^"\']{8,}["\']', re.IGNORECASE),
            "ip_address": re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            "url_with_credentials": re.compile(r'https?://[^:]+:[^@]+@'),
            "base64_encoded": re.compile(r'[A-Za-z0-9+/]{20,}={0,2}'),
            "jwt_token": re.compile(r'eyJ[A-Za-z0-9+/=]+\.eyJ[A-Za-z0-9+/=]+\.[A-Za-z0-9+/=]*'),
        }
    
    def _load_performance_thresholds(self) -> Dict[str, Any]:
        """Charge les seuils de performance."""
        return {
            "max_connection_pool_size": 200,
            "max_timeout_seconds": 300,
            "max_cache_size_mb": 1024,
            "max_log_level_production": "INFO",
            "max_workers": 50,
        }
    
    def validate_config(self, config: Dict[str, Any], environment: str) -> ValidationResult:
        """
        Valide une configuration complète.
        
        Args:
            config: Configuration à valider
            environment: Nom de l'environnement
            
        Returns:
            Résultat de validation
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], 
                                security_issues=[], performance_issues=[], compliance_issues=[])
        
        try:
            # Validation structurelle
            self._validate_structure(config, result)
            
            # Validation de sécurité
            self._validate_security(config, environment, result)
            
            # Validation de performance
            self._validate_performance(config, environment, result)
            
            # Validation de conformité
            self._validate_compliance(config, environment, result)
            
            # Validation des types de données
            self._validate_data_types(config, result)
            
            # Validation des dépendances
            self._validate_dependencies(config, result)
            
            # Validation environnement-spécifique
            self._validate_environment_specific(config, environment, result)
            
        except Exception as e:
            result.add_error(f"Erreur lors de la validation: {str(e)}")
            logger.error(f"Validation error: {e}", exc_info=True)
        
        return result
    
    def _validate_structure(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Valide la structure de base de la configuration."""
        required_sections = [
            "metadata",
            "application",
            "database", 
            "security",
            "observability"
        ]
        
        for section in required_sections:
            if section not in config:
                result.add_error(f"Section requise manquante: {section}")
        
        # Vérification de la metadata
        if "metadata" in config:
            metadata = config["metadata"]
            required_metadata = ["name", "description", "version", "environment_type"]
            
            for field in required_metadata:
                if field not in metadata:
                    result.add_error(f"Champ metadata requis manquant: {field}")
    
    def _validate_security(self, config: Dict[str, Any], environment: str, result: ValidationResult) -> None:
        """Valide les aspects de sécurité."""
        config_str = json.dumps(config)
        
        # Recherche de secrets hardcodés
        for pattern_name, pattern in self.security_patterns.items():
            if pattern.search(config_str):
                result.add_security_issue(f"Potentiel secret hardcodé détecté: {pattern_name}")
        
        # Validation des paramètres de sécurité
        security_config = config.get("security", {})
        
        # JWT Secret
        jwt_config = security_config.get("authentication", {}).get("jwt", {})
        if jwt_config.get("secret_key") in ["secret", "dev-secret", "test", ""]:
            if environment == "production":
                result.add_security_issue("JWT secret key faible en production")
            else:
                result.add_warning("JWT secret key faible pour environnement de développement")
        
        # TLS Configuration
        tls_config = security_config.get("encryption", {}).get("tls", {})
        if environment == "production" and not tls_config.get("enabled", False):
            result.add_security_issue("TLS non activé en production")
        
        # Rate Limiting
        rate_limiting = config.get("application", {}).get("api", {}).get("rate_limiting", {})
        if not rate_limiting.get("enabled", False) and environment == "production":
            result.add_security_issue("Rate limiting non activé en production")
    
    def _validate_performance(self, config: Dict[str, Any], environment: str, result: ValidationResult) -> None:
        """Valide les aspects de performance."""
        # Database connection pool
        db_config = config.get("database", {}).get("postgresql", {}).get("pool", {})
        max_size = db_config.get("max_size", 0)
        
        if max_size > self.performance_thresholds["max_connection_pool_size"]:
            result.add_performance_issue(f"Pool de connexions trop large: {max_size}")
        
        # Timeout configurations
        timeouts = self._extract_timeouts(config)
        for timeout_path, timeout_value in timeouts:
            if timeout_value > self.performance_thresholds["max_timeout_seconds"]:
                result.add_performance_issue(f"Timeout trop élevé: {timeout_path} = {timeout_value}s")
        
        # Log level in production
        if environment == "production":
            log_level = config.get("observability", {}).get("logging", {}).get("level", "INFO")
            if log_level == "DEBUG":
                result.add_performance_issue("Log level DEBUG en production peut impacter les performances")
    
    def _validate_compliance(self, config: Dict[str, Any], environment: str, result: ValidationResult) -> None:
        """Valide la conformité réglementaire."""
        # GDPR compliance
        if environment == "production":
            # Vérification de l'anonymisation des logs
            logging_config = config.get("observability", {}).get("logging", {})
            if not logging_config.get("anonymize_sensitive_data", False):
                result.add_compliance_issue("Anonymisation des données sensibles non configurée (GDPR)")
            
            # Vérification de la rétention des données
            if not logging_config.get("retention_policy"):
                result.add_compliance_issue("Politique de rétention des logs non définie (GDPR)")
        
        # SOC2 compliance
        if environment == "production":
            # Audit logging
            if not config.get("observability", {}).get("audit_logging", {}).get("enabled", False):
                result.add_compliance_issue("Audit logging non activé (SOC2)")
            
            # Encryption at rest
            encryption_config = config.get("security", {}).get("encryption", {})
            if not encryption_config.get("data_encryption", {}).get("enabled", False):
                result.add_compliance_issue("Chiffrement des données au repos non activé (SOC2)")
    
    def _validate_data_types(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Valide les types de données."""
        self._validate_types_recursive(config, "", result)
    
    def _validate_types_recursive(self, obj: Any, path: str, result: ValidationResult) -> None:
        """Validation récursive des types."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                self._validate_types_recursive(value, new_path, result)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                self._validate_types_recursive(item, new_path, result)
        elif isinstance(obj, str):
            # Validation des URLs
            if "url" in path.lower() or "endpoint" in path.lower():
                self._validate_url(obj, path, result)
            # Validation des IPs
            elif "host" in path.lower() and self._looks_like_ip(obj):
                self._validate_ip(obj, path, result)
    
    def _validate_url(self, url: str, path: str, result: ValidationResult) -> None:
        """Valide une URL."""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                result.add_error(f"URL invalide à {path}: {url}")
        except Exception:
            result.add_error(f"URL malformée à {path}: {url}")
    
    def _validate_ip(self, ip: str, path: str, result: ValidationResult) -> None:
        """Valide une adresse IP."""
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            result.add_error(f"Adresse IP invalide à {path}: {ip}")
    
    def _looks_like_ip(self, value: str) -> bool:
        """Vérifie si une chaîne ressemble à une IP."""
        return re.match(r'^\d+\.\d+\.\d+\.\d+$', value) is not None
    
    def _validate_dependencies(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Valide les dépendances entre configurations."""
        # Redis et cache
        redis_config = config.get("database", {}).get("redis", {})
        cache_config = config.get("caching", {})
        
        if cache_config.get("enabled", False) and not redis_config:
            result.add_error("Cache activé mais Redis non configuré")
        
        # ML services et GPU
        ml_config = config.get("external_services", {}).get("ml_services", {})
        tf_serving = ml_config.get("tensorflow_serving", {})
        
        if tf_serving.get("gpu_enabled", False):
            resources = config.get("resources", {}).get("compute", {})
            if not resources.get("gpu", {}).get("enabled", False):
                result.add_warning("TensorFlow Serving GPU activé mais ressources GPU non configurées")
    
    def _validate_environment_specific(self, config: Dict[str, Any], environment: str, result: ValidationResult) -> None:
        """Validation spécifique à l'environnement."""
        if environment == "development":
            # En dev, on peut avoir des configurations plus laxistes
            if not config.get("application", {}).get("general", {}).get("debug", False):
                result.add_warning("Mode debug non activé en développement")
                
        elif environment == "production":
            # En prod, on doit avoir des configurations strictes
            if config.get("application", {}).get("general", {}).get("debug", False):
                result.add_error("Mode debug activé en production")
            
            # Vérification des ressources
            resources = config.get("resources", {}).get("compute", {})
            if not resources.get("cpu", {}).get("limits"):
                result.add_warning("Limites CPU non définies en production")
    
    def _extract_timeouts(self, config: Dict[str, Any]) -> List[Tuple[str, int]]:
        """Extrait tous les timeouts de la configuration."""
        timeouts = []
        self._extract_timeouts_recursive(config, "", timeouts)
        return timeouts
    
    def _extract_timeouts_recursive(self, obj: Any, path: str, timeouts: List[Tuple[str, int]]) -> None:
        """Extraction récursive des timeouts."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                if "timeout" in key.lower() and isinstance(value, (int, float)):
                    timeouts.append((new_path, int(value)))
                else:
                    self._extract_timeouts_recursive(value, new_path, timeouts)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                self._extract_timeouts_recursive(item, new_path, timeouts)


class SchemaValidator:
    """Validateur basé sur des schémas JSON Schema."""
    
    def __init__(self):
        """Initialise le validateur de schémas."""
        self.schemas = self._load_schemas()
    
    def _load_schemas(self) -> Dict[str, Dict]:
        """Charge les schémas de validation."""
        base_schema = {
            "type": "object",
            "required": ["metadata", "application", "database", "security"],
            "properties": {
                "metadata": {
                    "type": "object",
                    "required": ["name", "description", "version", "environment_type"],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "description": {"type": "string", "minLength": 1},
                        "version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
                        "environment_type": {"type": "string", "enum": ["development", "staging", "production"]}
                    }
                },
                "application": {
                    "type": "object",
                    "properties": {
                        "api": {
                            "type": "object",
                            "properties": {
                                "fastapi": {
                                    "type": "object",
                                    "properties": {
                                        "host": {"type": "string"},
                                        "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                                        "workers": {"type": "integer", "minimum": 1, "maximum": 50}
                                    }
                                }
                            }
                        }
                    }
                },
                "database": {
                    "type": "object",
                    "properties": {
                        "postgresql": {
                            "type": "object",
                            "properties": {
                                "host": {"type": "string"},
                                "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                                "database": {"type": "string", "minLength": 1},
                                "pool": {
                                    "type": "object",
                                    "properties": {
                                        "min_size": {"type": "integer", "minimum": 1},
                                        "max_size": {"type": "integer", "minimum": 1, "maximum": 200}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return {
            "base": base_schema,
            "development": base_schema,
            "staging": base_schema,
            "production": base_schema
        }
    
    def validate_against_schema(self, config: Dict[str, Any], environment: str) -> ValidationResult:
        """Valide une configuration contre un schéma."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], 
                                security_issues=[], performance_issues=[], compliance_issues=[])
        
        schema = self.schemas.get(environment, self.schemas["base"])
        
        try:
            validate(instance=config, schema=schema)
        except ValidationError as e:
            result.add_error(f"Erreur de schéma: {e.message} à {e.absolute_path}")
        except Exception as e:
            result.add_error(f"Erreur de validation de schéma: {str(e)}")
        
        return result


def validate_configuration_file(file_path: str, environment: str = None) -> ValidationResult:
    """
    Valide un fichier de configuration.
    
    Args:
        file_path: Chemin vers le fichier de configuration
        environment: Environnement cible (optionnel, déduit du fichier)
        
    Returns:
        Résultat de validation
    """
    try:
        # Chargement du fichier
        path = Path(file_path)
        if not path.exists():
            result = ValidationResult(is_valid=False, errors=[], warnings=[], 
                                    security_issues=[], performance_issues=[], compliance_issues=[])
            result.add_error(f"Fichier non trouvé: {file_path}")
            return result
        
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() in ['.yml', '.yaml']:
                config = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                result = ValidationResult(is_valid=False, errors=[], warnings=[], 
                                        security_issues=[], performance_issues=[], compliance_issues=[])
                result.add_error(f"Format de fichier non supporté: {path.suffix}")
                return result
        
        # Déduction de l'environnement si non fourni
        if not environment:
            environment = config.get("metadata", {}).get("environment_type", "development")
        
        # Validation
        validator = ConfigValidator()
        return validator.validate_config(config, environment)
        
    except Exception as e:
        result = ValidationResult(is_valid=False, errors=[], warnings=[], 
                                security_issues=[], performance_issues=[], compliance_issues=[])
        result.add_error(f"Erreur lors du chargement du fichier: {str(e)}")
        return result


__all__ = [
    "ValidationLevel",
    "SecurityLevel", 
    "ValidationResult",
    "ConfigValidator",
    "SchemaValidator",
    "validate_configuration_file"
]
