"""
Validateur de Configuration Slack Ultra-Robuste
===============================================

Module de validation complète et sécurisée de toutes les configurations
Slack du système AlertManager avec vérifications avancées et rapports détaillés.

Développé par l'équipe Backend Senior sous la direction de Fahed Mlaiel.
"""

import re
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse
import ipaddress
from pydantic import BaseModel, Field, validator, ValidationError

from . import SlackSeverity, SlackChannelType
from .utils import SlackUtils

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Niveaux de validation."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"

class ValidationCategory(Enum):
    """Catégories de validation."""
    CONFIGURATION = "configuration"
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    INTEGRATION = "integration"

@dataclass
class ValidationResult:
    """Résultat d'une validation."""
    
    is_valid: bool
    category: ValidationCategory
    level: ValidationLevel
    field_path: str
    message: str
    suggestion: Optional[str] = None
    severity: str = "error"  # error, warning, info
    code: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit le résultat en dictionnaire."""
        return {
            'is_valid': self.is_valid,
            'category': self.category.value,
            'level': self.level.value,
            'field_path': self.field_path,
            'message': self.message,
            'suggestion': self.suggestion,
            'severity': self.severity,
            'code': self.code
        }

@dataclass
class ValidationReport:
    """Rapport de validation complet."""
    
    is_valid: bool
    validation_level: ValidationLevel
    total_checks: int
    passed_checks: int
    failed_checks: int
    warnings: int
    errors: List[ValidationResult] = field(default_factory=list)
    warnings_list: List[ValidationResult] = field(default_factory=list)
    info_list: List[ValidationResult] = field(default_factory=list)
    performance_score: int = 0  # 0-100
    security_score: int = 0     # 0-100
    compliance_score: int = 0   # 0-100
    validation_time: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    
    def add_result(self, result: ValidationResult):
        """Ajoute un résultat de validation."""
        if result.severity == "error":
            self.errors.append(result)
            self.failed_checks += 1
        elif result.severity == "warning":
            self.warnings_list.append(result)
            self.warnings += 1
        else:
            self.info_list.append(result)
        
        self.total_checks += 1
        if result.is_valid:
            self.passed_checks += 1
    
    def calculate_scores(self):
        """Calcule les scores de performance, sécurité et compliance."""
        total_categories = {cat: 0 for cat in ValidationCategory}
        passed_categories = {cat: 0 for cat in ValidationCategory}
        
        # Compter par catégorie
        for result in self.errors + self.warnings_list + self.info_list:
            total_categories[result.category] += 1
            if result.is_valid:
                passed_categories[result.category] += 1
        
        # Calculer les scores
        if total_categories[ValidationCategory.SECURITY] > 0:
            self.security_score = int(
                (passed_categories[ValidationCategory.SECURITY] / 
                 total_categories[ValidationCategory.SECURITY]) * 100
            )
        else:
            self.security_score = 100
        
        if total_categories[ValidationCategory.PERFORMANCE] > 0:
            self.performance_score = int(
                (passed_categories[ValidationCategory.PERFORMANCE] / 
                 total_categories[ValidationCategory.PERFORMANCE]) * 100
            )
        else:
            self.performance_score = 100
        
        if total_categories[ValidationCategory.COMPLIANCE] > 0:
            self.compliance_score = int(
                (passed_categories[ValidationCategory.COMPLIANCE] / 
                 total_categories[ValidationCategory.COMPLIANCE]) * 100
            )
        else:
            self.compliance_score = 100
        
        # Déterminer la validité globale
        self.is_valid = len(self.errors) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit le rapport en dictionnaire."""
        return {
            'is_valid': self.is_valid,
            'validation_level': self.validation_level.value,
            'summary': {
                'total_checks': self.total_checks,
                'passed_checks': self.passed_checks,
                'failed_checks': self.failed_checks,
                'warnings': self.warnings
            },
            'scores': {
                'performance': self.performance_score,
                'security': self.security_score,
                'compliance': self.compliance_score
            },
            'validation_time': self.validation_time,
            'errors': [error.to_dict() for error in self.errors],
            'warnings': [warning.to_dict() for warning in self.warnings_list],
            'info': [info.to_dict() for info in self.info_list],
            'recommendations': self.recommendations
        }

class SlackConfigValidator:
    """
    Validateur ultra-robuste pour les configurations Slack.
    
    Fonctionnalités:
    - Validation multi-niveaux (basic à paranoid)
    - Vérifications de sécurité avancées
    - Validation des performances
    - Conformité aux standards Slack
    - Validation des webhooks et tokens
    - Vérification des templates Jinja2
    - Analyse des règles de routage
    - Validation des escalades
    - Rapports détaillés avec recommandations
    - Cache de validation pour performances
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """
        Initialise le validateur.
        
        Args:
            validation_level: Niveau de validation par défaut
        """
        self.validation_level = validation_level
        
        # Patterns de validation
        self.webhook_pattern = re.compile(
            r'^https://hooks\.slack\.com/services/[A-Z0-9]+/[A-Z0-9]+/[A-Za-z0-9]+$'
        )
        self.channel_pattern = re.compile(r'^#[a-z0-9_-]+$')
        self.user_id_pattern = re.compile(r'^[UW][A-Z0-9]+$')
        self.token_pattern = re.compile(r'^xox[bpoa]-[A-Za-z0-9-]+$')
        
        # Limites Slack officielles
        self.slack_limits = {
            'message_text_length': 4000,
            'block_count': 50,
            'attachment_count': 20,
            'field_count': 10,
            'action_count': 25,
            'webhook_rate_limit': 1,  # par seconde
            'api_rate_limit': 100,    # par minute
            'channel_name_length': 21,
            'username_length': 21
        }
        
        # Mots-clés de sécurité sensibles
        self.sensitive_keywords = [
            'password', 'secret', 'key', 'token', 'credential',
            'private', 'confidential', 'internal', 'admin'
        ]
        
        # Cache de validation
        self._validation_cache: Dict[str, ValidationReport] = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Métriques
        self.metrics = {
            'validations_performed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'security_violations': 0,
            'performance_issues': 0,
            'compliance_failures': 0
        }
        
        logger.info(f"SlackConfigValidator initialisé (niveau: {validation_level.value})")
    
    async def validate_complete_config(self,
                                     config: Dict[str, Any],
                                     tenant_id: str,
                                     validation_level: Optional[ValidationLevel] = None) -> ValidationReport:
        """
        Valide une configuration complète.
        
        Args:
            config: Configuration à valider
            tenant_id: ID du tenant
            validation_level: Niveau de validation spécifique
            
        Returns:
            Rapport de validation complet
        """
        start_time = datetime.utcnow()
        level = validation_level or self.validation_level
        
        # Vérifier le cache
        cache_key = self._get_cache_key(config, tenant_id, level)
        cached_report = self._get_from_cache(cache_key)
        if cached_report:
            self.metrics['cache_hits'] += 1
            return cached_report
        
        self.metrics['cache_misses'] += 1
        self.metrics['validations_performed'] += 1
        
        # Créer le rapport
        report = ValidationReport(
            is_valid=True,
            validation_level=level,
            total_checks=0,
            passed_checks=0,
            failed_checks=0,
            warnings=0
        )
        
        try:
            # Validation de base
            await self._validate_basic_structure(config, report, level)
            
            # Validation de sécurité
            await self._validate_security(config, tenant_id, report, level)
            
            # Validation des performances
            await self._validate_performance(config, report, level)
            
            # Validation de compliance
            await self._validate_compliance(config, report, level)
            
            # Validation d'intégration
            if level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                await self._validate_integration(config, report, level)
            
            # Validation spécifique par composant
            await self._validate_webhooks(config, report, level)
            await self._validate_templates(config, report, level)
            await self._validate_routing_rules(config, report, level)
            await self._validate_escalation_rules(config, report, level)
            
            # Calculer les scores et recommendations
            report.calculate_scores()
            self._generate_recommendations(report)
            
            # Temps de validation
            report.validation_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Mettre en cache
            self._put_in_cache(cache_key, report)
            
            return report
            
        except Exception as e:
            logger.error(f"Erreur validation configuration: {e}")
            report.add_result(ValidationResult(
                is_valid=False,
                category=ValidationCategory.CONFIGURATION,
                level=level,
                field_path="global",
                message=f"Erreur validation: {e}",
                severity="error",
                code="VALIDATION_ERROR"
            ))
            return report
    
    async def _validate_basic_structure(self,
                                      config: Dict[str, Any],
                                      report: ValidationReport,
                                      level: ValidationLevel):
        """Valide la structure de base de la configuration."""
        
        # Vérifier les champs obligatoires
        required_fields = ['version', 'environments']
        for field in required_fields:
            if field not in config:
                report.add_result(ValidationResult(
                    is_valid=False,
                    category=ValidationCategory.CONFIGURATION,
                    level=level,
                    field_path=field,
                    message=f"Champ obligatoire manquant: {field}",
                    suggestion=f"Ajouter le champ '{field}' à la configuration",
                    severity="error",
                    code="MISSING_REQUIRED_FIELD"
                ))
            else:
                report.add_result(ValidationResult(
                    is_valid=True,
                    category=ValidationCategory.CONFIGURATION,
                    level=level,
                    field_path=field,
                    message=f"Champ obligatoire présent: {field}",
                    severity="info"
                ))
        
        # Valider la version
        if 'version' in config:
            version = config['version']
            if not isinstance(version, str) or not re.match(r'^\d+\.\d+\.\d+$', version):
                report.add_result(ValidationResult(
                    is_valid=False,
                    category=ValidationCategory.CONFIGURATION,
                    level=level,
                    field_path="version",
                    message="Format de version invalide",
                    suggestion="Utiliser le format semver (ex: 2.1.0)",
                    severity="error",
                    code="INVALID_VERSION_FORMAT"
                ))
        
        # Valider les environnements
        if 'environments' in config:
            environments = config['environments']
            if not isinstance(environments, dict):
                report.add_result(ValidationResult(
                    is_valid=False,
                    category=ValidationCategory.CONFIGURATION,
                    level=level,
                    field_path="environments",
                    message="Les environnements doivent être un objet",
                    severity="error",
                    code="INVALID_ENVIRONMENTS_TYPE"
                ))
            else:
                valid_envs = ['dev', 'staging', 'prod', 'test']
                for env_name in environments.keys():
                    if env_name not in valid_envs:
                        report.add_result(ValidationResult(
                            is_valid=False,
                            category=ValidationCategory.CONFIGURATION,
                            level=level,
                            field_path=f"environments.{env_name}",
                            message=f"Environnement invalide: {env_name}",
                            suggestion=f"Utiliser un des environnements valides: {valid_envs}",
                            severity="warning",
                            code="INVALID_ENVIRONMENT_NAME"
                        ))
    
    async def _validate_security(self,
                               config: Dict[str, Any],
                               tenant_id: str,
                               report: ValidationReport,
                               level: ValidationLevel):
        """Valide les aspects sécuritaires de la configuration."""
        
        # Vérifier les tokens et secrets
        await self._validate_tokens_security(config, report, level)
        
        # Vérifier les webhooks
        await self._validate_webhook_security(config, report, level)
        
        # Vérifier les données sensibles
        await self._validate_sensitive_data(config, report, level)
        
        # Validation stricte/paranoïaque
        if level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            await self._validate_advanced_security(config, tenant_id, report, level)
    
    async def _validate_tokens_security(self,
                                      config: Dict[str, Any],
                                      report: ValidationReport,
                                      level: ValidationLevel):
        """Valide la sécurité des tokens."""
        
        def check_token_recursive(obj: Any, path: str):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    
                    # Vérifier si c'est un token
                    if any(token_key in key.lower() for token_key in ['token', 'secret', 'key']):
                        if isinstance(value, str):
                            # Vérifier le format du token
                            if key.lower() == 'bot_token' and not self.token_pattern.match(value.replace('enc:', '')):
                                report.add_result(ValidationResult(
                                    is_valid=False,
                                    category=ValidationCategory.SECURITY,
                                    level=level,
                                    field_path=new_path,
                                    message="Format de token Slack invalide",
                                    suggestion="Utiliser un token Slack valide (xoxb-...)",
                                    severity="error",
                                    code="INVALID_TOKEN_FORMAT"
                                ))
                                self.metrics['security_violations'] += 1
                            
                            # Vérifier le chiffrement
                            if not value.startswith('enc:') and level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                                report.add_result(ValidationResult(
                                    is_valid=False,
                                    category=ValidationCategory.SECURITY,
                                    level=level,
                                    field_path=new_path,
                                    message="Token non chiffré détecté",
                                    suggestion="Chiffrer tous les tokens sensibles",
                                    severity="warning",
                                    code="UNENCRYPTED_TOKEN"
                                ))
                            
                            # Vérifier la longueur minimum
                            clean_value = value.replace('enc:', '')
                            if len(clean_value) < 20:
                                report.add_result(ValidationResult(
                                    is_valid=False,
                                    category=ValidationCategory.SECURITY,
                                    level=level,
                                    field_path=new_path,
                                    message="Token trop court (possible test/invalide)",
                                    severity="error",
                                    code="SHORT_TOKEN"
                                ))
                    
                    check_token_recursive(value, new_path)
            
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_token_recursive(item, f"{path}[{i}]")
        
        check_token_recursive(config, "")
    
    async def _validate_webhook_security(self,
                                       config: Dict[str, Any],
                                       report: ValidationReport,
                                       level: ValidationLevel):
        """Valide la sécurité des webhooks."""
        
        def check_webhooks_recursive(obj: Any, path: str):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    
                    if 'webhook' in key.lower() and isinstance(value, str):
                        # Vérifier le format
                        if not self.webhook_pattern.match(value):
                            report.add_result(ValidationResult(
                                is_valid=False,
                                category=ValidationCategory.SECURITY,
                                level=level,
                                field_path=new_path,
                                message="Format de webhook Slack invalide",
                                suggestion="Utiliser le format officiel Slack: https://hooks.slack.com/services/...",
                                severity="error",
                                code="INVALID_WEBHOOK_FORMAT"
                            ))
                            self.metrics['security_violations'] += 1
                        
                        # Vérifier HTTPS
                        if not value.startswith('https://'):
                            report.add_result(ValidationResult(
                                is_valid=False,
                                category=ValidationCategory.SECURITY,
                                level=level,
                                field_path=new_path,
                                message="Webhook non sécurisé (non HTTPS)",
                                suggestion="Utiliser uniquement des webhooks HTTPS",
                                severity="error",
                                code="INSECURE_WEBHOOK"
                            ))
                        
                        # Validation paranoïaque: vérifier le domaine
                        if level == ValidationLevel.PARANOID:
                            parsed_url = urlparse(value)
                            if not parsed_url.netloc.endswith('slack.com'):
                                report.add_result(ValidationResult(
                                    is_valid=False,
                                    category=ValidationCategory.SECURITY,
                                    level=level,
                                    field_path=new_path,
                                    message="Webhook pointant vers un domaine non-Slack",
                                    severity="error",
                                    code="UNTRUSTED_WEBHOOK_DOMAIN"
                                ))
                    
                    check_webhooks_recursive(value, new_path)
            
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_webhooks_recursive(item, f"{path}[{i}]")
        
        check_webhooks_recursive(config, "")
    
    async def _validate_sensitive_data(self,
                                     config: Dict[str, Any],
                                     report: ValidationReport,
                                     level: ValidationLevel):
        """Valide la présence de données sensibles."""
        
        def check_sensitive_recursive(obj: Any, path: str):
            if isinstance(obj, str):
                # Vérifier les mots-clés sensibles
                lower_obj = obj.lower()
                for keyword in self.sensitive_keywords:
                    if keyword in lower_obj and len(obj) > 10:
                        # Possible donnée sensible exposée
                        report.add_result(ValidationResult(
                            is_valid=False,
                            category=ValidationCategory.SECURITY,
                            level=level,
                            field_path=path,
                            message=f"Possible donnée sensible exposée (contient '{keyword}')",
                            suggestion="Vérifier et chiffrer si nécessaire",
                            severity="warning",
                            code="POTENTIAL_SENSITIVE_DATA"
                        ))
                
                # Vérifier les patterns d'adresses IP privées
                ip_pattern = re.compile(r'\b(?:10\.|172\.(?:1[6-9]|2[0-9]|3[01])\.|192\.168\.)\d+\.\d+\b')
                if ip_pattern.search(obj):
                    report.add_result(ValidationResult(
                        is_valid=False,
                        category=ValidationCategory.SECURITY,
                        level=level,
                        field_path=path,
                        message="Adresse IP privée détectée dans la configuration",
                        suggestion="Éviter d'exposer des adresses IP internes",
                        severity="warning",
                        code="PRIVATE_IP_EXPOSED"
                    ))
            
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    check_sensitive_recursive(value, new_path)
            
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_sensitive_recursive(item, f"{path}[{i}]")
        
        check_sensitive_recursive(config, "")
    
    async def _validate_performance(self,
                                  config: Dict[str, Any],
                                  report: ValidationReport,
                                  level: ValidationLevel):
        """Valide les aspects performance de la configuration."""
        
        # Vérifier les limites de rate limiting
        await self._validate_rate_limits(config, report, level)
        
        # Vérifier les timeouts
        await self._validate_timeouts(config, report, level)
        
        # Vérifier les configurations de cache
        await self._validate_cache_config(config, report, level)
    
    async def _validate_rate_limits(self,
                                  config: Dict[str, Any],
                                  report: ValidationReport,
                                  level: ValidationLevel):
        """Valide les configurations de rate limiting."""
        
        def check_rate_limits_recursive(obj: Any, path: str):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    
                    if 'rate_limit' in key.lower() and isinstance(value, (int, float)):
                        # Vérifier les valeurs raisonnables
                        if value > 1000:  # Plus de 1000 req/min semble excessif
                            report.add_result(ValidationResult(
                                is_valid=False,
                                category=ValidationCategory.PERFORMANCE,
                                level=level,
                                field_path=new_path,
                                message=f"Rate limit très élevé: {value}",
                                suggestion="Considérer une valeur plus conservative",
                                severity="warning",
                                code="HIGH_RATE_LIMIT"
                            ))
                            self.metrics['performance_issues'] += 1
                        
                        elif value < 1:  # Moins de 1 req/min semble trop restrictif
                            report.add_result(ValidationResult(
                                is_valid=False,
                                category=ValidationCategory.PERFORMANCE,
                                level=level,
                                field_path=new_path,
                                message=f"Rate limit très bas: {value}",
                                suggestion="Vérifier si cette valeur est intentionnelle",
                                severity="warning",
                                code="LOW_RATE_LIMIT"
                            ))
                    
                    check_rate_limits_recursive(value, new_path)
            
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_rate_limits_recursive(item, f"{path}[{i}]")
        
        check_rate_limits_recursive(config, "")
    
    async def _validate_timeouts(self,
                               config: Dict[str, Any],
                               report: ValidationReport,
                               level: ValidationLevel):
        """Valide les configurations de timeout."""
        
        def check_timeouts_recursive(obj: Any, path: str):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    
                    if 'timeout' in key.lower() and isinstance(value, (int, float)):
                        # Vérifier les valeurs raisonnables
                        if value > 300:  # Plus de 5 minutes
                            report.add_result(ValidationResult(
                                is_valid=False,
                                category=ValidationCategory.PERFORMANCE,
                                level=level,
                                field_path=new_path,
                                message=f"Timeout très élevé: {value}s",
                                suggestion="Considérer un timeout plus court pour éviter les blocages",
                                severity="warning",
                                code="HIGH_TIMEOUT"
                            ))
                        
                        elif value < 1:  # Moins de 1 seconde
                            report.add_result(ValidationResult(
                                is_valid=False,
                                category=ValidationCategory.PERFORMANCE,
                                level=level,
                                field_path=new_path,
                                message=f"Timeout très court: {value}s",
                                suggestion="Vérifier si ce timeout est suffisant",
                                severity="warning",
                                code="LOW_TIMEOUT"
                            ))
                    
                    check_timeouts_recursive(value, new_path)
            
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_timeouts_recursive(item, f"{path}[{i}]")
        
        check_timeouts_recursive(config, "")
    
    async def _validate_compliance(self,
                                 config: Dict[str, Any],
                                 report: ValidationReport,
                                 level: ValidationLevel):
        """Valide la conformité aux standards."""
        
        # Vérifier la structure des canaux
        await self._validate_channel_naming(config, report, level)
        
        # Vérifier les metadata requises
        await self._validate_required_metadata(config, report, level)
        
        # Vérifier la documentation
        await self._validate_documentation(config, report, level)
    
    async def _validate_channel_naming(self,
                                     config: Dict[str, Any],
                                     report: ValidationReport,
                                     level: ValidationLevel):
        """Valide la convention de nommage des canaux."""
        
        def check_channels_recursive(obj: Any, path: str):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    
                    if 'channel' in key.lower() and isinstance(value, str):
                        if value.startswith('#'):
                            channel_name = value[1:]  # Enlever le #
                            
                            # Vérifier la longueur
                            if len(channel_name) > self.slack_limits['channel_name_length']:
                                report.add_result(ValidationResult(
                                    is_valid=False,
                                    category=ValidationCategory.COMPLIANCE,
                                    level=level,
                                    field_path=new_path,
                                    message=f"Nom de canal trop long: {len(channel_name)} caractères",
                                    suggestion=f"Maximum {self.slack_limits['channel_name_length']} caractères",
                                    severity="error",
                                    code="CHANNEL_NAME_TOO_LONG"
                                ))
                                self.metrics['compliance_failures'] += 1
                            
                            # Vérifier les caractères autorisés
                            if not re.match(r'^[a-z0-9_-]+$', channel_name):
                                report.add_result(ValidationResult(
                                    is_valid=False,
                                    category=ValidationCategory.COMPLIANCE,
                                    level=level,
                                    field_path=new_path,
                                    message="Nom de canal contient des caractères invalides",
                                    suggestion="Utiliser uniquement: a-z, 0-9, _, -",
                                    severity="error",
                                    code="INVALID_CHANNEL_CHARACTERS"
                                ))
                            
                            # Vérifier les conventions de nommage
                            if level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                                if not any(prefix in channel_name for prefix in ['alert', 'monitor', 'incident', 'escalation']):
                                    report.add_result(ValidationResult(
                                        is_valid=False,
                                        category=ValidationCategory.COMPLIANCE,
                                        level=level,
                                        field_path=new_path,
                                        message="Canal ne suit pas les conventions de nommage",
                                        suggestion="Utiliser des préfixes comme 'alert-', 'monitor-', etc.",
                                        severity="warning",
                                        code="NAMING_CONVENTION"
                                    ))
                    
                    check_channels_recursive(value, new_path)
            
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_channels_recursive(item, f"{path}[{i}]")
        
        check_channels_recursive(config, "")
    
    async def _validate_webhooks(self,
                               config: Dict[str, Any],
                               report: ValidationReport,
                               level: ValidationLevel):
        """Valide spécifiquement les configurations de webhooks."""
        
        webhook_configs = self._extract_webhooks(config)
        
        for path, webhook_url in webhook_configs:
            # Format de base déjà vérifié dans _validate_webhook_security
            
            # Vérifier la structure de l'URL
            try:
                parsed = urlparse(webhook_url)
                path_parts = parsed.path.split('/')
                
                if len(path_parts) < 4:
                    report.add_result(ValidationResult(
                        is_valid=False,
                        category=ValidationCategory.INTEGRATION,
                        level=level,
                        field_path=path,
                        message="Structure d'URL webhook incomplète",
                        severity="error",
                        code="INCOMPLETE_WEBHOOK_URL"
                    ))
                
                # Vérifier que les composants semblent valides
                if level == ValidationLevel.PARANOID:
                    service_id = path_parts[2] if len(path_parts) > 2 else ""
                    if not re.match(r'^[A-Z0-9]+$', service_id):
                        report.add_result(ValidationResult(
                            is_valid=False,
                            category=ValidationCategory.INTEGRATION,
                            level=level,
                            field_path=path,
                            message="ID de service webhook invalide",
                            severity="warning",
                            code="INVALID_WEBHOOK_SERVICE_ID"
                        ))
            
            except Exception as e:
                report.add_result(ValidationResult(
                    is_valid=False,
                    category=ValidationCategory.INTEGRATION,
                    level=level,
                    field_path=path,
                    message=f"Erreur parsing URL webhook: {e}",
                    severity="error",
                    code="WEBHOOK_PARSE_ERROR"
                ))
    
    async def _validate_templates(self,
                                config: Dict[str, Any],
                                report: ValidationReport,
                                level: ValidationLevel):
        """Valide les templates Jinja2."""
        
        templates = self._extract_templates(config)
        
        for path, template_content in templates:
            try:
                # Valider la syntaxe Jinja2
                from jinja2 import Environment, TemplateSyntaxError
                env = Environment()
                env.from_string(template_content)
                
                report.add_result(ValidationResult(
                    is_valid=True,
                    category=ValidationCategory.CONFIGURATION,
                    level=level,
                    field_path=path,
                    message="Template Jinja2 valide",
                    severity="info"
                ))
                
            except TemplateSyntaxError as e:
                report.add_result(ValidationResult(
                    is_valid=False,
                    category=ValidationCategory.CONFIGURATION,
                    level=level,
                    field_path=path,
                    message=f"Erreur syntaxe template: {e}",
                    suggestion="Corriger la syntaxe Jinja2",
                    severity="error",
                    code="TEMPLATE_SYNTAX_ERROR"
                ))
            
            except Exception as e:
                report.add_result(ValidationResult(
                    is_valid=False,
                    category=ValidationCategory.CONFIGURATION,
                    level=level,
                    field_path=path,
                    message=f"Erreur validation template: {e}",
                    severity="error",
                    code="TEMPLATE_VALIDATION_ERROR"
                ))
            
            # Vérifications supplémentaires
            if level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                await self._validate_template_content(template_content, path, report, level)
    
    async def _validate_template_content(self,
                                       template_content: str,
                                       path: str,
                                       report: ValidationReport,
                                       level: ValidationLevel):
        """Valide le contenu des templates."""
        
        # Vérifier la longueur
        if len(template_content) > 10000:  # 10KB semble raisonnable
            report.add_result(ValidationResult(
                is_valid=False,
                category=ValidationCategory.PERFORMANCE,
                level=level,
                field_path=path,
                message="Template très volumineux",
                suggestion="Considérer diviser en templates plus petits",
                severity="warning",
                code="LARGE_TEMPLATE"
            ))
        
        # Vérifier la complexité (nombre de blocs)
        block_count = template_content.count('{%')
        if block_count > 50:
            report.add_result(ValidationResult(
                is_valid=False,
                category=ValidationCategory.PERFORMANCE,
                level=level,
                field_path=path,
                message=f"Template complexe: {block_count} blocs",
                suggestion="Simplifier le template pour de meilleures performances",
                severity="warning",
                code="COMPLEX_TEMPLATE"
            ))
        
        # Vérifier l'utilisation de variables non échappées
        if '{{' in template_content and '|safe' in template_content:
            report.add_result(ValidationResult(
                is_valid=False,
                category=ValidationCategory.SECURITY,
                level=level,
                field_path=path,
                message="Utilisation du filtre 'safe' détectée",
                suggestion="Vérifier que les données sont bien validées",
                severity="warning",
                code="UNSAFE_TEMPLATE_FILTER"
            ))
    
    async def _validate_routing_rules(self,
                                    config: Dict[str, Any],
                                    report: ValidationReport,
                                    level: ValidationLevel):
        """Valide les règles de routage."""
        
        routing_rules = self._extract_routing_rules(config)
        
        for path, rules in routing_rules:
            if not isinstance(rules, list):
                report.add_result(ValidationResult(
                    is_valid=False,
                    category=ValidationCategory.CONFIGURATION,
                    level=level,
                    field_path=path,
                    message="Les règles de routage doivent être une liste",
                    severity="error",
                    code="INVALID_ROUTING_RULES_TYPE"
                ))
                continue
            
            # Vérifier chaque règle
            for i, rule in enumerate(rules):
                rule_path = f"{path}[{i}]"
                await self._validate_single_routing_rule(rule, rule_path, report, level)
    
    async def _validate_single_routing_rule(self,
                                          rule: Dict[str, Any],
                                          path: str,
                                          report: ValidationReport,
                                          level: ValidationLevel):
        """Valide une règle de routage individuelle."""
        
        required_fields = ['id', 'name', 'conditions', 'target_channels']
        for field in required_fields:
            if field not in rule:
                report.add_result(ValidationResult(
                    is_valid=False,
                    category=ValidationCategory.CONFIGURATION,
                    level=level,
                    field_path=f"{path}.{field}",
                    message=f"Champ obligatoire manquant dans la règle: {field}",
                    severity="error",
                    code="MISSING_RULE_FIELD"
                ))
        
        # Valider les conditions
        if 'conditions' in rule:
            conditions = rule['conditions']
            if not isinstance(conditions, dict):
                report.add_result(ValidationResult(
                    is_valid=False,
                    category=ValidationCategory.CONFIGURATION,
                    level=level,
                    field_path=f"{path}.conditions",
                    message="Les conditions doivent être un objet",
                    severity="error",
                    code="INVALID_CONDITIONS_TYPE"
                ))
            else:
                # Vérifier la complexité des conditions
                if len(conditions) > 10:
                    report.add_result(ValidationResult(
                        is_valid=False,
                        category=ValidationCategory.PERFORMANCE,
                        level=level,
                        field_path=f"{path}.conditions",
                        message="Règle avec de nombreuses conditions (performance)",
                        suggestion="Simplifier ou diviser la règle",
                        severity="warning",
                        code="COMPLEX_ROUTING_RULE"
                    ))
        
        # Valider les canaux cibles
        if 'target_channels' in rule:
            channels = rule['target_channels']
            if not isinstance(channels, list):
                report.add_result(ValidationResult(
                    is_valid=False,
                    category=ValidationCategory.CONFIGURATION,
                    level=level,
                    field_path=f"{path}.target_channels",
                    message="Les canaux cibles doivent être une liste",
                    severity="error",
                    code="INVALID_TARGET_CHANNELS_TYPE"
                ))
            elif len(channels) == 0:
                report.add_result(ValidationResult(
                    is_valid=False,
                    category=ValidationCategory.CONFIGURATION,
                    level=level,
                    field_path=f"{path}.target_channels",
                    message="Aucun canal cible défini",
                    severity="error",
                    code="EMPTY_TARGET_CHANNELS"
                ))
    
    async def _validate_escalation_rules(self,
                                       config: Dict[str, Any],
                                       report: ValidationReport,
                                       level: ValidationLevel):
        """Valide les règles d'escalade."""
        
        escalation_rules = self._extract_escalation_rules(config)
        
        for path, rules in escalation_rules:
            if isinstance(rules, list):
                for i, rule in enumerate(rules):
                    rule_path = f"{path}[{i}]"
                    await self._validate_single_escalation_rule(rule, rule_path, report, level)
    
    async def _validate_single_escalation_rule(self,
                                             rule: Dict[str, Any],
                                             path: str,
                                             report: ValidationReport,
                                             level: ValidationLevel):
        """Valide une règle d'escalade individuelle."""
        
        # Vérifier les délais d'escalade
        if 'escalation_delays' in rule:
            delays = rule['escalation_delays']
            if isinstance(delays, list):
                for i, delay in enumerate(delays):
                    if not isinstance(delay, (int, float)) or delay < 0:
                        report.add_result(ValidationResult(
                            is_valid=False,
                            category=ValidationCategory.CONFIGURATION,
                            level=level,
                            field_path=f"{path}.escalation_delays[{i}]",
                            message="Délai d'escalade invalide",
                            severity="error",
                            code="INVALID_ESCALATION_DELAY"
                        ))
                    
                    # Vérifier des délais raisonnables
                    if delay > 3600:  # Plus d'1 heure
                        report.add_result(ValidationResult(
                            is_valid=False,
                            category=ValidationCategory.CONFIGURATION,
                            level=level,
                            field_path=f"{path}.escalation_delays[{i}]",
                            message=f"Délai d'escalade très long: {delay}s",
                            suggestion="Considérer un délai plus court",
                            severity="warning",
                            code="LONG_ESCALATION_DELAY"
                        ))
        
        # Vérifier les cibles par niveau
        if 'level_targets' in rule:
            level_targets = rule['level_targets']
            if isinstance(level_targets, dict):
                for level_name, targets in level_targets.items():
                    if not isinstance(targets, list) or len(targets) == 0:
                        report.add_result(ValidationResult(
                            is_valid=False,
                            category=ValidationCategory.CONFIGURATION,
                            level=level,
                            field_path=f"{path}.level_targets.{level_name}",
                            message=f"Aucune cible définie pour le niveau {level_name}",
                            severity="error",
                            code="NO_ESCALATION_TARGETS"
                        ))
    
    def _extract_webhooks(self, config: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Extrait tous les webhooks de la configuration."""
        webhooks = []
        
        def extract_recursive(obj: Any, path: str):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    if 'webhook' in key.lower() and isinstance(value, str):
                        webhooks.append((new_path, value))
                    extract_recursive(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_recursive(item, f"{path}[{i}]")
        
        extract_recursive(config, "")
        return webhooks
    
    def _extract_templates(self, config: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Extrait tous les templates de la configuration."""
        templates = []
        
        def extract_recursive(obj: Any, path: str):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    if 'template' in key.lower() and isinstance(value, str):
                        # Vérifier si c'est du contenu template (contient des {{}} ou {%%})
                        if '{{' in value or '{%' in value:
                            templates.append((new_path, value))
                    extract_recursive(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_recursive(item, f"{path}[{i}]")
        
        extract_recursive(config, "")
        return templates
    
    def _extract_routing_rules(self, config: Dict[str, Any]) -> List[Tuple[str, Any]]:
        """Extrait toutes les règles de routage."""
        rules = []
        
        def extract_recursive(obj: Any, path: str):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    if 'routing_rule' in key.lower():
                        rules.append((new_path, value))
                    extract_recursive(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_recursive(item, f"{path}[{i}]")
        
        extract_recursive(config, "")
        return rules
    
    def _extract_escalation_rules(self, config: Dict[str, Any]) -> List[Tuple[str, Any]]:
        """Extrait toutes les règles d'escalade."""
        rules = []
        
        def extract_recursive(obj: Any, path: str):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    if 'escalation_rule' in key.lower():
                        rules.append((new_path, value))
                    extract_recursive(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_recursive(item, f"{path}[{i}]")
        
        extract_recursive(config, "")
        return rules
    
    def _generate_recommendations(self, report: ValidationReport):
        """Génère des recommandations basées sur les résultats."""
        
        # Recommandations de sécurité
        if report.security_score < 80:
            report.recommendations.append("Améliorer la sécurité: chiffrer tous les tokens et secrets")
        
        if any(r.code == "UNENCRYPTED_TOKEN" for r in report.warnings_list):
            report.recommendations.append("Activer le chiffrement pour tous les tokens sensibles")
        
        # Recommandations de performance
        if report.performance_score < 70:
            report.recommendations.append("Optimiser les performances: revoir les timeouts et rate limits")
        
        if any(r.code == "COMPLEX_TEMPLATE" for r in report.warnings_list):
            report.recommendations.append("Simplifier les templates complexes pour de meilleures performances")
        
        # Recommandations de compliance
        if report.compliance_score < 90:
            report.recommendations.append("Améliorer la conformité: vérifier les conventions de nommage")
        
        # Recommandations générales
        if len(report.errors) > 0:
            report.recommendations.append(f"Corriger {len(report.errors)} erreur(s) critique(s)")
        
        if len(report.warnings_list) > 5:
            report.recommendations.append("Examiner et corriger les avertissements multiples")
    
    def _get_cache_key(self, config: Dict[str, Any], tenant_id: str, level: ValidationLevel) -> str:
        """Génère une clé de cache pour la validation."""
        import hashlib
        cache_data = json.dumps({
            'config_hash': hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            'tenant_id': tenant_id,
            'level': level.value
        }, sort_keys=True)
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[ValidationReport]:
        """Récupère un rapport du cache."""
        if cache_key in self._validation_cache:
            cached_item = self._validation_cache[cache_key]
            if datetime.utcnow() < cached_item['expires_at']:
                return cached_item['report']
            else:
                del self._validation_cache[cache_key]
        return None
    
    def _put_in_cache(self, cache_key: str, report: ValidationReport):
        """Met un rapport en cache."""
        from datetime import timedelta
        self._validation_cache[cache_key] = {
            'report': report,
            'expires_at': datetime.utcnow() + timedelta(seconds=self._cache_ttl)
        }
        
        # Nettoyage du cache si trop gros
        if len(self._validation_cache) > 100:
            self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Nettoie le cache expiré."""
        now = datetime.utcnow()
        expired_keys = [
            key for key, item in self._validation_cache.items()
            if now >= item['expires_at']
        ]
        
        for key in expired_keys:
            del self._validation_cache[key]
    
    async def validate_webhook_url(self, webhook_url: str) -> ValidationResult:
        """Valide une URL de webhook spécifique."""
        try:
            if not self.webhook_pattern.match(webhook_url):
                return ValidationResult(
                    is_valid=False,
                    category=ValidationCategory.INTEGRATION,
                    level=self.validation_level,
                    field_path="webhook_url",
                    message="Format de webhook invalide",
                    suggestion="Utiliser le format Slack officiel",
                    severity="error",
                    code="INVALID_WEBHOOK_FORMAT"
                )
            
            # Validation supplémentaire pour niveau strict
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                parsed = urlparse(webhook_url)
                if not parsed.netloc.endswith('slack.com'):
                    return ValidationResult(
                        is_valid=False,
                        category=ValidationCategory.SECURITY,
                        level=self.validation_level,
                        field_path="webhook_url",
                        message="Webhook ne pointe pas vers Slack",
                        severity="error",
                        code="UNTRUSTED_WEBHOOK_DOMAIN"
                    )
            
            return ValidationResult(
                is_valid=True,
                category=ValidationCategory.INTEGRATION,
                level=self.validation_level,
                field_path="webhook_url",
                message="Webhook valide",
                severity="info"
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                category=ValidationCategory.INTEGRATION,
                level=self.validation_level,
                field_path="webhook_url",
                message=f"Erreur validation webhook: {e}",
                severity="error",
                code="WEBHOOK_VALIDATION_ERROR"
            )
    
    async def validate_slack_token(self, token: str) -> ValidationResult:
        """Valide un token Slack."""
        try:
            # Enlever le préfixe de chiffrement si présent
            clean_token = token.replace('enc:', '')
            
            if not self.token_pattern.match(clean_token):
                return ValidationResult(
                    is_valid=False,
                    category=ValidationCategory.SECURITY,
                    level=self.validation_level,
                    field_path="token",
                    message="Format de token Slack invalide",
                    suggestion="Utiliser un token au format xoxb-... ou xoxa-...",
                    severity="error",
                    code="INVALID_TOKEN_FORMAT"
                )
            
            # Vérifier la longueur
            if len(clean_token) < 20:
                return ValidationResult(
                    is_valid=False,
                    category=ValidationCategory.SECURITY,
                    level=self.validation_level,
                    field_path="token",
                    message="Token trop court",
                    severity="error",
                    code="SHORT_TOKEN"
                )
            
            # Vérifier le chiffrement pour niveaux stricts
            if (self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID] and 
                not token.startswith('enc:')):
                return ValidationResult(
                    is_valid=False,
                    category=ValidationCategory.SECURITY,
                    level=self.validation_level,
                    field_path="token",
                    message="Token non chiffré",
                    suggestion="Chiffrer le token pour la sécurité",
                    severity="warning",
                    code="UNENCRYPTED_TOKEN"
                )
            
            return ValidationResult(
                is_valid=True,
                category=ValidationCategory.SECURITY,
                level=self.validation_level,
                field_path="token",
                message="Token valide",
                severity="info"
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                category=ValidationCategory.SECURITY,
                level=self.validation_level,
                field_path="token",
                message=f"Erreur validation token: {e}",
                severity="error",
                code="TOKEN_VALIDATION_ERROR"
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du validateur."""
        cache_hit_rate = 0
        if self.metrics['cache_hits'] + self.metrics['cache_misses'] > 0:
            cache_hit_rate = self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses'])
        
        return {
            **self.metrics,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self._validation_cache),
            'validation_level': self.validation_level.value,
            'supported_levels': [level.value for level in ValidationLevel],
            'validation_categories': [cat.value for cat in ValidationCategory]
        }
    
    def __repr__(self) -> str:
        return f"SlackConfigValidator(level={self.validation_level.value}, cache_size={len(self._validation_cache)})"
