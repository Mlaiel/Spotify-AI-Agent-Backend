"""
Validateur de Locales Avancé pour Spotify AI Agent
Système de validation et conformité des locales multi-tenant
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
import unicodedata
from collections import defaultdict
import asyncio

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Niveaux de validation"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    ENTERPRISE = "enterprise"


class ValidationResult(Enum):
    """Résultats de validation"""
    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Issue de validation"""
    level: ValidationResult
    code: str
    message: str
    key: Optional[str] = None
    value: Optional[str] = None
    suggestion: Optional[str] = None
    line_number: Optional[int] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Rapport de validation"""
    is_valid: bool
    issues: List[ValidationIssue]
    warnings_count: int = 0
    errors_count: int = 0
    critical_count: int = 0
    validation_time: float = 0.0
    locale_code: Optional[str] = None
    tenant_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Calcule les compteurs"""
        self.warnings_count = len([i for i in self.issues if i.level == ValidationResult.WARNING])
        self.errors_count = len([i for i in self.issues if i.level == ValidationResult.ERROR])
        self.critical_count = len([i for i in self.issues if i.level == ValidationResult.CRITICAL])


@dataclass
class ValidationConfig:
    """Configuration de validation"""
    level: ValidationLevel = ValidationLevel.STANDARD
    check_syntax: bool = True
    check_completeness: bool = True
    check_consistency: bool = True
    check_formatting: bool = True
    check_security: bool = True
    check_performance: bool = True
    check_accessibility: bool = True
    check_compliance: bool = True
    max_key_length: int = 100
    max_value_length: int = 1000
    required_keys: Set[str] = field(default_factory=set)
    forbidden_patterns: List[str] = field(default_factory=list)
    allowed_html_tags: Set[str] = field(default_factory=lambda: {
        'b', 'i', 'u', 'strong', 'em', 'br', 'span', 'a'
    })
    pluralization_required: bool = True
    encoding_check: bool = True
    rtl_support: bool = True


class ValidationRule(ABC):
    """Interface pour les règles de validation"""
    
    @property
    @abstractmethod
    def rule_code(self) -> str:
        """Code unique de la règle"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description de la règle"""
        pass
    
    @abstractmethod
    async def validate(
        self, 
        data: Dict[str, Any], 
        config: ValidationConfig,
        context: Dict[str, Any] = None
    ) -> List[ValidationIssue]:
        """Valide les données selon cette règle"""
        pass


class SyntaxValidationRule(ValidationRule):
    """Validation de la syntaxe"""
    
    @property
    def rule_code(self) -> str:
        return "SYNTAX"
    
    @property
    def description(self) -> str:
        return "Validation de la syntaxe des clés et valeurs"
    
    async def validate(
        self, 
        data: Dict[str, Any], 
        config: ValidationConfig,
        context: Dict[str, Any] = None
    ) -> List[ValidationIssue]:
        """Valide la syntaxe"""
        issues = []
        
        if not isinstance(data, dict):
            issues.append(ValidationIssue(
                level=ValidationResult.CRITICAL,
                code="SYNTAX_001",
                message="Les données doivent être un dictionnaire"
            ))
            return issues
        
        for key, value in data.items():
            # Validation des clés
            if not isinstance(key, str):
                issues.append(ValidationIssue(
                    level=ValidationResult.ERROR,
                    code="SYNTAX_002",
                    message="Les clés doivent être des chaînes de caractères",
                    key=str(key)
                ))
                continue
            
            if not key.strip():
                issues.append(ValidationIssue(
                    level=ValidationResult.ERROR,
                    code="SYNTAX_003",
                    message="Les clés ne peuvent pas être vides",
                    key=key
                ))
                continue
            
            if len(key) > config.max_key_length:
                issues.append(ValidationIssue(
                    level=ValidationResult.WARNING,
                    code="SYNTAX_004",
                    message=f"Clé trop longue ({len(key)} > {config.max_key_length})",
                    key=key,
                    suggestion=f"Raccourcir la clé à moins de {config.max_key_length} caractères"
                ))
            
            # Validation des caractères dans les clés
            if not re.match(r'^[a-zA-Z0-9._-]+$', key):
                issues.append(ValidationIssue(
                    level=ValidationResult.WARNING,
                    code="SYNTAX_005",
                    message="La clé contient des caractères non recommandés",
                    key=key,
                    suggestion="Utiliser uniquement des lettres, chiffres, points, underscores et tirets"
                ))
            
            # Validation des valeurs
            if isinstance(value, str):
                if len(value) > config.max_value_length:
                    issues.append(ValidationIssue(
                        level=ValidationResult.WARNING,
                        code="SYNTAX_006",
                        message=f"Valeur trop longue ({len(value)} > {config.max_value_length})",
                        key=key,
                        value=value[:100] + "..." if len(value) > 100 else value
                    ))
                
                # Vérification de l'encoding
                if config.encoding_check:
                    try:
                        value.encode('utf-8')
                    except UnicodeEncodeError as e:
                        issues.append(ValidationIssue(
                            level=ValidationResult.ERROR,
                            code="SYNTAX_007",
                            message=f"Problème d'encodage: {e}",
                            key=key,
                            value=value
                        ))
            
            elif not isinstance(value, (str, int, float, bool, list, dict)):
                issues.append(ValidationIssue(
                    level=ValidationResult.ERROR,
                    code="SYNTAX_008",
                    message="Type de valeur non supporté",
                    key=key,
                    value=str(type(value))
                ))
        
        return issues


class CompletenessValidationRule(ValidationRule):
    """Validation de la complétude"""
    
    @property
    def rule_code(self) -> str:
        return "COMPLETENESS"
    
    @property
    def description(self) -> str:
        return "Validation de la complétude des traductions"
    
    async def validate(
        self, 
        data: Dict[str, Any], 
        config: ValidationConfig,
        context: Dict[str, Any] = None
    ) -> List[ValidationIssue]:
        """Valide la complétude"""
        issues = []
        
        # Vérification des clés requises
        for required_key in config.required_keys:
            if required_key not in data:
                issues.append(ValidationIssue(
                    level=ValidationResult.ERROR,
                    code="COMPLETENESS_001",
                    message=f"Clé requise manquante: {required_key}",
                    key=required_key,
                    suggestion=f"Ajouter la clé '{required_key}' avec une valeur appropriée"
                ))
        
        # Vérification des valeurs vides
        empty_values = []
        for key, value in data.items():
            if isinstance(value, str) and not value.strip():
                empty_values.append(key)
            elif value is None:
                empty_values.append(key)
        
        if empty_values:
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                code="COMPLETENESS_002",
                message=f"Valeurs vides détectées: {', '.join(empty_values[:5])}",
                suggestion="Fournir des traductions pour toutes les clés"
            ))
        
        # Vérification de la pluralisation
        if config.pluralization_required:
            plural_keys = [k for k in data.keys() if any(
                suffix in k for suffix in ['_zero', '_one', '_two', '_few', '_many', '_other']
            )]
            
            if not plural_keys:
                issues.append(ValidationIssue(
                    level=ValidationResult.WARNING,
                    code="COMPLETENESS_003",
                    message="Aucune règle de pluralisation détectée",
                    suggestion="Ajouter des clés avec suffixes _zero, _one, _other pour la pluralisation"
                ))
        
        return issues


class ConsistencyValidationRule(ValidationRule):
    """Validation de la cohérence"""
    
    @property
    def rule_code(self) -> str:
        return "CONSISTENCY"
    
    @property
    def description(self) -> str:
        return "Validation de la cohérence des traductions"
    
    async def validate(
        self, 
        data: Dict[str, Any], 
        config: ValidationConfig,
        context: Dict[str, Any] = None
    ) -> List[ValidationIssue]:
        """Valide la cohérence"""
        issues = []
        
        # Vérification des placeholders
        placeholder_patterns = {}
        for key, value in data.items():
            if isinstance(value, str):
                placeholders = re.findall(r'\{[^}]+\}', value)
                if placeholders:
                    placeholder_patterns[key] = set(placeholders)
        
        # Comparaison avec la locale de référence
        reference_data = context.get('reference_data', {}) if context else {}
        for key, value in data.items():
            if key in reference_data and isinstance(value, str) and isinstance(reference_data[key], str):
                ref_placeholders = set(re.findall(r'\{[^}]+\}', reference_data[key]))
                curr_placeholders = set(re.findall(r'\{[^}]+\}', value))
                
                if ref_placeholders != curr_placeholders:
                    missing = ref_placeholders - curr_placeholders
                    extra = curr_placeholders - ref_placeholders
                    
                    message_parts = []
                    if missing:
                        message_parts.append(f"Placeholders manquants: {', '.join(missing)}")
                    if extra:
                        message_parts.append(f"Placeholders en trop: {', '.join(extra)}")
                    
                    issues.append(ValidationIssue(
                        level=ValidationResult.ERROR,
                        code="CONSISTENCY_001",
                        message=f"Incohérence des placeholders: {'; '.join(message_parts)}",
                        key=key,
                        value=value,
                        suggestion="Aligner les placeholders avec la locale de référence"
                    ))
        
        # Vérification de la cohérence terminologique
        term_variations = defaultdict(set)
        for key, value in data.items():
            if isinstance(value, str):
                words = re.findall(r'\b\w+\b', value.lower())
                for word in words:
                    if len(word) > 3:  # Ignorer les mots très courts
                        term_variations[word].add(key)
        
        # Détecter les variations de casse potentiellement problématiques
        case_issues = {}
        for key, value in data.items():
            if isinstance(value, str):
                # Vérifier la cohérence de la casse pour les termes récurrents
                words = re.findall(r'\b[A-Z][a-z]+\b', value)
                for word in words:
                    if word.lower() in term_variations and len(term_variations[word.lower()]) > 1:
                        if key not in case_issues:
                            case_issues[key] = []
                        case_issues[key].append(word)
        
        for key, words in case_issues.items():
            if len(words) > 2:  # Seulement si plusieurs occurrences
                issues.append(ValidationIssue(
                    level=ValidationResult.WARNING,
                    code="CONSISTENCY_002",
                    message=f"Variations de casse détectées: {', '.join(words[:3])}",
                    key=key,
                    suggestion="Vérifier la cohérence de la casse des termes"
                ))
        
        return issues


class SecurityValidationRule(ValidationRule):
    """Validation de sécurité"""
    
    @property
    def rule_code(self) -> str:
        return "SECURITY"
    
    @property
    def description(self) -> str:
        return "Validation de sécurité des traductions"
    
    async def validate(
        self, 
        data: Dict[str, Any], 
        config: ValidationConfig,
        context: Dict[str, Any] = None
    ) -> List[ValidationIssue]:
        """Valide la sécurité"""
        issues = []
        
        # Patterns dangereux
        dangerous_patterns = [
            (r'<script[^>]*>', 'SECURITY_001', 'Script tag détecté'),
            (r'javascript:', 'SECURITY_002', 'JavaScript URL détecté'),
            (r'on\w+\s*=', 'SECURITY_003', 'Event handler détecté'),
            (r'eval\s*\(', 'SECURITY_004', 'Fonction eval détectée'),
            (r'document\.(write|cookie)', 'SECURITY_005', 'Accès DOM sensible détecté'),
            (r'window\.(location|open)', 'SECURITY_006', 'Manipulation de fenêtre détectée'),
            (r'<!--.*?-->', 'SECURITY_007', 'Commentaire HTML détecté'),
            (r'<\?.*?\?>', 'SECURITY_008', 'Code serveur détecté')
        ]
        
        for key, value in data.items():
            if isinstance(value, str):
                for pattern, code, message in dangerous_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        issues.append(ValidationIssue(
                            level=ValidationResult.CRITICAL,
                            code=code,
                            message=message,
                            key=key,
                            value=value,
                            suggestion="Supprimer ou échapper le contenu dangereux"
                        ))
                
                # Vérification des balises HTML autorisées
                html_tags = re.findall(r'<(/?)(\w+)[^>]*>', value)
                for is_closing, tag in html_tags:
                    if tag.lower() not in config.allowed_html_tags:
                        issues.append(ValidationIssue(
                            level=ValidationResult.WARNING,
                            code="SECURITY_009",
                            message=f"Balise HTML non autorisée: {tag}",
                            key=key,
                            value=value,
                            suggestion=f"Utiliser uniquement ces balises: {', '.join(config.allowed_html_tags)}"
                        ))
                
                # Vérification des patterns interdits
                for forbidden_pattern in config.forbidden_patterns:
                    if re.search(forbidden_pattern, value, re.IGNORECASE):
                        issues.append(ValidationIssue(
                            level=ValidationResult.ERROR,
                            code="SECURITY_010",
                            message=f"Pattern interdit détecté: {forbidden_pattern}",
                            key=key,
                            value=value,
                            suggestion="Supprimer le contenu interdit"
                        ))
        
        return issues


class AccessibilityValidationRule(ValidationRule):
    """Validation d'accessibilité"""
    
    @property
    def rule_code(self) -> str:
        return "ACCESSIBILITY"
    
    @property
    def description(self) -> str:
        return "Validation d'accessibilité des traductions"
    
    async def validate(
        self, 
        data: Dict[str, Any], 
        config: ValidationConfig,
        context: Dict[str, Any] = None
    ) -> List[ValidationIssue]:
        """Valide l'accessibilité"""
        issues = []
        
        for key, value in data.items():
            if isinstance(value, str):
                # Vérification des liens sans texte descriptif
                links = re.findall(r'<a[^>]*>([^<]*)</a>', value, re.IGNORECASE)
                for link_text in links:
                    if not link_text.strip() or link_text.strip().lower() in ['here', 'click', 'link']:
                        issues.append(ValidationIssue(
                            level=ValidationResult.WARNING,
                            code="ACCESSIBILITY_001",
                            message="Lien sans texte descriptif",
                            key=key,
                            value=value,
                            suggestion="Utiliser un texte de lien descriptif"
                        ))
                
                # Vérification des images sans alt
                imgs = re.findall(r'<img[^>]*>', value, re.IGNORECASE)
                for img in imgs:
                    if 'alt=' not in img.lower():
                        issues.append(ValidationIssue(
                            level=ValidationResult.ERROR,
                            code="ACCESSIBILITY_002",
                            message="Image sans attribut alt",
                            key=key,
                            value=value,
                            suggestion="Ajouter un attribut alt descriptif"
                        ))
                
                # Vérification de la longueur des phrases
                sentences = re.split(r'[.!?]+', value)
                for sentence in sentences:
                    words = len(sentence.split())
                    if words > 25:
                        issues.append(ValidationIssue(
                            level=ValidationResult.WARNING,
                            code="ACCESSIBILITY_003",
                            message=f"Phrase longue ({words} mots)",
                            key=key,
                            suggestion="Diviser en phrases plus courtes pour améliorer la lisibilité"
                        ))
                
                # Vérification des contrastes (basique)
                if any(color in value.lower() for color in ['color:', 'background:']):
                    issues.append(ValidationIssue(
                        level=ValidationResult.WARNING,
                        code="ACCESSIBILITY_004",
                        message="Style de couleur détecté",
                        key=key,
                        suggestion="Vérifier le contraste des couleurs"
                    ))
        
        return issues


class LocaleValidator:
    """Validateur principal des locales"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self._rules = [
            SyntaxValidationRule(),
            CompletenessValidationRule(),
            ConsistencyValidationRule(),
            SecurityValidationRule(),
            AccessibilityValidationRule()
        ]
        self._stats = defaultdict(int)
    
    async def validate_locale(
        self, 
        data: Dict[str, Any],
        locale_code: Optional[str] = None,
        tenant_id: Optional[str] = None,
        reference_data: Optional[Dict[str, Any]] = None
    ) -> ValidationReport:
        """Valide une locale complète"""
        start_time = datetime.now()
        all_issues = []
        
        try:
            context = {
                'locale_code': locale_code,
                'tenant_id': tenant_id,
                'reference_data': reference_data
            }
            
            # Exécuter toutes les règles de validation
            validation_tasks = []
            for rule in self._rules:
                if self._should_run_rule(rule):
                    task = rule.validate(data, self.config, context)
                    validation_tasks.append(task)
            
            # Exécuter les validations en parallèle
            results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Validation rule error: {result}")
                    continue
                
                if isinstance(result, list):
                    all_issues.extend(result)
            
            # Calculer la validité
            has_critical = any(issue.level == ValidationResult.CRITICAL for issue in all_issues)
            has_errors = any(issue.level == ValidationResult.ERROR for issue in all_issues)
            
            is_valid = not (has_critical or (has_errors and self.config.level in [ValidationLevel.STRICT, ValidationLevel.ENTERPRISE]))
            
            validation_time = (datetime.now() - start_time).total_seconds()
            
            report = ValidationReport(
                is_valid=is_valid,
                issues=all_issues,
                validation_time=validation_time,
                locale_code=locale_code,
                tenant_id=tenant_id
            )
            
            # Mettre à jour les statistiques
            self._stats['validations_performed'] += 1
            if is_valid:
                self._stats['validations_passed'] += 1
            else:
                self._stats['validations_failed'] += 1
            
            return report
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            validation_time = (datetime.now() - start_time).total_seconds()
            
            return ValidationReport(
                is_valid=False,
                issues=[ValidationIssue(
                    level=ValidationResult.CRITICAL,
                    code="VALIDATOR_ERROR",
                    message=f"Erreur de validation: {e}"
                )],
                validation_time=validation_time,
                locale_code=locale_code,
                tenant_id=tenant_id
            )
    
    async def validate_multiple_locales(
        self, 
        locales_data: Dict[str, Dict[str, Any]],
        tenant_id: Optional[str] = None
    ) -> Dict[str, ValidationReport]:
        """Valide plusieurs locales"""
        try:
            # Prendre la première locale comme référence
            reference_locale = next(iter(locales_data.keys())) if locales_data else None
            reference_data = locales_data.get(reference_locale, {}) if reference_locale else None
            
            validation_tasks = []
            locale_codes = []
            
            for locale_code, data in locales_data.items():
                task = self.validate_locale(
                    data, 
                    locale_code, 
                    tenant_id, 
                    reference_data if locale_code != reference_locale else None
                )
                validation_tasks.append(task)
                locale_codes.append(locale_code)
            
            results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            reports = {}
            for locale_code, result in zip(locale_codes, results):
                if isinstance(result, Exception):
                    logger.error(f"Validation error for {locale_code}: {result}")
                    reports[locale_code] = ValidationReport(
                        is_valid=False,
                        issues=[ValidationIssue(
                            level=ValidationResult.CRITICAL,
                            code="VALIDATION_EXCEPTION",
                            message=str(result)
                        )],
                        locale_code=locale_code,
                        tenant_id=tenant_id
                    )
                else:
                    reports[locale_code] = result
            
            return reports
            
        except Exception as e:
            logger.error(f"Multi-locale validation error: {e}")
            return {}
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de validation"""
        return {
            'stats': dict(self._stats),
            'config': {
                'level': self.config.level.value,
                'checks_enabled': {
                    'syntax': self.config.check_syntax,
                    'completeness': self.config.check_completeness,
                    'consistency': self.config.check_consistency,
                    'formatting': self.config.check_formatting,
                    'security': self.config.check_security,
                    'performance': self.config.check_performance,
                    'accessibility': self.config.check_accessibility,
                    'compliance': self.config.check_compliance
                }
            },
            'rules_count': len(self._rules)
        }
    
    def _should_run_rule(self, rule: ValidationRule) -> bool:
        """Détermine si une règle doit être exécutée"""
        rule_mapping = {
            'SYNTAX': self.config.check_syntax,
            'COMPLETENESS': self.config.check_completeness,
            'CONSISTENCY': self.config.check_consistency,
            'SECURITY': self.config.check_security,
            'ACCESSIBILITY': self.config.check_accessibility
        }
        
        return rule_mapping.get(rule.rule_code, True)


class ComplianceValidator:
    """Validateur de conformité réglementaire"""
    
    def __init__(self):
        self._compliance_rules = {
            'GDPR': self._validate_gdpr,
            'CCPA': self._validate_ccpa,
            'WCAG': self._validate_wcag,
            'ISO27001': self._validate_iso27001
        }
    
    async def validate_compliance(
        self, 
        data: Dict[str, Any],
        standards: List[str] = None
    ) -> Dict[str, ValidationReport]:
        """Valide la conformité aux standards"""
        standards = standards or ['GDPR', 'WCAG']
        reports = {}
        
        for standard in standards:
            if standard in self._compliance_rules:
                try:
                    report = await self._compliance_rules[standard](data)
                    reports[standard] = report
                except Exception as e:
                    logger.error(f"Compliance validation error for {standard}: {e}")
                    reports[standard] = ValidationReport(
                        is_valid=False,
                        issues=[ValidationIssue(
                            level=ValidationResult.ERROR,
                            code=f"{standard}_ERROR",
                            message=f"Erreur de validation {standard}: {e}"
                        )]
                    )
        
        return reports
    
    async def _validate_gdpr(self, data: Dict[str, Any]) -> ValidationReport:
        """Valide la conformité GDPR"""
        issues = []
        
        # Vérifications GDPR basiques
        gdpr_required_keys = [
            'privacy_policy',
            'data_processing_consent',
            'data_deletion_request',
            'data_portability',
            'contact_dpo'
        ]
        
        for key in gdpr_required_keys:
            if key not in data:
                issues.append(ValidationIssue(
                    level=ValidationResult.ERROR,
                    code="GDPR_001",
                    message=f"Clé GDPR manquante: {key}",
                    suggestion=f"Ajouter la traduction pour {key}"
                ))
        
        return ValidationReport(
            is_valid=len(issues) == 0,
            issues=issues
        )
    
    async def _validate_ccpa(self, data: Dict[str, Any]) -> ValidationReport:
        """Valide la conformité CCPA"""
        issues = []
        
        ccpa_required_keys = [
            'do_not_sell',
            'personal_info_categories',
            'consumer_rights'
        ]
        
        for key in ccpa_required_keys:
            if key not in data:
                issues.append(ValidationIssue(
                    level=ValidationResult.ERROR,
                    code="CCPA_001",
                    message=f"Clé CCPA manquante: {key}",
                    suggestion=f"Ajouter la traduction pour {key}"
                ))
        
        return ValidationReport(
            is_valid=len(issues) == 0,
            issues=issues
        )
    
    async def _validate_wcag(self, data: Dict[str, Any]) -> ValidationReport:
        """Valide la conformité WCAG"""
        issues = []
        
        # Vérifications d'accessibilité WCAG
        for key, value in data.items():
            if isinstance(value, str):
                # Vérifier les images sans alt
                if '<img' in value and 'alt=' not in value:
                    issues.append(ValidationIssue(
                        level=ValidationResult.ERROR,
                        code="WCAG_001",
                        message="Image sans texte alternatif",
                        key=key
                    ))
                
                # Vérifier les contrastes (approximatif)
                if any(color in value.lower() for color in ['#fff', '#000', 'white', 'black']):
                    issues.append(ValidationIssue(
                        level=ValidationResult.WARNING,
                        code="WCAG_002",
                        message="Vérifier le contraste des couleurs",
                        key=key
                    ))
        
        return ValidationReport(
            is_valid=len([i for i in issues if i.level == ValidationResult.ERROR]) == 0,
            issues=issues
        )
    
    async def _validate_iso27001(self, data: Dict[str, Any]) -> ValidationReport:
        """Valide la conformité ISO 27001"""
        issues = []
        
        # Vérifications de sécurité ISO 27001
        security_keys = [
            'security_policy',
            'incident_response',
            'access_control',
            'data_classification'
        ]
        
        for key in security_keys:
            if key not in data:
                issues.append(ValidationIssue(
                    level=ValidationResult.WARNING,
                    code="ISO27001_001",
                    message=f"Clé de sécurité recommandée manquante: {key}",
                    suggestion=f"Considérer l'ajout de {key}"
                ))
        
        return ValidationReport(
            is_valid=True,  # Warnings seulement
            issues=issues
        )
