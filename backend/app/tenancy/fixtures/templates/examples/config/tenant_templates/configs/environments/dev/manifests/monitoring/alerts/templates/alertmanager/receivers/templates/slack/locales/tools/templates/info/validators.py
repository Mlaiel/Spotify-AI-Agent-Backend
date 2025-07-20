"""
✅ Advanced Content Validators - Production-Ready System
======================================================

Validateurs ultra-avancés pour contenu avec ML, sécurité et conformité.
Architecture industrielle pour validation multi-niveaux et détection intelligente.
"""

import re
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import langdetect
from profanity_check import predict as is_profane

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Niveaux de validation"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    ENTERPRISE = "enterprise"
    ULTRA_SECURE = "ultra_secure"


class ValidationType(Enum):
    """Types de validation"""
    CONTENT_SAFETY = "content_safety"
    LANGUAGE_DETECTION = "language_detection"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    SPAM_DETECTION = "spam_detection"
    COMPLIANCE_CHECK = "compliance_check"
    FORMAT_VALIDATION = "format_validation"
    LENGTH_VALIDATION = "length_validation"
    ENCODING_VALIDATION = "encoding_validation"
    SECURITY_SCAN = "security_scan"
    READABILITY_CHECK = "readability_check"


class ValidationSeverity(Enum):
    """Sévérité des problèmes détectés"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    BLOCKED = "blocked"


@dataclass
class ValidationResult:
    """Résultat de validation"""
    is_valid: bool
    severity: ValidationSeverity
    validation_type: ValidationType
    message: str
    details: Dict[str, Any]
    score: float = 0.0
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


@dataclass
class ValidationReport:
    """Rapport complet de validation"""
    content_id: str
    overall_valid: bool
    validation_level: ValidationLevel
    timestamp: datetime
    results: List[ValidationResult]
    
    # Scores globaux
    safety_score: float = 0.0
    quality_score: float = 0.0
    compliance_score: float = 0.0
    
    # Métriques
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    warnings: int = 0
    
    def __post_init__(self):
        self.total_checks = len(self.results)
        self.passed_checks = sum(1 for r in self.results if r.is_valid)
        self.failed_checks = self.total_checks - self.passed_checks
        self.warnings = sum(1 for r in self.results if r.severity == ValidationSeverity.WARNING)
        
        # Calcul des scores
        if self.results:
            self.safety_score = np.mean([r.score for r in self.results if r.validation_type in [
                ValidationType.CONTENT_SAFETY, ValidationType.SPAM_DETECTION, ValidationType.SECURITY_SCAN
            ]])
            self.quality_score = np.mean([r.score for r in self.results if r.validation_type in [
                ValidationType.READABILITY_CHECK, ValidationType.FORMAT_VALIDATION, ValidationType.LENGTH_VALIDATION
            ]])
            self.compliance_score = np.mean([r.score for r in self.results if r.validation_type == ValidationType.COMPLIANCE_CHECK])


class BaseValidator(ABC):
    """Validateur de base"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Configuration de validation
        self.validation_level = ValidationLevel(config.get('validation_level', 'standard'))
        self.enable_ml = config.get('enable_ml', True)
        self.strict_mode = config.get('strict_mode', False)
        
        # Cache des résultats
        self.validation_cache: Dict[str, ValidationResult] = {}
        
        # Métriques
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'blocked_content': 0,
            'avg_validation_time': 0.0
        }
    
    @abstractmethod
    async def validate(self, content: str, metadata: Dict[str, Any]) -> ValidationResult:
        """Validation du contenu (méthode abstraite)"""
        pass
    
    def _calculate_score(self, issues_found: int, total_checks: int) -> float:
        """Calcul du score de validation"""
        if total_checks == 0:
            return 1.0
        return max(0.0, 1.0 - (issues_found / total_checks))
    
    def _update_stats(self, success: bool, validation_time: float):
        """Mise à jour des statistiques"""
        self.validation_stats['total_validations'] += 1
        if success:
            self.validation_stats['successful_validations'] += 1
        
        # Calcul de la moyenne mobile du temps de validation
        current_avg = self.validation_stats['avg_validation_time']
        total = self.validation_stats['total_validations']
        new_avg = (current_avg * (total - 1) + validation_time) / total
        self.validation_stats['avg_validation_time'] = new_avg


class ContentSafetyValidator(BaseValidator):
    """Validateur de sécurité du contenu"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Modèles ML pour détection
        self._init_ml_models()
        
        # Listes de mots interdits
        self.profanity_enabled = config.get('profanity_check', True)
        self.custom_blocklist = config.get('custom_blocklist', [])
        
        # Patterns de détection
        self.security_patterns = [
            r'<script.*?>.*?</script>',  # Scripts
            r'javascript:',  # JavaScript URLs
            r'on\w+\s*=',  # Event handlers
            r'eval\s*\(',  # eval() calls
            r'expression\s*\(',  # CSS expressions
        ]
        
        # Patterns d'injection
        self.injection_patterns = [
            r'union\s+select',  # SQL injection
            r'drop\s+table',  # SQL drop
            r'exec\s*\(',  # Command execution
            r'system\s*\(',  # System calls
        ]
    
    def _init_ml_models(self):
        """Initialisation des modèles ML"""
        try:
            if self.enable_ml:
                # Modèle de détection de toxicité
                self.toxicity_classifier = pipeline(
                    "text-classification",
                    model="unitary/toxic-bert",
                    device=-1  # CPU
                )
                
                # Modèle de détection de haine
                self.hate_classifier = pipeline(
                    "text-classification", 
                    model="cardiffnlp/twitter-roberta-base-hate-latest",
                    device=-1
                )
                
                self.logger.info("ML models for content safety initialized")
            else:
                self.toxicity_classifier = None
                self.hate_classifier = None
                
        except Exception as e:
            self.logger.error(f"Failed to initialize ML models: {str(e)}")
            self.toxicity_classifier = None
            self.hate_classifier = None
    
    async def validate(self, content: str, metadata: Dict[str, Any]) -> ValidationResult:
        """Validation de la sécurité du contenu"""
        
        start_time = datetime.utcnow()
        issues = []
        suggestions = []
        
        try:
            # 1. Vérification des patterns de sécurité
            security_issues = self._check_security_patterns(content)
            issues.extend(security_issues)
            
            # 2. Vérification des injections
            injection_issues = self._check_injection_patterns(content)
            issues.extend(injection_issues)
            
            # 3. Vérification de la profanité
            if self.profanity_enabled:
                profanity_issues = await self._check_profanity(content)
                issues.extend(profanity_issues)
            
            # 4. Vérification ML (toxicité)
            if self.enable_ml and self.toxicity_classifier:
                ml_issues = await self._check_toxicity_ml(content)
                issues.extend(ml_issues)
            
            # 5. Vérification ML (haine)
            if self.enable_ml and self.hate_classifier:
                hate_issues = await self._check_hate_ml(content)
                issues.extend(hate_issues)
            
            # 6. Vérification de la liste personnalisée
            custom_issues = self._check_custom_blocklist(content)
            issues.extend(custom_issues)
            
            # Calcul du score et de la sévérité
            total_checks = 6
            score = self._calculate_score(len(issues), total_checks)
            
            # Détermination de la sévérité
            if len(issues) == 0:
                severity = ValidationSeverity.INFO
                is_valid = True
            elif len(issues) <= 2 and not any('critical' in issue for issue in issues):
                severity = ValidationSeverity.WARNING
                is_valid = True
                suggestions.append("Review content for potential safety concerns")
            elif len(issues) <= 4:
                severity = ValidationSeverity.ERROR
                is_valid = False
                suggestions.append("Significant safety issues detected, content modification required")
            else:
                severity = ValidationSeverity.BLOCKED
                is_valid = False
                suggestions.append("Critical safety violations, content blocked")
            
            # Mise à jour des statistiques
            validation_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_stats(is_valid, validation_time)
            
            return ValidationResult(
                is_valid=is_valid,
                severity=severity,
                validation_type=ValidationType.CONTENT_SAFETY,
                message=f"Content safety validation: {len(issues)} issues found",
                details={
                    'issues': issues,
                    'checks_performed': total_checks,
                    'validation_time_ms': validation_time * 1000
                },
                score=score,
                suggestions=suggestions
            )
            
        except Exception as e:
            self.logger.error(f"Content safety validation failed: {str(e)}")
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                validation_type=ValidationType.CONTENT_SAFETY,
                message=f"Validation error: {str(e)}",
                details={'error': str(e)},
                score=0.0,
                suggestions=["Retry validation or contact support"]
            )
    
    def _check_security_patterns(self, content: str) -> List[str]:
        """Vérification des patterns de sécurité"""
        issues = []
        
        for pattern in self.security_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                issues.append(f"Security pattern detected: {pattern}")
        
        return issues
    
    def _check_injection_patterns(self, content: str) -> List[str]:
        """Vérification des patterns d'injection"""
        issues = []
        
        for pattern in self.injection_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(f"Injection pattern detected: {pattern}")
        
        return issues
    
    async def _check_profanity(self, content: str) -> List[str]:
        """Vérification de la profanité"""
        issues = []
        
        try:
            if is_profane(content):
                issues.append("Profanity detected in content")
        except Exception as e:
            self.logger.warning(f"Profanity check failed: {str(e)}")
        
        return issues
    
    async def _check_toxicity_ml(self, content: str) -> List[str]:
        """Vérification de la toxicité avec ML"""
        issues = []
        
        try:
            result = self.toxicity_classifier(content)
            
            # Analyse du résultat
            for prediction in result:
                if prediction['label'] == 'TOXIC' and prediction['score'] > 0.7:
                    issues.append(f"High toxicity detected (score: {prediction['score']:.2f})")
                elif prediction['label'] == 'TOXIC' and prediction['score'] > 0.5:
                    issues.append(f"Moderate toxicity detected (score: {prediction['score']:.2f})")
                    
        except Exception as e:
            self.logger.warning(f"Toxicity ML check failed: {str(e)}")
        
        return issues
    
    async def _check_hate_ml(self, content: str) -> List[str]:
        """Vérification de la haine avec ML"""
        issues = []
        
        try:
            result = self.hate_classifier(content)
            
            # Analyse du résultat
            for prediction in result:
                if 'hate' in prediction['label'].lower() and prediction['score'] > 0.6:
                    issues.append(f"Hate speech detected (score: {prediction['score']:.2f})")
                    
        except Exception as e:
            self.logger.warning(f"Hate speech ML check failed: {str(e)}")
        
        return issues
    
    def _check_custom_blocklist(self, content: str) -> List[str]:
        """Vérification de la liste personnalisée"""
        issues = []
        
        content_lower = content.lower()
        for blocked_term in self.custom_blocklist:
            if blocked_term.lower() in content_lower:
                issues.append(f"Blocked term detected: {blocked_term}")
        
        return issues


class LanguageDetectionValidator(BaseValidator):
    """Validateur de détection de langue"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Langues supportées
        self.supported_languages = config.get('supported_languages', ['en', 'fr', 'de', 'es'])
        self.require_language_match = config.get('require_language_match', True)
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
    
    async def validate(self, content: str, metadata: Dict[str, Any]) -> ValidationResult:
        """Validation de la langue"""
        
        start_time = datetime.utcnow()
        issues = []
        suggestions = []
        
        try:
            # Détection de la langue
            detected_language = langdetect.detect(content)
            confidence = self._calculate_language_confidence(content, detected_language)
            
            expected_language = metadata.get('language', 'en')
            
            # Vérifications
            if detected_language not in self.supported_languages:
                issues.append(f"Unsupported language detected: {detected_language}")
                suggestions.append(f"Translate content to one of: {', '.join(self.supported_languages)}")
            
            if self.require_language_match and detected_language != expected_language:
                issues.append(f"Language mismatch: expected {expected_language}, detected {detected_language}")
                suggestions.append(f"Verify content language or update metadata")
            
            if confidence < self.confidence_threshold:
                issues.append(f"Low language detection confidence: {confidence:.2f}")
                suggestions.append("Review content clarity and language consistency")
            
            # Calcul du score
            score = confidence if len(issues) == 0 else confidence * 0.5
            
            # Détermination de la validité
            is_valid = len(issues) == 0
            severity = ValidationSeverity.INFO if is_valid else ValidationSeverity.WARNING
            
            validation_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_stats(is_valid, validation_time)
            
            return ValidationResult(
                is_valid=is_valid,
                severity=severity,
                validation_type=ValidationType.LANGUAGE_DETECTION,
                message=f"Language detection: {detected_language} (confidence: {confidence:.2f})",
                details={
                    'detected_language': detected_language,
                    'expected_language': expected_language,
                    'confidence': confidence,
                    'supported_languages': self.supported_languages,
                    'issues': issues
                },
                score=score,
                suggestions=suggestions
            )
            
        except Exception as e:
            self.logger.error(f"Language detection failed: {str(e)}")
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                validation_type=ValidationType.LANGUAGE_DETECTION,
                message=f"Language detection error: {str(e)}",
                details={'error': str(e)},
                score=0.0,
                suggestions=["Check content encoding and retry"]
            )
    
    def _calculate_language_confidence(self, content: str, language: str) -> float:
        """Calcul de la confiance de détection de langue"""
        
        try:
            # Utilisation de langdetect avec probabilités
            from langdetect import detect_langs
            
            probs = detect_langs(content)
            for prob in probs:
                if prob.lang == language:
                    return prob.prob
            
            return 0.0
            
        except Exception:
            # Fallback avec estimation basique
            if len(content) > 50:
                return 0.8
            elif len(content) > 20:
                return 0.6
            else:
                return 0.4


class ComplianceValidator(BaseValidator):
    """Validateur de conformité (GDPR, etc.)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Réglementations activées
        self.gdpr_enabled = config.get('gdpr_enabled', True)
        self.ccpa_enabled = config.get('ccpa_enabled', False)
        self.hipaa_enabled = config.get('hipaa_enabled', False)
        
        # Patterns de données sensibles
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'iban': r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b'
        }
        
        # Mots-clés de données sensibles
        self.sensitive_keywords = [
            'password', 'mdp', 'mot de passe', 'passwort',
            'secret', 'token', 'key', 'clé', 'schlüssel',
            'medical', 'médical', 'medizinisch',
            'health', 'santé', 'gesundheit'
        ]
    
    async def validate(self, content: str, metadata: Dict[str, Any]) -> ValidationResult:
        """Validation de conformité"""
        
        start_time = datetime.utcnow()
        issues = []
        suggestions = []
        
        try:
            # Détection de PII
            pii_found = self._detect_pii(content)
            if pii_found:
                issues.extend([f"PII detected: {pii_type}" for pii_type in pii_found])
                suggestions.append("Remove or anonymize personal information")
            
            # Détection de mots-clés sensibles
            sensitive_found = self._detect_sensitive_keywords(content)
            if sensitive_found:
                issues.extend([f"Sensitive keyword: {keyword}" for keyword in sensitive_found])
                suggestions.append("Review and redact sensitive information")
            
            # Vérifications spécifiques GDPR
            if self.gdpr_enabled:
                gdpr_issues = self._check_gdpr_compliance(content, metadata)
                issues.extend(gdpr_issues)
            
            # Vérifications spécifiques HIPAA
            if self.hipaa_enabled:
                hipaa_issues = self._check_hipaa_compliance(content)
                issues.extend(hipaa_issues)
            
            # Calcul du score
            total_checks = len(self.pii_patterns) + len(self.sensitive_keywords) + 2
            score = self._calculate_score(len(issues), total_checks)
            
            # Détermination de la sévérité
            if len(issues) == 0:
                severity = ValidationSeverity.INFO
                is_valid = True
            elif len(issues) <= 2:
                severity = ValidationSeverity.WARNING
                is_valid = True
            else:
                severity = ValidationSeverity.ERROR
                is_valid = False
                suggestions.append("Critical compliance violations detected")
            
            validation_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_stats(is_valid, validation_time)
            
            return ValidationResult(
                is_valid=is_valid,
                severity=severity,
                validation_type=ValidationType.COMPLIANCE_CHECK,
                message=f"Compliance validation: {len(issues)} issues found",
                details={
                    'pii_detected': pii_found,
                    'sensitive_keywords': sensitive_found,
                    'gdpr_enabled': self.gdpr_enabled,
                    'issues': issues
                },
                score=score,
                suggestions=suggestions
            )
            
        except Exception as e:
            self.logger.error(f"Compliance validation failed: {str(e)}")
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                validation_type=ValidationType.COMPLIANCE_CHECK,
                message=f"Compliance validation error: {str(e)}",
                details={'error': str(e)},
                score=0.0,
                suggestions=["Contact compliance team for review"]
            )
    
    def _detect_pii(self, content: str) -> List[str]:
        """Détection de données personnelles"""
        pii_found = []
        
        for pii_type, pattern in self.pii_patterns.items():
            if re.search(pattern, content):
                pii_found.append(pii_type)
        
        return pii_found
    
    def _detect_sensitive_keywords(self, content: str) -> List[str]:
        """Détection de mots-clés sensibles"""
        content_lower = content.lower()
        sensitive_found = []
        
        for keyword in self.sensitive_keywords:
            if keyword.lower() in content_lower:
                sensitive_found.append(keyword)
        
        return sensitive_found
    
    def _check_gdpr_compliance(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """Vérification GDPR"""
        issues = []
        
        # Vérification du consentement pour données personnelles
        if self._detect_pii(content) and not metadata.get('user_consent', False):
            issues.append("GDPR: Personal data without explicit consent")
        
        # Vérification de la finalité de traitement
        if not metadata.get('processing_purpose'):
            issues.append("GDPR: Processing purpose not specified")
        
        return issues
    
    def _check_hipaa_compliance(self, content: str) -> List[str]:
        """Vérification HIPAA"""
        issues = []
        
        # Patterns médicaux spécifiques
        medical_patterns = [
            r'\b(patient|diagnosis|treatment|medication)\b',
            r'\b(medical record|health record)\b',
            r'\b(doctor|physician|nurse)\b'
        ]
        
        for pattern in medical_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append("HIPAA: Medical information detected")
                break
        
        return issues


class InfoValidator:
    """Validateur principal pour le module info"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialisation des validateurs
        self.validators = {
            ValidationType.CONTENT_SAFETY: ContentSafetyValidator(config),
            ValidationType.LANGUAGE_DETECTION: LanguageDetectionValidator(config),
            ValidationType.COMPLIANCE_CHECK: ComplianceValidator(config)
        }
        
        # Configuration globale
        self.validation_level = ValidationLevel(config.get('validation_level', 'standard'))
        self.parallel_validation = config.get('parallel_validation', True)
        
        # Cache des rapports
        self.report_cache: Dict[str, ValidationReport] = {}
    
    async def validate_content(
        self, 
        content: str, 
        metadata: Dict[str, Any],
        validation_types: Optional[List[ValidationType]] = None
    ) -> ValidationReport:
        """Validation complète du contenu"""
        
        content_id = metadata.get('content_id', f"content_{hash(content)}")
        
        # Sélection des validateurs à utiliser
        if validation_types is None:
            validation_types = list(self.validators.keys())
        
        # Exécution des validations
        if self.parallel_validation:
            results = await self._parallel_validation(content, metadata, validation_types)
        else:
            results = await self._sequential_validation(content, metadata, validation_types)
        
        # Création du rapport
        report = ValidationReport(
            content_id=content_id,
            overall_valid=all(r.is_valid for r in results),
            validation_level=self.validation_level,
            timestamp=datetime.utcnow(),
            results=results
        )
        
        # Mise en cache
        self.report_cache[content_id] = report
        
        # Nettoyage du cache si nécessaire
        if len(self.report_cache) > 1000:
            await self._cleanup_cache()
        
        return report
    
    async def _parallel_validation(
        self, 
        content: str, 
        metadata: Dict[str, Any], 
        validation_types: List[ValidationType]
    ) -> List[ValidationResult]:
        """Validation en parallèle"""
        
        import asyncio
        
        # Création des tâches
        tasks = []
        for validation_type in validation_types:
            if validation_type in self.validators:
                validator = self.validators[validation_type]
                task = validator.validate(content, metadata)
                tasks.append(task)
        
        # Exécution parallèle
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Traitement des résultats
        validation_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Validation failed: {str(result)}")
                # Création d'un résultat d'erreur
                validation_results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    validation_type=validation_types[i],
                    message=f"Validation error: {str(result)}",
                    details={'error': str(result)},
                    score=0.0
                ))
            else:
                validation_results.append(result)
        
        return validation_results
    
    async def _sequential_validation(
        self, 
        content: str, 
        metadata: Dict[str, Any], 
        validation_types: List[ValidationType]
    ) -> List[ValidationResult]:
        """Validation séquentielle"""
        
        results = []
        
        for validation_type in validation_types:
            if validation_type in self.validators:
                try:
                    validator = self.validators[validation_type]
                    result = await validator.validate(content, metadata)
                    results.append(result)
                    
                    # Arrêt anticipé en cas d'erreur critique
                    if (result.severity == ValidationSeverity.BLOCKED and 
                        self.validation_level == ValidationLevel.ULTRA_SECURE):
                        break
                        
                except Exception as e:
                    self.logger.error(f"Validation {validation_type.value} failed: {str(e)}")
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        validation_type=validation_type,
                        message=f"Validation error: {str(e)}",
                        details={'error': str(e)},
                        score=0.0
                    ))
        
        return results
    
    async def _cleanup_cache(self):
        """Nettoyage du cache des rapports"""
        
        # Suppression des rapports les plus anciens
        sorted_reports = sorted(
            self.report_cache.items(), 
            key=lambda x: x[1].timestamp
        )
        
        # Garder seulement les 500 plus récents
        for content_id, _ in sorted_reports[:-500]:
            del self.report_cache[content_id]
        
        self.logger.info(f"Cache cleaned up: {len(sorted_reports) - 500} reports removed")


class ContentValidator:
    """Validateur de contenu avec règles métier"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Règles de validation
        self.length_rules = config.get('length_rules', {
            'min_length': 10,
            'max_length': 4096,
            'optimal_length': 150
        })
        
        self.format_rules = config.get('format_rules', {
            'allow_html': False,
            'allow_markdown': True,
            'require_structure': False
        })
        
        # Patterns requis/interdits
        self.required_patterns = config.get('required_patterns', [])
        self.forbidden_patterns = config.get('forbidden_patterns', [])
    
    async def validate_business_rules(
        self, 
        content: str, 
        metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validation des règles métier"""
        
        issues = []
        suggestions = []
        
        try:
            # Validation de longueur
            length_issues = self._validate_length(content)
            issues.extend(length_issues)
            
            # Validation de format
            format_issues = self._validate_format(content)
            issues.extend(format_issues)
            
            # Validation des patterns
            pattern_issues = self._validate_patterns(content)
            issues.extend(pattern_issues)
            
            # Validation de structure
            structure_issues = self._validate_structure(content)
            issues.extend(structure_issues)
            
            # Calcul du score
            total_checks = 4
            score = self._calculate_score(len(issues), total_checks)
            
            # Détermination de la validité
            is_valid = len(issues) == 0
            severity = ValidationSeverity.INFO if is_valid else ValidationSeverity.WARNING
            
            if len(issues) > 2:
                severity = ValidationSeverity.ERROR
                is_valid = False
            
            return ValidationResult(
                is_valid=is_valid,
                severity=severity,
                validation_type=ValidationType.FORMAT_VALIDATION,
                message=f"Business rules validation: {len(issues)} issues found",
                details={
                    'issues': issues,
                    'length': len(content),
                    'rules_checked': total_checks
                },
                score=score,
                suggestions=suggestions
            )
            
        except Exception as e:
            self.logger.error(f"Business rules validation failed: {str(e)}")
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                validation_type=ValidationType.FORMAT_VALIDATION,
                message=f"Validation error: {str(e)}",
                details={'error': str(e)},
                score=0.0
            )
    
    def _validate_length(self, content: str) -> List[str]:
        """Validation de longueur"""
        issues = []
        length = len(content)
        
        if length < self.length_rules['min_length']:
            issues.append(f"Content too short: {length} < {self.length_rules['min_length']}")
        
        if length > self.length_rules['max_length']:
            issues.append(f"Content too long: {length} > {self.length_rules['max_length']}")
        
        return issues
    
    def _validate_format(self, content: str) -> List[str]:
        """Validation de format"""
        issues = []
        
        # Vérification HTML
        if not self.format_rules['allow_html'] and '<' in content and '>' in content:
            if re.search(r'<[^>]+>', content):
                issues.append("HTML tags not allowed")
        
        # Vérification Markdown
        if not self.format_rules['allow_markdown']:
            markdown_patterns = [r'\*\*', r'__', r'##', r'\[.*\]\(.*\)']
            for pattern in markdown_patterns:
                if re.search(pattern, content):
                    issues.append("Markdown formatting not allowed")
                    break
        
        return issues
    
    def _validate_patterns(self, content: str) -> List[str]:
        """Validation des patterns"""
        issues = []
        
        # Patterns requis
        for pattern in self.required_patterns:
            if not re.search(pattern, content, re.IGNORECASE):
                issues.append(f"Required pattern missing: {pattern}")
        
        # Patterns interdits
        for pattern in self.forbidden_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(f"Forbidden pattern found: {pattern}")
        
        return issues
    
    def _validate_structure(self, content: str) -> List[str]:
        """Validation de structure"""
        issues = []
        
        if self.format_rules['require_structure']:
            # Vérification de la présence d'éléments structurels
            has_header = bool(re.search(r'^#+ ', content, re.MULTILINE))
            has_paragraphs = len(content.split('\n\n')) > 1
            
            if not has_header:
                issues.append("Structure requires header")
            
            if not has_paragraphs:
                issues.append("Structure requires multiple paragraphs")
        
        return issues
    
    def _calculate_score(self, issues_found: int, total_checks: int) -> float:
        """Calcul du score de validation"""
        if total_checks == 0:
            return 1.0
        return max(0.0, 1.0 - (issues_found / total_checks))
