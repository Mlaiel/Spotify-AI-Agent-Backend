"""
Validateurs de Données Localisées - Spotify AI Agent
==================================================

Module de validation sécurisée pour les données localisées avec protection
contre les injections, validation des formats et nettoyage automatique
des entrées utilisateur selon les standards de sécurité.

Fonctionnalités:
- Validation stricte des codes de locale
- Protection contre les injections XSS et SQL
- Validation des formats de données numériques
- Nettoyage et sanitization des chaînes
- Validation des templates d'alerte

Author: Fahed Mlaiel
"""

import re
import html
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
import logging
import unicodedata
import bleach

from . import LocaleType


class ValidationLevel(Enum):
    """Niveaux de validation"""
    STRICT = "strict"
    NORMAL = "normal"
    PERMISSIVE = "permissive"


class ValidationError(Exception):
    """Exception de validation personnalisée"""
    def __init__(self, message: str, field: str = "", value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)


@dataclass
class ValidationResult:
    """Résultat de validation"""
    is_valid: bool
    cleaned_value: Any = None
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class DataValidator:
    """Validateur principal pour les données localisées"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.NORMAL):
        self.logger = logging.getLogger(__name__)
        self.validation_level = validation_level
        
        # Patterns de validation
        self._patterns = {
            'locale_code': re.compile(r'^[a-z]{2}_[A-Z]{2}$'),
            'tenant_id': re.compile(r'^[a-zA-Z0-9_-]{1,64}$'),
            'alert_id': re.compile(r'^[a-zA-Z0-9_-]{1,128}$'),
            'template_string': re.compile(r'^[^<>{}]*$'),  # Pas de HTML ou JS
            'currency_code': re.compile(r'^[A-Z]{3}$'),
            'percentage': re.compile(r'^\d{1,3}(\.\d{1,2})?$'),
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'url': re.compile(r'^https?://[^\s<>"\']+$'),
            'ipv4': re.compile(r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'),
            'safe_string': re.compile(r'^[a-zA-Z0-9\s\-_.,!?()]+$')
        }
        
        # Tags HTML autorisés pour le nettoyage
        self._allowed_html_tags = {
            'b', 'i', 'u', 'strong', 'em', 'br', 'p', 'span'
        }
        
        # Attributs HTML autorisés
        self._allowed_html_attributes = {
            'class': ['highlight', 'warning', 'error'],
            'style': []  # Aucun style inline autorisé
        }
    
    def validate_locale_code(self, locale_code: str) -> ValidationResult:
        """Valide un code de locale"""
        try:
            # Vérifie le format de base
            if not isinstance(locale_code, str):
                return ValidationResult(
                    is_valid=False,
                    errors=["Locale code must be a string"]
                )
            
            # Vérifie le pattern
            if not self._patterns['locale_code'].match(locale_code):
                return ValidationResult(
                    is_valid=False,
                    errors=["Invalid locale code format. Expected format: xx_XX"]
                )
            
            # Vérifie si la locale est supportée
            try:
                locale_enum = LocaleType(locale_code)
                return ValidationResult(
                    is_valid=True,
                    cleaned_value=locale_enum
                )
            except ValueError:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Unsupported locale: {locale_code}"]
                )
                
        except Exception as e:
            self.logger.error(f"Locale validation error: {e}")
            return ValidationResult(
                is_valid=False,
                errors=["Locale validation failed"]
            )
    
    def validate_number(
        self, 
        value: Union[str, int, float], 
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_negative: bool = True
    ) -> ValidationResult:
        """Valide et nettoie une valeur numérique"""
        try:
            # Conversion en nombre
            if isinstance(value, str):
                # Nettoie la chaîne
                cleaned_str = re.sub(r'[^\d\-+.,]', '', value)
                
                # Gère les différents formats de séparateurs
                if ',' in cleaned_str and '.' in cleaned_str:
                    # Format avec séparateur de milliers
                    if cleaned_str.rfind(',') > cleaned_str.rfind('.'):
                        # Virgule comme décimal (format européen)
                        cleaned_str = cleaned_str.replace('.', '').replace(',', '.')
                    else:
                        # Point comme décimal (format US)
                        cleaned_str = cleaned_str.replace(',', '')
                elif ',' in cleaned_str:
                    # Seule virgule - peut être décimal ou milliers
                    parts = cleaned_str.split(',')
                    if len(parts) == 2 and len(parts[1]) <= 2:
                        # Probablement décimal
                        cleaned_str = cleaned_str.replace(',', '.')
                    else:
                        # Probablement séparateur de milliers
                        cleaned_str = cleaned_str.replace(',', '')
                
                try:
                    numeric_value = Decimal(cleaned_str)
                except InvalidOperation:
                    return ValidationResult(
                        is_valid=False,
                        errors=["Invalid number format"]
                    )
            else:
                numeric_value = Decimal(str(value))
            
            # Validations
            errors = []
            warnings = []
            
            if not allow_negative and numeric_value < 0:
                errors.append("Negative values not allowed")
            
            if min_value is not None and numeric_value < Decimal(str(min_value)):
                errors.append(f"Value must be >= {min_value}")
            
            if max_value is not None and numeric_value > Decimal(str(max_value)):
                errors.append(f"Value must be <= {max_value}")
            
            # Vérifie les valeurs extrêmes
            if abs(numeric_value) > Decimal('1e15'):
                warnings.append("Very large number detected")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                cleaned_value=float(numeric_value),
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            self.logger.error(f"Number validation error: {e}")
            return ValidationResult(
                is_valid=False,
                errors=["Number validation failed"]
            )
    
    def validate_string(
        self, 
        value: str,
        min_length: int = 0,
        max_length: int = 1000,
        allow_html: bool = False,
        pattern: Optional[str] = None
    ) -> ValidationResult:
        """Valide et nettoie une chaîne de caractères"""
        try:
            if not isinstance(value, str):
                return ValidationResult(
                    is_valid=False,
                    errors=["Value must be a string"]
                )
            
            errors = []
            warnings = []
            cleaned_value = value
            
            # Normalise Unicode
            cleaned_value = unicodedata.normalize('NFKC', cleaned_value)
            
            # Supprime les caractères de contrôle dangereux
            cleaned_value = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned_value)
            
            # Vérifie la longueur
            if len(cleaned_value) < min_length:
                errors.append(f"String too short (minimum: {min_length})")
            
            if len(cleaned_value) > max_length:
                if self.validation_level == ValidationLevel.STRICT:
                    errors.append(f"String too long (maximum: {max_length})")
                else:
                    cleaned_value = cleaned_value[:max_length]
                    warnings.append(f"String truncated to {max_length} characters")
            
            # Gestion HTML
            if not allow_html:
                # Échappe le HTML
                cleaned_value = html.escape(cleaned_value)
            else:
                # Nettoie le HTML avec bleach
                cleaned_value = bleach.clean(
                    cleaned_value,
                    tags=self._allowed_html_tags,
                    attributes=self._allowed_html_attributes,
                    strip=True
                )
            
            # Vérifie le pattern si fourni
            if pattern:
                if not re.match(pattern, cleaned_value):
                    errors.append("String does not match required pattern")
            
            # Détecte les tentatives d'injection
            injection_patterns = [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'vbscript:',
                r'onload\s*=',
                r'onerror\s*=',
                r'onclick\s*=',
                r'eval\s*\(',
                r'expression\s*\(',
                r'--',  # Commentaires SQL
                r';.*drop\s+table',
                r';.*delete\s+from',
                r';.*insert\s+into',
                r';.*update\s+.*set'
            ]
            
            for pattern_check in injection_patterns:
                if re.search(pattern_check, cleaned_value, re.IGNORECASE):
                    if self.validation_level == ValidationLevel.STRICT:
                        errors.append("Potential injection attempt detected")
                    else:
                        warnings.append("Suspicious content detected and cleaned")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                cleaned_value=cleaned_value,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            self.logger.error(f"String validation error: {e}")
            return ValidationResult(
                is_valid=False,
                errors=["String validation failed"]
            )
    
    def validate_template_string(self, template: str) -> ValidationResult:
        """Valide un template de message d'alerte"""
        try:
            # Validation de base
            string_result = self.validate_string(
                template,
                min_length=1,
                max_length=2000,
                allow_html=False
            )
            
            if not string_result.is_valid:
                return string_result
            
            cleaned_template = string_result.cleaned_value
            errors = string_result.errors[:]
            warnings = string_result.warnings[:]
            
            # Vérifie la syntaxe des placeholders
            placeholder_pattern = r'\{([^}]+)\}'
            placeholders = re.findall(placeholder_pattern, cleaned_template)
            
            for placeholder in placeholders:
                # Vérifie que le placeholder est sûr
                if not re.match(r'^[a-zA-Z0-9_]+$', placeholder):
                    errors.append(f"Invalid placeholder: {{{placeholder}}}")
            
            # Vérifie les accolades non appariées
            open_braces = cleaned_template.count('{')
            close_braces = cleaned_template.count('}')
            
            if open_braces != close_braces:
                errors.append("Mismatched braces in template")
            
            # Test de formatage avec des valeurs fictives
            try:
                test_values = {placeholder: "test" for placeholder in placeholders}
                test_formatted = cleaned_template.format(**test_values)
            except KeyError as e:
                errors.append(f"Missing placeholder value: {e}")
            except Exception as e:
                errors.append(f"Template formatting error: {e}")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                cleaned_value=cleaned_template,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            self.logger.error(f"Template validation error: {e}")
            return ValidationResult(
                is_valid=False,
                errors=["Template validation failed"]
            )
    
    def validate_json_data(self, data: Union[str, dict]) -> ValidationResult:
        """Valide des données JSON"""
        try:
            if isinstance(data, str):
                try:
                    parsed_data = json.loads(data)
                except json.JSONDecodeError as e:
                    return ValidationResult(
                        is_valid=False,
                        errors=[f"Invalid JSON: {e}"]
                    )
            else:
                parsed_data = data
            
            # Vérifie la profondeur de nesting
            def check_depth(obj, current_depth=0, max_depth=10):
                if current_depth > max_depth:
                    return False
                
                if isinstance(obj, dict):
                    return all(check_depth(v, current_depth + 1, max_depth) for v in obj.values())
                elif isinstance(obj, list):
                    return all(check_depth(item, current_depth + 1, max_depth) for item in obj)
                
                return True
            
            if not check_depth(parsed_data):
                return ValidationResult(
                    is_valid=False,
                    errors=["JSON structure too deeply nested"]
                )
            
            return ValidationResult(
                is_valid=True,
                cleaned_value=parsed_data
            )
            
        except Exception as e:
            self.logger.error(f"JSON validation error: {e}")
            return ValidationResult(
                is_valid=False,
                errors=["JSON validation failed"]
            )
    
    def validate_alert_parameters(self, parameters: Dict[str, Any]) -> ValidationResult:
        """Valide les paramètres d'une alerte"""
        try:
            errors = []
            warnings = []
            cleaned_parameters = {}
            
            for key, value in parameters.items():
                # Valide la clé
                key_result = self.validate_string(key, max_length=64, pattern=r'^[a-zA-Z0-9_]+$')
                if not key_result.is_valid:
                    errors.extend([f"Parameter key '{key}': {error}" for error in key_result.errors])
                    continue
                
                cleaned_key = key_result.cleaned_value
                
                # Valide la valeur selon son type
                if isinstance(value, (int, float)):
                    value_result = self.validate_number(value)
                    if value_result.is_valid:
                        cleaned_parameters[cleaned_key] = value_result.cleaned_value
                    else:
                        errors.extend([f"Parameter '{key}': {error}" for error in value_result.errors])
                
                elif isinstance(value, str):
                    value_result = self.validate_string(value, max_length=500)
                    if value_result.is_valid:
                        cleaned_parameters[cleaned_key] = value_result.cleaned_value
                    else:
                        errors.extend([f"Parameter '{key}': {error}" for error in value_result.errors])
                
                elif isinstance(value, bool):
                    cleaned_parameters[cleaned_key] = value
                
                elif isinstance(value, datetime):
                    cleaned_parameters[cleaned_key] = value
                
                else:
                    # Tente de convertir en string et valider
                    str_value = str(value)
                    value_result = self.validate_string(str_value, max_length=500)
                    if value_result.is_valid:
                        cleaned_parameters[cleaned_key] = value_result.cleaned_value
                        warnings.append(f"Parameter '{key}' converted to string")
                    else:
                        errors.append(f"Parameter '{key}' has unsupported type: {type(value)}")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                cleaned_value=cleaned_parameters,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            self.logger.error(f"Alert parameters validation error: {e}")
            return ValidationResult(
                is_valid=False,
                errors=["Alert parameters validation failed"]
            )


# Instance globale du validateur
data_validator = DataValidator()

__all__ = [
    "ValidationLevel",
    "ValidationError",
    "ValidationResult",
    "DataValidator",
    "data_validator"
]
