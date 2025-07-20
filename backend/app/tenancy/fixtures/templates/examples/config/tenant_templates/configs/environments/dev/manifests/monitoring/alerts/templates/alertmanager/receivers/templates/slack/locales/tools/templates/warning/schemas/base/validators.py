"""
Validateurs avancés - Spotify AI Agent
Système de validation métier sophistiqué avec règles personnalisables
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Callable, Pattern, Type
from uuid import UUID
import re
import ipaddress
import json
import ast
import sqlparse
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import validators as external_validators

from pydantic import BaseModel, Field, validator, ValidationError
from pydantic.types import EmailStr

from .enums import AlertLevel, Priority, SecurityLevel, Environment


class ValidationSeverity(str, Enum):
    """Sévérité des erreurs de validation"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationScope(str, Enum):
    """Portée de la validation"""
    FIELD = "field"
    MODEL = "model"
    BUSINESS = "business"
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"


@dataclass
class ValidationResult:
    """Résultat détaillé de validation"""
    is_valid: bool = True
    errors: List[str] = None
    warnings: List[str] = None
    info: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        self.errors = self.errors or []
        self.warnings = self.warnings or []
        self.info = self.info or []
        self.metadata = self.metadata or {}
    
    def add_error(self, message: str, field: Optional[str] = None, code: Optional[str] = None):
        """Ajoute une erreur de validation"""
        if field:
            message = f"{field}: {message}"
        if code:
            message = f"[{code}] {message}"
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str, field: Optional[str] = None, code: Optional[str] = None):
        """Ajoute un avertissement"""
        if field:
            message = f"{field}: {message}"
        if code:
            message = f"[{code}] {message}"
        self.warnings.append(message)
    
    def add_info(self, message: str, field: Optional[str] = None):
        """Ajoute une information"""
        if field:
            message = f"{field}: {message}"
        self.info.append(message)
    
    def merge(self, other: 'ValidationResult'):
        """Fusionne avec un autre résultat"""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)
        self.metadata.update(other.metadata)
        if not other.is_valid:
            self.is_valid = False
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0
    
    @property
    def total_issues(self) -> int:
        return len(self.errors) + len(self.warnings)


class BaseValidator:
    """Validateur de base avec configuration flexible"""
    
    def __init__(self, 
                 strict_mode: bool = False,
                 stop_on_first_error: bool = False,
                 max_errors: int = 100,
                 context: Optional[Dict[str, Any]] = None):
        self.strict_mode = strict_mode
        self.stop_on_first_error = stop_on_first_error
        self.max_errors = max_errors
        self.context = context or {}
        self.validation_rules: List[Callable] = []
    
    def add_rule(self, rule: Callable[[Any], ValidationResult]):
        """Ajoute une règle de validation"""
        self.validation_rules.append(rule)
    
    def validate(self, value: Any, field_name: Optional[str] = None) -> ValidationResult:
        """Valide une valeur avec toutes les règles"""
        result = ValidationResult()
        
        for rule in self.validation_rules:
            try:
                rule_result = rule(value)
                result.merge(rule_result)
                
                if self.stop_on_first_error and not rule_result.is_valid:
                    break
                
                if len(result.errors) >= self.max_errors:
                    result.add_warning("Maximum d'erreurs atteint, validation interrompue")
                    break
                    
            except Exception as e:
                result.add_error(f"Erreur lors de la validation: {str(e)}")
                if self.stop_on_first_error:
                    break
        
        return result


class StringValidator(BaseValidator):
    """Validateur spécialisé pour les chaînes de caractères"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_string_rules()
    
    def _setup_string_rules(self):
        """Configure les règles par défaut pour les chaînes"""
        self.add_rule(self._validate_not_empty)
        self.add_rule(self._validate_length)
        self.add_rule(self._validate_pattern)
        self.add_rule(self._validate_encoding)
        self.add_rule(self._validate_security)
    
    def _validate_not_empty(self, value: str) -> ValidationResult:
        """Valide que la chaîne n'est pas vide"""
        result = ValidationResult()
        if not value or not value.strip():
            result.add_error("La chaîne ne peut pas être vide")
        return result
    
    def _validate_length(self, value: str) -> ValidationResult:
        """Valide la longueur de la chaîne"""
        result = ValidationResult()
        min_length = self.context.get('min_length', 0)
        max_length = self.context.get('max_length', 10000)
        
        if len(value) < min_length:
            result.add_error(f"Longueur minimale requise: {min_length} (actuel: {len(value)})")
        
        if len(value) > max_length:
            result.add_error(f"Longueur maximale autorisée: {max_length} (actuel: {len(value)})")
        
        return result
    
    def _validate_pattern(self, value: str) -> ValidationResult:
        """Valide les patterns regex"""
        result = ValidationResult()
        patterns = self.context.get('patterns', [])
        
        for pattern_config in patterns:
            if isinstance(pattern_config, str):
                pattern = pattern_config
                message = f"Ne correspond pas au pattern: {pattern}"
            else:
                pattern = pattern_config.get('pattern')
                message = pattern_config.get('message', f"Ne correspond pas au pattern: {pattern}")
            
            if pattern and not re.match(pattern, value):
                result.add_error(message)
        
        return result
    
    def _validate_encoding(self, value: str) -> ValidationResult:
        """Valide l'encodage de la chaîne"""
        result = ValidationResult()
        allowed_encodings = self.context.get('allowed_encodings', ['utf-8'])
        
        for encoding in allowed_encodings:
            try:
                value.encode(encoding)
                return result  # Encodage valide trouvé
            except UnicodeEncodeError:
                continue
        
        result.add_error(f"Encodage non supporté. Encodages autorisés: {allowed_encodings}")
        return result
    
    def _validate_security(self, value: str) -> ValidationResult:
        """Valide la sécurité de la chaîne"""
        result = ValidationResult()
        
        # Détection d'injections potentielles
        dangerous_patterns = [
            (r'<script[^>]*>.*?</script>', 'Script JavaScript détecté'),
            (r'javascript:', 'URL JavaScript détectée'),
            (r'on\w+\s*=', 'Gestionnaire d\'événement HTML détecté'),
            (r'(union|select|insert|update|delete|drop)\s+', 'Commande SQL potentielle détectée'),
            (r'(\|\||&&|\||&)', 'Opérateur de commande shell détecté'),
        ]
        
        for pattern, message in dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                if self.strict_mode:
                    result.add_error(message)
                else:
                    result.add_warning(message)
        
        return result


class NumberValidator(BaseValidator):
    """Validateur pour les nombres"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_number_rules()
    
    def _setup_number_rules(self):
        """Configure les règles pour les nombres"""
        self.add_rule(self._validate_range)
        self.add_rule(self._validate_precision)
        self.add_rule(self._validate_special_values)
    
    def _validate_range(self, value: Union[int, float, Decimal]) -> ValidationResult:
        """Valide la plage de valeurs"""
        result = ValidationResult()
        min_value = self.context.get('min_value')
        max_value = self.context.get('max_value')
        
        if min_value is not None and value < min_value:
            result.add_error(f"Valeur trop petite: {value} < {min_value}")
        
        if max_value is not None and value > max_value:
            result.add_error(f"Valeur trop grande: {value} > {max_value}")
        
        return result
    
    def _validate_precision(self, value: Union[int, float, Decimal]) -> ValidationResult:
        """Valide la précision des nombres décimaux"""
        result = ValidationResult()
        
        if isinstance(value, float):
            max_decimal_places = self.context.get('max_decimal_places')
            if max_decimal_places is not None:
                decimal_str = str(value).split('.')
                if len(decimal_str) > 1 and len(decimal_str[1]) > max_decimal_places:
                    result.add_error(f"Trop de décimales: {len(decimal_str[1])} > {max_decimal_places}")
        
        return result
    
    def _validate_special_values(self, value: Union[int, float, Decimal]) -> ValidationResult:
        """Valide les valeurs spéciales (NaN, Infinity)"""
        result = ValidationResult()
        
        if isinstance(value, float):
            if value != value:  # NaN check
                result.add_error("Valeur NaN non autorisée")
            elif value == float('inf') or value == float('-inf'):
                if not self.context.get('allow_infinity', False):
                    result.add_error("Valeur infinie non autorisée")
        
        return result


class EmailValidator(BaseValidator):
    """Validateur pour les adresses email"""
    
    def validate_email(self, email: str) -> ValidationResult:
        """Valide une adresse email"""
        result = ValidationResult()
        
        try:
            # Validation basique avec pydantic
            EmailStr.validate(email)
        except ValidationError as e:
            result.add_error(f"Format email invalide: {str(e)}")
            return result
        
        # Validations supplémentaires
        self._validate_domain_rules(email, result)
        self._validate_business_rules(email, result)
        
        return result
    
    def _validate_domain_rules(self, email: str, result: ValidationResult):
        """Valide les règles de domaine"""
        domain = email.split('@')[1].lower()
        
        # Domaines interdits
        blocked_domains = self.context.get('blocked_domains', set())
        if domain in blocked_domains:
            result.add_error(f"Domaine interdit: {domain}")
        
        # Domaines requis
        required_domains = self.context.get('required_domains', set())
        if required_domains and domain not in required_domains:
            result.add_error(f"Domaine non autorisé: {domain}")
        
        # Validation DNS (optionnelle)
        if self.context.get('check_dns', False):
            try:
                import dns.resolver
                dns.resolver.resolve(domain, 'MX')
            except:
                result.add_warning(f"Aucun enregistrement MX trouvé pour {domain}")
    
    def _validate_business_rules(self, email: str, result: ValidationResult):
        """Valide les règles métier"""
        local_part = email.split('@')[0]
        
        # Longueur de la partie locale
        if len(local_part) > 64:
            result.add_error("Partie locale trop longue (>64 caractères)")
        
        # Caractères consécutifs
        if '..' in local_part:
            result.add_error("Points consécutifs non autorisés")
        
        # Début/fin par un point
        if local_part.startswith('.') or local_part.endswith('.'):
            result.add_error("La partie locale ne peut pas commencer ou finir par un point")


class URLValidator(BaseValidator):
    """Validateur pour les URLs"""
    
    def validate_url(self, url: str) -> ValidationResult:
        """Valide une URL"""
        result = ValidationResult()
        
        # Validation basique
        if not external_validators.url(url):
            result.add_error("Format URL invalide")
            return result
        
        # Validations spécifiques
        self._validate_scheme(url, result)
        self._validate_security(url, result)
        self._validate_accessibility(url, result)
        
        return result
    
    def _validate_scheme(self, url: str, result: ValidationResult):
        """Valide le schéma de l'URL"""
        allowed_schemes = self.context.get('allowed_schemes', {'http', 'https'})
        scheme = url.split('://')[0].lower()
        
        if scheme not in allowed_schemes:
            result.add_error(f"Schéma non autorisé: {scheme}")
        
        # HTTPS requis en production
        if self.context.get('require_https', False) and scheme != 'https':
            result.add_error("HTTPS requis")
    
    def _validate_security(self, url: str, result: ValidationResult):
        """Valide la sécurité de l'URL"""
        # URLs suspectes
        suspicious_patterns = [
            r'bit\.ly', r'tinyurl\.com', r'localhost', r'127\.0\.0\.1',
            r'192\.168\.', r'10\.', r'172\.(1[6-9]|2[0-9]|3[01])\.'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                result.add_warning(f"URL potentiellement suspecte: {pattern}")
    
    def _validate_accessibility(self, url: str, result: ValidationResult):
        """Valide l'accessibilité de l'URL"""
        if self.context.get('check_accessibility', False):
            try:
                import requests
                response = requests.head(url, timeout=5, allow_redirects=True)
                if response.status_code >= 400:
                    result.add_warning(f"URL non accessible: {response.status_code}")
            except Exception:
                result.add_warning("Impossible de vérifier l'accessibilité de l'URL")


class SQLValidator(BaseValidator):
    """Validateur pour les requêtes SQL"""
    
    def validate_sql(self, sql: str) -> ValidationResult:
        """Valide une requête SQL"""
        result = ValidationResult()
        
        # Validation syntaxique
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                result.add_error("Requête SQL vide ou invalide")
                return result
        except Exception as e:
            result.add_error(f"Erreur de syntaxe SQL: {str(e)}")
            return result
        
        # Validations de sécurité
        self._validate_sql_security(sql, result)
        self._validate_sql_performance(sql, result)
        self._validate_sql_complexity(sql, result)
        
        return result
    
    def _validate_sql_security(self, sql: str, result: ValidationResult):
        """Valide la sécurité de la requête SQL"""
        dangerous_keywords = [
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'CREATE', 'ALTER',
            'TRUNCATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE'
        ]
        
        sql_upper = sql.upper()
        for keyword in dangerous_keywords:
            if f' {keyword} ' in f' {sql_upper} ':
                if self.context.get('read_only', True):
                    result.add_error(f"Opération non autorisée: {keyword}")
                else:
                    result.add_warning(f"Opération potentiellement dangereuse: {keyword}")
        
        # Détection d'injection SQL
        injection_patterns = [
            r"';\s*(DROP|DELETE|UPDATE)",
            r"UNION\s+SELECT",
            r"--\s*$",
            r"/\*.*\*/"
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                result.add_error("Pattern d'injection SQL détecté")
    
    def _validate_sql_performance(self, sql: str, result: ValidationResult):
        """Valide les aspects de performance"""
        # Détection de requêtes potentiellement lentes
        performance_warnings = [
            (r'SELECT\s+\*', "SELECT * peut être inefficace"),
            (r'WHERE.*LIKE\s+\'%.*%\'', "LIKE avec % au début peut être lent"),
            (r'ORDER\s+BY.*RAND\(\)', "ORDER BY RAND() est très lent"),
            (r'GROUP\s+BY.*HAVING', "HAVING sans WHERE peut être inefficace")
        ]
        
        for pattern, message in performance_warnings:
            if re.search(pattern, sql, re.IGNORECASE):
                result.add_warning(message)
    
    def _validate_sql_complexity(self, sql: str, result: ValidationResult):
        """Valide la complexité de la requête"""
        max_joins = self.context.get('max_joins', 10)
        max_subqueries = self.context.get('max_subqueries', 5)
        
        # Compter les JOINs
        join_count = len(re.findall(r'\bJOIN\b', sql, re.IGNORECASE))
        if join_count > max_joins:
            result.add_warning(f"Trop de JOINs: {join_count} > {max_joins}")
        
        # Compter les sous-requêtes
        subquery_count = sql.count('(') - sql.count(')')
        if abs(subquery_count) > max_subqueries:
            result.add_warning(f"Trop de sous-requêtes: {abs(subquery_count)} > {max_subqueries}")


class BusinessRuleValidator(BaseValidator):
    """Validateur pour les règles métier"""
    
    def __init__(self, rules_config: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.rules_config = rules_config
    
    def validate_business_rule(self, entity: BaseModel, rule_name: str) -> ValidationResult:
        """Valide une règle métier spécifique"""
        result = ValidationResult()
        
        rule_config = self.rules_config.get(rule_name)
        if not rule_config:
            result.add_error(f"Règle métier inconnue: {rule_name}")
            return result
        
        # Exécuter les validations de la règle
        conditions = rule_config.get('conditions', [])
        for condition in conditions:
            condition_result = self._evaluate_condition(entity, condition)
            result.merge(condition_result)
        
        return result
    
    def _evaluate_condition(self, entity: BaseModel, condition: Dict[str, Any]) -> ValidationResult:
        """Évalue une condition métier"""
        result = ValidationResult()
        
        condition_type = condition.get('type')
        if condition_type == 'field_comparison':
            result = self._validate_field_comparison(entity, condition)
        elif condition_type == 'cross_field_validation':
            result = self._validate_cross_field(entity, condition)
        elif condition_type == 'custom_logic':
            result = self._validate_custom_logic(entity, condition)
        else:
            result.add_error(f"Type de condition inconnu: {condition_type}")
        
        return result
    
    def _validate_field_comparison(self, entity: BaseModel, condition: Dict[str, Any]) -> ValidationResult:
        """Valide une comparaison de champ"""
        result = ValidationResult()
        
        field_name = condition.get('field')
        operator = condition.get('operator')
        expected_value = condition.get('value')
        
        if not hasattr(entity, field_name):
            result.add_error(f"Champ introuvable: {field_name}")
            return result
        
        actual_value = getattr(entity, field_name)
        
        if operator == 'equals' and actual_value != expected_value:
            result.add_error(f"{field_name} doit être égal à {expected_value}")
        elif operator == 'not_equals' and actual_value == expected_value:
            result.add_error(f"{field_name} ne doit pas être égal à {expected_value}")
        elif operator == 'greater_than' and actual_value <= expected_value:
            result.add_error(f"{field_name} doit être supérieur à {expected_value}")
        elif operator == 'less_than' and actual_value >= expected_value:
            result.add_error(f"{field_name} doit être inférieur à {expected_value}")
        
        return result
    
    def _validate_cross_field(self, entity: BaseModel, condition: Dict[str, Any]) -> ValidationResult:
        """Valide une relation entre champs"""
        result = ValidationResult()
        
        field1 = condition.get('field1')
        field2 = condition.get('field2')
        relationship = condition.get('relationship')
        
        if not (hasattr(entity, field1) and hasattr(entity, field2)):
            result.add_error(f"Champs introuvables: {field1}, {field2}")
            return result
        
        value1 = getattr(entity, field1)
        value2 = getattr(entity, field2)
        
        if relationship == 'greater_than' and value1 <= value2:
            result.add_error(f"{field1} doit être supérieur à {field2}")
        elif relationship == 'less_than' and value1 >= value2:
            result.add_error(f"{field1} doit être inférieur à {field2}")
        elif relationship == 'equals' and value1 != value2:
            result.add_error(f"{field1} doit être égal à {field2}")
        
        return result
    
    def _validate_custom_logic(self, entity: BaseModel, condition: Dict[str, Any]) -> ValidationResult:
        """Exécute une logique métier personnalisée"""
        result = ValidationResult()
        
        logic_function = condition.get('function')
        if logic_function:
            try:
                # Exécution sécurisée de la fonction personnalisée
                custom_result = logic_function(entity)
                if isinstance(custom_result, ValidationResult):
                    result.merge(custom_result)
                elif not custom_result:
                    result.add_error("Règle métier personnalisée échouée")
            except Exception as e:
                result.add_error(f"Erreur dans la logique personnalisée: {str(e)}")
        
        return result


class CompositeValidator:
    """Validateur composite pour orchestrer plusieurs validateurs"""
    
    def __init__(self):
        self.validators: Dict[str, BaseValidator] = {}
        self.validation_order: List[str] = []
        self.dependencies: Dict[str, List[str]] = {}
    
    def add_validator(self, name: str, validator: BaseValidator, 
                     depends_on: Optional[List[str]] = None):
        """Ajoute un validateur avec ses dépendances"""
        self.validators[name] = validator
        if name not in self.validation_order:
            self.validation_order.append(name)
        
        if depends_on:
            self.dependencies[name] = depends_on
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Exécute tous les validateurs dans l'ordre correct"""
        result = ValidationResult()
        execution_order = self._calculate_execution_order()
        
        for validator_name in execution_order:
            validator = self.validators[validator_name]
            
            # Vérifier si les dépendances ont réussi
            if not self._dependencies_satisfied(validator_name, result):
                result.add_warning(f"Dépendances non satisfaites pour {validator_name}")
                continue
            
            validator_result = validator.validate(value)
            result.merge(validator_result)
            
            # Arrêter si erreur critique et mode strict
            if validator.strict_mode and not validator_result.is_valid:
                break
        
        return result
    
    def _calculate_execution_order(self) -> List[str]:
        """Calcule l'ordre d'exécution basé sur les dépendances"""
        # Tri topologique simple
        ordered = []
        visited = set()
        
        def visit(name):
            if name in visited:
                return
            visited.add(name)
            
            deps = self.dependencies.get(name, [])
            for dep in deps:
                if dep in self.validators:
                    visit(dep)
            
            ordered.append(name)
        
        for name in self.validation_order:
            visit(name)
        
        return ordered
    
    def _dependencies_satisfied(self, validator_name: str, result: ValidationResult) -> bool:
        """Vérifie si les dépendances d'un validateur sont satisfaites"""
        deps = self.dependencies.get(validator_name, [])
        
        for dep in deps:
            # Vérifier si la validation de la dépendance a réussi
            # (logique simplifiée, à améliorer selon les besoins)
            if dep not in result.metadata.get('completed_validators', set()):
                return False
        
        return True


__all__ = [
    'ValidationSeverity', 'ValidationScope', 'ValidationResult', 'BaseValidator',
    'StringValidator', 'NumberValidator', 'EmailValidator', 'URLValidator',
    'SQLValidator', 'BusinessRuleValidator', 'CompositeValidator'
]
