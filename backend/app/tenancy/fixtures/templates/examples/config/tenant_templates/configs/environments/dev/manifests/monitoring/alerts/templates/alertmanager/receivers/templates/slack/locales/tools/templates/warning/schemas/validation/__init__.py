"""
Validateurs personnalisés - Spotify AI Agent
Règles de validation avancées pour les schémas Pydantic
"""

import re
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Pattern, Set, Union
from uuid import UUID

from pydantic import validator, root_validator
from pydantic.fields import ModelField
from pydantic.types import EmailStr, HttpUrl

from ..base.enums import (
    AlertLevel, WarningCategory, NotificationChannel,
    VALIDATION_PATTERNS, SYSTEM_CONSTANTS
)


class ValidationRules:
    """Règles de validation centralisées"""
    
    # Patterns compilés pour performance
    TENANT_ID_PATTERN: Pattern = re.compile(VALIDATION_PATTERNS["TENANT_ID"])
    CORRELATION_ID_PATTERN: Pattern = re.compile(VALIDATION_PATTERNS["CORRELATION_ID"])
    EMAIL_PATTERN: Pattern = re.compile(VALIDATION_PATTERNS["EMAIL"])
    PHONE_PATTERN: Pattern = re.compile(VALIDATION_PATTERNS["PHONE"])
    URL_PATTERN: Pattern = re.compile(VALIDATION_PATTERNS["URL"])
    VERSION_PATTERN: Pattern = re.compile(VALIDATION_PATTERNS["VERSION"])
    HEX_COLOR_PATTERN: Pattern = re.compile(VALIDATION_PATTERNS["HEX_COLOR"])
    
    @classmethod
    def validate_tenant_id(cls, value: str) -> str:
        """Valide un ID de tenant"""
        if not value or not value.strip():
            raise ValueError("Tenant ID cannot be empty")
        
        value = value.strip().lower()
        
        if not cls.TENANT_ID_PATTERN.match(value):
            raise ValueError(
                "Tenant ID must contain only letters, numbers, hyphens and underscores"
            )
        
        if len(value) > 255:
            raise ValueError("Tenant ID cannot exceed 255 characters")
        
        # Vérification des mots réservés
        reserved_words = {"admin", "root", "system", "api", "test", "null", "undefined"}
        if value in reserved_words:
            raise ValueError(f"'{value}' is a reserved tenant ID")
        
        return value
    
    @classmethod
    def validate_alert_message(cls, value: str) -> str:
        """Valide un message d'alerte"""
        if not value or not value.strip():
            raise ValueError("Alert message cannot be empty")
        
        value = value.strip()
        
        if len(value) > SYSTEM_CONSTANTS["MAX_ALERT_MESSAGE_LENGTH"]:
            raise ValueError(
                f"Alert message cannot exceed {SYSTEM_CONSTANTS['MAX_ALERT_MESSAGE_LENGTH']} characters"
            )
        
        # Vérification de contenu malveillant basique
        suspicious_patterns = [
            r"<script[^>]*>",
            r"javascript:",
            r"data:text/html",
            r"vbscript:",
            r"on\w+\s*="
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                raise ValueError("Alert message contains potentially malicious content")
        
        return value
    
    @classmethod
    def validate_correlation_id(cls, value: Optional[str]) -> Optional[str]:
        """Valide un ID de corrélation"""
        if value is None:
            return None
        
        value = value.strip()
        if not value:
            return None
        
        if not cls.CORRELATION_ID_PATTERN.match(value):
            raise ValueError(
                "Correlation ID must contain only letters, numbers, hyphens and underscores"
            )
        
        return value
    
    @classmethod
    def validate_metadata_size(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        """Valide la taille des métadonnées"""
        if not value:
            return value
        
        # Calcul approximatif de la taille
        metadata_str = json.dumps(value, ensure_ascii=False)
        size_bytes = len(metadata_str.encode('utf-8'))
        
        if size_bytes > SYSTEM_CONSTANTS["MAX_METADATA_SIZE_BYTES"]:
            raise ValueError(
                f"Metadata size ({size_bytes} bytes) exceeds maximum "
                f"({SYSTEM_CONSTANTS['MAX_METADATA_SIZE_BYTES']} bytes)"
            )
        
        return value
    
    @classmethod
    def validate_severity_score(cls, value: Optional[float]) -> Optional[float]:
        """Valide un score de sévérité"""
        if value is None:
            return None
        
        if not 0.0 <= value <= 1.0:
            raise ValueError("Severity score must be between 0.0 and 1.0")
        
        # Arrondi à 3 décimales pour éviter les problèmes de précision
        return round(value, 3)
    
    @classmethod
    def validate_tags(cls, value: Dict[str, str]) -> Dict[str, str]:
        """Valide les tags"""
        if not value:
            return value
        
        validated_tags = {}
        
        for key, val in value.items():
            # Validation des clés
            if not key or not key.strip():
                raise ValueError("Tag key cannot be empty")
            
            key = key.strip().lower()
            
            if len(key) > 50:
                raise ValueError("Tag key cannot exceed 50 characters")
            
            if not re.match(r'^[a-z0-9_-]+$', key):
                raise ValueError(
                    "Tag key must contain only lowercase letters, numbers, hyphens and underscores"
                )
            
            # Validation des valeurs
            if not isinstance(val, str):
                val = str(val)
            
            val = val.strip()
            
            if len(val) > 200:
                raise ValueError("Tag value cannot exceed 200 characters")
            
            validated_tags[key] = val
        
        # Limite du nombre de tags
        if len(validated_tags) > 20:
            raise ValueError("Cannot have more than 20 tags")
        
        return validated_tags
    
    @classmethod
    def validate_recipients_list(cls, value: List[str], channel: NotificationChannel) -> List[str]:
        """Valide une liste de destinataires selon le canal"""
        if not value:
            raise ValueError("Recipients list cannot be empty")
        
        validated_recipients = []
        
        for recipient in value:
            if not recipient or not recipient.strip():
                continue
            
            recipient = recipient.strip()
            
            # Validation selon le type de canal
            if channel == NotificationChannel.EMAIL:
                if not cls.EMAIL_PATTERN.match(recipient):
                    raise ValueError(f"Invalid email address: {recipient}")
            
            elif channel == NotificationChannel.SLACK:
                # Format Slack: @username, #channel, ou user ID
                if not re.match(r'^[@#]?[a-zA-Z0-9._-]+$', recipient):
                    raise ValueError(f"Invalid Slack recipient format: {recipient}")
            
            elif channel == NotificationChannel.SMS:
                if not cls.PHONE_PATTERN.match(recipient):
                    raise ValueError(f"Invalid phone number: {recipient}")
            
            validated_recipients.append(recipient)
        
        # Limite du nombre de destinataires
        max_recipients = 100
        if len(validated_recipients) > max_recipients:
            raise ValueError(f"Cannot have more than {max_recipients} recipients")
        
        return validated_recipients
    
    @classmethod
    def validate_webhook_url(cls, value: HttpUrl) -> HttpUrl:
        """Valide une URL de webhook"""
        url_str = str(value)
        
        # Vérification du protocole
        if not url_str.startswith(('http://', 'https://')):
            raise ValueError("Webhook URL must use HTTP or HTTPS protocol")
        
        # Vérification des domaines interdits
        forbidden_domains = {
            'localhost', '127.0.0.1', '0.0.0.0',
            '192.168.', '10.', '172.16.', '172.17.',
            '172.18.', '172.19.', '172.20.', '172.21.',
            '172.22.', '172.23.', '172.24.', '172.25.',
            '172.26.', '172.27.', '172.28.', '172.29.',
            '172.30.', '172.31.'
        }
        
        for forbidden in forbidden_domains:
            if forbidden in url_str.lower():
                raise ValueError(f"Webhook URL cannot use forbidden domain: {forbidden}")
        
        return value
    
    @classmethod
    def validate_time_range(cls, start_time: Optional[datetime], 
                          end_time: Optional[datetime]) -> tuple:
        """Valide une plage temporelle"""
        if start_time is None and end_time is None:
            return start_time, end_time
        
        if start_time and end_time:
            if start_time >= end_time:
                raise ValueError("Start time must be before end time")
            
            # Vérification de la plage maximale (1 an)
            max_range_days = 365
            if (end_time - start_time).days > max_range_days:
                raise ValueError(f"Time range cannot exceed {max_range_days} days")
        
        return start_time, end_time
    
    @classmethod
    def validate_json_template(cls, value: str) -> str:
        """Valide un template JSON"""
        if not value or not value.strip():
            raise ValueError("JSON template cannot be empty")
        
        value = value.strip()
        
        # Vérification de la syntaxe JSON de base
        try:
            # Test avec des variables fictives
            test_template = value.replace('{{', '"test_').replace('}}', '_test"')
            json.loads(test_template)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON template syntax: {e}")
        
        # Vérification des variables Jinja2
        jinja_pattern = r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\}\}'
        variables = re.findall(jinja_pattern, value)
        
        # Validation des noms de variables
        for var in variables:
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_.]*$', var):
                raise ValueError(f"Invalid template variable name: {var}")
        
        return value
    
    @classmethod
    def validate_cron_expression(cls, value: str) -> str:
        """Valide une expression cron"""
        if not value or not value.strip():
            raise ValueError("Cron expression cannot be empty")
        
        value = value.strip()
        
        # Validation basique du format cron (5 ou 6 champs)
        fields = value.split()
        if len(fields) not in [5, 6]:
            raise ValueError("Cron expression must have 5 or 6 fields")
        
        # Validation de chaque champ
        field_ranges = [
            (0, 59),    # minutes
            (0, 23),    # hours
            (1, 31),    # day of month
            (1, 12),    # month
            (0, 6),     # day of week
        ]
        
        if len(fields) == 6:
            field_ranges.insert(0, (0, 59))  # seconds
        
        for i, (field, (min_val, max_val)) in enumerate(zip(fields, field_ranges)):
            if field == '*':
                continue
            
            # Gestion des listes (1,2,3)
            if ',' in field:
                values = field.split(',')
                for val in values:
                    if not val.isdigit() or not min_val <= int(val) <= max_val:
                        raise ValueError(f"Invalid cron field value: {val}")
                continue
            
            # Gestion des plages (1-5)
            if '-' in field:
                try:
                    start, end = map(int, field.split('-'))
                    if not (min_val <= start <= max_val and min_val <= end <= max_val):
                        raise ValueError(f"Invalid cron range: {field}")
                except ValueError:
                    raise ValueError(f"Invalid cron range format: {field}")
                continue
            
            # Gestion des pas (*/5)
            if '/' in field:
                base, step = field.split('/')
                if base != '*' and not base.isdigit():
                    raise ValueError(f"Invalid cron step base: {base}")
                if not step.isdigit() or int(step) <= 0:
                    raise ValueError(f"Invalid cron step value: {step}")
                continue
            
            # Valeur simple
            if not field.isdigit() or not min_val <= int(field) <= max_val:
                raise ValueError(f"Invalid cron field value: {field}")
        
        return value


# Décorateurs de validation pour faciliter l'usage
def validate_tenant_id_field():
    """Décorateur pour valider un champ tenant_id"""
    return validator('tenant_id', allow_reuse=True)(ValidationRules.validate_tenant_id)

def validate_alert_message_field():
    """Décorateur pour valider un champ message d'alerte"""
    return validator('message', allow_reuse=True)(ValidationRules.validate_alert_message)

def validate_metadata_field():
    """Décorateur pour valider un champ metadata"""
    return validator('metadata', allow_reuse=True)(ValidationRules.validate_metadata_size)

def validate_tags_field():
    """Décorateur pour valider un champ tags"""
    return validator('tags', allow_reuse=True)(ValidationRules.validate_tags)

def validate_severity_score_field():
    """Décorateur pour valider un champ severity_score"""
    return validator('severity_score', allow_reuse=True)(ValidationRules.validate_severity_score)


# Validateurs root pour validation inter-champs
def validate_time_range_fields(*field_names):
    """Décorateur pour valider une plage temporelle"""
    def decorator(func):
        @root_validator(allow_reuse=True)
        def validator_func(cls, values):
            if len(field_names) >= 2:
                start_field, end_field = field_names[0], field_names[1]
                start_time = values.get(start_field)
                end_time = values.get(end_field)
                
                validated_start, validated_end = ValidationRules.validate_time_range(
                    start_time, end_time
                )
                
                values[start_field] = validated_start
                values[end_field] = validated_end
            
            return values
        return validator_func
    return decorator


class DataSanitizer:
    """Utilitaires de sanitisation des données"""
    
    @staticmethod
    def sanitize_html(text: str) -> str:
        """Nettoie le HTML basique"""
        if not text:
            return text
        
        # Suppression des tags HTML dangereux
        dangerous_tags = [
            'script', 'iframe', 'object', 'embed', 'form',
            'input', 'button', 'textarea', 'select', 'option'
        ]
        
        for tag in dangerous_tags:
            pattern = rf'<{tag}[^>]*>.*?</{tag}>'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
            pattern = rf'<{tag}[^>]*/?>'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Suppression des attributs JavaScript
        text = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    @staticmethod
    def sanitize_sql(text: str) -> str:
        """Protection basique contre l'injection SQL"""
        if not text:
            return text
        
        # Mots-clés SQL dangereux
        sql_keywords = [
            'drop', 'delete', 'truncate', 'insert', 'update',
            'exec', 'execute', 'sp_', 'xp_', 'cmd', 'shell'
        ]
        
        for keyword in sql_keywords:
            pattern = rf'\b{keyword}\b'
            if re.search(pattern, text, re.IGNORECASE):
                raise ValueError(f"Potentially dangerous SQL keyword detected: {keyword}")
        
        return text
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalise les espaces blancs"""
        if not text:
            return text
        
        # Remplacement des caractères de contrôle
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalisation des espaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
        """Tronque un texte à la longueur maximale"""
        if not text or len(text) <= max_length:
            return text
        
        if len(suffix) >= max_length:
            return text[:max_length]
        
        return text[:max_length - len(suffix)] + suffix
