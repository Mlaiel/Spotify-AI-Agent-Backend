"""
Utilitaires Avancés pour Fixtures Slack
=======================================

Module contenant des utilitaires, validateurs, et helpers pour la gestion
avancée des fixtures d'alertes Slack dans l'environnement multi-tenant.

Fonctionnalités:
- Validation de schémas JSON complexes
- Formatage et transformation de données
- Utilitaires de chiffrement et sécurité
- Helpers pour templates Jinja2
- Métriques et observabilité
- Gestion des erreurs avancée

Auteur: Fahed Mlaiel - Lead Developer Achiri
Version: 2.5.0
"""

import re
import json
import base64
import hashlib
import secrets
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
import logging

from jsonschema import validate, ValidationError, Draft7Validator
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import prometheus_client
from opentelemetry import trace
import jinja2

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# Schémas JSON pour validation
SLACK_TEMPLATE_SCHEMA = {
    "type": "object",
    "required": ["channel"],
    "properties": {
        "channel": {
            "type": "string",
            "pattern": "^#[a-z0-9_-]+$",
            "description": "Canal Slack avec # au début"
        },
        "username": {
            "type": "string",
            "minLength": 1,
            "maxLength": 80,
            "description": "Nom d'utilisateur du bot"
        },
        "icon_emoji": {
            "type": "string",
            "pattern": "^:[a-z0-9_+-]+:$",
            "description": "Emoji d'icône au format :emoji:"
        },
        "icon_url": {
            "type": "string",
            "format": "uri",
            "description": "URL de l'icône du bot"
        },
        "text": {
            "type": "string",
            "description": "Texte principal du message"
        },
        "attachments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "color": {
                        "type": "string",
                        "pattern": "^(good|warning|danger|#[0-9a-fA-F]{6})$"
                    },
                    "title": {"type": "string"},
                    "text": {"type": "string"},
                    "fields": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["title", "value"],
                            "properties": {
                                "title": {"type": "string"},
                                "value": {"type": "string"},
                                "short": {"type": "boolean"}
                            }
                        }
                    },
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "text": {"type": "string"},
                                "url": {"type": "string", "format": "uri"}
                            }
                        }
                    },
                    "footer": {"type": "string"},
                    "footer_icon": {"type": "string", "format": "uri"},
                    "ts": {"type": ["string", "integer"]}
                }
            }
        },
        "blocks": {
            "type": "array",
            "description": "Slack Block Kit elements"
        }
    },
    "additionalProperties": False
}

FIXTURE_METADATA_SCHEMA = {
    "type": "object",
    "required": ["tenant_id", "environment", "locale", "alert_type"],
    "properties": {
        "tenant_id": {
            "type": "string",
            "pattern": "^[a-z0-9-]+$",
            "minLength": 3,
            "maxLength": 50
        },
        "environment": {
            "type": "string",
            "enum": ["dev", "staging", "prod"]
        },
        "locale": {
            "type": "string",
            "enum": ["fr", "en", "de", "es", "it"]
        },
        "alert_type": {
            "type": "string",
            "pattern": "^[a-z0-9_]+$",
            "minLength": 3,
            "maxLength": 50
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"}
        },
        "priority": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10
        },
        "expires_at": {
            "type": "string",
            "format": "date-time"
        }
    }
}

class SlackAlertValidator:
    """
    Validateur avancé pour les templates et configurations Slack.
    """
    
    def __init__(self):
        self.template_validator = Draft7Validator(SLACK_TEMPLATE_SCHEMA)
        self.metadata_validator = Draft7Validator(FIXTURE_METADATA_SCHEMA)
        
        # Métriques de validation
        self.validation_counter = prometheus_client.Counter(
            'slack_validations_total',
            'Total number of validations performed',
            ['type', 'result']
        )
        
    def validate_template(self, template: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Valide un template Slack.
        
        Args:
            template: Template à valider
            
        Returns:
            Tuple (is_valid, errors)
        """
        with tracer.start_as_current_span("validate_template"):
            errors = []
            
            try:
                # Validation du schéma JSON
                self.template_validator.validate(template)
                
                # Validations supplémentaires
                errors.extend(self._validate_template_content(template))
                errors.extend(self._validate_jinja2_syntax(template))
                
                is_valid = len(errors) == 0
                
                self.validation_counter.labels(
                    type='template',
                    result='success' if is_valid else 'error'
                ).inc()
                
                return is_valid, errors
                
            except ValidationError as e:
                errors.append(f"Erreur de schéma: {e.message}")
                self.validation_counter.labels(type='template', result='error').inc()
                return False, errors
            except Exception as e:
                errors.append(f"Erreur de validation: {str(e)}")
                self.validation_counter.labels(type='template', result='error').inc()
                return False, errors
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Valide les métadonnées d'une fixture.
        
        Args:
            metadata: Métadonnées à valider
            
        Returns:
            Tuple (is_valid, errors)
        """
        with tracer.start_as_current_span("validate_metadata"):
            errors = []
            
            try:
                self.metadata_validator.validate(metadata)
                
                # Validations métier supplémentaires
                errors.extend(self._validate_business_rules(metadata))
                
                is_valid = len(errors) == 0
                
                self.validation_counter.labels(
                    type='metadata',
                    result='success' if is_valid else 'error'
                ).inc()
                
                return is_valid, errors
                
            except ValidationError as e:
                errors.append(f"Erreur de schéma métadonnées: {e.message}")
                self.validation_counter.labels(type='metadata', result='error').inc()
                return False, errors
    
    def _validate_template_content(self, template: Dict[str, Any]) -> List[str]:
        """Valide le contenu spécifique du template."""
        errors = []
        
        # Validation du canal
        channel = template.get('channel', '')
        if not self._is_valid_slack_channel(channel):
            errors.append(f"Canal Slack invalide: {channel}")
        
        # Validation des URLs dans les actions
        if 'attachments' in template:
            for i, attachment in enumerate(template['attachments']):
                if 'actions' in attachment:
                    for j, action in enumerate(attachment['actions']):
                        if 'url' in action and not self._is_valid_url(action['url']):
                            errors.append(f"URL invalide dans attachment {i}, action {j}")
        
        return errors
    
    def _validate_jinja2_syntax(self, template: Dict[str, Any]) -> List[str]:
        """Valide la syntaxe Jinja2 dans le template."""
        errors = []
        
        try:
            env = jinja2.Environment()
            template_str = json.dumps(template)
            env.parse(template_str)
        except jinja2.TemplateSyntaxError as e:
            errors.append(f"Erreur de syntaxe Jinja2: {e}")
        except Exception as e:
            errors.append(f"Erreur de validation Jinja2: {e}")
        
        return errors
    
    def _validate_business_rules(self, metadata: Dict[str, Any]) -> List[str]:
        """Valide les règles métier spécifiques."""
        errors = []
        
        # Validation de l'expiration
        if 'expires_at' in metadata:
            try:
                expires_at = datetime.fromisoformat(metadata['expires_at'])
                if expires_at <= datetime.now(timezone.utc):
                    errors.append("La date d'expiration doit être dans le futur")
            except ValueError:
                errors.append("Format de date d'expiration invalide")
        
        # Validation des tags
        if 'tags' in metadata:
            for tag in metadata['tags']:
                if not re.match(r'^[a-z0-9_-]+$', tag):
                    errors.append(f"Tag invalide: {tag}")
        
        return errors
    
    def _is_valid_slack_channel(self, channel: str) -> bool:
        """Vérifie si un nom de canal Slack est valide."""
        return bool(re.match(r'^#[a-z0-9_-]+$', channel))
    
    def _is_valid_url(self, url: str) -> bool:
        """Vérifie si une URL est valide."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

class SecureTokenManager:
    """
    Gestionnaire sécurisé pour les tokens et données sensibles.
    """
    
    def __init__(self, master_key: Optional[bytes] = None):
        if master_key:
            self.cipher = Fernet(master_key)
        else:
            self.cipher = Fernet(Fernet.generate_key())
        
        self.token_cache: Dict[str, Dict[str, Any]] = {}
        
    def encrypt_token(self, token: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Chiffre un token avec métadonnées optionnelles.
        
        Args:
            token: Token à chiffrer
            metadata: Métadonnées optionnelles
            
        Returns:
            Token chiffré encodé en base64
        """
        with tracer.start_as_current_span("encrypt_token"):
            data = {
                'token': token,
                'metadata': metadata or {},
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            
            encrypted = self.cipher.encrypt(json.dumps(data).encode())
            return base64.b64encode(encrypted).decode()
    
    def decrypt_token(self, encrypted_token: str) -> Dict[str, Any]:
        """
        Déchiffre un token.
        
        Args:
            encrypted_token: Token chiffré
            
        Returns:
            Données déchiffrées
        """
        with tracer.start_as_current_span("decrypt_token"):
            try:
                encrypted_data = base64.b64decode(encrypted_token.encode())
                decrypted = self.cipher.decrypt(encrypted_data)
                return json.loads(decrypted.decode())
            except Exception as e:
                logger.error(f"Erreur de déchiffrement: {e}")
                raise ValueError("Token invalide ou corrompu")
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Génère un token sécurisé."""
        return secrets.token_urlsafe(length)
    
    def hash_data(self, data: str, salt: Optional[str] = None) -> str:
        """
        Hash des données avec un salt optionnel.
        
        Args:
            data: Données à hasher
            salt: Salt optionnel
            
        Returns:
            Hash hexadécimal
        """
        if salt is None:
            salt = secrets.token_hex(16)
        
        combined = f"{data}{salt}".encode()
        return hashlib.sha256(combined).hexdigest()

class FixtureCache:
    """
    Cache intelligent pour les fixtures avec TTL et invalidation.
    """
    
    def __init__(self, ttl: int = 3600, max_size: int = 1000):
        self.ttl = ttl
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, datetime] = {}
        
        # Métriques de cache
        self.cache_metrics = {
            'hits': prometheus_client.Counter(
                'fixture_cache_hits_total',
                'Total cache hits'
            ),
            'misses': prometheus_client.Counter(
                'fixture_cache_misses_total', 
                'Total cache misses'
            ),
            'evictions': prometheus_client.Counter(
                'fixture_cache_evictions_total',
                'Total cache evictions'
            )
        }
        
    async def initialize(self):
        """Initialise le cache et démarre les tâches de nettoyage."""
        asyncio.create_task(self._cleanup_task())
        logger.debug("Cache initialisé")
    
    async def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache."""
        with tracer.start_as_current_span("cache_get"):
            if key in self.cache:
                entry = self.cache[key]
                
                # Vérification TTL
                if datetime.now() - entry['created_at'] < timedelta(seconds=self.ttl):
                    self.access_times[key] = datetime.now()
                    self.cache_metrics['hits'].inc()
                    return entry['data']
                else:
                    # Expiration
                    del self.cache[key]
                    del self.access_times[key]
            
            self.cache_metrics['misses'].inc()
            return None
    
    async def set(self, key: str, value: Any) -> None:
        """Met une valeur en cache."""
        with tracer.start_as_current_span("cache_set"):
            # Éviction si nécessaire
            if len(self.cache) >= self.max_size:
                await self._evict_lru()
            
            self.cache[key] = {
                'data': value,
                'created_at': datetime.now()
            }
            self.access_times[key] = datetime.now()
    
    async def delete(self, key: str) -> None:
        """Supprime une entrée du cache."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
    
    async def clear(self) -> None:
        """Vide complètement le cache."""
        self.cache.clear()
        self.access_times.clear()
    
    async def _evict_lru(self) -> None:
        """Éviction LRU (Least Recently Used)."""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        await self.delete(lru_key)
        self.cache_metrics['evictions'].inc()
        logger.debug(f"Éviction LRU: {lru_key}")
    
    async def _cleanup_task(self) -> None:
        """Tâche de nettoyage périodique."""
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                
                now = datetime.now()
                expired_keys = [
                    key for key, entry in self.cache.items()
                    if now - entry['created_at'] > timedelta(seconds=self.ttl)
                ]
                
                for key in expired_keys:
                    await self.delete(key)
                
                if expired_keys:
                    logger.debug(f"Nettoyage cache: {len(expired_keys)} entrées expirées")
                    
            except Exception as e:
                logger.error(f"Erreur dans le nettoyage du cache: {e}")

class SlackMetricsCollector:
    """
    Collecteur de métriques avancées pour le monitoring Slack.
    """
    
    def __init__(self):
        self.metrics = {
            'alerts_sent': prometheus_client.Counter(
                'slack_alerts_sent_total',
                'Total Slack alerts sent',
                ['tenant_id', 'environment', 'locale', 'alert_type', 'severity']
            ),
            'alerts_failed': prometheus_client.Counter(
                'slack_alerts_failed_total',
                'Total failed Slack alerts',
                ['tenant_id', 'environment', 'error_type']
            ),
            'template_render_duration': prometheus_client.Histogram(
                'slack_template_render_duration_seconds',
                'Template rendering duration',
                ['template_type'],
                buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
            ),
            'fixture_load_duration': prometheus_client.Histogram(
                'slack_fixture_load_duration_seconds',
                'Fixture loading duration',
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
            ),
            'active_fixtures': prometheus_client.Gauge(
                'slack_active_fixtures_count',
                'Number of active fixtures',
                ['tenant_id', 'environment']
            ),
            'cache_size': prometheus_client.Gauge(
                'slack_cache_size_bytes',
                'Cache size in bytes'
            )
        }
        
    async def initialize(self):
        """Initialise le collecteur de métriques."""
        logger.debug("Collecteur de métriques initialisé")
    
    def record_alert_sent(self, tenant_id: str, environment: str, locale: str, 
                         alert_type: str, severity: str):
        """Enregistre une alerte envoyée."""
        self.metrics['alerts_sent'].labels(
            tenant_id=tenant_id,
            environment=environment,
            locale=locale,
            alert_type=alert_type,
            severity=severity
        ).inc()
    
    def record_alert_failed(self, tenant_id: str, environment: str, error_type: str):
        """Enregistre une alerte échouée."""
        self.metrics['alerts_failed'].labels(
            tenant_id=tenant_id,
            environment=environment,
            error_type=error_type
        ).inc()
    
    def record_template_render_duration(self, template_type: str, duration: float):
        """Enregistre la durée de rendu d'un template."""
        self.metrics['template_render_duration'].labels(
            template_type=template_type
        ).observe(duration)
    
    def record_fixture_load_duration(self, duration: float):
        """Enregistre la durée de chargement d'une fixture."""
        self.metrics['fixture_load_duration'].observe(duration)
    
    def update_active_fixtures_count(self, tenant_id: str, environment: str, count: int):
        """Met à jour le nombre de fixtures actives."""
        self.metrics['active_fixtures'].labels(
            tenant_id=tenant_id,
            environment=environment
        ).set(count)
    
    def update_cache_size(self, size_bytes: int):
        """Met à jour la taille du cache."""
        self.metrics['cache_size'].set(size_bytes)

# Fonctions utilitaires supplémentaires

def format_timestamp(timestamp: Union[int, float, datetime], 
                    locale: str = "fr") -> str:
    """
    Formate un timestamp selon la locale.
    
    Args:
        timestamp: Timestamp à formater
        locale: Locale pour le formatage
        
    Returns:
        Timestamp formaté
    """
    if isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    elif isinstance(timestamp, datetime):
        dt = timestamp
    else:
        raise ValueError("Type de timestamp non supporté")
    
    formats = {
        "fr": "%d/%m/%Y à %H:%M:%S",
        "en": "%Y-%m-%d at %H:%M:%S",
        "de": "%d.%m.%Y um %H:%M:%S",
        "es": "%d/%m/%Y a las %H:%M:%S",
        "it": "%d/%m/%Y alle %H:%M:%S"
    }
    
    return dt.strftime(formats.get(locale, formats["en"]))

def sanitize_string(text: str, max_length: int = 500) -> str:
    """
    Nettoie et limite une chaîne de caractères.
    
    Args:
        text: Texte à nettoyer
        max_length: Longueur maximale
        
    Returns:
        Texte nettoyé
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Suppression des caractères de contrôle
    cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Limitation de la longueur
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length-3] + "..."
    
    return cleaned

def extract_template_variables(template_str: str) -> List[str]:
    """
    Extrait les variables Jinja2 d'un template.
    
    Args:
        template_str: Chaîne de template
        
    Returns:
        Liste des variables trouvées
    """
    pattern = r'\{\{\s*([^}]+)\s*\}\}'
    matches = re.findall(pattern, template_str)
    
    variables = []
    for match in matches:
        # Nettoyage et extraction du nom de variable
        var_name = match.split('|')[0].strip()
        if '.' in var_name:
            var_name = var_name.split('.')[0]
        variables.append(var_name)
    
    return list(set(variables))

def validate_slack_webhook_url(url: str) -> bool:
    """
    Valide une URL de webhook Slack.
    
    Args:
        url: URL à valider
        
    Returns:
        True si valide, False sinon
    """
    slack_webhook_pattern = r'^https://hooks\.slack\.com/services/[A-Z0-9]+/[A-Z0-9]+/[a-zA-Z0-9]+$'
    return bool(re.match(slack_webhook_pattern, url))

# Export des classes et fonctions principales
__all__ = [
    "SlackAlertValidator",
    "SecureTokenManager", 
    "FixtureCache",
    "SlackMetricsCollector",
    "format_timestamp",
    "sanitize_string",
    "extract_template_variables",
    "validate_slack_webhook_url",
    "SLACK_TEMPLATE_SCHEMA",
    "FIXTURE_METADATA_SCHEMA"
]
