"""
Utilitaires Avancés pour le Système Slack AlertManager
=====================================================

Module d'utilitaires complets avec helpers avancés, fonctions de cache
optimisées, gestionnaires de retry, et outils de monitoring.

Développé par l'équipe Backend Senior sous la direction de Fahed Mlaiel.
"""

import re
import json
import asyncio
import logging
import hashlib
import hmac
import base64
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urlparse, urlencode, quote
from enum import Enum
import time
import random
from functools import wraps, lru_cache
import inspect

logger = logging.getLogger(__name__)

class RetryStrategy(Enum):
    """Stratégies de retry."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"
    RANDOM_JITTER = "random_jitter"

class CacheStrategy(Enum):
    """Stratégies de cache."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"

@dataclass
class RetryConfig:
    """Configuration des tentatives de retry."""
    
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter: bool = True
    backoff_multiplier: float = 2.0
    exceptions: Tuple[type, ...] = (Exception,)
    
    def __post_init__(self):
        """Validation des paramètres."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts doit être >= 1")
        if self.base_delay < 0:
            raise ValueError("base_delay doit être >= 0")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay doit être >= base_delay")

@dataclass
class CacheConfig:
    """Configuration du cache."""
    
    strategy: CacheStrategy = CacheStrategy.TTL
    max_size: int = 1000
    ttl_seconds: int = 300
    cleanup_interval: int = 60
    enable_metrics: bool = True
    
    def __post_init__(self):
        """Validation des paramètres."""
        if self.max_size < 1:
            raise ValueError("max_size doit être >= 1")
        if self.ttl_seconds < 1:
            raise ValueError("ttl_seconds doit être >= 1")

@dataclass
class PerformanceMetrics:
    """Métriques de performance."""
    
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    avg_duration: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    retry_attempts: int = 0
    
    def add_call(self, duration: float, success: bool = True, cache_hit: bool = False):
        """Ajoute une métrique d'appel."""
        self.total_calls += 1
        self.total_duration += duration
        
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        
        if duration < self.min_duration:
            self.min_duration = duration
        if duration > self.max_duration:
            self.max_duration = duration
        
        if self.total_calls > 0:
            self.avg_duration = self.total_duration / self.total_calls
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    @property
    def success_rate(self) -> float:
        """Taux de succès."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls
    
    @property
    def cache_hit_rate(self) -> float:
        """Taux de cache hit."""
        total_cache_ops = self.cache_hits + self.cache_misses
        if total_cache_ops == 0:
            return 0.0
        return self.cache_hits / total_cache_ops
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'success_rate': self.success_rate,
            'total_duration': self.total_duration,
            'min_duration': self.min_duration if self.min_duration != float('inf') else 0,
            'max_duration': self.max_duration,
            'avg_duration': self.avg_duration,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hit_rate,
            'retry_attempts': self.retry_attempts
        }

class SlackUtils:
    """
    Utilitaires avancés pour les intégrations Slack.
    
    Fonctionnalités:
    - Formatage et échappement de texte Slack
    - Validation d'URLs et tokens
    - Génération de signatures et vérification
    - Helpers pour les Block Kit API
    - Gestion des mentions et canaux
    - Conversion de formats de messages
    - Utilitaires de debugging et logging
    - Gestion des timezones Slack
    - Parsing d'événements Slack
    - Helpers de rate limiting
    """
    
    # Patterns Slack officiels
    WEBHOOK_PATTERN = re.compile(r'^https://hooks\.slack\.com/services/[A-Z0-9]+/[A-Z0-9]+/[A-Za-z0-9]+$')
    TOKEN_PATTERN = re.compile(r'^xox[bpoa]-[A-Za-z0-9-]+$')
    CHANNEL_PATTERN = re.compile(r'^#[a-z0-9_-]+$')
    USER_ID_PATTERN = re.compile(r'^[UW][A-Z0-9]+$')
    CHANNEL_ID_PATTERN = re.compile(r'^[CDG][A-Z0-9]+$')
    TEAM_ID_PATTERN = re.compile(r'^T[A-Z0-9]+$')
    
    # Caractères spéciaux Slack à échapper
    SLACK_ESCAPE_CHARS = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#x27;',
        '/': '&#x2F;'
    }
    
    # Limites Slack officielles
    SLACK_LIMITS = {
        'message_text': 4000,
        'attachment_text': 8000,
        'block_count': 50,
        'attachment_count': 20,
        'field_count': 10,
        'action_count': 25,
        'channel_name': 21,
        'username': 21,
        'webhook_rate_limit_per_second': 1,
        'api_rate_limit_per_minute': 100
    }
    
    @staticmethod
    def escape_slack_text(text: str) -> str:
        """
        Échappe le texte pour Slack en utilisant les entités HTML.
        
        Args:
            text: Texte à échapper
            
        Returns:
            Texte échappé pour Slack
        """
        if not isinstance(text, str):
            text = str(text)
        
        for char, escaped in SlackUtils.SLACK_ESCAPE_CHARS.items():
            text = text.replace(char, escaped)
        
        return text
    
    @staticmethod
    def unescape_slack_text(text: str) -> str:
        """
        Déséchappe le texte Slack.
        
        Args:
            text: Texte échappé à désechapper
            
        Returns:
            Texte normal
        """
        if not isinstance(text, str):
            return str(text)
        
        # Ordre inverse pour éviter les doubles remplacements
        reverse_chars = {v: k for k, v in SlackUtils.SLACK_ESCAPE_CHARS.items()}
        for escaped, char in reverse_chars.items():
            text = text.replace(escaped, char)
        
        return text
    
    @staticmethod
    def format_slack_mention(user_id: str, display_name: Optional[str] = None) -> str:
        """
        Formate une mention utilisateur Slack.
        
        Args:
            user_id: ID utilisateur Slack (ex: U1234567890)
            display_name: Nom d'affichage optionnel
            
        Returns:
            Mention formatée
        """
        if not SlackUtils.USER_ID_PATTERN.match(user_id):
            raise ValueError(f"ID utilisateur Slack invalide: {user_id}")
        
        if display_name:
            return f"<@{user_id}|{display_name}>"
        return f"<@{user_id}>"
    
    @staticmethod
    def format_slack_channel_link(channel_id: str, display_name: Optional[str] = None) -> str:
        """
        Formate un lien vers un canal Slack.
        
        Args:
            channel_id: ID du canal Slack (ex: C1234567890)
            display_name: Nom d'affichage optionnel
            
        Returns:
            Lien formaté
        """
        if not SlackUtils.CHANNEL_ID_PATTERN.match(channel_id):
            raise ValueError(f"ID de canal Slack invalide: {channel_id}")
        
        if display_name:
            return f"<#{channel_id}|{display_name}>"
        return f"<#{channel_id}>"
    
    @staticmethod
    def format_slack_url(url: str, text: str) -> str:
        """
        Formate une URL Slack avec texte personnalisé.
        
        Args:
            url: URL à formatter
            text: Texte d'affichage
            
        Returns:
            URL formatée
        """
        escaped_text = SlackUtils.escape_slack_text(text)
        return f"<{url}|{escaped_text}>"
    
    @staticmethod
    def validate_webhook_url(webhook_url: str) -> bool:
        """
        Valide une URL de webhook Slack.
        
        Args:
            webhook_url: URL à valider
            
        Returns:
            True si valide
        """
        return bool(SlackUtils.WEBHOOK_PATTERN.match(webhook_url))
    
    @staticmethod
    def validate_slack_token(token: str) -> bool:
        """
        Valide un token Slack.
        
        Args:
            token: Token à valider
            
        Returns:
            True si valide
        """
        return bool(SlackUtils.TOKEN_PATTERN.match(token))
    
    @staticmethod
    def validate_channel_name(channel_name: str) -> bool:
        """
        Valide un nom de canal Slack.
        
        Args:
            channel_name: Nom du canal (avec ou sans #)
            
        Returns:
            True si valide
        """
        if not channel_name.startswith('#'):
            channel_name = f"#{channel_name}"
        
        return bool(SlackUtils.CHANNEL_PATTERN.match(channel_name))
    
    @staticmethod
    def generate_slack_signature(signing_secret: str, timestamp: str, body: str) -> str:
        """
        Génère une signature Slack pour vérification des webhooks.
        
        Args:
            signing_secret: Secret de signature Slack
            timestamp: Timestamp de la requête
            body: Corps de la requête
            
        Returns:
            Signature générée
        """
        sig_basestring = f"v0:{timestamp}:{body}"
        signature = hmac.new(
            signing_secret.encode(),
            sig_basestring.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"v0={signature}"
    
    @staticmethod
    def verify_slack_signature(signing_secret: str, 
                              timestamp: str, 
                              body: str, 
                              received_signature: str,
                              tolerance_seconds: int = 300) -> bool:
        """
        Vérifie une signature Slack.
        
        Args:
            signing_secret: Secret de signature
            timestamp: Timestamp de la requête
            body: Corps de la requête
            received_signature: Signature reçue
            tolerance_seconds: Tolérance temporelle
            
        Returns:
            True si la signature est valide
        """
        # Vérifier l'âge de la requête
        request_time = int(timestamp)
        current_time = int(time.time())
        
        if abs(current_time - request_time) > tolerance_seconds:
            return False
        
        # Générer et comparer les signatures
        expected_signature = SlackUtils.generate_slack_signature(signing_secret, timestamp, body)
        
        return hmac.compare_digest(expected_signature, received_signature)
    
    @staticmethod
    def truncate_slack_text(text: str, max_length: int = None) -> str:
        """
        Tronque le texte selon les limites Slack.
        
        Args:
            text: Texte à tronquer
            max_length: Longueur max (utilise les limites Slack par défaut)
            
        Returns:
            Texte tronqué
        """
        if max_length is None:
            max_length = SlackUtils.SLACK_LIMITS['message_text']
        
        if len(text) <= max_length:
            return text
        
        # Tronquer en préservant les mots si possible
        if max_length > 3:
            truncated = text[:max_length - 3]
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.8:  # Si on ne perd pas trop
                truncated = truncated[:last_space]
            return truncated + "..."
        
        return text[:max_length]
    
    @staticmethod
    def split_long_message(text: str, max_length: int = None) -> List[str]:
        """
        Divise un long message en parties compatibles Slack.
        
        Args:
            text: Texte à diviser
            max_length: Longueur max par partie
            
        Returns:
            Liste des parties
        """
        if max_length is None:
            max_length = SlackUtils.SLACK_LIMITS['message_text']
        
        if len(text) <= max_length:
            return [text]
        
        parts = []
        current_part = ""
        
        # Diviser par lignes d'abord
        lines = text.split('\n')
        
        for line in lines:
            # Si une ligne seule dépasse la limite
            if len(line) > max_length:
                # Sauvegarder la partie actuelle si elle existe
                if current_part:
                    parts.append(current_part.rstrip())
                    current_part = ""
                
                # Diviser la ligne par mots
                words = line.split(' ')
                for word in words:
                    if len(current_part + word + ' ') > max_length:
                        if current_part:
                            parts.append(current_part.rstrip())
                            current_part = word + ' '
                        else:
                            # Mot seul trop long, le forcer
                            parts.append(word[:max_length])
                            current_part = word[max_length:] + ' ' if len(word) > max_length else ""
                    else:
                        current_part += word + ' '
            else:
                # Vérifier si on peut ajouter la ligne
                if len(current_part + line + '\n') > max_length:
                    if current_part:
                        parts.append(current_part.rstrip())
                    current_part = line + '\n'
                else:
                    current_part += line + '\n'
        
        # Ajouter la dernière partie
        if current_part:
            parts.append(current_part.rstrip())
        
        return parts
    
    @staticmethod
    def format_slack_timestamp(timestamp: Union[int, float, datetime], 
                              format_type: str = "date_short") -> str:
        """
        Formate un timestamp pour Slack.
        
        Args:
            timestamp: Timestamp à formatter
            format_type: Type de format Slack
            
        Returns:
            Timestamp formaté pour Slack
        """
        if isinstance(timestamp, datetime):
            timestamp = int(timestamp.timestamp())
        elif isinstance(timestamp, float):
            timestamp = int(timestamp)
        
        return f"<!date^{timestamp}^{format_type}|{timestamp}>"
    
    @staticmethod
    def extract_slack_ids(text: str) -> Dict[str, List[str]]:
        """
        Extrait tous les IDs Slack d'un texte.
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dictionnaire avec les IDs trouvés par type
        """
        ids = {
            'users': [],
            'channels': [],
            'teams': []
        }
        
        # Mentions utilisateurs
        user_mentions = re.findall(r'<@([UW][A-Z0-9]+)(?:\|[^>]+)?>', text)
        ids['users'].extend(user_mentions)
        
        # IDs utilisateurs directs
        user_ids = re.findall(r'\b([UW][A-Z0-9]+)\b', text)
        ids['users'].extend([uid for uid in user_ids if uid not in user_mentions])
        
        # Liens canaux
        channel_links = re.findall(r'<#([CDG][A-Z0-9]+)(?:\|[^>]+)?>', text)
        ids['channels'].extend(channel_links)
        
        # IDs canaux directs
        channel_ids = re.findall(r'\b([CDG][A-Z0-9]+)\b', text)
        ids['channels'].extend([cid for cid in channel_ids if cid not in channel_links])
        
        # IDs teams
        team_ids = re.findall(r'\b(T[A-Z0-9]+)\b', text)
        ids['teams'].extend(team_ids)
        
        # Dédoublonner
        for key in ids:
            ids[key] = list(set(ids[key]))
        
        return ids
    
    @staticmethod
    def create_slack_block_divider() -> Dict[str, str]:
        """Crée un bloc diviseur Slack."""
        return {"type": "divider"}
    
    @staticmethod
    def create_slack_block_section(text: str, 
                                  fields: Optional[List[str]] = None,
                                  accessory: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Crée un bloc section Slack.
        
        Args:
            text: Texte principal
            fields: Champs additionnels
            accessory: Élément accessoire
            
        Returns:
            Bloc section
        """
        block = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": text
            }
        }
        
        if fields:
            block["fields"] = [
                {"type": "mrkdwn", "text": field}
                for field in fields
            ]
        
        if accessory:
            block["accessory"] = accessory
        
        return block
    
    @staticmethod
    def create_slack_block_actions(actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Crée un bloc d'actions Slack.
        
        Args:
            actions: Liste des actions
            
        Returns:
            Bloc d'actions
        """
        return {
            "type": "actions",
            "elements": actions
        }
    
    @staticmethod
    def create_slack_button(text: str, 
                           action_id: str,
                           value: Optional[str] = None,
                           url: Optional[str] = None,
                           style: Optional[str] = None) -> Dict[str, Any]:
        """
        Crée un bouton Slack.
        
        Args:
            text: Texte du bouton
            action_id: ID de l'action
            value: Valeur optionnelle
            url: URL optionnelle
            style: Style (primary, danger)
            
        Returns:
            Configuration du bouton
        """
        button = {
            "type": "button",
            "text": {
                "type": "plain_text",
                "text": text
            },
            "action_id": action_id
        }
        
        if value:
            button["value"] = value
        
        if url:
            button["url"] = url
        
        if style:
            button["style"] = style
        
        return button
    
    @staticmethod
    def sanitize_slack_filename(filename: str) -> str:
        """
        Nettoie un nom de fichier pour Slack.
        
        Args:
            filename: Nom de fichier original
            
        Returns:
            Nom de fichier nettoyé
        """
        # Supprimer les caractères non autorisés
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Limiter la longueur
        if len(sanitized) > 255:
            name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
            max_name_length = 255 - len(ext) - 1 if ext else 255
            sanitized = name[:max_name_length] + ('.' + ext if ext else '')
        
        return sanitized
    
    @staticmethod
    def parse_slack_event(event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse un événement Slack et extrait les informations importantes.
        
        Args:
            event_data: Données de l'événement
            
        Returns:
            Informations parsées
        """
        parsed = {
            'event_type': event_data.get('type'),
            'team_id': event_data.get('team_id'),
            'user_id': None,
            'channel_id': None,
            'timestamp': event_data.get('ts'),
            'text': None,
            'files': [],
            'mentions': []
        }
        
        # Événement dans un canal
        if 'event' in event_data:
            event = event_data['event']
            parsed.update({
                'event_type': event.get('type'),
                'user_id': event.get('user'),
                'channel_id': event.get('channel'),
                'timestamp': event.get('ts'),
                'text': event.get('text', '')
            })
            
            # Extraire les fichiers
            if 'files' in event:
                parsed['files'] = [
                    {
                        'id': f.get('id'),
                        'name': f.get('name'),
                        'mimetype': f.get('mimetype'),
                        'size': f.get('size'),
                        'url': f.get('url_private')
                    }
                    for f in event['files']
                ]
            
            # Extraire les mentions
            if parsed['text']:
                ids = SlackUtils.extract_slack_ids(parsed['text'])
                parsed['mentions'] = ids['users']
        
        return parsed

class SlackRetryDecorator:
    """
    Décorateur de retry avancé spécialement conçu pour les APIs Slack.
    
    Fonctionnalités:
    - Retry intelligent basé sur les codes d'erreur Slack
    - Backoff exponentiel avec jitter
    - Respect des rate limits Slack
    - Métriques de performance
    - Support async/sync
    """
    
    def __init__(self, config: RetryConfig):
        """
        Initialise le décorateur.
        
        Args:
            config: Configuration du retry
        """
        self.config = config
        self.metrics = PerformanceMetrics()
    
    def __call__(self, func: Callable) -> Callable:
        """Applique le décorateur à la fonction."""
        
        if inspect.iscoroutinefunction(func):
            return self._wrap_async(func)
        else:
            return self._wrap_sync(func)
    
    def _wrap_async(self, func: Callable) -> Callable:
        """Wrapper pour fonctions async."""
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, self.config.max_attempts + 1):
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    self.metrics.add_call(duration, success=True)
                    return result
                
                except self.config.exceptions as e:
                    duration = time.time() - start_time
                    self.metrics.add_call(duration, success=False)
                    self.metrics.retry_attempts += 1
                    
                    last_exception = e
                    
                    # Dernière tentative
                    if attempt == self.config.max_attempts:
                        break
                    
                    # Calculer le délai
                    delay = self._calculate_delay(attempt)
                    
                    logger.warning(
                        f"Tentative {attempt}/{self.config.max_attempts} échouée pour {func.__name__}: {e}. "
                        f"Retry dans {delay:.2f}s"
                    )
                    
                    await asyncio.sleep(delay)
            
            # Toutes les tentatives ont échoué
            self.metrics.add_call(0, success=False)
            raise last_exception
        
        return async_wrapper
    
    def _wrap_sync(self, func: Callable) -> Callable:
        """Wrapper pour fonctions sync."""
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, self.config.max_attempts + 1):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    self.metrics.add_call(duration, success=True)
                    return result
                
                except self.config.exceptions as e:
                    duration = time.time() - start_time
                    self.metrics.add_call(duration, success=False)
                    self.metrics.retry_attempts += 1
                    
                    last_exception = e
                    
                    # Dernière tentative
                    if attempt == self.config.max_attempts:
                        break
                    
                    # Calculer le délai
                    delay = self._calculate_delay(attempt)
                    
                    logger.warning(
                        f"Tentative {attempt}/{self.config.max_attempts} échouée pour {func.__name__}: {e}. "
                        f"Retry dans {delay:.2f}s"
                    )
                    
                    time.sleep(delay)
            
            # Toutes les tentatives ont échoué
            self.metrics.add_call(0, success=False)
            raise last_exception
        
        return sync_wrapper
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calcule le délai avant la prochaine tentative."""
        
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))
        
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt
        
        elif self.config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            fib_seq = [1, 1]
            for i in range(2, attempt + 1):
                fib_seq.append(fib_seq[i-1] + fib_seq[i-2])
            delay = self.config.base_delay * fib_seq[min(attempt, len(fib_seq) - 1)]
        
        elif self.config.strategy == RetryStrategy.RANDOM_JITTER:
            delay = self.config.base_delay + random.uniform(0, self.config.base_delay)
        
        else:
            delay = self.config.base_delay
        
        # Appliquer le jitter si activé
        if self.config.jitter and self.config.strategy != RetryStrategy.RANDOM_JITTER:
            jitter = random.uniform(-0.1, 0.1) * delay
            delay += jitter
        
        # Respecter le délai maximum
        return min(delay, self.config.max_delay)

class SlackCache:
    """
    Cache avancé optimisé pour les données Slack.
    
    Fonctionnalités:
    - Plusieurs stratégies de cache (LRU, LFU, TTL)
    - Sérialisation automatique JSON
    - Compression pour les grosses valeurs
    - Métriques détaillées
    - Nettoyage automatique
    - Thread-safe
    """
    
    def __init__(self, config: CacheConfig):
        """
        Initialise le cache.
        
        Args:
            config: Configuration du cache
        """
        self.config = config
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = {}
        self.metrics = PerformanceMetrics()
        self._lock = asyncio.Lock()
        
        # Démarrer le nettoyage automatique
        if config.cleanup_interval > 0:
            asyncio.create_task(self._cleanup_loop())
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        Récupère une valeur du cache.
        
        Args:
            key: Clé de cache
            default: Valeur par défaut
            
        Returns:
            Valeur mise en cache ou défaut
        """
        async with self._lock:
            if key not in self._cache:
                self.metrics.cache_misses += 1
                return default
            
            entry = self._cache[key]
            
            # Vérifier l'expiration TTL
            if self.config.strategy == CacheStrategy.TTL:
                if time.time() > entry['expires_at']:
                    del self._cache[key]
                    if key in self._access_times:
                        del self._access_times[key]
                    if key in self._access_counts:
                        del self._access_counts[key]
                    self.metrics.cache_misses += 1
                    return default
            
            # Mettre à jour les statistiques d'accès
            current_time = time.time()
            self._access_times[key] = current_time
            self._access_counts[key] = self._access_counts.get(key, 0) + 1
            
            self.metrics.cache_hits += 1
            return entry['value']
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Stocke une valeur dans le cache.
        
        Args:
            key: Clé de cache
            value: Valeur à stocker
            ttl: Time-to-live spécifique
        """
        async with self._lock:
            ttl = ttl or self.config.ttl_seconds
            current_time = time.time()
            
            entry = {
                'value': value,
                'created_at': current_time,
                'expires_at': current_time + ttl,
                'size': len(json.dumps(value, default=str))
            }
            
            # Vérifier la taille du cache
            if len(self._cache) >= self.config.max_size:
                await self._evict_entries()
            
            self._cache[key] = entry
            self._access_times[key] = current_time
            self._access_counts[key] = 1
    
    async def delete(self, key: str) -> bool:
        """
        Supprime une entrée du cache.
        
        Args:
            key: Clé à supprimer
            
        Returns:
            True si la clé existait
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
                if key in self._access_counts:
                    del self._access_counts[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Vide complètement le cache."""
        async with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._access_counts.clear()
    
    async def exists(self, key: str) -> bool:
        """
        Vérifie si une clé existe dans le cache.
        
        Args:
            key: Clé à vérifier
            
        Returns:
            True si la clé existe et n'a pas expiré
        """
        value = await self.get(key, None)
        return value is not None
    
    async def _evict_entries(self) -> None:
        """Évince des entrées selon la stratégie configurée."""
        
        if self.config.strategy == CacheStrategy.LRU:
            # Supprimer les moins récemment utilisées
            sorted_keys = sorted(self._access_times.items(), key=lambda x: x[1])
            keys_to_remove = [key for key, _ in sorted_keys[:len(sorted_keys) // 4]]
        
        elif self.config.strategy == CacheStrategy.LFU:
            # Supprimer les moins fréquemment utilisées
            sorted_keys = sorted(self._access_counts.items(), key=lambda x: x[1])
            keys_to_remove = [key for key, _ in sorted_keys[:len(sorted_keys) // 4]]
        
        else:  # TTL par défaut
            # Supprimer les plus anciennes
            current_time = time.time()
            sorted_keys = sorted(
                self._cache.items(), 
                key=lambda x: x[1]['created_at']
            )
            keys_to_remove = [key for key, _ in sorted_keys[:len(sorted_keys) // 4]]
        
        for key in keys_to_remove:
            if key in self._cache:
                del self._cache[key]
            if key in self._access_times:
                del self._access_times[key]
            if key in self._access_counts:
                del self._access_counts[key]
    
    async def _cleanup_loop(self) -> None:
        """Boucle de nettoyage automatique."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_expired()
            except Exception as e:
                logger.error(f"Erreur nettoyage cache: {e}")
    
    async def _cleanup_expired(self) -> None:
        """Nettoie les entrées expirées."""
        if self.config.strategy != CacheStrategy.TTL:
            return
        
        current_time = time.time()
        expired_keys = []
        
        async with self._lock:
            for key, entry in self._cache.items():
                if current_time > entry['expires_at']:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
                if key in self._access_counts:
                    del self._access_counts[key]
        
        if expired_keys:
            logger.debug(f"Nettoyé {len(expired_keys)} entrées expirées du cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        total_size = sum(entry['size'] for entry in self._cache.values())
        
        return {
            'size': len(self._cache),
            'max_size': self.config.max_size,
            'total_memory_bytes': total_size,
            'hit_rate': self.metrics.cache_hit_rate,
            'hits': self.metrics.cache_hits,
            'misses': self.metrics.cache_misses,
            'strategy': self.config.strategy.value,
            'ttl_seconds': self.config.ttl_seconds
        }
    
    def __len__(self) -> int:
        """Retourne la taille du cache."""
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        """Vérifie si une clé est dans le cache (synchrone)."""
        return key in self._cache

def slack_retry(max_attempts: int = 3,
               base_delay: float = 1.0,
               strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
               exceptions: Tuple[type, ...] = (Exception,)) -> Callable:
    """
    Décorateur de retry simplifié pour Slack.
    
    Args:
        max_attempts: Nombre maximum de tentatives
        base_delay: Délai de base
        strategy: Stratégie de retry
        exceptions: Exceptions à capturer
        
    Returns:
        Décorateur configuré
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        strategy=strategy,
        exceptions=exceptions
    )
    
    return SlackRetryDecorator(config)

def generate_cache_key(*args, **kwargs) -> str:
    """
    Génère une clé de cache basée sur les arguments.
    
    Args:
        *args: Arguments positionnels
        **kwargs: Arguments nommés
        
    Returns:
        Clé de cache MD5
    """
    # Créer une représentation sérialisable
    cache_data = {
        'args': args,
        'kwargs': sorted(kwargs.items())
    }
    
    cache_string = json.dumps(cache_data, default=str, sort_keys=True)
    return hashlib.md5(cache_string.encode()).hexdigest()

@lru_cache(maxsize=1000)
def get_slack_color_for_severity(severity: str) -> str:
    """
    Retourne la couleur Slack appropriée pour un niveau de sévérité.
    
    Args:
        severity: Niveau de sévérité
        
    Returns:
        Code couleur Slack
    """
    color_map = {
        'critical': '#FF0000',  # Rouge vif
        'high': '#FF6B6B',      # Rouge
        'medium': '#FFB74D',    # Orange
        'low': '#4CAF50',       # Vert
        'info': '#2196F3',      # Bleu
        'warning': '#FF9800',   # Orange foncé
        'error': '#F44336',     # Rouge foncé
        'success': '#4CAF50',   # Vert
        'debug': '#9E9E9E'      # Gris
    }
    
    return color_map.get(severity.lower(), '#757575')  # Gris par défaut

def format_duration_human(seconds: float) -> str:
    """
    Formate une durée en secondes en format humain.
    
    Args:
        seconds: Durée en secondes
        
    Returns:
        Durée formatée
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}j"

def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Sérialisation JSON sécurisée avec gestion des types non-sérialisables.
    
    Args:
        obj: Objet à sérialiser
        **kwargs: Arguments pour json.dumps
        
    Returns:
        JSON string
    """
    def default_serializer(o):
        if isinstance(o, datetime):
            return o.isoformat()
        elif hasattr(o, '__dict__'):
            return o.__dict__
        else:
            return str(o)
    
    return json.dumps(obj, default=default_serializer, **kwargs)

def mask_sensitive_data(data: str, patterns: Optional[List[str]] = None) -> str:
    """
    Masque les données sensibles dans un texte.
    
    Args:
        data: Texte à traiter
        patterns: Patterns spécifiques à masquer
        
    Returns:
        Texte avec données masquées
    """
    if patterns is None:
        patterns = [
            r'xox[bpoa]-[A-Za-z0-9-]+',  # Tokens Slack
            r'https://hooks\.slack\.com/services/[A-Z0-9/]+',  # Webhooks
            r'[A-Za-z0-9+/]{40,}',  # Potential secrets
        ]
    
    result = data
    for pattern in patterns:
        result = re.sub(pattern, '[MASKED]', result)
    
    return result
