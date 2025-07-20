"""
Configuration Slack Ultra-Avancée pour AlertManager
===================================================

Module de configuration centralisée pour les notifications Slack
dans le système de monitoring AlertManager du Spotify AI Agent.

Fonctionnalités:
- Configuration multi-tenant
- Gestion dynamique des webhooks
- Templates personnalisables
- Routage intelligent par criticité
- Chiffrement des tokens
- Monitoring des performances
- Audit trail complet

Architecture développée par l'équipe Backend Senior sous la direction de Fahed Mlaiel.
"""

import os
import json
import yaml
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import base64
from cryptography.fernet import Fernet
import redis.asyncio as redis
from pydantic import BaseModel, Field, validator
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from . import SlackConfig, SlackSeverity, SlackChannelType
from .utils import SlackUtils

logger = logging.getLogger(__name__)

class SlackEnvironmentConfig(BaseModel):
    """Configuration Slack par environnement."""
    
    environment: str = Field(..., description="Nom de l'environnement")
    slack_config: Dict[str, Any] = Field(default_factory=dict)
    webhook_urls: Dict[str, str] = Field(default_factory=dict)
    channel_mappings: Dict[str, str] = Field(default_factory=dict)
    tenant_configs: Dict[str, Dict] = Field(default_factory=dict)
    security_settings: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('environment')
    def validate_environment(cls, v):
        allowed_envs = ['dev', 'staging', 'prod', 'test']
        if v not in allowed_envs:
            raise ValueError(f"Environment doit être dans {allowed_envs}")
        return v

class SlackTenantConfig(BaseModel):
    """Configuration Slack par tenant."""
    
    tenant_id: str = Field(..., description="ID du tenant")
    tenant_name: str = Field(..., description="Nom du tenant")
    workspace_id: str = Field(..., description="ID workspace Slack")
    bot_token: str = Field(..., description="Token bot Slack")
    channels: Dict[str, str] = Field(default_factory=dict)
    webhooks: Dict[str, str] = Field(default_factory=dict)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class SlackNotificationConfig:
    """
    Gestionnaire de configuration Slack ultra-avancé.
    
    Cette classe centralise toute la configuration Slack pour le système
    de monitoring AlertManager, avec support multi-tenant, chiffrement
    et gestion dynamique.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 redis_client: Optional[redis.Redis] = None,
                 encryption_key: Optional[str] = None):
        """
        Initialise le gestionnaire de configuration Slack.
        
        Args:
            config_path: Chemin vers le fichier de configuration
            redis_client: Client Redis pour le cache
            encryption_key: Clé de chiffrement pour les tokens
        """
        self.config_path = config_path or self._get_default_config_path()
        self.redis_client = redis_client
        self.encryption_key = encryption_key or self._generate_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key.encode() if isinstance(self.encryption_key, str) else self.encryption_key)
        
        # Configuration par défaut
        self.default_config = SlackConfig()
        
        # Cache configuration
        self._config_cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_reload = None
        
        # Métriques
        self.metrics = {
            'config_loads': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'encryption_operations': 0,
            'validation_errors': 0
        }
        
        # Initialisation
        self._initialize()
    
    def _get_default_config_path(self) -> str:
        """Retourne le chemin par défaut du fichier de configuration."""
        base_path = Path(__file__).parent
        return str(base_path / "slack_config.yaml")
    
    def _generate_encryption_key(self) -> bytes:
        """Génère une clé de chiffrement sécurisée."""
        return Fernet.generate_key()
    
    def _initialize(self):
        """Initialise la configuration."""
        try:
            self.load_configuration()
            logger.info("Configuration Slack initialisée avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation: {e}")
            self._use_default_config()
    
    def _use_default_config(self):
        """Utilise la configuration par défaut en cas d'erreur."""
        self._config_cache = {
            'default': asdict(self.default_config)
        }
        logger.warning("Utilisation de la configuration par défaut")
    
    async def load_configuration(self) -> Dict[str, Any]:
        """
        Charge la configuration depuis le fichier ou le cache Redis.
        
        Returns:
            Configuration Slack complète
        """
        self.metrics['config_loads'] += 1
        
        # Vérifier le cache local
        if self._is_cache_valid():
            self.metrics['cache_hits'] += 1
            return self._config_cache
        
        # Vérifier le cache Redis
        if self.redis_client:
            cached_config = await self._load_from_redis()
            if cached_config:
                self.metrics['cache_hits'] += 1
                self._config_cache = cached_config
                self._last_reload = datetime.utcnow()
                return cached_config
        
        # Charger depuis le fichier
        self.metrics['cache_misses'] += 1
        config = await self._load_from_file()
        
        # Mettre en cache
        await self._cache_configuration(config)
        
        return config
    
    def _is_cache_valid(self) -> bool:
        """Vérifie si le cache local est valide."""
        if not self._config_cache or not self._last_reload:
            return False
        
        return (datetime.utcnow() - self._last_reload).seconds < self._cache_ttl
    
    async def _load_from_redis(self) -> Optional[Dict[str, Any]]:
        """Charge la configuration depuis Redis."""
        try:
            if not self.redis_client:
                return None
            
            cached_data = await self.redis_client.get("slack_config")
            if cached_data:
                decrypted_data = self._decrypt_data(cached_data)
                return json.loads(decrypted_data)
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement Redis: {e}")
        
        return None
    
    async def _load_from_file(self) -> Dict[str, Any]:
        """Charge la configuration depuis le fichier."""
        try:
            config_path = Path(self.config_path)
            
            if not config_path.exists():
                # Créer le fichier de configuration par défaut
                await self._create_default_config_file()
            
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() == '.yaml':
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            # Valider et traiter la configuration
            processed_config = await self._process_configuration(config)
            
            self._config_cache = processed_config
            self._last_reload = datetime.utcnow()
            
            return processed_config
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement fichier: {e}")
            self.metrics['validation_errors'] += 1
            return asdict(self.default_config)
    
    async def _create_default_config_file(self):
        """Crée le fichier de configuration par défaut."""
        default_config = {
            'version': '2.1.0',
            'environments': {
                'dev': {
                    'slack_config': asdict(self.default_config),
                    'webhook_urls': {
                        'critical': os.getenv('SLACK_WEBHOOK_CRITICAL', ''),
                        'high': os.getenv('SLACK_WEBHOOK_HIGH', ''),
                        'medium': os.getenv('SLACK_WEBHOOK_MEDIUM', ''),
                        'low': os.getenv('SLACK_WEBHOOK_LOW', ''),
                        'info': os.getenv('SLACK_WEBHOOK_INFO', '')
                    },
                    'security': {
                        'encryption_enabled': True,
                        'token_rotation_days': 30,
                        'webhook_validation': True
                    }
                }
            },
            'global_settings': {
                'retry_policy': {
                    'max_attempts': 3,
                    'backoff_factor': 2,
                    'max_delay': 300
                },
                'rate_limiting': {
                    'requests_per_minute': 100,
                    'burst_limit': 20
                },
                'monitoring': {
                    'metrics_enabled': True,
                    'logging_level': 'INFO',
                    'audit_trail': True
                }
            }
        }
        
        config_path = Path(self.config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Fichier de configuration par défaut créé: {config_path}")
    
    async def _process_configuration(self, raw_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traite et valide la configuration brute.
        
        Args:
            raw_config: Configuration brute du fichier
            
        Returns:
            Configuration traitée et validée
        """
        processed = {}
        
        # Traiter chaque environnement
        for env_name, env_config in raw_config.get('environments', {}).items():
            try:
                env_model = SlackEnvironmentConfig(
                    environment=env_name,
                    **env_config
                )
                processed[env_name] = env_model.dict()
                
                # Déchiffrer les tokens si nécessaire
                if 'slack_config' in processed[env_name]:
                    await self._decrypt_tokens(processed[env_name]['slack_config'])
                
            except Exception as e:
                logger.error(f"Erreur traitement environnement {env_name}: {e}")
                self.metrics['validation_errors'] += 1
        
        # Ajouter les paramètres globaux
        processed['global_settings'] = raw_config.get('global_settings', {})
        processed['version'] = raw_config.get('version', '2.1.0')
        processed['loaded_at'] = datetime.utcnow().isoformat()
        
        return processed
    
    async def _decrypt_tokens(self, config: Dict[str, Any]):
        """Déchiffre les tokens dans la configuration."""
        try:
            encrypted_fields = ['bot_token', 'signing_secret', 'app_token']
            
            for field in encrypted_fields:
                if field in config and config[field]:
                    if config[field].startswith('enc:'):
                        encrypted_value = config[field][4:]  # Enlever 'enc:'
                        config[field] = self._decrypt_data(encrypted_value)
                        self.metrics['encryption_operations'] += 1
                        
        except Exception as e:
            logger.error(f"Erreur déchiffrement tokens: {e}")
    
    def _encrypt_data(self, data: str) -> str:
        """Chiffre une donnée."""
        try:
            encrypted = self.cipher_suite.encrypt(data.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Erreur chiffrement: {e}")
            return data
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """Déchiffre une donnée."""
        try:
            decoded = base64.b64decode(encrypted_data.encode())
            decrypted = self.cipher_suite.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Erreur déchiffrement: {e}")
            return encrypted_data
    
    async def _cache_configuration(self, config: Dict[str, Any]):
        """Met en cache la configuration."""
        try:
            if self.redis_client:
                encrypted_config = self._encrypt_data(json.dumps(config))
                await self.redis_client.setex(
                    "slack_config",
                    self._cache_ttl,
                    encrypted_config
                )
                self.metrics['encryption_operations'] += 1
                
        except Exception as e:
            logger.error(f"Erreur mise en cache: {e}")
    
    async def get_tenant_config(self, tenant_id: str, environment: str = 'dev') -> Optional[SlackTenantConfig]:
        """
        Récupère la configuration d'un tenant spécifique.
        
        Args:
            tenant_id: ID du tenant
            environment: Environnement cible
            
        Returns:
            Configuration du tenant ou None
        """
        try:
            config = await self.load_configuration()
            
            env_config = config.get(environment, {})
            tenant_configs = env_config.get('tenant_configs', {})
            
            if tenant_id in tenant_configs:
                tenant_data = tenant_configs[tenant_id]
                return SlackTenantConfig(**tenant_data)
            
            # Configuration par défaut pour le tenant
            return self._create_default_tenant_config(tenant_id)
            
        except Exception as e:
            logger.error(f"Erreur récupération config tenant {tenant_id}: {e}")
            return None
    
    def _create_default_tenant_config(self, tenant_id: str) -> SlackTenantConfig:
        """Crée une configuration par défaut pour un tenant."""
        return SlackTenantConfig(
            tenant_id=tenant_id,
            tenant_name=f"Tenant {tenant_id}",
            workspace_id=f"workspace_{tenant_id}",
            bot_token="",
            channels=self.default_config.severity_channels.copy(),
            webhooks={},
            preferences={
                'timezone': 'UTC',
                'date_format': '%Y-%m-%d %H:%M:%S',
                'language': 'en'
            }
        )
    
    async def update_tenant_config(self, 
                                 tenant_id: str, 
                                 tenant_config: SlackTenantConfig,
                                 environment: str = 'dev') -> bool:
        """
        Met à jour la configuration d'un tenant.
        
        Args:
            tenant_id: ID du tenant
            tenant_config: Nouvelle configuration
            environment: Environnement cible
            
        Returns:
            True si succès, False sinon
        """
        try:
            config = await self.load_configuration()
            
            if environment not in config:
                config[environment] = {
                    'slack_config': asdict(self.default_config),
                    'tenant_configs': {}
                }
            
            if 'tenant_configs' not in config[environment]:
                config[environment]['tenant_configs'] = {}
            
            # Chiffrer les tokens sensibles
            tenant_data = tenant_config.dict()
            if tenant_data.get('bot_token'):
                tenant_data['bot_token'] = f"enc:{self._encrypt_data(tenant_data['bot_token'])}"
            
            config[environment]['tenant_configs'][tenant_id] = tenant_data
            
            # Sauvegarder
            await self._save_configuration(config)
            
            # Invalider le cache
            self._invalidate_cache()
            
            logger.info(f"Configuration tenant {tenant_id} mise à jour")
            return True
            
        except Exception as e:
            logger.error(f"Erreur mise à jour config tenant {tenant_id}: {e}")
            return False
    
    async def _save_configuration(self, config: Dict[str, Any]):
        """Sauvegarde la configuration."""
        try:
            config_path = Path(self.config_path)
            
            # Backup de l'ancienne configuration
            if config_path.exists():
                backup_path = config_path.with_suffix(f"{config_path.suffix}.backup")
                config_path.rename(backup_path)
            
            # Sauvegarder la nouvelle configuration
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            # Mettre à jour le cache Redis
            await self._cache_configuration(config)
            
            logger.info("Configuration sauvegardée avec succès")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde configuration: {e}")
            raise
    
    def _invalidate_cache(self):
        """Invalide le cache local."""
        self._config_cache.clear()
        self._last_reload = None
    
    async def validate_configuration(self, config: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
        """
        Valide la configuration Slack.
        
        Args:
            config: Configuration à valider (optionnel)
            
        Returns:
            Tuple (is_valid, errors)
        """
        errors = []
        
        try:
            if config is None:
                config = await self.load_configuration()
            
            # Validation de la version
            version = config.get('version')
            if not version:
                errors.append("Version manquante dans la configuration")
            
            # Validation des environnements
            environments = config.get('environments', {})
            if not environments:
                errors.append("Aucun environnement configuré")
            
            for env_name, env_config in environments.items():
                env_errors = await self._validate_environment_config(env_name, env_config)
                errors.extend(env_errors)
            
            # Validation des paramètres globaux
            global_settings = config.get('global_settings', {})
            global_errors = self._validate_global_settings(global_settings)
            errors.extend(global_errors)
            
            is_valid = len(errors) == 0
            
            if not is_valid:
                self.metrics['validation_errors'] += len(errors)
            
            return is_valid, errors
            
        except Exception as e:
            logger.error(f"Erreur validation configuration: {e}")
            return False, [f"Erreur validation: {e}"]
    
    async def _validate_environment_config(self, env_name: str, env_config: Dict[str, Any]) -> List[str]:
        """Valide la configuration d'un environnement."""
        errors = []
        
        try:
            # Valider avec Pydantic
            SlackEnvironmentConfig(environment=env_name, **env_config)
            
            # Validations supplémentaires
            slack_config = env_config.get('slack_config', {})
            
            # Vérifier les canaux obligatoires
            severity_channels = slack_config.get('severity_channels', {})
            required_severities = ['critical', 'high', 'medium', 'low', 'info']
            
            for severity in required_severities:
                if severity not in severity_channels:
                    errors.append(f"Canal {severity} manquant pour {env_name}")
            
            # Vérifier les webhooks
            webhook_urls = env_config.get('webhook_urls', {})
            for severity, url in webhook_urls.items():
                if url and not self._is_valid_webhook_url(url):
                    errors.append(f"URL webhook invalide pour {severity} dans {env_name}")
            
        except Exception as e:
            errors.append(f"Configuration environnement {env_name} invalide: {e}")
        
        return errors
    
    def _validate_global_settings(self, global_settings: Dict[str, Any]) -> List[str]:
        """Valide les paramètres globaux."""
        errors = []
        
        # Validation retry policy
        retry_policy = global_settings.get('retry_policy', {})
        max_attempts = retry_policy.get('max_attempts', 3)
        if not isinstance(max_attempts, int) or max_attempts < 1:
            errors.append("retry_policy.max_attempts doit être un entier positif")
        
        # Validation rate limiting
        rate_limiting = global_settings.get('rate_limiting', {})
        rpm = rate_limiting.get('requests_per_minute', 100)
        if not isinstance(rpm, int) or rpm < 1:
            errors.append("rate_limiting.requests_per_minute doit être un entier positif")
        
        return errors
    
    def _is_valid_webhook_url(self, url: str) -> bool:
        """Valide une URL de webhook Slack."""
        if not url.startswith('https://hooks.slack.com/services/'):
            return False
        
        # Validation supplémentaire du format
        parts = url.split('/')
        return len(parts) >= 6 and all(part for part in parts[-3:])
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de performance."""
        cache_hit_rate = 0
        if self.metrics['cache_hits'] + self.metrics['cache_misses'] > 0:
            cache_hit_rate = self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses'])
        
        return {
            **self.metrics,
            'cache_hit_rate': cache_hit_rate,
            'last_reload': self._last_reload.isoformat() if self._last_reload else None,
            'cache_size': len(self._config_cache),
            'uptime': datetime.utcnow().isoformat()
        }
    
    async def reload_configuration(self) -> bool:
        """Force le rechargement de la configuration."""
        try:
            self._invalidate_cache()
            if self.redis_client:
                await self.redis_client.delete("slack_config")
            
            await self.load_configuration()
            logger.info("Configuration rechargée avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur rechargement configuration: {e}")
            return False
    
    def __repr__(self) -> str:
        return f"SlackNotificationConfig(config_path='{self.config_path}', cache_size={len(self._config_cache)})"
