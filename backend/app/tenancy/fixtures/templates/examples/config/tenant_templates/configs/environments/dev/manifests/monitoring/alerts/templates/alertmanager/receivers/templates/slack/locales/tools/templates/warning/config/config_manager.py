"""
Gestionnaire de Configuration Warning - Spotify AI Agent
======================================================

Gestionnaire centralisé ultra-avancé pour les configurations d'alertes Warning
avec support multi-tenant, cache distribué et validation sécurisée.

Auteur: Équipe d'experts dirigée par Fahed Mlaiel
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import redis
import asyncio
from cryptography.fernet import Fernet
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
import time

# Configuration du logging
logger = logging.getLogger(__name__)

@dataclass
class TenantProfile:
    """Profil de configuration pour un tenant."""
    tenant_id: str
    profile_type: str  # basic, premium, enterprise
    max_alerts_per_minute: int
    features: List[str]
    alert_retention_days: int
    escalation_enabled: bool
    notifications_enabled: bool
    created_at: datetime
    updated_at: datetime

@dataclass
class AlertConfig:
    """Configuration complète d'une alerte Warning."""
    alert_id: str
    tenant_id: str
    level: str
    priority: int
    color: str
    escalation_minutes: Optional[int]
    channels: List[str]
    template_id: str
    rate_limit: int
    cache_ttl: int
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

class ConfigEncryption:
    """Gestionnaire de chiffrement pour les configurations sensibles."""
    
    def __init__(self, encryption_key: str):
        self.fernet = Fernet(encryption_key.encode() if len(encryption_key) == 32 
                           else Fernet.generate_key())
    
    def encrypt(self, data: str) -> str:
        """Chiffre une chaîne de caractères."""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Déchiffre une chaîne de caractères."""
        return self.fernet.decrypt(encrypted_data.encode()).decode()

class ConfigCache:
    """Cache distribué Redis pour les configurations."""
    
    def __init__(self, redis_url: str, default_ttl: int = 3600):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.default_ttl = default_ttl
        self.local_cache = {}
        self.cache_lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache avec fallback local."""
        try:
            # Essai cache Redis d'abord
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.warning(f"Erreur cache Redis: {e}")
        
        # Fallback sur cache local
        with self.cache_lock:
            return self.local_cache.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Définit une valeur dans le cache."""
        ttl = ttl or self.default_ttl
        serialized_value = json.dumps(value, default=str)
        
        try:
            # Cache Redis principal
            self.redis_client.setex(key, ttl, serialized_value)
        except Exception as e:
            logger.warning(f"Erreur écriture cache Redis: {e}")
        
        # Cache local de secours
        with self.cache_lock:
            self.local_cache[key] = value
            # Nettoyage automatique du cache local
            if len(self.local_cache) > 1000:
                self.local_cache.clear()
        
        return True
    
    def delete(self, key: str) -> bool:
        """Supprime une clé du cache."""
        try:
            self.redis_client.delete(key)
        except Exception:
            pass
        
        with self.cache_lock:
            self.local_cache.pop(key, None)
        
        return True

class WarningConfigManager:
    """
    Gestionnaire principal de configuration pour les alertes Warning.
    
    Fonctionnalités:
    - Gestion multi-tenant avec isolation
    - Cache distribué Redis avec fallback local
    - Chiffrement des données sensibles
    - Validation et sanitisation des configurations
    - Support des profils utilisateur (basic, premium, enterprise)
    - Monitoring et métriques en temps réel
    """
    
    def __init__(self, config_path: str = None, redis_url: str = None):
        """Initialise le gestionnaire de configuration."""
        self.config_path = Path(config_path or os.getcwd())
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        
        # Initialisation des composants
        self._init_encryption()
        self._init_cache()
        self._init_config()
        self._init_metrics()
        
        # Pool de threads pour opérations asynchrones
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        logger.info("WarningConfigManager initialisé avec succès")
    
    def _init_encryption(self):
        """Initialise le système de chiffrement."""
        encryption_key = os.getenv('ENCRYPTION_KEY')
        if not encryption_key:
            encryption_key = Fernet.generate_key().decode()
            logger.warning("Clé de chiffrement générée automatiquement")
        
        self.encryption = ConfigEncryption(encryption_key)
    
    def _init_cache(self):
        """Initialise le cache distribué."""
        cache_ttl = int(os.getenv('CONFIG_CACHE_TTL', '7200'))
        self.cache = ConfigCache(self.redis_url, cache_ttl)
    
    def _init_config(self):
        """Charge les configurations de base."""
        self.default_config = self._load_default_config()
        self.tenant_profiles = self._load_tenant_profiles()
    
    def _init_metrics(self):
        """Initialise le système de métriques."""
        self.metrics = {
            'config_loads': 0,
            'config_saves': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'validation_errors': 0,
            'encryption_operations': 0
        }
        self.metrics_lock = threading.RLock()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Charge la configuration par défaut."""
        default_config_path = self.config_path / "settings.yml"
        
        if default_config_path.exists():
            with open(default_config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        return {
            "general": {
                "name": "Spotify AI Agent Warning System",
                "version": "1.0.0",
                "environment": "dev"
            },
            "alerting": {
                "levels": [
                    {"name": "WARNING", "priority": 3, "color": "#FFD700"}
                ]
            }
        }
    
    def _load_tenant_profiles(self) -> Dict[str, TenantProfile]:
        """Charge les profils de tenants."""
        profiles = {}
        
        # Profils par défaut
        default_profiles = {
            "basic": TenantProfile(
                tenant_id="default",
                profile_type="basic",
                max_alerts_per_minute=50,
                features=["basic_alerting", "slack_notifications"],
                alert_retention_days=7,
                escalation_enabled=False,
                notifications_enabled=True,
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            "premium": TenantProfile(
                tenant_id="default",
                profile_type="premium",
                max_alerts_per_minute=200,
                features=["advanced_alerting", "slack_notifications", "email_notifications"],
                alert_retention_days=30,
                escalation_enabled=True,
                notifications_enabled=True,
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            "enterprise": TenantProfile(
                tenant_id="default",
                profile_type="enterprise",
                max_alerts_per_minute=1000,
                features=["all_features_enabled", "custom_integrations"],
                alert_retention_days=90,
                escalation_enabled=True,
                notifications_enabled=True,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        }
        
        return default_profiles
    
    def get_tenant_config(self, tenant_id: str, profile_type: str = "basic") -> TenantProfile:
        """Récupère la configuration d'un tenant."""
        cache_key = f"tenant_config:{tenant_id}:{profile_type}"
        
        # Vérification du cache
        cached_config = self.cache.get(cache_key)
        if cached_config:
            self._increment_metric('cache_hits')
            return TenantProfile(**cached_config)
        
        self._increment_metric('cache_misses')
        
        # Création ou récupération de la configuration
        if profile_type in self.tenant_profiles:
            base_profile = self.tenant_profiles[profile_type]
            tenant_profile = TenantProfile(
                tenant_id=tenant_id,
                profile_type=base_profile.profile_type,
                max_alerts_per_minute=base_profile.max_alerts_per_minute,
                features=base_profile.features.copy(),
                alert_retention_days=base_profile.alert_retention_days,
                escalation_enabled=base_profile.escalation_enabled,
                notifications_enabled=base_profile.notifications_enabled,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        else:
            tenant_profile = self.tenant_profiles["basic"]
            tenant_profile.tenant_id = tenant_id
        
        # Mise en cache
        self.cache.set(cache_key, asdict(tenant_profile))
        self._increment_metric('config_loads')
        
        return tenant_profile
    
    def create_warning_config(self, tenant_id: str, level: str = "WARNING", 
                            channels: List[str] = None, escalation_enabled: bool = True,
                            custom_config: Dict[str, Any] = None) -> AlertConfig:
        """Crée une configuration d'alerte Warning personnalisée."""
        
        # Validation des paramètres
        if not self._validate_tenant_id(tenant_id):
            raise ValueError(f"Tenant ID invalide: {tenant_id}")
        
        if not self._validate_alert_level(level):
            raise ValueError(f"Niveau d'alerte invalide: {level}")
        
        # Configuration par défaut
        alert_config = AlertConfig(
            alert_id=self._generate_alert_id(tenant_id, level),
            tenant_id=tenant_id,
            level=level,
            priority=3,  # WARNING par défaut
            color="#FFD700",
            escalation_minutes=60 if escalation_enabled else None,
            channels=channels or ["slack"],
            template_id=f"warning_template_{tenant_id}",
            rate_limit=100,
            cache_ttl=3600,
            metadata=custom_config or {},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Application de la configuration tenant
        tenant_profile = self.get_tenant_config(tenant_id)
        alert_config.rate_limit = min(alert_config.rate_limit, tenant_profile.max_alerts_per_minute)
        
        # Validation et sauvegarde
        if self._validate_alert_config(alert_config):
            self._save_alert_config(alert_config)
            self._increment_metric('config_saves')
            
            logger.info(f"Configuration d'alerte créée: {alert_config.alert_id}")
            return alert_config
        else:
            self._increment_metric('validation_errors')
            raise ValueError("Configuration d'alerte invalide")
    
    def _validate_tenant_id(self, tenant_id: str) -> bool:
        """Valide un ID de tenant."""
        if not tenant_id or len(tenant_id) < 3:
            return False
        
        # Vérification des caractères autorisés
        allowed_chars = set('abcdefghijklmnopqrstuvwxyz0123456789_-')
        return all(c.lower() in allowed_chars for c in tenant_id)
    
    def _validate_alert_level(self, level: str) -> bool:
        """Valide un niveau d'alerte."""
        valid_levels = ["CRITICAL", "HIGH", "WARNING", "INFO", "DEBUG"]
        return level.upper() in valid_levels
    
    def _validate_alert_config(self, config: AlertConfig) -> bool:
        """Valide une configuration d'alerte complète."""
        try:
            # Validations de base
            if not config.alert_id or not config.tenant_id:
                return False
            
            if config.priority < 1 or config.priority > 5:
                return False
            
            if config.rate_limit < 1 or config.rate_limit > 10000:
                return False
            
            # Validation des canaux
            valid_channels = ["slack", "email", "webhook", "pagerduty"]
            if not all(channel in valid_channels for channel in config.channels):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur validation configuration: {e}")
            return False
    
    def _generate_alert_id(self, tenant_id: str, level: str) -> str:
        """Génère un ID unique pour l'alerte."""
        timestamp = int(time.time() * 1000)
        data = f"{tenant_id}:{level}:{timestamp}"
        hash_obj = hashlib.md5(data.encode())
        return f"warning_{hash_obj.hexdigest()[:12]}"
    
    def _save_alert_config(self, config: AlertConfig) -> bool:
        """Sauvegarde une configuration d'alerte."""
        cache_key = f"alert_config:{config.alert_id}"
        
        # Chiffrement des données sensibles
        sensitive_data = {
            "tenant_id": config.tenant_id,
            "metadata": config.metadata
        }
        
        encrypted_data = self.encryption.encrypt(json.dumps(sensitive_data))
        config_dict = asdict(config)
        config_dict["encrypted_data"] = encrypted_data
        
        # Sauvegarde en cache
        return self.cache.set(cache_key, config_dict, config.cache_ttl)
    
    def _increment_metric(self, metric_name: str):
        """Incrémente une métrique."""
        with self.metrics_lock:
            self.metrics[metric_name] = self.metrics.get(metric_name, 0) + 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du gestionnaire."""
        with self.metrics_lock:
            return self.metrics.copy()
    
    def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé du gestionnaire."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Vérification Redis
        try:
            self.cache.redis_client.ping()
            health_status["components"]["redis"] = "healthy"
        except Exception as e:
            health_status["components"]["redis"] = f"unhealthy: {e}"
            health_status["status"] = "degraded"
        
        # Vérification chiffrement
        try:
            test_data = "test"
            encrypted = self.encryption.encrypt(test_data)
            decrypted = self.encryption.decrypt(encrypted)
            health_status["components"]["encryption"] = "healthy" if decrypted == test_data else "unhealthy"
        except Exception as e:
            health_status["components"]["encryption"] = f"unhealthy: {e}"
            health_status["status"] = "degraded"
        
        return health_status
    
    def cleanup(self):
        """Nettoie les ressources."""
        self.executor.shutdown(wait=True)
        logger.info("WarningConfigManager nettoyé avec succès")

# Factory function pour faciliter l'utilisation
def create_warning_config_manager(config_path: str = None, redis_url: str = None) -> WarningConfigManager:
    """Factory function pour créer un gestionnaire de configuration."""
    return WarningConfigManager(config_path, redis_url)
