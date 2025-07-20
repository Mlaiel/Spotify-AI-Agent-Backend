#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spotify AI Agent - Locale Manager pour Alerting Multi-Tenant

Gestionnaire centralisé pour la localisation des alertes avec support
avancé pour l'écosystème Spotify AI Agent.

Fonctionnalités:
- Chargement dynamique des locales
- Cache intelligent avec TTL
- Fallback automatique vers langue par défaut
- Variables contextuelles pour IA musicale
- Templates Jinja2 avec fonctions custom
- Validation syntaxique des traductions

Architecture:
- Singleton pattern pour instance unique
- Lazy loading des ressources
- Cache LRU avec expiration
- Observer pattern pour reload à chaud

Performance:
- Cache Redis distribué (optionnel)
- Compilation des templates
- Métriques détaillées
- Pool de connexions optimisé

Utilisation:
    manager = LocaleManager()
    message = manager.get_localized_string(
        key="ai_model_degraded",
        locale="fr",
        context={
            "model_name": "MusicGenAI-v2",
            "accuracy_drop": 15.2,
            "tenant_name": "Universal Music"
        }
    )
"""

import json
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from functools import lru_cache, wraps
import logging

# Imports externes
import yaml
import jinja2
from jinja2 import Environment, FileSystemLoader, select_autoescape
import redis
from prometheus_client import Counter, Histogram, Gauge
import structlog

# Configuration du logger structuré
logger = structlog.get_logger(__name__)

# Métriques Prometheus
locale_cache_hits = Counter(
    'spotify_ai_locale_cache_hits_total',
    'Nombre de hits du cache locale',
    ['tenant_id', 'locale']
)

locale_cache_misses = Counter(
    'spotify_ai_locale_cache_misses_total', 
    'Nombre de miss du cache locale',
    ['tenant_id', 'locale']
)

locale_rendering_duration = Histogram(
    'spotify_ai_locale_rendering_seconds',
    'Durée de rendu des templates',
    ['template_type', 'locale']
)

active_locales = Gauge(
    'spotify_ai_active_locales',
    'Nombre de locales actives'
)

@dataclass
class LocaleConfig:
    """Configuration pour une locale spécifique."""
    code: str
    name: str
    direction: str = "ltr"  # ltr ou rtl
    fallback: Optional[str] = None
    enabled: bool = True
    priority: int = 0
    date_format: str = "%Y-%m-%d %H:%M:%S"
    number_format: str = "{:,.2f}"
    currency_symbol: str = "€"
    timezone: str = "UTC"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class CacheEntry:
    """Entrée du cache avec métadonnées."""
    value: Any
    created_at: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    ttl_seconds: int = 3600

    def is_expired(self) -> bool:
        """Vérifie si l'entrée a expiré."""
        return datetime.utcnow() > self.created_at + timedelta(seconds=self.ttl_seconds)

    def touch(self):
        """Met à jour l'heure d'accès."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1

class LocaleManager:
    """
    Gestionnaire centralisé des locales pour alerting multi-tenant.
    
    Implémente un pattern Singleton avec cache intelligent et
    support pour templates Jinja2 contextualisés.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    # Configuration des locales supportées
    SUPPORTED_LOCALES = {
        "fr": LocaleConfig(
            code="fr",
            name="Français",
            fallback="en",
            priority=10,
            date_format="%d/%m/%Y %H:%M:%S",
            currency_symbol="€",
            timezone="Europe/Paris"
        ),
        "en": LocaleConfig(
            code="en", 
            name="English",
            priority=100,  # Langue par défaut
            currency_symbol="$",
            timezone="UTC"
        ),
        "de": LocaleConfig(
            code="de",
            name="Deutsch", 
            fallback="en",
            priority=20,
            date_format="%d.%m.%Y %H:%M:%S",
            currency_symbol="€",
            timezone="Europe/Berlin"
        ),
        "es": LocaleConfig(
            code="es",
            name="Español",
            fallback="en", 
            priority=15,
            date_format="%d/%m/%Y %H:%M:%S",
            currency_symbol="€",
            timezone="Europe/Madrid"
        ),
        "it": LocaleConfig(
            code="it",
            name="Italiano",
            fallback="en",
            priority=12,
            date_format="%d/%m/%Y %H:%M:%S", 
            currency_symbol="€",
            timezone="Europe/Rome"
        )
    }
    
    DEFAULT_LOCALE = "en"
    CACHE_TTL = 3600  # 1 heure
    MAX_CACHE_SIZE = 10000
    
    def __new__(cls):
        """Pattern Singleton thread-safe."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialisation du gestionnaire de locales."""
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        
        # Cache en mémoire avec LRU
        self._cache: Dict[str, CacheEntry] = {}
        self._cache_lock = threading.RLock()
        
        # Configuration Redis (optionnel)
        self._redis_client = None
        self._enable_redis = os.getenv('LOCALE_REDIS_ENABLED', 'false').lower() == 'true'
        
        # Templates Jinja2
        self._jinja_env = None
        self._template_cache = {}
        
        # Statistiques
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'renders': 0,
            'errors': 0
        }
        
        # Initialisation
        self._init_redis()
        self._init_jinja()
        self._load_locale_files()
        
        # Métriques
        active_locales.set(len(self.SUPPORTED_LOCALES))
        
        logger.info(
            "LocaleManager initialisé",
            locales_count=len(self.SUPPORTED_LOCALES),
            redis_enabled=self._enable_redis,
            cache_ttl=self.CACHE_TTL
        )
    
    def _init_redis(self):
        """Initialise la connexion Redis si activée."""
        if not self._enable_redis:
            return
            
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            self._redis_client = redis.from_url(
                redis_url,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test de connexion
            self._redis_client.ping()
            logger.info("Connexion Redis établie", redis_url=redis_url)
            
        except Exception as e:
            logger.warning(
                "Échec connexion Redis, cache local utilisé",
                error=str(e)
            )
            self._enable_redis = False
            self._redis_client = None
    
    def _init_jinja(self):
        """Initialise l'environnement Jinja2."""
        # Chemin vers les templates
        template_dir = Path(__file__).parent / "templates"
        template_dir.mkdir(exist_ok=True)
        
        # Configuration Jinja2
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True,
            enable_async=False
        )
        
        # Fonctions custom pour templates
        self._jinja_env.globals.update({
            'format_number': self._format_number,
            'format_date': self._format_date,
            'format_currency': self._format_currency,
            'pluralize': self._pluralize,
            'truncate_smart': self._truncate_smart,
            'highlight_keywords': self._highlight_keywords
        })
        
        # Filtres custom
        self._jinja_env.filters.update({
            'spotify_format': self._spotify_format,
            'ai_context': self._ai_context_filter,
            'tenant_format': self._tenant_format_filter
        })
    
    def _load_locale_files(self):
        """Charge tous les fichiers de locale au démarrage."""
        locale_dir = Path(__file__).parent / "data"
        locale_dir.mkdir(exist_ok=True)
        
        for locale_code in self.SUPPORTED_LOCALES:
            locale_file = locale_dir / f"{locale_code}.yml"
            
            if locale_file.exists():
                try:
                    with open(locale_file, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                        cache_key = f"locale_data:{locale_code}"
                        self._set_cache(cache_key, data)
                        logger.debug(f"Locale {locale_code} chargée", keys_count=len(data))
                        
                except Exception as e:
                    logger.error(
                        f"Erreur chargement locale {locale_code}",
                        error=str(e),
                        file_path=str(locale_file)
                    )
            else:
                logger.warning(f"Fichier locale manquant", locale=locale_code, path=str(locale_file))
    
    def get_localized_string(
        self,
        key: str,
        locale: str = None,
        context: Dict[str, Any] = None,
        tenant_id: str = None,
        fallback_value: str = None
    ) -> str:
        """
        Récupère une chaîne localisée avec contexte et fallback.
        
        Args:
            key: Clé de traduction (ex: "alerts.ai_model.degraded")
            locale: Code locale (ex: "fr", "en")
            context: Variables pour template Jinja2
            tenant_id: ID du tenant pour cache/logging
            fallback_value: Valeur par défaut si traduction manquante
            
        Returns:
            Chaîne localisée et rendue
        """
        start_time = time.time()
        
        # Normalisation des paramètres
        locale = locale or self.DEFAULT_LOCALE
        context = context or {}
        tenant_id = tenant_id or "system"
        
        # Validation locale
        if locale not in self.SUPPORTED_LOCALES:
            logger.warning(f"Locale non supportée: {locale}, fallback vers {self.DEFAULT_LOCALE}")
            locale = self.DEFAULT_LOCALE
        
        # Clé de cache
        cache_key = f"localized:{tenant_id}:{locale}:{key}:{hash(str(context))}"
        
        try:
            # Vérification cache
            cached_value = self._get_cache(cache_key)
            if cached_value is not None:
                locale_cache_hits.labels(tenant_id=tenant_id, locale=locale).inc()
                self._stats['cache_hits'] += 1
                return cached_value
            
            # Cache miss
            locale_cache_misses.labels(tenant_id=tenant_id, locale=locale).inc()
            self._stats['cache_misses'] += 1
            
            # Chargement des données de locale
            locale_data = self._get_locale_data(locale)
            
            # Récupération de la valeur par clé imbriquée
            value = self._get_nested_value(locale_data, key)
            
            # Fallback si valeur manquante
            if value is None:
                fallback_locale = self.SUPPORTED_LOCALES[locale].fallback
                if fallback_locale and fallback_locale != locale:
                    logger.debug(f"Fallback vers locale {fallback_locale} pour clé {key}")
                    return self.get_localized_string(
                        key, fallback_locale, context, tenant_id, fallback_value
                    )
                else:
                    value = fallback_value or f"[{key}]"
                    logger.warning(f"Traduction manquante", key=key, locale=locale)
            
            # Rendu du template si nécessaire
            if isinstance(value, str) and context:
                try:
                    template = self._jinja_env.from_string(value)
                    rendered_value = template.render(**context)
                    
                    # Métriques de rendu
                    locale_rendering_duration.labels(
                        template_type="string",
                        locale=locale
                    ).observe(time.time() - start_time)
                    
                    self._stats['renders'] += 1
                    
                except Exception as e:
                    logger.error(f"Erreur rendu template", key=key, error=str(e))
                    rendered_value = value  # Valeur brute en cas d'erreur
                    self._stats['errors'] += 1
            else:
                rendered_value = value
            
            # Mise en cache
            self._set_cache(cache_key, rendered_value)
            
            return rendered_value
            
        except Exception as e:
            logger.error(
                "Erreur récupération chaîne localisée",
                key=key,
                locale=locale,
                tenant_id=tenant_id,
                error=str(e)
            )
            self._stats['errors'] += 1
            return fallback_value or f"[ERROR:{key}]"
    
    def _get_locale_data(self, locale: str) -> Dict[str, Any]:
        """Récupère les données d'une locale depuis le cache ou fichier."""
        cache_key = f"locale_data:{locale}"
        data = self._get_cache(cache_key)
        
        if data is not None:
            return data
        
        # Chargement depuis fichier
        locale_file = Path(__file__).parent / "data" / f"{locale}.yml"
        
        if locale_file.exists():
            try:
                with open(locale_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                    self._set_cache(cache_key, data)
                    return data
            except Exception as e:
                logger.error(f"Erreur chargement fichier locale", locale=locale, error=str(e))
        
        # Données par défaut vides
        return {}
    
    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Any:
        """Récupère une valeur imbriquée via notation pointée."""
        keys = key.split('.')
        current = data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        
        return current
    
    def _get_cache(self, key: str) -> Any:
        """Récupère une valeur du cache (Redis ou local)."""
        # Cache Redis en priorité
        if self._enable_redis and self._redis_client:
            try:
                value = self._redis_client.get(f"locale_cache:{key}")
                if value:
                    return json.loads(value)
            except Exception as e:
                logger.debug(f"Erreur lecture cache Redis", key=key, error=str(e))
        
        # Cache local
        with self._cache_lock:
            if key in self._cache:
                entry = self._cache[key]
                
                if not entry.is_expired():
                    entry.touch()
                    return entry.value
                else:
                    # Suppression entrée expirée
                    del self._cache[key]
        
        return None
    
    def _set_cache(self, key: str, value: Any, ttl: int = None):
        """Définit une valeur dans le cache."""
        ttl = ttl or self.CACHE_TTL
        
        # Cache Redis
        if self._enable_redis and self._redis_client:
            try:
                self._redis_client.setex(
                    f"locale_cache:{key}",
                    ttl,
                    json.dumps(value, ensure_ascii=False)
                )
            except Exception as e:
                logger.debug(f"Erreur écriture cache Redis", key=key, error=str(e))
        
        # Cache local
        with self._cache_lock:
            # Nettoyage si cache plein
            if len(self._cache) >= self.MAX_CACHE_SIZE:
                self._cleanup_cache()
            
            self._cache[key] = CacheEntry(
                value=value,
                created_at=datetime.utcnow(),
                ttl_seconds=ttl
            )
    
    def _cleanup_cache(self):
        """Nettoie le cache local (LRU)."""
        # Suppression des entrées expirées
        expired_keys = [
            k for k, v in self._cache.items()
            if v.is_expired()
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        # Si encore trop d'entrées, suppression LRU
        if len(self._cache) >= self.MAX_CACHE_SIZE:
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].last_accessed
            )
            
            # Suppression des 20% plus anciennes
            remove_count = len(sorted_entries) // 5
            for key, _ in sorted_entries[:remove_count]:
                del self._cache[key]
    
    # Fonctions utilitaires pour templates Jinja2
    
    def _format_number(self, value: Union[int, float], locale: str = None) -> str:
        """Formate un nombre selon la locale."""
        locale_config = self.SUPPORTED_LOCALES.get(locale or self.DEFAULT_LOCALE)
        return locale_config.number_format.format(value)
    
    def _format_date(self, value: datetime, locale: str = None) -> str:
        """Formate une date selon la locale."""
        locale_config = self.SUPPORTED_LOCALES.get(locale or self.DEFAULT_LOCALE)
        return value.strftime(locale_config.date_format)
    
    def _format_currency(self, value: Union[int, float], locale: str = None) -> str:
        """Formate une devise selon la locale."""
        locale_config = self.SUPPORTED_LOCALES.get(locale or self.DEFAULT_LOCALE)
        formatted_number = self._format_number(value, locale)
        return f"{formatted_number} {locale_config.currency_symbol}"
    
    def _pluralize(self, count: int, singular: str, plural: str = None) -> str:
        """Gère la pluralisation."""
        if count == 1:
            return singular
        return plural or f"{singular}s"
    
    def _truncate_smart(self, text: str, length: int = 100) -> str:
        """Troncature intelligente préservant les mots."""
        if len(text) <= length:
            return text
        
        truncated = text[:length]
        last_space = truncated.rfind(' ')
        
        if last_space > length * 0.7:  # Si espace proche de la fin
            return truncated[:last_space] + "..."
        
        return truncated + "..."
    
    def _highlight_keywords(self, text: str, keywords: List[str]) -> str:
        """Met en évidence des mots-clés (pour Slack)."""
        for keyword in keywords:
            text = text.replace(keyword, f"*{keyword}*")
        return text
    
    # Filtres Jinja2 spécialisés
    
    def _spotify_format(self, value: str) -> str:
        """Filtre de formatage spécifique Spotify."""
        # Formatage des noms d'artistes, tracks, etc.
        return value.title().replace("_", " ")
    
    def _ai_context_filter(self, value: str, ai_type: str = "music") -> str:
        """Filtre contextuel pour IA musicale."""
        context_mapping = {
            "music": "🎵",
            "recommendation": "🎯", 
            "generation": "🎼",
            "analysis": "📊"
        }
        
        emoji = context_mapping.get(ai_type, "🤖")
        return f"{emoji} {value}"
    
    def _tenant_format_filter(self, value: str, tenant_type: str = "artist") -> str:
        """Filtre de formatage tenant."""
        type_mapping = {
            "artist": "🎤",
            "label": "🏢",
            "studio": "🎚️",
            "platform": "🌐"
        }
        
        icon = type_mapping.get(tenant_type, "👤")
        return f"{icon} {value}"
    
    def get_supported_locales(self) -> Dict[str, LocaleConfig]:
        """Retourne la liste des locales supportées."""
        return self.SUPPORTED_LOCALES.copy()
    
    def reload_locale(self, locale: str) -> bool:
        """Recharge une locale spécifique."""
        try:
            # Suppression du cache
            cache_key = f"locale_data:{locale}"
            
            with self._cache_lock:
                if cache_key in self._cache:
                    del self._cache[cache_key]
            
            if self._enable_redis and self._redis_client:
                self._redis_client.delete(f"locale_cache:{cache_key}")
            
            # Rechargement
            self._get_locale_data(locale)
            
            logger.info(f"Locale rechargée avec succès", locale=locale)
            return True
            
        except Exception as e:
            logger.error(f"Erreur rechargement locale", locale=locale, error=str(e))
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        with self._cache_lock:
            cache_size = len(self._cache)
            expired_count = sum(1 for entry in self._cache.values() if entry.is_expired())
        
        return {
            'cache_size': cache_size,
            'expired_entries': expired_count,
            'max_cache_size': self.MAX_CACHE_SIZE,
            'redis_enabled': self._enable_redis,
            'stats': self._stats.copy()
        }
    
    def validate_locale_file(self, locale: str) -> Dict[str, Any]:
        """Valide la syntaxe d'un fichier de locale."""
        locale_file = Path(__file__).parent / "data" / f"{locale}.yml"
        
        result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'key_count': 0
        }
        
        if not locale_file.exists():
            result['errors'].append(f"Fichier de locale introuvable: {locale_file}")
            return result
        
        try:
            with open(locale_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
            if not isinstance(data, dict):
                result['errors'].append("Le fichier doit contenir un objet YAML")
                return result
            
            # Validation récursive des clés
            def validate_keys(obj, path=""):
                count = 0
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    if isinstance(value, dict):
                        count += validate_keys(value, current_path)
                    elif isinstance(value, str):
                        count += 1
                        # Validation template Jinja2
                        try:
                            self._jinja_env.from_string(value)
                        except Exception as e:
                            result['warnings'].append(
                                f"Template Jinja2 invalide à {current_path}: {e}"
                            )
                    else:
                        result['warnings'].append(
                            f"Type non supporté à {current_path}: {type(value)}"
                        )
                
                return count
            
            result['key_count'] = validate_keys(data)
            result['valid'] = len(result['errors']) == 0
            
        except Exception as e:
            result['errors'].append(f"Erreur parsing YAML: {e}")
        
        return result


# Instance globale (Singleton)
locale_manager = LocaleManager()
