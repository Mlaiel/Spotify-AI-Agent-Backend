"""
Gestionnaire de Locales Avancé pour Spotify AI Agent
Système de gestion centralisée et sécurisée des locales multi-tenant
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
import weakref
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from redis.asyncio import Redis
from pydantic import BaseModel, validator

logger = logging.getLogger(__name__)


@dataclass
class LocaleConfig:
    """Configuration avancée pour les locales"""
    default_locale: str = "en_US"
    supported_locales: Set[str] = field(default_factory=lambda: {
        "en_US", "fr_FR", "de_DE", "es_ES", "it_IT", "pt_BR", "ja_JP", "ko_KR", "zh_CN"
    })
    fallback_chain: List[str] = field(default_factory=lambda: ["en_US", "fr_FR"])
    cache_ttl: int = 3600
    preload_locales: Set[str] = field(default_factory=lambda: {"en_US", "fr_FR"})
    lazy_loading: bool = True
    compression_enabled: bool = True
    encryption_enabled: bool = True
    tenant_isolation: bool = True
    analytics_enabled: bool = True


@dataclass
class LocaleMetadata:
    """Métadonnées des locales"""
    locale_code: str
    display_name: str
    native_name: str
    language_code: str
    country_code: str
    script: Optional[str] = None
    variant: Optional[str] = None
    direction: str = "ltr"  # ltr ou rtl
    pluralization_rules: Dict[str, Any] = field(default_factory=dict)
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    number_format: Dict[str, str] = field(default_factory=dict)
    currency_code: str = "USD"
    encoding: str = "utf-8"
    collation: str = "unicode"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"


class LocaleProvider(ABC):
    """Interface abstraite pour les fournisseurs de locales"""
    
    @abstractmethod
    async def load_locale(self, locale_code: str, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Charge une locale"""
        pass
    
    @abstractmethod
    async def save_locale(self, locale_code: str, data: Dict[str, Any], tenant_id: Optional[str] = None) -> bool:
        """Sauvegarde une locale"""
        pass
    
    @abstractmethod
    async def delete_locale(self, locale_code: str, tenant_id: Optional[str] = None) -> bool:
        """Supprime une locale"""
        pass
    
    @abstractmethod
    async def list_locales(self, tenant_id: Optional[str] = None) -> List[str]:
        """Liste les locales disponibles"""
        pass


class FileSystemLocaleProvider(LocaleProvider):
    """Fournisseur de locales basé sur le système de fichiers"""
    
    def __init__(self, base_path: Path, tenant_aware: bool = True):
        self.base_path = base_path
        self.tenant_aware = tenant_aware
        self._file_cache = {}
        self._lock = threading.RLock()
    
    async def load_locale(self, locale_code: str, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Charge une locale depuis le système de fichiers"""
        try:
            file_path = self._get_locale_path(locale_code, tenant_id)
            
            with self._lock:
                cache_key = f"{tenant_id}:{locale_code}" if tenant_id else locale_code
                
                if cache_key in self._file_cache:
                    cached_data, cached_time = self._file_cache[cache_key]
                    if datetime.now() - cached_time < timedelta(minutes=10):
                        return cached_data
                
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    self._file_cache[cache_key] = (data, datetime.now())
                    return data
                
                logger.warning(f"Locale file not found: {file_path}")
                return {}
                
        except Exception as e:
            logger.error(f"Error loading locale {locale_code}: {e}")
            return {}
    
    async def save_locale(self, locale_code: str, data: Dict[str, Any], tenant_id: Optional[str] = None) -> bool:
        """Sauvegarde une locale sur le système de fichiers"""
        try:
            file_path = self._get_locale_path(locale_code, tenant_id)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Invalider le cache
            with self._lock:
                cache_key = f"{tenant_id}:{locale_code}" if tenant_id else locale_code
                self._file_cache.pop(cache_key, None)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving locale {locale_code}: {e}")
            return False
    
    async def delete_locale(self, locale_code: str, tenant_id: Optional[str] = None) -> bool:
        """Supprime une locale du système de fichiers"""
        try:
            file_path = self._get_locale_path(locale_code, tenant_id)
            
            if file_path.exists():
                file_path.unlink()
                
                # Invalider le cache
                with self._lock:
                    cache_key = f"{tenant_id}:{locale_code}" if tenant_id else locale_code
                    self._file_cache.pop(cache_key, None)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting locale {locale_code}: {e}")
            return False
    
    async def list_locales(self, tenant_id: Optional[str] = None) -> List[str]:
        """Liste les locales disponibles"""
        try:
            base_path = self.base_path
            if self.tenant_aware and tenant_id:
                base_path = base_path / "tenants" / tenant_id
            
            if not base_path.exists():
                return []
            
            locales = []
            for file_path in base_path.glob("*.json"):
                locale_code = file_path.stem
                locales.append(locale_code)
            
            return sorted(locales)
            
        except Exception as e:
            logger.error(f"Error listing locales: {e}")
            return []
    
    def _get_locale_path(self, locale_code: str, tenant_id: Optional[str] = None) -> Path:
        """Obtient le chemin du fichier de locale"""
        if self.tenant_aware and tenant_id:
            return self.base_path / "tenants" / tenant_id / f"{locale_code}.json"
        return self.base_path / f"{locale_code}.json"


class DatabaseLocaleProvider(LocaleProvider):
    """Fournisseur de locales basé sur la base de données"""
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
    
    async def load_locale(self, locale_code: str, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Charge une locale depuis la base de données"""
        try:
            # Implémentation de la requête SQL
            # Cette partie dépend du modèle de données exact
            pass
        except Exception as e:
            logger.error(f"Database error loading locale {locale_code}: {e}")
            return {}
    
    async def save_locale(self, locale_code: str, data: Dict[str, Any], tenant_id: Optional[str] = None) -> bool:
        """Sauvegarde une locale en base de données"""
        try:
            # Implémentation de la sauvegarde
            pass
        except Exception as e:
            logger.error(f"Database error saving locale {locale_code}: {e}")
            return False
    
    async def delete_locale(self, locale_code: str, tenant_id: Optional[str] = None) -> bool:
        """Supprime une locale de la base de données"""
        try:
            # Implémentation de la suppression
            pass
        except Exception as e:
            logger.error(f"Database error deleting locale {locale_code}: {e}")
            return False
    
    async def list_locales(self, tenant_id: Optional[str] = None) -> List[str]:
        """Liste les locales depuis la base de données"""
        try:
            # Implémentation de la liste
            pass
        except Exception as e:
            logger.error(f"Database error listing locales: {e}")
            return []


class LocaleManager:
    """Gestionnaire principal des locales avec support multi-tenant"""
    
    def __init__(
        self,
        config: LocaleConfig,
        providers: List[LocaleProvider],
        redis_client: Optional[Redis] = None
    ):
        self.config = config
        self.providers = providers
        self.redis_client = redis_client
        self._cache = {}
        self._metadata_cache = {}
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._observers = weakref.WeakSet()
        self._metrics = defaultdict(int)
        
        # Statistiques d'utilisation
        self._usage_stats = {
            'loads': defaultdict(int),
            'cache_hits': defaultdict(int),
            'cache_misses': defaultdict(int),
            'errors': defaultdict(int)
        }
    
    async def get_locale_data(
        self, 
        locale_code: str, 
        tenant_id: Optional[str] = None,
        keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Récupère les données de locale avec cache intelligent"""
        try:
            self._metrics['total_requests'] += 1
            cache_key = self._get_cache_key(locale_code, tenant_id)
            
            # Vérifier le cache Redis d'abord
            if self.redis_client:
                cached_data = await self._get_from_redis_cache(cache_key)
                if cached_data:
                    self._usage_stats['cache_hits'][locale_code] += 1
                    if keys:
                        return {k: cached_data.get(k) for k in keys if k in cached_data}
                    return cached_data
            
            # Vérifier le cache local
            async with self._lock:
                if cache_key in self._cache:
                    cached_data, timestamp = self._cache[cache_key]
                    if datetime.now() - timestamp < timedelta(seconds=self.config.cache_ttl):
                        self._usage_stats['cache_hits'][locale_code] += 1
                        if keys:
                            return {k: cached_data.get(k) for k in keys if k in cached_data}
                        return cached_data
            
            # Cache miss - charger depuis les providers
            self._usage_stats['cache_misses'][locale_code] += 1
            data = await self._load_from_providers(locale_code, tenant_id)
            
            if not data and locale_code != self.config.default_locale:
                # Fallback vers la locale par défaut
                for fallback_locale in self.config.fallback_chain:
                    if fallback_locale != locale_code:
                        data = await self._load_from_providers(fallback_locale, tenant_id)
                        if data:
                            logger.info(f"Using fallback locale {fallback_locale} for {locale_code}")
                            break
            
            if data:
                await self._cache_data(cache_key, data)
                self._usage_stats['loads'][locale_code] += 1
            
            if keys and data:
                return {k: data.get(k) for k in keys if k in data}
            
            return data or {}
            
        except Exception as e:
            self._usage_stats['errors'][locale_code] += 1
            logger.error(f"Error getting locale data for {locale_code}: {e}")
            return {}
    
    async def set_locale_data(
        self, 
        locale_code: str, 
        data: Dict[str, Any], 
        tenant_id: Optional[str] = None
    ) -> bool:
        """Met à jour les données de locale"""
        try:
            # Sauvegarder via les providers
            success = False
            for provider in self.providers:
                try:
                    if await provider.save_locale(locale_code, data, tenant_id):
                        success = True
                        break
                except Exception as e:
                    logger.warning(f"Provider {provider.__class__.__name__} failed: {e}")
            
            if success:
                # Invalider le cache
                cache_key = self._get_cache_key(locale_code, tenant_id)
                await self._invalidate_cache(cache_key)
                
                # Notifier les observateurs
                await self._notify_observers('locale_updated', {
                    'locale_code': locale_code,
                    'tenant_id': tenant_id,
                    'timestamp': datetime.now()
                })
            
            return success
            
        except Exception as e:
            logger.error(f"Error setting locale data for {locale_code}: {e}")
            return False
    
    async def delete_locale(self, locale_code: str, tenant_id: Optional[str] = None) -> bool:
        """Supprime une locale"""
        try:
            success = False
            for provider in self.providers:
                try:
                    if await provider.delete_locale(locale_code, tenant_id):
                        success = True
                        break
                except Exception as e:
                    logger.warning(f"Provider {provider.__class__.__name__} failed: {e}")
            
            if success:
                cache_key = self._get_cache_key(locale_code, tenant_id)
                await self._invalidate_cache(cache_key)
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting locale {locale_code}: {e}")
            return False
    
    async def list_available_locales(self, tenant_id: Optional[str] = None) -> List[str]:
        """Liste toutes les locales disponibles"""
        try:
            all_locales = set()
            
            for provider in self.providers:
                try:
                    locales = await provider.list_locales(tenant_id)
                    all_locales.update(locales)
                except Exception as e:
                    logger.warning(f"Provider {provider.__class__.__name__} failed: {e}")
            
            return sorted(list(all_locales))
            
        except Exception as e:
            logger.error(f"Error listing locales: {e}")
            return []
    
    async def preload_locales(self, tenant_id: Optional[str] = None):
        """Précharge les locales importantes"""
        try:
            preload_tasks = []
            for locale_code in self.config.preload_locales:
                task = self.get_locale_data(locale_code, tenant_id)
                preload_tasks.append(task)
            
            await asyncio.gather(*preload_tasks, return_exceptions=True)
            logger.info(f"Preloaded {len(self.config.preload_locales)} locales")
            
        except Exception as e:
            logger.error(f"Error preloading locales: {e}")
    
    async def get_usage_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques d'utilisation"""
        return {
            'usage_stats': dict(self._usage_stats),
            'metrics': dict(self._metrics),
            'cache_size': len(self._cache),
            'supported_locales': list(self.config.supported_locales),
            'preloaded_locales': list(self.config.preload_locales)
        }
    
    def add_observer(self, observer):
        """Ajoute un observateur pour les changements de locale"""
        self._observers.add(observer)
    
    def remove_observer(self, observer):
        """Supprime un observateur"""
        self._observers.discard(observer)
    
    async def _load_from_providers(self, locale_code: str, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Charge les données depuis les providers"""
        for provider in self.providers:
            try:
                data = await provider.load_locale(locale_code, tenant_id)
                if data:
                    return data
            except Exception as e:
                logger.warning(f"Provider {provider.__class__.__name__} failed: {e}")
        
        return {}
    
    async def _cache_data(self, cache_key: str, data: Dict[str, Any]):
        """Met en cache les données"""
        try:
            # Cache local
            async with self._lock:
                self._cache[cache_key] = (data, datetime.now())
            
            # Cache Redis
            if self.redis_client:
                await self._set_redis_cache(cache_key, data)
                
        except Exception as e:
            logger.warning(f"Error caching data: {e}")
    
    async def _invalidate_cache(self, cache_key: str):
        """Invalide le cache"""
        try:
            # Cache local
            async with self._lock:
                self._cache.pop(cache_key, None)
            
            # Cache Redis
            if self.redis_client:
                await self.redis_client.delete(f"locale:{cache_key}")
                
        except Exception as e:
            logger.warning(f"Error invalidating cache: {e}")
    
    async def _get_from_redis_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Récupère depuis le cache Redis"""
        try:
            data = await self.redis_client.get(f"locale:{cache_key}")
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Redis cache error: {e}")
        return None
    
    async def _set_redis_cache(self, cache_key: str, data: Dict[str, Any]):
        """Met en cache dans Redis"""
        try:
            await self.redis_client.setex(
                f"locale:{cache_key}",
                self.config.cache_ttl,
                json.dumps(data, ensure_ascii=False)
            )
        except Exception as e:
            logger.warning(f"Redis cache set error: {e}")
    
    def _get_cache_key(self, locale_code: str, tenant_id: Optional[str] = None) -> str:
        """Génère une clé de cache"""
        if tenant_id:
            return f"{tenant_id}:{locale_code}"
        return locale_code
    
    async def _notify_observers(self, event_type: str, data: Dict[str, Any]):
        """Notifie les observateurs"""
        try:
            for observer in list(self._observers):
                try:
                    if hasattr(observer, 'on_locale_event'):
                        await observer.on_locale_event(event_type, data)
                except Exception as e:
                    logger.warning(f"Observer notification error: {e}")
        except Exception as e:
            logger.error(f"Error notifying observers: {e}")


class TenantLocaleManager:
    """Gestionnaire de locales spécialisé pour les tenants"""
    
    def __init__(self, locale_manager: LocaleManager):
        self.locale_manager = locale_manager
        self._tenant_configs = {}
        self._tenant_locks = defaultdict(asyncio.Lock)
    
    async def get_tenant_locale_data(
        self, 
        tenant_id: str, 
        locale_code: str,
        keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Récupère les données de locale pour un tenant spécifique"""
        try:
            async with self._tenant_locks[tenant_id]:
                # Vérifier la configuration du tenant
                tenant_config = await self._get_tenant_config(tenant_id)
                
                # Vérifier si la locale est supportée par le tenant
                if locale_code not in tenant_config.get('supported_locales', set()):
                    locale_code = tenant_config.get('default_locale', 'en_US')
                
                return await self.locale_manager.get_locale_data(
                    locale_code, tenant_id, keys
                )
                
        except Exception as e:
            logger.error(f"Error getting tenant locale data: {e}")
            return {}
    
    async def set_tenant_locale_config(
        self, 
        tenant_id: str, 
        config: Dict[str, Any]
    ) -> bool:
        """Configure les locales pour un tenant"""
        try:
            async with self._tenant_locks[tenant_id]:
                self._tenant_configs[tenant_id] = config
                return True
                
        except Exception as e:
            logger.error(f"Error setting tenant locale config: {e}")
            return False
    
    async def _get_tenant_config(self, tenant_id: str) -> Dict[str, Any]:
        """Récupère la configuration du tenant"""
        return self._tenant_configs.get(tenant_id, {
            'default_locale': 'en_US',
            'supported_locales': {'en_US', 'fr_FR'},
            'fallback_chain': ['en_US']
        })
