"""
Tests pour le module cache.py du système Spleeter
"""

import pytest
import asyncio
import tempfile
import sqlite3
import pickle
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta

from spleeter.cache import (
    CacheEntry, CacheConfig, MemoryCache, DiskCache, 
    RedisCache, CacheManager
)
from spleeter.exceptions import CacheError


class TestCacheEntry:
    """Tests pour la classe CacheEntry"""
    
    def test_cache_entry_creation(self):
        """Test de création d'une entrée de cache"""
        data = {"test": "data"}
        entry = CacheEntry(
            key="test_key",
            data=data,
            size=100,
            ttl=3600
        )
        
        assert entry.key == "test_key"
        assert entry.data == data
        assert entry.size == 100
        assert entry.ttl == 3600
        assert entry.created_at is not None
        assert not entry.is_expired()
    
    def test_cache_entry_expiration(self):
        """Test d'expiration d'une entrée"""
        entry = CacheEntry(
            key="expiring_key",
            data="data",
            size=50,
            ttl=0.1  # 0.1 seconde
        )
        
        assert not entry.is_expired()
        
        # Simuler l'expiration
        entry.created_at = datetime.now() - timedelta(seconds=1)
        assert entry.is_expired()
    
    def test_cache_entry_no_ttl(self):
        """Test d'entrée sans TTL (ne devrait jamais expirer)"""
        entry = CacheEntry(
            key="permanent_key",
            data="data",
            size=50,
            ttl=None
        )
        
        # Même avec une ancienne date, ne devrait pas expirer
        entry.created_at = datetime.now() - timedelta(days=365)
        assert not entry.is_expired()
    
    def test_cache_entry_to_dict(self):
        """Test de conversion en dictionnaire"""
        entry = CacheEntry(
            key="dict_key",
            data={"nested": "data"},
            size=200,
            ttl=7200
        )
        
        entry_dict = entry.to_dict()
        
        assert entry_dict["key"] == "dict_key"
        assert entry_dict["data"] == {"nested": "data"}
        assert entry_dict["size"] == 200
        assert entry_dict["ttl"] == 7200
        assert "created_at" in entry_dict


class TestCacheConfig:
    """Tests pour la classe CacheConfig"""
    
    def test_default_config(self):
        """Test de la configuration par défaut"""
        config = CacheConfig()
        
        assert config.memory_size_mb == 512
        assert config.disk_size_mb == 2048
        assert config.default_ttl == 3600
        assert config.cleanup_interval == 300
        assert config.compression_enabled is True
        assert config.redis_url is None
    
    def test_custom_config(self):
        """Test de configuration personnalisée"""
        config = CacheConfig(
            memory_size_mb=1024,
            disk_size_mb=4096,
            default_ttl=7200,
            cleanup_interval=600,
            compression_enabled=False,
            redis_url="redis://localhost:6379/1"
        )
        
        assert config.memory_size_mb == 1024
        assert config.disk_size_mb == 4096
        assert config.default_ttl == 7200
        assert config.cleanup_interval == 600
        assert config.compression_enabled is False
        assert config.redis_url == "redis://localhost:6379/1"
    
    def test_memory_size_bytes(self):
        """Test de conversion en bytes"""
        config = CacheConfig(memory_size_mb=256)
        assert config.memory_size_bytes == 256 * 1024 * 1024
    
    def test_disk_size_bytes(self):
        """Test de conversion en bytes"""
        config = CacheConfig(disk_size_mb=1024)
        assert config.disk_size_bytes == 1024 * 1024 * 1024


class TestMemoryCache:
    """Tests pour la classe MemoryCache"""
    
    @pytest.fixture
    def memory_cache(self):
        """Fixture pour créer un cache mémoire"""
        config = CacheConfig(memory_size_mb=1)  # 1MB pour les tests
        return MemoryCache(config)
    
    @pytest.mark.asyncio
    async def test_memory_cache_basic_operations(self, memory_cache):
        """Test des opérations de base du cache mémoire"""
        # Test set/get
        await memory_cache.set("key1", "value1", ttl=3600)
        value = await memory_cache.get("key1")
        assert value == "value1"
        
        # Test inexistant
        value = await memory_cache.get("nonexistent")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_memory_cache_expiration(self, memory_cache):
        """Test d'expiration des entrées"""
        # Ajouter une entrée avec TTL court
        await memory_cache.set("expiring_key", "value", ttl=0.1)
        
        # Vérifier qu'elle existe
        value = await memory_cache.get("expiring_key")
        assert value == "value"
        
        # Attendre l'expiration
        await asyncio.sleep(0.2)
        
        # Vérifier qu'elle a expiré
        value = await memory_cache.get("expiring_key")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_memory_cache_lru_eviction(self, memory_cache):
        """Test d'éviction LRU"""
        # Remplir le cache avec des données qui dépassent la limite
        large_data = "x" * (100 * 1024)  # 100KB
        
        # Ajouter plusieurs entrées
        for i in range(15):  # Devrait dépasser 1MB
            await memory_cache.set(f"key_{i}", large_data)
        
        # Les premières clés devraient avoir été évincées
        value = await memory_cache.get("key_0")
        assert value is None
        
        # Les dernières clés devraient encore être là
        value = await memory_cache.get("key_14")
        assert value == large_data
    
    @pytest.mark.asyncio
    async def test_memory_cache_delete(self, memory_cache):
        """Test de suppression d'entrée"""
        await memory_cache.set("delete_me", "value")
        
        # Vérifier qu'elle existe
        value = await memory_cache.get("delete_me")
        assert value == "value"
        
        # Supprimer
        success = await memory_cache.delete("delete_me")
        assert success is True
        
        # Vérifier qu'elle n'existe plus
        value = await memory_cache.get("delete_me")
        assert value is None
        
        # Supprimer une clé inexistante
        success = await memory_cache.delete("nonexistent")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_memory_cache_clear(self, memory_cache):
        """Test de vidage du cache"""
        # Ajouter plusieurs entrées
        for i in range(5):
            await memory_cache.set(f"clear_key_{i}", f"value_{i}")
        
        # Vérifier qu'elles existent
        for i in range(5):
            value = await memory_cache.get(f"clear_key_{i}")
            assert value == f"value_{i}"
        
        # Vider le cache
        await memory_cache.clear()
        
        # Vérifier que toutes les entrées ont été supprimées
        for i in range(5):
            value = await memory_cache.get(f"clear_key_{i}")
            assert value is None
    
    @pytest.mark.asyncio
    async def test_memory_cache_stats(self, memory_cache):
        """Test des statistiques du cache"""
        stats = await memory_cache.get_stats()
        initial_requests = stats["total_requests"]
        
        # Effectuer quelques opérations
        await memory_cache.set("stats_key", "value")
        await memory_cache.get("stats_key")  # Hit
        await memory_cache.get("nonexistent")  # Miss
        
        stats = await memory_cache.get_stats()
        
        assert stats["total_requests"] == initial_requests + 2
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert stats["size"] >= 1
        assert "memory_usage" in stats
        assert "hit_rate" in stats
    
    @pytest.mark.asyncio
    async def test_memory_cache_cleanup(self, memory_cache):
        """Test du nettoyage automatique"""
        # Ajouter des entrées expirées
        await memory_cache.set("expired1", "value", ttl=0.1)
        await memory_cache.set("expired2", "value", ttl=0.1)
        await memory_cache.set("permanent", "value", ttl=None)
        
        # Attendre l'expiration
        await asyncio.sleep(0.2)
        
        # Déclencher le nettoyage
        cleaned = await memory_cache.cleanup()
        
        assert cleaned >= 2  # Au moins les 2 entrées expirées
        
        # Vérifier que l'entrée permanente existe toujours
        value = await memory_cache.get("permanent")
        assert value == "value"


class TestDiskCache:
    """Tests pour la classe DiskCache"""
    
    @pytest.fixture
    def disk_cache(self):
        """Fixture pour créer un cache disque"""
        config = CacheConfig(disk_size_mb=10)  # 10MB pour les tests
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DiskCache(config, cache_dir=temp_dir)
            yield cache
    
    @pytest.mark.asyncio
    async def test_disk_cache_basic_operations(self, disk_cache):
        """Test des opérations de base du cache disque"""
        # Test set/get
        await disk_cache.set("disk_key1", {"data": "value1"}, ttl=3600)
        value = await disk_cache.get("disk_key1")
        assert value == {"data": "value1"}
        
        # Test inexistant
        value = await disk_cache.get("nonexistent")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_disk_cache_persistence(self, disk_cache):
        """Test de persistance sur disque"""
        test_data = {"complex": {"nested": "data"}, "list": [1, 2, 3]}
        
        await disk_cache.set("persistent_key", test_data, ttl=3600)
        
        # Simuler une nouvelle instance (en réalité, les données devraient persister)
        value = await disk_cache.get("persistent_key")
        assert value == test_data
    
    @pytest.mark.asyncio
    async def test_disk_cache_compression(self, disk_cache):
        """Test de compression des données"""
        # Données répétitives qui se compriment bien
        repetitive_data = "A" * 10000
        
        await disk_cache.set("compressed_key", repetitive_data, ttl=3600)
        value = await disk_cache.get("compressed_key")
        
        assert value == repetitive_data
    
    @pytest.mark.asyncio
    async def test_disk_cache_large_data(self, disk_cache):
        """Test avec des données volumineuses"""
        large_data = {"big": "x" * (1024 * 1024)}  # 1MB
        
        await disk_cache.set("large_key", large_data, ttl=3600)
        value = await disk_cache.get("large_key")
        
        assert value == large_data
    
    @pytest.mark.asyncio
    async def test_disk_cache_metadata_operations(self, disk_cache):
        """Test des opérations sur les métadonnées"""
        await disk_cache.set("meta_key", "value", ttl=3600)
        
        # Test d'existence
        exists = await disk_cache.exists("meta_key")
        assert exists is True
        
        exists = await disk_cache.exists("nonexistent")
        assert exists is False
        
        # Test de suppression
        success = await disk_cache.delete("meta_key")
        assert success is True
        
        exists = await disk_cache.exists("meta_key")
        assert exists is False
    
    @pytest.mark.asyncio
    async def test_disk_cache_cleanup_expired(self, disk_cache):
        """Test de nettoyage des entrées expirées"""
        # Ajouter des entrées avec différents TTL
        await disk_cache.set("short_ttl", "value1", ttl=0.1)
        await disk_cache.set("long_ttl", "value2", ttl=3600)
        
        # Attendre l'expiration
        await asyncio.sleep(0.2)
        
        # Déclencher le nettoyage
        cleaned = await disk_cache.cleanup()
        assert cleaned >= 1
        
        # Vérifier les résultats
        value1 = await disk_cache.get("short_ttl")
        value2 = await disk_cache.get("long_ttl")
        
        assert value1 is None  # Expiré
        assert value2 == "value2"  # Toujours valide
    
    @pytest.mark.asyncio
    async def test_disk_cache_size_management(self, disk_cache):
        """Test de gestion de la taille du cache"""
        stats = await disk_cache.get_stats()
        initial_size = stats["disk_usage"]
        
        # Ajouter des données
        large_data = "x" * (100 * 1024)  # 100KB
        await disk_cache.set("size_test", large_data, ttl=3600)
        
        stats = await disk_cache.get_stats()
        assert stats["disk_usage"] > initial_size
        assert stats["total_entries"] >= 1


class TestRedisCache:
    """Tests pour la classe RedisCache"""
    
    @pytest.fixture
    def redis_cache(self):
        """Fixture pour créer un cache Redis (mocké)"""
        config = CacheConfig(redis_url="redis://localhost:6379/0")
        
        with patch('aioredis.from_url') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            
            cache = RedisCache(config)
            cache.redis = mock_redis_instance
            cache._connected = True
            
            yield cache, mock_redis_instance
    
    @pytest.mark.asyncio
    async def test_redis_cache_connection(self):
        """Test de connexion Redis"""
        config = CacheConfig(redis_url="redis://localhost:6379/0")
        
        with patch('aioredis.from_url') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.ping.return_value = True
            
            cache = RedisCache(config)
            await cache.connect()
            
            assert cache._connected is True
            mock_redis.assert_called_once_with("redis://localhost:6379/0")
            mock_redis_instance.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_redis_cache_basic_operations(self, redis_cache):
        """Test des opérations de base Redis"""
        cache, mock_redis = redis_cache
        
        # Mock des réponses Redis
        mock_redis.get.return_value = json.dumps("test_value").encode()
        mock_redis.set.return_value = True
        
        # Test set
        await cache.set("redis_key", "test_value", ttl=3600)
        mock_redis.setex.assert_called()
        
        # Test get
        value = await cache.get("redis_key")
        assert value == "test_value"
        mock_redis.get.assert_called_with("redis_key")
    
    @pytest.mark.asyncio
    async def test_redis_cache_complex_data(self, redis_cache):
        """Test avec des données complexes"""
        cache, mock_redis = redis_cache
        
        complex_data = {
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "number": 42.5
        }
        
        # Mock de la sérialisation/désérialisation
        mock_redis.get.return_value = json.dumps(complex_data).encode()
        mock_redis.setex.return_value = True
        
        await cache.set("complex_key", complex_data, ttl=3600)
        value = await cache.get("complex_key")
        
        assert value == complex_data
    
    @pytest.mark.asyncio
    async def test_redis_cache_delete(self, redis_cache):
        """Test de suppression Redis"""
        cache, mock_redis = redis_cache
        
        mock_redis.delete.return_value = 1  # 1 clé supprimée
        
        success = await cache.delete("delete_key")
        assert success is True
        mock_redis.delete.assert_called_with("delete_key")
        
        # Test suppression clé inexistante
        mock_redis.delete.return_value = 0
        success = await cache.delete("nonexistent")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_redis_cache_exists(self, redis_cache):
        """Test de vérification d'existence"""
        cache, mock_redis = redis_cache
        
        mock_redis.exists.return_value = 1
        exists = await cache.exists("existing_key")
        assert exists is True
        
        mock_redis.exists.return_value = 0
        exists = await cache.exists("nonexistent_key")
        assert exists is False
    
    @pytest.mark.asyncio
    async def test_redis_cache_error_handling(self, redis_cache):
        """Test de gestion d'erreurs Redis"""
        cache, mock_redis = redis_cache
        
        # Simuler une erreur de connexion
        mock_redis.get.side_effect = Exception("Connection error")
        
        with pytest.raises(CacheError):
            await cache.get("error_key")


class TestCacheManager:
    """Tests pour la classe CacheManager"""
    
    @pytest.fixture
    def cache_manager(self):
        """Fixture pour créer un gestionnaire de cache"""
        config = CacheConfig(
            memory_size_mb=1,
            disk_size_mb=10,
            redis_url=None  # Pas de Redis pour les tests
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheManager(config, cache_dir=temp_dir)
            yield manager
    
    @pytest.mark.asyncio
    async def test_cache_manager_initialization(self, cache_manager):
        """Test d'initialisation du gestionnaire"""
        await cache_manager.initialize()
        
        assert cache_manager.memory_cache is not None
        assert cache_manager.disk_cache is not None
        assert cache_manager.redis_cache is None  # Pas configuré
    
    @pytest.mark.asyncio
    async def test_cache_manager_auto_level_selection(self, cache_manager):
        """Test de sélection automatique du niveau de cache"""
        await cache_manager.initialize()
        
        # Petit objet -> mémoire
        small_data = "small"
        await cache_manager.set("small_key", small_data, cache_level="auto")
        
        # Vérifier qu'il est en mémoire
        memory_value = await cache_manager.memory_cache.get("small_key")
        assert memory_value == small_data
        
        # Gros objet -> disque
        large_data = "x" * (100 * 1024)  # 100KB
        await cache_manager.set("large_key", large_data, cache_level="auto")
        
        # Devrait aller sur disque (ou mémoire selon l'implémentation)
        value = await cache_manager.get("large_key")
        assert value == large_data
    
    @pytest.mark.asyncio
    async def test_cache_manager_specific_levels(self, cache_manager):
        """Test de sélection spécifique du niveau"""
        await cache_manager.initialize()
        
        data = "test_data"
        
        # Forcer en mémoire
        await cache_manager.set("memory_key", data, cache_level="memory")
        memory_value = await cache_manager.memory_cache.get("memory_key")
        assert memory_value == data
        
        # Forcer sur disque
        await cache_manager.set("disk_key", data, cache_level="disk")
        disk_value = await cache_manager.disk_cache.get("disk_key")
        assert disk_value == data
    
    @pytest.mark.asyncio
    async def test_cache_manager_fallback(self, cache_manager):
        """Test de fallback entre niveaux"""
        await cache_manager.initialize()
        
        # Mettre uniquement sur disque
        data = "fallback_test"
        await cache_manager.disk_cache.set("fallback_key", data, ttl=3600)
        
        # Le get général devrait le trouver
        value = await cache_manager.get("fallback_key")
        assert value == data
        
        # Et maintenant il devrait être promu en mémoire
        memory_value = await cache_manager.memory_cache.get("fallback_key")
        assert memory_value == data
    
    @pytest.mark.asyncio
    async def test_cache_manager_delete_all_levels(self, cache_manager):
        """Test de suppression sur tous les niveaux"""
        await cache_manager.initialize()
        
        data = "delete_test"
        key = "delete_all_key"
        
        # Mettre sur plusieurs niveaux
        await cache_manager.memory_cache.set(key, data, ttl=3600)
        await cache_manager.disk_cache.set(key, data, ttl=3600)
        
        # Supprimer
        success = await cache_manager.delete(key)
        assert success is True
        
        # Vérifier qu'il a été supprimé partout
        memory_value = await cache_manager.memory_cache.get(key)
        disk_value = await cache_manager.disk_cache.get(key)
        
        assert memory_value is None
        assert disk_value is None
    
    @pytest.mark.asyncio
    async def test_cache_manager_clear_all(self, cache_manager):
        """Test de vidage de tous les niveaux"""
        await cache_manager.initialize()
        
        # Ajouter des données sur plusieurs niveaux
        await cache_manager.set("mem_key", "mem_data", cache_level="memory")
        await cache_manager.set("disk_key", "disk_data", cache_level="disk")
        
        # Vider tout
        await cache_manager.clear_all()
        
        # Vérifier que tout est vide
        mem_value = await cache_manager.get("mem_key")
        disk_value = await cache_manager.get("disk_key")
        
        assert mem_value is None
        assert disk_value is None
    
    @pytest.mark.asyncio
    async def test_cache_manager_stats_aggregation(self, cache_manager):
        """Test d'agrégation des statistiques"""
        await cache_manager.initialize()
        
        # Ajouter des données pour générer des stats
        await cache_manager.set("stats_key1", "data1", cache_level="memory")
        await cache_manager.set("stats_key2", "data2", cache_level="disk")
        await cache_manager.get("stats_key1")
        await cache_manager.get("nonexistent")
        
        stats = await cache_manager.get_aggregated_stats()
        
        assert "memory" in stats
        assert "disk" in stats
        assert "total_hits" in stats
        assert "total_misses" in stats
        assert "total_requests" in stats
        assert "overall_hit_rate" in stats
    
    @pytest.mark.asyncio
    async def test_cache_manager_cleanup_all(self, cache_manager):
        """Test de nettoyage de tous les niveaux"""
        await cache_manager.initialize()
        
        # Ajouter des entrées expirées
        await cache_manager.set("exp1", "data1", ttl=0.1, cache_level="memory")
        await cache_manager.set("exp2", "data2", ttl=0.1, cache_level="disk")
        await cache_manager.set("perm", "data", ttl=None, cache_level="memory")
        
        # Attendre l'expiration
        await asyncio.sleep(0.2)
        
        # Nettoyer
        total_cleaned = await cache_manager.cleanup_all()
        assert total_cleaned >= 2
        
        # Vérifier que les données permanentes sont toujours là
        perm_value = await cache_manager.get("perm")
        assert perm_value == "data"
    
    @pytest.mark.asyncio
    async def test_cache_manager_cache_promotion(self, cache_manager):
        """Test de promotion de cache (disk -> memory)"""
        await cache_manager.initialize()
        
        # Mettre uniquement sur disque
        data = "promotion_test"
        await cache_manager.disk_cache.set("promote_key", data, ttl=3600)
        
        # Premier accès devrait promouvoir vers la mémoire
        value = await cache_manager.get("promote_key")
        assert value == data
        
        # Deuxième accès devrait venir de la mémoire
        memory_value = await cache_manager.memory_cache.get("promote_key")
        assert memory_value == data
    
    @pytest.mark.asyncio
    async def test_cache_manager_error_recovery(self, cache_manager):
        """Test de récupération d'erreur"""
        await cache_manager.initialize()
        
        # Simuler une erreur sur un niveau
        with patch.object(cache_manager.memory_cache, 'get', side_effect=Exception("Memory error")):
            # Mettre sur disque seulement
            await cache_manager.disk_cache.set("recovery_key", "recovery_data", ttl=3600)
            
            # Devrait récupérer depuis le disque malgré l'erreur mémoire
            value = await cache_manager.get("recovery_key")
            assert value == "recovery_data"
