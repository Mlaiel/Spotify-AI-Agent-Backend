"""
Tests Ultra-Avancés pour Cache Middleware Enterprise
================================================

Tests industriels complets avec patterns réels, benchmarks de performance,
tests de charge, failover, et validation ML pour le système de cache multi-niveaux.

Développé par l'équipe Test Engineering Expert sous la direction de Fahed Mlaiel.
Architecture: Enterprise Cache Testing Framework avec patterns industriels avancés.
"""

import pytest
import asyncio
import json
import time
import threading
import statistics
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import redis
import memcache
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc

# Import du middleware à tester
from app.api.middleware.cache_middleware import (
    AdvancedCacheManager,
    CacheMiddleware,
    CacheStrategy,
    CacheConfig,
    CacheMetrics,
    CacheInvalidator,
    CircuitBreaker,
    CompressionEngine,
    CacheCluster,
    create_cache_middleware,
    CacheLevel,
    CacheOperation,
    CacheType
)


# =============================================================================
# FIXTURES ENTERPRISE POUR CACHE TESTING
# =============================================================================

@pytest.fixture
def cache_config():
    """Configuration enterprise cache pour tests."""
    return CacheConfig(
        redis_url="redis://localhost:6379/1",
        memcached_servers=['localhost:11211'],
        default_ttl=3600,
        max_memory=100 * 1024 * 1024,  # 100MB
        compression_threshold=1024,
        circuit_breaker_threshold=5,
        circuit_breaker_timeout=30,
        metrics_enabled=True,
        cluster_enabled=False,
        cache_warming_enabled=True,
        background_cleanup=True
    )

@pytest.fixture
async def mock_redis():
    """Mock Redis avec comportement réaliste."""
    redis_mock = AsyncMock(spec=redis.Redis)
    
    # Simuler un stockage en mémoire
    storage = {}
    
    async def get_mock(key):
        data = storage.get(key)
        if data and data.get('expires_at', float('inf')) > time.time():
            return data['value']
        elif key in storage:
            del storage[key]
        return None
    
    async def set_mock(key, value, ex=None):
        expires_at = time.time() + ex if ex else float('inf')
        storage[key] = {'value': value, 'expires_at': expires_at}
        return True
    
    async def delete_mock(*keys):
        deleted = 0
        for key in keys:
            if key in storage:
                del storage[key]
                deleted += 1
        return deleted
    
    async def exists_mock(*keys):
        return sum(1 for key in keys if key in storage and 
                  storage[key].get('expires_at', float('inf')) > time.time())
    
    async def flushdb_mock():
        storage.clear()
        return True
    
    async def ping_mock():
        return True
    
    redis_mock.get = get_mock
    redis_mock.set = set_mock
    redis_mock.delete = delete_mock
    redis_mock.exists = exists_mock
    redis_mock.flushdb = flushdb_mock
    redis_mock.ping = ping_mock
    redis_mock.pipeline.return_value.__aenter__.return_value = redis_mock
    redis_mock.pipeline.return_value.__aexit__.return_value = None
    
    return redis_mock

@pytest.fixture
def mock_memcached():
    """Mock Memcached avec comportement réaliste."""
    mc_mock = Mock(spec=memcache.Client)
    
    storage = {}
    
    def get_mock(key):
        data = storage.get(key)
        if data and data.get('expires_at', float('inf')) > time.time():
            return data['value']
        elif key in storage:
            del storage[key]
        return None
    
    def set_mock(key, value, time_expire=0):
        expires_at = time.time() + time_expire if time_expire else float('inf')
        storage[key] = {'value': value, 'expires_at': expires_at}
        return True
    
    def delete_mock(key):
        return storage.pop(key, None) is not None
    
    def flush_all_mock():
        storage.clear()
        return True
    
    mc_mock.get = get_mock
    mc_mock.set = set_mock
    mc_mock.delete = delete_mock
    mc_mock.flush_all = flush_all_mock
    mc_mock.get_stats.return_value = [('localhost:11211', {'hits': 100, 'misses': 20})]
    
    return mc_mock

@pytest.fixture
async def cache_manager(cache_config, mock_redis, mock_memcached):
    """Gestionnaire de cache configuré pour tests."""
    with patch('redis.from_url', return_value=mock_redis), \
         patch('memcache.Client', return_value=mock_memcached):
        
        manager = AdvancedCacheManager(cache_config)
        await manager.initialize()
        yield manager
        await manager.cleanup()

@pytest.fixture
def sample_data():
    """Données d'exemple pour tests."""
    return {
        'user_id': 12345,
        'profile': {
            'name': 'Test User',
            'email': 'test@example.com',
            'preferences': ['rock', 'jazz', 'electronic'],
            'created_at': '2024-01-15T10:30:00Z'
        },
        'playlists': [
            {'id': 1, 'name': 'Favorites', 'tracks': 150},
            {'id': 2, 'name': 'Rock Classics', 'tracks': 75}
        ],
        'metadata': {
            'last_active': '2024-01-15T12:45:00Z',
            'session_id': 'sess_abcd1234'
        }
    }


# =============================================================================
# TESTS FONCTIONNELS ENTERPRISE
# =============================================================================

class TestAdvancedCacheManagerFunctionality:
    """Tests fonctionnels complets du gestionnaire de cache enterprise."""
    
    @pytest.mark.asyncio
    async def test_cache_manager_initialization(self, cache_config):
        """Test d'initialisation complète du gestionnaire."""
        manager = AdvancedCacheManager(cache_config)
        
        # Vérifier l'état initial
        assert manager.config == cache_config
        assert not manager.is_initialized
        
        # Initialiser avec mocks
        with patch('redis.from_url') as mock_redis_create, \
             patch('memcache.Client') as mock_mc_create:
            
            mock_redis = AsyncMock()
            mock_redis.ping.return_value = True
            mock_redis_create.return_value = mock_redis
            
            mock_mc = Mock()
            mock_mc.get_stats.return_value = [('localhost:11211', {})]
            mock_mc_create.return_value = mock_mc
            
            await manager.initialize()
            
            # Vérifier l'initialisation
            assert manager.is_initialized
            assert manager.redis_client is not None
            assert manager.memcached_client is not None
            assert manager.metrics is not None
            
            await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_multi_level_cache_operations(self, cache_manager, sample_data):
        """Test des opérations sur cache multi-niveaux."""
        key = "user:12345:profile"
        
        # Test SET - doit stocker dans tous les niveaux
        success = await cache_manager.set(key, sample_data, ttl=3600)
        assert success
        
        # Vérifier présence dans L1 (memory)
        l1_data = cache_manager.l1_cache.get(key)
        assert l1_data is not None
        assert l1_data['data'] == sample_data
        
        # Test GET - doit récupérer depuis le cache le plus rapide
        retrieved_data = await cache_manager.get(key)
        assert retrieved_data == sample_data
        
        # Test invalidation L1 pour forcer L2
        cache_manager.l1_cache.clear()
        retrieved_data = await cache_manager.get(key)
        assert retrieved_data == sample_data  # Récupéré depuis L2
        
        # Test DELETE - doit supprimer de tous les niveaux
        deleted = await cache_manager.delete(key)
        assert deleted
        
        # Vérifier suppression complète
        final_data = await cache_manager.get(key)
        assert final_data is None
    
    @pytest.mark.asyncio
    async def test_cache_compression_and_serialization(self, cache_manager):
        """Test compression et sérialisation avancées."""
        # Données volumineuses pour tester la compression
        large_data = [{
            'id': i,
            'data': 'x' * 1000,  # 1KB de données répétitives
            'timestamp': datetime.now().isoformat()
        } for i in range(100)]  # 100KB total
        key = "test:large_data"
        
        # Stocker avec compression automatique
        success = await cache_manager.set(key, large_data, ttl=3600)
        assert success
        
        # Vérifier que la compression a été appliquée
        compressed_size = len(cache_manager.compression_engine.compress(
            json.dumps(large_data).encode()
        ))
        original_size = len(json.dumps(large_data).encode())
        compression_ratio = compressed_size / original_size
        
        assert compression_ratio < 0.5  # Au moins 50% de compression
        
        # Récupérer et vérifier intégrité
        retrieved_data = await cache_manager.get(key)
        assert retrieved_data == large_data
    
    @pytest.mark.asyncio
    async def test_cache_strategies(self, cache_manager, sample_data):
        """Test des différentes stratégies de cache."""
        strategies = [
            CacheStrategy.LRU,
            CacheStrategy.LFU,
            CacheStrategy.FIFO,
            CacheStrategy.ADAPTIVE
        ]
        
        for strategy in strategies:
            cache_manager.config.strategy = strategy
            key = f"test:strategy:{strategy.value}"
            
            # Test stockage avec stratégie
            success = await cache_manager.set(key, sample_data, ttl=3600)
            assert success
            
            # Test récupération
            data = await cache_manager.get(key)
            assert data == sample_data
            
            # Nettoyer
            await cache_manager.delete(key)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, cache_manager):
        """Test du circuit breaker pour résistance aux pannes."""
        # Simuler des échecs Redis
        with patch.object(cache_manager.redis_client, 'get', 
                         side_effect=redis.ConnectionError("Connection failed")):
            
            # Premier échec - circuit fermé
            result = await cache_manager.get("test:key")
            assert result is None
            
            # Simuler plusieurs échecs pour ouvrir le circuit
            for _ in range(cache_manager.config.circuit_breaker_threshold):
                await cache_manager.get("test:key")
            
            # Circuit ouvert - doit retourner immédiatement
            start_time = time.time()
            result = await cache_manager.get("test:key")
            end_time = time.time()
            
            assert result is None
            assert (end_time - start_time) < 0.01  # Très rapide, pas d'appel Redis


# =============================================================================
# TESTS DE PERFORMANCE ET CHARGE
# =============================================================================

class TestCachePerformance:
    """Tests de performance et benchmarks pour le cache enterprise."""
    
    @pytest.mark.asyncio
    async def test_cache_throughput_benchmark(self, cache_manager):
        """Benchmark de débit du cache sous charge."""
        num_operations = 1000
        num_concurrent = 50
        
        # Préparer les données de test
        test_data = {f"key_{i}": f"value_{i}" * 100 for i in range(num_operations)}
        
        async def write_operation(items):
            """Opération d'écriture concurrente."""
            start_time = time.time()
            tasks = []
            for key, value in items:
                task = cache_manager.set(key, value, ttl=3600)
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            return time.time() - start_time
        
        async def read_operation(keys):
            """Opération de lecture concurrente."""
            start_time = time.time()
            tasks = []
            for key in keys:
                task = cache_manager.get(key)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return time.time() - start_time, results
        
        # Test d'écriture
        items = list(test_data.items())
        chunks = [items[i:i + num_operations//num_concurrent] 
                 for i in range(0, len(items), num_operations//num_concurrent)]
        
        write_start = time.time()
        write_tasks = [write_operation(chunk) for chunk in chunks]
        write_times = await asyncio.gather(*write_tasks)
        total_write_time = time.time() - write_start
        
        # Test de lecture
        keys = list(test_data.keys())
        key_chunks = [keys[i:i + num_operations//num_concurrent] 
                     for i in range(0, len(keys), num_operations//num_concurrent)]
        
        read_start = time.time()
        read_tasks = [read_operation(chunk) for chunk in key_chunks]
        read_results = await asyncio.gather(*read_tasks)
        total_read_time = time.time() - read_start
        
        # Calculer les métriques
        write_throughput = num_operations / total_write_time
        read_throughput = num_operations / total_read_time
        
        # Assertions de performance
        assert write_throughput > 500  # Min 500 ops/sec en écriture
        assert read_throughput > 1000  # Min 1000 ops/sec en lecture
        
        # Vérifier l'intégrité des données
        for _, (_, results) in read_results:
            assert all(result is not None for result in results)
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, cache_manager, sample_data):
        """Test d'efficacité mémoire et détection de fuites."""
        initial_memory = psutil.Process().memory_info().rss
        
        # Stocker beaucoup de données
        for i in range(1000):
            key = f"memory_test:{i}"
            data = {**sample_data, 'iteration': i}
            await cache_manager.set(key, data, ttl=3600)
        
        peak_memory = psutil.Process().memory_info().rss
        memory_increase = peak_memory - initial_memory
        
        # Nettoyer le cache
        await cache_manager.flush_all()
        gc.collect()  # Force garbage collection
        
        final_memory = psutil.Process().memory_info().rss
        memory_released = peak_memory - final_memory
        
        # Vérifications d'efficacité
        assert memory_increase < 50 * 1024 * 1024  # Max 50MB d'augmentation
        assert memory_released > memory_increase * 0.8  # Au moins 80% libéré
    
    @pytest.mark.asyncio
    async def test_cache_hit_ratio_optimization(self, cache_manager):
        """Test d'optimisation du taux de hit du cache."""
        # Simulation de patterns d'accès réalistes
        keys = [f"user:{i}" for i in range(100)]
        values = [f"data_{i}" for i in range(100)]
        
        # Phase 1: Populate cache
        for key, value in zip(keys, values):
            await cache_manager.set(key, value, ttl=3600)
        
        # Phase 2: Simulate access patterns (Zipf distribution)
        access_counts = {}
        total_accesses = 1000
        
        for _ in range(total_accesses):
            # Distribution Zipf - 80/20 rule
            if np.random.random() < 0.8:
                # Top 20% des clés
                key_idx = np.random.randint(0, 20)
            else:
                # Remaining 80% des clés
                key_idx = np.random.randint(20, 100)
            
            key = keys[key_idx]
            access_counts[key] = access_counts.get(key, 0) + 1
            
            result = await cache_manager.get(key)
            assert result is not None
        
        # Calculer le hit ratio
        metrics = cache_manager.get_metrics()
        hit_ratio = metrics.get('hit_ratio', 0)
        
        # Assertions de performance
        assert hit_ratio > 0.95  # Au moins 95% de hit ratio
        
        # Vérifier la distribution des accès
        most_accessed = max(access_counts.values())
        least_accessed = min(access_counts.values())
        assert most_accessed > least_accessed * 5  # Distribution claire


# =============================================================================
# TESTS DE RESILIENCE ET FAILOVER
# =============================================================================

class TestCacheResilience:
    """Tests de résilience et failover pour cache enterprise."""
    
    @pytest.mark.asyncio
    async def test_redis_failover_to_memcached(self, cache_manager, sample_data):
        """Test de basculement Redis vers Memcached."""
        key = "failover:test"
        
        # Stocker initialement dans Redis
        await cache_manager.set(key, sample_data, ttl=3600)
        
        # Vérifier que c'est dans Redis
        data = await cache_manager.get(key)
        assert data == sample_data
        
        # Simuler une panne Redis
        with patch.object(cache_manager.redis_client, 'get',
                         side_effect=redis.ConnectionError("Redis down")):
            
            # Doit basculer vers Memcached (L3)
            fallback_data = await cache_manager.get(key)
            assert fallback_data == sample_data
            
            # Vérifier que le circuit breaker s'active
            assert cache_manager.circuit_breaker.is_open()
    
    @pytest.mark.asyncio
    async def test_partial_cache_failure(self, cache_manager, sample_data):
        """Test de panne partielle du cache."""
        keys = [f"partial:test:{i}" for i in range(10)]
        
        # Stocker les données
        for key in keys:
            await cache_manager.set(key, sample_data, ttl=3600)
        
        # Simuler une panne intermittente
        call_count = 0
        original_get = cache_manager.redis_client.get
        
        async def failing_get(key):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Échec 1 fois sur 3
                raise redis.ConnectionError("Intermittent failure")
            return await original_get(key)
        
        with patch.object(cache_manager.redis_client, 'get', side_effect=failing_get):
            successful_gets = 0
            for key in keys:
                try:
                    data = await cache_manager.get(key)
                    if data is not None:
                        successful_gets += 1
                except Exception:
                    pass
            
            # Doit récupérer au moins 70% des données
            assert successful_gets >= 7
    
    @pytest.mark.asyncio
    async def test_cache_recovery_after_failure(self, cache_manager, sample_data):
        """Test de récupération après panne."""
        key = "recovery:test"
        
        # Simuler une panne puis récupération
        with patch.object(cache_manager.redis_client, 'get',
                         side_effect=redis.ConnectionError("Temporary failure")):
            
            # Panne - circuit breaker ouvert
            result = await cache_manager.get(key)
            assert result is None
            assert cache_manager.circuit_breaker.is_open()
        
        # Attendre la récupération du circuit breaker
        await asyncio.sleep(0.1)
        
        # Test de récupération
        with patch.object(cache_manager.redis_client, 'get',
                         return_value=json.dumps(sample_data).encode()):
            
            # Forcer la fermeture du circuit breaker
            cache_manager.circuit_breaker.reset()
            
            # Doit fonctionner normalement
            result = await cache_manager.get(key)
            assert result == sample_data
            assert not cache_manager.circuit_breaker.is_open()


# =============================================================================
# TESTS D'INTÉGRATION AVEC ML
# =============================================================================

class TestCacheMLIntegration:
    """Tests d'intégration avec les fonctionnalités ML."""
    
    @pytest.mark.asyncio
    async def test_adaptive_ttl_prediction(self, cache_manager):
        """Test de prédiction adaptive du TTL basée sur ML."""
        # Simuler des patterns d'accès historiques
        access_patterns = {
            'user:morning_user': [8, 9, 10, 11, 12],  # Actif le matin
            'user:evening_user': [18, 19, 20, 21, 22],  # Actif le soir
            'user:random_user': list(range(24))  # Accès aléatoire
        }
        
        for user_key, hours in access_patterns.items():
            # Simuler l'historique d'accès
            for hour in hours:
                access_time = datetime.now().replace(hour=hour % 24)
                await cache_manager._record_access_pattern(user_key, access_time)
        
        # Test de prédiction TTL
        morning_ttl = await cache_manager._predict_optimal_ttl('user:morning_user')
        evening_ttl = await cache_manager._predict_optimal_ttl('user:evening_user')
        random_ttl = await cache_manager._predict_optimal_ttl('user:random_user')
        
        # Les utilisateurs avec patterns spécifiques doivent avoir des TTL optimisés
        assert morning_ttl != random_ttl
        assert evening_ttl != random_ttl
        
        # TTL doit être raisonnable (entre 1h et 24h)
        for ttl in [morning_ttl, evening_ttl, random_ttl]:
            assert 3600 <= ttl <= 86400
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_in_cache_patterns(self, cache_manager):
        """Test de détection d'anomalies dans les patterns de cache."""
        # Créer un pattern normal
        normal_keys = [f"normal:user:{i}" for i in range(100)]
        for key in normal_keys:
            await cache_manager.set(key, {"type": "normal"}, ttl=3600)
            await cache_manager.get(key)  # Accès normal
        
        # Créer un pattern anormal (accès massif soudain)
        anomaly_keys = [f"anomaly:user:{i}" for i in range(1000)]
        start_time = time.time()
        
        for key in anomaly_keys:
            await cache_manager.set(key, {"type": "anomaly"}, ttl=3600)
            for _ in range(10):  # Accès répétés anormaux
                await cache_manager.get(key)
        
        anomaly_duration = time.time() - start_time
        
        # Analyser les métriques pour détecter l'anomalie
        metrics = cache_manager.get_metrics()
        recent_qps = metrics.get('queries_per_second', 0)
        
        # Doit détecter un pic anormal de QPS
        assert recent_qps > 1000  # QPS anormalement élevé
        
        # Le système doit s'adapter automatiquement
        adaptive_metrics = await cache_manager._analyze_performance_anomalies()
        assert 'high_qps_detected' in adaptive_metrics
        assert adaptive_metrics['recommended_action'] == 'scale_cache'


# =============================================================================
# TESTS DE CONFORMITÉ ET SÉCURITÉ
# =============================================================================

class TestCacheCompliance:
    """Tests de conformité et sécurité pour cache enterprise."""
    
    @pytest.mark.asyncio
    async def test_data_encryption_at_rest(self, cache_manager):
        """Test de chiffrement des données au repos."""
        sensitive_data = {
            'user_id': '12345',
            'email': 'user@example.com',
            'credit_card': '4111-1111-1111-1111',
            'ssn': '123-45-6789'
        }
        
        # Marquer comme données sensibles
        key = "sensitive:user:12345"
        await cache_manager.set(key, sensitive_data, ttl=3600, encrypt=True)
        
        # Vérifier que les données sont chiffrées en stockage
        raw_redis_data = await cache_manager.redis_client.get(key)
        assert raw_redis_data is not None
        
        # Les données brutes ne doivent pas contenir de texte en clair
        raw_str = raw_redis_data.decode() if isinstance(raw_redis_data, bytes) else str(raw_redis_data)
        assert '4111-1111-1111-1111' not in raw_str
        assert 'user@example.com' not in raw_str
        assert '123-45-6789' not in raw_str
        
        # Récupération normale doit déchiffrer automatiquement
        decrypted_data = await cache_manager.get(key)
        assert decrypted_data == sensitive_data
    
    @pytest.mark.asyncio
    async def test_pii_data_handling(self, cache_manager):
        """Test de gestion des données personnelles (PII)."""
        pii_data = {
            'first_name': 'John',
            'last_name': 'Doe',
            'birth_date': '1990-01-15',
            'address': '123 Main St, City, State',
            'phone': '+1-555-123-4567'
        }
        
        # Stocker avec marquage PII
        key = "pii:user:john_doe"
        await cache_manager.set(key, pii_data, ttl=3600, pii=True)
        
        # Vérifier l'audit trail
        audit_records = cache_manager.get_audit_records(key)
        assert len(audit_records) > 0
        assert audit_records[0]['action'] == 'PII_STORED'
        assert audit_records[0]['compliance'] == 'GDPR'
        
        # Test de suppression conforme GDPR
        await cache_manager.delete_pii_data(key, reason="user_request")
        
        # Vérifier suppression complète
        data = await cache_manager.get(key)
        assert data is None
        
        # Vérifier l'audit de suppression
        delete_audit = cache_manager.get_audit_records(key)
        assert any(record['action'] == 'PII_DELETED' for record in delete_audit)
    
    @pytest.mark.asyncio
    async def test_cache_access_logging(self, cache_manager, sample_data):
        """Test de logging des accès au cache."""
        key = "audit:test"
        
        # Opérations avec logging
        await cache_manager.set(key, sample_data, ttl=3600)
        await cache_manager.get(key)
        await cache_manager.get(key)  # Second accès
        await cache_manager.delete(key)
        
        # Vérifier les logs d'accès
        access_logs = cache_manager.get_access_logs(key)
        
        assert len(access_logs) >= 4  # Set, Get x2, Delete
        
        # Vérifier le contenu des logs
        set_log = next(log for log in access_logs if log['operation'] == 'SET')
        assert set_log['key'] == key
        assert 'timestamp' in set_log
        assert 'user_agent' in set_log or 'source_ip' in set_log
        
        get_logs = [log for log in access_logs if log['operation'] == 'GET']
        assert len(get_logs) >= 2
        
        delete_log = next(log for log in access_logs if log['operation'] == 'DELETE')
        assert delete_log['key'] == key


# =============================================================================
# TESTS DE BENCHMARK ET REGRESSION
# =============================================================================

class TestCacheBenchmarks:
    """Tests de benchmark et détection de régression."""
    
    @pytest.mark.asyncio
    async def test_performance_regression_detection(self, cache_manager):
        """Test de détection de régression de performance."""
        # Baseline performance
        baseline_operations = 100
        baseline_start = time.time()
        
        for i in range(baseline_operations):
            await cache_manager.set(f"baseline:{i}", f"value_{i}", ttl=3600)
            await cache_manager.get(f"baseline:{i}")
        
        baseline_time = time.time() - baseline_start
        baseline_ops_per_sec = baseline_operations / baseline_time
        
        # Test current performance
        test_operations = 100
        test_start = time.time()
        
        for i in range(test_operations):
            await cache_manager.set(f"test:{i}", f"value_{i}", ttl=3600)
            await cache_manager.get(f"test:{i}")
        
        test_time = time.time() - test_start
        test_ops_per_sec = test_operations / test_time
        
        # Détection de régression (tolérance de 20%)
        performance_ratio = test_ops_per_sec / baseline_ops_per_sec
        assert performance_ratio >= 0.8, f"Performance regression detected: {performance_ratio}"
        
        # Log performance metrics
        print(f"Baseline: {baseline_ops_per_sec:.2f} ops/sec")
        print(f"Current: {test_ops_per_sec:.2f} ops/sec")
        print(f"Ratio: {performance_ratio:.2f}")
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, cache_manager):
        """Test de détection de fuites mémoire."""
        initial_memory = psutil.Process().memory_info().rss
        
        # Cycle de créations/suppressions
        for cycle in range(10):
            # Créer beaucoup d'objets
            for i in range(100):
                key = f"memory_cycle:{cycle}:{i}"
                data = {'cycle': cycle, 'data': 'x' * 1000}
                await cache_manager.set(key, data, ttl=10)
            
            # Nettoyer
            await cache_manager.flush_all()
            gc.collect()
            
            # Vérifier la mémoire
            current_memory = psutil.Process().memory_info().rss
            memory_growth = current_memory - initial_memory
            
            # Pas plus de 10MB de croissance par cycle
            assert memory_growth < 10 * 1024 * 1024
        
        final_memory = psutil.Process().memory_info().rss
        total_growth = final_memory - initial_memory
        
        # Croissance totale acceptable
        assert total_growth < 50 * 1024 * 1024  # Max 50MB au total
    
    def test_cache_configuration_validation(self, cache_config):
        """Test de validation de configuration enterprise."""
        # Configuration valide
        manager = AdvancedCacheManager(cache_config)
        assert manager.config.redis_url is not None
        assert manager.config.default_ttl > 0
        assert manager.config.max_memory > 0
        
        # Configuration invalide
        invalid_config = CacheConfig(
            redis_url="",  # URL vide
            default_ttl=-1,  # TTL négatif
            max_memory=0  # Mémoire nulle
        )
        
        with pytest.raises(ValueError):
            AdvancedCacheManager(invalid_config)


# =============================================================================
# TESTS D'INTEGRATION COMPLETE
# =============================================================================

@pytest.mark.integration
class TestCacheIntegrationComplete:
    """Tests d'intégration complète du système de cache."""
    
    @pytest.mark.asyncio
    async def test_full_cache_workflow(self, cache_config):
        """Test de workflow complet enterprise."""
        # Initialisation
        with patch('redis.from_url') as mock_redis_create, \
             patch('memcache.Client') as mock_mc_create:
            
            # Configuration des mocks
            mock_redis = AsyncMock()
            mock_redis.ping.return_value = True
            mock_redis_create.return_value = mock_redis
            
            mock_mc = Mock()
            mock_mc.get_stats.return_value = [('localhost:11211', {})]
            mock_mc_create.return_value = mock_mc
            
            # Workflow complet
            manager = AdvancedCacheManager(cache_config)
            await manager.initialize()
            
            try:
                # 1. Stockage de données complexes
                user_data = {
                    'id': 12345,
                    'profile': {'name': 'Test User'},
                    'settings': {'theme': 'dark'},
                    'activity': [{'action': 'play', 'timestamp': time.time()}]
                }
                
                success = await manager.set('user:12345', user_data, ttl=3600)
                assert success
                
                # 2. Récupération et validation
                retrieved = await manager.get('user:12345')
                assert retrieved == user_data
                
                # 3. Mise à jour partielle
                user_data['settings']['theme'] = 'light'
                await manager.set('user:12345', user_data, ttl=3600)
                
                updated = await manager.get('user:12345')
                assert updated['settings']['theme'] == 'light'
                
                # 4. Operations en lot
                batch_data = {f'batch:{i}': f'value_{i}' for i in range(10)}
                await manager.set_multiple(batch_data, ttl=3600)
                
                batch_retrieved = await manager.get_multiple(list(batch_data.keys()))
                assert len(batch_retrieved) == len(batch_data)
                
                # 5. Nettoyage et métriques
                await manager.flush_all()
                metrics = manager.get_metrics()
                
                assert 'total_operations' in metrics
                assert 'hit_ratio' in metrics
                assert 'memory_usage' in metrics
                
            finally:
                await manager.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
