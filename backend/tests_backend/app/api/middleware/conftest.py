"""
Configuration Globale des Tests Enterprise pour Middleware
========================================================

Configuration centralisée pour tous les tests de middleware avec 
patterns enterprise, fixtures avancées, et utilitaires de test.

Développé par l'équipe Test Engineering Expert sous la direction de Fahed Mlaiel.
Architecture: Enterprise Testing Framework avec configuration modulaire.
"""

import pytest
import asyncio
import os
import sys
import time
import json
import logging
from typing import Dict, Any, List, Optional, Generator
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import tempfile
import threading
from pathlib import Path


# =============================================================================
# CONFIGURATION GLOBALE DES TESTS
# =============================================================================

# Configuration des timeouts
TEST_TIMEOUTS = {
    'unit_test': 10,          # 10 secondes pour tests unitaires
    'integration_test': 30,   # 30 secondes pour tests d'intégration
    'performance_test': 60,   # 1 minute pour tests de performance
    'load_test': 300,         # 5 minutes pour tests de charge
    'stress_test': 600        # 10 minutes pour tests de stress
}

# Configuration des seuils de performance
PERFORMANCE_THRESHOLDS = {
    'response_time_ms': {
        'excellent': 50,
        'good': 200,
        'acceptable': 500,
        'poor': 1000
    },
    'memory_usage_mb': {
        'excellent': 50,
        'good': 100,
        'acceptable': 200,
        'poor': 500
    },
    'cpu_usage_percent': {
        'excellent': 20,
        'good': 50,
        'acceptable': 70,
        'poor': 85
    },
    'throughput_qps': {
        'excellent': 1000,
        'good': 500,
        'acceptable': 200,
        'poor': 100
    }
}

# Configuration des mocks par défaut
DEFAULT_MOCKS = {
    'redis': {
        'enabled': True,
        'mock_responses': {
            'get': None,
            'set': True,
            'delete': True,
            'exists': False,
            'ttl': -1
        }
    },
    'prometheus': {
        'enabled': True,
        'mock_metrics': True
    },
    'database': {
        'enabled': True,
        'mock_queries': True
    },
    'external_apis': {
        'enabled': True,
        'response_delay_ms': 100
    }
}


# =============================================================================
# FIXTURES ENTERPRISE
# =============================================================================

@pytest.fixture(scope="session")
def test_environment():
    """Configuration de l'environnement de test."""
    return {
        'name': 'test',
        'debug': True,
        'log_level': 'DEBUG',
        'mock_external_services': True,
        'performance_monitoring': True,
        'test_data_isolation': True
    }

@pytest.fixture(scope="session")
def mock_redis():
    """Mock Redis configuré pour les tests."""
    class MockRedis:
        def __init__(self):
            self._data = {}
            self._ttl = {}
        
        def get(self, key):
            """Simulation get Redis."""
            if key in self._ttl:
                if time.time() > self._ttl[key]:
                    del self._data[key]
                    del self._ttl[key]
                    return None
            return self._data.get(key)
        
        def set(self, key, value, ex=None):
            """Simulation set Redis."""
            self._data[key] = value
            if ex:
                self._ttl[key] = time.time() + ex
            return True
        
        def delete(self, key):
            """Simulation delete Redis."""
            if key in self._data:
                del self._data[key]
            if key in self._ttl:
                del self._ttl[key]
            return True
        
        def exists(self, key):
            """Simulation exists Redis."""
            return key in self._data
        
        def ttl(self, key):
            """Simulation TTL Redis."""
            if key in self._ttl:
                remaining = self._ttl[key] - time.time()
                return max(0, int(remaining))
            return -1
        
        def flushall(self):
            """Vider toutes les données."""
            self._data.clear()
            self._ttl.clear()
        
        def pipeline(self):
            """Simulation pipeline Redis."""
            return MockRedisPipeline(self)
    
    class MockRedisPipeline:
        def __init__(self, redis_instance):
            self.redis = redis_instance
            self.commands = []
        
        def set(self, key, value, ex=None):
            self.commands.append(('set', key, value, ex))
            return self
        
        def get(self, key):
            self.commands.append(('get', key))
            return self
        
        def execute(self):
            results = []
            for cmd in self.commands:
                if cmd[0] == 'set':
                    result = self.redis.set(cmd[1], cmd[2], cmd[3] if len(cmd) > 3 else None)
                elif cmd[0] == 'get':
                    result = self.redis.get(cmd[1])
                else:
                    result = None
                results.append(result)
            self.commands.clear()
            return results
    
    return MockRedis()

@pytest.fixture(scope="function")
def mock_prometheus_metrics():
    """Mock Prometheus metrics pour les tests."""
    class MockPrometheusMetrics:
        def __init__(self):
            self.counters = {}
            self.histograms = {}
            self.gauges = {}
        
        def counter(self, name, description="", labelnames=None):
            """Mock Counter Prometheus."""
            class MockCounter:
                def __init__(self, metrics_instance, metric_name):
                    self.metrics = metrics_instance
                    self.name = metric_name
                    self.metrics.counters[metric_name] = 0
                
                def inc(self, amount=1):
                    self.metrics.counters[self.name] += amount
                
                def labels(self, **kwargs):
                    return self
            
            return MockCounter(self, name)
        
        def histogram(self, name, description="", labelnames=None):
            """Mock Histogram Prometheus."""
            class MockHistogram:
                def __init__(self, metrics_instance, metric_name):
                    self.metrics = metrics_instance
                    self.name = metric_name
                    self.metrics.histograms[metric_name] = []
                
                def observe(self, value):
                    self.metrics.histograms[self.name].append(value)
                
                def labels(self, **kwargs):
                    return self
                
                def time(self):
                    """Context manager pour mesurer le temps."""
                    return MockTimer(self)
            
            return MockHistogram(self, name)
        
        def gauge(self, name, description="", labelnames=None):
            """Mock Gauge Prometheus."""
            class MockGauge:
                def __init__(self, metrics_instance, metric_name):
                    self.metrics = metrics_instance
                    self.name = metric_name
                    self.metrics.gauges[metric_name] = 0
                
                def set(self, value):
                    self.metrics.gauges[self.name] = value
                
                def inc(self, amount=1):
                    self.metrics.gauges[self.name] += amount
                
                def dec(self, amount=1):
                    self.metrics.gauges[self.name] -= amount
                
                def labels(self, **kwargs):
                    return self
            
            return MockGauge(self, name)
        
        def get_sample_value(self, metric_name):
            """Récupère la valeur d'une métrique."""
            if metric_name in self.counters:
                return self.counters[metric_name]
            elif metric_name in self.gauges:
                return self.gauges[metric_name]
            elif metric_name in self.histograms:
                values = self.histograms[metric_name]
                return len(values)  # Nombre d'observations
            return None
    
    class MockTimer:
        def __init__(self, histogram):
            self.histogram = histogram
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.perf_counter()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time:
                duration = time.perf_counter() - self.start_time
                self.histogram.observe(duration)
    
    return MockPrometheusMetrics()

@pytest.fixture(scope="function")
def mock_database():
    """Mock base de données pour les tests."""
    class MockDatabase:
        def __init__(self):
            self.tables = {}
            self.queries_executed = []
            self.transaction_active = False
        
        def execute_query(self, query: str, params: Optional[List] = None):
            """Simulation d'exécution de requête."""
            self.queries_executed.append({
                'query': query,
                'params': params,
                'timestamp': datetime.now(),
                'execution_time_ms': 10 + (len(query) * 0.1)  # Simulation temps
            })
            
            # Simulation de résultats basés sur le type de requête
            if query.lower().startswith('select'):
                return [{'id': 1, 'name': 'test_record'}]
            elif query.lower().startswith('insert'):
                return {'rows_affected': 1, 'last_insert_id': 123}
            elif query.lower().startswith('update'):
                return {'rows_affected': 1}
            elif query.lower().startswith('delete'):
                return {'rows_affected': 1}
            else:
                return {'status': 'success'}
        
        def begin_transaction(self):
            """Démarrer une transaction."""
            self.transaction_active = True
        
        def commit_transaction(self):
            """Valider une transaction."""
            self.transaction_active = False
        
        def rollback_transaction(self):
            """Annuler une transaction."""
            self.transaction_active = False
        
        def get_query_stats(self):
            """Statistiques des requêtes."""
            return {
                'total_queries': len(self.queries_executed),
                'avg_execution_time': sum(q['execution_time_ms'] for q in self.queries_executed) / len(self.queries_executed) if self.queries_executed else 0,
                'queries_by_type': {}
            }
    
    return MockDatabase()

@pytest.fixture(scope="function")
def performance_monitor():
    """Monitor de performance pour les tests."""
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = {}
            self.start_times = {}
        
        def start_measurement(self, test_name: str):
            """Démarre la mesure de performance."""
            self.start_times[test_name] = {
                'wall_time': time.perf_counter(),
                'cpu_time': time.process_time(),
                'memory_start': self._get_memory_usage()
            }
        
        def end_measurement(self, test_name: str):
            """Termine la mesure de performance."""
            if test_name not in self.start_times:
                return None
            
            start_data = self.start_times[test_name]
            
            self.metrics[test_name] = {
                'wall_time_ms': (time.perf_counter() - start_data['wall_time']) * 1000,
                'cpu_time_ms': (time.process_time() - start_data['cpu_time']) * 1000,
                'memory_used_mb': self._get_memory_usage() - start_data['memory_start'],
                'timestamp': datetime.now()
            }
            
            del self.start_times[test_name]
            return self.metrics[test_name]
        
        def get_metrics(self, test_name: str) -> Optional[Dict[str, Any]]:
            """Récupère les métriques d'un test."""
            return self.metrics.get(test_name)
        
        def assert_performance_threshold(self, test_name: str, metric: str, threshold: float, operator: str = '<='):
            """Assert sur un seuil de performance."""
            if test_name not in self.metrics:
                pytest.fail(f"No metrics found for test {test_name}")
            
            value = self.metrics[test_name].get(metric)
            if value is None:
                pytest.fail(f"Metric {metric} not found for test {test_name}")
            
            if operator == '<=':
                assert value <= threshold, f"{metric} ({value}) exceeds threshold ({threshold})"
            elif operator == '>=':
                assert value >= threshold, f"{metric} ({value}) below threshold ({threshold})"
            elif operator == '<':
                assert value < threshold, f"{metric} ({value}) not less than threshold ({threshold})"
            elif operator == '>':
                assert value > threshold, f"{metric} ({value}) not greater than threshold ({threshold})"
            else:
                pytest.fail(f"Unknown operator {operator}")
        
        def _get_memory_usage(self) -> float:
            """Obtient l'utilisation mémoire actuelle (MB)."""
            try:
                import psutil
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024  # MB
            except ImportError:
                return 0.0  # Fallback si psutil n'est pas disponible
    
    return PerformanceMonitor()

@pytest.fixture(scope="function")
def test_data_factory():
    """Factory pour générer des données de test."""
    class TestDataFactory:
        def __init__(self):
            self.counters = {}
        
        def generate_request_data(self, **kwargs) -> Dict[str, Any]:
            """Génère des données de requête de test."""
            import uuid
            
            default_data = {
                'request_id': str(uuid.uuid4()),
                'method': 'GET',
                'path': '/api/test',
                'headers': {'Content-Type': 'application/json'},
                'query_params': {},
                'body': None,
                'timestamp': datetime.now().isoformat(),
                'user_id': 'test_user_123',
                'session_id': str(uuid.uuid4())
            }
            
            default_data.update(kwargs)
            return default_data
        
        def generate_performance_data(self, count: int = 100) -> List[Dict[str, Any]]:
            """Génère des données de performance de test."""
            import random
            
            data = []
            for i in range(count):
                data.append({
                    'timestamp': datetime.now() - timedelta(minutes=i),
                    'response_time_ms': random.uniform(50, 500),
                    'cpu_usage_percent': random.uniform(20, 80),
                    'memory_usage_mb': random.uniform(50, 200),
                    'requests_per_second': random.uniform(100, 1000),
                    'error_rate_percent': random.uniform(0, 5)
                })
            
            return data
        
        def generate_cache_data(self, size: int = 1000) -> Dict[str, Any]:
            """Génère des données de cache de test."""
            import random
            import string
            
            cache_data = {}
            for i in range(size):
                key = f"cache_key_{i}"
                value = ''.join(random.choices(string.ascii_letters + string.digits, k=50))
                cache_data[key] = {
                    'value': value,
                    'ttl': random.randint(60, 3600),
                    'created_at': datetime.now(),
                    'access_count': random.randint(1, 100)
                }
            
            return cache_data
        
        def generate_security_events(self, count: int = 50) -> List[Dict[str, Any]]:
            """Génère des événements de sécurité de test."""
            import random
            
            event_types = ['login_attempt', 'permission_denied', 'rate_limit_exceeded', 'suspicious_activity']
            severity_levels = ['low', 'medium', 'high', 'critical']
            
            events = []
            for i in range(count):
                events.append({
                    'event_id': f"sec_event_{i}",
                    'event_type': random.choice(event_types),
                    'severity': random.choice(severity_levels),
                    'timestamp': datetime.now() - timedelta(hours=random.randint(0, 24)),
                    'source_ip': f"192.168.1.{random.randint(1, 254)}",
                    'user_agent': 'Mozilla/5.0 (Test Browser)',
                    'details': {'test': True, 'iteration': i}
                })
            
            return events
        
        def get_next_id(self, prefix: str = 'test') -> str:
            """Génère un ID unique avec compteur."""
            if prefix not in self.counters:
                self.counters[prefix] = 0
            
            self.counters[prefix] += 1
            return f"{prefix}_{self.counters[prefix]:04d}"
    
    return TestDataFactory()

@pytest.fixture(scope="function")
def async_test_utils():
    """Utilitaires pour tests asynchrones."""
    class AsyncTestUtils:
        @staticmethod
        async def run_with_timeout(coro, timeout_seconds: float = 5.0):
            """Exécute une coroutine avec timeout."""
            try:
                return await asyncio.wait_for(coro, timeout=timeout_seconds)
            except asyncio.TimeoutError:
                pytest.fail(f"Async operation timed out after {timeout_seconds} seconds")
        
        @staticmethod
        async def run_concurrent(coroutines: List, max_concurrency: int = 10):
            """Exécute plusieurs coroutines en parallèle avec limite de concurrence."""
            semaphore = asyncio.Semaphore(max_concurrency)
            
            async def bounded_coro(coro):
                async with semaphore:
                    return await coro
            
            tasks = [bounded_coro(coro) for coro in coroutines]
            return await asyncio.gather(*tasks)
        
        @staticmethod
        def create_mock_async_context():
            """Crée un contexte asynchrone mock."""
            class MockAsyncContext:
                def __init__(self):
                    self.entered = False
                    self.exited = False
                
                async def __aenter__(self):
                    self.entered = True
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    self.exited = True
                    return False
            
            return MockAsyncContext()
    
    return AsyncTestUtils()


# =============================================================================
# MARKERS DE TEST PERSONNALISÉS
# =============================================================================

# Marqueurs pour catégoriser les tests
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.security = pytest.mark.security
pytest.mark.slow = pytest.mark.slow
pytest.mark.fast = pytest.mark.fast


# =============================================================================
# CONFIGURATION HOOKS PYTEST
# =============================================================================

def pytest_configure(config):
    """Configuration personnalisée pytest."""
    # Ajouter les marqueurs personnalisés
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "security: mark test as a security test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "fast: mark test as fast running")

def pytest_collection_modifyitems(config, items):
    """Modifie la collection de tests pour ajouter des marqueurs automatiques."""
    for item in items:
        # Marqueurs automatiques basés sur le nom
        if "performance" in item.name.lower():
            item.add_marker(pytest.mark.performance)
        if "security" in item.name.lower():
            item.add_marker(pytest.mark.security)
        if "integration" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
        
        # Marqueurs basés sur la durée estimée
        if any(keyword in item.name.lower() for keyword in ["load", "stress", "benchmark"]):
            item.add_marker(pytest.mark.slow)
        else:
            item.add_marker(pytest.mark.fast)

def pytest_runtest_setup(item):
    """Setup avant chaque test."""
    # Configuration des timeouts basés sur les marqueurs
    if item.get_closest_marker("performance"):
        item.config.timeout = TEST_TIMEOUTS['performance_test']
    elif item.get_closest_marker("integration"):
        item.config.timeout = TEST_TIMEOUTS['integration_test']
    else:
        item.config.timeout = TEST_TIMEOUTS['unit_test']

def pytest_runtest_teardown(item, nextitem):
    """Nettoyage après chaque test."""
    # Nettoyage automatique des mocks
    pass


# =============================================================================
# UTILITAIRES GLOBAUX DE TEST
# =============================================================================

class TestAssertions:
    """Assertions personnalisées pour les tests enterprise."""
    
    @staticmethod
    def assert_response_time(actual_ms: float, expected_ms: float, tolerance: float = 0.1):
        """Assert sur le temps de réponse avec tolérance."""
        tolerance_ms = expected_ms * tolerance
        assert abs(actual_ms - expected_ms) <= tolerance_ms, \
            f"Response time {actual_ms}ms not within {tolerance*100}% of expected {expected_ms}ms"
    
    @staticmethod
    def assert_memory_usage(actual_mb: float, max_mb: float):
        """Assert sur l'utilisation mémoire."""
        assert actual_mb <= max_mb, f"Memory usage {actual_mb}MB exceeds limit {max_mb}MB"
    
    @staticmethod
    def assert_throughput(actual_qps: float, min_qps: float):
        """Assert sur le débit."""
        assert actual_qps >= min_qps, f"Throughput {actual_qps} QPS below minimum {min_qps} QPS"
    
    @staticmethod
    def assert_error_rate(actual_rate: float, max_rate: float):
        """Assert sur le taux d'erreur."""
        assert actual_rate <= max_rate, f"Error rate {actual_rate}% exceeds maximum {max_rate}%"


class TestHelpers:
    """Helpers utilitaires pour les tests."""
    
    @staticmethod
    def wait_for_condition(condition_func, timeout_seconds: float = 10.0, check_interval: float = 0.1):
        """Attend qu'une condition soit vraie."""
        start_time = time.perf_counter()
        
        while time.perf_counter() - start_time < timeout_seconds:
            if condition_func():
                return True
            time.sleep(check_interval)
        
        return False
    
    @staticmethod
    def generate_load(target_qps: int, duration_seconds: int, request_func):
        """Génère une charge de test."""
        interval = 1.0 / target_qps
        end_time = time.perf_counter() + duration_seconds
        request_count = 0
        
        while time.perf_counter() < end_time:
            start = time.perf_counter()
            
            try:
                request_func()
                request_count += 1
            except Exception as e:
                # Log l'erreur mais continue
                print(f"Request failed: {e}")
            
            # Maintenir le débit
            elapsed = time.perf_counter() - start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        return request_count


if __name__ == "__main__":
    print("Configuration de test chargée avec succès")
    print(f"Seuils de performance: {PERFORMANCE_THRESHOLDS}")
    print(f"Timeouts de test: {TEST_TIMEOUTS}")
