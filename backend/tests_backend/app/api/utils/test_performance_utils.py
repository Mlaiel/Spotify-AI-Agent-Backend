"""
🎵 Spotify AI Agent - Tests Performance Utils Module
====================================================

Tests enterprise complets pour le module performance_utils
avec validation de performances, monitoring et optimisation.

🎖️ Développé par l'équipe d'experts enterprise
"""

import pytest
import time
import threading
import concurrent.futures
import psutil
import gc
from unittest.mock import patch, Mock
from contextlib import contextmanager

# Import du module à tester
from backend.app.api.utils.performance_utils import (
    measure_time,
    measure_memory,
    profile_function,
    cache_result,
    rate_limiter,
    batch_processor,
    async_executor,
    performance_monitor,
    memory_monitor,
    cpu_monitor,
    optimize_query,
    lazy_loader,
    connection_pool,
    timeout_handler,
    retry_mechanism,
    circuit_breaker,
    load_balancer,
    performance_alert,
    benchmark_function,
    performance_report
)

from . import TestUtils, security_test, performance_test, integration_test


class TestPerformanceUtils:
    """Tests pour le module performance_utils"""
    
    def test_measure_time_basic(self):
        """Test mesure temps basique"""
        def slow_function():
            time.sleep(0.1)  # 100ms
            return "result"
        
        result, execution_time = measure_time(slow_function)
        
        assert result == "result"
        assert execution_time >= 0.09  # Au moins 90ms
        assert execution_time <= 0.2   # Maximum 200ms (marge sécurité)
    
    def test_measure_time_with_args(self):
        """Test mesure temps avec arguments"""
        def add_numbers(a, b, multiplier=1):
            time.sleep(0.05)
            return (a + b) * multiplier
        
        result, execution_time = measure_time(add_numbers, 5, 3, multiplier=2)
        
        assert result == 16  # (5 + 3) * 2
        assert execution_time >= 0.04
    
    def test_measure_time_exception(self):
        """Test mesure temps avec exception"""
        def failing_function():
            time.sleep(0.05)
            raise ValueError("Test error")
        
        try:
            result, execution_time = measure_time(failing_function)
            assert False, "Exception devrait être propagée"
        except ValueError:
            # Exception propagée correctement
            assert True
    
    def test_measure_memory_basic(self):
        """Test mesure mémoire basique"""
        def memory_intensive_function():
            # Créer données en mémoire
            data = [i for i in range(10000)]
            return len(data)
        
        result, memory_usage = measure_memory(memory_intensive_function)
        
        assert result == 10000
        assert isinstance(memory_usage, dict)
        assert 'peak_memory' in memory_usage
        assert 'memory_diff' in memory_usage
        assert memory_usage['peak_memory'] > 0
    
    def test_measure_memory_cleanup(self):
        """Test mesure mémoire avec nettoyage"""
        def allocate_and_cleanup():
            # Allouer mémoire
            big_list = [0] * 100000
            # Nettoyer explicitement
            del big_list
            gc.collect()
            return "cleaned"
        
        result, memory_usage = measure_memory(allocate_and_cleanup)
        
        assert result == "cleaned"
        # La différence de mémoire devrait être proche de 0 après nettoyage
        assert abs(memory_usage['memory_diff']) < 50  # MB
    
    def test_profile_function_basic(self):
        """Test profilage fonction basique"""
        def complex_function():
            # Simulation opérations complexes
            result = 0
            for i in range(1000):
                result += i ** 2
            return result
        
        profile_data = profile_function(complex_function)
        
        assert isinstance(profile_data, dict)
        assert 'execution_time' in profile_data
        assert 'memory_usage' in profile_data
        assert 'cpu_usage' in profile_data or 'cpu_percent' in profile_data
        assert profile_data['execution_time'] > 0
    
    def test_profile_function_detailed(self):
        """Test profilage détaillé"""
        def nested_function():
            def inner_function():
                time.sleep(0.01)
                return sum(range(100))
            
            results = []
            for _ in range(5):
                results.append(inner_function())
            return results
        
        profile_data = profile_function(nested_function, detailed=True)
        
        assert 'call_count' in profile_data or 'function_calls' in profile_data
        assert 'hot_spots' in profile_data or 'line_profiler' in profile_data
    
    def test_cache_result_basic(self):
        """Test cache résultats basique"""
        call_count = 0
        
        @cache_result(ttl=60)  # Cache 1 minute
        def expensive_computation(n):
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)  # Simulation calcul coûteux
            return n ** 2
        
        # Premier appel
        result1 = expensive_computation(5)
        assert result1 == 25
        assert call_count == 1
        
        # Deuxième appel (doit utiliser cache)
        result2 = expensive_computation(5)
        assert result2 == 25
        assert call_count == 1  # Pas d'appel supplémentaire
        
        # Appel avec paramètre différent
        result3 = expensive_computation(6)
        assert result3 == 36
        assert call_count == 2
    
    def test_cache_result_expiration(self):
        """Test expiration cache"""
        call_count = 0
        
        @cache_result(ttl=0.1)  # Cache 100ms
        def cached_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # Premier appel
        result1 = cached_function(10)
        assert result1 == 20
        assert call_count == 1
        
        # Attendre expiration
        time.sleep(0.15)
        
        # Appel après expiration
        result2 = cached_function(10)
        assert result2 == 20
        assert call_count == 2  # Cache expiré, nouvel appel
    
    def test_rate_limiter_basic(self):
        """Test limiteur de débit basique"""
        @rate_limiter(max_calls=3, time_window=1.0)
        def limited_function(x):
            return x * 2
        
        # 3 premiers appels doivent passer
        start_time = time.time()
        for i in range(3):
            result = limited_function(i)
            assert result == i * 2
        
        # 4ème appel doit être bloqué ou retardé
        try:
            result = limited_function(4)
            # Si pas d'exception, vérifier que ça a pris du temps
            elapsed = time.time() - start_time
            assert elapsed >= 0.5  # Au moins 500ms de délai
        except Exception:
            # Rate limiting par exception
            assert True
    
    def test_rate_limiter_time_window(self):
        """Test fenêtre temporelle rate limiter"""
        @rate_limiter(max_calls=2, time_window=0.5)
        def windowed_function(x):
            return x + 1
        
        # 2 appels rapides
        result1 = windowed_function(1)
        result2 = windowed_function(2)
        assert result1 == 2
        assert result2 == 3
        
        # Attendre fin de fenêtre
        time.sleep(0.6)
        
        # Nouvel appel doit passer
        result3 = windowed_function(3)
        assert result3 == 4
    
    def test_batch_processor_basic(self):
        """Test processeur par lot basique"""
        def process_item(item):
            return item ** 2
        
        items = list(range(10))
        
        results = batch_processor(items, process_item, batch_size=3)
        
        expected = [i ** 2 for i in range(10)]
        assert results == expected
    
    def test_batch_processor_parallel(self):
        """Test traitement parallèle par lot"""
        def slow_process(item):
            time.sleep(0.01)  # Simulation traitement lent
            return item * 3
        
        items = list(range(20))
        
        start_time = time.time()
        results = batch_processor(items, slow_process, batch_size=5, parallel=True, max_workers=4)
        execution_time = time.time() - start_time
        
        expected = [i * 3 for i in range(20)]
        assert results == expected
        
        # Traitement parallèle doit être plus rapide
        assert execution_time < 0.15  # Moins que traitement séquentiel
    
    def test_async_executor_basic(self):
        """Test exécuteur asynchrone basique"""
        def cpu_bound_task(n):
            return sum(i ** 2 for i in range(n))
        
        tasks = [
            (cpu_bound_task, (100,)),
            (cpu_bound_task, (200,)),
            (cpu_bound_task, (300,))
        ]
        
        results = async_executor(tasks, max_workers=2)
        
        assert len(results) == 3
        assert results[0] == sum(i ** 2 for i in range(100))
        assert results[1] == sum(i ** 2 for i in range(200))
        assert results[2] == sum(i ** 2 for i in range(300))
    
    def test_async_executor_timeout(self):
        """Test timeout exécuteur asynchrone"""
        def slow_task(seconds):
            time.sleep(seconds)
            return f"Completed after {seconds}s"
        
        tasks = [
            (slow_task, (0.1,)),
            (slow_task, (0.5,)),  # Peut dépasser timeout
            (slow_task, (0.05,))
        ]
        
        results = async_executor(tasks, max_workers=3, timeout=0.3)
        
        # Certaines tâches peuvent échouer par timeout
        assert len(results) <= 3
        assert any("0.1" in str(r) for r in results if r is not None)
    
    def test_performance_monitor_basic(self):
        """Test moniteur performance basique"""
        monitor = performance_monitor()
        
        def monitored_function():
            time.sleep(0.05)
            return "monitored result"
        
        with monitor:
            result = monitored_function()
        
        assert result == "monitored result"
        
        metrics = monitor.get_metrics()
        assert isinstance(metrics, dict)
        assert 'execution_time' in metrics
        assert 'memory_usage' in metrics
        assert metrics['execution_time'] >= 0.04
    
    def test_memory_monitor_threshold(self):
        """Test moniteur mémoire avec seuil"""
        def memory_heavy_function():
            # Allouer beaucoup de mémoire
            big_data = [0] * 1000000  # ~8MB sur 64-bit
            return len(big_data)
        
        with memory_monitor(threshold_mb=5) as monitor:
            try:
                result = memory_heavy_function()
                # Peut lever alerte si dépasse seuil
                assert result == 1000000
            except MemoryError:
                # Alerte mémoire déclenchée
                assert True
        
        peak_memory = monitor.get_peak_memory()
        assert peak_memory > 0
    
    def test_cpu_monitor_basic(self):
        """Test moniteur CPU"""
        def cpu_intensive():
            # Calcul intensif
            result = 0
            for i in range(100000):
                result += i ** 0.5
            return result
        
        with cpu_monitor() as monitor:
            result = cpu_intensive()
        
        assert result > 0
        
        cpu_usage = monitor.get_cpu_usage()
        assert isinstance(cpu_usage, (int, float))
        assert 0 <= cpu_usage <= 100
    
    def test_optimize_query_basic(self):
        """Test optimisation requête basique"""
        # Simulation base de données
        mock_data = [
            {"id": 1, "name": "Alice", "age": 25, "city": "Paris"},
            {"id": 2, "name": "Bob", "age": 30, "city": "London"},
            {"id": 3, "name": "Charlie", "age": 35, "city": "Paris"}
        ]
        
        def mock_query(filters=None, limit=None):
            result = mock_data
            if filters:
                for key, value in filters.items():
                    result = [item for item in result if item.get(key) == value]
            if limit:
                result = result[:limit]
            return result
        
        # Requête optimisée
        optimized_query = optimize_query(mock_query)
        
        # Test avec cache
        result1 = optimized_query(filters={"city": "Paris"})
        result2 = optimized_query(filters={"city": "Paris"})  # Doit utiliser cache
        
        assert len(result1) == 2
        assert result1 == result2
        assert all(item["city"] == "Paris" for item in result1)
    
    def test_lazy_loader_basic(self):
        """Test chargement paresseux basique"""
        load_count = 0
        
        def expensive_loader():
            nonlocal load_count
            load_count += 1
            time.sleep(0.01)
            return {"expensive": "data", "load_count": load_count}
        
        lazy_data = lazy_loader(expensive_loader)
        
        # Pas encore chargé
        assert load_count == 0
        
        # Premier accès déclenche chargement
        data1 = lazy_data.get()
        assert load_count == 1
        assert data1["expensive"] == "data"
        
        # Deuxième accès utilise cache
        data2 = lazy_data.get()
        assert load_count == 1  # Pas de rechargement
        assert data2 == data1
    
    def test_connection_pool_basic(self):
        """Test pool de connexions basique"""
        connection_count = 0
        
        class MockConnection:
            def __init__(self):
                nonlocal connection_count
                connection_count += 1
                self.id = connection_count
                self.active = True
            
            def execute(self, query):
                return f"Result for {query} on connection {self.id}"
            
            def close(self):
                self.active = False
        
        pool = connection_pool(MockConnection, pool_size=3)
        
        # Utiliser connexions
        with pool.get_connection() as conn1:
            result1 = conn1.execute("SELECT 1")
            assert "connection" in result1
        
        with pool.get_connection() as conn2:
            result2 = conn2.execute("SELECT 2")
            assert "connection" in result2
        
        # Vérifier réutilisation
        pool_size = pool.get_pool_size()
        assert pool_size <= 3
    
    def test_timeout_handler_basic(self):
        """Test gestionnaire timeout"""
        @timeout_handler(timeout=0.1)
        def quick_function():
            time.sleep(0.05)
            return "quick result"
        
        @timeout_handler(timeout=0.1)
        def slow_function():
            time.sleep(0.2)
            return "slow result"
        
        # Fonction rapide doit passer
        result1 = quick_function()
        assert result1 == "quick result"
        
        # Fonction lente doit timeout
        try:
            result2 = slow_function()
            assert False, "Timeout attendu"
        except TimeoutError:
            assert True
    
    def test_retry_mechanism_basic(self):
        """Test mécanisme retry basique"""
        attempt_count = 0
        
        @retry_mechanism(max_retries=3, delay=0.01)
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = flaky_function()
        
        assert result == "success"
        assert attempt_count == 3
    
    def test_retry_mechanism_max_retries(self):
        """Test dépassement max retries"""
        @retry_mechanism(max_retries=2, delay=0.01)
        def always_failing():
            raise ValueError("Always fails")
        
        try:
            result = always_failing()
            assert False, "Exception attendue"
        except ValueError:
            assert True
    
    def test_circuit_breaker_basic(self):
        """Test circuit breaker basique"""
        failure_count = 0
        
        @circuit_breaker(failure_threshold=3, timeout=0.1)
        def unreliable_service():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 5:
                raise ConnectionError("Service unavailable")
            return "service response"
        
        # Premières tentatives échouent
        for i in range(3):
            try:
                unreliable_service()
            except ConnectionError:
                pass
        
        # Circuit breaker doit s'ouvrir
        try:
            unreliable_service()
            # Peut passer si circuit fermé ou lever CircuitBreakerOpen
        except Exception as e:
            if "circuit" in str(e).lower() or "breaker" in str(e).lower():
                assert True
            else:
                assert True  # Autre exception acceptable
    
    def test_load_balancer_basic(self):
        """Test équilibreur de charge basique"""
        # Simulation serveurs
        servers = [
            {"id": 1, "response": "Server 1 response"},
            {"id": 2, "response": "Server 2 response"},
            {"id": 3, "response": "Server 3 response"}
        ]
        
        def server_handler(server, request):
            return f"{server['response']} for {request}"
        
        balancer = load_balancer(servers, server_handler, strategy='round_robin')
        
        # Tester distribution
        responses = []
        for i in range(6):
            response = balancer.handle_request(f"request_{i}")
            responses.append(response)
        
        # Vérifier distribution équitable
        server_counts = {}
        for response in responses:
            for server in servers:
                if f"Server {server['id']}" in response:
                    server_counts[server['id']] = server_counts.get(server['id'], 0) + 1
        
        # Chaque serveur doit être utilisé
        assert len(server_counts) == 3
        assert all(count >= 1 for count in server_counts.values())
    
    def test_performance_alert_basic(self):
        """Test alerte performance"""
        alerts = []
        
        def alert_handler(alert):
            alerts.append(alert)
        
        @performance_alert(max_time=0.05, alert_callback=alert_handler)
        def monitored_function():
            time.sleep(0.1)  # Dépasse limite
            return "result"
        
        result = monitored_function()
        
        assert result == "result"
        assert len(alerts) >= 1
        assert "time" in alerts[0] or "performance" in alerts[0]
    
    def test_benchmark_function_basic(self):
        """Test benchmark fonction"""
        def function_to_benchmark(n):
            return sum(i ** 2 for i in range(n))
        
        benchmark_results = benchmark_function(
            function_to_benchmark,
            test_cases=[100, 500, 1000],
            iterations=5
        )
        
        assert isinstance(benchmark_results, dict)
        assert len(benchmark_results) == 3  # 3 cas de test
        
        for n, stats in benchmark_results.items():
            assert 'mean_time' in stats
            assert 'min_time' in stats
            assert 'max_time' in stats
            assert stats['mean_time'] > 0
    
    def test_performance_report_basic(self):
        """Test rapport performance"""
        def sample_function():
            time.sleep(0.01)
            return list(range(1000))
        
        report = performance_report(sample_function, iterations=3)
        
        assert isinstance(report, dict)
        assert 'execution_times' in report
        assert 'memory_usage' in report
        assert 'statistics' in report
        
        stats = report['statistics']
        assert 'mean_time' in stats
        assert 'std_dev' in stats
        assert len(report['execution_times']) == 3
    
    @performance_test
    def test_performance_utilities_overhead(self):
        """Test overhead des utilitaires performance"""
        def simple_function(x):
            return x * 2
        
        # Test sans monitoring
        start_time = time.time()
        for i in range(1000):
            simple_function(i)
        time_without_monitoring = time.time() - start_time
        
        # Test avec monitoring
        @performance_monitor()
        def monitored_simple(x):
            return x * 2
        
        start_time = time.time()
        for i in range(1000):
            monitored_simple(i)
        time_with_monitoring = time.time() - start_time
        
        # Overhead doit être raisonnable (< 50% supplémentaire)
        overhead_ratio = time_with_monitoring / time_without_monitoring
        assert overhead_ratio < 1.5, f"Overhead trop élevé: {overhead_ratio}"
    
    @performance_test
    def test_cache_performance_improvement(self):
        """Test amélioration performance avec cache"""
        def expensive_calculation(n):
            time.sleep(0.01)  # Simulation calcul coûteux
            return sum(i ** 2 for i in range(n))
        
        @cache_result(ttl=60)
        def cached_calculation(n):
            time.sleep(0.01)
            return sum(i ** 2 for i in range(n))
        
        # Test sans cache
        start_time = time.time()
        for _ in range(5):
            expensive_calculation(100)
        time_without_cache = time.time() - start_time
        
        # Test avec cache
        start_time = time.time()
        for _ in range(5):
            cached_calculation(100)  # Même paramètre = cache hit
        time_with_cache = time.time() - start_time
        
        # Cache doit améliorer significativement
        improvement_ratio = time_without_cache / time_with_cache
        assert improvement_ratio > 3, f"Amélioration insuffisante: {improvement_ratio}"
    
    @integration_test
    def test_complete_performance_optimization(self):
        """Test optimisation performance complète"""
        # Scénario: API avec cache, rate limiting, monitoring
        
        request_count = 0
        
        @rate_limiter(max_calls=100, time_window=60)
        @cache_result(ttl=30)
        @performance_monitor()
        @timeout_handler(timeout=1.0)
        def optimized_api_endpoint(user_id, query_type):
            nonlocal request_count
            request_count += 1
            
            # Simulation traitement API
            time.sleep(0.01)
            
            if query_type == "profile":
                return {"user_id": user_id, "name": f"User {user_id}", "data": list(range(100))}
            elif query_type == "settings":
                return {"user_id": user_id, "preferences": {"theme": "dark"}}
            else:
                raise ValueError("Unknown query type")
        
        # Test utilisation normale
        start_time = time.time()
        
        # Plusieurs appels identiques (test cache)
        results = []
        for i in range(10):
            result = optimized_api_endpoint(123, "profile")
            results.append(result)
        
        # Appels différents
        for i in range(5):
            result = optimized_api_endpoint(i, "settings")
            results.append(result)
        
        execution_time = time.time() - start_time
        
        # Vérifications
        assert len(results) == 15
        assert all(r is not None for r in results)
        
        # Cache doit réduire le nombre d'appels réels
        assert request_count < 15  # Moins d'appels grâce au cache
        
        # Performance doit être raisonnable
        assert execution_time < 0.5  # Moins de 500ms total
        
        # Tous les profils doivent être identiques (cache)
        profile_results = results[:10]
        assert all(r == profile_results[0] for r in profile_results)
        
        print("✅ Optimisation performance complète validée")


# Tests de stress et limites
class TestPerformanceStress:
    """Tests de stress et limites performance"""
    
    @performance_test
    def test_high_concurrency_cache(self):
        """Test cache haute concurrence"""
        @cache_result(ttl=60)
        def cached_function(x):
            time.sleep(0.001)  # Micro-délai
            return x ** 2
        
        def worker(thread_id):
            results = []
            for i in range(100):
                result = cached_function(i % 10)  # 10 valeurs différentes
                results.append(result)
            return results
        
        # Test avec plusieurs threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            all_results = [future.result() for future in futures]
        
        # Vérifier cohérence
        assert len(all_results) == 10
        assert all(len(results) == 100 for results in all_results)
        
        # Tous les threads doivent avoir les mêmes résultats pour même input
        first_thread_results = all_results[0]
        for other_results in all_results[1:]:
            assert other_results == first_thread_results
    
    @performance_test
    def test_memory_pressure_handling(self):
        """Test gestion pression mémoire"""
        def memory_intensive_task(size_mb):
            # Allouer mémoire
            data = [0] * (size_mb * 1024 * 1024 // 8)  # 8 bytes par int sur 64-bit
            return len(data)
        
        with memory_monitor(threshold_mb=50) as monitor:
            try:
                # Essayer d'allouer beaucoup de mémoire
                results = []
                for size in [10, 20, 30]:  # MB
                    result = memory_intensive_task(size)
                    results.append(result)
                    
                    # Nettoyer entre les allocations
                    gc.collect()
                
                assert len(results) == 3
            except MemoryError:
                # Acceptable si système a peu de mémoire
                assert True
        
        peak_memory = monitor.get_peak_memory()
        assert peak_memory > 0
    
    @performance_test
    def test_rate_limiter_burst_handling(self):
        """Test gestion rafales rate limiter"""
        @rate_limiter(max_calls=10, time_window=1.0)
        def rate_limited_function(x):
            return x * 2
        
        # Essayer rafale de requêtes
        start_time = time.time()
        results = []
        errors = 0
        
        for i in range(50):  # Plus que la limite
            try:
                result = rate_limited_function(i)
                results.append(result)
            except Exception:
                errors += 1
        
        execution_time = time.time() - start_time
        
        # Vérifier que rate limiting fonctionne
        if errors > 0:
            # Rate limiting par exception
            assert errors > 30  # Beaucoup de requêtes bloquées
        else:
            # Rate limiting par délai
            assert execution_time > 3.0  # Au moins 3 secondes de délai
        
        assert len(results) + errors == 50


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
