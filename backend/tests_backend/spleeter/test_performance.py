"""
🎵 Spotify AI Agent - Tests de Performance Spleeter
=================================================

Tests de performance, benchmarks et stress tests
pour valider les performances du module Spleeter.

🎖️ Développé par l'équipe d'experts enterprise
"""

import pytest
import asyncio
import time
import psutil
import numpy as np
from pathlib import Path
import tempfile
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch, AsyncMock

from spleeter import SpleeterEngine, SpleeterConfig
from spleeter.monitoring import MetricsCollector, PerformanceTimer, ResourceMonitor
from spleeter.cache import CacheManager
from spleeter.processor import BatchProcessor
from spleeter.utils import PerformanceOptimizer


class TestPerformanceBenchmarks:
    """Tests de benchmarks de performance"""
    
    @pytest.fixture
    def performance_config(self):
        """Configuration optimisée pour les tests de performance"""
        return SpleeterConfig(
            enable_gpu=False,  # CPU pour reproductibilité
            batch_size=8,
            worker_threads=4,
            cache_enabled=True,
            cache_size_mb=512,
            enable_monitoring=True,
            enable_preprocessing=True
        )
    
    @pytest.fixture
    async def performance_engine(self, performance_config, tmp_path):
        """Engine optimisé pour les tests de performance"""
        config = performance_config
        config.models_dir = str(tmp_path / "models")
        config.cache_dir = str(tmp_path / "cache")
        
        engine = SpleeterEngine(config=config)
        
        with patch('tensorflow.keras.models.load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = [
                np.random.random((1, 1024, 513, 1)),
                np.random.random((1, 1024, 513, 1))
            ]
            mock_load.return_value = mock_model
            
            await engine.initialize()
            yield engine
            await engine.cleanup()
    
    @pytest.fixture
    def sample_audio_data(self):
        """Données audio de test de différentes tailles"""
        sizes = {
            'small': 44100 * 10,    # 10 secondes
            'medium': 44100 * 60,   # 1 minute
            'large': 44100 * 300,   # 5 minutes
        }
        
        data = {}
        for size_name, samples in sizes.items():
            # Générer signal audio réaliste
            t = np.linspace(0, samples / 44100, samples)
            # Mélange de fréquences pour simuler de la musique
            signal = (np.sin(2 * np.pi * 440 * t) +
                     0.5 * np.sin(2 * np.pi * 880 * t) +
                     0.3 * np.sin(2 * np.pi * 1320 * t))
            # Ajout de bruit léger
            noise = np.random.normal(0, 0.01, samples)
            data[size_name] = signal + noise
        
        return data
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_separation_performance_benchmark(self, performance_engine, sample_audio_data, tmp_path):
        """Benchmark de performance de séparation"""
        results = {}
        
        for size_name, audio_data in sample_audio_data.items():
            # Créer fichier de test
            audio_file = tmp_path / f"test_{size_name}.wav"
            audio_file.write_bytes(b'MOCK_AUDIO_DATA')
            
            output_dir = tmp_path / f"output_{size_name}"
            output_dir.mkdir()
            
            # Mock des opérations audio avec délais réalistes
            def mock_load_with_delay(*args, **kwargs):
                # Simuler le temps de chargement basé sur la taille
                delay = len(audio_data) / 44100 * 0.1  # 0.1x temps réel
                time.sleep(delay)
                return audio_data, 44100
            
            def mock_stft_with_delay(*args, **kwargs):
                time.sleep(0.05)  # 50ms pour STFT
                return np.random.random((513, 1024)) + 1j * np.random.random((513, 1024))
            
            def mock_istft_with_delay(*args, **kwargs):
                time.sleep(0.03)  # 30ms pour iSTFT
                return audio_data
            
            with patch('librosa.load', side_effect=mock_load_with_delay), \
                 patch('soundfile.write') as mock_write, \
                 patch('librosa.stft', side_effect=mock_stft_with_delay), \
                 patch('librosa.istft', side_effect=mock_istft_with_delay):
                
                mock_write.return_value = None
                
                # Mesurer la performance
                start_time = time.perf_counter()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                result = await performance_engine.separate(
                    audio_path=str(audio_file),
                    model_name="spleeter:2stems-16kHz",
                    output_dir=str(output_dir)
                )
                
                end_time = time.perf_counter()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                processing_time = end_time - start_time
                memory_usage = end_memory - start_memory
                audio_duration = len(audio_data) / 44100
                
                results[size_name] = {
                    'audio_duration': audio_duration,
                    'processing_time': processing_time,
                    'memory_usage': memory_usage,
                    'real_time_factor': processing_time / audio_duration,
                    'success': result.success if result else False
                }
        
        # Vérifications de performance
        for size_name, metrics in results.items():
            print(f"\n{size_name.upper()} Audio Performance:")
            print(f"  Duration: {metrics['audio_duration']:.1f}s")
            print(f"  Processing: {metrics['processing_time']:.2f}s")
            print(f"  Real-time factor: {metrics['real_time_factor']:.2f}x")
            print(f"  Memory usage: {metrics['memory_usage']:.1f}MB")
            print(f"  Success: {metrics['success']}")
            
            # Assertions de performance
            assert metrics['success'], f"Processing failed for {size_name}"
            assert metrics['real_time_factor'] < 5.0, f"Too slow for {size_name}: {metrics['real_time_factor']}x"
            assert metrics['memory_usage'] < 1000, f"Too much memory for {size_name}: {metrics['memory_usage']}MB"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, performance_engine, tmp_path):
        """Test de performance du traitement par lots"""
        # Créer un lot de fichiers de test
        num_files = 10
        audio_files = []
        
        for i in range(num_files):
            audio_file = tmp_path / f"batch_test_{i}.wav"
            audio_file.write_bytes(b'MOCK_BATCH_AUDIO_DATA')
            audio_files.append(str(audio_file))
        
        output_dir = tmp_path / "batch_output"
        output_dir.mkdir()
        
        # Mock avec délais réalistes
        def mock_batch_load(*args, **kwargs):
            time.sleep(0.1)  # 100ms par fichier
            return np.random.random(88200), 44100  # 2 secondes d'audio
        
        with patch('librosa.load', side_effect=mock_batch_load), \
             patch('soundfile.write') as mock_write, \
             patch('librosa.stft') as mock_stft, \
             patch('librosa.istft') as mock_istft:
            
            mock_write.return_value = None
            mock_stft.return_value = np.random.random((513, 1024)) + 1j * np.random.random((513, 1024))
            mock_istft.return_value = np.random.random(88200)
            
            # Test séquentiel vs parallèle
            
            # 1. Traitement séquentiel
            start_time = time.perf_counter()
            sequential_results = []
            for audio_file in audio_files:
                result = await performance_engine.separate(
                    audio_path=audio_file,
                    model_name="spleeter:2stems-16kHz",
                    output_dir=str(output_dir / "sequential")
                )
                sequential_results.append(result)
            sequential_time = time.perf_counter() - start_time
            
            # 2. Traitement par lots
            start_time = time.perf_counter()
            batch_results = await performance_engine.batch_separate(
                audio_files=audio_files,
                model_name="spleeter:2stems-16kHz",
                output_dir=str(output_dir / "batch")
            )
            batch_time = time.perf_counter() - start_time
            
            # Analyse des résultats
            speedup = sequential_time / batch_time
            
            print(f"\nBatch Processing Performance:")
            print(f"  Files: {num_files}")
            print(f"  Sequential time: {sequential_time:.2f}s")
            print(f"  Batch time: {batch_time:.2f}s")
            print(f"  Speedup: {speedup:.2f}x")
            
            # Vérifications
            assert len(batch_results) == num_files
            assert speedup > 1.5, f"Batch processing not efficient enough: {speedup:.2f}x"
            successful_batch = sum(1 for r in batch_results if r.success)
            assert successful_batch >= num_files * 0.9, "Too many batch failures"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cache_performance_benchmark(self, tmp_path):
        """Benchmark de performance du cache"""
        cache_manager = CacheManager(
            cache_dir=str(tmp_path / "cache_bench"),
            memory_cache_size=100,
            disk_cache_size_mb=50,
            enable_redis=False
        )
        
        await cache_manager.initialize()
        
        # Données de test de différentes tailles
        test_data = {
            'small': {'data': np.random.random(1000).tolist()},
            'medium': {'data': np.random.random(10000).tolist()},
            'large': {'data': np.random.random(100000).tolist()}
        }
        
        results = {}
        
        for size_name, data in test_data.items():
            # Test d'écriture
            write_times = []
            for i in range(10):
                start_time = time.perf_counter()
                await cache_manager.set(f"{size_name}_key_{i}", data, ttl=300)
                write_time = time.perf_counter() - start_time
                write_times.append(write_time)
            
            # Test de lecture
            read_times = []
            for i in range(10):
                start_time = time.perf_counter()
                result = await cache_manager.get(f"{size_name}_key_{i}")
                read_time = time.perf_counter() - start_time
                read_times.append(read_time)
                assert result is not None
            
            results[size_name] = {
                'avg_write_time': statistics.mean(write_times),
                'avg_read_time': statistics.mean(read_times),
                'write_throughput': 1 / statistics.mean(write_times),
                'read_throughput': 1 / statistics.mean(read_times)
            }
        
        # Affichage des résultats
        for size_name, metrics in results.items():
            print(f"\n{size_name.upper()} Cache Performance:")
            print(f"  Avg write time: {metrics['avg_write_time']*1000:.2f}ms")
            print(f"  Avg read time: {metrics['avg_read_time']*1000:.2f}ms")
            print(f"  Write throughput: {metrics['write_throughput']:.1f} ops/s")
            print(f"  Read throughput: {metrics['read_throughput']:.1f} ops/s")
            
            # Assertions de performance
            assert metrics['avg_write_time'] < 0.1, f"Write too slow for {size_name}"
            assert metrics['avg_read_time'] < 0.05, f"Read too slow for {size_name}"
        
        # Test de performance du cache hit
        cache_stats = cache_manager.get_cache_stats()
        print(f"\nCache Statistics:")
        print(f"  Memory cache size: {cache_stats['memory_cache']['size']}")
        print(f"  Memory hit rate: {cache_stats['memory_cache']['hit_rate']:.1f}%")
        
        await cache_manager.cleanup()
    
    @pytest.mark.performance
    def test_monitoring_overhead_benchmark(self):
        """Benchmark de l'overhead du monitoring"""
        collector = MetricsCollector(enable_system_metrics=False)
        
        # Test sans monitoring
        start_time = time.perf_counter()
        for i in range(1000):
            # Simuler du travail
            _ = sum(range(100))
        no_monitoring_time = time.perf_counter() - start_time
        
        # Test avec monitoring
        start_time = time.perf_counter()
        for i in range(1000):
            # Simuler du travail avec métriques
            work_start = time.perf_counter()
            _ = sum(range(100))
            work_time = time.perf_counter() - work_start
            
            collector.record_metric(f"test_metric_{i % 10}", work_time, "seconds")
        monitoring_time = time.perf_counter() - start_time
        
        # Calcul de l'overhead
        overhead = ((monitoring_time - no_monitoring_time) / no_monitoring_time) * 100
        
        print(f"\nMonitoring Overhead Benchmark:")
        print(f"  Without monitoring: {no_monitoring_time:.4f}s")
        print(f"  With monitoring: {monitoring_time:.4f}s")
        print(f"  Overhead: {overhead:.2f}%")
        
        # Vérifications
        assert overhead < 20.0, f"Monitoring overhead too high: {overhead:.2f}%"
        
        # Vérifier que les métriques ont été collectées
        stats = collector.get_stats_summary()
        assert stats['buffer_size'] > 0
        assert len(stats['recent_metrics']) > 0
        
        collector.stop_system_monitoring()


class TestStressTests:
    """Tests de stress et de limite"""
    
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_concurrent_processing_stress(self, tmp_path):
        """Test de stress avec traitement concurrent intensif"""
        config = SpleeterConfig(
            models_dir=str(tmp_path / "models"),
            cache_dir=str(tmp_path / "cache"),
            enable_gpu=False,
            batch_size=4,
            worker_threads=8,  # Plus de workers pour le stress
            cache_enabled=True
        )
        
        engine = SpleeterEngine(config=config)
        
        with patch('tensorflow.keras.models.load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = [
                np.random.random((1, 1024, 513, 1)),
                np.random.random((1, 1024, 513, 1))
            ]
            mock_load.return_value = mock_model
            
            await engine.initialize()
            
            # Créer beaucoup de tâches concurrentes
            num_concurrent = 50
            tasks = []
            
            for i in range(num_concurrent):
                audio_file = tmp_path / f"stress_test_{i}.wav"
                audio_file.write_bytes(b'MOCK_STRESS_AUDIO_DATA')
                
                task = engine.separate(
                    audio_path=str(audio_file),
                    model_name="spleeter:2stems-16kHz",
                    output_dir=str(tmp_path / f"stress_output_{i}")
                )
                tasks.append(task)
            
            # Mock des opérations audio
            with patch('librosa.load') as mock_audio_load, \
                 patch('soundfile.write') as mock_write, \
                 patch('librosa.stft') as mock_stft, \
                 patch('librosa.istft') as mock_istft:
                
                mock_audio_load.return_value = (np.random.random(88200), 44100)
                mock_stft.return_value = np.random.random((513, 1024)) + 1j * np.random.random((513, 1024))
                mock_istft.return_value = np.random.random(88200)
                mock_write.return_value = None
                
                # Mesurer les ressources avant
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                initial_threads = process.num_threads()
                
                # Exécuter toutes les tâches
                start_time = time.perf_counter()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                total_time = time.perf_counter() - start_time
                
                # Mesurer les ressources après
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                final_threads = process.num_threads()
                
                # Analyser les résultats
                successful = sum(1 for r in results if not isinstance(r, Exception) and getattr(r, 'success', False))
                exceptions = [r for r in results if isinstance(r, Exception)]
                
                print(f"\nConcurrent Processing Stress Test:")
                print(f"  Concurrent tasks: {num_concurrent}")
                print(f"  Successful: {successful}")
                print(f"  Exceptions: {len(exceptions)}")
                print(f"  Total time: {total_time:.2f}s")
                print(f"  Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB")
                print(f"  Threads: {initial_threads} -> {final_threads}")
                
                # Vérifications de stress
                assert successful >= num_concurrent * 0.8, f"Too many failures: {successful}/{num_concurrent}"
                assert (final_memory - initial_memory) < 500, "Memory leak detected"
                assert final_threads < initial_threads + 20, "Thread leak detected"
            
            await engine.cleanup()
    
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_memory_pressure_stress(self, tmp_path):
        """Test de stress avec pression mémoire"""
        config = SpleeterConfig(
            models_dir=str(tmp_path / "models"),
            cache_dir=str(tmp_path / "cache"),
            enable_gpu=False,
            batch_size=16,  # Grande taille de batch
            worker_threads=4,
            cache_size_mb=100  # Cache limité
        )
        
        engine = SpleeterEngine(config=config)
        
        with patch('tensorflow.keras.models.load_model') as mock_load:
            mock_model = Mock()
            # Simuler des données plus volumineuses
            mock_model.predict.return_value = [
                np.random.random((1, 2048, 513, 1)),  # Plus grand
                np.random.random((1, 2048, 513, 1))
            ]
            mock_load.return_value = mock_model
            
            await engine.initialize()
            
            # Créer des fichiers de grande taille simulée
            large_files = []
            for i in range(20):
                audio_file = tmp_path / f"large_test_{i}.wav"
                audio_file.write_bytes(b'MOCK_LARGE_AUDIO_DATA' * 1000)  # Plus gros
                large_files.append(str(audio_file))
            
            # Mock pour simuler des données volumineuses
            def mock_large_load(*args, **kwargs):
                # Simuler un fichier de 10 minutes
                return np.random.random(44100 * 600), 44100
            
            with patch('librosa.load', side_effect=mock_large_load), \
                 patch('soundfile.write') as mock_write, \
                 patch('librosa.stft') as mock_stft, \
                 patch('librosa.istft') as mock_istft:
                
                mock_write.return_value = None
                mock_stft.return_value = np.random.random((513, 2048)) + 1j * np.random.random((513, 2048))
                mock_istft.return_value = np.random.random(44100 * 600)
                
                # Surveiller la mémoire
                process = psutil.Process()
                max_memory = 0
                
                # Traitement avec surveillance mémoire
                async def process_with_monitoring():
                    nonlocal max_memory
                    results = await engine.batch_separate(
                        audio_files=large_files,
                        model_name="spleeter:2stems-16kHz",
                        output_dir=str(tmp_path / "memory_stress_output")
                    )
                    
                    current_memory = process.memory_info().rss / 1024 / 1024
                    max_memory = max(max_memory, current_memory)
                    
                    return results
                
                # Exécuter avec surveillance
                start_memory = process.memory_info().rss / 1024 / 1024
                results = await process_with_monitoring()
                final_memory = process.memory_info().rss / 1024 / 1024
                
                print(f"\nMemory Pressure Stress Test:")
                print(f"  Large files: {len(large_files)}")
                print(f"  Start memory: {start_memory:.1f}MB")
                print(f"  Peak memory: {max_memory:.1f}MB")
                print(f"  Final memory: {final_memory:.1f}MB")
                print(f"  Successful: {sum(1 for r in results if r.success)}/{len(results)}")
                
                # Vérifications mémoire
                memory_growth = final_memory - start_memory
                assert memory_growth < 1000, f"Excessive memory growth: {memory_growth}MB"
                assert max_memory < 2000, f"Peak memory too high: {max_memory}MB"
            
            await engine.cleanup()
    
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_cache_stress_test(self, tmp_path):
        """Test de stress du système de cache"""
        cache_manager = CacheManager(
            cache_dir=str(tmp_path / "cache_stress"),
            memory_cache_size=50,  # Limité pour forcer l'éviction
            disk_cache_size_mb=10,  # Petit pour tester les limites
            enable_redis=False
        )
        
        await cache_manager.initialize()
        
        # Test avec beaucoup d'entrées
        num_entries = 1000
        entry_size = 1024  # 1KB par entrée
        
        # Phase 1: Écriture intensive
        write_start = time.perf_counter()
        for i in range(num_entries):
            data = {'id': i, 'data': np.random.random(entry_size).tolist()}
            await cache_manager.set(f"stress_key_{i}", data, ttl=300)
            
            # Vérifier périodiquement
            if i % 100 == 0:
                stats = cache_manager.get_cache_stats()
                print(f"Progress: {i}/{num_entries}, Memory entries: {stats['memory_cache']['size']}")
        
        write_time = time.perf_counter() - write_start
        
        # Phase 2: Lecture intensive avec accès aléatoire
        import random
        read_keys = random.sample(range(num_entries), min(200, num_entries))
        
        read_start = time.perf_counter()
        cache_hits = 0
        for key_id in read_keys:
            result = await cache_manager.get(f"stress_key_{key_id}")
            if result is not None:
                cache_hits += 1
        
        read_time = time.perf_counter() - read_start
        hit_rate = (cache_hits / len(read_keys)) * 100
        
        # Statistiques finales
        final_stats = cache_manager.get_cache_stats()
        
        print(f"\nCache Stress Test Results:")
        print(f"  Entries written: {num_entries}")
        print(f"  Write time: {write_time:.2f}s")
        print(f"  Write rate: {num_entries/write_time:.1f} ops/s")
        print(f"  Read samples: {len(read_keys)}")
        print(f"  Read time: {read_time:.2f}s")
        print(f"  Read rate: {len(read_keys)/read_time:.1f} ops/s")
        print(f"  Cache hit rate: {hit_rate:.1f}%")
        print(f"  Final memory entries: {final_stats['memory_cache']['size']}")
        
        # Vérifications
        assert write_time < 60.0, f"Write too slow: {write_time:.2f}s"
        assert read_time < 10.0, f"Read too slow: {read_time:.2f}s"
        assert hit_rate > 30.0, f"Hit rate too low: {hit_rate:.1f}%"
        
        await cache_manager.cleanup()


class TestResourceUtilization:
    """Tests d'utilisation des ressources"""
    
    @pytest.mark.performance
    def test_cpu_utilization_monitoring(self):
        """Test de surveillance de l'utilisation CPU"""
        from spleeter.monitoring import MetricsCollector
        
        collector = MetricsCollector(enable_system_metrics=True)
        
        # Simuler une charge CPU
        def cpu_intensive_work():
            total = 0
            for i in range(1000000):
                total += i * i
            return total
        
        # Mesurer avec monitoring
        with patch('time.sleep'):  # Accélérer les mesures
            start_time = time.perf_counter()
            
            # Travail intensif
            for _ in range(5):
                cpu_intensive_work()
            
            work_time = time.perf_counter() - start_time
            
            # Attendre un peu pour les métriques
            time.sleep(0.1)
        
        # Vérifier les métriques
        stats = collector.get_stats_summary()
        
        print(f"\nCPU Utilization Test:")
        print(f"  Work time: {work_time:.3f}s")
        print(f"  CPU metrics collected: {'system_cpu_usage' in stats['recent_metrics']}")
        
        if 'system_cpu_usage' in stats['recent_metrics']:
            cpu_usage = stats['recent_metrics']['system_cpu_usage_avg']
            print(f"  Average CPU usage: {cpu_usage:.1f}%")
            
            # Vérifications
            assert cpu_usage >= 0, "Invalid CPU usage"
            assert cpu_usage <= 100, "CPU usage over 100%"
        
        collector.stop_system_monitoring()
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_resource_monitoring_integration(self, tmp_path):
        """Test d'intégration de surveillance des ressources"""
        from spleeter.monitoring import ResourceMonitor
        
        monitor = ResourceMonitor(
            memory_threshold_mb=500,
            gpu_threshold_percent=80
        )
        
        # Simuler une opération qui utilise des ressources
        async with monitor.monitor_operation("resource_test"):
            # Simuler utilisation mémoire
            large_arrays = []
            for i in range(10):
                array = np.random.random((100, 100, 10))  # ~8MB
                large_arrays.append(array)
                await asyncio.sleep(0.01)
            
            # Simuler travail CPU
            for _ in range(5):
                _ = sum(range(10000))
                await asyncio.sleep(0.01)
            
            # Nettoyer progressivement
            for _ in range(5):
                if large_arrays:
                    large_arrays.pop()
                await asyncio.sleep(0.01)
        
        print(f"\nResource Monitoring Test:")
        print(f"  Monitoring completed successfully")
        
        # Le monitoring devrait s'être arrêté proprement
        assert not monitor.monitoring
    
    @pytest.mark.performance
    def test_performance_optimizer_recommendations(self):
        """Test des recommandations de l'optimiseur de performance"""
        from spleeter.utils import PerformanceOptimizer
        
        # Obtenir la configuration optimale
        optimal_config = PerformanceOptimizer.detect_optimal_config()
        
        print(f"\nPerformance Optimizer Recommendations:")
        print(f"  CPU cores: {optimal_config['cpu_count']}")
        print(f"  GPU available: {optimal_config['gpu_available']}")
        print(f"  Memory (GB): {optimal_config['memory_gb']:.1f}")
        print(f"  Recommended batch size: {optimal_config['recommended_batch_size']}")
        print(f"  Recommended workers: {optimal_config['recommended_workers']}")
        print(f"  Enable GPU: {optimal_config['enable_gpu']}")
        
        # Vérifications
        assert optimal_config['cpu_count'] > 0
        assert optimal_config['memory_gb'] > 0
        assert optimal_config['recommended_batch_size'] > 0
        assert optimal_config['recommended_workers'] > 0
        
        # Test d'estimation de temps
        audio_duration = 180.0  # 3 minutes
        estimated_time = PerformanceOptimizer.estimate_processing_time(
            audio_duration=audio_duration,
            model_complexity="2stems",
            use_gpu=optimal_config['gpu_available']
        )
        
        print(f"  Estimated processing time for 3min audio: {estimated_time:.1f}s")
        
        assert estimated_time > 0
        assert estimated_time < audio_duration * 10  # Pas plus de 10x temps réel
        
        # Test des besoins mémoire
        memory_req = PerformanceOptimizer.get_memory_requirements(
            audio_duration=audio_duration,
            sample_rate=44100,
            model_complexity="2stems"
        )
        
        print(f"  Memory requirements:")
        print(f"    Audio: {memory_req['audio_mb']:.1f}MB")
        print(f"    Model: {memory_req['model_mb']:.1f}MB")
        print(f"    Processing: {memory_req['processing_mb']:.1f}MB")
        print(f"    Total: {memory_req['total_mb']:.1f}MB")
        
        assert all(req > 0 for req in memory_req.values())
        assert memory_req['total_mb'] < 10000  # Moins de 10GB


# Utilitaires pour les tests de performance
def measure_time(func):
    """Décorateur pour mesurer le temps d'exécution"""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper


def measure_memory(func):
    """Décorateur pour mesurer l'utilisation mémoire"""
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        result = func(*args, **kwargs)
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"{func.__name__} used {end_memory - start_memory:.1f}MB")
        return result
    return wrapper
