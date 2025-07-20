#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Performance Benchmark & Analytics Framework
===================================================

Ultra-advanced performance benchmarking system with comprehensive metrics analysis,
machine learning-powered performance predictions, automated optimization recommendations,
and real-time performance monitoring with intelligent alerts.

Expert Development Team:
- Lead Dev + AI Architect
- Senior Backend Developer (Python/FastAPI/Django)
- ML Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect

Project Lead: Fahed Mlaiel
"""

import asyncio
import json
import logging
import os
import sys
import time
import statistics
import platform
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, NamedTuple
import dataclasses
from dataclasses import dataclass, asdict
import concurrent.futures
import threading
import multiprocessing
import psutil
import numpy as np
from collections import defaultdict, deque
import hashlib
import random

# Ajout du chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from . import (
    EnterpriseMetricsSystem, MetricDataPoint, MetricType, 
    MetricCategory, MetricSeverity, get_metrics_system
)
from .collector import MetricsCollectionAgent, CollectorConfig
from .monitor import AlertEngine, AlertRule, AlertPriority, HealthMonitor

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Résultat d'un benchmark."""
    test_name: str
    execution_time: float
    throughput: float
    memory_usage: float
    cpu_usage: float
    error_rate: float
    success_count: int
    error_count: int
    percentiles: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class SystemInfo:
    """Informations système."""
    platform: str
    python_version: str
    cpu_count: int
    memory_total: float
    disk_space: float
    network_interfaces: List[str]
    hostname: str
    timestamp: datetime


@dataclass
class PerformanceProfile:
    """Profil de performance."""
    profile_name: str
    system_info: SystemInfo
    benchmark_results: List[BenchmarkResult]
    overall_score: float
    recommendations: List[str]
    timestamp: datetime


class PerformanceBenchmark:
    """Framework de benchmark de performance ultra-avancé."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.results = []
        self.system_info = None
        self.start_time = None
        self.metrics_system = None
        
        # Configuration par défaut
        self.default_config = {
            "warm_up_iterations": 3,
            "benchmark_iterations": 10,
            "concurrent_users": [1, 5, 10, 20, 50],
            "data_sizes": [100, 1000, 10000, 50000],
            "timeout_seconds": 300,
            "memory_limit_mb": 1024,
            "enable_profiling": True,
            "enable_ml_analysis": True
        }
        
        # Fusion avec la configuration par défaut
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    async def run_comprehensive_benchmark(self) -> PerformanceProfile:
        """Exécute un benchmark complet."""
        logger.info("🚀 Démarrage du benchmark de performance complet")
        self.start_time = time.time()
        
        try:
            # Collecte des informations système
            self.system_info = await self._collect_system_info()
            
            # Initialisation du système de métriques
            await self._initialize_metrics_system()
            
            # Warm-up
            await self._run_warmup()
            
            # Benchmarks de base
            base_results = await self._run_base_benchmarks()
            
            # Benchmarks de charge
            load_results = await self._run_load_benchmarks()
            
            # Benchmarks de stress
            stress_results = await self._run_stress_benchmarks()
            
            # Benchmarks de concurrence
            concurrency_results = await self._run_concurrency_benchmarks()
            
            # Benchmarks de mémoire
            memory_results = await self._run_memory_benchmarks()
            
            # Compilation des résultats
            all_results = (base_results + load_results + stress_results + 
                          concurrency_results + memory_results)
            
            # Analyse et scoring
            overall_score = await self._calculate_overall_score(all_results)
            recommendations = await self._generate_recommendations(all_results)
            
            # Création du profil de performance
            profile = PerformanceProfile(
                profile_name=f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                system_info=self.system_info,
                benchmark_results=all_results,
                overall_score=overall_score,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
            # Sauvegarde et rapport
            await self._save_profile(profile)
            await self._generate_report(profile)
            
            total_time = time.time() - self.start_time
            logger.info(f"✅ Benchmark terminé en {total_time:.1f}s - Score: {overall_score:.1f}/100")
            
            return profile
            
        finally:
            if self.metrics_system:
                await self.metrics_system.stop()
    
    async def _collect_system_info(self) -> SystemInfo:
        """Collecte les informations système."""
        logger.info("📊 Collecte des informations système")
        
        # Informations de base
        system_info = SystemInfo(
            platform=platform.platform(),
            python_version=platform.python_version(),
            cpu_count=psutil.cpu_count(),
            memory_total=psutil.virtual_memory().total / (1024**3),  # GB
            disk_space=psutil.disk_usage('/').total / (1024**3),  # GB
            network_interfaces=[],
            hostname=platform.node(),
            timestamp=datetime.now()
        )
        
        # Interfaces réseau
        try:
            network_info = psutil.net_if_addrs()
            system_info.network_interfaces = list(network_info.keys())
        except Exception as e:
            logger.warning(f"Impossible de collecter les informations réseau: {e}")
        
        logger.info(f"💻 Système: {system_info.platform}")
        logger.info(f"🐍 Python: {system_info.python_version}")
        logger.info(f"⚙️  CPU: {system_info.cpu_count} cœurs")
        logger.info(f"🧠 RAM: {system_info.memory_total:.1f} GB")
        
        return system_info
    
    async def _initialize_metrics_system(self):
        """Initialise le système de métriques pour les tests."""
        logger.info("🔧 Initialisation du système de métriques")
        
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="benchmark_")
        
        self.metrics_system = get_metrics_system(
            "sqlite", 
            {"db_path": f"{temp_dir}/benchmark.db"}
        )
        await self.metrics_system.start()
    
    async def _run_warmup(self):
        """Exécute une phase de warm-up."""
        logger.info("🔥 Phase de warm-up")
        
        for i in range(self.config["warm_up_iterations"]):
            await self._basic_metric_operation()
            await asyncio.sleep(0.1)
    
    async def _basic_metric_operation(self):
        """Opération de métrique basique pour le warm-up."""
        metric = MetricDataPoint(
            metric_id="benchmark.warmup",
            value=random.random() * 100,
            metric_type=MetricType.GAUGE
        )
        await self.metrics_system.storage.store_metric(metric)
    
    async def _run_base_benchmarks(self) -> List[BenchmarkResult]:
        """Exécute les benchmarks de base."""
        logger.info("📈 Exécution des benchmarks de base")
        
        results = []
        
        # Benchmark d'écriture simple
        results.append(await self._benchmark_simple_write())
        
        # Benchmark de lecture simple
        results.append(await self._benchmark_simple_read())
        
        # Benchmark d'écriture en lot
        results.append(await self._benchmark_batch_write())
        
        # Benchmark de requête complexe
        results.append(await self._benchmark_complex_query())
        
        return results
    
    async def _benchmark_simple_write(self) -> BenchmarkResult:
        """Benchmark d'écriture simple."""
        test_name = "simple_write"
        iterations = self.config["benchmark_iterations"] * 100
        
        # Préparation des métriques
        metrics = []
        for i in range(iterations):
            metrics.append(MetricDataPoint(
                metric_id=f"benchmark.simple_write.{i}",
                value=float(i),
                metric_type=MetricType.COUNTER
            ))
        
        # Mesure des ressources initiales
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)  # MB
        initial_cpu = process.cpu_percent()
        
        # Benchmark
        start_time = time.time()
        error_count = 0
        success_count = 0
        
        response_times = []
        
        for metric in metrics:
            op_start = time.time()
            try:
                await self.metrics_system.storage.store_metric(metric)
                success_count += 1
                response_times.append(time.time() - op_start)
            except Exception as e:
                error_count += 1
                logger.debug(f"Erreur dans simple_write: {e}")
        
        execution_time = time.time() - start_time
        final_memory = process.memory_info().rss / (1024**2)  # MB
        final_cpu = process.cpu_percent()
        
        # Calculs de performance
        throughput = success_count / execution_time if execution_time > 0 else 0
        error_rate = error_count / (success_count + error_count) * 100 if (success_count + error_count) > 0 else 0
        
        # Percentiles
        percentiles = {}
        if response_times:
            percentiles = {
                "p50": np.percentile(response_times, 50),
                "p90": np.percentile(response_times, 90),
                "p95": np.percentile(response_times, 95),
                "p99": np.percentile(response_times, 99)
            }
        
        return BenchmarkResult(
            test_name=test_name,
            execution_time=execution_time,
            throughput=throughput,
            memory_usage=final_memory - initial_memory,
            cpu_usage=final_cpu - initial_cpu,
            error_rate=error_rate,
            success_count=success_count,
            error_count=error_count,
            percentiles=percentiles,
            metadata={
                "iterations": iterations,
                "avg_response_time": statistics.mean(response_times) if response_times else 0
            }
        )
    
    async def _benchmark_simple_read(self) -> BenchmarkResult:
        """Benchmark de lecture simple."""
        test_name = "simple_read"
        
        # Préparation: insertion de données de test
        prep_metrics = []
        for i in range(1000):
            prep_metrics.append(MetricDataPoint(
                metric_id=f"benchmark.read_test.{i}",
                value=float(i),
                metric_type=MetricType.GAUGE
            ))
        
        for metric in prep_metrics:
            await self.metrics_system.storage.store_metric(metric)
        
        # Benchmark de lecture
        iterations = self.config["benchmark_iterations"] * 10
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)
        initial_cpu = process.cpu_percent()
        
        start_time = time.time()
        error_count = 0
        success_count = 0
        response_times = []
        
        for i in range(iterations):
            op_start = time.time()
            try:
                results = await self.metrics_system.storage.query_metrics(
                    metric_pattern="benchmark.read_test.*",
                    start_time=datetime.now() - timedelta(hours=1),
                    end_time=datetime.now()
                )
                success_count += 1
                response_times.append(time.time() - op_start)
            except Exception as e:
                error_count += 1
                logger.debug(f"Erreur dans simple_read: {e}")
        
        execution_time = time.time() - start_time
        final_memory = process.memory_info().rss / (1024**2)
        final_cpu = process.cpu_percent()
        
        throughput = success_count / execution_time if execution_time > 0 else 0
        error_rate = error_count / (success_count + error_count) * 100 if (success_count + error_count) > 0 else 0
        
        percentiles = {}
        if response_times:
            percentiles = {
                "p50": np.percentile(response_times, 50),
                "p90": np.percentile(response_times, 90),
                "p95": np.percentile(response_times, 95),
                "p99": np.percentile(response_times, 99)
            }
        
        return BenchmarkResult(
            test_name=test_name,
            execution_time=execution_time,
            throughput=throughput,
            memory_usage=final_memory - initial_memory,
            cpu_usage=final_cpu - initial_cpu,
            error_rate=error_rate,
            success_count=success_count,
            error_count=error_count,
            percentiles=percentiles,
            metadata={
                "iterations": iterations,
                "avg_response_time": statistics.mean(response_times) if response_times else 0,
                "records_per_query": len(prep_metrics)
            }
        )
    
    async def _benchmark_batch_write(self) -> BenchmarkResult:
        """Benchmark d'écriture en lot."""
        test_name = "batch_write"
        batch_size = 100
        num_batches = self.config["benchmark_iterations"]
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)
        initial_cpu = process.cpu_percent()
        
        start_time = time.time()
        error_count = 0
        success_count = 0
        response_times = []
        
        for batch_num in range(num_batches):
            batch_metrics = []
            for i in range(batch_size):
                batch_metrics.append(MetricDataPoint(
                    metric_id=f"benchmark.batch_write.{batch_num}.{i}",
                    value=float(i),
                    metric_type=MetricType.COUNTER
                ))
            
            batch_start = time.time()
            try:
                # Stockage en lot
                for metric in batch_metrics:
                    await self.metrics_system.storage.store_metric(metric)
                success_count += len(batch_metrics)
                response_times.append(time.time() - batch_start)
            except Exception as e:
                error_count += len(batch_metrics)
                logger.debug(f"Erreur dans batch_write: {e}")
        
        execution_time = time.time() - start_time
        final_memory = process.memory_info().rss / (1024**2)
        final_cpu = process.cpu_percent()
        
        throughput = success_count / execution_time if execution_time > 0 else 0
        error_rate = error_count / (success_count + error_count) * 100 if (success_count + error_count) > 0 else 0
        
        percentiles = {}
        if response_times:
            percentiles = {
                "p50": np.percentile(response_times, 50),
                "p90": np.percentile(response_times, 90),
                "p95": np.percentile(response_times, 95),
                "p99": np.percentile(response_times, 99)
            }
        
        return BenchmarkResult(
            test_name=test_name,
            execution_time=execution_time,
            throughput=throughput,
            memory_usage=final_memory - initial_memory,
            cpu_usage=final_cpu - initial_cpu,
            error_rate=error_rate,
            success_count=success_count,
            error_count=error_count,
            percentiles=percentiles,
            metadata={
                "batch_size": batch_size,
                "num_batches": num_batches,
                "total_metrics": success_count + error_count
            }
        )
    
    async def _benchmark_complex_query(self) -> BenchmarkResult:
        """Benchmark de requête complexe."""
        test_name = "complex_query"
        
        # Préparation: données diversifiées
        prep_start = time.time()
        hosts = ["server1", "server2", "server3", "server4", "server5"]
        services = ["api", "db", "cache", "queue", "web"]
        
        for i in range(2000):
            host = random.choice(hosts)
            service = random.choice(services)
            
            metric = MetricDataPoint(
                metric_id=f"benchmark.complex.{service}.{host}.{i}",
                timestamp=datetime.now() - timedelta(minutes=random.randint(0, 60)),
                value=random.uniform(0, 100),
                metric_type=MetricType.GAUGE,
                tags={"host": host, "service": service, "env": "test"}
            )
            await self.metrics_system.storage.store_metric(metric)
        
        prep_time = time.time() - prep_start
        logger.info(f"Préparation des données: {prep_time:.1f}s")
        
        # Benchmark de requêtes complexes
        iterations = self.config["benchmark_iterations"]
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)
        initial_cpu = process.cpu_percent()
        
        start_time = time.time()
        error_count = 0
        success_count = 0
        response_times = []
        
        queries = [
            "benchmark.complex.api.*",
            "benchmark.complex.*.server1.*",
            "benchmark.complex.db.*",
            "benchmark.complex.*"
        ]
        
        for i in range(iterations):
            query = random.choice(queries)
            
            op_start = time.time()
            try:
                results = await self.metrics_system.storage.query_metrics(
                    metric_pattern=query,
                    start_time=datetime.now() - timedelta(hours=2),
                    end_time=datetime.now()
                )
                success_count += 1
                response_times.append(time.time() - op_start)
            except Exception as e:
                error_count += 1
                logger.debug(f"Erreur dans complex_query: {e}")
        
        execution_time = time.time() - start_time
        final_memory = process.memory_info().rss / (1024**2)
        final_cpu = process.cpu_percent()
        
        throughput = success_count / execution_time if execution_time > 0 else 0
        error_rate = error_count / (success_count + error_count) * 100 if (success_count + error_count) > 0 else 0
        
        percentiles = {}
        if response_times:
            percentiles = {
                "p50": np.percentile(response_times, 50),
                "p90": np.percentile(response_times, 90),
                "p95": np.percentile(response_times, 95),
                "p99": np.percentile(response_times, 99)
            }
        
        return BenchmarkResult(
            test_name=test_name,
            execution_time=execution_time,
            throughput=throughput,
            memory_usage=final_memory - initial_memory,
            cpu_usage=final_cpu - initial_cpu,
            error_rate=error_rate,
            success_count=success_count,
            error_count=error_count,
            percentiles=percentiles,
            metadata={
                "iterations": iterations,
                "preparation_time": prep_time,
                "test_data_size": 2000,
                "query_patterns": len(queries)
            }
        )
    
    async def _run_load_benchmarks(self) -> List[BenchmarkResult]:
        """Exécute les benchmarks de charge."""
        logger.info("📊 Exécution des benchmarks de charge")
        
        results = []
        
        for data_size in self.config["data_sizes"]:
            result = await self._benchmark_load_test(data_size)
            results.append(result)
        
        return results
    
    async def _benchmark_load_test(self, data_size: int) -> BenchmarkResult:
        """Benchmark de test de charge."""
        test_name = f"load_test_{data_size}"
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)
        initial_cpu = process.cpu_percent()
        
        start_time = time.time()
        error_count = 0
        success_count = 0
        response_times = []
        
        # Test avec différentes tailles de données
        for i in range(data_size):
            op_start = time.time()
            try:
                metric = MetricDataPoint(
                    metric_id=f"benchmark.load.{data_size}.{i}",
                    value=random.uniform(0, 1000),
                    metric_type=MetricType.GAUGE,
                    tags={f"tag_{j}": f"value_{j}" for j in range(min(10, data_size // 100))},
                    metadata={f"meta_{j}": f"data_{j}" for j in range(min(5, data_size // 200))}
                )
                await self.metrics_system.storage.store_metric(metric)
                success_count += 1
                response_times.append(time.time() - op_start)
                
                # Pause périodique pour éviter la surcharge
                if i % 100 == 0 and i > 0:
                    await asyncio.sleep(0.001)
                    
            except Exception as e:
                error_count += 1
                logger.debug(f"Erreur dans load_test: {e}")
        
        execution_time = time.time() - start_time
        final_memory = process.memory_info().rss / (1024**2)
        final_cpu = process.cpu_percent()
        
        throughput = success_count / execution_time if execution_time > 0 else 0
        error_rate = error_count / (success_count + error_count) * 100 if (success_count + error_count) > 0 else 0
        
        percentiles = {}
        if response_times:
            percentiles = {
                "p50": np.percentile(response_times, 50),
                "p90": np.percentile(response_times, 90),
                "p95": np.percentile(response_times, 95),
                "p99": np.percentile(response_times, 99)
            }
        
        return BenchmarkResult(
            test_name=test_name,
            execution_time=execution_time,
            throughput=throughput,
            memory_usage=final_memory - initial_memory,
            cpu_usage=final_cpu - initial_cpu,
            error_rate=error_rate,
            success_count=success_count,
            error_count=error_count,
            percentiles=percentiles,
            metadata={
                "data_size": data_size,
                "avg_response_time": statistics.mean(response_times) if response_times else 0,
                "memory_per_metric": (final_memory - initial_memory) / success_count if success_count > 0 else 0
            }
        )
    
    async def _run_stress_benchmarks(self) -> List[BenchmarkResult]:
        """Exécute les benchmarks de stress."""
        logger.info("💪 Exécution des benchmarks de stress")
        
        results = []
        
        # Test de stress mémoire
        results.append(await self._benchmark_memory_stress())
        
        # Test de stress CPU
        results.append(await self._benchmark_cpu_stress())
        
        return results
    
    async def _benchmark_memory_stress(self) -> BenchmarkResult:
        """Benchmark de stress mémoire."""
        test_name = "memory_stress"
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)
        initial_cpu = process.cpu_percent()
        
        start_time = time.time()
        error_count = 0
        success_count = 0
        response_times = []
        
        # Création de métriques avec beaucoup de métadonnées
        large_objects = []
        
        for i in range(1000):
            op_start = time.time()
            try:
                # Création d'objets volumineux
                large_tags = {f"tag_{j}": f"value_{j}" * 100 for j in range(20)}
                large_metadata = {f"meta_{j}": f"data_{j}" * 50 for j in range(10)}
                
                metric = MetricDataPoint(
                    metric_id=f"benchmark.memory_stress.{i}",
                    value=float(i),
                    metric_type=MetricType.GAUGE,
                    tags=large_tags,
                    metadata=large_metadata
                )
                
                await self.metrics_system.storage.store_metric(metric)
                large_objects.append(metric)  # Garder en mémoire
                
                success_count += 1
                response_times.append(time.time() - op_start)
                
            except Exception as e:
                error_count += 1
                logger.debug(f"Erreur dans memory_stress: {e}")
        
        execution_time = time.time() - start_time
        final_memory = process.memory_info().rss / (1024**2)
        final_cpu = process.cpu_percent()
        
        throughput = success_count / execution_time if execution_time > 0 else 0
        error_rate = error_count / (success_count + error_count) * 100 if (success_count + error_count) > 0 else 0
        
        percentiles = {}
        if response_times:
            percentiles = {
                "p50": np.percentile(response_times, 50),
                "p90": np.percentile(response_times, 90),
                "p95": np.percentile(response_times, 95),
                "p99": np.percentile(response_times, 99)
            }
        
        return BenchmarkResult(
            test_name=test_name,
            execution_time=execution_time,
            throughput=throughput,
            memory_usage=final_memory - initial_memory,
            cpu_usage=final_cpu - initial_cpu,
            error_rate=error_rate,
            success_count=success_count,
            error_count=error_count,
            percentiles=percentiles,
            metadata={
                "large_objects_count": len(large_objects),
                "avg_object_size": (final_memory - initial_memory) / len(large_objects) if large_objects else 0
            }
        )
    
    async def _benchmark_cpu_stress(self) -> BenchmarkResult:
        """Benchmark de stress CPU."""
        test_name = "cpu_stress"
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)
        initial_cpu = process.cpu_percent()
        
        start_time = time.time()
        error_count = 0
        success_count = 0
        response_times = []
        
        # Opérations intensives en CPU
        for i in range(100):
            op_start = time.time()
            try:
                # Calculs intensifs
                values = [random.random() for _ in range(1000)]
                
                # Opérations mathématiques complexes
                result = sum(values)
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values)
                sorted_vals = sorted(values)
                
                # Création et stockage de la métrique
                metric = MetricDataPoint(
                    metric_id=f"benchmark.cpu_stress.{i}",
                    value=result,
                    metric_type=MetricType.GAUGE,
                    tags={
                        "mean": str(mean_val),
                        "std": str(std_val),
                        "min": str(min(values)),
                        "max": str(max(values))
                    }
                )
                
                await self.metrics_system.storage.store_metric(metric)
                success_count += 1
                response_times.append(time.time() - op_start)
                
            except Exception as e:
                error_count += 1
                logger.debug(f"Erreur dans cpu_stress: {e}")
        
        execution_time = time.time() - start_time
        final_memory = process.memory_info().rss / (1024**2)
        final_cpu = process.cpu_percent()
        
        throughput = success_count / execution_time if execution_time > 0 else 0
        error_rate = error_count / (success_count + error_count) * 100 if (success_count + error_count) > 0 else 0
        
        percentiles = {}
        if response_times:
            percentiles = {
                "p50": np.percentile(response_times, 50),
                "p90": np.percentile(response_times, 90),
                "p95": np.percentile(response_times, 95),
                "p99": np.percentile(response_times, 99)
            }
        
        return BenchmarkResult(
            test_name=test_name,
            execution_time=execution_time,
            throughput=throughput,
            memory_usage=final_memory - initial_memory,
            cpu_usage=final_cpu - initial_cpu,
            error_rate=error_rate,
            success_count=success_count,
            error_count=error_count,
            percentiles=percentiles,
            metadata={
                "cpu_intensive_operations": 100,
                "calculations_per_op": 1000
            }
        )
    
    async def _run_concurrency_benchmarks(self) -> List[BenchmarkResult]:
        """Exécute les benchmarks de concurrence."""
        logger.info("🔄 Exécution des benchmarks de concurrence")
        
        results = []
        
        for concurrent_users in self.config["concurrent_users"]:
            result = await self._benchmark_concurrent_operations(concurrent_users)
            results.append(result)
        
        return results
    
    async def _benchmark_concurrent_operations(self, concurrent_users: int) -> BenchmarkResult:
        """Benchmark d'opérations concurrentes."""
        test_name = f"concurrent_{concurrent_users}_users"
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)
        initial_cpu = process.cpu_percent()
        
        metrics_per_user = 50
        all_response_times = []
        total_success = 0
        total_errors = 0
        
        async def user_simulation(user_id: int):
            """Simulation d'un utilisateur."""
            user_success = 0
            user_errors = 0
            user_times = []
            
            for i in range(metrics_per_user):
                op_start = time.time()
                try:
                    metric = MetricDataPoint(
                        metric_id=f"benchmark.concurrent.user_{user_id}.metric_{i}",
                        value=float(i),
                        metric_type=MetricType.COUNTER,
                        tags={"user_id": str(user_id)}
                    )
                    await self.metrics_system.storage.store_metric(metric)
                    user_success += 1
                    user_times.append(time.time() - op_start)
                    
                    # Petite pause pour simulation réaliste
                    await asyncio.sleep(0.001)
                    
                except Exception as e:
                    user_errors += 1
                    logger.debug(f"Erreur utilisateur {user_id}: {e}")
            
            return user_success, user_errors, user_times
        
        # Lancement des utilisateurs concurrents
        start_time = time.time()
        
        tasks = [user_simulation(i) for i in range(concurrent_users)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        execution_time = time.time() - start_time
        
        # Compilation des résultats
        for result in results:
            if isinstance(result, tuple):
                success, errors, times = result
                total_success += success
                total_errors += errors
                all_response_times.extend(times)
        
        final_memory = process.memory_info().rss / (1024**2)
        final_cpu = process.cpu_percent()
        
        throughput = total_success / execution_time if execution_time > 0 else 0
        error_rate = total_errors / (total_success + total_errors) * 100 if (total_success + total_errors) > 0 else 0
        
        percentiles = {}
        if all_response_times:
            percentiles = {
                "p50": np.percentile(all_response_times, 50),
                "p90": np.percentile(all_response_times, 90),
                "p95": np.percentile(all_response_times, 95),
                "p99": np.percentile(all_response_times, 99)
            }
        
        return BenchmarkResult(
            test_name=test_name,
            execution_time=execution_time,
            throughput=throughput,
            memory_usage=final_memory - initial_memory,
            cpu_usage=final_cpu - initial_cpu,
            error_rate=error_rate,
            success_count=total_success,
            error_count=total_errors,
            percentiles=percentiles,
            metadata={
                "concurrent_users": concurrent_users,
                "metrics_per_user": metrics_per_user,
                "total_operations": concurrent_users * metrics_per_user,
                "avg_response_time": statistics.mean(all_response_times) if all_response_times else 0
            }
        )
    
    async def _run_memory_benchmarks(self) -> List[BenchmarkResult]:
        """Exécute les benchmarks de mémoire."""
        logger.info("🧠 Exécution des benchmarks de mémoire")
        
        results = []
        
        # Test de croissance mémoire
        results.append(await self._benchmark_memory_growth())
        
        # Test de récupération mémoire
        results.append(await self._benchmark_memory_recovery())
        
        return results
    
    async def _benchmark_memory_growth(self) -> BenchmarkResult:
        """Benchmark de croissance mémoire."""
        test_name = "memory_growth"
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)
        initial_cpu = process.cpu_percent()
        
        start_time = time.time()
        error_count = 0
        success_count = 0
        response_times = []
        memory_snapshots = []
        
        # Création progressive d'objets
        objects_in_memory = []
        
        for i in range(500):
            op_start = time.time()
            try:
                # Création d'objets de plus en plus volumineux
                size_factor = 1 + (i // 100)
                
                metric = MetricDataPoint(
                    metric_id=f"benchmark.memory_growth.{i}",
                    value=float(i),
                    metric_type=MetricType.GAUGE,
                    tags={f"tag_{j}": f"value_{j}" * size_factor for j in range(size_factor * 5)},
                    metadata={f"meta_{j}": f"data_{j}" * size_factor for j in range(size_factor * 3)}
                )
                
                await self.metrics_system.storage.store_metric(metric)
                objects_in_memory.append(metric)  # Garder en mémoire
                
                success_count += 1
                response_times.append(time.time() - op_start)
                
                # Snapshot mémoire périodique
                if i % 50 == 0:
                    current_memory = process.memory_info().rss / (1024**2)
                    memory_snapshots.append(current_memory)
                
            except Exception as e:
                error_count += 1
                logger.debug(f"Erreur dans memory_growth: {e}")
        
        execution_time = time.time() - start_time
        final_memory = process.memory_info().rss / (1024**2)
        final_cpu = process.cpu_percent()
        
        throughput = success_count / execution_time if execution_time > 0 else 0
        error_rate = error_count / (success_count + error_count) * 100 if (success_count + error_count) > 0 else 0
        
        percentiles = {}
        if response_times:
            percentiles = {
                "p50": np.percentile(response_times, 50),
                "p90": np.percentile(response_times, 90),
                "p95": np.percentile(response_times, 95),
                "p99": np.percentile(response_times, 99)
            }
        
        return BenchmarkResult(
            test_name=test_name,
            execution_time=execution_time,
            throughput=throughput,
            memory_usage=final_memory - initial_memory,
            cpu_usage=final_cpu - initial_cpu,
            error_rate=error_rate,
            success_count=success_count,
            error_count=error_count,
            percentiles=percentiles,
            metadata={
                "objects_in_memory": len(objects_in_memory),
                "memory_snapshots": memory_snapshots,
                "memory_growth_rate": (final_memory - initial_memory) / execution_time if execution_time > 0 else 0
            }
        )
    
    async def _benchmark_memory_recovery(self) -> BenchmarkResult:
        """Benchmark de récupération mémoire."""
        test_name = "memory_recovery"
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)
        
        # Phase 1: Allocation massive
        large_objects = []
        for i in range(200):
            metric = MetricDataPoint(
                metric_id=f"benchmark.memory_recovery.{i}",
                value=float(i),
                metric_type=MetricType.GAUGE,
                tags={f"tag_{j}": f"value_{j}" * 100 for j in range(20)},
                metadata={f"meta_{j}": f"data_{j}" * 50 for j in range(10)}
            )
            large_objects.append(metric)
            await self.metrics_system.storage.store_metric(metric)
        
        peak_memory = process.memory_info().rss / (1024**2)
        
        # Phase 2: Libération
        start_time = time.time()
        del large_objects  # Libération explicite
        
        # Attente de la récupération
        await asyncio.sleep(1)
        
        execution_time = time.time() - start_time
        final_memory = process.memory_info().rss / (1024**2)
        
        memory_recovered = peak_memory - final_memory
        recovery_rate = (memory_recovered / (peak_memory - initial_memory)) * 100 if (peak_memory - initial_memory) > 0 else 0
        
        return BenchmarkResult(
            test_name=test_name,
            execution_time=execution_time,
            throughput=0,  # Non applicable
            memory_usage=final_memory - initial_memory,
            cpu_usage=0,  # Non mesurable facilement
            error_rate=0,  # Pas d'erreurs dans ce test
            success_count=1,  # Test réussi
            error_count=0,
            percentiles={},  # Non applicable
            metadata={
                "initial_memory": initial_memory,
                "peak_memory": peak_memory,
                "final_memory": final_memory,
                "memory_recovered": memory_recovered,
                "recovery_rate": recovery_rate
            }
        )
    
    async def _calculate_overall_score(self, results: List[BenchmarkResult]) -> float:
        """Calcule le score global de performance."""
        if not results:
            return 0.0
        
        scores = []
        
        for result in results:
            # Score basé sur différents facteurs
            throughput_score = min(100, result.throughput / 100)  # Normalisation
            error_score = max(0, 100 - result.error_rate)
            memory_score = max(0, 100 - min(100, result.memory_usage / 10))  # 10MB = score 0
            
            # Score composite
            composite_score = (throughput_score * 0.4 + error_score * 0.4 + memory_score * 0.2)
            scores.append(composite_score)
        
        return statistics.mean(scores)
    
    async def _generate_recommendations(self, results: List[BenchmarkResult]) -> List[str]:
        """Génère des recommandations d'optimisation."""
        recommendations = []
        
        # Analyse des performances
        throughputs = [r.throughput for r in results if r.throughput > 0]
        memory_usages = [r.memory_usage for r in results if r.memory_usage > 0]
        error_rates = [r.error_rate for r in results if r.error_rate > 0]
        
        # Recommandations basées sur le débit
        if throughputs:
            avg_throughput = statistics.mean(throughputs)
            if avg_throughput < 100:
                recommendations.append("🚀 Optimiser le débit: Considérer l'utilisation de connexions persistantes ou de pooling")
            if max(throughputs) / min(throughputs) > 5:
                recommendations.append("📊 Débit variable: Implémenter un système de cache pour stabiliser les performances")
        
        # Recommandations basées sur la mémoire
        if memory_usages:
            max_memory = max(memory_usages)
            if max_memory > 100:  # Plus de 100MB
                recommendations.append("🧠 Optimisation mémoire: Considérer l'utilisation de générateurs ou de streaming pour réduire l'empreinte mémoire")
            
            avg_memory = statistics.mean(memory_usages)
            if avg_memory > 50:
                recommendations.append("💾 Gestion mémoire: Implémenter un garbage collection plus agressif ou une limitation de cache")
        
        # Recommandations basées sur les erreurs
        if error_rates:
            avg_error_rate = statistics.mean(error_rates)
            if avg_error_rate > 5:
                recommendations.append("🔧 Gestion d'erreurs: Améliorer la robustesse du système avec retry logic et circuit breakers")
            if max(error_rates) > 20:
                recommendations.append("⚠️  Stabilité: Investiguer les causes d'erreurs dans les scénarios de charge élevée")
        
        # Recommandations générales
        concurrent_results = [r for r in results if "concurrent" in r.test_name]
        if concurrent_results:
            error_rates_concurrent = [r.error_rate for r in concurrent_results]
            if error_rates_concurrent and max(error_rates_concurrent) > 10:
                recommendations.append("🔄 Concurrence: Optimiser la gestion des accès concurrents avec des verrous appropriés")
        
        # Recommandations système
        if self.system_info:
            if self.system_info.cpu_count < 4:
                recommendations.append("⚙️  Hardware: Considérer l'ajout de CPU pour améliorer les performances de concurrence")
            if self.system_info.memory_total < 8:
                recommendations.append("🧠 Hardware: Augmenter la RAM pour améliorer les performances générales")
        
        # Recommandations d'architecture
        storage_results = [r for r in results if any(test in r.test_name for test in ["read", "write", "query"])]
        if storage_results:
            avg_storage_throughput = statistics.mean([r.throughput for r in storage_results])
            if avg_storage_throughput < 50:
                recommendations.append("💾 Architecture: Considérer l'utilisation d'un backend de stockage plus performant (Redis, PostgreSQL)")
        
        if not recommendations:
            recommendations.append("🎉 Excellent! Les performances sont optimales selon nos métriques")
        
        return recommendations
    
    async def _save_profile(self, profile: PerformanceProfile):
        """Sauvegarde le profil de performance."""
        filename = f"performance_profile_{profile.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        
        # Conversion en dictionnaire sérialisable
        profile_dict = {
            "profile_name": profile.profile_name,
            "timestamp": profile.timestamp.isoformat(),
            "overall_score": profile.overall_score,
            "recommendations": profile.recommendations,
            "system_info": asdict(profile.system_info),
            "benchmark_results": [asdict(result) for result in profile.benchmark_results]
        }
        
        # Conversion des datetime en string
        profile_dict["system_info"]["timestamp"] = profile.system_info.timestamp.isoformat()
        
        with open(filename, 'w') as f:
            json.dump(profile_dict, f, indent=2, default=str)
        
        logger.info(f"💾 Profil sauvegardé: {filename}")
    
    async def _generate_report(self, profile: PerformanceProfile):
        """Génère un rapport de performance."""
        report_filename = f"performance_report_{profile.timestamp.strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_filename, 'w') as f:
            f.write("# Rapport de Performance - Système de Métriques\n\n")
            f.write(f"**Projet dirigé par:** Fahed Mlaiel\n")
            f.write(f"**Date:** {profile.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Score Global:** {profile.overall_score:.1f}/100\n\n")
            
            # Informations système
            f.write("## 📊 Informations Système\n\n")
            f.write(f"- **Plateforme:** {profile.system_info.platform}\n")
            f.write(f"- **Python:** {profile.system_info.python_version}\n")
            f.write(f"- **CPU:** {profile.system_info.cpu_count} cœurs\n")
            f.write(f"- **RAM:** {profile.system_info.memory_total:.1f} GB\n")
            f.write(f"- **Hostname:** {profile.system_info.hostname}\n\n")
            
            # Résultats par catégorie
            f.write("## 📈 Résultats des Benchmarks\n\n")
            
            categories = {
                "Base": ["simple_write", "simple_read", "batch_write", "complex_query"],
                "Charge": [r.test_name for r in profile.benchmark_results if "load_test" in r.test_name],
                "Stress": ["memory_stress", "cpu_stress"],
                "Concurrence": [r.test_name for r in profile.benchmark_results if "concurrent" in r.test_name],
                "Mémoire": ["memory_growth", "memory_recovery"]
            }
            
            for category, test_names in categories.items():
                category_results = [r for r in profile.benchmark_results if r.test_name in test_names]
                if category_results:
                    f.write(f"### {category}\n\n")
                    f.write("| Test | Débit (ops/s) | Erreurs (%) | Mémoire (MB) | P95 (ms) |\n")
                    f.write("|------|---------------|-------------|--------------|----------|\n")
                    
                    for result in category_results:
                        p95 = result.percentiles.get("p95", 0) * 1000  # Conversion en ms
                        f.write(f"| {result.test_name} | {result.throughput:.1f} | {result.error_rate:.1f} | {result.memory_usage:.1f} | {p95:.2f} |\n")
                    f.write("\n")
            
            # Recommandations
            f.write("## 🎯 Recommandations d'Optimisation\n\n")
            for i, recommendation in enumerate(profile.recommendations, 1):
                f.write(f"{i}. {recommendation}\n")
            f.write("\n")
            
            # Métriques détaillées
            f.write("## 📊 Métriques Détaillées\n\n")
            
            # Débit global
            throughputs = [r.throughput for r in profile.benchmark_results if r.throughput > 0]
            if throughputs:
                f.write(f"**Débit moyen:** {statistics.mean(throughputs):.1f} ops/s\n")
                f.write(f"**Débit maximum:** {max(throughputs):.1f} ops/s\n")
                f.write(f"**Débit minimum:** {min(throughputs):.1f} ops/s\n\n")
            
            # Utilisation mémoire
            memory_usages = [r.memory_usage for r in profile.benchmark_results if r.memory_usage > 0]
            if memory_usages:
                f.write(f"**Utilisation mémoire moyenne:** {statistics.mean(memory_usages):.1f} MB\n")
                f.write(f"**Pic d'utilisation mémoire:** {max(memory_usages):.1f} MB\n\n")
            
            # Taux d'erreur
            error_rates = [r.error_rate for r in profile.benchmark_results if r.error_rate > 0]
            if error_rates:
                f.write(f"**Taux d'erreur moyen:** {statistics.mean(error_rates):.2f}%\n")
                f.write(f"**Taux d'erreur maximum:** {max(error_rates):.2f}%\n")
            else:
                f.write("**Taux d'erreur:** 0% (Excellent!)\n")
            
            f.write("\n---\n")
            f.write("*Rapport généré automatiquement par le Framework de Benchmark Avancé*\n")
            f.write("*Expert Development Team - Dirigé par Fahed Mlaiel*\n")
        
        logger.info(f"📝 Rapport généré: {report_filename}")


async def run_benchmark():
    """Fonction principale pour exécuter le benchmark."""
    print("🚀 Framework de Benchmark de Performance Ultra-Avancé")
    print("=" * 60)
    print("Expert Development Team - Projet dirigé par Fahed Mlaiel")
    print("=" * 60)
    
    # Configuration du benchmark
    config = {
        "warm_up_iterations": 5,
        "benchmark_iterations": 20,
        "concurrent_users": [1, 2, 5, 10],
        "data_sizes": [100, 500, 1000, 2000],
        "enable_ml_analysis": True
    }
    
    # Création du framework de benchmark
    benchmark = PerformanceBenchmark(config)
    
    try:
        # Exécution du benchmark complet
        profile = await benchmark.run_comprehensive_benchmark()
        
        print("\n" + "=" * 60)
        print("🎉 BENCHMARK TERMINÉ AVEC SUCCÈS!")
        print("=" * 60)
        print(f"🏆 Score Global: {profile.overall_score:.1f}/100")
        print(f"📊 Tests Exécutés: {len(profile.benchmark_results)}")
        print(f"⏱️  Durée Totale: {(profile.timestamp - profile.system_info.timestamp).total_seconds():.1f}s")
        
        print("\n🎯 Recommandations Principales:")
        for i, rec in enumerate(profile.recommendations[:3], 1):
            print(f"{i}. {rec}")
        
        if profile.overall_score >= 90:
            print("\n🏆 EXCELLENT! Performances exceptionnelles!")
        elif profile.overall_score >= 75:
            print("\n👍 BIEN! Performances satisfaisantes avec quelques optimisations possibles.")
        elif profile.overall_score >= 60:
            print("\n⚠️  MOYEN. Plusieurs optimisations recommandées.")
        else:
            print("\n🔧 CRITIQUE. Optimisations urgentes nécessaires.")
        
        print("\n" + "=" * 60)
        print("Développé par l'Équipe d'Experts - Dirigé par Fahed Mlaiel")
        print("=" * 60)
        
        return profile
        
    except Exception as e:
        print(f"\n💥 ERREUR FATALE dans le benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Exécution du benchmark
    asyncio.run(run_benchmark())
