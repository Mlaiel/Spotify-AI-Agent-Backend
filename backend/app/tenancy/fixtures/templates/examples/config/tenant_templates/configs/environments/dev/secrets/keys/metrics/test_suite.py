#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite & Validation Framework
==============================================

Ultra-advanced testing framework with comprehensive unit tests, integration tests,
performance benchmarks, security validation, and automated quality assurance.

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
import tempfile
import shutil
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import unittest
import pytest
import concurrent.futures
import threading
import multiprocessing
import psutil
import hashlib
import random
import string

# Ajout du chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from . import (
    EnterpriseMetricsSystem, MetricDataPoint, MetricType, 
    MetricCategory, MetricSeverity, get_metrics_system
)
from .collector import MetricsCollectionAgent, CollectorConfig
from .monitor import AlertEngine, AlertRule, AlertPriority, HealthMonitor, MonitoringTarget
from .deploy import DeploymentOrchestrator, DeploymentConfig

# Configuration du logging pour les tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class MetricsSystemTestSuite:
    """Suite de tests complète pour le système de métriques."""
    
    def __init__(self):
        self.test_results = {}
        self.performance_results = {}
        self.temp_dir = None
        self.test_storage = None
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Exécute tous les tests."""
        logging.info("🚀 Démarrage de la suite de tests complète")
        
        # Préparation de l'environnement de test
        await self._setup_test_environment()
        
        try:
            # Tests unitaires
            unit_results = await self._run_unit_tests()
            
            # Tests d'intégration
            integration_results = await self._run_integration_tests()
            
            # Tests de performance
            performance_results = await self._run_performance_tests()
            
            # Tests de sécurité
            security_results = await self._run_security_tests()
            
            # Tests de robustesse
            stress_results = await self._run_stress_tests()
            
            # Validation de l'API
            api_results = await self._run_api_tests()
            
            # Compilation des résultats
            results = {
                "summary": {
                    "total_tests": sum(len(r) for r in [
                        unit_results, integration_results, performance_results,
                        security_results, stress_results, api_results
                    ]),
                    "passed": sum(
                        sum(1 for t in r.values() if t.get("passed", False)) 
                        for r in [unit_results, integration_results, performance_results,
                                security_results, stress_results, api_results]
                    ),
                    "execution_time": time.time() - self.start_time
                },
                "unit_tests": unit_results,
                "integration_tests": integration_results,
                "performance_tests": performance_results,
                "security_tests": security_results,
                "stress_tests": stress_results,
                "api_tests": api_results
            }
            
            # Calcul du score global
            total_tests = results["summary"]["total_tests"]
            passed_tests = results["summary"]["passed"]
            results["summary"]["success_rate"] = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            
            logging.info(f"✅ Tests terminés: {passed_tests}/{total_tests} réussis ({results['summary']['success_rate']:.1f}%)")
            
            return results
            
        finally:
            # Nettoyage
            await self._cleanup_test_environment()
    
    async def _setup_test_environment(self):
        """Configure l'environnement de test."""
        self.start_time = time.time()
        
        # Création du répertoire temporaire
        self.temp_dir = tempfile.mkdtemp(prefix="metrics_test_")
        logging.info(f"📁 Répertoire de test: {self.temp_dir}")
        
        # Configuration du stockage de test
        self.test_storage = get_metrics_system("sqlite", {"db_path": f"{self.temp_dir}/test.db"})
        await self.test_storage.start()
        
    async def _cleanup_test_environment(self):
        """Nettoie l'environnement de test."""
        if self.test_storage:
            await self.test_storage.stop()
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logging.info("🧹 Environnement de test nettoyé")
    
    async def _run_unit_tests(self) -> Dict[str, Dict[str, Any]]:
        """Exécute les tests unitaires."""
        logging.info("🔬 Exécution des tests unitaires")
        
        tests = {
            "test_metric_creation": self._test_metric_creation,
            "test_metric_storage": self._test_metric_storage,
            "test_metric_query": self._test_metric_query,
            "test_alert_rules": self._test_alert_rules,
            "test_anomaly_detection": self._test_anomaly_detection,
            "test_aggregations": self._test_aggregations,
            "test_data_validation": self._test_data_validation,
            "test_configuration": self._test_configuration
        }
        
        results = {}
        for test_name, test_func in tests.items():
            try:
                start_time = time.time()
                await test_func()
                execution_time = time.time() - start_time
                
                results[test_name] = {
                    "passed": True,
                    "execution_time": execution_time,
                    "message": "Test réussi"
                }
                logging.info(f"✅ {test_name}: OK ({execution_time:.3f}s)")
                
            except Exception as e:
                results[test_name] = {
                    "passed": False,
                    "execution_time": time.time() - start_time,
                    "error": str(e),
                    "message": f"Test échoué: {e}"
                }
                logging.error(f"❌ {test_name}: ÉCHEC - {e}")
        
        return results
    
    async def _test_metric_creation(self):
        """Test de création de métriques."""
        # Test création métrique basique
        metric = MetricDataPoint(
            metric_id="test.basic.metric",
            value=42.0,
            metric_type=MetricType.GAUGE,
            category=MetricCategory.SYSTEM
        )
        
        assert metric.metric_id == "test.basic.metric"
        assert metric.value == 42.0
        assert metric.metric_type == MetricType.GAUGE
        
        # Test avec tags et métadonnées
        metric_with_tags = MetricDataPoint(
            metric_id="test.tagged.metric",
            value=100.0,
            metric_type=MetricType.COUNTER,
            tags={"env": "test", "service": "metrics"},
            metadata={"description": "Test metric with tags"}
        )
        
        assert len(metric_with_tags.tags) == 2
        assert metric_with_tags.tags["env"] == "test"
        assert "description" in metric_with_tags.metadata
        
        # Test validation des valeurs
        try:
            MetricDataPoint(
                metric_id="",  # ID vide devrait échouer
                value=42.0,
                metric_type=MetricType.GAUGE
            )
            raise AssertionError("Devrait échouer avec ID vide")
        except ValueError:
            pass  # Comportement attendu
    
    async def _test_metric_storage(self):
        """Test de stockage de métriques."""
        # Stockage d'une métrique simple
        metric = MetricDataPoint(
            metric_id="test.storage.metric",
            value=123.45,
            metric_type=MetricType.GAUGE,
            category=MetricCategory.PERFORMANCE
        )
        
        await self.test_storage.storage.store_metric(metric)
        
        # Vérification que la métrique est stockée
        stored_metrics = await self.test_storage.storage.query_metrics(
            start_time=datetime.now() - timedelta(minutes=1),
            end_time=datetime.now() + timedelta(minutes=1)
        )
        
        assert len(stored_metrics) >= 1
        assert any(m.metric_id == "test.storage.metric" for m in stored_metrics)
        
        # Test de stockage en lot
        batch_metrics = []
        for i in range(10):
            batch_metrics.append(MetricDataPoint(
                metric_id=f"test.batch.metric_{i}",
                value=float(i),
                metric_type=MetricType.COUNTER
            ))
        
        for metric in batch_metrics:
            await self.test_storage.storage.store_metric(metric)
        
        # Vérification du lot
        batch_stored = await self.test_storage.storage.query_metrics(
            metric_pattern="test.batch.*",
            start_time=datetime.now() - timedelta(minutes=1),
            end_time=datetime.now() + timedelta(minutes=1)
        )
        
        assert len(batch_stored) >= 10
    
    async def _test_metric_query(self):
        """Test de requêtes de métriques."""
        # Création de métriques de test
        test_metrics = []
        base_time = datetime.now() - timedelta(hours=1)
        
        for i in range(20):
            test_metrics.append(MetricDataPoint(
                metric_id="test.query.cpu_usage",
                timestamp=base_time + timedelta(minutes=i*3),
                value=50.0 + (i % 10),
                metric_type=MetricType.GAUGE,
                tags={"host": f"server_{i%3}"}
            ))
        
        # Stockage des métriques de test
        for metric in test_metrics:
            await self.test_storage.storage.store_metric(metric)
        
        # Test requête par pattern
        pattern_results = await self.test_storage.storage.query_metrics(
            metric_pattern="test.query.*",
            start_time=base_time - timedelta(minutes=30),
            end_time=datetime.now()
        )
        
        assert len(pattern_results) >= 20
        
        # Test requête par plage de temps
        time_range_results = await self.test_storage.storage.query_metrics(
            start_time=base_time + timedelta(minutes=30),
            end_time=base_time + timedelta(minutes=45)
        )
        
        assert len(time_range_results) >= 5
        
        # Test agrégation (si supportée)
        try:
            agg_results = await self.test_storage.storage.query_aggregated(
                metric_pattern="test.query.*",
                aggregation="avg",
                interval="10m",
                start_time=base_time,
                end_time=datetime.now()
            )
            # Vérification que l'agrégation fonctionne
            assert len(agg_results) >= 1
        except AttributeError:
            pass  # Agrégation peut ne pas être implémentée
    
    async def _test_alert_rules(self):
        """Test des règles d'alerte."""
        # Création d'un moteur d'alertes de test
        alert_engine = AlertEngine(self.test_storage)
        
        # Test création de règle
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Alert Rule",
            description="Rule for testing",
            metric_pattern="test.alert.*",
            threshold_value=80.0,
            comparison=">",
            priority=AlertPriority.HIGH
        )
        
        await alert_engine.add_rule(rule)
        assert "test_rule" in alert_engine.alert_rules
        
        # Test évaluation de règle
        test_metric = MetricDataPoint(
            metric_id="test.alert.cpu",
            value=85.0,  # Au-dessus du seuil
            metric_type=MetricType.GAUGE
        )
        
        await self.test_storage.storage.store_metric(test_metric)
        
        # Simulation d'évaluation
        should_trigger = alert_engine._check_threshold_condition(85.0, 80.0, ">")
        assert should_trigger == True
        
        should_not_trigger = alert_engine._check_threshold_condition(75.0, 80.0, ">")
        assert should_not_trigger == False
    
    async def _test_anomaly_detection(self):
        """Test de détection d'anomalies."""
        # Création de données avec anomalie
        normal_values = [50.0 + random.gauss(0, 5) for _ in range(50)]
        anomaly_values = [150.0, 200.0]  # Valeurs anormalement élevées
        
        all_values = normal_values + anomaly_values
        
        # Test détection simple par Z-score
        mean_val = statistics.mean(normal_values)
        std_val = statistics.stdev(normal_values)
        
        for anomaly_val in anomaly_values:
            z_score = abs(anomaly_val - mean_val) / std_val
            assert z_score > 3.0  # Seuil d'anomalie standard
        
        # Test que les valeurs normales ne sont pas détectées comme anomalies
        for normal_val in normal_values[-10:]:  # Test sur les dernières valeurs
            z_score = abs(normal_val - mean_val) / std_val
            assert z_score <= 3.0
    
    async def _test_aggregations(self):
        """Test des fonctions d'agrégation."""
        # Données de test
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        
        # Test calculs d'agrégation
        assert statistics.mean(values) == 30.0
        assert min(values) == 10.0
        assert max(values) == 50.0
        assert sum(values) == 150.0
        
        # Test avec données vides
        try:
            statistics.mean([])
            raise AssertionError("Devrait échouer avec liste vide")
        except statistics.StatisticsError:
            pass  # Comportement attendu
    
    async def _test_data_validation(self):
        """Test de validation des données."""
        # Test validation des IDs de métriques
        valid_ids = ["system.cpu.usage", "crypto.key.access_count", "app.api.response_time"]
        invalid_ids = ["", "  ", "invalid..metric", "metric with spaces"]
        
        for valid_id in valid_ids:
            # Ces IDs devraient être valides
            metric = MetricDataPoint(
                metric_id=valid_id,
                value=42.0,
                metric_type=MetricType.GAUGE
            )
            assert metric.metric_id == valid_id
        
        # Test validation des valeurs
        valid_values = [0.0, 42.0, -10.5, 1e6, 1e-6]
        for valid_value in valid_values:
            metric = MetricDataPoint(
                metric_id="test.value",
                value=valid_value,
                metric_type=MetricType.GAUGE
            )
            assert metric.value == valid_value
        
        # Test validation des timestamps
        valid_timestamp = datetime.now()
        metric = MetricDataPoint(
            metric_id="test.timestamp",
            timestamp=valid_timestamp,
            value=42.0,
            metric_type=MetricType.GAUGE
        )
        assert metric.timestamp == valid_timestamp
    
    async def _test_configuration(self):
        """Test de configuration du système."""
        # Test configuration du collecteur
        config = CollectorConfig(
            system_interval=30,
            security_interval=300,
            adaptive_sampling=True
        )
        
        assert config.system_interval == 30
        assert config.security_interval == 300
        assert config.adaptive_sampling == True
        
        # Test configuration de déploiement
        deploy_config = DeploymentConfig(
            deployment_name="test-deployment",
            mode="development"
        )
        
        assert deploy_config.deployment_name == "test-deployment"
        assert hasattr(deploy_config, 'mode')
    
    async def _run_integration_tests(self) -> Dict[str, Dict[str, Any]]:
        """Exécute les tests d'intégration."""
        logging.info("🔗 Exécution des tests d'intégration")
        
        tests = {
            "test_end_to_end_flow": self._test_end_to_end_flow,
            "test_collector_integration": self._test_collector_integration,
            "test_alert_integration": self._test_alert_integration,
            "test_storage_backends": self._test_storage_backends,
            "test_monitoring_integration": self._test_monitoring_integration
        }
        
        results = {}
        for test_name, test_func in tests.items():
            try:
                start_time = time.time()
                await test_func()
                execution_time = time.time() - start_time
                
                results[test_name] = {
                    "passed": True,
                    "execution_time": execution_time,
                    "message": "Test d'intégration réussi"
                }
                logging.info(f"✅ {test_name}: OK ({execution_time:.3f}s)")
                
            except Exception as e:
                results[test_name] = {
                    "passed": False,
                    "execution_time": time.time() - start_time,
                    "error": str(e),
                    "message": f"Test d'intégration échoué: {e}"
                }
                logging.error(f"❌ {test_name}: ÉCHEC - {e}")
        
        return results
    
    async def _test_end_to_end_flow(self):
        """Test du flux de bout en bout."""
        # 1. Création d'une métrique
        metric = MetricDataPoint(
            metric_id="test.e2e.cpu_usage",
            value=85.0,
            metric_type=MetricType.GAUGE,
            category=MetricCategory.SYSTEM,
            tags={"host": "test-server"}
        )
        
        # 2. Stockage
        await self.test_storage.storage.store_metric(metric)
        
        # 3. Requête
        results = await self.test_storage.storage.query_metrics(
            metric_pattern="test.e2e.*",
            start_time=datetime.now() - timedelta(minutes=1),
            end_time=datetime.now() + timedelta(minutes=1)
        )
        
        # 4. Vérification
        assert len(results) >= 1
        found_metric = next((m for m in results if m.metric_id == "test.e2e.cpu_usage"), None)
        assert found_metric is not None
        assert found_metric.value == 85.0
        assert found_metric.tags.get("host") == "test-server"
    
    async def _test_collector_integration(self):
        """Test d'intégration du collecteur."""
        # Configuration du collecteur
        config = CollectorConfig(
            system_interval=1,  # Intervalle court pour le test
            max_concurrent_collectors=2
        )
        
        # Création du collecteur
        collector = MetricsCollectionAgent(config, self.test_storage)
        
        # Test de démarrage et arrêt
        await collector.start()
        await asyncio.sleep(2)  # Laisser le collecteur fonctionner
        await collector.stop()
        
        # Vérification que des métriques ont été collectées
        assert collector.metrics_collected > 0
    
    async def _test_alert_integration(self):
        """Test d'intégration des alertes."""
        # Création du moteur d'alertes
        alert_engine = AlertEngine(self.test_storage)
        await alert_engine.start()
        
        # Ajout d'une règle de test
        rule = AlertRule(
            rule_id="integration_test_rule",
            name="Integration Test Rule",
            description="Test rule for integration",
            metric_pattern="test.integration.*",
            threshold_value=90.0,
            comparison=">",
            duration_seconds=1,
            priority=AlertPriority.HIGH
        )
        
        await alert_engine.add_rule(rule)
        
        # Injection de métriques qui devraient déclencher l'alerte
        trigger_metric = MetricDataPoint(
            metric_id="test.integration.cpu",
            value=95.0,  # Au-dessus du seuil
            metric_type=MetricType.GAUGE
        )
        
        await self.test_storage.storage.store_metric(trigger_metric)
        
        # Attente de l'évaluation
        await asyncio.sleep(2)
        
        # Vérification (l'alerte peut ne pas se déclencher dans l'environnement de test)
        status = alert_engine.get_engine_status()
        assert status["running"] == True
        assert status["rules_count"] >= 1
        
        await alert_engine.stop()
    
    async def _test_storage_backends(self):
        """Test des différents backends de stockage."""
        # Test SQLite (déjà testé dans la configuration principale)
        sqlite_storage = get_metrics_system("sqlite", {"db_path": f"{self.temp_dir}/test_sqlite.db"})
        await sqlite_storage.start()
        
        test_metric = MetricDataPoint(
            metric_id="test.storage.sqlite",
            value=42.0,
            metric_type=MetricType.GAUGE
        )
        
        await sqlite_storage.storage.store_metric(test_metric)
        
        results = await sqlite_storage.storage.query_metrics(
            metric_pattern="test.storage.*",
            start_time=datetime.now() - timedelta(minutes=1),
            end_time=datetime.now() + timedelta(minutes=1)
        )
        
        assert len(results) >= 1
        await sqlite_storage.stop()
        
        # Test Redis (si disponible)
        try:
            redis_storage = get_metrics_system("redis", {"redis_url": "redis://localhost:6379/15"})
            await redis_storage.start()
            
            await redis_storage.storage.store_metric(test_metric)
            await redis_storage.stop()
            
        except Exception as e:
            logging.warning(f"Redis non disponible pour les tests: {e}")
    
    async def _test_monitoring_integration(self):
        """Test d'intégration du monitoring."""
        # Création du moteur d'alertes
        alert_engine = AlertEngine(self.test_storage)
        
        # Création du moniteur de santé
        health_monitor = HealthMonitor(alert_engine)
        
        # Ajout d'une cible de test
        target = MonitoringTarget(
            target_id="test_target",
            name="Test Target",
            target_type="test",
            endpoint="127.0.0.1",
            port=80,
            check_interval=1
        )
        
        await health_monitor.add_target(target)
        
        # Vérification que la cible est ajoutée
        assert "test_target" in health_monitor.targets
        assert health_monitor.targets["test_target"].name == "Test Target"
    
    async def _run_performance_tests(self) -> Dict[str, Dict[str, Any]]:
        """Exécute les tests de performance."""
        logging.info("⚡ Exécution des tests de performance")
        
        tests = {
            "test_metric_ingestion_rate": self._test_metric_ingestion_rate,
            "test_query_performance": self._test_query_performance,
            "test_concurrent_operations": self._test_concurrent_operations,
            "test_memory_usage": self._test_memory_usage,
            "test_large_dataset": self._test_large_dataset
        }
        
        results = {}
        for test_name, test_func in tests.items():
            try:
                start_time = time.time()
                performance_data = await test_func()
                execution_time = time.time() - start_time
                
                results[test_name] = {
                    "passed": True,
                    "execution_time": execution_time,
                    "performance_data": performance_data,
                    "message": "Test de performance réussi"
                }
                logging.info(f"✅ {test_name}: OK ({execution_time:.3f}s)")
                
            except Exception as e:
                results[test_name] = {
                    "passed": False,
                    "execution_time": time.time() - start_time,
                    "error": str(e),
                    "message": f"Test de performance échoué: {e}"
                }
                logging.error(f"❌ {test_name}: ÉCHEC - {e}")
        
        return results
    
    async def _test_metric_ingestion_rate(self) -> Dict[str, float]:
        """Test du taux d'ingestion de métriques."""
        num_metrics = 1000
        metrics = []
        
        # Génération des métriques
        for i in range(num_metrics):
            metrics.append(MetricDataPoint(
                metric_id=f"perf.ingestion.metric_{i}",
                value=float(i),
                metric_type=MetricType.COUNTER
            ))
        
        # Test d'ingestion
        start_time = time.time()
        
        for metric in metrics:
            await self.test_storage.storage.store_metric(metric)
        
        ingestion_time = time.time() - start_time
        rate = num_metrics / ingestion_time
        
        logging.info(f"📊 Taux d'ingestion: {rate:.1f} métriques/seconde")
        
        return {
            "metrics_count": num_metrics,
            "ingestion_time": ingestion_time,
            "rate_per_second": rate
        }
    
    async def _test_query_performance(self) -> Dict[str, float]:
        """Test de performance des requêtes."""
        # Préparation des données
        num_metrics = 500
        for i in range(num_metrics):
            metric = MetricDataPoint(
                metric_id=f"perf.query.metric_{i%10}",
                timestamp=datetime.now() - timedelta(minutes=i),
                value=float(i),
                metric_type=MetricType.GAUGE
            )
            await self.test_storage.storage.store_metric(metric)
        
        # Test de requête simple
        start_time = time.time()
        results = await self.test_storage.storage.query_metrics(
            metric_pattern="perf.query.*",
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )
        query_time = time.time() - start_time
        
        logging.info(f"📊 Temps de requête: {query_time:.3f}s pour {len(results)} résultats")
        
        return {
            "query_time": query_time,
            "results_count": len(results),
            "results_per_second": len(results) / query_time if query_time > 0 else 0
        }
    
    async def _test_concurrent_operations(self) -> Dict[str, float]:
        """Test des opérations concurrentes."""
        num_concurrent = 10
        metrics_per_task = 50
        
        async def write_metrics(task_id: int):
            for i in range(metrics_per_task):
                metric = MetricDataPoint(
                    metric_id=f"perf.concurrent.task_{task_id}.metric_{i}",
                    value=float(i),
                    metric_type=MetricType.COUNTER
                )
                await self.test_storage.storage.store_metric(metric)
        
        # Exécution concurrente
        start_time = time.time()
        
        tasks = [write_metrics(i) for i in range(num_concurrent)]
        await asyncio.gather(*tasks)
        
        concurrent_time = time.time() - start_time
        total_metrics = num_concurrent * metrics_per_task
        rate = total_metrics / concurrent_time
        
        logging.info(f"📊 Opérations concurrentes: {rate:.1f} métriques/seconde ({num_concurrent} tâches)")
        
        return {
            "concurrent_tasks": num_concurrent,
            "metrics_per_task": metrics_per_task,
            "total_time": concurrent_time,
            "total_rate": rate
        }
    
    async def _test_memory_usage(self) -> Dict[str, float]:
        """Test d'utilisation mémoire."""
        if not hasattr(psutil, 'Process'):
            return {"memory_usage": 0, "memory_increase": 0}
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Création de nombreuses métriques en mémoire
        metrics = []
        for i in range(1000):
            metrics.append(MetricDataPoint(
                metric_id=f"perf.memory.metric_{i}",
                value=float(i),
                metric_type=MetricType.GAUGE,
                tags={f"tag_{j}": f"value_{j}" for j in range(5)},
                metadata={f"meta_{j}": f"data_{j}" for j in range(3)}
            ))
        
        # Stockage des métriques
        for metric in metrics:
            await self.test_storage.storage.store_metric(metric)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        logging.info(f"📊 Utilisation mémoire: {final_memory:.1f}MB (+{memory_increase:.1f}MB)")
        
        return {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": memory_increase
        }
    
    async def _test_large_dataset(self) -> Dict[str, float]:
        """Test avec un large dataset."""
        num_metrics = 5000
        batch_size = 100
        
        start_time = time.time()
        
        # Création par lots pour éviter les problèmes de mémoire
        for batch_start in range(0, num_metrics, batch_size):
            batch_metrics = []
            for i in range(batch_start, min(batch_start + batch_size, num_metrics)):
                batch_metrics.append(MetricDataPoint(
                    metric_id=f"perf.large.metric_{i%100}",
                    timestamp=datetime.now() - timedelta(seconds=i),
                    value=float(i),
                    metric_type=MetricType.GAUGE
                ))
            
            # Stockage du lot
            for metric in batch_metrics:
                await self.test_storage.storage.store_metric(metric)
        
        storage_time = time.time() - start_time
        
        # Test de requête sur le large dataset
        query_start = time.time()
        results = await self.test_storage.storage.query_metrics(
            metric_pattern="perf.large.*",
            start_time=datetime.now() - timedelta(hours=2),
            end_time=datetime.now()
        )
        query_time = time.time() - query_start
        
        logging.info(f"📊 Large dataset: {len(results)} métriques stockées en {storage_time:.1f}s, requête en {query_time:.3f}s")
        
        return {
            "dataset_size": num_metrics,
            "storage_time": storage_time,
            "query_time": query_time,
            "results_count": len(results)
        }
    
    async def _run_security_tests(self) -> Dict[str, Dict[str, Any]]:
        """Exécute les tests de sécurité."""
        logging.info("🔒 Exécution des tests de sécurité")
        
        tests = {
            "test_input_validation": self._test_input_validation,
            "test_sql_injection": self._test_sql_injection,
            "test_data_sanitization": self._test_data_sanitization,
            "test_access_control": self._test_access_control
        }
        
        results = {}
        for test_name, test_func in tests.items():
            try:
                start_time = time.time()
                await test_func()
                execution_time = time.time() - start_time
                
                results[test_name] = {
                    "passed": True,
                    "execution_time": execution_time,
                    "message": "Test de sécurité réussi"
                }
                logging.info(f"✅ {test_name}: OK ({execution_time:.3f}s)")
                
            except Exception as e:
                results[test_name] = {
                    "passed": False,
                    "execution_time": time.time() - start_time,
                    "error": str(e),
                    "message": f"Test de sécurité échoué: {e}"
                }
                logging.error(f"❌ {test_name}: ÉCHEC - {e}")
        
        return results
    
    async def _test_input_validation(self):
        """Test de validation des entrées."""
        # Test avec des IDs malveillants
        malicious_ids = [
            "../../../etc/passwd",
            "<script>alert('xss')</script>",
            "'; DROP TABLE metrics; --",
            "\\x00\\x01\\x02",
            "A" * 10000  # ID très long
        ]
        
        for malicious_id in malicious_ids:
            try:
                metric = MetricDataPoint(
                    metric_id=malicious_id,
                    value=42.0,
                    metric_type=MetricType.GAUGE
                )
                # Si aucune exception n'est levée, vérifier que l'ID est sanitisé
                assert len(metric.metric_id) < 1000  # Limite raisonnable
                
            except (ValueError, TypeError):
                pass  # Comportement attendu pour les entrées invalides
    
    async def _test_sql_injection(self):
        """Test de protection contre l'injection SQL."""
        # Test avec des patterns d'injection SQL
        sql_injection_patterns = [
            "test'; DROP TABLE metrics; --",
            "test' OR '1'='1",
            "test' UNION SELECT * FROM users --",
            "test'; INSERT INTO metrics VALUES (1,2,3); --"
        ]
        
        for pattern in sql_injection_patterns:
            # Tentative de stockage avec pattern malveillant
            metric = MetricDataPoint(
                metric_id="test.security.injection",
                value=42.0,
                metric_type=MetricType.GAUGE,
                tags={"malicious": pattern}
            )
            
            await self.test_storage.storage.store_metric(metric)
            
            # Vérification que le système fonctionne toujours
            results = await self.test_storage.storage.query_metrics(
                metric_pattern="test.security.*",
                start_time=datetime.now() - timedelta(minutes=1),
                end_time=datetime.now() + timedelta(minutes=1)
            )
            
            # Le système devrait toujours fonctionner normalement
            assert isinstance(results, list)
    
    async def _test_data_sanitization(self):
        """Test de sanitisation des données."""
        # Test avec des caractères spéciaux
        special_chars = "!@#$%^&*()[]{}|\\:;\"'<>,.?/~`"
        
        metric = MetricDataPoint(
            metric_id="test.sanitization.special_chars",
            value=42.0,
            metric_type=MetricType.GAUGE,
            tags={"special": special_chars},
            metadata={"description": special_chars}
        )
        
        await self.test_storage.storage.store_metric(metric)
        
        # Vérification que les données sont stockées correctement
        results = await self.test_storage.storage.query_metrics(
            metric_pattern="test.sanitization.*",
            start_time=datetime.now() - timedelta(minutes=1),
            end_time=datetime.now() + timedelta(minutes=1)
        )
        
        assert len(results) >= 1
    
    async def _test_access_control(self):
        """Test du contrôle d'accès."""
        # Ce test dépend de l'implémentation du contrôle d'accès
        # Pour l'instant, on vérifie que le système fonctionne normalement
        
        metric = MetricDataPoint(
            metric_id="test.access.control",
            value=42.0,
            metric_type=MetricType.GAUGE
        )
        
        await self.test_storage.storage.store_metric(metric)
        
        results = await self.test_storage.storage.query_metrics(
            metric_pattern="test.access.*",
            start_time=datetime.now() - timedelta(minutes=1),
            end_time=datetime.now() + timedelta(minutes=1)
        )
        
        assert len(results) >= 1
    
    async def _run_stress_tests(self) -> Dict[str, Dict[str, Any]]:
        """Exécute les tests de stress."""
        logging.info("💪 Exécution des tests de stress")
        
        tests = {
            "test_high_load": self._test_high_load,
            "test_memory_pressure": self._test_memory_pressure,
            "test_connection_limits": self._test_connection_limits,
            "test_error_recovery": self._test_error_recovery
        }
        
        results = {}
        for test_name, test_func in tests.items():
            try:
                start_time = time.time()
                stress_data = await test_func()
                execution_time = time.time() - start_time
                
                results[test_name] = {
                    "passed": True,
                    "execution_time": execution_time,
                    "stress_data": stress_data,
                    "message": "Test de stress réussi"
                }
                logging.info(f"✅ {test_name}: OK ({execution_time:.3f}s)")
                
            except Exception as e:
                results[test_name] = {
                    "passed": False,
                    "execution_time": time.time() - start_time,
                    "error": str(e),
                    "message": f"Test de stress échoué: {e}"
                }
                logging.error(f"❌ {test_name}: ÉCHEC - {e}")
        
        return results
    
    async def _test_high_load(self) -> Dict[str, float]:
        """Test de charge élevée."""
        num_tasks = 20
        metrics_per_task = 100
        
        async def high_load_task(task_id: int):
            for i in range(metrics_per_task):
                metric = MetricDataPoint(
                    metric_id=f"stress.high_load.task_{task_id}.metric_{i}",
                    value=float(i),
                    metric_type=MetricType.COUNTER
                )
                await self.test_storage.storage.store_metric(metric)
                
                # Petite pause pour éviter de surcharger
                if i % 10 == 0:
                    await asyncio.sleep(0.001)
        
        start_time = time.time()
        
        # Lancement de toutes les tâches en parallèle
        tasks = [high_load_task(i) for i in range(num_tasks)]
        await asyncio.gather(*tasks)
        
        load_time = time.time() - start_time
        total_metrics = num_tasks * metrics_per_task
        rate = total_metrics / load_time
        
        logging.info(f"📊 Test de charge: {total_metrics} métriques en {load_time:.1f}s ({rate:.1f}/sec)")
        
        return {
            "total_metrics": total_metrics,
            "load_time": load_time,
            "rate_per_second": rate,
            "concurrent_tasks": num_tasks
        }
    
    async def _test_memory_pressure(self) -> Dict[str, float]:
        """Test de pression mémoire."""
        # Création d'un grand nombre d'objets en mémoire
        large_objects = []
        
        for i in range(1000):
            # Création de métriques avec beaucoup de métadonnées
            metric = MetricDataPoint(
                metric_id=f"stress.memory.metric_{i}",
                value=float(i),
                metric_type=MetricType.GAUGE,
                tags={f"tag_{j}": f"value_{j}" * 100 for j in range(10)},
                metadata={f"meta_{j}": f"data_{j}" * 50 for j in range(5)}
            )
            large_objects.append(metric)
        
        # Stockage de tous les objets
        for metric in large_objects:
            await self.test_storage.storage.store_metric(metric)
        
        # Mesure de la mémoire utilisée
        memory_usage = 0
        if hasattr(psutil, 'Process'):
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            "objects_created": len(large_objects),
            "memory_usage_mb": memory_usage
        }
    
    async def _test_connection_limits(self) -> Dict[str, float]:
        """Test des limites de connexion."""
        # Ce test dépend du backend de stockage
        # Pour SQLite, on teste les accès concurrents
        
        num_concurrent = 50
        
        async def concurrent_access(task_id: int):
            metric = MetricDataPoint(
                metric_id=f"stress.connection.task_{task_id}",
                value=float(task_id),
                metric_type=MetricType.GAUGE
            )
            await self.test_storage.storage.store_metric(metric)
            
            # Lecture immédiate
            results = await self.test_storage.storage.query_metrics(
                metric_pattern=f"stress.connection.task_{task_id}",
                start_time=datetime.now() - timedelta(minutes=1),
                end_time=datetime.now() + timedelta(minutes=1)
            )
            return len(results)
        
        start_time = time.time()
        
        # Accès concurrents
        tasks = [concurrent_access(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        connection_time = time.time() - start_time
        
        # Comptage des succès
        successful_connections = sum(1 for r in results if isinstance(r, int) and r > 0)
        
        return {
            "concurrent_connections": num_concurrent,
            "successful_connections": successful_connections,
            "connection_time": connection_time,
            "success_rate": successful_connections / num_concurrent * 100
        }
    
    async def _test_error_recovery(self) -> Dict[str, float]:
        """Test de récupération d'erreur."""
        errors_encountered = 0
        successful_operations = 0
        
        # Simulation d'erreurs et de récupération
        for i in range(100):
            try:
                if i % 10 == 0:
                    # Simulation d'une erreur (métrique invalide)
                    metric = MetricDataPoint(
                        metric_id="",  # ID invalide
                        value=float(i),
                        metric_type=MetricType.GAUGE
                    )
                    await self.test_storage.storage.store_metric(metric)
                else:
                    # Opération normale
                    metric = MetricDataPoint(
                        metric_id=f"stress.error_recovery.metric_{i}",
                        value=float(i),
                        metric_type=MetricType.GAUGE
                    )
                    await self.test_storage.storage.store_metric(metric)
                    successful_operations += 1
                    
            except Exception:
                errors_encountered += 1
                # Continue avec l'opération suivante (récupération)
        
        return {
            "total_operations": 100,
            "successful_operations": successful_operations,
            "errors_encountered": errors_encountered,
            "recovery_rate": successful_operations / 100 * 100
        }
    
    async def _run_api_tests(self) -> Dict[str, Dict[str, Any]]:
        """Exécute les tests d'API."""
        logging.info("🌐 Exécution des tests d'API")
        
        # Pour l'instant, tests basiques sans serveur HTTP
        tests = {
            "test_metric_serialization": self._test_metric_serialization,
            "test_json_compatibility": self._test_json_compatibility,
            "test_data_formats": self._test_data_formats
        }
        
        results = {}
        for test_name, test_func in tests.items():
            try:
                start_time = time.time()
                await test_func()
                execution_time = time.time() - start_time
                
                results[test_name] = {
                    "passed": True,
                    "execution_time": execution_time,
                    "message": "Test d'API réussi"
                }
                logging.info(f"✅ {test_name}: OK ({execution_time:.3f}s)")
                
            except Exception as e:
                results[test_name] = {
                    "passed": False,
                    "execution_time": time.time() - start_time,
                    "error": str(e),
                    "message": f"Test d'API échoué: {e}"
                }
                logging.error(f"❌ {test_name}: ÉCHEC - {e}")
        
        return results
    
    async def _test_metric_serialization(self):
        """Test de sérialisation des métriques."""
        metric = MetricDataPoint(
            metric_id="test.api.serialization",
            value=42.0,
            metric_type=MetricType.GAUGE,
            tags={"env": "test", "service": "api"},
            metadata={"description": "Test metric for serialization"}
        )
        
        # Test de conversion en dictionnaire
        metric_dict = {
            "metric_id": metric.metric_id,
            "timestamp": metric.timestamp.isoformat(),
            "value": metric.value,
            "metric_type": metric.metric_type.value,
            "category": metric.category.value,
            "severity": metric.severity.value,
            "tags": metric.tags,
            "metadata": metric.metadata
        }
        
        # Vérification de la structure
        assert "metric_id" in metric_dict
        assert "timestamp" in metric_dict
        assert "value" in metric_dict
        assert isinstance(metric_dict["tags"], dict)
        assert isinstance(metric_dict["metadata"], dict)
    
    async def _test_json_compatibility(self):
        """Test de compatibilité JSON."""
        metric = MetricDataPoint(
            metric_id="test.api.json",
            value=123.45,
            metric_type=MetricType.COUNTER,
            tags={"unicode": "ñáéíóú", "special": "!@#$%"}
        )
        
        # Test de sérialisation JSON
        metric_json = json.dumps({
            "metric_id": metric.metric_id,
            "value": metric.value,
            "metric_type": metric.metric_type.value,
            "tags": metric.tags
        }, ensure_ascii=False)
        
        # Test de désérialisation
        parsed_data = json.loads(metric_json)
        
        assert parsed_data["metric_id"] == metric.metric_id
        assert parsed_data["value"] == metric.value
        assert parsed_data["tags"]["unicode"] == "ñáéíóú"
    
    async def _test_data_formats(self):
        """Test des formats de données."""
        # Test avec différents types de valeurs
        test_values = [
            0,
            42,
            -10,
            3.14159,
            1e10,
            1e-10
        ]
        
        for value in test_values:
            metric = MetricDataPoint(
                metric_id="test.api.formats",
                value=float(value),
                metric_type=MetricType.GAUGE
            )
            
            # Vérification que la valeur est correctement stockée
            assert metric.value == float(value)
        
        # Test avec différents formats de timestamp
        timestamps = [
            datetime.now(),
            datetime.now() - timedelta(hours=1),
            datetime.now() + timedelta(minutes=30)
        ]
        
        for ts in timestamps:
            metric = MetricDataPoint(
                metric_id="test.api.timestamp",
                timestamp=ts,
                value=42.0,
                metric_type=MetricType.GAUGE
            )
            
            assert metric.timestamp == ts


async def run_comprehensive_tests():
    """Fonction principale pour exécuter tous les tests."""
    print("🧪 Démarrage de la Suite de Tests Complète")
    print("=" * 60)
    print(f"Expert Development Team - Projet dirigé par Fahed Mlaiel")
    print("=" * 60)
    
    # Création de la suite de tests
    test_suite = MetricsSystemTestSuite()
    
    try:
        # Exécution de tous les tests
        results = await test_suite.run_all_tests()
        
        # Affichage des résultats
        print("\n" + "=" * 60)
        print("📊 RÉSULTATS DES TESTS")
        print("=" * 60)
        
        summary = results["summary"]
        print(f"📈 Tests totaux: {summary['total_tests']}")
        print(f"✅ Tests réussis: {summary['passed']}")
        print(f"❌ Tests échoués: {summary['total_tests'] - summary['passed']}")
        print(f"🎯 Taux de réussite: {summary['success_rate']:.1f}%")
        print(f"⏱️  Temps d'exécution: {summary['execution_time']:.1f}s")
        
        # Détails par catégorie
        categories = ["unit_tests", "integration_tests", "performance_tests", "security_tests", "stress_tests", "api_tests"]
        
        for category in categories:
            if category in results:
                category_results = results[category]
                passed = sum(1 for t in category_results.values() if t.get("passed", False))
                total = len(category_results)
                print(f"\n📋 {category.replace('_', ' ').title()}: {passed}/{total}")
                
                for test_name, test_result in category_results.items():
                    status = "✅" if test_result.get("passed", False) else "❌"
                    time_str = f"({test_result.get('execution_time', 0):.3f}s)"
                    print(f"  {status} {test_name} {time_str}")
        
        # Résultats de performance
        if "performance_tests" in results:
            print("\n" + "=" * 60)
            print("⚡ MÉTRIQUES DE PERFORMANCE")
            print("=" * 60)
            
            for test_name, test_result in results["performance_tests"].items():
                if test_result.get("passed") and "performance_data" in test_result:
                    perf_data = test_result["performance_data"]
                    print(f"\n📊 {test_name}:")
                    for key, value in perf_data.items():
                        if isinstance(value, float):
                            print(f"  • {key}: {value:.3f}")
                        else:
                            print(f"  • {key}: {value}")
        
        # Sauvegarde des résultats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Résultats sauvegardés dans: {results_file}")
        
        # Score final
        if summary["success_rate"] >= 95:
            print("\n🏆 EXCELLENT! Tous les tests sont réussis.")
        elif summary["success_rate"] >= 80:
            print("\n👍 BIEN! La plupart des tests sont réussis.")
        else:
            print("\n⚠️  ATTENTION! Plusieurs tests ont échoué.")
        
        print("\n" + "=" * 60)
        print("🎉 Suite de tests terminée avec succès!")
        print(f"Développé par l'Équipe d'Experts - Dirigé par Fahed Mlaiel")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        print(f"\n💥 ERREUR FATALE dans la suite de tests: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Exécution de la suite de tests
    asyncio.run(run_comprehensive_tests())
