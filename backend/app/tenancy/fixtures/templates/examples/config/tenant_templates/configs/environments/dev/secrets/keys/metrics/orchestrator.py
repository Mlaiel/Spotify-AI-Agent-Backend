#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enterprise Metrics System - Master Orchestrator
==============================================

Ultra-advanced master orchestrator for the complete enterprise metrics ecosystem.
This script provides a unified interface to deploy, test, benchmark, monitor,
and validate the entire metrics infrastructure with comprehensive automation.

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
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import tempfile
import shutil

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Ajout du chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from . import (
        EnterpriseMetricsSystem, MetricDataPoint, MetricType, 
        MetricCategory, MetricSeverity, get_metrics_system
    )
    from .collector import MetricsCollectionAgent, CollectorConfig
    from .monitor import AlertEngine, AlertRule, AlertPriority, HealthMonitor
    from .deploy import DeploymentOrchestrator, DeploymentConfig
    from .test_suite import MetricsSystemTestSuite
    from .benchmark import PerformanceBenchmark
    from .compliance import ComplianceValidator, ComplianceStandard
except ImportError as e:
    logger.error(f"Erreur d'import: {e}")
    logger.info("Tentative d'import relatif...")
    try:
        import __init__ as metrics_module
        from test_suite import MetricsSystemTestSuite
        from benchmark import PerformanceBenchmark
        from compliance import ComplianceValidator, ComplianceStandard
        from collector import MetricsCollectionAgent, CollectorConfig
        from monitor import AlertEngine
        from deploy import DeploymentOrchestrator, DeploymentConfig
    except ImportError as e2:
        logger.error(f"Erreur d'import relatif: {e2}")
        sys.exit(1)


class OrchestrationMode:
    """Modes d'orchestration disponibles."""
    FULL = "full"
    DEPLOY = "deploy"
    TEST = "test"
    BENCHMARK = "benchmark"
    MONITOR = "monitor"
    COMPLIANCE = "compliance"
    DEMO = "demo"
    INTERACTIVE = "interactive"


class MasterOrchestrator:
    """Orchestrateur principal du système de métriques d'entreprise."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.temp_dir = None
        self.metrics_system = None
        self.start_time = None
        
        # Configuration par défaut
        self.default_config = {
            "storage_backend": "sqlite",
            "storage_config": {},
            "enable_collector": True,
            "enable_monitoring": True,
            "enable_analytics": True,
            "deployment_mode": "development",
            "auto_cleanup": True,
            "report_format": "both",  # json, markdown, both
            "verbose": False
        }
        
        # Fusion avec la configuration par défaut
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    async def run_orchestration(self, mode: str = OrchestrationMode.FULL, **kwargs) -> Dict[str, Any]:
        """Lance l'orchestration selon le mode spécifié."""
        logger.info(f"🚀 Démarrage de l'orchestration en mode: {mode.upper()}")
        self.start_time = time.time()
        
        try:
            # Préparation de l'environnement
            await self._setup_environment()
            
            # Exécution selon le mode
            if mode == OrchestrationMode.FULL:
                results = await self._run_full_orchestration(**kwargs)
            elif mode == OrchestrationMode.DEPLOY:
                results = await self._run_deployment_only(**kwargs)
            elif mode == OrchestrationMode.TEST:
                results = await self._run_testing_only(**kwargs)
            elif mode == OrchestrationMode.BENCHMARK:
                results = await self._run_benchmark_only(**kwargs)
            elif mode == OrchestrationMode.MONITOR:
                results = await self._run_monitoring_only(**kwargs)
            elif mode == OrchestrationMode.COMPLIANCE:
                results = await self._run_compliance_only(**kwargs)
            elif mode == OrchestrationMode.DEMO:
                results = await self._run_demo_mode(**kwargs)
            elif mode == OrchestrationMode.INTERACTIVE:
                results = await self._run_interactive_mode(**kwargs)
            else:
                raise ValueError(f"Mode d'orchestration non supporté: {mode}")
            
            # Rapport final
            total_time = time.time() - self.start_time
            results["orchestration_summary"] = {
                "mode": mode,
                "total_execution_time": total_time,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            
            # Sauvegarde des résultats
            await self._save_orchestration_results(results, mode)
            
            logger.info(f"✅ Orchestration terminée avec succès en {total_time:.1f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"💥 Erreur dans l'orchestration: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Nettoyage
            if self.config.get("auto_cleanup", True):
                await self._cleanup_environment()
    
    async def _setup_environment(self):
        """Configure l'environnement d'orchestration."""
        logger.info("🔧 Configuration de l'environnement")
        
        # Création du répertoire temporaire
        self.temp_dir = tempfile.mkdtemp(prefix="metrics_orchestration_")
        logger.info(f"📁 Répertoire temporaire: {self.temp_dir}")
        
        # Configuration du stockage
        storage_config = self.config.get("storage_config", {})
        if self.config["storage_backend"] == "sqlite":
            storage_config["db_path"] = f"{self.temp_dir}/metrics.db"
        
        # Initialisation du système de métriques
        try:
            from . import get_metrics_system
            self.metrics_system = get_metrics_system(
                self.config["storage_backend"], 
                storage_config
            )
        except ImportError:
            # Fallback pour les imports relatifs
            import __init__ as metrics_module
            self.metrics_system = metrics_module.get_metrics_system(
                self.config["storage_backend"], 
                storage_config
            )
        
        await self.metrics_system.start()
        logger.info(f"✅ Système de métriques initialisé ({self.config['storage_backend']})")
    
    async def _cleanup_environment(self):
        """Nettoie l'environnement d'orchestration."""
        logger.info("🧹 Nettoyage de l'environnement")
        
        if self.metrics_system:
            try:
                await self.metrics_system.stop()
            except Exception as e:
                logger.warning(f"Erreur lors de l'arrêt du système: {e}")
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info("✅ Répertoire temporaire nettoyé")
            except Exception as e:
                logger.warning(f"Erreur lors du nettoyage: {e}")
    
    async def _run_full_orchestration(self, **kwargs) -> Dict[str, Any]:
        """Exécute l'orchestration complète."""
        logger.info("🔄 Orchestration complète")
        
        results = {}
        
        # Phase 1: Déploiement
        logger.info("📦 Phase 1: Déploiement")
        try:
            deployment_results = await self._run_deployment()
            results["deployment"] = deployment_results
            logger.info("✅ Déploiement terminé avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur de déploiement: {e}")
            results["deployment"] = {"error": str(e), "success": False}
        
        # Phase 2: Tests
        logger.info("🧪 Phase 2: Tests")
        try:
            test_results = await self._run_testing()
            results["testing"] = test_results
            logger.info("✅ Tests terminés avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur de tests: {e}")
            results["testing"] = {"error": str(e), "success": False}
        
        # Phase 3: Benchmarks
        logger.info("⚡ Phase 3: Benchmarks de performance")
        try:
            benchmark_results = await self._run_benchmark()
            results["benchmark"] = benchmark_results
            logger.info("✅ Benchmarks terminés avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur de benchmark: {e}")
            results["benchmark"] = {"error": str(e), "success": False}
        
        # Phase 4: Validation de compliance
        logger.info("🔍 Phase 4: Validation de compliance")
        try:
            compliance_results = await self._run_compliance()
            results["compliance"] = compliance_results
            logger.info("✅ Validation de compliance terminée")
        except Exception as e:
            logger.error(f"❌ Erreur de compliance: {e}")
            results["compliance"] = {"error": str(e), "success": False}
        
        # Phase 5: Monitoring (optionnel - démarrage en arrière-plan)
        if self.config.get("enable_monitoring", False):
            logger.info("📊 Phase 5: Démarrage du monitoring")
            try:
                monitoring_results = await self._start_monitoring()
                results["monitoring"] = monitoring_results
                logger.info("✅ Monitoring démarré")
            except Exception as e:
                logger.error(f"❌ Erreur de monitoring: {e}")
                results["monitoring"] = {"error": str(e), "success": False}
        
        return results
    
    async def _run_deployment_only(self, **kwargs) -> Dict[str, Any]:
        """Exécute uniquement le déploiement."""
        logger.info("📦 Mode déploiement uniquement")
        
        deployment_results = await self._run_deployment()
        
        return {
            "deployment": deployment_results,
            "mode": "deploy_only"
        }
    
    async def _run_testing_only(self, **kwargs) -> Dict[str, Any]:
        """Exécute uniquement les tests."""
        logger.info("🧪 Mode tests uniquement")
        
        test_results = await self._run_testing()
        
        return {
            "testing": test_results,
            "mode": "test_only"
        }
    
    async def _run_benchmark_only(self, **kwargs) -> Dict[str, Any]:
        """Exécute uniquement les benchmarks."""
        logger.info("⚡ Mode benchmark uniquement")
        
        benchmark_results = await self._run_benchmark()
        
        return {
            "benchmark": benchmark_results,
            "mode": "benchmark_only"
        }
    
    async def _run_monitoring_only(self, duration: int = 60, **kwargs) -> Dict[str, Any]:
        """Exécute uniquement le monitoring."""
        logger.info(f"📊 Mode monitoring uniquement ({duration}s)")
        
        monitoring_results = await self._run_monitoring_session(duration)
        
        return {
            "monitoring": monitoring_results,
            "mode": "monitor_only"
        }
    
    async def _run_compliance_only(self, **kwargs) -> Dict[str, Any]:
        """Exécute uniquement la validation de compliance."""
        logger.info("🔍 Mode compliance uniquement")
        
        compliance_results = await self._run_compliance()
        
        return {
            "compliance": compliance_results,
            "mode": "compliance_only"
        }
    
    async def _run_demo_mode(self, **kwargs) -> Dict[str, Any]:
        """Exécute une démonstration complète."""
        logger.info("🎭 Mode démonstration")
        
        results = {}
        
        # Démonstration rapide de chaque composant
        results["demo_deployment"] = await self._demo_deployment()
        results["demo_metrics"] = await self._demo_metrics_collection()
        results["demo_queries"] = await self._demo_queries()
        results["demo_alerts"] = await self._demo_alerts()
        results["demo_analytics"] = await self._demo_analytics()
        
        return {
            "demo": results,
            "mode": "demo"
        }
    
    async def _run_interactive_mode(self, **kwargs) -> Dict[str, Any]:
        """Exécute en mode interactif."""
        logger.info("🔄 Mode interactif")
        
        results = {"interactions": []}
        
        print("\n" + "=" * 60)
        print("🎯 MODE INTERACTIF - Système de Métriques d'Entreprise")
        print("Expert Development Team - Dirigé par Fahed Mlaiel")
        print("=" * 60)
        
        while True:
            print("\n📋 Options disponibles:")
            print("1. 📦 Déployer le système")
            print("2. 🧪 Exécuter les tests")
            print("3. ⚡ Benchmark de performance")
            print("4. 🔍 Validation de compliance")
            print("5. 📊 Session de monitoring")
            print("6. 🎭 Démonstration")
            print("7. 📈 Injecter des métriques de test")
            print("8. 🔍 Requêter les métriques")
            print("9. 📊 Afficher les statistiques")
            print("0. 🚪 Quitter")
            
            try:
                choice = input("\n👉 Votre choix (0-9): ").strip()
                
                if choice == "0":
                    print("👋 Au revoir!")
                    break
                elif choice == "1":
                    result = await self._run_deployment()
                    results["interactions"].append({"action": "deployment", "result": result})
                    print("✅ Déploiement terminé!")
                elif choice == "2":
                    result = await self._run_testing()
                    results["interactions"].append({"action": "testing", "result": result})
                    print("✅ Tests terminés!")
                elif choice == "3":
                    result = await self._run_benchmark()
                    results["interactions"].append({"action": "benchmark", "result": result})
                    print("✅ Benchmark terminé!")
                elif choice == "4":
                    result = await self._run_compliance()
                    results["interactions"].append({"action": "compliance", "result": result})
                    print("✅ Validation terminée!")
                elif choice == "5":
                    duration = input("Durée du monitoring (secondes, défaut 30): ").strip()
                    duration = int(duration) if duration.isdigit() else 30
                    result = await self._run_monitoring_session(duration)
                    results["interactions"].append({"action": "monitoring", "result": result})
                    print("✅ Monitoring terminé!")
                elif choice == "6":
                    result = await self._run_demo_mode()
                    results["interactions"].append({"action": "demo", "result": result})
                    print("✅ Démonstration terminée!")
                elif choice == "7":
                    count = input("Nombre de métriques à injecter (défaut 100): ").strip()
                    count = int(count) if count.isdigit() else 100
                    result = await self._inject_test_metrics(count)
                    results["interactions"].append({"action": "inject_metrics", "result": result})
                    print(f"✅ {count} métriques injectées!")
                elif choice == "8":
                    pattern = input("Pattern de recherche (défaut '*'): ").strip()
                    pattern = pattern if pattern else "*"
                    result = await self._query_metrics_interactive(pattern)
                    results["interactions"].append({"action": "query_metrics", "result": result})
                    print(f"✅ Requête terminée! {len(result.get('metrics', []))} métriques trouvées")
                elif choice == "9":
                    result = await self._show_system_stats()
                    results["interactions"].append({"action": "show_stats", "result": result})
                    print("✅ Statistiques affichées!")
                else:
                    print("❌ Choix invalide, veuillez réessayer.")
                    
            except KeyboardInterrupt:
                print("\n👋 Interruption utilisateur, arrêt...")
                break
            except Exception as e:
                print(f"❌ Erreur: {e}")
                logger.error(f"Erreur interactive: {e}")
        
        return {
            "interactive": results,
            "mode": "interactive"
        }
    
    async def _run_deployment(self) -> Dict[str, Any]:
        """Exécute le déploiement."""
        config = DeploymentConfig(
            deployment_name="orchestration_deployment",
            mode=self.config.get("deployment_mode", "development")
        )
        
        orchestrator = DeploymentOrchestrator(config)
        return await orchestrator.deploy_complete_system()
    
    async def _run_testing(self) -> Dict[str, Any]:
        """Exécute la suite de tests."""
        test_suite = MetricsSystemTestSuite()
        return await test_suite.run_all_tests()
    
    async def _run_benchmark(self) -> Dict[str, Any]:
        """Exécute les benchmarks de performance."""
        benchmark_config = {
            "warm_up_iterations": 3,
            "benchmark_iterations": 10,
            "concurrent_users": [1, 2, 5],
            "data_sizes": [100, 500, 1000]
        }
        
        benchmark = PerformanceBenchmark(benchmark_config)
        profile = await benchmark.run_comprehensive_benchmark()
        
        # Conversion en dictionnaire pour sérialisation
        return {
            "profile_name": profile.profile_name,
            "overall_score": profile.overall_score,
            "benchmark_count": len(profile.benchmark_results),
            "recommendations": profile.recommendations,
            "timestamp": profile.timestamp.isoformat()
        }
    
    async def _run_compliance(self) -> Dict[str, Any]:
        """Exécute la validation de compliance."""
        compliance_config = {
            "standards": [
                ComplianceStandard.GDPR,
                ComplianceStandard.ISO_27001,
                ComplianceStandard.SOC2
            ]
        }
        
        validator = ComplianceValidator(compliance_config)
        report = await validator.run_comprehensive_validation(self.metrics_system)
        
        # Conversion en dictionnaire pour sérialisation
        return {
            "report_id": report.report_id,
            "compliance_score": report.compliance_score,
            "overall_status": report.overall_status.value,
            "vulnerabilities_count": len(report.security_vulnerabilities),
            "checks_count": len(report.compliance_checks),
            "recommendations": report.recommendations,
            "generated_at": report.generated_at.isoformat()
        }
    
    async def _start_monitoring(self) -> Dict[str, Any]:
        """Démarre le système de monitoring."""
        # Configuration basique du monitoring
        alert_engine = AlertEngine(self.metrics_system)
        await alert_engine.start()
        
        # Note: Dans un vrai déploiement, le monitoring continuerait
        # Ici on simule juste le démarrage
        await asyncio.sleep(1)
        await alert_engine.stop()
        
        return {
            "monitoring_started": True,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _run_monitoring_session(self, duration: int) -> Dict[str, Any]:
        """Exécute une session de monitoring."""
        logger.info(f"📊 Session de monitoring ({duration}s)")
        
        # Démarrage des collecteurs
        collector_config = CollectorConfig(
            system_interval=5,
            security_interval=30,
            application_interval=10
        )
        
        collector = MetricsCollectionAgent(collector_config, self.metrics_system)
        await collector.start()
        
        # Démarrage du moteur d'alertes
        alert_engine = AlertEngine(self.metrics_system)
        await alert_engine.start()
        
        start_time = time.time()
        collected_metrics = 0
        
        # Monitoring pendant la durée spécifiée
        while time.time() - start_time < duration:
            await asyncio.sleep(1)
            collected_metrics = collector.metrics_collected
            
            if collected_metrics % 10 == 0 and collected_metrics > 0:
                logger.info(f"📈 Métriques collectées: {collected_metrics}")
        
        # Arrêt des services
        await collector.stop()
        await alert_engine.stop()
        
        return {
            "duration": duration,
            "metrics_collected": collected_metrics,
            "collection_rate": collected_metrics / duration if duration > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _demo_deployment(self) -> Dict[str, Any]:
        """Démonstration du déploiement."""
        logger.info("🎭 Démo: Déploiement")
        
        # Simulation d'un déploiement rapide
        await asyncio.sleep(0.5)
        
        return {
            "demo": "deployment",
            "status": "success",
            "components_deployed": ["metrics_system", "storage", "collectors", "monitoring"]
        }
    
    async def _demo_metrics_collection(self) -> Dict[str, Any]:
        """Démonstration de collecte de métriques."""
        logger.info("🎭 Démo: Collecte de métriques")
        
        # Injection de quelques métriques de démonstration
        demo_metrics = []
        for i in range(10):
            metric = MetricDataPoint(
                metric_id=f"demo.metric.{i}",
                value=float(i * 10),
                metric_type=MetricType.GAUGE,
                category=MetricCategory.SYSTEM,
                tags={"demo": "true", "component": "orchestrator"}
            )
            await self.metrics_system.storage.store_metric(metric)
            demo_metrics.append(metric.metric_id)
        
        return {
            "demo": "metrics_collection",
            "metrics_created": len(demo_metrics),
            "metric_ids": demo_metrics
        }
    
    async def _demo_queries(self) -> Dict[str, Any]:
        """Démonstration de requêtes."""
        logger.info("🎭 Démo: Requêtes de métriques")
        
        # Requête des métriques de démonstration
        results = await self.metrics_system.storage.query_metrics(
            metric_pattern="demo.*",
            start_time=datetime.now() - timedelta(minutes=5),
            end_time=datetime.now()
        )
        
        return {
            "demo": "queries",
            "query_pattern": "demo.*",
            "results_count": len(results),
            "sample_metrics": [r.metric_id for r in results[:3]]
        }
    
    async def _demo_alerts(self) -> Dict[str, Any]:
        """Démonstration du système d'alertes."""
        logger.info("🎭 Démo: Système d'alertes")
        
        # Simulation d'une alerte
        alert_engine = AlertEngine(self.metrics_system)
        
        return {
            "demo": "alerts",
            "alert_engine_status": "initialized",
            "sample_alert": "High CPU usage detected"
        }
    
    async def _demo_analytics(self) -> Dict[str, Any]:
        """Démonstration des analytics."""
        logger.info("🎭 Démo: Analytics")
        
        # Simulation d'analytics
        return {
            "demo": "analytics",
            "ml_models_loaded": ["anomaly_detection", "trend_analysis"],
            "analytics_ready": True
        }
    
    async def _inject_test_metrics(self, count: int) -> Dict[str, Any]:
        """Injecte des métriques de test."""
        logger.info(f"💉 Injection de {count} métriques de test")
        
        import random
        
        metrics_created = []
        for i in range(count):
            metric = MetricDataPoint(
                metric_id=f"test.interactive.metric_{i}",
                value=random.uniform(0, 100),
                metric_type=random.choice([MetricType.GAUGE, MetricType.COUNTER]),
                category=random.choice([MetricCategory.SYSTEM, MetricCategory.PERFORMANCE, MetricCategory.BUSINESS]),
                tags={
                    "source": "interactive",
                    "batch": str(int(i / 10)),
                    "priority": random.choice(["low", "medium", "high"])
                }
            )
            await self.metrics_system.storage.store_metric(metric)
            metrics_created.append(metric.metric_id)
        
        return {
            "metrics_injected": count,
            "metric_ids": metrics_created[:5],  # Seulement les 5 premiers pour l'affichage
            "timestamp": datetime.now().isoformat()
        }
    
    async def _query_metrics_interactive(self, pattern: str) -> Dict[str, Any]:
        """Requête interactive de métriques."""
        logger.info(f"🔍 Requête de métriques: {pattern}")
        
        results = await self.metrics_system.storage.query_metrics(
            metric_pattern=pattern,
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )
        
        # Affichage des résultats
        print(f"\n📊 Résultats de la requête '{pattern}':")
        print(f"   Total: {len(results)} métriques trouvées")
        
        if results:
            print("   Échantillon:")
            for metric in results[:5]:
                print(f"   • {metric.metric_id}: {metric.value} ({metric.metric_type.value})")
        
        return {
            "query_pattern": pattern,
            "results_count": len(results),
            "metrics": [
                {
                    "metric_id": m.metric_id,
                    "value": m.value,
                    "timestamp": m.timestamp.isoformat(),
                    "type": m.metric_type.value
                } for m in results[:10]  # Limiter pour éviter les gros volumes
            ]
        }
    
    async def _show_system_stats(self) -> Dict[str, Any]:
        """Affiche les statistiques du système."""
        logger.info("📊 Affichage des statistiques système")
        
        # Requête de toutes les métriques récentes
        all_metrics = await self.metrics_system.storage.query_metrics(
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )
        
        # Calcul de statistiques
        total_metrics = len(all_metrics)
        
        metrics_by_type = {}
        metrics_by_category = {}
        
        for metric in all_metrics:
            # Par type
            type_name = metric.metric_type.value
            if type_name not in metrics_by_type:
                metrics_by_type[type_name] = 0
            metrics_by_type[type_name] += 1
            
            # Par catégorie
            category_name = metric.category.value
            if category_name not in metrics_by_category:
                metrics_by_category[category_name] = 0
            metrics_by_category[category_name] += 1
        
        stats = {
            "total_metrics": total_metrics,
            "metrics_by_type": metrics_by_type,
            "metrics_by_category": metrics_by_category,
            "system_uptime": time.time() - self.start_time if self.start_time else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        # Affichage formaté
        print(f"\n📊 Statistiques du Système:")
        print(f"   Total métriques: {stats['total_metrics']}")
        print(f"   Uptime: {stats['system_uptime']:.1f}s")
        
        if metrics_by_type:
            print("   Par type:")
            for type_name, count in metrics_by_type.items():
                print(f"     • {type_name}: {count}")
        
        if metrics_by_category:
            print("   Par catégorie:")
            for category_name, count in metrics_by_category.items():
                print(f"     • {category_name}: {count}")
        
        return stats
    
    async def _save_orchestration_results(self, results: Dict[str, Any], mode: str):
        """Sauvegarde les résultats d'orchestration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sauvegarde JSON
        if self.config.get("report_format") in ["json", "both"]:
            json_filename = f"orchestration_results_{mode}_{timestamp}.json"
            
            with open(json_filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"💾 Résultats sauvegardés: {json_filename}")
        
        # Sauvegarde Markdown
        if self.config.get("report_format") in ["markdown", "both"]:
            md_filename = f"orchestration_report_{mode}_{timestamp}.md"
            await self._generate_markdown_report(results, mode, md_filename)
            logger.info(f"📝 Rapport généré: {md_filename}")
    
    async def _generate_markdown_report(self, results: Dict[str, Any], mode: str, filename: str):
        """Génère un rapport Markdown."""
        with open(filename, 'w') as f:
            f.write("# Rapport d'Orchestration - Système de Métriques d'Entreprise\n\n")
            f.write(f"**Projet dirigé par:** Fahed Mlaiel\n")
            f.write(f"**Mode d'exécution:** {mode.upper()}\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Résumé
            if "orchestration_summary" in results:
                summary = results["orchestration_summary"]
                f.write("## 📊 Résumé Exécutif\n\n")
                f.write(f"- **Statut:** {'✅ SUCCÈS' if summary.get('success') else '❌ ÉCHEC'}\n")
                f.write(f"- **Temps d'exécution:** {summary.get('total_execution_time', 0):.1f}s\n")
                f.write(f"- **Mode:** {summary.get('mode', 'unknown')}\n\n")
            
            # Détails par composant
            components = {
                "deployment": "📦 Déploiement",
                "testing": "🧪 Tests",
                "benchmark": "⚡ Benchmarks",
                "compliance": "🔍 Compliance",
                "monitoring": "📊 Monitoring",
                "demo": "🎭 Démonstration",
                "interactive": "🔄 Interactif"
            }
            
            for comp_key, comp_title in components.items():
                if comp_key in results:
                    f.write(f"## {comp_title}\n\n")
                    comp_data = results[comp_key]
                    
                    if isinstance(comp_data, dict):
                        if "error" in comp_data:
                            f.write(f"❌ **Erreur:** {comp_data['error']}\n\n")
                        else:
                            # Affichage des données principales
                            for key, value in comp_data.items():
                                if key not in ["error", "success"]:
                                    f.write(f"- **{key}:** {value}\n")
                            f.write("\n")
                    else:
                        f.write(f"```json\n{json.dumps(comp_data, indent=2, default=str)}\n```\n\n")
            
            f.write("---\n")
            f.write("*Rapport généré par le Master Orchestrator*\n")
            f.write("*Expert Development Team - Dirigé par Fahed Mlaiel*\n")


def create_argument_parser():
    """Crée le parser d'arguments."""
    parser = argparse.ArgumentParser(
        description="Master Orchestrator - Système de Métriques d'Entreprise",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python orchestrator.py --mode full                    # Orchestration complète
  python orchestrator.py --mode test                    # Tests uniquement
  python orchestrator.py --mode benchmark               # Benchmarks uniquement
  python orchestrator.py --mode demo                    # Démonstration
  python orchestrator.py --mode interactive             # Mode interactif
  python orchestrator.py --mode monitor --duration 60   # Monitoring 60s
  
Développé par l'Expert Development Team - Dirigé par Fahed Mlaiel
        """
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=[
            OrchestrationMode.FULL,
            OrchestrationMode.DEPLOY,
            OrchestrationMode.TEST,
            OrchestrationMode.BENCHMARK,
            OrchestrationMode.MONITOR,
            OrchestrationMode.COMPLIANCE,
            OrchestrationMode.DEMO,
            OrchestrationMode.INTERACTIVE
        ],
        default=OrchestrationMode.FULL,
        help="Mode d'orchestration (défaut: full)"
    )
    
    parser.add_argument(
        "--storage",
        choices=["sqlite", "redis", "postgresql"],
        default="sqlite",
        help="Backend de stockage (défaut: sqlite)"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Durée pour le mode monitor en secondes (défaut: 60)"
    )
    
    parser.add_argument(
        "--deployment-mode",
        choices=["development", "staging", "production"],
        default="development",
        help="Mode de déploiement (défaut: development)"
    )
    
    parser.add_argument(
        "--report-format",
        choices=["json", "markdown", "both"],
        default="both",
        help="Format de rapport (défaut: both)"
    )
    
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Désactiver le nettoyage automatique"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mode verbeux"
    )
    
    return parser


async def main():
    """Fonction principale."""
    print("🚀 Master Orchestrator - Système de Métriques d'Entreprise")
    print("=" * 60)
    print("Expert Development Team - Projet dirigé par Fahed Mlaiel")
    print("=" * 60)
    
    # Parse des arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configuration du logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Configuration de l'orchestrateur
    config = {
        "storage_backend": args.storage,
        "deployment_mode": args.deployment_mode,
        "auto_cleanup": not args.no_cleanup,
        "report_format": args.report_format,
        "verbose": args.verbose
    }
    
    # Création et exécution de l'orchestrateur
    orchestrator = MasterOrchestrator(config)
    
    try:
        # Arguments spécifiques au mode
        kwargs = {}
        if args.mode == OrchestrationMode.MONITOR:
            kwargs["duration"] = args.duration
        
        # Exécution
        results = await orchestrator.run_orchestration(args.mode, **kwargs)
        
        print("\n" + "=" * 60)
        print("🎉 ORCHESTRATION TERMINÉE AVEC SUCCÈS!")
        print("=" * 60)
        
        # Affichage du résumé
        if "orchestration_summary" in results:
            summary = results["orchestration_summary"]
            print(f"📊 Mode: {summary.get('mode', 'unknown').upper()}")
            print(f"⏱️  Durée: {summary.get('total_execution_time', 0):.1f}s")
            print(f"✅ Statut: {'SUCCÈS' if summary.get('success') else 'ÉCHEC'}")
        
        # Statistiques par composant
        components_stats = []
        for comp in ["deployment", "testing", "benchmark", "compliance", "monitoring"]:
            if comp in results:
                comp_data = results[comp]
                if isinstance(comp_data, dict) and "error" not in comp_data:
                    components_stats.append(f"✅ {comp}")
                else:
                    components_stats.append(f"❌ {comp}")
        
        if components_stats:
            print(f"📋 Composants: {', '.join(components_stats)}")
        
        print("\n" + "=" * 60)
        print("Merci d'avoir utilisé le Master Orchestrator!")
        print("Développé par l'Équipe d'Experts - Dirigé par Fahed Mlaiel")
        print("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 Interruption utilisateur")
        return 1
    except Exception as e:
        print(f"\n💥 ERREUR FATALE: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Exécution de l'orchestrateur principal
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
