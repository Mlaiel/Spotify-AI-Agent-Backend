#!/usr/bin/env python3
"""
Configuration Management Scripts Module
======================================

Module principal pour la gestion automatisée des configurations Kubernetes.
Fournit une interface Python pour tous les outils de gestion des configurations.

Ce module orchestre:
- Génération automatisée des configurations
- Validation multi-niveaux avec conformité sécurité
- Déploiement intelligent avec rollback automatique
- Surveillance en temps réel avec alertes
- Export de métriques pour observabilité

Classes principales:
- ConfigurationManager: Gestionnaire principal
- ConfigurationPipeline: Pipeline d'automation complète
- MetricsCollector: Collecteur de métriques avancé
- SecurityValidator: Validateur de sécurité spécialisé
- DeploymentOrchestrator: Orchestrateur de déploiements

Version: 2.0.0
Python: >=3.8
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
from contextlib import contextmanager

# Configuration du logging avancé
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/config_management.log')
    ]
)

logger = logging.getLogger(__name__)

class ConfigurationEnvironment(Enum):
    """Environnements de configuration supportés."""
    LOCAL = "local"
    DEVELOPMENT = "dev"
    STAGING = "staging"
    PRODUCTION = "prod"
    TEST = "test"

class DeploymentStrategy(Enum):
    """Stratégies de déploiement."""
    ROLLING_UPDATE = "rolling_update"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"

class SecurityLevel(Enum):
    """Niveaux de sécurité."""
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ConfigurationContext:
    """Contexte de configuration pour les opérations."""
    environment: ConfigurationEnvironment
    namespace: str
    security_level: SecurityLevel = SecurityLevel.STANDARD
    deployment_strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE
    dry_run: bool = False
    backup_enabled: bool = True
    monitoring_enabled: bool = True
    compliance_checks: bool = True
    custom_labels: Dict[str, str] = field(default_factory=dict)
    custom_annotations: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validation post-initialisation."""
        if not self.namespace:
            raise ValueError("Namespace cannot be empty")
        
        # Labels par défaut
        default_labels = {
            "app": "spotify-ai-agent",
            "environment": self.environment.value,
            "managed-by": "config-management-suite",
            "version": "2.0.0"
        }
        self.custom_labels = {**default_labels, **self.custom_labels}
        
        # Annotations par défaut
        default_annotations = {
            "config.spotify-ai-agent.io/managed": "true",
            "config.spotify-ai-agent.io/created-at": datetime.now().isoformat(),
            "config.spotify-ai-agent.io/security-level": self.security_level.value
        }
        self.custom_annotations = {**default_annotations, **self.custom_annotations}

class ConfigurationResult:
    """Résultat d'une opération de configuration."""
    
    def __init__(self, success: bool = True, message: str = "", 
                 data: Optional[Dict[str, Any]] = None, 
                 errors: Optional[List[str]] = None,
                 warnings: Optional[List[str]] = None):
        self.success = success
        self.message = message
        self.data = data or {}
        self.errors = errors or []
        self.warnings = warnings or []
        self.timestamp = datetime.now()
        self.execution_time = None
    
    def add_error(self, error: str):
        """Ajoute une erreur au résultat."""
        self.errors.append(error)
        self.success = False
    
    def add_warning(self, warning: str):
        """Ajoute un avertissement au résultat."""
        self.warnings.append(warning)
    
    def set_execution_time(self, start_time: datetime):
        """Définit le temps d'exécution."""
        self.execution_time = (datetime.now() - start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "errors": self.errors,
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat(),
            "execution_time": self.execution_time
        }

class ConfigurationManager:
    """Gestionnaire principal des configurations."""
    
    def __init__(self, context: ConfigurationContext):
        self.context = context
        self.logger = logging.getLogger(f"{__name__}.ConfigurationManager")
        self._operation_queue = queue.Queue()
        self._metrics_history = []
        self._active_operations = {}
        
        # Initialisation des composants
        self._init_components()
    
    def _init_components(self):
        """Initialise les composants internes."""
        self.logger.info("Initialisation du gestionnaire de configurations")
        self.logger.info(f"Environnement: {self.context.environment.value}")
        self.logger.info(f"Namespace: {self.context.namespace}")
        self.logger.info(f"Niveau de sécurité: {self.context.security_level.value}")
    
    async def generate_configurations(self, **kwargs) -> ConfigurationResult:
        """Génère les configurations de manière asynchrone."""
        start_time = datetime.now()
        result = ConfigurationResult()
        
        try:
            self.logger.info("Démarrage de la génération des configurations")
            
            # Import dynamique pour éviter les dépendances circulaires
            from .generate_configs import ConfigurationGenerator
            
            generator = ConfigurationGenerator(self.context)
            await generator.generate_all(**kwargs)
            
            result.message = "Configurations générées avec succès"
            result.data["generated_files"] = generator.get_generated_files()
            
        except Exception as e:
            result.add_error(f"Erreur lors de la génération: {str(e)}")
            self.logger.error(f"Génération échouée: {e}")
        
        result.set_execution_time(start_time)
        return result
    
    async def validate_configurations(self, **kwargs) -> ConfigurationResult:
        """Valide les configurations de manière asynchrone."""
        start_time = datetime.now()
        result = ConfigurationResult()
        
        try:
            self.logger.info("Démarrage de la validation des configurations")
            
            from .validate_configs import ComprehensiveValidator
            
            validator = ComprehensiveValidator(self.context)
            validation_result = await validator.validate_all(**kwargs)
            
            if validation_result.overall_score >= 80:
                result.message = f"Validation réussie (score: {validation_result.overall_score}%)"
            else:
                result.add_warning(f"Score de validation faible: {validation_result.overall_score}%")
            
            result.data["validation_score"] = validation_result.overall_score
            result.data["validation_details"] = validation_result.to_dict()
            
        except Exception as e:
            result.add_error(f"Erreur lors de la validation: {str(e)}")
            self.logger.error(f"Validation échouée: {e}")
        
        result.set_execution_time(start_time)
        return result
    
    async def deploy_configurations(self, **kwargs) -> ConfigurationResult:
        """Déploie les configurations de manière asynchrone."""
        start_time = datetime.now()
        result = ConfigurationResult()
        
        try:
            self.logger.info("Démarrage du déploiement des configurations")
            
            from .deploy_configs import ConfigurationDeployer
            
            deployer = ConfigurationDeployer(
                namespace=self.context.namespace,
                dry_run=self.context.dry_run
            )
            
            # Vérifications préalables
            if not deployer.check_prerequisites():
                result.add_error("Prérequis de déploiement non satisfaits")
                return result
            
            # Déploiement
            configs = deployer.load_configurations(Path(kwargs.get("config_dir", "./configs")))
            if deployer.deploy_configurations(configs):
                result.message = "Déploiement réussi"
                result.data["deployed_resources"] = deployer.applied_resources
            else:
                result.add_error("Échec du déploiement")
                result.data["failed_resources"] = deployer.failed_resources
            
        except Exception as e:
            result.add_error(f"Erreur lors du déploiement: {str(e)}")
            self.logger.error(f"Déploiement échoué: {e}")
        
        result.set_execution_time(start_time)
        return result
    
    async def monitor_configurations(self, duration: Optional[int] = None) -> ConfigurationResult:
        """Surveille les configurations de manière asynchrone."""
        start_time = datetime.now()
        result = ConfigurationResult()
        
        try:
            self.logger.info("Démarrage de la surveillance des configurations")
            
            from .monitor_configs import ConfigurationMonitor
            
            monitor = ConfigurationMonitor(namespace=self.context.namespace)
            status = monitor.get_configuration_status()
            
            result.message = f"Surveillance terminée (score de santé: {status['health_score']:.1f}%)"
            result.data["monitoring_status"] = status
            
            if status["health_score"] < 50:
                result.add_warning("Score de santé critique détecté")
            
        except Exception as e:
            result.add_error(f"Erreur lors de la surveillance: {str(e)}")
            self.logger.error(f"Surveillance échouée: {e}")
        
        result.set_execution_time(start_time)
        return result

class ConfigurationPipeline:
    """Pipeline d'automation complète pour les configurations."""
    
    def __init__(self, context: ConfigurationContext):
        self.context = context
        self.manager = ConfigurationManager(context)
        self.logger = logging.getLogger(f"{__name__}.ConfigurationPipeline")
        self._pipeline_results = []
    
    async def execute_full_pipeline(self, **kwargs) -> List[ConfigurationResult]:
        """Exécute le pipeline complet de gestion des configurations."""
        self.logger.info("Démarrage du pipeline complet")
        
        results = []
        
        # Étape 1: Génération
        self.logger.info("=== ÉTAPE 1: GÉNÉRATION ===")
        gen_result = await self.manager.generate_configurations(**kwargs)
        results.append(gen_result)
        
        if not gen_result.success:
            self.logger.error("Arrêt du pipeline - génération échouée")
            return results
        
        # Étape 2: Validation
        self.logger.info("=== ÉTAPE 2: VALIDATION ===")
        val_result = await self.manager.validate_configurations(**kwargs)
        results.append(val_result)
        
        if not val_result.success:
            self.logger.error("Arrêt du pipeline - validation échouée")
            return results
        
        # Étape 3: Déploiement (si pas en dry-run)
        if not self.context.dry_run:
            self.logger.info("=== ÉTAPE 3: DÉPLOIEMENT ===")
            deploy_result = await self.manager.deploy_configurations(**kwargs)
            results.append(deploy_result)
            
            if deploy_result.success and self.context.monitoring_enabled:
                # Étape 4: Surveillance
                self.logger.info("=== ÉTAPE 4: SURVEILLANCE ===")
                monitor_result = await self.manager.monitor_configurations(duration=60)
                results.append(monitor_result)
        else:
            self.logger.info("=== MODE DRY-RUN: Déploiement sauté ===")
        
        self._pipeline_results = results
        self.logger.info("Pipeline complet terminé")
        
        return results
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Retourne un résumé du pipeline."""
        if not self._pipeline_results:
            return {"status": "not_executed"}
        
        total_success = all(result.success for result in self._pipeline_results)
        total_warnings = sum(len(result.warnings) for result in self._pipeline_results)
        total_errors = sum(len(result.errors) for result in self._pipeline_results)
        
        return {
            "status": "success" if total_success else "failed",
            "total_steps": len(self._pipeline_results),
            "successful_steps": sum(1 for result in self._pipeline_results if result.success),
            "total_warnings": total_warnings,
            "total_errors": total_errors,
            "execution_time": sum(
                result.execution_time for result in self._pipeline_results 
                if result.execution_time
            ),
            "results": [result.to_dict() for result in self._pipeline_results]
        }

class MetricsCollector:
    """Collecteur de métriques avancé."""
    
    def __init__(self, context: ConfigurationContext):
        self.context = context
        self.logger = logging.getLogger(f"{__name__}.MetricsCollector")
        self._metrics_buffer = []
        self._collectors = {}
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collecte les métriques système."""
        import psutil
        
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
        }
    
    def collect_kubernetes_metrics(self) -> Dict[str, Any]:
        """Collecte les métriques Kubernetes."""
        try:
            from kubernetes import client, config
            
            # Chargement de la configuration
            try:
                config.load_incluster_config()
            except:
                config.load_kube_config()
            
            v1 = client.CoreV1Api()
            
            # Métriques des pods
            pods = v1.list_namespaced_pod(namespace=self.context.namespace)
            pod_phases = {}
            for pod in pods.items:
                phase = pod.status.phase
                pod_phases[phase] = pod_phases.get(phase, 0) + 1
            
            return {
                "total_pods": len(pods.items),
                "pod_phases": pod_phases,
                "namespace": self.context.namespace
            }
            
        except Exception as e:
            self.logger.warning(f"Impossible de collecter les métriques Kubernetes: {e}")
            return {}
    
    def export_prometheus_metrics(self) -> str:
        """Exporte les métriques au format Prometheus."""
        lines = []
        timestamp = int(datetime.now().timestamp() * 1000)
        
        # Métriques système
        system_metrics = self.collect_system_metrics()
        for metric_name, value in system_metrics.items():
            if value is not None:
                lines.append(f'config_management_{metric_name}{{namespace="{self.context.namespace}"}} {value} {timestamp}')
        
        # Métriques Kubernetes
        k8s_metrics = self.collect_kubernetes_metrics()
        for metric_name, value in k8s_metrics.items():
            if isinstance(value, (int, float)):
                lines.append(f'config_management_k8s_{metric_name}{{namespace="{self.context.namespace}"}} {value} {timestamp}')
        
        return '\n'.join(lines)

# Fonctions utilitaires
@contextmanager
def configuration_context(environment: str, namespace: str, **kwargs):
    """Context manager pour les opérations de configuration."""
    env = ConfigurationEnvironment(environment)
    context = ConfigurationContext(
        environment=env,
        namespace=namespace,
        **kwargs
    )
    
    manager = ConfigurationManager(context)
    try:
        yield manager
    finally:
        # Nettoyage si nécessaire
        pass

async def run_configuration_pipeline(environment: str, namespace: str, **kwargs) -> List[ConfigurationResult]:
    """Fonction de commodité pour exécuter un pipeline complet."""
    env = ConfigurationEnvironment(environment)
    context = ConfigurationContext(
        environment=env,
        namespace=namespace,
        **kwargs
    )
    
    pipeline = ConfigurationPipeline(context)
    return await pipeline.execute_full_pipeline(**kwargs)

def get_configuration_status(namespace: str) -> Dict[str, Any]:
    """Fonction de commodité pour obtenir le statut des configurations."""
    from .monitor_configs import ConfigurationMonitor
    
    monitor = ConfigurationMonitor(namespace=namespace)
    return monitor.get_configuration_status()

# Exportation des classes principales
__all__ = [
    'ConfigurationManager',
    'ConfigurationPipeline', 
    'ConfigurationContext',
    'ConfigurationResult',
    'ConfigurationEnvironment',
    'DeploymentStrategy',
    'SecurityLevel',
    'MetricsCollector',
    'configuration_context',
    'run_configuration_pipeline',
    'get_configuration_status'
]

# Information du module
__version__ = "2.0.0"
__author__ = "Configuration Management Team"
__email__ = "devops@spotify-ai-agent.com"
__description__ = "Suite complète de gestion des configurations Kubernetes"

# Configuration du module
if __name__ == "__main__":
    print(f"Configuration Management Suite v{__version__}")
    print("Ce module fournit une interface complète pour la gestion des configurations Kubernetes.")
    print("Utilisez les scripts individuels ou importez ce module pour une intégration Python.")
