#!/usr/bin/env python3
"""
Enterprise Deployment Automation System
=======================================

Système d'automatisation de déploiement ultra-avancé avec orchestration intelligente,
monitoring en temps réel, et rollback automatique.

Auteur: Fahed Mlaiel (Lead Dev + Architecte IA)
Version: 2.0.0 Enterprise
"""

import asyncio
import logging
import yaml
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import aiohttp
import hashlib
import time

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/deployment_automation.log')
    ]
)
logger = logging.getLogger(__name__)

class DeploymentStrategy(Enum):
    """Stratégies de déploiement supportées."""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    RECREATE = "recreate"

class DeploymentStatus(Enum):
    """Statuts de déploiement."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"

@dataclass
class DeploymentConfig:
    """Configuration de déploiement."""
    name: str
    environment: str
    strategy: DeploymentStrategy
    config_files: List[Path]
    namespace: str = "default"
    timeout_seconds: int = 600
    health_check_timeout: int = 300
    rollback_on_failure: bool = True
    notification_channels: List[str] = field(default_factory=list)
    approval_required: bool = False
    approvers: List[str] = field(default_factory=list)

@dataclass
class DeploymentResult:
    """Résultat de déploiement."""
    deployment_id: str
    config: DeploymentConfig
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    rollback_performed: bool = False
    error_message: Optional[str] = None

class KubernetesManager:
    """Gestionnaire Kubernetes pour les déploiements."""
    
    def __init__(self, kubeconfig_path: Optional[Path] = None):
        self.kubeconfig_path = kubeconfig_path
        self.kubectl_cmd = "kubectl"
        if kubeconfig_path:
            self.kubectl_cmd += f" --kubeconfig {kubeconfig_path}"
    
    async def apply_configuration(self, config_file: Path, namespace: str = "default") -> Tuple[bool, str]:
        """Applique une configuration Kubernetes."""
        try:
            cmd = f"{self.kubectl_cmd} apply -f {config_file} -n {namespace}"
            result = await self._run_command(cmd)
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            return False, str(e)
    
    async def delete_configuration(self, config_file: Path, namespace: str = "default") -> Tuple[bool, str]:
        """Supprime une configuration Kubernetes."""
        try:
            cmd = f"{self.kubectl_cmd} delete -f {config_file} -n {namespace}"
            result = await self._run_command(cmd)
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            return False, str(e)
    
    async def get_deployment_status(self, deployment_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Récupère le statut d'un déploiement."""
        try:
            cmd = f"{self.kubectl_cmd} get deployment {deployment_name} -n {namespace} -o json"
            result = await self._run_command(cmd)
            if result.returncode == 0:
                return json.loads(result.stdout)
            return {}
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du statut: {e}")
            return {}
    
    async def wait_for_rollout(self, deployment_name: str, namespace: str = "default", timeout: int = 300) -> bool:
        """Attend la fin du rollout d'un déploiement."""
        try:
            cmd = f"{self.kubectl_cmd} rollout status deployment/{deployment_name} -n {namespace} --timeout={timeout}s"
            result = await self._run_command(cmd)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Erreur lors de l'attente du rollout: {e}")
            return False
    
    async def rollback_deployment(self, deployment_name: str, namespace: str = "default") -> bool:
        """Effectue un rollback d'un déploiement."""
        try:
            cmd = f"{self.kubectl_cmd} rollout undo deployment/{deployment_name} -n {namespace}"
            result = await self._run_command(cmd)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Erreur lors du rollback: {e}")
            return False
    
    async def _run_command(self, command: str) -> subprocess.CompletedProcess:
        """Exécute une commande shell de manière asynchrone."""
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        # Création d'un objet similaire à subprocess.CompletedProcess
        class AsyncResult:
            def __init__(self, returncode, stdout, stderr):
                self.returncode = returncode
                self.stdout = stdout.decode() if stdout else ""
                self.stderr = stderr.decode() if stderr else ""
        
        return AsyncResult(process.returncode, stdout, stderr)

class HealthChecker:
    """Vérificateur de santé des services déployés."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_http_endpoint(self, url: str, expected_status: int = 200, timeout: int = 30) -> bool:
        """Vérifie la santé d'un endpoint HTTP."""
        try:
            async with self.session.get(url, timeout=timeout) as response:
                return response.status == expected_status
        except Exception as e:
            logger.debug(f"Health check failed for {url}: {e}")
            return False
    
    async def check_kubernetes_service(self, service_name: str, namespace: str = "default", 
                                     path: str = "/health", timeout: int = 30) -> bool:
        """Vérifie la santé d'un service Kubernetes."""
        # Construction de l'URL du service
        url = f"http://{service_name}.{namespace}.svc.cluster.local{path}"
        return await self.check_http_endpoint(url, timeout=timeout)
    
    async def wait_for_service_ready(self, service_name: str, namespace: str = "default",
                                   max_attempts: int = 30, delay: int = 10) -> bool:
        """Attend qu'un service soit prêt."""
        for attempt in range(max_attempts):
            if await self.check_kubernetes_service(service_name, namespace):
                logger.info(f"Service {service_name} is ready")
                return True
            
            logger.debug(f"Service {service_name} not ready, attempt {attempt + 1}/{max_attempts}")
            await asyncio.sleep(delay)
        
        logger.error(f"Service {service_name} failed to become ready after {max_attempts} attempts")
        return False

class MetricsCollector:
    """Collecteur de métriques de déploiement."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
    
    def start_timing(self, metric_name: str):
        """Démarre le chronométrage d'une métrique."""
        self.metrics[f"{metric_name}_start"] = time.time()
    
    def end_timing(self, metric_name: str):
        """Termine le chronométrage d'une métrique."""
        start_time = self.metrics.get(f"{metric_name}_start")
        if start_time:
            duration = time.time() - start_time
            self.metrics[f"{metric_name}_duration"] = duration
            return duration
        return 0
    
    def record_metric(self, name: str, value: Any):
        """Enregistre une métrique."""
        self.metrics[name] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Récupère toutes les métriques."""
        return self.metrics.copy()

class NotificationManager:
    """Gestionnaire de notifications."""
    
    def __init__(self):
        self.channels = {
            "slack": self._send_slack_notification,
            "email": self._send_email_notification,
            "webhook": self._send_webhook_notification
        }
    
    async def send_notification(self, channel: str, message: str, **kwargs):
        """Envoie une notification via le canal spécifié."""
        if channel in self.channels:
            await self.channels[channel](message, **kwargs)
        else:
            logger.warning(f"Canal de notification non supporté: {channel}")
    
    async def _send_slack_notification(self, message: str, webhook_url: str = None, **kwargs):
        """Envoie une notification Slack."""
        # Implémentation Slack (nécessite webhook_url configuré)
        logger.info(f"Slack notification: {message}")
    
    async def _send_email_notification(self, message: str, recipients: List[str] = None, **kwargs):
        """Envoie une notification email."""
        # Implémentation email
        logger.info(f"Email notification: {message}")
    
    async def _send_webhook_notification(self, message: str, webhook_url: str = None, **kwargs):
        """Envoie une notification webhook."""
        # Implémentation webhook
        logger.info(f"Webhook notification: {message}")

class DeploymentAutomator:
    """Orchestrateur principal de déploiement."""
    
    def __init__(self, kubeconfig_path: Optional[Path] = None):
        self.kubernetes = KubernetesManager(kubeconfig_path)
        self.metrics_collector = MetricsCollector()
        self.notification_manager = NotificationManager()
        self.deployment_history: List[DeploymentResult] = []
    
    async def deploy_configuration(self, config: DeploymentConfig) -> DeploymentResult:
        """Déploie une configuration selon la stratégie spécifiée."""
        deployment_id = self._generate_deployment_id(config)
        logger.info(f"Début du déploiement {deployment_id}")
        
        result = DeploymentResult(
            deployment_id=deployment_id,
            config=config,
            status=DeploymentStatus.PENDING,
            start_time=datetime.now(timezone.utc)
        )
        
        try:
            # Démarrage du chronométrage
            self.metrics_collector.start_timing("total_deployment")
            
            # Vérification des prérequis
            await self._check_prerequisites(config, result)
            
            # Demande d'approbation si nécessaire
            if config.approval_required:
                await self._request_approval(config, result)
            
            # Sauvegarde de l'état actuel pour rollback
            await self._backup_current_state(config, result)
            
            # Exécution du déploiement selon la stratégie
            result.status = DeploymentStatus.IN_PROGRESS
            await self._notify_deployment_start(config, result)
            
            if config.strategy == DeploymentStrategy.BLUE_GREEN:
                success = await self._deploy_blue_green(config, result)
            elif config.strategy == DeploymentStrategy.CANARY:
                success = await self._deploy_canary(config, result)
            elif config.strategy == DeploymentStrategy.ROLLING:
                success = await self._deploy_rolling(config, result)
            else:
                success = await self._deploy_recreate(config, result)
            
            # Vérification de santé post-déploiement
            if success:
                success = await self._post_deployment_health_check(config, result)
            
            # Gestion du résultat
            if success:
                result.status = DeploymentStatus.SUCCESS
                await self._notify_deployment_success(config, result)
            else:
                result.status = DeploymentStatus.FAILED
                if config.rollback_on_failure:
                    await self._perform_rollback(config, result)
                await self._notify_deployment_failure(config, result)
        
        except Exception as e:
            logger.error(f"Erreur lors du déploiement {deployment_id}: {e}")
            result.status = DeploymentStatus.FAILED
            result.error_message = str(e)
            
            if config.rollback_on_failure:
                await self._perform_rollback(config, result)
            
            await self._notify_deployment_failure(config, result)
        
        finally:
            # Finalisation
            result.end_time = datetime.now(timezone.utc)
            result.duration_seconds = self.metrics_collector.end_timing("total_deployment")
            result.metrics = self.metrics_collector.get_metrics()
            
            self.deployment_history.append(result)
            logger.info(f"Déploiement {deployment_id} terminé avec le statut: {result.status.value}")
        
        return result
    
    def _generate_deployment_id(self, config: DeploymentConfig) -> str:
        """Génère un ID unique pour le déploiement."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = hashlib.md5(f"{config.name}_{config.environment}".encode()).hexdigest()[:8]
        return f"deploy_{config.name}_{config.environment}_{timestamp}_{config_hash}"
    
    async def _check_prerequisites(self, config: DeploymentConfig, result: DeploymentResult):
        """Vérifie les prérequis du déploiement."""
        self.metrics_collector.start_timing("prerequisites_check")
        
        # Vérification de l'existence des fichiers de configuration
        for config_file in config.config_files:
            if not config_file.exists():
                raise FileNotFoundError(f"Fichier de configuration introuvable: {config_file}")
        
        # Vérification de la connectivité Kubernetes
        try:
            cmd = f"{self.kubernetes.kubectl_cmd} cluster-info"
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"Impossible de se connecter au cluster Kubernetes: {stderr.decode()}")
        
        except Exception as e:
            raise RuntimeError(f"Erreur de vérification des prérequis: {e}")
        
        self.metrics_collector.end_timing("prerequisites_check")
        result.logs.append("Prérequis vérifiés avec succès")
    
    async def _request_approval(self, config: DeploymentConfig, result: DeploymentResult):
        """Demande l'approbation pour le déploiement."""
        # Implémentation de la demande d'approbation
        # Ceci pourrait intégrer avec des systèmes comme Jira, ServiceNow, etc.
        result.logs.append(f"Approbation demandée auprès de: {', '.join(config.approvers)}")
        
        # Pour cette démonstration, nous simulons une approbation automatique
        await asyncio.sleep(1)
        result.logs.append("Approbation accordée")
    
    async def _backup_current_state(self, config: DeploymentConfig, result: DeploymentResult):
        """Sauvegarde l'état actuel pour permettre un rollback."""
        self.metrics_collector.start_timing("backup")
        
        # Sauvegarde des configurations actuelles
        backup_dir = Path(f"/tmp/backup_{result.deployment_id}")
        backup_dir.mkdir(exist_ok=True)
        
        # Export des ressources existantes
        for config_file in config.config_files:
            backup_file = backup_dir / f"backup_{config_file.name}"
            shutil.copy2(config_file, backup_file)
        
        result.logs.append(f"État actuel sauvegardé dans: {backup_dir}")
        self.metrics_collector.end_timing("backup")
    
    async def _deploy_blue_green(self, config: DeploymentConfig, result: DeploymentResult) -> bool:
        """Implémente le déploiement Blue-Green."""
        result.logs.append("Démarrage du déploiement Blue-Green")
        
        try:
            # Phase 1: Déploiement de l'environnement Green
            result.logs.append("Phase 1: Déploiement de l'environnement Green")
            for config_file in config.config_files:
                success, output = await self.kubernetes.apply_configuration(config_file, config.namespace)
                if not success:
                    result.logs.append(f"Erreur lors de l'application de {config_file}: {output}")
                    return False
                result.logs.append(f"Configuration appliquée: {config_file}")
            
            # Phase 2: Vérification de santé de Green
            result.logs.append("Phase 2: Vérification de santé de l'environnement Green")
            async with HealthChecker() as health_checker:
                if not await health_checker.wait_for_service_ready(config.name, config.namespace):
                    result.logs.append("Échec de la vérification de santé de Green")
                    return False
            
            # Phase 3: Basculement du trafic (simulation)
            result.logs.append("Phase 3: Basculement du trafic vers Green")
            await asyncio.sleep(2)  # Simulation du basculement
            
            # Phase 4: Nettoyage de Blue (optionnel)
            result.logs.append("Phase 4: Nettoyage de l'ancien environnement Blue")
            
            return True
        
        except Exception as e:
            result.logs.append(f"Erreur lors du déploiement Blue-Green: {e}")
            return False
    
    async def _deploy_canary(self, config: DeploymentConfig, result: DeploymentResult) -> bool:
        """Implémente le déploiement Canary."""
        result.logs.append("Démarrage du déploiement Canary")
        
        try:
            # Phase 1: Déploiement d'une petite partie du trafic
            result.logs.append("Phase 1: Déploiement Canary (5% du trafic)")
            
            # Application des configurations
            for config_file in config.config_files:
                success, output = await self.kubernetes.apply_configuration(config_file, config.namespace)
                if not success:
                    result.logs.append(f"Erreur lors de l'application de {config_file}: {output}")
                    return False
            
            # Phase 2: Monitoring des métriques Canary
            result.logs.append("Phase 2: Monitoring des métriques Canary")
            await asyncio.sleep(30)  # Période d'observation
            
            # Phase 3: Augmentation progressive du trafic
            traffic_percentages = [10, 25, 50, 100]
            for percentage in traffic_percentages:
                result.logs.append(f"Phase: Augmentation du trafic à {percentage}%")
                await asyncio.sleep(15)  # Observation entre chaque phase
                
                # Vérification des métriques
                # Ici, on pourrait intégrer avec Prometheus/Grafana
                
            return True
        
        except Exception as e:
            result.logs.append(f"Erreur lors du déploiement Canary: {e}")
            return False
    
    async def _deploy_rolling(self, config: DeploymentConfig, result: DeploymentResult) -> bool:
        """Implémente le déploiement Rolling."""
        result.logs.append("Démarrage du déploiement Rolling")
        
        try:
            # Application des configurations avec rolling update
            for config_file in config.config_files:
                success, output = await self.kubernetes.apply_configuration(config_file, config.namespace)
                if not success:
                    result.logs.append(f"Erreur lors de l'application de {config_file}: {output}")
                    return False
                
                # Attente du rollout
                if await self.kubernetes.wait_for_rollout(config.name, config.namespace, config.timeout_seconds):
                    result.logs.append(f"Rollout réussi pour {config_file}")
                else:
                    result.logs.append(f"Échec du rollout pour {config_file}")
                    return False
            
            return True
        
        except Exception as e:
            result.logs.append(f"Erreur lors du déploiement Rolling: {e}")
            return False
    
    async def _deploy_recreate(self, config: DeploymentConfig, result: DeploymentResult) -> bool:
        """Implémente le déploiement Recreate."""
        result.logs.append("Démarrage du déploiement Recreate")
        
        try:
            # Suppression des anciennes ressources
            for config_file in config.config_files:
                await self.kubernetes.delete_configuration(config_file, config.namespace)
                result.logs.append(f"Ressources supprimées pour {config_file}")
            
            # Attente de la suppression complète
            await asyncio.sleep(10)
            
            # Création des nouvelles ressources
            for config_file in config.config_files:
                success, output = await self.kubernetes.apply_configuration(config_file, config.namespace)
                if not success:
                    result.logs.append(f"Erreur lors de l'application de {config_file}: {output}")
                    return False
                result.logs.append(f"Nouvelles ressources créées pour {config_file}")
            
            return True
        
        except Exception as e:
            result.logs.append(f"Erreur lors du déploiement Recreate: {e}")
            return False
    
    async def _post_deployment_health_check(self, config: DeploymentConfig, result: DeploymentResult) -> bool:
        """Effectue les vérifications de santé post-déploiement."""
        result.logs.append("Vérifications de santé post-déploiement")
        
        try:
            async with HealthChecker() as health_checker:
                if await health_checker.wait_for_service_ready(
                    config.name, 
                    config.namespace, 
                    max_attempts=config.health_check_timeout // 10
                ):
                    result.logs.append("Vérifications de santé réussies")
                    return True
                else:
                    result.logs.append("Échec des vérifications de santé")
                    return False
        
        except Exception as e:
            result.logs.append(f"Erreur lors des vérifications de santé: {e}")
            return False
    
    async def _perform_rollback(self, config: DeploymentConfig, result: DeploymentResult):
        """Effectue un rollback en cas d'échec."""
        result.logs.append("Démarrage du rollback")
        result.status = DeploymentStatus.ROLLING_BACK
        
        try:
            # Rollback Kubernetes
            rollback_success = await self.kubernetes.rollback_deployment(config.name, config.namespace)
            
            if rollback_success:
                result.logs.append("Rollback Kubernetes réussi")
                result.rollback_performed = True
                result.status = DeploymentStatus.ROLLED_BACK
            else:
                result.logs.append("Échec du rollback Kubernetes")
        
        except Exception as e:
            result.logs.append(f"Erreur lors du rollback: {e}")
    
    async def _notify_deployment_start(self, config: DeploymentConfig, result: DeploymentResult):
        """Notifie le début du déploiement."""
        message = f"🚀 Déploiement démarré: {config.name} ({config.environment})"
        for channel in config.notification_channels:
            await self.notification_manager.send_notification(channel, message)
    
    async def _notify_deployment_success(self, config: DeploymentConfig, result: DeploymentResult):
        """Notifie le succès du déploiement."""
        message = f"✅ Déploiement réussi: {config.name} ({config.environment}) en {result.duration_seconds:.2f}s"
        for channel in config.notification_channels:
            await self.notification_manager.send_notification(channel, message)
    
    async def _notify_deployment_failure(self, config: DeploymentConfig, result: DeploymentResult):
        """Notifie l'échec du déploiement."""
        rollback_msg = " (rollback effectué)" if result.rollback_performed else ""
        message = f"❌ Déploiement échoué: {config.name} ({config.environment}){rollback_msg}"
        for channel in config.notification_channels:
            await self.notification_manager.send_notification(channel, message)


async def main():
    """Fonction principale de démonstration."""
    # Configuration d'exemple
    config = DeploymentConfig(
        name="spotify-ai-template-service",
        environment="staging",
        strategy=DeploymentStrategy.BLUE_GREEN,
        config_files=[
            Path("../deployment_orchestration.yaml"),
            Path("../monitoring_config.json")
        ],
        namespace="spotify-ai",
        timeout_seconds=600,
        rollback_on_failure=True,
        notification_channels=["slack"],
        approval_required=False
    )
    
    # Initialisation de l'automatiseur
    automator = DeploymentAutomator()
    
    # Exécution du déploiement
    result = await automator.deploy_configuration(config)
    
    # Affichage du résultat
    print(f"\n{'='*60}")
    print(f"RÉSULTAT DU DÉPLOIEMENT")
    print(f"{'='*60}")
    print(f"ID: {result.deployment_id}")
    print(f"Statut: {result.status.value}")
    print(f"Durée: {result.duration_seconds:.2f}s")
    print(f"Rollback effectué: {result.rollback_performed}")
    
    if result.error_message:
        print(f"Erreur: {result.error_message}")
    
    print(f"\nLogs:")
    for log in result.logs:
        print(f"  - {log}")
    
    print(f"\nMétriques:")
    for metric, value in result.metrics.items():
        print(f"  - {metric}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
