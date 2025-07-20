#!/usr/bin/env python3
# =============================================================================
# Monitoring Deployment & Maintenance Orchestrator - Enterprise
# =============================================================================
# 
# Script de déploiement et maintenance automatisé pour le système de monitoring
# enterprise avec validation, monitoring de santé et opérations DevOps.
#
# Développé par l'équipe d'experts techniques:
# - Lead Developer + AI Architect (Orchestration et architecture)
# - Backend Senior Developer (Python/FastAPI/Django)
# - DevOps Senior Engineer (Déploiement et infrastructure)
# - Spécialiste Sécurité Backend (Validation et audit)
# - Architecte Microservices (Coordination des services)
#
# Direction Technique: Fahed Mlaiel
# =============================================================================

import asyncio
import argparse
import sys
import os
import time
import json
import yaml
import subprocess
import shutil
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import structlog
import psutil
import docker
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Imports locaux
from config_manager import ConfigurationManager, setup_monitoring_config, Environment
from __init__ import (
    EnterpriseMonitoringOrchestrator, 
    MultiTenantMonitoringManager,
    MonitoringFactory,
    initialize_monitoring
)

logger = structlog.get_logger(__name__)

# =============================================================================
# MODÈLES DE DÉPLOIEMENT
# =============================================================================

class DeploymentMode(Enum):
    """Modes de déploiement"""
    STANDALONE = "standalone"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    SYSTEMD = "systemd"

class OperationType(Enum):
    """Types d'opérations"""
    DEPLOY = "deploy"
    START = "start"
    STOP = "stop"
    RESTART = "restart"
    UPDATE = "update"
    BACKUP = "backup"
    RESTORE = "restore"
    VALIDATE = "validate"
    CLEANUP = "cleanup"
    MIGRATE = "migrate"

class ServiceStatus(Enum):
    """Statuts des services"""
    UNKNOWN = "unknown"
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    DEGRADED = "degraded"

@dataclass
class ServiceInfo:
    """Informations d'un service"""
    name: str
    type: str
    port: int
    status: ServiceStatus = ServiceStatus.UNKNOWN
    pid: Optional[int] = None
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    uptime_seconds: int = 0
    health_url: Optional[str] = None
    config_path: Optional[str] = None
    log_path: Optional[str] = None
    data_path: Optional[str] = None

@dataclass
class DeploymentPlan:
    """Plan de déploiement"""
    mode: DeploymentMode
    environment: Environment
    services: List[ServiceInfo] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    validation_steps: List[str] = field(default_factory=list)
    rollback_enabled: bool = True
    backup_before_deploy: bool = True

# =============================================================================
# ORCHESTRATEUR DE DÉPLOIEMENT
# =============================================================================

class MonitoringDeploymentOrchestrator:
    """
    Orchestrateur de déploiement et maintenance du système de monitoring.
    """
    
    def __init__(self, environment: str = "dev", mode: str = "standalone"):
        self.environment = Environment.DEVELOPMENT if environment == "dev" else Environment.PRODUCTION
        self.deployment_mode = DeploymentMode(mode)
        
        # Configuration
        self.config_manager = setup_monitoring_config(environment)
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        self.logs_dir = self.base_dir / "logs"
        self.backups_dir = self.base_dir / "backups"
        
        # Services
        self.services: Dict[str, ServiceInfo] = {}
        self.deployment_plan: Optional[DeploymentPlan] = None
        
        # État
        self.docker_client: Optional[docker.DockerClient] = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self._ensure_directories()
        logger.info(f"MonitoringDeploymentOrchestrator initialisé ({environment}/{mode})")

    def _ensure_directories(self):
        """Création des répertoires nécessaires"""
        for directory in [self.data_dir, self.logs_dir, self.backups_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    async def create_deployment_plan(self) -> DeploymentPlan:
        """Création du plan de déploiement"""
        
        plan = DeploymentPlan(
            mode=self.deployment_mode,
            environment=self.environment
        )
        
        # Définition des services selon l'environnement
        if self.deployment_mode == DeploymentMode.DOCKER:
            plan.services = await self._create_docker_services()
        elif self.deployment_mode == DeploymentMode.STANDALONE:
            plan.services = await self._create_standalone_services()
        elif self.deployment_mode == DeploymentMode.KUBERNETES:
            plan.services = await self._create_k8s_services()
        
        # Définition des dépendances
        plan.dependencies = {
            "prometheus": [],
            "grafana": ["prometheus"],
            "alertmanager": ["prometheus"],
            "monitoring-api": ["prometheus", "grafana"],
            "redis": [],
            "postgres": []
        }
        
        # Étapes de validation
        plan.validation_steps = [
            "check_ports_available",
            "validate_configs",
            "check_dependencies",
            "verify_disk_space",
            "test_network_connectivity"
        ]
        
        self.deployment_plan = plan
        return plan

    async def _create_docker_services(self) -> List[ServiceInfo]:
        """Création des services Docker"""
        
        prometheus_config = self.config_manager.load_config("prometheus")
        grafana_config = self.config_manager.load_config("grafana")
        
        services = [
            ServiceInfo(
                name="prometheus",
                type="docker",
                port=prometheus_config["port"],
                health_url=f"http://localhost:{prometheus_config['port']}/-/healthy",
                config_path=str(self.base_dir / "configs" / "prometheus.yml"),
                data_path=str(self.data_dir / "prometheus")
            ),
            ServiceInfo(
                name="grafana",
                type="docker",
                port=grafana_config["port"],
                health_url=f"http://localhost:{grafana_config['port']}/api/health",
                config_path=str(self.base_dir / "configs" / "grafana.ini"),
                data_path=str(self.data_dir / "grafana")
            ),
            ServiceInfo(
                name="alertmanager",
                type="docker",
                port=9093,
                health_url="http://localhost:9093/-/healthy",
                config_path=str(self.base_dir / "configs" / "alertmanager.yml"),
                data_path=str(self.data_dir / "alertmanager")
            ),
            ServiceInfo(
                name="redis",
                type="docker",
                port=6379,
                data_path=str(self.data_dir / "redis")
            ),
            ServiceInfo(
                name="postgres",
                type="docker",
                port=5432,
                data_path=str(self.data_dir / "postgres")
            )
        ]
        
        return services

    async def _create_standalone_services(self) -> List[ServiceInfo]:
        """Création des services standalone"""
        
        # Configuration similaire mais pour exécution directe
        prometheus_config = self.config_manager.load_config("prometheus")
        grafana_config = self.config_manager.load_config("grafana")
        
        services = [
            ServiceInfo(
                name="monitoring-api",
                type="python",
                port=8000,
                health_url="http://localhost:8000/health",
                config_path=str(self.base_dir / "configs" / "api.yaml")
            ),
            ServiceInfo(
                name="prometheus",
                type="binary",
                port=prometheus_config["port"],
                health_url=f"http://localhost:{prometheus_config['port']}/-/healthy",
                config_path=str(self.base_dir / "configs" / "prometheus.yml")
            ),
            ServiceInfo(
                name="grafana-server",
                type="binary",
                port=grafana_config["port"],
                health_url=f"http://localhost:{grafana_config['port']}/api/health",
                config_path=str(self.base_dir / "configs" / "grafana.ini")
            )
        ]
        
        return services

    async def _create_k8s_services(self) -> List[ServiceInfo]:
        """Création des services Kubernetes"""
        
        # Configuration pour Kubernetes
        services = [
            ServiceInfo(
                name="prometheus-deployment",
                type="k8s",
                port=9090,
                config_path=str(self.base_dir / "k8s" / "prometheus.yaml")
            ),
            ServiceInfo(
                name="grafana-deployment",
                type="k8s",
                port=3000,
                config_path=str(self.base_dir / "k8s" / "grafana.yaml")
            ),
            ServiceInfo(
                name="monitoring-api-deployment",
                type="k8s",
                port=8000,
                config_path=str(self.base_dir / "k8s" / "api.yaml")
            )
        ]
        
        return services

    async def validate_environment(self) -> Dict[str, bool]:
        """Validation de l'environnement de déploiement"""
        
        results = {}
        
        # Validation des ports
        results["ports_available"] = await self._check_ports_available()
        
        # Validation des configurations
        results["configs_valid"] = await self._validate_configs()
        
        # Validation des dépendances
        results["dependencies_ok"] = await self._check_dependencies()
        
        # Validation de l'espace disque
        results["disk_space_ok"] = await self._verify_disk_space()
        
        # Validation de la connectivité réseau
        results["network_ok"] = await self._test_network_connectivity()
        
        # Validation des permissions
        results["permissions_ok"] = await self._check_permissions()
        
        return results

    async def _check_ports_available(self) -> bool:
        """Vérification de la disponibilité des ports"""
        
        if not self.deployment_plan:
            return False
        
        import socket
        
        for service in self.deployment_plan.services:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                result = sock.connect_ex(('localhost', service.port))
                if result == 0:
                    logger.warning(f"Port {service.port} déjà utilisé pour {service.name}")
                    return False
            finally:
                sock.close()
        
        return True

    async def _validate_configs(self) -> bool:
        """Validation des configurations"""
        
        try:
            validation_results = self.config_manager.validate_all_configs()
            return all(validation_results.values())
        except Exception as e:
            logger.error(f"Erreur validation configs: {e}")
            return False

    async def _check_dependencies(self) -> bool:
        """Vérification des dépendances"""
        
        # Vérification Python
        if sys.version_info < (3, 8):
            logger.error("Python 3.8+ requis")
            return False
        
        # Vérification des packages Python requis
        required_packages = [
            'fastapi', 'uvicorn', 'pydantic', 'structlog',
            'prometheus-client', 'aioredis', 'asyncpg'
        ]
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                logger.error(f"Package manquant: {package}")
                return False
        
        # Vérification Docker si nécessaire
        if self.deployment_mode == DeploymentMode.DOCKER:
            try:
                import docker
                client = docker.from_env()
                client.ping()
            except Exception as e:
                logger.error(f"Docker non disponible: {e}")
                return False
        
        return True

    async def _verify_disk_space(self, min_gb: int = 10) -> bool:
        """Vérification de l'espace disque"""
        
        disk_usage = shutil.disk_usage(self.base_dir)
        free_gb = disk_usage.free / (1024 ** 3)
        
        if free_gb < min_gb:
            logger.error(f"Espace disque insuffisant: {free_gb:.1f}GB < {min_gb}GB")
            return False
        
        return True

    async def _test_network_connectivity(self) -> bool:
        """Test de connectivité réseau"""
        
        test_urls = [
            "https://prometheus.io",
            "https://grafana.com",
            "https://github.com"
        ]
        
        for url in test_urls:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code != 200:
                    logger.warning(f"Connectivité limitée: {url}")
            except Exception as e:
                logger.warning(f"Erreur connectivité {url}: {e}")
                # Ne pas échouer pour les problèmes de connectivité externe
        
        return True

    async def _check_permissions(self) -> bool:
        """Vérification des permissions"""
        
        # Test d'écriture dans les répertoires
        test_dirs = [self.data_dir, self.logs_dir, self.backups_dir]
        
        for directory in test_dirs:
            test_file = directory / f"test_{int(time.time())}.tmp"
            try:
                test_file.write_text("test")
                test_file.unlink()
            except Exception as e:
                logger.error(f"Permissions insuffisantes pour {directory}: {e}")
                return False
        
        return True

    async def deploy(self, operation: OperationType = OperationType.DEPLOY) -> bool:
        """Déploiement du système de monitoring"""
        
        logger.info(f"Démarrage du déploiement ({operation.value})")
        
        try:
            # Création du plan de déploiement
            if not self.deployment_plan:
                await self.create_deployment_plan()
            
            # Validation de l'environnement
            validation_results = await self.validate_environment()
            if not all(validation_results.values()):
                failed_checks = [k for k, v in validation_results.items() if not v]
                raise Exception(f"Validation échouée: {', '.join(failed_checks)}")
            
            # Sauvegarde si nécessaire
            if self.deployment_plan.backup_before_deploy and operation in [OperationType.DEPLOY, OperationType.UPDATE]:
                await self.backup()
            
            # Déploiement selon le mode
            if self.deployment_mode == DeploymentMode.DOCKER:
                success = await self._deploy_docker()
            elif self.deployment_mode == DeploymentMode.STANDALONE:
                success = await self._deploy_standalone()
            elif self.deployment_mode == DeploymentMode.KUBERNETES:
                success = await self._deploy_kubernetes()
            else:
                raise Exception(f"Mode de déploiement non supporté: {self.deployment_mode}")
            
            if success:
                # Post-déploiement
                await self._post_deployment_tasks()
                logger.info("Déploiement réussi")
            else:
                logger.error("Échec du déploiement")
            
            return success
            
        except Exception as e:
            logger.error(f"Erreur déploiement: {e}")
            
            # Rollback si activé
            if self.deployment_plan and self.deployment_plan.rollback_enabled:
                await self._rollback()
            
            return False

    async def _deploy_docker(self) -> bool:
        """Déploiement avec Docker"""
        
        try:
            # Initialisation du client Docker
            self.docker_client = docker.from_env()
            
            # Génération des configurations Docker
            await self._generate_docker_configs()
            
            # Démarrage des services
            for service in self.deployment_plan.services:
                success = await self._start_docker_service(service)
                if not success:
                    return False
            
            # Vérification de santé
            await asyncio.sleep(10)  # Attente du démarrage
            return await self._check_services_health()
            
        except Exception as e:
            logger.error(f"Erreur déploiement Docker: {e}")
            return False

    async def _deploy_standalone(self) -> bool:
        """Déploiement standalone"""
        
        try:
            # Génération des configurations
            await self._generate_standalone_configs()
            
            # Démarrage des services
            for service in self.deployment_plan.services:
                success = await self._start_standalone_service(service)
                if not success:
                    return False
            
            # Vérification de santé
            await asyncio.sleep(5)
            return await self._check_services_health()
            
        except Exception as e:
            logger.error(f"Erreur déploiement standalone: {e}")
            return False

    async def _deploy_kubernetes(self) -> bool:
        """Déploiement sur Kubernetes"""
        
        try:
            # Génération des manifests K8s
            await self._generate_k8s_manifests()
            
            # Application des manifests
            for service in self.deployment_plan.services:
                success = await self._apply_k8s_manifest(service)
                if not success:
                    return False
            
            # Attente du déploiement
            await asyncio.sleep(30)
            return await self._check_k8s_deployments()
            
        except Exception as e:
            logger.error(f"Erreur déploiement K8s: {e}")
            return False

    async def _generate_docker_configs(self):
        """Génération des configurations Docker"""
        
        # Docker Compose pour l'orchestration
        compose_config = {
            'version': '3.8',
            'services': {},
            'volumes': {},
            'networks': {
                'monitoring': {
                    'driver': 'bridge'
                }
            }
        }
        
        # Configuration Prometheus
        prometheus_config = self.config_manager.load_config("prometheus")
        compose_config['services']['prometheus'] = {
            'image': 'prom/prometheus:latest',
            'ports': [f"{prometheus_config['port']}:9090"],
            'volumes': [
                f"{self.base_dir}/configs/prometheus.yml:/etc/prometheus/prometheus.yml",
                f"{self.data_dir}/prometheus:/prometheus"
            ],
            'command': [
                '--config.file=/etc/prometheus/prometheus.yml',
                '--storage.tsdb.path=/prometheus',
                '--web.console.libraries=/etc/prometheus/console_libraries',
                '--web.console.templates=/etc/prometheus/consoles',
                '--storage.tsdb.retention.time=30d',
                '--web.enable-lifecycle'
            ],
            'networks': ['monitoring']
        }
        
        # Configuration Grafana
        grafana_config = self.config_manager.load_config("grafana")
        compose_config['services']['grafana'] = {
            'image': 'grafana/grafana:latest',
            'ports': [f"{grafana_config['port']}:3000"],
            'volumes': [
                f"{self.data_dir}/grafana:/var/lib/grafana"
            ],
            'environment': {
                'GF_SECURITY_ADMIN_USER': grafana_config['admin_user'],
                'GF_SECURITY_ADMIN_PASSWORD': grafana_config['admin_password']
            },
            'networks': ['monitoring']
        }
        
        # Sauvegarde du docker-compose.yml
        compose_file = self.base_dir / "docker-compose.yml"
        with open(compose_file, 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False)
        
        logger.info("Configurations Docker générées")

    async def _start_docker_service(self, service: ServiceInfo) -> bool:
        """Démarrage d'un service Docker"""
        
        try:
            # Utilisation de docker-compose
            cmd = [
                "docker-compose", "-f", str(self.base_dir / "docker-compose.yml"),
                "up", "-d", service.name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Service Docker {service.name} démarré")
                return True
            else:
                logger.error(f"Erreur démarrage {service.name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Erreur démarrage service Docker {service.name}: {e}")
            return False

    async def _generate_standalone_configs(self):
        """Génération des configurations standalone"""
        
        configs_dir = self.base_dir / "configs"
        configs_dir.mkdir(exist_ok=True)
        
        # Configuration Prometheus
        prometheus_config = self.config_manager.load_config("prometheus")
        prometheus_yml = {
            'global': {
                'scrape_interval': prometheus_config['scrape_interval'],
                'evaluation_interval': prometheus_config['evaluation_interval']
            },
            'scrape_configs': [
                {
                    'job_name': 'monitoring-api',
                    'static_configs': [{'targets': ['localhost:8000']}]
                },
                {
                    'job_name': 'prometheus',
                    'static_configs': [{'targets': ['localhost:9090']}]
                }
            ]
        }
        
        with open(configs_dir / "prometheus.yml", 'w') as f:
            yaml.dump(prometheus_yml, f, default_flow_style=False)
        
        # Configuration API de monitoring
        api_config = {
            'server': {
                'host': '0.0.0.0',
                'port': 8000,
                'reload': self.environment == Environment.DEVELOPMENT
            },
            'monitoring': self.config_manager.export_configuration(include_secrets=False)
        }
        
        with open(configs_dir / "api.yaml", 'w') as f:
            yaml.dump(api_config, f, default_flow_style=False)
        
        logger.info("Configurations standalone générées")

    async def _start_standalone_service(self, service: ServiceInfo) -> bool:
        """Démarrage d'un service standalone"""
        
        try:
            if service.name == "monitoring-api":
                # Démarrage de l'API avec uvicorn
                cmd = [
                    sys.executable, "-m", "uvicorn",
                    "monitoring_api:app",
                    "--host", "0.0.0.0",
                    "--port", str(service.port),
                    "--log-level", "info"
                ]
                
                if self.environment == Environment.DEVELOPMENT:
                    cmd.append("--reload")
                
            elif service.name == "prometheus":
                # Démarrage de Prometheus
                cmd = [
                    "prometheus",
                    f"--config.file={service.config_path}",
                    f"--storage.tsdb.path={self.data_dir}/prometheus",
                    f"--web.listen-address=:{service.port}",
                    "--storage.tsdb.retention.time=30d"
                ]
                
            else:
                logger.error(f"Service standalone non supporté: {service.name}")
                return False
            
            # Démarrage en arrière-plan
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.base_dir
            )
            
            service.pid = process.pid
            logger.info(f"Service {service.name} démarré (PID: {service.pid})")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur démarrage service {service.name}: {e}")
            return False

    async def _check_services_health(self) -> bool:
        """Vérification de santé des services"""
        
        if not self.deployment_plan:
            return False
        
        healthy_services = 0
        total_services = len(self.deployment_plan.services)
        
        for service in self.deployment_plan.services:
            if service.health_url:
                try:
                    response = requests.get(service.health_url, timeout=10)
                    if response.status_code == 200:
                        service.status = ServiceStatus.RUNNING
                        healthy_services += 1
                        logger.info(f"Service {service.name}: HEALTHY")
                    else:
                        service.status = ServiceStatus.DEGRADED
                        logger.warning(f"Service {service.name}: DEGRADED ({response.status_code})")
                except Exception as e:
                    service.status = ServiceStatus.ERROR
                    logger.error(f"Service {service.name}: ERROR - {e}")
            else:
                # Vérification par port pour les services sans health check
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    result = sock.connect_ex(('localhost', service.port))
                    if result == 0:
                        service.status = ServiceStatus.RUNNING
                        healthy_services += 1
                        logger.info(f"Service {service.name}: RUNNING")
                    else:
                        service.status = ServiceStatus.ERROR
                        logger.error(f"Service {service.name}: PORT CLOSED")
                finally:
                    sock.close()
        
        success_rate = healthy_services / total_services
        logger.info(f"Santé des services: {healthy_services}/{total_services} ({success_rate:.1%})")
        
        return success_rate >= 0.8  # 80% des services doivent être sains

    async def _post_deployment_tasks(self):
        """Tâches post-déploiement"""
        
        # Initialisation du monitoring
        await self._initialize_monitoring_system()
        
        # Configuration des dashboards par défaut
        await self._setup_default_dashboards()
        
        # Configuration des alertes par défaut
        await self._setup_default_alerts()
        
        # Validation finale
        await self._final_validation()

    async def _initialize_monitoring_system(self):
        """Initialisation du système de monitoring"""
        
        try:
            # Création d'une configuration de monitoring
            monitoring_config = {
                'prometheus': self.config_manager.load_config("prometheus"),
                'grafana': self.config_manager.load_config("grafana"),
                'alerting': self.config_manager.load_config("alerting")
            }
            
            # Initialisation avec la factory
            orchestrator = await initialize_monitoring(monitoring_config)
            
            logger.info("Système de monitoring initialisé")
            
        except Exception as e:
            logger.error(f"Erreur initialisation monitoring: {e}")

    async def backup(self) -> str:
        """Sauvegarde du système"""
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_name = f"monitoring_backup_{timestamp}"
        backup_path = self.backups_dir / backup_name
        
        try:
            backup_path.mkdir(exist_ok=True)
            
            # Sauvegarde des configurations
            configs_backup = backup_path / "configs"
            shutil.copytree(self.base_dir / "configs", configs_backup, dirs_exist_ok=True)
            
            # Sauvegarde des données
            if self.data_dir.exists():
                data_backup = backup_path / "data"
                shutil.copytree(self.data_dir, data_backup, dirs_exist_ok=True)
            
            # Métadonnées de sauvegarde
            metadata = {
                'timestamp': timestamp,
                'environment': self.environment.value,
                'deployment_mode': self.deployment_mode.value,
                'services': [asdict(service) for service in (self.deployment_plan.services if self.deployment_plan else [])]
            }
            
            with open(backup_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Sauvegarde créée: {backup_name}")
            return backup_name
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde: {e}")
            return ""

    async def get_system_status(self) -> Dict[str, Any]:
        """Récupération du statut du système"""
        
        status = {
            'timestamp': datetime.utcnow().isoformat(),
            'environment': self.environment.value,
            'deployment_mode': self.deployment_mode.value,
            'services': {},
            'system_resources': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
        }
        
        # Statut des services
        if self.deployment_plan:
            for service in self.deployment_plan.services:
                status['services'][service.name] = {
                    'status': service.status.value,
                    'port': service.port,
                    'pid': service.pid,
                    'memory_mb': service.memory_mb,
                    'cpu_percent': service.cpu_percent,
                    'uptime_seconds': service.uptime_seconds
                }
        
        return status

    async def _rollback(self):
        """Rollback en cas d'échec"""
        
        logger.warning("Démarrage du rollback...")
        
        # Arrêt des services déployés
        await self.stop_all_services()
        
        # Restauration depuis la dernière sauvegarde
        backups = sorted(self.backups_dir.glob("monitoring_backup_*"), reverse=True)
        if backups:
            latest_backup = backups[0]
            await self.restore(latest_backup.name)
            logger.info(f"Rollback effectué depuis {latest_backup.name}")
        else:
            logger.error("Aucune sauvegarde disponible pour le rollback")

    async def stop_all_services(self):
        """Arrêt de tous les services"""
        
        if self.deployment_mode == DeploymentMode.DOCKER:
            cmd = ["docker-compose", "-f", str(self.base_dir / "docker-compose.yml"), "down"]
            subprocess.run(cmd)
        
        elif self.deployment_mode == DeploymentMode.STANDALONE:
            if self.deployment_plan:
                for service in self.deployment_plan.services:
                    if service.pid:
                        try:
                            os.kill(service.pid, signal.SIGTERM)
                            logger.info(f"Service {service.name} arrêté")
                        except Exception as e:
                            logger.error(f"Erreur arrêt {service.name}: {e}")

    async def restore(self, backup_name: str) -> bool:
        """Restauration depuis une sauvegarde"""
        
        backup_path = self.backups_dir / backup_name
        
        if not backup_path.exists():
            logger.error(f"Sauvegarde non trouvée: {backup_name}")
            return False
        
        try:
            # Arrêt des services
            await self.stop_all_services()
            
            # Restauration des configurations
            configs_backup = backup_path / "configs"
            if configs_backup.exists():
                if (self.base_dir / "configs").exists():
                    shutil.rmtree(self.base_dir / "configs")
                shutil.copytree(configs_backup, self.base_dir / "configs")
            
            # Restauration des données
            data_backup = backup_path / "data"
            if data_backup.exists():
                if self.data_dir.exists():
                    shutil.rmtree(self.data_dir)
                shutil.copytree(data_backup, self.data_dir)
            
            logger.info(f"Restauration réussie depuis {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur restauration: {e}")
            return False

# =============================================================================
# CLI ET POINT D'ENTRÉE
# =============================================================================

def create_cli_parser() -> argparse.ArgumentParser:
    """Création du parser CLI"""
    
    parser = argparse.ArgumentParser(
        description="Orchestrateur de déploiement et maintenance du monitoring enterprise",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  %(prog)s deploy --environment prod --mode docker
  %(prog)s start --environment dev
  %(prog)s backup
  %(prog)s status
  %(prog)s validate --environment staging
        """
    )
    
    parser.add_argument(
        'operation',
        choices=[op.value for op in OperationType],
        help='Opération à effectuer'
    )
    
    parser.add_argument(
        '--environment', '-e',
        choices=['dev', 'staging', 'prod', 'test'],
        default='dev',
        help='Environnement cible'
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=[mode.value for mode in DeploymentMode],
        default='standalone',
        help='Mode de déploiement'
    )
    
    parser.add_argument(
        '--backup-name',
        help='Nom de la sauvegarde pour restauration'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Forcer l\'opération sans confirmation'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mode verbeux'
    )
    
    return parser

async def main():
    """Point d'entrée principal"""
    
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Configuration du logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Création de l'orchestrateur
    orchestrator = MonitoringDeploymentOrchestrator(
        environment=args.environment,
        mode=args.mode
    )
    
    operation = OperationType(args.operation)
    
    try:
        if operation == OperationType.DEPLOY:
            success = await orchestrator.deploy(OperationType.DEPLOY)
            sys.exit(0 if success else 1)
            
        elif operation == OperationType.START:
            success = await orchestrator.deploy(OperationType.START)
            sys.exit(0 if success else 1)
            
        elif operation == OperationType.STOP:
            await orchestrator.stop_all_services()
            
        elif operation == OperationType.BACKUP:
            backup_name = await orchestrator.backup()
            if backup_name:
                print(f"Sauvegarde créée: {backup_name}")
            else:
                sys.exit(1)
                
        elif operation == OperationType.RESTORE:
            if not args.backup_name:
                print("--backup-name requis pour la restauration")
                sys.exit(1)
            success = await orchestrator.restore(args.backup_name)
            sys.exit(0 if success else 1)
            
        elif operation == OperationType.VALIDATE:
            validation_results = await orchestrator.validate_environment()
            
            print("Résultats de validation:")
            for check, result in validation_results.items():
                status = "✓" if result else "✗"
                print(f"  {status} {check}")
            
            if not all(validation_results.values()):
                sys.exit(1)
                
        elif operation == OperationType.UPDATE:
            success = await orchestrator.deploy(OperationType.UPDATE)
            sys.exit(0 if success else 1)
            
        else:
            # Status par défaut
            status = await orchestrator.get_system_status()
            print(json.dumps(status, indent=2))
    
    except KeyboardInterrupt:
        logger.info("Opération interrompue")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Erreur: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
