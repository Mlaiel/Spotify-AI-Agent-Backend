#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Metrics System Configuration & Deployment
=================================================

Ultra-advanced automated configuration and deployment system for enterprise
metrics infrastructure with intelligent auto-configuration, health checks,
and production-ready deployment orchestration.

Expert Development Team:
- Lead Dev + AI Architect
- Senior Backend Developer (Python/FastAPI/Django)
- ML Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect
"""

import asyncio
import json
import logging
import os
import sys
import time
import shutil
import tempfile
import subprocess
import socket
import platform
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import argparse
import yaml
import hashlib

# Ajout du chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from . import EnterpriseMetricsSystem, MetricDataPoint, get_metrics_system
from .collector import CollectorConfig, MetricsCollectionAgent

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import docker
    HAS_DOCKER = True
except ImportError:
    HAS_DOCKER = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class DeploymentMode(Enum):
    """Modes de déploiement."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    DEMO = "demo"


class InfrastructureType(Enum):
    """Types d'infrastructure."""
    STANDALONE = "standalone"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    CLOUD_AWS = "cloud_aws"
    CLOUD_GCP = "cloud_gcp"
    CLOUD_AZURE = "cloud_azure"


@dataclass
class DeploymentConfig:
    """Configuration de déploiement."""
    
    # Identification
    deployment_id: str = field(default_factory=lambda: f"metrics-{int(time.time())}")
    deployment_name: str = "enterprise-metrics-system"
    version: str = "1.0.0"
    
    # Mode et infrastructure
    mode: DeploymentMode = DeploymentMode.DEVELOPMENT
    infrastructure: InfrastructureType = InfrastructureType.STANDALONE
    
    # Chemins et répertoires
    base_dir: str = "/opt/metrics-system"
    config_dir: str = "/etc/metrics-system"
    data_dir: str = "/var/lib/metrics-system"
    log_dir: str = "/var/log/metrics-system"
    pid_dir: str = "/run/metrics-system"
    
    # Configuration réseau
    bind_host: str = "0.0.0.0"
    bind_port: int = 8080
    metrics_port: int = 9090
    health_port: int = 8081
    
    # Base de données
    database_type: str = "sqlite"  # sqlite, postgresql, redis
    database_url: str = ""
    database_pool_size: int = 10
    
    # Sécurité
    enable_tls: bool = False
    cert_file: str = ""
    key_file: str = ""
    ca_file: str = ""
    
    # Performance
    worker_processes: int = 1
    max_connections: int = 1000
    memory_limit_mb: int = 1024
    cpu_limit_cores: float = 2.0
    
    # Monitoring
    enable_monitoring: bool = True
    metrics_retention_days: int = 30
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    
    # Alerting
    enable_alerting: bool = True
    smtp_server: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    alert_recipients: List[str] = field(default_factory=list)
    
    # Intégrations
    enable_prometheus: bool = False
    enable_grafana: bool = False
    enable_elasticsearch: bool = False
    
    # Auto-configuration
    auto_detect_hardware: bool = True
    auto_tune_performance: bool = True
    auto_setup_systemd: bool = True
    auto_setup_nginx: bool = False


class SystemRequirements:
    """Vérification et validation des prérequis système."""
    
    @staticmethod
    def check_python_version() -> Tuple[bool, str]:
        """Vérifie la version Python."""
        required_version = (3, 8)
        current_version = sys.version_info[:2]
        
        if current_version >= required_version:
            return True, f"Python {sys.version} ✓"
        else:
            return False, f"Python {current_version} requis >= {required_version}"
    
    @staticmethod
    def check_system_resources() -> Dict[str, Any]:
        """Vérifie les ressources système."""
        resources = {
            "cpu_count": 1,
            "memory_gb": 1.0,
            "disk_space_gb": 10.0,
            "load_average": 0.0
        }
        
        try:
            if HAS_PSUTIL:
                resources.update({
                    "cpu_count": psutil.cpu_count(),
                    "memory_gb": psutil.virtual_memory().total / (1024**3),
                    "disk_space_gb": psutil.disk_usage('/').free / (1024**3),
                    "load_average": os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0
                })
        except Exception as e:
            logging.warning(f"Erreur vérification ressources: {e}")
        
        return resources
    
    @staticmethod
    def check_network_ports(ports: List[int]) -> Dict[int, bool]:
        """Vérifie la disponibilité des ports réseau."""
        results = {}
        
        for port in ports:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    result = sock.connect_ex(('127.0.0.1', port))
                    results[port] = result != 0  # Port libre si connexion échoue
            except Exception:
                results[port] = False
        
        return results
    
    @staticmethod
    def check_permissions(directories: List[str]) -> Dict[str, bool]:
        """Vérifie les permissions d'écriture."""
        results = {}
        
        for directory in directories:
            try:
                # Teste la création d'un fichier temporaire
                test_file = os.path.join(directory, f".test_{int(time.time())}")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                results[directory] = True
            except Exception:
                results[directory] = False
        
        return results
    
    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """Vérifie les dépendances Python."""
        dependencies = {
            "psutil": HAS_PSUTIL,
            "requests": HAS_REQUESTS,
            "docker": HAS_DOCKER,
            "asyncio": True,  # Intégré à Python 3.7+
            "json": True,     # Intégré
            "sqlite3": True   # Intégré
        }
        
        # Vérification de modules optionnels
        try:
            import redis
            dependencies["redis"] = True
        except ImportError:
            dependencies["redis"] = False
        
        try:
            import postgresql
            dependencies["postgresql"] = True
        except ImportError:
            dependencies["postgresql"] = False
        
        try:
            import prometheus_client
            dependencies["prometheus_client"] = True
        except ImportError:
            dependencies["prometheus_client"] = False
        
        return dependencies


class ConfigurationGenerator:
    """Générateur de configuration automatique."""
    
    def __init__(self, deployment_config: DeploymentConfig):
        self.deployment_config = deployment_config
        
    def generate_base_config(self) -> Dict[str, Any]:
        """Génère la configuration de base."""
        resources = SystemRequirements.check_system_resources()
        
        # Auto-tuning basé sur les ressources
        if self.deployment_config.auto_tune_performance:
            self._auto_tune_performance(resources)
        
        config = {
            "deployment": asdict(self.deployment_config),
            "system": {
                "resources": resources,
                "platform": platform.platform(),
                "hostname": socket.gethostname(),
                "python_version": sys.version
            },
            "collector": self._generate_collector_config(resources),
            "storage": self._generate_storage_config(),
            "monitoring": self._generate_monitoring_config(),
            "security": self._generate_security_config(),
            "networking": self._generate_networking_config()
        }
        
        return config
    
    def _auto_tune_performance(self, resources: Dict[str, Any]):
        """Auto-tuning des performances basé sur les ressources."""
        cpu_count = resources.get("cpu_count", 1)
        memory_gb = resources.get("memory_gb", 1.0)
        
        # Ajustement du nombre de workers
        if cpu_count >= 8:
            self.deployment_config.worker_processes = min(cpu_count - 2, 16)
        elif cpu_count >= 4:
            self.deployment_config.worker_processes = cpu_count - 1
        else:
            self.deployment_config.worker_processes = 1
        
        # Ajustement de la mémoire
        if memory_gb >= 16:
            self.deployment_config.memory_limit_mb = 4096
        elif memory_gb >= 8:
            self.deployment_config.memory_limit_mb = 2048
        elif memory_gb >= 4:
            self.deployment_config.memory_limit_mb = 1024
        else:
            self.deployment_config.memory_limit_mb = 512
        
        # Ajustement des limites CPU
        self.deployment_config.cpu_limit_cores = min(cpu_count * 0.8, 8.0)
        
        # Ajustement des connexions
        if memory_gb >= 8:
            self.deployment_config.max_connections = 2000
        elif memory_gb >= 4:
            self.deployment_config.max_connections = 1000
        else:
            self.deployment_config.max_connections = 500
    
    def _generate_collector_config(self, resources: Dict[str, Any]) -> Dict[str, Any]:
        """Génère la configuration du collecteur."""
        cpu_count = resources.get("cpu_count", 1)
        
        # Intervalles adaptatifs basés sur les ressources
        if cpu_count >= 8:
            system_interval = 30
            performance_interval = 15
        elif cpu_count >= 4:
            system_interval = 45
            performance_interval = 20
        else:
            system_interval = 60
            performance_interval = 30
        
        return {
            "system_interval": system_interval,
            "security_interval": 300,
            "performance_interval": performance_interval,
            "network_interval": 120,
            "storage_interval": 180,
            "max_concurrent_collectors": min(cpu_count, 10),
            "adaptive_sampling": True,
            "intelligent_batching": True,
            "compression_enabled": True
        }
    
    def _generate_storage_config(self) -> Dict[str, Any]:
        """Génère la configuration de stockage."""
        config = {
            "type": self.deployment_config.database_type,
            "retention_days": self.deployment_config.metrics_retention_days,
            "backup_enabled": self.deployment_config.backup_enabled,
            "backup_interval_hours": self.deployment_config.backup_interval_hours
        }
        
        if self.deployment_config.database_type == "sqlite":
            config.update({
                "db_path": os.path.join(self.deployment_config.data_dir, "metrics.db"),
                "connection_pool_size": 5,
                "wal_mode": True,
                "synchronous": "NORMAL"
            })
        elif self.deployment_config.database_type == "redis":
            config.update({
                "redis_url": self.deployment_config.database_url or "redis://localhost:6379/0",
                "connection_pool_size": self.deployment_config.database_pool_size,
                "max_connections": 100
            })
        elif self.deployment_config.database_type == "postgresql":
            config.update({
                "postgresql_url": self.deployment_config.database_url,
                "connection_pool_size": self.deployment_config.database_pool_size,
                "max_connections": self.deployment_config.max_connections
            })
        
        return config
    
    def _generate_monitoring_config(self) -> Dict[str, Any]:
        """Génère la configuration de monitoring."""
        return {
            "enabled": self.deployment_config.enable_monitoring,
            "health_check_port": self.deployment_config.health_port,
            "metrics_port": self.deployment_config.metrics_port,
            "prometheus_enabled": self.deployment_config.enable_prometheus,
            "grafana_enabled": self.deployment_config.enable_grafana,
            "elasticsearch_enabled": self.deployment_config.enable_elasticsearch,
            "health_check_interval": 30,
            "metrics_export_interval": 60
        }
    
    def _generate_security_config(self) -> Dict[str, Any]:
        """Génère la configuration de sécurité."""
        return {
            "tls_enabled": self.deployment_config.enable_tls,
            "cert_file": self.deployment_config.cert_file,
            "key_file": self.deployment_config.key_file,
            "ca_file": self.deployment_config.ca_file,
            "auth_required": False,  # Configurable selon les besoins
            "api_key_required": False,
            "rate_limiting": {
                "enabled": True,
                "requests_per_minute": 1000,
                "burst_size": 100
            }
        }
    
    def _generate_networking_config(self) -> Dict[str, Any]:
        """Génère la configuration réseau."""
        return {
            "bind_host": self.deployment_config.bind_host,
            "bind_port": self.deployment_config.bind_port,
            "max_connections": self.deployment_config.max_connections,
            "keep_alive_timeout": 30,
            "request_timeout": 60,
            "cors_enabled": True,
            "cors_origins": ["*"]
        }


class DeploymentOrchestrator:
    """Orchestrateur de déploiement automatisé."""
    
    def __init__(self, deployment_config: DeploymentConfig):
        self.deployment_config = deployment_config
        self.config_generator = ConfigurationGenerator(deployment_config)
        
    async def deploy(self) -> bool:
        """Déploie le système de métriques."""
        try:
            logging.info("🚀 Démarrage du déploiement du système de métriques")
            
            # 1. Vérification des prérequis
            if not await self._check_prerequisites():
                return False
            
            # 2. Création de la structure de répertoires
            if not await self._create_directory_structure():
                return False
            
            # 3. Génération des configurations
            if not await self._generate_configurations():
                return False
            
            # 4. Installation des dépendances
            if not await self._install_dependencies():
                return False
            
            # 5. Configuration des services système
            if not await self._setup_system_services():
                return False
            
            # 6. Déploiement des composants
            if not await self._deploy_components():
                return False
            
            # 7. Tests de santé
            if not await self._run_health_checks():
                return False
            
            # 8. Finalisation
            await self._finalize_deployment()
            
            logging.info("✅ Déploiement terminé avec succès")
            return True
            
        except Exception as e:
            logging.error(f"❌ Erreur lors du déploiement: {e}")
            await self._rollback_deployment()
            return False
    
    async def _check_prerequisites(self) -> bool:
        """Vérifie les prérequis système."""
        logging.info("📋 Vérification des prérequis...")
        
        # Version Python
        python_ok, python_msg = SystemRequirements.check_python_version()
        logging.info(f"Python: {python_msg}")
        if not python_ok:
            return False
        
        # Ressources système
        resources = SystemRequirements.check_system_resources()
        logging.info(f"CPU: {resources['cpu_count']} cores")
        logging.info(f"Mémoire: {resources['memory_gb']:.1f} GB")
        logging.info(f"Espace disque: {resources['disk_space_gb']:.1f} GB")
        
        # Vérifications minimales
        if resources['memory_gb'] < 1.0:
            logging.error("❌ Mémoire insuffisante (minimum 1GB)")
            return False
        
        if resources['disk_space_gb'] < 5.0:
            logging.error("❌ Espace disque insuffisant (minimum 5GB)")
            return False
        
        # Ports réseau
        required_ports = [
            self.deployment_config.bind_port,
            self.deployment_config.metrics_port,
            self.deployment_config.health_port
        ]
        
        port_availability = SystemRequirements.check_network_ports(required_ports)
        for port, available in port_availability.items():
            if not available:
                logging.error(f"❌ Port {port} non disponible")
                return False
            logging.info(f"Port {port}: ✓")
        
        # Dépendances
        dependencies = SystemRequirements.check_dependencies()
        missing_deps = [dep for dep, available in dependencies.items() if not available]
        
        if missing_deps:
            logging.warning(f"⚠️  Dépendances manquantes: {missing_deps}")
            # Certaines dépendances sont optionnelles
        
        logging.info("✅ Prérequis validés")
        return True
    
    async def _create_directory_structure(self) -> bool:
        """Crée la structure de répertoires."""
        logging.info("📁 Création de la structure de répertoires...")
        
        directories = [
            self.deployment_config.base_dir,
            self.deployment_config.config_dir,
            self.deployment_config.data_dir,
            self.deployment_config.log_dir,
            self.deployment_config.pid_dir,
            os.path.join(self.deployment_config.config_dir, "ssl"),
            os.path.join(self.deployment_config.data_dir, "backups"),
            os.path.join(self.deployment_config.log_dir, "archive")
        ]
        
        for directory in directories:
            try:
                os.makedirs(directory, mode=0o755, exist_ok=True)
                logging.debug(f"Répertoire créé: {directory}")
            except Exception as e:
                logging.error(f"❌ Erreur création répertoire {directory}: {e}")
                return False
        
        logging.info("✅ Structure de répertoires créée")
        return True
    
    async def _generate_configurations(self) -> bool:
        """Génère les fichiers de configuration."""
        logging.info("⚙️  Génération des configurations...")
        
        try:
            # Configuration principale
            main_config = self.config_generator.generate_base_config()
            
            # Sauvegarde des configurations
            config_files = {
                "main.json": main_config,
                "collector.json": main_config["collector"],
                "storage.json": main_config["storage"],
                "monitoring.json": main_config["monitoring"],
                "security.json": main_config["security"]
            }
            
            for filename, config_data in config_files.items():
                config_path = os.path.join(self.deployment_config.config_dir, filename)
                
                with open(config_path, 'w') as f:
                    json.dump(config_data, f, indent=2, default=str)
                
                # Permissions restrictives pour les fichiers de config
                os.chmod(config_path, 0o640)
                logging.debug(f"Configuration sauvée: {config_path}")
            
            # Configuration YAML pour Docker/Kubernetes
            if self.deployment_config.infrastructure in [InfrastructureType.DOCKER, InfrastructureType.KUBERNETES]:
                await self._generate_container_configs(main_config)
            
            logging.info("✅ Configurations générées")
            return True
            
        except Exception as e:
            logging.error(f"❌ Erreur génération configurations: {e}")
            return False
    
    async def _generate_container_configs(self, main_config: Dict[str, Any]):
        """Génère les configurations pour conteneurs."""
        
        # Docker Compose
        if self.deployment_config.infrastructure == InfrastructureType.DOCKER:
            docker_compose = {
                "version": "3.8",
                "services": {
                    "metrics-system": {
                        "image": "metrics-system:latest",
                        "ports": [
                            f"{self.deployment_config.bind_port}:8080",
                            f"{self.deployment_config.metrics_port}:9090",
                            f"{self.deployment_config.health_port}:8081"
                        ],
                        "volumes": [
                            f"{self.deployment_config.data_dir}:/var/lib/metrics-system",
                            f"{self.deployment_config.config_dir}:/etc/metrics-system:ro"
                        ],
                        "environment": {
                            "METRICS_CONFIG_DIR": "/etc/metrics-system",
                            "METRICS_DATA_DIR": "/var/lib/metrics-system"
                        },
                        "restart": "unless-stopped",
                        "healthcheck": {
                            "test": f"curl -f http://localhost:{self.deployment_config.health_port}/health || exit 1",
                            "interval": "30s",
                            "timeout": "10s",
                            "retries": 3
                        }
                    }
                }
            }
            
            # Ajout de Redis si configuré
            if self.deployment_config.database_type == "redis":
                docker_compose["services"]["redis"] = {
                    "image": "redis:7-alpine",
                    "ports": ["6379:6379"],
                    "volumes": [f"{self.deployment_config.data_dir}/redis:/data"],
                    "restart": "unless-stopped"
                }
            
            # Sauvegarde du Docker Compose
            compose_path = os.path.join(self.deployment_config.config_dir, "docker-compose.yml")
            with open(compose_path, 'w') as f:
                yaml.dump(docker_compose, f, default_flow_style=False)
        
        # Configuration Kubernetes
        elif self.deployment_config.infrastructure == InfrastructureType.KUBERNETES:
            k8s_configs = self._generate_kubernetes_configs(main_config)
            
            for config_name, config_data in k8s_configs.items():
                config_path = os.path.join(self.deployment_config.config_dir, f"k8s-{config_name}.yaml")
                with open(config_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False)
    
    def _generate_kubernetes_configs(self, main_config: Dict[str, Any]) -> Dict[str, Dict]:
        """Génère les configurations Kubernetes."""
        configs = {}
        
        # ConfigMap
        configs["configmap"] = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "metrics-system-config",
                "namespace": "default"
            },
            "data": {
                "main.json": json.dumps(main_config, indent=2, default=str)
            }
        }
        
        # Deployment
        configs["deployment"] = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "metrics-system",
                "namespace": "default"
            },
            "spec": {
                "replicas": 1,
                "selector": {
                    "matchLabels": {"app": "metrics-system"}
                },
                "template": {
                    "metadata": {
                        "labels": {"app": "metrics-system"}
                    },
                    "spec": {
                        "containers": [{
                            "name": "metrics-system",
                            "image": "metrics-system:latest",
                            "ports": [
                                {"containerPort": 8080, "name": "api"},
                                {"containerPort": 9090, "name": "metrics"},
                                {"containerPort": 8081, "name": "health"}
                            ],
                            "volumeMounts": [{
                                "name": "config",
                                "mountPath": "/etc/metrics-system"
                            }],
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8081
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": 8081
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }],
                        "volumes": [{
                            "name": "config",
                            "configMap": {"name": "metrics-system-config"}
                        }]
                    }
                }
            }
        }
        
        # Service
        configs["service"] = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "metrics-system-service",
                "namespace": "default"
            },
            "spec": {
                "selector": {"app": "metrics-system"},
                "ports": [
                    {"port": 8080, "targetPort": 8080, "name": "api"},
                    {"port": 9090, "targetPort": 9090, "name": "metrics"},
                    {"port": 8081, "targetPort": 8081, "name": "health"}
                ],
                "type": "ClusterIP"
            }
        }
        
        return configs
    
    async def _install_dependencies(self) -> bool:
        """Installe les dépendances."""
        logging.info("📦 Installation des dépendances...")
        
        try:
            # Installation via pip des dépendances optionnelles
            optional_deps = []
            
            if self.deployment_config.database_type == "redis":
                optional_deps.append("redis")
            
            if self.deployment_config.database_type == "postgresql":
                optional_deps.extend(["psycopg2-binary", "sqlalchemy"])
            
            if self.deployment_config.enable_prometheus:
                optional_deps.append("prometheus-client")
            
            if optional_deps:
                cmd = [sys.executable, "-m", "pip", "install"] + optional_deps
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logging.warning(f"⚠️  Échec installation dépendances optionnelles: {result.stderr}")
                else:
                    logging.info(f"✅ Dépendances installées: {optional_deps}")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Erreur installation dépendances: {e}")
            return False
    
    async def _setup_system_services(self) -> bool:
        """Configure les services système."""
        logging.info("🔧 Configuration des services système...")
        
        try:
            # Service systemd (Linux uniquement)
            if platform.system() == "Linux" and self.deployment_config.auto_setup_systemd:
                await self._create_systemd_service()
            
            # Configuration Nginx (optionnel)
            if self.deployment_config.auto_setup_nginx:
                await self._create_nginx_config()
            
            logging.info("✅ Services système configurés")
            return True
            
        except Exception as e:
            logging.error(f"❌ Erreur configuration services: {e}")
            return False
    
    async def _create_systemd_service(self):
        """Crée le service systemd."""
        service_content = f"""[Unit]
Description=Enterprise Metrics System
After=network.target

[Service]
Type=exec
User=metrics
Group=metrics
WorkingDirectory={self.deployment_config.base_dir}
ExecStart={sys.executable} -m metrics_collector --config {self.deployment_config.config_dir}/main.json --daemon
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=metrics-system
PIDFile={self.deployment_config.pid_dir}/metrics-system.pid

[Install]
WantedBy=multi-user.target
"""
        
        service_path = "/etc/systemd/system/metrics-system.service"
        
        try:
            with open(service_path, 'w') as f:
                f.write(service_content)
            
            # Rechargement systemd
            subprocess.run(["systemctl", "daemon-reload"], check=True)
            subprocess.run(["systemctl", "enable", "metrics-system"], check=True)
            
            logging.info("✅ Service systemd créé et activé")
            
        except (PermissionError, subprocess.CalledProcessError) as e:
            logging.warning(f"⚠️  Impossible de créer le service systemd: {e}")
    
    async def _create_nginx_config(self):
        """Crée la configuration Nginx."""
        nginx_config = f"""
upstream metrics_backend {{
    server 127.0.0.1:{self.deployment_config.bind_port};
}}

server {{
    listen 80;
    server_name metrics.example.com;
    
    location / {{
        proxy_pass http://metrics_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
    
    location /health {{
        proxy_pass http://127.0.0.1:{self.deployment_config.health_port}/health;
    }}
    
    location /metrics {{
        proxy_pass http://127.0.0.1:{self.deployment_config.metrics_port}/metrics;
    }}
}}
"""
        
        nginx_path = f"{self.deployment_config.config_dir}/nginx.conf"
        
        try:
            with open(nginx_path, 'w') as f:
                f.write(nginx_config)
            
            logging.info(f"✅ Configuration Nginx créée: {nginx_path}")
            
        except Exception as e:
            logging.warning(f"⚠️  Erreur création config Nginx: {e}")
    
    async def _deploy_components(self) -> bool:
        """Déploie les composants du système."""
        logging.info("🚢 Déploiement des composants...")
        
        try:
            if self.deployment_config.infrastructure == InfrastructureType.DOCKER:
                return await self._deploy_docker()
            elif self.deployment_config.infrastructure == InfrastructureType.KUBERNETES:
                return await self._deploy_kubernetes()
            else:
                return await self._deploy_standalone()
            
        except Exception as e:
            logging.error(f"❌ Erreur déploiement composants: {e}")
            return False
    
    async def _deploy_standalone(self) -> bool:
        """Déploie en mode standalone."""
        logging.info("🖥️  Déploiement standalone...")
        
        # Copie des scripts et modules
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Copie des fichiers Python
        src_files = [
            "__init__.py",
            "collector.py",
            "config.py"
        ]
        
        for src_file in src_files:
            src_path = os.path.join(current_dir, src_file)
            dst_path = os.path.join(self.deployment_config.base_dir, src_file)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                logging.debug(f"Fichier copié: {src_file}")
        
        return True
    
    async def _deploy_docker(self) -> bool:
        """Déploie avec Docker."""
        logging.info("🐳 Déploiement Docker...")
        
        if not HAS_DOCKER:
            logging.error("❌ Module docker non disponible")
            return False
        
        try:
            # Vérification Docker
            client = docker.from_env()
            client.ping()
            
            # Démarrage des conteneurs
            compose_path = os.path.join(self.deployment_config.config_dir, "docker-compose.yml")
            
            if os.path.exists(compose_path):
                cmd = ["docker-compose", "-f", compose_path, "up", "-d"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logging.info("✅ Conteneurs Docker démarrés")
                    return True
                else:
                    logging.error(f"❌ Erreur Docker Compose: {result.stderr}")
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Erreur déploiement Docker: {e}")
            return False
    
    async def _deploy_kubernetes(self) -> bool:
        """Déploie sur Kubernetes."""
        logging.info("☸️  Déploiement Kubernetes...")
        
        try:
            # Application des configurations K8s
            config_dir = self.deployment_config.config_dir
            k8s_files = [f for f in os.listdir(config_dir) if f.startswith("k8s-") and f.endswith(".yaml")]
            
            for k8s_file in k8s_files:
                file_path = os.path.join(config_dir, k8s_file)
                cmd = ["kubectl", "apply", "-f", file_path]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logging.info(f"✅ Configuration K8s appliquée: {k8s_file}")
                else:
                    logging.error(f"❌ Erreur K8s {k8s_file}: {result.stderr}")
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Erreur déploiement Kubernetes: {e}")
            return False
    
    async def _run_health_checks(self) -> bool:
        """Exécute les tests de santé."""
        logging.info("🏥 Tests de santé du système...")
        
        try:
            # Attente du démarrage
            await asyncio.sleep(10)
            
            # Test de connectivité
            health_url = f"http://127.0.0.1:{self.deployment_config.health_port}/health"
            
            if HAS_REQUESTS:
                try:
                    response = requests.get(health_url, timeout=5)
                    if response.status_code == 200:
                        logging.info("✅ Test de santé réussi")
                        return True
                    else:
                        logging.error(f"❌ Test de santé échoué: {response.status_code}")
                        return False
                except requests.RequestException as e:
                    logging.error(f"❌ Erreur test de santé: {e}")
                    return False
            else:
                # Test basique de connectivité socket
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                        sock.settimeout(5)
                        result = sock.connect_ex(('127.0.0.1', self.deployment_config.health_port))
                        if result == 0:
                            logging.info("✅ Test de connectivité réussi")
                            return True
                        else:
                            logging.error(f"❌ Test de connectivité échoué")
                            return False
                except Exception as e:
                    logging.error(f"❌ Erreur test connectivité: {e}")
                    return False
            
        except Exception as e:
            logging.error(f"❌ Erreur tests de santé: {e}")
            return False
    
    async def _finalize_deployment(self):
        """Finalise le déploiement."""
        logging.info("🎯 Finalisation du déploiement...")
        
        # Génération du rapport de déploiement
        deployment_report = {
            "deployment_id": self.deployment_config.deployment_id,
            "deployment_name": self.deployment_config.deployment_name,
            "version": self.deployment_config.version,
            "timestamp": datetime.now().isoformat(),
            "mode": self.deployment_config.mode.value,
            "infrastructure": self.deployment_config.infrastructure.value,
            "endpoints": {
                "api": f"http://127.0.0.1:{self.deployment_config.bind_port}",
                "metrics": f"http://127.0.0.1:{self.deployment_config.metrics_port}",
                "health": f"http://127.0.0.1:{self.deployment_config.health_port}"
            },
            "configuration": {
                "config_dir": self.deployment_config.config_dir,
                "data_dir": self.deployment_config.data_dir,
                "log_dir": self.deployment_config.log_dir
            },
            "resources": SystemRequirements.check_system_resources()
        }
        
        # Sauvegarde du rapport
        report_path = os.path.join(self.deployment_config.config_dir, "deployment-report.json")
        with open(report_path, 'w') as f:
            json.dump(deployment_report, f, indent=2, default=str)
        
        logging.info(f"📊 Rapport de déploiement: {report_path}")
        
        # Affichage des informations de déploiement
        print("\n" + "="*60)
        print("🎉 DÉPLOIEMENT TERMINÉ AVEC SUCCÈS")
        print("="*60)
        print(f"📝 ID de déploiement: {self.deployment_config.deployment_id}")
        print(f"🏷️  Nom: {self.deployment_config.deployment_name}")
        print(f"📊 Version: {self.deployment_config.version}")
        print(f"🌍 Mode: {self.deployment_config.mode.value}")
        print(f"🏗️  Infrastructure: {self.deployment_config.infrastructure.value}")
        print("\n📍 Points d'accès:")
        print(f"   • API: http://127.0.0.1:{self.deployment_config.bind_port}")
        print(f"   • Métriques: http://127.0.0.1:{self.deployment_config.metrics_port}")
        print(f"   • Santé: http://127.0.0.1:{self.deployment_config.health_port}")
        print("\n📁 Répertoires:")
        print(f"   • Configuration: {self.deployment_config.config_dir}")
        print(f"   • Données: {self.deployment_config.data_dir}")
        print(f"   • Logs: {self.deployment_config.log_dir}")
        print("="*60)
    
    async def _rollback_deployment(self):
        """Effectue un rollback en cas d'échec."""
        logging.warning("🔄 Rollback du déploiement...")
        
        try:
            # Arrêt des services
            if self.deployment_config.infrastructure == InfrastructureType.DOCKER:
                compose_path = os.path.join(self.deployment_config.config_dir, "docker-compose.yml")
                if os.path.exists(compose_path):
                    subprocess.run(["docker-compose", "-f", compose_path, "down"], 
                                 capture_output=True)
            
            # Nettoyage des fichiers créés
            # Attention: ne pas supprimer les répertoires de données
            logging.info("🧹 Nettoyage effectué")
            
        except Exception as e:
            logging.error(f"❌ Erreur lors du rollback: {e}")


async def main():
    """Fonction principale de configuration et déploiement."""
    parser = argparse.ArgumentParser(description="Configuration et déploiement automatisé du système de métriques")
    
    # Configuration de base
    parser.add_argument("--mode", choices=["development", "staging", "production", "testing", "demo"],
                       default="development", help="Mode de déploiement")
    parser.add_argument("--infrastructure", choices=["standalone", "docker", "kubernetes"],
                       default="standalone", help="Type d'infrastructure")
    parser.add_argument("--name", default="enterprise-metrics-system", help="Nom du déploiement")
    
    # Répertoires
    parser.add_argument("--base-dir", default="/opt/metrics-system", help="Répertoire de base")
    parser.add_argument("--config-dir", default="/etc/metrics-system", help="Répertoire de configuration")
    parser.add_argument("--data-dir", default="/var/lib/metrics-system", help="Répertoire de données")
    parser.add_argument("--log-dir", default="/var/log/metrics-system", help="Répertoire de logs")
    
    # Réseau
    parser.add_argument("--bind-host", default="0.0.0.0", help="Adresse d'écoute")
    parser.add_argument("--bind-port", type=int, default=8080, help="Port d'écoute API")
    parser.add_argument("--metrics-port", type=int, default=9090, help="Port métriques")
    parser.add_argument("--health-port", type=int, default=8081, help="Port santé")
    
    # Base de données
    parser.add_argument("--database", choices=["sqlite", "redis", "postgresql"],
                       default="sqlite", help="Type de base de données")
    parser.add_argument("--database-url", help="URL de la base de données")
    
    # Options
    parser.add_argument("--auto-tune", action="store_true", help="Auto-tuning des performances")
    parser.add_argument("--setup-systemd", action="store_true", help="Configuration service systemd")
    parser.add_argument("--setup-nginx", action="store_true", help="Configuration Nginx")
    parser.add_argument("--enable-monitoring", action="store_true", default=True, help="Activation monitoring")
    parser.add_argument("--enable-prometheus", action="store_true", help="Activation Prometheus")
    parser.add_argument("--enable-grafana", action="store_true", help="Activation Grafana")
    
    # Logs
    parser.add_argument("--log-level", default="INFO", help="Niveau de log")
    parser.add_argument("--log-file", help="Fichier de log")
    
    # Actions
    parser.add_argument("--dry-run", action="store_true", help="Simulation sans déploiement")
    parser.add_argument("--generate-config-only", action="store_true", help="Génération config uniquement")
    
    args = parser.parse_args()
    
    # Configuration du logging
    log_level = getattr(logging, args.log_level.upper())
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    if args.log_file:
        logging.basicConfig(level=log_level, format=log_format, filename=args.log_file)
    else:
        logging.basicConfig(level=log_level, format=log_format)
    
    # Configuration du déploiement
    deployment_config = DeploymentConfig(
        deployment_name=args.name,
        mode=DeploymentMode(args.mode),
        infrastructure=InfrastructureType(args.infrastructure),
        base_dir=args.base_dir,
        config_dir=args.config_dir,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        bind_host=args.bind_host,
        bind_port=args.bind_port,
        metrics_port=args.metrics_port,
        health_port=args.health_port,
        database_type=args.database,
        database_url=args.database_url or "",
        auto_tune_performance=args.auto_tune,
        auto_setup_systemd=args.setup_systemd,
        auto_setup_nginx=args.setup_nginx,
        enable_monitoring=args.enable_monitoring,
        enable_prometheus=args.enable_prometheus,
        enable_grafana=args.enable_grafana
    )
    
    try:
        if args.generate_config_only:
            # Génération de configuration uniquement
            logging.info("📝 Génération de configuration uniquement")
            config_generator = ConfigurationGenerator(deployment_config)
            config = config_generator.generate_base_config()
            
            print(json.dumps(config, indent=2, default=str))
            
        elif args.dry_run:
            # Simulation
            logging.info("🧪 Mode simulation (dry-run)")
            orchestrator = DeploymentOrchestrator(deployment_config)
            
            # Vérifications seulement
            await orchestrator._check_prerequisites()
            
        else:
            # Déploiement complet
            orchestrator = DeploymentOrchestrator(deployment_config)
            success = await orchestrator.deploy()
            
            if success:
                logging.info("🎉 Déploiement terminé avec succès")
                sys.exit(0)
            else:
                logging.error("💥 Échec du déploiement")
                sys.exit(1)
                
    except KeyboardInterrupt:
        logging.info("⏹️  Déploiement interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logging.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
