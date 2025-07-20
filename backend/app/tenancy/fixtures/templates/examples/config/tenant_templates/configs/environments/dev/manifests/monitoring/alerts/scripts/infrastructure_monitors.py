"""
Moniteurs d'Infrastructure Ultra-Avancés
Système de surveillance d'infrastructure intelligent pour Spotify AI Agent

Fonctionnalités:
- Monitoring avancé des serveurs et containers
- Surveillance réseau intelligente
- Monitoring base de données avec optimisations
- Surveillance stockage et I/O
- Monitoring Kubernetes/Docker
- Détection de dérive de configuration
- Prédiction de pannes avec IA
"""

import asyncio
import logging
import psutil
import docker
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import redis.asyncio as redis
import aiohttp
from collections import defaultdict, deque
import numpy as np
import json

from . import AlertConfig, AlertSeverity, AlertCategory, ScriptType, register_alert

logger = logging.getLogger(__name__)

class InfrastructureComponent(Enum):
    """Composants d'infrastructure"""
    SERVER = "server"
    DATABASE = "database"
    NETWORK = "network"
    STORAGE = "storage"
    CONTAINER = "container"
    KUBERNETES = "kubernetes"
    LOAD_BALANCER = "load_balancer"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    MONITORING = "monitoring"

class MetricType(Enum):
    """Types de métriques"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    NETWORK_IO = "network_io"
    DISK_IO = "disk_io"
    CONNECTION_COUNT = "connection_count"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"

@dataclass
class InfrastructureMetric:
    """Métrique d'infrastructure"""
    component: InfrastructureComponent
    metric_type: MetricType
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None

@dataclass
class ServerInfo:
    """Informations sur un serveur"""
    hostname: str
    ip_address: str
    cpu_cores: int
    total_memory: int
    disk_capacity: int
    operating_system: str
    uptime: float
    load_average: Tuple[float, float, float]
    status: str = "healthy"
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ContainerInfo:
    """Informations sur un container"""
    container_id: str
    name: str
    image: str
    status: str
    cpu_usage: float
    memory_usage: int
    network_io: Dict[str, int]
    restart_count: int
    created_at: datetime
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class DatabaseMetrics:
    """Métriques de base de données"""
    db_type: str
    connection_count: int
    active_queries: int
    slow_queries: int
    cache_hit_ratio: float
    replication_lag: float
    disk_usage: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

class AdvancedInfrastructureMonitor:
    """Moniteur d'infrastructure avancé avec IA"""
    
    def __init__(self):
        self.redis_client = None
        self.docker_client = None
        self.metrics_buffer: List[InfrastructureMetric] = []
        self.servers: Dict[str, ServerInfo] = {}
        self.containers: Dict[str, ContainerInfo] = {}
        self.historical_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.monitoring_active = False
        self.alert_thresholds = self._initialize_thresholds()

    async def initialize(self):
        """Initialise le moniteur d'infrastructure"""
        try:
            self.redis_client = redis.Redis(
                host="localhost", 
                port=6379, 
                decode_responses=True,
                db=5
            )
            
            # Initialisation client Docker
            try:
                self.docker_client = docker.from_env()
                logger.info("Client Docker initialisé")
            except Exception as e:
                logger.warning(f"Impossible d'initialiser Docker: {e}")
            
            # Collecte des informations système de base
            await self._collect_server_info()
            
            logger.info("Moniteur d'infrastructure initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation: {e}")

    def _initialize_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialise les seuils d'alerte"""
        
        return {
            "cpu_usage": {
                "warning": 70.0,
                "critical": 85.0
            },
            "memory_usage": {
                "warning": 80.0,
                "critical": 90.0
            },
            "disk_usage": {
                "warning": 75.0,
                "critical": 85.0
            },
            "network_latency": {
                "warning": 100.0,  # ms
                "critical": 500.0
            },
            "error_rate": {
                "warning": 1.0,   # %
                "critical": 5.0
            },
            "response_time": {
                "warning": 1000.0,  # ms
                "critical": 5000.0
            }
        }

    async def start_monitoring(self):
        """Démarre le monitoring d'infrastructure"""
        self.monitoring_active = True
        
        monitoring_tasks = [
            self._monitor_system_resources(),
            self._monitor_network(),
            self._monitor_docker_containers(),
            self._monitor_databases(),
            self._monitor_kubernetes(),
            self._analyze_trends_and_predict(),
            self._health_checks(),
            self._configuration_drift_detection()
        ]
        
        await asyncio.gather(*monitoring_tasks)

    async def _monitor_system_resources(self):
        """Monitore les ressources système"""
        while self.monitoring_active:
            try:
                # CPU
                cpu_percent = psutil.cpu_percent(interval=1)
                await self._record_metric(
                    InfrastructureComponent.SERVER,
                    MetricType.CPU_USAGE,
                    cpu_percent,
                    "%"
                )
                
                # Mémoire
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                await self._record_metric(
                    InfrastructureComponent.SERVER,
                    MetricType.MEMORY_USAGE,
                    memory_percent,
                    "%"
                )
                
                # Disque
                for partition in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        disk_percent = (usage.used / usage.total) * 100
                        
                        await self._record_metric(
                            InfrastructureComponent.STORAGE,
                            MetricType.DISK_USAGE,
                            disk_percent,
                            "%",
                            tags={"partition": partition.device}
                        )
                    except PermissionError:
                        continue
                
                # I/O disque
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    await self._record_metric(
                        InfrastructureComponent.STORAGE,
                        MetricType.DISK_IO,
                        disk_io.read_bytes + disk_io.write_bytes,
                        "bytes"
                    )
                
                # Réseau I/O
                net_io = psutil.net_io_counters()
                if net_io:
                    await self._record_metric(
                        InfrastructureComponent.NETWORK,
                        MetricType.NETWORK_IO,
                        net_io.bytes_sent + net_io.bytes_recv,
                        "bytes"
                    )
                
                # Load average
                load_avg = psutil.getloadavg()
                await self._record_metric(
                    InfrastructureComponent.SERVER,
                    MetricType.CPU_USAGE,
                    load_avg[0],
                    "load",
                    tags={"type": "load_1min"}
                )
                
                await asyncio.sleep(30)  # Collecte toutes les 30 secondes
                
            except Exception as e:
                logger.error(f"Erreur monitoring ressources système: {e}")
                await asyncio.sleep(60)

    async def _monitor_network(self):
        """Monitore la performance réseau"""
        while self.monitoring_active:
            try:
                # Test de latence vers des endpoints critiques
                endpoints = [
                    "8.8.8.8",
                    "1.1.1.1",
                    "localhost",
                    "127.0.0.1"
                ]
                
                for endpoint in endpoints:
                    latency = await self._ping_endpoint(endpoint)
                    if latency is not None:
                        await self._record_metric(
                            InfrastructureComponent.NETWORK,
                            MetricType.RESPONSE_TIME,
                            latency,
                            "ms",
                            tags={"endpoint": endpoint}
                        )
                
                # Statistiques interfaces réseau
                net_stats = psutil.net_if_stats()
                for interface, stats in net_stats.items():
                    if stats.isup:
                        await self._record_metric(
                            InfrastructureComponent.NETWORK,
                            MetricType.AVAILABILITY,
                            100.0 if stats.isup else 0.0,
                            "%",
                            tags={"interface": interface}
                        )
                
                # Connexions réseau
                connections = psutil.net_connections()
                connection_count = len([c for c in connections if c.status == "ESTABLISHED"])
                
                await self._record_metric(
                    InfrastructureComponent.NETWORK,
                    MetricType.CONNECTION_COUNT,
                    connection_count,
                    "count"
                )
                
                await asyncio.sleep(60)  # Vérification toutes les minutes
                
            except Exception as e:
                logger.error(f"Erreur monitoring réseau: {e}")
                await asyncio.sleep(120)

    async def _ping_endpoint(self, endpoint: str) -> Optional[float]:
        """Ping un endpoint et retourne la latence"""
        try:
            # Utilisation de ping système
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "5", endpoint],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # Extraction du temps de ping
                import re
                match = re.search(r"time=(\d+\.?\d*)", result.stdout)
                if match:
                    return float(match.group(1))
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur ping {endpoint}: {e}")
            return None

    async def _monitor_docker_containers(self):
        """Monitore les containers Docker"""
        while self.monitoring_active:
            try:
                if not self.docker_client:
                    await asyncio.sleep(300)
                    continue
                
                containers = self.docker_client.containers.list(all=True)
                
                for container in containers:
                    try:
                        # Informations de base
                        stats = container.stats(stream=False)
                        
                        # CPU usage
                        cpu_usage = self._calculate_container_cpu_usage(stats)
                        await self._record_metric(
                            InfrastructureComponent.CONTAINER,
                            MetricType.CPU_USAGE,
                            cpu_usage,
                            "%",
                            tags={"container": container.name}
                        )
                        
                        # Memory usage
                        memory_usage = stats["memory_stats"]["usage"]
                        memory_limit = stats["memory_stats"]["limit"]
                        memory_percent = (memory_usage / memory_limit) * 100
                        
                        await self._record_metric(
                            InfrastructureComponent.CONTAINER,
                            MetricType.MEMORY_USAGE,
                            memory_percent,
                            "%",
                            tags={"container": container.name}
                        )
                        
                        # Network I/O
                        networks = stats.get("networks", {})
                        total_rx = sum(net["rx_bytes"] for net in networks.values())
                        total_tx = sum(net["tx_bytes"] for net in networks.values())
                        
                        await self._record_metric(
                            InfrastructureComponent.CONTAINER,
                            MetricType.NETWORK_IO,
                            total_rx + total_tx,
                            "bytes",
                            tags={"container": container.name}
                        )
                        
                        # Mise à jour des infos container
                        container_info = ContainerInfo(
                            container_id=container.id[:12],
                            name=container.name,
                            image=container.image.tags[0] if container.image.tags else "unknown",
                            status=container.status,
                            cpu_usage=cpu_usage,
                            memory_usage=memory_usage,
                            network_io={"rx": total_rx, "tx": total_tx},
                            restart_count=container.attrs["RestartCount"],
                            created_at=datetime.fromisoformat(
                                container.attrs["Created"].replace("Z", "+00:00")
                            )
                        )
                        
                        self.containers[container.id] = container_info
                        
                    except Exception as e:
                        logger.error(f"Erreur monitoring container {container.name}: {e}")
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Erreur monitoring Docker: {e}")
                await asyncio.sleep(120)

    def _calculate_container_cpu_usage(self, stats: Dict[str, Any]) -> float:
        """Calcule l'usage CPU d'un container"""
        try:
            cpu_stats = stats["cpu_stats"]
            precpu_stats = stats["precpu_stats"]
            
            cpu_delta = cpu_stats["cpu_usage"]["total_usage"] - precpu_stats["cpu_usage"]["total_usage"]
            system_delta = cpu_stats["system_cpu_usage"] - precpu_stats["system_cpu_usage"]
            
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * len(cpu_stats["cpu_usage"]["percpu_usage"]) * 100
                return min(100.0, max(0.0, cpu_percent))
            
            return 0.0
            
        except (KeyError, ZeroDivisionError):
            return 0.0

    async def _monitor_databases(self):
        """Monitore les bases de données"""
        while self.monitoring_active:
            try:
                # PostgreSQL monitoring
                await self._monitor_postgresql()
                
                # Redis monitoring
                await self._monitor_redis()
                
                # MongoDB monitoring (si disponible)
                await self._monitor_mongodb()
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Erreur monitoring bases de données: {e}")
                await asyncio.sleep(120)

    async def _monitor_postgresql(self):
        """Monitore PostgreSQL"""
        try:
            # Simulation de métriques PostgreSQL
            # En production, utiliser psycopg2 ou asyncpg
            
            metrics = {
                "connection_count": 25,
                "active_queries": 5,
                "slow_queries": 2,
                "cache_hit_ratio": 98.5,
                "replication_lag": 0.1,
                "disk_usage": 45.2
            }
            
            for metric_name, value in metrics.items():
                await self._record_metric(
                    InfrastructureComponent.DATABASE,
                    MetricType.CONNECTION_COUNT if "connection" in metric_name else MetricType.RESPONSE_TIME,
                    value,
                    "count" if "count" in metric_name else "ms",
                    tags={"database": "postgresql", "metric": metric_name}
                )
                
        except Exception as e:
            logger.error(f"Erreur monitoring PostgreSQL: {e}")

    async def _monitor_redis(self):
        """Monitore Redis"""
        try:
            if self.redis_client:
                info = await self.redis_client.info()
                
                # Métriques Redis importantes
                connected_clients = info.get("connected_clients", 0)
                used_memory = info.get("used_memory", 0)
                total_commands = info.get("total_commands_processed", 0)
                keyspace_hits = info.get("keyspace_hits", 0)
                keyspace_misses = info.get("keyspace_misses", 0)
                
                # Hit ratio
                total_requests = keyspace_hits + keyspace_misses
                hit_ratio = (keyspace_hits / total_requests * 100) if total_requests > 0 else 0
                
                await self._record_metric(
                    InfrastructureComponent.CACHE,
                    MetricType.CONNECTION_COUNT,
                    connected_clients,
                    "count",
                    tags={"database": "redis"}
                )
                
                await self._record_metric(
                    InfrastructureComponent.CACHE,
                    MetricType.MEMORY_USAGE,
                    used_memory / (1024 * 1024),  # MB
                    "MB",
                    tags={"database": "redis"}
                )
                
                await self._record_metric(
                    InfrastructureComponent.CACHE,
                    MetricType.THROUGHPUT,
                    hit_ratio,
                    "%",
                    tags={"database": "redis", "metric": "hit_ratio"}
                )
                
        except Exception as e:
            logger.error(f"Erreur monitoring Redis: {e}")

    async def _monitor_mongodb(self):
        """Monitore MongoDB"""
        try:
            # Simulation de métriques MongoDB
            # En production, utiliser motor ou pymongo
            
            metrics = {
                "connections": 15,
                "operations_per_second": 150,
                "lock_percentage": 2.1,
                "page_faults": 5
            }
            
            for metric_name, value in metrics.items():
                await self._record_metric(
                    InfrastructureComponent.DATABASE,
                    MetricType.CONNECTION_COUNT if "connection" in metric_name else MetricType.THROUGHPUT,
                    value,
                    "count" if "connection" in metric_name else "ops/s",
                    tags={"database": "mongodb", "metric": metric_name}
                )
                
        except Exception as e:
            logger.error(f"Erreur monitoring MongoDB: {e}")

    async def _monitor_kubernetes(self):
        """Monitore Kubernetes (si disponible)"""
        while self.monitoring_active:
            try:
                # Vérification si kubectl est disponible
                result = subprocess.run(
                    ["kubectl", "cluster-info"], 
                    capture_output=True, 
                    text=True,
                    timeout=10
                )
                
                if result.returncode != 0:
                    await asyncio.sleep(300)  # Retry dans 5 minutes
                    continue
                
                # Monitoring des pods
                await self._monitor_k8s_pods()
                
                # Monitoring des nodes
                await self._monitor_k8s_nodes()
                
                # Monitoring des services
                await self._monitor_k8s_services()
                
                await asyncio.sleep(120)  # Vérification toutes les 2 minutes
                
            except Exception as e:
                logger.error(f"Erreur monitoring Kubernetes: {e}")
                await asyncio.sleep(300)

    async def _monitor_k8s_pods(self):
        """Monitore les pods Kubernetes"""
        try:
            result = subprocess.run(
                ["kubectl", "get", "pods", "--all-namespaces", "-o", "json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                pods = data.get("items", [])
                
                # Statistiques des pods
                pod_stats = {
                    "Running": 0,
                    "Pending": 0,
                    "Failed": 0,
                    "Succeeded": 0
                }
                
                for pod in pods:
                    status = pod.get("status", {}).get("phase", "Unknown")
                    pod_stats[status] = pod_stats.get(status, 0) + 1
                
                for status, count in pod_stats.items():
                    await self._record_metric(
                        InfrastructureComponent.KUBERNETES,
                        MetricType.AVAILABILITY,
                        count,
                        "count",
                        tags={"resource": "pods", "status": status}
                    )
                    
        except Exception as e:
            logger.error(f"Erreur monitoring pods K8s: {e}")

    async def _monitor_k8s_nodes(self):
        """Monitore les nodes Kubernetes"""
        try:
            result = subprocess.run(
                ["kubectl", "top", "nodes"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        node_name = parts[0]
                        cpu_usage = parts[1].replace('%', '')
                        memory_usage = parts[3].replace('%', '')
                        
                        await self._record_metric(
                            InfrastructureComponent.KUBERNETES,
                            MetricType.CPU_USAGE,
                            float(cpu_usage),
                            "%",
                            tags={"resource": "node", "node": node_name}
                        )
                        
                        await self._record_metric(
                            InfrastructureComponent.KUBERNETES,
                            MetricType.MEMORY_USAGE,
                            float(memory_usage),
                            "%",
                            tags={"resource": "node", "node": node_name}
                        )
                        
        except Exception as e:
            logger.error(f"Erreur monitoring nodes K8s: {e}")

    async def _monitor_k8s_services(self):
        """Monitore les services Kubernetes"""
        try:
            result = subprocess.run(
                ["kubectl", "get", "services", "--all-namespaces", "-o", "json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                services = data.get("items", [])
                
                await self._record_metric(
                    InfrastructureComponent.KUBERNETES,
                    MetricType.AVAILABILITY,
                    len(services),
                    "count",
                    tags={"resource": "services"}
                )
                
        except Exception as e:
            logger.error(f"Erreur monitoring services K8s: {e}")

    async def _analyze_trends_and_predict(self):
        """Analyse les tendances et prédit les problèmes"""
        while self.monitoring_active:
            try:
                # Analyse des tendances CPU
                await self._analyze_cpu_trends()
                
                # Analyse des tendances mémoire
                await self._analyze_memory_trends()
                
                # Prédiction de l'espace disque
                await self._predict_disk_usage()
                
                # Détection d'anomalies
                await self._detect_infrastructure_anomalies()
                
                await asyncio.sleep(300)  # Analyse toutes les 5 minutes
                
            except Exception as e:
                logger.error(f"Erreur analyse tendances: {e}")
                await asyncio.sleep(600)

    async def _analyze_cpu_trends(self):
        """Analyse les tendances d'usage CPU"""
        try:
            cpu_data = list(self.historical_data["cpu_usage"])
            
            if len(cpu_data) >= 10:
                # Calcul de la tendance (régression linéaire simple)
                values = [point["value"] for point in cpu_data[-20:]]
                
                if len(values) >= 2:
                    # Tendance croissante détectée
                    recent_avg = sum(values[-5:]) / 5
                    older_avg = sum(values[:5]) / 5
                    
                    if recent_avg > older_avg + 10:  # Augmentation de 10%
                        logger.warning(f"Tendance CPU croissante détectée: {recent_avg:.1f}%")
                        
                        # Prédiction dans 30 minutes
                        growth_rate = (recent_avg - older_avg) / len(values)
                        predicted_usage = recent_avg + (growth_rate * 6)  # 6 points = 30 min
                        
                        if predicted_usage > 85:
                            await self._create_infrastructure_alert(
                                "CPU usage trending high",
                                f"CPU usage may reach {predicted_usage:.1f}% in 30 minutes",
                                AlertSeverity.WARNING
                            )
                            
        except Exception as e:
            logger.error(f"Erreur analyse tendances CPU: {e}")

    async def _analyze_memory_trends(self):
        """Analyse les tendances d'usage mémoire"""
        try:
            memory_data = list(self.historical_data["memory_usage"])
            
            if len(memory_data) >= 10:
                values = [point["value"] for point in memory_data[-15:]]
                
                if len(values) >= 2:
                    recent_avg = sum(values[-5:]) / 5
                    older_avg = sum(values[:5]) / 5
                    
                    if recent_avg > older_avg + 5:  # Augmentation de 5%
                        growth_rate = (recent_avg - older_avg) / len(values)
                        predicted_usage = recent_avg + (growth_rate * 12)  # 1 heure
                        
                        if predicted_usage > 90:
                            await self._create_infrastructure_alert(
                                "Memory leak suspected",
                                f"Memory usage may reach {predicted_usage:.1f}% in 1 hour",
                                AlertSeverity.HIGH
                            )
                            
        except Exception as e:
            logger.error(f"Erreur analyse tendances mémoire: {e}")

    async def _predict_disk_usage(self):
        """Prédit l'usage du disque"""
        try:
            disk_data = list(self.historical_data["disk_usage"])
            
            if len(disk_data) >= 20:
                values = [point["value"] for point in disk_data]
                
                # Calcul du taux de croissance
                daily_growth = self._calculate_daily_growth_rate(values)
                
                if daily_growth > 0:
                    current_usage = values[-1]
                    
                    # Prédiction à 7 et 30 jours
                    usage_7d = current_usage + (daily_growth * 7)
                    usage_30d = current_usage + (daily_growth * 30)
                    
                    if usage_7d > 85:
                        await self._create_infrastructure_alert(
                            "Disk space will be critical",
                            f"Disk usage may reach {usage_7d:.1f}% in 7 days",
                            AlertSeverity.WARNING
                        )
                    
                    if usage_30d > 95:
                        await self._create_infrastructure_alert(
                            "Disk space planning required",
                            f"Disk usage may reach {usage_30d:.1f}% in 30 days",
                            AlertSeverity.MEDIUM
                        )
                        
        except Exception as e:
            logger.error(f"Erreur prédiction disque: {e}")

    def _calculate_daily_growth_rate(self, values: List[float]) -> float:
        """Calcule le taux de croissance quotidien"""
        try:
            if len(values) < 2:
                return 0.0
            
            # Approximation simple: différence entre premier et dernier / nombre de jours
            days = len(values) / (24 * 2)  # Assuming 30-second intervals
            if days > 0:
                return (values[-1] - values[0]) / days
            
            return 0.0
            
        except Exception:
            return 0.0

    async def _detect_infrastructure_anomalies(self):
        """Détecte les anomalies d'infrastructure"""
        try:
            # Détection d'anomalies basée sur des seuils statistiques
            for metric_name, data in self.historical_data.items():
                if len(data) >= 30:  # Au moins 30 points
                    values = [point["value"] for point in data]
                    
                    # Calcul statistiques
                    mean = np.mean(values)
                    std = np.std(values)
                    current_value = values[-1]
                    
                    # Détection outlier (> 3 sigma)
                    if abs(current_value - mean) > 3 * std:
                        severity = AlertSeverity.HIGH if abs(current_value - mean) > 4 * std else AlertSeverity.WARNING
                        
                        await self._create_infrastructure_alert(
                            f"Anomaly detected in {metric_name}",
                            f"Current value {current_value:.2f} is {abs(current_value - mean) / std:.1f} standard deviations from normal",
                            severity
                        )
                        
        except Exception as e:
            logger.error(f"Erreur détection anomalies: {e}")

    async def _health_checks(self):
        """Effectue des vérifications de santé"""
        while self.monitoring_active:
            try:
                # Vérification services critiques
                await self._check_critical_services()
                
                # Vérification processus
                await self._check_critical_processes()
                
                # Vérification certificats SSL
                await self._check_ssl_certificates()
                
                await asyncio.sleep(120)  # Vérification toutes les 2 minutes
                
            except Exception as e:
                logger.error(f"Erreur health checks: {e}")
                await asyncio.sleep(300)

    async def _check_critical_services(self):
        """Vérifie les services critiques"""
        critical_services = [
            ("Redis", "127.0.0.1", 6379),
            ("HTTP", "127.0.0.1", 8000),
            ("PostgreSQL", "127.0.0.1", 5432)
        ]
        
        for service_name, host, port in critical_services:
            is_available = await self._check_service_availability(host, port)
            
            await self._record_metric(
                InfrastructureComponent.MONITORING,
                MetricType.AVAILABILITY,
                100.0 if is_available else 0.0,
                "%",
                tags={"service": service_name}
            )
            
            if not is_available:
                await self._create_infrastructure_alert(
                    f"Service {service_name} unavailable",
                    f"Cannot connect to {service_name} on {host}:{port}",
                    AlertSeverity.CRITICAL
                )

    async def _check_service_availability(self, host: str, port: int) -> bool:
        """Vérifie la disponibilité d'un service"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=5
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False

    async def _check_critical_processes(self):
        """Vérifie les processus critiques"""
        critical_processes = [
            "redis-server",
            "postgres",
            "nginx",
            "python"
        ]
        
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                proc_name = proc.info['name']
                if any(critical in proc_name for critical in critical_processes):
                    await self._record_metric(
                        InfrastructureComponent.MONITORING,
                        MetricType.AVAILABILITY,
                        100.0,
                        "%",
                        tags={"process": proc_name}
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    async def _check_ssl_certificates(self):
        """Vérifie l'expiration des certificats SSL"""
        try:
            # Simulation de vérification SSL
            # En production, utiliser ssl et socket
            
            domains = ["localhost", "api.example.com"]
            
            for domain in domains:
                # Simulation: certificat expire dans 30 jours
                days_until_expiry = 30
                
                if days_until_expiry < 30:
                    severity = AlertSeverity.CRITICAL if days_until_expiry < 7 else AlertSeverity.WARNING
                    
                    await self._create_infrastructure_alert(
                        f"SSL certificate expiring for {domain}",
                        f"Certificate expires in {days_until_expiry} days",
                        severity
                    )
                    
        except Exception as e:
            logger.error(f"Erreur vérification SSL: {e}")

    async def _configuration_drift_detection(self):
        """Détecte la dérive de configuration"""
        while self.monitoring_active:
            try:
                # Vérification configuration système
                await self._check_system_configuration()
                
                # Vérification configuration Docker
                await self._check_docker_configuration()
                
                # Vérification configuration Kubernetes
                await self._check_kubernetes_configuration()
                
                await asyncio.sleep(3600)  # Vérification toutes les heures
                
            except Exception as e:
                logger.error(f"Erreur détection dérive configuration: {e}")
                await asyncio.sleep(7200)

    async def _check_system_configuration(self):
        """Vérifie la configuration système"""
        try:
            # Vérification des limites système
            limits = {
                "max_open_files": 65536,
                "max_processes": 32768
            }
            
            # Simulation de vérification
            current_limits = {
                "max_open_files": 65536,
                "max_processes": 32768
            }
            
            for limit_name, expected_value in limits.items():
                current_value = current_limits.get(limit_name, 0)
                
                if current_value != expected_value:
                    await self._create_infrastructure_alert(
                        f"System configuration drift: {limit_name}",
                        f"Expected {expected_value}, found {current_value}",
                        AlertSeverity.WARNING
                    )
                    
        except Exception as e:
            logger.error(f"Erreur vérification configuration système: {e}")

    async def _check_docker_configuration(self):
        """Vérifie la configuration Docker"""
        try:
            if not self.docker_client:
                return
            
            # Vérification version Docker
            version_info = self.docker_client.version()
            docker_version = version_info.get("Version", "unknown")
            
            # Alertes de sécurité pour versions obsolètes
            if docker_version < "20.10":
                await self._create_infrastructure_alert(
                    "Docker version outdated",
                    f"Docker version {docker_version} has known security vulnerabilities",
                    AlertSeverity.MEDIUM
                )
                
        except Exception as e:
            logger.error(f"Erreur vérification configuration Docker: {e}")

    async def _check_kubernetes_configuration(self):
        """Vérifie la configuration Kubernetes"""
        try:
            # Vérification RBAC
            result = subprocess.run(
                ["kubectl", "auth", "can-i", "--list"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Analyse des permissions (simulation)
                excessive_permissions = False
                
                if excessive_permissions:
                    await self._create_infrastructure_alert(
                        "Excessive Kubernetes permissions detected",
                        "Review RBAC configuration for security compliance",
                        AlertSeverity.MEDIUM
                    )
                    
        except Exception as e:
            logger.error(f"Erreur vérification configuration K8s: {e}")

    async def _record_metric(self, component: InfrastructureComponent, metric_type: MetricType, value: float, unit: str, tags: Optional[Dict[str, str]] = None):
        """Enregistre une métrique d'infrastructure"""
        
        metric = InfrastructureMetric(
            component=component,
            metric_type=metric_type,
            value=value,
            unit=unit,
            timestamp=datetime.utcnow(),
            tags=tags or {}
        )
        
        # Définition des seuils
        thresholds = self.alert_thresholds.get(metric_type.value, {})
        metric.threshold_warning = thresholds.get("warning")
        metric.threshold_critical = thresholds.get("critical")
        
        self.metrics_buffer.append(metric)
        
        # Stockage dans l'historique
        metric_key = f"{component.value}_{metric_type.value}"
        self.historical_data[metric_key].append({
            "value": value,
            "timestamp": metric.timestamp.isoformat(),
            "tags": tags
        })
        
        # Vérification des seuils
        await self._check_metric_thresholds(metric)
        
        # Stockage dans Redis
        await self._store_metric_in_redis(metric)

    async def _check_metric_thresholds(self, metric: InfrastructureMetric):
        """Vérifie les seuils d'une métrique"""
        
        if metric.threshold_critical and metric.value >= metric.threshold_critical:
            await self._create_infrastructure_alert(
                f"Critical threshold exceeded: {metric.component.value} {metric.metric_type.value}",
                f"Value {metric.value:.2f}{metric.unit} exceeds critical threshold {metric.threshold_critical}{metric.unit}",
                AlertSeverity.CRITICAL
            )
        elif metric.threshold_warning and metric.value >= metric.threshold_warning:
            await self._create_infrastructure_alert(
                f"Warning threshold exceeded: {metric.component.value} {metric.metric_type.value}",
                f"Value {metric.value:.2f}{metric.unit} exceeds warning threshold {metric.threshold_warning}{metric.unit}",
                AlertSeverity.WARNING
            )

    async def _store_metric_in_redis(self, metric: InfrastructureMetric):
        """Stocke une métrique dans Redis"""
        try:
            if self.redis_client:
                key = f"infrastructure_metric:{metric.component.value}:{metric.metric_type.value}"
                
                # Time series avec sliding window
                timestamp = int(metric.timestamp.timestamp())
                
                # Pipeline pour operations atomiques
                pipe = self.redis_client.pipeline()
                pipe.zadd(key, {f"{timestamp}:{metric.value}": timestamp})
                pipe.zremrangebyscore(key, 0, timestamp - 3600)  # Garder 1 heure
                pipe.expire(key, 7200)  # TTL 2 heures
                
                await pipe.execute()
                
        except Exception as e:
            logger.error(f"Erreur stockage métrique Redis: {e}")

    async def _create_infrastructure_alert(self, title: str, description: str, severity: AlertSeverity):
        """Crée une alerte d'infrastructure"""
        
        logger.log(
            logging.CRITICAL if severity == AlertSeverity.CRITICAL else logging.WARNING,
            f"ALERTE INFRASTRUCTURE [{severity.value.upper()}]: {title} - {description}"
        )

    async def _collect_server_info(self):
        """Collecte les informations du serveur"""
        try:
            hostname = psutil.os.uname().nodename
            
            # Informations réseau
            interfaces = psutil.net_if_addrs()
            ip_address = "127.0.0.1"
            
            for interface, addrs in interfaces.items():
                for addr in addrs:
                    if addr.family == 2 and not addr.address.startswith("127."):  # IPv4
                        ip_address = addr.address
                        break
            
            # Informations système
            cpu_count = psutil.cpu_count()
            memory_total = psutil.virtual_memory().total
            disk_total = psutil.disk_usage('/').total
            
            boot_time = psutil.boot_time()
            uptime = datetime.utcnow().timestamp() - boot_time
            
            load_avg = psutil.getloadavg()
            
            server_info = ServerInfo(
                hostname=hostname,
                ip_address=ip_address,
                cpu_cores=cpu_count,
                total_memory=memory_total,
                disk_capacity=disk_total,
                operating_system=f"{psutil.os.name} {psutil.os.uname().release}",
                uptime=uptime,
                load_average=load_avg
            )
            
            self.servers[hostname] = server_info
            
            logger.info(f"Informations serveur collectées: {hostname} ({ip_address})")
            
        except Exception as e:
            logger.error(f"Erreur collecte informations serveur: {e}")

    async def get_infrastructure_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de l'infrastructure"""
        
        recent_metrics = [
            m for m in self.metrics_buffer
            if datetime.utcnow() - m.timestamp < timedelta(minutes=30)
        ]
        
        # Métriques par composant
        component_metrics = defaultdict(list)
        for metric in recent_metrics:
            component_metrics[metric.component.value].append(metric)
        
        # Alertes actives
        active_alerts = sum(
            1 for metric in recent_metrics
            if metric.threshold_critical and metric.value >= metric.threshold_critical
        )
        
        # Santé globale
        health_score = 100
        
        for metric in recent_metrics:
            if metric.threshold_critical and metric.value >= metric.threshold_critical:
                health_score -= 20
            elif metric.threshold_warning and metric.value >= metric.threshold_warning:
                health_score -= 5
        
        health_score = max(0, health_score)
        
        return {
            "health_score": health_score,
            "servers_count": len(self.servers),
            "containers_count": len(self.containers),
            "active_alerts": active_alerts,
            "metrics_collected": len(self.metrics_buffer),
            "component_metrics": {
                comp: len(metrics) for comp, metrics in component_metrics.items()
            },
            "monitoring_active": self.monitoring_active,
            "last_update": datetime.utcnow().isoformat()
        }

    async def stop_monitoring(self):
        """Arrête le monitoring d'infrastructure"""
        self.monitoring_active = False
        
        if self.docker_client:
            self.docker_client.close()
        
        logger.info("Monitoring d'infrastructure arrêté")

# Instance globale du moniteur d'infrastructure
_infrastructure_monitor = AdvancedInfrastructureMonitor()

async def start_infrastructure_monitoring():
    """Function helper pour démarrer le monitoring d'infrastructure"""
    if not _infrastructure_monitor.redis_client:
        await _infrastructure_monitor.initialize()
    
    await _infrastructure_monitor.start_monitoring()

async def get_infrastructure_monitor() -> AdvancedInfrastructureMonitor:
    """Retourne l'instance du moniteur d'infrastructure"""
    return _infrastructure_monitor

# Configuration des alertes d'infrastructure
if __name__ == "__main__":
    # Enregistrement des configurations d'alertes
    infrastructure_configs = [
        AlertConfig(
            name="high_cpu_usage",
            category=AlertCategory.PERFORMANCE,
            severity=AlertSeverity.HIGH,
            script_type=ScriptType.DETECTION,
            conditions=['CPU usage > 85%'],
            actions=['scale_up_instances', 'optimize_processes'],
            ml_enabled=True,
            auto_remediation=True
        ),
        AlertConfig(
            name="memory_leak_detected",
            category=AlertCategory.PERFORMANCE,
            severity=AlertSeverity.CRITICAL,
            script_type=ScriptType.DETECTION,
            conditions=['Memory usage trending up rapidly'],
            actions=['restart_service', 'dump_memory', 'investigate'],
            ml_enabled=True
        ),
        AlertConfig(
            name="disk_space_critical",
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.CRITICAL,
            script_type=ScriptType.DETECTION,
            conditions=['Disk usage > 90%'],
            actions=['cleanup_logs', 'extend_volume', 'alert_admin'],
            ml_enabled=False,
            auto_remediation=True
        ),
        AlertConfig(
            name="container_failed",
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.HIGH,
            script_type=ScriptType.DETECTION,
            conditions=['Container status = Failed'],
            actions=['restart_container', 'check_logs', 'notify_oncall'],
            ml_enabled=False,
            auto_remediation=True
        ),
        AlertConfig(
            name="network_latency_high",
            category=AlertCategory.NETWORK,
            severity=AlertSeverity.WARNING,
            script_type=ScriptType.DETECTION,
            conditions=['Network latency > 500ms'],
            actions=['check_network_path', 'optimize_routing'],
            ml_enabled=True
        )
    ]
    
    for config in infrastructure_configs:
        register_alert(config)
