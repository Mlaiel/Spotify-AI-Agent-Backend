"""
Analyseur de Performance et Métriques Système Avancé
===================================================

Auteur: Fahed Mlaiel
Rôles: Lead Dev + Architecte IA, Architecte Microservices

Ce module implémente des analyseurs de performance sophistiqués pour le monitoring
en temps réel des métriques système, application et infrastructure.
"""

import asyncio
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import redis
import aioredis
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import docker
from kubernetes import client, config
import requests
import socket
from collections import deque, defaultdict
import threading
import time

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types de métriques"""
    SYSTEM = "system"           # CPU, RAM, Disk, Network
    APPLICATION = "application" # Response time, throughput, errors
    DATABASE = "database"       # Connection pool, query time, locks
    CACHE = "cache"            # Hit rate, memory usage, evictions
    NETWORK = "network"        # Latency, packet loss, bandwidth
    CONTAINER = "container"    # Docker/K8s metrics
    CUSTOM = "custom"          # Métriques métier

class PerformanceThreshold(Enum):
    """Seuils de performance prédéfinis"""
    EXCELLENT = "excellent"    # 95th percentile
    GOOD = "good"             # 90th percentile
    ACCEPTABLE = "acceptable"  # 75th percentile
    POOR = "poor"             # 50th percentile
    CRITICAL = "critical"     # Below 50th percentile

@dataclass
class PerformanceMetric:
    """Métrique de performance"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    threshold_level: Optional[PerformanceThreshold] = None
    trend_direction: Optional[str] = None  # "up", "down", "stable"
    
@dataclass
class PerformanceAlert:
    """Alerte de performance"""
    metric_name: str
    current_value: float
    threshold_value: float
    severity: str
    message: str
    recommendations: List[str]
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)

class SystemMetricsCollector:
    """Collecteur de métriques système avancé"""
    
    def __init__(self, collection_interval: int = 60):
        self.collection_interval = collection_interval
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.is_collecting = False
        self.redis_client = None
        
        # Métriques Prometheus
        self.registry = CollectorRegistry()
        self.cpu_usage = Gauge('system_cpu_usage_percent', 
                              'CPU usage percentage', 
                              registry=self.registry)
        self.memory_usage = Gauge('system_memory_usage_percent', 
                                 'Memory usage percentage', 
                                 registry=self.registry)
        self.disk_usage = Gauge('system_disk_usage_percent', 
                               'Disk usage percentage', 
                               ['device'], registry=self.registry)
        self.network_io = Counter('system_network_io_bytes_total', 
                                 'Network I/O bytes', 
                                 ['direction'], registry=self.registry)
        
    async def initialize(self, redis_url: str = "redis://localhost:6379/3"):
        """Initialise le collecteur"""
        self.redis_client = aioredis.from_url(redis_url)
        logger.info("Collecteur de métriques système initialisé")
    
    async def start_collection(self):
        """Démarre la collecte de métriques"""
        self.is_collecting = True
        asyncio.create_task(self._collection_loop())
        logger.info("Collecte de métriques démarrée")
    
    async def stop_collection(self):
        """Arrête la collecte de métriques"""
        self.is_collecting = False
        logger.info("Collecte de métriques arrêtée")
    
    async def _collection_loop(self):
        """Boucle principale de collecte"""
        while self.is_collecting:
            try:
                await self._collect_all_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Erreur collecte métriques: {e}")
                await asyncio.sleep(10)  # Retry after 10 seconds
    
    async def _collect_all_metrics(self):
        """Collecte toutes les métriques"""
        timestamp = datetime.now()
        
        # Métriques système
        system_metrics = await self._collect_system_metrics()
        
        # Métriques réseau
        network_metrics = await self._collect_network_metrics()
        
        # Métriques processus
        process_metrics = await self._collect_process_metrics()
        
        # Métriques Docker si disponible
        try:
            docker_metrics = await self._collect_docker_metrics()
        except Exception:
            docker_metrics = []
        
        # Combiner toutes les métriques
        all_metrics = system_metrics + network_metrics + process_metrics + docker_metrics
        
        # Stocker en mémoire et Redis
        for metric in all_metrics:
            self.metrics_history[metric.name].append(metric)
            await self._store_metric_in_redis(metric)
        
        # Mettre à jour Prometheus
        await self._update_prometheus_metrics(all_metrics)
    
    async def _collect_system_metrics(self) -> List[PerformanceMetric]:
        """Collecte les métriques système de base"""
        metrics = []
        timestamp = datetime.now()
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        metrics.extend([
            PerformanceMetric("cpu_usage_percent", cpu_percent, "%", timestamp),
            PerformanceMetric("cpu_count", cpu_count, "cores", timestamp),
            PerformanceMetric("cpu_frequency_mhz", cpu_freq.current if cpu_freq else 0, "MHz", timestamp)
        ])
        
        # Mémoire
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        metrics.extend([
            PerformanceMetric("memory_usage_percent", memory.percent, "%", timestamp),
            PerformanceMetric("memory_used_bytes", memory.used, "bytes", timestamp),
            PerformanceMetric("memory_available_bytes", memory.available, "bytes", timestamp),
            PerformanceMetric("swap_usage_percent", swap.percent, "%", timestamp),
            PerformanceMetric("swap_used_bytes", swap.used, "bytes", timestamp)
        ])
        
        # Disque
        for partition in psutil.disk_partitions():
            try:
                disk_usage = psutil.disk_usage(partition.mountpoint)
                device_name = partition.device.replace('/', '_')
                
                metrics.extend([
                    PerformanceMetric(
                        f"disk_usage_percent", 
                        disk_usage.percent, 
                        "%", 
                        timestamp,
                        labels={"device": device_name, "mountpoint": partition.mountpoint}
                    ),
                    PerformanceMetric(
                        f"disk_free_bytes", 
                        disk_usage.free, 
                        "bytes", 
                        timestamp,
                        labels={"device": device_name}
                    )
                ])
            except PermissionError:
                continue
        
        # I/O Disque
        disk_io = psutil.disk_io_counters()
        if disk_io:
            metrics.extend([
                PerformanceMetric("disk_read_bytes", disk_io.read_bytes, "bytes", timestamp),
                PerformanceMetric("disk_write_bytes", disk_io.write_bytes, "bytes", timestamp),
                PerformanceMetric("disk_read_count", disk_io.read_count, "ops", timestamp),
                PerformanceMetric("disk_write_count", disk_io.write_count, "ops", timestamp)
            ])
        
        return metrics
    
    async def _collect_network_metrics(self) -> List[PerformanceMetric]:
        """Collecte les métriques réseau"""
        metrics = []
        timestamp = datetime.now()
        
        # I/O Réseau
        net_io = psutil.net_io_counters()
        if net_io:
            metrics.extend([
                PerformanceMetric("network_bytes_sent", net_io.bytes_sent, "bytes", timestamp),
                PerformanceMetric("network_bytes_recv", net_io.bytes_recv, "bytes", timestamp),
                PerformanceMetric("network_packets_sent", net_io.packets_sent, "packets", timestamp),
                PerformanceMetric("network_packets_recv", net_io.packets_recv, "packets", timestamp),
                PerformanceMetric("network_errors_in", net_io.errin, "errors", timestamp),
                PerformanceMetric("network_errors_out", net_io.errout, "errors", timestamp)
            ])
        
        # Connexions réseau
        connections = psutil.net_connections()
        conn_status = defaultdict(int)
        for conn in connections:
            conn_status[conn.status] += 1
        
        for status, count in conn_status.items():
            metrics.append(
                PerformanceMetric(
                    "network_connections", 
                    count, 
                    "connections", 
                    timestamp,
                    labels={"status": status}
                )
            )
        
        return metrics
    
    async def _collect_process_metrics(self) -> List[PerformanceMetric]:
        """Collecte les métriques des processus"""
        metrics = []
        timestamp = datetime.now()
        
        # Processus courant
        current_process = psutil.Process()
        
        metrics.extend([
            PerformanceMetric("process_cpu_percent", current_process.cpu_percent(), "%", timestamp),
            PerformanceMetric("process_memory_percent", current_process.memory_percent(), "%", timestamp),
            PerformanceMetric("process_memory_rss", current_process.memory_info().rss, "bytes", timestamp),
            PerformanceMetric("process_memory_vms", current_process.memory_info().vms, "bytes", timestamp),
            PerformanceMetric("process_num_threads", current_process.num_threads(), "threads", timestamp),
            PerformanceMetric("process_num_fds", current_process.num_fds() if hasattr(current_process, 'num_fds') else 0, "fds", timestamp)
        ])
        
        # Processus système
        total_processes = len(psutil.pids())
        running_processes = len([p for p in psutil.process_iter() if p.status() == psutil.STATUS_RUNNING])
        
        metrics.extend([
            PerformanceMetric("system_total_processes", total_processes, "processes", timestamp),
            PerformanceMetric("system_running_processes", running_processes, "processes", timestamp)
        ])
        
        return metrics
    
    async def _collect_docker_metrics(self) -> List[PerformanceMetric]:
        """Collecte les métriques Docker"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            docker_client = docker.from_env()
            containers = docker_client.containers.list()
            
            for container in containers:
                container_name = container.name
                stats = container.stats(stream=False)
                
                # CPU
                cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
                system_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
                cpu_percent = (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100.0
                
                # Mémoire
                memory_usage = stats['memory_stats']['usage']
                memory_limit = stats['memory_stats']['limit']
                memory_percent = (memory_usage / memory_limit) * 100.0
                
                # Réseau
                network_rx = sum(net['rx_bytes'] for net in stats['networks'].values())
                network_tx = sum(net['tx_bytes'] for net in stats['networks'].values())
                
                metrics.extend([
                    PerformanceMetric(
                        "container_cpu_percent", 
                        cpu_percent, 
                        "%", 
                        timestamp,
                        labels={"container": container_name}
                    ),
                    PerformanceMetric(
                        "container_memory_percent", 
                        memory_percent, 
                        "%", 
                        timestamp,
                        labels={"container": container_name}
                    ),
                    PerformanceMetric(
                        "container_memory_usage", 
                        memory_usage, 
                        "bytes", 
                        timestamp,
                        labels={"container": container_name}
                    ),
                    PerformanceMetric(
                        "container_network_rx", 
                        network_rx, 
                        "bytes", 
                        timestamp,
                        labels={"container": container_name}
                    ),
                    PerformanceMetric(
                        "container_network_tx", 
                        network_tx, 
                        "bytes", 
                        timestamp,
                        labels={"container": container_name}
                    )
                ])
        
        except Exception as e:
            logger.debug(f"Docker non disponible: {e}")
        
        return metrics
    
    async def _store_metric_in_redis(self, metric: PerformanceMetric):
        """Stocke une métrique dans Redis"""
        if not self.redis_client:
            return
        
        try:
            metric_data = {
                'value': metric.value,
                'unit': metric.unit,
                'timestamp': metric.timestamp.isoformat(),
                'labels': json.dumps(metric.labels)
            }
            
            key = f"metrics:{metric.name}:{metric.timestamp.strftime('%Y%m%d%H%M')}"
            await self.redis_client.hmset(key, metric_data)
            await self.redis_client.expire(key, 86400)  # 24 heures
        
        except Exception as e:
            logger.warning(f"Erreur stockage Redis: {e}")
    
    async def _update_prometheus_metrics(self, metrics: List[PerformanceMetric]):
        """Met à jour les métriques Prometheus"""
        for metric in metrics:
            try:
                if metric.name == "cpu_usage_percent":
                    self.cpu_usage.set(metric.value)
                elif metric.name == "memory_usage_percent":
                    self.memory_usage.set(metric.value)
                elif metric.name == "disk_usage_percent" and "device" in metric.labels:
                    self.disk_usage.labels(device=metric.labels["device"]).set(metric.value)
                elif metric.name in ["network_bytes_sent", "network_bytes_recv"]:
                    direction = "sent" if "sent" in metric.name else "recv"
                    self.network_io.labels(direction=direction).inc(metric.value)
            except Exception as e:
                logger.debug(f"Erreur Prometheus pour {metric.name}: {e}")
    
    def get_prometheus_metrics(self) -> str:
        """Retourne les métriques au format Prometheus"""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_metric_history(self, metric_name: str, duration_minutes: int = 60) -> List[PerformanceMetric]:
        """Récupère l'historique d'une métrique"""
        if metric_name not in self.metrics_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        return [m for m in self.metrics_history[metric_name] if m.timestamp >= cutoff_time]

class PerformanceAnalyzer:
    """Analyseur de performance avancé"""
    
    def __init__(self, metrics_collector: SystemMetricsCollector):
        self.metrics_collector = metrics_collector
        self.performance_thresholds = {
            'cpu_usage_percent': {'critical': 90, 'high': 80, 'medium': 70, 'low': 60},
            'memory_usage_percent': {'critical': 95, 'high': 85, 'medium': 75, 'low': 65},
            'disk_usage_percent': {'critical': 95, 'high': 90, 'medium': 80, 'low': 70},
            'response_time_ms': {'critical': 5000, 'high': 2000, 'medium': 1000, 'low': 500},
            'error_rate_percent': {'critical': 10, 'high': 5, 'medium': 2, 'low': 1}
        }
        
        self.trend_window = 20  # Nombre de points pour analyse de tendance
        
    async def analyze_performance(self, duration_minutes: int = 30) -> Dict[str, Any]:
        """Analyse globale des performances"""
        analysis_result = {
            'timestamp': datetime.now().isoformat(),
            'duration_minutes': duration_minutes,
            'overall_health': 'unknown',
            'alerts': [],
            'recommendations': [],
            'metrics_summary': {},
            'trends': {},
            'predictions': {}
        }
        
        # Analyse par type de métrique
        metric_types = ['cpu_usage_percent', 'memory_usage_percent', 'disk_usage_percent']
        
        for metric_name in metric_types:
            metric_analysis = await self._analyze_single_metric(metric_name, duration_minutes)
            analysis_result['metrics_summary'][metric_name] = metric_analysis
            
            # Générer des alertes si nécessaire
            alerts = self._generate_alerts_for_metric(metric_name, metric_analysis)
            analysis_result['alerts'].extend(alerts)
            
            # Analyser les tendances
            trend_analysis = await self._analyze_trend(metric_name, duration_minutes)
            analysis_result['trends'][metric_name] = trend_analysis
        
        # Santé globale du système
        analysis_result['overall_health'] = self._calculate_overall_health(analysis_result['alerts'])
        
        # Recommandations générales
        analysis_result['recommendations'] = self._generate_general_recommendations(analysis_result)
        
        # Prédictions
        analysis_result['predictions'] = await self._generate_predictions(duration_minutes)
        
        return analysis_result
    
    async def _analyze_single_metric(self, metric_name: str, duration_minutes: int) -> Dict[str, Any]:
        """Analyse une métrique spécifique"""
        history = self.metrics_collector.get_metric_history(metric_name, duration_minutes)
        
        if not history:
            return {'status': 'no_data', 'message': 'Aucune donnée disponible'}
        
        values = [m.value for m in history]
        timestamps = [m.timestamp for m in history]
        
        analysis = {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'percentiles': {
                '50': np.percentile(values, 50),
                '75': np.percentile(values, 75),
                '90': np.percentile(values, 90),
                '95': np.percentile(values, 95),
                '99': np.percentile(values, 99)
            },
            'current_value': values[-1] if values else 0,
            'threshold_status': self._evaluate_threshold(metric_name, values[-1] if values else 0),
            'stability': self._calculate_stability(values),
            'anomalies': self._detect_metric_anomalies(values)
        }
        
        return analysis
    
    def _evaluate_threshold(self, metric_name: str, value: float) -> str:
        """Évalue le seuil pour une valeur donnée"""
        if metric_name not in self.performance_thresholds:
            return 'unknown'
        
        thresholds = self.performance_thresholds[metric_name]
        
        if value >= thresholds['critical']:
            return 'critical'
        elif value >= thresholds['high']:
            return 'high'
        elif value >= thresholds['medium']:
            return 'medium'
        elif value >= thresholds['low']:
            return 'low'
        else:
            return 'excellent'
    
    def _calculate_stability(self, values: List[float]) -> Dict[str, float]:
        """Calcule la stabilité d'une métrique"""
        if len(values) < 2:
            return {'score': 1.0, 'coefficient_variation': 0.0}
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Coefficient de variation
        cv = std_val / mean_val if mean_val != 0 else 0
        
        # Score de stabilité (plus c'est proche de 1, plus c'est stable)
        stability_score = max(0, 1 - cv)
        
        return {
            'score': stability_score,
            'coefficient_variation': cv,
            'interpretation': self._interpret_stability(stability_score)
        }
    
    def _interpret_stability(self, score: float) -> str:
        """Interprète le score de stabilité"""
        if score >= 0.9:
            return 'très stable'
        elif score >= 0.7:
            return 'stable'
        elif score >= 0.5:
            return 'modérément stable'
        elif score >= 0.3:
            return 'instable'
        else:
            return 'très instable'
    
    def _detect_metric_anomalies(self, values: List[float]) -> List[Dict[str, Any]]:
        """Détecte les anomalies dans une série de valeurs"""
        if len(values) < 10:
            return []
        
        anomalies = []
        
        # Z-Score pour détecter les outliers
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        for i, value in enumerate(values):
            if std_val > 0:
                z_score = abs(value - mean_val) / std_val
                if z_score > 3:  # Seuil standard pour outliers
                    anomalies.append({
                        'index': i,
                        'value': value,
                        'z_score': z_score,
                        'type': 'outlier'
                    })
        
        # Détection de pics soudains
        if len(values) >= 3:
            for i in range(1, len(values) - 1):
                prev_val = values[i-1]
                curr_val = values[i]
                next_val = values[i+1]
                
                # Pic vers le haut
                if curr_val > prev_val * 2 and curr_val > next_val * 2:
                    anomalies.append({
                        'index': i,
                        'value': curr_val,
                        'type': 'spike_up'
                    })
                
                # Pic vers le bas
                elif curr_val < prev_val * 0.5 and curr_val < next_val * 0.5:
                    anomalies.append({
                        'index': i,
                        'value': curr_val,
                        'type': 'spike_down'
                    })
        
        return anomalies
    
    def _generate_alerts_for_metric(self, metric_name: str, analysis: Dict[str, Any]) -> List[PerformanceAlert]:
        """Génère des alertes pour une métrique"""
        alerts = []
        
        current_value = analysis.get('current_value', 0)
        threshold_status = analysis.get('threshold_status', 'unknown')
        
        if threshold_status in ['critical', 'high']:
            severity = threshold_status
            threshold_value = self.performance_thresholds.get(metric_name, {}).get(severity, 0)
            
            alert = PerformanceAlert(
                metric_name=metric_name,
                current_value=current_value,
                threshold_value=threshold_value,
                severity=severity,
                message=f"{metric_name} a atteint un niveau {severity}: {current_value}",
                recommendations=self._get_metric_recommendations(metric_name, threshold_status),
                timestamp=datetime.now(),
                context={
                    'analysis': analysis,
                    'threshold_status': threshold_status
                }
            )
            alerts.append(alert)
        
        # Alertes pour instabilité
        stability = analysis.get('stability', {})
        if stability.get('score', 1.0) < 0.5:
            alert = PerformanceAlert(
                metric_name=metric_name,
                current_value=current_value,
                threshold_value=0.5,
                severity='medium',
                message=f"{metric_name} présente une instabilité: {stability.get('interpretation', 'inconnue')}",
                recommendations=[f"Investiguer les causes d'instabilité de {metric_name}"],
                timestamp=datetime.now(),
                context={'stability': stability}
            )
            alerts.append(alert)
        
        return alerts
    
    def _get_metric_recommendations(self, metric_name: str, status: str) -> List[str]:
        """Obtient des recommandations spécifiques pour une métrique"""
        recommendations = {
            'cpu_usage_percent': {
                'critical': [
                    "Identifier les processus consommant le plus de CPU",
                    "Optimiser les algorithmes ou ajouter des ressources",
                    "Considérer la répartition de charge"
                ],
                'high': [
                    "Surveiller l'évolution de la charge CPU",
                    "Planifier une optimisation des performances"
                ]
            },
            'memory_usage_percent': {
                'critical': [
                    "Vérifier les fuites mémoire",
                    "Optimiser l'utilisation de la mémoire",
                    "Augmenter la RAM disponible"
                ],
                'high': [
                    "Analyser l'allocation mémoire",
                    "Nettoyer les caches inutiles"
                ]
            },
            'disk_usage_percent': {
                'critical': [
                    "Libérer de l'espace disque immédiatement",
                    "Archiver ou supprimer les anciens fichiers",
                    "Ajouter des disques supplémentaires"
                ],
                'high': [
                    "Planifier le nettoyage des disques",
                    "Analyser l'utilisation de l'espace"
                ]
            }
        }
        
        return recommendations.get(metric_name, {}).get(status, ["Surveiller la métrique"])
    
    async def _analyze_trend(self, metric_name: str, duration_minutes: int) -> Dict[str, Any]:
        """Analyse la tendance d'une métrique"""
        history = self.metrics_collector.get_metric_history(metric_name, duration_minutes)
        
        if len(history) < self.trend_window:
            return {'status': 'insufficient_data'}
        
        # Prendre les derniers points pour l'analyse
        recent_values = [m.value for m in history[-self.trend_window:]]
        
        # Régression linéaire simple
        x = np.arange(len(recent_values))
        z = np.polyfit(x, recent_values, 1)
        slope = z[0]
        
        # Déterminer la direction de la tendance
        if slope > 0.1:
            direction = 'increasing'
        elif slope < -0.1:
            direction = 'decreasing'
        else:
            direction = 'stable'
        
        # Calculer la force de la tendance
        correlation = np.corrcoef(x, recent_values)[0, 1]
        trend_strength = abs(correlation)
        
        return {
            'direction': direction,
            'slope': slope,
            'strength': trend_strength,
            'strength_interpretation': self._interpret_trend_strength(trend_strength),
            'prediction_next_hour': recent_values[-1] + slope * 60  # Projection 1h
        }
    
    def _interpret_trend_strength(self, strength: float) -> str:
        """Interprète la force de la tendance"""
        if strength >= 0.8:
            return 'très forte'
        elif strength >= 0.6:
            return 'forte'
        elif strength >= 0.4:
            return 'modérée'
        elif strength >= 0.2:
            return 'faible'
        else:
            return 'très faible'
    
    def _calculate_overall_health(self, alerts: List[PerformanceAlert]) -> str:
        """Calcule la santé globale du système"""
        if not alerts:
            return 'excellent'
        
        critical_count = sum(1 for alert in alerts if alert.severity == 'critical')
        high_count = sum(1 for alert in alerts if alert.severity == 'high')
        
        if critical_count > 0:
            return 'critical'
        elif high_count > 2:
            return 'poor'
        elif high_count > 0:
            return 'fair'
        else:
            return 'good'
    
    def _generate_general_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Génère des recommandations générales"""
        recommendations = []
        
        overall_health = analysis['overall_health']
        alerts_count = len(analysis['alerts'])
        
        if overall_health == 'critical':
            recommendations.extend([
                "Action immédiate requise - Systèmes critiques",
                "Escalader vers l'équipe d'infrastructure",
                "Considérer la mise en maintenance"
            ])
        elif overall_health == 'poor':
            recommendations.extend([
                "Planifier des optimisations urgentes",
                "Surveiller étroitement les métriques",
                "Préparer un plan de récupération"
            ])
        elif alerts_count > 0:
            recommendations.append("Investiguer les alertes actives")
        
        # Recommandations basées sur les tendances
        for metric_name, trend in analysis.get('trends', {}).items():
            if trend.get('direction') == 'increasing' and trend.get('strength', 0) > 0.6:
                recommendations.append(f"Surveiller la tendance croissante de {metric_name}")
        
        return recommendations or ["Système en bon état - Maintenir la surveillance"]
    
    async def _generate_predictions(self, duration_minutes: int) -> Dict[str, Any]:
        """Génère des prédictions de performance"""
        predictions = {}
        
        metric_types = ['cpu_usage_percent', 'memory_usage_percent', 'disk_usage_percent']
        
        for metric_name in metric_types:
            trend = await self._analyze_trend(metric_name, duration_minutes)
            
            if trend.get('status') != 'insufficient_data':
                current_history = self.metrics_collector.get_metric_history(metric_name, duration_minutes)
                if current_history:
                    current_value = current_history[-1].value
                    slope = trend.get('slope', 0)
                    
                    predictions[metric_name] = {
                        'current_value': current_value,
                        'predicted_1h': max(0, current_value + slope * 60),
                        'predicted_24h': max(0, current_value + slope * 60 * 24),
                        'trend_direction': trend.get('direction', 'stable'),
                        'confidence': trend.get('strength', 0)
                    }
        
        return predictions
