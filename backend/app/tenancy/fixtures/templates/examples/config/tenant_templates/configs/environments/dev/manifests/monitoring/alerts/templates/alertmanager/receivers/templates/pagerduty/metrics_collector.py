"""
Advanced Metrics Collector for PagerDuty Integration

Ce module fournit une collecte de métriques sophistiquée, un monitoring temps réel,
et des analyses de performance pour l'écosystème PagerDuty.

Fonctionnalités:
- Collecte de métriques temps réel avec Prometheus
- KPI et SLA tracking automatisé
- Alerting proactif basé sur les tendances
- Dashboards interactifs avec Grafana
- Analyse de performance avec APM
- Rapports automatisés et insights IA

Version: 4.0.0
Développé par l'équipe Spotify AI Agent
"""

import asyncio
import json
import time
import statistics
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import structlog
import aiofiles
import aioredis
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary
import psutil
import numpy as np
import pandas as pd

from . import (
    IncidentData, IncidentStatus, IncidentSeverity, 
    MetricsData, logger
)

# ============================================================================
# Configuration des Métriques
# ============================================================================

@dataclass
class MetricsConfig:
    """Configuration du collecteur de métriques"""
    collection_interval: int = 60  # secondes
    retention_period: int = 7 * 24 * 3600  # 7 jours en secondes
    aggregation_windows: List[int] = field(default_factory=lambda: [300, 900, 3600])  # 5m, 15m, 1h
    enable_system_metrics: bool = True
    enable_business_metrics: bool = True
    enable_sla_tracking: bool = True
    prometheus_port: int = 8000
    redis_key_prefix: str = "pagerduty:metrics"

class MetricType(Enum):
    """Types de métriques"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class SLAType(Enum):
    """Types de SLA trackés"""
    INCIDENT_RESPONSE = "incident_response"
    INCIDENT_RESOLUTION = "incident_resolution"
    API_AVAILABILITY = "api_availability"
    NOTIFICATION_DELIVERY = "notification_delivery"

# ============================================================================
# Métriques Prometheus
# ============================================================================

# Métriques d'incidents
INCIDENT_TOTAL = Counter(
    'pagerduty_incidents_total',
    'Total number of incidents',
    ['severity', 'status', 'service', 'source']
)

INCIDENT_DURATION = Histogram(
    'pagerduty_incident_duration_seconds',
    'Incident duration from creation to resolution',
    ['severity', 'service'],
    buckets=[60, 300, 900, 1800, 3600, 7200, 14400]  # 1m à 4h
)

INCIDENT_RESPONSE_TIME = Histogram(
    'pagerduty_incident_response_time_seconds',
    'Time from incident creation to acknowledgment',
    ['severity', 'urgency'],
    buckets=[30, 60, 180, 300, 600, 1200]  # 30s à 20m
)

# Métriques API
API_REQUESTS_TOTAL = Counter(
    'pagerduty_api_requests_total',
    'Total API requests to PagerDuty',
    ['method', 'endpoint', 'status_code']
)

API_REQUEST_DURATION = Histogram(
    'pagerduty_api_request_duration_seconds',
    'API request duration',
    ['endpoint'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

API_RATE_LIMIT = Gauge(
    'pagerduty_api_rate_limit_remaining',
    'Remaining API rate limit'
)

# Métriques de notifications
NOTIFICATION_TOTAL = Counter(
    'pagerduty_notifications_total',
    'Total notifications sent',
    ['channel', 'status']
)

NOTIFICATION_DELIVERY_TIME = Histogram(
    'pagerduty_notification_delivery_seconds',
    'Notification delivery time',
    ['channel'],
    buckets=[1, 5, 10, 30, 60, 300]
)

# Métriques de performance système
SYSTEM_CPU_USAGE = Gauge(
    'pagerduty_system_cpu_usage_percent',
    'System CPU usage percentage'
)

SYSTEM_MEMORY_USAGE = Gauge(
    'pagerduty_system_memory_usage_percent',
    'System memory usage percentage'
)

CACHE_HIT_RATE = Gauge(
    'pagerduty_cache_hit_rate',
    'Cache hit rate percentage'
)

# Métriques IA
AI_PREDICTION_ACCURACY = Gauge(
    'pagerduty_ai_prediction_accuracy',
    'AI prediction accuracy',
    ['model_type']
)

AI_PROCESSING_TIME = Histogram(
    'pagerduty_ai_processing_seconds',
    'AI processing time',
    ['operation'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# ============================================================================
# Collecteur de Métriques Principal
# ============================================================================

class MetricsCollector:
    """Collecteur de métriques avancé pour PagerDuty"""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.redis_pool = None
        self.running = False
        self.collection_task = None
        self.metrics_buffer = defaultdict(deque)
        self.sla_tracker = SLATracker(config)
        self.performance_monitor = PerformanceMonitor()
        
    async def initialize(self, redis_url: str):
        """Initialise le collecteur de métriques"""
        try:
            # Connexion Redis
            self.redis_pool = aioredis.ConnectionPool.from_url(redis_url)
            
            # Démarrage du serveur Prometheus
            prometheus_client.start_http_server(self.config.prometheus_port)
            
            # Initialisation du tracker SLA
            await self.sla_tracker.initialize(redis_url)
            
            logger.info(
                "Metrics collector initialized",
                prometheus_port=self.config.prometheus_port
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize metrics collector: {e}")
            raise
            
    async def start_collection(self):
        """Démarre la collecte de métriques"""
        if self.running:
            return
            
        self.running = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Metrics collection started")
        
    async def stop_collection(self):
        """Arrête la collecte de métriques"""
        if not self.running:
            return
            
        self.running = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Metrics collection stopped")
        
    async def _collection_loop(self):
        """Boucle principale de collecte"""
        while self.running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.config.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(5)  # Attente avant retry
                
    async def _collect_metrics(self):
        """Collecte toutes les métriques"""
        timestamp = datetime.now(timezone.utc)
        
        # Métriques système
        if self.config.enable_system_metrics:
            await self._collect_system_metrics(timestamp)
            
        # Métriques business
        if self.config.enable_business_metrics:
            await self._collect_business_metrics(timestamp)
            
        # Métriques SLA
        if self.config.enable_sla_tracking:
            await self._collect_sla_metrics(timestamp)
            
        # Métriques de performance
        await self._collect_performance_metrics(timestamp)
        
        # Sauvegarde en Redis
        await self._persist_metrics(timestamp)
        
    async def _collect_system_metrics(self, timestamp: datetime):
        """Collecte les métriques système"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            SYSTEM_CPU_USAGE.set(cpu_percent)
            
            # Mémoire
            memory = psutil.virtual_memory()
            SYSTEM_MEMORY_USAGE.set(memory.percent)
            
            # Stockage des métriques brutes
            self.metrics_buffer['system_cpu'].append((timestamp, cpu_percent))
            self.metrics_buffer['system_memory'].append((timestamp, memory.percent))
            
        except Exception as e:
            logger.warning(f"System metrics collection failed: {e}")
            
    async def _collect_business_metrics(self, timestamp: datetime):
        """Collecte les métriques business"""
        try:
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                # Compteurs d'incidents par statut
                for status in IncidentStatus:
                    count = await redis.get(f"incidents:count:{status.value}")
                    if count:
                        self.metrics_buffer[f'incidents_{status.value}'].append((timestamp, int(count)))
                        
                # Temps de réponse moyen
                avg_response_time = await redis.get("incidents:avg_response_time")
                if avg_response_time:
                    self.metrics_buffer['avg_response_time'].append((timestamp, float(avg_response_time)))
                    
                # Taux de résolution automatique
                auto_resolution_rate = await redis.get("incidents:auto_resolution_rate")
                if auto_resolution_rate:
                    self.metrics_buffer['auto_resolution_rate'].append((timestamp, float(auto_resolution_rate)))
                    
        except Exception as e:
            logger.warning(f"Business metrics collection failed: {e}")
            
    async def _collect_sla_metrics(self, timestamp: datetime):
        """Collecte les métriques SLA"""
        try:
            sla_data = await self.sla_tracker.get_current_sla_status()
            
            for sla_type, metrics in sla_data.items():
                self.metrics_buffer[f'sla_{sla_type}'].append((timestamp, metrics))
                
        except Exception as e:
            logger.warning(f"SLA metrics collection failed: {e}")
            
    async def _collect_performance_metrics(self, timestamp: datetime):
        """Collecte les métriques de performance"""
        try:
            perf_data = await self.performance_monitor.get_performance_snapshot()
            
            for metric_name, value in perf_data.items():
                self.metrics_buffer[f'performance_{metric_name}'].append((timestamp, value))
                
        except Exception as e:
            logger.warning(f"Performance metrics collection failed: {e}")
            
    async def _persist_metrics(self, timestamp: datetime):
        """Persiste les métriques en Redis"""
        try:
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                pipe = redis.pipeline()
                
                for metric_name, data_points in self.metrics_buffer.items():
                    # Limite la taille du buffer
                    while len(data_points) > 1000:
                        data_points.popleft()
                        
                    # Sauvegarde en Redis avec TTL
                    key = f"{self.config.redis_key_prefix}:{metric_name}"
                    value = json.dumps([(ts.isoformat(), value) for ts, value in data_points])
                    pipe.setex(key, self.config.retention_period, value)
                    
                await pipe.execute()
                
        except Exception as e:
            logger.warning(f"Metrics persistence failed: {e}")
            
    # ============================================================================
    # Méthodes d'enregistrement des événements
    # ============================================================================
    
    def record_incident_created(self, incident: IncidentData):
        """Enregistre la création d'un incident"""
        INCIDENT_TOTAL.labels(
            severity=incident.severity.value,
            status=incident.status.value,
            service=incident.service_id,
            source=incident.source.value
        ).inc()
        
    def record_incident_acknowledged(self, incident: IncidentData, response_time: float):
        """Enregistre l'acknowledgment d'un incident"""
        INCIDENT_RESPONSE_TIME.labels(
            severity=incident.severity.value,
            urgency=incident.urgency.value
        ).observe(response_time)
        
    def record_incident_resolved(self, incident: IncidentData, duration: float):
        """Enregistre la résolution d'un incident"""
        INCIDENT_DURATION.labels(
            severity=incident.severity.value,
            service=incident.service_id
        ).observe(duration)
        
    def record_api_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Enregistre une requête API"""
        API_REQUESTS_TOTAL.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        API_REQUEST_DURATION.labels(endpoint=endpoint).observe(duration)
        
    def record_notification_sent(self, channel: str, status: str, delivery_time: float):
        """Enregistre l'envoi d'une notification"""
        NOTIFICATION_TOTAL.labels(channel=channel, status=status).inc()
        NOTIFICATION_DELIVERY_TIME.labels(channel=channel).observe(delivery_time)
        
    def update_rate_limit(self, remaining: int):
        """Met à jour le rate limit restant"""
        API_RATE_LIMIT.set(remaining)
        
    def update_cache_hit_rate(self, rate: float):
        """Met à jour le taux de cache hit"""
        CACHE_HIT_RATE.set(rate)
        
    def record_ai_prediction(self, model_type: str, accuracy: float, processing_time: float):
        """Enregistre une prédiction IA"""
        AI_PREDICTION_ACCURACY.labels(model_type=model_type).set(accuracy)
        AI_PROCESSING_TIME.labels(operation=model_type).observe(processing_time)
        
    # ============================================================================
    # Méthodes d'analyse et de reporting
    # ============================================================================
    
    async def get_metrics_summary(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Récupère un résumé des métriques pour une période"""
        try:
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                summary = {}
                
                # Récupération des métriques principales
                for metric_name in ['incidents_resolved', 'avg_response_time', 'auto_resolution_rate']:
                    key = f"{self.config.redis_key_prefix}:{metric_name}"
                    data = await redis.get(key)
                    
                    if data:
                        points = json.loads(data)
                        filtered_points = [
                            (ts, value) for ts, value in points
                            if start_time <= datetime.fromisoformat(ts) <= end_time
                        ]
                        
                        if filtered_points:
                            values = [value for _, value in filtered_points]
                            summary[metric_name] = {
                                'min': min(values),
                                'max': max(values),
                                'avg': sum(values) / len(values),
                                'count': len(values)
                            }
                            
                return summary
                
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {}
            
    async def generate_daily_report(self, date: datetime) -> Dict[str, Any]:
        """Génère un rapport quotidien"""
        start_time = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(days=1)
        
        report = {
            'date': date.isoformat(),
            'summary': await self.get_metrics_summary(start_time, end_time),
            'sla_status': await self.sla_tracker.get_daily_sla_report(date),
            'performance': await self.performance_monitor.get_daily_performance_report(date),
            'trends': await self._analyze_trends(start_time, end_time)
        }
        
        return report
        
    async def _analyze_trends(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Analyse les tendances pour une période"""
        trends = {}
        
        try:
            # Calcul des tendances pour les métriques clés
            metrics_to_analyze = ['incidents_total', 'avg_response_time', 'auto_resolution_rate']
            
            for metric in metrics_to_analyze:
                data = await self._get_metric_data(metric, start_time, end_time)
                if len(data) > 1:
                    values = [point[1] for point in data]
                    trend = self._calculate_trend(values)
                    trends[metric] = {
                        'direction': trend['direction'],
                        'slope': trend['slope'],
                        'correlation': trend['correlation']
                    }
                    
        except Exception as e:
            logger.warning(f"Trend analysis failed: {e}")
            
        return trends
        
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calcule la tendance d'une série de valeurs"""
        if len(values) < 2:
            return {'direction': 'stable', 'slope': 0, 'correlation': 0}
            
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        correlation = np.corrcoef(x, values)[0, 1]
        
        if slope > 0.1:
            direction = 'increasing'
        elif slope < -0.1:
            direction = 'decreasing'
        else:
            direction = 'stable'
            
        return {
            'direction': direction,
            'slope': float(slope),
            'correlation': float(correlation)
        }
        
    async def _get_metric_data(self, metric_name: str, start_time: datetime, end_time: datetime) -> List[Tuple[datetime, float]]:
        """Récupère les données d'une métrique pour une période"""
        try:
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                key = f"{self.config.redis_key_prefix}:{metric_name}"
                data = await redis.get(key)
                
                if data:
                    points = json.loads(data)
                    return [
                        (datetime.fromisoformat(ts), value)
                        for ts, value in points
                        if start_time <= datetime.fromisoformat(ts) <= end_time
                    ]
                    
        except Exception as e:
            logger.warning(f"Failed to get metric data for {metric_name}: {e}")
            
        return []

# ============================================================================
# Tracker SLA
# ============================================================================

class SLATracker:
    """Tracker pour les SLA et KPI"""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.redis_pool = None
        self.sla_targets = {
            SLAType.INCIDENT_RESPONSE: 300,  # 5 minutes
            SLAType.INCIDENT_RESOLUTION: 1800,  # 30 minutes
            SLAType.API_AVAILABILITY: 99.9,  # 99.9%
            SLAType.NOTIFICATION_DELIVERY: 60  # 1 minute
        }
        
    async def initialize(self, redis_url: str):
        """Initialise le tracker SLA"""
        self.redis_pool = aioredis.ConnectionPool.from_url(redis_url)
        
    async def track_incident_sla(self, incident: IncidentData, event_type: str, timestamp: datetime):
        """Track les SLA pour un incident"""
        try:
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                key = f"sla:incident:{incident.id}:{event_type}"
                await redis.setex(key, 86400, timestamp.isoformat())  # 24h TTL
                
                # Calcul du SLA si c'est une résolution
                if event_type == 'resolved':
                    await self._calculate_incident_sla(incident, redis)
                    
        except Exception as e:
            logger.warning(f"SLA tracking failed: {e}")
            
    async def _calculate_incident_sla(self, incident: IncidentData, redis):
        """Calcule les SLA pour un incident résolu"""
        try:
            # Récupération des timestamps
            created_key = f"sla:incident:{incident.id}:created"
            acknowledged_key = f"sla:incident:{incident.id}:acknowledged"
            resolved_key = f"sla:incident:{incident.id}:resolved"
            
            created_ts = await redis.get(created_key)
            acknowledged_ts = await redis.get(acknowledged_key)
            resolved_ts = await redis.get(resolved_key)
            
            if created_ts and resolved_ts:
                created_time = datetime.fromisoformat(created_ts)
                resolved_time = datetime.fromisoformat(resolved_ts)
                
                # Temps de résolution total
                resolution_time = (resolved_time - created_time).total_seconds()
                
                # Vérification du SLA
                sla_met = resolution_time <= self.sla_targets[SLAType.INCIDENT_RESOLUTION]
                
                # Enregistrement du résultat
                sla_key = f"sla:results:{datetime.now().date()}"
                await redis.hincrby(sla_key, 'total_incidents', 1)
                if sla_met:
                    await redis.hincrby(sla_key, 'sla_met', 1)
                    
                # TTL de 30 jours
                await redis.expire(sla_key, 30 * 86400)
                
        except Exception as e:
            logger.warning(f"SLA calculation failed: {e}")
            
    async def get_current_sla_status(self) -> Dict[str, Dict[str, float]]:
        """Récupère le statut SLA actuel"""
        try:
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                today = datetime.now().date()
                sla_key = f"sla:results:{today}"
                
                results = await redis.hgetall(sla_key)
                
                if results:
                    total = int(results.get('total_incidents', 0))
                    met = int(results.get('sla_met', 0))
                    
                    if total > 0:
                        percentage = (met / total) * 100
                        return {
                            'incident_resolution': {
                                'percentage': percentage,
                                'total_incidents': total,
                                'sla_met': met,
                                'target': 95.0  # 95% target
                            }
                        }
                        
        except Exception as e:
            logger.warning(f"SLA status retrieval failed: {e}")
            
        return {}
        
    async def get_daily_sla_report(self, date: datetime) -> Dict[str, Any]:
        """Génère un rapport SLA quotidien"""
        try:
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                sla_key = f"sla:results:{date.date()}"
                results = await redis.hgetall(sla_key)
                
                if results:
                    total = int(results.get('total_incidents', 0))
                    met = int(results.get('sla_met', 0))
                    
                    return {
                        'date': date.date().isoformat(),
                        'total_incidents': total,
                        'sla_met': met,
                        'sla_percentage': (met / total * 100) if total > 0 else 0,
                        'target_percentage': 95.0,
                        'status': 'met' if (met / total * 100) >= 95.0 else 'missed' if total > 0 else 'no_data'
                    }
                    
        except Exception as e:
            logger.warning(f"Daily SLA report failed: {e}")
            
        return {'date': date.date().isoformat(), 'status': 'no_data'}

# ============================================================================
# Monitor de Performance
# ============================================================================

class PerformanceMonitor:
    """Monitor de performance système et applicatif"""
    
    def __init__(self):
        self.performance_history = deque(maxlen=1000)
        
    async def get_performance_snapshot(self) -> Dict[str, float]:
        """Récupère un snapshot de performance"""
        try:
            # Métriques système
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Métriques réseau
            network = psutil.net_io_counters()
            
            snapshot = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available': memory.available,
                'disk_percent': (disk.used / disk.total) * 100,
                'disk_free': disk.free,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'timestamp': time.time()
            }
            
            self.performance_history.append(snapshot)
            return snapshot
            
        except Exception as e:
            logger.warning(f"Performance snapshot failed: {e}")
            return {}
            
    async def get_daily_performance_report(self, date: datetime) -> Dict[str, Any]:
        """Génère un rapport de performance quotidien"""
        # Filtrage des données pour la date
        start_timestamp = date.timestamp()
        end_timestamp = (date + timedelta(days=1)).timestamp()
        
        daily_data = [
            snapshot for snapshot in self.performance_history
            if start_timestamp <= snapshot.get('timestamp', 0) <= end_timestamp
        ]
        
        if not daily_data:
            return {'date': date.date().isoformat(), 'status': 'no_data'}
            
        # Calculs statistiques
        cpu_values = [d['cpu_percent'] for d in daily_data if 'cpu_percent' in d]
        memory_values = [d['memory_percent'] for d in daily_data if 'memory_percent' in d]
        
        report = {
            'date': date.date().isoformat(),
            'cpu': {
                'avg': statistics.mean(cpu_values) if cpu_values else 0,
                'max': max(cpu_values) if cpu_values else 0,
                'min': min(cpu_values) if cpu_values else 0
            },
            'memory': {
                'avg': statistics.mean(memory_values) if memory_values else 0,
                'max': max(memory_values) if memory_values else 0,
                'min': min(memory_values) if memory_values else 0
            },
            'samples_count': len(daily_data)
        }
        
        return report

# ============================================================================
# Interface Publique
# ============================================================================

__all__ = [
    'MetricsCollector',
    'MetricsConfig',
    'SLATracker',
    'PerformanceMonitor',
    'MetricType',
    'SLAType'
]
