"""
Core Analytics Engine - Moteur Principal d'Analytics
===================================================

Ce module contient le moteur principal d'analytics qui orchestre
tous les composants du système d'analytics avancé.

Composants:
- AnalyticsEngine: Orchestrateur principal
- MetricsCollector: Collecteur de métriques
- AlertManager: Gestionnaire d'alertes
- EventBus: Bus d'événements
- StateManager: Gestionnaire d'état
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
import influxdb_client
from prometheus_client import Counter, Histogram, Gauge
from elasticsearch import AsyncElasticsearch

from ..config import AnalyticsConfig
from ..models import Metric, Event, Alert, Tenant
from ..utils import Logger, Timer, RateLimiter


class EngineState(Enum):
    """États du moteur d'analytics."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class EngineMetrics:
    """Métriques du moteur d'analytics."""
    processed_metrics: int = 0
    processed_events: int = 0
    active_alerts: int = 0
    uptime_seconds: float = 0
    memory_usage_mb: float = 0
    cpu_usage_percent: float = 0
    errors_total: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'processed_metrics': self.processed_metrics,
            'processed_events': self.processed_events,
            'active_alerts': self.active_alerts,
            'uptime_seconds': self.uptime_seconds,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'errors_total': self.errors_total
        }


class EventBus:
    """Bus d'événements pour la communication inter-composants."""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.logger = Logger(__name__)
        self.subscribers: Dict[str, List[Callable]] = {}
        self.redis_client: Optional[aioredis.Redis] = None
        
    async def initialize(self):
        """Initialise le bus d'événements."""
        try:
            self.redis_client = aioredis.from_url(
                self.config.redis_url,
                decode_responses=True
            )
            await self.redis_client.ping()
            self.logger.info("Bus d'événements initialisé")
        except Exception as e:
            self.logger.error(f"Erreur initialisation bus d'événements: {e}")
            raise
    
    async def publish(self, event_type: str, data: Dict[str, Any]):
        """Publie un événement."""
        event = {
            'id': str(uuid.uuid4()),
            'type': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            'data': data
        }
        
        # Publication locale
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    await callback(event)
                except Exception as e:
                    self.logger.error(f"Erreur callback {event_type}: {e}")
        
        # Publication Redis
        if self.redis_client:
            try:
                await self.redis_client.publish(
                    f"analytics:events:{event_type}",
                    str(event)
                )
            except Exception as e:
                self.logger.error(f"Erreur publication Redis: {e}")
    
    def subscribe(self, event_type: str, callback: Callable):
        """S'abonne à un type d'événement."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    async def shutdown(self):
        """Arrête le bus d'événements."""
        if self.redis_client:
            await self.redis_client.close()


class StateManager:
    """Gestionnaire d'état du système d'analytics."""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.logger = Logger(__name__)
        self.state: Dict[str, Any] = {}
        self.redis_client: Optional[aioredis.Redis] = None
        
    async def initialize(self):
        """Initialise le gestionnaire d'état."""
        try:
            self.redis_client = aioredis.from_url(
                self.config.redis_url,
                decode_responses=True
            )
            await self.redis_client.ping()
            self.logger.info("Gestionnaire d'état initialisé")
        except Exception as e:
            self.logger.error(f"Erreur initialisation état: {e}")
            raise
    
    async def set_state(self, key: str, value: Any, ttl: Optional[int] = None):
        """Définit une valeur d'état."""
        self.state[key] = value
        
        if self.redis_client:
            try:
                if ttl:
                    await self.redis_client.setex(
                        f"analytics:state:{key}",
                        ttl,
                        str(value)
                    )
                else:
                    await self.redis_client.set(
                        f"analytics:state:{key}",
                        str(value)
                    )
            except Exception as e:
                self.logger.error(f"Erreur sauvegarde état Redis: {e}")
    
    async def get_state(self, key: str, default: Any = None) -> Any:
        """Récupère une valeur d'état."""
        # D'abord vérifier le cache local
        if key in self.state:
            return self.state[key]
        
        # Puis Redis
        if self.redis_client:
            try:
                value = await self.redis_client.get(f"analytics:state:{key}")
                if value is not None:
                    self.state[key] = value
                    return value
            except Exception as e:
                self.logger.error(f"Erreur lecture état Redis: {e}")
        
        return default
    
    async def delete_state(self, key: str):
        """Supprime une valeur d'état."""
        self.state.pop(key, None)
        
        if self.redis_client:
            try:
                await self.redis_client.delete(f"analytics:state:{key}")
            except Exception as e:
                self.logger.error(f"Erreur suppression état Redis: {e}")


class MetricsCollector:
    """Collecteur de métriques avancé."""
    
    def __init__(self, config: AnalyticsConfig, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.logger = Logger(__name__)
        self.rate_limiter = RateLimiter(max_requests=1000, window=60)
        
        # Métriques Prometheus
        self.metrics_collected = Counter(
            'analytics_metrics_collected_total',
            'Total des métriques collectées',
            ['tenant_id', 'metric_type']
        )
        
        self.collection_duration = Histogram(
            'analytics_collection_duration_seconds',
            'Durée de collecte des métriques',
            ['tenant_id', 'metric_type']
        )
        
    async def collect_metric(
        self,
        tenant_id: str,
        metric_name: str,
        value: Union[int, float],
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Collecte une métrique."""
        if not await self.rate_limiter.is_allowed(f"metric:{tenant_id}"):
            self.logger.warning(f"Rate limit dépassé pour tenant {tenant_id}")
            return False
        
        with Timer() as timer:
            try:
                metric = Metric(
                    name=metric_name,
                    value=value,
                    tenant_id=tenant_id,
                    tags=tags or {},
                    timestamp=timestamp or datetime.utcnow()
                )
                
                # Publier l'événement
                await self.event_bus.publish('metric_collected', {
                    'metric': metric.to_dict(),
                    'tenant_id': tenant_id
                })
                
                # Mettre à jour les métriques Prometheus
                self.metrics_collected.labels(
                    tenant_id=tenant_id,
                    metric_type=metric_name
                ).inc()
                
                self.collection_duration.labels(
                    tenant_id=tenant_id,
                    metric_type=metric_name
                ).observe(timer.elapsed)
                
                return True
                
            except Exception as e:
                self.logger.error(f"Erreur collecte métrique: {e}")
                return False
    
    async def collect_system_metrics(self) -> Dict[str, float]:
        """Collecte les métriques système."""
        import psutil
        
        metrics = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'network_bytes_sent': psutil.net_io_counters().bytes_sent,
            'network_bytes_recv': psutil.net_io_counters().bytes_recv,
        }
        
        # Collecter chaque métrique
        for name, value in metrics.items():
            await self.collect_metric(
                tenant_id="system",
                metric_name=name,
                value=value,
                tags={'source': 'system'}
            )
        
        return metrics
    
    async def collect_business_metrics(
        self,
        tenant_id: str,
        metrics: Dict[str, Union[int, float]]
    ) -> bool:
        """Collecte des métriques business."""
        try:
            for name, value in metrics.items():
                await self.collect_metric(
                    tenant_id=tenant_id,
                    metric_name=name,
                    value=value,
                    tags={'type': 'business'}
                )
            return True
        except Exception as e:
            self.logger.error(f"Erreur collecte métriques business: {e}")
            return False


class AlertManager:
    """Gestionnaire d'alertes intelligent."""
    
    def __init__(self, config: AnalyticsConfig, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.logger = Logger(__name__)
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self):
        """Initialise le gestionnaire d'alertes."""
        # S'abonner aux événements de métriques
        self.event_bus.subscribe('metric_collected', self._process_metric_alert)
        self.logger.info("Gestionnaire d'alertes initialisé")
    
    async def create_alert_rule(
        self,
        name: str,
        condition: str,
        severity: str,
        channels: List[str],
        cooldown: int = 300
    ):
        """Crée une règle d'alerte."""
        self.alert_rules[name] = {
            'condition': condition,
            'severity': severity,
            'channels': channels,
            'cooldown': cooldown,
            'last_triggered': None
        }
        
        self.logger.info(f"Règle d'alerte créée: {name}")
    
    async def _process_metric_alert(self, event: Dict[str, Any]):
        """Traite les alertes basées sur les métriques."""
        metric_data = event['data']['metric']
        
        for rule_name, rule in self.alert_rules.items():
            try:
                # Évaluer la condition (simplifié)
                if await self._evaluate_condition(rule['condition'], metric_data):
                    await self._trigger_alert(rule_name, rule, metric_data)
            except Exception as e:
                self.logger.error(f"Erreur évaluation alerte {rule_name}: {e}")
    
    async def _evaluate_condition(
        self,
        condition: str,
        metric_data: Dict[str, Any]
    ) -> bool:
        """Évalue une condition d'alerte."""
        # Implémentation simplifiée - à enrichir avec un parseur d'expressions
        try:
            # Remplacer les variables dans la condition
            context = {
                'value': metric_data['value'],
                'name': metric_data['name']
            }
            
            # Évaluation sécurisée (attention aux injections)
            return eval(condition, {"__builtins__": {}}, context)
        except Exception:
            return False
    
    async def _trigger_alert(
        self,
        rule_name: str,
        rule: Dict[str, Any],
        metric_data: Dict[str, Any]
    ):
        """Déclenche une alerte."""
        now = datetime.utcnow()
        
        # Vérifier le cooldown
        if rule['last_triggered']:
            if (now - rule['last_triggered']).seconds < rule['cooldown']:
                return
        
        alert = Alert(
            id=str(uuid.uuid4()),
            name=rule_name,
            severity=rule['severity'],
            message=f"Alerte {rule_name} déclenchée pour {metric_data['name']}",
            tenant_id=metric_data.get('tenant_id', 'unknown'),
            timestamp=now,
            status='active'
        )
        
        self.active_alerts[alert.id] = alert
        rule['last_triggered'] = now
        
        # Publier l'événement d'alerte
        await self.event_bus.publish('alert_triggered', {
            'alert': alert.to_dict(),
            'rule': rule_name,
            'metric': metric_data
        })
        
        self.logger.warning(f"Alerte déclenchée: {alert.name}")


class AnalyticsEngine:
    """Moteur principal d'analytics."""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.logger = Logger(__name__)
        self.state = EngineState.STOPPED
        self.start_time: Optional[datetime] = None
        self.metrics = EngineMetrics()
        
        # Composants
        self.event_bus = EventBus(config)
        self.state_manager = StateManager(config)
        self.metrics_collector = MetricsCollector(config, self.event_bus)
        self.alert_manager = AlertManager(config, self.event_bus)
        
        # Tâches background
        self.background_tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Démarre le moteur d'analytics."""
        if self.state != EngineState.STOPPED:
            raise RuntimeError(f"Moteur déjà en état {self.state.value}")
        
        self.state = EngineState.STARTING
        self.start_time = datetime.utcnow()
        
        try:
            # Initialiser les composants
            await self.event_bus.initialize()
            await self.state_manager.initialize()
            await self.alert_manager.initialize()
            
            # Démarrer les tâches background
            await self._start_background_tasks()
            
            self.state = EngineState.RUNNING
            self.logger.info("Moteur d'analytics démarré")
            
            # Publier l'événement de démarrage
            await self.event_bus.publish('engine_started', {
                'timestamp': self.start_time.isoformat()
            })
            
        except Exception as e:
            self.state = EngineState.ERROR
            self.logger.error(f"Erreur démarrage moteur: {e}")
            raise
    
    async def stop(self):
        """Arrête le moteur d'analytics."""
        if self.state not in [EngineState.RUNNING, EngineState.ERROR]:
            return
        
        self.state = EngineState.STOPPING
        
        try:
            # Arrêter les tâches background
            for task in self.background_tasks:
                task.cancel()
            
            if self.background_tasks:
                await asyncio.gather(
                    *self.background_tasks,
                    return_exceptions=True
                )
            
            # Arrêter les composants
            await self.event_bus.shutdown()
            
            self.state = EngineState.STOPPED
            self.logger.info("Moteur d'analytics arrêté")
            
        except Exception as e:
            self.state = EngineState.ERROR
            self.logger.error(f"Erreur arrêt moteur: {e}")
            raise
    
    async def _start_background_tasks(self):
        """Démarre les tâches background."""
        self.background_tasks = [
            asyncio.create_task(self._metrics_update_loop()),
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._cleanup_loop())
        ]
    
    async def _metrics_update_loop(self):
        """Boucle de mise à jour des métriques."""
        while self.state == EngineState.RUNNING:
            try:
                # Mettre à jour les métriques du moteur
                if self.start_time:
                    self.metrics.uptime_seconds = (
                        datetime.utcnow() - self.start_time
                    ).total_seconds()
                
                # Collecter les métriques système
                await self.metrics_collector.collect_system_metrics()
                
                # Sauvegarder l'état
                await self.state_manager.set_state(
                    'engine_metrics',
                    self.metrics.to_dict(),
                    ttl=300
                )
                
                await asyncio.sleep(60)  # Mise à jour chaque minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Erreur boucle métriques: {e}")
                await asyncio.sleep(10)
    
    async def _health_check_loop(self):
        """Boucle de vérification de santé."""
        while self.state == EngineState.RUNNING:
            try:
                # Vérifier la santé des composants
                health_status = await self._check_health()
                
                await self.state_manager.set_state(
                    'health_status',
                    health_status,
                    ttl=60
                )
                
                if not health_status.get('healthy', False):
                    self.logger.warning("Problème de santé détecté")
                
                await asyncio.sleep(30)  # Vérification toutes les 30 secondes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Erreur vérification santé: {e}")
                await asyncio.sleep(10)
    
    async def _cleanup_loop(self):
        """Boucle de nettoyage."""
        while self.state == EngineState.RUNNING:
            try:
                # Nettoyer les alertes expirées
                await self._cleanup_expired_alerts()
                
                # Nettoyer le cache
                await self._cleanup_cache()
                
                await asyncio.sleep(3600)  # Nettoyage chaque heure
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Erreur nettoyage: {e}")
                await asyncio.sleep(60)
    
    async def _check_health(self) -> Dict[str, Any]:
        """Vérifie la santé du système."""
        health = {
            'healthy': True,
            'components': {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            # Vérifier Redis
            if self.event_bus.redis_client:
                await self.event_bus.redis_client.ping()
                health['components']['redis'] = 'healthy'
            else:
                health['components']['redis'] = 'unhealthy'
                health['healthy'] = False
                
        except Exception:
            health['components']['redis'] = 'unhealthy'
            health['healthy'] = False
        
        return health
    
    async def _cleanup_expired_alerts(self):
        """Nettoie les alertes expirées."""
        now = datetime.utcnow()
        expired_alerts = []
        
        for alert_id, alert in self.alert_manager.active_alerts.items():
            if (now - alert.timestamp).total_seconds() > 3600:  # 1 heure
                expired_alerts.append(alert_id)
        
        for alert_id in expired_alerts:
            del self.alert_manager.active_alerts[alert_id]
        
        if expired_alerts:
            self.logger.info(f"Nettoyé {len(expired_alerts)} alertes expirées")
    
    async def _cleanup_cache(self):
        """Nettoie le cache."""
        # Implémentation du nettoyage de cache
        self.logger.debug("Nettoyage du cache effectué")
    
    def get_status(self) -> Dict[str, Any]:
        """Retourne le statut du moteur."""
        return {
            'state': self.state.value,
            'uptime_seconds': self.metrics.uptime_seconds,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'metrics': self.metrics.to_dict(),
            'active_alerts': len(self.alert_manager.active_alerts)
        }

    @asynccontextmanager
    async def get_engine():
        """Context manager pour le moteur d'analytics."""
        engine = None
        try:
            config = AnalyticsConfig()
            engine = AnalyticsEngine(config)
            await engine.start()
            yield engine
        finally:
            if engine:
                await engine.stop()
