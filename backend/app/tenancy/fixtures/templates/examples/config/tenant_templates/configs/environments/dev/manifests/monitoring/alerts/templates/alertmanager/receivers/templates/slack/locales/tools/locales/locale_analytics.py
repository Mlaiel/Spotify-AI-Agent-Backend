"""
Analyseur de Locales Avancé pour Spotify AI Agent
Système d'analytics et de tracking des locales multi-tenant
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from abc import ABC, abstractmethod
import statistics
import threading
import weakref
from enum import Enum

logger = logging.getLogger(__name__)


class AnalyticsEvent(Enum):
    """Types d'événements analytiques"""
    LOCALE_ACCESSED = "locale_accessed"
    LOCALE_CACHED = "locale_cached"
    LOCALE_UPDATED = "locale_updated"
    LOCALE_VALIDATED = "locale_validated"
    LOCALE_ERROR = "locale_error"
    TENANT_SWITCH = "tenant_switch"
    PERFORMANCE_ISSUE = "performance_issue"


@dataclass
class UsageEvent:
    """Événement d'utilisation des locales"""
    event_type: AnalyticsEvent
    locale_code: str
    tenant_id: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Métriques de performance"""
    avg_load_time: float
    max_load_time: float
    min_load_time: float
    total_requests: int
    error_rate: float
    cache_hit_rate: float
    memory_usage: int
    cpu_usage: float
    throughput_per_second: float
    percentile_95: float
    percentile_99: float


@dataclass
class LocaleUsageStats:
    """Statistiques d'utilisation d'une locale"""
    locale_code: str
    total_accesses: int
    unique_tenants: int
    avg_response_time: float
    error_count: int
    last_accessed: datetime
    popularity_score: float
    trending_score: float
    geographic_distribution: Dict[str, int] = field(default_factory=dict)
    time_distribution: Dict[int, int] = field(default_factory=dict)  # Heure -> Count


class AnalyticsCollector(ABC):
    """Interface pour les collecteurs d'analytics"""
    
    @abstractmethod
    async def collect_event(self, event: UsageEvent):
        """Collecte un événement"""
        pass
    
    @abstractmethod
    async def get_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Récupère les métriques"""
        pass


class MemoryAnalyticsCollector(AnalyticsCollector):
    """Collecteur d'analytics en mémoire"""
    
    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self._events = []
        self._lock = threading.RLock()
        self._event_index = defaultdict(list)
    
    async def collect_event(self, event: UsageEvent):
        """Collecte un événement en mémoire"""
        try:
            with self._lock:
                self._events.append(event)
                
                # Indexer par locale et tenant
                self._event_index[event.locale_code].append(len(self._events) - 1)
                if event.tenant_id:
                    self._event_index[f"tenant:{event.tenant_id}"].append(len(self._events) - 1)
                
                # Limiter la taille
                if len(self._events) > self.max_events:
                    # Supprimer les plus anciens
                    removed_count = len(self._events) - self.max_events
                    self._events = self._events[removed_count:]
                    
                    # Réindexer
                    self._rebuild_index()
                    
        except Exception as e:
            logger.error(f"Error collecting event: {e}")
    
    async def get_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Récupère les métriques depuis la mémoire"""
        try:
            with self._lock:
                # Filtrer les événements par période
                filtered_events = [
                    event for event in self._events
                    if start_time <= event.timestamp <= end_time
                ]
                
                if not filtered_events:
                    return {}
                
                # Calculer les métriques
                metrics = await self._calculate_metrics(filtered_events)
                return metrics
                
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {}
    
    def _rebuild_index(self):
        """Reconstruit l'index des événements"""
        self._event_index.clear()
        for i, event in enumerate(self._events):
            self._event_index[event.locale_code].append(i)
            if event.tenant_id:
                self._event_index[f"tenant:{event.tenant_id}"].append(i)
    
    async def _calculate_metrics(self, events: List[UsageEvent]) -> Dict[str, Any]:
        """Calcule les métriques à partir des événements"""
        if not events:
            return {}
        
        # Métriques de base
        total_events = len(events)
        locale_counts = Counter(event.locale_code for event in events)
        tenant_counts = Counter(event.tenant_id for event in events if event.tenant_id)
        event_type_counts = Counter(event.event_type for event in events)
        
        # Métriques de performance
        durations = [event.duration_ms for event in events if event.duration_ms is not None]
        
        performance_metrics = {}
        if durations:
            performance_metrics = {
                'avg_duration': statistics.mean(durations),
                'max_duration': max(durations),
                'min_duration': min(durations),
                'median_duration': statistics.median(durations),
                'p95_duration': self._percentile(durations, 95),
                'p99_duration': self._percentile(durations, 99)
            }
        
        # Métriques temporelles
        time_distribution = defaultdict(int)
        for event in events:
            hour = event.timestamp.hour
            time_distribution[hour] += 1
        
        # Taux d'erreur
        error_events = len([e for e in events if e.event_type == AnalyticsEvent.LOCALE_ERROR])
        error_rate = error_events / total_events if total_events > 0 else 0
        
        return {
            'total_events': total_events,
            'unique_locales': len(locale_counts),
            'unique_tenants': len(tenant_counts),
            'most_popular_locale': locale_counts.most_common(1)[0] if locale_counts else None,
            'most_active_tenant': tenant_counts.most_common(1)[0] if tenant_counts else None,
            'event_type_distribution': dict(event_type_counts),
            'error_rate': error_rate,
            'performance_metrics': performance_metrics,
            'time_distribution': dict(time_distribution),
            'locale_distribution': dict(locale_counts.most_common(10))
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calcule un percentile"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]


class DatabaseAnalyticsCollector(AnalyticsCollector):
    """Collecteur d'analytics avec persistance en base"""
    
    def __init__(self, db_session):
        self.db_session = db_session
    
    async def collect_event(self, event: UsageEvent):
        """Collecte un événement en base de données"""
        # Implémentation de la sauvegarde en base
        # Dépend du modèle de données exact
        pass
    
    async def get_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Récupère les métriques depuis la base"""
        # Implémentation des requêtes SQL
        pass


class LocaleAnalytics:
    """Système d'analytics principal pour les locales"""
    
    def __init__(self, collectors: List[AnalyticsCollector]):
        self.collectors = collectors
        self._observers = weakref.WeakSet()
        self._aggregated_stats = {}
        self._lock = threading.RLock()
        self._running = False
        self._aggregation_task = None
    
    async def start(self):
        """Démarre le système d'analytics"""
        if not self._running:
            self._running = True
            self._aggregation_task = asyncio.create_task(self._aggregation_loop())
            logger.info("Locale analytics started")
    
    async def stop(self):
        """Arrête le système d'analytics"""
        self._running = False
        if self._aggregation_task:
            self._aggregation_task.cancel()
            try:
                await self._aggregation_task
            except asyncio.CancelledError:
                pass
        logger.info("Locale analytics stopped")
    
    async def track_locale_access(
        self,
        locale_code: str,
        tenant_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        metadata: Dict[str, Any] = None
    ):
        """Trace un accès à une locale"""
        event = UsageEvent(
            event_type=AnalyticsEvent.LOCALE_ACCESSED,
            locale_code=locale_code,
            tenant_id=tenant_id,
            timestamp=datetime.now(),
            metadata=metadata or {},
            duration_ms=duration_ms
        )
        
        await self._collect_event(event)
    
    async def track_locale_error(
        self,
        locale_code: str,
        error_type: str,
        tenant_id: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Trace une erreur de locale"""
        event = UsageEvent(
            event_type=AnalyticsEvent.LOCALE_ERROR,
            locale_code=locale_code,
            tenant_id=tenant_id,
            timestamp=datetime.now(),
            metadata={**(metadata or {}), 'error_type': error_type}
        )
        
        await self._collect_event(event)
    
    async def track_performance_issue(
        self,
        locale_code: str,
        issue_type: str,
        severity: str,
        tenant_id: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Trace un problème de performance"""
        event = UsageEvent(
            event_type=AnalyticsEvent.PERFORMANCE_ISSUE,
            locale_code=locale_code,
            tenant_id=tenant_id,
            timestamp=datetime.now(),
            metadata={
                **(metadata or {}),
                'issue_type': issue_type,
                'severity': severity
            }
        )
        
        await self._collect_event(event)
    
    async def get_locale_usage_stats(
        self,
        locale_code: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> LocaleUsageStats:
        """Récupère les statistiques d'utilisation d'une locale"""
        try:
            if not start_time:
                start_time = datetime.now() - timedelta(days=30)
            if not end_time:
                end_time = datetime.now()
            
            # Collecter les métriques de tous les collecteurs
            all_metrics = []
            for collector in self.collectors:
                try:
                    metrics = await collector.get_metrics(start_time, end_time)
                    if metrics:
                        all_metrics.append(metrics)
                except Exception as e:
                    logger.warning(f"Collector error: {e}")
            
            # Agréger les métriques
            return await self._aggregate_locale_stats(locale_code, all_metrics)
            
        except Exception as e:
            logger.error(f"Error getting locale usage stats: {e}")
            return LocaleUsageStats(
                locale_code=locale_code,
                total_accesses=0,
                unique_tenants=0,
                avg_response_time=0.0,
                error_count=0,
                last_accessed=datetime.now(),
                popularity_score=0.0,
                trending_score=0.0
            )
    
    async def get_global_analytics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Récupère les analytics globales"""
        try:
            if not start_time:
                start_time = datetime.now() - timedelta(days=7)
            if not end_time:
                end_time = datetime.now()
            
            # Collecter toutes les métriques
            global_metrics = {
                'period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                },
                'collectors': [],
                'summary': {}
            }
            
            total_events = 0
            all_locales = set()
            all_tenants = set()
            
            for collector in self.collectors:
                try:
                    metrics = await collector.get_metrics(start_time, end_time)
                    if metrics:
                        global_metrics['collectors'].append({
                            'type': collector.__class__.__name__,
                            'metrics': metrics
                        })
                        
                        # Agréger pour le résumé
                        total_events += metrics.get('total_events', 0)
                        all_locales.update(metrics.get('locale_distribution', {}).keys())
                        
                except Exception as e:
                    logger.warning(f"Collector error: {e}")
            
            global_metrics['summary'] = {
                'total_events': total_events,
                'unique_locales': len(all_locales),
                'unique_tenants': len(all_tenants),
                'avg_events_per_locale': total_events / len(all_locales) if all_locales else 0
            }
            
            return global_metrics
            
        except Exception as e:
            logger.error(f"Error getting global analytics: {e}")
            return {}
    
    async def get_tenant_analytics(
        self,
        tenant_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Récupère les analytics spécifiques à un tenant"""
        try:
            if not start_time:
                start_time = datetime.now() - timedelta(days=30)
            if not end_time:
                end_time = datetime.now()
            
            tenant_analytics = {
                'tenant_id': tenant_id,
                'period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                },
                'locale_usage': {},
                'performance': {},
                'trends': {}
            }
            
            # Obtenir les métriques pour ce tenant
            for collector in self.collectors:
                try:
                    metrics = await collector.get_metrics(start_time, end_time)
                    # Filtrer par tenant (implémentation dépendante du collecteur)
                    # Pour l'instant, retourner les métriques globales
                    tenant_analytics['performance'].update(
                        metrics.get('performance_metrics', {})
                    )
                except Exception as e:
                    logger.warning(f"Tenant analytics collector error: {e}")
            
            return tenant_analytics
            
        except Exception as e:
            logger.error(f"Error getting tenant analytics: {e}")
            return {}
    
    async def generate_usage_report(
        self,
        format: str = "json",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Génère un rapport d'utilisation complet"""
        try:
            if not start_time:
                start_time = datetime.now() - timedelta(days=7)
            if not end_time:
                end_time = datetime.now()
            
            # Collecter toutes les données
            global_analytics = await self.get_global_analytics(start_time, end_time)
            
            # Construire le rapport
            report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'period': {
                        'start': start_time.isoformat(),
                        'end': end_time.isoformat()
                    },
                    'format': format
                },
                'executive_summary': await self._generate_executive_summary(global_analytics),
                'detailed_analytics': global_analytics,
                'recommendations': await self._generate_recommendations(global_analytics),
                'charts_data': await self._prepare_chart_data(global_analytics)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating usage report: {e}")
            return {}
    
    def add_observer(self, observer):
        """Ajoute un observateur pour les événements analytics"""
        self._observers.add(observer)
    
    def remove_observer(self, observer):
        """Supprime un observateur"""
        self._observers.discard(observer)
    
    async def _collect_event(self, event: UsageEvent):
        """Collecte un événement via tous les collecteurs"""
        tasks = []
        for collector in self.collectors:
            task = collector.collect_event(event)
            tasks.append(task)
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Notifier les observateurs
            await self._notify_observers(event)
            
        except Exception as e:
            logger.error(f"Error collecting event: {e}")
    
    async def _notify_observers(self, event: UsageEvent):
        """Notifie les observateurs d'un événement"""
        for observer in list(self._observers):
            try:
                if hasattr(observer, 'on_analytics_event'):
                    await observer.on_analytics_event(event)
            except Exception as e:
                logger.warning(f"Observer notification error: {e}")
    
    async def _aggregation_loop(self):
        """Boucle d'agrégation des statistiques"""
        while self._running:
            try:
                await asyncio.sleep(300)  # Toutes les 5 minutes
                await self._aggregate_statistics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Aggregation loop error: {e}")
    
    async def _aggregate_statistics(self):
        """Agrège les statistiques périodiquement"""
        try:
            # Agréger les données des dernières 24h
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            
            with self._lock:
                self._aggregated_stats = await self.get_global_analytics(start_time, end_time)
            
            logger.debug("Statistics aggregated")
            
        except Exception as e:
            logger.error(f"Statistics aggregation error: {e}")
    
    async def _aggregate_locale_stats(
        self,
        locale_code: str,
        metrics_list: List[Dict[str, Any]]
    ) -> LocaleUsageStats:
        """Agrège les statistiques d'une locale"""
        if not metrics_list:
            return LocaleUsageStats(
                locale_code=locale_code,
                total_accesses=0,
                unique_tenants=0,
                avg_response_time=0.0,
                error_count=0,
                last_accessed=datetime.now(),
                popularity_score=0.0,
                trending_score=0.0
            )
        
        # Agréger les métriques
        total_accesses = 0
        all_tenants = set()
        response_times = []
        error_count = 0
        
        for metrics in metrics_list:
            locale_dist = metrics.get('locale_distribution', {})
            total_accesses += locale_dist.get(locale_code, 0)
            
            if 'performance_metrics' in metrics:
                perf = metrics['performance_metrics']
                if 'avg_duration' in perf:
                    response_times.append(perf['avg_duration'])
            
            error_count += metrics.get('error_rate', 0) * metrics.get('total_events', 0)
        
        avg_response_time = statistics.mean(response_times) if response_times else 0.0
        
        # Calculer les scores
        popularity_score = min(total_accesses / 1000, 1.0)  # Normaliser sur 1000 accès
        trending_score = popularity_score * 0.8  # Score de tendance basique
        
        return LocaleUsageStats(
            locale_code=locale_code,
            total_accesses=total_accesses,
            unique_tenants=len(all_tenants),
            avg_response_time=avg_response_time,
            error_count=int(error_count),
            last_accessed=datetime.now(),
            popularity_score=popularity_score,
            trending_score=trending_score
        )
    
    async def _generate_executive_summary(self, analytics: Dict[str, Any]) -> Dict[str, Any]:
        """Génère un résumé exécutif"""
        summary = analytics.get('summary', {})
        
        return {
            'total_locale_requests': summary.get('total_events', 0),
            'unique_locales_used': summary.get('unique_locales', 0),
            'average_response_time': 'N/A',  # Calculer depuis les métriques de performance
            'error_rate': 'N/A',
            'most_popular_locale': 'N/A',
            'key_insights': [
                f"Total de {summary.get('total_events', 0)} requêtes de localisation",
                f"{summary.get('unique_locales', 0)} locales différentes utilisées",
                "Performance stable sur la période analysée"
            ]
        }
    
    async def _generate_recommendations(self, analytics: Dict[str, Any]) -> List[str]:
        """Génère des recommandations basées sur les analytics"""
        recommendations = []
        
        summary = analytics.get('summary', {})
        
        # Recommandations basées sur l'utilisation
        if summary.get('unique_locales', 0) > 20:
            recommendations.append("Considérer l'optimisation du cache pour les locales populaires")
        
        if summary.get('total_events', 0) > 10000:
            recommendations.append("Implémenter un système de cache distribué")
        
        recommendations.append("Surveiller les patterns d'utilisation pour optimiser les performances")
        
        return recommendations
    
    async def _prepare_chart_data(self, analytics: Dict[str, Any]) -> Dict[str, Any]:
        """Prépare les données pour les graphiques"""
        chart_data = {
            'locale_usage_pie': {},
            'time_series': {},
            'performance_metrics': {}
        }
        
        # Préparer les données depuis les analytics
        for collector_data in analytics.get('collectors', []):
            metrics = collector_data.get('metrics', {})
            
            # Distribution des locales
            if 'locale_distribution' in metrics:
                chart_data['locale_usage_pie'] = metrics['locale_distribution']
            
            # Distribution temporelle
            if 'time_distribution' in metrics:
                chart_data['time_series'] = metrics['time_distribution']
        
        return chart_data


class UsageTracker:
    """Tracker d'utilisation simplifié"""
    
    def __init__(self, analytics: LocaleAnalytics):
        self.analytics = analytics
        self._session_data = {}
        self._lock = threading.RLock()
    
    async def track_session_start(self, session_id: str, tenant_id: str, locale_code: str):
        """Démarre le tracking d'une session"""
        with self._lock:
            self._session_data[session_id] = {
                'tenant_id': tenant_id,
                'locale_code': locale_code,
                'start_time': datetime.now(),
                'locale_switches': 0,
                'requests_count': 0
            }
    
    async def track_locale_request(self, session_id: str, locale_code: str, response_time: float):
        """Trace une requête de locale"""
        with self._lock:
            if session_id in self._session_data:
                session = self._session_data[session_id]
                session['requests_count'] += 1
                
                if session['locale_code'] != locale_code:
                    session['locale_switches'] += 1
                    session['locale_code'] = locale_code
        
        # Envoyer à l'analytics
        await self.analytics.track_locale_access(
            locale_code=locale_code,
            tenant_id=self._session_data.get(session_id, {}).get('tenant_id'),
            duration_ms=response_time * 1000,
            metadata={'session_id': session_id}
        )
    
    async def track_session_end(self, session_id: str):
        """Termine le tracking d'une session"""
        with self._lock:
            if session_id in self._session_data:
                session_data = self._session_data.pop(session_id)
                
                # Calculer la durée de session
                duration = datetime.now() - session_data['start_time']
                
                # Log des métriques de session
                logger.info(f"Session {session_id} ended: "
                          f"Duration={duration.total_seconds()}s, "
                          f"Requests={session_data['requests_count']}, "
                          f"Locale switches={session_data['locale_switches']}")
    
    def get_active_sessions(self) -> Dict[str, Any]:
        """Retourne les sessions actives"""
        with self._lock:
            return {
                'active_count': len(self._session_data),
                'sessions': dict(self._session_data)
            }
