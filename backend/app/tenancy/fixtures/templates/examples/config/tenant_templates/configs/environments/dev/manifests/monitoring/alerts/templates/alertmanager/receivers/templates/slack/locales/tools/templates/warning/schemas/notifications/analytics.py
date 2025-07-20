"""
Système d'analytics avancé pour notifications
============================================

Analytics en temps réel avec ML, prédictions, et insights automatiques.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, asc
import aioredis
from prometheus_client import Counter, Histogram, Gauge, Summary
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

from .models import *
from .schemas import *


@dataclass
class NotificationInsight:
    """Insight automatique sur les notifications"""
    type: str  # 'performance', 'anomaly', 'trend', 'optimization'
    severity: str  # 'info', 'warning', 'critical'
    title: str
    description: str
    metrics: Dict[str, Any]
    recommendations: List[str]
    confidence_score: float
    created_at: datetime


class InsightType(str, Enum):
    """Types d'insights"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DELIVERY_ANOMALY = "delivery_anomaly"
    USAGE_SPIKE = "usage_spike"
    ERROR_PATTERN = "error_pattern"
    OPTIMIZATION_OPPORTUNITY = "optimization_opportunity"
    TREND_CHANGE = "trend_change"
    CAPACITY_WARNING = "capacity_warning"


class NotificationAnalyticsService:
    """Service d'analytics avancé pour notifications"""
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: aioredis.Redis,
        config: Dict[str, Any] = None
    ):
        self.db = db_session
        self.redis = redis_client
        self.config = config or {}
        self.logger = logging.getLogger("NotificationAnalytics")
        
        # Modèles ML pour détection d'anomalies
        self.anomaly_detector = None
        self.clustering_model = None
        self.scaler = StandardScaler()
        
        # Cache pour métriques temps réel
        self._metrics_cache = {}
        self._cache_ttl = 60  # 1 minute
        
        # Seuils pour alertes
        self.thresholds = {
            'error_rate_critical': 0.1,  # 10% d'erreurs
            'error_rate_warning': 0.05,  # 5% d'erreurs
            'delivery_time_critical': 30000,  # 30 secondes
            'delivery_time_warning': 10000,   # 10 secondes
            'queue_size_critical': 10000,
            'queue_size_warning': 5000,
        }
        
        # Buffer pour streaming analytics
        self._event_buffer = []
        self._buffer_size = 1000
    
    async def initialize(self):
        """Initialisation du service analytics"""
        await self._load_ml_models()
        await self._setup_real_time_processors()
        
        # Démarrer les tâches de fond
        asyncio.create_task(self._continuous_analysis_loop())
        asyncio.create_task(self._metrics_aggregation_loop())
    
    async def _load_ml_models(self):
        """Charger les modèles ML pré-entraînés"""
        try:
            # Modèle de détection d'anomalies
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            # Modèle de clustering
            self.clustering_model = KMeans(
                n_clusters=5,
                random_state=42,
                n_init=10
            )
            
            # Entraîner sur des données historiques
            await self._train_models()
            
        except Exception as e:
            self.logger.error(f"Erreur chargement modèles ML: {e}")
    
    async def _train_models(self):
        """Entraîner les modèles sur les données historiques"""
        try:
            # Récupérer les données des 30 derniers jours
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=30)
            
            query = select(NotificationMetrics).where(
                and_(
                    NotificationMetrics.time_bucket >= start_date,
                    NotificationMetrics.time_bucket <= end_date
                )
            )
            
            result = await self.db.execute(query)
            historical_data = result.scalars().all()
            
            if len(historical_data) < 100:  # Pas assez de données
                self.logger.warning("Pas assez de données historiques pour entraîner les modèles")
                return
            
            # Préparer les features
            features = []
            for metric in historical_data:
                feature_vector = [
                    metric.total_sent,
                    metric.total_delivered,
                    metric.total_failed,
                    metric.avg_delivery_time_ms,
                    metric.retry_rate,
                    metric.success_rate,
                    metric.read_rate,
                    len(metric.error_breakdown) if metric.error_breakdown else 0
                ]
                features.append(feature_vector)
            
            features_array = np.array(features)
            
            # Normaliser les features
            features_scaled = self.scaler.fit_transform(features_array)
            
            # Entraîner le détecteur d'anomalies
            self.anomaly_detector.fit(features_scaled)
            
            # Entraîner le modèle de clustering
            self.clustering_model.fit(features_scaled)
            
            self.logger.info(f"Modèles ML entraînés sur {len(historical_data)} échantillons")
            
        except Exception as e:
            self.logger.error(f"Erreur entraînement modèles: {e}")
    
    async def _setup_real_time_processors(self):
        """Configurer les processeurs temps réel"""
        # Configuration des streams Redis pour événements temps réel
        self.event_streams = {
            'notifications': 'notification_events',
            'deliveries': 'delivery_events',
            'errors': 'error_events'
        }
    
    async def track_notification_event(
        self,
        event_type: str,
        notification_id: str,
        tenant_id: str,
        metadata: Dict[str, Any]
    ):
        """Tracker un événement de notification en temps réel"""
        
        event = {
            'event_type': event_type,
            'notification_id': notification_id,
            'tenant_id': tenant_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'metadata': metadata
        }
        
        # Ajouter au buffer pour traitement
        self._event_buffer.append(event)
        
        # Publier sur Redis stream
        stream_name = self.event_streams.get('notifications', 'notification_events')
        await self.redis.xadd(stream_name, event)
        
        # Traitement immédiat pour événements critiques
        if event_type in ['delivery_failed', 'timeout', 'rate_limited']:
            await self._process_critical_event(event)
    
    async def _process_critical_event(self, event: Dict[str, Any]):
        """Traiter immédiatement les événements critiques"""
        
        # Détecter les patterns d'erreur
        if event['event_type'] == 'delivery_failed':
            await self._analyze_error_pattern(event)
        
        # Détecter les problèmes de performance
        if event['event_type'] == 'timeout':
            await self._analyze_performance_issue(event)
        
        # Détecter les limites de taux
        if event['event_type'] == 'rate_limited':
            await self._analyze_rate_limit_issue(event)
    
    async def get_real_time_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Obtenir les métriques temps réel"""
        
        cache_key = f"realtime_metrics:{tenant_id}"
        
        # Vérifier le cache
        cached = await self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Calculer les métriques
        now = datetime.now(timezone.utc)
        
        # Métriques de la dernière heure
        hour_ago = now - timedelta(hours=1)
        
        # Notifications envoyées
        sent_query = select(func.count(Notification.id)).where(
            and_(
                Notification.tenant_id == tenant_id,
                Notification.created_at >= hour_ago,
                Notification.status.in_([
                    NotificationStatus.SENT,
                    NotificationStatus.DELIVERED,
                    NotificationStatus.READ
                ])
            )
        )
        
        sent_result = await self.db.execute(sent_query)
        sent_count = sent_result.scalar() or 0
        
        # Notifications échouées
        failed_query = select(func.count(Notification.id)).where(
            and_(
                Notification.tenant_id == tenant_id,
                Notification.created_at >= hour_ago,
                Notification.status == NotificationStatus.FAILED
            )
        )
        
        failed_result = await self.db.execute(failed_query)
        failed_count = failed_result.scalar() or 0
        
        # Temps de livraison moyen
        delivery_time_query = select(func.avg(Notification.delivery_time_ms)).where(
            and_(
                Notification.tenant_id == tenant_id,
                Notification.sent_at >= hour_ago,
                Notification.delivery_time_ms.isnot(None)
            )
        )
        
        delivery_time_result = await self.db.execute(delivery_time_query)
        avg_delivery_time = delivery_time_result.scalar() or 0
        
        # Queue size actuelle
        queue_query = select(func.count(NotificationQueue.id)).where(
            and_(
                NotificationQueue.tenant_id == tenant_id,
                NotificationQueue.status == 'pending'
            )
        )
        
        queue_result = await self.db.execute(queue_query)
        queue_size = queue_result.scalar() or 0
        
        # Calculer les taux
        total_notifications = sent_count + failed_count
        success_rate = (sent_count / total_notifications) if total_notifications > 0 else 1.0
        error_rate = (failed_count / total_notifications) if total_notifications > 0 else 0.0
        
        metrics = {
            'timestamp': now.isoformat(),
            'period': 'last_hour',
            'notifications_sent': sent_count,
            'notifications_failed': failed_count,
            'success_rate': round(success_rate, 3),
            'error_rate': round(error_rate, 3),
            'avg_delivery_time_ms': round(avg_delivery_time, 2) if avg_delivery_time else 0,
            'queue_size': queue_size,
            'throughput_per_minute': round((sent_count + failed_count) / 60, 2),
        }
        
        # Ajouter les alertes
        metrics['alerts'] = await self._generate_real_time_alerts(metrics)
        
        # Mettre en cache
        await self.redis.setex(cache_key, self._cache_ttl, json.dumps(metrics))
        
        return metrics
    
    async def _generate_real_time_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Générer des alertes basées sur les métriques temps réel"""
        
        alerts = []
        
        # Alerte taux d'erreur
        if metrics['error_rate'] >= self.thresholds['error_rate_critical']:
            alerts.append({
                'type': 'error_rate',
                'severity': 'critical',
                'message': f"Taux d'erreur critique: {metrics['error_rate']:.1%}",
                'threshold': self.thresholds['error_rate_critical']
            })
        elif metrics['error_rate'] >= self.thresholds['error_rate_warning']:
            alerts.append({
                'type': 'error_rate',
                'severity': 'warning',
                'message': f"Taux d'erreur élevé: {metrics['error_rate']:.1%}",
                'threshold': self.thresholds['error_rate_warning']
            })
        
        # Alerte temps de livraison
        if metrics['avg_delivery_time_ms'] >= self.thresholds['delivery_time_critical']:
            alerts.append({
                'type': 'delivery_time',
                'severity': 'critical',
                'message': f"Temps de livraison critique: {metrics['avg_delivery_time_ms']:.0f}ms",
                'threshold': self.thresholds['delivery_time_critical']
            })
        elif metrics['avg_delivery_time_ms'] >= self.thresholds['delivery_time_warning']:
            alerts.append({
                'type': 'delivery_time',
                'severity': 'warning',
                'message': f"Temps de livraison élevé: {metrics['avg_delivery_time_ms']:.0f}ms",
                'threshold': self.thresholds['delivery_time_warning']
            })
        
        # Alerte taille de queue
        if metrics['queue_size'] >= self.thresholds['queue_size_critical']:
            alerts.append({
                'type': 'queue_size',
                'severity': 'critical',
                'message': f"Queue saturée: {metrics['queue_size']} notifications",
                'threshold': self.thresholds['queue_size_critical']
            })
        elif metrics['queue_size'] >= self.thresholds['queue_size_warning']:
            alerts.append({
                'type': 'queue_size',
                'severity': 'warning',
                'message': f"Queue chargée: {metrics['queue_size']} notifications",
                'threshold': self.thresholds['queue_size_warning']
            })
        
        return alerts
    
    async def get_analytics_dashboard_data(
        self,
        tenant_id: str,
        period: str = 'last_24h',
        channels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Obtenir les données pour le dashboard analytics"""
        
        # Calculer la période
        now = datetime.now(timezone.utc)
        if period == 'last_hour':
            start_time = now - timedelta(hours=1)
            time_bucket_size = 'minute'
        elif period == 'last_24h':
            start_time = now - timedelta(days=1)
            time_bucket_size = 'hour'
        elif period == 'last_7d':
            start_time = now - timedelta(days=7)
            time_bucket_size = 'day'
        elif period == 'last_30d':
            start_time = now - timedelta(days=30)
            time_bucket_size = 'day'
        else:
            start_time = now - timedelta(days=1)
            time_bucket_size = 'hour'
        
        # Requête de base pour les métriques
        query = select(NotificationMetrics).where(
            and_(
                NotificationMetrics.tenant_id == tenant_id,
                NotificationMetrics.time_bucket >= start_time
            )
        )
        
        if channels:
            query = query.where(NotificationMetrics.channel_type.in_(channels))
        
        query = query.order_by(NotificationMetrics.time_bucket)
        
        result = await self.db.execute(query)
        metrics_data = result.scalars().all()
        
        # Agréger les données
        dashboard_data = {
            'period': period,
            'start_time': start_time.isoformat(),
            'end_time': now.isoformat(),
            'summary': await self._calculate_summary_metrics(metrics_data),
            'trends': await self._calculate_trend_data(metrics_data, time_bucket_size),
            'channels': await self._calculate_channel_breakdown(metrics_data),
            'performance': await self._calculate_performance_metrics(metrics_data),
            'errors': await self._calculate_error_analysis(metrics_data),
            'insights': await self._generate_insights(tenant_id, metrics_data)
        }
        
        return dashboard_data
    
    async def _calculate_summary_metrics(self, metrics_data: List[NotificationMetrics]) -> Dict[str, Any]:
        """Calculer les métriques de résumé"""
        
        if not metrics_data:
            return {
                'total_sent': 0,
                'total_delivered': 0,
                'total_failed': 0,
                'success_rate': 0,
                'avg_delivery_time': 0,
                'read_rate': 0
            }
        
        total_sent = sum(m.total_sent for m in metrics_data)
        total_delivered = sum(m.total_delivered for m in metrics_data)
        total_failed = sum(m.total_failed for m in metrics_data)
        total_read = sum(m.total_read for m in metrics_data)
        
        # Moyenne pondérée des temps de livraison
        weighted_delivery_time = sum(
            m.avg_delivery_time_ms * m.total_delivered 
            for m in metrics_data if m.total_delivered > 0
        )
        total_delivered_for_time = sum(
            m.total_delivered for m in metrics_data if m.total_delivered > 0
        )
        
        avg_delivery_time = (
            weighted_delivery_time / total_delivered_for_time 
            if total_delivered_for_time > 0 else 0
        )
        
        success_rate = total_delivered / total_sent if total_sent > 0 else 0
        read_rate = total_read / total_delivered if total_delivered > 0 else 0
        
        return {
            'total_sent': total_sent,
            'total_delivered': total_delivered,
            'total_failed': total_failed,
            'total_read': total_read,
            'success_rate': round(success_rate, 3),
            'avg_delivery_time_ms': round(avg_delivery_time, 2),
            'read_rate': round(read_rate, 3)
        }
    
    async def _calculate_trend_data(
        self,
        metrics_data: List[NotificationMetrics],
        time_bucket_size: str
    ) -> Dict[str, List]:
        """Calculer les données de tendance"""
        
        # Grouper par bucket de temps
        time_buckets = defaultdict(lambda: {
            'sent': 0,
            'delivered': 0,
            'failed': 0,
            'delivery_times': []
        })
        
        for metric in metrics_data:
            bucket_key = self._get_time_bucket_key(metric.time_bucket, time_bucket_size)
            bucket = time_buckets[bucket_key]
            
            bucket['sent'] += metric.total_sent
            bucket['delivered'] += metric.total_delivered
            bucket['failed'] += metric.total_failed
            
            if metric.avg_delivery_time_ms > 0:
                bucket['delivery_times'].append(metric.avg_delivery_time_ms)
        
        # Convertir en listes pour graphiques
        timestamps = sorted(time_buckets.keys())
        sent_data = []
        delivered_data = []
        failed_data = []
        success_rate_data = []
        delivery_time_data = []
        
        for timestamp in timestamps:
            bucket = time_buckets[timestamp]
            
            sent_data.append(bucket['sent'])
            delivered_data.append(bucket['delivered'])
            failed_data.append(bucket['failed'])
            
            # Taux de succès
            total = bucket['sent']
            success_rate = (bucket['delivered'] / total) if total > 0 else 0
            success_rate_data.append(round(success_rate, 3))
            
            # Temps de livraison moyen
            avg_delivery = (
                statistics.mean(bucket['delivery_times']) 
                if bucket['delivery_times'] else 0
            )
            delivery_time_data.append(round(avg_delivery, 2))
        
        return {
            'timestamps': timestamps,
            'sent': sent_data,
            'delivered': delivered_data,
            'failed': failed_data,
            'success_rate': success_rate_data,
            'delivery_time': delivery_time_data
        }
    
    async def _calculate_channel_breakdown(self, metrics_data: List[NotificationMetrics]) -> Dict[str, Any]:
        """Calculer la répartition par canal"""
        
        channel_stats = defaultdict(lambda: {
            'sent': 0,
            'delivered': 0,
            'failed': 0,
            'delivery_times': []
        })
        
        for metric in metrics_data:
            channel = metric.channel_type
            stats = channel_stats[channel]
            
            stats['sent'] += metric.total_sent
            stats['delivered'] += metric.total_delivered
            stats['failed'] += metric.total_failed
            
            if metric.avg_delivery_time_ms > 0:
                stats['delivery_times'].append(metric.avg_delivery_time_ms)
        
        # Convertir en format final
        channel_breakdown = {}
        for channel, stats in channel_stats.items():
            total = stats['sent']
            success_rate = (stats['delivered'] / total) if total > 0 else 0
            avg_delivery = (
                statistics.mean(stats['delivery_times']) 
                if stats['delivery_times'] else 0
            )
            
            channel_breakdown[channel] = {
                'sent': stats['sent'],
                'delivered': stats['delivered'],
                'failed': stats['failed'],
                'success_rate': round(success_rate, 3),
                'avg_delivery_time_ms': round(avg_delivery, 2)
            }
        
        return channel_breakdown
    
    async def _generate_insights(
        self,
        tenant_id: str,
        metrics_data: List[NotificationMetrics]
    ) -> List[NotificationInsight]:
        """Générer des insights automatiques"""
        
        insights = []
        
        # Insight sur les performances
        performance_insight = await self._analyze_performance_trends(metrics_data)
        if performance_insight:
            insights.append(performance_insight)
        
        # Insight sur les anomalies
        anomaly_insight = await self._detect_anomalies(metrics_data)
        if anomaly_insight:
            insights.append(anomaly_insight)
        
        # Insight sur l'optimisation
        optimization_insight = await self._suggest_optimizations(tenant_id, metrics_data)
        if optimization_insight:
            insights.append(optimization_insight)
        
        # Insight sur les tendances
        trend_insight = await self._analyze_usage_trends(metrics_data)
        if trend_insight:
            insights.append(trend_insight)
        
        return insights
    
    async def _analyze_performance_trends(self, metrics_data: List[NotificationMetrics]) -> Optional[NotificationInsight]:
        """Analyser les tendances de performance"""
        
        if len(metrics_data) < 10:
            return None
        
        # Analyser l'évolution du temps de livraison
        delivery_times = [m.avg_delivery_time_ms for m in metrics_data if m.avg_delivery_time_ms > 0]
        
        if len(delivery_times) < 5:
            return None
        
        # Calculer la tendance (régression linéaire simple)
        x = np.arange(len(delivery_times))
        coeffs = np.polyfit(x, delivery_times, 1)
        trend = coeffs[0]  # Pente
        
        # Déterminer la sévérité
        if trend > 50:  # Augmentation de plus de 50ms par période
            severity = 'critical'
            title = "Dégradation critique des performances"
            description = f"Le temps de livraison augmente de {trend:.0f}ms par période"
            recommendations = [
                "Vérifier la charge des services de notification",
                "Analyser les goulots d'étranglement réseau",
                "Considérer l'ajout de capacité"
            ]
        elif trend > 20:
            severity = 'warning'
            title = "Dégradation des performances détectée"
            description = f"Le temps de livraison augmente de {trend:.0f}ms par période"
            recommendations = [
                "Surveiller l'évolution de la performance",
                "Analyser les logs pour identifier les causes"
            ]
        elif trend < -20:  # Amélioration significative
            severity = 'info'
            title = "Amélioration des performances"
            description = f"Le temps de livraison diminue de {abs(trend):.0f}ms par période"
            recommendations = [
                "Documenter les changements récents",
                "Maintenir les bonnes pratiques"
            ]
        else:
            return None
        
        return NotificationInsight(
            type='performance',
            severity=severity,
            title=title,
            description=description,
            metrics={
                'trend_slope': trend,
                'current_avg': delivery_times[-1],
                'previous_avg': delivery_times[0],
                'samples': len(delivery_times)
            },
            recommendations=recommendations,
            confidence_score=min(0.9, len(delivery_times) / 20),
            created_at=datetime.now(timezone.utc)
        )
    
    async def _detect_anomalies(self, metrics_data: List[NotificationMetrics]) -> Optional[NotificationInsight]:
        """Détecter les anomalies avec ML"""
        
        if not self.anomaly_detector or len(metrics_data) < 10:
            return None
        
        try:
            # Préparer les features
            features = []
            for metric in metrics_data:
                feature_vector = [
                    metric.total_sent,
                    metric.total_delivered,
                    metric.total_failed,
                    metric.avg_delivery_time_ms,
                    metric.retry_rate,
                    metric.success_rate,
                    metric.read_rate,
                    len(metric.error_breakdown) if metric.error_breakdown else 0
                ]
                features.append(feature_vector)
            
            features_array = np.array(features)
            features_scaled = self.scaler.transform(features_array)
            
            # Détecter les anomalies
            anomaly_scores = self.anomaly_detector.decision_function(features_scaled)
            anomalies = self.anomaly_detector.predict(features_scaled)
            
            # Identifier les anomalies significatives
            anomaly_indices = np.where(anomalies == -1)[0]
            
            if len(anomaly_indices) == 0:
                return None
            
            # Analyser les anomalies récentes (dernières 20%)
            recent_threshold = int(len(metrics_data) * 0.8)
            recent_anomalies = [i for i in anomaly_indices if i >= recent_threshold]
            
            if not recent_anomalies:
                return None
            
            # Créer l'insight
            anomaly_count = len(recent_anomalies)
            severity = 'critical' if anomaly_count > 2 else 'warning'
            
            return NotificationInsight(
                type='anomaly',
                severity=severity,
                title=f"{anomaly_count} anomalie(s) détectée(s)",
                description=f"Comportement inhabituel détecté dans les {anomaly_count} dernières périodes",
                metrics={
                    'anomaly_count': anomaly_count,
                    'anomaly_indices': recent_anomalies.tolist(),
                    'min_score': float(min(anomaly_scores[recent_anomalies])),
                    'avg_score': float(np.mean(anomaly_scores[recent_anomalies]))
                },
                recommendations=[
                    "Analyser les logs des périodes anormales",
                    "Vérifier les changements de configuration",
                    "Surveiller de près les prochaines heures"
                ],
                confidence_score=0.8,
                created_at=datetime.now(timezone.utc)
            )
        
        except Exception as e:
            self.logger.error(f"Erreur détection anomalies: {e}")
            return None
    
    def _get_time_bucket_key(self, timestamp: datetime, bucket_size: str) -> str:
        """Obtenir la clé de bucket de temps"""
        if bucket_size == 'minute':
            return timestamp.strftime('%Y-%m-%d %H:%M')
        elif bucket_size == 'hour':
            return timestamp.strftime('%Y-%m-%d %H:00')
        elif bucket_size == 'day':
            return timestamp.strftime('%Y-%m-%d')
        else:
            return timestamp.isoformat()
    
    async def _continuous_analysis_loop(self):
        """Boucle d'analyse continue en arrière-plan"""
        while True:
            try:
                # Traiter les événements du buffer
                if self._event_buffer:
                    events_batch = self._event_buffer[:self._buffer_size]
                    self._event_buffer = self._event_buffer[self._buffer_size:]
                    
                    await self._process_events_batch(events_batch)
                
                await asyncio.sleep(30)  # Traiter toutes les 30 secondes
                
            except Exception as e:
                self.logger.error(f"Erreur dans la boucle d'analyse continue: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_aggregation_loop(self):
        """Boucle d'agrégation des métriques"""
        while True:
            try:
                # Agréger les métriques horaires
                await self._aggregate_hourly_metrics()
                
                # Attendre jusqu'à la prochaine heure
                now = datetime.now(timezone.utc)
                next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                sleep_seconds = (next_hour - now).total_seconds()
                
                await asyncio.sleep(sleep_seconds)
                
            except Exception as e:
                self.logger.error(f"Erreur dans l'agrégation des métriques: {e}")
                await asyncio.sleep(3600)  # Attendre 1 heure en cas d'erreur
    
    async def _aggregate_hourly_metrics(self):
        """Agréger les métriques par heure"""
        
        # Période à agréger (heure précédente)
        now = datetime.now(timezone.utc)
        end_time = now.replace(minute=0, second=0, microsecond=0)
        start_time = end_time - timedelta(hours=1)
        
        # Requête pour obtenir toutes les notifications de cette période
        query = select(Notification).where(
            and_(
                Notification.created_at >= start_time,
                Notification.created_at < end_time
            )
        )
        
        result = await self.db.execute(query)
        notifications = result.scalars().all()
        
        if not notifications:
            return
        
        # Grouper par tenant, canal, priorité, type de destinataire
        groups = defaultdict(list)
        
        for notification in notifications:
            key = (
                notification.tenant_id,
                notification.channel_type,
                notification.priority,
                notification.recipient_type
            )
            groups[key].append(notification)
        
        # Créer les métriques agrégées
        for (tenant_id, channel_type, priority, recipient_type), group_notifications in groups.items():
            
            # Calculer les métriques
            total_sent = len([n for n in group_notifications if n.status in [
                NotificationStatus.SENT, NotificationStatus.DELIVERED, NotificationStatus.READ
            ]])
            
            total_delivered = len([n for n in group_notifications if n.status in [
                NotificationStatus.DELIVERED, NotificationStatus.READ
            ]])
            
            total_failed = len([n for n in group_notifications if n.status == NotificationStatus.FAILED])
            total_read = len([n for n in group_notifications if n.status == NotificationStatus.READ])
            
            # Temps de livraison
            delivery_times = [n.delivery_time_ms for n in group_notifications if n.delivery_time_ms]
            avg_delivery_time = statistics.mean(delivery_times) if delivery_times else 0
            p95_delivery_time = np.percentile(delivery_times, 95) if delivery_times else 0
            p99_delivery_time = np.percentile(delivery_times, 99) if delivery_times else 0
            
            # Temps de lecture
            read_times = [n.read_time_ms for n in group_notifications if n.read_time_ms]
            avg_read_time = statistics.mean(read_times) if read_times else 0
            
            # Taux
            retry_count = sum(n.retry_count for n in group_notifications)
            retry_rate = retry_count / len(group_notifications) if group_notifications else 0
            
            success_rate = total_delivered / total_sent if total_sent > 0 else 0
            read_rate = total_read / total_delivered if total_delivered > 0 else 0
            
            # Répartition des erreurs
            error_breakdown = Counter()
            for notification in group_notifications:
                if notification.status == NotificationStatus.FAILED:
                    # Récupérer la dernière tentative pour obtenir l'erreur
                    if notification.delivery_attempts:
                        last_attempt = max(notification.delivery_attempts, key=lambda a: a.attempt_number)
                        error_code = last_attempt.error_code or 'unknown'
                        error_breakdown[error_code] += 1
            
            # Créer l'enregistrement de métrique
            metric = NotificationMetrics(
                time_bucket=start_time,
                tenant_id=tenant_id,
                channel_type=channel_type,
                priority=priority,
                recipient_type=recipient_type,
                total_sent=total_sent,
                total_delivered=total_delivered,
                total_failed=total_failed,
                total_read=total_read,
                avg_delivery_time_ms=avg_delivery_time,
                avg_read_time_ms=avg_read_time,
                p95_delivery_time_ms=p95_delivery_time,
                p99_delivery_time_ms=p99_delivery_time,
                retry_rate=retry_rate,
                success_rate=success_rate,
                read_rate=read_rate,
                error_breakdown=dict(error_breakdown)
            )
            
            self.db.add(metric)
        
        await self.db.commit()
        self.logger.info(f"Métriques agrégées pour la période {start_time} - {end_time}")
