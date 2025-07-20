"""
Advanced Analytics and Performance Manager

This module provides sophisticated analytics tracking, performance monitoring,
A/B testing, and email campaign optimization for email templates.

Version: 3.0.0
Developed by Spotify AI Agent Team
"""

import time
import asyncio
import hashlib
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import structlog
import aiofiles
import aioredis
import json
import statistics
from collections import defaultdict, Counter

logger = structlog.get_logger(__name__)

# ============================================================================
# Analytics Configuration Classes
# ============================================================================

class EventType(Enum):
    """Types d'événements analytiques"""
    EMAIL_SENT = "email_sent"
    EMAIL_OPENED = "email_opened"
    EMAIL_CLICKED = "email_clicked"
    EMAIL_BOUNCED = "email_bounced"
    EMAIL_UNSUBSCRIBED = "email_unsubscribed"
    EMAIL_SPAM = "email_spam"
    TEMPLATE_RENDERED = "template_rendered"
    TEMPLATE_ERROR = "template_error"
    PERFORMANCE_METRIC = "performance_metric"
    AB_TEST_IMPRESSION = "ab_test_impression"
    AB_TEST_CONVERSION = "ab_test_conversion"

class MetricType(Enum):
    """Types de métriques"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class ABTestStatus(Enum):
    """Statuts des tests A/B"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class AnalyticsEvent:
    """Événement analytique"""
    event_type: EventType
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    email_id: Optional[str] = None
    template_id: Optional[str] = None
    campaign_id: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetric:
    """Métrique de performance"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None

@dataclass
class ABTestVariant:
    """Variante de test A/B"""
    id: str
    name: str
    template_data: Dict[str, Any]
    weight: float = 0.5
    impressions: int = 0
    conversions: int = 0
    conversion_rate: float = 0.0

@dataclass
class ABTest:
    """Test A/B"""
    id: str
    name: str
    description: str
    status: ABTestStatus
    variants: List[ABTestVariant]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    target_metric: str = "conversion_rate"
    confidence_level: float = 0.95
    min_sample_size: int = 100
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class CampaignStats:
    """Statistiques de campagne"""
    campaign_id: str
    emails_sent: int = 0
    emails_delivered: int = 0
    emails_opened: int = 0
    emails_clicked: int = 0
    emails_bounced: int = 0
    emails_unsubscribed: int = 0
    emails_spam: int = 0
    delivery_rate: float = 0.0
    open_rate: float = 0.0
    click_rate: float = 0.0
    bounce_rate: float = 0.0
    unsubscribe_rate: float = 0.0
    spam_rate: float = 0.0

# ============================================================================
# Advanced Analytics Manager
# ============================================================================

class AdvancedAnalyticsManager:
    """Gestionnaire d'analytics avancé"""
    
    def __init__(self,
                 storage_dir: str,
                 redis_url: Optional[str] = None,
                 enable_real_time: bool = True,
                 retention_days: int = 90):
        
        self.storage_dir = Path(storage_dir)
        self.redis_url = redis_url
        self.enable_real_time = enable_real_time
        self.retention_days = retention_days
        
        # Stockage
        self.redis_client: Optional[aioredis.Redis] = None
        self.events_buffer: List[AnalyticsEvent] = []
        self.metrics_buffer: List[PerformanceMetric] = []
        
        # Tests A/B
        self.ab_tests: Dict[str, ABTest] = {}
        
        # Cache des statistiques
        self.stats_cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Configuration
        self.batch_size = 100
        self.flush_interval = 30  # secondes
        
        # Initialize
        asyncio.create_task(self._initialize())
        
        logger.info("Advanced Analytics Manager initialized")
    
    async def _initialize(self):
        """Initialisation du gestionnaire"""
        
        # Création des répertoires
        await self._ensure_directories()
        
        # Connexion Redis
        if self.redis_url and self.enable_real_time:
            try:
                self.redis_client = await aioredis.from_url(self.redis_url)
                await self.redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None
        
        # Chargement des tests A/B
        await self._load_ab_tests()
        
        # Démarrage du processus de flush périodique
        asyncio.create_task(self._periodic_flush())
        
        # Nettoyage périodique
        asyncio.create_task(self._periodic_cleanup())
        
        logger.info("Analytics Manager initialization completed")
    
    async def _ensure_directories(self):
        """Assure que les répertoires nécessaires existent"""
        
        directories = [
            self.storage_dir,
            self.storage_dir / "events",
            self.storage_dir / "metrics",
            self.storage_dir / "ab_tests",
            self.storage_dir / "reports",
            self.storage_dir / "exports"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def track_event(self, event: AnalyticsEvent):
        """Enregistre un événement analytique"""
        
        # Ajout à la queue
        self.events_buffer.append(event)
        
        # Stockage temps réel si Redis disponible
        if self.redis_client:
            await self._store_event_redis(event)
        
        # Flush si le buffer est plein
        if len(self.events_buffer) >= self.batch_size:
            await self._flush_events()
        
        # Mise à jour des statistiques en temps réel
        await self._update_real_time_stats(event)
        
        logger.debug(f"Tracked event: {event.event_type.value}")
    
    async def _store_event_redis(self, event: AnalyticsEvent):
        """Stocke un événement dans Redis"""
        
        try:
            # Clé Redis basée sur la date et le type
            date_key = event.timestamp.strftime("%Y-%m-%d")
            redis_key = f"events:{date_key}:{event.event_type.value}"
            
            # Sérialisation de l'événement
            event_data = {
                "timestamp": event.timestamp.isoformat(),
                "user_id": event.user_id,
                "session_id": event.session_id,
                "email_id": event.email_id,
                "template_id": event.template_id,
                "campaign_id": event.campaign_id,
                "properties": event.properties,
                "metadata": event.metadata
            }
            
            # Stockage avec expiration
            await self.redis_client.lpush(redis_key, json.dumps(event_data))
            await self.redis_client.expire(redis_key, self.retention_days * 86400)
            
        except Exception as e:
            logger.error(f"Failed to store event in Redis: {e}")
    
    async def record_metric(self, metric: PerformanceMetric):
        """Enregistre une métrique de performance"""
        
        # Ajout à la queue
        self.metrics_buffer.append(metric)
        
        # Stockage temps réel si Redis disponible
        if self.redis_client:
            await self._store_metric_redis(metric)
        
        # Flush si le buffer est plein
        if len(self.metrics_buffer) >= self.batch_size:
            await self._flush_metrics()
        
        logger.debug(f"Recorded metric: {metric.name} = {metric.value}")
    
    async def _store_metric_redis(self, metric: PerformanceMetric):
        """Stocke une métrique dans Redis"""
        
        try:
            # Clé Redis pour la métrique
            tags_str = ",".join([f"{k}={v}" for k, v in metric.tags.items()])
            redis_key = f"metrics:{metric.name}:{tags_str}"
            
            # Données de la métrique
            metric_data = {
                "value": metric.value,
                "timestamp": metric.timestamp.isoformat(),
                "type": metric.metric_type.value,
                "unit": metric.unit
            }
            
            # Stockage selon le type de métrique
            if metric.metric_type == MetricType.COUNTER:
                await self.redis_client.incrbyfloat(f"{redis_key}:total", metric.value)
            elif metric.metric_type == MetricType.GAUGE:
                await self.redis_client.set(f"{redis_key}:current", metric.value)
            elif metric.metric_type in [MetricType.HISTOGRAM, MetricType.TIMER]:
                await self.redis_client.lpush(f"{redis_key}:values", metric.value)
                await self.redis_client.ltrim(f"{redis_key}:values", 0, 999)  # Garde les 1000 dernières valeurs
            
            # Métadonnées
            await self.redis_client.lpush(f"{redis_key}:history", json.dumps(metric_data))
            await self.redis_client.expire(f"{redis_key}:history", self.retention_days * 86400)
            
        except Exception as e:
            logger.error(f"Failed to store metric in Redis: {e}")
    
    async def _flush_events(self):
        """Flush les événements vers le stockage persistant"""
        
        if not self.events_buffer:
            return
        
        # Groupage par date
        events_by_date = defaultdict(list)
        for event in self.events_buffer:
            date_key = event.timestamp.strftime("%Y-%m-%d")
            events_by_date[date_key].append(event)
        
        # Sauvegarde par fichier de date
        for date_key, events in events_by_date.items():
            await self._save_events_to_file(date_key, events)
        
        # Nettoyage du buffer
        self.events_buffer.clear()
        
        logger.debug(f"Flushed {sum(len(events) for events in events_by_date.values())} events")
    
    async def _flush_metrics(self):
        """Flush les métriques vers le stockage persistant"""
        
        if not self.metrics_buffer:
            return
        
        # Groupage par date
        metrics_by_date = defaultdict(list)
        for metric in self.metrics_buffer:
            date_key = metric.timestamp.strftime("%Y-%m-%d")
            metrics_by_date[date_key].append(metric)
        
        # Sauvegarde par fichier de date
        for date_key, metrics in metrics_by_date.items():
            await self._save_metrics_to_file(date_key, metrics)
        
        # Nettoyage du buffer
        self.metrics_buffer.clear()
        
        logger.debug(f"Flushed {sum(len(metrics) for metrics in metrics_by_date.values())} metrics")
    
    async def _save_events_to_file(self, date_key: str, events: List[AnalyticsEvent]):
        """Sauvegarde les événements dans un fichier"""
        
        file_path = self.storage_dir / "events" / f"{date_key}.jsonl"
        
        try:
            async with aiofiles.open(file_path, 'a', encoding='utf-8') as f:
                for event in events:
                    event_dict = asdict(event)
                    event_dict['event_type'] = event.event_type.value
                    event_dict['timestamp'] = event.timestamp.isoformat()
                    
                    await f.write(json.dumps(event_dict) + '\n')
                    
        except Exception as e:
            logger.error(f"Failed to save events to file {file_path}: {e}")
    
    async def _save_metrics_to_file(self, date_key: str, metrics: List[PerformanceMetric]):
        """Sauvegarde les métriques dans un fichier"""
        
        file_path = self.storage_dir / "metrics" / f"{date_key}.jsonl"
        
        try:
            async with aiofiles.open(file_path, 'a', encoding='utf-8') as f:
                for metric in metrics:
                    metric_dict = asdict(metric)
                    metric_dict['metric_type'] = metric.metric_type.value
                    metric_dict['timestamp'] = metric.timestamp.isoformat()
                    
                    await f.write(json.dumps(metric_dict) + '\n')
                    
        except Exception as e:
            logger.error(f"Failed to save metrics to file {file_path}: {e}")
    
    async def _update_real_time_stats(self, event: AnalyticsEvent):
        """Met à jour les statistiques en temps réel"""
        
        if not event.campaign_id:
            return
        
        # Clé de cache pour la campagne
        cache_key = f"campaign_stats:{event.campaign_id}"
        
        # Récupération ou initialisation des stats
        if cache_key not in self.stats_cache:
            self.stats_cache[cache_key] = CampaignStats(campaign_id=event.campaign_id)
        
        stats = self.stats_cache[cache_key]
        
        # Mise à jour selon le type d'événement
        if event.event_type == EventType.EMAIL_SENT:
            stats.emails_sent += 1
        elif event.event_type == EventType.EMAIL_OPENED:
            stats.emails_opened += 1
        elif event.event_type == EventType.EMAIL_CLICKED:
            stats.emails_clicked += 1
        elif event.event_type == EventType.EMAIL_BOUNCED:
            stats.emails_bounced += 1
        elif event.event_type == EventType.EMAIL_UNSUBSCRIBED:
            stats.emails_unsubscribed += 1
        elif event.event_type == EventType.EMAIL_SPAM:
            stats.emails_spam += 1
        
        # Calcul des taux
        if stats.emails_sent > 0:
            stats.emails_delivered = stats.emails_sent - stats.emails_bounced
            stats.delivery_rate = stats.emails_delivered / stats.emails_sent
            stats.bounce_rate = stats.emails_bounced / stats.emails_sent
            
            if stats.emails_delivered > 0:
                stats.open_rate = stats.emails_opened / stats.emails_delivered
                stats.unsubscribe_rate = stats.emails_unsubscribed / stats.emails_delivered
                stats.spam_rate = stats.emails_spam / stats.emails_delivered
                
                if stats.emails_opened > 0:
                    stats.click_rate = stats.emails_clicked / stats.emails_opened
    
    async def get_campaign_stats(self, campaign_id: str) -> CampaignStats:
        """Obtient les statistiques d'une campagne"""
        
        cache_key = f"campaign_stats:{campaign_id}"
        
        # Vérification du cache
        if cache_key in self.stats_cache:
            return self.stats_cache[cache_key]
        
        # Calcul depuis les événements stockés
        stats = await self._calculate_campaign_stats(campaign_id)
        self.stats_cache[cache_key] = stats
        
        return stats
    
    async def _calculate_campaign_stats(self, campaign_id: str) -> CampaignStats:
        """Calcule les statistiques d'une campagne depuis les événements"""
        
        stats = CampaignStats(campaign_id=campaign_id)
        
        # Lecture des fichiers d'événements
        events_dir = self.storage_dir / "events"
        
        for event_file in events_dir.glob("*.jsonl"):
            try:
                async with aiofiles.open(event_file, 'r', encoding='utf-8') as f:
                    async for line in f:
                        event_data = json.loads(line.strip())
                        
                        if event_data.get('campaign_id') == campaign_id:
                            event_type = EventType(event_data['event_type'])
                            
                            if event_type == EventType.EMAIL_SENT:
                                stats.emails_sent += 1
                            elif event_type == EventType.EMAIL_OPENED:
                                stats.emails_opened += 1
                            elif event_type == EventType.EMAIL_CLICKED:
                                stats.emails_clicked += 1
                            elif event_type == EventType.EMAIL_BOUNCED:
                                stats.emails_bounced += 1
                            elif event_type == EventType.EMAIL_UNSUBSCRIBED:
                                stats.emails_unsubscribed += 1
                            elif event_type == EventType.EMAIL_SPAM:
                                stats.emails_spam += 1
                                
            except Exception as e:
                logger.error(f"Error reading event file {event_file}: {e}")
        
        # Calcul des taux
        if stats.emails_sent > 0:
            stats.emails_delivered = stats.emails_sent - stats.emails_bounced
            stats.delivery_rate = stats.emails_delivered / stats.emails_sent
            stats.bounce_rate = stats.emails_bounced / stats.emails_sent
            
            if stats.emails_delivered > 0:
                stats.open_rate = stats.emails_opened / stats.emails_delivered
                stats.unsubscribe_rate = stats.emails_unsubscribed / stats.emails_delivered
                stats.spam_rate = stats.emails_spam / stats.emails_delivered
                
                if stats.emails_opened > 0:
                    stats.click_rate = stats.emails_clicked / stats.emails_opened
        
        return stats
    
    async def create_ab_test(self,
                           name: str,
                           description: str,
                           variants: List[ABTestVariant],
                           target_metric: str = "conversion_rate",
                           confidence_level: float = 0.95) -> ABTest:
        """Crée un nouveau test A/B"""
        
        test_id = str(uuid.uuid4())
        
        ab_test = ABTest(
            id=test_id,
            name=name,
            description=description,
            status=ABTestStatus.DRAFT,
            variants=variants,
            target_metric=target_metric,
            confidence_level=confidence_level
        )
        
        self.ab_tests[test_id] = ab_test
        
        # Sauvegarde
        await self._save_ab_test(ab_test)
        
        logger.info(f"Created A/B test: {name} ({test_id})")
        return ab_test
    
    async def start_ab_test(self, test_id: str) -> bool:
        """Démarre un test A/B"""
        
        if test_id not in self.ab_tests:
            return False
        
        ab_test = self.ab_tests[test_id]
        ab_test.status = ABTestStatus.ACTIVE
        ab_test.start_date = datetime.now()
        
        await self._save_ab_test(ab_test)
        
        logger.info(f"Started A/B test: {ab_test.name}")
        return True
    
    async def stop_ab_test(self, test_id: str) -> bool:
        """Arrête un test A/B"""
        
        if test_id not in self.ab_tests:
            return False
        
        ab_test = self.ab_tests[test_id]
        ab_test.status = ABTestStatus.COMPLETED
        ab_test.end_date = datetime.now()
        
        await self._save_ab_test(ab_test)
        
        logger.info(f"Stopped A/B test: {ab_test.name}")
        return True
    
    async def get_ab_test_variant(self, test_id: str, user_id: str) -> Optional[ABTestVariant]:
        """Obtient la variante pour un utilisateur donné"""
        
        if test_id not in self.ab_tests:
            return None
        
        ab_test = self.ab_tests[test_id]
        
        if ab_test.status != ABTestStatus.ACTIVE:
            return None
        
        # Hash déterministe basé sur l'ID utilisateur
        hash_input = f"{test_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        normalized_hash = (hash_value % 10000) / 10000.0  # 0.0 to 1.0
        
        # Sélection de la variante selon les poids
        cumulative_weight = 0.0
        for variant in ab_test.variants:
            cumulative_weight += variant.weight
            if normalized_hash <= cumulative_weight:
                # Enregistrement de l'impression
                await self.track_event(AnalyticsEvent(
                    event_type=EventType.AB_TEST_IMPRESSION,
                    timestamp=datetime.now(),
                    user_id=user_id,
                    properties={
                        "test_id": test_id,
                        "variant_id": variant.id
                    }
                ))
                
                return variant
        
        # Fallback vers la première variante
        return ab_test.variants[0] if ab_test.variants else None
    
    async def record_ab_test_conversion(self,
                                      test_id: str,
                                      user_id: str,
                                      variant_id: str):
        """Enregistre une conversion pour un test A/B"""
        
        await self.track_event(AnalyticsEvent(
            event_type=EventType.AB_TEST_CONVERSION,
            timestamp=datetime.now(),
            user_id=user_id,
            properties={
                "test_id": test_id,
                "variant_id": variant_id
            }
        ))
        
        # Mise à jour des statistiques du test
        if test_id in self.ab_tests:
            ab_test = self.ab_tests[test_id]
            for variant in ab_test.variants:
                if variant.id == variant_id:
                    variant.conversions += 1
                    if variant.impressions > 0:
                        variant.conversion_rate = variant.conversions / variant.impressions
                    break
            
            await self._save_ab_test(ab_test)
    
    async def analyze_ab_test(self, test_id: str) -> Dict[str, Any]:
        """Analyse les résultats d'un test A/B"""
        
        if test_id not in self.ab_tests:
            return {"error": "Test not found"}
        
        ab_test = self.ab_tests[test_id]
        
        # Calcul des statistiques
        analysis = {
            "test_id": test_id,
            "name": ab_test.name,
            "status": ab_test.status.value,
            "variants": [],
            "statistical_significance": False,
            "winning_variant": None,
            "confidence_level": ab_test.confidence_level
        }
        
        total_impressions = sum(v.impressions for v in ab_test.variants)
        total_conversions = sum(v.conversions for v in ab_test.variants)
        
        for variant in ab_test.variants:
            variant_stats = {
                "id": variant.id,
                "name": variant.name,
                "impressions": variant.impressions,
                "conversions": variant.conversions,
                "conversion_rate": variant.conversion_rate,
                "weight": variant.weight
            }
            
            if total_impressions > 0:
                variant_stats["impression_share"] = variant.impressions / total_impressions
            
            if total_conversions > 0:
                variant_stats["conversion_share"] = variant.conversions / total_conversions
            
            analysis["variants"].append(variant_stats)
        
        # Test de significativité statistique (simplifiée)
        if len(ab_test.variants) >= 2 and total_impressions >= ab_test.min_sample_size:
            best_variant = max(ab_test.variants, key=lambda v: v.conversion_rate)
            analysis["winning_variant"] = best_variant.id
            
            # Calcul simplifiée de la significativité
            if best_variant.impressions >= ab_test.min_sample_size / len(ab_test.variants):
                analysis["statistical_significance"] = True
        
        return analysis
    
    async def _save_ab_test(self, ab_test: ABTest):
        """Sauvegarde un test A/B"""
        
        file_path = self.storage_dir / "ab_tests" / f"{ab_test.id}.json"
        
        try:
            # Sérialisation
            test_dict = asdict(ab_test)
            test_dict['status'] = ab_test.status.value
            
            if ab_test.start_date:
                test_dict['start_date'] = ab_test.start_date.isoformat()
            if ab_test.end_date:
                test_dict['end_date'] = ab_test.end_date.isoformat()
            
            test_dict['created_at'] = ab_test.created_at.isoformat()
            
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(test_dict, indent=2))
                
        except Exception as e:
            logger.error(f"Failed to save A/B test {ab_test.id}: {e}")
    
    async def _load_ab_tests(self):
        """Charge les tests A/B"""
        
        ab_tests_dir = self.storage_dir / "ab_tests"
        
        for test_file in ab_tests_dir.glob("*.json"):
            try:
                async with aiofiles.open(test_file, 'r', encoding='utf-8') as f:
                    test_data = json.loads(await f.read())
                
                # Reconstruction de l'objet ABTest
                test_data['status'] = ABTestStatus(test_data['status'])
                
                if test_data.get('start_date'):
                    test_data['start_date'] = datetime.fromisoformat(test_data['start_date'])
                if test_data.get('end_date'):
                    test_data['end_date'] = datetime.fromisoformat(test_data['end_date'])
                
                test_data['created_at'] = datetime.fromisoformat(test_data['created_at'])
                
                # Reconstruction des variantes
                variants = []
                for variant_data in test_data['variants']:
                    variants.append(ABTestVariant(**variant_data))
                test_data['variants'] = variants
                
                ab_test = ABTest(**test_data)
                self.ab_tests[ab_test.id] = ab_test
                
            except Exception as e:
                logger.error(f"Failed to load A/B test from {test_file}: {e}")
    
    async def _periodic_flush(self):
        """Flush périodique des données"""
        
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_events()
                await self._flush_metrics()
                
            except Exception as e:
                logger.error(f"Periodic flush error: {e}")
    
    async def _periodic_cleanup(self):
        """Nettoyage périodique des données anciennes"""
        
        while True:
            try:
                # Nettoyage quotidien
                await asyncio.sleep(86400)
                
                cutoff_date = datetime.now() - timedelta(days=self.retention_days)
                
                # Nettoyage des fichiers d'événements
                events_dir = self.storage_dir / "events"
                for event_file in events_dir.glob("*.jsonl"):
                    try:
                        date_str = event_file.stem
                        file_date = datetime.strptime(date_str, "%Y-%m-%d")
                        
                        if file_date < cutoff_date:
                            event_file.unlink()
                            logger.info(f"Cleaned up old event file: {event_file}")
                            
                    except Exception as e:
                        logger.error(f"Error cleaning event file {event_file}: {e}")
                
                # Nettoyage des fichiers de métriques
                metrics_dir = self.storage_dir / "metrics"
                for metric_file in metrics_dir.glob("*.jsonl"):
                    try:
                        date_str = metric_file.stem
                        file_date = datetime.strptime(date_str, "%Y-%m-%d")
                        
                        if file_date < cutoff_date:
                            metric_file.unlink()
                            logger.info(f"Cleaned up old metric file: {metric_file}")
                            
                    except Exception as e:
                        logger.error(f"Error cleaning metric file {metric_file}: {e}")
                
            except Exception as e:
                logger.error(f"Periodic cleanup error: {e}")
    
    async def get_performance_report(self,
                                   start_date: datetime,
                                   end_date: datetime) -> Dict[str, Any]:
        """Génère un rapport de performance"""
        
        report = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "email_metrics": {},
            "template_metrics": {},
            "performance_metrics": {},
            "ab_tests": []
        }
        
        # Agrégation des événements email
        email_events = defaultdict(int)
        template_usage = defaultdict(int)
        
        # Lecture des fichiers dans la période
        current_date = start_date.date()
        end_date_only = end_date.date()
        
        while current_date <= end_date_only:
            date_str = current_date.strftime("%Y-%m-%d")
            event_file = self.storage_dir / "events" / f"{date_str}.jsonl"
            
            if event_file.exists():
                try:
                    async with aiofiles.open(event_file, 'r', encoding='utf-8') as f:
                        async for line in f:
                            event_data = json.loads(line.strip())
                            event_time = datetime.fromisoformat(event_data['timestamp'])
                            
                            if start_date <= event_time <= end_date:
                                event_type = event_data['event_type']
                                email_events[event_type] += 1
                                
                                if event_data.get('template_id'):
                                    template_usage[event_data['template_id']] += 1
                                    
                except Exception as e:
                    logger.error(f"Error reading event file {event_file}: {e}")
            
            current_date += timedelta(days=1)
        
        report["email_metrics"] = dict(email_events)
        report["template_metrics"] = dict(template_usage)
        
        # Tests A/B actifs dans la période
        for ab_test in self.ab_tests.values():
            if (ab_test.start_date and ab_test.start_date <= end_date and
                (not ab_test.end_date or ab_test.end_date >= start_date)):
                
                analysis = await self.analyze_ab_test(ab_test.id)
                report["ab_tests"].append(analysis)
        
        return report
    
    async def cleanup(self):
        """Nettoyage des ressources"""
        
        # Flush final
        await self._flush_events()
        await self._flush_metrics()
        
        # Fermeture Redis
        if self.redis_client:
            await self.redis_client.close()

# ============================================================================
# Performance Context Manager
# ============================================================================

class PerformanceTimer:
    """Context manager pour mesurer les performances"""
    
    def __init__(self, analytics_manager: AdvancedAnalyticsManager, metric_name: str, **tags):
        self.analytics_manager = analytics_manager
        self.metric_name = metric_name
        self.tags = tags
        self.start_time = None
    
    async def __aenter__(self):
        self.start_time = time.perf_counter()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.perf_counter() - self.start_time
            
            metric = PerformanceMetric(
                name=self.metric_name,
                value=duration,
                metric_type=MetricType.TIMER,
                timestamp=datetime.now(),
                tags=self.tags,
                unit="seconds"
            )
            
            await self.analytics_manager.record_metric(metric)

# ============================================================================
# Factory Functions
# ============================================================================

def create_analytics_manager(
    storage_dir: str,
    redis_url: Optional[str] = None,
    enable_real_time: bool = True
) -> AdvancedAnalyticsManager:
    """Factory pour créer un gestionnaire d'analytics"""
    
    return AdvancedAnalyticsManager(
        storage_dir=storage_dir,
        redis_url=redis_url,
        enable_real_time=enable_real_time
    )

def create_ab_test_variant(
    variant_id: str,
    name: str,
    template_data: Dict[str, Any],
    weight: float = 0.5
) -> ABTestVariant:
    """Crée une variante de test A/B"""
    
    return ABTestVariant(
        id=variant_id,
        name=name,
        template_data=template_data,
        weight=weight
    )

# Export des classes principales
__all__ = [
    "AdvancedAnalyticsManager",
    "PerformanceTimer",
    "AnalyticsEvent",
    "PerformanceMetric",
    "ABTest",
    "ABTestVariant",
    "CampaignStats",
    "EventType",
    "MetricType",
    "ABTestStatus",
    "create_analytics_manager",
    "create_ab_test_variant"
]
