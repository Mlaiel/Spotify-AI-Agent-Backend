"""
Storage Module - Systèmes de Stockage Analytics
==============================================

Ce module gère tous les systèmes de stockage pour les données analytics,
incluant les séries temporelles, métriques, événements et cache.

Classes:
- BaseStorage: Stockage de base
- TimeSeriesStorage: Stockage séries temporelles (InfluxDB)
- MetricsStorage: Stockage métriques (Prometheus)
- EventStorage: Stockage événements (Elasticsearch)
- CacheStorage: Stockage cache (Redis)
- PostgreSQLStorage: Stockage relationnel
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import redis.asyncio as aioredis
from elasticsearch import AsyncElasticsearch
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from ..config import AnalyticsConfig
from ..models import Metric, Event, Alert, Dashboard
from ..utils import Logger, Timer, retry_async


@dataclass
class StorageStats:
    """Statistiques de stockage."""
    total_writes: int = 0
    total_reads: int = 0
    total_deletes: int = 0
    total_errors: int = 0
    average_write_time: float = 0
    average_read_time: float = 0
    storage_size_bytes: int = 0
    connection_pool_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_writes': self.total_writes,
            'total_reads': self.total_reads,
            'total_deletes': self.total_deletes,
            'total_errors': self.total_errors,
            'average_write_time': self.average_write_time,
            'average_read_time': self.average_read_time,
            'storage_size_bytes': self.storage_size_bytes,
            'connection_pool_size': self.connection_pool_size
        }


class BaseStorage(ABC):
    """Classe de base pour tous les systèmes de stockage."""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.logger = Logger(self.__class__.__name__)
        self.is_connected = False
        self.stats = StorageStats()
        self._write_times = []
        self._read_times = []
    
    @abstractmethod
    async def connect(self):
        """Établit la connexion."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Ferme la connexion."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Vérifie la santé de la connexion."""
        pass
    
    def _update_write_stats(self, duration: float):
        """Met à jour les stats d'écriture."""
        self.stats.total_writes += 1
        self._write_times.append(duration)
        if len(self._write_times) > 100:
            self._write_times.pop(0)
        self.stats.average_write_time = sum(self._write_times) / len(self._write_times)
    
    def _update_read_stats(self, duration: float):
        """Met à jour les stats de lecture."""
        self.stats.total_reads += 1
        self._read_times.append(duration)
        if len(self._read_times) > 100:
            self._read_times.pop(0)
        self.stats.average_read_time = sum(self._read_times) / len(self._read_times)
    
    def _update_error_stats(self):
        """Met à jour les stats d'erreur."""
        self.stats.total_errors += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques."""
        return self.stats.to_dict()


class TimeSeriesStorage(BaseStorage):
    """Stockage de séries temporelles avec InfluxDB."""
    
    def __init__(self, config: AnalyticsConfig):
        super().__init__(config)
        self.client: Optional[InfluxDBClient] = None
        self.write_api = None
        self.query_api = None
        self.bucket = config.database.influx_bucket
        self.org = config.database.influx_org
    
    async def connect(self):
        """Connexion à InfluxDB."""
        try:
            self.client = InfluxDBClient(
                url=f"http://{self.config.database.influx_host}:{self.config.database.influx_port}",
                token=self.config.database.influx_token,
                org=self.org,
                timeout=self.config.database.influx_timeout * 1000
            )
            
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.client.query_api()
            
            # Test de connexion
            await self.health_check()
            
            self.is_connected = True
            self.logger.info("Connexion InfluxDB établie")
            
        except Exception as e:
            self.logger.error(f"Erreur connexion InfluxDB: {e}")
            raise
    
    async def disconnect(self):
        """Déconnexion d'InfluxDB."""
        if self.client:
            self.client.close()
            self.is_connected = False
            self.logger.info("Connexion InfluxDB fermée")
    
    async def health_check(self) -> bool:
        """Vérifie la santé d'InfluxDB."""
        try:
            if not self.client:
                return False
            
            # Test avec une requête simple
            health = self.client.health()
            return health.status == "pass"
            
        except Exception as e:
            self.logger.error(f"Health check InfluxDB échoué: {e}")
            return False
    
    @retry_async(max_attempts=3)
    async def write_metric(self, metric: Metric):
        """Écrit une métrique dans InfluxDB."""
        if not self.is_connected:
            raise RuntimeError("InfluxDB non connecté")
        
        start_time = time.time()
        
        try:
            # Créer un point InfluxDB
            point = Point(metric.name) \
                .tag("tenant_id", metric.tenant_id) \
                .field("value", float(metric.value)) \
                .time(metric.timestamp)
            
            # Ajouter les tags
            for key, value in metric.tags.items():
                point = point.tag(key, str(value))
            
            # Ajouter les métadonnées comme champs
            for key, value in metric.metadata.items():
                if isinstance(value, (int, float)):
                    point = point.field(f"meta_{key}", value)
                else:
                    point = point.field(f"meta_{key}", str(value))
            
            # Écriture
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            
            duration = time.time() - start_time
            self._update_write_stats(duration)
            
        except Exception as e:
            self._update_error_stats()
            self.logger.error(f"Erreur écriture métrique InfluxDB: {e}")
            raise
    
    async def write_metrics_batch(self, metrics: List[Metric]):
        """Écrit un batch de métriques."""
        if not self.is_connected:
            raise RuntimeError("InfluxDB non connecté")
        
        start_time = time.time()
        
        try:
            points = []
            for metric in metrics:
                point = Point(metric.name) \
                    .tag("tenant_id", metric.tenant_id) \
                    .field("value", float(metric.value)) \
                    .time(metric.timestamp)
                
                # Tags
                for key, value in metric.tags.items():
                    point = point.tag(key, str(value))
                
                points.append(point)
            
            # Écriture en batch
            self.write_api.write(bucket=self.bucket, org=self.org, record=points)
            
            duration = time.time() - start_time
            self._update_write_stats(duration)
            
        except Exception as e:
            self._update_error_stats()
            self.logger.error(f"Erreur écriture batch InfluxDB: {e}")
            raise
    
    async def query_metrics(
        self,
        tenant_id: str,
        metric_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """Requête de métriques."""
        if not self.is_connected:
            raise RuntimeError("InfluxDB non connecté")
        
        query_start = time.time()
        
        try:
            # Construire la requête Flux
            query = f'from(bucket: "{self.bucket}") |> range(start: '
            
            if start_time:
                query += start_time.isoformat()
            else:
                query += "-1h"
            
            if end_time:
                query += f", stop: {end_time.isoformat()}"
            
            query += ') |> filter(fn: (r) => r._measurement == "'
            
            if metric_name:
                query += metric_name
            else:
                query += 'metrics'
            
            query += f'") |> filter(fn: (r) => r.tenant_id == "{tenant_id}")'
            
            # Filtres par tags
            if tags:
                for key, value in tags.items():
                    query += f' |> filter(fn: (r) => r.{key} == "{value}")'
            
            # Exécuter la requête
            result = self.query_api.query(org=self.org, query=query)
            
            # Convertir en format standard
            metrics = []
            for table in result:
                for record in table.records:
                    metrics.append({
                        'name': record.get_measurement(),
                        'value': record.get_value(),
                        'timestamp': record.get_time(),
                        'tenant_id': record.values.get('tenant_id'),
                        'tags': {k: v for k, v in record.values.items() 
                                if k not in ['_measurement', '_field', '_value', '_time', 'tenant_id']}
                    })
            
            duration = time.time() - query_start
            self._update_read_stats(duration)
            
            return metrics
            
        except Exception as e:
            self._update_error_stats()
            self.logger.error(f"Erreur requête InfluxDB: {e}")
            raise
    
    async def get_metric_aggregations(
        self,
        tenant_id: str,
        metric_name: str,
        aggregation: str = "mean",
        window: str = "1h",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Récupère des agrégations de métriques."""
        query_start = time.time()
        
        try:
            query = f'''
            from(bucket: "{self.bucket}")
                |> range(start: {start_time.isoformat() if start_time else "-24h"}
                         {f", stop: {end_time.isoformat()}" if end_time else ""})
                |> filter(fn: (r) => r._measurement == "{metric_name}")
                |> filter(fn: (r) => r.tenant_id == "{tenant_id}")
                |> aggregateWindow(every: {window}, fn: {aggregation})
                |> yield(name: "{aggregation}")
            '''
            
            result = self.query_api.query(org=self.org, query=query)
            
            aggregations = []
            for table in result:
                for record in table.records:
                    aggregations.append({
                        'timestamp': record.get_time(),
                        'value': record.get_value(),
                        'aggregation': aggregation,
                        'window': window
                    })
            
            duration = time.time() - query_start
            self._update_read_stats(duration)
            
            return aggregations
            
        except Exception as e:
            self._update_error_stats()
            self.logger.error(f"Erreur agrégation InfluxDB: {e}")
            raise


class EventStorage(BaseStorage):
    """Stockage d'événements avec Elasticsearch."""
    
    def __init__(self, config: AnalyticsConfig):
        super().__init__(config)
        self.client: Optional[AsyncElasticsearch] = None
        self.index_prefix = config.database.elastic_index_prefix
    
    async def connect(self):
        """Connexion à Elasticsearch."""
        try:
            self.client = AsyncElasticsearch(
                hosts=self.config.database.elastic_hosts,
                http_auth=(
                    self.config.database.elastic_username,
                    self.config.database.elastic_password
                ),
                use_ssl=self.config.database.elastic_ssl,
                verify_certs=self.config.database.elastic_verify_certs,
                timeout=self.config.database.elastic_timeout
            )
            
            # Test de connexion
            await self.health_check()
            
            self.is_connected = True
            self.logger.info("Connexion Elasticsearch établie")
            
        except Exception as e:
            self.logger.error(f"Erreur connexion Elasticsearch: {e}")
            raise
    
    async def disconnect(self):
        """Déconnexion d'Elasticsearch."""
        if self.client:
            await self.client.close()
            self.is_connected = False
            self.logger.info("Connexion Elasticsearch fermée")
    
    async def health_check(self) -> bool:
        """Vérifie la santé d'Elasticsearch."""
        try:
            if not self.client:
                return False
            
            health = await self.client.cluster.health()
            return health['status'] in ['green', 'yellow']
            
        except Exception as e:
            self.logger.error(f"Health check Elasticsearch échoué: {e}")
            return False
    
    @retry_async(max_attempts=3)
    async def write_event(self, event: Event):
        """Écrit un événement dans Elasticsearch."""
        if not self.is_connected:
            raise RuntimeError("Elasticsearch non connecté")
        
        start_time = time.time()
        
        try:
            index_name = self._get_index_name(event.tenant_id, event.timestamp)
            
            # Document à indexer
            doc = {
                'id': event.id,
                'name': event.name,
                'event_type': event.event_type,
                'tenant_id': event.tenant_id,
                'user_id': event.user_id,
                'session_id': event.session_id,
                'source': event.source,
                'data': event.data,
                'context': event.context,
                'tags': event.tags,
                'timestamp': event.timestamp,
                'processed': event.processed,
                'created_at': event.created_at
            }
            
            # Indexation
            await self.client.index(
                index=index_name,
                id=event.id,
                document=doc
            )
            
            duration = time.time() - start_time
            self._update_write_stats(duration)
            
        except Exception as e:
            self._update_error_stats()
            self.logger.error(f"Erreur écriture événement Elasticsearch: {e}")
            raise
    
    async def write_events_batch(self, events: List[Event]):
        """Écrit un batch d'événements."""
        if not self.is_connected:
            raise RuntimeError("Elasticsearch non connecté")
        
        start_time = time.time()
        
        try:
            # Préparer les actions bulk
            actions = []
            for event in events:
                index_name = self._get_index_name(event.tenant_id, event.timestamp)
                
                action = {
                    "_index": index_name,
                    "_id": event.id,
                    "_source": {
                        'id': event.id,
                        'name': event.name,
                        'event_type': event.event_type,
                        'tenant_id': event.tenant_id,
                        'user_id': event.user_id,
                        'session_id': event.session_id,
                        'source': event.source,
                        'data': event.data,
                        'context': event.context,
                        'tags': event.tags,
                        'timestamp': event.timestamp,
                        'processed': event.processed,
                        'created_at': event.created_at
                    }
                }
                actions.append(action)
            
            # Bulk insert
            await self.client.bulk(operations=actions)
            
            duration = time.time() - start_time
            self._update_write_stats(duration)
            
        except Exception as e:
            self._update_error_stats()
            self.logger.error(f"Erreur écriture batch Elasticsearch: {e}")
            raise
    
    async def search_events(
        self,
        tenant_id: str,
        query: Optional[str] = None,
        event_types: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        size: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Recherche d'événements."""
        if not self.is_connected:
            raise RuntimeError("Elasticsearch non connecté")
        
        query_start = time.time()
        
        try:
            # Construire la requête
            search_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"tenant_id": tenant_id}}
                        ]
                    }
                },
                "sort": [
                    {"timestamp": {"order": "desc"}}
                ],
                "size": size,
                "from": offset
            }
            
            # Filtre par type d'événement
            if event_types:
                search_query["query"]["bool"]["must"].append({
                    "terms": {"event_type": event_types}
                })
            
            # Filtre temporel
            if start_time or end_time:
                time_filter = {"range": {"timestamp": {}}}
                if start_time:
                    time_filter["range"]["timestamp"]["gte"] = start_time.isoformat()
                if end_time:
                    time_filter["range"]["timestamp"]["lte"] = end_time.isoformat()
                search_query["query"]["bool"]["must"].append(time_filter)
            
            # Recherche textuelle
            if query:
                search_query["query"]["bool"]["must"].append({
                    "multi_match": {
                        "query": query,
                        "fields": ["name", "source", "data.*", "context.*"]
                    }
                })
            
            # Index pattern
            index_pattern = f"{self.index_prefix}-events-{tenant_id}-*"
            
            # Exécuter la recherche
            response = await self.client.search(
                index=index_pattern,
                body=search_query
            )
            
            duration = time.time() - query_start
            self._update_read_stats(duration)
            
            return {
                'total': response['hits']['total']['value'],
                'events': [hit['_source'] for hit in response['hits']['hits']],
                'took': response['took']
            }
            
        except Exception as e:
            self._update_error_stats()
            self.logger.error(f"Erreur recherche Elasticsearch: {e}")
            raise
    
    def _get_index_name(self, tenant_id: str, timestamp: datetime) -> str:
        """Génère le nom d'index basé sur la date."""
        date_str = timestamp.strftime('%Y-%m-%d')
        return f"{self.index_prefix}-events-{tenant_id}-{date_str}"


class CacheStorage(BaseStorage):
    """Stockage cache avec Redis."""
    
    def __init__(self, config: AnalyticsConfig):
        super().__init__(config)
        self.client: Optional[aioredis.Redis] = None
        self.default_ttl = config.cache.default_ttl
    
    async def connect(self):
        """Connexion à Redis."""
        try:
            self.client = aioredis.from_url(
                self.config.cache.redis_url,
                decode_responses=True,
                max_connections=self.config.cache.redis_max_connections,
                retry_on_timeout=self.config.cache.redis_retry_on_timeout,
                socket_timeout=self.config.cache.redis_socket_timeout,
                socket_connect_timeout=self.config.cache.redis_socket_connect_timeout,
                health_check_interval=self.config.cache.redis_health_check_interval
            )
            
            # Test de connexion
            await self.health_check()
            
            self.is_connected = True
            self.logger.info("Connexion Redis établie")
            
        except Exception as e:
            self.logger.error(f"Erreur connexion Redis: {e}")
            raise
    
    async def disconnect(self):
        """Déconnexion de Redis."""
        if self.client:
            await self.client.close()
            self.is_connected = False
            self.logger.info("Connexion Redis fermée")
    
    async def health_check(self) -> bool:
        """Vérifie la santé de Redis."""
        try:
            if not self.client:
                return False
            
            await self.client.ping()
            return True
            
        except Exception as e:
            self.logger.error(f"Health check Redis échoué: {e}")
            return False
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Stocke une valeur dans Redis."""
        if not self.is_connected:
            raise RuntimeError("Redis non connecté")
        
        start_time = time.time()
        
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value, default=str)
            
            ttl = ttl or self.default_ttl
            
            await self.client.setex(key, ttl, str(value))
            
            duration = time.time() - start_time
            self._update_write_stats(duration)
            
        except Exception as e:
            self._update_error_stats()
            self.logger.error(f"Erreur écriture Redis: {e}")
            raise
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Récupère une valeur de Redis."""
        if not self.is_connected:
            raise RuntimeError("Redis non connecté")
        
        start_time = time.time()
        
        try:
            value = await self.client.get(key)
            
            if value is None:
                return default
            
            # Tenter de désérialiser le JSON
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
            
            duration = time.time() - start_time
            self._update_read_stats(duration)
            
        except Exception as e:
            self._update_error_stats()
            self.logger.error(f"Erreur lecture Redis: {e}")
            return default
    
    async def delete(self, key: str):
        """Supprime une clé de Redis."""
        if not self.is_connected:
            raise RuntimeError("Redis non connecté")
        
        try:
            await self.client.delete(key)
            self.stats.total_deletes += 1
            
        except Exception as e:
            self._update_error_stats()
            self.logger.error(f"Erreur suppression Redis: {e}")
            raise
    
    async def exists(self, key: str) -> bool:
        """Vérifie si une clé existe."""
        if not self.is_connected:
            return False
        
        try:
            return bool(await self.client.exists(key))
        except Exception:
            return False
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Incrémente une valeur."""
        if not self.is_connected:
            raise RuntimeError("Redis non connecté")
        
        try:
            return await self.client.incrby(key, amount)
        except Exception as e:
            self._update_error_stats()
            self.logger.error(f"Erreur incrémentation Redis: {e}")
            raise
    
    async def expire(self, key: str, ttl: int):
        """Définit un TTL sur une clé."""
        if not self.is_connected:
            raise RuntimeError("Redis non connecté")
        
        try:
            await self.client.expire(key, ttl)
        except Exception as e:
            self._update_error_stats()
            self.logger.error(f"Erreur expiration Redis: {e}")
            raise


# Storage Manager pour orchestrer tous les storages
class StorageManager:
    """Gestionnaire de tous les systèmes de stockage."""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.logger = Logger(__name__)
        
        # Initialiser les storages
        self.timeseries = TimeSeriesStorage(config)
        self.events = EventStorage(config)
        self.cache = CacheStorage(config)
        
        self.storages = [self.timeseries, self.events, self.cache]
    
    async def connect_all(self):
        """Connecte tous les storages."""
        for storage in self.storages:
            try:
                await storage.connect()
            except Exception as e:
                self.logger.error(f"Échec connexion {storage.__class__.__name__}: {e}")
    
    async def disconnect_all(self):
        """Déconnecte tous les storages."""
        for storage in self.storages:
            try:
                await storage.disconnect()
            except Exception as e:
                self.logger.error(f"Échec déconnexion {storage.__class__.__name__}: {e}")
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Vérifie la santé de tous les storages."""
        health = {}
        for storage in self.storages:
            try:
                health[storage.__class__.__name__] = await storage.health_check()
            except Exception:
                health[storage.__class__.__name__] = False
        
        return health
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Récupère les stats de tous les storages."""
        stats = {}
        for storage in self.storages:
            stats[storage.__class__.__name__] = storage.get_stats()
        
        return stats


# Context manager pour la gestion automatique des connexions
@asynccontextmanager
async def storage_context(config: AnalyticsConfig):
    """Context manager pour les storages."""
    manager = StorageManager(config)
    try:
        await manager.connect_all()
        yield manager
    finally:
        await manager.disconnect_all()
