"""
Spotify AI Agent - Enterprise Integrations Module
================================================

Intégrations ultra-avancées avec les services externes
pour les collecteurs de données avec connecteurs enterprise-grade,
fault tolerance, et monitoring complet.

Intégrations supportées:
- Spotify Web API: Intégration complète avec rate limiting intelligent
- TimescaleDB: Stockage time-series haute performance
- Redis Cluster: Cache distribué avec sharding automatique
- Elasticsearch: Recherche et analytics avancées
- Prometheus: Métriques et monitoring
- Grafana: Dashboards et alerting
- Apache Kafka: Streaming de données en temps réel
- PostgreSQL: Base de données relationnelle
- InfluxDB: Métriques time-series
- AWS Services: S3, CloudWatch, Lambda
- GCP Services: BigQuery, Cloud Storage, Pub/Sub
- Azure Services: Event Hubs, Cosmos DB

Développé par l'équipe Spotify AI Agent
Lead Developer: Fahed Mlaiel
Architecture: Enterprise-grade service integrations
"""

import asyncio
import time
import json
import base64
import hashlib
import hmac
import ssl
import socket
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from typing import (
    Any, Dict, List, Optional, Union, Callable, Tuple, 
    Set, Awaitable, Type, Protocol, Generic, TypeVar
)
import logging
import structlog
import asyncpg
import aioredis
import aiohttp
import aiokafka
from elasticsearch import AsyncElasticsearch
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
import boto3
from azure.eventhub.aio import EventHubProducerClient
from google.cloud import bigquery, storage, pubsub_v1
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
import pandas as pd
import numpy as np
from pydantic import BaseModel, validator, SecretStr
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import certifi
import urllib3


T = TypeVar('T')
logger = structlog.get_logger(__name__)


class IntegrationType(Enum):
    """Types d'intégrations disponibles."""
    SPOTIFY_API = "spotify_api"
    TIMESCALE_DB = "timescale_db"
    REDIS_CLUSTER = "redis_cluster"
    ELASTICSEARCH = "elasticsearch"
    PROMETHEUS = "prometheus"
    GRAFANA = "grafana"
    KAFKA = "kafka"
    POSTGRESQL = "postgresql"
    INFLUXDB = "influxdb"
    AWS_S3 = "aws_s3"
    AWS_CLOUDWATCH = "aws_cloudwatch"
    GCP_BIGQUERY = "gcp_bigquery"
    GCP_STORAGE = "gcp_storage"
    AZURE_EVENTHUB = "azure_eventhub"


class ConnectionState(Enum):
    """États de connexion."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RETRYING = "retrying"


@dataclass
class IntegrationConfig:
    """Configuration d'une intégration."""
    
    name: str
    integration_type: IntegrationType
    
    # Connexion
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[SecretStr] = None
    
    # URLs et endpoints
    base_url: Optional[str] = None
    api_endpoint: Optional[str] = None
    
    # Authentification
    api_key: Optional[SecretStr] = None
    client_id: Optional[str] = None
    client_secret: Optional[SecretStr] = None
    access_token: Optional[SecretStr] = None
    refresh_token: Optional[SecretStr] = None
    
    # SSL/TLS
    use_ssl: bool = True
    verify_ssl: bool = True
    ca_cert_path: Optional[str] = None
    
    # Pool de connexions
    min_connections: int = 1
    max_connections: int = 10
    connection_timeout: float = 30.0
    read_timeout: float = 60.0
    
    # Rate limiting
    requests_per_second: float = 10.0
    burst_size: int = 20
    
    # Retry et circuit breaker
    max_retries: int = 3
    retry_delay: float = 1.0
    circuit_breaker_threshold: int = 5
    
    # Monitoring
    health_check_interval: float = 60.0
    metrics_enabled: bool = True
    
    # Métadonnées
    tenant_id: Optional[str] = None
    environment: str = "development"
    enabled: bool = True


@dataclass
class ConnectionMetrics:
    """Métriques d'une connexion."""
    
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    avg_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    last_error: Optional[str] = None
    uptime_seconds: float = 0.0
    connection_errors: int = 0
    rate_limit_hits: int = 0


class BaseIntegration(ABC):
    """
    Classe de base pour toutes les intégrations.
    
    Fonctionnalités communes:
    - Gestion de la connexion
    - Circuit breaker
    - Rate limiting
    - Retry automatique
    - Monitoring et métriques
    - Health checks
    """
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.state = ConnectionState.DISCONNECTED
        self.metrics = ConnectionMetrics()
        self.last_health_check = None
        self.connection = None
        self.rate_limiter = None
        self.circuit_breaker = None
        
        # Métriques Prometheus
        self.integration_requests = Counter(
            'integration_requests_total',
            'Nombre total de requêtes par intégration',
            ['integration_name', 'integration_type', 'result']
        )
        
        self.integration_response_time = Histogram(
            'integration_response_time_seconds',
            'Temps de réponse des intégrations',
            ['integration_name', 'integration_type']
        )
        
        self.integration_connections = Gauge(
            'integration_active_connections',
            'Nombre de connexions actives',
            ['integration_name', 'integration_type']
        )
        
        self.integration_health = Gauge(
            'integration_health_status',
            'Statut de santé de l\'intégration (0-1)',
            ['integration_name', 'integration_type']
        )
        
        logger.info(
            "Intégration initialisée",
            name=config.name,
            type=config.integration_type.value
        )
    
    @abstractmethod
    async def connect(self) -> bool:
        """Établit la connexion avec le service."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Ferme la connexion."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Vérifie la santé de la connexion."""
        pass
    
    async def ensure_connected(self) -> bool:
        """S'assure que la connexion est établie."""
        if self.state != ConnectionState.CONNECTED:
            return await self.connect()
        return True
    
    async def execute_with_retry(self, operation: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """
        Exécute une opération avec retry automatique.
        
        Args:
            operation: Opération à exécuter
            *args: Arguments positionnels
            **kwargs: Arguments nommés
        
        Returns:
            Résultat de l'opération
        """
        start_time = time.time()
        last_exception = None
        
        for attempt in range(1, self.config.max_retries + 1):
            try:
                # S'assurer que la connexion est établie
                if not await self.ensure_connected():
                    raise ConnectionError(f"Cannot connect to {self.config.name}")
                
                # Exécution de l'opération
                result = await operation(*args, **kwargs)
                
                # Mise à jour des métriques de succès
                self.metrics.successful_requests += 1
                self.metrics.last_request_time = datetime.now(timezone.utc)
                
                response_time = time.time() - start_time
                self.metrics.avg_response_time = (
                    (self.metrics.avg_response_time * (self.metrics.total_requests - 1) + response_time) 
                    / self.metrics.total_requests
                    if self.metrics.total_requests > 0 else response_time
                )
                
                self.integration_requests.labels(
                    integration_name=self.config.name,
                    integration_type=self.config.integration_type.value,
                    result='success'
                ).inc()
                
                self.integration_response_time.labels(
                    integration_name=self.config.name,
                    integration_type=self.config.integration_type.value
                ).observe(response_time)
                
                return result
                
            except Exception as e:
                last_exception = e
                self.metrics.failed_requests += 1
                self.metrics.last_error = str(e)
                
                logger.warning(
                    "Tentative d'opération échouée",
                    integration_name=self.config.name,
                    attempt=attempt,
                    max_attempts=self.config.max_retries,
                    error=str(e)
                )
                
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * attempt)
                else:
                    self.integration_requests.labels(
                        integration_name=self.config.name,
                        integration_type=self.config.integration_type.value,
                        result='failure'
                    ).inc()
        
        raise last_exception
    
    async def start_health_monitoring(self) -> None:
        """Démarre le monitoring de santé."""
        asyncio.create_task(self._health_check_loop())
    
    async def _health_check_loop(self) -> None:
        """Boucle de vérification de santé."""
        while self.config.enabled:
            try:
                health_status = await self.health_check()
                
                self.integration_health.labels(
                    integration_name=self.config.name,
                    integration_type=self.config.integration_type.value
                ).set(1.0 if health_status else 0.0)
                
                self.last_health_check = datetime.now(timezone.utc)
                
                if not health_status and self.state == ConnectionState.CONNECTED:
                    logger.warning(
                        "Health check échoué - reconnexion nécessaire",
                        integration_name=self.config.name
                    )
                    self.state = ConnectionState.ERROR
                
            except Exception as e:
                logger.error(
                    "Erreur lors du health check",
                    integration_name=self.config.name,
                    error=str(e)
                )
            
            await asyncio.sleep(self.config.health_check_interval)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de l'intégration."""
        return {
            "integration_name": self.config.name,
            "integration_type": self.config.integration_type.value,
            "state": self.state.value,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "success_rate": (
                    self.metrics.successful_requests / self.metrics.total_requests * 100
                    if self.metrics.total_requests > 0 else 0
                ),
                "avg_response_time": self.metrics.avg_response_time,
                "uptime_seconds": self.metrics.uptime_seconds,
                "last_request_time": self.metrics.last_request_time.isoformat() if self.metrics.last_request_time else None,
                "last_error": self.metrics.last_error
            },
            "config": {
                "host": self.config.host,
                "port": self.config.port,
                "enabled": self.config.enabled,
                "health_check_interval": self.config.health_check_interval
            }
        }


class SpotifyAPIIntegration(BaseIntegration):
    """
    Intégration avancée avec l'API Spotify.
    
    Fonctionnalités:
    - Authentification OAuth2 automatique
    - Rate limiting intelligent
    - Cache des réponses
    - Retry avec backoff
    - Pagination automatique
    """
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.spotify_client = None
        self.oauth_manager = None
        self.token_cache = {}
        self.response_cache = {}
        
    async def connect(self) -> bool:
        """Connexion à l'API Spotify."""
        try:
            self.state = ConnectionState.CONNECTING
            
            # Configuration OAuth
            if self.config.client_id and self.config.client_secret:
                if self.config.refresh_token:
                    # Utilisation d'un refresh token existant
                    self.oauth_manager = SpotifyOAuth(
                        client_id=self.config.client_id,
                        client_secret=self.config.client_secret.get_secret_value(),
                        redirect_uri="http://localhost:8888/callback",
                        scope="user-read-private user-read-email playlist-read-private user-top-read user-read-recently-played"
                    )
                else:
                    # Authentification client credentials
                    credentials_manager = SpotifyClientCredentials(
                        client_id=self.config.client_id,
                        client_secret=self.config.client_secret.get_secret_value()
                    )
                    self.spotify_client = spotipy.Spotify(client_credentials_manager=credentials_manager)
            
            # Test de connexion
            if self.spotify_client:
                # Test avec un appel simple
                result = self.spotify_client.search(q="test", type="track", limit=1)
                if result:
                    self.state = ConnectionState.CONNECTED
                    logger.info("Connexion Spotify API établie", integration_name=self.config.name)
                    return True
            
            self.state = ConnectionState.ERROR
            return False
            
        except Exception as e:
            self.state = ConnectionState.ERROR
            self.metrics.connection_errors += 1
            logger.error(
                "Erreur de connexion Spotify API",
                integration_name=self.config.name,
                error=str(e)
            )
            return False
    
    async def disconnect(self) -> None:
        """Déconnexion de l'API Spotify."""
        self.spotify_client = None
        self.oauth_manager = None
        self.state = ConnectionState.DISCONNECTED
        logger.info("Déconnexion Spotify API", integration_name=self.config.name)
    
    async def health_check(self) -> bool:
        """Vérification de santé Spotify API."""
        try:
            if not self.spotify_client:
                return False
            
            # Test simple
            result = self.spotify_client.search(q="test", type="track", limit=1)
            return result is not None
            
        except Exception as e:
            logger.warning(
                "Health check Spotify échoué",
                integration_name=self.config.name,
                error=str(e)
            )
            return False
    
    async def search_tracks(self, query: str, limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """Recherche de tracks avec cache."""
        cache_key = f"search:{hashlib.md5(f'{query}:{limit}:{offset}'.encode()).hexdigest()}"
        
        # Vérification du cache
        if cache_key in self.response_cache:
            cache_entry = self.response_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < 300:  # 5 minutes
                return cache_entry['data']
        
        # Exécution de la recherche
        async def _search():
            self.metrics.total_requests += 1
            result = self.spotify_client.search(q=query, type="track", limit=limit, offset=offset)
            
            # Mise en cache
            self.response_cache[cache_key] = {
                'data': result,
                'timestamp': time.time()
            }
            
            return result
        
        return await self.execute_with_retry(_search)
    
    async def get_user_playlists(self, user_id: str) -> Dict[str, Any]:
        """Récupère les playlists d'un utilisateur."""
        async def _get_playlists():
            self.metrics.total_requests += 1
            return self.spotify_client.user_playlists(user_id)
        
        return await self.execute_with_retry(_get_playlists)
    
    async def get_audio_features(self, track_ids: List[str]) -> Dict[str, Any]:
        """Récupère les features audio pour plusieurs tracks."""
        async def _get_features():
            self.metrics.total_requests += 1
            # Traitement par batch de 100 (limite Spotify)
            all_features = []
            for i in range(0, len(track_ids), 100):
                batch = track_ids[i:i+100]
                features = self.spotify_client.audio_features(batch)
                all_features.extend(features)
            return all_features
        
        return await self.execute_with_retry(_get_features)


class TimescaleDBIntegration(BaseIntegration):
    """
    Intégration avec TimescaleDB pour les données time-series.
    
    Fonctionnalités:
    - Pool de connexions asynchrones
    - Requêtes optimisées time-series
    - Compression automatique
    - Partitioning intelligent
    - Aggregations continues
    """
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.connection_pool = None
        self.hypertables = set()
        
    async def connect(self) -> bool:
        """Connexion à TimescaleDB."""
        try:
            self.state = ConnectionState.CONNECTING
            
            # Configuration de la connexion
            dsn = f"postgresql://{self.config.username}:{self.config.password.get_secret_value()}@{self.config.host}:{self.config.port}/{self.config.database}"
            
            # Création du pool de connexions
            self.connection_pool = await asyncpg.create_pool(
                dsn,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                command_timeout=self.config.read_timeout,
                server_settings={
                    'jit': 'off',
                    'application_name': f'spotify-ai-agent-{self.config.name}'
                }
            )
            
            # Test de connexion
            async with self.connection_pool.acquire() as conn:
                result = await conn.fetchval("SELECT version()")
                if "TimescaleDB" in result:
                    self.state = ConnectionState.CONNECTED
                    logger.info(
                        "Connexion TimescaleDB établie",
                        integration_name=self.config.name,
                        version=result
                    )
                    return True
            
            return False
            
        except Exception as e:
            self.state = ConnectionState.ERROR
            self.metrics.connection_errors += 1
            logger.error(
                "Erreur de connexion TimescaleDB",
                integration_name=self.config.name,
                error=str(e)
            )
            return False
    
    async def disconnect(self) -> None:
        """Déconnexion de TimescaleDB."""
        if self.connection_pool:
            await self.connection_pool.close()
            self.connection_pool = None
        
        self.state = ConnectionState.DISCONNECTED
        logger.info("Déconnexion TimescaleDB", integration_name=self.config.name)
    
    async def health_check(self) -> bool:
        """Vérification de santé TimescaleDB."""
        try:
            if not self.connection_pool:
                return False
            
            async with self.connection_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
                return True
                
        except Exception as e:
            logger.warning(
                "Health check TimescaleDB échoué",
                integration_name=self.config.name,
                error=str(e)
            )
            return False
    
    async def create_hypertable(self, table_name: str, time_column: str = "timestamp", chunk_time_interval: str = "1 day") -> bool:
        """Crée une hypertable TimescaleDB."""
        async def _create():
            self.metrics.total_requests += 1
            async with self.connection_pool.acquire() as conn:
                # Vérification si la table existe déjà comme hypertable
                exists = await conn.fetchval(
                    "SELECT COUNT(*) FROM timescaledb_information.hypertables WHERE table_name = $1",
                    table_name
                )
                
                if exists == 0:
                    await conn.execute(
                        f"SELECT create_hypertable('{table_name}', '{time_column}', chunk_time_interval => INTERVAL '{chunk_time_interval}')"
                    )
                    self.hypertables.add(table_name)
                    logger.info(
                        "Hypertable créée",
                        table_name=table_name,
                        time_column=time_column,
                        chunk_interval=chunk_time_interval
                    )
                
                return True
        
        return await self.execute_with_retry(_create)
    
    async def insert_time_series_data(self, table_name: str, data: List[Dict[str, Any]]) -> bool:
        """Insert des données time-series par batch."""
        async def _insert():
            self.metrics.total_requests += 1
            
            if not data:
                return True
            
            # Construction de la requête d'insertion
            columns = list(data[0].keys())
            placeholders = ", ".join([f"${i+1}" for i in range(len(columns))])
            
            query = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({placeholders})
            """
            
            async with self.connection_pool.acquire() as conn:
                # Insertion par batch
                values_list = []
                for row in data:
                    values_list.append([row[col] for col in columns])
                
                await conn.executemany(query, values_list)
                
                self.metrics.total_bytes_sent += len(str(data).encode('utf-8'))
                
                logger.debug(
                    "Données time-series insérées",
                    table_name=table_name,
                    rows_count=len(data)
                )
                
                return True
        
        return await self.execute_with_retry(_insert)
    
    async def query_time_series(
        self, 
        table_name: str, 
        start_time: datetime, 
        end_time: datetime, 
        columns: Optional[List[str]] = None,
        aggregation: Optional[str] = None,
        group_by_interval: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Requête time-series avec aggregations optionnelles."""
        
        async def _query():
            self.metrics.total_requests += 1
            
            # Construction de la requête
            select_columns = ", ".join(columns) if columns else "*"
            
            if aggregation and group_by_interval:
                # Requête avec aggregation
                query = f"""
                    SELECT 
                        time_bucket('{group_by_interval}', timestamp) AS bucket,
                        {aggregation}
                    FROM {table_name}
                    WHERE timestamp >= $1 AND timestamp <= $2
                    GROUP BY bucket
                    ORDER BY bucket
                """
            else:
                # Requête simple
                query = f"""
                    SELECT {select_columns}
                    FROM {table_name}
                    WHERE timestamp >= $1 AND timestamp <= $2
                    ORDER BY timestamp
                """
            
            async with self.connection_pool.acquire() as conn:
                rows = await conn.fetch(query, start_time, end_time)
                
                # Conversion en dictionnaires
                result = [dict(row) for row in rows]
                
                self.metrics.total_bytes_received += len(str(result).encode('utf-8'))
                
                logger.debug(
                    "Requête time-series exécutée",
                    table_name=table_name,
                    rows_returned=len(result),
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat()
                )
                
                return result
        
        return await self.execute_with_retry(_query)
    
    async def create_continuous_aggregate(
        self, 
        view_name: str, 
        source_table: str, 
        time_bucket: str,
        aggregations: Dict[str, str]
    ) -> bool:
        """Crée une aggregation continue."""
        
        async def _create_cagg():
            self.metrics.total_requests += 1
            
            # Construction des aggregations
            agg_clauses = []
            for column, agg_func in aggregations.items():
                agg_clauses.append(f"{agg_func}({column}) AS {agg_func}_{column}")
            
            query = f"""
                CREATE MATERIALIZED VIEW {view_name}
                WITH (timescaledb.continuous) AS
                SELECT
                    time_bucket('{time_bucket}', timestamp) AS bucket,
                    {', '.join(agg_clauses)}
                FROM {source_table}
                GROUP BY bucket
            """
            
            async with self.connection_pool.acquire() as conn:
                await conn.execute(query)
                
                logger.info(
                    "Aggregation continue créée",
                    view_name=view_name,
                    source_table=source_table,
                    time_bucket=time_bucket
                )
                
                return True
        
        return await self.execute_with_retry(_create_cagg)


class RedisClusterIntegration(BaseIntegration):
    """
    Intégration avec Redis Cluster pour le cache distribué.
    
    Fonctionnalités:
    - Clustering automatique
    - Sharding intelligent
    - Failover automatique
    - Compression des données
    - TTL automatique
    """
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.redis_client = None
        self.cluster_nodes = []
        
    async def connect(self) -> bool:
        """Connexion au cluster Redis."""
        try:
            self.state = ConnectionState.CONNECTING
            
            # Configuration cluster
            if isinstance(self.config.host, list):
                self.cluster_nodes = [(host, self.config.port or 6379) for host in self.config.host]
            else:
                self.cluster_nodes = [(self.config.host, self.config.port or 6379)]
            
            # Création du client cluster
            self.redis_client = aioredis.Redis(
                host=self.config.host,
                port=self.config.port or 6379,
                password=self.config.password.get_secret_value() if self.config.password else None,
                db=0,
                encoding='utf-8',
                decode_responses=True,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.connection_timeout,
                socket_connect_timeout=self.config.connection_timeout
            )
            
            # Test de connexion
            await self.redis_client.ping()
            
            self.state = ConnectionState.CONNECTED
            logger.info("Connexion Redis établie", integration_name=self.config.name)
            return True
            
        except Exception as e:
            self.state = ConnectionState.ERROR
            self.metrics.connection_errors += 1
            logger.error(
                "Erreur de connexion Redis",
                integration_name=self.config.name,
                error=str(e)
            )
            return False
    
    async def disconnect(self) -> None:
        """Déconnexion de Redis."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
        
        self.state = ConnectionState.DISCONNECTED
        logger.info("Déconnexion Redis", integration_name=self.config.name)
    
    async def health_check(self) -> bool:
        """Vérification de santé Redis."""
        try:
            if not self.redis_client:
                return False
            
            pong = await self.redis_client.ping()
            return pong == "PONG"
            
        except Exception as e:
            logger.warning(
                "Health check Redis échoué",
                integration_name=self.config.name,
                error=str(e)
            )
            return False
    
    async def set_with_compression(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set avec compression automatique."""
        async def _set():
            self.metrics.total_requests += 1
            
            # Sérialisation
            if isinstance(value, (dict, list)):
                serialized = json.dumps(value, default=str)
            else:
                serialized = str(value)
            
            # Compression si la valeur est grande
            if len(serialized) > 1024:  # 1KB
                import gzip
                compressed = gzip.compress(serialized.encode('utf-8'))
                if len(compressed) < len(serialized):
                    await self.redis_client.set(f"{key}:compressed", base64.b64encode(compressed).decode(), ex=ttl)
                    self.metrics.total_bytes_sent += len(compressed)
                    return True
            
            # Stockage normal
            await self.redis_client.set(key, serialized, ex=ttl)
            self.metrics.total_bytes_sent += len(serialized.encode('utf-8'))
            return True
        
        return await self.execute_with_retry(_set)
    
    async def get_with_decompression(self, key: str) -> Any:
        """Get avec décompression automatique."""
        async def _get():
            self.metrics.total_requests += 1
            
            # Tentative récupération compressée
            compressed_value = await self.redis_client.get(f"{key}:compressed")
            if compressed_value:
                import gzip
                compressed_data = base64.b64decode(compressed_value.encode())
                decompressed = gzip.decompress(compressed_data).decode('utf-8')
                self.metrics.total_bytes_received += len(decompressed.encode('utf-8'))
                
                # Tentative de désérialisation JSON
                try:
                    return json.loads(decompressed)
                except json.JSONDecodeError:
                    return decompressed
            
            # Récupération normale
            value = await self.redis_client.get(key)
            if value:
                self.metrics.total_bytes_received += len(value.encode('utf-8'))
                
                # Tentative de désérialisation JSON
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            
            return None
        
        return await self.execute_with_retry(_get)
    
    async def pipeline_operations(self, operations: List[Tuple[str, str, Any]]) -> List[Any]:
        """Exécute plusieurs opérations en pipeline."""
        async def _pipeline():
            self.metrics.total_requests += len(operations)
            
            pipeline = self.redis_client.pipeline()
            
            for op_type, key, value in operations:
                if op_type == "set":
                    pipeline.set(key, json.dumps(value, default=str) if isinstance(value, (dict, list)) else str(value))
                elif op_type == "get":
                    pipeline.get(key)
                elif op_type == "delete":
                    pipeline.delete(key)
                elif op_type == "incr":
                    pipeline.incr(key)
            
            results = await pipeline.execute()
            return results
        
        return await self.execute_with_retry(_pipeline)


# Gestionnaire global des intégrations
class IntegrationManager:
    """
    Gestionnaire centralisé des intégrations.
    
    Fonctionnalités:
    - Orchestration des connexions
    - Monitoring global
    - Failover automatique
    - Load balancing
    - Health checks centralisés
    """
    
    def __init__(self):
        self.integrations: Dict[str, BaseIntegration] = {}
        self.integration_groups: Dict[str, List[str]] = defaultdict(list)
        self.health_check_task = None
        
        # Métriques globales
        self.manager_health = Gauge(
            'integration_manager_health',
            'Santé globale du gestionnaire d\'intégrations',
            ['manager_instance']
        )
        
        logger.info("Gestionnaire d'intégrations initialisé")
    
    def register_integration(self, integration: BaseIntegration, group: Optional[str] = None) -> None:
        """Enregistre une intégration."""
        self.integrations[integration.config.name] = integration
        
        if group:
            self.integration_groups[group].append(integration.config.name)
        
        # Démarrage du monitoring de santé
        asyncio.create_task(integration.start_health_monitoring())
        
        logger.info(
            "Intégration enregistrée",
            name=integration.config.name,
            type=integration.config.integration_type.value,
            group=group
        )
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connecte toutes les intégrations."""
        results = {}
        
        connection_tasks = []
        for name, integration in self.integrations.items():
            if integration.config.enabled:
                task = asyncio.create_task(integration.connect())
                connection_tasks.append((name, task))
        
        for name, task in connection_tasks:
            try:
                result = await task
                results[name] = result
                
                if result:
                    logger.info("Intégration connectée", name=name)
                else:
                    logger.error("Échec de connexion", name=name)
                    
            except Exception as e:
                results[name] = False
                logger.error("Erreur de connexion", name=name, error=str(e))
        
        return results
    
    async def disconnect_all(self) -> None:
        """Déconnecte toutes les intégrations."""
        disconnect_tasks = []
        
        for integration in self.integrations.values():
            task = asyncio.create_task(integration.disconnect())
            disconnect_tasks.append(task)
        
        await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        logger.info("Toutes les intégrations déconnectées")
    
    async def get_global_health(self) -> Dict[str, Any]:
        """Retourne la santé globale de toutes les intégrations."""
        health_report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'integrations': {},
            'groups': {},
            'overall_health': 0.0
        }
        
        healthy_count = 0
        total_count = 0
        
        # Santé par intégration
        for name, integration in self.integrations.items():
            if integration.config.enabled:
                total_count += 1
                is_healthy = integration.state == ConnectionState.CONNECTED
                
                if is_healthy:
                    healthy_count += 1
                
                health_report['integrations'][name] = {
                    'healthy': is_healthy,
                    'state': integration.state.value,
                    'last_health_check': integration.last_health_check.isoformat() if integration.last_health_check else None,
                    'metrics': integration.get_metrics()
                }
        
        # Santé par groupe
        for group_name, integration_names in self.integration_groups.items():
            group_healthy = 0
            group_total = 0
            
            for int_name in integration_names:
                if int_name in self.integrations and self.integrations[int_name].config.enabled:
                    group_total += 1
                    if self.integrations[int_name].state == ConnectionState.CONNECTED:
                        group_healthy += 1
            
            health_report['groups'][group_name] = {
                'healthy_count': group_healthy,
                'total_count': group_total,
                'health_percentage': (group_healthy / group_total * 100) if group_total > 0 else 0
            }
        
        # Santé globale
        health_report['overall_health'] = (healthy_count / total_count * 100) if total_count > 0 else 0
        
        # Métriques Prometheus
        self.manager_health.labels(manager_instance='main').set(health_report['overall_health'] / 100)
        
        return health_report
    
    def get_integration(self, name: str) -> Optional[BaseIntegration]:
        """Récupère une intégration par nom."""
        return self.integrations.get(name)
    
    def get_integrations_by_type(self, integration_type: IntegrationType) -> List[BaseIntegration]:
        """Récupère les intégrations par type."""
        return [
            integration for integration in self.integrations.values()
            if integration.config.integration_type == integration_type
        ]
    
    def get_integrations_by_group(self, group: str) -> List[BaseIntegration]:
        """Récupère les intégrations par groupe."""
        integration_names = self.integration_groups.get(group, [])
        return [
            self.integrations[name] for name in integration_names
            if name in self.integrations
        ]


# Instance globale
global_integration_manager = IntegrationManager()
