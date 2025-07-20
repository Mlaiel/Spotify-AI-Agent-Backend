#!/usr/bin/env python3
"""
Integration Hub Enterprise

Hub d'intégration intelligent pour la connectivité multi-services, APIs externes
et orchestration des échanges de données avec gouvernance automatisée.

Architecture:
✅ Lead Dev + Architecte IA - Intégrations intelligentes distribuées
✅ Développeur Backend Senior - APIs async haute performance
✅ Ingénieur Machine Learning - Mapping automatique et ML routing
✅ DBA & Data Engineer - Pipelines de données intégrées
✅ Spécialiste Sécurité Backend - Sécurité des intégrations
✅ Architecte Microservices - Service mesh et API gateway

Fonctionnalités Enterprise:
- API Gateway intelligent avec routing ML
- Service mesh avec discovery automatique
- Data mapping et transformation automatiques
- Connecteurs universels multi-protocoles
- Monitoring et observabilité des intégrations
- Rate limiting et circuit breakers
- Authentication et authorization centralisées
- Event streaming et messaging distribué
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Set, Protocol
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import aiohttp
from urllib.parse import urljoin, urlparse
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import ssl
import certifi

# Imports pour messaging et streaming
import aioredis
import aiocache
from asyncio import Queue, Event, Lock, Semaphore
import websockets
import pika
import kafka
from kafka import KafkaProducer, KafkaConsumer

# Imports ML et transformation
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Imports sécurité et crypto
import hashlib
import hmac
import base64
import jwt
from cryptography.fernet import Fernet

# Configuration du logging
logger = logging.getLogger(__name__)

class IntegrationType(Enum):
    """Types d'intégrations supportées."""
    REST_API = "rest_api"
    GRAPHQL = "graphql"
    WEBSOCKET = "websocket"
    WEBHOOK = "webhook"
    MESSAGE_QUEUE = "message_queue"
    KAFKA = "kafka"
    REDIS_STREAM = "redis_stream"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    GRPC = "grpc"
    CUSTOM = "custom"

class AuthenticationType(Enum):
    """Types d'authentification supportés."""
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    JWT = "jwt"
    CUSTOM = "custom"

class DataFormat(Enum):
    """Formats de données supportés."""
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    YAML = "yaml"
    PROTOBUF = "protobuf"
    AVRO = "avro"
    PARQUET = "parquet"
    BINARY = "binary"

class ConnectionStatus(Enum):
    """Statuts de connexion."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"
    RETRYING = "retrying"
    AUTHENTICATED = "authenticated"

@dataclass
class EndpointConfig:
    """Configuration d'un endpoint d'intégration."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    type: IntegrationType = IntegrationType.REST_API
    url: str = ""
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    auth_type: AuthenticationType = AuthenticationType.NONE
    auth_config: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    rate_limit: Optional[int] = None
    data_format: DataFormat = DataFormat.JSON
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataMapping:
    """Configuration de mapping de données."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    source_format: DataFormat = DataFormat.JSON
    target_format: DataFormat = DataFormat.JSON
    transformation_rules: Dict[str, Any] = field(default_factory=dict)
    field_mappings: Dict[str, str] = field(default_factory=dict)
    validation_schema: Optional[Dict[str, Any]] = None
    filters: List[Dict[str, Any]] = field(default_factory=list)
    enrichment_rules: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class IntegrationMetrics:
    """Métriques d'intégration."""
    endpoint_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None
    rate_limit_hits: int = 0
    circuit_breaker_trips: int = 0
    data_volume_bytes: int = 0

class CircuitBreaker:
    """Circuit breaker pour protection des intégrations."""
    
    def __init__(self, 
                 failure_threshold: int = 5, 
                 timeout: float = 60.0,
                 success_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self._lock = threading.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Exécute une fonction avec protection circuit breaker."""
        with self._lock:
            if self.state == "open":
                if (self.last_failure_time and 
                    (datetime.utcnow() - self.last_failure_time).total_seconds() > self.timeout):
                    self.state = "half-open"
                    self.success_count = 0
                else:
                    raise Exception("Circuit breaker is open")
            
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                if self.state == "half-open":
                    self.success_count += 1
                    if self.success_count >= self.success_threshold:
                        self.state = "closed"
                        self.failure_count = 0
                
                return result
            
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = datetime.utcnow()
                
                if self.state == "half-open" or self.failure_count >= self.failure_threshold:
                    self.state = "open"
                
                raise e

class RateLimiter:
    """Rate limiter pour les intégrations."""
    
    def __init__(self, max_requests: int, time_window: float = 60.0):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self._lock = threading.Lock()
    
    async def acquire(self) -> bool:
        """Tente d'acquérir un slot pour une requête."""
        with self._lock:
            now = datetime.utcnow()
            
            # Nettoyage des requêtes anciennes
            while self.requests and (now - self.requests[0]).total_seconds() > self.time_window:
                self.requests.popleft()
            
            # Vérification de la limite
            if len(self.requests) >= self.max_requests:
                return False
            
            # Ajout de la requête actuelle
            self.requests.append(now)
            return True

class DataTransformer:
    """Transformateur de données intelligent avec ML."""
    
    def __init__(self):
        self.field_similarity_cache = {}
        self.transformation_patterns = {}
        self.vectorizer = TfidfVectorizer()
        self._lock = threading.Lock()
    
    async def transform_data(self, 
                           data: Any, 
                           mapping: DataMapping) -> Any:
        """Transforme les données selon le mapping fourni."""
        try:
            # Conversion du format source
            normalized_data = await self._normalize_input_data(data, mapping.source_format)
            
            # Application des filtres
            filtered_data = await self._apply_filters(normalized_data, mapping.filters)
            
            # Mapping des champs
            mapped_data = await self._map_fields(filtered_data, mapping.field_mappings)
            
            # Enrichissement
            enriched_data = await self._enrich_data(mapped_data, mapping.enrichment_rules)
            
            # Validation
            if mapping.validation_schema:
                await self._validate_data(enriched_data, mapping.validation_schema)
            
            # Conversion au format cible
            final_data = await self._format_output_data(enriched_data, mapping.target_format)
            
            return final_data
        
        except Exception as e:
            logger.error(f"Erreur de transformation de données: {e}")
            raise
    
    async def auto_discover_mapping(self, 
                                   source_schema: Dict[str, Any], 
                                   target_schema: Dict[str, Any]) -> Dict[str, str]:
        """Découverte automatique de mapping entre schémas."""
        mapping = {}
        
        source_fields = self._extract_field_names(source_schema)
        target_fields = self._extract_field_names(target_schema)
        
        if not source_fields or not target_fields:
            return mapping
        
        # Calcul de similarité entre champs
        try:
            # Vectorisation des noms de champs
            all_fields = source_fields + target_fields
            field_vectors = self.vectorizer.fit_transform(all_fields)
            
            source_vectors = field_vectors[:len(source_fields)]
            target_vectors = field_vectors[len(source_fields):]
            
            # Calcul des similarités
            similarities = cosine_similarity(source_vectors, target_vectors)
            
            # Mapping basé sur la similarité maximale
            for i, source_field in enumerate(source_fields):
                max_similarity_idx = np.argmax(similarities[i])
                max_similarity = similarities[i][max_similarity_idx]
                
                if max_similarity > 0.7:  # Seuil de similarité
                    target_field = target_fields[max_similarity_idx]
                    mapping[source_field] = target_field
        
        except Exception as e:
            logger.warning(f"Erreur dans la découverte automatique de mapping: {e}")
        
        return mapping
    
    def _extract_field_names(self, schema: Dict[str, Any]) -> List[str]:
        """Extrait les noms de champs d'un schéma."""
        fields = []
        
        def extract_recursive(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    field_name = f"{prefix}.{key}" if prefix else key
                    fields.append(field_name)
                    
                    if isinstance(value, dict):
                        extract_recursive(value, field_name)
        
        extract_recursive(schema)
        return fields
    
    async def _normalize_input_data(self, data: Any, format: DataFormat) -> Dict[str, Any]:
        """Normalise les données d'entrée au format dict."""
        if format == DataFormat.JSON:
            if isinstance(data, str):
                return json.loads(data)
            return data
        elif format == DataFormat.XML:
            # Conversion XML vers dict (implémentation simplifiée)
            return {"xml_data": str(data)}
        elif format == DataFormat.CSV:
            # Conversion CSV vers dict
            return {"csv_data": str(data)}
        else:
            return {"raw_data": data}
    
    async def _apply_filters(self, 
                           data: Dict[str, Any], 
                           filters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Applique les filtres aux données."""
        filtered_data = data.copy()
        
        for filter_rule in filters:
            filter_type = filter_rule.get("type")
            
            if filter_type == "exclude_fields":
                fields_to_exclude = filter_rule.get("fields", [])
                for field in fields_to_exclude:
                    filtered_data.pop(field, None)
            
            elif filter_type == "include_only":
                fields_to_include = filter_rule.get("fields", [])
                filtered_data = {k: v for k, v in filtered_data.items() if k in fields_to_include}
            
            elif filter_type == "value_filter":
                field = filter_rule.get("field")
                condition = filter_rule.get("condition")
                value = filter_rule.get("value")
                
                if field in filtered_data:
                    field_value = filtered_data[field]
                    
                    if condition == "equals" and field_value != value:
                        filtered_data.pop(field, None)
                    elif condition == "not_equals" and field_value == value:
                        filtered_data.pop(field, None)
        
        return filtered_data
    
    async def _map_fields(self, 
                         data: Dict[str, Any], 
                         mappings: Dict[str, str]) -> Dict[str, Any]:
        """Mappe les champs selon les règles fournies."""
        mapped_data = {}
        
        for source_field, target_field in mappings.items():
            if source_field in data:
                mapped_data[target_field] = data[source_field]
        
        # Ajout des champs non mappés
        for key, value in data.items():
            if key not in mappings and key not in mapped_data:
                mapped_data[key] = value
        
        return mapped_data
    
    async def _enrich_data(self, 
                          data: Dict[str, Any], 
                          enrichment_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enrichit les données selon les règles."""
        enriched_data = data.copy()
        
        for rule in enrichment_rules:
            rule_type = rule.get("type")
            
            if rule_type == "add_timestamp":
                field_name = rule.get("field", "timestamp")
                enriched_data[field_name] = datetime.utcnow().isoformat()
            
            elif rule_type == "add_uuid":
                field_name = rule.get("field", "id")
                enriched_data[field_name] = str(uuid.uuid4())
            
            elif rule_type == "concatenate_fields":
                source_fields = rule.get("source_fields", [])
                target_field = rule.get("target_field")
                separator = rule.get("separator", " ")
                
                if target_field and all(field in enriched_data for field in source_fields):
                    values = [str(enriched_data[field]) for field in source_fields]
                    enriched_data[target_field] = separator.join(values)
            
            elif rule_type == "calculate_field":
                expression = rule.get("expression")
                target_field = rule.get("target_field")
                
                if expression and target_field:
                    try:
                        # Évaluation sécurisée d'expression (à améliorer)
                        result = eval(expression, {"__builtins__": {}}, enriched_data)
                        enriched_data[target_field] = result
                    except Exception as e:
                        logger.warning(f"Erreur dans le calcul de champ: {e}")
        
        return enriched_data
    
    async def _validate_data(self, 
                           data: Dict[str, Any], 
                           schema: Dict[str, Any]) -> None:
        """Valide les données contre un schéma."""
        # Implémentation basique de validation
        required_fields = schema.get("required", [])
        
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Champ requis manquant: {field}")
        
        # Validation des types
        field_types = schema.get("properties", {})
        for field, type_def in field_types.items():
            if field in data:
                expected_type = type_def.get("type")
                field_value = data[field]
                
                if expected_type == "string" and not isinstance(field_value, str):
                    raise ValueError(f"Type incorrect pour {field}: attendu string")
                elif expected_type == "number" and not isinstance(field_value, (int, float)):
                    raise ValueError(f"Type incorrect pour {field}: attendu number")
                elif expected_type == "boolean" and not isinstance(field_value, bool):
                    raise ValueError(f"Type incorrect pour {field}: attendu boolean")
    
    async def _format_output_data(self, 
                                 data: Dict[str, Any], 
                                 format: DataFormat) -> Any:
        """Formate les données de sortie."""
        if format == DataFormat.JSON:
            return data
        elif format == DataFormat.XML:
            # Conversion vers XML (implémentation simplifiée)
            return f"<root>{json.dumps(data)}</root>"
        elif format == DataFormat.CSV:
            # Conversion vers CSV (implémentation simplifiée)
            if isinstance(data, dict):
                return ",".join([f"{k}:{v}" for k, v in data.items()])
        else:
            return str(data)

class IntegrationConnector:
    """Connecteur générique pour intégrations."""
    
    def __init__(self, config: EndpointConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter(config.rate_limit) if config.rate_limit else None
        self.metrics = IntegrationMetrics(endpoint_id=config.id)
        self.status = ConnectionStatus.DISCONNECTED
        self._lock = Lock()
    
    async def initialize(self) -> None:
        """Initialise le connecteur."""
        try:
            # Configuration SSL
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            
            # Configuration du timeout
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            
            # Création de la session
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(ssl=ssl_context)
            )
            
            self.status = ConnectionStatus.CONNECTED
            logger.info(f"Connecteur {self.config.name} initialisé")
            
        except Exception as e:
            self.status = ConnectionStatus.FAILED
            logger.error(f"Erreur d'initialisation du connecteur {self.config.name}: {e}")
            raise
    
    async def execute_request(self, 
                             data: Optional[Any] = None,
                             override_params: Optional[Dict[str, Any]] = None) -> Any:
        """Exécute une requête via le connecteur."""
        if not self.session:
            await self.initialize()
        
        # Vérification du rate limiting
        if self.rate_limiter and not await self.rate_limiter.acquire():
            self.metrics.rate_limit_hits += 1
            raise Exception("Rate limit exceeded")
        
        start_time = datetime.utcnow()
        
        try:
            # Exécution avec circuit breaker
            result = await self.circuit_breaker.call(self._execute_http_request, data, override_params)
            
            # Mise à jour des métriques de succès
            self.metrics.successful_requests += 1
            self.metrics.last_success_time = datetime.utcnow()
            
            return result
        
        except Exception as e:
            # Mise à jour des métriques d'échec
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = datetime.utcnow()
            
            if "Circuit breaker is open" in str(e):
                self.metrics.circuit_breaker_trips += 1
            
            logger.error(f"Erreur dans l'exécution de requête {self.config.name}: {e}")
            raise
        
        finally:
            # Mise à jour des métriques générales
            self.metrics.total_requests += 1
            self.metrics.last_request_time = datetime.utcnow()
            
            # Calcul du temps de réponse moyen
            response_time = (datetime.utcnow() - start_time).total_seconds()
            if self.metrics.total_requests > 1:
                self.metrics.avg_response_time = (
                    (self.metrics.avg_response_time * (self.metrics.total_requests - 1) + response_time) /
                    self.metrics.total_requests
                )
            else:
                self.metrics.avg_response_time = response_time
    
    async def _execute_http_request(self, 
                                   data: Optional[Any] = None,
                                   override_params: Optional[Dict[str, Any]] = None) -> Any:
        """Exécute une requête HTTP."""
        if not self.session:
            raise Exception("Session non initialisée")
        
        # Préparation des headers
        headers = self.config.headers.copy()
        await self._add_authentication_headers(headers)
        
        # Préparation des paramètres
        params = self.config.parameters.copy()
        if override_params:
            params.update(override_params)
        
        # Préparation du body
        request_data = None
        if data is not None:
            if self.config.data_format == DataFormat.JSON:
                request_data = json.dumps(data) if not isinstance(data, str) else data
                headers['Content-Type'] = 'application/json'
            else:
                request_data = str(data)
        
        # Exécution de la requête
        async with self.session.request(
            method=self.config.method,
            url=self.config.url,
            headers=headers,
            params=params,
            data=request_data
        ) as response:
            
            # Vérification du statut
            response.raise_for_status()
            
            # Parsing de la réponse
            response_text = await response.text()
            self.metrics.data_volume_bytes += len(response_text.encode('utf-8'))
            
            if self.config.data_format == DataFormat.JSON:
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    return response_text
            else:
                return response_text
    
    async def _add_authentication_headers(self, headers: Dict[str, str]) -> None:
        """Ajoute les headers d'authentification."""
        auth_config = self.config.auth_config
        
        if self.config.auth_type == AuthenticationType.API_KEY:
            key_name = auth_config.get('key_name', 'X-API-Key')
            api_key = auth_config.get('api_key')
            if api_key:
                headers[key_name] = api_key
        
        elif self.config.auth_type == AuthenticationType.BEARER_TOKEN:
            token = auth_config.get('token')
            if token:
                headers['Authorization'] = f'Bearer {token}'
        
        elif self.config.auth_type == AuthenticationType.BASIC_AUTH:
            username = auth_config.get('username')
            password = auth_config.get('password')
            if username and password:
                credentials = base64.b64encode(f'{username}:{password}'.encode()).decode()
                headers['Authorization'] = f'Basic {credentials}'
        
        elif self.config.auth_type == AuthenticationType.JWT:
            token = auth_config.get('jwt_token')
            if token:
                headers['Authorization'] = f'Bearer {token}'
    
    async def test_connection(self) -> bool:
        """Teste la connexion au endpoint."""
        try:
            await self.execute_request()
            return True
        except Exception as e:
            logger.warning(f"Test de connexion échoué pour {self.config.name}: {e}")
            return False
    
    async def close(self) -> None:
        """Ferme le connecteur."""
        if self.session:
            await self.session.close()
            self.session = None
        
        self.status = ConnectionStatus.DISCONNECTED
        logger.info(f"Connecteur {self.config.name} fermé")

class IntegrationHub:
    """
    Hub d'intégration Enterprise avec intelligence artificielle.
    
    Fonctionnalités:
    - Gestion centralisée des intégrations multi-protocoles
    - Transformation automatique des données avec ML
    - Monitoring et observabilité complète
    - Circuit breakers et rate limiting
    - Authentification et autorisation centralisées
    - Découverte automatique de services
    - Event streaming et messaging distribué
    """
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 enable_ml_mapping: bool = True):
        self.redis_url = redis_url
        self.enable_ml_mapping = enable_ml_mapping
        
        # Composants principaux
        self.data_transformer = DataTransformer()
        
        # Registres
        self.endpoints: Dict[str, EndpointConfig] = {}
        self.connectors: Dict[str, IntegrationConnector] = {}
        self.data_mappings: Dict[str, DataMapping] = {}
        
        # Métriques globales
        self.global_metrics = {
            'total_integrations': 0,
            'active_connections': 0,
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'data_volume_mb': 0.0
        }
        
        # Composants async
        self.redis_client: Optional[aioredis.Redis] = None
        self.cache = aiocache.Cache(aiocache.Cache.MEMORY)
        
        # Event system
        self.event_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_queue: Queue = Queue()
        
        # Synchronisation
        self._hub_lock = Lock()
        self.shutdown_event = Event()
        
        # Workers
        self.worker_executor = ThreadPoolExecutor(max_workers=5)
        
        logger.info("Hub d'intégration initialisé")
    
    async def initialize(self) -> None:
        """Initialise le hub d'intégration."""
        try:
            # Connexion Redis
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Chargement des configurations persistées
            await self._load_configurations()
            
            # Démarrage des workers
            self._start_workers()
            
            logger.info("Hub d'intégration initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur d'initialisation du hub d'intégration: {e}")
            raise
    
    def _start_workers(self) -> None:
        """Démarre les workers du hub."""
        # Worker d'événements
        asyncio.create_task(self._event_worker())
        
        # Worker de monitoring
        asyncio.create_task(self._monitoring_worker())
        
        # Worker de health checks
        asyncio.create_task(self._health_check_worker())
        
        # Worker de métriques
        asyncio.create_task(self._metrics_worker())
    
    async def register_endpoint(self, config: EndpointConfig) -> str:
        """Enregistre un nouveau endpoint d'intégration."""
        async with self._hub_lock:
            # Validation de la configuration
            if not await self._validate_endpoint_config(config):
                raise ValueError("Configuration d'endpoint invalide")
            
            # Stockage de la configuration
            self.endpoints[config.id] = config
            
            # Création du connecteur
            connector = IntegrationConnector(config)
            await connector.initialize()
            self.connectors[config.id] = connector
            
            # Persistance Redis
            if self.redis_client:
                config_data = await self._serialize_endpoint_config(config)
                await self.redis_client.hset(
                    "integration:endpoints",
                    config.id,
                    json.dumps(config_data, default=str)
                )
            
            # Mise à jour des métriques
            self.global_metrics['total_integrations'] += 1
            self.global_metrics['active_connections'] += 1
            
            # Événement
            await self._emit_event("endpoint_registered", {
                "endpoint_id": config.id,
                "endpoint_name": config.name,
                "endpoint_type": config.type.value
            })
            
            logger.info(f"Endpoint {config.name} enregistré (ID: {config.id})")
            return config.id
    
    async def unregister_endpoint(self, endpoint_id: str) -> bool:
        """Désenregistre un endpoint."""
        async with self._hub_lock:
            if endpoint_id not in self.endpoints:
                return False
            
            # Fermeture du connecteur
            if endpoint_id in self.connectors:
                await self.connectors[endpoint_id].close()
                del self.connectors[endpoint_id]
            
            # Suppression de la configuration
            config = self.endpoints.pop(endpoint_id)
            
            # Suppression Redis
            if self.redis_client:
                await self.redis_client.hdel("integration:endpoints", endpoint_id)
            
            # Mise à jour des métriques
            self.global_metrics['active_connections'] -= 1
            
            # Événement
            await self._emit_event("endpoint_unregistered", {
                "endpoint_id": endpoint_id,
                "endpoint_name": config.name
            })
            
            logger.info(f"Endpoint {config.name} désenregistré")
            return True
    
    async def execute_integration(self, 
                                 endpoint_id: str,
                                 data: Optional[Any] = None,
                                 mapping_id: Optional[str] = None,
                                 parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Exécute une intégration."""
        if endpoint_id not in self.connectors:
            raise ValueError(f"Endpoint {endpoint_id} non trouvé")
        
        connector = self.connectors[endpoint_id]
        start_time = datetime.utcnow()
        
        try:
            # Transformation des données si mapping fourni
            transformed_data = data
            if mapping_id and mapping_id in self.data_mappings:
                mapping = self.data_mappings[mapping_id]
                transformed_data = await self.data_transformer.transform_data(data, mapping)
            
            # Exécution de la requête
            result = await connector.execute_request(transformed_data, parameters)
            
            # Mise à jour des métriques globales
            self.global_metrics['successful_requests'] += 1
            self.global_metrics['total_requests'] += 1
            
            # Événement de succès
            await self._emit_event("integration_success", {
                "endpoint_id": endpoint_id,
                "execution_time": (datetime.utcnow() - start_time).total_seconds(),
                "data_size": len(str(result))
            })
            
            return result
        
        except Exception as e:
            # Mise à jour des métriques d'échec
            self.global_metrics['failed_requests'] += 1
            self.global_metrics['total_requests'] += 1
            
            # Événement d'échec
            await self._emit_event("integration_failure", {
                "endpoint_id": endpoint_id,
                "error": str(e),
                "execution_time": (datetime.utcnow() - start_time).total_seconds()
            })
            
            logger.error(f"Erreur d'intégration {endpoint_id}: {e}")
            raise
    
    async def create_data_mapping(self, mapping: DataMapping) -> str:
        """Crée un mapping de données."""
        # Stockage du mapping
        self.data_mappings[mapping.id] = mapping
        
        # Persistance Redis
        if self.redis_client:
            mapping_data = await self._serialize_data_mapping(mapping)
            await self.redis_client.hset(
                "integration:mappings",
                mapping.id,
                json.dumps(mapping_data, default=str)
            )
        
        logger.info(f"Mapping de données {mapping.name} créé (ID: {mapping.id})")
        return mapping.id
    
    async def auto_discover_integration(self, 
                                       base_url: str,
                                       auth_config: Optional[Dict[str, Any]] = None) -> List[EndpointConfig]:
        """Découvre automatiquement les endpoints d'intégration."""
        discovered_endpoints = []
        
        try:
            # Tentative de découverte d'API REST
            rest_endpoints = await self._discover_rest_endpoints(base_url, auth_config)
            discovered_endpoints.extend(rest_endpoints)
            
            # Tentative de découverte GraphQL
            graphql_endpoint = await self._discover_graphql_endpoint(base_url, auth_config)
            if graphql_endpoint:
                discovered_endpoints.append(graphql_endpoint)
            
            # Tentative de découverte WebSocket
            ws_endpoint = await self._discover_websocket_endpoint(base_url, auth_config)
            if ws_endpoint:
                discovered_endpoints.append(ws_endpoint)
            
            logger.info(f"Découverte automatique terminée: {len(discovered_endpoints)} endpoints trouvés")
            
        except Exception as e:
            logger.error(f"Erreur lors de la découverte automatique: {e}")
        
        return discovered_endpoints
    
    async def get_integration_metrics(self, endpoint_id: Optional[str] = None) -> Dict[str, Any]:
        """Récupère les métriques d'intégration."""
        if endpoint_id:
            if endpoint_id in self.connectors:
                connector = self.connectors[endpoint_id]
                return {
                    "endpoint_id": endpoint_id,
                    "endpoint_name": self.endpoints[endpoint_id].name,
                    "status": connector.status.value,
                    "metrics": {
                        "total_requests": connector.metrics.total_requests,
                        "successful_requests": connector.metrics.successful_requests,
                        "failed_requests": connector.metrics.failed_requests,
                        "success_rate": (connector.metrics.successful_requests / max(connector.metrics.total_requests, 1)) * 100,
                        "avg_response_time": connector.metrics.avg_response_time,
                        "rate_limit_hits": connector.metrics.rate_limit_hits,
                        "circuit_breaker_trips": connector.metrics.circuit_breaker_trips,
                        "data_volume_bytes": connector.metrics.data_volume_bytes,
                        "last_request_time": connector.metrics.last_request_time.isoformat() if connector.metrics.last_request_time else None
                    }
                }
            else:
                return {"error": f"Endpoint {endpoint_id} non trouvé"}
        else:
            # Métriques globales
            total_requests = sum(c.metrics.total_requests for c in self.connectors.values())
            if total_requests > 0:
                self.global_metrics['avg_response_time'] = sum(
                    c.metrics.avg_response_time * c.metrics.total_requests 
                    for c in self.connectors.values()
                ) / total_requests
            
            # Volume de données en MB
            total_bytes = sum(c.metrics.data_volume_bytes for c in self.connectors.values())
            self.global_metrics['data_volume_mb'] = total_bytes / (1024 * 1024)
            
            return self.global_metrics.copy()
    
    async def subscribe_to_events(self, event_type: str, callback: Callable) -> None:
        """S'abonne aux événements du hub."""
        self.event_subscribers[event_type].append(callback)
        logger.info(f"Nouvel abonné pour les événements {event_type}")
    
    async def batch_execute_integrations(self, 
                                        requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Exécute plusieurs intégrations en batch."""
        results = []
        
        # Exécution parallèle des requêtes
        tasks = []
        for i, request in enumerate(requests):
            task = asyncio.create_task(
                self._execute_single_batch_request(i, request)
            )
            tasks.append(task)
        
        # Attente des résultats
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                results.append({
                    "request_index": i,
                    "success": False,
                    "error": str(result)
                })
            else:
                results.append(result)
        
        return results
    
    # Workers asynchrones
    async def _event_worker(self) -> None:
        """Worker pour le traitement des événements."""
        while not self.shutdown_event.is_set():
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._process_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Erreur dans le worker d'événements: {e}")
    
    async def _monitoring_worker(self) -> None:
        """Worker de monitoring des intégrations."""
        while not self.shutdown_event.is_set():
            try:
                # Monitoring des connecteurs
                for endpoint_id, connector in self.connectors.items():
                    await self._monitor_connector_health(endpoint_id, connector)
                
                await asyncio.sleep(60)  # Toutes les minutes
                
            except Exception as e:
                logger.error(f"Erreur dans le worker de monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _health_check_worker(self) -> None:
        """Worker de vérification de santé."""
        while not self.shutdown_event.is_set():
            try:
                # Health checks sur tous les endpoints
                health_results = {}
                
                for endpoint_id, connector in self.connectors.items():
                    try:
                        is_healthy = await connector.test_connection()
                        health_results[endpoint_id] = {
                            "healthy": is_healthy,
                            "checked_at": datetime.utcnow().isoformat()
                        }
                    except Exception as e:
                        health_results[endpoint_id] = {
                            "healthy": False,
                            "error": str(e),
                            "checked_at": datetime.utcnow().isoformat()
                        }
                
                # Stockage des résultats
                if self.redis_client and health_results:
                    await self.redis_client.setex(
                        "integration:health_status",
                        300,  # TTL 5 minutes
                        json.dumps(health_results)
                    )
                
                await asyncio.sleep(300)  # Toutes les 5 minutes
                
            except Exception as e:
                logger.error(f"Erreur dans le worker de health check: {e}")
                await asyncio.sleep(600)
    
    async def _metrics_worker(self) -> None:
        """Worker de collection des métriques."""
        while not self.shutdown_event.is_set():
            try:
                # Collection et agrégation des métriques
                all_metrics = {}
                
                for endpoint_id, connector in self.connectors.items():
                    endpoint_metrics = {
                        "total_requests": connector.metrics.total_requests,
                        "successful_requests": connector.metrics.successful_requests,
                        "failed_requests": connector.metrics.failed_requests,
                        "avg_response_time": connector.metrics.avg_response_time,
                        "data_volume_bytes": connector.metrics.data_volume_bytes,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    all_metrics[endpoint_id] = endpoint_metrics
                
                # Stockage des métriques
                if self.redis_client and all_metrics:
                    await self.redis_client.setex(
                        "integration:metrics",
                        300,
                        json.dumps(all_metrics)
                    )
                
                await asyncio.sleep(120)  # Toutes les 2 minutes
                
            except Exception as e:
                logger.error(f"Erreur dans le worker de métriques: {e}")
                await asyncio.sleep(240)
    
    # Méthodes utilitaires
    async def _emit_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Émet un événement vers les abonnés."""
        await self.event_queue.put({
            "type": event_type,
            "data": event_data,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def _process_event(self, event: Dict[str, Any]) -> None:
        """Traite un événement."""
        event_type = event["type"]
        
        if event_type in self.event_subscribers:
            for callback in self.event_subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Erreur dans le callback d'événement {event_type}: {e}")
    
    async def shutdown(self) -> None:
        """Arrête le hub d'intégration."""
        self.shutdown_event.set()
        
        # Fermeture de tous les connecteurs
        for connector in self.connectors.values():
            await connector.close()
        
        # Fermeture des ressources
        if self.redis_client:
            await self.redis_client.close()
        
        self.worker_executor.shutdown(wait=True)
        
        logger.info("Hub d'intégration arrêté")

# Factory et utilitaires
def create_integration_hub(config: Dict[str, Any]) -> IntegrationHub:
    """
    Factory pour créer un hub d'intégration configuré.
    
    Args:
        config: Configuration du hub
        
    Returns:
        Instance du hub d'intégration
    """
    return IntegrationHub(
        redis_url=config.get("redis_url", "redis://localhost:6379"),
        enable_ml_mapping=config.get("enable_ml_mapping", True)
    )

# Templates de configuration prédéfinis
PREDEFINED_ENDPOINT_TEMPLATES = {
    "slack_webhook": {
        "name": "Slack Webhook",
        "type": "webhook",
        "method": "POST",
        "data_format": "json",
        "headers": {
            "Content-Type": "application/json"
        },
        "auth_type": "none"
    },
    "rest_api_with_apikey": {
        "name": "REST API with API Key",
        "type": "rest_api",
        "method": "GET",
        "data_format": "json",
        "auth_type": "api_key",
        "headers": {
            "Accept": "application/json"
        }
    },
    "oauth2_rest_api": {
        "name": "OAuth2 REST API",
        "type": "rest_api",
        "method": "GET",
        "data_format": "json",
        "auth_type": "oauth2",
        "headers": {
            "Accept": "application/json"
        }
    }
}
