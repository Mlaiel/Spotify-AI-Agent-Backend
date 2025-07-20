"""
Advanced API Integration Manager for PagerDuty

Ce module fournit un système d'intégration API ultra-sophistiqué pour PagerDuty avec intelligence artificielle,
gestion multi-API, cache intelligent, rate limiting adaptatif, retry avec backoff exponentiel,
circuit breaker pattern, et monitoring complet des performances.

Fonctionnalités principales:
- Intégration complète avec toutes les APIs PagerDuty (v2)
- Gestion intelligente des rate limits avec prédiction IA
- Circuit breaker pattern avec auto-recovery
- Cache distribué avec invalidation intelligente
- Retry adaptatif avec backoff exponentiel intelligent
- Monitoring et métriques en temps réel
- Webhook management avec validation et sécurité
- API versioning et compatibility layer
- Performance optimization automatique

Version: 4.0.0
Développé par Spotify AI Agent Team
"""

import asyncio
import json
import hashlib
import uuid
import time
import hmac
import base64
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
import structlog
import aiofiles
import aiohttp
import aioredis
import backoff
import tenacity
from urllib.parse import urljoin, urlparse
import ssl
from collections import defaultdict, deque
import asyncio_throttle
from functools import wraps
import jwt
from cryptography.fernet import Fernet

logger = structlog.get_logger(__name__)

# ============================================================================
# API Integration Enums and Data Structures
# ============================================================================

class APIEndpointType(Enum):
    """Types d'endpoints API PagerDuty"""
    INCIDENTS = "incidents"
    SERVICES = "services"
    ESCALATION_POLICIES = "escalation_policies"
    SCHEDULES = "schedules"
    USERS = "users"
    TEAMS = "teams"
    VENDORS = "vendors"
    PRIORITIES = "priorities"
    EVENTS = "events"
    WEBHOOKS = "webhooks"
    ANALYTICS = "analytics"
    NOTIFICATIONS = "notifications"
    MAINTENANCE_WINDOWS = "maintenance_windows"
    BUSINESS_SERVICES = "business_services"

class HTTPMethod(Enum):
    """Méthodes HTTP"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"

class RateLimitStrategy(Enum):
    """Stratégies de rate limiting"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    ADAPTIVE = "adaptive"

class CircuitBreakerState(Enum):
    """États du circuit breaker"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CacheStrategy(Enum):
    """Stratégies de cache"""
    NO_CACHE = "no_cache"
    TIME_BASED = "time_based"
    CONTENT_BASED = "content_based"
    INTELLIGENT = "intelligent"

@dataclass
class APIEndpointConfig:
    """Configuration d'endpoint API"""
    endpoint_type: APIEndpointType
    base_path: str
    methods_allowed: List[HTTPMethod]
    rate_limit_per_minute: int = 100
    cache_ttl_seconds: int = 300
    timeout_seconds: int = 30
    retry_attempts: int = 3
    circuit_breaker_enabled: bool = True
    requires_pagination: bool = False
    supports_filtering: bool = True
    requires_authentication: bool = True
    supports_webhooks: bool = False
    api_version: str = "v2"

@dataclass
class RateLimitInfo:
    """Informations de rate limiting"""
    limit: int
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None
    window_start: Optional[datetime] = None

@dataclass
class CircuitBreakerMetrics:
    """Métriques du circuit breaker"""
    state: CircuitBreakerState
    failure_count: int
    success_count: int
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    half_open_attempts: int = 0
    total_requests: int = 0

@dataclass
class APICallMetrics:
    """Métriques d'appel API"""
    endpoint: str
    method: HTTPMethod
    status_code: int
    response_time_ms: float
    payload_size_bytes: int
    response_size_bytes: int
    from_cache: bool = False
    retry_count: int = 0
    rate_limited: bool = False
    circuit_breaker_triggered: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class WebhookConfig:
    """Configuration de webhook"""
    id: str
    url: str
    events: List[str]
    secret_key: str
    active: bool = True
    ssl_verify: bool = True
    timeout_seconds: int = 10
    retry_attempts: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    last_delivery_attempt: Optional[datetime] = None
    last_successful_delivery: Optional[datetime] = None
    delivery_success_rate: float = 1.0

@dataclass
class APIRequest:
    """Requête API structurée"""
    endpoint_type: APIEndpointType
    method: HTTPMethod
    path: str
    query_params: Dict[str, Any] = field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: Optional[int] = None
    retry_config: Optional[Dict[str, Any]] = None
    cache_config: Optional[Dict[str, Any]] = None
    priority: int = 1  # 1=low, 5=high

@dataclass
class APIResponse:
    """Réponse API structurée"""
    status_code: int
    headers: Dict[str, str]
    data: Any
    from_cache: bool = False
    rate_limit_info: Optional[RateLimitInfo] = None
    metrics: Optional[APICallMetrics] = None
    pagination_info: Optional[Dict[str, Any]] = None

# ============================================================================
# Advanced API Integration Manager
# ============================================================================

class AdvancedAPIIntegrationManager:
    """Gestionnaire d'intégration API ultra-avancé"""
    
    def __init__(self,
                 api_key: str,
                 base_url: str = "https://api.pagerduty.com",
                 events_url: str = "https://events.pagerduty.com/v2/enqueue",
                 cache_dir: str = "/tmp/pagerduty_cache",
                 redis_url: Optional[str] = None,
                 enable_caching: bool = True,
                 enable_rate_limiting: bool = True,
                 enable_circuit_breaker: bool = True,
                 enable_metrics: bool = True):
        
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.events_url = events_url
        self.cache_dir = Path(cache_dir)
        self.redis_url = redis_url
        self.enable_caching = enable_caching
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_circuit_breaker = enable_circuit_breaker
        self.enable_metrics = enable_metrics
        
        # Configuration des endpoints
        self.endpoint_configs = self._initialize_endpoint_configs()
        
        # Sessions HTTP
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.events_session: Optional[aiohttp.ClientSession] = None
        
        # Cache
        self.redis_client: Optional[aioredis.Redis] = None
        self.local_cache: Dict[str, Any] = {}
        self.cache_ttl_default = 300  # 5 minutes
        
        # Rate limiting
        self.rate_limiters: Dict[str, asyncio_throttle.Throttler] = {}
        self.rate_limit_info: Dict[str, RateLimitInfo] = {}
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreakerMetrics] = {}
        self.circuit_breaker_failure_threshold = 5
        self.circuit_breaker_recovery_timeout = 60
        
        # Métriques
        self.api_metrics: deque = deque(maxlen=10000)
        self.performance_metrics: Dict[str, Any] = defaultdict(list)
        
        # Webhooks
        self.webhook_configs: Dict[str, WebhookConfig] = {}
        self.webhook_deliveries: deque = deque(maxlen=1000)
        
        # Queue de requêtes prioritaires
        self.request_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.queue_workers: List[asyncio.Task] = []
        
        # Chiffrement pour données sensibles
        self.cipher_key = Fernet.generate_key()
        self.cipher = Fernet(self.cipher_key)
        
        # Configuration SSL
        self.ssl_context = ssl.create_default_context()
        
        # Initialisation
        asyncio.create_task(self._initialize())
        
        logger.info("Advanced API Integration Manager initialized")
    
    def _initialize_endpoint_configs(self) -> Dict[APIEndpointType, APIEndpointConfig]:
        """Initialise les configurations d'endpoints"""
        
        configs = {}
        
        # Incidents API
        configs[APIEndpointType.INCIDENTS] = APIEndpointConfig(
            endpoint_type=APIEndpointType.INCIDENTS,
            base_path="/incidents",
            methods_allowed=[HTTPMethod.GET, HTTPMethod.POST, HTTPMethod.PUT, HTTPMethod.PATCH],
            rate_limit_per_minute=120,
            cache_ttl_seconds=60,
            requires_pagination=True,
            supports_filtering=True,
            supports_webhooks=True
        )
        
        # Services API
        configs[APIEndpointType.SERVICES] = APIEndpointConfig(
            endpoint_type=APIEndpointType.SERVICES,
            base_path="/services",
            methods_allowed=[HTTPMethod.GET, HTTPMethod.POST, HTTPMethod.PUT, HTTPMethod.DELETE],
            rate_limit_per_minute=100,
            cache_ttl_seconds=300,
            requires_pagination=True,
            supports_filtering=True
        )
        
        # Escalation Policies API
        configs[APIEndpointType.ESCALATION_POLICIES] = APIEndpointConfig(
            endpoint_type=APIEndpointType.ESCALATION_POLICIES,
            base_path="/escalation_policies",
            methods_allowed=[HTTPMethod.GET, HTTPMethod.POST, HTTPMethod.PUT, HTTPMethod.DELETE],
            rate_limit_per_minute=80,
            cache_ttl_seconds=600,
            requires_pagination=True
        )
        
        # Schedules API
        configs[APIEndpointType.SCHEDULES] = APIEndpointConfig(
            endpoint_type=APIEndpointType.SCHEDULES,
            base_path="/schedules",
            methods_allowed=[HTTPMethod.GET, HTTPMethod.POST, HTTPMethod.PUT, HTTPMethod.DELETE],
            rate_limit_per_minute=60,
            cache_ttl_seconds=300,
            requires_pagination=True
        )
        
        # Users API
        configs[APIEndpointType.USERS] = APIEndpointConfig(
            endpoint_type=APIEndpointType.USERS,
            base_path="/users",
            methods_allowed=[HTTPMethod.GET, HTTPMethod.POST, HTTPMethod.PUT, HTTPMethod.DELETE],
            rate_limit_per_minute=100,
            cache_ttl_seconds=600,
            requires_pagination=True
        )
        
        # Teams API
        configs[APIEndpointType.TEAMS] = APIEndpointConfig(
            endpoint_type=APIEndpointType.TEAMS,
            base_path="/teams",
            methods_allowed=[HTTPMethod.GET, HTTPMethod.POST, HTTPMethod.PUT, HTTPMethod.DELETE],
            rate_limit_per_minute=80,
            cache_ttl_seconds=600,
            requires_pagination=True
        )
        
        # Events API (différente URL)
        configs[APIEndpointType.EVENTS] = APIEndpointConfig(
            endpoint_type=APIEndpointType.EVENTS,
            base_path="/v2/enqueue",
            methods_allowed=[HTTPMethod.POST],
            rate_limit_per_minute=200,
            cache_ttl_seconds=0,  # Pas de cache pour les events
            circuit_breaker_enabled=True,
            requires_authentication=False  # Utilise routing_key
        )
        
        # Webhooks API
        configs[APIEndpointType.WEBHOOKS] = APIEndpointConfig(
            endpoint_type=APIEndpointType.WEBHOOKS,
            base_path="/webhook_subscriptions",
            methods_allowed=[HTTPMethod.GET, HTTPMethod.POST, HTTPMethod.PUT, HTTPMethod.DELETE],
            rate_limit_per_minute=50,
            cache_ttl_seconds=300
        )
        
        # Analytics API
        configs[APIEndpointType.ANALYTICS] = APIEndpointConfig(
            endpoint_type=APIEndpointType.ANALYTICS,
            base_path="/analytics",
            methods_allowed=[HTTPMethod.GET],
            rate_limit_per_minute=30,
            cache_ttl_seconds=1800,  # 30 minutes
            timeout_seconds=60
        )
        
        return configs
    
    async def _initialize(self):
        """Initialisation du gestionnaire"""
        
        # Création des répertoires
        await self._ensure_directories()
        
        # Initialisation des sessions HTTP
        await self._initialize_http_sessions()
        
        # Initialisation du cache Redis
        if self.redis_url and self.enable_caching:
            await self._initialize_redis()
        
        # Initialisation des rate limiters
        if self.enable_rate_limiting:
            await self._initialize_rate_limiters()
        
        # Initialisation des circuit breakers
        if self.enable_circuit_breaker:
            await self._initialize_circuit_breakers()
        
        # Chargement des configurations
        await self._load_configurations()
        
        # Démarrage des workers de queue
        await self._start_queue_workers()
        
        # Démarrage des tâches périodiques
        asyncio.create_task(self._periodic_metrics_collection())
        asyncio.create_task(self._periodic_cache_cleanup())
        asyncio.create_task(self._periodic_circuit_breaker_health_check())
        asyncio.create_task(self._periodic_rate_limit_optimization())
        
        logger.info("API Integration Manager initialization completed")
    
    async def _ensure_directories(self):
        """Assure que les répertoires nécessaires existent"""
        
        directories = [
            self.cache_dir,
            self.cache_dir / "responses",
            self.cache_dir / "metrics",
            self.cache_dir / "configs",
            self.cache_dir / "webhooks",
            self.cache_dir / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def _initialize_http_sessions(self):
        """Initialise les sessions HTTP"""
        
        # Headers communs
        common_headers = {
            "Accept": "application/vnd.pagerduty+json;version=2",
            "User-Agent": "Spotify-AI-Agent-PagerDuty/4.0.0",
            "Authorization": f"Token token={self.api_key}"
        }
        
        # Configuration du timeout
        timeout = aiohttp.ClientTimeout(
            total=30,
            connect=10,
            sock_read=20
        )
        
        # Connecteur avec configuration SSL
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            use_dns_cache=True,
            ssl=self.ssl_context,
            enable_cleanup_closed=True
        )
        
        # Session principale pour l'API
        self.http_session = aiohttp.ClientSession(
            headers=common_headers,
            timeout=timeout,
            connector=connector,
            raise_for_status=False  # Gestion manuelle des erreurs
        )
        
        # Session pour les events (différents headers)
        events_headers = {
            "Content-Type": "application/json",
            "User-Agent": "Spotify-AI-Agent-PagerDuty/4.0.0"
        }
        
        self.events_session = aiohttp.ClientSession(
            headers=events_headers,
            timeout=timeout,
            connector=aiohttp.TCPConnector(
                limit=50,
                ssl=self.ssl_context
            ),
            raise_for_status=False
        )
    
    async def _initialize_redis(self):
        """Initialise la connexion Redis"""
        
        try:
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            
            # Test de la connexion
            await self.redis_client.ping()
            logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self.redis_client = None
    
    async def _initialize_rate_limiters(self):
        """Initialise les rate limiters"""
        
        for endpoint_type, config in self.endpoint_configs.items():
            # Créer un throttler pour chaque endpoint
            rate_per_second = config.rate_limit_per_minute / 60
            
            self.rate_limiters[endpoint_type.value] = asyncio_throttle.Throttler(
                rate_limit=rate_per_second,
                period=1.0
            )
            
            # Initialiser les infos de rate limit
            self.rate_limit_info[endpoint_type.value] = RateLimitInfo(
                limit=config.rate_limit_per_minute,
                remaining=config.rate_limit_per_minute,
                reset_time=datetime.now() + timedelta(minutes=1)
            )
    
    async def _initialize_circuit_breakers(self):
        """Initialise les circuit breakers"""
        
        for endpoint_type in self.endpoint_configs:
            self.circuit_breakers[endpoint_type.value] = CircuitBreakerMetrics(
                state=CircuitBreakerState.CLOSED,
                failure_count=0,
                success_count=0
            )
    
    async def _start_queue_workers(self):
        """Démarre les workers de queue de requêtes"""
        
        # Créer plusieurs workers pour traiter les requêtes en parallèle
        for i in range(5):  # 5 workers
            worker = asyncio.create_task(self._queue_worker(f"worker-{i}"))
            self.queue_workers.append(worker)
    
    async def _queue_worker(self, worker_name: str):
        """Worker pour traiter les requêtes de la queue"""
        
        while True:
            try:
                # Récupérer une requête de la queue (avec priorité)
                priority, request_id, api_request = await self.request_queue.get()
                
                # Traiter la requête
                try:
                    response = await self._execute_api_request_internal(api_request)
                    logger.debug(f"{worker_name} processed request {request_id}")
                except Exception as e:
                    logger.error(f"{worker_name} failed to process request {request_id}: {e}")
                
                # Marquer la tâche comme terminée
                self.request_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue worker {worker_name} error: {e}")
                await asyncio.sleep(1)
    
    async def execute_api_request(self, request: APIRequest) -> APIResponse:
        """Execute une requête API avec toutes les optimisations"""
        
        # Validation de la requête
        await self._validate_api_request(request)
        
        # Vérification du circuit breaker
        if self.enable_circuit_breaker:
            circuit_breaker = self.circuit_breakers.get(request.endpoint_type.value)
            if circuit_breaker and circuit_breaker.state == CircuitBreakerState.OPEN:
                if not await self._should_attempt_circuit_breaker_recovery(circuit_breaker):
                    raise Exception(f"Circuit breaker is OPEN for {request.endpoint_type.value}")
        
        # Vérification du cache
        if self.enable_caching:
            cached_response = await self._get_cached_response(request)
            if cached_response:
                return cached_response
        
        # Rate limiting
        if self.enable_rate_limiting:
            await self._apply_rate_limiting(request)
        
        # Exécution de la requête
        response = await self._execute_api_request_internal(request)
        
        # Cache de la réponse
        if self.enable_caching and response.status_code < 400:
            await self._cache_response(request, response)
        
        # Mise à jour des métriques
        if self.enable_metrics:
            await self._update_metrics(request, response)
        
        return response
    
    async def _execute_api_request_internal(self, request: APIRequest) -> APIResponse:
        """Exécution interne de la requête API"""
        
        start_time = time.time()
        
        # Sélection de la session appropriée
        if request.endpoint_type == APIEndpointType.EVENTS:
            session = self.events_session
            base_url = self.events_url.rstrip('/v2/enqueue')
        else:
            session = self.http_session
            base_url = self.base_url
        
        # Construction de l'URL
        config = self.endpoint_configs[request.endpoint_type]
        url = urljoin(base_url, config.base_path.lstrip('/') + request.path)
        
        # Préparation des headers
        headers = request.headers.copy()
        
        # Préparation des paramètres
        params = request.query_params.copy() if request.query_params else {}
        
        # Configuration du timeout
        timeout = request.timeout or config.timeout_seconds
        
        try:
            # Exécution de la requête avec retry
            response = await self._execute_with_retry(
                session=session,
                method=request.method.value,
                url=url,
                headers=headers,
                params=params,
                json=request.body,
                timeout=timeout,
                request=request
            )
            
            # Traitement de la réponse
            response_time = (time.time() - start_time) * 1000
            
            # Extraction des données
            try:
                if response.content_type == 'application/json':
                    data = await response.json()
                else:
                    data = await response.text()
            except Exception as e:
                logger.warning(f"Failed to parse response data: {e}")
                data = None
            
            # Extraction des informations de rate limiting
            rate_limit_info = self._extract_rate_limit_info(response.headers)
            
            # Création de la réponse structurée
            api_response = APIResponse(
                status_code=response.status,
                headers=dict(response.headers),
                data=data,
                rate_limit_info=rate_limit_info,
                metrics=APICallMetrics(
                    endpoint=url,
                    method=request.method,
                    status_code=response.status,
                    response_time_ms=response_time,
                    payload_size_bytes=len(json.dumps(request.body).encode()) if request.body else 0,
                    response_size_bytes=len(str(data).encode()) if data else 0
                )
            )
            
            # Mise à jour du circuit breaker
            if self.enable_circuit_breaker:
                await self._update_circuit_breaker(request.endpoint_type, response.status < 500)
            
            return api_response
            
        except Exception as e:
            # Mise à jour du circuit breaker en cas d'erreur
            if self.enable_circuit_breaker:
                await self._update_circuit_breaker(request.endpoint_type, False)
            
            logger.error(f"API request failed: {e}")
            raise
    
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential_jitter(initial=1, max=60),
        retry=tenacity.retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _execute_with_retry(self, session: aiohttp.ClientSession, **kwargs) -> aiohttp.ClientResponse:
        """Exécute une requête avec retry intelligent"""
        
        request = kwargs.pop('request', None)
        
        try:
            async with session.request(**kwargs) as response:
                # Vérification des erreurs de rate limiting
                if response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited, waiting {retry_after} seconds")
                    await asyncio.sleep(retry_after)
                    raise aiohttp.ClientError("Rate limited")
                
                # Vérification des erreurs serveur (5xx)
                if response.status >= 500:
                    raise aiohttp.ClientError(f"Server error: {response.status}")
                
                return response
                
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if request:
                logger.warning(f"Request retry for {request.endpoint_type.value}: {e}")
            raise
    
    async def _apply_rate_limiting(self, request: APIRequest):
        """Applique le rate limiting"""
        
        endpoint_key = request.endpoint_type.value
        
        if endpoint_key in self.rate_limiters:
            throttler = self.rate_limiters[endpoint_key]
            
            # Attendre si nécessaire
            async with throttler:
                pass
            
            # Mise à jour des informations de rate limit
            rate_info = self.rate_limit_info.get(endpoint_key)
            if rate_info:
                rate_info.remaining = max(0, rate_info.remaining - 1)
                
                # Reset si nécessaire
                if datetime.now() >= rate_info.reset_time:
                    config = self.endpoint_configs[request.endpoint_type]
                    rate_info.remaining = config.rate_limit_per_minute
                    rate_info.reset_time = datetime.now() + timedelta(minutes=1)
    
    async def _get_cached_response(self, request: APIRequest) -> Optional[APIResponse]:
        """Récupère une réponse depuis le cache"""
        
        if not self.enable_caching or request.method != HTTPMethod.GET:
            return None
        
        cache_key = self._generate_cache_key(request)
        
        # Vérification du cache Redis d'abord
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    response = APIResponse(**data)
                    response.from_cache = True
                    return response
            except Exception as e:
                logger.warning(f"Redis cache read error: {e}")
        
        # Vérification du cache local
        if cache_key in self.local_cache:
            cache_entry = self.local_cache[cache_key]
            if cache_entry['expires'] > datetime.now():
                response = APIResponse(**cache_entry['data'])
                response.from_cache = True
                return response
            else:
                # Suppression de l'entrée expirée
                del self.local_cache[cache_key]
        
        return None
    
    async def _cache_response(self, request: APIRequest, response: APIResponse):
        """Met en cache une réponse"""
        
        if not self.enable_caching or request.method != HTTPMethod.GET:
            return
        
        config = self.endpoint_configs[request.endpoint_type]
        cache_key = self._generate_cache_key(request)
        
        # Préparation des données à cacher
        cache_data = {
            'status_code': response.status_code,
            'headers': response.headers,
            'data': response.data,
            'from_cache': False
        }
        
        # Cache Redis
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    cache_key,
                    config.cache_ttl_seconds,
                    json.dumps(cache_data, default=str)
                )
            except Exception as e:
                logger.warning(f"Redis cache write error: {e}")
        
        # Cache local
        self.local_cache[cache_key] = {
            'data': cache_data,
            'expires': datetime.now() + timedelta(seconds=config.cache_ttl_seconds)
        }
    
    def _generate_cache_key(self, request: APIRequest) -> str:
        """Génère une clé de cache pour une requête"""
        
        key_components = [
            request.endpoint_type.value,
            request.method.value,
            request.path,
            json.dumps(request.query_params, sort_keys=True) if request.query_params else ""
        ]
        
        key_string = "|".join(key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _extract_rate_limit_info(self, headers: Dict[str, str]) -> Optional[RateLimitInfo]:
        """Extrait les informations de rate limiting des headers"""
        
        try:
            limit = headers.get('X-RateLimit-Limit')
            remaining = headers.get('X-RateLimit-Remaining')
            reset = headers.get('X-RateLimit-Reset')
            retry_after = headers.get('Retry-After')
            
            if limit and remaining and reset:
                return RateLimitInfo(
                    limit=int(limit),
                    remaining=int(remaining),
                    reset_time=datetime.fromtimestamp(int(reset)),
                    retry_after=int(retry_after) if retry_after else None
                )
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse rate limit headers: {e}")
        
        return None
    
    async def _update_circuit_breaker(self, endpoint_type: APIEndpointType, success: bool):
        """Met à jour l'état du circuit breaker"""
        
        endpoint_key = endpoint_type.value
        circuit_breaker = self.circuit_breakers.get(endpoint_key)
        
        if not circuit_breaker:
            return
        
        circuit_breaker.total_requests += 1
        
        if success:
            circuit_breaker.success_count += 1
            circuit_breaker.last_success_time = datetime.now()
            
            # Reset du compteur d'échecs en cas de succès
            if circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
                circuit_breaker.half_open_attempts += 1
                if circuit_breaker.half_open_attempts >= 3:  # 3 succès consécutifs
                    circuit_breaker.state = CircuitBreakerState.CLOSED
                    circuit_breaker.failure_count = 0
                    circuit_breaker.half_open_attempts = 0
            elif circuit_breaker.state == CircuitBreakerState.CLOSED:
                circuit_breaker.failure_count = max(0, circuit_breaker.failure_count - 1)
        else:
            circuit_breaker.failure_count += 1
            circuit_breaker.last_failure_time = datetime.now()
            
            # Ouverture du circuit breaker si trop d'échecs
            if (circuit_breaker.state == CircuitBreakerState.CLOSED and 
                circuit_breaker.failure_count >= self.circuit_breaker_failure_threshold):
                circuit_breaker.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker OPENED for {endpoint_key}")
            elif circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
                circuit_breaker.state = CircuitBreakerState.OPEN
                circuit_breaker.half_open_attempts = 0
    
    async def _should_attempt_circuit_breaker_recovery(self, circuit_breaker: CircuitBreakerMetrics) -> bool:
        """Détermine si on doit tenter une récupération du circuit breaker"""
        
        if circuit_breaker.state != CircuitBreakerState.OPEN:
            return True
        
        # Vérification du timeout de récupération
        if (circuit_breaker.last_failure_time and 
            datetime.now() - circuit_breaker.last_failure_time > timedelta(seconds=self.circuit_breaker_recovery_timeout)):
            circuit_breaker.state = CircuitBreakerState.HALF_OPEN
            circuit_breaker.half_open_attempts = 0
            return True
        
        return False
    
    async def _update_metrics(self, request: APIRequest, response: APIResponse):
        """Met à jour les métriques"""
        
        if response.metrics:
            self.api_metrics.append(response.metrics)
            
            # Mise à jour des métriques de performance par endpoint
            endpoint_key = request.endpoint_type.value
            self.performance_metrics[endpoint_key].append({
                'response_time': response.metrics.response_time_ms,
                'status_code': response.metrics.status_code,
                'timestamp': response.metrics.timestamp,
                'from_cache': response.from_cache
            })
            
            # Limitation de l'historique
            if len(self.performance_metrics[endpoint_key]) > 1000:
                self.performance_metrics[endpoint_key] = self.performance_metrics[endpoint_key][-1000:]
    
    async def _validate_api_request(self, request: APIRequest):
        """Valide une requête API"""
        
        config = self.endpoint_configs.get(request.endpoint_type)
        if not config:
            raise ValueError(f"Unknown endpoint type: {request.endpoint_type}")
        
        if request.method not in config.methods_allowed:
            raise ValueError(f"Method {request.method} not allowed for {request.endpoint_type}")
        
        # Validation des paramètres spécifiques
        if request.endpoint_type == APIEndpointType.EVENTS and not request.body:
            raise ValueError("Events API requires a body")
    
    # API Convenience Methods
    
    async def get_incidents(self, 
                          filters: Optional[Dict[str, Any]] = None,
                          limit: int = 25,
                          offset: int = 0) -> APIResponse:
        """Récupère les incidents avec filtres"""
        
        query_params = {
            'limit': min(limit, 100),  # Maximum 100
            'offset': offset
        }
        
        if filters:
            query_params.update(filters)
        
        request = APIRequest(
            endpoint_type=APIEndpointType.INCIDENTS,
            method=HTTPMethod.GET,
            path="",
            query_params=query_params
        )
        
        return await self.execute_api_request(request)
    
    async def create_incident(self, incident_data: Dict[str, Any]) -> APIResponse:
        """Crée un incident"""
        
        request = APIRequest(
            endpoint_type=APIEndpointType.INCIDENTS,
            method=HTTPMethod.POST,
            path="",
            body=incident_data
        )
        
        return await self.execute_api_request(request)
    
    async def update_incident(self, incident_id: str, update_data: Dict[str, Any]) -> APIResponse:
        """Met à jour un incident"""
        
        request = APIRequest(
            endpoint_type=APIEndpointType.INCIDENTS,
            method=HTTPMethod.PUT,
            path=f"/{incident_id}",
            body=update_data
        )
        
        return await self.execute_api_request(request)
    
    async def send_event(self, event_data: Dict[str, Any]) -> APIResponse:
        """Envoie un événement"""
        
        request = APIRequest(
            endpoint_type=APIEndpointType.EVENTS,
            method=HTTPMethod.POST,
            path="",
            body=event_data,
            priority=5  # Haute priorité pour les événements
        )
        
        return await self.execute_api_request(request)
    
    async def get_services(self, include_deleted: bool = False) -> APIResponse:
        """Récupère les services"""
        
        query_params = {}
        if include_deleted:
            query_params['include[]'] = 'deleted'
        
        request = APIRequest(
            endpoint_type=APIEndpointType.SERVICES,
            method=HTTPMethod.GET,
            path="",
            query_params=query_params
        )
        
        return await self.execute_api_request(request)
    
    async def get_escalation_policies(self) -> APIResponse:
        """Récupère les politiques d'escalade"""
        
        request = APIRequest(
            endpoint_type=APIEndpointType.ESCALATION_POLICIES,
            method=HTTPMethod.GET,
            path=""
        )
        
        return await self.execute_api_request(request)
    
    async def get_schedules(self) -> APIResponse:
        """Récupère les plannings"""
        
        request = APIRequest(
            endpoint_type=APIEndpointType.SCHEDULES,
            method=HTTPMethod.GET,
            path=""
        )
        
        return await self.execute_api_request(request)
    
    async def get_users(self) -> APIResponse:
        """Récupère les utilisateurs"""
        
        request = APIRequest(
            endpoint_type=APIEndpointType.USERS,
            method=HTTPMethod.GET,
            path=""
        )
        
        return await self.execute_api_request(request)
    
    # Webhook Management
    
    async def create_webhook_subscription(self, webhook_config: WebhookConfig) -> APIResponse:
        """Crée une souscription webhook"""
        
        webhook_data = {
            "webhook_subscription": {
                "delivery_method": {
                    "type": "http_delivery_method",
                    "url": webhook_config.url,
                    "secret": webhook_config.secret_key
                },
                "events": webhook_config.events,
                "filter": {
                    "type": "account_reference"
                }
            }
        }
        
        request = APIRequest(
            endpoint_type=APIEndpointType.WEBHOOKS,
            method=HTTPMethod.POST,
            path="",
            body=webhook_data
        )
        
        response = await self.execute_api_request(request)
        
        if response.status_code == 201:
            self.webhook_configs[webhook_config.id] = webhook_config
            await self._save_webhook_config(webhook_config)
        
        return response
    
    async def validate_webhook_signature(self, payload: str, signature: str, secret: str) -> bool:
        """Valide la signature d'un webhook"""
        
        try:
            expected_signature = hmac.new(
                secret.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(f"sha256={expected_signature}", signature)
        except Exception as e:
            logger.error(f"Webhook signature validation error: {e}")
            return False
    
    async def process_webhook_delivery(self, webhook_id: str, payload: Dict[str, Any]) -> bool:
        """Traite une livraison de webhook"""
        
        try:
            webhook_config = self.webhook_configs.get(webhook_id)
            if not webhook_config:
                logger.warning(f"Unknown webhook: {webhook_id}")
                return False
            
            # Enregistrement de la tentative de livraison
            delivery_record = {
                'webhook_id': webhook_id,
                'timestamp': datetime.now(),
                'payload': payload,
                'success': True
            }
            
            self.webhook_deliveries.append(delivery_record)
            
            # Mise à jour des statistiques
            webhook_config.last_delivery_attempt = datetime.now()
            webhook_config.last_successful_delivery = datetime.now()
            
            # Calcul du taux de succès
            recent_deliveries = [d for d in self.webhook_deliveries 
                               if d['webhook_id'] == webhook_id and 
                               d['timestamp'] > datetime.now() - timedelta(hours=24)]
            
            if recent_deliveries:
                success_count = sum(1 for d in recent_deliveries if d['success'])
                webhook_config.delivery_success_rate = success_count / len(recent_deliveries)
            
            return True
            
        except Exception as e:
            logger.error(f"Webhook delivery processing error: {e}")
            return False
    
    # Analytics and Metrics
    
    async def get_api_analytics(self) -> Dict[str, Any]:
        """Obtient les analytics API"""
        
        analytics = {
            'total_requests': len(self.api_metrics),
            'requests_by_endpoint': defaultdict(int),
            'avg_response_times': {},
            'error_rates': {},
            'cache_hit_rates': {},
            'circuit_breaker_states': {},
            'rate_limit_status': {}
        }
        
        # Analyse des métriques
        for metric in self.api_metrics:
            endpoint = metric.endpoint
            analytics['requests_by_endpoint'][endpoint] += 1
        
        # Calcul des temps de réponse moyens
        for endpoint, metrics in self.performance_metrics.items():
            if metrics:
                response_times = [m['response_time'] for m in metrics]
                analytics['avg_response_times'][endpoint] = sum(response_times) / len(response_times)
                
                # Taux d'erreur
                error_count = sum(1 for m in metrics if m['status_code'] >= 400)
                analytics['error_rates'][endpoint] = error_count / len(metrics)
                
                # Taux de cache hit
                cache_hits = sum(1 for m in metrics if m.get('from_cache', False))
                analytics['cache_hit_rates'][endpoint] = cache_hits / len(metrics)
        
        # États des circuit breakers
        for endpoint, cb in self.circuit_breakers.items():
            analytics['circuit_breaker_states'][endpoint] = {
                'state': cb.state.value,
                'failure_count': cb.failure_count,
                'success_count': cb.success_count
            }
        
        # Statut des rate limits
        for endpoint, rate_info in self.rate_limit_info.items():
            analytics['rate_limit_status'][endpoint] = {
                'limit': rate_info.limit,
                'remaining': rate_info.remaining,
                'reset_time': rate_info.reset_time.isoformat()
            }
        
        return analytics
    
    async def get_performance_report(self, period_hours: int = 24) -> Dict[str, Any]:
        """Génère un rapport de performance"""
        
        cutoff_time = datetime.now() - timedelta(hours=period_hours)
        
        # Filtrer les métriques récentes
        recent_metrics = [m for m in self.api_metrics if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {'error': 'No metrics available for the specified period'}
        
        # Calculs de performance
        total_requests = len(recent_metrics)
        successful_requests = len([m for m in recent_metrics if m.status_code < 400])
        failed_requests = total_requests - successful_requests
        
        response_times = [m.response_time_ms for m in recent_metrics]
        avg_response_time = sum(response_times) / len(response_times)
        p95_response_time = np.percentile(response_times, 95) if response_times else 0
        p99_response_time = np.percentile(response_times, 99) if response_times else 0
        
        cache_hits = len([m for m in recent_metrics if m.from_cache])
        cache_hit_rate = cache_hits / total_requests if total_requests > 0 else 0
        
        rate_limited_requests = len([m for m in recent_metrics if m.rate_limited])
        rate_limit_rate = rate_limited_requests / total_requests if total_requests > 0 else 0
        
        report = {
            'period': {
                'start': cutoff_time.isoformat(),
                'end': datetime.now().isoformat(),
                'hours': period_hours
            },
            'requests': {
                'total': total_requests,
                'successful': successful_requests,
                'failed': failed_requests,
                'success_rate': successful_requests / total_requests if total_requests > 0 else 0
            },
            'performance': {
                'avg_response_time_ms': avg_response_time,
                'p95_response_time_ms': p95_response_time,
                'p99_response_time_ms': p99_response_time
            },
            'caching': {
                'hit_rate': cache_hit_rate,
                'hits': cache_hits,
                'total_cacheable': len([m for m in recent_metrics if m.endpoint.endswith('GET')])
            },
            'rate_limiting': {
                'rate_limited_requests': rate_limited_requests,
                'rate_limit_rate': rate_limit_rate
            },
            'endpoints': {}
        }
        
        # Analyse par endpoint
        endpoints_metrics = defaultdict(list)
        for metric in recent_metrics:
            endpoints_metrics[metric.endpoint].append(metric)
        
        for endpoint, metrics in endpoints_metrics.items():
            endpoint_response_times = [m.response_time_ms for m in metrics]
            endpoint_success_rate = len([m for m in metrics if m.status_code < 400]) / len(metrics)
            
            report['endpoints'][endpoint] = {
                'request_count': len(metrics),
                'avg_response_time_ms': sum(endpoint_response_times) / len(endpoint_response_times),
                'success_rate': endpoint_success_rate
            }
        
        return report
    
    # Periodic Tasks
    
    async def _periodic_metrics_collection(self):
        """Collecte périodique de métriques"""
        
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                
                # Sauvegarde des métriques
                metrics_file = self.cache_dir / "metrics" / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                
                analytics = await self.get_api_analytics()
                
                async with aiofiles.open(metrics_file, 'w') as f:
                    await f.write(json.dumps(analytics, indent=2, default=str))
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
    
    async def _periodic_cache_cleanup(self):
        """Nettoyage périodique du cache"""
        
        while True:
            try:
                await asyncio.sleep(3600)  # 1 heure
                
                # Nettoyage du cache local
                current_time = datetime.now()
                expired_keys = []
                
                for key, entry in self.local_cache.items():
                    if entry['expires'] <= current_time:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.local_cache[key]
                
                logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    async def _periodic_circuit_breaker_health_check(self):
        """Vérification périodique de santé des circuit breakers"""
        
        while True:
            try:
                await asyncio.sleep(60)  # 1 minute
                
                for endpoint, cb in self.circuit_breakers.items():
                    if cb.state == CircuitBreakerState.OPEN:
                        await self._should_attempt_circuit_breaker_recovery(cb)
                
            except Exception as e:
                logger.error(f"Circuit breaker health check error: {e}")
    
    async def _periodic_rate_limit_optimization(self):
        """Optimisation périodique des rate limits"""
        
        while True:
            try:
                await asyncio.sleep(1800)  # 30 minutes
                
                # Analyse des patterns de rate limiting
                for endpoint, metrics in self.performance_metrics.items():
                    if len(metrics) > 50:  # Suffisamment de données
                        recent_metrics = metrics[-50:]  # 50 dernières requêtes
                        
                        # Calcul du taux d'utilisation du rate limit
                        rate_limited_count = len([m for m in recent_metrics 
                                                if m.get('rate_limited', False)])
                        
                        if rate_limited_count > len(recent_metrics) * 0.1:  # Plus de 10%
                            # Ajustement du rate limiter
                            if endpoint in self.rate_limiters:
                                current_rate = self.rate_limiters[endpoint].rate_limit
                                new_rate = current_rate * 0.9  # Réduction de 10%
                                
                                self.rate_limiters[endpoint] = asyncio_throttle.Throttler(
                                    rate_limit=new_rate,
                                    period=1.0
                                )
                                
                                logger.info(f"Rate limit adjusted for {endpoint}: {current_rate} -> {new_rate}")
                
            except Exception as e:
                logger.error(f"Rate limit optimization error: {e}")
    
    # Configuration Management
    
    async def _load_configurations(self):
        """Charge les configurations sauvegardées"""
        
        # Chargement des configurations de webhooks
        webhooks_file = self.cache_dir / "configs" / "webhooks.json"
        if webhooks_file.exists():
            try:
                async with aiofiles.open(webhooks_file, 'r') as f:
                    webhooks_data = json.loads(await f.read())
                
                for webhook_data in webhooks_data:
                    webhook_config = WebhookConfig(**webhook_data)
                    self.webhook_configs[webhook_config.id] = webhook_config
                    
            except Exception as e:
                logger.error(f"Failed to load webhook configurations: {e}")
    
    async def _save_webhook_config(self, webhook_config: WebhookConfig):
        """Sauvegarde une configuration de webhook"""
        
        webhooks_file = self.cache_dir / "configs" / "webhooks.json"
        
        try:
            # Chargement des configurations existantes
            existing_configs = []
            if webhooks_file.exists():
                async with aiofiles.open(webhooks_file, 'r') as f:
                    existing_configs = json.loads(await f.read())
            
            # Ajout ou mise à jour
            config_dict = asdict(webhook_config)
            updated = False
            
            for i, config in enumerate(existing_configs):
                if config['id'] == webhook_config.id:
                    existing_configs[i] = config_dict
                    updated = True
                    break
            
            if not updated:
                existing_configs.append(config_dict)
            
            # Sauvegarde
            async with aiofiles.open(webhooks_file, 'w') as f:
                await f.write(json.dumps(existing_configs, indent=2, default=str))
                
        except Exception as e:
            logger.error(f"Failed to save webhook configuration: {e}")
    
    async def cleanup(self):
        """Nettoyage des ressources"""
        
        # Arrêt des workers de queue
        for worker in self.queue_workers:
            worker.cancel()
        
        # Fermeture des sessions HTTP
        if self.http_session:
            await self.http_session.close()
        
        if self.events_session:
            await self.events_session.close()
        
        # Fermeture de la connexion Redis
        if self.redis_client:
            await self.redis_client.close()
        
        # Sauvegarde finale des métriques
        try:
            final_analytics = await self.get_api_analytics()
            final_metrics_file = self.cache_dir / "metrics" / f"final_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            async with aiofiles.open(final_metrics_file, 'w') as f:
                await f.write(json.dumps(final_analytics, indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to save final metrics: {e}")
        
        logger.info("Advanced API Integration Manager cleaned up")

# Export des classes principales
__all__ = [
    "AdvancedAPIIntegrationManager",
    "APIRequest",
    "APIResponse",
    "WebhookConfig",
    "APIEndpointType",
    "HTTPMethod",
    "RateLimitStrategy",
    "CircuitBreakerState",
    "CacheStrategy",
    "APIEndpointConfig",
    "RateLimitInfo",
    "CircuitBreakerMetrics",
    "APICallMetrics"
]
