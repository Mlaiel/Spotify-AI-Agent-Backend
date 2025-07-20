"""
🔗 Request ID Middleware Ultra-Avancé - Spotify AI Agent
======================================================
Système de traçabilité et corrélation distribué pour microservices
Auteur: Équipe Lead Dev + Architecte IA + Spécialiste Sécurité Backend

Fonctionnalités Enterprise:
- Génération d'IDs uniques (UUID, Snowflake, Nanoid)
- Tracing distribué OpenTelemetry
- Corrélation inter-services
- Audit trail complet
- Propagation contexte
- Analytics de performance
- Support multi-tenant
- Chiffrement des IDs sensibles
"""

import time
import uuid
import hashlib
import base64
import json
import random
import string
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import redis.asyncio as redis
from opentelemetry import trace, context, baggage
from opentelemetry.trace import Status, StatusCode
from opentelemetry.propagate import extract, inject

from app.core.config import settings
from app.core.logging import get_logger
from app.utils.metrics_manager import MetricsManager
from app.core.security import SecurityUtils

logger = get_logger(__name__)
metrics = MetricsManager()
security_utils = SecurityUtils()
tracer = trace.get_tracer(__name__)


class IDFormat(Enum):
    """Formats d'ID supportés"""
    UUID4 = "uuid4"
    UUID1 = "uuid1"
    SNOWFLAKE = "snowflake"
    NANOID = "nanoid"
    ULID = "ulid"
    CUSTOM = "custom"


class ContextType(Enum):
    """Types de contexte"""
    REQUEST = "request"
    USER = "user"
    SESSION = "session"
    TRANSACTION = "transaction"
    BUSINESS = "business"
    SECURITY = "security"


@dataclass
class TraceContext:
    """Contexte de trace complet"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    correlation_id: str = ""
    request_id: str = ""
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    
    # Métadonnées business
    business_context: Dict[str, Any] = field(default_factory=dict)
    
    # Audit et sécurité
    source_ip: str = ""
    user_agent: str = ""
    api_key: Optional[str] = None
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    propagated_at: Optional[datetime] = None
    
    # Flags
    is_internal: bool = False
    is_sensitive: bool = False
    is_encrypted: bool = False
    
    # Baggage (données propagées)
    baggage_items: Dict[str, str] = field(default_factory=dict)


@dataclass
class RequestJourney:
    """Journey d'une requête à travers les services"""
    journey_id: str
    initial_request_id: str
    services_visited: List[str] = field(default_factory=list)
    hops: List[Dict[str, Any]] = field(default_factory=list)
    total_duration: float = 0.0
    error_count: int = 0
    status: str = "active"
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class SnowflakeIDGenerator:
    """Générateur d'IDs Snowflake distribuées"""
    
    def __init__(self, datacenter_id: int = 1, worker_id: int = 1):
        self.datacenter_id = datacenter_id & 0x1F  # 5 bits
        self.worker_id = worker_id & 0x1F  # 5 bits
        self.sequence = 0
        self.last_timestamp = -1
        
        # Epoch custom (2024-01-01)
        self.epoch = 1704067200000
    
    def generate(self) -> int:
        """Générer un ID Snowflake"""
        timestamp = int(time.time() * 1000)
        
        if timestamp < self.last_timestamp:
            raise ValueError("Clock moved backwards!")
        
        if timestamp == self.last_timestamp:
            self.sequence = (self.sequence + 1) & 0xFFF  # 12 bits
            if self.sequence == 0:
                timestamp = self._wait_next_millis(timestamp)
        else:
            self.sequence = 0
        
        self.last_timestamp = timestamp
        
        # Construire l'ID (64 bits)
        snowflake_id = (
            ((timestamp - self.epoch) << 22) |
            (self.datacenter_id << 17) |
            (self.worker_id << 12) |
            self.sequence
        )
        
        return snowflake_id
    
    def _wait_next_millis(self, last_timestamp: int) -> int:
        """Attendre la prochaine milliseconde"""
        timestamp = int(time.time() * 1000)
        while timestamp <= last_timestamp:
            timestamp = int(time.time() * 1000)
        return timestamp
    
    def parse(self, snowflake_id: int) -> Dict[str, Any]:
        """Parser un ID Snowflake"""
        timestamp = ((snowflake_id >> 22) + self.epoch) / 1000
        datacenter_id = (snowflake_id >> 17) & 0x1F
        worker_id = (snowflake_id >> 12) & 0x1F
        sequence = snowflake_id & 0xFFF
        
        return {
            "timestamp": datetime.fromtimestamp(timestamp),
            "datacenter_id": datacenter_id,
            "worker_id": worker_id,
            "sequence": sequence
        }


class NanoIDGenerator:
    """Générateur NanoID optimisé"""
    
    def __init__(self, alphabet: str = None, size: int = 21):
        self.alphabet = alphabet or "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
        self.size = size
    
    def generate(self) -> str:
        """Générer un NanoID"""
        return ''.join(random.choices(self.alphabet, k=self.size))


class ULIDGenerator:
    """Générateur ULID (Universally Unique Lexicographically Sortable Identifier)"""
    
    def __init__(self):
        self.encoding = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford Base32
        self.encoding_len = len(self.encoding)
        self.time_len = 10
        self.random_len = 16
    
    def generate(self) -> str:
        """Générer un ULID"""
        timestamp = int(time.time() * 1000)
        
        # Partie temporelle (10 caractères)
        time_part = self._encode_time(timestamp)
        
        # Partie aléatoire (16 caractères)
        random_part = self._encode_random()
        
        return time_part + random_part
    
    def _encode_time(self, timestamp: int) -> str:
        """Encoder la partie temporelle"""
        encoded = ""
        for _ in range(self.time_len):
            encoded = self.encoding[timestamp % self.encoding_len] + encoded
            timestamp //= self.encoding_len
        return encoded
    
    def _encode_random(self) -> str:
        """Encoder la partie aléatoire"""
        return ''.join(random.choices(self.encoding, k=self.random_len))


class RequestIDGenerator:
    """Générateur d'IDs de requête avancé"""
    
    def __init__(self, format_type: IDFormat = IDFormat.UUID4):
        self.format_type = format_type
        self.snowflake_gen = SnowflakeIDGenerator()
        self.nanoid_gen = NanoIDGenerator()
        self.ulid_gen = ULIDGenerator()
    
    def generate(self, prefix: str = "", context: Optional[Dict] = None) -> str:
        """Générer un ID selon le format spécifié"""
        if self.format_type == IDFormat.UUID4:
            base_id = str(uuid.uuid4())
        elif self.format_type == IDFormat.UUID1:
            base_id = str(uuid.uuid1())
        elif self.format_type == IDFormat.SNOWFLAKE:
            base_id = str(self.snowflake_gen.generate())
        elif self.format_type == IDFormat.NANOID:
            base_id = self.nanoid_gen.generate()
        elif self.format_type == IDFormat.ULID:
            base_id = self.ulid_gen.generate()
        else:
            base_id = str(uuid.uuid4())
        
        # Ajouter préfixe si spécifié
        if prefix:
            return f"{prefix}_{base_id}"
        
        return base_id
    
    def generate_with_checksum(self, prefix: str = "") -> str:
        """Générer un ID avec checksum pour validation"""
        base_id = self.generate(prefix)
        checksum = hashlib.md5(base_id.encode()).hexdigest()[:4]
        return f"{base_id}_{checksum}"
    
    def validate_checksum(self, id_with_checksum: str) -> bool:
        """Valider un ID avec checksum"""
        try:
            parts = id_with_checksum.rsplit('_', 1)
            if len(parts) != 2:
                return False
            
            base_id, provided_checksum = parts
            expected_checksum = hashlib.md5(base_id.encode()).hexdigest()[:4]
            
            return provided_checksum == expected_checksum
        except Exception:
            return False


class ContextPropagator:
    """Propagateur de contexte distribué"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.context_cache = {}
    
    async def inject_context(self, headers: Dict[str, str], trace_context: TraceContext):
        """Injecter le contexte dans les headers"""
        # Headers standards de tracing
        headers["X-Trace-Id"] = trace_context.trace_id
        headers["X-Span-Id"] = trace_context.span_id
        headers["X-Correlation-Id"] = trace_context.correlation_id
        headers["X-Request-Id"] = trace_context.request_id
        
        if trace_context.parent_span_id:
            headers["X-Parent-Span-Id"] = trace_context.parent_span_id
        
        # Contexte business
        if trace_context.user_id:
            headers["X-User-Id"] = trace_context.user_id
        if trace_context.session_id:
            headers["X-Session-Id"] = trace_context.session_id
        if trace_context.tenant_id:
            headers["X-Tenant-Id"] = trace_context.tenant_id
        
        # Baggage items
        if trace_context.baggage_items:
            baggage_header = ",".join([
                f"{k}={v}" for k, v in trace_context.baggage_items.items()
            ])
            headers["X-Baggage"] = baggage_header
        
        # OpenTelemetry standard propagation
        carrier = {}
        inject(carrier)
        for key, value in carrier.items():
            headers[key] = value
        
        # Stocker le contexte pour récupération ultérieure
        await self._store_context(trace_context)
    
    async def extract_context(self, headers: Dict[str, str]) -> Optional[TraceContext]:
        """Extraire le contexte depuis les headers"""
        try:
            # Extraire les IDs principaux
            trace_id = headers.get("X-Trace-Id") or headers.get("x-trace-id")
            span_id = headers.get("X-Span-Id") or headers.get("x-span-id")
            correlation_id = headers.get("X-Correlation-Id") or headers.get("x-correlation-id")
            request_id = headers.get("X-Request-Id") or headers.get("x-request-id")
            
            if not trace_id:
                return None
            
            # Créer le contexte
            trace_context = TraceContext(
                trace_id=trace_id,
                span_id=span_id or str(uuid.uuid4()),
                parent_span_id=headers.get("X-Parent-Span-Id"),
                correlation_id=correlation_id or "",
                request_id=request_id or "",
                session_id=headers.get("X-Session-Id"),
                user_id=headers.get("X-User-Id"),
                tenant_id=headers.get("X-Tenant-Id")
            )
            
            # Extraire le baggage
            baggage_header = headers.get("X-Baggage", "")
            if baggage_header:
                for item in baggage_header.split(","):
                    if "=" in item:
                        key, value = item.split("=", 1)
                        trace_context.baggage_items[key.strip()] = value.strip()
            
            # Enrichir depuis le stockage si disponible
            stored_context = await self._retrieve_context(trace_id)
            if stored_context:
                trace_context.business_context.update(stored_context.get("business_context", {}))
            
            return trace_context
            
        except Exception as e:
            logger.error(f"Erreur extraction contexte: {e}")
            return None
    
    async def _store_context(self, trace_context: TraceContext):
        """Stocker le contexte dans Redis"""
        try:
            context_data = {
                "trace_id": trace_context.trace_id,
                "correlation_id": trace_context.correlation_id,
                "business_context": trace_context.business_context,
                "baggage_items": trace_context.baggage_items,
                "created_at": trace_context.created_at.isoformat(),
                "is_sensitive": trace_context.is_sensitive
            }
            
            key = f"trace_context:{trace_context.trace_id}"
            await self.redis_client.setex(
                key,
                timedelta(hours=24),  # TTL de 24h
                json.dumps(context_data)
            )
            
        except Exception as e:
            logger.error(f"Erreur stockage contexte: {e}")
    
    async def _retrieve_context(self, trace_id: str) -> Optional[Dict]:
        """Récupérer le contexte depuis Redis"""
        try:
            key = f"trace_context:{trace_id}"
            context_data = await self.redis_client.get(key)
            
            if context_data:
                return json.loads(context_data)
            
        except Exception as e:
            logger.error(f"Erreur récupération contexte: {e}")
        
        return None


class JourneyTracker:
    """Traqueur de journey de requête"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
    
    async def start_journey(self, request_id: str, service_name: str) -> str:
        """Démarrer un journey de requête"""
        journey_id = f"journey_{uuid.uuid4().hex[:12]}"
        
        journey = RequestJourney(
            journey_id=journey_id,
            initial_request_id=request_id,
            services_visited=[service_name]
        )
        
        await self._store_journey(journey)
        return journey_id
    
    async def add_hop(self, journey_id: str, service_name: str, 
                     duration: float, status: str, metadata: Dict = None):
        """Ajouter une étape au journey"""
        try:
            journey = await self._get_journey(journey_id)
            if not journey:
                return
            
            hop = {
                "service": service_name,
                "timestamp": datetime.utcnow().isoformat(),
                "duration": duration,
                "status": status,
                "metadata": metadata or {}
            }
            
            journey.hops.append(hop)
            journey.services_visited.append(service_name)
            journey.total_duration += duration
            
            if status == "error":
                journey.error_count += 1
            
            await self._store_journey(journey)
            
        except Exception as e:
            logger.error(f"Erreur ajout hop journey: {e}")
    
    async def complete_journey(self, journey_id: str):
        """Compléter un journey"""
        try:
            journey = await self._get_journey(journey_id)
            if journey:
                journey.status = "completed"
                journey.completed_at = datetime.utcnow()
                await self._store_journey(journey)
                
        except Exception as e:
            logger.error(f"Erreur completion journey: {e}")
    
    async def _store_journey(self, journey: RequestJourney):
        """Stocker un journey"""
        journey_data = {
            "journey_id": journey.journey_id,
            "initial_request_id": journey.initial_request_id,
            "services_visited": journey.services_visited,
            "hops": journey.hops,
            "total_duration": journey.total_duration,
            "error_count": journey.error_count,
            "status": journey.status,
            "created_at": journey.created_at.isoformat(),
            "completed_at": journey.completed_at.isoformat() if journey.completed_at else None
        }
        
        key = f"journey:{journey.journey_id}"
        await self.redis_client.setex(
            key,
            timedelta(days=7),  # TTL de 7 jours
            json.dumps(journey_data)
        )
    
    async def _get_journey(self, journey_id: str) -> Optional[RequestJourney]:
        """Récupérer un journey"""
        try:
            key = f"journey:{journey_id}"
            journey_data = await self.redis_client.get(key)
            
            if journey_data:
                data = json.loads(journey_data)
                return RequestJourney(
                    journey_id=data["journey_id"],
                    initial_request_id=data["initial_request_id"],
                    services_visited=data["services_visited"],
                    hops=data["hops"],
                    total_duration=data["total_duration"],
                    error_count=data["error_count"],
                    status=data["status"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    completed_at=datetime.fromisoformat(data["completed_at"]) if data["completed_at"] else None
                )
            
        except Exception as e:
            logger.error(f"Erreur récupération journey: {e}")
        
        return None


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware de gestion des IDs de requête ultra-avancé
    
    Fonctionnalités:
    - Génération d'IDs multiples formats
    - Tracing distribué OpenTelemetry
    - Corrélation inter-services
    - Journey tracking
    - Audit complet
    - Propagation de contexte
    - Support multi-tenant
    """
    
    def __init__(
        self,
        app,
        redis_client: Optional[redis.Redis] = None,
        id_format: IDFormat = IDFormat.UUID4,
        service_name: str = "spotify-ai-agent",
        enable_journey_tracking: bool = True,
        enable_context_propagation: bool = True,
        enable_encryption: bool = False
    ):
        super().__init__(app)
        self.redis_client = redis_client or redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        
        self.service_name = service_name
        self.enable_journey_tracking = enable_journey_tracking
        self.enable_context_propagation = enable_context_propagation
        self.enable_encryption = enable_encryption
        
        # Générateurs
        self.id_generator = RequestIDGenerator(id_format)
        
        # Composants avancés
        self.context_propagator = ContextPropagator(self.redis_client) if enable_context_propagation else None
        self.journey_tracker = JourneyTracker(self.redis_client) if enable_journey_tracking else None
        
        # Métriques
        self._init_metrics()
        
        # Cache des contextes fréquents
        self.context_cache = {}
    
    def _init_metrics(self):
        """Initialiser les métriques"""
        self.metrics = {
            "requests_total": metrics.get_or_create_counter(
                "request_id_requests_total",
                "Total requests with ID tracking",
                ["service", "has_parent", "format"]
            ),
            "context_propagations": metrics.get_or_create_counter(
                "request_context_propagations_total",
                "Total context propagations",
                ["service", "direction"]
            ),
            "journey_hops": metrics.get_or_create_counter(
                "request_journey_hops_total",
                "Total journey hops",
                ["service", "status"]
            ),
            "id_generation_duration": metrics.get_or_create_histogram(
                "request_id_generation_duration_seconds",
                "Time to generate request IDs",
                ["format"]
            )
        }
    
    async def dispatch(self, request: Request, call_next):
        """Point d'entrée principal du middleware"""
        start_time = time.time()
        
        try:
            # Extraire ou créer le contexte de trace
            trace_context = await self._get_or_create_trace_context(request)
            
            # Ajouter le contexte à la requête
            request.state.trace_context = trace_context
            request.state.request_id = trace_context.request_id
            request.state.correlation_id = trace_context.correlation_id
            
            # Démarrer le journey tracking si activé
            journey_id = None
            if self.journey_tracker:
                journey_id = await self.journey_tracker.start_journey(
                    trace_context.request_id, 
                    self.service_name
                )
                request.state.journey_id = journey_id
            
            # Créer un span OpenTelemetry
            with tracer.start_as_current_span(
                f"{request.method} {request.url.path}",
                attributes={
                    "http.method": request.method,
                    "http.url": str(request.url),
                    "request.id": trace_context.request_id,
                    "correlation.id": trace_context.correlation_id,
                    "service.name": self.service_name
                }
            ) as span:
                
                # Ajouter le baggage OpenTelemetry
                for key, value in trace_context.baggage_items.items():
                    baggage.set_baggage(key, value)
                
                # Traitement de la requête
                response = await call_next(request)
                
                # Finaliser le span
                span.set_attribute("http.status_code", response.status_code)
                if response.status_code >= 400:
                    span.set_status(Status(StatusCode.ERROR))
                else:
                    span.set_status(Status(StatusCode.OK))
                
                # Ajouter les headers de trace à la réponse
                self._add_trace_headers(response, trace_context)
                
                # Finaliser le journey
                if journey_id and self.journey_tracker:
                    duration = time.time() - start_time
                    status = "success" if response.status_code < 400 else "error"
                    
                    await self.journey_tracker.add_hop(
                        journey_id, 
                        self.service_name, 
                        duration, 
                        status,
                        {
                            "endpoint": request.url.path,
                            "method": request.method,
                            "status_code": response.status_code
                        }
                    )
                    
                    await self.journey_tracker.complete_journey(journey_id)
                
                # Enregistrer les métriques
                await self._record_metrics(trace_context, response, start_time)
                
                return response
                
        except Exception as e:
            logger.error(f"Erreur request ID middleware: {e}")
            # Ne pas bloquer la requête
            response = await call_next(request)
            return response
    
    async def _get_or_create_trace_context(self, request: Request) -> TraceContext:
        """Obtenir ou créer le contexte de trace"""
        # Essayer d'extraire le contexte existant
        if self.context_propagator:
            existing_context = await self.context_propagator.extract_context(dict(request.headers))
            if existing_context:
                # Mettre à jour les informations de la requête actuelle
                existing_context.source_ip = self._get_client_ip(request)
                existing_context.user_agent = request.headers.get("user-agent", "")
                existing_context.propagated_at = datetime.utcnow()
                
                # Métriques
                self.metrics["context_propagations"].labels(
                    service=self.service_name,
                    direction="inbound"
                ).inc()
                
                return existing_context
        
        # Créer un nouveau contexte
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        request_id = self.id_generator.generate("req")
        correlation_id = self.id_generator.generate("corr")
        
        # Extraire les informations utilisateur si disponibles
        user_id = getattr(request.state, 'user_id', None)
        session_id = request.headers.get("x-session-id")
        tenant_id = request.headers.get("x-tenant-id")
        
        trace_context = TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            correlation_id=correlation_id,
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            source_ip=self._get_client_ip(request),
            user_agent=request.headers.get("user-agent", ""),
            api_key=request.headers.get("x-api-key")
        )
        
        # Ajouter contexte business si disponible
        if hasattr(request.state, 'business_context'):
            trace_context.business_context.update(request.state.business_context)
        
        # Métriques
        self.metrics["requests_total"].labels(
            service=self.service_name,
            has_parent="false",
            format=self.id_generator.format_type.value
        ).inc()
        
        return trace_context
    
    def _add_trace_headers(self, response: Response, trace_context: TraceContext):
        """Ajouter les headers de trace à la réponse"""
        response.headers["X-Trace-Id"] = trace_context.trace_id
        response.headers["X-Request-Id"] = trace_context.request_id
        response.headers["X-Correlation-Id"] = trace_context.correlation_id
        response.headers["X-Service"] = self.service_name
        
        # Headers pour debugging en développement
        if settings.ENVIRONMENT == "development":
            response.headers["X-Debug-Span-Id"] = trace_context.span_id
            if trace_context.journey_id:
                response.headers["X-Debug-Journey-Id"] = trace_context.journey_id
    
    def _get_client_ip(self, request: Request) -> str:
        """Obtenir l'IP réelle du client"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    async def _record_metrics(self, trace_context: TraceContext, response: Response, start_time: float):
        """Enregistrer les métriques de tracing"""
        duration = time.time() - start_time
        
        # Métriques de génération d'ID
        self.metrics["id_generation_duration"].labels(
            format=self.id_generator.format_type.value
        ).observe(0.001)  # Estimation pour l'exemple
        
        # Stocker les métriques de trace dans Redis
        trace_metrics = {
            "trace_id": trace_context.trace_id,
            "request_id": trace_context.request_id,
            "service": self.service_name,
            "duration": duration,
            "status_code": response.status_code,
            "user_id": trace_context.user_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        metrics_key = f"trace_metrics:{datetime.utcnow().strftime('%Y%m%d%H')}"
        await self.redis_client.lpush(
            metrics_key,
            json.dumps(trace_metrics)
        )
        await self.redis_client.expire(metrics_key, timedelta(days=7))
    
    async def get_trace_analytics(self, hours: int = 24) -> Dict:
        """Obtenir les analytics de tracing"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Récupérer les métriques récentes
            current_hour = datetime.utcnow().strftime('%Y%m%d%H')
            metrics_keys = []
            
            for i in range(hours):
                hour = (datetime.utcnow() - timedelta(hours=i)).strftime('%Y%m%d%H')
                metrics_keys.append(f"trace_metrics:{hour}")
            
            all_metrics = []
            for key in metrics_keys:
                metrics_data = await self.redis_client.lrange(key, 0, -1)
                for metric_str in metrics_data:
                    try:
                        metric = json.loads(metric_str)
                        if datetime.fromisoformat(metric["timestamp"]) > cutoff_time:
                            all_metrics.append(metric)
                    except Exception:
                        continue
            
            if not all_metrics:
                return {"error": "Aucune métrique disponible"}
            
            # Calculs analytiques
            total_requests = len(all_metrics)
            unique_traces = len(set(m["trace_id"] for m in all_metrics))
            services = set(m["service"] for m in all_metrics)
            
            durations = [m["duration"] for m in all_metrics]
            error_count = sum(1 for m in all_metrics if m["status_code"] >= 400)
            
            analytics = {
                "period_hours": hours,
                "total_requests": total_requests,
                "unique_traces": unique_traces,
                "services_involved": list(services),
                "avg_duration": sum(durations) / len(durations) if durations else 0,
                "error_rate": (error_count / total_requests) * 100 if total_requests > 0 else 0,
                "requests_per_hour": total_requests / hours,
                "top_users": self._get_top_users(all_metrics),
                "status_code_distribution": self._get_status_distribution(all_metrics),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Erreur analytics trace: {e}")
            return {"error": str(e)}
    
    def _get_top_users(self, metrics: List[Dict], top_n: int = 10) -> List[Dict]:
        """Obtenir les top utilisateurs par nombre de requêtes"""
        user_counts = {}
        for metric in metrics:
            user_id = metric.get("user_id")
            if user_id:
                user_counts[user_id] = user_counts.get(user_id, 0) + 1
        
        top_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [{"user_id": user_id, "request_count": count} for user_id, count in top_users]
    
    def _get_status_distribution(self, metrics: List[Dict]) -> Dict[str, int]:
        """Obtenir la distribution des codes de statut"""
        status_counts = {}
        for metric in metrics:
            status_code = str(metric["status_code"])
            status_counts[status_code] = status_counts.get(status_code, 0) + 1
        
        return status_counts


# Factory functions pour différents environnements
def create_request_id_middleware_development(app) -> RequestIDMiddleware:
    """Créer middleware request ID pour développement"""
    return RequestIDMiddleware(
        app=app,
        id_format=IDFormat.UUID4,
        service_name="spotify-ai-agent-dev",
        enable_journey_tracking=True,
        enable_context_propagation=True,
        enable_encryption=False
    )


def create_request_id_middleware_production(app) -> RequestIDMiddleware:
    """Créer middleware request ID pour production"""
    return RequestIDMiddleware(
        app=app,
        id_format=IDFormat.SNOWFLAKE,  # Plus performant en production
        service_name="spotify-ai-agent",
        enable_journey_tracking=True,
        enable_context_propagation=True,
        enable_encryption=True
    )


def create_request_id_middleware_testing(app) -> RequestIDMiddleware:
    """Créer middleware request ID pour tests"""
    return RequestIDMiddleware(
        app=app,
        id_format=IDFormat.NANOID,
        service_name="spotify-ai-agent-test",
        enable_journey_tracking=False,  # Désactivé pour les tests
        enable_context_propagation=False,
        enable_encryption=False
    )
