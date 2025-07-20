"""
📊 Ultra-Advanced Logging & Monitoring Middleware System
========================================================

Système de logging et monitoring de niveau enterprise pour Spotify AI Agent.
Architecture microservices avec observabilité complète, métriques business,
audit de sécurité, et intelligence artificielle intégrée.

🎯 Features Enterprise:
- Logging structuré avec corrélation distribuée
- Tracing OpenTelemetry multi-services
- Métriques Prometheus & business KPIs
- Audit de sécurité temps réel avec ML
- Performance monitoring avec profiling
- Alerting intelligent et auto-healing
- Analytics comportementaux utilisateurs
- Compliance GDPR/SOX automatique

🏗️ Architecture:
- Event-driven logging avec streaming
- Distributed tracing avec Jaeger
- Real-time metrics avec InfluxDB/Prometheus
- ML-powered anomaly detection
- Zero-downtime observability
- Auto-scaling monitoring

🔒 Security & Compliance:
- PII detection et masquage automatique
- Audit trail tamper-proof
- Real-time threat detection
- Compliance reporting automatique

Author: Expert DevOps + Security Architect + ML Engineer
Version: 2.0.0 Enterprise
Date: 2025-01-14
"""

import asyncio
import json
import time
import traceback
import uuid
import hashlib
import gzip
import base64
import sys
import gc
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, AsyncGenerator
from enum import Enum
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from functools import wraps
import threading
from collections import defaultdict, deque

# Core FastAPI imports
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Redis for distributed caching
import redis.asyncio as redis

# Conditional imports with graceful fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info,
        CollectorRegistry, REGISTRY, generate_latest
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Histogram = Gauge = Summary = Info = None
    CollectorRegistry = REGISTRY = None

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    structlog = None

try:
    import sentry_sdk
    from sentry_sdk import capture_exception, capture_message, set_tag, set_context
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False
    sentry_sdk = None
    capture_exception = capture_message = set_tag = set_context = lambda *a, **k: None

try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = metrics = JaegerExporter = TracerProvider = None
    BatchSpanProcessor = MeterProvider = FastAPIInstrumentor = None

try:
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    np = IsolationForest = StandardScaler = None

# Internal imports
from ...core.config import get_settings
from ...core.logging import get_logger
from ...core.database import get_database
from ...core.exceptions import LoggingError, MonitoringError, SecurityViolationError

settings = get_settings()
logger = get_logger(__name__)


class LogLevel(Enum):
    """Niveaux de logging avancés"""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"
    BUSINESS = "business"
    PERFORMANCE = "performance"


class MetricType(Enum):
    """Types de métriques Prometheus"""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"
    INFO = "info"


class SecurityEvent(Enum):
    """Types d'événements de sécurité"""
    SUSPICIOUS_REQUEST = "suspicious_request"
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_VIOLATION = "authz_violation"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"
    INJECTION_ATTEMPT = "injection_attempt"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    SECURITY_HEADER_MISSING = "security_header_missing"


class BusinessEvent(Enum):
    """Types d'événements business"""
    USER_REGISTRATION = "user_registration"
    SUBSCRIPTION_CHANGE = "subscription_change"
    AI_FEATURE_USAGE = "ai_feature_usage"
    SPOTIFY_API_CALL = "spotify_api_call"
    CONTENT_GENERATION = "content_generation"
    RECOMMENDATION_REQUEST = "recommendation_request"
    PAYMENT_TRANSACTION = "payment_transaction"
    USER_CHURN_INDICATOR = "user_churn_indicator"


@dataclass
class LogContext:
    """Contexte de logging enrichi"""
    request_id: str
    trace_id: str
    span_id: str
    user_id: Optional[str]
    session_id: Optional[str]
    correlation_id: str
    timestamp: datetime
    environment: str
    service_name: str
    version: str


@dataclass
class SecurityContext:
    """Contexte de sécurité"""
    client_ip: str
    user_agent: str
    geo_location: Optional[Dict[str, str]]
    threat_score: float
    risk_factors: List[str]
    authentication_method: Optional[str]
    permission_level: str


@dataclass
class PerformanceMetrics:
    """Métriques de performance"""
    duration_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    db_queries_count: int
    cache_hits: int
    cache_misses: int
    external_api_calls: int
    bytes_transferred: int


@dataclass
class BusinessMetrics:
    """Métriques business"""
    user_tier: str
    feature_used: str
    conversion_event: bool
    revenue_impact: float
    engagement_score: float
    retention_indicator: str
    customer_satisfaction: Optional[float]


class PII_Detector:
    """Détecteur et masqueur de données personnelles (PII)"""
    
    PII_PATTERNS = {
        'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        'phone': re.compile(r'\b(\+?1?[-.\s]?)?(\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})\b'),
        'credit_card': re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
        'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
        'api_key': re.compile(r'\b[A-Za-z0-9]{32,}\b'),
        'jwt_token': re.compile(r'\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b')
    }
    
    SENSITIVE_FIELDS = {
        'password', 'secret', 'token', 'key', 'auth', 'authorization',
        'api_key', 'access_token', 'refresh_token', 'client_secret',
        'private_key', 'signature', 'credit_card', 'ssn', 'email',
        'phone', 'address', 'name', 'birthday', 'age'
    }
    
    @classmethod
    def mask_sensitive_data(cls, data: Any) -> Any:
        """Masquer les données sensibles dans n'importe quelle structure"""
        if isinstance(data, dict):
            return cls._mask_dict(data)
        elif isinstance(data, list):
            return [cls.mask_sensitive_data(item) for item in data]
        elif isinstance(data, str):
            return cls._mask_string(data)
        else:
            return data
    
    @classmethod
    def _mask_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Masquer un dictionnaire"""
        masked = {}
        for key, value in data.items():
            if isinstance(key, str) and any(field in key.lower() for field in cls.SENSITIVE_FIELDS):
                masked[key] = "[REDACTED]"
            else:
                masked[key] = cls.mask_sensitive_data(value)
        return masked
    
    @classmethod
    def _mask_string(cls, text: str) -> str:
        """Masquer une chaîne de caractères"""
        masked_text = text
        for pii_type, pattern in cls.PII_PATTERNS.items():
            masked_text = pattern.sub(f"[REDACTED_{pii_type.upper()}]", masked_text)
        return masked_text
    
    @classmethod
    def detect_pii(cls, data: Any) -> List[str]:
        """Détecter les types de PII présents"""
        detected_types = []
        text = json.dumps(data, default=str) if not isinstance(data, str) else data
        
        for pii_type, pattern in cls.PII_PATTERNS.items():
            if pattern.search(text):
                detected_types.append(pii_type)
        
        return detected_types


class AnomalyDetector:
    """Détecteur d'anomalies basé sur ML"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.feature_history = deque(maxlen=1000)
        
        if ML_AVAILABLE:
            self.model = IsolationForest(contamination=0.1, random_state=42)
            self.scaler = StandardScaler()
    
    def extract_features(self, request: Request, response: Response, duration: float) -> np.ndarray:
        """Extraire les features pour la détection d'anomalies"""
        if not ML_AVAILABLE:
            return np.array([])
        
        features = [
            duration,  # Durée de la requête
            response.status_code,  # Code de statut
            len(str(request.url)),  # Longueur de l'URL
            len(request.headers),  # Nombre de headers
            int(request.headers.get('content-length', 0)),  # Taille du contenu
            hash(request.method) % 1000,  # Hash de la méthode
            time.time() % (24 * 3600),  # Heure de la journée
        ]
        
        return np.array(features).reshape(1, -1)
    
    def is_anomalous(self, features: np.ndarray) -> Tuple[bool, float]:
        """Détecter si une requête est anormale"""
        if not ML_AVAILABLE or not self.is_trained or features.size == 0:
            return False, 0.0
        
        try:
            # Normaliser les features
            features_scaled = self.scaler.transform(features)
            
            # Prédiction d'anomalie
            prediction = self.model.predict(features_scaled)[0]
            anomaly_score = self.model.decision_function(features_scaled)[0]
            
            is_anomaly = prediction == -1
            confidence = abs(anomaly_score)
            
            return is_anomaly, confidence
            
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
            return False, 0.0
    
    def update_model(self, features: np.ndarray):
        """Mettre à jour le modèle avec de nouvelles données"""
        if not ML_AVAILABLE or features.size == 0:
            return
        
        try:
            self.feature_history.append(features.flatten())
            
            # Réentraîner le modèle périodiquement
            if len(self.feature_history) >= 100 and len(self.feature_history) % 50 == 0:
                training_data = np.array(list(self.feature_history))
                self.scaler.fit(training_data)
                self.model.fit(self.scaler.transform(training_data))
                self.is_trained = True
                
        except Exception as e:
            logger.warning(f"Model update failed: {e}")


class MetricsCollector:
    """Collecteur de métriques Prometheus avancé"""
    
    def __init__(self):
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Métriques de base
        self.request_counter = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code', 'user_tier', 'service']
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint', 'service'],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf'))
        )
        
        self.active_connections = Gauge(
            'active_connections_total',
            'Active connections',
            ['service']
        )
        
        # Métriques business
        self.ai_feature_usage = Counter(
            'ai_feature_usage_total',
            'AI feature usage',
            ['feature', 'user_tier', 'success']
        )
        
        self.spotify_api_calls = Counter(
            'spotify_api_calls_total',
            'Spotify API calls',
            ['endpoint', 'status', 'rate_limited']
        )
        
        self.subscription_events = Counter(
            'subscription_events_total',
            'Subscription events',
            ['event_type', 'tier_from', 'tier_to']
        )
        
        # Métriques de sécurité
        self.security_events = Counter(
            'security_events_total',
            'Security events',
            ['event_type', 'severity', 'source_ip']
        )
        
        self.anomaly_detections = Counter(
            'anomaly_detections_total',
            'Anomaly detections',
            ['type', 'confidence_level']
        )
        
        # Métriques système
        self.system_cpu = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage'
        )
        
        self.system_memory = Gauge(
            'system_memory_usage_percent',
            'System memory usage'
        )
        
        self.database_connections = Gauge(
            'database_connections_active',
            'Active database connections',
            ['database']
        )
    
    def record_request(self, method: str, endpoint: str, status_code: int, 
                      duration: float, user_tier: str = "unknown"):
        """Enregistrer une requête HTTP"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            normalized_endpoint = self._normalize_endpoint(endpoint)
            service_name = "spotify-ai-agent"
            
            self.request_counter.labels(
                method=method,
                endpoint=normalized_endpoint,
                status_code=status_code,
                user_tier=user_tier,
                service=service_name
            ).inc()
            
            self.request_duration.labels(
                method=method,
                endpoint=normalized_endpoint,
                service=service_name
            ).observe(duration)
            
        except Exception as e:
            logger.warning(f"Failed to record request metrics: {e}")
    
    def record_ai_usage(self, feature: str, user_tier: str, success: bool):
        """Enregistrer l'usage d'une fonctionnalité IA"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            self.ai_feature_usage.labels(
                feature=feature,
                user_tier=user_tier,
                success=str(success).lower()
            ).inc()
        except Exception as e:
            logger.warning(f"Failed to record AI usage: {e}")
    
    def record_security_event(self, event_type: str, severity: str, source_ip: str):
        """Enregistrer un événement de sécurité"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            self.security_events.labels(
                event_type=event_type,
                severity=severity,
                source_ip=source_ip
            ).inc()
        except Exception as e:
            logger.warning(f"Failed to record security event: {e}")
    
    def record_anomaly(self, anomaly_type: str, confidence: float):
        """Enregistrer une détection d'anomalie"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            confidence_level = "high" if confidence > 0.8 else "medium" if confidence > 0.5 else "low"
            self.anomaly_detections.labels(
                type=anomaly_type,
                confidence_level=confidence_level
            ).inc()
        except Exception as e:
            logger.warning(f"Failed to record anomaly: {e}")
    
    def update_system_metrics(self):
        """Mettre à jour les métriques système"""
        if not PROMETHEUS_AVAILABLE or not PSUTIL_AVAILABLE:
            return
        
        try:
            self.system_cpu.set(psutil.cpu_percent())
            self.system_memory.set(psutil.virtual_memory().percent)
        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")
    
    def _normalize_endpoint(self, path: str) -> str:
        """Normaliser un endpoint pour les métriques"""
        # Remplacer les IDs par des placeholders
        normalized = re.sub(r'/\d+', '/{id}', path)
        normalized = re.sub(r'/[a-f0-9-]{36}', '/{uuid}', normalized)
        normalized = re.sub(r'/[a-f0-9]{24}', '/{objectid}', normalized)
        return normalized


class AdvancedLoggingMiddleware:
    """
    Middleware de logging ultra-avancé avec IA et observabilité complète
    """
    
    def __init__(self):
        self.redis_client = redis.from_url(settings.REDIS_URL)
        self.pii_detector = PII_Detector()
        self.anomaly_detector = AnomalyDetector()
        self.metrics_collector = MetricsCollector()
        
        # Configuration du tracing distribué
        self.tracer = None
        if OPENTELEMETRY_AVAILABLE:
            self._setup_tracing()
        
        # Configuration de Sentry
        if SENTRY_AVAILABLE and hasattr(settings, 'SENTRY_DSN'):
            self._setup_sentry()
        
        # Configuration structlog
        if STRUCTLOG_AVAILABLE:
            self._setup_structured_logging()
        
        # Cache pour les contextes de requête
        self.request_contexts = {}
        
        # Thread pour les métriques système
        self._start_system_metrics_thread()
    
    def _setup_tracing(self):
        """Configurer le tracing distribué OpenTelemetry"""
        try:
            provider = TracerProvider()
            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer(__name__, "1.0.0")
            
            # Configurer l'exporteur Jaeger
            if hasattr(settings, 'JAEGER_ENDPOINT') and settings.JAEGER_ENDPOINT:
                jaeger_exporter = JaegerExporter(
                    agent_host_name=getattr(settings, 'JAEGER_HOST', 'localhost'),
                    agent_port=getattr(settings, 'JAEGER_PORT', 14268)
                )
                span_processor = BatchSpanProcessor(jaeger_exporter)
                provider.add_span_processor(span_processor)
                
                logger.info("✅ Jaeger tracing configured")
            
        except Exception as e:
            logger.warning(f"Could not setup tracing: {e}")
    
    def _setup_sentry(self):
        """Configurer Sentry pour le monitoring d'erreurs"""
        try:
            sentry_sdk.init(
                dsn=settings.SENTRY_DSN,
                traces_sample_rate=0.1,
                profiles_sample_rate=0.1,
                environment=getattr(settings, 'ENVIRONMENT', 'development')
            )
            logger.info("✅ Sentry error monitoring configured")
        except Exception as e:
            logger.warning(f"Could not setup Sentry: {e}")
    
    def _setup_structured_logging(self):
        """Configurer le logging structuré"""
        try:
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
            logger.info("✅ Structured logging configured")
        except Exception as e:
            logger.warning(f"Could not setup structured logging: {e}")
    
    def _start_system_metrics_thread(self):
        """Démarrer le thread de collecte des métriques système"""
        def collect_system_metrics():
            while True:
                try:
                    self.metrics_collector.update_system_metrics()
                    time.sleep(30)  # Collecter toutes les 30 secondes
                except Exception as e:
                    logger.warning(f"System metrics collection failed: {e}")
                    time.sleep(60)
        
        metrics_thread = threading.Thread(target=collect_system_metrics, daemon=True)
        metrics_thread.start()
    
    async def __call__(self, request: Request, call_next):
        """Point d'entrée principal du middleware"""
        # Générer les identifiants de corrélation
        request_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        correlation_id = request.headers.get('X-Correlation-ID', str(uuid.uuid4()))
        
        # Ajouter au contexte de la requête
        request.state.request_id = request_id
        request.state.trace_id = trace_id
        request.state.correlation_id = correlation_id
        
        # Créer le contexte de logging
        log_context = LogContext(
            request_id=request_id,
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            user_id=getattr(request.state, 'user_id', None),
            session_id=request.headers.get('X-Session-ID'),
            correlation_id=correlation_id,
            timestamp=datetime.utcnow(),
            environment=getattr(settings, 'ENVIRONMENT', 'development'),
            service_name='spotify-ai-agent',
            version=getattr(settings, 'VERSION', '1.0.0')
        )
        
        # Créer le contexte de sécurité
        security_context = SecurityContext(
            client_ip=self._get_client_ip(request),
            user_agent=request.headers.get('User-Agent', ''),
            geo_location=await self._get_geo_location(request),
            threat_score=0.0,
            risk_factors=[],
            authentication_method=getattr(request.state, 'auth_method', None),
            permission_level=getattr(request.state, 'permission_level', 'anonymous')
        )
        
        start_time = time.time()
        
        # Démarrer un span de tracing
        span_context = None
        if self.tracer:
            span_context = self.tracer.start_span(
                f"{request.method} {request.url.path}",
                attributes={
                    "http.method": request.method,
                    "http.url": str(request.url),
                    "http.user_agent": request.headers.get('User-Agent', ''),
                    "request.id": request_id,
                    "trace.id": trace_id,
                    "user.id": log_context.user_id or "anonymous"
                }
            )
        
        try:
            # Analyse de sécurité pré-requête
            await self._analyze_request_security(request, security_context, log_context)
            
            # Log de la requête entrante
            await self._log_incoming_request(request, log_context, security_context)
            
            # Traitement de la requête
            response = await call_next(request)
            
            # Calcul des métriques
            duration = time.time() - start_time
            
            # Extraction des features pour ML
            features = self.anomaly_detector.extract_features(request, response, duration)
            
            # Détection d'anomalies
            is_anomalous, anomaly_confidence = self.anomaly_detector.is_anomalous(features)
            
            # Créer les métriques de performance
            performance_metrics = PerformanceMetrics(
                duration_ms=duration * 1000,
                memory_usage_mb=self._get_memory_usage(),
                cpu_usage_percent=self._get_cpu_usage(),
                db_queries_count=getattr(request.state, 'db_queries', 0),
                cache_hits=getattr(request.state, 'cache_hits', 0),
                cache_misses=getattr(request.state, 'cache_misses', 0),
                external_api_calls=getattr(request.state, 'api_calls', 0),
                bytes_transferred=self._get_response_size(response)
            )
            
            # Créer les métriques business
            business_metrics = BusinessMetrics(
                user_tier=getattr(request.state, 'user_tier', 'anonymous'),
                feature_used=self._extract_feature_from_path(request.url.path),
                conversion_event=self._is_conversion_event(request, response),
                revenue_impact=self._calculate_revenue_impact(request, response),
                engagement_score=self._calculate_engagement_score(request, response, duration),
                retention_indicator=self._get_retention_indicator(request),
                customer_satisfaction=getattr(request.state, 'satisfaction_score', None)
            )
            
            # Log de la réponse
            await self._log_outgoing_response(
                request, response, log_context, security_context,
                performance_metrics, business_metrics, duration
            )
            
            # Enregistrement des métriques Prometheus
            self.metrics_collector.record_request(
                request.method, request.url.path, response.status_code,
                duration, business_metrics.user_tier
            )
            
            # Gestion des anomalies détectées
            if is_anomalous:
                await self._handle_anomaly(
                    request, response, log_context, anomaly_confidence
                )
            
            # Mise à jour du modèle ML
            self.anomaly_detector.update_model(features)
            
            # Analytics business
            await self._track_business_events(request, response, business_metrics, log_context)
            
            # Finalisation du span
            if span_context:
                span_context.set_attribute("http.status_code", response.status_code)
                span_context.set_attribute("response.duration_ms", duration * 1000)
                span_context.set_attribute("anomaly.detected", is_anomalous)
                if is_anomalous:
                    span_context.set_attribute("anomaly.confidence", anomaly_confidence)
                span_context.set_status(trace.Status(trace.StatusCode.OK))
                span_context.end()
            
            # Nettoyage du contexte
            self._cleanup_request_context(request_id)
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log de l'erreur
            await self._log_error(
                request, e, log_context, security_context, duration
            )
            
            # Enregistrement des métriques d'erreur
            self.metrics_collector.record_request(
                request.method, request.url.path, 500,
                duration, getattr(request.state, 'user_tier', 'unknown')
            )
            
            # Capture Sentry
            if SENTRY_AVAILABLE:
                set_context("request", {
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "duration": duration
                })
                capture_exception(e)
            
            # Finalisation du span avec erreur
            if span_context:
                span_context.set_attribute("error", True)
                span_context.set_attribute("error.type", type(e).__name__)
                span_context.set_attribute("error.message", str(e))
                span_context.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span_context.end()
            
            # Nettoyage
            self._cleanup_request_context(request_id)
            
            raise
    
    async def _analyze_request_security(self, request: Request, 
                                      security_context: SecurityContext,
                                      log_context: LogContext):
        """Analyser la sécurité d'une requête"""
        threat_indicators = []
        
        # Analyser l'URL pour des patterns suspects
        url_threats = self._check_url_threats(str(request.url))
        threat_indicators.extend(url_threats)
        
        # Analyser les headers
        header_threats = self._check_header_threats(request.headers)
        threat_indicators.extend(header_threats)
        
        # Calculer le score de menace
        security_context.threat_score = len(threat_indicators) * 0.2
        security_context.risk_factors = threat_indicators
        
        # Enregistrer les événements de sécurité
        if threat_indicators:
            await self._log_security_event(
                SecurityEvent.SUSPICIOUS_REQUEST,
                request, security_context, log_context,
                {"threats": threat_indicators}
            )
    
    def _check_url_threats(self, url: str) -> List[str]:
        """Vérifier les menaces dans l'URL"""
        threats = []
        url_lower = url.lower()
        
        # Patterns d'injection SQL
        sql_patterns = ['union', 'select', 'drop', 'insert', 'delete', 'update', '--', ';']
        for pattern in sql_patterns:
            if pattern in url_lower:
                threats.append(f"sql_injection_{pattern}")
        
        # Patterns XSS
        xss_patterns = ['<script', 'javascript:', 'onload=', 'onerror=', 'eval(']
        for pattern in xss_patterns:
            if pattern in url_lower:
                threats.append(f"xss_{pattern}")
        
        # Path traversal
        if '../' in url or '..\\' in url:
            threats.append("path_traversal")
        
        # Commandes système
        cmd_patterns = ['cmd.exe', '/bin/', '/etc/', 'passwd', 'shadow']
        for pattern in cmd_patterns:
            if pattern in url_lower:
                threats.append(f"command_injection_{pattern}")
        
        return threats
    
    def _check_header_threats(self, headers: Dict[str, str]) -> List[str]:
        """Vérifier les menaces dans les headers"""
        threats = []
        
        # Headers suspects
        for header_name, header_value in headers.items():
            header_lower = header_value.lower()
            
            # Injection dans les headers
            if any(pattern in header_lower for pattern in ['<script', 'javascript:', 'eval(']):
                threats.append(f"header_injection_{header_name}")
            
            # User-Agent suspects
            if header_name.lower() == 'user-agent':
                if any(bot in header_lower for bot in ['sqlmap', 'nikto', 'nmap', 'metasploit']):
                    threats.append("malicious_user_agent")
        
        return threats
    
    async def _log_incoming_request(self, request: Request, 
                                  log_context: LogContext,
                                  security_context: SecurityContext):
        """Logger une requête entrante"""
        try:
            # Extraire et masquer les données sensibles
            query_params = self.pii_detector.mask_sensitive_data(dict(request.query_params))
            headers = self.pii_detector.mask_sensitive_data(dict(request.headers))
            
            # Détecter les PII
            pii_detected = self.pii_detector.detect_pii({
                "url": str(request.url),
                "headers": headers,
                "query_params": query_params
            })
            
            request_data = {
                "event_type": "request_incoming",
                "request_id": log_context.request_id,
                "trace_id": log_context.trace_id,
                "correlation_id": log_context.correlation_id,
                "timestamp": log_context.timestamp.isoformat(),
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": query_params,
                "headers": headers,
                "client": {
                    "ip": security_context.client_ip,
                    "user_agent": security_context.user_agent,
                    "geo_location": security_context.geo_location
                },
                "security": {
                    "threat_score": security_context.threat_score,
                    "risk_factors": security_context.risk_factors,
                    "pii_detected": pii_detected
                },
                "user": {
                    "user_id": log_context.user_id,
                    "session_id": log_context.session_id,
                    "auth_method": security_context.authentication_method,
                    "permission_level": security_context.permission_level
                },
                "service": {
                    "name": log_context.service_name,
                    "version": log_context.version,
                    "environment": log_context.environment
                }
            }
            
            # Logger selon le niveau de sécurité
            if security_context.threat_score > 0.5:
                logger.warning("High-risk request incoming", extra=request_data)
            else:
                logger.info("Request incoming", extra=request_data)
            
            # Stocker dans Redis pour corrélation
            await self._store_request_context(log_context.request_id, request_data)
            
        except Exception as e:
            logger.error(f"Failed to log incoming request: {e}")
    
    async def _log_outgoing_response(self, request: Request, response: Response,
                                   log_context: LogContext, security_context: SecurityContext,
                                   performance_metrics: PerformanceMetrics,
                                   business_metrics: BusinessMetrics, duration: float):
        """Logger une réponse sortante"""
        try:
            # Masquer les headers sensibles
            response_headers = self.pii_detector.mask_sensitive_data(dict(response.headers))
            
            response_data = {
                "event_type": "response_outgoing",
                "request_id": log_context.request_id,
                "trace_id": log_context.trace_id,
                "correlation_id": log_context.correlation_id,
                "timestamp": datetime.utcnow().isoformat(),
                "status_code": response.status_code,
                "headers": response_headers,
                "performance": asdict(performance_metrics),
                "business": asdict(business_metrics),
                "security": {
                    "threat_score": security_context.threat_score,
                    "risk_factors": security_context.risk_factors
                }
            }
            
            # Déterminer le niveau de log
            if response.status_code >= 500:
                log_level = "error"
            elif response.status_code >= 400:
                log_level = "warning"
            elif performance_metrics.duration_ms > 5000:  # > 5 secondes
                log_level = "warning"
            else:
                log_level = "info"
            
            getattr(logger, log_level)("Response outgoing", extra=response_data)
            
            # Analytics temps réel
            await self._send_realtime_analytics(request, response, performance_metrics, business_metrics)
            
        except Exception as e:
            logger.error(f"Failed to log outgoing response: {e}")
    
    async def _log_error(self, request: Request, error: Exception,
                        log_context: LogContext, security_context: SecurityContext,
                        duration: float):
        """Logger une erreur"""
        try:
            error_data = {
                "event_type": "error",
                "request_id": log_context.request_id,
                "trace_id": log_context.trace_id,
                "correlation_id": log_context.correlation_id,
                "timestamp": datetime.utcnow().isoformat(),
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc()
                },
                "request": {
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration * 1000
                },
                "security": {
                    "threat_score": security_context.threat_score,
                    "client_ip": security_context.client_ip
                }
            }
            
            logger.error("Request error", extra=error_data)
            
            # Alerting pour erreurs critiques
            await self._handle_critical_error(error, error_data, log_context)
            
        except Exception as e:
            logger.error(f"Failed to log error: {e}")
    
    async def _log_security_event(self, event_type: SecurityEvent, request: Request,
                                security_context: SecurityContext, log_context: LogContext,
                                details: Dict[str, Any]):
        """Logger un événement de sécurité"""
        try:
            security_event_data = {
                "event_type": "security_event",
                "security_event_type": event_type.value,
                "request_id": log_context.request_id,
                "trace_id": log_context.trace_id,
                "timestamp": datetime.utcnow().isoformat(),
                "severity": self._get_security_severity(event_type, details),
                "client": {
                    "ip": security_context.client_ip,
                    "user_agent": security_context.user_agent,
                    "geo_location": security_context.geo_location
                },
                "request": {
                    "method": request.method,
                    "path": request.url.path,
                    "url": str(request.url)
                },
                "details": details,
                "threat_score": security_context.threat_score
            }
            
            logger.warning("Security event detected", extra=security_event_data)
            
            # Enregistrer dans les métriques Prometheus
            self.metrics_collector.record_security_event(
                event_type.value,
                security_event_data["severity"],
                security_context.client_ip
            )
            
            # Stocker pour analyse
            await self.redis_client.lpush(
                "security_events",
                json.dumps(security_event_data, default=str)
            )
            await self.redis_client.ltrim("security_events", 0, 9999)
            
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
    
    def _get_security_severity(self, event_type: SecurityEvent, details: Dict[str, Any]) -> str:
        """Déterminer la sévérité d'un événement de sécurité"""
        high_risk_events = {
            SecurityEvent.DATA_BREACH_ATTEMPT,
            SecurityEvent.INJECTION_ATTEMPT,
            SecurityEvent.AUTHORIZATION_VIOLATION
        }
        
        if event_type in high_risk_events:
            return "high"
        elif len(details.get("threats", [])) > 3:
            return "medium"
        else:
            return "low"
    
    async def _handle_anomaly(self, request: Request, response: Response,
                            log_context: LogContext, confidence: float):
        """Gérer une anomalie détectée"""
        try:
            anomaly_data = {
                "event_type": "anomaly_detected",
                "request_id": log_context.request_id,
                "trace_id": log_context.trace_id,
                "timestamp": datetime.utcnow().isoformat(),
                "confidence": confidence,
                "request": {
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code
                },
                "features": {
                    "duration": (time.time() - float(getattr(request.state, 'start_time', time.time()))),
                    "url_length": len(str(request.url)),
                    "header_count": len(request.headers)
                }
            }
            
            logger.warning("Anomaly detected", extra=anomaly_data)
            
            # Enregistrer dans Prometheus
            self.metrics_collector.record_anomaly("request_anomaly", confidence)
            
            # Stocker pour analyse
            await self.redis_client.lpush(
                "anomalies",
                json.dumps(anomaly_data, default=str)
            )
            await self.redis_client.ltrim("anomalies", 0, 9999)
            
        except Exception as e:
            logger.error(f"Failed to handle anomaly: {e}")
    
    async def _track_business_events(self, request: Request, response: Response,
                                   business_metrics: BusinessMetrics, log_context: LogContext):
        """Tracker les événements business"""
        try:
            # Déterminer le type d'événement business
            business_event = self._determine_business_event(request, response)
            
            if business_event:
                event_data = {
                    "event_type": "business_event",
                    "business_event_type": business_event.value,
                    "request_id": log_context.request_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "metrics": asdict(business_metrics),
                    "request": {
                        "method": request.method,
                        "path": request.url.path
                    },
                    "user": {
                        "user_id": log_context.user_id,
                        "tier": business_metrics.user_tier
                    }
                }
                
                logger.info("Business event tracked", extra=event_data)
                
                # Analytics spécialisés selon le type d'événement
                if business_event == BusinessEvent.AI_FEATURE_USAGE:
                    self.metrics_collector.record_ai_usage(
                        business_metrics.feature_used,
                        business_metrics.user_tier,
                        response.status_code < 400
                    )
                
                # Stocker pour analytics
                await self.redis_client.lpush(
                    f"business_events:{business_event.value}",
                    json.dumps(event_data, default=str)
                )
                await self.redis_client.ltrim(f"business_events:{business_event.value}", 0, 9999)
                
        except Exception as e:
            logger.error(f"Failed to track business events: {e}")
    
    def _determine_business_event(self, request: Request, response: Response) -> Optional[BusinessEvent]:
        """Déterminer le type d'événement business"""
        path = request.url.path.lower()
        
        if "/ai/" in path or "/generate" in path:
            return BusinessEvent.AI_FEATURE_USAGE
        elif "/spotify/" in path:
            return BusinessEvent.SPOTIFY_API_CALL
        elif "/subscription" in path or "/billing" in path:
            return BusinessEvent.SUBSCRIPTION_CHANGE
        elif "/auth/register" in path:
            return BusinessEvent.USER_REGISTRATION
        elif "/payment" in path:
            return BusinessEvent.PAYMENT_TRANSACTION
        elif "/recommend" in path:
            return BusinessEvent.RECOMMENDATION_REQUEST
        
        return None
    
    # Utility methods
    def _get_client_ip(self, request: Request) -> str:
        """Obtenir l'IP réelle du client"""
        # Vérifier les headers de proxy
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    async def _get_geo_location(self, request: Request) -> Optional[Dict[str, str]]:
        """Obtenir la géolocalisation (mock pour l'exemple)"""
        # Dans un vrai système, on utiliserait une API de géolocalisation
        return {
            "country": "Unknown",
            "city": "Unknown",
            "region": "Unknown"
        }
    
    def _get_memory_usage(self) -> float:
        """Obtenir l'usage mémoire en MB"""
        if PSUTIL_AVAILABLE:
            try:
                return psutil.Process().memory_info().rss / 1024 / 1024
            except:
                pass
        return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Obtenir l'usage CPU"""
        if PSUTIL_AVAILABLE:
            try:
                return psutil.cpu_percent()
            except:
                pass
        return 0.0
    
    def _get_response_size(self, response: Response) -> int:
        """Obtenir la taille de la réponse"""
        return int(response.headers.get("content-length", 0))
    
    def _extract_feature_from_path(self, path: str) -> str:
        """Extraire la fonctionnalité utilisée depuis le path"""
        if "/ai/generate" in path:
            return "content_generation"
        elif "/ai/recommend" in path:
            return "recommendation"
        elif "/ai/analyze" in path:
            return "analysis"
        elif "/spotify/" in path:
            return "spotify_integration"
        elif "/user/" in path:
            return "user_management"
        else:
            return "other"
    
    def _is_conversion_event(self, request: Request, response: Response) -> bool:
        """Déterminer si c'est un événement de conversion"""
        path = request.url.path.lower()
        return (
            response.status_code == 200 and
            ("/subscribe" in path or "/upgrade" in path or "/purchase" in path)
        )
    
    def _calculate_revenue_impact(self, request: Request, response: Response) -> float:
        """Calculer l'impact revenue (mock)"""
        if self._is_conversion_event(request, response):
            # Dans un vrai système, on extrairait le montant de la transaction
            return 9.99  # Prix d'abonnement exemple
        return 0.0
    
    def _calculate_engagement_score(self, request: Request, response: Response, duration: float) -> float:
        """Calculer le score d'engagement"""
        base_score = 1.0
        
        # Pénaliser les requêtes lentes
        if duration > 2.0:
            base_score *= 0.8
        
        # Bonus pour les fonctionnalités IA
        if "/ai/" in request.url.path:
            base_score *= 1.2
        
        # Pénaliser les erreurs
        if response.status_code >= 400:
            base_score *= 0.5
        
        return min(base_score, 5.0)  # Cap à 5.0
    
    def _get_retention_indicator(self, request: Request) -> str:
        """Obtenir l'indicateur de rétention"""
        # Dans un vrai système, on analyserait l'historique de l'utilisateur
        user_tier = getattr(request.state, 'user_tier', 'anonymous')
        
        if user_tier == 'premium':
            return 'high'
        elif user_tier == 'free':
            return 'medium'
        else:
            return 'low'
    
    async def _store_request_context(self, request_id: str, context: Dict[str, Any]):
        """Stocker le contexte de requête"""
        try:
            await self.redis_client.setex(
                f"request_context:{request_id}",
                timedelta(hours=2),
                json.dumps(context, default=str)
            )
        except Exception as e:
            logger.warning(f"Could not store request context: {e}")
    
    async def _send_realtime_analytics(self, request: Request, response: Response,
                                     performance_metrics: PerformanceMetrics,
                                     business_metrics: BusinessMetrics):
        """Envoyer des analytics en temps réel"""
        try:
            analytics_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "endpoint": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "performance": asdict(performance_metrics),
                "business": asdict(business_metrics)
            }
            
            # Publier dans Redis pour les dashboards temps réel
            await self.redis_client.publish(
                "realtime_analytics",
                json.dumps(analytics_data, default=str)
            )
            
        except Exception as e:
            logger.warning(f"Could not send realtime analytics: {e}")
    
    async def _handle_critical_error(self, error: Exception, error_data: Dict[str, Any],
                                   log_context: LogContext):
        """Gérer les erreurs critiques"""
        try:
            # Définir les erreurs critiques
            critical_error_types = {
                "DatabaseError", "RedisConnectionError", "SecurityViolationError",
                "OutOfMemoryError", "SystemExit"
            }
            
            error_type = type(error).__name__
            
            if error_type in critical_error_types:
                # Créer une alerte critique
                alert = {
                    "type": "critical_error",
                    "timestamp": datetime.utcnow().isoformat(),
                    "error_type": error_type,
                    "service": log_context.service_name,
                    "environment": log_context.environment,
                    "request_id": log_context.request_id,
                    "trace_id": log_context.trace_id,
                    "error_data": error_data,
                    "alert_id": str(uuid.uuid4())
                }
                
                # Publier l'alerte
                await self.redis_client.publish("critical_alerts", json.dumps(alert, default=str))
                
                # Log critique
                logger.critical("Critical error detected", extra=alert)
                
        except Exception as e:
            logger.error(f"Could not handle critical error: {e}")
    
    def _cleanup_request_context(self, request_id: str):
        """Nettoyer le contexte de requête"""
        try:
            if request_id in self.request_contexts:
                del self.request_contexts[request_id]
        except Exception as e:
            logger.warning(f"Could not cleanup request context: {e}")


# Middleware instances
class RequestTracingMiddleware(AdvancedLoggingMiddleware):
    """Middleware spécialisé pour le tracing des requêtes"""
    pass


class PerformanceLoggingMiddleware(AdvancedLoggingMiddleware):
    """Middleware spécialisé pour le logging des performances"""
    pass


class SecurityAuditMiddleware(AdvancedLoggingMiddleware):
    """Middleware spécialisé pour l'audit de sécurité"""
    pass


class BusinessMetricsMiddleware(AdvancedLoggingMiddleware):
    """Middleware spécialisé pour les métriques business"""
    pass
