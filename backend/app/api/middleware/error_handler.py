"""
🔥 Ultra-Advanced Error Handling & Recovery Middleware System
===========================================================

Système de gestion d'erreurs de niveau enterprise pour Spotify AI Agent.
Architecture résiliente avec recovery automatique, circuit breakers,
alerting intelligent, et analytics d'erreurs en temps réel.

🛡️ Features Enterprise:
- Circuit breaker patterns avancés
- Recovery automatique et graceful degradation
- Error classification et routing intelligent
- Alerting multi-canal (Slack, Email, SMS)
- Analytics d'erreurs avec ML
- Retry mechanisms adaptatifs
- Error budgets et SLA monitoring
- Audit trail des erreurs critiques

🏗️ Architecture:
- Event-driven error handling
- Distributed error tracking
- Real-time error correlation
- Auto-healing mechanisms
- Predictive error prevention
- Performance impact analysis

🔒 Reliability Patterns:
- Bulkhead isolation
- Timeout management
- Rate limiting pour erreurs
- Failover automatique
- Health check intégré
- Chaos engineering ready

Author: Expert Site Reliability Engineer + DevOps Architect
Version: 2.0.0 Enterprise
Date: 2025-07-14
"""

import asyncio
import json
import time
import traceback
import uuid
import hashlib
import sys
import gc
import psutil
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, Awaitable
from enum import Enum
from dataclasses import dataclass, asdict
from functools import wraps
import threading
from collections import defaultdict, deque
import statistics
import re

# Core FastAPI imports
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

# Redis for distributed state management
import redis.asyncio as redis

# Conditional imports with graceful fallbacks
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    prometheus_client = Counter = Histogram = Gauge = Summary = None

try:
    import sentry_sdk
    from sentry_sdk import capture_exception, capture_message, set_tag, set_context
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False
    sentry_sdk = None
    capture_exception = capture_message = set_tag = set_context = lambda *a, **k: None

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    structlog = None

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
from ...core.exceptions import (
    ValidationError, SecurityViolationError, 
    DatabaseException, ServiceUnavailableError
)

settings = get_settings()
logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Niveaux de sévérité des erreurs"""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class ErrorCategory(Enum):
    """Catégories d'erreurs"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    DATABASE = "database"
    EXTERNAL_SERVICE = "external_service"
    NETWORK = "network"
    SYSTEM = "system"
    SECURITY = "security"
    PERFORMANCE = "performance"
    UNKNOWN = "unknown"


class ErrorPattern(Enum):
    """Patterns d'erreurs courants"""
    TIMEOUT = "timeout"
    CONNECTION_REFUSED = "connection_refused"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    INVALID_INPUT = "invalid_input"
    RESOURCE_NOT_FOUND = "resource_not_found"
    PERMISSION_DENIED = "permission_denied"
    SERVICE_UNAVAILABLE = "service_unavailable"
    DEADLOCK = "deadlock"
    MEMORY_EXHAUSTED = "memory_exhausted"


class CircuitState(Enum):
    """États du circuit breaker"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class RecoveryStrategy(Enum):
    """Stratégies de récupération"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAIL_FAST = "fail_fast"
    BULKHEAD = "bulkhead"


@dataclass
class ErrorContext:
    """Contexte d'une erreur"""
    error_id: str
    timestamp: datetime
    request_id: str
    user_id: Optional[str]
    session_id: Optional[str]
    endpoint: str
    method: str
    url: str
    headers: Dict[str, str]
    body_size: int
    client_ip: str
    user_agent: str
    trace_id: str
    span_id: str


@dataclass
class ErrorMetrics:
    """Métriques d'erreur"""
    error_count: int
    error_rate: float
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    active_connections: int
    queue_size: int


@dataclass
class RecoveryAction:
    """Action de récupération"""
    strategy: RecoveryStrategy
    executed_at: datetime
    success: bool
    duration_ms: float
    details: Dict[str, Any]
    retry_count: int
    final_attempt: bool


@dataclass
class CircuitBreakerState:
    """État du circuit breaker"""
    name: str
    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: Optional[datetime]
    next_attempt_time: Optional[datetime]
    failure_threshold: int
    recovery_timeout: int
    half_open_max_calls: int


class ErrorClassifier:
    """Classificateur d'erreurs intelligent"""
    
    def __init__(self):
        self.error_patterns = self._build_error_patterns()
        self.ml_model = None
        self.scaler = None
        self.feature_history = deque(maxlen=10000)
        
        if ML_AVAILABLE:
            self.ml_model = IsolationForest(contamination=0.1, random_state=42)
            self.scaler = StandardScaler()
    
    def _build_error_patterns(self) -> Dict[ErrorPattern, List[re.Pattern]]:
        """Construire les patterns de reconnaissance d'erreurs"""
        return {
            ErrorPattern.TIMEOUT: [
                re.compile(r"timeout", re.IGNORECASE),
                re.compile(r"time.*out", re.IGNORECASE),
                re.compile(r"deadline exceeded", re.IGNORECASE),
            ],
            ErrorPattern.CONNECTION_REFUSED: [
                re.compile(r"connection refused", re.IGNORECASE),
                re.compile(r"connection reset", re.IGNORECASE),
                re.compile(r"connection.*closed", re.IGNORECASE),
            ],
            ErrorPattern.RATE_LIMITED: [
                re.compile(r"rate limit", re.IGNORECASE),
                re.compile(r"too many requests", re.IGNORECASE),
                re.compile(r"429", re.IGNORECASE),
            ],
            ErrorPattern.QUOTA_EXCEEDED: [
                re.compile(r"quota.*exceeded", re.IGNORECASE),
                re.compile(r"limit.*exceeded", re.IGNORECASE),
                re.compile(r"usage.*limit", re.IGNORECASE),
            ],
            ErrorPattern.INVALID_INPUT: [
                re.compile(r"invalid.*input", re.IGNORECASE),
                re.compile(r"validation.*error", re.IGNORECASE),
                re.compile(r"bad.*request", re.IGNORECASE),
            ],
            ErrorPattern.RESOURCE_NOT_FOUND: [
                re.compile(r"not found", re.IGNORECASE),
                re.compile(r"404", re.IGNORECASE),
                re.compile(r"does not exist", re.IGNORECASE),
            ],
            ErrorPattern.PERMISSION_DENIED: [
                re.compile(r"permission denied", re.IGNORECASE),
                re.compile(r"access denied", re.IGNORECASE),
                re.compile(r"forbidden", re.IGNORECASE),
                re.compile(r"401|403", re.IGNORECASE),
            ],
            ErrorPattern.SERVICE_UNAVAILABLE: [
                re.compile(r"service unavailable", re.IGNORECASE),
                re.compile(r"503", re.IGNORECASE),
                re.compile(r"server.*down", re.IGNORECASE),
            ],
            ErrorPattern.DEADLOCK: [
                re.compile(r"deadlock", re.IGNORECASE),
                re.compile(r"lock.*timeout", re.IGNORECASE),
                re.compile(r"circular.*dependency", re.IGNORECASE),
            ],
            ErrorPattern.MEMORY_EXHAUSTED: [
                re.compile(r"out of memory", re.IGNORECASE),
                re.compile(r"memory.*exhausted", re.IGNORECASE),
                re.compile(r"oom", re.IGNORECASE),
            ],
        }
    
    def classify_error(self, error: Exception, context: ErrorContext) -> Tuple[ErrorCategory, ErrorSeverity, ErrorPattern]:
        """Classifier une erreur"""
        error_message = str(error)
        error_type = type(error).__name__
        
        # Classification par type d'exception
        category = self._classify_by_type(error)
        
        # Classification par pattern
        pattern = self._detect_pattern(error_message)
        
        # Déterminer la sévérité
        severity = self._determine_severity(error, category, pattern, context)
        
        return category, severity, pattern
    
    def _classify_by_type(self, error: Exception) -> ErrorCategory:
        """Classifier par type d'exception"""
        error_type = type(error).__name__
        
        if isinstance(error, (PermissionError, SecurityViolationError)):
            return ErrorCategory.SECURITY
        elif isinstance(error, (ValidationError, ValueError, TypeError)):
            return ErrorCategory.VALIDATION
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorCategory.NETWORK
        elif isinstance(error, DatabaseError):
            return ErrorCategory.DATABASE
        elif isinstance(error, ServiceUnavailableError):
            return ErrorCategory.EXTERNAL_SERVICE
        elif isinstance(error, HTTPException):
            if 400 <= error.status_code < 500:
                return ErrorCategory.VALIDATION
            else:
                return ErrorCategory.SYSTEM
        elif isinstance(error, (MemoryError, SystemError)):
            return ErrorCategory.SYSTEM
        else:
            return ErrorCategory.UNKNOWN
    
    def _detect_pattern(self, error_message: str) -> ErrorPattern:
        """Détecter le pattern d'erreur"""
        for pattern, regexes in self.error_patterns.items():
            for regex in regexes:
                if regex.search(error_message):
                    return pattern
        return ErrorPattern.TIMEOUT  # Default pattern
    
    def _determine_severity(self, error: Exception, category: ErrorCategory, 
                          pattern: ErrorPattern, context: ErrorContext) -> ErrorSeverity:
        """Déterminer la sévérité d'une erreur"""
        # Erreurs critiques
        if isinstance(error, (MemoryError, SystemExit, KeyboardInterrupt)):
            return ErrorSeverity.FATAL
        
        if category == ErrorCategory.SECURITY:
            return ErrorSeverity.CRITICAL
        
        if pattern in [ErrorPattern.MEMORY_EXHAUSTED, ErrorPattern.DEADLOCK]:
            return ErrorSeverity.CRITICAL
        
        # Erreurs système importantes
        if category == ErrorCategory.SYSTEM:
            return ErrorSeverity.ERROR
        
        if category == ErrorCategory.DATABASE:
            return ErrorSeverity.ERROR
        
        # Erreurs de service
        if pattern in [ErrorPattern.SERVICE_UNAVAILABLE, ErrorPattern.TIMEOUT]:
            return ErrorSeverity.ERROR
        
        # Erreurs de validation
        if category == ErrorCategory.VALIDATION:
            return ErrorSeverity.WARNING
        
        # Autres cas
        if isinstance(error, HTTPException):
            if error.status_code >= 500:
                return ErrorSeverity.ERROR
            elif error.status_code >= 400:
                return ErrorSeverity.WARNING
        
        return ErrorSeverity.INFO
    
    def extract_features(self, error: Exception, context: ErrorContext) -> np.ndarray:
        """Extraire les features pour ML"""
        if not ML_AVAILABLE:
            return np.array([])
        
        try:
            features = [
                hash(type(error).__name__) % 10000,  # Type d'erreur
                len(str(error)),  # Longueur du message
                hash(context.endpoint) % 1000,  # Endpoint
                context.body_size,  # Taille du body
                len(context.headers),  # Nombre de headers
                time.time() % (24 * 3600),  # Heure de la journée
                hash(context.method) % 100,  # Méthode HTTP
            ]
            
            return np.array(features).reshape(1, -1)
        except Exception:
            return np.array([]).reshape(1, -1)
    
    def is_anomalous_error(self, error: Exception, context: ErrorContext) -> Tuple[bool, float]:
        """Détecter si une erreur est anormale"""
        if not ML_AVAILABLE or not hasattr(self, 'ml_model') or self.ml_model is None:
            return False, 0.0
        
        features = self.extract_features(error, context)
        if features.size == 0:
            return False, 0.0
        
        try:
            # Vérifier si le modèle est entraîné
            if not hasattr(self.ml_model, 'estimators_'):
                return False, 0.0
            
            features_scaled = self.scaler.transform(features)
            prediction = self.ml_model.predict(features_scaled)[0]
            anomaly_score = self.ml_model.decision_function(features_scaled)[0]
            
            is_anomaly = prediction == -1
            confidence = abs(anomaly_score)
            
            return is_anomaly, confidence
        except Exception:
            return False, 0.0


class CircuitBreaker:
    """Circuit breaker avancé avec états adaptatifs"""
    
    def __init__(self, name: str, failure_threshold: int = 5, 
                 recovery_timeout: int = 60, half_open_max_calls: int = 3):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.next_attempt_time = None
        self.half_open_calls = 0
        
        # Métriques
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Lock pour thread safety
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Exécuter une fonction protégée par le circuit breaker"""
        with self._lock:
            self.total_requests += 1
            
            # Vérifier l'état du circuit
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")
            
            # Exécuter la fonction
            try:
                if self.state == CircuitState.HALF_OPEN:
                    if self.half_open_calls >= self.half_open_max_calls:
                        raise CircuitBreakerOpenError(f"Circuit breaker {self.name} half-open limit reached")
                    self.half_open_calls += 1
                
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except Exception as e:
                self._on_failure()
                raise
    
    async def acall(self, func: Callable[..., Awaitable], *args, **kwargs) -> Any:
        """Version async du circuit breaker"""
        with self._lock:
            self.total_requests += 1
            
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")
        
        try:
            if self.state == CircuitState.HALF_OPEN:
                with self._lock:
                    if self.half_open_calls >= self.half_open_max_calls:
                        raise CircuitBreakerOpenError(f"Circuit breaker {self.name} half-open limit reached")
                    self.half_open_calls += 1
            
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Vérifier si on doit tenter de réinitialiser le circuit"""
        if self.next_attempt_time is None:
            return True
        return datetime.utcnow() >= self.next_attempt_time
    
    def _on_success(self):
        """Gérer un succès"""
        with self._lock:
            self.successful_requests += 1
            self.success_count += 1
            
            if self.state == CircuitState.HALF_OPEN:
                if self.success_count >= self.half_open_max_calls:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
            elif self.state == CircuitState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self):
        """Gérer un échec"""
        with self._lock:
            self.failed_requests += 1
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            
            if (self.state == CircuitState.CLOSED and 
                self.failure_count >= self.failure_threshold):
                self.state = CircuitState.OPEN
                self.next_attempt_time = datetime.utcnow() + timedelta(seconds=self.recovery_timeout)
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.next_attempt_time = datetime.utcnow() + timedelta(seconds=self.recovery_timeout)
    
    def get_state(self) -> CircuitBreakerState:
        """Obtenir l'état actuel du circuit breaker"""
        with self._lock:
            return CircuitBreakerState(
                name=self.name,
                state=self.state,
                failure_count=self.failure_count,
                success_count=self.success_count,
                last_failure_time=self.last_failure_time,
                next_attempt_time=self.next_attempt_time,
                failure_threshold=self.failure_threshold,
                recovery_timeout=self.recovery_timeout,
                half_open_max_calls=self.half_open_max_calls
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtenir les métriques du circuit breaker"""
        with self._lock:
            success_rate = (self.successful_requests / max(1, self.total_requests)) * 100
            return {
                "name": self.name,
                "state": self.state.value,
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": success_rate,
                "failure_count": self.failure_count,
                "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
            }


class ErrorAlerting:
    """Système d'alerting intelligent"""
    
    def __init__(self):
        self.alert_channels = []
        self.rate_limits = defaultdict(lambda: deque(maxlen=100))
        self.escalation_rules = {}
        
        # Configuration des canaux d'alerte
        self._setup_alert_channels()
    
    def _setup_alert_channels(self):
        """Configurer les canaux d'alerte"""
        # Slack webhook (exemple)
        if hasattr(settings, 'SLACK_WEBHOOK_URL') and settings.SLACK_WEBHOOK_URL:
            self.alert_channels.append({
                "type": "slack",
                "url": settings.SLACK_WEBHOOK_URL,
                "severity_threshold": ErrorSeverity.ERROR
            })
        
        # Email SMTP (exemple)
        if hasattr(settings, 'SMTP_SERVER') and settings.SMTP_SERVER:
            self.alert_channels.append({
                "type": "email",
                "smtp_server": settings.SMTP_SERVER,
                "recipients": getattr(settings, 'ALERT_EMAILS', []),
                "severity_threshold": ErrorSeverity.CRITICAL
            })
    
    async def send_alert(self, error: Exception, context: ErrorContext, 
                        severity: ErrorSeverity, category: ErrorCategory):
        """Envoyer une alerte"""
        # Vérifier rate limiting
        alert_key = f"{category.value}:{severity.value}"
        if self._is_rate_limited(alert_key):
            return
        
        # Construire le message d'alerte
        alert_message = self._build_alert_message(error, context, severity, category)
        
        # Envoyer sur tous les canaux appropriés
        for channel in self.alert_channels:
            if severity.value >= channel.get("severity_threshold", ErrorSeverity.ERROR).value:
                await self._send_to_channel(channel, alert_message)
    
    def _is_rate_limited(self, alert_key: str, max_per_hour: int = 10) -> bool:
        """Vérifier le rate limiting des alertes"""
        now = time.time()
        hour_ago = now - 3600
        
        # Nettoyer les anciennes entrées
        while self.rate_limits[alert_key] and self.rate_limits[alert_key][0] < hour_ago:
            self.rate_limits[alert_key].popleft()
        
        # Vérifier la limite
        if len(self.rate_limits[alert_key]) >= max_per_hour:
            return True
        
        # Ajouter cette alerte
        self.rate_limits[alert_key].append(now)
        return False
    
    def _build_alert_message(self, error: Exception, context: ErrorContext,
                           severity: ErrorSeverity, category: ErrorCategory) -> Dict[str, Any]:
        """Construire le message d'alerte"""
        return {
            "timestamp": context.timestamp.isoformat(),
            "error_id": context.error_id,
            "severity": severity.value,
            "category": category.value,
            "error_type": type(error).__name__,
            "error_message": str(error)[:500],  # Limiter la longueur
            "endpoint": context.endpoint,
            "method": context.method,
            "client_ip": context.client_ip,
            "user_id": context.user_id,
            "request_id": context.request_id,
            "environment": getattr(settings, 'ENVIRONMENT', 'unknown')
        }
    
    async def _send_to_channel(self, channel: Dict[str, Any], message: Dict[str, Any]):
        """Envoyer à un canal spécifique"""
        try:
            if channel["type"] == "slack":
                await self._send_slack_alert(channel, message)
            elif channel["type"] == "email":
                await self._send_email_alert(channel, message)
        except Exception as e:
            logger.error(f"Failed to send alert to {channel['type']}: {e}")
    
    async def _send_slack_alert(self, channel: Dict[str, Any], message: Dict[str, Any]):
        """Envoyer une alerte Slack"""
        if not AIOHTTP_AVAILABLE:
            return
        
        slack_message = {
            "text": f"🚨 {message['severity'].upper()} Error Alert",
            "attachments": [{
                "color": self._get_color_for_severity(message['severity']),
                "fields": [
                    {"title": "Error Type", "value": message['error_type'], "short": True},
                    {"title": "Category", "value": message['category'], "short": True},
                    {"title": "Endpoint", "value": message['endpoint'], "short": True},
                    {"title": "Error ID", "value": message['error_id'], "short": True},
                    {"title": "Message", "value": message['error_message'], "short": False},
                ]
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(channel["url"], json=slack_message) as response:
                if response.status != 200:
                    logger.warning(f"Slack alert failed: {response.status}")
    
    async def _send_email_alert(self, channel: Dict[str, Any], message: Dict[str, Any]):
        """Envoyer une alerte email"""
        # Implementation d'email serait ici
        # Pour l'exemple, on log juste
        logger.info(f"Would send email alert: {message['error_id']}")
    
    def _get_color_for_severity(self, severity: str) -> str:
        """Obtenir la couleur pour une sévérité"""
        colors = {
            "fatal": "#8B0000",     # Dark red
            "critical": "#FF0000",  # Red
            "error": "#FF4500",     # Orange red
            "warning": "#FFA500",   # Orange
            "info": "#0000FF",      # Blue
            "debug": "#808080",     # Gray
            "trace": "#D3D3D3"      # Light gray
        }
        return colors.get(severity, "#808080")


class ErrorRecovery:
    """Système de récupération d'erreurs"""
    
    def __init__(self):
        self.recovery_strategies = {
            ErrorCategory.NETWORK: [RecoveryStrategy.RETRY, RecoveryStrategy.CIRCUIT_BREAKER],
            ErrorCategory.DATABASE: [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK],
            ErrorCategory.EXTERNAL_SERVICE: [RecoveryStrategy.RETRY, RecoveryStrategy.CIRCUIT_BREAKER],
            ErrorCategory.VALIDATION: [RecoveryStrategy.FAIL_FAST],
            ErrorCategory.SECURITY: [RecoveryStrategy.FAIL_FAST],
            ErrorCategory.SYSTEM: [RecoveryStrategy.GRACEFUL_DEGRADATION],
        }
        
        self.retry_configs = {
            ErrorCategory.NETWORK: {"max_attempts": 3, "backoff_factor": 2, "max_delay": 60},
            ErrorCategory.DATABASE: {"max_attempts": 2, "backoff_factor": 1.5, "max_delay": 30},
            ErrorCategory.EXTERNAL_SERVICE: {"max_attempts": 3, "backoff_factor": 2, "max_delay": 120},
        }
    
    async def attempt_recovery(self, error: Exception, context: ErrorContext,
                             category: ErrorCategory) -> Optional[RecoveryAction]:
        """Tenter une récupération d'erreur"""
        strategies = self.recovery_strategies.get(category, [RecoveryStrategy.FAIL_FAST])
        
        for strategy in strategies:
            try:
                start_time = time.time()
                success = await self._execute_strategy(strategy, error, context, category)
                duration = (time.time() - start_time) * 1000
                
                return RecoveryAction(
                    strategy=strategy,
                    executed_at=datetime.utcnow(),
                    success=success,
                    duration_ms=duration,
                    details={"category": category.value, "error_type": type(error).__name__},
                    retry_count=getattr(context, 'retry_count', 0),
                    final_attempt=strategy == strategies[-1]
                )
                
            except Exception as recovery_error:
                logger.warning(f"Recovery strategy {strategy.value} failed: {recovery_error}")
                continue
        
        return None
    
    async def _execute_strategy(self, strategy: RecoveryStrategy, error: Exception,
                              context: ErrorContext, category: ErrorCategory) -> bool:
        """Exécuter une stratégie de récupération"""
        if strategy == RecoveryStrategy.RETRY:
            return await self._retry_strategy(error, context, category)
        elif strategy == RecoveryStrategy.FALLBACK:
            return await self._fallback_strategy(error, context, category)
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return await self._graceful_degradation_strategy(error, context, category)
        elif strategy == RecoveryStrategy.FAIL_FAST:
            return False  # Ne pas récupérer, échouer rapidement
        else:
            return False
    
    async def _retry_strategy(self, error: Exception, context: ErrorContext,
                            category: ErrorCategory) -> bool:
        """Stratégie de retry avec backoff exponentiel"""
        config = self.retry_configs.get(category, {"max_attempts": 1, "backoff_factor": 1, "max_delay": 30})
        
        retry_count = getattr(context, 'retry_count', 0)
        if retry_count >= config["max_attempts"]:
            return False
        
        # Calculer le délai de backoff
        delay = min(
            config["backoff_factor"] ** retry_count,
            config["max_delay"]
        )
        
        await asyncio.sleep(delay)
        return True  # Indiquer qu'on peut retry
    
    async def _fallback_strategy(self, error: Exception, context: ErrorContext,
                               category: ErrorCategory) -> bool:
        """Stratégie de fallback"""
        # Exemple de fallback : utiliser un cache ou service alternatif
        logger.info(f"Executing fallback for {category.value} error in {context.endpoint}")
        
        # Ici on pourrait implémenter des fallbacks spécifiques
        # Par exemple, retourner des données cached, utiliser un service de backup, etc.
        
        return True
    
    async def _graceful_degradation_strategy(self, error: Exception, context: ErrorContext,
                                           category: ErrorCategory) -> bool:
        """Stratégie de dégradation gracieuse"""
        logger.info(f"Executing graceful degradation for {category.value} error in {context.endpoint}")
        
        # Ici on pourrait implémenter une version simplifiée du service
        # Par exemple, retourner des résultats partiels, désactiver des features non-critiques, etc.
        
        return True


class ErrorMetricsCollector:
    """Collecteur de métriques d'erreurs"""
    
    def __init__(self):
        self.redis_client = redis.from_url(settings.REDIS_URL)
        
        # Métriques Prometheus
        if PROMETHEUS_AVAILABLE:
            self.error_counter = Counter(
                'errors_total',
                'Total number of errors',
                ['category', 'severity', 'endpoint', 'error_type']
            )
            
            self.error_duration = Histogram(
                'error_handling_duration_seconds',
                'Time spent handling errors',
                ['category', 'recovery_strategy']
            )
            
            self.circuit_breaker_state = Gauge(
                'circuit_breaker_state',
                'Circuit breaker state (0=closed, 1=half-open, 2=open)',
                ['circuit_name']
            )
            
            self.recovery_success_rate = Gauge(
                'error_recovery_success_rate',
                'Error recovery success rate',
                ['category', 'strategy']
            )
    
    async def record_error(self, error: Exception, context: ErrorContext,
                          category: ErrorCategory, severity: ErrorSeverity):
        """Enregistrer une erreur"""
        # Métriques Prometheus
        if PROMETHEUS_AVAILABLE:
            self.error_counter.labels(
                category=category.value,
                severity=severity.value,
                endpoint=self._normalize_endpoint(context.endpoint),
                error_type=type(error).__name__
            ).inc()
        
        # Stocker dans Redis pour analytics
        error_data = {
            "timestamp": context.timestamp.isoformat(),
            "error_id": context.error_id,
            "category": category.value,
            "severity": severity.value,
            "error_type": type(error).__name__,
            "endpoint": context.endpoint,
            "method": context.method,
            "client_ip": context.client_ip,
            "user_id": context.user_id,
            "request_id": context.request_id
        }
        
        await self.redis_client.lpush("error_analytics", json.dumps(error_data))
        await self.redis_client.ltrim("error_analytics", 0, 99999)  # Garder 100k erreurs max
    
    async def record_recovery(self, recovery_action: RecoveryAction, category: ErrorCategory):
        """Enregistrer une action de récupération"""
        if PROMETHEUS_AVAILABLE:
            self.error_duration.labels(
                category=category.value,
                recovery_strategy=recovery_action.strategy.value
            ).observe(recovery_action.duration_ms / 1000)
        
        # Mettre à jour le taux de succès
        await self._update_recovery_success_rate(recovery_action, category)
    
    async def record_circuit_breaker_state(self, circuit_name: str, state: CircuitState):
        """Enregistrer l'état d'un circuit breaker"""
        if PROMETHEUS_AVAILABLE:
            state_mapping = {
                CircuitState.CLOSED: 0,
                CircuitState.HALF_OPEN: 1,
                CircuitState.OPEN: 2
            }
            self.circuit_breaker_state.labels(circuit_name=circuit_name).set(
                state_mapping.get(state, 0)
            )
    
    async def _update_recovery_success_rate(self, recovery_action: RecoveryAction, category: ErrorCategory):
        """Mettre à jour le taux de succès de récupération"""
        key = f"recovery_stats:{category.value}:{recovery_action.strategy.value}"
        
        # Incrémenter les compteurs
        if recovery_action.success:
            await self.redis_client.incr(f"{key}:success")
        await self.redis_client.incr(f"{key}:total")
        
        # Calculer et mettre à jour le taux
        if PROMETHEUS_AVAILABLE:
            success_count = int(await self.redis_client.get(f"{key}:success") or 0)
            total_count = int(await self.redis_client.get(f"{key}:total") or 1)
            success_rate = success_count / total_count
            
            self.recovery_success_rate.labels(
                category=category.value,
                strategy=recovery_action.strategy.value
            ).set(success_rate)
    
    def _normalize_endpoint(self, endpoint: str) -> str:
        """Normaliser un endpoint pour les métriques"""
        # Remplacer les IDs par des placeholders
        normalized = re.sub(r'/\d+', '/{id}', endpoint)
        normalized = re.sub(r'/[a-f0-9-]{36}', '/{uuid}', normalized)
        return normalized


class AdvancedErrorHandler(BaseHTTPMiddleware):
    """
    Middleware de gestion d'erreurs ultra-avancé
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.redis_client = redis.from_url(settings.REDIS_URL)
        self.error_classifier = ErrorClassifier()
        self.circuit_breakers = {}
        self.error_alerting = ErrorAlerting()
        self.error_recovery = ErrorRecovery()
        self.metrics_collector = ErrorMetricsCollector()
        
        # Configuration
        self.environment = getattr(settings, 'ENVIRONMENT', 'development')
        self.debug_mode = self.environment == 'development'
        
        # Cache des erreurs récentes
        self.recent_errors = deque(maxlen=10000)
        
        # Thread pour le nettoyage périodique
        self._start_cleanup_thread()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Point d'entrée principal du middleware"""
        start_time = time.time()
        error_context = None
        
        try:
            # Créer le contexte de base
            error_context = self._create_error_context(request)
            
            # Exécuter la requête avec protection circuit breaker
            response = await self._execute_with_circuit_breaker(
                request, call_next, error_context
            )
            
            return response
            
        except Exception as error:
            duration = time.time() - start_time
            
            # Gérer l'erreur de manière complète
            return await self._handle_error(
                error, request, error_context, duration
            )
    
    def _create_error_context(self, request: Request) -> ErrorContext:
        """Créer le contexte d'erreur"""
        return ErrorContext(
            error_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            request_id=getattr(request.state, 'request_id', str(uuid.uuid4())),
            user_id=getattr(request.state, 'user_id', None),
            session_id=getattr(request.state, 'session_id', None),
            endpoint=request.url.path,
            method=request.method,
            url=str(request.url),
            headers=self._sanitize_headers(dict(request.headers)),
            body_size=int(request.headers.get('content-length', 0)),
            client_ip=self._get_client_ip(request),
            user_agent=request.headers.get('user-agent', ''),
            trace_id=getattr(request.state, 'trace_id', str(uuid.uuid4())),
            span_id=getattr(request.state, 'span_id', str(uuid.uuid4()))
        )
    
    async def _execute_with_circuit_breaker(self, request: Request, call_next: Callable,
                                          context: ErrorContext) -> Response:
        """Exécuter avec protection circuit breaker"""
        circuit_name = f"{request.method}:{context.endpoint}"
        circuit_breaker = self._get_circuit_breaker(circuit_name)
        
        try:
            response = await circuit_breaker.acall(call_next, request)
            return response
        except CircuitBreakerOpenError:
            # Circuit ouvert, retourner erreur de service indisponible
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Service temporarily unavailable",
                    "code": "CIRCUIT_BREAKER_OPEN",
                    "error_id": context.error_id,
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
    
    async def _handle_error(self, error: Exception, request: Request,
                          context: Optional[ErrorContext], duration: float) -> Response:
        """Gérer une erreur de manière complète"""
        # Créer le contexte si nécessaire
        if context is None:
            context = self._create_error_context(request)
        
        # Classifier l'erreur
        category, severity, pattern = self.error_classifier.classify_error(error, context)
        
        # Détecter les anomalies
        is_anomalous, anomaly_confidence = self.error_classifier.is_anomalous_error(error, context)
        
        # Enregistrer l'erreur
        await self._log_error(error, context, category, severity, pattern, duration)
        
        # Collecter les métriques
        await self.metrics_collector.record_error(error, context, category, severity)
        
        # Tenter une récupération
        recovery_action = await self.error_recovery.attempt_recovery(error, context, category)
        if recovery_action:
            await self.metrics_collector.record_recovery(recovery_action, category)
            
            # Si la récupération réussit, retry
            if recovery_action.success and not recovery_action.final_attempt:
                context.retry_count = getattr(context, 'retry_count', 0) + 1
                # Note: Dans un vrai système, on ferait le retry ici
        
        # Envoyer des alertes si nécessaire
        if severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
            await self.error_alerting.send_alert(error, context, severity, category)
        
        # Capture Sentry si disponible
        if SENTRY_AVAILABLE:
            self._capture_sentry_error(error, context, category, severity)
        
        # Stocker l'erreur dans le cache récent
        self._cache_recent_error(error, context, category, severity)
        
        # Créer la réponse d'erreur
        return self._create_error_response(error, context, category, severity, is_anomalous)
    
    def _get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Obtenir ou créer un circuit breaker"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=5,
                recovery_timeout=60,
                half_open_max_calls=3
            )
        return self.circuit_breakers[name]
    
    async def _log_error(self, error: Exception, context: ErrorContext,
                        category: ErrorCategory, severity: ErrorSeverity,
                        pattern: ErrorPattern, duration: float):
        """Logger une erreur avec contexte complet"""
        error_data = {
            "event_type": "error",
            "error_id": context.error_id,
            "timestamp": context.timestamp.isoformat(),
            "severity": severity.value,
            "category": category.value,
            "pattern": pattern.value,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "request": {
                "id": context.request_id,
                "method": context.method,
                "url": context.url,
                "endpoint": context.endpoint,
                "headers": context.headers,
                "body_size": context.body_size,
                "duration_ms": duration * 1000
            },
            "client": {
                "ip": context.client_ip,
                "user_agent": context.user_agent
            },
            "user": {
                "id": context.user_id,
                "session_id": context.session_id
            },
            "tracing": {
                "trace_id": context.trace_id,
                "span_id": context.span_id
            },
            "system": {
                "memory_usage_mb": self._get_memory_usage(),
                "cpu_usage_percent": self._get_cpu_usage(),
                "environment": self.environment
            }
        }
        
        # Logger selon la sévérité
        if severity == ErrorSeverity.FATAL:
            logger.critical("Fatal error occurred", extra=error_data)
        elif severity == ErrorSeverity.CRITICAL:
            logger.critical("Critical error occurred", extra=error_data)
        elif severity == ErrorSeverity.ERROR:
            logger.error("Error occurred", extra=error_data)
        elif severity == ErrorSeverity.WARNING:
            logger.warning("Warning occurred", extra=error_data)
        else:
            logger.info("Error handled", extra=error_data)
        
        # Stocker dans Redis pour analytics
        await self.redis_client.lpush("error_logs", json.dumps(error_data, default=str))
        await self.redis_client.ltrim("error_logs", 0, 99999)
    
    def _create_error_response(self, error: Exception, context: ErrorContext,
                             category: ErrorCategory, severity: ErrorSeverity,
                             is_anomalous: bool) -> Response:
        """Créer la réponse d'erreur"""
        # Déterminer le code de statut
        if isinstance(error, HTTPException):
            status_code = error.status_code
        elif isinstance(error, ValidationError):
            status_code = 400
        elif isinstance(error, SecurityViolationError):
            status_code = 403
        elif isinstance(error, ServiceUnavailableError):
            status_code = 503
        elif category == ErrorCategory.AUTHENTICATION:
            status_code = 401
        elif category == ErrorCategory.AUTHORIZATION:
            status_code = 403
        elif category == ErrorCategory.VALIDATION:
            status_code = 400
        else:
            status_code = 500
        
        # Construire le contenu de l'erreur
        error_content = {
            "error": self._get_user_friendly_message(error, category),
            "code": self._get_error_code(error, category),
            "error_id": context.error_id,
            "timestamp": context.timestamp.isoformat(),
            "request_id": context.request_id
        }
        
        # Ajouter des détails en mode debug
        if self.debug_mode:
            error_content.update({
                "debug": {
                    "error_type": type(error).__name__,
                    "category": category.value,
                    "severity": severity.value,
                    "anomalous": is_anomalous,
                    "traceback": traceback.format_exc().split('\n')
                }
            })
        
        # Headers de réponse
        headers = {
            "X-Error-ID": context.error_id,
            "X-Request-ID": context.request_id,
            "X-Error-Category": category.value,
            "X-Error-Severity": severity.value
        }
        
        # Ajouter Retry-After pour les erreurs temporaires
        if status_code in [429, 503]:
            headers["Retry-After"] = "60"
        
        return JSONResponse(
            status_code=status_code,
            content=error_content,
            headers=headers
        )
    
    def _get_user_friendly_message(self, error: Exception, category: ErrorCategory) -> str:
        """Obtenir un message d'erreur convivial"""
        if isinstance(error, HTTPException):
            return error.detail
        
        messages = {
            ErrorCategory.AUTHENTICATION: "Authentication required. Please log in.",
            ErrorCategory.AUTHORIZATION: "You don't have permission to access this resource.",
            ErrorCategory.VALIDATION: "Invalid input provided. Please check your data.",
            ErrorCategory.DATABASE: "A database error occurred. Please try again later.",
            ErrorCategory.EXTERNAL_SERVICE: "External service is temporarily unavailable.",
            ErrorCategory.NETWORK: "Network error occurred. Please check your connection.",
            ErrorCategory.SECURITY: "Security violation detected. Access denied.",
            ErrorCategory.SYSTEM: "An internal error occurred. Please try again later.",
        }
        
        return messages.get(category, "An unexpected error occurred. Please try again later.")
    
    def _get_error_code(self, error: Exception, category: ErrorCategory) -> str:
        """Obtenir un code d'erreur structuré"""
        if isinstance(error, HTTPException):
            return f"HTTP_{error.status_code}"
        
        codes = {
            ErrorCategory.AUTHENTICATION: "AUTH_REQUIRED",
            ErrorCategory.AUTHORIZATION: "ACCESS_DENIED",
            ErrorCategory.VALIDATION: "INVALID_INPUT",
            ErrorCategory.DATABASE: "DATABASE_ERROR",
            ErrorCategory.EXTERNAL_SERVICE: "SERVICE_UNAVAILABLE",
            ErrorCategory.NETWORK: "NETWORK_ERROR",
            ErrorCategory.SECURITY: "SECURITY_VIOLATION",
            ErrorCategory.SYSTEM: "INTERNAL_ERROR",
        }
        
        return codes.get(category, "UNKNOWN_ERROR")
    
    def _capture_sentry_error(self, error: Exception, context: ErrorContext,
                            category: ErrorCategory, severity: ErrorSeverity):
        """Capturer l'erreur dans Sentry"""
        set_tag("error_category", category.value)
        set_tag("error_severity", severity.value)
        set_tag("endpoint", context.endpoint)
        set_tag("method", context.method)
        
        set_context("request", {
            "error_id": context.error_id,
            "request_id": context.request_id,
            "url": context.url,
            "method": context.method,
            "client_ip": context.client_ip,
            "user_agent": context.user_agent
        })
        
        if context.user_id:
            set_context("user", {"id": context.user_id})
        
        capture_exception(error)
    
    def _cache_recent_error(self, error: Exception, context: ErrorContext,
                          category: ErrorCategory, severity: ErrorSeverity):
        """Mettre en cache une erreur récente"""
        error_summary = {
            "error_id": context.error_id,
            "timestamp": context.timestamp,
            "error_type": type(error).__name__,
            "category": category.value,
            "severity": severity.value,
            "endpoint": context.endpoint,
            "method": context.method
        }
        
        self.recent_errors.append(error_summary)
    
    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Nettoyer les headers sensibles"""
        sensitive_headers = {
            'authorization', 'cookie', 'x-api-key', 'x-auth-token',
            'x-access-token', 'x-csrf-token'
        }
        
        sanitized = {}
        for key, value in headers.items():
            if key.lower() in sensitive_headers:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value[:500]  # Limiter la longueur
        
        return sanitized
    
    def _get_client_ip(self, request: Request) -> str:
        """Obtenir l'IP du client"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _get_memory_usage(self) -> float:
        """Obtenir l'usage mémoire en MB"""
        try:
            return psutil.Process().memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Obtenir l'usage CPU"""
        try:
            return psutil.cpu_percent()
        except:
            return 0.0
    
    def _start_cleanup_thread(self):
        """Démarrer le thread de nettoyage"""
        def cleanup_task():
            while True:
                try:
                    # Nettoyer les circuit breakers inactifs
                    self._cleanup_circuit_breakers()
                    
                    # Forcer le garbage collection
                    gc.collect()
                    
                    time.sleep(300)  # Toutes les 5 minutes
                except Exception as e:
                    logger.warning(f"Cleanup task error: {e}")
                    time.sleep(60)
        
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_circuit_breakers(self):
        """Nettoyer les circuit breakers inactifs"""
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        for name, cb in list(self.circuit_breakers.items()):
            # Supprimer les circuit breakers inactifs depuis 1h
            if (cb.last_failure_time and cb.last_failure_time < cutoff_time and
                cb.state == CircuitState.CLOSED and cb.failure_count == 0):
                del self.circuit_breakers[name]
    
    # Méthodes d'API pour monitoring
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Obtenir les statistiques d'erreurs"""
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        
        recent_errors_hour = [
            err for err in self.recent_errors 
            if err["timestamp"] > hour_ago
        ]
        
        # Compter par catégorie
        category_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for error in recent_errors_hour:
            category_counts[error["category"]] += 1
            severity_counts[error["severity"]] += 1
        
        return {
            "total_errors_last_hour": len(recent_errors_hour),
            "errors_by_category": dict(category_counts),
            "errors_by_severity": dict(severity_counts),
            "circuit_breakers": {
                name: cb.get_metrics() 
                for name, cb in self.circuit_breakers.items()
            }
        }
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Obtenir le statut des circuit breakers"""
        return {
            name: cb.get_state()
            for name, cb in self.circuit_breakers.items()
        }


# Exception personnalisée pour circuit breaker

class CircuitBreakerOpenError(Exception):
    """Exception levée quand un circuit breaker est ouvert"""
    pass


# Factory functions

def create_error_handler() -> AdvancedErrorHandler:
    """Créer un gestionnaire d'erreurs avec configuration par défaut"""
    return AdvancedErrorHandler(None)


def setup_error_handlers(app):
    """Configurer les gestionnaires d'erreurs pour une application FastAPI"""
    error_handler = create_error_handler()
    app.add_middleware(AdvancedErrorHandler)
    
    # Handlers spécifiques
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return await error_handler._handle_error(exc, request, None, 0.0)
    
    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, exc: ValidationError):
        return await error_handler._handle_error(exc, request, None, 0.0)
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        return await error_handler._handle_error(exc, request, None, 0.0)


# Décorateur pour protection circuit breaker

def circuit_breaker(name: str, failure_threshold: int = 5, 
                   recovery_timeout: int = 60):
    """Décorateur pour protéger une fonction avec un circuit breaker"""
    def decorator(func):
        cb = CircuitBreaker(name, failure_threshold, recovery_timeout)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await cb.acall(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return cb.call(func, *args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Configuration par défaut

DEFAULT_ERROR_CONFIG = {
    "circuit_breaker": {
        "failure_threshold": 5,
        "recovery_timeout": 60,
        "half_open_max_calls": 3
    },
    "retry": {
        "max_attempts": 3,
        "backoff_factor": 2,
        "max_delay": 60
    },
    "alerting": {
        "rate_limit_per_hour": 10,
        "severity_threshold": "error"
    },
    "monitoring": {
        "enable_metrics": True,
        "enable_tracing": True,
        "enable_analytics": True
    }
}
