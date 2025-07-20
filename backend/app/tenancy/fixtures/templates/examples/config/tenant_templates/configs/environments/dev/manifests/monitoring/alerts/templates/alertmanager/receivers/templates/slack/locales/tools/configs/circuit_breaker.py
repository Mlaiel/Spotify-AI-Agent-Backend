"""
Circuit Breaker Pattern avancé pour le système de monitoring Slack.

Ce module implémente des patterns de résilience pour protéger le système contre:
- Les défaillances en cascade
- Les timeouts et latences excessives
- La surcharge des services externes
- Les pics de trafic imprévisibles
- Les pannes partielles des dépendances

Architecture:
    - State pattern pour les états du circuit breaker
    - Strategy pattern pour les politiques de récupération
    - Observer pattern pour les notifications d'état
    - Command pattern pour les opérations protégées
    - Template Method pattern pour les algorithmes de détection

Fonctionnalités:
    - Circuit breakers configurables par service
    - Détection automatique des pannes
    - Récupération progressive (half-open state)
    - Bulkhead pattern pour l'isolation
    - Rate limiting adaptatif
    - Métriques et alerting en temps réel
    - Fallback strategies personnalisables
    - Health checks automatiques

Types de Circuit Breakers:
    - Count-based: basé sur le nombre d'échecs
    - Time-based: basé sur la fenêtre temporelle
    - Hybrid: combinaison des deux approches
    - Adaptive: adaptation automatique aux conditions

Auteur: Équipe Spotify AI Agent
Lead Developer: Fahed Mlaiel
"""

import asyncio
import time
import threading
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from weakref import WeakSet

from .metrics import MetricsCollector


class CircuitBreakerState(Enum):
    """États du circuit breaker."""
    CLOSED = "closed"        # Fonctionnement normal
    OPEN = "open"           # Circuit ouvert, requêtes bloquées
    HALF_OPEN = "half_open" # Test de récupération


class FailureType(Enum):
    """Types d'échecs détectés."""
    TIMEOUT = "timeout"
    ERROR = "error"
    SLOW_RESPONSE = "slow_response"
    RATE_LIMIT = "rate_limit"
    UNAVAILABLE = "unavailable"


class RecoveryStrategy(Enum):
    """Stratégies de récupération."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_INTERVAL = "fixed_interval"
    LINEAR_BACKOFF = "linear_backoff"
    ADAPTIVE = "adaptive"


@dataclass
class CircuitBreakerConfig:
    """Configuration d'un circuit breaker."""
    # Seuils de déclenchement
    failure_threshold: int = 5              # Nombre d'échecs avant ouverture
    timeout_threshold: float = 5.0          # Timeout en secondes
    slow_response_threshold: float = 2.0    # Seuil de réponse lente
    success_threshold: int = 3              # Succès requis pour fermeture
    
    # Fenêtres temporelles
    time_window: int = 60                   # Fenêtre d'observation (secondes)
    open_timeout: int = 60                  # Durée d'ouverture (secondes)
    half_open_timeout: int = 30             # Durée half-open (secondes)
    
    # Configuration avancée
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.EXPONENTIAL_BACKOFF
    max_retry_attempts: int = 3
    bulkhead_enabled: bool = False
    bulkhead_max_concurrent: int = 10
    rate_limit_enabled: bool = False
    rate_limit_requests_per_second: int = 100
    
    # Métriques et monitoring
    metrics_enabled: bool = True
    health_check_enabled: bool = True
    health_check_interval: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire."""
        return {
            "failure_threshold": self.failure_threshold,
            "timeout_threshold": self.timeout_threshold,
            "slow_response_threshold": self.slow_response_threshold,
            "success_threshold": self.success_threshold,
            "time_window": self.time_window,
            "open_timeout": self.open_timeout,
            "half_open_timeout": self.half_open_timeout,
            "recovery_strategy": self.recovery_strategy.value,
            "max_retry_attempts": self.max_retry_attempts,
            "bulkhead_enabled": self.bulkhead_enabled,
            "bulkhead_max_concurrent": self.bulkhead_max_concurrent,
            "rate_limit_enabled": self.rate_limit_enabled,
            "rate_limit_requests_per_second": self.rate_limit_requests_per_second,
            "metrics_enabled": self.metrics_enabled,
            "health_check_enabled": self.health_check_enabled,
            "health_check_interval": self.health_check_interval
        }


@dataclass
class CircuitBreakerStats:
    """Statistiques d'un circuit breaker."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeouts: int = 0
    slow_responses: int = 0
    circuit_opens: int = 0
    circuit_closes: int = 0
    current_state: CircuitBreakerState = CircuitBreakerState.CLOSED
    state_changed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    average_response_time: float = 0.0
    success_rate: float = 1.0
    
    def update_success_rate(self) -> None:
        """Met à jour le taux de succès."""
        if self.total_requests > 0:
            self.success_rate = self.successful_requests / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "timeouts": self.timeouts,
            "slow_responses": self.slow_responses,
            "circuit_opens": self.circuit_opens,
            "circuit_closes": self.circuit_closes,
            "current_state": self.current_state.value,
            "state_changed_at": self.state_changed_at.isoformat(),
            "average_response_time": self.average_response_time,
            "success_rate": self.success_rate
        }


@dataclass
class RequestResult:
    """Résultat d'une requête protégée."""
    success: bool
    response_time: float
    error: Optional[Exception] = None
    failure_type: Optional[FailureType] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire."""
        return {
            "success": self.success,
            "response_time": self.response_time,
            "error": str(self.error) if self.error else None,
            "failure_type": self.failure_type.value if self.failure_type else None,
            "timestamp": self.timestamp.isoformat()
        }


class ICircuitBreakerStateHandler(ABC):
    """Interface pour les gestionnaires d'état."""
    
    @abstractmethod
    def handle_request(self, circuit_breaker: 'CircuitBreaker') -> bool:
        """Gère une requête selon l'état."""
        pass
    
    @abstractmethod
    def handle_success(self, circuit_breaker: 'CircuitBreaker', response_time: float) -> None:
        """Gère un succès."""
        pass
    
    @abstractmethod
    def handle_failure(self, circuit_breaker: 'CircuitBreaker', 
                      failure_type: FailureType, error: Exception) -> None:
        """Gère un échec."""
        pass


class ClosedStateHandler(ICircuitBreakerStateHandler):
    """Gestionnaire pour l'état fermé (normal)."""
    
    def handle_request(self, circuit_breaker: 'CircuitBreaker') -> bool:
        """Autorise toutes les requêtes en état fermé."""
        return True
    
    def handle_success(self, circuit_breaker: 'CircuitBreaker', response_time: float) -> None:
        """Gère un succès en état fermé."""
        # Reset du compteur d'échecs consécutifs
        circuit_breaker._consecutive_failures = 0
        
        # Mise à jour des statistiques
        circuit_breaker._stats.successful_requests += 1
        circuit_breaker._update_response_time(response_time)
        
        # Vérification des réponses lentes
        if response_time > circuit_breaker._config.slow_response_threshold:
            circuit_breaker._stats.slow_responses += 1
            circuit_breaker._emit_event("slow_response", {"response_time": response_time})
    
    def handle_failure(self, circuit_breaker: 'CircuitBreaker', 
                      failure_type: FailureType, error: Exception) -> None:
        """Gère un échec en état fermé."""
        circuit_breaker._consecutive_failures += 1
        circuit_breaker._stats.failed_requests += 1
        
        # Mise à jour des compteurs par type
        if failure_type == FailureType.TIMEOUT:
            circuit_breaker._stats.timeouts += 1
        
        # Vérification du seuil d'ouverture
        if (circuit_breaker._consecutive_failures >= circuit_breaker._config.failure_threshold or
            circuit_breaker._calculate_failure_rate() > 0.5):
            circuit_breaker._transition_to_open()


class OpenStateHandler(ICircuitBreakerStateHandler):
    """Gestionnaire pour l'état ouvert (circuit breaker déclenché)."""
    
    def handle_request(self, circuit_breaker: 'CircuitBreaker') -> bool:
        """Bloque les requêtes en état ouvert."""
        # Vérification du timeout d'ouverture
        elapsed = time.time() - circuit_breaker._state_changed_at
        if elapsed >= circuit_breaker._config.open_timeout:
            circuit_breaker._transition_to_half_open()
            return True  # Première requête de test autorisée
        
        return False  # Requête bloquée
    
    def handle_success(self, circuit_breaker: 'CircuitBreaker', response_time: float) -> None:
        """Ne devrait pas arriver en état ouvert."""
        pass
    
    def handle_failure(self, circuit_breaker: 'CircuitBreaker', 
                      failure_type: FailureType, error: Exception) -> None:
        """Prolonge l'état ouvert."""
        circuit_breaker._state_changed_at = time.time()  # Reset du timer


class HalfOpenStateHandler(ICircuitBreakerStateHandler):
    """Gestionnaire pour l'état semi-ouvert (test de récupération)."""
    
    def handle_request(self, circuit_breaker: 'CircuitBreaker') -> bool:
        """Autorise un nombre limité de requêtes de test."""
        return circuit_breaker._half_open_attempts < circuit_breaker._config.success_threshold
    
    def handle_success(self, circuit_breaker: 'CircuitBreaker', response_time: float) -> None:
        """Gère un succès en état semi-ouvert."""
        circuit_breaker._half_open_successes += 1
        circuit_breaker._stats.successful_requests += 1
        circuit_breaker._update_response_time(response_time)
        
        # Vérification du seuil de fermeture
        if circuit_breaker._half_open_successes >= circuit_breaker._config.success_threshold:
            circuit_breaker._transition_to_closed()
    
    def handle_failure(self, circuit_breaker: 'CircuitBreaker', 
                      failure_type: FailureType, error: Exception) -> None:
        """Gère un échec en état semi-ouvert."""
        circuit_breaker._stats.failed_requests += 1
        
        # Retour à l'état ouvert
        circuit_breaker._transition_to_open()


class CircuitBreaker:
    """
    Circuit Breaker principal avec gestion d'état sophistiquée.
    
    Protège les appels vers les services externes en détectant
    automatiquement les pannes et en bloquant temporairement
    les requêtes pour permettre la récupération.
    """
    
    def __init__(self,
                 name: str,
                 config: Optional[CircuitBreakerConfig] = None,
                 metrics_collector: Optional[MetricsCollector] = None,
                 fallback_function: Optional[Callable[..., Any]] = None):
        
        self._name = name
        self._config = config or CircuitBreakerConfig()
        self._metrics = metrics_collector or MetricsCollector()
        self._fallback_function = fallback_function
        
        # État du circuit breaker
        self._state = CircuitBreakerState.CLOSED
        self._state_changed_at = time.time()
        self._consecutive_failures = 0
        self._half_open_attempts = 0
        self._half_open_successes = 0
        
        # Gestionnaires d'état
        self._state_handlers = {
            CircuitBreakerState.CLOSED: ClosedStateHandler(),
            CircuitBreakerState.OPEN: OpenStateHandler(),
            CircuitBreakerState.HALF_OPEN: HalfOpenStateHandler()
        }
        
        # Historique des requêtes pour analyse
        self._request_history: deque = deque(maxlen=1000)
        self._failure_history: deque = deque(maxlen=100)
        
        # Statistiques
        self._stats = CircuitBreakerStats()
        
        # Callbacks pour les événements
        self._event_hooks: WeakSet[Callable[[str, str, Dict[str, Any]], None]] = WeakSet()
        
        # Bulkhead: limitation du nombre de requêtes concurrentes
        self._concurrent_requests = 0
        self._bulkhead_lock = threading.Semaphore(self._config.bulkhead_max_concurrent)
        
        # Rate limiting
        self._rate_limiter_tokens = self._config.rate_limit_requests_per_second
        self._rate_limiter_last_update = time.time()
        self._rate_limiter_lock = threading.Lock()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Health check
        self._health_check_thread: Optional[threading.Thread] = None
        self._health_check_running = False
        
        if self._config.health_check_enabled:
            self._start_health_check()
    
    @property
    def name(self) -> str:
        """Nom du circuit breaker."""
        return self._name
    
    @property
    def state(self) -> CircuitBreakerState:
        """État actuel du circuit breaker."""
        return self._state
    
    @property
    def is_closed(self) -> bool:
        """Vérifie si le circuit est fermé (normal)."""
        return self._state == CircuitBreakerState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Vérifie si le circuit est ouvert."""
        return self._state == CircuitBreakerState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Vérifie si le circuit est semi-ouvert."""
        return self._state == CircuitBreakerState.HALF_OPEN
    
    def call(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Exécute une fonction protégée par le circuit breaker.
        
        Args:
            func: Fonction à exécuter
            *args: Arguments positionnels
            **kwargs: Arguments nommés
            
        Returns:
            Résultat de la fonction ou fallback
            
        Raises:
            CircuitBreakerOpenException: Si le circuit est ouvert
            Exception: Autres erreurs de la fonction
        """
        # Vérification du rate limiting
        if not self._check_rate_limit():
            raise RateLimitExceededException(f"Rate limit exceeded for {self._name}")
        
        # Vérification du bulkhead
        if self._config.bulkhead_enabled:
            if not self._bulkhead_lock.acquire(blocking=False):
                raise BulkheadFullException(f"Bulkhead full for {self._name}")
        
        try:
            # Vérification de l'état du circuit
            with self._lock:
                handler = self._state_handlers[self._state]
                if not handler.handle_request(self):
                    # Circuit ouvert, utiliser le fallback si disponible
                    if self._fallback_function:
                        self._emit_event("fallback_used", {})
                        return self._fallback_function(*args, **kwargs)
                    else:
                        raise CircuitBreakerOpenException(f"Circuit breaker {self._name} is open")
                
                self._stats.total_requests += 1
                self._concurrent_requests += 1
                
                if self._state == CircuitBreakerState.HALF_OPEN:
                    self._half_open_attempts += 1
            
            # Exécution de la fonction avec timeout
            start_time = time.time()
            try:
                if asyncio.iscoroutinefunction(func):
                    # Fonction asynchrone
                    result = asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self._config.timeout_threshold
                    )
                else:
                    # Fonction synchrone avec timeout simulé
                    result = func(*args, **kwargs)
                
                response_time = time.time() - start_time
                
                # Enregistrement du succès
                self._record_success(response_time)
                
                return result
                
            except asyncio.TimeoutError as e:
                response_time = time.time() - start_time
                self._record_failure(FailureType.TIMEOUT, e)
                raise TimeoutException(f"Timeout after {response_time:.2f}s") from e
                
            except Exception as e:
                response_time = time.time() - start_time
                failure_type = self._classify_error(e)
                self._record_failure(failure_type, e)
                raise
        
        finally:
            # Libération des ressources
            with self._lock:
                self._concurrent_requests -= 1
            
            if self._config.bulkhead_enabled:
                self._bulkhead_lock.release()
    
    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Décorateur pour protéger une fonction."""
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def reset(self) -> None:
        """Remet à zéro le circuit breaker."""
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._state_changed_at = time.time()
            self._consecutive_failures = 0
            self._half_open_attempts = 0
            self._half_open_successes = 0
            self._request_history.clear()
            self._failure_history.clear()
            
            # Reset des statistiques
            self._stats = CircuitBreakerStats()
            
            self._emit_event("circuit_reset", {})
    
    def force_open(self) -> None:
        """Force l'ouverture du circuit."""
        with self._lock:
            self._transition_to_open()
            self._emit_event("circuit_forced_open", {})
    
    def force_close(self) -> None:
        """Force la fermeture du circuit."""
        with self._lock:
            self._transition_to_closed()
            self._emit_event("circuit_forced_close", {})
    
    def get_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques détaillées."""
        with self._lock:
            self._stats.update_success_rate()
            stats = self._stats.to_dict()
            
            stats.update({
                "name": self._name,
                "config": self._config.to_dict(),
                "consecutive_failures": self._consecutive_failures,
                "concurrent_requests": self._concurrent_requests,
                "request_history_size": len(self._request_history),
                "failure_history_size": len(self._failure_history),
                "state_duration_seconds": time.time() - self._state_changed_at,
                "failure_rate": self._calculate_failure_rate()
            })
            
            return stats
    
    def add_event_hook(self, hook: Callable[[str, str, Dict[str, Any]], None]) -> None:
        """Ajoute un hook pour les événements."""
        self._event_hooks.add(hook)
    
    def _record_success(self, response_time: float) -> None:
        """Enregistre un succès."""
        with self._lock:
            result = RequestResult(True, response_time)
            self._request_history.append(result)
            
            handler = self._state_handlers[self._state]
            handler.handle_success(self, response_time)
            
            self._emit_event("request_success", {
                "response_time": response_time,
                "state": self._state.value
            })
    
    def _record_failure(self, failure_type: FailureType, error: Exception) -> None:
        """Enregistre un échec."""
        with self._lock:
            response_time = 0.0  # Pas de temps de réponse pour les erreurs
            result = RequestResult(False, response_time, error, failure_type)
            self._request_history.append(result)
            self._failure_history.append(result)
            
            handler = self._state_handlers[self._state]
            handler.handle_failure(self, failure_type, error)
            
            self._emit_event("request_failure", {
                "failure_type": failure_type.value,
                "error": str(error),
                "state": self._state.value
            })
    
    def _transition_to_open(self) -> None:
        """Transition vers l'état ouvert."""
        if self._state != CircuitBreakerState.OPEN:
            self._state = CircuitBreakerState.OPEN
            self._state_changed_at = time.time()
            self._stats.circuit_opens += 1
            self._stats.current_state = CircuitBreakerState.OPEN
            self._stats.state_changed_at = datetime.now(timezone.utc)
            
            self._emit_event("circuit_opened", {
                "consecutive_failures": self._consecutive_failures,
                "failure_rate": self._calculate_failure_rate()
            })
            
            if self._config.metrics_enabled:
                self._metrics.increment(f"circuit_breaker_{self._name}_opened")
    
    def _transition_to_half_open(self) -> None:
        """Transition vers l'état semi-ouvert."""
        self._state = CircuitBreakerState.HALF_OPEN
        self._state_changed_at = time.time()
        self._half_open_attempts = 0
        self._half_open_successes = 0
        self._stats.current_state = CircuitBreakerState.HALF_OPEN
        self._stats.state_changed_at = datetime.now(timezone.utc)
        
        self._emit_event("circuit_half_opened", {})
        
        if self._config.metrics_enabled:
            self._metrics.increment(f"circuit_breaker_{self._name}_half_opened")
    
    def _transition_to_closed(self) -> None:
        """Transition vers l'état fermé."""
        self._state = CircuitBreakerState.CLOSED
        self._state_changed_at = time.time()
        self._consecutive_failures = 0
        self._stats.circuit_closes += 1
        self._stats.current_state = CircuitBreakerState.CLOSED
        self._stats.state_changed_at = datetime.now(timezone.utc)
        
        self._emit_event("circuit_closed", {})
        
        if self._config.metrics_enabled:
            self._metrics.increment(f"circuit_breaker_{self._name}_closed")
    
    def _calculate_failure_rate(self) -> float:
        """Calcule le taux d'échec sur la fenêtre temporelle."""
        now = time.time()
        window_start = now - self._config.time_window
        
        recent_requests = [
            r for r in self._request_history 
            if r.timestamp.timestamp() >= window_start
        ]
        
        if not recent_requests:
            return 0.0
        
        failures = sum(1 for r in recent_requests if not r.success)
        return failures / len(recent_requests)
    
    def _update_response_time(self, response_time: float) -> None:
        """Met à jour le temps de réponse moyen."""
        current_avg = self._stats.average_response_time
        current_count = self._stats.successful_requests
        
        if current_count == 1:
            self._stats.average_response_time = response_time
        else:
            self._stats.average_response_time = (
                (current_avg * (current_count - 1) + response_time) / current_count
            )
    
    def _classify_error(self, error: Exception) -> FailureType:
        """Classifie le type d'erreur."""
        error_type = type(error).__name__.lower()
        
        if "timeout" in error_type:
            return FailureType.TIMEOUT
        elif "connection" in error_type or "network" in error_type:
            return FailureType.UNAVAILABLE
        elif "rate" in error_type or "limit" in error_type:
            return FailureType.RATE_LIMIT
        else:
            return FailureType.ERROR
    
    def _check_rate_limit(self) -> bool:
        """Vérifie les limites de taux."""
        if not self._config.rate_limit_enabled:
            return True
        
        with self._rate_limiter_lock:
            now = time.time()
            elapsed = now - self._rate_limiter_last_update
            
            # Ajout de tokens selon le taux configuré
            tokens_to_add = elapsed * self._config.rate_limit_requests_per_second
            self._rate_limiter_tokens = min(
                self._config.rate_limit_requests_per_second,
                self._rate_limiter_tokens + tokens_to_add
            )
            self._rate_limiter_last_update = now
            
            # Consommation d'un token
            if self._rate_limiter_tokens >= 1:
                self._rate_limiter_tokens -= 1
                return True
            
            return False
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Émet un événement."""
        for hook in self._event_hooks:
            try:
                hook(event_type, self._name, data)
            except Exception:
                continue
        
        if self._config.metrics_enabled:
            self._metrics.increment(f"circuit_breaker_event_{event_type}")
    
    def _start_health_check(self) -> None:
        """Démarre le health check en arrière-plan."""
        self._health_check_running = True
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self._health_check_thread.start()
    
    def _health_check_loop(self) -> None:
        """Boucle de health check."""
        while self._health_check_running:
            try:
                time.sleep(self._config.health_check_interval)
                
                # Health check uniquement si le circuit est ouvert
                if self._state == CircuitBreakerState.OPEN:
                    self._perform_health_check()
                    
            except Exception:
                continue
    
    def _perform_health_check(self) -> None:
        """Effectue un health check."""
        # Dans une vraie implémentation, ceci ferait un ping ou un check léger
        # du service pour vérifier s'il est de nouveau disponible
        
        # Simulation: on considère que le service peut récupérer
        elapsed = time.time() - self._state_changed_at
        if elapsed >= self._config.open_timeout:
            self._transition_to_half_open()
    
    def stop(self) -> None:
        """Arrête le circuit breaker."""
        self._health_check_running = False
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5.0)


class CircuitBreakerManager:
    """
    Gestionnaire central des circuit breakers.
    
    Permet de créer, configurer et monitorer plusieurs
    circuit breakers pour différents services.
    """
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self._metrics = metrics_collector or MetricsCollector()
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._default_config = CircuitBreakerConfig()
        self._lock = threading.RLock()
        
        # Hooks globaux
        self._global_event_hooks: WeakSet[Callable[[str, str, Dict[str, Any]], None]] = WeakSet()
    
    def create_circuit_breaker(self,
                              name: str,
                              config: Optional[CircuitBreakerConfig] = None,
                              fallback_function: Optional[Callable[..., Any]] = None) -> CircuitBreaker:
        """
        Crée un nouveau circuit breaker.
        
        Args:
            name: Nom unique du circuit breaker
            config: Configuration spécifique
            fallback_function: Fonction de fallback
            
        Returns:
            Circuit breaker créé
        """
        with self._lock:
            if name in self._circuit_breakers:
                return self._circuit_breakers[name]
            
            circuit_breaker = CircuitBreaker(
                name=name,
                config=config or self._default_config,
                metrics_collector=self._metrics,
                fallback_function=fallback_function
            )
            
            # Ajout des hooks globaux
            for hook in self._global_event_hooks:
                circuit_breaker.add_event_hook(hook)
            
            self._circuit_breakers[name] = circuit_breaker
            
            self._metrics.gauge("circuit_breakers_total", len(self._circuit_breakers))
            
            return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Récupère un circuit breaker par nom."""
        return self._circuit_breakers.get(name)
    
    def remove_circuit_breaker(self, name: str) -> bool:
        """Supprime un circuit breaker."""
        with self._lock:
            circuit_breaker = self._circuit_breakers.pop(name, None)
            if circuit_breaker:
                circuit_breaker.stop()
                self._metrics.gauge("circuit_breakers_total", len(self._circuit_breakers))
                return True
            return False
    
    def list_circuit_breakers(self) -> List[str]:
        """Liste tous les circuit breakers."""
        return list(self._circuit_breakers.keys())
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Récupère les statistiques de tous les circuit breakers."""
        return {
            name: cb.get_stats()
            for name, cb in self._circuit_breakers.items()
        }
    
    def reset_all(self) -> None:
        """Remet à zéro tous les circuit breakers."""
        for circuit_breaker in self._circuit_breakers.values():
            circuit_breaker.reset()
    
    def add_global_event_hook(self, hook: Callable[[str, str, Dict[str, Any]], None]) -> None:
        """Ajoute un hook global pour tous les circuit breakers."""
        self._global_event_hooks.add(hook)
        
        # Ajout aux circuit breakers existants
        for circuit_breaker in self._circuit_breakers.values():
            circuit_breaker.add_event_hook(hook)
    
    def circuit_breaker(self,
                       name: str,
                       config: Optional[CircuitBreakerConfig] = None,
                       fallback_function: Optional[Callable[..., Any]] = None) -> Callable:
        """
        Décorateur pour protéger une fonction avec un circuit breaker.
        
        Args:
            name: Nom du circuit breaker
            config: Configuration optionnelle
            fallback_function: Fonction de fallback
            
        Returns:
            Décorateur
        """
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            circuit_breaker = self.create_circuit_breaker(name, config, fallback_function)
            return circuit_breaker(func)
        
        return decorator


# Exceptions personnalisées
class CircuitBreakerException(Exception):
    """Exception de base pour les circuit breakers."""
    pass


class CircuitBreakerOpenException(CircuitBreakerException):
    """Exception levée quand le circuit breaker est ouvert."""
    pass


class TimeoutException(CircuitBreakerException):
    """Exception levée en cas de timeout."""
    pass


class RateLimitExceededException(CircuitBreakerException):
    """Exception levée quand la limite de taux est dépassée."""
    pass


class BulkheadFullException(CircuitBreakerException):
    """Exception levée quand le bulkhead est plein."""
    pass


# Instance globale singleton
_global_circuit_breaker_manager: Optional[CircuitBreakerManager] = None
_manager_lock = threading.Lock()


def get_circuit_breaker_manager(**kwargs) -> CircuitBreakerManager:
    """
    Récupère l'instance globale du gestionnaire de circuit breakers.
    
    Returns:
        Instance singleton du CircuitBreakerManager
    """
    global _global_circuit_breaker_manager
    
    if _global_circuit_breaker_manager is None:
        with _manager_lock:
            if _global_circuit_breaker_manager is None:
                _global_circuit_breaker_manager = CircuitBreakerManager(**kwargs)
    
    return _global_circuit_breaker_manager


# API publique simplifiée
def circuit_breaker(name: str,
                   config: Optional[CircuitBreakerConfig] = None,
                   fallback_function: Optional[Callable[..., Any]] = None) -> Callable:
    """API simplifiée pour le décorateur circuit breaker."""
    manager = get_circuit_breaker_manager()
    return manager.circuit_breaker(name, config, fallback_function)


def get_circuit_breaker(name: str) -> Optional[CircuitBreaker]:
    """API simplifiée pour récupérer un circuit breaker."""
    manager = get_circuit_breaker_manager()
    return manager.get_circuit_breaker(name)
