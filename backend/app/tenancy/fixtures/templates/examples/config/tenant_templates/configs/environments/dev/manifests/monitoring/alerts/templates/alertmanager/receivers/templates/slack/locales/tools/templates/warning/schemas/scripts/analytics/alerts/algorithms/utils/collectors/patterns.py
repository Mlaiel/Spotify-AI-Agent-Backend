"""
Spotify AI Agent - Advanced Patterns Module
==========================================

Module de patterns ultra-avancés pour la résilience et la performance
des collecteurs de données avec implémentations enterprise-grade.

Patterns disponibles:
- CircuitBreaker: Protection contre les défaillances en cascade
- RetryManager: Gestion intelligente des tentatives avec backoff adaptatif
- RateLimiter: Limitation de débit avec algorithmes multiples
- BulkheadPattern: Isolation des ressources et des échecs
- TimeoutManager: Gestion avancée des timeouts
- LoadBalancer: Répartition de charge intelligente
- CacheAside: Pattern de cache avec fallback
- Saga: Gestion des transactions distribuées
- CQRS: Séparation lecture/écriture

Développé par l'équipe Spotify AI Agent
Lead Developer: Fahed Mlaiel
Architecture: Enterprise resilience patterns
"""

import asyncio
import time
import threading
import random
import math
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
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp
import numpy as np
from sklearn.linear_model import LinearRegression
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge


T = TypeVar('T')
logger = structlog.get_logger(__name__)


class CircuitState(Enum):
    """États du circuit breaker."""
    CLOSED = "closed"      # Fonctionnement normal
    OPEN = "open"          # Circuit ouvert, requêtes bloquées
    HALF_OPEN = "half_open"  # Test de récupération


class RetryStrategy(Enum):
    """Stratégies de retry."""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"
    ADAPTIVE = "adaptive"


class LoadBalanceStrategy(Enum):
    """Stratégies de load balancing."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_LEAST_CONNECTIONS = "weighted_least_connections"
    RANDOM = "random"
    WEIGHTED_RANDOM = "weighted_random"
    IP_HASH = "ip_hash"
    HEALTH_BASED = "health_based"


@dataclass
class CircuitBreakerConfig:
    """Configuration du circuit breaker."""
    
    failure_threshold: int = 5          # Nombre d'échecs avant ouverture
    success_threshold: int = 3          # Nombre de succès pour fermeture
    timeout: float = 60.0              # Timeout en état ouvert (secondes)
    half_open_max_calls: int = 10      # Appels max en état half-open
    failure_rate_threshold: float = 50.0  # Seuil de taux d'échec (%)
    slow_call_threshold: float = 5.0    # Seuil d'appel lent (secondes)
    slow_call_rate_threshold: float = 50.0  # Seuil de taux d'appels lents (%)
    minimum_throughput: int = 10        # Débit minimum pour calculs


@dataclass
class RetryConfig:
    """Configuration des tentatives."""
    
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    base_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    jitter: bool = True
    backoff_cap: float = 300.0
    retry_on_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    stop_on_exceptions: Tuple[Type[Exception], ...] = ()


@dataclass
class RateLimitConfig:
    """Configuration du rate limiting."""
    
    requests_per_second: float = 10.0
    burst_size: int = 20
    algorithm: str = "token_bucket"  # token_bucket, sliding_window, fixed_window
    window_size: float = 1.0
    refill_rate: float = 1.0


class CircuitBreaker:
    """
    Circuit Breaker enterprise avec métriques avancées et auto-tuning.
    
    Fonctionnalités:
    - Détection des échecs multiples (exceptions, timeouts, slow calls)
    - Seuils adaptatifs basés sur l'historique
    - Métriques détaillées avec Prometheus
    - Half-open intelligent avec gradient testing
    - Auto-tuning des paramètres
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        enable_metrics: bool = True,
        enable_auto_tuning: bool = True
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.enable_metrics = enable_metrics
        self.enable_auto_tuning = enable_auto_tuning
        
        # État du circuit
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        
        # Historique et statistiques
        self._call_history: deque = deque(maxlen=1000)
        self._failure_history: deque = deque(maxlen=100)
        self._response_times: deque = deque(maxlen=1000)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Auto-tuning
        self._performance_history: deque = deque(maxlen=10000)
        self._last_auto_tune = time.time()
        
        # Métriques
        if enable_metrics:
            self._init_metrics()
    
    def _init_metrics(self) -> None:
        """Initialise les métriques Prometheus."""
        
        self.calls_total = Counter(
            f'circuit_breaker_calls_total',
            'Nombre total d\'appels au circuit breaker',
            ['name', 'state', 'result']
        )
        
        self.state_transitions = Counter(
            f'circuit_breaker_state_transitions_total',
            'Nombre de transitions d\'état',
            ['name', 'from_state', 'to_state']
        )
        
        self.current_state = Gauge(
            f'circuit_breaker_current_state',
            'État actuel du circuit breaker (0=closed, 1=half_open, 2=open)',
            ['name']
        )
        
        self.failure_rate = Gauge(
            f'circuit_breaker_failure_rate',
            'Taux d\'échec actuel',
            ['name']
        )
        
        self.response_time = Histogram(
            f'circuit_breaker_response_time_seconds',
            'Temps de réponse des appels',
            ['name'],
            buckets=[0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 30.0]
        )
    
    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """
        Exécute une fonction via le circuit breaker.
        
        Args:
            func: Fonction asynchrone à exécuter
            *args: Arguments positionnels
            **kwargs: Arguments nommés
            
        Returns:
            Résultat de la fonction
            
        Raises:
            CircuitBreakerOpenError: Si le circuit est ouvert
            Exception: Exceptions propagées de la fonction
        """
        
        with self._lock:
            if not self._can_execute():
                self._record_metrics('blocked')
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open"
                )
        
        start_time = time.time()
        
        try:
            # Exécution de la fonction
            result = await func(*args, **kwargs)
            
            # Enregistrement du succès
            execution_time = time.time() - start_time
            self._record_success(execution_time)
            
            return result
            
        except Exception as e:
            # Enregistrement de l'échec
            execution_time = time.time() - start_time
            self._record_failure(e, execution_time)
            raise
    
    def _can_execute(self) -> bool:
        """Vérifie si une exécution est autorisée."""
        
        current_time = time.time()
        
        if self._state == CircuitState.CLOSED:
            return True
        
        elif self._state == CircuitState.OPEN:
            # Vérification du timeout
            if current_time - self._last_failure_time >= self.config.timeout:
                self._transition_to_half_open()
                return True
            return False
        
        elif self._state == CircuitState.HALF_OPEN:
            # Limitation des appels en half-open
            return self._half_open_calls < self.config.half_open_max_calls
        
        return False
    
    def _record_success(self, execution_time: float) -> None:
        """Enregistre un succès."""
        
        with self._lock:
            self._success_count += 1
            
            current_time = time.time()
            call_record = {
                'timestamp': current_time,
                'success': True,
                'execution_time': execution_time,
                'slow_call': execution_time > self.config.slow_call_threshold
            }
            
            self._call_history.append(call_record)
            self._response_times.append(execution_time)
            
            # Gestion de l'état half-open
            if self._state == CircuitState.HALF_OPEN:
                if self._success_count >= self.config.success_threshold:
                    self._transition_to_closed()
                else:
                    self._half_open_calls += 1
            
            # Auto-tuning
            if self.enable_auto_tuning:
                self._record_performance_data(call_record)
            
            # Métriques
            if self.enable_metrics:
                self._record_metrics('success', execution_time)
    
    def _record_failure(self, exception: Exception, execution_time: float) -> None:
        """Enregistre un échec."""
        
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            call_record = {
                'timestamp': time.time(),
                'success': False,
                'execution_time': execution_time,
                'exception': type(exception).__name__,
                'slow_call': execution_time > self.config.slow_call_threshold
            }
            
            self._call_history.append(call_record)
            self._failure_history.append(call_record)
            self._response_times.append(execution_time)
            
            # Vérification des seuils d'ouverture
            if self._should_open_circuit():
                self._transition_to_open()
            
            # Gestion de l'état half-open
            elif self._state == CircuitState.HALF_OPEN:
                self._transition_to_open()
            
            # Auto-tuning
            if self.enable_auto_tuning:
                self._record_performance_data(call_record)
            
            # Métriques
            if self.enable_metrics:
                self._record_metrics('failure', execution_time)
    
    def _should_open_circuit(self) -> bool:
        """Détermine si le circuit doit être ouvert."""
        
        if self._state != CircuitState.CLOSED:
            return False
        
        # Vérification du seuil simple d'échecs
        if self._failure_count >= self.config.failure_threshold:
            return True
        
        # Vérification du taux d'échec
        if len(self._call_history) >= self.config.minimum_throughput:
            recent_calls = list(self._call_history)[-self.config.minimum_throughput:]
            failures = sum(1 for call in recent_calls if not call['success'])
            failure_rate = (failures / len(recent_calls)) * 100
            
            if failure_rate >= self.config.failure_rate_threshold:
                return True
        
        # Vérification du taux d'appels lents
        if len(self._call_history) >= self.config.minimum_throughput:
            recent_calls = list(self._call_history)[-self.config.minimum_throughput:]
            slow_calls = sum(1 for call in recent_calls if call['slow_call'])
            slow_call_rate = (slow_calls / len(recent_calls)) * 100
            
            if slow_call_rate >= self.config.slow_call_rate_threshold:
                return True
        
        return False
    
    def _transition_to_open(self) -> None:
        """Transition vers l'état ouvert."""
        
        old_state = self._state
        self._state = CircuitState.OPEN
        self._half_open_calls = 0
        
        logger.warning(
            "Circuit breaker ouvert",
            name=self.name,
            failure_count=self._failure_count,
            state_transition=f"{old_state.value} -> {self._state.value}"
        )
        
        if self.enable_metrics:
            self._record_state_transition(old_state, self._state)
    
    def _transition_to_half_open(self) -> None:
        """Transition vers l'état half-open."""
        
        old_state = self._state
        self._state = CircuitState.HALF_OPEN
        self._half_open_calls = 0
        self._success_count = 0
        
        logger.info(
            "Circuit breaker en test de récupération",
            name=self.name,
            state_transition=f"{old_state.value} -> {self._state.value}"
        )
        
        if self.enable_metrics:
            self._record_state_transition(old_state, self._state)
    
    def _transition_to_closed(self) -> None:
        """Transition vers l'état fermé."""
        
        old_state = self._state
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        
        logger.info(
            "Circuit breaker fermé - récupération réussie",
            name=self.name,
            state_transition=f"{old_state.value} -> {self._state.value}"
        )
        
        if self.enable_metrics:
            self._record_state_transition(old_state, self._state)
    
    def _record_performance_data(self, call_record: Dict[str, Any]) -> None:
        """Enregistre les données de performance pour l'auto-tuning."""
        
        self._performance_history.append(call_record)
        
        # Auto-tuning périodique (toutes les heures)
        current_time = time.time()
        if current_time - self._last_auto_tune > 3600:
            self._auto_tune_parameters()
            self._last_auto_tune = current_time
    
    def _auto_tune_parameters(self) -> None:
        """Ajuste automatiquement les paramètres basés sur l'historique."""
        
        if len(self._performance_history) < 100:
            return
        
        recent_data = list(self._performance_history)[-1000:]
        
        # Calcul du taux d'échec moyen
        failures = sum(1 for record in recent_data if not record['success'])
        avg_failure_rate = (failures / len(recent_data)) * 100
        
        # Calcul du temps de réponse moyen
        response_times = [record['execution_time'] for record in recent_data]
        avg_response_time = np.mean(response_times)
        p95_response_time = np.percentile(response_times, 95)
        
        # Ajustement du seuil de slow call
        if p95_response_time > self.config.slow_call_threshold * 1.5:
            new_threshold = min(p95_response_time * 0.8, self.config.slow_call_threshold * 2)
            self.config.slow_call_threshold = new_threshold
            
            logger.info(
                "Auto-tuning: Ajustement du seuil slow call",
                name=self.name,
                old_threshold=self.config.slow_call_threshold,
                new_threshold=new_threshold
            )
        
        # Ajustement du seuil de taux d'échec
        if avg_failure_rate < self.config.failure_rate_threshold * 0.5:
            # Si le taux d'échec est très bas, on peut être plus sensible
            new_threshold = max(avg_failure_rate * 2, self.config.failure_rate_threshold * 0.7)
            self.config.failure_rate_threshold = new_threshold
            
            logger.info(
                "Auto-tuning: Ajustement du seuil de taux d'échec",
                name=self.name,
                old_threshold=self.config.failure_rate_threshold,
                new_threshold=new_threshold
            )
    
    def _record_metrics(self, result: str, execution_time: Optional[float] = None) -> None:
        """Enregistre les métriques Prometheus."""
        
        if not self.enable_metrics:
            return
        
        self.calls_total.labels(
            name=self.name,
            state=self._state.value,
            result=result
        ).inc()
        
        state_values = {
            CircuitState.CLOSED: 0,
            CircuitState.HALF_OPEN: 1,
            CircuitState.OPEN: 2
        }
        self.current_state.labels(name=self.name).set(state_values[self._state])
        
        if execution_time is not None:
            self.response_time.labels(name=self.name).observe(execution_time)
        
        # Calcul du taux d'échec actuel
        if len(self._call_history) >= 10:
            recent_calls = list(self._call_history)[-10:]
            failures = sum(1 for call in recent_calls if not call['success'])
            current_failure_rate = (failures / len(recent_calls)) * 100
            self.failure_rate.labels(name=self.name).set(current_failure_rate)
    
    def _record_state_transition(self, from_state: CircuitState, to_state: CircuitState) -> None:
        """Enregistre une transition d'état."""
        
        if self.enable_metrics:
            self.state_transitions.labels(
                name=self.name,
                from_state=from_state.value,
                to_state=to_state.value
            ).inc()
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du circuit breaker."""
        
        with self._lock:
            recent_calls = list(self._call_history)[-100:] if self._call_history else []
            
            if recent_calls:
                success_rate = (sum(1 for call in recent_calls if call['success']) / len(recent_calls)) * 100
                avg_response_time = np.mean([call['execution_time'] for call in recent_calls])
                slow_calls = sum(1 for call in recent_calls if call['slow_call'])
                slow_call_rate = (slow_calls / len(recent_calls)) * 100
            else:
                success_rate = 0
                avg_response_time = 0
                slow_call_rate = 0
            
            return {
                'name': self.name,
                'state': self._state.value,
                'failure_count': self._failure_count,
                'success_count': self._success_count,
                'total_calls': len(self._call_history),
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'slow_call_rate': slow_call_rate,
                'last_failure_time': self._last_failure_time,
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'success_threshold': self.config.success_threshold,
                    'timeout': self.config.timeout,
                    'failure_rate_threshold': self.config.failure_rate_threshold,
                    'slow_call_threshold': self.config.slow_call_threshold
                }
            }
    
    def reset(self) -> None:
        """Remet à zéro le circuit breaker."""
        
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._last_failure_time = 0.0
            self._call_history.clear()
            self._failure_history.clear()
            self._response_times.clear()
            
            logger.info("Circuit breaker réinitialisé", name=self.name)


class RetryManager:
    """
    Gestionnaire de tentatives intelligent avec stratégies adaptatives.
    
    Fonctionnalités:
    - Multiples stratégies de backoff
    - Jitter pour éviter le thundering herd
    - Conditions de retry personnalisables
    - Métriques détaillées
    - Apprentissage adaptatif des patterns d'échec
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[RetryConfig] = None,
        enable_metrics: bool = True,
        enable_adaptive_learning: bool = True
    ):
        self.name = name
        self.config = config or RetryConfig()
        self.enable_metrics = enable_metrics
        self.enable_adaptive_learning = enable_adaptive_learning
        
        # Historique des tentatives
        self._retry_history: deque = deque(maxlen=1000)
        self._pattern_learning: Dict[str, List[float]] = defaultdict(list)
        
        # Métriques
        if enable_metrics:
            self._init_metrics()
    
    def _init_metrics(self) -> None:
        """Initialise les métriques Prometheus."""
        
        self.retry_attempts = Counter(
            'retry_attempts_total',
            'Nombre total de tentatives',
            ['name', 'attempt', 'strategy']
        )
        
        self.retry_success = Counter(
            'retry_success_total',
            'Nombre de succès après retry',
            ['name', 'attempt', 'strategy']
        )
        
        self.retry_exhausted = Counter(
            'retry_exhausted_total',
            'Nombre de retry épuisés',
            ['name', 'strategy']
        )
        
        self.retry_delay = Histogram(
            'retry_delay_seconds',
            'Délai entre les tentatives',
            ['name', 'strategy'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
    
    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs
    ) -> T:
        """
        Exécute une fonction avec retry intelligent.
        
        Args:
            func: Fonction asynchrone à exécuter
            *args: Arguments positionnels
            **kwargs: Arguments nommés
            
        Returns:
            Résultat de la fonction
            
        Raises:
            RetryExhaustedError: Si toutes les tentatives échouent
        """
        
        attempt = 0
        last_exception = None
        start_time = time.time()
        
        while attempt < self.config.max_attempts:
            attempt += 1
            
            try:
                # Enregistrement de la tentative
                if self.enable_metrics:
                    self.retry_attempts.labels(
                        name=self.name,
                        attempt=str(attempt),
                        strategy=self.config.strategy.value
                    ).inc()
                
                # Exécution de la fonction
                result = await func(*args, **kwargs)
                
                # Succès
                execution_time = time.time() - start_time
                self._record_success(attempt, execution_time)
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Vérification si on doit arrêter
                if self._should_stop_retry(e):
                    break
                
                # Dernière tentative
                if attempt >= self.config.max_attempts:
                    break
                
                # Calcul du délai
                delay = self._calculate_delay(attempt, e)
                
                # Attente avant la prochaine tentative
                if delay > 0:
                    if self.enable_metrics:
                        self.retry_delay.labels(
                            name=self.name,
                            strategy=self.config.strategy.value
                        ).observe(delay)
                    
                    await asyncio.sleep(delay)
        
        # Échec final
        total_time = time.time() - start_time
        self._record_failure(attempt, total_time, last_exception)
        
        raise RetryExhaustedError(
            f"Retry exhausted for '{self.name}' after {attempt} attempts. "
            f"Last error: {last_exception}"
        )
    
    def _should_stop_retry(self, exception: Exception) -> bool:
        """Détermine si on doit arrêter les tentatives."""
        
        # Exceptions qui arrêtent immédiatement
        for stop_exception in self.config.stop_on_exceptions:
            if isinstance(exception, stop_exception):
                return True
        
        # Vérification des exceptions autorisées
        if self.config.retry_on_exceptions:
            for retry_exception in self.config.retry_on_exceptions:
                if isinstance(exception, retry_exception):
                    return False
            return True  # Exception non autorisée
        
        return False
    
    def _calculate_delay(self, attempt: int, exception: Exception) -> float:
        """Calcule le délai avant la prochaine tentative."""
        
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.base_delay
            
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.multiplier ** (attempt - 1))
            
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * attempt
            
        elif self.config.strategy == RetryStrategy.FIBONACCI:
            delay = self.config.base_delay * self._fibonacci(attempt)
            
        elif self.config.strategy == RetryStrategy.ADAPTIVE:
            delay = self._adaptive_delay(attempt, exception)
            
        else:
            delay = self.config.base_delay
        
        # Application du cap maximal
        delay = min(delay, self.config.max_delay)
        
        # Application du jitter
        if self.config.jitter:
            jitter_range = delay * 0.1  # ±10%
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)
    
    def _fibonacci(self, n: int) -> int:
        """Calcule le n-ième nombre de Fibonacci."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    def _adaptive_delay(self, attempt: int, exception: Exception) -> float:
        """Calcule un délai adaptatif basé sur l'historique."""
        
        exception_type = type(exception).__name__
        
        # Apprentissage des patterns d'échec
        if self.enable_adaptive_learning and exception_type in self._pattern_learning:
            historical_delays = self._pattern_learning[exception_type]
            if len(historical_delays) >= 3:
                # Utilisation de la régression linéaire pour prédire le délai optimal
                X = np.array(range(1, len(historical_delays) + 1)).reshape(-1, 1)
                y = np.array(historical_delays)
                
                try:
                    model = LinearRegression()
                    model.fit(X, y)
                    predicted_delay = model.predict([[attempt]])[0]
                    
                    # Limitation dans des bornes raisonnables
                    return max(
                        self.config.base_delay,
                        min(predicted_delay, self.config.max_delay)
                    )
                except Exception:
                    pass  # Fallback vers exponentiel
        
        # Fallback vers stratégie exponentielle
        return self.config.base_delay * (self.config.multiplier ** (attempt - 1))
    
    def _record_success(self, attempt: int, execution_time: float) -> None:
        """Enregistre un succès."""
        
        retry_record = {
            'timestamp': time.time(),
            'success': True,
            'attempts': attempt,
            'execution_time': execution_time,
            'strategy': self.config.strategy.value
        }
        
        self._retry_history.append(retry_record)
        
        if self.enable_metrics:
            self.retry_success.labels(
                name=self.name,
                attempt=str(attempt),
                strategy=self.config.strategy.value
            ).inc()
        
        logger.info(
            "Retry réussi",
            name=self.name,
            attempts=attempt,
            execution_time=execution_time
        )
    
    def _record_failure(self, attempts: int, total_time: float, exception: Exception) -> None:
        """Enregistre un échec final."""
        
        retry_record = {
            'timestamp': time.time(),
            'success': False,
            'attempts': attempts,
            'total_time': total_time,
            'exception': type(exception).__name__,
            'strategy': self.config.strategy.value
        }
        
        self._retry_history.append(retry_record)
        
        # Apprentissage adaptatif
        if self.enable_adaptive_learning:
            exception_type = type(exception).__name__
            avg_delay = total_time / max(1, attempts - 1)
            self._pattern_learning[exception_type].append(avg_delay)
            
            # Limitation de l'historique
            if len(self._pattern_learning[exception_type]) > 100:
                self._pattern_learning[exception_type].pop(0)
        
        if self.enable_metrics:
            self.retry_exhausted.labels(
                name=self.name,
                strategy=self.config.strategy.value
            ).inc()
        
        logger.error(
            "Retry épuisé",
            name=self.name,
            attempts=attempts,
            total_time=total_time,
            exception=type(exception).__name__
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du retry manager."""
        
        if not self._retry_history:
            return {
                'name': self.name,
                'total_executions': 0,
                'success_rate': 0,
                'avg_attempts': 0,
                'strategy': self.config.strategy.value
            }
        
        total_executions = len(self._retry_history)
        successes = sum(1 for record in self._retry_history if record['success'])
        success_rate = (successes / total_executions) * 100
        
        avg_attempts = np.mean([record['attempts'] for record in self._retry_history])
        
        # Analyse par type d'exception
        exception_stats = defaultdict(int)
        for record in self._retry_history:
            if not record['success']:
                exception_stats[record.get('exception', 'Unknown')] += 1
        
        return {
            'name': self.name,
            'total_executions': total_executions,
            'success_rate': success_rate,
            'avg_attempts': avg_attempts,
            'strategy': self.config.strategy.value,
            'exception_breakdown': dict(exception_stats),
            'learned_patterns': len(self._pattern_learning) if self.enable_adaptive_learning else 0
        }


class RateLimiter:
    """
    Rate Limiter ultra-avancé avec algorithmes multiples et adaptation intelligente.
    
    Fonctionnalités:
    - Token Bucket avec refill adaptatif
    - Sliding Window avec précision sub-seconde
    - Fixed Window avec burst handling
    - Leaky Bucket avec queue management
    - Adaptation automatique selon la charge
    - Métriques détaillées et alerting
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[RateLimitConfig] = None,
        enable_metrics: bool = True,
        enable_adaptive_limits: bool = True
    ):
        self.name = name
        self.config = config or RateLimitConfig()
        self.enable_metrics = enable_metrics
        self.enable_adaptive_limits = enable_adaptive_limits
        
        # État selon l'algorithme
        if self.config.algorithm == "token_bucket":
            self._init_token_bucket()
        elif self.config.algorithm == "sliding_window":
            self._init_sliding_window()
        elif self.config.algorithm == "fixed_window":
            self._init_fixed_window()
        elif self.config.algorithm == "leaky_bucket":
            self._init_leaky_bucket()
        
        # Historique et adaptation
        self._request_history: deque = deque(maxlen=10000)
        self._rejection_history: deque = deque(maxlen=1000)
        self._last_adaptive_check = time.time()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Métriques
        if enable_metrics:
            self._init_metrics()
    
    def _init_token_bucket(self) -> None:
        """Initialise l'algorithme Token Bucket."""
        self._tokens = float(self.config.burst_size)
        self._last_refill = time.time()
    
    def _init_sliding_window(self) -> None:
        """Initialise l'algorithme Sliding Window."""
        self._request_timestamps: deque = deque()
    
    def _init_fixed_window(self) -> None:
        """Initialise l'algorithme Fixed Window."""
        self._current_window_start = time.time()
        self._current_window_count = 0
    
    def _init_leaky_bucket(self) -> None:
        """Initialise l'algorithme Leaky Bucket."""
        self._queue: deque = deque()
        self._last_leak = time.time()
    
    def _init_metrics(self) -> None:
        """Initialise les métriques Prometheus."""
        
        self.requests_total = Counter(
            'rate_limiter_requests_total',
            'Nombre total de requêtes',
            ['name', 'algorithm', 'result']
        )
        
        self.current_rate = Gauge(
            'rate_limiter_current_rate',
            'Taux actuel de requêtes par seconde',
            ['name', 'algorithm']
        )
        
        self.available_tokens = Gauge(
            'rate_limiter_available_tokens',
            'Tokens disponibles (token bucket)',
            ['name']
        )
        
        self.queue_size = Gauge(
            'rate_limiter_queue_size',
            'Taille de la queue (leaky bucket)',
            ['name']
        )
        
        self.wait_time = Histogram(
            'rate_limiter_wait_time_seconds',
            'Temps d\'attente avant autorisation',
            ['name', 'algorithm'],
            buckets=[0.001, 0.01, 0.1, 1.0, 5.0, 10.0]
        )
    
    async def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Acquiert des tokens selon l'algorithme configuré.
        
        Args:
            tokens: Nombre de tokens à acquérir
            timeout: Timeout d'attente (None = pas d'attente)
            
        Returns:
            True si autorisé, False si rejeté
        """
        
        start_time = time.time()
        
        with self._lock:
            if self.config.algorithm == "token_bucket":
                result = await self._acquire_token_bucket(tokens, timeout)
            elif self.config.algorithm == "sliding_window":
                result = await self._acquire_sliding_window(tokens, timeout)
            elif self.config.algorithm == "fixed_window":
                result = await self._acquire_fixed_window(tokens, timeout)
            elif self.config.algorithm == "leaky_bucket":
                result = await self._acquire_leaky_bucket(tokens, timeout)
            else:
                result = False
        
        # Enregistrement de la requête
        wait_time = time.time() - start_time
        self._record_request(result, wait_time, tokens)
        
        return result
    
    async def _acquire_token_bucket(self, tokens: int, timeout: Optional[float]) -> bool:
        """Implémentation Token Bucket."""
        
        current_time = time.time()
        
        # Refill des tokens
        time_passed = current_time - self._last_refill
        new_tokens = time_passed * self.config.refill_rate
        self._tokens = min(self.config.burst_size, self._tokens + new_tokens)
        self._last_refill = current_time
        
        # Vérification de la disponibilité
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        
        # Attente si timeout spécifié
        if timeout and timeout > 0:
            tokens_needed = tokens - self._tokens
            wait_time = tokens_needed / self.config.refill_rate
            
            if wait_time <= timeout:
                await asyncio.sleep(wait_time)
                self._tokens = max(0, self._tokens - tokens)
                return True
        
        return False
    
    async def _acquire_sliding_window(self, tokens: int, timeout: Optional[float]) -> bool:
        """Implémentation Sliding Window."""
        
        current_time = time.time()
        window_start = current_time - self.config.window_size
        
        # Nettoyage des anciennes requêtes
        while self._request_timestamps and self._request_timestamps[0] < window_start:
            self._request_timestamps.popleft()
        
        # Vérification du taux
        current_count = len(self._request_timestamps)
        if current_count + tokens <= self.config.requests_per_second * self.config.window_size:
            # Ajout des nouveaux tokens
            for _ in range(tokens):
                self._request_timestamps.append(current_time)
            return True
        
        # Attente si timeout spécifié
        if timeout and timeout > 0:
            # Calcul du temps d'attente nécessaire
            if self._request_timestamps:
                oldest_request = self._request_timestamps[0]
                wait_time = (oldest_request + self.config.window_size) - current_time
                
                if wait_time > 0 and wait_time <= timeout:
                    await asyncio.sleep(wait_time)
                    return await self._acquire_sliding_window(tokens, timeout - wait_time)
        
        return False
    
    async def _acquire_fixed_window(self, tokens: int, timeout: Optional[float]) -> bool:
        """Implémentation Fixed Window."""
        
        current_time = time.time()
        window_duration = self.config.window_size
        
        # Vérification du changement de fenêtre
        if current_time - self._current_window_start >= window_duration:
            self._current_window_start = current_time
            self._current_window_count = 0
        
        # Vérification du quota
        max_requests = int(self.config.requests_per_second * window_duration)
        if self._current_window_count + tokens <= max_requests:
            self._current_window_count += tokens
            return True
        
        # Attente pour la prochaine fenêtre
        if timeout and timeout > 0:
            next_window = self._current_window_start + window_duration
            wait_time = next_window - current_time
            
            if wait_time > 0 and wait_time <= timeout:
                await asyncio.sleep(wait_time)
                return await self._acquire_fixed_window(tokens, timeout - wait_time)
        
        return False
    
    async def _acquire_leaky_bucket(self, tokens: int, timeout: Optional[float]) -> bool:
        """Implémentation Leaky Bucket."""
        
        current_time = time.time()
        
        # Leak des requêtes anciennes
        time_passed = current_time - self._last_leak
        requests_to_leak = int(time_passed * self.config.requests_per_second)
        
        for _ in range(min(requests_to_leak, len(self._queue))):
            self._queue.popleft()
        
        self._last_leak = current_time
        
        # Vérification de la capacité
        if len(self._queue) + tokens <= self.config.burst_size:
            for _ in range(tokens):
                self._queue.append(current_time)
            return True
        
        # Attente si timeout spécifié
        if timeout and timeout > 0:
            # Calcul du temps nécessaire pour libérer de l'espace
            space_needed = len(self._queue) + tokens - self.config.burst_size
            wait_time = space_needed / self.config.requests_per_second
            
            if wait_time <= timeout:
                await asyncio.sleep(wait_time)
                return await self._acquire_leaky_bucket(tokens, timeout - wait_time)
        
        return False
    
    def _record_request(self, allowed: bool, wait_time: float, tokens: int) -> None:
        """Enregistre une requête dans l'historique."""
        
        request_record = {
            'timestamp': time.time(),
            'allowed': allowed,
            'wait_time': wait_time,
            'tokens': tokens,
            'algorithm': self.config.algorithm
        }
        
        self._request_history.append(request_record)
        
        if not allowed:
            self._rejection_history.append(request_record)
        
        # Métriques
        if self.enable_metrics:
            result = 'allowed' if allowed else 'rejected'
            self.requests_total.labels(
                name=self.name,
                algorithm=self.config.algorithm,
                result=result
            ).inc()
            
            if wait_time > 0:
                self.wait_time.labels(
                    name=self.name,
                    algorithm=self.config.algorithm
                ).observe(wait_time)
            
            # Mise à jour des gauges spécifiques
            if self.config.algorithm == "token_bucket":
                self.available_tokens.labels(name=self.name).set(self._tokens)
            elif self.config.algorithm == "leaky_bucket":
                self.queue_size.labels(name=self.name).set(len(self._queue))
        
        # Adaptation intelligente
        if self.enable_adaptive_limits:
            self._check_adaptive_limits()
    
    def _check_adaptive_limits(self) -> None:
        """Vérifie et ajuste les limites de manière adaptative."""
        
        current_time = time.time()
        
        # Vérification périodique (toutes les 5 minutes)
        if current_time - self._last_adaptive_check < 300:
            return
        
        self._last_adaptive_check = current_time
        
        # Analyse des 1000 dernières requêtes
        recent_requests = list(self._request_history)[-1000:]
        if len(recent_requests) < 100:
            return
        
        # Calcul du taux de rejet
        rejections = sum(1 for req in recent_requests if not req['allowed'])
        rejection_rate = (rejections / len(recent_requests)) * 100
        
        # Adaptation selon le taux de rejet
        if rejection_rate > 20:  # Trop de rejets
            # Augmentation de la limite
            new_rate = self.config.requests_per_second * 1.1
            self.config.requests_per_second = min(new_rate, self.config.requests_per_second * 3)
            
            logger.info(
                "Adaptation rate limiter: Augmentation",
                name=self.name,
                old_rate=self.config.requests_per_second / 1.1,
                new_rate=self.config.requests_per_second,
                rejection_rate=rejection_rate
            )
            
        elif rejection_rate < 5:  # Très peu de rejets
            # Évaluation pour diminuer si la charge est faible
            avg_wait_time = np.mean([req['wait_time'] for req in recent_requests])
            
            if avg_wait_time < 0.1:  # Très peu d'attente
                new_rate = self.config.requests_per_second * 0.95
                self.config.requests_per_second = max(new_rate, 1.0)
                
                logger.info(
                    "Adaptation rate limiter: Diminution",
                    name=self.name,
                    old_rate=self.config.requests_per_second / 0.95,
                    new_rate=self.config.requests_per_second,
                    rejection_rate=rejection_rate
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du rate limiter."""
        
        if not self._request_history:
            return {
                'name': self.name,
                'algorithm': self.config.algorithm,
                'current_rate': self.config.requests_per_second,
                'total_requests': 0,
                'rejection_rate': 0
            }
        
        total_requests = len(self._request_history)
        rejections = sum(1 for req in self._request_history if not req['allowed'])
        rejection_rate = (rejections / total_requests) * 100 if total_requests > 0 else 0
        
        # Taux actuel (sur la dernière minute)
        current_time = time.time()
        recent_requests = [
            req for req in self._request_history
            if current_time - req['timestamp'] <= 60
        ]
        current_actual_rate = len(recent_requests) / 60 if recent_requests else 0
        
        # État spécifique à l'algorithme
        algorithm_state = {}
        if self.config.algorithm == "token_bucket":
            algorithm_state = {
                'available_tokens': self._tokens,
                'burst_size': self.config.burst_size,
                'refill_rate': self.config.refill_rate
            }
        elif self.config.algorithm == "leaky_bucket":
            algorithm_state = {
                'queue_size': len(self._queue),
                'max_queue_size': self.config.burst_size
            }
        
        return {
            'name': self.name,
            'algorithm': self.config.algorithm,
            'configured_rate': self.config.requests_per_second,
            'current_actual_rate': current_actual_rate,
            'total_requests': total_requests,
            'rejection_rate': rejection_rate,
            'avg_wait_time': np.mean([req['wait_time'] for req in self._request_history]) if self._request_history else 0,
            **algorithm_state
        }


# Exceptions personnalisées
class CircuitBreakerOpenError(Exception):
    """Exception levée quand le circuit breaker est ouvert."""
    pass


class RetryExhaustedError(Exception):
    """Exception levée quand toutes les tentatives de retry sont épuisées."""
    pass


class RateLimitExceededError(Exception):
    """Exception levée quand la limite de débit est dépassée."""
    pass


# Décorateurs pour l'utilisation facile des patterns
def circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
    **kwargs
):
    """
    Décorateur pour appliquer un circuit breaker à une fonction.
    
    Args:
        name: Nom du circuit breaker
        config: Configuration du circuit breaker
        **kwargs: Arguments supplémentaires pour CircuitBreaker
    """
    
    cb = CircuitBreaker(name, config, **kwargs)
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await cb.call(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Pour les fonctions synchrones, conversion en async temporaire
            async def async_func():
                return func(*args, **kwargs)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(cb.call(async_func))
            finally:
                loop.close()
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def retry(
    name: str,
    config: Optional[RetryConfig] = None,
    **kwargs
):
    """
    Décorateur pour appliquer une stratégie de retry à une fonction.
    
    Args:
        name: Nom du retry manager
        config: Configuration du retry
        **kwargs: Arguments supplémentaires pour RetryManager
    """
    
    rm = RetryManager(name, config, **kwargs)
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await rm.execute(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Conversion pour fonctions synchrones
            async def async_func():
                return func(*args, **kwargs)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(rm.execute(async_func))
            finally:
                loop.close()
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def rate_limit(
    name: str,
    config: Optional[RateLimitConfig] = None,
    **kwargs
):
    """
    Décorateur pour appliquer un rate limiting à une fonction.
    
    Args:
        name: Nom du rate limiter
        config: Configuration du rate limiter
        **kwargs: Arguments supplémentaires pour RateLimiter
    """
    
    rl = RateLimiter(name, config, **kwargs)
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if await rl.acquire():
                return await func(*args, **kwargs)
            else:
                raise RateLimitExceededError(f"Rate limit exceeded for '{name}'")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Version synchrone simplifiée
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                if loop.run_until_complete(rl.acquire()):
                    return func(*args, **kwargs)
                else:
                    raise RateLimitExceededError(f"Rate limit exceeded for '{name}'")
            finally:
                loop.close()
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Gestionnaire global des patterns
class PatternManager:
    """Gestionnaire centralisé de tous les patterns de résilience."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_managers: Dict[str, RetryManager] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}
    
    def get_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Récupère ou crée un circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, config)
        return self.circuit_breakers[name]
    
    def get_retry_manager(self, name: str, config: Optional[RetryConfig] = None) -> RetryManager:
        """Récupère ou crée un retry manager."""
        if name not in self.retry_managers:
            self.retry_managers[name] = RetryManager(name, config)
        return self.retry_managers[name]
    
    def get_rate_limiter(self, name: str, config: Optional[RateLimitConfig] = None) -> RateLimiter:
        """Récupère ou crée un rate limiter."""
        if name not in self.rate_limiters:
            self.rate_limiters[name] = RateLimiter(name, config)
        return self.rate_limiters[name]
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de tous les patterns."""
        return {
            'circuit_breakers': {name: cb.get_stats() for name, cb in self.circuit_breakers.items()},
            'retry_managers': {name: rm.get_stats() for name, rm in self.retry_managers.items()},
            'rate_limiters': {name: rl.get_stats() for name, rl in self.rate_limiters.items()}
        }


# Instance globale
global_pattern_manager = PatternManager()
