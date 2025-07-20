"""
Circuit Breaker Pattern pour le Système de Cache
===============================================

Implémentation complète du pattern Circuit Breaker pour la gestion
des failures et la protection contre les cascades d'erreurs dans
le système de cache multi-niveaux.

Fonctionnalités:
- États de circuit avec transitions automatiques
- Seuils configurables de failure
- Récupération progressive (half-open state)
- Monitoring des métriques de santé
- Alerting intégré
- Support pour backends multiples
- Statistiques détaillées

Auteurs: Équipe Spotify AI Agent - Direction technique Fahed Mlaiel
"""

import time
import threading
import logging
from typing import Any, Dict, Optional, Callable, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import asyncio
from contextlib import contextmanager, asynccontextmanager

from .exceptions import CacheCircuitBreakerError


class CircuitState(Enum):
    """États du circuit breaker"""
    CLOSED = "closed"           # Opérations normales
    OPEN = "open"              # Circuit ouvert, rejette les requêtes
    HALF_OPEN = "half_open"    # Test de récupération


@dataclass
class CircuitBreakerConfig:
    """Configuration du circuit breaker"""
    failure_threshold: int = 5          # Nombre d'échecs avant ouverture
    recovery_timeout: float = 30.0      # Temps avant test de récupération (secondes)
    success_threshold: int = 3          # Succès requis pour fermeture en half-open
    timeout: float = 5.0               # Timeout des opérations
    expected_exception: type = Exception # Type d'exception à capturer
    name: str = "default"              # Nom du circuit breaker
    enable_monitoring: bool = True      # Activation du monitoring
    enable_alerting: bool = True       # Activation des alertes


@dataclass
class CircuitBreakerMetrics:
    """Métriques du circuit breaker"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    timeout_calls: int = 0
    state_changes: int = 0
    current_failure_streak: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_state_change: Optional[datetime] = None
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def success_rate(self) -> float:
        """Taux de succès en pourcentage"""
        total = self.successful_calls + self.failed_calls
        return (self.successful_calls / total * 100) if total > 0 else 0.0
    
    @property
    def failure_rate(self) -> float:
        """Taux d'échec en pourcentage"""
        return 100.0 - self.success_rate
    
    @property
    def avg_response_time(self) -> float:
        """Temps de réponse moyen"""
        return sum(self.response_times) / len(self.response_times) if self.response_times else 0.0


class CircuitBreaker:
    """Circuit Breaker pour protection contre les failures en cascade"""
    
    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.lock = threading.RLock()
        self.logger = logging.getLogger(f"{__name__}.{self.config.name}")
        
        # Callbacks pour événements
        self.on_state_change: Optional[Callable] = None
        self.on_failure: Optional[Callable] = None
        self.on_success: Optional[Callable] = None
        
        # Dernière tentative en half-open
        self.last_attempt_time: Optional[datetime] = None
        self.half_open_successes = 0
    
    @contextmanager
    def __call__(self):
        """Context manager pour synchrone"""
        if not self._should_allow_request():
            self._record_rejected_call()
            raise CacheCircuitBreakerError(
                self.config.name,
                self.state.value,
                self.metrics.current_failure_streak,
                self.metrics.last_failure_time
            )
        
        start_time = time.time()
        try:
            yield
            self._record_success(time.time() - start_time)
        except self.config.expected_exception as e:
            self._record_failure(time.time() - start_time)
            raise
        except Exception as e:
            # Exception inattendue, on la traite comme un échec
            self._record_failure(time.time() - start_time)
            raise
    
    @asynccontextmanager
    async def async_call(self):
        """Context manager pour asynchrone"""
        if not self._should_allow_request():
            self._record_rejected_call()
            raise CacheCircuitBreakerError(
                self.config.name,
                self.state.value,
                self.metrics.current_failure_streak,
                self.metrics.last_failure_time
            )
        
        start_time = time.time()
        try:
            yield
            self._record_success(time.time() - start_time)
        except self.config.expected_exception as e:
            self._record_failure(time.time() - start_time)
            raise
        except Exception as e:
            self._record_failure(time.time() - start_time)
            raise
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Exécute une fonction protégée par le circuit breaker"""
        with self:
            return func(*args, **kwargs)
    
    async def async_call_func(self, func: Callable, *args, **kwargs) -> Any:
        """Exécute une fonction async protégée par le circuit breaker"""
        async with self.async_call():
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
    
    def _should_allow_request(self) -> bool:
        """Détermine si une requête doit être autorisée"""
        with self.lock:
            current_time = datetime.now()
            
            if self.state == CircuitState.CLOSED:
                return True
            
            elif self.state == CircuitState.OPEN:
                # Vérifier si on peut passer en half-open
                if (self.metrics.last_failure_time and 
                    (current_time - self.metrics.last_failure_time).total_seconds() >= self.config.recovery_timeout):
                    self._transition_to_half_open()
                    return True
                return False
            
            elif self.state == CircuitState.HALF_OPEN:
                # Permettre seulement une requête à la fois en half-open
                if (self.last_attempt_time is None or 
                    (current_time - self.last_attempt_time).total_seconds() >= 1.0):
                    self.last_attempt_time = current_time
                    return True
                return False
    
    def _record_success(self, response_time: float):
        """Enregistre un succès"""
        with self.lock:
            self.metrics.total_calls += 1
            self.metrics.successful_calls += 1
            self.metrics.last_success_time = datetime.now()
            self.metrics.response_times.append(response_time)
            self.metrics.current_failure_streak = 0
            
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_successes += 1
                if self.half_open_successes >= self.config.success_threshold:
                    self._transition_to_closed()
            
            if self.on_success:
                self.on_success(self.metrics)
    
    def _record_failure(self, response_time: float):
        """Enregistre un échec"""
        with self.lock:
            self.metrics.total_calls += 1
            self.metrics.failed_calls += 1
            self.metrics.last_failure_time = datetime.now()
            self.metrics.response_times.append(response_time)
            self.metrics.current_failure_streak += 1
            
            if self.state == CircuitState.CLOSED:
                if self.metrics.current_failure_streak >= self.config.failure_threshold:
                    self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()
            
            if self.on_failure:
                self.on_failure(self.metrics)
    
    def _record_rejected_call(self):
        """Enregistre un appel rejeté"""
        with self.lock:
            self.metrics.total_calls += 1
            self.metrics.rejected_calls += 1
    
    def _transition_to_open(self):
        """Transition vers l'état OPEN"""
        if self.state != CircuitState.OPEN:
            self.logger.warning(f"Circuit breaker '{self.config.name}' opening")
            self._change_state(CircuitState.OPEN)
    
    def _transition_to_half_open(self):
        """Transition vers l'état HALF_OPEN"""
        if self.state != CircuitState.HALF_OPEN:
            self.logger.info(f"Circuit breaker '{self.config.name}' transitioning to half-open")
            self.half_open_successes = 0
            self._change_state(CircuitState.HALF_OPEN)
    
    def _transition_to_closed(self):
        """Transition vers l'état CLOSED"""
        if self.state != CircuitState.CLOSED:
            self.logger.info(f"Circuit breaker '{self.config.name}' closing")
            self.half_open_successes = 0
            self._change_state(CircuitState.CLOSED)
    
    def _change_state(self, new_state: CircuitState):
        """Change l'état du circuit breaker"""
        old_state = self.state
        self.state = new_state
        self.metrics.state_changes += 1
        self.metrics.last_state_change = datetime.now()
        
        if self.on_state_change:
            self.on_state_change(old_state, new_state, self.metrics)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du circuit breaker"""
        with self.lock:
            return {
                "name": self.config.name,
                "state": self.state.value,
                "total_calls": self.metrics.total_calls,
                "successful_calls": self.metrics.successful_calls,
                "failed_calls": self.metrics.failed_calls,
                "rejected_calls": self.metrics.rejected_calls,
                "success_rate": self.metrics.success_rate,
                "failure_rate": self.metrics.failure_rate,
                "current_failure_streak": self.metrics.current_failure_streak,
                "avg_response_time": self.metrics.avg_response_time,
                "state_changes": self.metrics.state_changes,
                "last_failure_time": self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
                "last_success_time": self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None,
                "last_state_change": self.metrics.last_state_change.isoformat() if self.metrics.last_state_change else None
            }
    
    def reset(self):
        """Remet à zéro le circuit breaker"""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.metrics = CircuitBreakerMetrics()
            self.half_open_successes = 0
            self.last_attempt_time = None
            self.logger.info(f"Circuit breaker '{self.config.name}' reset")
    
    def force_open(self):
        """Force l'ouverture du circuit breaker"""
        with self.lock:
            self._transition_to_open()
            self.logger.warning(f"Circuit breaker '{self.config.name}' forced open")
    
    def force_close(self):
        """Force la fermeture du circuit breaker"""
        with self.lock:
            self._transition_to_closed()
            self.logger.info(f"Circuit breaker '{self.config.name}' forced closed")


class CircuitBreakerManager:
    """Gestionnaire centralisé de circuit breakers"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.global_metrics = defaultdict(int)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Monitoring périodique
        self._monitoring_enabled = False
        self._monitoring_interval = 30  # secondes
        self._monitoring_task = None
    
    def get_or_create(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Récupère ou crée un circuit breaker"""
        with self.lock:
            if name not in self.circuit_breakers:
                if config is None:
                    config = CircuitBreakerConfig(name=name)
                else:
                    config.name = name
                
                circuit_breaker = CircuitBreaker(config)
                
                # Enregistrement des callbacks pour métriques globales
                circuit_breaker.on_state_change = self._on_state_change
                circuit_breaker.on_failure = self._on_failure
                circuit_breaker.on_success = self._on_success
                
                self.circuit_breakers[name] = circuit_breaker
                self.logger.info(f"Created circuit breaker: {name}")
            
            return self.circuit_breakers[name]
    
    def remove(self, name: str) -> bool:
        """Supprime un circuit breaker"""
        with self.lock:
            if name in self.circuit_breakers:
                del self.circuit_breakers[name]
                self.logger.info(f"Removed circuit breaker: {name}")
                return True
            return False
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de tous les circuit breakers"""
        with self.lock:
            metrics = {
                "global": dict(self.global_metrics),
                "circuit_breakers": {}
            }
            
            for name, cb in self.circuit_breakers.items():
                metrics["circuit_breakers"][name] = cb.get_metrics()
            
            return metrics
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de la santé des circuit breakers"""
        with self.lock:
            total_cbs = len(self.circuit_breakers)
            open_cbs = sum(1 for cb in self.circuit_breakers.values() if cb.state == CircuitState.OPEN)
            half_open_cbs = sum(1 for cb in self.circuit_breakers.values() if cb.state == CircuitState.HALF_OPEN)
            closed_cbs = total_cbs - open_cbs - half_open_cbs
            
            # Circuit breakers problématiques
            problematic_cbs = []
            for name, cb in self.circuit_breakers.items():
                if cb.state != CircuitState.CLOSED or cb.metrics.failure_rate > 10:
                    problematic_cbs.append({
                        "name": name,
                        "state": cb.state.value,
                        "failure_rate": cb.metrics.failure_rate,
                        "current_failure_streak": cb.metrics.current_failure_streak
                    })
            
            return {
                "total_circuit_breakers": total_cbs,
                "closed": closed_cbs,
                "open": open_cbs,
                "half_open": half_open_cbs,
                "health_status": "healthy" if open_cbs == 0 else "degraded" if open_cbs < total_cbs * 0.5 else "critical",
                "problematic_circuit_breakers": problematic_cbs
            }
    
    def reset_all(self):
        """Remet à zéro tous les circuit breakers"""
        with self.lock:
            for cb in self.circuit_breakers.values():
                cb.reset()
            self.global_metrics.clear()
            self.logger.info("Reset all circuit breakers")
    
    def start_monitoring(self, interval: int = 30):
        """Démarre le monitoring périodique"""
        if not self._monitoring_enabled:
            self._monitoring_enabled = True
            self._monitoring_interval = interval
            
            if asyncio.get_event_loop().is_running():
                self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            else:
                # Fallback pour environnement sans event loop
                threading.Thread(target=self._sync_monitoring_loop, daemon=True).start()
            
            self.logger.info(f"Started circuit breaker monitoring (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Arrête le monitoring périodique"""
        self._monitoring_enabled = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
        self.logger.info("Stopped circuit breaker monitoring")
    
    async def _monitoring_loop(self):
        """Boucle de monitoring asynchrone"""
        while self._monitoring_enabled:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self._monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5)
    
    def _sync_monitoring_loop(self):
        """Boucle de monitoring synchrone"""
        while self._monitoring_enabled:
            try:
                asyncio.run(self._perform_health_check())
                time.sleep(self._monitoring_interval)
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(5)
    
    async def _perform_health_check(self):
        """Effectue un contrôle de santé"""
        health_summary = self.get_health_summary()
        
        if health_summary["health_status"] != "healthy":
            self.logger.warning(f"Circuit breaker health check: {health_summary}")
            
            # Ici, on pourrait déclencher des alertes
            for problematic in health_summary["problematic_circuit_breakers"]:
                if problematic["state"] == "open":
                    self.logger.error(
                        f"Circuit breaker '{problematic['name']}' is OPEN "
                        f"(failure rate: {problematic['failure_rate']:.1f}%)"
                    )
    
    def _on_state_change(self, old_state: CircuitState, new_state: CircuitState, metrics):
        """Callback pour changement d'état"""
        self.global_metrics[f"state_change_{old_state.value}_to_{new_state.value}"] += 1
        
        if new_state == CircuitState.OPEN:
            self.global_metrics["total_opens"] += 1
        elif new_state == CircuitState.CLOSED and old_state != CircuitState.CLOSED:
            self.global_metrics["total_recoveries"] += 1
    
    def _on_failure(self, metrics):
        """Callback pour échec"""
        self.global_metrics["total_failures"] += 1
    
    def _on_success(self, metrics):
        """Callback pour succès"""
        self.global_metrics["total_successes"] += 1


# Instance globale du gestionnaire de circuit breakers
circuit_breaker_manager = CircuitBreakerManager()


def circuit_breaker(name: str, config: CircuitBreakerConfig = None):
    """Décorateur pour circuit breaker"""
    def decorator(func):
        cb = circuit_breaker_manager.get_or_create(name, config)
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                return await cb.async_call_func(func, *args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                return cb.call(func, *args, **kwargs)
            return sync_wrapper
    
    return decorator
