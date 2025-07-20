#!/usr/bin/env python3
"""
Circuit Breaker Pattern Implementation for PagerDuty Integration.

Advanced circuit breaker implementation for fault tolerance and resilience
in external API calls and service integrations.

Features:
- Multiple circuit breaker states (Closed, Open, Half-Open)
- Configurable failure thresholds and timeouts
- Exponential backoff for recovery attempts
- Health monitoring and metrics collection
- Thread-safe implementation
- Async and sync support
- Customizable failure detection
- Circuit breaker analytics and reporting
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from functools import wraps
import threading
from collections import deque, defaultdict

from .metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Number of failures to open circuit
    recovery_timeout: float = 60.0  # Seconds to wait before half-open
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout: float = 30.0  # Request timeout in seconds
    expected_exception: type = Exception  # Exception type that counts as failure
    fallback_function: Optional[Callable] = None  # Fallback function
    monitor_window: int = 300  # Window for failure rate monitoring (seconds)
    max_failures_in_window: int = 10  # Max failures in monitoring window


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    circuit_open_count: int = 0
    circuit_half_open_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    average_response_time: float = 0.0
    failure_rate: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls


@dataclass
class CallRecord:
    """Record of a function call attempt."""
    timestamp: datetime
    success: bool
    duration: float
    exception: Optional[Exception] = None


class CircuitBreaker:
    """
    Advanced circuit breaker implementation.
    
    The circuit breaker operates in three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing state, requests are blocked and fail fast
    - HALF_OPEN: Testing state, limited requests allowed to test recovery
    
    Features:
    - Configurable failure thresholds and timeouts
    - Exponential backoff for recovery
    - Comprehensive metrics and monitoring
    - Thread-safe operation
    - Async and sync support
    - Customizable failure detection
    """
    
    def __init__(self, 
                 name: str = "default",
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 success_threshold: int = 3,
                 timeout: float = 30.0,
                 expected_exception: type = Exception,
                 fallback_function: Optional[Callable] = None,
                 monitor_window: int = 300,
                 max_failures_in_window: int = 10):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name for identification
            failure_threshold: Number of failures to open circuit
            recovery_timeout: Seconds to wait before half-open
            success_threshold: Successes needed to close from half-open
            timeout: Request timeout in seconds
            expected_exception: Exception type that counts as failure
            fallback_function: Function to call when circuit is open
            monitor_window: Window for failure rate monitoring (seconds)
            max_failures_in_window: Max failures in monitoring window
        """
        self.name = name
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            timeout=timeout,
            expected_exception=expected_exception,
            fallback_function=fallback_function,
            monitor_window=monitor_window,
            max_failures_in_window=max_failures_in_window
        )
        
        # State management
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._last_success_time: Optional[datetime] = None
        self._next_attempt_time: Optional[datetime] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Call history for monitoring
        self._call_history: deque = deque(maxlen=1000)
        self._recent_calls: Dict[str, List[CallRecord]] = defaultdict(list)
        
        # Metrics
        self.metrics = CircuitBreakerMetrics()
        self.metrics_collector = MetricsCollector()
        
        # Recovery configuration
        self._recovery_backoff_factor = 2.0
        self._max_recovery_timeout = 300.0  # 5 minutes max
        
        logger.info(f"Circuit breaker '{name}' initialized with threshold {failure_threshold}")
    
    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        with self._lock:
            return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitBreakerState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing)."""
        return self.state == CircuitBreakerState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing)."""
        return self.state == CircuitBreakerState.HALF_OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        if self._state != CircuitBreakerState.OPEN:
            return False
        
        if self._next_attempt_time is None:
            return True
        
        return datetime.utcnow() >= self._next_attempt_time
    
    def _calculate_next_attempt_time(self) -> datetime:
        """Calculate next attempt time with exponential backoff."""
        if self._last_failure_time is None:
            return datetime.utcnow() + timedelta(seconds=self.config.recovery_timeout)
        
        # Calculate backoff time
        failures_since_last_success = self._failure_count
        backoff_time = min(
            self.config.recovery_timeout * (self._recovery_backoff_factor ** failures_since_last_success),
            self._max_recovery_timeout
        )
        
        return datetime.utcnow() + timedelta(seconds=backoff_time)
    
    def _record_success(self, duration: float):
        """Record a successful call."""
        with self._lock:
            self._success_count += 1
            self._last_success_time = datetime.utcnow()
            
            # Update metrics
            self.metrics.total_calls += 1
            self.metrics.successful_calls += 1
            self.metrics.last_success_time = self._last_success_time
            
            # Update average response time
            total_time = self.metrics.average_response_time * (self.metrics.total_calls - 1)
            self.metrics.average_response_time = (total_time + duration) / self.metrics.total_calls
            
            # Record call
            call_record = CallRecord(
                timestamp=self._last_success_time,
                success=True,
                duration=duration
            )
            self._call_history.append(call_record)
            
            # State transitions
            if self._state == CircuitBreakerState.HALF_OPEN:
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitBreakerState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' closed after recovery")
                    self.metrics_collector.increment(f'circuit_breaker_{self.name}_closed')
            
            elif self._state == CircuitBreakerState.OPEN:
                # This shouldn't happen, but reset if it does
                self._state = CircuitBreakerState.CLOSED
                self._failure_count = 0
                logger.warning(f"Circuit breaker '{self.name}' unexpectedly reset to closed")
    
    def _record_failure(self, exception: Exception, duration: float):
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.utcnow()
            
            # Update metrics
            self.metrics.total_calls += 1
            self.metrics.failed_calls += 1
            self.metrics.last_failure_time = self._last_failure_time
            
            # Calculate failure rate
            if self.metrics.total_calls > 0:
                self.metrics.failure_rate = self.metrics.failed_calls / self.metrics.total_calls
            
            # Record call
            call_record = CallRecord(
                timestamp=self._last_failure_time,
                success=False,
                duration=duration,
                exception=exception
            )
            self._call_history.append(call_record)
            
            # State transitions
            if self._state == CircuitBreakerState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitBreakerState.OPEN
                    self._next_attempt_time = self._calculate_next_attempt_time()
                    self.metrics.circuit_open_count += 1
                    logger.warning(f"Circuit breaker '{self.name}' opened after {self._failure_count} failures")
                    self.metrics_collector.increment(f'circuit_breaker_{self.name}_opened')
            
            elif self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.OPEN
                self._next_attempt_time = self._calculate_next_attempt_time()
                self.metrics.circuit_open_count += 1
                logger.warning(f"Circuit breaker '{self.name}' reopened during half-open test")
                self.metrics_collector.increment(f'circuit_breaker_{self.name}_reopened')
    
    def _check_recent_failure_rate(self) -> bool:
        """Check if recent failure rate exceeds threshold."""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.config.monitor_window)
        
        # Count recent failures
        recent_failures = sum(
            1 for record in self._call_history
            if record.timestamp >= window_start and not record.success
        )
        
        return recent_failures >= self.config.max_failures_in_window
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection (async).
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: When circuit is open
            Original exception: When function fails
        """
        with self._lock:
            # Check if we should attempt reset
            if self._state == CircuitBreakerState.OPEN and self._should_attempt_reset():
                self._state = CircuitBreakerState.HALF_OPEN
                self._success_count = 0
                self.metrics.circuit_half_open_count += 1
                logger.info(f"Circuit breaker '{self.name}' entering half-open state")
                self.metrics_collector.increment(f'circuit_breaker_{self.name}_half_open')
            
            # Fail fast if circuit is open
            if self._state == CircuitBreakerState.OPEN:
                retry_after = None
                if self._next_attempt_time:
                    retry_after = (self._next_attempt_time - datetime.utcnow()).total_seconds()
                    retry_after = max(0, retry_after)
                
                # Try fallback function if available
                if self.config.fallback_function:
                    try:
                        logger.info(f"Circuit breaker '{self.name}' using fallback function")
                        return await self.config.fallback_function(*args, **kwargs)
                    except Exception as e:
                        logger.error(f"Fallback function failed: {e}")
                
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open", 
                    retry_after
                )
        
        # Execute function with timeout
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs), 
                    timeout=self.config.timeout
                )
            else:
                result = func(*args, **kwargs)
            
            duration = time.time() - start_time
            self._record_success(duration)
            
            return result
            
        except self.config.expected_exception as e:
            duration = time.time() - start_time
            self._record_failure(e, duration)
            raise
        
        except asyncio.TimeoutError as e:
            duration = time.time() - start_time
            self._record_failure(e, duration)
            raise
    
    def call_sync(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection (sync).
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        return asyncio.run(self.call_async(func, *args, **kwargs))
    
    def reset(self):
        """Manually reset circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._next_attempt_time = None
            
            logger.info(f"Circuit breaker '{self.name}' manually reset")
            self.metrics_collector.increment(f'circuit_breaker_{self.name}_reset')
    
    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get current metrics."""
        with self._lock:
            return self.metrics
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status information."""
        with self._lock:
            return {
                'name': self.name,
                'state': self._state.value,
                'failure_count': self._failure_count,
                'success_count': self._success_count,
                'last_failure_time': self._last_failure_time.isoformat() if self._last_failure_time else None,
                'last_success_time': self._last_success_time.isoformat() if self._last_success_time else None,
                'next_attempt_time': self._next_attempt_time.isoformat() if self._next_attempt_time else None,
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'recovery_timeout': self.config.recovery_timeout,
                    'success_threshold': self.config.success_threshold,
                    'timeout': self.config.timeout
                },
                'metrics': {
                    'total_calls': self.metrics.total_calls,
                    'successful_calls': self.metrics.successful_calls,
                    'failed_calls': self.metrics.failed_calls,
                    'success_rate': self.metrics.success_rate,
                    'failure_rate': self.metrics.failure_rate,
                    'average_response_time': self.metrics.average_response_time,
                    'circuit_open_count': self.metrics.circuit_open_count,
                    'circuit_half_open_count': self.metrics.circuit_half_open_count
                }
            }


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """
    Get or create a circuit breaker by name.
    
    Args:
        name: Circuit breaker name
        **kwargs: Circuit breaker configuration
        
    Returns:
        CircuitBreaker instance
    """
    with _registry_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name=name, **kwargs)
        return _circuit_breakers[name]


def list_circuit_breakers() -> List[str]:
    """List all registered circuit breaker names."""
    with _registry_lock:
        return list(_circuit_breakers.keys())


def get_all_circuit_breaker_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all circuit breakers."""
    with _registry_lock:
        return {name: cb.get_status() for name, cb in _circuit_breakers.items()}


# Decorator for circuit breaker protection
def circuit_breaker(name: Optional[str] = None,
                   failure_threshold: int = 5,
                   recovery_timeout: float = 60.0,
                   success_threshold: int = 3,
                   timeout: float = 30.0,
                   expected_exception: type = Exception,
                   fallback_function: Optional[Callable] = None):
    """
    Decorator to protect function with circuit breaker.
    
    Args:
        name: Circuit breaker name (function name if not provided)
        failure_threshold: Number of failures to open circuit
        recovery_timeout: Seconds to wait before half-open
        success_threshold: Successes needed to close from half-open
        timeout: Request timeout in seconds
        expected_exception: Exception type that counts as failure
        fallback_function: Function to call when circuit is open
    """
    def decorator(func):
        breaker_name = name or func.__name__
        cb = get_circuit_breaker(
            breaker_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            timeout=timeout,
            expected_exception=expected_exception,
            fallback_function=fallback_function
        )
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await cb.call_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return cb.call_sync(func, *args, **kwargs)
            return sync_wrapper
    
    return decorator


# Health check function
async def check_circuit_breaker_health() -> Dict[str, Any]:
    """Check health of all circuit breakers."""
    with _registry_lock:
        health_status = {
            'healthy': True,
            'circuit_breakers': {},
            'summary': {
                'total': len(_circuit_breakers),
                'closed': 0,
                'open': 0,
                'half_open': 0
            }
        }
        
        for name, cb in _circuit_breakers.items():
            status = cb.get_status()
            health_status['circuit_breakers'][name] = status
            
            state = status['state']
            health_status['summary'][state] += 1
            
            # Mark as unhealthy if any circuit is open
            if state == 'open':
                health_status['healthy'] = False
        
        return health_status
