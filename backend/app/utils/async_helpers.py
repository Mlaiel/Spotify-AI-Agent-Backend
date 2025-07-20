"""
Enterprise Async Helpers
========================
Advanced asynchronous utilities for high-performance Spotify AI Agent streaming platform.

Expert Team Implementation:
- Lead Developer + AI Architect: AI-powered async optimization and intelligent task scheduling
- Senior Backend Developer: High-performance async patterns and concurrency control
- Microservices Architect: Distributed async processing and service orchestration
- DBA & Data Engineer: Async database operations and connection pooling
- Performance Engineer: Async optimization, monitoring, and load balancing
"""

import asyncio
import logging
import time
import json
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Coroutine, TypeVar, Generic, Awaitable, Tuple
from abc import ABC, abstractmethod
from enum import Enum
import weakref
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools
import inspect
import contextvars
import sys

logger = logging.getLogger(__name__)

T = TypeVar('T')
P = TypeVar('P')

# === Async Types and Enums ===
class TaskPriority(Enum):
    """Task execution priority levels."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BACKGROUND = 4

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class RetryStrategy(Enum):
    """Retry strategy types."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class TaskResult(Generic[T]):
    """Result of async task execution."""
    success: bool
    result: Optional[T] = None
    error: Optional[Exception] = None
    execution_time_ms: float = 0.0
    task_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskMetrics:
    """Task execution metrics."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_execution_time_ms: float = 0.0
    peak_execution_time_ms: float = 0.0
    min_execution_time_ms: float = float('inf')
    success_rate: float = 0.0
    throughput_per_second: float = 0.0

@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    max_requests: int
    time_window_seconds: int
    burst_size: Optional[int] = None
    backoff_strategy: str = "exponential"

# === Advanced Task Manager ===
class TaskManager:
    """Advanced asynchronous task management with priorities and scheduling."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.max_concurrent_tasks = config.get('max_concurrent_tasks', 100)
        self.default_timeout = config.get('default_timeout', 30.0)
        
        # Task queues by priority
        self.task_queues = {
            TaskPriority.CRITICAL: asyncio.Queue(),
            TaskPriority.HIGH: asyncio.Queue(),
            TaskPriority.MEDIUM: asyncio.Queue(),
            TaskPriority.LOW: asyncio.Queue()
        }
        
        # Active tasks tracking
        self.active_tasks = {}
        self.task_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
        # Metrics and monitoring
        self.metrics = TaskMetrics()
        self.execution_history = deque(maxlen=1000)
        
        # Worker control
        self.workers_running = False
        self.worker_tasks = []
        
        # Task scheduling
        self.scheduler = TaskScheduler()
        
    async def start_workers(self, num_workers: int = None):
        """Start task worker coroutines."""
        if self.workers_running:
            return
        
        num_workers = num_workers or min(10, self.max_concurrent_tasks)
        self.workers_running = True
        
        for i in range(num_workers):
            worker_task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self.worker_tasks.append(worker_task)
        
        logger.info(f"Started {num_workers} task workers")
    
    async def stop_workers(self):
        """Stop all task workers."""
        self.workers_running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for workers to finish
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        self.worker_tasks.clear()
        logger.info("Stopped all task workers")
    
    async def submit_task(self, 
                         coro: Awaitable[T], 
                         priority: TaskPriority = TaskPriority.MEDIUM,
                         timeout: float = None,
                         metadata: Dict[str, Any] = None) -> str:
        """Submit async task for execution."""
        task_id = str(uuid.uuid4())
        timeout = timeout or self.default_timeout
        metadata = metadata or {}
        
        task_info = {
            'id': task_id,
            'coro': coro,
            'priority': priority,
            'timeout': timeout,
            'metadata': metadata,
            'submitted_at': datetime.now(),
            'future': asyncio.Future()
        }
        
        # Add to appropriate priority queue
        await self.task_queues[priority].put(task_info)
        self.metrics.total_tasks += 1
        
        logger.debug(f"Submitted task {task_id} with priority {priority.name}")
        return task_id
    
    async def get_task_result(self, task_id: str, timeout: float = None) -> TaskResult[T]:
        """Get result of submitted task."""
        if task_id not in self.active_tasks:
            return TaskResult(
                success=False,
                error=ValueError(f"Task {task_id} not found"),
                task_id=task_id
            )
        
        try:
            task_info = self.active_tasks[task_id]
            result = await asyncio.wait_for(task_info['future'], timeout=timeout)
            return result
        except asyncio.TimeoutError:
            return TaskResult(
                success=False,
                error=asyncio.TimeoutError(f"Task {task_id} timed out"),
                task_id=task_id
            )
    
    async def _worker_loop(self, worker_name: str):
        """Main worker loop for processing tasks."""
        logger.debug(f"Worker {worker_name} started")
        
        while self.workers_running:
            try:
                # Get task from highest priority queue with tasks
                task_info = await self._get_next_task()
                
                if task_info is None:
                    await asyncio.sleep(0.1)  # Brief sleep if no tasks
                    continue
                
                # Execute task
                async with self.task_semaphore:
                    await self._execute_task(task_info)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1)  # Brief delay on error
        
        logger.debug(f"Worker {worker_name} stopped")
    
    async def _get_next_task(self):
        """Get next task from priority queues."""
        # Check queues in priority order
        for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, TaskPriority.MEDIUM, TaskPriority.LOW]:
            queue = self.task_queues[priority]
            
            try:
                task_info = queue.get_nowait()
                queue.task_done()
                return task_info
            except asyncio.QueueEmpty:
                continue
        
        return None
    
    async def _execute_task(self, task_info: Dict[str, Any]):
        """Execute individual task."""
        task_id = task_info['id']
        start_time = time.time()
        
        try:
            # Add to active tasks
            self.active_tasks[task_id] = task_info
            
            # Execute with timeout
            result = await asyncio.wait_for(
                task_info['coro'], 
                timeout=task_info['timeout']
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            # Create successful result
            task_result = TaskResult(
                success=True,
                result=result,
                execution_time_ms=execution_time,
                task_id=task_id,
                metadata=task_info['metadata']
            )
            
            # Update metrics
            self.metrics.completed_tasks += 1
            self._update_execution_metrics(execution_time)
            
            # Set future result
            if not task_info['future'].done():
                task_info['future'].set_result(task_result)
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            # Create error result
            task_result = TaskResult(
                success=False,
                error=e,
                execution_time_ms=execution_time,
                task_id=task_id,
                metadata=task_info['metadata']
            )
            
            # Update metrics
            self.metrics.failed_tasks += 1
            self._update_execution_metrics(execution_time)
            
            # Set future result
            if not task_info['future'].done():
                task_info['future'].set_result(task_result)
            
            logger.error(f"Task {task_id} failed: {e}")
        
        finally:
            # Remove from active tasks
            self.active_tasks.pop(task_id, None)
            
            # Record execution
            self.execution_history.append({
                'task_id': task_id,
                'execution_time_ms': execution_time,
                'success': task_result.success,
                'timestamp': datetime.now()
            })
    
    def _update_execution_metrics(self, execution_time_ms: float):
        """Update execution time metrics."""
        # Update min/max
        if execution_time_ms < self.metrics.min_execution_time_ms:
            self.metrics.min_execution_time_ms = execution_time_ms
        
        if execution_time_ms > self.metrics.peak_execution_time_ms:
            self.metrics.peak_execution_time_ms = execution_time_ms
        
        # Update average
        total_completed = self.metrics.completed_tasks + self.metrics.failed_tasks
        if total_completed > 0:
            current_avg = self.metrics.avg_execution_time_ms
            self.metrics.avg_execution_time_ms = (
                (current_avg * (total_completed - 1) + execution_time_ms) / total_completed
            )
        
        # Update success rate
        if self.metrics.total_tasks > 0:
            self.metrics.success_rate = self.metrics.completed_tasks / self.metrics.total_tasks
        
        # Update throughput (tasks per second over last minute)
        recent_tasks = [
            t for t in self.execution_history 
            if (datetime.now() - t['timestamp']).total_seconds() < 60
        ]
        self.metrics.throughput_per_second = len(recent_tasks) / 60.0
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get task queue statistics."""
        stats = {}
        for priority, queue in self.task_queues.items():
            stats[priority.name] = {
                'pending_tasks': queue.qsize(),
                'max_size': queue.maxsize if queue.maxsize > 0 else 'unlimited'
            }
        
        stats['active_tasks'] = len(self.active_tasks)
        stats['total_workers'] = len(self.worker_tasks)
        stats['workers_running'] = self.workers_running
        
        return stats

# === Task Scheduler ===
class TaskScheduler:
    """Advanced task scheduling with cron-like functionality."""
    
    def __init__(self):
        self.scheduled_tasks = {}
        self.running = False
        self.scheduler_task = None
    
    async def start(self):
        """Start the scheduler."""
        if self.running:
            return
        
        self.running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Task scheduler started")
    
    async def stop(self):
        """Stop the scheduler."""
        self.running = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Task scheduler stopped")
    
    def schedule_recurring(self, 
                          name: str,
                          coro_func: Callable[[], Awaitable],
                          interval_seconds: float,
                          delay_seconds: float = 0):
        """Schedule recurring task."""
        self.scheduled_tasks[name] = {
            'type': 'recurring',
            'coro_func': coro_func,
            'interval_seconds': interval_seconds,
            'delay_seconds': delay_seconds,
            'next_run': datetime.now() + timedelta(seconds=delay_seconds),
            'last_run': None,
            'run_count': 0
        }
        
        logger.info(f"Scheduled recurring task '{name}' every {interval_seconds}s")
    
    def schedule_at(self, 
                   name: str,
                   coro_func: Callable[[], Awaitable],
                   run_at: datetime):
        """Schedule task to run at specific time."""
        self.scheduled_tasks[name] = {
            'type': 'once',
            'coro_func': coro_func,
            'run_at': run_at,
            'completed': False
        }
        
        logger.info(f"Scheduled task '{name}' to run at {run_at}")
    
    def unschedule(self, name: str) -> bool:
        """Remove scheduled task."""
        if name in self.scheduled_tasks:
            del self.scheduled_tasks[name]
            logger.info(f"Unscheduled task '{name}'")
            return True
        return False
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            try:
                now = datetime.now()
                
                # Check all scheduled tasks
                for name, task_info in list(self.scheduled_tasks.items()):
                    try:
                        if task_info['type'] == 'recurring':
                            if now >= task_info['next_run']:
                                await self._execute_scheduled_task(name, task_info)
                        
                        elif task_info['type'] == 'once':
                            if not task_info['completed'] and now >= task_info['run_at']:
                                await self._execute_scheduled_task(name, task_info)
                                task_info['completed'] = True
                    
                    except Exception as e:
                        logger.error(f"Error checking scheduled task '{name}': {e}")
                
                # Sleep for a short interval
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(5)
    
    async def _execute_scheduled_task(self, name: str, task_info: Dict[str, Any]):
        """Execute scheduled task."""
        try:
            logger.debug(f"Executing scheduled task '{name}'")
            
            # Execute the coroutine
            await task_info['coro_func']()
            
            # Update task info for recurring tasks
            if task_info['type'] == 'recurring':
                task_info['last_run'] = datetime.now()
                task_info['run_count'] += 1
                task_info['next_run'] = datetime.now() + timedelta(seconds=task_info['interval_seconds'])
            
            logger.debug(f"Completed scheduled task '{name}'")
            
        except Exception as e:
            logger.error(f"Scheduled task '{name}' failed: {e}")

# === Rate Limiter ===
class RateLimiter:
    """Advanced rate limiting with different strategies."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_timestamps = deque()
        self.current_burst = 0
        self.last_request_time = 0
        self.backoff_until = 0
        
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire rate limit tokens."""
        now = time.time()
        
        # Check if we're in backoff period
        if now < self.backoff_until:
            return False
        
        # Clean old timestamps
        cutoff_time = now - self.config.time_window_seconds
        while self.request_timestamps and self.request_timestamps[0] < cutoff_time:
            self.request_timestamps.popleft()
        
        # Check rate limit
        if len(self.request_timestamps) + tokens > self.config.max_requests:
            # Rate limit exceeded
            await self._apply_backoff(now)
            return False
        
        # Check burst limit if configured
        if self.config.burst_size:
            time_since_last = now - self.last_request_time
            if time_since_last >= 1.0:  # Reset burst counter every second
                self.current_burst = 0
            
            if self.current_burst + tokens > self.config.burst_size:
                return False
            
            self.current_burst += tokens
        
        # Add timestamps for tokens
        for _ in range(tokens):
            self.request_timestamps.append(now)
        
        self.last_request_time = now
        return True
    
    async def _apply_backoff(self, current_time: float):
        """Apply backoff strategy when rate limited."""
        if self.config.backoff_strategy == "exponential":
            # Exponential backoff
            backoff_seconds = min(60, 2 ** len(self.request_timestamps) % 10)
        elif self.config.backoff_strategy == "linear":
            # Linear backoff
            backoff_seconds = min(60, len(self.request_timestamps) * 0.1)
        else:
            # Fixed backoff
            backoff_seconds = 1.0
        
        self.backoff_until = current_time + backoff_seconds
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        now = time.time()
        cutoff_time = now - self.config.time_window_seconds
        
        # Count recent requests
        recent_requests = sum(1 for ts in self.request_timestamps if ts >= cutoff_time)
        
        return {
            'recent_requests': recent_requests,
            'max_requests': self.config.max_requests,
            'time_window_seconds': self.config.time_window_seconds,
            'current_burst': self.current_burst,
            'burst_limit': self.config.burst_size,
            'backoff_until': self.backoff_until,
            'requests_remaining': max(0, self.config.max_requests - recent_requests)
        }

# === Circuit Breaker ===
class CircuitBreaker:
    """Circuit breaker pattern for async operations."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        
        # Statistics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
    
    async def call(self, coro: Awaitable[T]) -> T:
        """Execute coroutine with circuit breaker protection."""
        self.total_calls += 1
        
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")
        
        try:
            result = await coro
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset circuit."""
        return (self.last_failure_time is not None and 
                time.time() - self.last_failure_time >= self.recovery_timeout)
    
    def _on_success(self):
        """Handle successful call."""
        self.successful_calls += 1
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failed call."""
        self.failed_calls += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        success_rate = (self.successful_calls / self.total_calls * 100) if self.total_calls > 0 else 0
        
        return {
            'state': self.state.value,
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'failure_count': self.failure_count,
            'success_rate_percent': success_rate,
            'failure_threshold': self.failure_threshold,
            'recovery_timeout': self.recovery_timeout
        }

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass

# === Async Context Managers ===
@asynccontextmanager
async def timeout_context(seconds: float):
    """Async context manager with timeout."""
    try:
        async with asyncio.timeout(seconds):
            yield
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {seconds} seconds")
        raise

@asynccontextmanager
async def retry_context(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Async context manager with retry logic."""
    attempt = 0
    while attempt < max_attempts:
        try:
            yield attempt
            break
        except Exception as e:
            attempt += 1
            if attempt >= max_attempts:
                logger.error(f"Max retry attempts ({max_attempts}) exceeded")
                raise e
            
            logger.warning(f"Attempt {attempt} failed, retrying in {delay}s: {e}")
            await asyncio.sleep(delay)
            delay *= backoff

# === Async Utilities ===
class AsyncUtils:
    """Collection of async utility functions."""
    
    @staticmethod
    async def gather_with_limit(coroutines: List[Awaitable[T]], limit: int = 10) -> List[T]:
        """Execute coroutines with concurrency limit."""
        semaphore = asyncio.Semaphore(limit)
        
        async def limited_coro(coro):
            async with semaphore:
                return await coro
        
        limited_coroutines = [limited_coro(coro) for coro in coroutines]
        return await asyncio.gather(*limited_coroutines)
    
    @staticmethod
    async def gather_with_timeout(coroutines: List[Awaitable[T]], timeout: float) -> List[Optional[T]]:
        """Execute coroutines with timeout, returning None for timed out tasks."""
        async def timeout_wrapper(coro):
            try:
                return await asyncio.wait_for(coro, timeout=timeout)
            except asyncio.TimeoutError:
                return None
        
        wrapped_coroutines = [timeout_wrapper(coro) for coro in coroutines]
        return await asyncio.gather(*wrapped_coroutines, return_exceptions=True)
    
    @staticmethod
    async def run_with_fallback(primary_coro: Awaitable[T], fallback_coro: Awaitable[T]) -> T:
        """Run primary coroutine with fallback on failure."""
        try:
            return await primary_coro
        except Exception as e:
            logger.warning(f"Primary coroutine failed, using fallback: {e}")
            return await fallback_coro
    
    @staticmethod
    async def batch_process(items: List[Any], 
                          process_func: Callable[[Any], Awaitable[T]], 
                          batch_size: int = 10,
                          delay_between_batches: float = 0) -> List[T]:
        """Process items in batches."""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_coroutines = [process_func(item) for item in batch]
            batch_results = await asyncio.gather(*batch_coroutines, return_exceptions=True)
            results.extend(batch_results)
            
            if delay_between_batches > 0 and i + batch_size < len(items):
                await asyncio.sleep(delay_between_batches)
        
        return results

# === Async Connection Pool ===
class AsyncConnectionPool(Generic[T]):
    """Generic async connection pool."""
    
    def __init__(self, 
                 create_connection: Callable[[], Awaitable[T]],
                 close_connection: Callable[[T], Awaitable[None]],
                 max_size: int = 10,
                 min_size: int = 1,
                 max_idle_time: float = 300):
        self.create_connection = create_connection
        self.close_connection = close_connection
        self.max_size = max_size
        self.min_size = min_size
        self.max_idle_time = max_idle_time
        
        self.pool = asyncio.Queue(maxsize=max_size)
        self.active_connections = set()
        self.connection_timestamps = {}
        self.total_created = 0
        self.total_closed = 0
        
        self._cleanup_task = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the connection pool."""
        if self._initialized:
            return
        
        # Create minimum connections
        for _ in range(self.min_size):
            connection = await self.create_connection()
            await self.pool.put(connection)
            self.total_created += 1
            self.connection_timestamps[id(connection)] = time.time()
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_idle_connections())
        self._initialized = True
        
        logger.info(f"Initialized connection pool with {self.min_size} connections")
    
    async def acquire(self) -> T:
        """Acquire connection from pool."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Try to get existing connection
            connection = self.pool.get_nowait()
        except asyncio.QueueEmpty:
            # Create new connection if under limit
            if len(self.active_connections) < self.max_size:
                connection = await self.create_connection()
                self.total_created += 1
                self.connection_timestamps[id(connection)] = time.time()
            else:
                # Wait for available connection
                connection = await self.pool.get()
        
        self.active_connections.add(connection)
        return connection
    
    async def release(self, connection: T):
        """Release connection back to pool."""
        self.active_connections.discard(connection)
        self.connection_timestamps[id(connection)] = time.time()
        
        try:
            self.pool.put_nowait(connection)
        except asyncio.QueueFull:
            # Pool is full, close connection
            await self.close_connection(connection)
            self.total_closed += 1
            self.connection_timestamps.pop(id(connection), None)
    
    async def close(self):
        """Close all connections and cleanup."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Close active connections
        for connection in list(self.active_connections):
            await self.close_connection(connection)
            self.total_closed += 1
        
        # Close pooled connections
        while not self.pool.empty():
            try:
                connection = self.pool.get_nowait()
                await self.close_connection(connection)
                self.total_closed += 1
            except asyncio.QueueEmpty:
                break
        
        self.active_connections.clear()
        self.connection_timestamps.clear()
        self._initialized = False
        
        logger.info("Closed connection pool")
    
    async def _cleanup_idle_connections(self):
        """Cleanup idle connections periodically."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                current_time = time.time()
                connections_to_close = []
                
                # Check pooled connections for idle timeout
                temp_connections = []
                while not self.pool.empty():
                    try:
                        connection = self.pool.get_nowait()
                        connection_time = self.connection_timestamps.get(id(connection), current_time)
                        
                        if current_time - connection_time > self.max_idle_time:
                            connections_to_close.append(connection)
                        else:
                            temp_connections.append(connection)
                    except asyncio.QueueEmpty:
                        break
                
                # Put back non-expired connections
                for connection in temp_connections:
                    try:
                        self.pool.put_nowait(connection)
                    except asyncio.QueueFull:
                        connections_to_close.append(connection)
                
                # Close expired connections
                for connection in connections_to_close:
                    await self.close_connection(connection)
                    self.total_closed += 1
                    self.connection_timestamps.pop(id(connection), None)
                
                if connections_to_close:
                    logger.debug(f"Closed {len(connections_to_close)} idle connections")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Connection cleanup error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            'pool_size': self.pool.qsize(),
            'active_connections': len(self.active_connections),
            'max_size': self.max_size,
            'min_size': self.min_size,
            'total_created': self.total_created,
            'total_closed': self.total_closed,
            'utilization_percent': (len(self.active_connections) / self.max_size) * 100
        }

# === Async Decorators ===
def async_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, exceptions: Tuple[type, ...] = (Exception,)):
    """Decorator for async function retry logic."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay
            
            while attempt < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise e
                    
                    logger.warning(f"Attempt {attempt} failed for {func.__name__}, retrying in {current_delay}s: {e}")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
        return wrapper
    return decorator

def async_timeout(seconds: float):
    """Decorator for async function timeout."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error(f"Function {func.__name__} timed out after {seconds} seconds")
                raise
        return wrapper
    return decorator

def async_rate_limit(rate_limiter: RateLimiter):
    """Decorator for async function rate limiting."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not await rate_limiter.acquire():
                raise RateLimitExceededError("Rate limit exceeded")
            return await func(*args, **kwargs)
        return wrapper
    return decorator

class RateLimitExceededError(Exception):
    """Exception raised when rate limit is exceeded."""
    pass

# === Factory Functions ===
def create_task_manager(config: Dict[str, Any] = None) -> TaskManager:
    """Create task manager instance."""
    return TaskManager(config)

def create_rate_limiter(max_requests: int, time_window_seconds: int, **kwargs) -> RateLimiter:
    """Create rate limiter instance."""
    config = RateLimitConfig(
        max_requests=max_requests,
        time_window_seconds=time_window_seconds,
        **kwargs
    )
    return RateLimiter(config)

def create_circuit_breaker(failure_threshold: int = 5, recovery_timeout: float = 60, **kwargs) -> CircuitBreaker:
    """Create circuit breaker instance."""
    return CircuitBreaker(failure_threshold, recovery_timeout, **kwargs)

def create_connection_pool(create_func: Callable[[], Awaitable[T]], 
                          close_func: Callable[[T], Awaitable[None]], 
                          **kwargs) -> AsyncConnectionPool[T]:
    """Create async connection pool instance."""
    return AsyncConnectionPool(create_func, close_func, **kwargs)

# === Export Classes ===
__all__ = [
    'TaskManager', 'TaskScheduler', 'RateLimiter', 'CircuitBreaker', 
    'AsyncUtils', 'AsyncConnectionPool',
    'TaskPriority', 'CircuitState',
    'TaskResult', 'TaskMetrics', 'RateLimitConfig',
    'timeout_context', 'retry_context',
    'async_retry', 'async_timeout', 'async_rate_limit',
    'CircuitBreakerOpenError', 'RateLimitExceededError',
    'create_task_manager', 'create_rate_limiter', 'create_circuit_breaker', 'create_connection_pool'
]
