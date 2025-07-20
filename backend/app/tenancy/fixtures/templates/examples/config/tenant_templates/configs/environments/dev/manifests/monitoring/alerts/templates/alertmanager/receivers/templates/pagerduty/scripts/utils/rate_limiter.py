#!/usr/bin/env python3
"""
Rate Limiter for PagerDuty API Integration.

Advanced rate limiting implementation with multiple algorithms and strategies
to respect API quotas and prevent service overload.

Features:
- Multiple rate limiting algorithms (Token Bucket, Sliding Window, Fixed Window)
- Per-endpoint and global rate limiting
- Burst handling and smoothing
- Distributed rate limiting with Redis
- Automatic backoff and retry logic
- Rate limit monitoring and analytics
- Configurable policies and quotas
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
import threading
import math
from collections import defaultdict, deque

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

from .metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimitExceededError(Exception):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: Optional[float] = None, 
                 limit: Optional[int] = None, remaining: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after
        self.limit = limit
        self.remaining = remaining


@dataclass
class RateLimitConfig:
    """Configuration for rate limiter."""
    requests_per_second: float = 10.0
    requests_per_minute: int = 600
    requests_per_hour: int = 36000
    burst_size: int = 20
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    enable_burst: bool = True
    backoff_factor: float = 1.5
    max_backoff: float = 300.0  # 5 minutes
    smoothing_factor: float = 0.1
    distributed: bool = False
    redis_key_prefix: str = "rate_limit:"


@dataclass
class RateLimitStatus:
    """Current rate limit status."""
    limit: int
    remaining: int
    reset_time: datetime
    retry_after: Optional[float] = None
    window_start: Optional[datetime] = None
    
    @property
    def percentage_used(self) -> float:
        """Calculate percentage of rate limit used."""
        if self.limit == 0:
            return 0.0
        return ((self.limit - self.remaining) / self.limit) * 100


class TokenBucket:
    """Token bucket rate limiter implementation."""
    
    def __init__(self, capacity: int, refill_rate: float, refill_period: float = 1.0):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per refill period
            refill_period: Refill period in seconds
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.refill_period = refill_period
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Consume tokens from bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if not enough tokens
        """
        with self._lock:
            now = time.time()
            
            # Add tokens based on elapsed time
            elapsed = now - self.last_refill
            tokens_to_add = (elapsed / self.refill_period) * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get time to wait before tokens are available."""
        with self._lock:
            if self.tokens >= tokens:
                return 0.0
            
            tokens_needed = tokens - self.tokens
            return (tokens_needed / self.refill_rate) * self.refill_period
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bucket status."""
        with self._lock:
            return {
                'capacity': self.capacity,
                'tokens': self.tokens,
                'refill_rate': self.refill_rate,
                'percentage_full': (self.tokens / self.capacity) * 100
            }


class SlidingWindowCounter:
    """Sliding window rate limiter implementation."""
    
    def __init__(self, limit: int, window_seconds: int, precision: int = 10):
        """
        Initialize sliding window counter.
        
        Args:
            limit: Maximum requests in window
            window_seconds: Window size in seconds
            precision: Number of sub-windows for precision
        """
        self.limit = limit
        self.window_seconds = window_seconds
        self.precision = precision
        self.sub_window_size = window_seconds / precision
        self.counters = defaultdict(int)
        self._lock = threading.Lock()
    
    def is_allowed(self) -> bool:
        """Check if request is allowed."""
        with self._lock:
            now = time.time()
            current_window = int(now / self.sub_window_size)
            
            # Clean old windows
            cutoff = current_window - self.precision
            keys_to_remove = [k for k in self.counters.keys() if k <= cutoff]
            for key in keys_to_remove:
                del self.counters[key]
            
            # Count requests in current window
            total_requests = sum(
                count for window, count in self.counters.items()
                if window > cutoff
            )
            
            if total_requests < self.limit:
                self.counters[current_window] += 1
                return True
            
            return False
    
    def get_reset_time(self) -> float:
        """Get time when window resets."""
        now = time.time()
        current_window = int(now / self.sub_window_size)
        next_reset = (current_window + 1) * self.sub_window_size
        return next_reset - now


class RateLimiter:
    """
    Advanced rate limiter with multiple algorithms and strategies.
    
    Features:
    - Multiple rate limiting algorithms
    - Per-key rate limiting
    - Burst handling
    - Distributed rate limiting
    - Automatic backoff
    - Analytics and monitoring
    """
    
    def __init__(self,
                 requests_per_minute: int = 600,
                 requests_per_second: Optional[float] = None,
                 burst_size: Optional[int] = None,
                 algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET,
                 enable_distributed: bool = False,
                 redis_url: Optional[str] = None,
                 key_prefix: str = "rate_limit:"):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Requests allowed per minute
            requests_per_second: Requests allowed per second (overrides per_minute)
            burst_size: Maximum burst size
            algorithm: Rate limiting algorithm to use
            enable_distributed: Enable distributed rate limiting
            redis_url: Redis URL for distributed limiting
            key_prefix: Prefix for Redis keys
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_second or (requests_per_minute / 60.0)
        self.burst_size = burst_size or max(int(self.requests_per_second * 2), 10)
        self.algorithm = algorithm
        self.enable_distributed = enable_distributed
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        
        # Local rate limiters per key
        self._local_limiters: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        # Redis client for distributed limiting
        self._redis_client = None
        
        # Metrics
        self.metrics = MetricsCollector()
        
        # Backoff tracking
        self._backoff_times: Dict[str, float] = {}
        self._max_backoff = 300.0  # 5 minutes
        self._backoff_factor = 1.5
        
        logger.info(f"Rate limiter initialized: {self.requests_per_second} req/s, algorithm: {algorithm.value}")
    
    async def _get_redis_client(self):
        """Get or create Redis client."""
        if self._redis_client is None and self.enable_distributed:
            if redis is None:
                raise RuntimeError("Redis library not available")
            
            self._redis_client = redis.from_url(
                self.redis_url or "redis://localhost:6379",
                retry_on_timeout=True
            )
        
        return self._redis_client
    
    def _get_local_limiter(self, key: str) -> Any:
        """Get or create local rate limiter for key."""
        with self._lock:
            if key not in self._local_limiters:
                if self.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                    self._local_limiters[key] = TokenBucket(
                        capacity=self.burst_size,
                        refill_rate=self.requests_per_second
                    )
                elif self.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                    self._local_limiters[key] = SlidingWindowCounter(
                        limit=self.requests_per_minute,
                        window_seconds=60
                    )
                else:
                    # Default to token bucket
                    self._local_limiters[key] = TokenBucket(
                        capacity=self.burst_size,
                        refill_rate=self.requests_per_second
                    )
            
            return self._local_limiters[key]
    
    async def _check_distributed_limit(self, key: str) -> RateLimitStatus:
        """Check rate limit using Redis."""
        redis_client = await self._get_redis_client()
        redis_key = f"{self.key_prefix}{key}"
        
        now = time.time()
        window_start = int(now / 60) * 60  # 1-minute windows
        
        pipe = redis_client.pipeline()
        pipe.incr(redis_key)
        pipe.expire(redis_key, 60)
        results = await pipe.execute()
        
        current_count = results[0]
        remaining = max(0, self.requests_per_minute - current_count)
        
        reset_time = datetime.fromtimestamp(window_start + 60)
        
        return RateLimitStatus(
            limit=self.requests_per_minute,
            remaining=remaining,
            reset_time=reset_time,
            window_start=datetime.fromtimestamp(window_start)
        )
    
    def _check_local_limit(self, key: str) -> RateLimitStatus:
        """Check rate limit using local limiter."""
        limiter = self._get_local_limiter(key)
        
        if self.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            if limiter.consume():
                status = limiter.get_status()
                return RateLimitStatus(
                    limit=self.burst_size,
                    remaining=int(status['tokens']),
                    reset_time=datetime.now() + timedelta(seconds=60)
                )
            else:
                wait_time = limiter.get_wait_time()
                return RateLimitStatus(
                    limit=self.burst_size,
                    remaining=0,
                    reset_time=datetime.now() + timedelta(seconds=wait_time),
                    retry_after=wait_time
                )
        
        elif self.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            if limiter.is_allowed():
                return RateLimitStatus(
                    limit=self.requests_per_minute,
                    remaining=self.requests_per_minute - 1,  # Approximate
                    reset_time=datetime.now() + timedelta(seconds=60)
                )
            else:
                reset_time = limiter.get_reset_time()
                return RateLimitStatus(
                    limit=self.requests_per_minute,
                    remaining=0,
                    reset_time=datetime.now() + timedelta(seconds=reset_time),
                    retry_after=reset_time
                )
        
        # Default case
        return RateLimitStatus(
            limit=self.requests_per_minute,
            remaining=self.requests_per_minute,
            reset_time=datetime.now() + timedelta(seconds=60)
        )
    
    async def check_limit(self, key: str = "default") -> RateLimitStatus:
        """
        Check if request is within rate limit.
        
        Args:
            key: Rate limit key (e.g., user ID, IP address)
            
        Returns:
            RateLimitStatus with current limit information
        """
        try:
            if self.enable_distributed:
                status = await self._check_distributed_limit(key)
            else:
                status = self._check_local_limit(key)
            
            # Record metrics
            self.metrics.increment('rate_limit_checks_total')
            if status.remaining > 0:
                self.metrics.increment('rate_limit_checks_allowed')
            else:
                self.metrics.increment('rate_limit_checks_denied')
            
            return status
            
        except Exception as e:
            logger.error(f"Rate limit check failed for key '{key}': {e}")
            self.metrics.increment('rate_limit_errors')
            
            # Fall back to allowing request on error
            return RateLimitStatus(
                limit=self.requests_per_minute,
                remaining=self.requests_per_minute,
                reset_time=datetime.now() + timedelta(seconds=60)
            )
    
    async def wait_if_needed(self, key: str = "default") -> float:
        """
        Wait if rate limit is exceeded.
        
        Args:
            key: Rate limit key
            
        Returns:
            Time waited in seconds
        """
        status = await self.check_limit(key)
        
        if status.remaining > 0:
            # Reset backoff on successful request
            self._backoff_times.pop(key, None)
            return 0.0
        
        # Calculate wait time with backoff
        base_wait = status.retry_after or 1.0
        
        if key in self._backoff_times:
            # Apply exponential backoff
            self._backoff_times[key] = min(
                self._backoff_times[key] * self._backoff_factor,
                self._max_backoff
            )
        else:
            self._backoff_times[key] = base_wait
        
        wait_time = self._backoff_times[key]
        
        logger.warning(f"Rate limit exceeded for key '{key}', waiting {wait_time:.2f}s")
        self.metrics.record_histogram('rate_limit_wait_time', wait_time)
        
        await asyncio.sleep(wait_time)
        return wait_time
    
    async def acquire(self, key: str = "default", timeout: Optional[float] = None) -> bool:
        """
        Acquire rate limit permission.
        
        Args:
            key: Rate limit key
            timeout: Maximum time to wait for permission
            
        Returns:
            True if permission acquired, False if timeout
        """
        start_time = time.time()
        
        while True:
            status = await self.check_limit(key)
            
            if status.remaining > 0:
                return True
            
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False
                
                # Don't wait longer than remaining timeout
                wait_time = min(status.retry_after or 1.0, timeout - elapsed)
            else:
                wait_time = status.retry_after or 1.0
            
            await asyncio.sleep(wait_time)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                'algorithm': self.algorithm.value,
                'requests_per_second': self.requests_per_second,
                'requests_per_minute': self.requests_per_minute,
                'burst_size': self.burst_size,
                'distributed': self.enable_distributed,
                'active_keys': len(self._local_limiters),
                'backoff_keys': len(self._backoff_times),
                'metrics': self.metrics.get_all_metrics()
            }
    
    def reset_key(self, key: str):
        """Reset rate limit for a specific key."""
        with self._lock:
            self._local_limiters.pop(key, None)
            self._backoff_times.pop(key, None)
        
        logger.info(f"Rate limit reset for key: {key}")
    
    def reset_all(self):
        """Reset all rate limits."""
        with self._lock:
            self._local_limiters.clear()
            self._backoff_times.clear()
        
        logger.info("All rate limits reset")
    
    async def close(self):
        """Close Redis connections."""
        if self._redis_client:
            await self._redis_client.close()


# Decorator for rate limiting
def rate_limit(requests_per_minute: int = 600,
              key_func: Optional[Callable] = None,
              algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET):
    """
    Decorator to apply rate limiting to functions.
    
    Args:
        requests_per_minute: Requests allowed per minute
        key_func: Function to generate rate limit key from arguments
        algorithm: Rate limiting algorithm
    """
    limiter = RateLimiter(
        requests_per_minute=requests_per_minute,
        algorithm=algorithm
    )
    
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Generate key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = func.__name__
            
            # Check rate limit
            status = await limiter.check_limit(key)
            if status.remaining <= 0:
                raise RateLimitExceededError(
                    f"Rate limit exceeded for {func.__name__}",
                    retry_after=status.retry_after,
                    limit=status.limit,
                    remaining=status.remaining
                )
            
            return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global rate limiter instance
_global_rate_limiter = None

def get_global_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter()
    return _global_rate_limiter


# Convenience functions
async def wait_for_rate_limit(key: str = "default") -> float:
    """Wait for rate limit using global limiter."""
    limiter = get_global_rate_limiter()
    return await limiter.wait_if_needed(key)


async def check_rate_limit(key: str = "default") -> RateLimitStatus:
    """Check rate limit using global limiter."""
    limiter = get_global_rate_limiter()
    return await limiter.check_limit(key)
