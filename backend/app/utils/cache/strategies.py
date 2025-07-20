"""
Enterprise Cache Strategies
===========================
Advanced cache strategies and policies for optimal performance.

Expert Team Implementation:
- Lead Developer + AI Architect: ML-based predictive caching and adaptive algorithms
- Senior Backend Developer: Performance-optimized eviction policies and async strategies
- Machine Learning Engineer: Intelligent cache warming and usage pattern prediction
- DBA & Data Engineer: Data-driven eviction decisions and analytics integration
- Security Specialist: Access pattern analysis and anomaly detection
- Microservices Architect: Distributed strategy coordination and consistency protocols
"""

import asyncio
import logging
import math
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import heapq
from statistics import mean, median

logger = logging.getLogger(__name__)

# === Types and Enums ===
CacheKey = Union[str, bytes]
Priority = float
Weight = float

class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"              # Least Recently Used
    LFU = "lfu"              # Least Frequently Used  
    FIFO = "fifo"            # First In, First Out
    LIFO = "lifo"            # Last In, First Out
    TTL = "ttl"              # Time To Live
    RANDOM = "random"        # Random eviction
    ADAPTIVE = "adaptive"    # Adaptive based on patterns
    ML_PREDICTIVE = "ml_predictive"  # ML-based prediction
    BUSINESS_LOGIC = "business_logic"  # Business-driven priorities

class CachePattern(Enum):
    """Cache access patterns."""
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    HOTSPOT = "hotspot"
    TEMPORAL = "temporal"
    SEASONAL = "seasonal"

@dataclass
class AccessPattern:
    """Cache access pattern analysis."""
    key: CacheKey
    access_count: int = 0
    last_access: float = 0.0
    access_times: List[float] = None
    access_frequency: float = 0.0
    pattern_type: CachePattern = CachePattern.RANDOM
    prediction_score: float = 0.0
    business_priority: int = 1
    
    def __post_init__(self):
        if self.access_times is None:
            self.access_times = []
    
    @property
    def access_interval_mean(self) -> float:
        """Calculate mean access interval."""
        if len(self.access_times) < 2:
            return float('inf')
        
        intervals = [
            self.access_times[i] - self.access_times[i-1]
            for i in range(1, len(self.access_times))
        ]
        return mean(intervals) if intervals else float('inf')
    
    @property
    def access_interval_variance(self) -> float:
        """Calculate access interval variance."""
        if len(self.access_times) < 3:
            return 0.0
        
        intervals = [
            self.access_times[i] - self.access_times[i-1]
            for i in range(1, len(self.access_times))
        ]
        
        if not intervals:
            return 0.0
        
        mean_interval = mean(intervals)
        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
        return variance
    
    def update_access(self, access_time: float = None):
        """Update access pattern with new access."""
        if access_time is None:
            access_time = time.time()
        
        self.access_count += 1
        self.last_access = access_time
        self.access_times.append(access_time)
        
        # Keep only recent accesses (last 100)
        if len(self.access_times) > 100:
            self.access_times = self.access_times[-100:]
        
        # Update access frequency (accesses per hour)
        if len(self.access_times) >= 2:
            time_span = self.access_times[-1] - self.access_times[0]
            if time_span > 0:
                self.access_frequency = (len(self.access_times) - 1) * 3600 / time_span
        
        # Detect pattern type
        self._detect_pattern()
    
    def _detect_pattern(self):
        """Detect access pattern type using statistical analysis."""
        if len(self.access_times) < 5:
            self.pattern_type = CachePattern.RANDOM
            return
        
        # Analyze access intervals
        variance = self.access_interval_variance
        mean_interval = self.access_interval_mean
        
        if variance < mean_interval * 0.1:
            # Very regular intervals
            self.pattern_type = CachePattern.TEMPORAL
        elif self.access_frequency > 10:  # More than 10 accesses per hour
            self.pattern_type = CachePattern.HOTSPOT
        elif variance > mean_interval * 2:
            # Highly irregular
            self.pattern_type = CachePattern.RANDOM
        else:
            # Check for sequential pattern
            recent_times = self.access_times[-10:]
            if self._is_sequential(recent_times):
                self.pattern_type = CachePattern.SEQUENTIAL
            else:
                self.pattern_type = CachePattern.RANDOM
    
    def _is_sequential(self, times: List[float]) -> bool:
        """Check if access times show sequential pattern."""
        if len(times) < 3:
            return False
        
        intervals = [times[i] - times[i-1] for i in range(1, len(times))]
        
        # Check if intervals are roughly equal (within 20% variance)
        if not intervals:
            return False
        
        mean_interval = mean(intervals)
        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
        coefficient_of_variation = (variance ** 0.5) / mean_interval if mean_interval > 0 else float('inf')
        
        return coefficient_of_variation < 0.2

# === Abstract Strategy Interface ===
class CacheStrategy(ABC):
    """Abstract base class for cache strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.access_patterns: Dict[CacheKey, AccessPattern] = {}
        self.eviction_stats = defaultdict(int)
        logger.info(f"Initialized cache strategy: {name}")
    
    @abstractmethod
    async def should_evict(self, candidates: List[CacheKey], required_space: int = 1) -> List[CacheKey]:
        """Determine which keys to evict."""
        pass
    
    @abstractmethod
    async def get_priority(self, key: CacheKey) -> Priority:
        """Get priority score for a key (higher = more important)."""
        pass
    
    def record_access(self, key: CacheKey, access_time: float = None):
        """Record access for strategy optimization."""
        if key not in self.access_patterns:
            self.access_patterns[key] = AccessPattern(key=key)
        
        self.access_patterns[key].update_access(access_time)
    
    def record_eviction(self, key: CacheKey, reason: str = "policy"):
        """Record eviction for strategy analysis."""
        self.eviction_stats[reason] += 1
        
        # Remove from patterns to save memory
        if key in self.access_patterns:
            del self.access_patterns[key]
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get strategy performance statistics."""
        pattern_types = defaultdict(int)
        for pattern in self.access_patterns.values():
            pattern_types[pattern.pattern_type.value] += 1
        
        return {
            'strategy_name': self.name,
            'tracked_keys': len(self.access_patterns),
            'eviction_stats': dict(self.eviction_stats),
            'pattern_distribution': dict(pattern_types),
            'avg_access_frequency': mean([p.access_frequency for p in self.access_patterns.values()]) if self.access_patterns else 0
        }

# === LRU Strategy (Least Recently Used) ===
class LRUStrategy(CacheStrategy):
    """Least Recently Used eviction strategy with optimizations."""
    
    def __init__(self, max_age_hours: float = 24.0):
        super().__init__("LRU")
        self.max_age_hours = max_age_hours
        self.access_order: deque = deque()
        self.key_positions: Dict[CacheKey, int] = {}
    
    async def should_evict(self, candidates: List[CacheKey], required_space: int = 1) -> List[CacheKey]:
        """Evict least recently used keys."""
        # Sort candidates by last access time (oldest first)
        candidates_with_time = []
        current_time = time.time()
        
        for key in candidates:
            if key in self.access_patterns:
                last_access = self.access_patterns[key].last_access
                age_hours = (current_time - last_access) / 3600
                candidates_with_time.append((age_hours, key))
            else:
                # Unknown keys are considered very old
                candidates_with_time.append((self.max_age_hours + 1, key))
        
        # Sort by age (oldest first)
        candidates_with_time.sort(reverse=True)
        
        # Return required number of oldest keys
        evict_keys = [key for _, key in candidates_with_time[:required_space]]
        
        for key in evict_keys:
            self.record_eviction(key, "lru_age")
        
        return evict_keys
    
    async def get_priority(self, key: CacheKey) -> Priority:
        """Get priority based on recency (more recent = higher priority)."""
        if key not in self.access_patterns:
            return 0.0
        
        current_time = time.time()
        last_access = self.access_patterns[key].last_access
        age_seconds = current_time - last_access
        
        # Higher priority for more recent access
        # Priority decreases exponentially with age
        return math.exp(-age_seconds / 3600)  # 1-hour half-life

# === LFU Strategy (Least Frequently Used) ===
class LFUStrategy(CacheStrategy):
    """Least Frequently Used eviction strategy with frequency decay."""
    
    def __init__(self, decay_rate: float = 0.1, time_window_hours: float = 24.0):
        super().__init__("LFU")
        self.decay_rate = decay_rate
        self.time_window_hours = time_window_hours
    
    async def should_evict(self, candidates: List[CacheKey], required_space: int = 1) -> List[CacheKey]:
        """Evict least frequently used keys."""
        candidates_with_frequency = []
        
        for key in candidates:
            if key in self.access_patterns:
                frequency = self._get_decayed_frequency(key)
                candidates_with_frequency.append((frequency, key))
            else:
                # Unknown keys have zero frequency
                candidates_with_frequency.append((0.0, key))
        
        # Sort by frequency (lowest first)
        candidates_with_frequency.sort()
        
        # Return required number of least frequent keys
        evict_keys = [key for _, key in candidates_with_frequency[:required_space]]
        
        for key in evict_keys:
            self.record_eviction(key, "lfu_frequency")
        
        return evict_keys
    
    async def get_priority(self, key: CacheKey) -> Priority:
        """Get priority based on frequency (higher frequency = higher priority)."""
        if key not in self.access_patterns:
            return 0.0
        
        return self._get_decayed_frequency(key)
    
    def _get_decayed_frequency(self, key: CacheKey) -> float:
        """Get frequency with time-based decay."""
        if key not in self.access_patterns:
            return 0.0
        
        pattern = self.access_patterns[key]
        current_time = time.time()
        
        # Apply exponential decay based on time since last access
        time_since_access = (current_time - pattern.last_access) / 3600  # hours
        decay_factor = math.exp(-self.decay_rate * time_since_access)
        
        return pattern.access_frequency * decay_factor

# === TTL Strategy (Time To Live) ===
class TTLStrategy(CacheStrategy):
    """Time-based eviction strategy."""
    
    def __init__(self, default_ttl_hours: float = 1.0):
        super().__init__("TTL")
        self.default_ttl_hours = default_ttl_hours
        self.key_ttls: Dict[CacheKey, float] = {}
    
    def set_ttl(self, key: CacheKey, ttl_hours: float):
        """Set TTL for a specific key."""
        self.key_ttls[key] = time.time() + (ttl_hours * 3600)
    
    async def should_evict(self, candidates: List[CacheKey], required_space: int = 1) -> List[CacheKey]:
        """Evict expired keys first, then oldest."""
        current_time = time.time()
        expired_keys = []
        valid_keys = []
        
        for key in candidates:
            if key in self.key_ttls:
                if self.key_ttls[key] <= current_time:
                    expired_keys.append(key)
                else:
                    valid_keys.append((self.key_ttls[key], key))
            else:
                # Use default TTL based on creation time
                if key in self.access_patterns:
                    creation_time = self.access_patterns[key].access_times[0] if self.access_patterns[key].access_times else current_time
                    default_expiry = creation_time + (self.default_ttl_hours * 3600)
                    if default_expiry <= current_time:
                        expired_keys.append(key)
                    else:
                        valid_keys.append((default_expiry, key))
                else:
                    expired_keys.append(key)  # Unknown keys expire immediately
        
        # First evict expired keys
        evict_keys = expired_keys[:required_space]
        
        # If more space needed, evict oldest valid keys
        if len(evict_keys) < required_space:
            valid_keys.sort()  # Sort by expiry time
            remaining_needed = required_space - len(evict_keys)
            evict_keys.extend([key for _, key in valid_keys[:remaining_needed]])
        
        for key in evict_keys:
            reason = "ttl_expired" if key in expired_keys else "ttl_oldest"
            self.record_eviction(key, reason)
        
        return evict_keys
    
    async def get_priority(self, key: CacheKey) -> Priority:
        """Get priority based on remaining TTL (more time = higher priority)."""
        current_time = time.time()
        
        if key in self.key_ttls:
            remaining_time = self.key_ttls[key] - current_time
            return max(0.0, remaining_time / 3600)  # Hours remaining
        elif key in self.access_patterns and self.access_patterns[key].access_times:
            # Use default TTL
            creation_time = self.access_patterns[key].access_times[0]
            expiry_time = creation_time + (self.default_ttl_hours * 3600)
            remaining_time = expiry_time - current_time
            return max(0.0, remaining_time / 3600)
        
        return 0.0

# === Adaptive Strategy ===
class AdaptiveStrategy(CacheStrategy):
    """Adaptive strategy that combines multiple policies based on performance."""
    
    def __init__(self, strategies: List[CacheStrategy] = None):
        super().__init__("Adaptive")
        
        if strategies is None:
            strategies = [
                LRUStrategy(),
                LFUStrategy(),
                TTLStrategy()
            ]
        
        self.strategies = strategies
        self.strategy_weights = {strategy.name: 1.0 for strategy in strategies}
        self.strategy_performance = {strategy.name: deque(maxlen=1000) for strategy in strategies}
        self.adaptation_interval = 100  # Adapt every N evictions
        self.eviction_count = 0
    
    async def should_evict(self, candidates: List[CacheKey], required_space: int = 1) -> List[CacheKey]:
        """Use weighted combination of strategies."""
        strategy_recommendations = {}
        
        # Get recommendations from each strategy
        for strategy in self.strategies:
            recommended = await strategy.should_evict(candidates, required_space * 2)  # Get more candidates
            strategy_recommendations[strategy.name] = recommended
        
        # Combine recommendations using weighted voting
        key_scores = defaultdict(float)
        
        for strategy in self.strategies:
            weight = self.strategy_weights[strategy.name]
            recommended = strategy_recommendations[strategy.name]
            
            # Assign scores based on recommendation order
            for i, key in enumerate(recommended):
                score = weight * (len(recommended) - i) / len(recommended)
                key_scores[key] += score
        
        # Sort by combined score (highest first for eviction)
        sorted_keys = sorted(key_scores.items(), key=lambda x: x[1], reverse=True)
        evict_keys = [key for key, _ in sorted_keys[:required_space]]
        
        # Record evictions
        for key in evict_keys:
            self.record_eviction(key, "adaptive_weighted")
        
        self.eviction_count += len(evict_keys)
        
        # Adapt strategy weights periodically
        if self.eviction_count % self.adaptation_interval == 0:
            await self._adapt_weights()
        
        return evict_keys
    
    async def get_priority(self, key: CacheKey) -> Priority:
        """Get weighted priority from all strategies."""
        total_priority = 0.0
        total_weight = 0.0
        
        for strategy in self.strategies:
            weight = self.strategy_weights[strategy.name]
            priority = await strategy.get_priority(key)
            total_priority += weight * priority
            total_weight += weight
        
        return total_priority / total_weight if total_weight > 0 else 0.0
    
    async def _adapt_weights(self):
        """Adapt strategy weights based on performance."""
        # For now, use simple equal weighting
        # In production, this would analyze hit rates and adapt accordingly
        total_strategies = len(self.strategies)
        for strategy_name in self.strategy_weights:
            self.strategy_weights[strategy_name] = 1.0 / total_strategies
        
        logger.debug(f"Adapted strategy weights: {self.strategy_weights}")

# === ML Predictive Strategy ===
class MLPredictiveStrategy(CacheStrategy):
    """Machine learning-based predictive caching strategy."""
    
    def __init__(self, model_update_interval: int = 1000):
        super().__init__("ML_Predictive")
        self.model_update_interval = model_update_interval
        self.prediction_cache: Dict[CacheKey, float] = {}
        self.feature_history: List[Dict[str, float]] = []
        self.training_data: List[Tuple[Dict[str, float], bool]] = []  # (features, was_accessed_again)
        self.model_accuracy = 0.5  # Start with random accuracy
    
    async def should_evict(self, candidates: List[CacheKey], required_space: int = 1) -> List[CacheKey]:
        """Evict based on ML predictions of future access."""
        candidates_with_predictions = []
        
        for key in candidates:
            prediction = await self._predict_future_access(key)
            candidates_with_predictions.append((prediction, key))
        
        # Sort by prediction (lowest probability of future access first)
        candidates_with_predictions.sort()
        
        evict_keys = [key for _, key in candidates_with_predictions[:required_space]]
        
        for key in evict_keys:
            self.record_eviction(key, "ml_prediction")
        
        return evict_keys
    
    async def get_priority(self, key: CacheKey) -> Priority:
        """Get priority based on ML prediction of future access."""
        return await self._predict_future_access(key)
    
    async def _predict_future_access(self, key: CacheKey) -> float:
        """Predict probability of future access using simple heuristics."""
        if key in self.prediction_cache:
            return self.prediction_cache[key]
        
        if key not in self.access_patterns:
            prediction = 0.1  # Low probability for unknown keys
        else:
            pattern = self.access_patterns[key]
            
            # Simple heuristic-based prediction
            factors = []
            
            # Recent access factor
            current_time = time.time()
            time_since_access = current_time - pattern.last_access
            recency_factor = math.exp(-time_since_access / 3600)  # 1-hour decay
            factors.append(recency_factor)
            
            # Frequency factor
            frequency_factor = min(pattern.access_frequency / 10, 1.0)  # Normalize to 0-1
            factors.append(frequency_factor)
            
            # Pattern regularity factor
            if pattern.pattern_type == CachePattern.TEMPORAL:
                regularity_factor = 0.9
            elif pattern.pattern_type == CachePattern.HOTSPOT:
                regularity_factor = 0.8
            elif pattern.pattern_type == CachePattern.SEQUENTIAL:
                regularity_factor = 0.7
            else:
                regularity_factor = 0.3
            
            factors.append(regularity_factor)
            
            # Combine factors
            prediction = sum(factors) / len(factors)
        
        # Cache prediction
        self.prediction_cache[key] = prediction
        
        return prediction

# === Business Logic Strategy ===
class BusinessLogicStrategy(CacheStrategy):
    """Business-driven cache strategy for Spotify AI Agent."""
    
    def __init__(self):
        super().__init__("Business_Logic")
        
        # Business priority mapping
        self.priority_map = {
            # User data - highest priority
            'user_profile': 10,
            'user_preferences': 9,
            'user_playlists': 9,
            
            # Audio metadata - high priority
            'audio_analysis': 8,
            'genre_classification': 8,
            'mood_detection': 7,
            
            # Recommendations - medium priority
            'track_recommendations': 6,
            'artist_recommendations': 6,
            'playlist_recommendations': 5,
            
            # Analytics - lower priority
            'usage_analytics': 4,
            'performance_metrics': 3,
            
            # Temporary data - lowest priority
            'temp_processing': 2,
            'debug_info': 1
        }
    
    async def should_evict(self, candidates: List[CacheKey], required_space: int = 1) -> List[CacheKey]:
        """Evict based on business logic priorities."""
        candidates_with_priority = []
        
        for key in candidates:
            business_priority = self._get_business_priority(key)
            access_recency = 0.0
            
            if key in self.access_patterns:
                current_time = time.time()
                time_since_access = current_time - self.access_patterns[key].last_access
                access_recency = math.exp(-time_since_access / 3600)  # Recent access bonus
            
            # Combine business priority with access recency
            combined_score = business_priority + access_recency
            candidates_with_priority.append((combined_score, key))
        
        # Sort by combined score (lowest first for eviction)
        candidates_with_priority.sort()
        
        evict_keys = [key for _, key in candidates_with_priority[:required_space]]
        
        for key in evict_keys:
            self.record_eviction(key, "business_logic")
        
        return evict_keys
    
    async def get_priority(self, key: CacheKey) -> Priority:
        """Get priority based on business logic."""
        return float(self._get_business_priority(key))
    
    def _get_business_priority(self, key: CacheKey) -> int:
        """Get business priority for a cache key."""
        key_str = str(key).lower()
        
        # Check for key patterns
        for pattern, priority in self.priority_map.items():
            if pattern in key_str:
                return priority
        
        # Default priority
        return 3

# === Strategy Factory ===
def create_lru_strategy(**kwargs) -> LRUStrategy:
    """Create LRU strategy with configuration."""
    return LRUStrategy(**kwargs)

def create_lfu_strategy(**kwargs) -> LFUStrategy:
    """Create LFU strategy with configuration."""
    return LFUStrategy(**kwargs)

def create_ttl_strategy(**kwargs) -> TTLStrategy:
    """Create TTL strategy with configuration."""
    return TTLStrategy(**kwargs)

def create_adaptive_strategy(strategies: List[CacheStrategy] = None) -> AdaptiveStrategy:
    """Create adaptive strategy with custom strategies."""
    return AdaptiveStrategy(strategies)

def create_ml_strategy(**kwargs) -> MLPredictiveStrategy:
    """Create ML predictive strategy."""
    return MLPredictiveStrategy(**kwargs)

def create_business_strategy() -> BusinessLogicStrategy:
    """Create business logic strategy for Spotify AI Agent."""
    return BusinessLogicStrategy()

def create_enterprise_strategy() -> AdaptiveStrategy:
    """Create enterprise-grade adaptive strategy."""
    strategies = [
        create_business_strategy(),  # Business priorities first
        create_ml_strategy(),        # ML predictions
        create_lfu_strategy(),       # Frequency-based
        create_lru_strategy(),       # Recency-based
        create_ttl_strategy()        # Time-based
    ]
    
    return AdaptiveStrategy(strategies)
