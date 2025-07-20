"""
Enterprise Performance Optimization Schemas
Advanced performance monitoring, optimization and scaling configurations.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
from decimal import Decimal


class PerformanceTier(str, Enum):
    """Performance optimization tiers."""
    BASIC = "basic"
    STANDARD = "standard" 
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    ULTRA = "ultra"


class ScalingStrategy(str, Enum):
    """Auto-scaling strategies."""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    SCHEDULED = "scheduled"
    HYBRID = "hybrid"


class CacheStrategy(str, Enum):
    """Caching strategies."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    RANDOM = "random"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class OptimizationLevel(str, Enum):
    """Optimization levels."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"


class PerformanceMetricSchema(BaseModel):
    """Performance metric definition and thresholds."""
    
    metric_name: str = Field(..., description="Metric identifier")
    display_name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Metric description")
    unit: str = Field(..., description="Measurement unit")
    
    # Thresholds
    warning_threshold: float = Field(..., description="Warning threshold value")
    critical_threshold: float = Field(..., description="Critical threshold value")
    target_value: Optional[float] = Field(None, description="Optimal target value")
    
    # Collection settings
    collection_interval: int = Field(60, ge=1, le=3600, description="Seconds between collections")
    retention_days: int = Field(90, ge=1, le=2555, description="Data retention period")
    aggregation_window: int = Field(300, ge=60, le=3600, description="Aggregation window in seconds")
    
    # Analysis
    trend_analysis: bool = True
    anomaly_detection: bool = True
    forecasting_enabled: bool = True
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ResourceLimitsSchema(BaseModel):
    """Resource limits and quotas configuration."""
    
    # CPU limits
    cpu_cores_min: float = Field(0.1, ge=0.1, le=128.0)
    cpu_cores_max: float = Field(4.0, ge=0.1, le=128.0)
    cpu_utilization_target: float = Field(70.0, ge=10.0, le=95.0)
    
    # Memory limits
    memory_mb_min: int = Field(128, ge=128, le=131072)
    memory_mb_max: int = Field(4096, ge=128, le=131072)
    memory_utilization_target: float = Field(80.0, ge=10.0, le=95.0)
    
    # Storage limits
    storage_gb_quota: int = Field(100, ge=1, le=10240)
    iops_limit: int = Field(1000, ge=100, le=50000)
    bandwidth_mbps: int = Field(100, ge=10, le=10000)
    
    # Network limits
    requests_per_second: int = Field(1000, ge=1, le=100000)
    concurrent_connections: int = Field(1000, ge=10, le=50000)
    data_transfer_gb: int = Field(1000, ge=1, le=102400)
    
    @validator('cpu_cores_max')
    def validate_cpu_max(cls, v, values):
        if 'cpu_cores_min' in values and v < values['cpu_cores_min']:
            raise ValueError('CPU max must be greater than min')
        return v
        
    @validator('memory_mb_max')
    def validate_memory_max(cls, v, values):
        if 'memory_mb_min' in values and v < values['memory_mb_min']:
            raise ValueError('Memory max must be greater than min')
        return v


class AutoScalingConfigSchema(BaseModel):
    """Advanced auto-scaling configuration."""
    
    enabled: bool = True
    strategy: ScalingStrategy = ScalingStrategy.HYBRID
    
    # Scaling triggers
    cpu_scale_up_threshold: float = Field(80.0, ge=50.0, le=95.0)
    cpu_scale_down_threshold: float = Field(30.0, ge=5.0, le=50.0)
    memory_scale_up_threshold: float = Field(85.0, ge=50.0, le=95.0)
    memory_scale_down_threshold: float = Field(40.0, ge=5.0, le=50.0)
    
    # Response time triggers
    response_time_threshold_ms: int = Field(500, ge=100, le=10000)
    queue_length_threshold: int = Field(100, ge=10, le=10000)
    error_rate_threshold: float = Field(5.0, ge=0.1, le=50.0)
    
    # Scaling parameters
    scale_up_cooldown_seconds: int = Field(300, ge=60, le=3600)
    scale_down_cooldown_seconds: int = Field(600, ge=300, le=7200)
    min_instances: int = Field(2, ge=1, le=100)
    max_instances: int = Field(50, ge=1, le=1000)
    scale_increment: int = Field(2, ge=1, le=10)
    
    # Predictive scaling
    predictive_enabled: bool = True
    prediction_window_hours: int = Field(24, ge=1, le=168)
    learning_period_days: int = Field(14, ge=7, le=90)
    prediction_confidence: float = Field(0.8, ge=0.5, le=0.99)
    
    # Scheduled scaling
    scheduled_scaling: List[Dict[str, Any]] = Field(default_factory=list)
    
    @validator('max_instances')
    def validate_max_instances(cls, v, values):
        if 'min_instances' in values and v < values['min_instances']:
            raise ValueError('Max instances must be greater than min instances')
        return v


class CacheConfigSchema(BaseModel):
    """Advanced caching configuration."""
    
    enabled: bool = True
    strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    
    # Cache levels
    l1_cache_enabled: bool = True  # Application cache
    l2_cache_enabled: bool = True  # Redis/Memcached
    l3_cache_enabled: bool = True  # CDN cache
    
    # Size limits
    l1_cache_size_mb: int = Field(256, ge=64, le=4096)
    l2_cache_size_mb: int = Field(2048, ge=256, le=16384)
    l3_cache_size_gb: int = Field(100, ge=10, le=1024)
    
    # TTL settings
    default_ttl_seconds: int = Field(3600, ge=60, le=86400)
    max_ttl_seconds: int = Field(86400, ge=3600, le=604800)
    min_ttl_seconds: int = Field(60, ge=1, le=3600)
    
    # Eviction policies
    l1_eviction_policy: str = "lru"
    l2_eviction_policy: str = "allkeys-lru"
    cache_hit_ratio_target: float = Field(0.85, ge=0.5, le=0.99)
    
    # Warming strategies
    cache_warming_enabled: bool = True
    preload_popular_content: bool = True
    background_refresh: bool = True
    
    # Monitoring
    hit_ratio_monitoring: bool = True
    latency_monitoring: bool = True
    memory_usage_monitoring: bool = True


class DatabaseOptimizationSchema(BaseModel):
    """Database performance optimization configuration."""
    
    # Connection pooling
    connection_pool_size: int = Field(20, ge=5, le=100)
    max_overflow: int = Field(10, ge=0, le=50)
    pool_timeout_seconds: int = Field(30, ge=5, le=300)
    pool_recycle_seconds: int = Field(3600, ge=300, le=86400)
    
    # Query optimization
    query_timeout_seconds: int = Field(30, ge=1, le=300)
    slow_query_threshold_seconds: float = Field(1.0, ge=0.1, le=60.0)
    explain_analyze_enabled: bool = True
    query_plan_caching: bool = True
    
    # Indexing strategy
    auto_index_creation: bool = True
    index_usage_monitoring: bool = True
    unused_index_cleanup: bool = True
    index_rebuild_threshold: float = Field(0.3, ge=0.1, le=0.8)
    
    # Partitioning
    table_partitioning_enabled: bool = True
    partition_strategy: str = "time_based"
    partition_size_gb: int = Field(10, ge=1, le=100)
    
    # Backup and maintenance
    auto_vacuum_enabled: bool = True
    statistics_update_frequency: str = "daily"
    backup_optimization: bool = True
    
    # Read replicas
    read_replica_count: int = Field(2, ge=0, le=10)
    read_write_split: bool = True
    replica_lag_threshold_seconds: int = Field(5, ge=1, le=60)


class OptimizationConfigSchema(BaseModel):
    """Main performance optimization configuration."""
    
    tenant_id: str = Field(..., description="Tenant identifier")
    environment: str = Field(..., regex=r"^(dev|staging|prod)$")
    performance_tier: PerformanceTier = PerformanceTier.STANDARD
    optimization_level: OptimizationLevel = OptimizationLevel.MODERATE
    
    # Core components
    resource_limits: ResourceLimitsSchema
    auto_scaling: AutoScalingConfigSchema
    caching: CacheConfigSchema
    database: DatabaseOptimizationSchema
    
    # Application-level optimizations
    compression_enabled: bool = True
    minification_enabled: bool = True
    bundling_enabled: bool = True
    lazy_loading_enabled: bool = True
    
    # CDN and networking
    cdn_enabled: bool = True
    gzip_compression: bool = True
    brotli_compression: bool = True
    http2_enabled: bool = True
    http3_enabled: bool = False
    
    # Monitoring and alerting
    performance_monitoring: bool = True
    real_time_alerts: bool = True
    performance_budgets: Dict[str, float] = Field(default_factory=lambda: {
        "page_load_time": 3.0,
        "api_response_time": 0.5,
        "database_query_time": 0.1,
        "cache_hit_ratio": 0.85
    })
    
    # Cost optimization
    cost_optimization_enabled: bool = True
    resource_rightsizing: bool = True
    spot_instance_usage: bool = False
    reserved_instance_planning: bool = True
    
    # Metrics to track
    tracked_metrics: List[PerformanceMetricSchema] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "tenant_id": "tenant_001",
                "environment": "prod",
                "performance_tier": "premium",
                "optimization_level": "aggressive",
                "resource_limits": {
                    "cpu_cores_min": 2.0,
                    "cpu_cores_max": 16.0,
                    "memory_mb_min": 4096,
                    "memory_mb_max": 32768
                }
            }
        }


class PerformanceReportSchema(BaseModel):
    """Performance analysis and reporting."""
    
    report_id: str = Field(..., description="Unique report identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    report_type: str = Field(..., regex=r"^(daily|weekly|monthly|custom)$")
    
    # Time range
    start_date: datetime
    end_date: datetime
    
    # Performance summary
    overall_score: float = Field(..., ge=0.0, le=100.0)
    availability_percentage: float = Field(..., ge=0.0, le=100.0)
    avg_response_time_ms: float = Field(..., ge=0.0)
    error_rate_percentage: float = Field(..., ge=0.0, le=100.0)
    
    # Resource utilization
    avg_cpu_utilization: float = Field(..., ge=0.0, le=100.0)
    avg_memory_utilization: float = Field(..., ge=0.0, le=100.0)
    avg_storage_utilization: float = Field(..., ge=0.0, le=100.0)
    avg_network_utilization: float = Field(..., ge=0.0, le=100.0)
    
    # Cost analysis
    total_cost: Decimal = Field(..., ge=0)
    cost_per_request: Decimal = Field(..., ge=0)
    cost_optimization_savings: Decimal = Field(0, ge=0)
    
    # Recommendations
    optimization_recommendations: List[str] = Field(default_factory=list)
    scaling_recommendations: List[str] = Field(default_factory=list)
    cost_reduction_recommendations: List[str] = Field(default_factory=list)
    
    # Trends
    performance_trend: str = Field(..., regex=r"^(improving|stable|degrading)$")
    capacity_forecast: Dict[str, Any] = Field(default_factory=dict)
    
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            Decimal: str
        }
