"""
Spotify AI Agent - Fixture Monitoring
====================================

Enterprise monitoring and performance tracking system 
for fixture operations with real-time metrics and alerting.
"""

import asyncio
import logging
import time
import psutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable
from uuid import UUID, uuid4

from prometheus_client import Counter, Histogram, Gauge, Summary
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Metric type enumeration."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricValue:
    """Individual metric value with metadata."""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "labels": self.labels,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description
        }


@dataclass
class PerformanceMetrics:
    """Collection of performance metrics for fixture operations."""
    fixture_id: UUID
    fixture_name: str
    tenant_id: Optional[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    
    # Processing metrics
    records_processed: int = 0
    records_per_second: float = 0.0
    success_rate: float = 0.0
    error_count: int = 0
    
    # Resource metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    disk_io_mb: float = 0.0
    network_io_mb: float = 0.0
    
    # Database metrics
    db_connections_used: int = 0
    db_query_count: int = 0
    db_avg_query_time: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def calculate_derived_metrics(self) -> None:
        """Calculate derived metrics from base measurements."""
        if self.end_time and self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
            
            if self.duration > 0:
                self.records_per_second = self.records_processed / self.duration
        
        total_operations = self.records_processed + self.error_count
        if total_operations > 0:
            self.success_rate = self.records_processed / total_operations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "fixture_id": str(self.fixture_id),
            "fixture_name": self.fixture_name,
            "tenant_id": self.tenant_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "processing": {
                "records_processed": self.records_processed,
                "records_per_second": self.records_per_second,
                "success_rate": self.success_rate,
                "error_count": self.error_count
            },
            "resources": {
                "memory_usage_mb": self.memory_usage_mb,
                "cpu_usage_percent": self.cpu_usage_percent,
                "disk_io_mb": self.disk_io_mb,
                "network_io_mb": self.network_io_mb
            },
            "database": {
                "connections_used": self.db_connections_used,
                "query_count": self.db_query_count,
                "avg_query_time": self.db_avg_query_time
            },
            "custom_metrics": self.custom_metrics
        }


@dataclass
class Alert:
    """Alert notification for monitoring events."""
    id: UUID = field(default_factory=uuid4)
    level: AlertLevel = AlertLevel.INFO
    title: str = ""
    message: str = ""
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    fixture_id: Optional[UUID] = None
    tenant_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "fixture_id": str(self.fixture_id) if self.fixture_id else None,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at.isoformat(),
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by
        }


class MetricCollector(ABC):
    """Abstract base class for metric collectors."""
    
    @abstractmethod
    async def collect_metrics(self, context: Dict[str, Any]) -> List[MetricValue]:
        """Collect metrics and return list of metric values."""
        pass


class SystemMetricCollector(MetricCollector):
    """Collector for system-level metrics."""
    
    async def collect_metrics(self, context: Dict[str, Any]) -> List[MetricValue]:
        """Collect system metrics."""
        metrics = []
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.append(MetricValue(
            name="system_memory_usage_percent",
            value=memory.percent,
            metric_type=MetricType.GAUGE,
            description="System memory usage percentage"
        ))
        
        metrics.append(MetricValue(
            name="system_memory_available_mb",
            value=memory.available / 1024 / 1024,
            metric_type=MetricType.GAUGE,
            description="Available system memory in MB"
        ))
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.append(MetricValue(
            name="system_cpu_usage_percent",
            value=cpu_percent,
            metric_type=MetricType.GAUGE,
            description="System CPU usage percentage"
        ))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics.append(MetricValue(
            name="system_disk_usage_percent",
            value=(disk.used / disk.total) * 100,
            metric_type=MetricType.GAUGE,
            description="System disk usage percentage"
        ))
        
        return metrics


class DatabaseMetricCollector(MetricCollector):
    """Collector for database-related metrics."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def collect_metrics(self, context: Dict[str, Any]) -> List[MetricValue]:
        """Collect database metrics."""
        metrics = []
        
        try:
            # Connection pool metrics
            engine = self.session.get_bind()
            pool = engine.pool
            
            metrics.append(MetricValue(
                name="db_pool_size",
                value=pool.size(),
                metric_type=MetricType.GAUGE,
                description="Database connection pool size"
            ))
            
            metrics.append(MetricValue(
                name="db_pool_checked_out",
                value=pool.checkedout(),
                metric_type=MetricType.GAUGE,
                description="Database connections checked out"
            ))
            
            # Query performance metrics would be collected here
            # This would require instrumentation of SQL queries
            
        except Exception as e:
            logger.error(f"Error collecting database metrics: {e}")
        
        return metrics


class FixtureMetricCollector(MetricCollector):
    """Collector for fixture-specific metrics."""
    
    def __init__(self, fixture_id: UUID, fixture_name: str):
        self.fixture_id = fixture_id
        self.fixture_name = fixture_name
    
    async def collect_metrics(self, context: Dict[str, Any]) -> List[MetricValue]:
        """Collect fixture-specific metrics."""
        metrics = []
        
        labels = {
            "fixture_id": str(self.fixture_id),
            "fixture_name": self.fixture_name
        }
        
        tenant_id = context.get("tenant_id")
        if tenant_id:
            labels["tenant_id"] = tenant_id
        
        # Processing metrics
        records_processed = context.get("records_processed", 0)
        metrics.append(MetricValue(
            name="fixture_records_processed_total",
            value=records_processed,
            metric_type=MetricType.COUNTER,
            labels=labels,
            description="Total records processed by fixture"
        ))
        
        # Error metrics
        error_count = context.get("error_count", 0)
        metrics.append(MetricValue(
            name="fixture_errors_total",
            value=error_count,
            metric_type=MetricType.COUNTER,
            labels=labels,
            description="Total errors encountered by fixture"
        ))
        
        # Duration metrics
        duration = context.get("duration", 0)
        if duration > 0:
            metrics.append(MetricValue(
                name="fixture_duration_seconds",
                value=duration,
                metric_type=MetricType.HISTOGRAM,
                labels=labels,
                description="Fixture execution duration in seconds"
            ))
        
        return metrics


class PerformanceTracker:
    """
    Tracks performance metrics for fixture operations.
    
    Provides:
    - Real-time metric collection
    - Performance trend analysis
    - Resource usage monitoring
    - Alert generation
    """
    
    def __init__(self, fixture_id: UUID, fixture_name: str, tenant_id: Optional[str] = None):
        self.fixture_id = fixture_id
        self.fixture_name = fixture_name
        self.tenant_id = tenant_id
        
        self.metrics = PerformanceMetrics(
            fixture_id=fixture_id,
            fixture_name=fixture_name,
            tenant_id=tenant_id,
            start_time=datetime.now(timezone.utc)
        )
        
        self.collectors: List[MetricCollector] = []
        self.start_resources = self._get_current_resources()
        
        self.logger = logging.getLogger(f"{__name__}.PerformanceTracker")
    
    def start_tracking(self) -> None:
        """Start performance tracking."""
        self.metrics.start_time = datetime.now(timezone.utc)
        self.start_resources = self._get_current_resources()
        self.logger.info(f"Started performance tracking for fixture: {self.fixture_name}")
    
    def stop_tracking(self) -> PerformanceMetrics:
        """Stop tracking and calculate final metrics."""
        self.metrics.end_time = datetime.now(timezone.utc)
        
        # Calculate resource usage
        end_resources = self._get_current_resources()
        self.metrics.memory_usage_mb = end_resources.get("memory_mb", 0) - self.start_resources.get("memory_mb", 0)
        self.metrics.cpu_usage_percent = end_resources.get("cpu_percent", 0)
        
        # Calculate derived metrics
        self.metrics.calculate_derived_metrics()
        
        self.logger.info(
            f"Stopped performance tracking for fixture: {self.fixture_name} "
            f"(Duration: {self.metrics.duration:.2f}s, Records: {self.metrics.records_processed})"
        )
        
        return self.metrics
    
    def update_processing_metrics(
        self,
        records_processed: int,
        error_count: int = 0
    ) -> None:
        """Update processing metrics."""
        self.metrics.records_processed = records_processed
        self.metrics.error_count = error_count
    
    def update_database_metrics(
        self,
        connections_used: int,
        query_count: int,
        avg_query_time: float
    ) -> None:
        """Update database metrics."""
        self.metrics.db_connections_used = connections_used
        self.metrics.db_query_count = query_count
        self.metrics.db_avg_query_time = avg_query_time
    
    def add_custom_metric(self, name: str, value: float) -> None:
        """Add a custom metric."""
        self.metrics.custom_metrics[name] = value
    
    def _get_current_resources(self) -> Dict[str, float]:
        """Get current resource usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "memory_mb": memory_info.rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
                "disk_io_mb": sum(psutil.disk_io_counters()[:2]) / 1024 / 1024,
                "network_io_mb": sum(psutil.net_io_counters()[:2]) / 1024 / 1024
            }
        except Exception as e:
            self.logger.error(f"Error getting resource usage: {e}")
            return {}


class FixtureMonitor:
    """
    Main monitoring system for fixture operations.
    
    Provides:
    - Centralized metric collection
    - Alert management
    - Performance analysis
    - Health monitoring
    """
    
    def __init__(self, session: Optional[AsyncSession] = None):
        self.session = session
        self.collectors: List[MetricCollector] = []
        self.active_trackers: Dict[UUID, PerformanceTracker] = {}
        self.alerts: List[Alert] = []
        self.alert_thresholds: Dict[str, Dict[str, float]] = {}
        
        # Initialize default collectors
        self._initialize_collectors()
        self._initialize_alert_thresholds()
        
        self.logger = logging.getLogger(f"{__name__}.FixtureMonitor")
    
    def _initialize_collectors(self) -> None:
        """Initialize default metric collectors."""
        self.collectors = [
            SystemMetricCollector()
        ]
        
        if self.session:
            self.collectors.append(DatabaseMetricCollector(self.session))
    
    def _initialize_alert_thresholds(self) -> None:
        """Initialize default alert thresholds."""
        self.alert_thresholds = {
            "system_memory_usage_percent": {
                "warning": 80.0,
                "critical": 95.0
            },
            "system_cpu_usage_percent": {
                "warning": 80.0,
                "critical": 95.0
            },
            "system_disk_usage_percent": {
                "warning": 85.0,
                "critical": 95.0
            },
            "fixture_duration_seconds": {
                "warning": 300.0,  # 5 minutes
                "critical": 1800.0  # 30 minutes
            },
            "fixture_error_rate": {
                "warning": 0.05,  # 5%
                "critical": 0.20   # 20%
            }
        }
    
    def start_fixture_monitoring(
        self,
        fixture_id: UUID,
        fixture_name: str,
        tenant_id: Optional[str] = None
    ) -> PerformanceTracker:
        """Start monitoring a fixture operation."""
        tracker = PerformanceTracker(fixture_id, fixture_name, tenant_id)
        tracker.start_tracking()
        
        self.active_trackers[fixture_id] = tracker
        
        # Add fixture-specific collector
        fixture_collector = FixtureMetricCollector(fixture_id, fixture_name)
        self.collectors.append(fixture_collector)
        
        self.logger.info(f"Started monitoring fixture: {fixture_name} (ID: {fixture_id})")
        return tracker
    
    def stop_fixture_monitoring(self, fixture_id: UUID) -> Optional[PerformanceMetrics]:
        """Stop monitoring a fixture operation."""
        tracker = self.active_trackers.get(fixture_id)
        if not tracker:
            return None
        
        metrics = tracker.stop_tracking()
        del self.active_trackers[fixture_id]
        
        # Remove fixture-specific collector
        self.collectors = [
            c for c in self.collectors 
            if not (isinstance(c, FixtureMetricCollector) and c.fixture_id == fixture_id)
        ]
        
        # Check for alerts based on final metrics
        await self._check_fixture_alerts(metrics)
        
        self.logger.info(f"Stopped monitoring fixture: {tracker.fixture_name} (ID: {fixture_id})")
        return metrics
    
    async def collect_all_metrics(self) -> List[MetricValue]:
        """Collect metrics from all collectors."""
        all_metrics = []
        
        for collector in self.collectors:
            try:
                context = {
                    "active_fixtures": len(self.active_trackers),
                    "alert_count": len(self.alerts)
                }
                
                metrics = await collector.collect_metrics(context)
                all_metrics.extend(metrics)
                
            except Exception as e:
                self.logger.error(f"Error collecting metrics from {collector.__class__.__name__}: {e}")
        
        return all_metrics
    
    async def _check_fixture_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check fixture metrics against alert thresholds."""
        # Duration alert
        if metrics.duration:
            await self._check_threshold_alert(
                "fixture_duration_seconds",
                metrics.duration,
                fixture_id=metrics.fixture_id,
                tenant_id=metrics.tenant_id
            )
        
        # Error rate alert
        if metrics.records_processed > 0:
            error_rate = metrics.error_count / (metrics.records_processed + metrics.error_count)
            await self._check_threshold_alert(
                "fixture_error_rate",
                error_rate,
                fixture_id=metrics.fixture_id,
                tenant_id=metrics.tenant_id
            )
        
        # Performance alerts
        if metrics.records_per_second > 0 and metrics.records_per_second < 10:
            alert = Alert(
                level=AlertLevel.WARNING,
                title="Low Processing Performance",
                message=f"Fixture {metrics.fixture_name} processing at {metrics.records_per_second:.1f} records/sec",
                fixture_id=metrics.fixture_id,
                tenant_id=metrics.tenant_id
            )
            self.alerts.append(alert)
    
    async def _check_threshold_alert(
        self,
        metric_name: str,
        value: float,
        fixture_id: Optional[UUID] = None,
        tenant_id: Optional[str] = None
    ) -> None:
        """Check metric value against thresholds and create alerts."""
        thresholds = self.alert_thresholds.get(metric_name, {})
        
        alert_level = None
        threshold_value = None
        
        if "critical" in thresholds and value >= thresholds["critical"]:
            alert_level = AlertLevel.CRITICAL
            threshold_value = thresholds["critical"]
        elif "warning" in thresholds and value >= thresholds["warning"]:
            alert_level = AlertLevel.WARNING
            threshold_value = thresholds["warning"]
        
        if alert_level:
            alert = Alert(
                level=alert_level,
                title=f"{metric_name} threshold exceeded",
                message=f"{metric_name} value {value} exceeded {alert_level.value} threshold {threshold_value}",
                metric_name=metric_name,
                metric_value=value,
                threshold=threshold_value,
                fixture_id=fixture_id,
                tenant_id=tenant_id
            )
            self.alerts.append(alert)
            
            self.logger.warning(
                f"Alert created: {alert.title} - "
                f"Value: {value}, Threshold: {threshold_value}"
            )
    
    def get_active_alerts(self, acknowledged: bool = False) -> List[Alert]:
        """Get active alerts, optionally filtered by acknowledgment status."""
        return [alert for alert in self.alerts if alert.acknowledged == acknowledged]
    
    def acknowledge_alert(self, alert_id: UUID, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.now(timezone.utc)
                alert.acknowledged_by = acknowledged_by
                self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
        return False
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        # This would typically query stored metrics from a time-series database
        # For now, return summary of current active trackers
        
        active_fixtures = len(self.active_trackers)
        total_alerts = len(self.alerts)
        unacknowledged_alerts = len([a for a in self.alerts if not a.acknowledged])
        
        return {
            "period_hours": hours,
            "active_fixtures": active_fixtures,
            "total_alerts": total_alerts,
            "unacknowledged_alerts": unacknowledged_alerts,
            "alert_breakdown": {
                level.value: len([a for a in self.alerts if a.level == level])
                for level in AlertLevel
            }
        }
    
    def set_alert_threshold(
        self,
        metric_name: str,
        warning_threshold: Optional[float] = None,
        critical_threshold: Optional[float] = None
    ) -> None:
        """Set custom alert thresholds for a metric."""
        if metric_name not in self.alert_thresholds:
            self.alert_thresholds[metric_name] = {}
        
        if warning_threshold is not None:
            self.alert_thresholds[metric_name]["warning"] = warning_threshold
        
        if critical_threshold is not None:
            self.alert_thresholds[metric_name]["critical"] = critical_threshold
        
        self.logger.info(f"Updated alert thresholds for {metric_name}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {},
            "metrics": {}
        }
        
        try:
            # System health
            memory = psutil.virtual_memory()
            health_status["components"]["system"] = {
                "status": "healthy" if memory.percent < 90 else "degraded",
                "memory_usage": memory.percent,
                "cpu_usage": psutil.cpu_percent(interval=1)
            }
            
            # Database health
            if self.session:
                try:
                    await self.session.execute("SELECT 1")
                    health_status["components"]["database"] = {
                        "status": "healthy"
                    }
                except Exception as e:
                    health_status["components"]["database"] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    health_status["status"] = "degraded"
            
            # Active monitoring
            health_status["components"]["monitoring"] = {
                "status": "healthy",
                "active_trackers": len(self.active_trackers),
                "total_alerts": len(self.alerts),
                "unacknowledged_alerts": len([a for a in self.alerts if not a.acknowledged])
            }
            
            # Check if any critical alerts exist
            critical_alerts = [a for a in self.alerts if a.level == AlertLevel.CRITICAL and not a.acknowledged]
            if critical_alerts:
                health_status["status"] = "unhealthy"
                health_status["critical_alerts"] = len(critical_alerts)
        
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
            self.logger.error(f"Health check failed: {e}")
        
        return health_status
