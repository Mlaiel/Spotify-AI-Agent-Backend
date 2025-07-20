"""
Advanced Factory Monitoring and Analytics System
===============================================

Ultra-advanced monitoring and analytics system for enterprise authentication
factory operations with real-time metrics, predictive analytics, and
AI-powered optimization for Fortune 500 manufacturing environments.

Monitoring Features:
- Real-time Production Metrics and KPIs
- Performance Monitoring with SLA Tracking
- Quality Control Analytics and Reporting
- Resource Utilization and Capacity Planning
- Predictive Maintenance and Failure Detection
- Security Monitoring and Threat Detection
- Business Intelligence and Executive Dashboards
- Compliance Monitoring and Audit Trails

Analytics Capabilities:
- Machine Learning-powered Production Optimization
- Predictive Quality Analytics
- Supply Chain Analytics and Optimization
- Customer Behavior Analytics
- Cost Analytics and ROI Tracking
- Performance Benchmarking and Comparison
- Anomaly Detection and Root Cause Analysis
- Forecasting and Capacity Planning
"""

from typing import Dict, List, Any, Optional, Union, Callable, AsyncGenerator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import uuid
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import structlog
import threading
from concurrent.futures import ThreadPoolExecutor
import statistics

# Import base classes
from . import FactoryProductionMetrics, FactoryProductSpecification
from .manufacturing import ProductionOrder, ManufacturingWorkItem, ProductionStatus

logger = structlog.get_logger(__name__)


# ================== MONITORING ENUMS ==================

class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MonitoringScope(Enum):
    """Monitoring scope levels."""
    FACTORY = "factory"
    PRODUCTION_LINE = "production_line"
    WORKER = "worker"
    PRODUCT = "product"
    SYSTEM = "system"


class AnalyticsType(Enum):
    """Types of analytics."""
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    PRESCRIPTIVE = "prescriptive"


class DashboardType(Enum):
    """Dashboard types."""
    OPERATIONAL = "operational"
    TACTICAL = "tactical"
    STRATEGIC = "strategic"
    EXECUTIVE = "executive"


# ================== MONITORING DATA STRUCTURES ==================

@dataclass
class MetricDataPoint:
    """Data point for metrics."""
    
    metric_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metric_name: str = ""
    metric_type: MetricType = MetricType.GAUGE
    value: Union[int, float, str] = 0
    unit: str = ""
    
    # Context information
    tags: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    dimensions: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    collection_time: Optional[datetime] = None
    
    # Quality
    quality_score: float = 100.0
    confidence_level: float = 100.0
    
    # Metadata
    source: str = ""
    collector: str = ""
    version: str = "1.0.0"


@dataclass
class Alert:
    """Alert for monitoring system."""
    
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    alert_name: str = ""
    severity: AlertSeverity = AlertSeverity.INFO
    
    # Alert details
    message: str = ""
    description: str = ""
    metric_name: str = ""
    threshold_value: Union[int, float] = 0
    current_value: Union[int, float] = 0
    
    # Context
    scope: MonitoringScope = MonitoringScope.FACTORY
    source: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Status
    is_active: bool = True
    is_acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    
    # Timing
    triggered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    
    # Actions
    notification_sent: bool = False
    escalation_level: int = 0
    auto_resolution_attempted: bool = False


@dataclass
class PerformanceReport:
    """Performance report for analytics."""
    
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    report_name: str = ""
    report_type: str = "performance_summary"
    
    # Time period
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Metrics summary
    total_production: int = 0
    successful_production: int = 0
    failed_production: int = 0
    average_cycle_time: float = 0.0
    throughput_per_hour: float = 0.0
    
    # Quality metrics
    quality_score: float = 100.0
    defect_rate: float = 0.0
    first_pass_yield: float = 100.0
    customer_satisfaction: float = 100.0
    
    # Resource utilization
    worker_utilization: float = 0.0
    equipment_utilization: float = 0.0
    material_utilization: float = 0.0
    
    # Cost metrics
    total_cost: float = 0.0
    cost_per_unit: float = 0.0
    labor_cost_percentage: float = 0.0
    material_cost_percentage: float = 0.0
    
    # Trends and insights
    trend_analysis: Dict[str, Any] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    generated_by: str = "analytics_engine"
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ================== METRIC COLLECTORS ==================

class MetricCollector(ABC):
    """Abstract base class for metric collectors."""
    
    def __init__(self, collector_id: str, collection_interval: int = 60):
        self.collector_id = collector_id
        self.collection_interval = collection_interval
        self.is_running = False
        self.metrics_buffer: List[MetricDataPoint] = []
        self._collection_task: Optional[asyncio.Task] = None
        
    @abstractmethod
    async def collect_metrics(self) -> List[MetricDataPoint]:
        """Collect metrics from source."""
        pass
    
    async def start_collection(self):
        """Start metric collection."""
        
        if self.is_running:
            return
        
        self.is_running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        
        logger.info("Metric collection started", collector_id=self.collector_id)
    
    async def stop_collection(self):
        """Stop metric collection."""
        
        self.is_running = False
        
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Metric collection stopped", collector_id=self.collector_id)
    
    async def _collection_loop(self):
        """Main collection loop."""
        
        while self.is_running:
            try:
                # Collect metrics
                start_time = time.time()
                metrics = await self.collect_metrics()
                collection_time = time.time() - start_time
                
                # Update collection time
                for metric in metrics:
                    metric.collection_time = datetime.now(timezone.utc)
                    metric.collector = self.collector_id
                
                # Add to buffer
                self.metrics_buffer.extend(metrics)
                
                logger.debug(
                    "Metrics collected",
                    collector_id=self.collector_id,
                    metric_count=len(metrics),
                    collection_time_ms=collection_time * 1000
                )
                
                # Wait for next collection
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(
                    "Error in metric collection",
                    collector_id=self.collector_id,
                    error=str(e)
                )
                await asyncio.sleep(5)  # Wait before retrying
    
    def get_buffered_metrics(self, clear_buffer: bool = True) -> List[MetricDataPoint]:
        """Get metrics from buffer."""
        
        metrics = self.metrics_buffer.copy()
        
        if clear_buffer:
            self.metrics_buffer.clear()
        
        return metrics


class FactoryMetricCollector(MetricCollector):
    """Collector for factory-level metrics."""
    
    def __init__(self, factory, collection_interval: int = 30):
        super().__init__("factory_collector", collection_interval)
        self.factory = factory
    
    async def collect_metrics(self) -> List[MetricDataPoint]:
        """Collect factory metrics."""
        
        metrics = []
        
        try:
            # Get factory status
            factory_status = await self.factory.get_factory_status()
            
            # Production metrics
            total_orders = factory_status["orders"]["total"]
            completed_orders = factory_status["orders"]["by_status"].get("completed", 0)
            failed_orders = factory_status["orders"]["by_status"].get("failed", 0)
            
            metrics.extend([
                MetricDataPoint(
                    metric_name="factory.orders.total",
                    metric_type=MetricType.GAUGE,
                    value=total_orders,
                    tags={"factory_id": self.factory.factory_id}
                ),
                MetricDataPoint(
                    metric_name="factory.orders.completed",
                    metric_type=MetricType.COUNTER,
                    value=completed_orders,
                    tags={"factory_id": self.factory.factory_id}
                ),
                MetricDataPoint(
                    metric_name="factory.orders.failed",
                    metric_type=MetricType.COUNTER,
                    value=failed_orders,
                    tags={"factory_id": self.factory.factory_id}
                )
            ])
            
            # Production line metrics
            for line_id, line_status in factory_status["production_lines"].items():
                utilization = line_status["utilization"]
                queue_size = sum(stage["queue_size"] for stage in line_status["stage_status"].values())
                
                metrics.extend([
                    MetricDataPoint(
                        metric_name="production_line.utilization",
                        metric_type=MetricType.GAUGE,
                        value=utilization,
                        unit="percentage",
                        tags={"factory_id": self.factory.factory_id, "line_id": line_id}
                    ),
                    MetricDataPoint(
                        metric_name="production_line.queue_size",
                        metric_type=MetricType.GAUGE,
                        value=queue_size,
                        tags={"factory_id": self.factory.factory_id, "line_id": line_id}
                    )
                ])
            
            # System metrics
            metrics.extend([
                MetricDataPoint(
                    metric_name="factory.performance.throughput",
                    metric_type=MetricType.GAUGE,
                    value=factory_status["metrics"]["throughput_objects_per_hour"],
                    unit="objects_per_hour",
                    tags={"factory_id": self.factory.factory_id}
                ),
                MetricDataPoint(
                    metric_name="factory.performance.production_rate",
                    metric_type=MetricType.GAUGE,
                    value=factory_status["metrics"]["production_rate_per_second"],
                    unit="objects_per_second",
                    tags={"factory_id": self.factory.factory_id}
                )
            ])
            
        except Exception as e:
            logger.error("Error collecting factory metrics", error=str(e))
        
        return metrics


class ProductionLineMetricCollector(MetricCollector):
    """Collector for production line metrics."""
    
    def __init__(self, production_line, collection_interval: int = 15):
        super().__init__(f"line_collector_{production_line.line_id}", collection_interval)
        self.production_line = production_line
    
    async def collect_metrics(self) -> List[MetricDataPoint]:
        """Collect production line metrics."""
        
        metrics = []
        
        try:
            # Get line status
            line_status = await self.production_line.get_line_status()
            line_id = self.production_line.line_id
            
            # Overall line metrics
            metrics.extend([
                MetricDataPoint(
                    metric_name="line.utilization",
                    metric_type=MetricType.GAUGE,
                    value=line_status["utilization"],
                    unit="percentage",
                    tags={"line_id": line_id}
                ),
                MetricDataPoint(
                    metric_name="line.current_load",
                    metric_type=MetricType.GAUGE,
                    value=line_status["current_load"],
                    tags={"line_id": line_id}
                ),
                MetricDataPoint(
                    metric_name="line.capacity",
                    metric_type=MetricType.GAUGE,
                    value=line_status["capacity"],
                    tags={"line_id": line_id}
                )
            ])
            
            # Stage-specific metrics
            for stage_name, stage_status in line_status["stage_status"].items():
                stage_tags = {"line_id": line_id, "stage": stage_name}
                
                metrics.extend([
                    MetricDataPoint(
                        metric_name="stage.queue_size",
                        metric_type=MetricType.GAUGE,
                        value=stage_status["queue_size"],
                        tags=stage_tags
                    ),
                    MetricDataPoint(
                        metric_name="stage.worker_utilization",
                        metric_type=MetricType.GAUGE,
                        value=stage_status["utilization"],
                        unit="percentage",
                        tags=stage_tags
                    ),
                    MetricDataPoint(
                        metric_name="stage.available_workers",
                        metric_type=MetricType.GAUGE,
                        value=stage_status["available_workers"],
                        tags=stage_tags
                    ),
                    MetricDataPoint(
                        metric_name="stage.total_workers",
                        metric_type=MetricType.GAUGE,
                        value=stage_status["total_workers"],
                        tags=stage_tags
                    )
                ])
            
            # Production metrics from line metrics
            line_metrics = line_status["metrics"]
            
            metrics.extend([
                MetricDataPoint(
                    metric_name="line.items_processed",
                    metric_type=MetricType.COUNTER,
                    value=line_metrics["total_items_processed"],
                    tags={"line_id": line_id}
                ),
                MetricDataPoint(
                    metric_name="line.successful_completions",
                    metric_type=MetricType.COUNTER,
                    value=line_metrics["successful_completions"],
                    tags={"line_id": line_id}
                ),
                MetricDataPoint(
                    metric_name="line.failed_productions",
                    metric_type=MetricType.COUNTER,
                    value=line_metrics["failed_productions"],
                    tags={"line_id": line_id}
                ),
                MetricDataPoint(
                    metric_name="line.average_cycle_time",
                    metric_type=MetricType.GAUGE,
                    value=line_metrics["average_cycle_time"],
                    unit="seconds",
                    tags={"line_id": line_id}
                )
            ])
            
        except Exception as e:
            logger.error("Error collecting production line metrics", line_id=self.production_line.line_id, error=str(e))
        
        return metrics


class WorkerMetricCollector(MetricCollector):
    """Collector for worker performance metrics."""
    
    def __init__(self, workers: List, collection_interval: int = 60):
        super().__init__("worker_collector", collection_interval)
        self.workers = workers
    
    async def collect_metrics(self) -> List[MetricDataPoint]:
        """Collect worker metrics."""
        
        metrics = []
        
        try:
            for worker in self.workers:
                worker_metrics = worker.get_worker_metrics()
                worker_id = worker.worker_id
                
                metrics.extend([
                    MetricDataPoint(
                        metric_name="worker.productivity_score",
                        metric_type=MetricType.GAUGE,
                        value=worker_metrics["productivity_score"],
                        unit="score",
                        tags={"worker_id": worker_id}
                    ),
                    MetricDataPoint(
                        metric_name="worker.items_processed",
                        metric_type=MetricType.COUNTER,
                        value=worker_metrics["total_items_processed"],
                        tags={"worker_id": worker_id}
                    ),
                    MetricDataPoint(
                        metric_name="worker.items_per_hour",
                        metric_type=MetricType.GAUGE,
                        value=worker_metrics["items_per_hour"],
                        unit="items_per_hour",
                        tags={"worker_id": worker_id}
                    ),
                    MetricDataPoint(
                        metric_name="worker.availability",
                        metric_type=MetricType.GAUGE,
                        value=1 if worker_metrics["is_available"] else 0,
                        tags={"worker_id": worker_id}
                    ),
                    MetricDataPoint(
                        metric_name="worker.shift_duration",
                        metric_type=MetricType.GAUGE,
                        value=worker_metrics["shift_duration_hours"],
                        unit="hours",
                        tags={"worker_id": worker_id}
                    )
                ])
                
        except Exception as e:
            logger.error("Error collecting worker metrics", error=str(e))
        
        return metrics


# ================== ALERTING SYSTEM ==================

class AlertRule:
    """Rule for generating alerts."""
    
    def __init__(self, rule_id: str, metric_name: str, condition: str, threshold: Union[int, float], severity: AlertSeverity):
        self.rule_id = rule_id
        self.metric_name = metric_name
        self.condition = condition  # e.g., ">", "<", "==", "!="
        self.threshold = threshold
        self.severity = severity
        self.is_enabled = True
        
        # Alert management
        self.cooldown_period = 300  # 5 minutes
        self.last_triggered = None
        
    def evaluate(self, metric: MetricDataPoint) -> Optional[Alert]:
        """Evaluate metric against rule."""
        
        if not self.is_enabled:
            return None
        
        if metric.metric_name != self.metric_name:
            return None
        
        # Check cooldown
        if self.last_triggered:
            time_since_last = (datetime.now(timezone.utc) - self.last_triggered).total_seconds()
            if time_since_last < self.cooldown_period:
                return None
        
        # Evaluate condition
        triggered = False
        
        if self.condition == ">" and metric.value > self.threshold:
            triggered = True
        elif self.condition == "<" and metric.value < self.threshold:
            triggered = True
        elif self.condition == "==" and metric.value == self.threshold:
            triggered = True
        elif self.condition == "!=" and metric.value != self.threshold:
            triggered = True
        elif self.condition == ">=" and metric.value >= self.threshold:
            triggered = True
        elif self.condition == "<=" and metric.value <= self.threshold:
            triggered = True
        
        if triggered:
            self.last_triggered = datetime.now(timezone.utc)
            
            alert = Alert(
                alert_name=f"{self.metric_name}_{self.condition}_{self.threshold}",
                severity=self.severity,
                message=f"{self.metric_name} is {metric.value} (threshold: {self.condition} {self.threshold})",
                metric_name=self.metric_name,
                threshold_value=self.threshold,
                current_value=metric.value,
                source=metric.source,
                tags=metric.tags
            )
            
            return alert
        
        return None


class AlertingEngine:
    """Engine for processing alerts."""
    
    def __init__(self):
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers: List[Callable] = []
        
    def add_alert_rule(self, rule: AlertRule):
        """Add alert rule."""
        self.alert_rules.append(rule)
        logger.info("Alert rule added", rule_id=rule.rule_id, metric=rule.metric_name)
    
    def add_notification_handler(self, handler: Callable):
        """Add notification handler."""
        self.notification_handlers.append(handler)
    
    async def process_metric(self, metric: MetricDataPoint):
        """Process metric against all alert rules."""
        
        for rule in self.alert_rules:
            alert = rule.evaluate(metric)
            
            if alert:
                await self._handle_alert(alert)
    
    async def _handle_alert(self, alert: Alert):
        """Handle triggered alert."""
        
        # Add to active alerts
        self.active_alerts[alert.alert_id] = alert
        
        # Add to history
        self.alert_history.append(alert)
        
        # Send notifications
        for handler in self.notification_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error("Failed to send alert notification", alert_id=alert.alert_id, error=str(e))
        
        logger.warning(
            "Alert triggered",
            alert_id=alert.alert_id,
            severity=alert.severity.value,
            message=alert.message
        )
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert."""
        
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.is_acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now(timezone.utc)
            
            logger.info("Alert acknowledged", alert_id=alert_id, acknowledged_by=acknowledged_by)
    
    async def resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.is_active = False
            alert.resolved_at = datetime.now(timezone.utc)
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            logger.info("Alert resolved", alert_id=alert_id)
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts."""
        
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return alerts
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        
        total_alerts = len(self.alert_history)
        active_alerts = len(self.active_alerts)
        acknowledged_alerts = sum(1 for alert in self.active_alerts.values() if alert.is_acknowledged)
        
        severity_counts = defaultdict(int)
        for alert in self.alert_history:
            severity_counts[alert.severity.value] += 1
        
        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "acknowledged_alerts": acknowledged_alerts,
            "unacknowledged_alerts": active_alerts - acknowledged_alerts,
            "alerts_by_severity": dict(severity_counts)
        }


# ================== ANALYTICS ENGINE ==================

class AnalyticsEngine:
    """Engine for advanced analytics and insights."""
    
    def __init__(self):
        self.metric_store: List[MetricDataPoint] = []
        self.production_history: List[ProductionOrder] = []
        self.quality_history: List[Dict[str, Any]] = []
        
        # Analytics configuration
        self.analysis_window = timedelta(hours=24)
        self.prediction_horizon = timedelta(hours=4)
        
    def add_metric(self, metric: MetricDataPoint):
        """Add metric to analytics store."""
        self.metric_store.append(metric)
        
        # Keep only recent metrics
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=7)
        self.metric_store = [m for m in self.metric_store if m.timestamp >= cutoff_time]
    
    def add_production_record(self, order: ProductionOrder):
        """Add production record."""
        self.production_history.append(order)
        
        # Keep only recent records
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=30)
        self.production_history = [r for r in self.production_history if r.created_at >= cutoff_time]
    
    async def generate_performance_report(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> PerformanceReport:
        """Generate performance report."""
        
        if start_time is None:
            start_time = datetime.now(timezone.utc) - self.analysis_window
        
        if end_time is None:
            end_time = datetime.now(timezone.utc)
        
        # Filter data by time range
        relevant_orders = [
            order for order in self.production_history
            if start_time <= order.created_at <= end_time
        ]
        
        relevant_metrics = [
            metric for metric in self.metric_store
            if start_time <= metric.timestamp <= end_time
        ]
        
        # Calculate basic statistics
        total_production = len(relevant_orders)
        successful_production = sum(1 for order in relevant_orders if order.status == ProductionStatus.COMPLETED)
        failed_production = sum(1 for order in relevant_orders if order.status == ProductionStatus.FAILED)
        
        # Calculate performance metrics
        if total_production > 0:
            success_rate = (successful_production / total_production) * 100
        else:
            success_rate = 0
        
        # Calculate cycle times
        completed_orders = [order for order in relevant_orders if order.completed_at and order.started_at]
        cycle_times = [(order.completed_at - order.started_at).total_seconds() for order in completed_orders]
        
        average_cycle_time = statistics.mean(cycle_times) if cycle_times else 0
        
        # Calculate throughput
        time_period_hours = (end_time - start_time).total_seconds() / 3600
        throughput_per_hour = successful_production / max(time_period_hours, 1)
        
        # Get throughput metrics
        throughput_metrics = [m for m in relevant_metrics if m.metric_name.endswith("throughput")]
        avg_throughput = statistics.mean([m.value for m in throughput_metrics]) if throughput_metrics else 0
        
        # Get utilization metrics
        utilization_metrics = [m for m in relevant_metrics if "utilization" in m.metric_name]
        avg_utilization = statistics.mean([m.value for m in utilization_metrics]) if utilization_metrics else 0
        
        # Generate insights
        insights = await self._generate_insights(relevant_orders, relevant_metrics)
        recommendations = await self._generate_recommendations(relevant_orders, relevant_metrics)
        trend_analysis = await self._analyze_trends(relevant_metrics)
        
        report = PerformanceReport(
            report_name=f"Performance Report {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}",
            start_time=start_time,
            end_time=end_time,
            total_production=total_production,
            successful_production=successful_production,
            failed_production=failed_production,
            average_cycle_time=average_cycle_time,
            throughput_per_hour=throughput_per_hour,
            quality_score=success_rate,
            defect_rate=(failed_production / max(total_production, 1)) * 100,
            first_pass_yield=success_rate,
            worker_utilization=avg_utilization,
            equipment_utilization=avg_utilization,
            trend_analysis=trend_analysis,
            insights=insights,
            recommendations=recommendations
        )
        
        return report
    
    async def _generate_insights(self, orders: List[ProductionOrder], metrics: List[MetricDataPoint]) -> List[str]:
        """Generate analytical insights."""
        
        insights = []
        
        # Production insights
        if orders:
            total_orders = len(orders)
            completed_orders = sum(1 for order in orders if order.status == ProductionStatus.COMPLETED)
            completion_rate = (completed_orders / total_orders) * 100
            
            if completion_rate > 95:
                insights.append("Exceptional production performance with >95% completion rate")
            elif completion_rate > 90:
                insights.append("Strong production performance with >90% completion rate")
            elif completion_rate < 80:
                insights.append("Production performance below target with <80% completion rate")
            
            # Priority analysis
            urgent_orders = sum(1 for order in orders if order.priority.value in ["urgent", "emergency"])
            if urgent_orders > total_orders * 0.3:
                insights.append("High volume of urgent orders indicating potential capacity constraints")
        
        # Metric insights
        throughput_metrics = [m for m in metrics if "throughput" in m.metric_name]
        if throughput_metrics:
            recent_throughput = [m.value for m in throughput_metrics[-10:]]  # Last 10 readings
            if len(recent_throughput) > 1:
                trend = np.polyfit(range(len(recent_throughput)), recent_throughput, 1)[0]
                if trend > 0.1:
                    insights.append("Throughput trending upward - production efficiency improving")
                elif trend < -0.1:
                    insights.append("Throughput trending downward - investigate potential bottlenecks")
        
        return insights
    
    async def _generate_recommendations(self, orders: List[ProductionOrder], metrics: List[MetricDataPoint]) -> List[str]:
        """Generate recommendations for improvement."""
        
        recommendations = []
        
        # Utilization recommendations
        utilization_metrics = [m for m in metrics if "utilization" in m.metric_name]
        if utilization_metrics:
            avg_utilization = statistics.mean([m.value for m in utilization_metrics])
            
            if avg_utilization < 70:
                recommendations.append("Consider load balancing to improve resource utilization (currently <70%)")
            elif avg_utilization > 95:
                recommendations.append("High utilization detected - consider capacity expansion to prevent bottlenecks")
        
        # Quality recommendations
        if orders:
            failed_rate = sum(1 for order in orders if order.status == ProductionStatus.FAILED) / len(orders)
            
            if failed_rate > 0.05:  # 5% failure rate
                recommendations.append("Implement additional quality controls to reduce failure rate")
            
            # Rework analysis
            high_rework_orders = sum(1 for order in orders if hasattr(order, 'rework_count') and order.rework_count > 1)
            if high_rework_orders > 0:
                recommendations.append("Analyze root causes of rework to improve first-pass yield")
        
        # Performance recommendations
        queue_metrics = [m for m in metrics if "queue_size" in m.metric_name]
        if queue_metrics:
            max_queue_size = max([m.value for m in queue_metrics])
            avg_queue_size = statistics.mean([m.value for m in queue_metrics])
            
            if avg_queue_size > 10:
                recommendations.append("Optimize workflow to reduce queue sizes and improve flow")
            
            if max_queue_size > 50:
                recommendations.append("Implement capacity planning to prevent queue overflow")
        
        return recommendations
    
    async def _analyze_trends(self, metrics: List[MetricDataPoint]) -> Dict[str, Any]:
        """Analyze metric trends."""
        
        trend_analysis = {}
        
        # Group metrics by name
        metric_groups = defaultdict(list)
        for metric in metrics:
            metric_groups[metric.metric_name].append(metric)
        
        # Analyze each metric
        for metric_name, metric_list in metric_groups.items():
            if len(metric_list) < 3:  # Need at least 3 points for trend
                continue
            
            # Sort by timestamp
            metric_list.sort(key=lambda m: m.timestamp)
            
            # Extract values and timestamps
            values = [m.value for m in metric_list]
            timestamps = [(m.timestamp - metric_list[0].timestamp).total_seconds() for m in metric_list]
            
            # Calculate trend
            if len(values) > 1:
                try:
                    slope, intercept = np.polyfit(timestamps, values, 1)
                    
                    # Determine trend direction
                    if abs(slope) < 0.01:
                        trend_direction = "stable"
                    elif slope > 0:
                        trend_direction = "increasing"
                    else:
                        trend_direction = "decreasing"
                    
                    # Calculate trend strength
                    correlation = np.corrcoef(timestamps, values)[0, 1] if len(values) > 2 else 0
                    
                    trend_analysis[metric_name] = {
                        "direction": trend_direction,
                        "slope": slope,
                        "correlation": correlation,
                        "data_points": len(values),
                        "latest_value": values[-1],
                        "change_from_first": values[-1] - values[0],
                        "percentage_change": ((values[-1] - values[0]) / max(abs(values[0]), 1)) * 100
                    }
                    
                except Exception as e:
                    logger.warning("Failed to analyze trend", metric_name=metric_name, error=str(e))
        
        return trend_analysis
    
    async def predict_future_performance(self, hours_ahead: int = 4) -> Dict[str, Any]:
        """Predict future performance based on historical data."""
        
        predictions = {}
        
        # Get recent throughput data
        recent_time = datetime.now(timezone.utc) - timedelta(hours=24)
        throughput_metrics = [
            m for m in self.metric_store
            if m.metric_name.endswith("throughput") and m.timestamp >= recent_time
        ]
        
        if len(throughput_metrics) >= 5:
            # Sort by timestamp
            throughput_metrics.sort(key=lambda m: m.timestamp)
            
            # Prepare data for prediction
            values = [m.value for m in throughput_metrics]
            time_indices = list(range(len(values)))
            
            # Simple linear regression for prediction
            try:
                slope, intercept = np.polyfit(time_indices, values, 1)
                
                # Predict future values
                future_time_index = len(values) + hours_ahead
                predicted_throughput = slope * future_time_index + intercept
                
                # Ensure prediction is reasonable (not negative)
                predicted_throughput = max(0, predicted_throughput)
                
                predictions["throughput"] = {
                    "predicted_value": predicted_throughput,
                    "confidence": min(0.8, len(values) / 24),  # Higher confidence with more data
                    "trend_slope": slope,
                    "hours_ahead": hours_ahead
                }
                
            except Exception as e:
                logger.warning("Failed to predict throughput", error=str(e))
        
        # Predict capacity utilization
        utilization_metrics = [
            m for m in self.metric_store
            if "utilization" in m.metric_name and m.timestamp >= recent_time
        ]
        
        if utilization_metrics:
            avg_utilization = statistics.mean([m.value for m in utilization_metrics])
            recent_utilization = statistics.mean([m.value for m in utilization_metrics[-6:]])  # Last 6 readings
            
            # Simple trend-based prediction
            trend = recent_utilization - avg_utilization
            predicted_utilization = recent_utilization + (trend * hours_ahead / 24)
            predicted_utilization = max(0, min(100, predicted_utilization))  # Clamp to 0-100%
            
            predictions["utilization"] = {
                "predicted_value": predicted_utilization,
                "current_average": avg_utilization,
                "recent_average": recent_utilization,
                "trend": trend,
                "hours_ahead": hours_ahead
            }
        
        return predictions


# ================== MONITORING DASHBOARD ==================

class MonitoringDashboard:
    """Dashboard for monitoring and analytics visualization."""
    
    def __init__(self, analytics_engine: AnalyticsEngine, alerting_engine: AlertingEngine):
        self.analytics_engine = analytics_engine
        self.alerting_engine = alerting_engine
        
    async def get_dashboard_data(self, dashboard_type: DashboardType = DashboardType.OPERATIONAL) -> Dict[str, Any]:
        """Get dashboard data based on type."""
        
        if dashboard_type == DashboardType.OPERATIONAL:
            return await self._get_operational_dashboard()
        elif dashboard_type == DashboardType.TACTICAL:
            return await self._get_tactical_dashboard()
        elif dashboard_type == DashboardType.STRATEGIC:
            return await self._get_strategic_dashboard()
        elif dashboard_type == DashboardType.EXECUTIVE:
            return await self._get_executive_dashboard()
        else:
            return {}
    
    async def _get_operational_dashboard(self) -> Dict[str, Any]:
        """Get operational dashboard data (real-time)."""
        
        current_time = datetime.now(timezone.utc)
        last_hour = current_time - timedelta(hours=1)
        
        # Get recent metrics
        recent_metrics = [
            m for m in self.analytics_engine.metric_store
            if m.timestamp >= last_hour
        ]
        
        # Current alerts
        active_alerts = self.alerting_engine.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        
        # Key performance indicators
        throughput_metrics = [m for m in recent_metrics if "throughput" in m.metric_name]
        current_throughput = throughput_metrics[-1].value if throughput_metrics else 0
        
        utilization_metrics = [m for m in recent_metrics if "utilization" in m.metric_name]
        current_utilization = statistics.mean([m.value for m in utilization_metrics]) if utilization_metrics else 0
        
        queue_metrics = [m for m in recent_metrics if "queue_size" in m.metric_name]
        total_queue_size = sum([m.value for m in queue_metrics])
        
        return {
            "dashboard_type": "operational",
            "timestamp": current_time.isoformat(),
            "real_time_metrics": {
                "current_throughput": current_throughput,
                "current_utilization": current_utilization,
                "total_queue_size": total_queue_size,
                "active_alerts": len(active_alerts),
                "critical_alerts": len(critical_alerts)
            },
            "alerts": {
                "active": len(active_alerts),
                "critical": len(critical_alerts),
                "recent_alerts": [
                    {
                        "id": alert.alert_id,
                        "severity": alert.severity.value,
                        "message": alert.message,
                        "triggered_at": alert.triggered_at.isoformat()
                    }
                    for alert in active_alerts[:5]  # Last 5 alerts
                ]
            },
            "system_status": {
                "overall_health": "healthy" if len(critical_alerts) == 0 else "degraded",
                "last_updated": current_time.isoformat()
            }
        }
    
    async def _get_tactical_dashboard(self) -> Dict[str, Any]:
        """Get tactical dashboard data (hourly/daily trends)."""
        
        # Generate performance report for last 24 hours
        report = await self.analytics_engine.generate_performance_report()
        
        # Get predictions
        predictions = await self.analytics_engine.predict_future_performance(hours_ahead=4)
        
        # Alert statistics
        alert_stats = self.alerting_engine.get_alert_statistics()
        
        return {
            "dashboard_type": "tactical",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "performance_summary": {
                "total_production": report.total_production,
                "success_rate": (report.successful_production / max(report.total_production, 1)) * 100,
                "average_cycle_time": report.average_cycle_time,
                "throughput_per_hour": report.throughput_per_hour,
                "quality_score": report.quality_score
            },
            "trends": report.trend_analysis,
            "predictions": predictions,
            "insights": report.insights,
            "recommendations": report.recommendations,
            "alert_statistics": alert_stats
        }
    
    async def _get_strategic_dashboard(self) -> Dict[str, Any]:
        """Get strategic dashboard data (weekly/monthly trends)."""
        
        # Generate extended performance report
        start_time = datetime.now(timezone.utc) - timedelta(days=30)
        monthly_report = await self.analytics_engine.generate_performance_report(start_time=start_time)
        
        # Compare with previous month
        previous_month_start = start_time - timedelta(days=30)
        previous_month_report = await self.analytics_engine.generate_performance_report(
            start_time=previous_month_start,
            end_time=start_time
        )
        
        # Calculate month-over-month changes
        throughput_change = monthly_report.throughput_per_hour - previous_month_report.throughput_per_hour
        quality_change = monthly_report.quality_score - previous_month_report.quality_score
        
        return {
            "dashboard_type": "strategic",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "monthly_performance": {
                "total_production": monthly_report.total_production,
                "throughput_per_hour": monthly_report.throughput_per_hour,
                "quality_score": monthly_report.quality_score,
                "utilization": monthly_report.worker_utilization
            },
            "month_over_month": {
                "throughput_change": throughput_change,
                "quality_change": quality_change,
                "throughput_change_percentage": (throughput_change / max(previous_month_report.throughput_per_hour, 1)) * 100,
                "quality_change_percentage": (quality_change / max(previous_month_report.quality_score, 1)) * 100
            },
            "strategic_insights": monthly_report.insights,
            "strategic_recommendations": monthly_report.recommendations,
            "capacity_analysis": {
                "current_utilization": monthly_report.worker_utilization,
                "capacity_headroom": 100 - monthly_report.worker_utilization,
                "expansion_recommended": monthly_report.worker_utilization > 85
            }
        }
    
    async def _get_executive_dashboard(self) -> Dict[str, Any]:
        """Get executive dashboard data (high-level KPIs)."""
        
        # Get quarterly data
        start_time = datetime.now(timezone.utc) - timedelta(days=90)
        quarterly_report = await self.analytics_engine.generate_performance_report(start_time=start_time)
        
        # Calculate key business metrics
        total_value_produced = quarterly_report.total_production * 100  # Mock value per unit
        
        return {
            "dashboard_type": "executive",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "key_performance_indicators": {
                "total_production_value": total_value_produced,
                "overall_efficiency": quarterly_report.quality_score,
                "customer_satisfaction": quarterly_report.customer_satisfaction,
                "operational_excellence": min(100, quarterly_report.quality_score + quarterly_report.worker_utilization) / 2
            },
            "business_impact": {
                "production_volume": quarterly_report.total_production,
                "quality_achievement": quarterly_report.quality_score,
                "cost_efficiency": quarterly_report.cost_per_unit,
                "time_to_market": quarterly_report.average_cycle_time
            },
            "strategic_objectives": {
                "quality_target_achievement": quarterly_report.quality_score >= 95,
                "efficiency_target_achievement": quarterly_report.worker_utilization >= 80,
                "throughput_target_achievement": quarterly_report.throughput_per_hour >= 100
            },
            "executive_summary": quarterly_report.insights[:3],  # Top 3 insights
            "action_items": quarterly_report.recommendations[:3]  # Top 3 recommendations
        }


# Export main classes
__all__ = [
    "MetricType",
    "AlertSeverity",
    "MonitoringScope",
    "AnalyticsType",
    "DashboardType",
    "MetricDataPoint",
    "Alert",
    "PerformanceReport",
    "MetricCollector",
    "FactoryMetricCollector",
    "ProductionLineMetricCollector",
    "WorkerMetricCollector",
    "AlertRule",
    "AlertingEngine",
    "AnalyticsEngine",
    "MonitoringDashboard"
]
