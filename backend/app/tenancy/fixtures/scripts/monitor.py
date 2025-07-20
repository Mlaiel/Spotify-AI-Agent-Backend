#!/usr/bin/env python3
"""
Spotify AI Agent - Fixture Monitoring Script
===========================================

Advanced monitoring script that provides:
- Real-time fixture health monitoring
- Performance metrics and analytics
- Alert system for critical issues
- Dashboard generation and reporting
- Automated health checks and recovery

Usage:
    python -m app.tenancy.fixtures.scripts.monitor --mode dashboard --tenant-id mycompany
    python monitor.py --health-check --alert-threshold critical --auto-recovery

Author: Expert Development Team (Fahed Mlaiel)
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable

import psutil
from sqlalchemy import text, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session
from app.core.cache import get_redis_client
from app.tenancy.fixtures.base import FixtureManager
from app.tenancy.fixtures.monitoring import FixtureMonitor
from app.tenancy.fixtures.utils import FixtureUtils, TenantUtils
from app.tenancy.fixtures.exceptions import FixtureError, FixtureMonitoringError
from app.tenancy.fixtures.constants import MONITORING_THRESHOLDS, HEALTH_CHECK_INTERVALS

logger = logging.getLogger(__name__)


class HealthMetrics:
    """Container for health metrics."""
    
    def __init__(self):
        self.timestamp = datetime.now(timezone.utc)
        self.database_metrics = {}
        self.cache_metrics = {}
        self.system_metrics = {}
        self.application_metrics = {}
        self.tenant_metrics = {}
        self.overall_health = "unknown"
        self.alerts = []
        self.recommendations = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "database_metrics": self.database_metrics,
            "cache_metrics": self.cache_metrics,
            "system_metrics": self.system_metrics,
            "application_metrics": self.application_metrics,
            "tenant_metrics": self.tenant_metrics,
            "overall_health": self.overall_health,
            "alerts": self.alerts,
            "recommendations": self.recommendations
        }


class AlertManager:
    """Manage alerts and notifications."""
    
    def __init__(self):
        self.alert_history = []
        self.alert_rules = {
            "database_connection_failure": {
                "severity": "critical",
                "threshold": 1,
                "notification_channels": ["email", "slack"]
            },
            "high_memory_usage": {
                "severity": "warning", 
                "threshold": 80,  # percentage
                "notification_channels": ["email"]
            },
            "cache_miss_rate_high": {
                "severity": "warning",
                "threshold": 50,  # percentage
                "notification_channels": ["slack"]
            },
            "tenant_fixture_failure": {
                "severity": "error",
                "threshold": 3,  # failed operations
                "notification_channels": ["email", "slack"]
            },
            "performance_degradation": {
                "severity": "warning",
                "threshold": 2.0,  # seconds
                "notification_channels": ["email"]
            }
        }
    
    async def evaluate_alerts(self, metrics: HealthMetrics) -> List[Dict[str, Any]]:
        """Evaluate metrics against alert rules."""
        alerts = []
        
        # Database connection alerts
        if not metrics.database_metrics.get("connection_healthy", True):
            alerts.append({
                "type": "database_connection_failure",
                "severity": "critical",
                "message": "Database connection failed",
                "timestamp": metrics.timestamp.isoformat(),
                "metrics": metrics.database_metrics
            })
        
        # Memory usage alerts
        memory_usage = metrics.system_metrics.get("memory_percent", 0)
        if memory_usage > self.alert_rules["high_memory_usage"]["threshold"]:
            alerts.append({
                "type": "high_memory_usage",
                "severity": "warning",
                "message": f"High memory usage: {memory_usage:.1f}%",
                "timestamp": metrics.timestamp.isoformat(),
                "value": memory_usage
            })
        
        # Cache performance alerts
        cache_miss_rate = metrics.cache_metrics.get("miss_rate_percent", 0)
        if cache_miss_rate > self.alert_rules["cache_miss_rate_high"]["threshold"]:
            alerts.append({
                "type": "cache_miss_rate_high",
                "severity": "warning", 
                "message": f"High cache miss rate: {cache_miss_rate:.1f}%",
                "timestamp": metrics.timestamp.isoformat(),
                "value": cache_miss_rate
            })
        
        # Performance degradation alerts
        avg_response_time = metrics.application_metrics.get("avg_response_time", 0)
        if avg_response_time > self.alert_rules["performance_degradation"]["threshold"]:
            alerts.append({
                "type": "performance_degradation",
                "severity": "warning",
                "message": f"Performance degradation: {avg_response_time:.2f}s avg response time",
                "timestamp": metrics.timestamp.isoformat(),
                "value": avg_response_time
            })
        
        # Tenant-specific alerts
        for tenant_id, tenant_data in metrics.tenant_metrics.items():
            failed_operations = tenant_data.get("failed_operations", 0)
            if failed_operations > self.alert_rules["tenant_fixture_failure"]["threshold"]:
                alerts.append({
                    "type": "tenant_fixture_failure",
                    "severity": "error",
                    "message": f"Multiple fixture failures for tenant {tenant_id}: {failed_operations}",
                    "timestamp": metrics.timestamp.isoformat(),
                    "tenant_id": tenant_id,
                    "value": failed_operations
                })
        
        # Store alert history
        self.alert_history.extend(alerts)
        
        return alerts
    
    async def send_notifications(self, alerts: List[Dict[str, Any]]) -> None:
        """Send notifications for alerts."""
        for alert in alerts:
            alert_type = alert["type"]
            rule = self.alert_rules.get(alert_type, {})
            channels = rule.get("notification_channels", [])
            
            for channel in channels:
                await self._send_notification(channel, alert)
    
    async def _send_notification(self, channel: str, alert: Dict[str, Any]) -> None:
        """Send notification to specific channel."""
        # Mock implementation - would integrate with actual notification services
        logger.info(f"Alert notification [{channel}]: {alert['message']}")


class PerformanceAnalyzer:
    """Analyze performance metrics and trends."""
    
    def __init__(self):
        self.metrics_history = []
        self.baseline_metrics = {}
    
    async def analyze_performance_trends(
        self,
        metrics: HealthMetrics,
        lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        
        # Store current metrics
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        self.metrics_history = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        if len(self.metrics_history) < 2:
            return {"status": "insufficient_data", "message": "Need more data points for analysis"}
        
        analysis = {
            "trend_analysis": await self._analyze_trends(),
            "performance_regression": await self._detect_regression(),
            "capacity_planning": await self._capacity_analysis(),
            "optimization_opportunities": await self._find_optimizations(),
            "health_score_trend": await self._calculate_health_trend()
        }
        
        return analysis
    
    async def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze metric trends."""
        trends = {}
        
        # Response time trend
        response_times = [
            m.application_metrics.get("avg_response_time", 0) 
            for m in self.metrics_history
        ]
        trends["response_time"] = {
            "direction": "improving" if response_times[-1] < response_times[0] else "degrading",
            "change_percent": ((response_times[-1] - response_times[0]) / max(response_times[0], 0.001)) * 100
        }
        
        # Memory usage trend
        memory_usage = [
            m.system_metrics.get("memory_percent", 0)
            for m in self.metrics_history
        ]
        trends["memory_usage"] = {
            "direction": "increasing" if memory_usage[-1] > memory_usage[0] else "decreasing",
            "change_percent": ((memory_usage[-1] - memory_usage[0]) / max(memory_usage[0], 0.001)) * 100
        }
        
        # Cache hit rate trend
        cache_hit_rates = [
            m.cache_metrics.get("hit_rate_percent", 0)
            for m in self.metrics_history
        ]
        trends["cache_performance"] = {
            "direction": "improving" if cache_hit_rates[-1] > cache_hit_rates[0] else "degrading",
            "change_percent": ((cache_hit_rates[-1] - cache_hit_rates[0]) / max(cache_hit_rates[0], 0.001)) * 100
        }
        
        return trends
    
    async def _detect_regression(self) -> Dict[str, Any]:
        """Detect performance regression."""
        if len(self.metrics_history) < 10:
            return {"status": "insufficient_data"}
        
        # Calculate baseline from first half of data
        baseline_size = len(self.metrics_history) // 2
        baseline_metrics = self.metrics_history[:baseline_size]
        recent_metrics = self.metrics_history[baseline_size:]
        
        # Calculate averages
        baseline_response = sum(
            m.application_metrics.get("avg_response_time", 0) 
            for m in baseline_metrics
        ) / len(baseline_metrics)
        
        recent_response = sum(
            m.application_metrics.get("avg_response_time", 0)
            for m in recent_metrics
        ) / len(recent_metrics)
        
        regression_threshold = 0.2  # 20% increase
        regression_detected = (recent_response - baseline_response) / max(baseline_response, 0.001) > regression_threshold
        
        return {
            "regression_detected": regression_detected,
            "baseline_response_time": baseline_response,
            "recent_response_time": recent_response,
            "performance_change_percent": ((recent_response - baseline_response) / max(baseline_response, 0.001)) * 100
        }
    
    async def _capacity_analysis(self) -> Dict[str, Any]:
        """Analyze capacity and scaling needs."""
        latest_metrics = self.metrics_history[-1]
        
        # Calculate resource utilization
        cpu_usage = latest_metrics.system_metrics.get("cpu_percent", 0)
        memory_usage = latest_metrics.system_metrics.get("memory_percent", 0)
        
        # Predict scaling needs
        scaling_needed = False
        scaling_recommendations = []
        
        if cpu_usage > 80:
            scaling_needed = True
            scaling_recommendations.append("Consider adding CPU resources or horizontal scaling")
        
        if memory_usage > 85:
            scaling_needed = True
            scaling_recommendations.append("Memory usage is high - consider increasing memory allocation")
        
        # Estimate capacity runway
        if self.metrics_history:
            # Simple linear projection
            memory_trend = [m.system_metrics.get("memory_percent", 0) for m in self.metrics_history[-10:]]
            if len(memory_trend) > 1:
                memory_growth_rate = (memory_trend[-1] - memory_trend[0]) / len(memory_trend)
                if memory_growth_rate > 0:
                    days_to_capacity = (95 - memory_usage) / (memory_growth_rate * 24)  # Assuming hourly measurements
                else:
                    days_to_capacity = float('inf')
            else:
                days_to_capacity = float('inf')
        else:
            days_to_capacity = float('inf')
        
        return {
            "current_utilization": {
                "cpu_percent": cpu_usage,
                "memory_percent": memory_usage
            },
            "scaling_needed": scaling_needed,
            "recommendations": scaling_recommendations,
            "estimated_days_to_capacity": min(days_to_capacity, 365) if days_to_capacity != float('inf') else None
        }
    
    async def _find_optimizations(self) -> List[Dict[str, Any]]:
        """Find optimization opportunities."""
        optimizations = []
        latest_metrics = self.metrics_history[-1]
        
        # Cache optimization
        cache_hit_rate = latest_metrics.cache_metrics.get("hit_rate_percent", 0)
        if cache_hit_rate < 80:
            optimizations.append({
                "type": "cache_optimization",
                "priority": "medium",
                "description": f"Cache hit rate is {cache_hit_rate:.1f}% - consider cache tuning",
                "potential_impact": "Improved response times and reduced database load"
            })
        
        # Database optimization
        db_connections = latest_metrics.database_metrics.get("active_connections", 0)
        max_connections = latest_metrics.database_metrics.get("max_connections", 100)
        if db_connections / max_connections > 0.8:
            optimizations.append({
                "type": "database_optimization",
                "priority": "high",
                "description": "High database connection usage - consider connection pooling optimization",
                "potential_impact": "Better database performance and connection efficiency"
            })
        
        # Query optimization
        slow_queries = latest_metrics.database_metrics.get("slow_queries", [])
        if slow_queries:
            optimizations.append({
                "type": "query_optimization",
                "priority": "high",
                "description": f"Found {len(slow_queries)} slow queries - consider indexing or query optimization",
                "potential_impact": "Significantly improved query performance"
            })
        
        return optimizations
    
    async def _calculate_health_trend(self) -> Dict[str, Any]:
        """Calculate overall health score trend."""
        if len(self.metrics_history) < 5:
            return {"status": "insufficient_data"}
        
        # Calculate health scores for recent metrics
        health_scores = []
        for metrics in self.metrics_history[-5:]:
            score = await self._calculate_health_score(metrics)
            health_scores.append(score)
        
        trend = "stable"
        if health_scores[-1] > health_scores[0] + 5:
            trend = "improving"
        elif health_scores[-1] < health_scores[0] - 5:
            trend = "degrading"
        
        return {
            "current_score": health_scores[-1],
            "trend": trend,
            "score_change": health_scores[-1] - health_scores[0],
            "score_history": health_scores
        }
    
    async def _calculate_health_score(self, metrics: HealthMetrics) -> float:
        """Calculate overall health score (0-100)."""
        score = 100.0
        
        # Database health (25% weight)
        if not metrics.database_metrics.get("connection_healthy", True):
            score -= 25
        else:
            slow_queries = len(metrics.database_metrics.get("slow_queries", []))
            score -= min(slow_queries * 2, 10)  # -2 points per slow query, max -10
        
        # System health (25% weight)
        memory_usage = metrics.system_metrics.get("memory_percent", 0)
        cpu_usage = metrics.system_metrics.get("cpu_percent", 0)
        score -= max(0, memory_usage - 70) * 0.5  # -0.5 points per % over 70%
        score -= max(0, cpu_usage - 70) * 0.3     # -0.3 points per % over 70%
        
        # Cache performance (20% weight)
        cache_hit_rate = metrics.cache_metrics.get("hit_rate_percent", 0)
        score -= max(0, 90 - cache_hit_rate) * 0.2  # -0.2 points per % under 90%
        
        # Application performance (30% weight)
        response_time = metrics.application_metrics.get("avg_response_time", 0)
        score -= max(0, response_time - 1.0) * 10   # -10 points per second over 1s
        
        return max(0, min(100, score))


class FixtureMonitoringSystem:
    """
    Comprehensive fixture monitoring system.
    
    Provides:
    - Real-time health monitoring
    - Performance analytics
    - Alert management
    - Recovery automation
    - Dashboard generation
    """
    
    def __init__(self, session: AsyncSession, redis_client=None):
        self.session = session
        self.redis_client = redis_client
        self.fixture_manager = FixtureManager(session, redis_client)
        self.fixture_monitor = FixtureMonitor(session, redis_client)
        self.alert_manager = AlertManager()
        self.performance_analyzer = PerformanceAnalyzer()
        self.monitoring_active = False
    
    async def start_monitoring(
        self,
        interval_seconds: int = 60,
        health_check_enabled: bool = True,
        auto_recovery: bool = False
    ) -> None:
        """Start continuous monitoring."""
        logger.info(f"Starting fixture monitoring (interval: {interval_seconds}s)")
        self.monitoring_active = True
        
        try:
            while self.monitoring_active:
                try:
                    # Collect metrics
                    metrics = await self.collect_health_metrics()
                    
                    # Evaluate alerts
                    alerts = await self.alert_manager.evaluate_alerts(metrics)
                    
                    # Send notifications
                    if alerts:
                        await self.alert_manager.send_notifications(alerts)
                    
                    # Analyze performance
                    performance_analysis = await self.performance_analyzer.analyze_performance_trends(metrics)
                    
                    # Auto-recovery if enabled
                    if auto_recovery and alerts:
                        await self._attempt_auto_recovery(alerts, metrics)
                    
                    # Log status
                    logger.info(f"Health check completed - Status: {metrics.overall_health}")
                    if alerts:
                        logger.warning(f"Active alerts: {len(alerts)}")
                    
                except Exception as e:
                    logger.error(f"Monitoring cycle failed: {e}")
                
                # Wait for next cycle
                await asyncio.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        finally:
            self.monitoring_active = False
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self.monitoring_active = False
        logger.info("Monitoring stopped")
    
    async def collect_health_metrics(self, tenant_id: Optional[str] = None) -> HealthMetrics:
        """Collect comprehensive health metrics."""
        metrics = HealthMetrics()
        
        try:
            # Database metrics
            metrics.database_metrics = await self._collect_database_metrics()
            
            # Cache metrics
            if self.redis_client:
                metrics.cache_metrics = await self._collect_cache_metrics()
            
            # System metrics
            metrics.system_metrics = await self._collect_system_metrics()
            
            # Application metrics
            metrics.application_metrics = await self._collect_application_metrics()
            
            # Tenant-specific metrics
            if tenant_id:
                metrics.tenant_metrics[tenant_id] = await self._collect_tenant_metrics(tenant_id)
            else:
                # Collect for all tenants
                tenant_list = await self._get_tenant_list()
                for tid in tenant_list[:10]:  # Limit to first 10 tenants for performance
                    try:
                        metrics.tenant_metrics[tid] = await self._collect_tenant_metrics(tid)
                    except Exception as e:
                        logger.error(f"Failed to collect metrics for tenant {tid}: {e}")
            
            # Calculate overall health
            metrics.overall_health = await self._calculate_overall_health(metrics)
            
            # Generate recommendations
            metrics.recommendations = await self._generate_recommendations(metrics)
            
        except Exception as e:
            logger.error(f"Failed to collect health metrics: {e}")
            metrics.overall_health = "error"
            metrics.alerts.append({
                "type": "metrics_collection_failure",
                "message": f"Failed to collect metrics: {e}",
                "timestamp": metrics.timestamp.isoformat()
            })
        
        return metrics
    
    async def _collect_database_metrics(self) -> Dict[str, Any]:
        """Collect database-related metrics."""
        metrics = {}
        
        try:
            # Test connection
            await self.session.execute(text("SELECT 1"))
            metrics["connection_healthy"] = True
            
            # Active connections
            result = await self.session.execute(
                text("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
            )
            metrics["active_connections"] = result.scalar() or 0
            
            # Max connections
            result = await self.session.execute(text("SHOW max_connections"))
            metrics["max_connections"] = int(result.scalar() or 100)
            
            # Database size
            result = await self.session.execute(
                text("SELECT pg_size_pretty(pg_database_size(current_database()))")
            )
            metrics["database_size"] = result.scalar() or "unknown"
            
            # Slow queries (queries running > 5 seconds)
            result = await self.session.execute(
                text("""
                SELECT query, query_start, state, wait_event_type, wait_event
                FROM pg_stat_activity 
                WHERE state = 'active' 
                AND query_start < NOW() - INTERVAL '5 seconds'
                AND query NOT LIKE '%pg_stat_activity%'
                """)
            )
            
            slow_queries = []
            for row in result:
                slow_queries.append({
                    "query": row[0][:100] + "..." if len(row[0]) > 100 else row[0],
                    "query_start": row[1].isoformat() if row[1] else None,
                    "state": row[2],
                    "wait_event_type": row[3],
                    "wait_event": row[4]
                })
            
            metrics["slow_queries"] = slow_queries
            
            # Connection utilization
            metrics["connection_utilization_percent"] = (
                metrics["active_connections"] / metrics["max_connections"]
            ) * 100
            
        except Exception as e:
            logger.error(f"Database metrics collection failed: {e}")
            metrics["connection_healthy"] = False
            metrics["error"] = str(e)
        
        return metrics
    
    async def _collect_cache_metrics(self) -> Dict[str, Any]:
        """Collect cache-related metrics."""
        metrics = {}
        
        try:
            # Redis info
            info = await self.redis_client.info()
            
            metrics["memory_used"] = info.get("used_memory_human", "unknown")
            metrics["memory_peak"] = info.get("used_memory_peak_human", "unknown") 
            metrics["connected_clients"] = info.get("connected_clients", 0)
            metrics["total_commands_processed"] = info.get("total_commands_processed", 0)
            metrics["instantaneous_ops_per_sec"] = info.get("instantaneous_ops_per_sec", 0)
            
            # Calculate hit/miss rates from stats
            keyspace_hits = info.get("keyspace_hits", 0)
            keyspace_misses = info.get("keyspace_misses", 0)
            total_requests = keyspace_hits + keyspace_misses
            
            if total_requests > 0:
                metrics["hit_rate_percent"] = (keyspace_hits / total_requests) * 100
                metrics["miss_rate_percent"] = (keyspace_misses / total_requests) * 100
            else:
                metrics["hit_rate_percent"] = 0
                metrics["miss_rate_percent"] = 0
            
            # Key count
            metrics["total_keys"] = await self.redis_client.dbsize()
            
            # Memory fragmentation
            memory_rss = info.get("used_memory_rss", 0)
            memory_used = info.get("used_memory", 0)
            if memory_used > 0:
                metrics["memory_fragmentation_ratio"] = memory_rss / memory_used
            
        except Exception as e:
            logger.error(f"Cache metrics collection failed: {e}")
            metrics["error"] = str(e)
        
        return metrics
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level metrics."""
        metrics = {}
        
        try:
            # CPU usage
            metrics["cpu_percent"] = psutil.cpu_percent(interval=1)
            metrics["cpu_count"] = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics["memory_total"] = memory.total
            metrics["memory_used"] = memory.used
            metrics["memory_percent"] = memory.percent
            metrics["memory_available"] = memory.available
            
            # Disk usage
            disk = psutil.disk_usage('/')
            metrics["disk_total"] = disk.total
            metrics["disk_used"] = disk.used
            metrics["disk_percent"] = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            metrics["network_bytes_sent"] = network.bytes_sent
            metrics["network_bytes_recv"] = network.bytes_recv
            
            # Load average (Unix only)
            try:
                load_avg = psutil.getloadavg()
                metrics["load_average_1m"] = load_avg[0]
                metrics["load_average_5m"] = load_avg[1]
                metrics["load_average_15m"] = load_avg[2]
            except AttributeError:
                # Windows doesn't have load average
                pass
            
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
            metrics["error"] = str(e)
        
        return metrics
    
    async def _collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application-specific metrics."""
        metrics = {}
        
        try:
            # Response time simulation (would be real metrics in production)
            # This would typically come from application performance monitoring
            
            metrics["avg_response_time"] = 0.85  # Mock value
            metrics["requests_per_minute"] = 145  # Mock value
            metrics["error_rate_percent"] = 0.5   # Mock value
            
            # Fixture operation metrics
            fixture_metrics = await self.fixture_monitor.get_performance_metrics()
            metrics.update(fixture_metrics)
            
        except Exception as e:
            logger.error(f"Application metrics collection failed: {e}")
            metrics["error"] = str(e)
        
        return metrics
    
    async def _collect_tenant_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Collect tenant-specific metrics."""
        metrics = {}
        
        try:
            schema_name = TenantUtils.get_tenant_schema_name(tenant_id)
            
            # Data counts
            result = await self.session.execute(
                text(f"SELECT COUNT(*) FROM {schema_name}.users")
            )
            metrics["user_count"] = result.scalar() or 0
            
            result = await self.session.execute(
                text(f"SELECT COUNT(*) FROM {schema_name}.ai_sessions")
            )
            metrics["session_count"] = result.scalar() or 0
            
            result = await self.session.execute(
                text(f"SELECT COUNT(*) FROM {schema_name}.content_generated")
            )
            metrics["content_count"] = result.scalar() or 0
            
            # Recent activity (last 24 hours)
            result = await self.session.execute(
                text(f"""
                SELECT COUNT(*) FROM {schema_name}.ai_sessions 
                WHERE created_at > NOW() - INTERVAL '24 hours'
                """)
            )
            metrics["recent_sessions"] = result.scalar() or 0
            
            # Failed operations (mock - would track real failures)
            metrics["failed_operations"] = 0  # Mock value
            
            # Storage usage
            result = await self.session.execute(
                text(f"""
                SELECT pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename))
                FROM pg_tables 
                WHERE schemaname = '{schema_name}'
                """)
            )
            
            schema_size = 0
            for row in result:
                # Parse size (simplified)
                size_str = row[0] or "0 bytes"
                # This would need proper parsing in production
                schema_size += 1  # Mock calculation
            
            metrics["schema_size_mb"] = schema_size
            
        except Exception as e:
            logger.error(f"Tenant metrics collection failed for {tenant_id}: {e}")
            metrics["error"] = str(e)
        
        return metrics
    
    async def _calculate_overall_health(self, metrics: HealthMetrics) -> str:
        """Calculate overall system health status."""
        
        # Check critical issues first
        if not metrics.database_metrics.get("connection_healthy", True):
            return "critical"
        
        if metrics.cache_metrics.get("error"):
            return "degraded"
        
        # Check system resources
        memory_percent = metrics.system_metrics.get("memory_percent", 0)
        cpu_percent = metrics.system_metrics.get("cpu_percent", 0)
        
        if memory_percent > 90 or cpu_percent > 90:
            return "critical"
        elif memory_percent > 80 or cpu_percent > 80:
            return "warning"
        
        # Check application performance
        response_time = metrics.application_metrics.get("avg_response_time", 0)
        error_rate = metrics.application_metrics.get("error_rate_percent", 0)
        
        if response_time > 3.0 or error_rate > 5:
            return "degraded"
        elif response_time > 2.0 or error_rate > 2:
            return "warning"
        
        # Check cache performance
        cache_hit_rate = metrics.cache_metrics.get("hit_rate_percent", 100)
        if cache_hit_rate < 70:
            return "warning"
        
        return "healthy"
    
    async def _generate_recommendations(self, metrics: HealthMetrics) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []
        
        # Memory recommendations
        memory_percent = metrics.system_metrics.get("memory_percent", 0)
        if memory_percent > 85:
            recommendations.append("Consider increasing memory allocation or optimizing memory usage")
        
        # Cache recommendations
        cache_hit_rate = metrics.cache_metrics.get("hit_rate_percent", 100)
        if cache_hit_rate < 80:
            recommendations.append("Cache hit rate is low - consider cache optimization or TTL tuning")
        
        # Database recommendations
        active_conn = metrics.database_metrics.get("active_connections", 0)
        max_conn = metrics.database_metrics.get("max_connections", 100)
        if active_conn / max_conn > 0.8:
            recommendations.append("Database connection pool is highly utilized - consider optimization")
        
        slow_queries = metrics.database_metrics.get("slow_queries", [])
        if slow_queries:
            recommendations.append(f"Found {len(slow_queries)} slow queries - consider query optimization")
        
        # Performance recommendations
        response_time = metrics.application_metrics.get("avg_response_time", 0)
        if response_time > 1.5:
            recommendations.append("Response times are elevated - investigate performance bottlenecks")
        
        return recommendations
    
    async def _get_tenant_list(self) -> List[str]:
        """Get list of all tenants."""
        try:
            result = await self.session.execute(
                text("""
                SELECT schema_name 
                FROM information_schema.schemata 
                WHERE schema_name LIKE 'tenant_%'
                """)
            )
            
            tenant_schemas = [row[0] for row in result]
            return [schema.replace('tenant_', '') for schema in tenant_schemas]
            
        except Exception as e:
            logger.error(f"Error getting tenant list: {e}")
            return []
    
    async def _attempt_auto_recovery(
        self,
        alerts: List[Dict[str, Any]],
        metrics: HealthMetrics
    ) -> None:
        """Attempt automatic recovery for certain issues."""
        
        for alert in alerts:
            alert_type = alert["type"]
            
            try:
                if alert_type == "cache_miss_rate_high":
                    await self._recovery_cache_warmup()
                elif alert_type == "high_memory_usage":
                    await self._recovery_memory_cleanup()
                elif alert_type == "tenant_fixture_failure":
                    tenant_id = alert.get("tenant_id")
                    if tenant_id:
                        await self._recovery_tenant_fixtures(tenant_id)
                
                logger.info(f"Auto-recovery attempted for alert: {alert_type}")
                
            except Exception as e:
                logger.error(f"Auto-recovery failed for {alert_type}: {e}")
    
    async def _recovery_cache_warmup(self) -> None:
        """Perform cache warmup to improve hit rates."""
        # Mock implementation - would implement real cache warming
        logger.info("Performing cache warmup for improved hit rates")
    
    async def _recovery_memory_cleanup(self) -> None:
        """Perform memory cleanup to reduce usage."""
        # Mock implementation - would implement memory optimization
        logger.info("Performing memory cleanup to reduce usage")
    
    async def _recovery_tenant_fixtures(self, tenant_id: str) -> None:
        """Attempt to recover tenant fixture issues."""
        try:
            # Validate tenant data
            from .validate_data import validate_data
            validation_result = await validate_data(tenant_id=tenant_id, auto_fix=True)
            
            if validation_result.get("auto_fixes_applied"):
                logger.info(f"Applied auto-fixes for tenant {tenant_id}")
            
        except Exception as e:
            logger.error(f"Tenant recovery failed for {tenant_id}: {e}")
    
    async def generate_dashboard(
        self,
        output_format: str = "json",
        include_charts: bool = False
    ) -> Dict[str, Any]:
        """Generate monitoring dashboard data."""
        
        # Collect current metrics
        metrics = await self.collect_health_metrics()
        
        # Get performance analysis
        performance_analysis = await self.performance_analyzer.analyze_performance_trends(metrics)
        
        # Get alert history
        recent_alerts = self.alert_manager.alert_history[-50:]  # Last 50 alerts
        
        dashboard_data = {
            "dashboard_generated_at": datetime.now(timezone.utc).isoformat(),
            "overall_status": {
                "health": metrics.overall_health,
                "active_alerts": len([a for a in recent_alerts if a.get("severity") in ["critical", "error"]]),
                "system_uptime": "99.9%",  # Mock value
                "last_incident": None  # Mock value
            },
            "current_metrics": metrics.to_dict(),
            "performance_analysis": performance_analysis,
            "alert_summary": {
                "total_alerts_24h": len([a for a in recent_alerts if 
                    datetime.fromisoformat(a["timestamp"].replace('Z', '+00:00')) > 
                    datetime.now(timezone.utc) - timedelta(hours=24)]),
                "critical_alerts": len([a for a in recent_alerts if a.get("severity") == "critical"]),
                "warning_alerts": len([a for a in recent_alerts if a.get("severity") == "warning"]),
                "recent_alerts": recent_alerts[-10:]  # Last 10 alerts
            },
            "tenant_overview": {
                "total_tenants": len(metrics.tenant_metrics),
                "healthy_tenants": len([t for t in metrics.tenant_metrics.values() 
                                      if t.get("failed_operations", 0) == 0]),
                "tenants_with_issues": len([t for t in metrics.tenant_metrics.values() 
                                          if t.get("failed_operations", 0) > 0])
            },
            "recommendations": metrics.recommendations
        }
        
        # Add charts data if requested
        if include_charts:
            dashboard_data["charts"] = await self._generate_chart_data(metrics)
        
        return dashboard_data
    
    async def _generate_chart_data(self, metrics: HealthMetrics) -> Dict[str, Any]:
        """Generate chart data for dashboard."""
        
        # Mock chart data - would generate real charts in production
        charts = {
            "response_time_trend": {
                "type": "line",
                "data": [0.8, 0.9, 0.7, 0.85, 0.9, 0.8],  # Mock data
                "labels": ["6h ago", "5h ago", "4h ago", "3h ago", "2h ago", "1h ago"]
            },
            "memory_usage": {
                "type": "gauge",
                "current": metrics.system_metrics.get("memory_percent", 0),
                "max": 100,
                "thresholds": [70, 85, 95]
            },
            "cache_performance": {
                "type": "donut",
                "data": {
                    "hits": metrics.cache_metrics.get("hit_rate_percent", 0),
                    "misses": metrics.cache_metrics.get("miss_rate_percent", 0)
                }
            },
            "tenant_distribution": {
                "type": "bar",
                "data": [len(metrics.tenant_metrics)],  # Simplified
                "labels": ["Active Tenants"]
            }
        }
        
        return charts


async def monitor_fixtures(
    mode: str = "health-check",
    tenant_id: Optional[str] = None,
    interval: int = 60,
    auto_recovery: bool = False,
    alert_threshold: str = "warning",
    output_format: str = "json"
) -> Dict[str, Any]:
    """
    Main function to monitor fixtures.
    
    Args:
        mode: Monitoring mode (health-check, dashboard, continuous)
        tenant_id: Specific tenant or None for all
        interval: Monitoring interval in seconds
        auto_recovery: Enable auto-recovery
        alert_threshold: Alert threshold level
        output_format: Output format (json, table)
        
    Returns:
        Monitoring results
    """
    async with get_async_session() as session:
        redis_client = await get_redis_client()
        
        try:
            monitoring_system = FixtureMonitoringSystem(session, redis_client)
            
            if mode == "health-check":
                # Single health check
                metrics = await monitoring_system.collect_health_metrics(tenant_id)
                alerts = await monitoring_system.alert_manager.evaluate_alerts(metrics)
                
                return {
                    "mode": mode,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metrics": metrics.to_dict(),
                    "alerts": alerts,
                    "health_status": metrics.overall_health
                }
            
            elif mode == "dashboard":
                # Generate dashboard
                dashboard_data = await monitoring_system.generate_dashboard(
                    output_format=output_format,
                    include_charts=True
                )
                return dashboard_data
            
            elif mode == "continuous":
                # Start continuous monitoring
                await monitoring_system.start_monitoring(
                    interval_seconds=interval,
                    health_check_enabled=True,
                    auto_recovery=auto_recovery
                )
                return {"status": "monitoring_stopped"}
            
            else:
                raise ValueError(f"Unknown monitoring mode: {mode}")
            
        finally:
            if redis_client:
                await redis_client.close()


def main():
    """Command line interface for fixture monitoring."""
    parser = argparse.ArgumentParser(
        description="Monitor fixture health and performance"
    )
    
    parser.add_argument(
        "--mode",
        choices=["health-check", "dashboard", "continuous"],
        default="health-check",
        help="Monitoring mode"
    )
    
    parser.add_argument(
        "--tenant-id",
        help="Specific tenant to monitor (default: all tenants)"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Monitoring interval in seconds (for continuous mode)"
    )
    
    parser.add_argument(
        "--auto-recovery",
        action="store_true",
        help="Enable automatic recovery for certain issues"
    )
    
    parser.add_argument(
        "--alert-threshold",
        choices=["info", "warning", "error", "critical"],
        default="warning",
        help="Alert threshold level"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["json", "table"],
        default="json",
        help="Output format"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        # Run monitoring
        result = asyncio.run(
            monitor_fixtures(
                mode=args.mode,
                tenant_id=args.tenant_id,
                interval=args.interval,
                auto_recovery=args.auto_recovery,
                alert_threshold=args.alert_threshold,
                output_format=args.output_format
            )
        )
        
        # Display results
        if args.output_format == "json":
            print(json.dumps(result, indent=2, default=str))
        else:
            # Table format
            if args.mode == "health-check":
                print(f"\n🏥 Health Check Results")
                print(f"Status: {result['health_status']}")
                print(f"Timestamp: {result['timestamp']}")
                
                metrics = result["metrics"]
                print(f"\n📊 System Metrics:")
                print(f"  Memory: {metrics['system_metrics'].get('memory_percent', 0):.1f}%")
                print(f"  CPU: {metrics['system_metrics'].get('cpu_percent', 0):.1f}%")
                print(f"  Database: {'✅' if metrics['database_metrics'].get('connection_healthy') else '❌'}")
                
                if result["alerts"]:
                    print(f"\n🚨 Active Alerts ({len(result['alerts'])}):")
                    for alert in result["alerts"][:5]:
                        severity_icon = {"critical": "🔴", "error": "🟠", "warning": "🟡"}.get(alert.get("severity"), "ℹ️")
                        print(f"  {severity_icon} {alert['message']}")
                else:
                    print(f"\n✅ No active alerts")
            
            elif args.mode == "dashboard":
                print(f"\n📊 Dashboard Summary")
                status = result["overall_status"]
                print(f"Health: {status['health']}")
                print(f"Active Alerts: {status['active_alerts']}")
                print(f"System Uptime: {status['system_uptime']}")
                
                tenant_overview = result["tenant_overview"]
                print(f"\n🏢 Tenant Overview:")
                print(f"  Total: {tenant_overview['total_tenants']}")
                print(f"  Healthy: {tenant_overview['healthy_tenants']}")
                print(f"  With Issues: {tenant_overview['tenants_with_issues']}")
                
                if result["recommendations"]:
                    print(f"\n💡 Recommendations:")
                    for rec in result["recommendations"][:3]:
                        print(f"  • {rec}")
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n⚠️  Monitoring interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Monitoring failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
