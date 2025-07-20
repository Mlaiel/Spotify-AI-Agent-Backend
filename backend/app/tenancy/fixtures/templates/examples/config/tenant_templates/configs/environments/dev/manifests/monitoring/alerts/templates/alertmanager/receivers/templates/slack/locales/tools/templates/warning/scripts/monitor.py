#!/usr/bin/env python3
"""
Enterprise Performance Monitoring & Health Checking System
Real-time monitoring with ML-based anomaly detection and predictive analytics
Advanced SLA monitoring with automated alerting and remediation
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import aiohttp
import psutil
import redis
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
from prometheus_client import Gauge, Counter, Histogram, start_http_server
import aioinfluxdb
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

@dataclass
class HealthCheckConfig:
    """Advanced health check configuration"""
    endpoint: str
    method: str = "GET"
    expected_status: int = 200
    timeout: int = 30
    headers: Dict[str, str] = None
    body: Optional[str] = None
    expected_response_time: float = 1.0
    critical: bool = True
    tags: List[str] = None

@dataclass
class MetricThreshold:
    """Metric threshold configuration with SLA definitions"""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison: str  # gt, lt, eq
    duration: int  # seconds before triggering
    sla_target: float
    business_impact: str

class PerformanceMonitor:
    """
    Enterprise Performance Monitor with:
    - Real-time metrics collection
    - ML-based anomaly detection
    - Predictive performance analytics
    - SLA monitoring and reporting
    - Automated scaling recommendations
    """
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.redis_client = redis.Redis.from_url(self.config['redis_url'])
        self.influxdb_client = None
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        self.metrics_history = []
        self.alerts_sent = {}
        
        # Prometheus metrics
        self.response_time_histogram = Histogram(
            'warning_system_response_time_seconds',
            'Response time of warning system operations',
            ['operation', 'tenant', 'environment']
        )
        
        self.error_counter = Counter(
            'warning_system_errors_total',
            'Total number of errors in warning system',
            ['error_type', 'component', 'tenant']
        )
        
        self.active_alerts_gauge = Gauge(
            'warning_system_active_alerts',
            'Number of active alerts',
            ['severity', 'tenant']
        )
        
        self.throughput_gauge = Gauge(
            'warning_system_throughput_per_second',
            'System throughput in operations per second',
            ['operation', 'tenant']
        )
        
        # Initialize ML models
        self._initialize_anomaly_detection()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load monitoring configuration with validation"""
        import yaml
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config.get('monitoring', {})
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging system"""
        logger = logging.getLogger('performance_monitor')
        logger.setLevel(logging.INFO)
        
        # Structured logging handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _initialize_anomaly_detection(self):
        """Initialize ML-based anomaly detection"""
        
        # Isolation Forest for anomaly detection
        self.anomaly_detector = IsolationForest(
            contamination=0.1,  # 10% anomalies expected
            random_state=42,
            n_estimators=100
        )
        
        # Load historical data for training if available
        self._load_historical_data()
    
    def _load_historical_data(self):
        """Load historical performance data for ML training"""
        try:
            # Try to load from Redis/InfluxDB
            historical_data = self.redis_client.get('performance_history')
            
            if historical_data:
                data = json.loads(historical_data)
                df = pd.DataFrame(data)
                
                if len(df) > 100:  # Minimum data for training
                    features = ['cpu_usage', 'memory_usage', 'response_time', 'throughput']
                    X = df[features].values
                    
                    # Scale data
                    X_scaled = self.scaler.fit_transform(X)
                    
                    # Train anomaly detector
                    self.anomaly_detector.fit(X_scaled)
                    self.logger.info("Anomaly detector trained with historical data")
                
        except Exception as e:
            self.logger.warning(f"Could not load historical data: {str(e)}")
    
    async def start_monitoring(self):
        """Start comprehensive monitoring system"""
        
        self.logger.info("Starting performance monitoring system")
        
        # Start Prometheus metrics server
        start_http_server(8000)
        
        # Initialize InfluxDB connection
        await self._initialize_influxdb()
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._collect_system_metrics()),
            asyncio.create_task(self._monitor_application_metrics()),
            asyncio.create_task(self._run_health_checks()),
            asyncio.create_task(self._detect_anomalies()),
            asyncio.create_task(self._monitor_sla_compliance()),
            asyncio.create_task(self._generate_performance_reports())
        ]
        
        await asyncio.gather(*tasks)
    
    async def _initialize_influxdb(self):
        """Initialize InfluxDB connection for time-series data"""
        try:
            self.influxdb_client = aioinfluxdb.InfluxDBClient(
                host=self.config.get('influxdb_host', 'localhost'),
                port=self.config.get('influxdb_port', 8086),
                username=self.config.get('influxdb_username'),
                password=self.config.get('influxdb_password'),
                database=self.config.get('influxdb_database', 'monitoring')
            )
            
            await self.influxdb_client.create_database('monitoring')
            self.logger.info("InfluxDB connection initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize InfluxDB: {str(e)}")
    
    async def _collect_system_metrics(self):
        """Collect system-level performance metrics"""
        
        while True:
            try:
                # CPU metrics
                cpu_usage = psutil.cpu_percent(interval=1)
                cpu_per_core = psutil.cpu_percent(percpu=True)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()
                
                # Disk metrics
                disk_usage = psutil.disk_usage('/')
                disk_io = psutil.disk_io_counters()
                
                # Network metrics
                network_io = psutil.net_io_counters()
                
                # Process metrics
                process_count = len(psutil.pids())
                
                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_usage': cpu_usage,
                    'cpu_per_core': cpu_per_core,
                    'memory_usage': memory.percent,
                    'memory_available': memory.available,
                    'swap_usage': swap.percent,
                    'disk_usage': disk_usage.percent,
                    'disk_read_bytes': disk_io.read_bytes if disk_io else 0,
                    'disk_write_bytes': disk_io.write_bytes if disk_io else 0,
                    'network_bytes_sent': network_io.bytes_sent,
                    'network_bytes_recv': network_io.bytes_recv,
                    'process_count': process_count
                }
                
                # Store metrics
                await self._store_metrics('system', metrics)
                
                # Update Prometheus metrics
                self.throughput_gauge.labels(
                    operation='system_monitoring',
                    tenant='global'
                ).set(1)
                
                # Check thresholds
                await self._check_system_thresholds(metrics)
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {str(e)}")
                await asyncio.sleep(60)
    
    async def _monitor_application_metrics(self):
        """Monitor application-specific metrics"""
        
        while True:
            try:
                # Warning system metrics
                warning_metrics = await self._collect_warning_metrics()
                
                # Database metrics
                db_metrics = await self._collect_database_metrics()
                
                # Redis metrics
                redis_metrics = await self._collect_redis_metrics()
                
                # Combine all application metrics
                app_metrics = {
                    'timestamp': datetime.now().isoformat(),
                    **warning_metrics,
                    **db_metrics,
                    **redis_metrics
                }
                
                # Store metrics
                await self._store_metrics('application', app_metrics)
                
                # Update Prometheus metrics
                self.active_alerts_gauge.labels(
                    severity='high',
                    tenant='global'
                ).set(app_metrics.get('active_high_alerts', 0))
                
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                self.logger.error(f"Error collecting application metrics: {str(e)}")
                await asyncio.sleep(120)
    
    async def _collect_warning_metrics(self) -> Dict[str, Any]:
        """Collect warning system specific metrics"""
        
        try:
            # Get metrics from Redis
            pipeline = self.redis_client.pipeline()
            
            # Alert counts by severity
            pipeline.scard('alerts:critical')
            pipeline.scard('alerts:high')
            pipeline.scard('alerts:medium')
            pipeline.scard('alerts:low')
            
            # Processing metrics
            pipeline.get('metrics:warnings_processed_hour')
            pipeline.get('metrics:avg_processing_time')
            pipeline.get('metrics:failed_notifications')
            
            results = pipeline.execute()
            
            return {
                'active_critical_alerts': results[0] or 0,
                'active_high_alerts': results[1] or 0,
                'active_medium_alerts': results[2] or 0,
                'active_low_alerts': results[3] or 0,
                'warnings_processed_hour': int(results[4] or 0),
                'avg_processing_time': float(results[5] or 0),
                'failed_notifications': int(results[6] or 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting warning metrics: {str(e)}")
            return {}
    
    async def _collect_database_metrics(self) -> Dict[str, Any]:
        """Collect database performance metrics"""
        
        try:
            from sqlalchemy import create_engine, text
            
            engine = create_engine(self.config['database_url'])
            
            with engine.connect() as conn:
                # Active connections
                result = conn.execute(text("SELECT count(*) FROM pg_stat_activity"))
                active_connections = result.scalar()
                
                # Database size
                result = conn.execute(text("""
                    SELECT pg_size_pretty(pg_database_size(current_database()))
                """))
                db_size = result.scalar()
                
                # Slow queries
                result = conn.execute(text("""
                    SELECT count(*) FROM pg_stat_statements 
                    WHERE mean_time > 1000
                """))
                slow_queries = result.scalar() or 0
                
                return {
                    'db_active_connections': active_connections,
                    'db_size': db_size,
                    'db_slow_queries': slow_queries
                }
                
        except Exception as e:
            self.logger.error(f"Error collecting database metrics: {str(e)}")
            return {}
    
    async def _collect_redis_metrics(self) -> Dict[str, Any]:
        """Collect Redis performance metrics"""
        
        try:
            info = self.redis_client.info()
            
            return {
                'redis_connected_clients': info.get('connected_clients', 0),
                'redis_used_memory': info.get('used_memory', 0),
                'redis_keyspace_hits': info.get('keyspace_hits', 0),
                'redis_keyspace_misses': info.get('keyspace_misses', 0),
                'redis_operations_per_sec': info.get('instantaneous_ops_per_sec', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting Redis metrics: {str(e)}")
            return {}
    
    async def _run_health_checks(self):
        """Execute comprehensive health checks"""
        
        health_checks = self.config.get('health_checks', [])
        
        while True:
            try:
                results = []
                
                for check_config in health_checks:
                    check = HealthCheckConfig(**check_config)
                    result = await self._execute_health_check(check)
                    results.append(result)
                
                # Store health check results
                await self._store_metrics('health_checks', {
                    'timestamp': datetime.now().isoformat(),
                    'checks': results,
                    'total_checks': len(results),
                    'passed_checks': len([r for r in results if r['status'] == 'healthy']),
                    'failed_checks': len([r for r in results if r['status'] == 'unhealthy'])
                })
                
                # Alert on failed critical checks
                critical_failures = [r for r in results if r['critical'] and r['status'] == 'unhealthy']
                
                if critical_failures:
                    await self._send_critical_alert(critical_failures)
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                self.logger.error(f"Error running health checks: {str(e)}")
                await asyncio.sleep(120)
    
    async def _execute_health_check(self, check: HealthCheckConfig) -> Dict[str, Any]:
        """Execute individual health check"""
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=check.method,
                    url=check.endpoint,
                    headers=check.headers or {},
                    data=check.body,
                    timeout=aiohttp.ClientTimeout(total=check.timeout)
                ) as response:
                    
                    response_time = time.time() - start_time
                    
                    # Record response time
                    self.response_time_histogram.labels(
                        operation='health_check',
                        tenant='global',
                        environment='production'
                    ).observe(response_time)
                    
                    # Check status and response time
                    status_ok = response.status == check.expected_status
                    time_ok = response_time <= check.expected_response_time
                    
                    overall_status = 'healthy' if (status_ok and time_ok) else 'unhealthy'
                    
                    return {
                        'endpoint': check.endpoint,
                        'status': overall_status,
                        'http_status': response.status,
                        'response_time': response_time,
                        'critical': check.critical,
                        'tags': check.tags or [],
                        'timestamp': datetime.now().isoformat()
                    }
                    
        except Exception as e:
            response_time = time.time() - start_time
            
            self.error_counter.labels(
                error_type='health_check_failure',
                component='monitor',
                tenant='global'
            ).inc()
            
            return {
                'endpoint': check.endpoint,
                'status': 'unhealthy',
                'error': str(e),
                'response_time': response_time,
                'critical': check.critical,
                'tags': check.tags or [],
                'timestamp': datetime.now().isoformat()
            }
    
    async def _detect_anomalies(self):
        """ML-based anomaly detection on performance metrics"""
        
        while True:
            try:
                # Get recent metrics
                recent_metrics = await self._get_recent_metrics(hours=1)
                
                if len(recent_metrics) < 10:
                    await asyncio.sleep(300)  # Wait for more data
                    continue
                
                # Prepare data for anomaly detection
                df = pd.DataFrame(recent_metrics)
                
                features = ['cpu_usage', 'memory_usage', 'response_time', 'throughput']
                available_features = [f for f in features if f in df.columns]
                
                if len(available_features) < 2:
                    await asyncio.sleep(300)
                    continue
                
                X = df[available_features].values
                
                # Scale data
                try:
                    X_scaled = self.scaler.transform(X)
                except:
                    # Fit if not fitted yet
                    X_scaled = self.scaler.fit_transform(X)
                
                # Detect anomalies
                anomalies = self.anomaly_detector.predict(X_scaled)
                anomaly_scores = self.anomaly_detector.decision_function(X_scaled)
                
                # Process detected anomalies
                for i, (is_anomaly, score) in enumerate(zip(anomalies, anomaly_scores)):
                    if is_anomaly == -1:  # Anomaly detected
                        await self._handle_anomaly(recent_metrics[i], score)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in anomaly detection: {str(e)}")
                await asyncio.sleep(600)
    
    async def _handle_anomaly(self, metric_data: Dict[str, Any], anomaly_score: float):
        """Handle detected performance anomaly"""
        
        alert_key = f"anomaly_{metric_data.get('timestamp', time.time())}"
        
        # Avoid duplicate alerts
        if alert_key in self.alerts_sent:
            return
        
        self.alerts_sent[alert_key] = time.time()
        
        anomaly_alert = {
            'type': 'performance_anomaly',
            'severity': 'high' if anomaly_score < -0.5 else 'medium',
            'timestamp': datetime.now().isoformat(),
            'metric_data': metric_data,
            'anomaly_score': anomaly_score,
            'message': f"Performance anomaly detected with score {anomaly_score:.3f}"
        }
        
        await self._send_anomaly_alert(anomaly_alert)
        
        self.logger.warning(f"Anomaly detected: {anomaly_alert}")

class HealthChecker:
    """
    Specialized health checking system with:
    - Multi-tier health checks (infrastructure, application, business logic)
    - Custom health check definitions
    - Health trend analysis
    - Automatic remediation suggestions
    """
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger('health_checker')
        self.health_history = []
        
    async def comprehensive_health_check(self) -> Dict[str, Any]:
        """Run comprehensive multi-tier health check"""
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'infrastructure': await self._check_infrastructure_health(),
            'application': await self._check_application_health(),
            'business_logic': await self._check_business_logic_health(),
            'dependencies': await self._check_dependencies_health()
        }
        
        # Determine overall status
        tier_statuses = [
            results['infrastructure']['status'],
            results['application']['status'],
            results['business_logic']['status'],
            results['dependencies']['status']
        ]
        
        if 'critical' in tier_statuses:
            results['overall_status'] = 'critical'
        elif 'unhealthy' in tier_statuses:
            results['overall_status'] = 'unhealthy'
        elif 'degraded' in tier_statuses:
            results['overall_status'] = 'degraded'
        
        # Store health check history
        self.health_history.append(results)
        if len(self.health_history) > 1000:
            self.health_history = self.health_history[-1000:]
        
        return results
    
    async def _check_infrastructure_health(self) -> Dict[str, Any]:
        """Check infrastructure layer health"""
        
        checks = []
        
        # System resources
        cpu_check = await self._check_cpu_health()
        memory_check = await self._check_memory_health()
        disk_check = await self._check_disk_health()
        network_check = await self._check_network_health()
        
        checks.extend([cpu_check, memory_check, disk_check, network_check])
        
        # Determine infrastructure status
        critical_failures = [c for c in checks if c['status'] == 'critical']
        unhealthy_checks = [c for c in checks if c['status'] == 'unhealthy']
        
        if critical_failures:
            status = 'critical'
        elif unhealthy_checks:
            status = 'unhealthy'
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'checks': checks,
            'summary': f"{len(checks)} infrastructure checks, {len(critical_failures)} critical, {len(unhealthy_checks)} unhealthy"
        }

if __name__ == "__main__":
    # Example usage
    async def main():
        monitor = PerformanceMonitor('monitoring_config.yml')
        await monitor.start_monitoring()
    
    asyncio.run(main())
