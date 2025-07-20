"""
Real-time Streaming Processor for Spotify AI Agent
==================================================

High-performance streaming alert processor for real-time analysis and
response to incoming alert data streams.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Callable, Any, AsyncGenerator
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import aioredis
import aiokafka
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time

@dataclass
class StreamingAlert:
    """Real-time alert data structure."""
    id: str
    tenant_id: str
    metric_name: str
    value: float
    threshold: float
    severity: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any]

@dataclass
class ProcessingResult:
    """Result of streaming processing."""
    alert_id: str
    processing_time: float
    anomaly_detected: bool
    correlation_found: bool
    action_taken: str
    confidence: float
    recommendations: List[str]

class StreamingProcessor:
    """
    High-performance real-time alert stream processor.
    
    Features:
    - Kafka integration for alert streaming
    - Redis for real-time state management
    - Asynchronous processing pipeline
    - Real-time anomaly detection
    - Instant correlation analysis
    - Automated response actions
    - Performance monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.consumer = None
        self.producer = None
        self.redis_client = None
        self.processing_stats = {
            'total_processed': 0,
            'anomalies_detected': 0,
            'correlations_found': 0,
            'avg_processing_time': 0.0,
            'last_processed': None
        }
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        
    async def initialize(self):
        """Initialize streaming components."""
        await self._setup_kafka()
        await self._setup_redis()
        self.logger.info("Streaming processor initialized")
        
    async def _setup_kafka(self):
        """Setup Kafka consumer and producer."""
        kafka_config = self.config.get('kafka', {})
        
        self.consumer = AIOKafkaConsumer(
            kafka_config.get('alert_topic', 'alerts'),
            bootstrap_servers=kafka_config.get('bootstrap_servers', 'localhost:9092'),
            group_id=kafka_config.get('group_id', 'alert_processor'),
            auto_offset_reset='latest',
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        self.producer = AIOKafkaProducer(
            bootstrap_servers=kafka_config.get('bootstrap_servers', 'localhost:9092'),
            value_serializer=lambda x: json.dumps(x, default=str).encode('utf-8')
        )
        
        await self.consumer.start()
        await self.producer.start()
        
    async def _setup_redis(self):
        """Setup Redis for real-time state management."""
        redis_config = self.config.get('redis', {})
        
        self.redis_client = await aioredis.create_redis_pool(
            f"redis://{redis_config.get('host', 'localhost')}:{redis_config.get('port', 6379)}",
            db=redis_config.get('db', 0),
            password=redis_config.get('password'),
            maxsize=redis_config.get('max_connections', 10)
        )
        
    async def start_processing(self):
        """Start the streaming processing pipeline."""
        self.is_running = True
        self.logger.info("Starting alert stream processing")
        
        # Start multiple processing tasks
        tasks = [
            asyncio.create_task(self._process_alert_stream()),
            asyncio.create_task(self._monitoring_task()),
            asyncio.create_task(self._cleanup_task())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Error in streaming processing: {e}")
        finally:
            await self.cleanup()
            
    async def stop_processing(self):
        """Stop the streaming processing pipeline."""
        self.is_running = False
        self.logger.info("Stopping alert stream processing")
        
    async def _process_alert_stream(self):
        """Main alert processing loop."""
        batch_size = self.config.get('batch_size', 100)
        batch_timeout = self.config.get('batch_timeout_ms', 1000)
        
        alert_batch = []
        last_batch_time = time.time()
        
        async for message in self.consumer:
            if not self.is_running:
                break
                
            try:
                # Parse incoming alert
                alert_data = message.value
                alert = self._parse_alert(alert_data)
                
                if alert:
                    alert_batch.append(alert)
                    
                    # Process batch when full or timeout reached
                    current_time = time.time()
                    if (len(alert_batch) >= batch_size or 
                        (current_time - last_batch_time) * 1000 >= batch_timeout):
                        
                        await self._process_alert_batch(alert_batch)
                        alert_batch = []
                        last_batch_time = current_time
                        
            except Exception as e:
                self.logger.error(f"Error processing alert message: {e}")
                
    def _parse_alert(self, alert_data: Dict[str, Any]) -> Optional[StreamingAlert]:
        """Parse incoming alert data."""
        try:
            return StreamingAlert(
                id=alert_data.get('id', ''),
                tenant_id=alert_data.get('tenant_id', ''),
                metric_name=alert_data.get('metric_name', ''),
                value=float(alert_data.get('value', 0)),
                threshold=float(alert_data.get('threshold', 0)),
                severity=alert_data.get('severity', 'info'),
                timestamp=datetime.fromisoformat(alert_data.get('timestamp', datetime.now().isoformat())),
                source=alert_data.get('source', 'unknown'),
                metadata=alert_data.get('metadata', {})
            )
        except (ValueError, KeyError) as e:
            self.logger.error(f"Error parsing alert data: {e}")
            return None
            
    async def _process_alert_batch(self, alerts: List[StreamingAlert]):
        """Process a batch of alerts."""
        start_time = time.time()
        
        # Process alerts in parallel
        processing_tasks = [
            self._process_single_alert(alert) for alert in alerts
        ]
        
        results = await asyncio.gather(*processing_tasks, return_exceptions=True)
        
        # Update statistics
        processing_time = time.time() - start_time
        await self._update_processing_stats(alerts, results, processing_time)
        
        # Send results to output topics
        await self._publish_results(results)
        
    async def _process_single_alert(self, alert: StreamingAlert) -> ProcessingResult:
        """Process a single alert through the analysis pipeline."""
        start_time = time.time()
        
        try:
            # Real-time anomaly detection
            anomaly_detected = await self._detect_real_time_anomaly(alert)
            
            # Real-time correlation analysis
            correlation_found = await self._analyze_real_time_correlation(alert)
            
            # Determine action
            action_taken = await self._determine_action(alert, anomaly_detected, correlation_found)
            
            # Calculate confidence
            confidence = await self._calculate_processing_confidence(
                alert, anomaly_detected, correlation_found
            )
            
            # Generate recommendations
            recommendations = await self._generate_real_time_recommendations(
                alert, anomaly_detected, correlation_found
            )
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                alert_id=alert.id,
                processing_time=processing_time,
                anomaly_detected=anomaly_detected,
                correlation_found=correlation_found,
                action_taken=action_taken,
                confidence=confidence,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error processing alert {alert.id}: {e}")
            return ProcessingResult(
                alert_id=alert.id,
                processing_time=time.time() - start_time,
                anomaly_detected=False,
                correlation_found=False,
                action_taken="error",
                confidence=0.0,
                recommendations=["Processing error occurred"]
            )
            
    async def _detect_real_time_anomaly(self, alert: StreamingAlert) -> bool:
        """Detect anomalies in real-time using sliding window analysis."""
        try:
            # Get recent metric history from Redis
            history_key = f"metric_history:{alert.tenant_id}:{alert.metric_name}"
            history_data = await self.redis_client.lrange(history_key, 0, 99)
            
            if len(history_data) < 10:
                # Not enough history for anomaly detection
                return False
                
            # Convert to numeric values
            values = []
            for data_point in history_data:
                try:
                    point = json.loads(data_point)
                    values.append(float(point['value']))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
                    
            if len(values) < 10:
                return False
                
            # Simple statistical anomaly detection
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # Z-score based detection
            z_score = abs(alert.value - mean_val) / max(std_val, 0.1)
            
            # Store current value in history
            await self._store_metric_value(alert)
            
            return z_score > 3.0  # 3-sigma rule
            
        except Exception as e:
            self.logger.error(f"Error in real-time anomaly detection: {e}")
            return False
            
    async def _analyze_real_time_correlation(self, alert: StreamingAlert) -> bool:
        """Analyze correlations with recent alerts in real-time."""
        try:
            # Get recent alerts from Redis
            recent_alerts_key = f"recent_alerts:{alert.tenant_id}"
            recent_data = await self.redis_client.lrange(recent_alerts_key, 0, 49)
            
            correlation_window = self.config.get('correlation_window_minutes', 5)
            cutoff_time = alert.timestamp - timedelta(minutes=correlation_window)
            
            correlated_count = 0
            
            for data_point in recent_data:
                try:
                    recent_alert = json.loads(data_point)
                    recent_timestamp = datetime.fromisoformat(recent_alert['timestamp'])
                    
                    if recent_timestamp >= cutoff_time:
                        # Check for correlation conditions
                        if (recent_alert['metric_name'] != alert.metric_name and
                            recent_alert['severity'] in ['warning', 'critical']):
                            correlated_count += 1
                            
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
                    
            # Store current alert in recent alerts
            await self._store_recent_alert(alert)
            
            # Consider correlated if multiple related alerts in time window
            return correlated_count >= 2
            
        except Exception as e:
            self.logger.error(f"Error in real-time correlation analysis: {e}")
            return False
            
    async def _store_metric_value(self, alert: StreamingAlert):
        """Store metric value in Redis for history."""
        history_key = f"metric_history:{alert.tenant_id}:{alert.metric_name}"
        
        data_point = {
            'value': alert.value,
            'timestamp': alert.timestamp.isoformat(),
            'threshold': alert.threshold
        }
        
        # Add to list and keep only last 100 values
        await self.redis_client.lpush(history_key, json.dumps(data_point))
        await self.redis_client.ltrim(history_key, 0, 99)
        await self.redis_client.expire(history_key, 3600)  # 1 hour TTL
        
    async def _store_recent_alert(self, alert: StreamingAlert):
        """Store alert in recent alerts for correlation analysis."""
        recent_alerts_key = f"recent_alerts:{alert.tenant_id}"
        
        alert_data = {
            'id': alert.id,
            'metric_name': alert.metric_name,
            'severity': alert.severity,
            'timestamp': alert.timestamp.isoformat(),
            'value': alert.value
        }
        
        # Add to list and keep only last 50 alerts
        await self.redis_client.lpush(recent_alerts_key, json.dumps(alert_data))
        await self.redis_client.ltrim(recent_alerts_key, 0, 49)
        await self.redis_client.expire(recent_alerts_key, 1800)  # 30 minutes TTL
        
    async def _determine_action(self, 
                              alert: StreamingAlert,
                              anomaly_detected: bool,
                              correlation_found: bool) -> str:
        """Determine what action to take based on analysis results."""
        if alert.severity == 'critical':
            if anomaly_detected and correlation_found:
                return "escalate_immediately"
            elif anomaly_detected or correlation_found:
                return "escalate_urgent"
            else:
                return "alert_critical"
        elif alert.severity == 'warning':
            if anomaly_detected and correlation_found:
                return "escalate_urgent"
            elif anomaly_detected or correlation_found:
                return "investigate"
            else:
                return "monitor"
        else:
            if anomaly_detected:
                return "investigate"
            else:
                return "log_only"
                
    async def _calculate_processing_confidence(self, 
                                             alert: StreamingAlert,
                                             anomaly_detected: bool,
                                             correlation_found: bool) -> float:
        """Calculate confidence in the processing results."""
        base_confidence = 0.5
        
        # Increase confidence based on analysis results
        if anomaly_detected:
            base_confidence += 0.3
            
        if correlation_found:
            base_confidence += 0.2
            
        # Adjust based on alert severity
        severity_weights = {'critical': 1.0, 'warning': 0.8, 'info': 0.6}
        severity_factor = severity_weights.get(alert.severity, 0.6)
        
        return min(1.0, base_confidence * severity_factor)
        
    async def _generate_real_time_recommendations(self, 
                                                alert: StreamingAlert,
                                                anomaly_detected: bool,
                                                correlation_found: bool) -> List[str]:
        """Generate real-time recommendations."""
        recommendations = []
        
        if anomaly_detected:
            recommendations.append("Anomaly detected - investigate metric behavior")
            recommendations.append("Check for recent system changes")
            
        if correlation_found:
            recommendations.append("Correlated alerts found - investigate system-wide issues")
            recommendations.append("Review infrastructure dependencies")
            
        if alert.severity == 'critical':
            recommendations.append("Critical alert - immediate attention required")
            recommendations.append("Follow incident response procedures")
            
        # Metric-specific recommendations
        metric_recommendations = {
            'cpu_usage': ["Check CPU-intensive processes", "Consider scaling resources"],
            'memory_usage': ["Investigate memory leaks", "Monitor garbage collection"],
            'disk_usage': ["Clean up disk space", "Archive old data"],
            'response_time': ["Optimize database queries", "Check network latency"]
        }
        
        for metric_type, advice in metric_recommendations.items():
            if metric_type in alert.metric_name.lower():
                recommendations.extend(advice)
                break
                
        return recommendations[:5]  # Limit to top 5 recommendations
        
    async def _update_processing_stats(self, 
                                     alerts: List[StreamingAlert],
                                     results: List[ProcessingResult],
                                     processing_time: float):
        """Update processing statistics."""
        valid_results = [r for r in results if isinstance(r, ProcessingResult)]
        
        self.processing_stats['total_processed'] += len(alerts)
        self.processing_stats['anomalies_detected'] += sum(
            1 for r in valid_results if r.anomaly_detected
        )
        self.processing_stats['correlations_found'] += sum(
            1 for r in valid_results if r.correlation_found
        )
        
        # Update average processing time
        if valid_results:
            avg_time = sum(r.processing_time for r in valid_results) / len(valid_results)
            self.processing_stats['avg_processing_time'] = (
                (self.processing_stats['avg_processing_time'] + avg_time) / 2
            )
            
        self.processing_stats['last_processed'] = datetime.now()
        
        # Store stats in Redis
        await self.redis_client.setex(
            'processing_stats',
            300,  # 5 minutes TTL
            json.dumps(self.processing_stats, default=str)
        )
        
    async def _publish_results(self, results: List[ProcessingResult]):
        """Publish processing results to output topics."""
        output_topic = self.config.get('output_topic', 'alert_results')
        
        for result in results:
            if isinstance(result, ProcessingResult):
                try:
                    await self.producer.send(
                        output_topic,
                        value=asdict(result)
                    )
                except Exception as e:
                    self.logger.error(f"Error publishing result: {e}")
                    
    async def _monitoring_task(self):
        """Background task for monitoring system health."""
        while self.is_running:
            try:
                # Log processing statistics
                stats = self.processing_stats.copy()
                self.logger.info(f"Processing stats: {stats}")
                
                # Check Redis connection
                await self.redis_client.ping()
                
                # Check Kafka connection
                if self.consumer and hasattr(self.consumer, '_client'):
                    # This is a simplified check
                    pass
                    
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in monitoring task: {e}")
                await asyncio.sleep(10)
                
    async def _cleanup_task(self):
        """Background task for cleaning up old data."""
        while self.is_running:
            try:
                # Clean up old metric history
                cursor = 0
                while True:
                    cursor, keys = await self.redis_client.scan(
                        cursor, match="metric_history:*", count=100
                    )
                    
                    for key in keys:
                        # Check if key is old
                        ttl = await self.redis_client.ttl(key)
                        if ttl < 0:  # No TTL set
                            await self.redis_client.expire(key, 3600)
                            
                    if cursor == 0:
                        break
                        
                # Clean up old alert data
                cursor = 0
                while True:
                    cursor, keys = await self.redis_client.scan(
                        cursor, match="recent_alerts:*", count=100
                    )
                    
                    for key in keys:
                        ttl = await self.redis_client.ttl(key)
                        if ttl < 0:
                            await self.redis_client.expire(key, 1800)
                            
                    if cursor == 0:
                        break
                        
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)
                
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        try:
            # Get from Redis for most up-to-date stats
            stats_data = await self.redis_client.get('processing_stats')
            if stats_data:
                return json.loads(stats_data)
        except Exception as e:
            self.logger.error(f"Error getting stats from Redis: {e}")
            
        return self.processing_stats.copy()
        
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of streaming components."""
        health = {
            'status': 'healthy',
            'components': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Check Redis
        try:
            await self.redis_client.ping()
            health['components']['redis'] = 'healthy'
        except Exception as e:
            health['components']['redis'] = f'unhealthy: {e}'
            health['status'] = 'degraded'
            
        # Check Kafka
        try:
            if self.consumer and self.producer:
                health['components']['kafka'] = 'healthy'
            else:
                health['components']['kafka'] = 'unhealthy: not connected'
                health['status'] = 'degraded'
        except Exception as e:
            health['components']['kafka'] = f'unhealthy: {e}'
            health['status'] = 'degraded'
            
        # Check processing
        if not self.is_running:
            health['components']['processor'] = 'stopped'
            health['status'] = 'unhealthy'
        else:
            health['components']['processor'] = 'running'
            
        return health
        
    async def cleanup(self):
        """Cleanup resources."""
        self.is_running = False
        
        if self.consumer:
            await self.consumer.stop()
            
        if self.producer:
            await self.producer.stop()
            
        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()
            
        self.executor.shutdown(wait=True)
        
        self.logger.info("Streaming processor cleanup completed")
