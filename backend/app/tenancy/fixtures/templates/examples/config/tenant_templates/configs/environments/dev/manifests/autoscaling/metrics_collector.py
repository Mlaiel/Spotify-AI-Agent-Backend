"""
Collecteur de métriques multi-source pour l'autoscaling intelligent
Support Prometheus, InfluxDB, CloudWatch, et métriques custom
Développé par l'équipe d'experts dirigée par Fahed Mlaiel
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp
import json
import numpy as np
from urllib.parse import urlencode
import boto3
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from prometheus_client.parser import text_string_to_metric_families

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Point de métrique avec timestamp"""
    timestamp: datetime
    value: float
    labels: Dict[str, str]

@dataclass
class MetricSeries:
    """Série de métriques"""
    name: str
    points: List[MetricPoint]
    unit: str
    source: str

class PrometheusCollector:
    """Collecteur de métriques Prometheus"""
    
    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url.rstrip('/')
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def query(self, query: str, time: Optional[datetime] = None) -> List[MetricPoint]:
        """Exécute une requête Prometheus"""
        params = {'query': query}
        if time:
            params['time'] = time.timestamp()
        
        url = f"{self.prometheus_url}/api/v1/query"
        
        try:
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                
                if data['status'] != 'success':
                    logger.error(f"Prometheus query failed: {data}")
                    return []
                
                return self._parse_prometheus_response(data['data'])
                
        except Exception as e:
            logger.error(f"Prometheus query error: {e}")
            return []
    
    async def query_range(self, query: str, start: datetime, end: datetime, 
                         step: str = "1m") -> List[MetricPoint]:
        """Exécute une requête Prometheus sur une plage"""
        params = {
            'query': query,
            'start': start.timestamp(),
            'end': end.timestamp(),
            'step': step
        }
        
        url = f"{self.prometheus_url}/api/v1/query_range"
        
        try:
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                
                if data['status'] != 'success':
                    logger.error(f"Prometheus range query failed: {data}")
                    return []
                
                return self._parse_prometheus_range_response(data['data'])
                
        except Exception as e:
            logger.error(f"Prometheus range query error: {e}")
            return []
    
    def _parse_prometheus_response(self, data: Dict) -> List[MetricPoint]:
        """Parse la réponse Prometheus instantanée"""
        points = []
        
        for result in data.get('result', []):
            labels = result.get('metric', {})
            value_data = result.get('value', [])
            
            if len(value_data) == 2:
                timestamp = datetime.fromtimestamp(float(value_data[0]))
                value = float(value_data[1])
                
                points.append(MetricPoint(
                    timestamp=timestamp,
                    value=value,
                    labels=labels
                ))
        
        return points
    
    def _parse_prometheus_range_response(self, data: Dict) -> List[MetricPoint]:
        """Parse la réponse Prometheus de plage"""
        points = []
        
        for result in data.get('result', []):
            labels = result.get('metric', {})
            values = result.get('values', [])
            
            for value_data in values:
                if len(value_data) == 2:
                    timestamp = datetime.fromtimestamp(float(value_data[0]))
                    value = float(value_data[1])
                    
                    points.append(MetricPoint(
                        timestamp=timestamp,
                        value=value,
                        labels=labels
                    ))
        
        return points

class InfluxDBCollector:
    """Collecteur de métriques InfluxDB"""
    
    def __init__(self, url: str, token: str, org: str):
        self.url = url
        self.token = token
        self.org = org
        self.client = None
    
    async def __aenter__(self):
        self.client = InfluxDBClientAsync(
            url=self.url,
            token=self.token,
            org=self.org
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.close()
    
    async def query(self, flux_query: str) -> List[MetricPoint]:
        """Exécute une requête Flux"""
        try:
            query_api = self.client.query_api()
            tables = await query_api.query(flux_query)
            
            points = []
            for table in tables:
                for record in table.records:
                    points.append(MetricPoint(
                        timestamp=record.get_time(),
                        value=float(record.get_value()),
                        labels=record.values
                    ))
            
            return points
            
        except Exception as e:
            logger.error(f"InfluxDB query error: {e}")
            return []

class CloudWatchCollector:
    """Collecteur de métriques AWS CloudWatch"""
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
    
    async def get_metric_statistics(self, namespace: str, metric_name: str,
                                  dimensions: List[Dict], start_time: datetime,
                                  end_time: datetime, period: int = 300,
                                  statistic: str = 'Average') -> List[MetricPoint]:
        """Récupère les statistiques CloudWatch"""
        try:
            response = self.cloudwatch.get_metric_statistics(
                Namespace=namespace,
                MetricName=metric_name,
                Dimensions=dimensions,
                StartTime=start_time,
                EndTime=end_time,
                Period=period,
                Statistics=[statistic]
            )
            
            points = []
            for datapoint in response['Datapoints']:
                points.append(MetricPoint(
                    timestamp=datapoint['Timestamp'],
                    value=float(datapoint[statistic]),
                    labels={'namespace': namespace, 'metric': metric_name}
                ))
            
            return sorted(points, key=lambda x: x.timestamp)
            
        except Exception as e:
            logger.error(f"CloudWatch query error: {e}")
            return []

class MetricsCollector:
    """Collecteur de métriques unifié multi-source"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prometheus_collector = None
        self.influxdb_collector = None
        self.cloudwatch_collector = None
        self.custom_metrics_cache: Dict[str, List[MetricPoint]] = {}
        self._initialize_collectors()
    
    def _initialize_collectors(self):
        """Initialise les collecteurs selon la configuration"""
        
        # Prometheus
        if 'prometheus' in self.config:
            prom_config = self.config['prometheus']
            self.prometheus_collector = PrometheusCollector(prom_config['url'])
        
        # InfluxDB
        if 'influxdb' in self.config:
            influx_config = self.config['influxdb']
            self.influxdb_collector = InfluxDBCollector(
                influx_config['url'],
                influx_config['token'],
                influx_config['org']
            )
        
        # CloudWatch
        if 'cloudwatch' in self.config:
            cw_config = self.config['cloudwatch']
            self.cloudwatch_collector = CloudWatchCollector(
                cw_config.get('region', 'us-east-1')
            )
    
    async def get_service_metrics(self, tenant_id: str, service_name: str,
                                namespace: str = "default") -> Dict[str, float]:
        """Collecte toutes les métriques d'un service"""
        
        metrics = {}
        
        # Métriques système de base
        system_metrics = await self._get_system_metrics(tenant_id, service_name, namespace)
        metrics.update(system_metrics)
        
        # Métriques application
        app_metrics = await self._get_application_metrics(tenant_id, service_name, namespace)
        metrics.update(app_metrics)
        
        # Métriques business/custom
        business_metrics = await self._get_business_metrics(tenant_id, service_name)
        metrics.update(business_metrics)
        
        return metrics
    
    async def _get_system_metrics(self, tenant_id: str, service_name: str,
                                namespace: str) -> Dict[str, float]:
        """Collecte les métriques système (CPU, mémoire, etc.)"""
        
        metrics = {}
        
        if not self.prometheus_collector:
            return metrics
        
        async with self.prometheus_collector:
            # CPU utilization
            cpu_query = f'rate(container_cpu_usage_seconds_total{{namespace="{namespace}",pod=~"{service_name}-.*"}}[5m]) * 100'
            cpu_points = await self.prometheus_collector.query(cpu_query)
            if cpu_points:
                metrics['cpu_utilization_percentage'] = np.mean([p.value for p in cpu_points])
            
            # Memory utilization
            memory_query = f'container_memory_usage_bytes{{namespace="{namespace}",pod=~"{service_name}-.*"}}'
            memory_points = await self.prometheus_collector.query(memory_query)
            if memory_points:
                metrics['memory_usage_bytes'] = np.mean([p.value for p in memory_points])
                
                # Memory utilization percentage
                memory_limit_query = f'container_spec_memory_limit_bytes{{namespace="{namespace}",pod=~"{service_name}-.*"}}'
                memory_limit_points = await self.prometheus_collector.query(memory_limit_query)
                if memory_limit_points:
                    avg_usage = np.mean([p.value for p in memory_points])
                    avg_limit = np.mean([p.value for p in memory_limit_points])
                    if avg_limit > 0:
                        metrics['memory_utilization_percentage'] = (avg_usage / avg_limit) * 100
            
            # Network I/O
            network_rx_query = f'rate(container_network_receive_bytes_total{{namespace="{namespace}",pod=~"{service_name}-.*"}}[5m])'
            network_rx_points = await self.prometheus_collector.query(network_rx_query)
            if network_rx_points:
                metrics['network_rx_bytes_per_sec'] = np.mean([p.value for p in network_rx_points])
            
            network_tx_query = f'rate(container_network_transmit_bytes_total{{namespace="{namespace}",pod=~"{service_name}-.*"}}[5m])'
            network_tx_points = await self.prometheus_collector.query(network_tx_query)
            if network_tx_points:
                metrics['network_tx_bytes_per_sec'] = np.mean([p.value for p in network_tx_points])
            
            # Disk I/O
            disk_read_query = f'rate(container_fs_reads_bytes_total{{namespace="{namespace}",pod=~"{service_name}-.*"}}[5m])'
            disk_read_points = await self.prometheus_collector.query(disk_read_query)
            if disk_read_points:
                metrics['disk_read_bytes_per_sec'] = np.mean([p.value for p in disk_read_points])
            
            disk_write_query = f'rate(container_fs_writes_bytes_total{{namespace="{namespace}",pod=~"{service_name}-.*"}}[5m])'
            disk_write_points = await self.prometheus_collector.query(disk_write_query)
            if disk_write_points:
                metrics['disk_write_bytes_per_sec'] = np.mean([p.value for p in disk_write_points])
        
        return metrics
    
    async def _get_application_metrics(self, tenant_id: str, service_name: str,
                                     namespace: str) -> Dict[str, float]:
        """Collecte les métriques application (requêtes, latence, erreurs)"""
        
        metrics = {}
        
        if not self.prometheus_collector:
            return metrics
        
        async with self.prometheus_collector:
            # Request rate
            request_rate_query = f'rate(http_requests_total{{service="{service_name}",tenant="{tenant_id}"}}[5m])'
            request_points = await self.prometheus_collector.query(request_rate_query)
            if request_points:
                metrics['requests_per_second'] = np.sum([p.value for p in request_points])
            
            # Response time / Latency
            latency_query = f'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{{service="{service_name}",tenant="{tenant_id}"}}[5m])) * 1000'
            latency_points = await self.prometheus_collector.query(latency_query)
            if latency_points:
                metrics['avg_response_time_ms'] = np.mean([p.value for p in latency_points])
            
            # Error rate
            error_rate_query = f'rate(http_requests_total{{service="{service_name}",tenant="{tenant_id}",status=~"5.."}}[5m])'
            error_points = await self.prometheus_collector.query(error_rate_query)
            if error_points and request_points:
                total_errors = np.sum([p.value for p in error_points])
                total_requests = np.sum([p.value for p in request_points])
                if total_requests > 0:
                    metrics['error_rate_percentage'] = (total_errors / total_requests) * 100
            
            # Active connections
            connections_query = f'http_requests_in_flight{{service="{service_name}",tenant="{tenant_id}"}}'
            conn_points = await self.prometheus_collector.query(connections_query)
            if conn_points:
                metrics['active_connections'] = np.mean([p.value for p in conn_points])
        
        return metrics
    
    async def _get_business_metrics(self, tenant_id: str, service_name: str) -> Dict[str, float]:
        """Collecte les métriques métier spécifiques au Spotify AI Agent"""
        
        metrics = {}
        
        if not self.prometheus_collector:
            return metrics
        
        async with self.prometheus_collector:
            # Audio processing queue length
            if service_name == "audio-processor":
                queue_query = f'audio_processing_queue_length{{tenant="{tenant_id}"}}'
                queue_points = await self.prometheus_collector.query(queue_query)
                if queue_points:
                    metrics['audio_queue_length'] = np.mean([p.value for p in queue_points])
                
                # Audio processing time
                processing_time_query = f'histogram_quantile(0.95, rate(audio_processing_duration_seconds_bucket{{tenant="{tenant_id}"}}[5m])) * 1000'
                processing_points = await self.prometheus_collector.query(processing_time_query)
                if processing_points:
                    metrics['audio_processing_time_ms'] = np.mean([p.value for p in processing_points])
            
            # ML model inference metrics
            if service_name == "ml-service":
                inference_query = f'rate(ml_inference_total{{tenant="{tenant_id}"}}[5m])'
                inference_points = await self.prometheus_collector.query(inference_query)
                if inference_points:
                    metrics['ml_inferences_per_second'] = np.sum([p.value for p in inference_points])
                
                # Model accuracy
                accuracy_query = f'ml_model_accuracy{{tenant="{tenant_id}"}}'
                accuracy_points = await self.prometheus_collector.query(accuracy_query)
                if accuracy_points:
                    metrics['ml_model_accuracy'] = np.mean([p.value for p in accuracy_points])
                
                # GPU utilization (si applicable)
                gpu_query = f'nvidia_gpu_utilization_percentage{{tenant="{tenant_id}"}}'
                gpu_points = await self.prometheus_collector.query(gpu_query)
                if gpu_points:
                    metrics['gpu_utilization_percentage'] = np.mean([p.value for p in gpu_points])
            
            # User session metrics
            sessions_query = f'active_user_sessions{{tenant="{tenant_id}"}}'
            session_points = await self.prometheus_collector.query(sessions_query)
            if session_points:
                metrics['active_user_sessions'] = np.mean([p.value for p in session_points])
            
            # Database connection pool
            db_pool_query = f'database_connections_active{{tenant="{tenant_id}",service="{service_name}"}}'
            db_points = await self.prometheus_collector.query(db_pool_query)
            if db_points:
                metrics['db_connections_active'] = np.mean([p.value for p in db_points])
        
        return metrics
    
    async def get_historical_metrics(self, tenant_id: str, service_name: str,
                                   start_time: datetime, end_time: datetime) -> List[Tuple[datetime, Dict[str, float]]]:
        """Collecte les métriques historiques sur une période"""
        
        historical_data = []
        
        if not self.prometheus_collector:
            return historical_data
        
        # Intervalle de collecte (5 minutes)
        step = "5m"
        
        async with self.prometheus_collector:
            # Requêtes principales pour les métriques historiques
            queries = {
                'cpu_usage_cores': f'rate(container_cpu_usage_seconds_total{{pod=~"{service_name}-.*"}}[5m])',
                'memory_usage_bytes': f'container_memory_usage_bytes{{pod=~"{service_name}-.*"}}',
                'requests_per_second': f'rate(http_requests_total{{service="{service_name}",tenant="{tenant_id}"}}[5m])',
                'response_time_ms': f'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{{service="{service_name}",tenant="{tenant_id}"}}[5m])) * 1000'
            }
            
            # Collecte pour chaque métrique
            metrics_data = {}
            for metric_name, query in queries.items():
                points = await self.prometheus_collector.query_range(query, start_time, end_time, step)
                
                # Groupement par timestamp
                for point in points:
                    timestamp = point.timestamp
                    if timestamp not in metrics_data:
                        metrics_data[timestamp] = {}
                    
                    if metric_name not in metrics_data[timestamp]:
                        metrics_data[timestamp][metric_name] = []
                    metrics_data[timestamp][metric_name].append(point.value)
            
            # Agrégation et création de la liste finale
            for timestamp in sorted(metrics_data.keys()):
                aggregated_metrics = {}
                for metric_name, values in metrics_data[timestamp].items():
                    if values:
                        aggregated_metrics[metric_name] = np.mean(values)
                
                if aggregated_metrics:
                    historical_data.append((timestamp, aggregated_metrics))
        
        return historical_data
    
    async def add_custom_metric(self, metric_name: str, value: float, 
                              labels: Dict[str, str], timestamp: Optional[datetime] = None):
        """Ajoute une métrique custom au cache"""
        
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        point = MetricPoint(
            timestamp=timestamp,
            value=value,
            labels=labels
        )
        
        if metric_name not in self.custom_metrics_cache:
            self.custom_metrics_cache[metric_name] = []
        
        self.custom_metrics_cache[metric_name].append(point)
        
        # Nettoyage du cache (garde 1 heure)
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        self.custom_metrics_cache[metric_name] = [
            p for p in self.custom_metrics_cache[metric_name]
            if p.timestamp > cutoff_time
        ]
    
    async def get_custom_metrics(self, metric_name: str, 
                               time_range: Optional[Tuple[datetime, datetime]] = None) -> List[MetricPoint]:
        """Récupère les métriques custom du cache"""
        
        if metric_name not in self.custom_metrics_cache:
            return []
        
        points = self.custom_metrics_cache[metric_name]
        
        if time_range:
            start_time, end_time = time_range
            points = [
                p for p in points
                if start_time <= p.timestamp <= end_time
            ]
        
        return sorted(points, key=lambda x: x.timestamp)
    
    def get_metrics_summary(self, tenant_id: str) -> Dict[str, Any]:
        """Génère un résumé des métriques collectées"""
        
        summary = {
            'collectors_status': {
                'prometheus': self.prometheus_collector is not None,
                'influxdb': self.influxdb_collector is not None,
                'cloudwatch': self.cloudwatch_collector is not None
            },
            'custom_metrics_count': len(self.custom_metrics_cache),
            'last_collection': datetime.utcnow().isoformat(),
            'tenant_id': tenant_id
        }
        
        return summary
    
    async def validate_metrics_availability(self) -> Dict[str, bool]:
        """Valide la disponibilité des sources de métriques"""
        
        status = {}
        
        # Test Prometheus
        if self.prometheus_collector:
            try:
                async with self.prometheus_collector:
                    test_points = await self.prometheus_collector.query('up')
                    status['prometheus'] = len(test_points) > 0
            except Exception as e:
                logger.error(f"Prometheus validation failed: {e}")
                status['prometheus'] = False
        else:
            status['prometheus'] = False
        
        # Test InfluxDB
        if self.influxdb_collector:
            try:
                async with self.influxdb_collector:
                    # Test query simple
                    test_points = await self.influxdb_collector.query('from(bucket: "test") |> range(start: -1m) |> limit(n: 1)')
                    status['influxdb'] = True
            except Exception as e:
                logger.error(f"InfluxDB validation failed: {e}")
                status['influxdb'] = False
        else:
            status['influxdb'] = False
        
        # Test CloudWatch
        if self.cloudwatch_collector:
            try:
                # Test simple avec CloudWatch
                test_metrics = await self.cloudwatch_collector.get_metric_statistics(
                    namespace='AWS/EC2',
                    metric_name='CPUUtilization',
                    dimensions=[],
                    start_time=datetime.utcnow() - timedelta(minutes=5),
                    end_time=datetime.utcnow()
                )
                status['cloudwatch'] = True
            except Exception as e:
                logger.error(f"CloudWatch validation failed: {e}")
                status['cloudwatch'] = False
        else:
            status['cloudwatch'] = False
        
        return status
