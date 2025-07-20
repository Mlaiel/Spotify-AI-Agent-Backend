"""
Advanced InfluxDB Multi-Tenant Time Series Exporter
==================================================

Exportateur haute performance pour InfluxDB avec support multi-tenant,
optimisé pour les métriques temporelles et l'analytics en temps réel.

Fonctionnalités:
- Time series haute performance
- Downsampling automatique
- Rétention intelligente
- Continuous queries
- Flux queries avancées
- Alerting intégré
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from influxdb_client import Point, WritePrecision
from influxdb_client.client.write_api_async import WriteApiAsync
from influxdb_client.client.query_api_async import QueryApiAsync
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class InfluxDBConfig:
    """Configuration pour InfluxDB multi-tenant."""
    url: str = "http://localhost:8086"
    token: str = ""
    org: str = "spotify-ai"
    timeout: int = 30000
    enable_gzip: bool = True
    verify_ssl: bool = True


@dataclass
class TenantBucketConfig:
    """Configuration des buckets pour un tenant."""
    tenant_id: str
    bucket_prefix: str = "spotify-ai"
    retention_period: str = "90d"
    replication: int = 1
    shard_group_duration: str = "1d"
    enable_downsampling: bool = True
    downsample_rules: List[Dict[str, str]] = field(default_factory=lambda: [
        {"every": "1h", "for": "7d", "aggregation": "mean"},
        {"every": "1d", "for": "30d", "aggregation": "mean"},
        {"every": "1w", "for": "1y", "aggregation": "mean"}
    ])


@dataclass
class TimeSeriesPoint:
    """Point de données time series."""
    measurement: str
    tags: Dict[str, str]
    fields: Dict[str, Union[int, float, str, bool]]
    timestamp: Optional[datetime] = None
    precision: WritePrecision = WritePrecision.NS


class InfluxDBMetricsExporter:
    """
    Exportateur InfluxDB avancé avec support multi-tenant.
    
    Fonctionnalités:
    - Buckets isolés par tenant
    - Downsampling automatique
    - Queries Flux optimisées
    - Continuous queries
    - Monitoring des performances
    """
    
    def __init__(
        self,
        config: InfluxDBConfig,
        bucket_config: TenantBucketConfig
    ):
        self.config = config
        self.bucket_config = bucket_config
        self.client: Optional[InfluxDBClientAsync] = None
        self.write_api: Optional[WriteApiAsync] = None
        self.query_api: Optional[QueryApiAsync] = None
        
        # Noms des buckets
        self.buckets = {
            'raw': f"{bucket_config.bucket_prefix}-{bucket_config.tenant_id}-raw",
            'hourly': f"{bucket_config.bucket_prefix}-{bucket_config.tenant_id}-hourly",
            'daily': f"{bucket_config.bucket_prefix}-{bucket_config.tenant_id}-daily",
            'weekly': f"{bucket_config.bucket_prefix}-{bucket_config.tenant_id}-weekly"
        }
        
        # Métriques internes
        self.stats = {
            'points_written': 0,
            'queries_executed': 0,
            'buckets_created': 0,
            'downsampling_tasks': 0,
            'errors': 0
        }
        
        # Cache des queries
        self.query_cache = {}
        
    async def initialize(self):
        """Initialise l'exportateur InfluxDB."""
        try:
            # Créer le client InfluxDB
            self.client = InfluxDBClientAsync(
                url=self.config.url,
                token=self.config.token,
                org=self.config.org,
                timeout=self.config.timeout,
                enable_gzip=self.config.enable_gzip,
                verify_ssl=self.config.verify_ssl
            )
            
            # APIs
            self.write_api = self.client.write_api()
            self.query_api = self.client.query_api()
            
            # Vérifier la connexion
            await self._check_influxdb_connection()
            
            # Créer les buckets
            await self._setup_tenant_buckets()
            
            # Configurer le downsampling
            if self.bucket_config.enable_downsampling:
                await self._setup_downsampling_tasks()
                
            # Configurer les continuous queries
            await self._setup_continuous_queries()
            
            logger.info(
                "InfluxDBMetricsExporter initialized successfully",
                tenant_id=self.bucket_config.tenant_id
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize InfluxDB exporter: {e}")
            raise
            
    async def _check_influxdb_connection(self):
        """Vérifie la connexion à InfluxDB."""
        health = await self.client.health()
        if health.status != "pass":
            raise Exception(f"InfluxDB health check failed: {health.status}")
            
        logger.info("Connected to InfluxDB successfully")
        
    async def _setup_tenant_buckets(self):
        """Configure les buckets pour le tenant."""
        buckets_api = self.client.buckets_api()
        
        for bucket_type, bucket_name in self.buckets.items():
            try:
                # Vérifier si le bucket existe
                existing_bucket = await buckets_api.find_bucket_by_name(bucket_name)
                
                if not existing_bucket:
                    # Déterminer la rétention selon le type
                    if bucket_type == 'raw':
                        retention = self.bucket_config.retention_period
                    elif bucket_type == 'hourly':
                        retention = "1y"
                    elif bucket_type == 'daily':
                        retention = "5y"
                    else:  # weekly
                        retention = "10y"
                        
                    # Créer le bucket
                    bucket = await buckets_api.create_bucket(
                        bucket_name=bucket_name,
                        org=self.config.org,
                        retention_rules=[{
                            "type": "expire",
                            "everySeconds": self._parse_duration(retention)
                        }],
                        description=f"Spotify AI metrics for tenant {self.bucket_config.tenant_id} - {bucket_type}"
                    )
                    
                    self.stats['buckets_created'] += 1
                    
                    logger.info(
                        f"Created bucket: {bucket_name}",
                        tenant_id=self.bucket_config.tenant_id,
                        retention=retention
                    )
                    
            except Exception as e:
                logger.error(
                    f"Failed to setup bucket {bucket_name}: {e}",
                    tenant_id=self.bucket_config.tenant_id
                )
                raise
                
    def _parse_duration(self, duration_str: str) -> int:
        """Parse une durée en secondes."""
        unit_multipliers = {
            's': 1,
            'm': 60,
            'h': 3600,
            'd': 86400,
            'w': 604800,
            'y': 31536000
        }
        
        try:
            if duration_str[-1] in unit_multipliers:
                value = int(duration_str[:-1])
                unit = duration_str[-1]
                return value * unit_multipliers[unit]
            else:
                return int(duration_str)
        except (ValueError, IndexError):
            return 86400  # Default to 1 day
            
    async def _setup_downsampling_tasks(self):
        """Configure les tâches de downsampling."""
        tasks_api = self.client.tasks_api()
        
        for rule in self.bucket_config.downsample_rules:
            every = rule['every']
            for_duration = rule['for']
            aggregation = rule['aggregation']
            
            # Déterminer le bucket source et destination
            if every == "1h":
                source_bucket = self.buckets['raw']
                dest_bucket = self.buckets['hourly']
            elif every == "1d":
                source_bucket = self.buckets['hourly']
                dest_bucket = self.buckets['daily']
            elif every == "1w":
                source_bucket = self.buckets['daily']
                dest_bucket = self.buckets['weekly']
            else:
                continue
                
            # Flux query pour le downsampling
            flux_query = f'''
                from(bucket: "{source_bucket}")
                  |> range(start: -{for_duration})
                  |> filter(fn: (r) => r._measurement != "")
                  |> aggregateWindow(every: {every}, fn: {aggregation}, createEmpty: false)
                  |> to(bucket: "{dest_bucket}")
            '''
            
            task_name = f"downsample_{self.bucket_config.tenant_id}_{every}"
            
            try:
                # Créer la tâche de downsampling
                task = await tasks_api.create_task(
                    name=task_name,
                    flux=flux_query,
                    every=every,
                    org=self.config.org,
                    description=f"Downsample {every} for tenant {self.bucket_config.tenant_id}"
                )
                
                self.stats['downsampling_tasks'] += 1
                
                logger.info(
                    f"Created downsampling task: {task_name}",
                    tenant_id=self.bucket_config.tenant_id,
                    every=every
                )
                
            except Exception as e:
                logger.warning(
                    f"Failed to create downsampling task {task_name}: {e}",
                    tenant_id=self.bucket_config.tenant_id
                )
                
    async def _setup_continuous_queries(self):
        """Configure les continuous queries pour l'analytics en temps réel."""
        
        # Query pour calculer les moyennes mobiles
        moving_average_query = f'''
            from(bucket: "{self.buckets['raw']}")
              |> range(start: -1h)
              |> filter(fn: (r) => r._measurement == "ai_inference_time")
              |> filter(fn: (r) => r.tenant_id == "{self.bucket_config.tenant_id}")
              |> movingAverage(n: 10)
              |> set(key: "_measurement", value: "ai_inference_time_ma")
              |> to(bucket: "{self.buckets['raw']}")
        '''
        
        # Query pour détecter les anomalies
        anomaly_detection_query = f'''
            import "experimental/anomalydetection"
            
            from(bucket: "{self.buckets['raw']}")
              |> range(start: -1h)
              |> filter(fn: (r) => r._measurement == "ai_inference_time")
              |> filter(fn: (r) => r.tenant_id == "{self.bucket_config.tenant_id}")
              |> anomalydetection.mad(threshold: 3.0)
              |> filter(fn: (r) => r._value > 0)
              |> set(key: "_measurement", value: "ai_anomalies")
              |> to(bucket: "{self.buckets['raw']}")
        '''
        
        # Ces queries seraient normalement configurées comme des tâches
        # Pour la démonstration, on les stocke pour utilisation ultérieure
        self.continuous_queries = {
            'moving_average': moving_average_query,
            'anomaly_detection': anomaly_detection_query
        }
        
    async def write_ai_metrics(self, metrics: List[Dict[str, Any]]):
        """
        Écrit des métriques IA dans InfluxDB.
        
        Args:
            metrics: Liste des métriques IA à écrire
        """
        try:
            points = []
            
            for metric in metrics:
                # Point pour le temps d'inférence
                if 'inference_time' in metric:
                    point = Point("ai_inference_time") \
                        .tag("tenant_id", self.bucket_config.tenant_id) \
                        .tag("model_name", metric.get('model_name', 'unknown')) \
                        .tag("model_version", metric.get('model_version', 'v1.0')) \
                        .tag("algorithm", metric.get('algorithm', 'unknown')) \
                        .tag("tensor_size", metric.get('tensor_size', 'medium')) \
                        .field("value", float(metric['inference_time'])) \
                        .field("accuracy", float(metric.get('accuracy', 0))) \
                        .time(metric.get('timestamp', datetime.utcnow()), WritePrecision.NS)
                    points.append(point)
                    
                # Point pour la précision des recommandations
                if 'recommendation_accuracy' in metric:
                    point = Point("recommendation_accuracy") \
                        .tag("tenant_id", self.bucket_config.tenant_id) \
                        .tag("algorithm", metric.get('algorithm', 'collaborative')) \
                        .tag("dataset_version", metric.get('dataset_version', 'latest')) \
                        .field("accuracy", float(metric['recommendation_accuracy'])) \
                        .field("confidence", float(metric.get('confidence', 0.8))) \
                        .time(metric.get('timestamp', datetime.utcnow()), WritePrecision.NS)
                    points.append(point)
                    
                # Point pour l'engagement utilisateur
                if 'user_engagement' in metric:
                    point = Point("user_engagement") \
                        .tag("tenant_id", self.bucket_config.tenant_id) \
                        .tag("artist_tier", metric.get('artist_tier', 'emerging')) \
                        .tag("region", metric.get('region', 'global')) \
                        .field("score", float(metric['user_engagement'])) \
                        .field("interactions", int(metric.get('interactions', 0))) \
                        .time(metric.get('timestamp', datetime.utcnow()), WritePrecision.NS)
                    points.append(point)
                    
            # Écriture en lot
            await self.write_api.write(
                bucket=self.buckets['raw'],
                org=self.config.org,
                record=points
            )
            
            self.stats['points_written'] += len(points)
            
            logger.info(
                f"Wrote {len(points)} AI metric points",
                tenant_id=self.bucket_config.tenant_id,
                bucket=self.buckets['raw']
            )
            
        except Exception as e:
            logger.error(
                f"Failed to write AI metrics: {e}",
                tenant_id=self.bucket_config.tenant_id
            )
            self.stats['errors'] += 1
            raise
            
    async def write_business_metrics(self, metrics: List[Dict[str, Any]]):
        """
        Écrit des métriques business dans InfluxDB.
        
        Args:
            metrics: Liste des métriques business à écrire
        """
        try:
            points = []
            
            for metric in metrics:
                timestamp = metric.get('timestamp', datetime.utcnow())
                
                # Point pour les pistes générées
                if 'tracks_generated' in metric:
                    point = Point("tracks_generated") \
                        .tag("tenant_id", self.bucket_config.tenant_id) \
                        .tag("genre", metric.get('genre', 'pop')) \
                        .tag("collaboration_type", metric.get('collaboration_type', 'single')) \
                        .tag("artist_tier", metric.get('artist_tier', 'emerging')) \
                        .field("count", int(metric['tracks_generated'])) \
                        .field("duration_avg", float(metric.get('avg_duration', 0))) \
                        .time(timestamp, WritePrecision.NS)
                    points.append(point)
                    
                # Point pour l'impact revenue
                if 'revenue_impact' in metric:
                    point = Point("revenue_impact") \
                        .tag("tenant_id", self.bucket_config.tenant_id) \
                        .tag("stream_type", metric.get('stream_type', 'premium')) \
                        .tag("monetization_channel", metric.get('channel', 'streaming')) \
                        .tag("region", metric.get('region', 'global')) \
                        .field("euros", float(metric['revenue_impact'])) \
                        .field("streams", int(metric.get('streams', 0))) \
                        .time(timestamp, WritePrecision.NS)
                    points.append(point)
                    
                # Point pour les collaborations
                if 'collaboration_success_rate' in metric:
                    point = Point("collaboration_success") \
                        .tag("tenant_id", self.bucket_config.tenant_id) \
                        .tag("collaboration_type", metric.get('collaboration_type', 'ai_assisted')) \
                        .tag("genre_match", metric.get('genre_match', 'high')) \
                        .field("success_rate", float(metric['collaboration_success_rate'])) \
                        .field("participants", int(metric.get('participants', 2))) \
                        .field("duration_days", int(metric.get('duration_days', 7))) \
                        .time(timestamp, WritePrecision.NS)
                    points.append(point)
                    
                # Point pour l'engagement des artistes
                if 'artist_engagement' in metric:
                    point = Point("artist_engagement") \
                        .tag("tenant_id", self.bucket_config.tenant_id) \
                        .tag("artist_id", metric.get('artist_id', 'unknown')) \
                        .tag("genre", metric.get('genre', 'pop')) \
                        .field("engagement_score", float(metric['artist_engagement'])) \
                        .field("followers_growth", int(metric.get('followers_growth', 0))) \
                        .time(timestamp, WritePrecision.NS)
                    points.append(point)
                    
            # Écriture en lot
            await self.write_api.write(
                bucket=self.buckets['raw'],
                org=self.config.org,
                record=points
            )
            
            self.stats['points_written'] += len(points)
            
            logger.info(
                f"Wrote {len(points)} business metric points",
                tenant_id=self.bucket_config.tenant_id,
                bucket=self.buckets['raw']
            )
            
        except Exception as e:
            logger.error(
                f"Failed to write business metrics: {e}",
                tenant_id=self.bucket_config.tenant_id
            )
            self.stats['errors'] += 1
            raise
            
    async def query_ai_performance(
        self,
        time_range: str = "-1h",
        aggregation_window: str = "5m"
    ) -> Dict[str, Any]:
        """
        Query les performances des modèles IA.
        
        Args:
            time_range: Période de query (ex: "-1h", "-1d")
            aggregation_window: Fenêtre d'agrégation
            
        Returns:
            Résultats de performance
        """
        try:
            # Query pour les temps d'inférence
            inference_query = f'''
                from(bucket: "{self.buckets['raw']}")
                  |> range(start: {time_range})
                  |> filter(fn: (r) => r._measurement == "ai_inference_time")
                  |> filter(fn: (r) => r.tenant_id == "{self.bucket_config.tenant_id}")
                  |> aggregateWindow(every: {aggregation_window}, fn: mean, createEmpty: false)
                  |> group(columns: ["model_name", "algorithm"])
                  |> mean()
            '''
            
            # Query pour la précision
            accuracy_query = f'''
                from(bucket: "{self.buckets['raw']}")
                  |> range(start: {time_range})
                  |> filter(fn: (r) => r._measurement == "recommendation_accuracy")
                  |> filter(fn: (r) => r.tenant_id == "{self.bucket_config.tenant_id}")
                  |> aggregateWindow(every: {aggregation_window}, fn: mean, createEmpty: false)
                  |> group(columns: ["algorithm"])
                  |> mean()
            '''
            
            # Exécuter les queries
            inference_results = await self.query_api.query(inference_query, org=self.config.org)
            accuracy_results = await self.query_api.query(accuracy_query, org=self.config.org)
            
            self.stats['queries_executed'] += 2
            
            # Formater les résultats
            performance_data = {
                "inference_times": [],
                "accuracy_scores": [],
                "summary": {
                    "avg_inference_time": 0,
                    "avg_accuracy": 0,
                    "models_count": 0
                }
            }
            
            # Traiter les résultats d'inférence
            inference_sum = 0
            inference_count = 0
            models = set()
            
            for table in inference_results:
                for record in table.records:
                    performance_data["inference_times"].append({
                        "model_name": record.values.get("model_name"),
                        "algorithm": record.values.get("algorithm"),
                        "avg_inference_time": record.get_value(),
                        "timestamp": record.get_time()
                    })
                    inference_sum += record.get_value()
                    inference_count += 1
                    models.add(record.values.get("model_name"))
                    
            # Traiter les résultats de précision
            accuracy_sum = 0
            accuracy_count = 0
            
            for table in accuracy_results:
                for record in table.records:
                    performance_data["accuracy_scores"].append({
                        "algorithm": record.values.get("algorithm"),
                        "avg_accuracy": record.get_value(),
                        "timestamp": record.get_time()
                    })
                    accuracy_sum += record.get_value()
                    accuracy_count += 1
                    
            # Calculer les moyennes
            if inference_count > 0:
                performance_data["summary"]["avg_inference_time"] = inference_sum / inference_count
            if accuracy_count > 0:
                performance_data["summary"]["avg_accuracy"] = accuracy_sum / accuracy_count
            performance_data["summary"]["models_count"] = len(models)
            
            return performance_data
            
        except Exception as e:
            logger.error(
                f"Failed to query AI performance: {e}",
                tenant_id=self.bucket_config.tenant_id
            )
            self.stats['errors'] += 1
            raise
            
    async def query_business_impact(
        self,
        time_range: str = "-24h",
        group_by: str = "genre"
    ) -> Dict[str, Any]:
        """
        Query l'impact business.
        
        Args:
            time_range: Période de query
            group_by: Critère de groupement (genre, region, collaboration_type)
            
        Returns:
            Résultats business
        """
        try:
            # Query pour les pistes générées
            tracks_query = f'''
                from(bucket: "{self.buckets['raw']}")
                  |> range(start: {time_range})
                  |> filter(fn: (r) => r._measurement == "tracks_generated")
                  |> filter(fn: (r) => r.tenant_id == "{self.bucket_config.tenant_id}")
                  |> group(columns: ["{group_by}"])
                  |> sum(column: "_value")
            '''
            
            # Query pour l'impact revenue
            revenue_query = f'''
                from(bucket: "{self.buckets['raw']}")
                  |> range(start: {time_range})
                  |> filter(fn: (r) => r._measurement == "revenue_impact")
                  |> filter(fn: (r) => r.tenant_id == "{self.bucket_config.tenant_id}")
                  |> group(columns: ["{group_by}"])
                  |> sum(column: "_value")
            '''
            
            # Query pour le succès des collaborations
            collaboration_query = f'''
                from(bucket: "{self.buckets['raw']}")
                  |> range(start: {time_range})
                  |> filter(fn: (r) => r._measurement == "collaboration_success")
                  |> filter(fn: (r) => r.tenant_id == "{self.bucket_config.tenant_id}")
                  |> group(columns: ["collaboration_type"])
                  |> mean(column: "_value")
            '''
            
            # Exécuter les queries
            tracks_results = await self.query_api.query(tracks_query, org=self.config.org)
            revenue_results = await self.query_api.query(revenue_query, org=self.config.org)
            collaboration_results = await self.query_api.query(collaboration_query, org=self.config.org)
            
            self.stats['queries_executed'] += 3
            
            # Formater les résultats
            business_data = {
                "tracks_by_group": {},
                "revenue_by_group": {},
                "collaboration_success": {},
                "totals": {
                    "total_tracks": 0,
                    "total_revenue": 0,
                    "avg_collaboration_success": 0
                }
            }
            
            # Traiter les résultats de pistes
            total_tracks = 0
            for table in tracks_results:
                for record in table.records:
                    group_value = record.values.get(group_by, "unknown")
                    tracks_count = record.get_value()
                    business_data["tracks_by_group"][group_value] = tracks_count
                    total_tracks += tracks_count
                    
            # Traiter les résultats de revenue
            total_revenue = 0
            for table in revenue_results:
                for record in table.records:
                    group_value = record.values.get(group_by, "unknown")
                    revenue_amount = record.get_value()
                    business_data["revenue_by_group"][group_value] = revenue_amount
                    total_revenue += revenue_amount
                    
            # Traiter les résultats de collaboration
            collaboration_rates = []
            for table in collaboration_results:
                for record in table.records:
                    collab_type = record.values.get("collaboration_type", "unknown")
                    success_rate = record.get_value()
                    business_data["collaboration_success"][collab_type] = success_rate
                    collaboration_rates.append(success_rate)
                    
            # Calculer les totaux
            business_data["totals"]["total_tracks"] = total_tracks
            business_data["totals"]["total_revenue"] = total_revenue
            if collaboration_rates:
                business_data["totals"]["avg_collaboration_success"] = sum(collaboration_rates) / len(collaboration_rates)
                
            return business_data
            
        except Exception as e:
            logger.error(
                f"Failed to query business impact: {e}",
                tenant_id=self.bucket_config.tenant_id
            )
            self.stats['errors'] += 1
            raise
            
    async def detect_anomalies(
        self,
        measurement: str,
        time_range: str = "-1h",
        threshold: float = 3.0
    ) -> List[Dict[str, Any]]:
        """
        Détecte les anomalies dans les métriques.
        
        Args:
            measurement: Nom de la mesure à analyser
            time_range: Période d'analyse
            threshold: Seuil de détection (écarts-types)
            
        Returns:
            Liste des anomalies détectées
        """
        try:
            # Query de détection d'anomalies avec MAD (Median Absolute Deviation)
            anomaly_query = f'''
                import "experimental/anomalydetection"
                
                from(bucket: "{self.buckets['raw']}")
                  |> range(start: {time_range})
                  |> filter(fn: (r) => r._measurement == "{measurement}")
                  |> filter(fn: (r) => r.tenant_id == "{self.bucket_config.tenant_id}")
                  |> anomalydetection.mad(threshold: {threshold})
                  |> filter(fn: (r) => r._value > 0)
                  |> yield(name: "anomalies")
            '''
            
            results = await self.query_api.query(anomaly_query, org=self.config.org)
            self.stats['queries_executed'] += 1
            
            anomalies = []
            for table in results:
                for record in table.records:
                    anomaly = {
                        "timestamp": record.get_time(),
                        "measurement": measurement,
                        "value": record.get_value(),
                        "tags": {k: v for k, v in record.values.items() 
                                if k not in ['_time', '_value', '_field', '_measurement', 'result', 'table']},
                        "severity": "high" if record.get_value() > threshold * 2 else "medium"
                    }
                    anomalies.append(anomaly)
                    
            logger.info(
                f"Detected {len(anomalies)} anomalies in {measurement}",
                tenant_id=self.bucket_config.tenant_id,
                time_range=time_range
            )
            
            return anomalies
            
        except Exception as e:
            logger.error(
                f"Failed to detect anomalies: {e}",
                tenant_id=self.bucket_config.tenant_id,
                measurement=measurement
            )
            self.stats['errors'] += 1
            raise
            
    async def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Retourne les données pour un dashboard temps réel."""
        try:
            # Données des 5 dernières minutes
            recent_ai_data = await self.query_ai_performance("-5m", "1m")
            recent_business_data = await self.query_business_impact("-1h", "genre")
            
            # Détection d'anomalies récentes
            inference_anomalies = await self.detect_anomalies("ai_inference_time", "-15m")
            accuracy_anomalies = await self.detect_anomalies("recommendation_accuracy", "-15m")
            
            dashboard_data = {
                "timestamp": datetime.now().isoformat(),
                "tenant_id": self.bucket_config.tenant_id,
                "ai_performance": recent_ai_data,
                "business_metrics": recent_business_data,
                "anomalies": {
                    "inference_time": inference_anomalies,
                    "accuracy": accuracy_anomalies
                },
                "health_status": "healthy" if len(inference_anomalies + accuracy_anomalies) == 0 else "warning",
                "stats": self.stats
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(
                f"Failed to get real-time dashboard data: {e}",
                tenant_id=self.bucket_config.tenant_id
            )
            raise
            
    async def export_tenant_data(
        self,
        time_range: str = "-30d",
        bucket_type: str = "raw"
    ) -> Dict[str, Any]:
        """
        Exporte toutes les données d'un tenant.
        
        Args:
            time_range: Période à exporter
            bucket_type: Type de bucket (raw, hourly, daily, weekly)
            
        Returns:
            Données exportées
        """
        try:
            bucket_name = self.buckets[bucket_type]
            
            export_query = f'''
                from(bucket: "{bucket_name}")
                  |> range(start: {time_range})
                  |> filter(fn: (r) => r.tenant_id == "{self.bucket_config.tenant_id}")
                  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            results = await self.query_api.query(export_query, org=self.config.org)
            self.stats['queries_executed'] += 1
            
            export_data = {
                "tenant_id": self.bucket_config.tenant_id,
                "export_timestamp": datetime.now().isoformat(),
                "time_range": time_range,
                "bucket_type": bucket_type,
                "data": []
            }
            
            for table in results:
                for record in table.records:
                    data_point = {
                        "timestamp": record.get_time().isoformat(),
                        "measurement": record.values.get("_measurement"),
                        "values": {k: v for k, v in record.values.items() 
                                 if k not in ['_time', '_measurement', 'result', 'table', 'tenant_id']},
                    }
                    export_data["data"].append(data_point)
                    
            return export_data
            
        except Exception as e:
            logger.error(
                f"Failed to export tenant data: {e}",
                tenant_id=self.bucket_config.tenant_id
            )
            raise
            
    async def cleanup(self):
        """Nettoie les ressources."""
        if self.client:
            await self.client.close()
            
        logger.info(
            "InfluxDBMetricsExporter cleaned up",
            tenant_id=self.bucket_config.tenant_id
        )


# Factory pour créer des exportateurs InfluxDB
class InfluxDBExporterFactory:
    """Factory pour créer des exportateurs InfluxDB configurés."""
    
    @staticmethod
    def create_spotify_ai_exporter(
        tenant_id: str,
        influxdb_url: str = "http://localhost:8086",
        token: str = "",
        org: str = "spotify-ai"
    ) -> InfluxDBMetricsExporter:
        """Crée un exportateur configuré pour Spotify AI."""
        influx_config = InfluxDBConfig(
            url=influxdb_url,
            token=token,
            org=org,
            timeout=30000,
            enable_gzip=True
        )
        
        bucket_config = TenantBucketConfig(
            tenant_id=tenant_id,
            bucket_prefix="spotify-ai",
            retention_period="90d",
            enable_downsampling=True
        )
        
        return InfluxDBMetricsExporter(influx_config, bucket_config)


# Usage example
if __name__ == "__main__":
    async def main():
        # Configuration pour un artiste Spotify
        exporter = InfluxDBExporterFactory.create_spotify_ai_exporter(
            tenant_id="spotify_artist_daft_punk",
            influxdb_url="http://localhost:8086",
            token="your-influxdb-token",
            org="spotify-ai"
        )
        
        await exporter.initialize()
        
        # Écriture de métriques IA
        ai_metrics = [
            {
                "inference_time": 0.045,
                "accuracy": 0.94,
                "model_name": "collaborative_filter_v2",
                "model_version": "2.1.0",
                "algorithm": "deep_learning",
                "tensor_size": "medium"
            },
            {
                "recommendation_accuracy": 0.89,
                "confidence": 0.85,
                "algorithm": "content_based",
                "dataset_version": "v2.3"
            }
        ]
        
        await exporter.write_ai_metrics(ai_metrics)
        
        # Écriture de métriques business
        business_metrics = [
            {
                "tracks_generated": 25,
                "genre": "electronic",
                "collaboration_type": "ai_assisted",
                "avg_duration": 4.2
            },
            {
                "revenue_impact": 15000.50,
                "stream_type": "premium",
                "channel": "streaming",
                "region": "europe",
                "streams": 50000
            }
        ]
        
        await exporter.write_business_metrics(business_metrics)
        
        # Query des performances IA
        ai_performance = await exporter.query_ai_performance("-1h", "5m")
        print(f"AI Performance: {ai_performance}")
        
        # Query de l'impact business
        business_impact = await exporter.query_business_impact("-24h", "genre")
        print(f"Business Impact: {business_impact}")
        
        # Détection d'anomalies
        anomalies = await exporter.detect_anomalies("ai_inference_time", "-1h")
        print(f"Anomalies detected: {len(anomalies)}")
        
        # Dashboard temps réel
        dashboard_data = await exporter.get_real_time_dashboard_data()
        print(f"Dashboard data: {dashboard_data['health_status']}")
        
        await exporter.cleanup()
        
    asyncio.run(main())
