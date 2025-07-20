#!/usr/bin/env python3
"""
Advanced Analytics Processing Scripts for Spotify AI Agent
========================================================

Ultra-sophisticated analytics processing scripts with real-time data pipeline,
ML model training, and comprehensive reporting capabilities.

Author: Fahed Mlaiel
Roles: Lead Dev + Architecte IA, Ingénieur Machine Learning, DBA & Data Engineer
"""

import asyncio
import argparse
import logging
import json
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import aioredis
import aiofiles
from dataclasses import asdict

# Import our analytics modules
sys.path.append(str(Path(__file__).parent))
from analytics import analytics_engine, AnalyticsMetric, MetricType
from analytics.algorithms import initialize_algorithms, anomaly_detector, trend_forecaster, recommender
from analytics.alerts import alert_manager
from analytics.utils import data_processor, performance_monitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/spotify-analytics.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AnalyticsProcessor:
    """Main analytics processing orchestrator."""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.redis_client = None
        self.processing_stats = {
            'processed_records': 0,
            'failed_records': 0,
            'processing_time': 0.0,
            'last_run': None
        }
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file or return defaults."""
        default_config = {
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            },
            'processing': {
                'batch_size': 1000,
                'max_workers': 8,
                'timeout': 300
            },
            'ml': {
                'retrain_frequency': 86400,  # 24 hours
                'model_accuracy_threshold': 0.85,
                'anomaly_threshold': 0.1
            },
            'alerts': {
                'evaluation_interval': 60,
                'max_alerts_per_hour': 100
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                return {**default_config, **config}
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                
        return default_config

    async def initialize(self) -> None:
        """Initialize analytics processor."""
        try:
            # Initialize Redis connection
            self.redis_client = aioredis.from_url(
                f"redis://{self.config['redis']['host']}:{self.config['redis']['port']}",
                db=self.config['redis']['db'],
                decode_responses=True
            )
            
            # Initialize analytics engine
            analytics_engine.redis_client = self.redis_client
            
            # Initialize ML algorithms
            initialize_algorithms()
            
            # Start alert manager
            await alert_manager.start()
            
            logger.info("Analytics processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing analytics processor: {e}")
            raise

    async def process_realtime_data(self, data_source: str) -> None:
        """Process real-time streaming data."""
        logger.info(f"Starting real-time data processing from {data_source}")
        
        start_time = datetime.now()
        processed_count = 0
        failed_count = 0
        
        try:
            # This would typically connect to a streaming source like Kafka
            # For demonstration, we'll simulate streaming data
            async for batch in self._simulate_streaming_data():
                try:
                    # Process batch
                    await self._process_data_batch(batch)
                    processed_count += len(batch)
                    
                    # Update processing stats every 1000 records
                    if processed_count % 1000 == 0:
                        await self._update_processing_stats(processed_count, failed_count)
                        
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    failed_count += len(batch)
                    
        except Exception as e:
            logger.error(f"Error in real-time processing: {e}")
            
        finally:
            # Final stats update
            total_time = (datetime.now() - start_time).total_seconds()
            self.processing_stats.update({
                'processed_records': processed_count,
                'failed_records': failed_count,
                'processing_time': total_time,
                'last_run': datetime.now().isoformat()
            })
            
            logger.info(f"Processed {processed_count} records in {total_time:.2f} seconds")

    async def _simulate_streaming_data(self):
        """Simulate streaming data for demonstration."""
        batch_size = self.config['processing']['batch_size']
        
        while True:
            batch = []
            for i in range(batch_size):
                # Simulate user listening data
                record = {
                    'user_id': f"user_{np.random.randint(1, 10000)}",
                    'track_id': f"track_{np.random.randint(1, 1000000)}",
                    'artist_id': f"artist_{np.random.randint(1, 100000)}",
                    'listen_duration': np.random.exponential(180),  # seconds
                    'skip_rate': np.random.beta(2, 8),
                    'timestamp': datetime.now().isoformat(),
                    'tenant_id': f"tenant_{np.random.randint(1, 100)}",
                    'region': np.random.choice(['us-east', 'us-west', 'eu-central', 'asia-pacific']),
                    'device_type': np.random.choice(['mobile', 'desktop', 'smart_speaker', 'tv']),
                    'user_tier': np.random.choice(['free', 'premium', 'family'])
                }
                batch.append(record)
                
            yield batch
            await asyncio.sleep(1)  # 1 second between batches

    async def _process_data_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Process a batch of data records."""
        try:
            # Convert to DataFrame for processing
            df = pd.DataFrame(batch)
            
            # Data quality assessment
            quality_report = await data_processor.assess_data_quality(df, "streaming_data")
            
            if quality_report.quality_score < 0.7:
                logger.warning(f"Low data quality score: {quality_report.quality_score}")
                
            # Process individual records
            for record in batch:
                await self._process_record(record)
                
            # Aggregate metrics
            await self._generate_aggregate_metrics(df)
            
        except Exception as e:
            logger.error(f"Error processing data batch: {e}")
            raise

    async def _process_record(self, record: Dict[str, Any]) -> None:
        """Process individual record and generate metrics."""
        try:
            tenant_id = record['tenant_id']
            timestamp = datetime.fromisoformat(record['timestamp'])
            
            # Generate various metrics
            metrics = [
                AnalyticsMetric(
                    name="user_listen_duration",
                    value=record['listen_duration'],
                    timestamp=timestamp,
                    tenant_id=tenant_id,
                    metric_type=MetricType.HISTOGRAM,
                    labels={
                        'region': record['region'],
                        'device_type': record['device_type'],
                        'user_tier': record['user_tier']
                    }
                ),
                AnalyticsMetric(
                    name="user_skip_rate",
                    value=record['skip_rate'],
                    timestamp=timestamp,
                    tenant_id=tenant_id,
                    metric_type=MetricType.GAUGE,
                    labels={
                        'region': record['region'],
                        'device_type': record['device_type']
                    }
                ),
                AnalyticsMetric(
                    name="track_play_count",
                    value=1,
                    timestamp=timestamp,
                    tenant_id=tenant_id,
                    metric_type=MetricType.COUNTER,
                    labels={
                        'track_id': record['track_id'],
                        'artist_id': record['artist_id']
                    }
                )
            ]
            
            # Record metrics
            for metric in metrics:
                await analytics_engine.record_metric(metric)
                
        except Exception as e:
            logger.error(f"Error processing record: {e}")

    async def _generate_aggregate_metrics(self, df: pd.DataFrame) -> None:
        """Generate aggregate metrics from batch data."""
        try:
            # Calculate batch-level metrics
            avg_listen_duration = df['listen_duration'].mean()
            avg_skip_rate = df['skip_rate'].mean()
            unique_users = df['user_id'].nunique()
            unique_tracks = df['track_id'].nunique()
            
            # Record aggregate metrics
            aggregate_metrics = [
                AnalyticsMetric(
                    name="batch_avg_listen_duration",
                    value=avg_listen_duration,
                    timestamp=datetime.now(),
                    tenant_id="system",
                    metric_type=MetricType.GAUGE
                ),
                AnalyticsMetric(
                    name="batch_avg_skip_rate",
                    value=avg_skip_rate,
                    timestamp=datetime.now(),
                    tenant_id="system",
                    metric_type=MetricType.GAUGE
                ),
                AnalyticsMetric(
                    name="batch_unique_users",
                    value=unique_users,
                    timestamp=datetime.now(),
                    tenant_id="system",
                    metric_type=MetricType.GAUGE
                ),
                AnalyticsMetric(
                    name="batch_unique_tracks",
                    value=unique_tracks,
                    timestamp=datetime.now(),
                    tenant_id="system",
                    metric_type=MetricType.GAUGE
                )
            ]
            
            for metric in aggregate_metrics:
                await analytics_engine.record_metric(metric)
                
        except Exception as e:
            logger.error(f"Error generating aggregate metrics: {e}")

    async def train_ml_models(self, tenant_id: str = None) -> None:
        """Train or retrain ML models."""
        logger.info("Starting ML model training")
        
        try:
            # Get training data
            training_data = await self._get_training_data(tenant_id)
            
            if training_data.empty:
                logger.warning("No training data available")
                return
                
            # Train anomaly detection model
            if anomaly_detector:
                logger.info("Training anomaly detection model")
                anomaly_metrics = await anomaly_detector.train(training_data)
                logger.info(f"Anomaly detection metrics: {anomaly_metrics}")
                
            # Train trend forecasting model
            if trend_forecaster:
                logger.info("Training trend forecasting model")
                trend_metrics = await trend_forecaster.train(training_data)
                logger.info(f"Trend forecasting metrics: {trend_metrics}")
                
            # Train recommendation model
            if recommender:
                logger.info("Training recommendation model")
                rec_metrics = await recommender.train(training_data)
                logger.info(f"Recommendation metrics: {rec_metrics}")
                
            logger.info("ML model training completed")
            
        except Exception as e:
            logger.error(f"Error training ML models: {e}")

    async def _get_training_data(self, tenant_id: str = None) -> pd.DataFrame:
        """Get training data for ML models."""
        try:
            # This would typically query a data warehouse or time-series database
            # For demonstration, create synthetic training data
            n_samples = 10000
            
            data = {
                'user_id': [f"user_{i}" for i in range(n_samples)],
                'track_id': [f"track_{np.random.randint(1, 1000)}" for _ in range(n_samples)],
                'listen_duration': np.random.exponential(180, n_samples),
                'skip_rate': np.random.beta(2, 8, n_samples),
                'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(n_samples)],
                'user_tier': np.random.choice(['free', 'premium'], n_samples),
                'region': np.random.choice(['us', 'eu', 'asia'], n_samples)
            }
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error getting training data: {e}")
            return pd.DataFrame()

    async def generate_analytics_report(self, tenant_id: str, 
                                       time_range: timedelta = timedelta(hours=24)) -> Dict[str, Any]:
        """Generate comprehensive analytics report."""
        logger.info(f"Generating analytics report for tenant {tenant_id}")
        
        try:
            # Get dashboard data
            dashboard_data = await analytics_engine.get_analytics_dashboard(tenant_id, time_range)
            
            # Get alert statistics
            alert_stats = await alert_manager.get_alert_statistics(tenant_id, time_range)
            
            # Generate ML insights
            ml_insights = await self._generate_ml_insights(tenant_id, time_range)
            
            # Compile comprehensive report
            report = {
                'tenant_id': tenant_id,
                'report_generated': datetime.now().isoformat(),
                'time_range': {
                    'duration_hours': time_range.total_seconds() / 3600,
                    'start_time': (datetime.now() - time_range).isoformat(),
                    'end_time': datetime.now().isoformat()
                },
                'dashboard_data': dashboard_data,
                'alert_statistics': alert_stats,
                'ml_insights': ml_insights,
                'processing_stats': self.processing_stats,
                'recommendations': await self._generate_recommendations(dashboard_data, alert_stats)
            }
            
            # Save report
            await self._save_report(report, tenant_id)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating analytics report: {e}")
            return {}

    async def _generate_ml_insights(self, tenant_id: str, time_range: timedelta) -> Dict[str, Any]:
        """Generate ML-powered insights."""
        insights = {
            'anomalies_detected': 0,
            'trend_predictions': [],
            'recommendations': [],
            'model_performance': {}
        }
        
        try:
            # Get recent data for analysis
            recent_data = await self._get_training_data(tenant_id)
            
            if not recent_data.empty:
                # Anomaly detection insights
                if anomaly_detector and anomaly_detector.is_trained:
                    sample_features = {
                        'listen_duration': recent_data['listen_duration'].mean(),
                        'skip_rate': recent_data['skip_rate'].mean()
                    }
                    anomaly_result = await anomaly_detector.predict(sample_features)
                    insights['anomalies_detected'] = int(anomaly_result.prediction)
                
                # Trend forecasting insights
                if trend_forecaster and trend_forecaster.is_trained:
                    trend_features = {
                        'historical_avg': recent_data['listen_duration'].mean(),
                        'trend_slope': np.polyfit(range(len(recent_data)), recent_data['listen_duration'], 1)[0]
                    }
                    trend_result = await trend_forecaster.predict(trend_features)
                    insights['trend_predictions'].append({
                        'metric': 'listen_duration',
                        'predicted_value': trend_result.prediction,
                        'confidence': trend_result.confidence
                    })
                
                # Recommendation insights
                if recommender and recommender.is_trained:
                    rec_features = {'user_id': 0, 'n_recommendations': 10}
                    rec_result = await recommender.predict(rec_features)
                    insights['recommendations'] = rec_result.prediction
            
        except Exception as e:
            logger.error(f"Error generating ML insights: {e}")
            
        return insights

    async def _generate_recommendations(self, dashboard_data: Dict[str, Any], 
                                       alert_stats: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on data."""
        recommendations = []
        
        try:
            # Performance recommendations
            if dashboard_data.get('metrics', {}).get('error_rate', 0) > 0.05:
                recommendations.append("High error rate detected. Investigate API endpoints and infrastructure.")
            
            if dashboard_data.get('metrics', {}).get('average_response_time', 0) > 2.0:
                recommendations.append("Response times are high. Consider scaling infrastructure or optimizing queries.")
            
            # Alert recommendations
            if alert_stats.get('escalated_alerts', 0) > 5:
                recommendations.append("Multiple alerts escalated. Review alert thresholds and response procedures.")
            
            # Business recommendations
            if dashboard_data.get('metrics', {}).get('active_users', 0) < 1000:
                recommendations.append("User engagement is low. Consider promotional campaigns or feature improvements.")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            
        return recommendations

    async def _save_report(self, report: Dict[str, Any], tenant_id: str) -> None:
        """Save analytics report to file."""
        try:
            reports_dir = Path('/var/reports/analytics')
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"analytics_report_{tenant_id}_{timestamp}.json"
            filepath = reports_dir / filename
            
            async with aiofiles.open(filepath, 'w') as f:
                await f.write(json.dumps(report, indent=2, default=str))
                
            logger.info(f"Analytics report saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")

    async def _update_processing_stats(self, processed: int, failed: int) -> None:
        """Update processing statistics."""
        self.processing_stats.update({
            'processed_records': processed,
            'failed_records': failed,
            'last_update': datetime.now().isoformat()
        })
        
        # Store stats in Redis
        if self.redis_client:
            await self.redis_client.hset(
                "analytics:processing_stats",
                mapping=self.processing_stats
            )

async def main():
    """Main entry point for analytics processing."""
    parser = argparse.ArgumentParser(description='Spotify AI Agent Analytics Processor')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--mode', type=str, choices=['realtime', 'train', 'report'], 
                       default='realtime', help='Processing mode')
    parser.add_argument('--tenant-id', type=str, help='Tenant ID for specific operations')
    parser.add_argument('--data-source', type=str, default='kafka', help='Data source for real-time processing')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = AnalyticsProcessor(args.config)
    await processor.initialize()
    
    try:
        if args.mode == 'realtime':
            logger.info("Starting real-time analytics processing")
            await processor.process_realtime_data(args.data_source)
            
        elif args.mode == 'train':
            logger.info("Starting ML model training")
            await processor.train_ml_models(args.tenant_id)
            
        elif args.mode == 'report':
            if not args.tenant_id:
                logger.error("Tenant ID required for report generation")
                sys.exit(1)
            
            logger.info(f"Generating analytics report for tenant {args.tenant_id}")
            report = await processor.generate_analytics_report(args.tenant_id)
            print(json.dumps(report, indent=2, default=str))
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        await alert_manager.stop()
        if processor.redis_client:
            await processor.redis_client.close()

if __name__ == "__main__":
    asyncio.run(main())
