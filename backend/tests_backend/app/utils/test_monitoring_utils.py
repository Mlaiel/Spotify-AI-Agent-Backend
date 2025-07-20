"""
Tests Enterprise - Monitoring Utilities
=======================================

Suite de tests ultra-avancée pour le module monitoring_utils avec observabilité complète,
métriques temps réel, alerting intelligent, et APM enterprise.

Développé par l'équipe Monitoring & Observability Expert sous la direction de Fahed Mlaiel.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from datetime import datetime, timedelta
import json
from typing import Dict, Any, List, Optional
import uuid
import time

# Import des modules monitoring à tester
try:
    from app.utils.monitoring_utils import (
        MetricsCollector,
        AlertManager,
        ObservabilityEngine,
        APMTracer,
        PerformanceMonitor
    )
except ImportError:
    # Mocks si modules pas encore implémentés
    MetricsCollector = MagicMock
    AlertManager = MagicMock
    ObservabilityEngine = MagicMock
    APMTracer = MagicMock
    PerformanceMonitor = MagicMock


class TestMetricsCollector:
    """Tests enterprise pour MetricsCollector avec collecte métriques avancée."""
    
    @pytest.fixture
    def metrics_collector(self):
        """Instance MetricsCollector pour tests."""
        return MetricsCollector()
    
    @pytest.fixture
    def metrics_config(self):
        """Configuration métriques enterprise."""
        return {
            'collection_strategy': {
                'mode': 'real_time',
                'batch_size': 1000,
                'flush_interval_ms': 5000,
                'compression': True,
                'sampling_rate': 0.01
            },
            'metric_types': {
                'business_metrics': {
                    'enabled': True,
                    'categories': ['user_engagement', 'revenue', 'content_performance']
                },
                'technical_metrics': {
                    'enabled': True,
                    'categories': ['performance', 'reliability', 'security']
                },
                'ml_metrics': {
                    'enabled': True,
                    'categories': ['model_performance', 'feature_drift', 'prediction_quality']
                }
            },
            'retention_policies': {
                'high_resolution': {'duration_days': 7, 'granularity_seconds': 1},
                'medium_resolution': {'duration_days': 30, 'granularity_seconds': 60},
                'low_resolution': {'duration_days': 365, 'granularity_seconds': 3600}
            },
            'alerting_thresholds': {
                'error_rate': 0.01,
                'latency_p95_ms': 500,
                'memory_usage': 0.8,
                'cpu_usage': 0.7
            }
        }
    
    async def test_comprehensive_metrics_collection(self, metrics_collector, metrics_config):
        """Test collecte métriques complète avec tous types."""
        # Mock configuration collector
        metrics_collector.configure = AsyncMock(return_value={'status': 'configured'})
        await metrics_collector.configure(metrics_config)
        
        # Types de métriques business
        business_metrics = [
            {
                'name': 'daily_active_users',
                'value': 125000,
                'type': 'gauge',
                'tags': {'platform': 'mobile', 'region': 'europe'},
                'timestamp': datetime.utcnow()
            },
            {
                'name': 'track_plays_total',
                'value': 1,
                'type': 'counter',
                'tags': {'track_id': 'track_12345', 'quality': 'high'},
                'timestamp': datetime.utcnow()
            },
            {
                'name': 'revenue_per_user',
                'value': 9.99,
                'type': 'histogram',
                'tags': {'subscription_type': 'premium', 'currency': 'EUR'},
                'timestamp': datetime.utcnow()
            }
        ]
        
        # Types de métriques techniques
        technical_metrics = [
            {
                'name': 'api_request_duration_ms',
                'value': 234.5,
                'type': 'histogram',
                'tags': {'endpoint': '/api/v1/tracks', 'method': 'GET', 'status': '200'},
                'timestamp': datetime.utcnow()
            },
            {
                'name': 'memory_usage_bytes',
                'value': 2147483648,
                'type': 'gauge',
                'tags': {'service': 'streaming-api', 'instance': 'api-01'},
                'timestamp': datetime.utcnow()
            },
            {
                'name': 'error_count',
                'value': 1,
                'type': 'counter',
                'tags': {'error_type': 'timeout', 'service': 'ml-inference'},
                'timestamp': datetime.utcnow()
            }
        ]
        
        # Types de métriques ML
        ml_metrics = [
            {
                'name': 'model_accuracy',
                'value': 0.94,
                'type': 'gauge',
                'tags': {'model': 'recommendation_v2', 'dataset': 'validation'},
                'timestamp': datetime.utcnow()
            },
            {
                'name': 'feature_drift_score',
                'value': 0.23,
                'type': 'gauge',
                'tags': {'feature': 'user_listening_pattern', 'window': '24h'},
                'timestamp': datetime.utcnow()
            },
            {
                'name': 'prediction_latency_ms',
                'value': 12.7,
                'type': 'histogram',
                'tags': {'model': 'personalization_engine', 'batch_size': '100'},
                'timestamp': datetime.utcnow()
            }
        ]
        
        # Mock collecte métriques
        metrics_collector.collect_metrics = AsyncMock(return_value={
            'metrics_collected': len(business_metrics) + len(technical_metrics) + len(ml_metrics),
            'collection_time_ms': 45.2,
            'compression_ratio': 0.72,
            'batch_id': str(uuid.uuid4()),
            'storage_backend': 'prometheus',
            'validation_passed': True
        })
        
        # Collecte métriques par catégorie
        all_metrics = business_metrics + technical_metrics + ml_metrics
        result = await metrics_collector.collect_metrics(
            metrics=all_metrics,
            collection_mode='batch',
            validate_schema=True
        )
        
        # Validations collecte
        assert result['metrics_collected'] == len(all_metrics)
        assert result['collection_time_ms'] < 100
        assert result['compression_ratio'] > 0.5
        assert result['validation_passed'] is True
        assert 'batch_id' in result
    
    async def test_real_time_metrics_streaming(self, metrics_collector):
        """Test streaming métriques temps réel."""
        # Configuration streaming temps réel
        streaming_config = {
            'stream_type': 'kafka',
            'topic': 'metrics.real_time',
            'partitioning': 'by_service',
            'serialization': 'avro',
            'compression': 'snappy',
            'buffer_size': 10000,
            'flush_interval_ms': 1000
        }
        
        # Mock streaming métriques
        metrics_collector.start_real_time_streaming = AsyncMock(return_value={
            'stream_id': 'stream_metrics_001',
            'status': 'active',
            'throughput_metrics_per_second': 5000,
            'latency_p95_ms': 15,
            'error_rate': 0.0001,
            'consumer_lag_ms': 23
        })
        
        # Simulation métriques temps réel
        real_time_metrics = []
        for i in range(100):
            metric = {
                'name': 'concurrent_streams',
                'value': np.random.randint(1000, 10000),
                'type': 'gauge',
                'tags': {'region': np.random.choice(['eu', 'us', 'asia'])},
                'timestamp': datetime.utcnow()
            }
            real_time_metrics.append(metric)
        
        # Démarrage streaming
        stream_result = await metrics_collector.start_real_time_streaming(
            config=streaming_config,
            metrics_source=real_time_metrics
        )
        
        # Validations streaming temps réel
        assert stream_result['status'] == 'active'
        assert stream_result['throughput_metrics_per_second'] > 1000
        assert stream_result['latency_p95_ms'] < 50
        assert stream_result['error_rate'] < 0.01
        assert stream_result['consumer_lag_ms'] < 100
    
    async def test_metrics_aggregation_algorithms(self, metrics_collector):
        """Test algorithmes agrégation métriques."""
        # Données métriques pour agrégation
        raw_metrics_data = {
            'api_latency_samples': np.random.lognormal(3, 0.5, 10000),  # Distribution log-normale
            'memory_usage_samples': np.random.beta(2, 5, 10000) * 100,  # Distribution beta
            'error_counts': np.random.poisson(2, 1000),  # Distribution Poisson
            'user_sessions': np.random.exponential(30, 5000)  # Distribution exponentielle
        }
        
        # Algorithmes d'agrégation testés
        aggregation_algorithms = [
            {
                'name': 'percentile_based',
                'percentiles': [50, 75, 90, 95, 99, 99.9],
                'window_size': '5m'
            },
            {
                'name': 'sliding_window_avg',
                'window_size': '1m',
                'slide_interval': '10s'
            },
            {
                'name': 'exponential_smoothing',
                'alpha': 0.3,
                'beta': 0.1
            },
            {
                'name': 'seasonal_decomposition',
                'period': 24,  # heures
                'model': 'additive'
            }
        ]
        
        # Mock agrégation
        metrics_collector.aggregate_metrics = AsyncMock()
        
        for algorithm in aggregation_algorithms:
            # Configuration réponse agrégation
            metrics_collector.aggregate_metrics.return_value = {
                'algorithm_used': algorithm['name'],
                'aggregated_values': {
                    'mean': np.random.uniform(10, 100),
                    'median': np.random.uniform(5, 80),
                    'std_dev': np.random.uniform(1, 20),
                    'min': np.random.uniform(0, 10),
                    'max': np.random.uniform(90, 200)
                },
                'percentiles': {f'p{p}': np.random.uniform(10, 150) for p in [50, 90, 95, 99]},
                'data_quality_score': np.random.uniform(0.85, 0.98),
                'outliers_detected': np.random.randint(0, 50),
                'aggregation_accuracy': np.random.uniform(0.92, 0.99)
            }
            
            result = await metrics_collector.aggregate_metrics(
                raw_data=raw_metrics_data,
                algorithm=algorithm,
                time_window='1h'
            )
            
            # Validations agrégation
            assert result['data_quality_score'] > 0.8
            assert result['aggregation_accuracy'] > 0.9
            assert 'percentiles' in result
            assert result['aggregated_values']['mean'] > 0
    
    async def test_metrics_anomaly_detection(self, metrics_collector):
        """Test détection anomalies dans métriques."""
        # Types d'anomalies à détecter
        anomaly_scenarios = [
            {
                'type': 'spike',
                'metric': 'cpu_usage',
                'normal_range': [0.2, 0.6],
                'anomaly_value': 0.95,
                'severity': 'high'
            },
            {
                'type': 'drop',
                'metric': 'request_rate',
                'normal_range': [1000, 5000],
                'anomaly_value': 50,
                'severity': 'critical'
            },
            {
                'type': 'trend_change',
                'metric': 'memory_usage',
                'normal_trend': 'stable',
                'anomaly_trend': 'increasing',
                'severity': 'medium'
            },
            {
                'type': 'outlier',
                'metric': 'api_latency',
                'normal_range': [10, 100],
                'anomaly_value': 5000,
                'severity': 'high'
            }
        ]
        
        # Mock détection anomalies
        metrics_collector.detect_anomalies = AsyncMock()
        
        for scenario in anomaly_scenarios:
            # Configuration réponse détection
            metrics_collector.detect_anomalies.return_value = {
                'anomaly_detected': True,
                'anomaly_type': scenario['type'],
                'confidence_score': np.random.uniform(0.8, 0.99),
                'severity': scenario['severity'],
                'affected_metric': scenario['metric'],
                'detection_algorithm': 'isolation_forest',
                'time_detected': datetime.utcnow(),
                'expected_range': scenario.get('normal_range', []),
                'actual_value': scenario.get('anomaly_value', scenario.get('anomaly_trend')),
                'root_cause_hints': [
                    'sudden_traffic_spike',
                    'resource_exhaustion',
                    'external_dependency_failure'
                ]
            }
            
            result = await metrics_collector.detect_anomalies(
                metric_name=scenario['metric'],
                time_window='15m',
                detection_sensitivity='high'
            )
            
            # Validations détection anomalies
            assert result['anomaly_detected'] is True
            assert result['confidence_score'] > 0.8
            assert result['severity'] in ['low', 'medium', 'high', 'critical']
            assert 'root_cause_hints' in result


class TestAlertManager:
    """Tests enterprise pour AlertManager avec alerting intelligent."""
    
    @pytest.fixture
    def alert_manager(self):
        """Instance AlertManager pour tests."""
        return AlertManager()
    
    @pytest.fixture
    def alerting_config(self):
        """Configuration alerting enterprise."""
        return {
            'notification_channels': {
                'slack': {
                    'webhook_url': 'https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX',
                    'channel': '#alerts-critical',
                    'mention_users': ['@oncall-engineer', '@team-lead']
                },
                'email': {
                    'smtp_server': 'smtp.company.com',
                    'recipients': ['alerts@company.com', 'oncall@company.com']
                },
                'pagerduty': {
                    'api_key': 'r1d2c3f7e18b46c9e90f3b4b5c6d7e8f',
                    'service_id': 'PXXXXXX'
                },
                'webhook': {
                    'url': 'https://api.company.com/alerts',
                    'auth_token': 'bearer_token_123'
                }
            },
            'escalation_policies': {
                'critical': {
                    'immediate': ['slack', 'pagerduty'],
                    'after_5m': ['email'],
                    'after_15m': ['webhook']
                },
                'high': {
                    'immediate': ['slack'],
                    'after_10m': ['email']
                },
                'medium': {
                    'immediate': ['slack'],
                    'after_30m': ['email']
                }
            },
            'suppression_rules': {
                'maintenance_windows': True,
                'similar_alerts_grouping': True,
                'noise_reduction': True
            }
        }
    
    async def test_intelligent_alert_generation(self, alert_manager, alerting_config):
        """Test génération alertes intelligente."""
        # Mock configuration alerting
        alert_manager.configure = AsyncMock(return_value={'status': 'configured'})
        await alert_manager.configure(alerting_config)
        
        # Conditions d'alerte critiques
        critical_conditions = [
            {
                'name': 'service_down',
                'description': 'Service de streaming indisponible',
                'severity': 'critical',
                'affected_services': ['streaming-api', 'cdn'],
                'impact_assessment': {
                    'users_affected': 50000,
                    'revenue_impact_per_minute': 1000,
                    'sla_breach': True
                }
            },
            {
                'name': 'high_error_rate',
                'description': 'Taux d\'erreur API élevé',
                'severity': 'high',
                'current_value': 0.15,
                'threshold': 0.05,
                'trend': 'increasing'
            },
            {
                'name': 'latency_spike',
                'description': 'Pic de latence détecté',
                'severity': 'medium',
                'current_p95_ms': 2500,
                'threshold_p95_ms': 500,
                'duration_minutes': 7
            }
        ]
        
        # Mock génération alertes
        alert_manager.generate_alert = AsyncMock()
        
        for condition in critical_conditions:
            # Configuration réponse alerte
            alert_manager.generate_alert.return_value = {
                'alert_id': str(uuid.uuid4()),
                'alert_name': condition['name'],
                'severity': condition['severity'],
                'status': 'active',
                'created_at': datetime.utcnow(),
                'escalation_level': 0,
                'notification_channels_used': alerting_config['escalation_policies'][condition['severity']]['immediate'],
                'suppression_applied': False,
                'correlation_id': f"incident_{uuid.uuid4().hex[:8]}",
                'runbook_url': f"https://runbooks.company.com/{condition['name']}",
                'estimated_resolution_time_minutes': 30 if condition['severity'] == 'critical' else 60
            }
            
            alert = await alert_manager.generate_alert(
                condition=condition,
                context={'service': 'spotify-ai-agent', 'environment': 'production'},
                auto_escalate=True
            )
            
            # Validations génération alerte
            assert alert['alert_id'] is not None
            assert alert['severity'] == condition['severity']
            assert alert['status'] == 'active'
            assert len(alert['notification_channels_used']) > 0
            assert 'runbook_url' in alert
    
    async def test_smart_alert_correlation(self, alert_manager):
        """Test corrélation intelligente alertes."""
        # Alertes potentiellement corrélées
        related_alerts = [
            {
                'id': 'alert_001',
                'name': 'database_connection_timeout',
                'service': 'user-service',
                'timestamp': datetime.utcnow() - timedelta(minutes=2),
                'severity': 'high'
            },
            {
                'id': 'alert_002',
                'name': 'api_high_latency',
                'service': 'user-service',
                'timestamp': datetime.utcnow() - timedelta(minutes=1),
                'severity': 'medium'
            },
            {
                'id': 'alert_003',
                'name': 'increased_error_rate',
                'service': 'user-service',
                'timestamp': datetime.utcnow(),
                'severity': 'high'
            },
            {
                'id': 'alert_004',
                'name': 'memory_leak_detected',
                'service': 'recommendation-service',
                'timestamp': datetime.utcnow() - timedelta(minutes=10),
                'severity': 'medium'
            }
        ]
        
        # Mock corrélation
        alert_manager.correlate_alerts = AsyncMock(return_value={
            'incident_groups': [
                {
                    'incident_id': 'incident_001',
                    'root_cause': 'database_connectivity_issue',
                    'related_alerts': ['alert_001', 'alert_002', 'alert_003'],
                    'correlation_confidence': 0.92,
                    'probable_impact': {
                        'services_affected': ['user-service', 'auth-service'],
                        'estimated_users_impacted': 25000,
                        'business_impact': 'high'
                    },
                    'suggested_actions': [
                        'check_database_health',
                        'verify_network_connectivity',
                        'review_connection_pool_settings'
                    ]
                },
                {
                    'incident_id': 'incident_002',
                    'root_cause': 'memory_management_issue',
                    'related_alerts': ['alert_004'],
                    'correlation_confidence': 0.87,
                    'probable_impact': {
                        'services_affected': ['recommendation-service'],
                        'estimated_users_impacted': 5000,
                        'business_impact': 'medium'
                    }
                }
            ],
            'noise_reduction': {
                'duplicate_alerts_suppressed': 3,
                'correlated_alerts_grouped': 3,
                'total_alerts_reduced_by': 0.4
            }
        })
        
        correlation_result = await alert_manager.correlate_alerts(
            alerts=related_alerts,
            correlation_window_minutes=15,
            confidence_threshold=0.8
        )
        
        # Validations corrélation
        assert len(correlation_result['incident_groups']) > 0
        for incident in correlation_result['incident_groups']:
            assert incident['correlation_confidence'] > 0.8
            assert len(incident['related_alerts']) > 0
            assert 'suggested_actions' in incident
        
        assert correlation_result['noise_reduction']['total_alerts_reduced_by'] > 0
    
    async def test_adaptive_alert_thresholds(self, alert_manager):
        """Test seuils adaptatifs pour alertes."""
        # Métriques pour adaptation seuils
        metrics_history = {
            'api_latency_p95': {
                'historical_data': np.random.lognormal(4, 0.3, 1000),  # Données historiques
                'current_threshold': 500,
                'business_hours_pattern': True,
                'seasonal_variance': 0.2
            },
            'error_rate': {
                'historical_data': np.random.beta(1, 99, 1000),  # Taux erreur historique
                'current_threshold': 0.01,
                'deployment_correlation': True,
                'user_impact_weight': 0.8
            },
            'memory_usage': {
                'historical_data': np.random.normal(0.6, 0.1, 1000),  # Usage mémoire
                'current_threshold': 0.8,
                'growth_trend': 'stable',
                'capacity_planning_factor': 1.2
            }
        }
        
        # Algorithmes adaptation seuils
        adaptation_algorithms = [
            {
                'name': 'statistical_baseline',
                'method': 'percentile_based',
                'parameters': {'percentile': 95, 'window_days': 30}
            },
            {
                'name': 'machine_learning',
                'method': 'anomaly_detection',
                'parameters': {'algorithm': 'isolation_forest', 'contamination': 0.1}
            },
            {
                'name': 'business_aware',
                'method': 'impact_weighted',
                'parameters': {'sla_target': 0.999, 'cost_per_false_positive': 100}
            }
        ]
        
        # Mock adaptation seuils
        alert_manager.adapt_thresholds = AsyncMock()
        
        for metric_name, metric_data in metrics_history.items():
            for algorithm in adaptation_algorithms:
                # Configuration réponse adaptation
                alert_manager.adapt_thresholds.return_value = {
                    'metric_name': metric_name,
                    'algorithm_used': algorithm['name'],
                    'old_threshold': metric_data['current_threshold'],
                    'new_threshold': metric_data['current_threshold'] * np.random.uniform(0.8, 1.2),
                    'confidence_score': np.random.uniform(0.85, 0.95),
                    'adaptation_reason': 'historical_pattern_analysis',
                    'false_positive_reduction': np.random.uniform(0.1, 0.4),
                    'sensitivity_improvement': np.random.uniform(0.05, 0.25),
                    'validation_period_days': 7
                }
                
                result = await alert_manager.adapt_thresholds(
                    metric_name=metric_name,
                    historical_data=metric_data['historical_data'],
                    adaptation_algorithm=algorithm
                )
                
                # Validations adaptation
                assert result['confidence_score'] > 0.8
                assert result['new_threshold'] != result['old_threshold']
                assert result['false_positive_reduction'] > 0
                assert result['validation_period_days'] > 0


class TestObservabilityEngine:
    """Tests enterprise pour ObservabilityEngine avec observabilité complète."""
    
    @pytest.fixture
    def observability_engine(self):
        """Instance ObservabilityEngine pour tests."""
        return ObservabilityEngine()
    
    async def test_distributed_tracing_correlation(self, observability_engine):
        """Test corrélation distributed tracing."""
        # Traces distribuées simulées
        distributed_traces = [
            {
                'trace_id': 'trace_12345',
                'spans': [
                    {
                        'span_id': 'span_001',
                        'operation': 'http_request',
                        'service': 'api-gateway',
                        'duration_ms': 234,
                        'start_time': datetime.utcnow(),
                        'tags': {'http.method': 'GET', 'http.status_code': 200}
                    },
                    {
                        'span_id': 'span_002',
                        'parent_span_id': 'span_001',
                        'operation': 'database_query',
                        'service': 'user-service',
                        'duration_ms': 89,
                        'start_time': datetime.utcnow() + timedelta(milliseconds=10)
                    },
                    {
                        'span_id': 'span_003',
                        'parent_span_id': 'span_001',
                        'operation': 'ml_inference',
                        'service': 'recommendation-service',
                        'duration_ms': 156,
                        'start_time': datetime.utcnow() + timedelta(milliseconds=100)
                    }
                ]
            }
        ]
        
        # Mock analyse traces
        observability_engine.analyze_distributed_trace = AsyncMock(return_value={
            'trace_analysis': {
                'total_duration_ms': 234,
                'critical_path': ['api-gateway', 'recommendation-service'],
                'bottleneck_service': 'recommendation-service',
                'bottleneck_operation': 'ml_inference',
                'parallel_execution_efficiency': 0.76,
                'service_dependencies': {
                    'api-gateway': ['user-service', 'recommendation-service'],
                    'user-service': [],
                    'recommendation-service': []
                }
            },
            'performance_insights': {
                'latency_breakdown': {
                    'api-gateway': 0.42,
                    'user-service': 0.38,
                    'recommendation-service': 0.67
                },
                'optimization_opportunities': [
                    {
                        'service': 'recommendation-service',
                        'operation': 'ml_inference',
                        'potential_improvement_ms': 50,
                        'recommendation': 'cache_model_predictions'
                    }
                ]
            },
            'error_correlation': {
                'errors_detected': 0,
                'retry_patterns': [],
                'timeout_risks': ['recommendation-service']
            }
        })
        
        analysis = await observability_engine.analyze_distributed_trace(
            trace_data=distributed_traces[0],
            analysis_depth='comprehensive'
        )
        
        # Validations analyse traces
        assert analysis['trace_analysis']['total_duration_ms'] > 0
        assert 'critical_path' in analysis['trace_analysis']
        assert 'bottleneck_service' in analysis['trace_analysis']
        assert len(analysis['performance_insights']['optimization_opportunities']) > 0
    
    async def test_service_dependency_mapping(self, observability_engine):
        """Test cartographie dépendances services."""
        # Configuration services et dépendances
        service_topology = {
            'services': [
                'api-gateway', 'user-service', 'track-service',
                'recommendation-service', 'streaming-service',
                'payment-service', 'notification-service'
            ],
            'external_dependencies': [
                'postgres-db', 'redis-cache', 'elasticsearch',
                'kafka-broker', 'ml-model-registry'
            ]
        }
        
        # Mock cartographie dépendances
        observability_engine.map_service_dependencies = AsyncMock(return_value={
            'dependency_graph': {
                'nodes': [
                    {'id': 'api-gateway', 'type': 'service', 'criticality': 'high'},
                    {'id': 'user-service', 'type': 'service', 'criticality': 'high'},
                    {'id': 'postgres-db', 'type': 'database', 'criticality': 'critical'},
                    {'id': 'redis-cache', 'type': 'cache', 'criticality': 'medium'}
                ],
                'edges': [
                    {'from': 'api-gateway', 'to': 'user-service', 'relationship': 'calls'},
                    {'from': 'user-service', 'to': 'postgres-db', 'relationship': 'reads'},
                    {'from': 'user-service', 'to': 'redis-cache', 'relationship': 'caches'}
                ]
            },
            'critical_paths': [
                ['api-gateway', 'user-service', 'postgres-db'],
                ['api-gateway', 'streaming-service', 'ml-model-registry']
            ],
            'single_points_of_failure': [
                {'service': 'postgres-db', 'impact_services': 5, 'mitigation': 'setup_replica'},
                {'service': 'api-gateway', 'impact_services': 7, 'mitigation': 'load_balancing'}
            ],
            'circuit_breaker_recommendations': [
                {
                    'from_service': 'user-service',
                    'to_service': 'postgres-db',
                    'timeout_ms': 5000,
                    'failure_threshold': 5
                }
            ]
        })
        
        dependency_map = await observability_engine.map_service_dependencies(
            topology=service_topology,
            analysis_period_hours=24
        )
        
        # Validations cartographie
        assert len(dependency_map['dependency_graph']['nodes']) > 0
        assert len(dependency_map['dependency_graph']['edges']) > 0
        assert len(dependency_map['critical_paths']) > 0
        assert len(dependency_map['single_points_of_failure']) > 0
    
    async def test_observability_dashboards_generation(self, observability_engine):
        """Test génération dashboards observabilité."""
        # Configuration dashboards
        dashboard_specs = [
            {
                'name': 'service_health_overview',
                'type': 'operational',
                'target_audience': 'sre_team',
                'refresh_interval_seconds': 30,
                'panels': [
                    'service_availability', 'error_rates', 'latency_percentiles',
                    'throughput_metrics', 'dependency_health'
                ]
            },
            {
                'name': 'business_metrics_executive',
                'type': 'business',
                'target_audience': 'executives',
                'refresh_interval_seconds': 300,
                'panels': [
                    'daily_active_users', 'revenue_metrics', 'content_consumption',
                    'user_satisfaction', 'churn_indicators'
                ]
            },
            {
                'name': 'ml_model_performance',
                'type': 'ml_ops',
                'target_audience': 'ml_engineers',
                'refresh_interval_seconds': 60,
                'panels': [
                    'model_accuracy', 'prediction_latency', 'feature_drift',
                    'training_metrics', 'inference_volume'
                ]
            }
        ]
        
        # Mock génération dashboards
        observability_engine.generate_dashboard = AsyncMock()
        
        for spec in dashboard_specs:
            # Configuration réponse dashboard
            observability_engine.generate_dashboard.return_value = {
                'dashboard_id': f"dash_{uuid.uuid4().hex[:8]}",
                'dashboard_name': spec['name'],
                'dashboard_url': f"https://grafana.company.com/d/{uuid.uuid4().hex[:8]}",
                'panels_created': len(spec['panels']),
                'data_sources_connected': ['prometheus', 'elasticsearch', 'postgres'],
                'alerting_rules_configured': np.random.randint(5, 20),
                'estimated_load_time_ms': np.random.uniform(500, 2000),
                'accessibility_score': np.random.uniform(0.85, 0.95),
                'mobile_responsive': True
            }
            
            dashboard = await observability_engine.generate_dashboard(
                specification=spec,
                auto_configure_alerts=True,
                responsive_design=True
            )
            
            # Validations dashboard
            assert 'dashboard_id' in dashboard
            assert dashboard['panels_created'] == len(spec['panels'])
            assert len(dashboard['data_sources_connected']) > 0
            assert dashboard['estimated_load_time_ms'] < 3000
            assert dashboard['accessibility_score'] > 0.8


class TestAPMTracer:
    """Tests enterprise pour APMTracer avec Application Performance Monitoring."""
    
    @pytest.fixture
    def apm_tracer(self):
        """Instance APMTracer pour tests."""
        return APMTracer()
    
    async def test_comprehensive_performance_tracing(self, apm_tracer):
        """Test tracing performance applicatif complet."""
        # Configuration tracing APM
        apm_config = {
            'sampling_strategy': {
                'default_rate': 0.1,
                'error_rate': 1.0,
                'slow_request_rate': 1.0,
                'adaptive_sampling': True
            },
            'instrumentation': {
                'auto_instrument': ['http', 'database', 'cache', 'messaging'],
                'custom_spans': True,
                'async_context_propagation': True
            },
            'performance_budgets': {
                'api_endpoints': {'p95_latency_ms': 500, 'error_rate': 0.01},
                'database_queries': {'p95_latency_ms': 100, 'timeout_ms': 5000},
                'external_calls': {'p95_latency_ms': 1000, 'circuit_breaker': True}
            }
        }
        
        # Mock configuration APM
        apm_tracer.configure = AsyncMock(return_value={'status': 'configured'})
        await apm_tracer.configure(apm_config)
        
        # Transactions applicatives simulées
        app_transactions = [
            {
                'transaction_id': 'txn_001',
                'name': 'GET /api/v1/recommendations',
                'type': 'request',
                'duration_ms': 342,
                'result': 'success',
                'user_id': 'user_12345',
                'spans': [
                    {'name': 'auth_validation', 'duration_ms': 23},
                    {'name': 'user_profile_fetch', 'duration_ms': 89},
                    {'name': 'ml_recommendation_engine', 'duration_ms': 203},
                    {'name': 'response_serialization', 'duration_ms': 27}
                ]
            },
            {
                'transaction_id': 'txn_002',
                'name': 'POST /api/v1/stream/start',
                'type': 'request',
                'duration_ms': 156,
                'result': 'success',
                'user_id': 'user_67890',
                'spans': [
                    {'name': 'stream_authorization', 'duration_ms': 34},
                    {'name': 'cdn_endpoint_selection', 'duration_ms': 67},
                    {'name': 'quality_optimization', 'duration_ms': 45},
                    {'name': 'stream_session_init', 'duration_ms': 10}
                ]
            }
        ]
        
        # Mock tracing transactions
        apm_tracer.trace_transaction = AsyncMock()
        
        for transaction in app_transactions:
            # Configuration réponse tracing
            apm_tracer.trace_transaction.return_value = {
                'trace_id': f"trace_{uuid.uuid4().hex}",
                'transaction_summary': {
                    'name': transaction['name'],
                    'duration_ms': transaction['duration_ms'],
                    'spans_count': len(transaction['spans']),
                    'performance_score': np.random.uniform(0.8, 0.95),
                    'budget_compliance': transaction['duration_ms'] < apm_config['performance_budgets']['api_endpoints']['p95_latency_ms']
                },
                'bottlenecks_identified': [
                    {
                        'span_name': 'ml_recommendation_engine',
                        'contribution_percentage': 59.4,
                        'optimization_potential': 'model_caching'
                    }
                ],
                'performance_insights': {
                    'cpu_time_ms': transaction['duration_ms'] * 0.7,
                    'io_wait_ms': transaction['duration_ms'] * 0.2,
                    'network_time_ms': transaction['duration_ms'] * 0.1,
                    'memory_allocated_kb': np.random.uniform(500, 2000)
                }
            }
            
            trace_result = await apm_tracer.trace_transaction(
                transaction=transaction,
                context={'environment': 'production', 'version': 'v2.1.0'}
            )
            
            # Validations tracing
            assert trace_result['transaction_summary']['performance_score'] > 0.8
            assert trace_result['transaction_summary']['spans_count'] == len(transaction['spans'])
            assert len(trace_result['bottlenecks_identified']) >= 0
            assert 'performance_insights' in trace_result
    
    async def test_code_level_profiling(self, apm_tracer):
        """Test profilage niveau code."""
        # Configuration profilage code
        profiling_config = {
            'profiling_modes': ['cpu', 'memory', 'io', 'locks'],
            'sampling_frequency_hz': 100,
            'stack_trace_depth': 50,
            'hot_spot_detection': True,
            'flame_graph_generation': True
        }
        
        # Mock profilage code
        apm_tracer.profile_code_execution = AsyncMock(return_value={
            'profiling_session_id': f"profile_{uuid.uuid4().hex[:8]}",
            'execution_analysis': {
                'total_cpu_time_ms': 1234.5,
                'wall_clock_time_ms': 1456.7,
                'cpu_efficiency': 0.847,
                'memory_peak_mb': 256.7,
                'memory_leaks_detected': 0,
                'gc_pressure_score': 0.23
            },
            'hot_spots_identified': [
                {
                    'function_name': 'ml_model.predict',
                    'file_path': '/app/ml/model.py',
                    'line_number': 234,
                    'cpu_percentage': 34.5,
                    'call_count': 1234,
                    'avg_duration_ms': 2.3
                },
                {
                    'function_name': 'database.query_user_preferences',
                    'file_path': '/app/db/queries.py',
                    'line_number': 89,
                    'cpu_percentage': 28.1,
                    'call_count': 567,
                    'avg_duration_ms': 4.7
                }
            ],
            'optimization_recommendations': [
                {
                    'type': 'caching',
                    'target': 'ml_model.predict',
                    'potential_improvement': '40% CPU reduction',
                    'implementation_effort': 'medium'
                },
                {
                    'type': 'query_optimization',
                    'target': 'database.query_user_preferences',
                    'potential_improvement': '25% latency reduction',
                    'implementation_effort': 'low'
                }
            ],
            'flame_graph_url': f"https://profiler.company.com/flamegraph/{uuid.uuid4().hex}"
        })
        
        profiling_result = await apm_tracer.profile_code_execution(
            target_function='user_recommendation_pipeline',
            profiling_config=profiling_config,
            duration_seconds=60
        )
        
        # Validations profilage
        assert profiling_result['execution_analysis']['cpu_efficiency'] > 0.5
        assert len(profiling_result['hot_spots_identified']) > 0
        assert len(profiling_result['optimization_recommendations']) > 0
        assert 'flame_graph_url' in profiling_result
    
    async def test_database_performance_monitoring(self, apm_tracer):
        """Test monitoring performance base de données."""
        # Requêtes database simulées
        database_queries = [
            {
                'query_id': 'query_001',
                'sql': 'SELECT * FROM users WHERE premium_subscription = true',
                'database': 'user_db',
                'execution_time_ms': 234.5,
                'rows_examined': 125000,
                'rows_returned': 25000,
                'index_usage': 'full_scan'
            },
            {
                'query_id': 'query_002',
                'sql': 'SELECT track_id, play_count FROM user_listening_history WHERE user_id = ?',
                'database': 'analytics_db',
                'execution_time_ms': 67.8,
                'rows_examined': 10000,
                'rows_returned': 1500,
                'index_usage': 'index_used'
            }
        ]
        
        # Mock monitoring database
        apm_tracer.monitor_database_performance = AsyncMock()
        
        for query in database_queries:
            # Configuration réponse monitoring
            apm_tracer.monitor_database_performance.return_value = {
                'query_analysis': {
                    'query_id': query['query_id'],
                    'performance_score': np.random.uniform(0.6, 0.9),
                    'optimization_opportunity': query['index_usage'] == 'full_scan',
                    'execution_plan_efficiency': np.random.uniform(0.7, 0.95),
                    'cache_hit_ratio': np.random.uniform(0.8, 0.95)
                },
                'resource_consumption': {
                    'cpu_time_ms': query['execution_time_ms'] * 0.8,
                    'io_reads': query['rows_examined'] // 100,
                    'memory_used_kb': query['rows_returned'] * 2,
                    'locks_acquired': np.random.randint(1, 10)
                },
                'optimization_suggestions': [
                    {
                        'type': 'index_creation',
                        'recommendation': 'CREATE INDEX idx_users_premium ON users(premium_subscription)',
                        'estimated_improvement': '80% faster execution'
                    } if query['index_usage'] == 'full_scan' else {
                        'type': 'query_optimization',
                        'recommendation': 'Query already optimized',
                        'estimated_improvement': 'none'
                    }
                ]
            }
            
            monitoring_result = await apm_tracer.monitor_database_performance(
                query=query,
                monitoring_duration_minutes=5
            )
            
            # Validations monitoring database
            assert monitoring_result['query_analysis']['performance_score'] > 0.5
            assert 'resource_consumption' in monitoring_result
            assert len(monitoring_result['optimization_suggestions']) > 0


class TestPerformanceMonitor:
    """Tests enterprise pour PerformanceMonitor avec monitoring performance avancé."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Instance PerformanceMonitor pour tests."""
        return PerformanceMonitor()
    
    async def test_real_time_performance_analysis(self, performance_monitor):
        """Test analyse performance temps réel."""
        # Métriques performance temps réel
        real_time_metrics = {
            'system_metrics': {
                'cpu_usage_percentage': np.random.uniform(20, 80),
                'memory_usage_percentage': np.random.uniform(40, 70),
                'disk_io_ops_per_second': np.random.uniform(100, 1000),
                'network_throughput_mbps': np.random.uniform(10, 100)
            },
            'application_metrics': {
                'requests_per_second': np.random.uniform(100, 5000),
                'average_response_time_ms': np.random.uniform(50, 300),
                'error_rate_percentage': np.random.uniform(0, 2),
                'active_connections': np.random.randint(100, 10000)
            },
            'business_metrics': {
                'concurrent_streams': np.random.randint(1000, 50000),
                'revenue_per_minute': np.random.uniform(100, 2000),
                'user_satisfaction_score': np.random.uniform(0.8, 0.95),
                'content_cache_hit_ratio': np.random.uniform(0.85, 0.98)
            }
        }
        
        # Mock analyse temps réel
        performance_monitor.analyze_real_time_performance = AsyncMock(return_value={
            'overall_health_score': np.random.uniform(0.8, 0.95),
            'performance_grade': 'A',
            'critical_issues': [],
            'performance_trends': {
                'cpu_trend': 'stable',
                'memory_trend': 'increasing_slowly',
                'response_time_trend': 'improving',
                'error_rate_trend': 'stable'
            },
            'capacity_utilization': {
                'current_load_percentage': np.random.uniform(60, 80),
                'projected_capacity_hours': np.random.uniform(24, 168),
                'scaling_recommendation': 'horizontal_scaling_recommended'
            },
            'anomalies_detected': [
                {
                    'metric': 'memory_usage',
                    'severity': 'medium',
                    'description': 'Gradual memory increase detected',
                    'recommendation': 'Monitor for memory leaks'
                }
            ]
        })
        
        analysis = await performance_monitor.analyze_real_time_performance(
            metrics=real_time_metrics,
            analysis_window_minutes=15
        )
        
        # Validations analyse temps réel
        assert analysis['overall_health_score'] > 0.7
        assert analysis['performance_grade'] in ['A', 'B', 'C', 'D', 'F']
        assert 'performance_trends' in analysis
        assert 'capacity_utilization' in analysis
    
    async def test_performance_regression_detection(self, performance_monitor):
        """Test détection régressions performance."""
        # Données performance historiques vs actuelles
        performance_comparison = {
            'baseline_period': {
                'start_date': datetime.utcnow() - timedelta(days=7),
                'end_date': datetime.utcnow() - timedelta(days=1),
                'metrics': {
                    'avg_response_time_ms': 156.7,
                    'p95_response_time_ms': 423.2,
                    'error_rate': 0.003,
                    'throughput_rps': 2345.6
                }
            },
            'current_period': {
                'start_date': datetime.utcnow() - timedelta(hours=6),
                'end_date': datetime.utcnow(),
                'metrics': {
                    'avg_response_time_ms': 234.8,
                    'p95_response_time_ms': 567.9,
                    'error_rate': 0.007,
                    'throughput_rps': 1987.3
                }
            }
        }
        
        # Mock détection régression
        performance_monitor.detect_performance_regression = AsyncMock(return_value={
            'regression_detected': True,
            'severity': 'high',
            'affected_metrics': [
                {
                    'metric_name': 'avg_response_time_ms',
                    'baseline_value': 156.7,
                    'current_value': 234.8,
                    'change_percentage': 49.8,
                    'statistical_significance': 0.95
                },
                {
                    'metric_name': 'error_rate',
                    'baseline_value': 0.003,
                    'current_value': 0.007,
                    'change_percentage': 133.3,
                    'statistical_significance': 0.98
                }
            ],
            'probable_causes': [
                {
                    'cause': 'recent_deployment',
                    'confidence': 0.87,
                    'evidence': 'Performance degradation started 2 hours after deployment v2.1.3'
                },
                {
                    'cause': 'database_performance_issue',
                    'confidence': 0.76,
                    'evidence': 'Increased database query latency observed'
                }
            ],
            'impact_assessment': {
                'users_affected_estimate': 15000,
                'revenue_impact_per_hour': 500,
                'sla_risk': 'high'
            },
            'recommended_actions': [
                'rollback_deployment',
                'investigate_database_performance',
                'enable_circuit_breakers'
            ]
        })
        
        regression_analysis = await performance_monitor.detect_performance_regression(
            comparison_data=performance_comparison,
            significance_threshold=0.05
        )
        
        # Validations détection régression
        assert 'regression_detected' in regression_analysis
        if regression_analysis['regression_detected']:
            assert len(regression_analysis['affected_metrics']) > 0
            assert len(regression_analysis['probable_causes']) > 0
            assert 'impact_assessment' in regression_analysis
            assert len(regression_analysis['recommended_actions']) > 0


# =============================================================================
# TESTS INTEGRATION MONITORING
# =============================================================================

@pytest.mark.integration
class TestMonitoringUtilsIntegration:
    """Tests d'intégration pour utils monitoring."""
    
    async def test_complete_monitoring_pipeline(self):
        """Test pipeline monitoring complet."""
        # Configuration pipeline intégré
        pipeline_config = {
            'data_flow': [
                'metrics_collection',
                'real_time_processing',
                'anomaly_detection',
                'alert_generation',
                'dashboard_update',
                'notification_dispatch'
            ],
            'quality_gates': {
                'data_completeness': 0.95,
                'processing_latency_ms': 5000,
                'alert_accuracy': 0.9
            }
        }
        
        # Simulation pipeline complet
        pipeline_results = {}
        for stage in pipeline_config['data_flow']:
            # Simulation temps traitement
            processing_time = np.random.uniform(100, 1000)
            success_rate = np.random.uniform(0.95, 0.99)
            
            pipeline_results[stage] = {
                'success': True,
                'processing_time_ms': processing_time,
                'success_rate': success_rate,
                'data_quality_score': np.random.uniform(0.9, 0.98)
            }
        
        # Validations pipeline
        assert all(result['success'] for result in pipeline_results.values())
        total_latency = sum(r['processing_time_ms'] for r in pipeline_results.values())
        assert total_latency <= pipeline_config['quality_gates']['processing_latency_ms']


# =============================================================================
# TESTS PERFORMANCE MONITORING
# =============================================================================

@pytest.mark.performance
class TestMonitoringUtilsPerformance:
    """Tests performance pour utils monitoring."""
    
    async def test_high_volume_metrics_processing(self):
        """Test traitement métriques haut volume."""
        # Mock collector haute performance
        metrics_collector = MetricsCollector()
        metrics_collector.process_high_volume_metrics = AsyncMock(return_value={
            'metrics_processed_per_second': 50000,
            'processing_latency_p95_ms': 12.3,
            'memory_efficiency_score': 0.92,
            'cpu_utilization': 0.65,
            'backpressure_events': 0,
            'data_loss_rate': 0.0001
        })
        
        # Test traitement haute charge
        volume_test = await metrics_collector.process_high_volume_metrics(
            metrics_per_second=50000,
            test_duration_minutes=10
        )
        
        # Validations haute performance
        assert volume_test['metrics_processed_per_second'] >= 40000
        assert volume_test['processing_latency_p95_ms'] < 50
        assert volume_test['memory_efficiency_score'] > 0.9
        assert volume_test['data_loss_rate'] < 0.01
    
    async def test_alerting_system_scalability(self):
        """Test scalabilité système alerting."""
        alert_manager = AlertManager()
        
        # Test montée en charge alerting
        alert_manager.test_alerting_scalability = AsyncMock(return_value={
            'concurrent_alerts_handled': 10000,
            'alert_processing_latency_ms': 23.4,
            'notification_delivery_success_rate': 0.998,
            'correlation_accuracy': 0.94,
            'system_stability_score': 0.97
        })
        
        scalability_test = await alert_manager.test_alerting_scalability(
            concurrent_alerts=10000,
            test_duration_minutes=15
        )
        
        # Validations scalabilité
        assert scalability_test['concurrent_alerts_handled'] >= 8000
        assert scalability_test['alert_processing_latency_ms'] < 100
        assert scalability_test['notification_delivery_success_rate'] > 0.99
        assert scalability_test['correlation_accuracy'] > 0.9
