"""
Tests Ultra-Avancés pour Data Pipeline Middleware Enterprise
========================================================

Tests industriels complets pour pipeline de données avec streaming temps réel,
ETL/ELT, Kafka, validation de schémas, et patterns de test enterprise.

Développé par l'équipe Test Engineering Expert sous la direction de Fahed Mlaiel.
Architecture: Enterprise Data Pipeline Testing Framework avec Big Data patterns.
"""

import pytest
import asyncio
import time
import json
import uuid
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import psutil
import statistics
from dataclasses import dataclass

# Import du middleware à tester
from app.api.middleware.data_pipeline_middleware import (
    DataPipelineMiddleware,
    DataPipelineProcessor,
    MessageQueueManager,
    DataTransformer,
    SchemaValidator,
    DataQualityChecker,
    StreamProcessor,
    BatchProcessor,
    create_data_pipeline_middleware,
    DataPipelineConfig,
    PipelineStage,
    DataFormat,
    QualityRule,
    TransformationRule,
    ProcessingMode
)


# =============================================================================
# FIXTURES ENTERPRISE POUR DATA PIPELINE TESTING
# =============================================================================

@pytest.fixture
def pipeline_config():
    """Configuration enterprise data pipeline pour tests."""
    return DataPipelineConfig(
        kafka_bootstrap_servers=['localhost:9092'],
        redis_url="redis://localhost:6379/2",
        processing_mode=ProcessingMode.HYBRID,  # Stream + Batch
        batch_size=1000,
        batch_timeout_seconds=30,
        max_workers=4,
        enable_schema_validation=True,
        enable_data_quality_checks=True,
        enable_real_time_processing=True,
        enable_dead_letter_queue=True,
        compression_enabled=True,
        encryption_enabled=True,
        monitoring_enabled=True,
        backup_enabled=True,
        pipeline_stages=[
            'ingestion',
            'validation',
            'transformation',
            'enrichment',
            'quality_check',
            'output'
        ],
        supported_formats=[
            DataFormat.JSON,
            DataFormat.AVRO,
            DataFormat.PARQUET,
            DataFormat.CSV
        ]
    )

@pytest.fixture
def mock_kafka_producer():
    """Mock producteur Kafka avec comportement réaliste."""
    producer = Mock()
    
    # Stockage simulé des messages
    sent_messages = []
    
    async def send_mock(topic, value, key=None, headers=None):
        message = {
            'topic': topic,
            'value': value,
            'key': key,
            'headers': headers,
            'timestamp': time.time(),
            'offset': len(sent_messages)
        }
        sent_messages.append(message)
        
        # Simuler un future complété
        future = Mock()
        future.get.return_value = Mock(offset=len(sent_messages)-1, timestamp=time.time())
        return future
    
    async def flush_mock():
        return True
    
    producer.send = send_mock
    producer.flush = flush_mock
    producer.get_sent_messages = lambda: sent_messages.copy()
    
    return producer

@pytest.fixture
def mock_kafka_consumer():
    """Mock consommateur Kafka avec simulation de messages."""
    consumer = Mock()
    
    # Messages simulés
    message_queue = []
    
    def add_message(topic, value, key=None, headers=None):
        message = Mock()
        message.topic = topic
        message.value = value if isinstance(value, bytes) else json.dumps(value).encode()
        message.key = key
        message.headers = headers or {}
        message.offset = len(message_queue)
        message.timestamp = time.time()
        message_queue.append(message)
    
    async def poll_mock(timeout_ms=1000):
        if message_queue:
            return {Mock(): [message_queue.pop(0)]}
        return {}
    
    def subscribe_mock(topics):
        consumer.subscribed_topics = topics
    
    consumer.poll = poll_mock
    consumer.subscribe = subscribe_mock
    consumer.add_message = add_message
    consumer.commit = Mock()
    
    return consumer

@pytest.fixture
def mock_redis_client():
    """Mock client Redis pour queues et cache."""
    redis_client = AsyncMock()
    
    # Stockage simulé
    storage = {}
    queues = {}
    
    async def set_mock(key, value, ex=None):
        storage[key] = {'value': value, 'expires_at': time.time() + ex if ex else float('inf')}
        return True
    
    async def get_mock(key):
        data = storage.get(key)
        if data and data['expires_at'] > time.time():
            return data['value']
        elif key in storage:
            del storage[key]
        return None
    
    async def lpush_mock(queue_name, *values):
        if queue_name not in queues:
            queues[queue_name] = []
        queues[queue_name].extend(reversed(values))
        return len(queues[queue_name])
    
    async def rpop_mock(queue_name):
        if queue_name in queues and queues[queue_name]:
            return queues[queue_name].pop()
        return None
    
    async def llen_mock(queue_name):
        return len(queues.get(queue_name, []))
    
    redis_client.set = set_mock
    redis_client.get = get_mock
    redis_client.lpush = lpush_mock
    redis_client.rpop = rpop_mock
    redis_client.llen = llen_mock
    redis_client.flushdb = AsyncMock(return_value=True)
    
    return redis_client

@pytest.fixture
async def data_pipeline_middleware(pipeline_config, mock_kafka_producer, mock_kafka_consumer, mock_redis_client):
    """Middleware de pipeline de données configuré pour tests."""
    with patch('kafka.KafkaProducer', return_value=mock_kafka_producer), \
         patch('kafka.KafkaConsumer', return_value=mock_kafka_consumer), \
         patch('redis.from_url', return_value=mock_redis_client):
        
        middleware = DataPipelineMiddleware(pipeline_config)
        await middleware.initialize()
        yield middleware
        await middleware.cleanup()

@pytest.fixture
def sample_user_data():
    """Données utilisateur d'exemple pour tests."""
    return {
        'user_id': '12345',
        'timestamp': datetime.now().isoformat(),
        'event_type': 'user_interaction',
        'data': {
            'action': 'play_song',
            'song_id': 'song_67890',
            'artist': 'Test Artist',
            'album': 'Test Album',
            'duration_ms': 180000,
            'position_ms': 0,
            'context': {
                'playlist_id': 'playlist_123',
                'device_type': 'mobile',
                'location': 'US',
                'user_agent': 'SpotifyApp/1.0'
            }
        },
        'metadata': {
            'session_id': 'session_abc123',
            'request_id': 'req_def456',
            'api_version': 'v1',
            'processing_timestamp': time.time()
        }
    }

@pytest.fixture
def sample_analytics_data():
    """Données analytics d'exemple pour tests."""
    return {
        'event_id': str(uuid.uuid4()),
        'timestamp': datetime.now().isoformat(),
        'event_type': 'recommendation_interaction',
        'user_id': '12345',
        'recommendation_id': 'rec_789',
        'interaction_type': 'click',
        'recommendation_context': {
            'algorithm': 'collaborative_filtering',
            'model_version': '2.1.3',
            'confidence_score': 0.87,
            'recommendation_position': 3,
            'total_recommendations': 10
        },
        'user_context': {
            'listening_history_size': 1500,
            'favorite_genres': ['rock', 'jazz', 'electronic'],
            'premium_user': True,
            'country': 'US',
            'device_type': 'desktop'
        }
    }


# =============================================================================
# TESTS FONCTIONNELS ENTERPRISE
# =============================================================================

class TestDataPipelineMiddlewareFunctionality:
    """Tests fonctionnels complets du middleware de pipeline de données."""
    
    @pytest.mark.asyncio
    async def test_middleware_initialization(self, pipeline_config):
        """Test d'initialisation complète du middleware."""
        middleware = DataPipelineMiddleware(pipeline_config)
        
        # Vérifier l'état initial
        assert middleware.config == pipeline_config
        assert not middleware.is_initialized
        
        # Initialiser avec mocks
        with patch('kafka.KafkaProducer') as mock_producer, \
             patch('kafka.KafkaConsumer') as mock_consumer, \
             patch('redis.from_url') as mock_redis:
            
            await middleware.initialize()
            
            # Vérifier l'initialisation
            assert middleware.is_initialized
            assert middleware.kafka_producer is not None
            assert middleware.kafka_consumer is not None
            assert middleware.redis_client is not None
            assert middleware.processor is not None
            assert middleware.schema_validator is not None
            assert middleware.quality_checker is not None
            
            await middleware.cleanup()
    
    @pytest.mark.asyncio
    async def test_data_ingestion_and_processing(self, data_pipeline_middleware, sample_user_data):
        """Test d'ingestion et traitement de données."""
        # Ingérer des données
        ingestion_result = await data_pipeline_middleware.ingest_data(
            data=sample_user_data,
            source='user_events',
            format=DataFormat.JSON
        )
        
        assert ingestion_result['status'] == 'success'
        assert 'pipeline_id' in ingestion_result
        assert 'stage' in ingestion_result
        
        # Vérifier que les données sont dans le pipeline
        pipeline_status = await data_pipeline_middleware.get_pipeline_status(
            ingestion_result['pipeline_id']
        )
        
        assert pipeline_status is not None
        assert pipeline_status['current_stage'] in ['ingestion', 'validation']
        assert pipeline_status['total_stages'] == len(data_pipeline_middleware.config.pipeline_stages)
        
        # Traiter les données à travers le pipeline
        processing_result = await data_pipeline_middleware.process_pipeline_stage(
            pipeline_id=ingestion_result['pipeline_id'],
            stage='validation'
        )
        
        assert processing_result['status'] == 'success'
        assert processing_result['stage'] == 'validation'
    
    @pytest.mark.asyncio
    async def test_schema_validation(self, data_pipeline_middleware):
        """Test de validation de schéma."""
        # Définir un schéma de test
        user_event_schema = {
            'type': 'object',
            'required': ['user_id', 'timestamp', 'event_type', 'data'],
            'properties': {
                'user_id': {'type': 'string'},
                'timestamp': {'type': 'string', 'format': 'date-time'},
                'event_type': {'type': 'string', 'enum': ['user_interaction', 'system_event']},
                'data': {
                    'type': 'object',
                    'required': ['action'],
                    'properties': {
                        'action': {'type': 'string'},
                        'song_id': {'type': 'string'},
                        'duration_ms': {'type': 'integer', 'minimum': 0}
                    }
                }
            }
        }
        
        # Enregistrer le schéma
        await data_pipeline_middleware.register_schema(
            schema_name='user_event_v1',
            schema_definition=user_event_schema
        )
        
        # Tester avec des données valides
        valid_data = {
            'user_id': '12345',
            'timestamp': datetime.now().isoformat(),
            'event_type': 'user_interaction',
            'data': {
                'action': 'play_song',
                'song_id': 'song_123',
                'duration_ms': 180000
            }
        }
        
        validation_result = await data_pipeline_middleware.validate_schema(
            data=valid_data,
            schema_name='user_event_v1'
        )
        
        assert validation_result['is_valid'] is True
        assert len(validation_result['errors']) == 0
        
        # Tester avec des données invalides
        invalid_data = {
            'user_id': '12345',
            'event_type': 'invalid_type',  # Type non autorisé
            'data': {
                'action': 'play_song',
                'duration_ms': -100  # Valeur négative
            }
            # Manque 'timestamp' (requis)
        }
        
        validation_result = await data_pipeline_middleware.validate_schema(
            data=invalid_data,
            schema_name='user_event_v1'
        )
        
        assert validation_result['is_valid'] is False
        assert len(validation_result['errors']) > 0
        
        # Vérifier les types d'erreurs
        error_types = [error['type'] for error in validation_result['errors']]
        assert 'required_field_missing' in error_types
        assert 'enum_validation_failed' in error_types or 'value_validation_failed' in error_types
    
    @pytest.mark.asyncio
    async def test_data_transformation(self, data_pipeline_middleware, sample_user_data):
        """Test de transformation de données."""
        # Définir des règles de transformation
        transformation_rules = [
            {
                'name': 'normalize_timestamp',
                'type': 'field_transform',
                'source_field': 'timestamp',
                'target_field': 'normalized_timestamp',
                'function': 'to_unix_timestamp'
            },
            {
                'name': 'extract_country',
                'type': 'field_extract',
                'source_field': 'data.context.location',
                'target_field': 'country_code',
                'function': 'identity'
            },
            {
                'name': 'calculate_listening_session',
                'type': 'computed_field',
                'target_field': 'session_duration_estimate',
                'function': 'estimate_session_duration',
                'dependencies': ['data.duration_ms', 'data.position_ms']
            },
            {
                'name': 'enrich_user_tier',
                'type': 'lookup_enrichment',
                'source_field': 'user_id',
                'target_field': 'user_tier',
                'lookup_source': 'user_profile_cache'
            }
        ]
        
        # Enregistrer les règles
        for rule in transformation_rules:
            await data_pipeline_middleware.register_transformation_rule(rule)
        
        # Appliquer les transformations
        transformed_data = await data_pipeline_middleware.transform_data(
            data=sample_user_data,
            transformation_set='user_event_enrichment'
        )
        
        assert transformed_data is not None
        assert 'normalized_timestamp' in transformed_data
        assert 'country_code' in transformed_data
        assert 'session_duration_estimate' in transformed_data
        
        # Vérifier les transformations
        assert isinstance(transformed_data['normalized_timestamp'], (int, float))
        assert transformed_data['country_code'] == 'US'
        assert transformed_data['session_duration_estimate'] > 0
    
    @pytest.mark.asyncio
    async def test_data_quality_checks(self, data_pipeline_middleware):
        """Test de vérification de qualité des données."""
        # Définir des règles de qualité
        quality_rules = [
            {
                'name': 'user_id_not_null',
                'type': 'not_null',
                'field': 'user_id',
                'severity': 'critical'
            },
            {
                'name': 'timestamp_format_valid',
                'type': 'format_validation',
                'field': 'timestamp',
                'format': 'iso_datetime',
                'severity': 'critical'
            },
            {
                'name': 'duration_reasonable',
                'type': 'range_check',
                'field': 'data.duration_ms',
                'min_value': 1000,  # Au moins 1 seconde
                'max_value': 600000,  # Max 10 minutes
                'severity': 'warning'
            },
            {
                'name': 'event_type_whitelist',
                'type': 'whitelist_check',
                'field': 'event_type',
                'allowed_values': ['user_interaction', 'system_event', 'analytics_event'],
                'severity': 'error'
            }
        ]
        
        # Enregistrer les règles
        for rule in quality_rules:
            await data_pipeline_middleware.register_quality_rule(rule)
        
        # Tester avec des données de bonne qualité
        good_data = {
            'user_id': '12345',
            'timestamp': datetime.now().isoformat(),
            'event_type': 'user_interaction',
            'data': {
                'action': 'play_song',
                'duration_ms': 180000
            }
        }
        
        quality_result = await data_pipeline_middleware.check_data_quality(
            data=good_data,
            rule_set='user_event_quality'
        )
        
        assert quality_result['overall_score'] >= 90  # Score élevé
        assert quality_result['critical_issues'] == 0
        assert quality_result['status'] == 'pass'
        
        # Tester avec des données de mauvaise qualité
        bad_data = {
            'user_id': None,  # Null (critique)
            'timestamp': 'invalid_date',  # Format invalide (critique)
            'event_type': 'unknown_event',  # Pas dans whitelist (erreur)
            'data': {
                'action': 'play_song',
                'duration_ms': 1200000  # Trop long (warning)
            }
        }
        
        quality_result = await data_pipeline_middleware.check_data_quality(
            data=bad_data,
            rule_set='user_event_quality'
        )
        
        assert quality_result['overall_score'] < 50  # Score faible
        assert quality_result['critical_issues'] > 0
        assert quality_result['status'] == 'fail'
        
        # Vérifier les détails des problèmes
        issues = quality_result['issues']
        issue_types = [issue['rule_name'] for issue in issues]
        assert 'user_id_not_null' in issue_types
        assert 'timestamp_format_valid' in issue_types
        assert 'event_type_whitelist' in issue_types


# =============================================================================
# TESTS DE STREAMING ET TEMPS REEL
# =============================================================================

class TestRealTimeDataProcessing:
    """Tests de traitement de données en temps réel."""
    
    @pytest.mark.asyncio
    async def test_real_time_stream_processing(self, data_pipeline_middleware, mock_kafka_consumer):
        """Test de traitement de flux temps réel."""
        # Ajouter des messages de test au consumer
        test_messages = [
            {
                'user_id': f'user_{i}',
                'timestamp': datetime.now().isoformat(),
                'event_type': 'user_interaction',
                'data': {'action': 'play_song', 'song_id': f'song_{i}'}
            }
            for i in range(100)
        ]
        
        for msg in test_messages:
            mock_kafka_consumer.add_message(
                topic='user_events',
                value=msg,
                key=msg['user_id']
            )
        
        # Démarrer le traitement temps réel
        processed_messages = []
        
        async def message_handler(message_data):
            """Handler pour traiter les messages."""
            processed_messages.append(message_data)
            return {'status': 'processed', 'timestamp': time.time()}
        
        await data_pipeline_middleware.start_real_time_processing(
            topics=['user_events'],
            message_handler=message_handler,
            max_messages=100
        )
        
        # Attendre le traitement
        await asyncio.sleep(1.0)
        
        # Vérifier que tous les messages ont été traités
        assert len(processed_messages) == len(test_messages)
        
        # Vérifier l'ordre et l'intégrité
        for i, processed in enumerate(processed_messages):
            assert processed['user_id'] == f'user_{i}'
            assert processed['event_type'] == 'user_interaction'
        
        # Arrêter le traitement
        await data_pipeline_middleware.stop_real_time_processing()
    
    @pytest.mark.asyncio
    async def test_stream_aggregation(self, data_pipeline_middleware):
        """Test d'agrégation de flux en temps réel."""
        # Définir des fenêtres d'agrégation
        aggregation_config = {
            'window_size_seconds': 10,
            'slide_interval_seconds': 5,
            'aggregations': [
                {'field': 'data.duration_ms', 'function': 'avg', 'alias': 'avg_duration'},
                {'field': 'user_id', 'function': 'count_distinct', 'alias': 'unique_users'},
                {'field': 'data.action', 'function': 'count', 'alias': 'total_actions'},
                {'field': 'data.duration_ms', 'function': 'sum', 'alias': 'total_listening_time'}
            ],
            'group_by': ['data.action', 'data.context.device_type']
        }
        
        # Configurer l'agrégation
        await data_pipeline_middleware.configure_stream_aggregation(
            stream_name='user_interactions',
            config=aggregation_config
        )
        
        # Simuler des événements sur une période
        events = []
        for i in range(50):
            event = {
                'user_id': f'user_{i % 10}',  # 10 utilisateurs uniques
                'timestamp': (datetime.now() + timedelta(seconds=i*0.2)).isoformat(),
                'event_type': 'user_interaction',
                'data': {
                    'action': 'play_song' if i % 2 == 0 else 'skip_song',
                    'duration_ms': np.random.randint(30000, 300000),
                    'context': {
                        'device_type': 'mobile' if i % 3 == 0 else 'desktop'
                    }
                }
            }
            events.append(event)
            
            # Ingérer l'événement
            await data_pipeline_middleware.ingest_stream_event(
                stream_name='user_interactions',
                event=event
            )
        
        # Attendre l'agrégation
        await asyncio.sleep(1.0)
        
        # Récupérer les résultats d'agrégation
        aggregation_results = await data_pipeline_middleware.get_stream_aggregation_results(
            stream_name='user_interactions'
        )
        
        assert len(aggregation_results) > 0
        
        # Vérifier la structure des résultats
        for result in aggregation_results:
            assert 'window_start' in result
            assert 'window_end' in result
            assert 'group_keys' in result
            assert 'aggregated_values' in result
            
            agg_values = result['aggregated_values']
            assert 'avg_duration' in agg_values
            assert 'unique_users' in agg_values
            assert 'total_actions' in agg_values
            assert 'total_listening_time' in agg_values
            
            # Vérifier la logique des agrégations
            assert agg_values['avg_duration'] > 0
            assert 1 <= agg_values['unique_users'] <= 10
            assert agg_values['total_actions'] > 0
    
    @pytest.mark.asyncio
    async def test_event_deduplication(self, data_pipeline_middleware):
        """Test de déduplication d'événements."""
        # Créer des événements avec duplicatas
        base_event = {
            'user_id': '12345',
            'timestamp': datetime.now().isoformat(),
            'event_type': 'user_interaction',
            'data': {
                'action': 'play_song',
                'song_id': 'song_123'
            },
            'metadata': {
                'session_id': 'session_abc',
                'request_id': 'req_def'
            }
        }
        
        # Créer des versions légèrement différentes du même événement
        duplicate_events = []
        for i in range(5):
            event = base_event.copy()
            event['metadata'] = base_event['metadata'].copy()
            event['metadata']['processing_attempt'] = i + 1
            event['metadata']['processing_timestamp'] = time.time() + i * 0.1
            duplicate_events.append(event)
        
        # Configurer la déduplication
        deduplication_config = {
            'deduplication_key_fields': ['user_id', 'data.action', 'data.song_id', 'metadata.session_id'],
            'deduplication_window_seconds': 60,
            'strategy': 'keep_first'  # Garder le premier événement
        }
        
        await data_pipeline_middleware.configure_deduplication(
            stream_name='user_interactions',
            config=deduplication_config
        )
        
        # Ingérer tous les événements duplicata
        ingestion_results = []
        for event in duplicate_events:
            result = await data_pipeline_middleware.ingest_data(
                data=event,
                source='user_events',
                format=DataFormat.JSON,
                enable_deduplication=True
            )
            ingestion_results.append(result)
        
        # Vérifier que seul le premier événement a été accepté
        accepted_events = [r for r in ingestion_results if r['status'] == 'success']
        duplicated_events = [r for r in ingestion_results if r['status'] == 'duplicate']
        
        assert len(accepted_events) == 1
        assert len(duplicated_events) == 4
        
        # Vérifier les métadonnées de déduplication
        for dup_result in duplicated_events:
            assert 'deduplication_info' in dup_result
            assert dup_result['deduplication_info']['reason'] == 'duplicate_key_found'
            assert 'original_event_id' in dup_result['deduplication_info']


# =============================================================================
# TESTS DE TRAITEMENT BATCH
# =============================================================================

class TestBatchDataProcessing:
    """Tests de traitement de données par lot."""
    
    @pytest.mark.asyncio
    async def test_batch_processing_workflow(self, data_pipeline_middleware):
        """Test de workflow de traitement par lot."""
        # Créer un lot de données de test
        batch_data = []
        for i in range(1000):
            event = {
                'user_id': f'user_{i % 100}',
                'timestamp': (datetime.now() - timedelta(hours=24) + timedelta(minutes=i)).isoformat(),
                'event_type': 'user_interaction',
                'data': {
                    'action': np.random.choice(['play_song', 'skip_song', 'like_song', 'add_to_playlist']),
                    'song_id': f'song_{np.random.randint(1, 1000)}',
                    'artist_id': f'artist_{np.random.randint(1, 100)}',
                    'duration_ms': np.random.randint(30000, 300000),
                    'context': {
                        'device_type': np.random.choice(['mobile', 'desktop', 'tablet']),
                        'location': np.random.choice(['US', 'UK', 'DE', 'FR', 'CA'])
                    }
                }
            }
            batch_data.append(event)
        
        # Configurer le traitement par lot
        batch_config = {
            'batch_size': 100,
            'processing_stages': [
                'validation',
                'deduplication',
                'transformation',
                'quality_check',
                'aggregation',
                'output'
            ],
            'output_format': DataFormat.PARQUET,
            'partition_by': ['data.context.location', 'date'],
            'enable_parallel_processing': True,
            'max_workers': 4
        }
        
        # Soumettre le lot pour traitement
        batch_job = await data_pipeline_middleware.submit_batch_job(
            data=batch_data,
            job_name='daily_user_interactions_processing',
            config=batch_config
        )
        
        assert batch_job['job_id'] is not None
        assert batch_job['status'] == 'submitted'
        assert batch_job['total_records'] == len(batch_data)
        
        # Surveiller le progrès du traitement
        max_wait_time = 30  # 30 secondes max
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            job_status = await data_pipeline_middleware.get_batch_job_status(batch_job['job_id'])
            
            if job_status['status'] in ['completed', 'failed']:
                break
            
            await asyncio.sleep(1)
        
        # Vérifier le résultat final
        final_status = await data_pipeline_middleware.get_batch_job_status(batch_job['job_id'])
        
        assert final_status['status'] == 'completed'
        assert final_status['processed_records'] > 0
        assert final_status['failed_records'] <= final_status['total_records'] * 0.05  # Max 5% d'échec
        
        # Vérifier les résultats de sortie
        output_summary = final_status['output_summary']
        assert 'partitions_created' in output_summary
        assert 'total_output_size_bytes' in output_summary
        assert output_summary['partitions_created'] > 0
    
    @pytest.mark.asyncio
    async def test_batch_data_quality_reporting(self, data_pipeline_middleware):
        """Test de reporting de qualité pour traitement par lot."""
        # Créer des données avec problèmes de qualité connus
        batch_data = []
        
        # Données de bonne qualité (70%)
        for i in range(700):
            event = {
                'user_id': f'valid_user_{i}',
                'timestamp': datetime.now().isoformat(),
                'event_type': 'user_interaction',
                'data': {
                    'action': 'play_song',
                    'song_id': f'song_{i}',
                    'duration_ms': np.random.randint(30000, 300000)
                }
            }
            batch_data.append(event)
        
        # Données avec problèmes (30%)
        for i in range(300):
            if i % 3 == 0:
                # User ID manquant
                event = {
                    'user_id': None,
                    'timestamp': datetime.now().isoformat(),
                    'event_type': 'user_interaction',
                    'data': {'action': 'play_song', 'song_id': f'song_{i}'}
                }
            elif i % 3 == 1:
                # Timestamp invalide
                event = {
                    'user_id': f'user_{i}',
                    'timestamp': 'invalid_timestamp',
                    'event_type': 'user_interaction',
                    'data': {'action': 'play_song', 'song_id': f'song_{i}'}
                }
            else:
                # Durée négative
                event = {
                    'user_id': f'user_{i}',
                    'timestamp': datetime.now().isoformat(),
                    'event_type': 'user_interaction',
                    'data': {
                        'action': 'play_song',
                        'song_id': f'song_{i}',
                        'duration_ms': -1000
                    }
                }
            batch_data.append(event)
        
        # Traiter le lot avec reporting de qualité
        batch_job = await data_pipeline_middleware.submit_batch_job(
            data=batch_data,
            job_name='quality_test_batch',
            config={
                'enable_quality_reporting': True,
                'quality_thresholds': {
                    'min_overall_score': 80,
                    'max_critical_issues_percent': 5,
                    'max_error_rate_percent': 10
                }
            }
        )
        
        # Attendre la completion
        await asyncio.sleep(2)
        
        # Récupérer le rapport de qualité
        quality_report = await data_pipeline_middleware.get_batch_quality_report(batch_job['job_id'])
        
        assert quality_report is not None
        assert 'overall_quality_score' in quality_report
        assert 'issue_summary' in quality_report
        assert 'recommendations' in quality_report
        
        # Vérifier les métriques de qualité
        assert quality_report['overall_quality_score'] < 80  # Score faible à cause des problèmes
        assert quality_report['total_records'] == len(batch_data)
        assert quality_report['valid_records'] == 700
        assert quality_report['invalid_records'] == 300
        
        # Vérifier le résumé des problèmes
        issue_summary = quality_report['issue_summary']
        assert 'null_user_id' in [issue['type'] for issue in issue_summary]
        assert 'invalid_timestamp_format' in [issue['type'] for issue in issue_summary]
        assert 'negative_duration' in [issue['type'] for issue in issue_summary]


# =============================================================================
# TESTS DE PERFORMANCE ET SCALABILITE
# =============================================================================

class TestDataPipelinePerformance:
    """Tests de performance et scalabilité du pipeline."""
    
    @pytest.mark.asyncio
    async def test_throughput_benchmark(self, data_pipeline_middleware):
        """Test de débit du pipeline de données."""
        num_records = 10000
        batch_size = 1000
        
        # Générer des données de test
        test_data = []
        for i in range(num_records):
            record = {
                'id': i,
                'timestamp': datetime.now().isoformat(),
                'data': {
                    'value': np.random.random(),
                    'category': f'cat_{i % 10}',
                    'metadata': {'batch_id': i // batch_size}
                }
            }
            test_data.append(record)
        
        # Mesurer le débit d'ingestion
        start_time = time.time()
        
        ingestion_tasks = []
        for i in range(0, num_records, batch_size):
            batch = test_data[i:i + batch_size]
            task = data_pipeline_middleware.ingest_data_batch(
                data=batch,
                source='performance_test',
                format=DataFormat.JSON
            )
            ingestion_tasks.append(task)
        
        # Attendre toutes les ingestions
        results = await asyncio.gather(*ingestion_tasks)
        ingestion_time = time.time() - start_time
        
        # Calculer les métriques de performance
        ingestion_throughput = num_records / ingestion_time
        
        # Vérifier les résultats
        successful_ingestions = sum(1 for r in results if r['status'] == 'success')
        assert successful_ingestions == len(ingestion_tasks)
        
        # Assertions de performance
        assert ingestion_throughput > 1000  # Au moins 1000 records/sec
        
        print(f"Ingestion throughput: {ingestion_throughput:.2f} records/sec")
        
        # Mesurer le débit de traitement
        start_time = time.time()
        
        processing_results = await data_pipeline_middleware.process_all_pending_data()
        processing_time = time.time() - start_time
        
        processing_throughput = num_records / processing_time
        assert processing_throughput > 500  # Au moins 500 records/sec en traitement
        
        print(f"Processing throughput: {processing_throughput:.2f} records/sec")
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, data_pipeline_middleware):
        """Test d'efficacité mémoire du pipeline."""
        import gc
        
        initial_memory = psutil.Process().memory_info().rss
        
        # Traiter beaucoup de données par vagues
        for wave in range(5):
            # Générer une vague de données
            wave_data = []
            for i in range(5000):
                record = {
                    'wave': wave,
                    'id': i,
                    'timestamp': datetime.now().isoformat(),
                    'data': {
                        'payload': 'x' * 1000,  # 1KB par record
                        'metadata': {'wave_id': wave}
                    }
                }
                wave_data.append(record)
            
            # Traiter la vague
            await data_pipeline_middleware.ingest_data_batch(
                data=wave_data,
                source='memory_test',
                format=DataFormat.JSON
            )
            
            # Forcer le nettoyage de la vague précédente
            await data_pipeline_middleware.cleanup_processed_data(
                retention_seconds=1
            )
            
            # Mesurer la mémoire
            current_memory = psutil.Process().memory_info().rss
            memory_growth = current_memory - initial_memory
            
            # La croissance mémoire ne doit pas être linéaire avec les données
            max_acceptable_growth = 100 * 1024 * 1024  # 100MB max
            assert memory_growth < max_acceptable_growth
            
            # Forcer garbage collection
            gc.collect()
        
        final_memory = psutil.Process().memory_info().rss
        total_growth = final_memory - initial_memory
        
        print(f"Total memory growth: {total_growth / 1024 / 1024:.2f} MB")
        assert total_growth < 150 * 1024 * 1024  # Max 150MB au total
    
    @pytest.mark.asyncio
    async def test_concurrent_pipeline_processing(self, data_pipeline_middleware):
        """Test de traitement concurrent de multiples pipelines."""
        num_pipelines = 10
        records_per_pipeline = 1000
        
        async def process_pipeline(pipeline_id):
            """Traiter un pipeline individuel."""
            pipeline_data = []
            for i in range(records_per_pipeline):
                record = {
                    'pipeline_id': pipeline_id,
                    'record_id': i,
                    'timestamp': datetime.now().isoformat(),
                    'data': {
                        'value': np.random.random(),
                        'pipeline_specific': f'pipeline_{pipeline_id}_value_{i}'
                    }
                }
                pipeline_data.append(record)
            
            # Traitement complet du pipeline
            ingestion_result = await data_pipeline_middleware.ingest_data_batch(
                data=pipeline_data,
                source=f'pipeline_{pipeline_id}',
                format=DataFormat.JSON
            )
            
            processing_result = await data_pipeline_middleware.process_pipeline_data(
                source=f'pipeline_{pipeline_id}',
                config={'enable_validation': True, 'enable_transformation': True}
            )
            
            return {
                'pipeline_id': pipeline_id,
                'ingestion_status': ingestion_result['status'],
                'processing_status': processing_result['status'],
                'records_processed': len(pipeline_data)
            }
        
        # Exécuter tous les pipelines en parallèle
        start_time = time.time()
        
        pipeline_tasks = [process_pipeline(i) for i in range(num_pipelines)]
        results = await asyncio.gather(*pipeline_tasks)
        
        total_time = time.time() - start_time
        
        # Vérifier les résultats
        successful_pipelines = sum(1 for r in results 
                                 if r['ingestion_status'] == 'success' 
                                 and r['processing_status'] == 'success')
        
        assert successful_pipelines == num_pipelines
        
        total_records = sum(r['records_processed'] for r in results)
        overall_throughput = total_records / total_time
        
        assert overall_throughput > 2000  # Au moins 2000 records/sec en parallèle
        
        print(f"Concurrent processing throughput: {overall_throughput:.2f} records/sec")
        print(f"Pipelines processed: {successful_pipelines}/{num_pipelines}")


# =============================================================================
# TESTS D'INTEGRATION COMPLETE
# =============================================================================

@pytest.mark.integration
class TestDataPipelineIntegrationComplete:
    """Tests d'intégration complète du système de pipeline de données."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_data_pipeline(self, pipeline_config):
        """Test de pipeline de données complet bout-en-bout."""
        with patch('kafka.KafkaProducer') as mock_producer, \
             patch('kafka.KafkaConsumer') as mock_consumer, \
             patch('redis.from_url') as mock_redis:
            
            # Configuration des mocks
            mock_producer.return_value = Mock()
            mock_consumer.return_value = Mock()
            mock_redis.return_value = AsyncMock()
            
            # Initialisation du pipeline
            middleware = DataPipelineMiddleware(pipeline_config)
            await middleware.initialize()
            
            try:
                # 1. Configuration du pipeline complet
                pipeline_config_full = {
                    'stages': [
                        {'name': 'ingestion', 'enabled': True},
                        {'name': 'validation', 'enabled': True},
                        {'name': 'deduplication', 'enabled': True},
                        {'name': 'transformation', 'enabled': True},
                        {'name': 'quality_check', 'enabled': True},
                        {'name': 'enrichment', 'enabled': True},
                        {'name': 'aggregation', 'enabled': True},
                        {'name': 'output', 'enabled': True}
                    ],
                    'processing_mode': 'hybrid',  # Stream + Batch
                    'monitoring_enabled': True
                }
                
                await middleware.configure_pipeline('comprehensive_test', pipeline_config_full)
                
                # 2. Ingestion de données diversifiées
                data_sources = [
                    {
                        'name': 'user_events',
                        'format': DataFormat.JSON,
                        'volume': 1000
                    },
                    {
                        'name': 'analytics_events',
                        'format': DataFormat.AVRO,
                        'volume': 500
                    },
                    {
                        'name': 'system_logs',
                        'format': DataFormat.JSON,
                        'volume': 2000
                    }
                ]
                
                total_ingested = 0
                
                for source in data_sources:
                    # Générer des données de test
                    test_data = []
                    for i in range(source['volume']):
                        record = {
                            'source': source['name'],
                            'id': f"{source['name']}_{i}",
                            'timestamp': datetime.now().isoformat(),
                            'data': {
                                'value': np.random.random(),
                                'category': f"cat_{i % 5}",
                                'metadata': {'source_type': source['name']}
                            }
                        }
                        test_data.append(record)
                    
                    # Ingérer les données
                    ingestion_result = await middleware.ingest_data_batch(
                        data=test_data,
                        source=source['name'],
                        format=source['format']
                    )
                    
                    assert ingestion_result['status'] == 'success'
                    total_ingested += source['volume']
                
                # 3. Traitement complet du pipeline
                processing_result = await middleware.execute_full_pipeline(
                    pipeline_name='comprehensive_test'
                )
                
                assert processing_result['status'] == 'completed'
                assert processing_result['total_records_processed'] == total_ingested
                
                # 4. Validation des résultats de sortie
                output_summary = await middleware.get_pipeline_output_summary(
                    pipeline_name='comprehensive_test'
                )
                
                assert 'total_output_records' in output_summary
                assert 'data_quality_score' in output_summary
                assert 'processing_time_seconds' in output_summary
                
                # Vérifier la qualité des données
                assert output_summary['data_quality_score'] > 85  # Au moins 85% de qualité
                
                # 5. Métriques et monitoring
                pipeline_metrics = await middleware.get_pipeline_metrics(
                    pipeline_name='comprehensive_test'
                )
                
                assert 'throughput' in pipeline_metrics
                assert 'latency_percentiles' in pipeline_metrics
                assert 'error_rates' in pipeline_metrics
                assert 'resource_utilization' in pipeline_metrics
                
                # Vérifier les performances
                assert pipeline_metrics['throughput']['records_per_second'] > 100
                assert pipeline_metrics['latency_percentiles']['p95'] < 1000  # < 1s
                assert pipeline_metrics['error_rates']['overall'] < 0.05  # < 5%
                
            finally:
                await middleware.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
