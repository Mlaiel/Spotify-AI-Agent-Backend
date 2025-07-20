# 🧪 Tests pour Event Streaming Engine
# ====================================
# 
# Tests complets pour le moteur de streaming d'événements
# avec tests de performance, fiabilité et intégration ML.
#
# 🎖️ Expert: Event Streaming Specialist + ML Test Engineer
#
# 👨‍💻 Développé par: Fahed Mlaiel
# ====================================

"""
🌊 Event Streaming Tests
=======================

Comprehensive test suite for the Event Streaming Engine:
- Event processing pipeline tests
- Kafka integration and reliability tests
- ML handler and recommendation tests
- Dead letter queue and error handling
- Performance and throughput tests
- Event aggregation and analytics tests
- Real-time processing validation
- Fault tolerance and recovery tests
"""

import asyncio
import json
import pytest
import time
import uuid
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import du module à tester
from app.realtime.event_streaming import (
    EventStreamEngine,
    StreamEvent,
    EventHandler,
    MusicPlayHandler,
    RecommendationHandler,
    EventAggregator,
    EventProcessor,
    DeadLetterQueue,
    KafkaProducer,
    KafkaConsumer,
    EventType,
    ProcessingStatus
)

from . import TestUtils, REDIS_TEST_URL


class TestStreamEvent:
    """Tests pour StreamEvent"""
    
    def test_event_creation(self):
        """Test de création d'événement"""
        event_data = {
            "user_id": "test_user",
            "track_id": "test_track",
            "action": "play"
        }
        
        event = StreamEvent(
            event_id="test_event_1",
            event_type=EventType.MUSIC_PLAY,
            data=event_data,
            user_id="test_user"
        )
        
        assert event.event_id == "test_event_1"
        assert event.event_type == EventType.MUSIC_PLAY
        assert event.data == event_data
        assert event.user_id == "test_user"
        assert event.created_at is not None
        assert event.processing_status == ProcessingStatus.PENDING
    
    def test_event_serialization(self):
        """Test de sérialisation d'événement"""
        event = StreamEvent(
            event_id="test_event_1",
            event_type=EventType.USER_ACTION,
            data={"action": "click"},
            user_id="test_user"
        )
        
        serialized = event.to_dict()
        
        assert "event_id" in serialized
        assert "event_type" in serialized
        assert "data" in serialized
        assert "user_id" in serialized
        assert "created_at" in serialized
        assert "processing_status" in serialized
    
    def test_event_deserialization(self):
        """Test de désérialisation d'événement"""
        event_dict = {
            "event_id": "test_event_1",
            "event_type": "user_action",
            "data": {"action": "click"},
            "user_id": "test_user",
            "created_at": datetime.utcnow().isoformat(),
            "processing_status": "pending"
        }
        
        event = StreamEvent.from_dict(event_dict)
        
        assert event.event_id == "test_event_1"
        assert event.event_type == EventType.USER_ACTION
        assert event.data == {"action": "click"}
        assert event.user_id == "test_user"


class TestEventHandler:
    """Tests pour EventHandler de base"""
    
    @pytest.fixture
    def handler(self):
        """Handler de test"""
        return EventHandler(handler_id="test_handler")
    
    @pytest.mark.asyncio
    async def test_handler_initialization(self, handler):
        """Test d'initialisation du handler"""
        assert handler.handler_id == "test_handler"
        assert handler.processed_count == 0
        assert handler.error_count == 0
        assert handler.is_enabled is True
    
    @pytest.mark.asyncio
    async def test_handler_can_handle(self, handler):
        """Test de vérification de capacité de traitement"""
        event = StreamEvent(
            event_id="test",
            event_type=EventType.USER_ACTION,
            data={},
            user_id="test"
        )
        
        # Par défaut, peut traiter tous les événements
        can_handle = await handler.can_handle(event)
        assert can_handle is True
    
    @pytest.mark.asyncio
    async def test_handler_process_success(self, handler):
        """Test de traitement réussi"""
        event = StreamEvent(
            event_id="test",
            event_type=EventType.USER_ACTION,
            data={},
            user_id="test"
        )
        
        result = await handler.process(event)
        
        assert result.success is True
        assert handler.processed_count == 1
        assert handler.error_count == 0
    
    @pytest.mark.asyncio
    async def test_handler_process_error(self, handler):
        """Test de gestion d'erreur dans le traitement"""
        # Override process pour lever une exception
        async def failing_process(event):
            raise Exception("Processing failed")
        
        handler.process = failing_process
        
        event = StreamEvent(
            event_id="test",
            event_type=EventType.USER_ACTION,
            data={},
            user_id="test"
        )
        
        result = await handler.process(event)
        
        assert result.success is False
        assert "Processing failed" in result.error_message


class TestMusicPlayHandler:
    """Tests pour MusicPlayHandler"""
    
    @pytest.fixture
    async def handler(self):
        """Handler de lecture musicale de test"""
        config = {
            "redis_url": REDIS_TEST_URL,
            "analytics_enabled": True
        }
        
        handler = MusicPlayHandler(config)
        await handler.initialize()
        
        yield handler
        
        await handler.cleanup()
    
    @pytest.mark.asyncio
    async def test_music_play_processing(self, handler):
        """Test de traitement d'événement de lecture"""
        event = StreamEvent(
            event_id="test_play",
            event_type=EventType.MUSIC_PLAY,
            data={
                "track_id": "track_123",
                "playlist_id": "playlist_456", 
                "position": 0,
                "duration": 180,
                "quality": "high"
            },
            user_id="user_789"
        )
        
        result = await handler.process(event)
        
        assert result.success is True
        assert result.data["track_id"] == "track_123"
        assert result.data["user_id"] == "user_789"
        assert "play_session_id" in result.data
    
    @pytest.mark.asyncio
    async def test_play_analytics_update(self, handler):
        """Test de mise à jour des analytics de lecture"""
        track_id = "track_test_123"
        user_id = "user_test_456"
        
        event = StreamEvent(
            event_id="test_analytics",
            event_type=EventType.MUSIC_PLAY,
            data={
                "track_id": track_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            user_id=user_id
        )
        
        await handler.process(event)
        
        # Vérifier que les analytics sont mises à jour
        stats = await handler.get_track_stats(track_id)
        assert stats["total_plays"] >= 1
        assert stats["unique_listeners"] >= 1
    
    @pytest.mark.asyncio
    async def test_concurrent_plays(self, handler):
        """Test de lectures concurrentes"""
        tasks = []
        
        for i in range(10):
            event = StreamEvent(
                event_id=f"concurrent_play_{i}",
                event_type=EventType.MUSIC_PLAY,
                data={
                    "track_id": f"track_{i}",
                    "user_id": f"user_{i}"
                },
                user_id=f"user_{i}"
            )
            
            tasks.append(handler.process(event))
        
        results = await asyncio.gather(*tasks)
        
        # Tous les traitements devraient réussir
        assert all(result.success for result in results)
    
    @pytest.mark.asyncio
    async def test_play_session_tracking(self, handler):
        """Test de suivi de session de lecture"""
        user_id = "session_test_user"
        
        # Démarrer une session
        play_event = StreamEvent(
            event_id="session_start",
            event_type=EventType.MUSIC_PLAY,
            data={"track_id": "track_1"},
            user_id=user_id
        )
        
        result = await handler.process(play_event)
        session_id = result.data.get("play_session_id")
        
        # Vérifier la session
        session_data = await handler.get_user_session(user_id)
        assert session_data["session_id"] == session_id
        assert session_data["current_track"] == "track_1"


class TestRecommendationHandler:
    """Tests pour RecommendationHandler"""
    
    @pytest.fixture
    async def handler(self):
        """Handler de recommandations de test"""
        config = {
            "redis_url": REDIS_TEST_URL,
            "ml_enabled": True,
            "recommendation_threshold": 0.7
        }
        
        handler = RecommendationHandler(config)
        await handler.initialize()
        
        # Mock du modèle ML
        handler.recommendation_model = Mock()
        handler.recommendation_model.predict = Mock(return_value=[0.8, 0.9, 0.6])
        
        yield handler
        
        await handler.cleanup()
    
    @pytest.mark.asyncio
    async def test_recommendation_generation(self, handler):
        """Test de génération de recommandations"""
        event = StreamEvent(
            event_id="rec_test",
            event_type=EventType.RECOMMENDATION_REQUEST,
            data={
                "user_id": "user_123",
                "context": "playlist_creation",
                "preferences": {"genre": "rock", "mood": "energetic"}
            },
            user_id="user_123"
        )
        
        result = await handler.process(event)
        
        assert result.success is True
        assert "recommendations" in result.data
        assert len(result.data["recommendations"]) > 0
    
    @pytest.mark.asyncio
    async def test_personalized_recommendations(self, handler):
        """Test de recommandations personnalisées"""
        user_id = "personalized_user"
        
        # Simuler l'historique utilisateur
        await handler.update_user_history(user_id, {
            "liked_tracks": ["track_1", "track_2"],
            "genres": ["rock", "pop"],
            "mood_preferences": ["energetic", "happy"]
        })
        
        event = StreamEvent(
            event_id="personalized_rec",
            event_type=EventType.RECOMMENDATION_REQUEST,
            data={"context": "daily_mix"},
            user_id=user_id
        )
        
        result = await handler.process(event)
        
        assert result.success is True
        recommendations = result.data["recommendations"]
        
        # Vérifier que les recommandations sont personnalisées
        assert any("rock" in rec.get("genre", "") for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_recommendation_feedback(self, handler):
        """Test de feedback sur les recommandations"""
        user_id = "feedback_user"
        track_id = "recommended_track"
        
        # Événement de feedback
        feedback_event = StreamEvent(
            event_id="feedback_test",
            event_type=EventType.RECOMMENDATION_FEEDBACK,
            data={
                "track_id": track_id,
                "feedback": "liked",
                "recommendation_id": "rec_123"
            },
            user_id=user_id
        )
        
        result = await handler.process(feedback_event)
        
        assert result.success is True
        
        # Vérifier que le feedback est enregistré
        feedback_data = await handler.get_recommendation_feedback(user_id)
        assert track_id in feedback_data["liked_tracks"]
    
    @pytest.mark.asyncio
    async def test_cold_start_recommendations(self, handler):
        """Test de recommandations pour nouveaux utilisateurs"""
        new_user_id = "new_user_123"
        
        event = StreamEvent(
            event_id="cold_start",
            event_type=EventType.RECOMMENDATION_REQUEST,
            data={"context": "new_user_onboarding"},
            user_id=new_user_id
        )
        
        result = await handler.process(event)
        
        assert result.success is True
        
        # Les nouveaux utilisateurs devraient recevoir des recommandations populaires
        recommendations = result.data["recommendations"]
        assert len(recommendations) > 0
        assert all(rec.get("popularity_score", 0) > 0.5 for rec in recommendations)


class TestEventAggregator:
    """Tests pour EventAggregator"""
    
    @pytest.fixture
    async def aggregator(self):
        """Agrégateur d'événements de test"""
        config = {
            "redis_url": REDIS_TEST_URL,
            "window_size": 60,  # 1 minute
            "aggregation_interval": 10  # 10 secondes
        }
        
        aggregator = EventAggregator(config)
        await aggregator.initialize()
        
        yield aggregator
        
        await aggregator.cleanup()
    
    @pytest.mark.asyncio
    async def test_event_aggregation(self, aggregator):
        """Test d'agrégation d'événements"""
        # Ajouter plusieurs événements
        events = []
        for i in range(5):
            event = StreamEvent(
                event_id=f"agg_test_{i}",
                event_type=EventType.USER_ACTION,
                data={"action": "click", "value": i * 10},
                user_id=f"user_{i % 2}"  # 2 utilisateurs différents
            )
            events.append(event)
            await aggregator.add_event(event)
        
        # Déclencher l'agrégation
        aggregations = await aggregator.aggregate_events()
        
        assert len(aggregations) > 0
        
        # Vérifier les métriques agrégées
        user_metrics = aggregations.get("user_metrics", {})
        assert len(user_metrics) == 2  # 2 utilisateurs
    
    @pytest.mark.asyncio
    async def test_time_window_aggregation(self, aggregator):
        """Test d'agrégation par fenêtre temporelle"""
        now = datetime.utcnow()
        
        # Événements dans différentes fenêtres temporelles
        old_event = StreamEvent(
            event_id="old_event",
            event_type=EventType.USER_ACTION,
            data={"action": "old_action"},
            user_id="user_1"
        )
        old_event.created_at = now - timedelta(minutes=2)
        
        recent_event = StreamEvent(
            event_id="recent_event",
            event_type=EventType.USER_ACTION,
            data={"action": "recent_action"},
            user_id="user_1"
        )
        
        await aggregator.add_event(old_event)
        await aggregator.add_event(recent_event)
        
        # L'agrégation ne devrait inclure que les événements récents
        aggregations = await aggregator.aggregate_events()
        recent_events = aggregations.get("recent_events", [])
        
        # Seul l'événement récent devrait être inclus
        assert len(recent_events) == 1
        assert recent_events[0]["event_id"] == "recent_event"
    
    @pytest.mark.asyncio
    async def test_metric_calculations(self, aggregator):
        """Test de calculs de métriques"""
        # Ajouter des événements avec des valeurs numériques
        values = [10, 20, 30, 40, 50]
        for i, value in enumerate(values):
            event = StreamEvent(
                event_id=f"metric_test_{i}",
                event_type=EventType.USER_ACTION,
                data={"metric_value": value},
                user_id="metric_user"
            )
            await aggregator.add_event(event)
        
        aggregations = await aggregator.aggregate_events()
        metrics = aggregations.get("calculated_metrics", {})
        
        # Vérifier les calculs
        assert metrics.get("total") == sum(values)
        assert metrics.get("average") == sum(values) / len(values)
        assert metrics.get("count") == len(values)


class TestDeadLetterQueue:
    """Tests pour Dead Letter Queue"""
    
    @pytest.fixture
    async def dlq(self):
        """Dead Letter Queue de test"""
        config = {
            "redis_url": REDIS_TEST_URL,
            "max_retries": 3,
            "retry_delay": 1
        }
        
        dlq = DeadLetterQueue(config)
        await dlq.initialize()
        
        yield dlq
        
        await dlq.cleanup()
    
    @pytest.mark.asyncio
    async def test_failed_event_handling(self, dlq):
        """Test de gestion d'événements échoués"""
        failed_event = StreamEvent(
            event_id="failed_event",
            event_type=EventType.USER_ACTION,
            data={"action": "failed_action"},
            user_id="failed_user"
        )
        
        error_info = {
            "error_message": "Processing failed",
            "handler_id": "test_handler",
            "retry_count": 0
        }
        
        await dlq.add_failed_event(failed_event, error_info)
        
        # Vérifier que l'événement est dans la DLQ
        dlq_events = await dlq.get_failed_events()
        assert len(dlq_events) == 1
        assert dlq_events[0]["event_id"] == "failed_event"
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, dlq):
        """Test du mécanisme de retry"""
        # Mock handler qui échoue puis réussit
        retry_count = 0
        
        async def mock_handler(event):
            nonlocal retry_count
            retry_count += 1
            if retry_count < 3:
                raise Exception("Retry test failure")
            return {"success": True}
        
        failed_event = StreamEvent(
            event_id="retry_event",
            event_type=EventType.USER_ACTION,
            data={},
            user_id="retry_user"
        )
        
        # Ajouter à la DLQ
        await dlq.add_failed_event(failed_event, {
            "error_message": "Initial failure",
            "handler_id": "retry_handler",
            "retry_count": 0
        })
        
        # Simuler le retry
        success = await dlq.retry_failed_event("retry_event", mock_handler)
        
        # Devrait réussir après quelques tentatives
        assert success is True
        assert retry_count >= 2
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, dlq):
        """Test de dépassement du nombre maximum de tentatives"""
        async def always_failing_handler(event):
            raise Exception("Always fails")
        
        failed_event = StreamEvent(
            event_id="max_retry_event",
            event_type=EventType.USER_ACTION,
            data={},
            user_id="max_retry_user"
        )
        
        # Ajouter avec le nombre max de tentatives
        await dlq.add_failed_event(failed_event, {
            "error_message": "Max retry test",
            "handler_id": "failing_handler",
            "retry_count": 3  # Déjà au maximum
        })
        
        success = await dlq.retry_failed_event("max_retry_event", always_failing_handler)
        
        # Ne devrait pas réessayer
        assert success is False


class TestEventStreamEngine:
    """Tests pour EventStreamEngine complet"""
    
    @pytest.fixture
    async def engine(self):
        """Moteur de streaming d'événements de test"""
        config = {
            "redis_url": REDIS_TEST_URL,
            "kafka_bootstrap_servers": "localhost:9092",
            "enable_kafka": False,  # Désactiver Kafka pour les tests
            "max_workers": 4,
            "batch_size": 10,
            "processing_timeout": 30
        }
        
        engine = EventStreamEngine(config)
        await engine.initialize()
        
        yield engine
        
        await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine):
        """Test d'initialisation du moteur"""
        assert engine.redis_client is not None
        assert len(engine.handlers) > 0
        assert engine.aggregator is not None
        assert engine.dead_letter_queue is not None
    
    @pytest.mark.asyncio
    async def test_event_processing_pipeline(self, engine):
        """Test du pipeline de traitement d'événements"""
        event = StreamEvent(
            event_id="pipeline_test",
            event_type=EventType.MUSIC_PLAY,
            data={
                "track_id": "track_123",
                "user_id": "user_456"
            },
            user_id="user_456"
        )
        
        # Publier l'événement
        result = await engine.publish_event(event)
        assert result is True
        
        # Attendre le traitement
        await asyncio.sleep(0.5)
        
        # Vérifier que l'événement a été traité
        processed_events = await engine.get_processed_events()
        assert any(e["event_id"] == "pipeline_test" for e in processed_events)
    
    @pytest.mark.asyncio
    async def test_handler_registration(self, engine):
        """Test d'enregistrement de handlers"""
        # Handler personnalisé
        class CustomHandler(EventHandler):
            def __init__(self):
                super().__init__("custom_handler")
            
            async def can_handle(self, event):
                return event.event_type == EventType.USER_ACTION
            
            async def process(self, event):
                return {"success": True, "custom": True}
        
        custom_handler = CustomHandler()
        engine.register_handler(custom_handler)
        
        assert "custom_handler" in engine.handlers
        
        # Test avec un événement personnalisé
        event = StreamEvent(
            event_id="custom_test",
            event_type=EventType.USER_ACTION,
            data={"action": "custom_action"},
            user_id="custom_user"
        )
        
        await engine.publish_event(event)
        await asyncio.sleep(0.2)
        
        # Vérifier que le handler personnalisé a traité l'événement
        assert custom_handler.processed_count == 1
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, engine):
        """Test de traitement par batch"""
        # Publier plusieurs événements rapidement
        events = []
        for i in range(15):  # Plus que la taille du batch
            event = StreamEvent(
                event_id=f"batch_test_{i}",
                event_type=EventType.USER_ACTION,
                data={"action": f"action_{i}"},
                user_id=f"user_{i % 3}"
            )
            events.append(event)
            await engine.publish_event(event)
        
        # Attendre le traitement
        await asyncio.sleep(1.0)
        
        # Vérifier que tous les événements ont été traités
        processed_events = await engine.get_processed_events()
        processed_ids = [e["event_id"] for e in processed_events]
        
        for event in events:
            assert event.event_id in processed_ids
    
    @pytest.mark.asyncio
    async def test_error_handling_pipeline(self, engine):
        """Test de gestion d'erreurs dans le pipeline"""
        # Handler qui échoue
        class FailingHandler(EventHandler):
            def __init__(self):
                super().__init__("failing_handler")
            
            async def can_handle(self, event):
                return event.data.get("should_fail", False)
            
            async def process(self, event):
                raise Exception("Intentional failure")
        
        failing_handler = FailingHandler()
        engine.register_handler(failing_handler)
        
        # Événement qui va échouer
        failing_event = StreamEvent(
            event_id="failing_test",
            event_type=EventType.USER_ACTION,
            data={"should_fail": True},
            user_id="failing_user"
        )
        
        await engine.publish_event(failing_event)
        await asyncio.sleep(0.5)
        
        # Vérifier que l'événement est dans la DLQ
        dlq_events = await engine.dead_letter_queue.get_failed_events()
        assert any(e["event_id"] == "failing_test" for e in dlq_events)
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_high_throughput_processing(self, engine):
        """Test de traitement haute performance"""
        num_events = 100
        start_time = time.time()
        
        # Publier beaucoup d'événements
        tasks = []
        for i in range(num_events):
            event = StreamEvent(
                event_id=f"throughput_test_{i}",
                event_type=EventType.USER_ACTION,
                data={"sequence": i},
                user_id=f"user_{i % 10}"
            )
            tasks.append(engine.publish_event(event))
        
        await asyncio.gather(*tasks)
        publish_time = time.time() - start_time
        
        # Attendre le traitement
        await asyncio.sleep(2.0)
        
        processing_time = time.time() - start_time
        
        # Vérifier les performances
        assert publish_time < 5.0  # Publication rapide
        assert processing_time < 10.0  # Traitement dans un délai raisonnable
        
        # Vérifier que la plupart des événements ont été traités
        processed_events = await engine.get_processed_events()
        processed_count = len([e for e in processed_events if "throughput_test" in e["event_id"]])
        
        assert processed_count >= num_events * 0.9  # Au moins 90% traités


@pytest.mark.integration
class TestEventStreamingIntegration:
    """Tests d'intégration pour le streaming d'événements"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_music_workflow(self):
        """Test de workflow musical complet"""
        config = {
            "redis_url": REDIS_TEST_URL,
            "enable_kafka": False,
            "analytics_enabled": True
        }
        
        engine = EventStreamEngine(config)
        await engine.initialize()
        
        try:
            user_id = TestUtils.generate_test_user_id()
            track_id = TestUtils.generate_test_track_id()
            
            # 1. Événement de lecture
            play_event = StreamEvent(
                event_id="e2e_play",
                event_type=EventType.MUSIC_PLAY,
                data={
                    "track_id": track_id,
                    "playlist_id": "playlist_123",
                    "position": 0
                },
                user_id=user_id
            )
            
            await engine.publish_event(play_event)
            
            # 2. Événement de progression
            progress_event = StreamEvent(
                event_id="e2e_progress",
                event_type=EventType.MUSIC_PROGRESS,
                data={
                    "track_id": track_id,
                    "position": 30,
                    "duration": 180
                },
                user_id=user_id
            )
            
            await engine.publish_event(progress_event)
            
            # 3. Événement de fin
            complete_event = StreamEvent(
                event_id="e2e_complete",
                event_type=EventType.MUSIC_COMPLETE,
                data={
                    "track_id": track_id,
                    "completion_rate": 0.95
                },
                user_id=user_id
            )
            
            await engine.publish_event(complete_event)
            
            # Attendre le traitement
            await asyncio.sleep(1.0)
            
            # Vérifier les résultats
            processed_events = await engine.get_processed_events()
            event_ids = [e["event_id"] for e in processed_events]
            
            assert "e2e_play" in event_ids
            assert "e2e_progress" in event_ids
            assert "e2e_complete" in event_ids
            
        finally:
            await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_real_time_analytics_integration(self):
        """Test d'intégration analytics temps réel"""
        config = {
            "redis_url": REDIS_TEST_URL,
            "enable_kafka": False,
            "analytics_enabled": True,
            "aggregation_interval": 1  # 1 seconde pour les tests
        }
        
        engine = EventStreamEngine(config)
        await engine.initialize()
        
        try:
            # Générer plusieurs événements d'analyse
            users = [TestUtils.generate_test_user_id() for _ in range(5)]
            tracks = [TestUtils.generate_test_track_id() for _ in range(3)]
            
            # Événements de lecture pour différents utilisateurs et tracks
            for i, (user, track) in enumerate(zip(users * 2, tracks * 3)):
                event = StreamEvent(
                    event_id=f"analytics_test_{i}",
                    event_type=EventType.MUSIC_PLAY,
                    data={
                        "track_id": track,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    user_id=user
                )
                await engine.publish_event(event)
            
            # Attendre l'agrégation
            await asyncio.sleep(2.0)
            
            # Vérifier les analytics
            aggregations = await engine.aggregator.aggregate_events()
            
            assert "user_metrics" in aggregations
            assert "track_metrics" in aggregations
            assert len(aggregations["user_metrics"]) == len(users)
            
        finally:
            await engine.shutdown()


# Utilitaires pour les tests de streaming
class StreamTestUtils:
    """Utilitaires pour les tests de streaming"""
    
    @staticmethod
    async def generate_music_events(count=10):
        """Génère des événements musicaux de test"""
        events = []
        for i in range(count):
            event = StreamEvent(
                event_id=f"music_event_{i}",
                event_type=EventType.MUSIC_PLAY,
                data={
                    "track_id": f"track_{i % 5}",
                    "genre": ["rock", "pop", "jazz", "classical"][i % 4],
                    "duration": 180 + (i * 10),
                    "quality": "high"
                },
                user_id=f"user_{i % 3}"
            )
            events.append(event)
        return events
    
    @staticmethod
    async def wait_for_processing(engine, timeout=5):
        """Attend que le traitement soit terminé"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if engine.processing_queue.empty():
                await asyncio.sleep(0.1)
                return True
            await asyncio.sleep(0.1)
        return False


# Export des classes de test
__all__ = [
    "TestStreamEvent",
    "TestEventHandler",
    "TestMusicPlayHandler",
    "TestRecommendationHandler",
    "TestEventAggregator",
    "TestDeadLetterQueue",
    "TestEventStreamEngine",
    "TestEventStreamingIntegration",
    "StreamTestUtils"
]
