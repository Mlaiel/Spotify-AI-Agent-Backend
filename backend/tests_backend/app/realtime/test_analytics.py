# 🧪 Tests pour Analytics Engine
# ==============================
# 
# Tests complets pour le moteur d'analytics temps réel
# avec tests de performance, ML et business intelligence.
#
# 🎖️ Expert: Analytics Engineer + Data Scientist + BI Specialist
#
# 👨‍💻 Développé par: Fahed Mlaiel
# ==============================

"""
📊 Analytics Engine Tests
=========================

Comprehensive test suite for the Real-Time Analytics Engine:
- Stream processing and aggregation tests
- User behavior analysis and ML tests
- Music analytics and recommendation tests
- Performance monitoring and alerting tests
- Time series data management tests
- Dashboard and visualization tests
- GDPR compliance and privacy tests
- Business intelligence and insights tests
"""

import asyncio
import json
import pytest
import time
import uuid
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np
import pandas as pd

# Import du module à tester
from app.realtime.analytics import (
    RealTimeAnalyticsEngine,
    AnalyticsEvent,
    MetricDefinition,
    TimeSeries,
    StreamProcessor,
    UserBehaviorAnalyzer,
    MusicAnalytics,
    PerformanceMonitor,
    EventCategory,
    MetricType,
    AggregationType,
    track_user_event,
    track_music_event
)

from . import TestUtils, REDIS_TEST_URL


class TestAnalyticsEvent:
    """Tests pour AnalyticsEvent"""
    
    def test_event_creation(self):
        """Test de création d'événement analytics"""
        event = AnalyticsEvent(
            event_id="test_analytics_1",
            category=EventCategory.USER_BEHAVIOR,
            event_type="page_view",
            user_id="user_123",
            session_id="session_456",
            properties={"page": "home", "referrer": "direct"},
            metrics={"load_time": 1.2, "scroll_depth": 0.8}
        )
        
        assert event.event_id == "test_analytics_1"
        assert event.category == EventCategory.USER_BEHAVIOR
        assert event.event_type == "page_view"
        assert event.user_id == "user_123"
        assert event.session_id == "session_456"
        assert event.properties["page"] == "home"
        assert event.metrics["load_time"] == 1.2
        assert event.timestamp is not None
        assert event.consent_given is True
    
    def test_event_serialization(self):
        """Test de sérialisation d'événement"""
        event = AnalyticsEvent(
            event_id="serialize_test",
            category=EventCategory.MUSIC_INTERACTION,
            event_type="track_play",
            user_id="user_789",
            properties={"track_id": "track_123"}
        )
        
        serialized = event.to_dict()
        
        assert "event_id" in serialized
        assert "category" in serialized
        assert "event_type" in serialized
        assert "user_id" in serialized
        assert "properties" in serialized
        assert "timestamp" in serialized
        assert serialized["category"] == "music_interaction"
    
    def test_event_privacy_compliance(self):
        """Test de conformité GDPR"""
        # Événement sans consentement
        event_no_consent = AnalyticsEvent(
            event_id="privacy_test_1",
            category=EventCategory.USER_BEHAVIOR,
            event_type="click",
            user_id="user_privacy",
            consent_given=False
        )
        
        serialized = event_no_consent.to_dict()
        
        # L'user_id ne devrait pas être inclus sans consentement
        assert serialized["user_id"] is None
        assert serialized["consent_given"] is False
        
        # Événement anonyme
        event_anonymous = AnalyticsEvent(
            event_id="privacy_test_2",
            category=EventCategory.USER_BEHAVIOR,
            event_type="click",
            user_id="user_anonymous",
            is_anonymous=True
        )
        
        serialized_anon = event_anonymous.to_dict()
        assert serialized_anon["is_anonymous"] is True


class TestMetricDefinition:
    """Tests pour MetricDefinition"""
    
    def test_metric_definition_creation(self):
        """Test de création de définition de métrique"""
        metric_def = MetricDefinition(
            name="page_views",
            type=MetricType.COUNTER,
            aggregation=AggregationType.SUM,
            dimensions=["page", "user_segment"],
            window_size=300,
            threshold_warning=1000.0,
            threshold_critical=5000.0
        )
        
        assert metric_def.name == "page_views"
        assert metric_def.type == MetricType.COUNTER
        assert metric_def.aggregation == AggregationType.SUM
        assert "page" in metric_def.dimensions
        assert metric_def.window_size == 300
        assert metric_def.threshold_warning == 1000.0


class TestTimeSeries:
    """Tests pour TimeSeries"""
    
    def test_time_series_creation(self):
        """Test de création de série temporelle"""
        ts = TimeSeries(
            metric_name="response_time",
            dimension_values={"endpoint": "/api/tracks", "method": "GET"}
        )
        
        assert ts.metric_name == "response_time"
        assert ts.dimension_values["endpoint"] == "/api/tracks"
        assert len(ts.data_points) == 0
    
    def test_add_data_points(self):
        """Test d'ajout de points de données"""
        ts = TimeSeries("test_metric")
        
        now = datetime.utcnow()
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        
        for i, value in enumerate(values):
            timestamp = now + timedelta(seconds=i)
            ts.add_point(timestamp, value)
        
        assert len(ts.data_points) == 5
        assert ts.data_points[-1][1] == 50.0  # Dernière valeur
    
    def test_aggregation_calculations(self):
        """Test de calculs d'agrégation"""
        ts = TimeSeries("aggregation_test")
        
        now = datetime.utcnow()
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        
        for i, value in enumerate(values):
            timestamp = now + timedelta(seconds=i)
            ts.add_point(timestamp, value)
        
        # Test des différentes agrégations
        assert ts.calculate_aggregation(AggregationType.SUM) == 150.0
        assert ts.calculate_aggregation(AggregationType.COUNT) == 5
        assert ts.calculate_aggregation(AggregationType.AVERAGE) == 30.0
        assert ts.calculate_aggregation(AggregationType.MIN) == 10.0
        assert ts.calculate_aggregation(AggregationType.MAX) == 50.0
        assert ts.calculate_aggregation(AggregationType.MEDIAN) == 30.0
    
    def test_time_window_filtering(self):
        """Test de filtrage par fenêtre temporelle"""
        ts = TimeSeries("window_test")
        
        now = datetime.utcnow()
        
        # Ajouter des points sur 10 secondes
        for i in range(10):
            timestamp = now + timedelta(seconds=i)
            ts.add_point(timestamp, float(i))
        
        # Filtrer les 5 dernières secondes
        start_time = now + timedelta(seconds=5)
        recent_values = ts.get_values(start_time)
        
        assert len(recent_values) == 5
        assert recent_values[0][1] == 5.0  # Première valeur dans la fenêtre


class TestStreamProcessor:
    """Tests pour StreamProcessor"""
    
    @pytest.fixture
    def stream_processor(self):
        """Processeur de flux de test"""
        return StreamProcessor(window_size=60)
    
    @pytest.mark.asyncio
    async def test_event_processing(self, stream_processor):
        """Test de traitement d'événement"""
        event = AnalyticsEvent(
            event_id="stream_test_1",
            category=EventCategory.USER_BEHAVIOR,
            event_type="click",
            user_id="user_stream",
            properties={"button": "play"}
        )
        
        # Enregistrer un processeur simple
        async def simple_processor(events):
            return {"processed_count": len(events)}
        
        stream_processor.register_processor(simple_processor)
        
        result = await stream_processor.process_event(event)
        
        assert "processor_0" in result
        assert result["processor_0"]["processed_count"] == 1
    
    @pytest.mark.asyncio
    async def test_window_management(self, stream_processor):
        """Test de gestion des fenêtres"""
        # Ajouter plusieurs événements
        for i in range(5):
            event = AnalyticsEvent(
                event_id=f"window_test_{i}",
                category=EventCategory.USER_BEHAVIOR,
                event_type="action",
                user_id=f"user_{i}"
            )
            await stream_processor.process_event(event)
        
        # Vérifier les statistiques de fenêtre
        window_key = "user_behavior:action"
        stats = stream_processor.get_window_stats(window_key)
        
        assert stats["count"] == 5
        assert stats["unique_users"] == 5
    
    @pytest.mark.asyncio
    async def test_window_cleanup(self, stream_processor):
        """Test de nettoyage des fenêtres"""
        # Créer un processeur avec une petite fenêtre
        small_window_processor = StreamProcessor(window_size=1)  # 1 seconde
        
        # Ajouter un événement ancien
        old_event = AnalyticsEvent(
            event_id="old_event",
            category=EventCategory.USER_BEHAVIOR,
            event_type="old_action",
            user_id="old_user"
        )
        old_event.timestamp = datetime.utcnow() - timedelta(seconds=2)
        
        await small_window_processor.process_event(old_event)
        
        # Ajouter un événement récent
        new_event = AnalyticsEvent(
            event_id="new_event",
            category=EventCategory.USER_BEHAVIOR,
            event_type="old_action",
            user_id="new_user"
        )
        
        await small_window_processor.process_event(new_event)
        
        # La fenêtre ne devrait contenir que l'événement récent
        window_key = "user_behavior:old_action"
        stats = small_window_processor.get_window_stats(window_key)
        
        assert stats["count"] == 1  # Seul l'événement récent


class TestUserBehaviorAnalyzer:
    """Tests pour UserBehaviorAnalyzer"""
    
    @pytest.fixture
    def behavior_analyzer(self):
        """Analyseur de comportement de test"""
        return UserBehaviorAnalyzer()
    
    @pytest.mark.asyncio
    async def test_user_session_tracking(self, behavior_analyzer):
        """Test de suivi de session utilisateur"""
        user_id = TestUtils.generate_test_user_id()
        session_id = f"session_{uuid.uuid4()}"
        
        # Événements de session
        events = [
            AnalyticsEvent(
                event_id="session_start",
                category=EventCategory.USER_BEHAVIOR,
                event_type="page_view",
                user_id=user_id,
                session_id=session_id,
                properties={"page_name": "home"}
            ),
            AnalyticsEvent(
                event_id="session_action",
                category=EventCategory.USER_BEHAVIOR,
                event_type="click",
                user_id=user_id,
                session_id=session_id,
                properties={"element": "play_button"}
            )
        ]
        
        # Traiter les événements
        for event in events:
            await behavior_analyzer.analyze_user_event(event)
        
        # Vérifier la session
        session = behavior_analyzer.user_sessions.get(session_id)
        assert session is not None
        assert session["user_id"] == user_id
        assert len(session["events"]) == 2
        assert "home" in session["pages_visited"]
    
    @pytest.mark.asyncio
    async def test_engagement_score_calculation(self, behavior_analyzer):
        """Test de calcul du score d'engagement"""
        user_id = TestUtils.generate_test_user_id()
        
        # Événement d'engagement élevé
        high_engagement_event = AnalyticsEvent(
            event_id="high_engagement",
            category=EventCategory.MUSIC_INTERACTION,
            event_type="track_complete",
            user_id=user_id,
            properties={"completion_rate": 0.95}
        )
        
        result = await behavior_analyzer.analyze_user_event(high_engagement_event)
        
        assert "engagement_score" in result
        assert result["engagement_score"] > 0.0
    
    @pytest.mark.asyncio
    async def test_user_segmentation(self, behavior_analyzer):
        """Test de segmentation utilisateur"""
        user_id = TestUtils.generate_test_user_id()
        
        # Simuler un utilisateur actif
        for i in range(25):  # Beaucoup d'événements
            event = AnalyticsEvent(
                event_id=f"active_event_{i}",
                category=EventCategory.MUSIC_INTERACTION,
                event_type="track_play",
                user_id=user_id,
                properties={"track_id": f"track_{i}"}
            )
            await behavior_analyzer.analyze_user_event(event)
        
        # Vérifier la segmentation
        result = await behavior_analyzer.analyze_user_event(
            AnalyticsEvent(
                event_id="segment_test",
                category=EventCategory.USER_BEHAVIOR,
                event_type="action",
                user_id=user_id
            )
        )
        
        # Devrait être segmenté comme utilisateur actif
        assert result["user_segment"] in ["power_user", "regular_user"]
    
    @pytest.mark.asyncio
    async def test_anomaly_detection(self, behavior_analyzer):
        """Test de détection d'anomalies"""
        user_id = TestUtils.generate_test_user_id()
        
        # Créer un pattern normal
        for i in range(10):
            normal_event = AnalyticsEvent(
                event_id=f"normal_{i}",
                category=EventCategory.USER_BEHAVIOR,
                event_type="page_view",
                user_id=user_id
            )
            normal_event.timestamp = datetime.utcnow().replace(hour=14)  # Activité normale l'après-midi
            await behavior_analyzer.analyze_user_event(normal_event)
        
        # Événement potentiellement anormal
        anomalous_event = AnalyticsEvent(
            event_id="anomalous",
            category=EventCategory.USER_BEHAVIOR,
            event_type="page_view",
            user_id=user_id
        )
        anomalous_event.timestamp = datetime.utcnow().replace(hour=3)  # Activité à 3h du matin
        
        result = await behavior_analyzer.analyze_user_event(anomalous_event)
        
        # Le système devrait détecter une anomalie potentielle
        assert "is_anomaly" in result


class TestMusicAnalytics:
    """Tests pour MusicAnalytics"""
    
    @pytest.fixture
    def music_analytics(self):
        """Analytics musicaux de test"""
        return MusicAnalytics()
    
    @pytest.mark.asyncio
    async def test_track_play_analysis(self, music_analytics):
        """Test d'analyse de lecture de track"""
        track_id = TestUtils.generate_test_track_id()
        
        play_event = AnalyticsEvent(
            event_id="track_play_test",
            category=EventCategory.MUSIC_INTERACTION,
            event_type="track_play",
            user_id="user_music_test",
            properties={
                "track_id": track_id,
                "genre": "rock",
                "duration": 180
            }
        )
        
        result = await music_analytics.analyze_music_event(play_event)
        
        assert result["track_id"] == track_id
        assert result["total_plays"] == 1
        assert "popularity_score" in result
    
    @pytest.mark.asyncio
    async def test_track_skip_analysis(self, music_analytics):
        """Test d'analyse de skip de track"""
        track_id = TestUtils.generate_test_track_id()
        
        # D'abord une lecture
        play_event = AnalyticsEvent(
            event_id="play_before_skip",
            category=EventCategory.MUSIC_INTERACTION,
            event_type="track_play",
            user_id="skip_test_user",
            properties={"track_id": track_id}
        )
        await music_analytics.analyze_music_event(play_event)
        
        # Puis un skip
        skip_event = AnalyticsEvent(
            event_id="track_skip_test",
            category=EventCategory.MUSIC_INTERACTION,
            event_type="track_skip",
            user_id="skip_test_user",
            properties={
                "track_id": track_id,
                "position": 30  # Skip après 30 secondes
            }
        )
        
        result = await music_analytics.analyze_music_event(skip_event)
        
        assert result["track_id"] == track_id
        assert "skip_rate" in result
        assert "avg_skip_position" in result
        assert result["avg_skip_position"] == 30.0
    
    @pytest.mark.asyncio
    async def test_popularity_calculation(self, music_analytics):
        """Test de calcul de popularité"""
        track_id = TestUtils.generate_test_track_id()
        
        # Simuler plusieurs lectures et complétions
        for i in range(10):
            # Lecture
            play_event = AnalyticsEvent(
                event_id=f"pop_play_{i}",
                category=EventCategory.MUSIC_INTERACTION,
                event_type="track_play",
                user_id=f"pop_user_{i}",
                properties={"track_id": track_id}
            )
            await music_analytics.analyze_music_event(play_event)
            
            # La plupart complètent
            if i < 8:
                complete_event = AnalyticsEvent(
                    event_id=f"pop_complete_{i}",
                    category=EventCategory.MUSIC_INTERACTION,
                    event_type="track_complete",
                    user_id=f"pop_user_{i}",
                    properties={
                        "track_id": track_id,
                        "duration_listened": 180
                    }
                )
                await music_analytics.analyze_music_event(complete_event)
        
        # Calculer la popularité
        popularity = music_analytics._calculate_track_popularity(track_id)
        
        # Devrait être élevé (bon taux de completion)
        assert popularity > 0.5
    
    @pytest.mark.asyncio
    async def test_trending_detection(self, music_analytics):
        """Test de détection de tendances"""
        track_id = TestUtils.generate_test_track_id()
        
        # Simuler beaucoup de lectures récentes
        for i in range(150):
            play_event = AnalyticsEvent(
                event_id=f"trending_play_{i}",
                category=EventCategory.MUSIC_INTERACTION,
                event_type="track_play",
                user_id=f"trending_user_{i}",
                properties={"track_id": track_id}
            )
            # Toutes récentes
            play_event.timestamp = datetime.utcnow() - timedelta(minutes=i // 10)
            await music_analytics.analyze_music_event(play_event)
        
        # Vérifier si c'est trending
        is_trending = music_analytics._is_track_trending(track_id)
        assert is_trending is True


class TestPerformanceMonitor:
    """Tests pour PerformanceMonitor"""
    
    @pytest.fixture
    def performance_monitor(self):
        """Moniteur de performance de test"""
        return PerformanceMonitor()
    
    @pytest.mark.asyncio
    async def test_metric_recording(self, performance_monitor):
        """Test d'enregistrement de métrique"""
        result = await performance_monitor.record_metric(
            "response_time",
            1.5,
            {"endpoint": "/api/tracks", "method": "GET"}
        )
        
        assert result["metric_name"] == "response_time"
        assert result["value"] == 1.5
        assert result["dimensions"]["endpoint"] == "/api/tracks"
    
    @pytest.mark.asyncio
    async def test_alert_generation(self, performance_monitor):
        """Test de génération d'alertes"""
        # Enregistrer une métrique qui dépasse le seuil critique
        result = await performance_monitor.record_metric(
            "response_time",
            5.0,  # Au-dessus du seuil critique de 3.0
            {"endpoint": "/api/slow"}
        )
        
        assert result["alert"] is not None
        assert result["alert"]["level"] == "critical"
        assert result["alert"]["metric_name"] == "response_time"
    
    @pytest.mark.asyncio
    async def test_metric_statistics(self, performance_monitor):
        """Test de statistiques de métriques"""
        # Enregistrer plusieurs valeurs
        values = [1.0, 1.5, 2.0, 2.5, 3.0]
        for value in values:
            await performance_monitor.record_metric(
                "test_metric",
                value,
                {"service": "test"}
            )
        
        # Récupérer les statistiques
        stats = performance_monitor.get_metric_stats(
            "test_metric",
            {"service": "test"}
        )
        
        assert stats["count"] == 5
        assert stats["min"] == 1.0
        assert stats["max"] == 3.0
        assert stats["avg"] == 2.0
        assert stats["median"] == 2.0
    
    def test_active_alerts_filtering(self, performance_monitor):
        """Test de filtrage des alertes actives"""
        # Ajouter des alertes de différents niveaux
        warning_alert = {
            "id": "warning_1",
            "level": "warning",
            "metric_name": "cpu_usage",
            "value": 0.85
        }
        
        critical_alert = {
            "id": "critical_1",
            "level": "critical",
            "metric_name": "memory_usage",
            "value": 0.98
        }
        
        performance_monitor.alerts.extend([warning_alert, critical_alert])
        
        # Filtrer par niveau
        critical_alerts = performance_monitor.get_active_alerts("critical")
        warning_alerts = performance_monitor.get_active_alerts("warning")
        all_alerts = performance_monitor.get_active_alerts()
        
        assert len(critical_alerts) == 1
        assert len(warning_alerts) == 1
        assert len(all_alerts) == 2


class TestRealTimeAnalyticsEngine:
    """Tests pour RealTimeAnalyticsEngine complet"""
    
    @pytest.fixture
    async def analytics_engine(self):
        """Moteur d'analytics de test"""
        engine = RealTimeAnalyticsEngine(redis_url=REDIS_TEST_URL)
        await engine.initialize()
        
        yield engine
        
        await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, analytics_engine):
        """Test d'initialisation du moteur"""
        assert analytics_engine.redis_client is not None
        assert analytics_engine.stream_processor is not None
        assert analytics_engine.user_analyzer is not None
        assert analytics_engine.music_analytics is not None
        assert analytics_engine.performance_monitor is not None
        assert len(analytics_engine.metric_definitions) > 0
    
    @pytest.mark.asyncio
    async def test_event_tracking(self, analytics_engine):
        """Test de tracking d'événement"""
        event = AnalyticsEvent(
            event_id="engine_test_1",
            category=EventCategory.USER_BEHAVIOR,
            event_type="page_view",
            user_id="engine_test_user",
            properties={"page": "discover"},
            metrics={"load_time": 0.8}
        )
        
        result = await analytics_engine.track_event(event)
        
        assert result["status"] == "queued"
        assert result["event_id"] == "engine_test_1"
    
    @pytest.mark.asyncio
    async def test_metric_retrieval(self, analytics_engine):
        """Test de récupération de métrique"""
        # Simuler quelques événements pour créer des métriques
        for i in range(5):
            event = AnalyticsEvent(
                event_id=f"metric_test_{i}",
                category=EventCategory.MUSIC_INTERACTION,
                event_type="track_play",
                user_id=f"metric_user_{i}",
                metrics={"track_plays": 1}
            )
            await analytics_engine.track_event(event)
        
        # Attendre le traitement
        await asyncio.sleep(0.5)
        
        # Récupérer la métrique
        value = await analytics_engine.get_metric_value(
            "track_plays",
            aggregation=AggregationType.SUM
        )
        
        # Devrait avoir une valeur
        assert value is not None
    
    @pytest.mark.asyncio
    async def test_dashboard_data_generation(self, analytics_engine):
        """Test de génération de données de dashboard"""
        # Générer quelques événements
        events = [
            AnalyticsEvent(
                event_id="dash_user_1",
                category=EventCategory.USER_BEHAVIOR,
                event_type="session_start",
                user_id="dash_user_1",
                metrics={"session_duration": 600}
            ),
            AnalyticsEvent(
                event_id="dash_music_1",
                category=EventCategory.MUSIC_INTERACTION,
                event_type="track_play",
                user_id="dash_user_1",
                properties={"track_id": "track_dashboard_1"},
                metrics={"track_plays": 1}
            )
        ]
        
        for event in events:
            await analytics_engine.track_event(event)
        
        # Attendre le traitement
        await asyncio.sleep(0.5)
        
        # Récupérer les données de dashboard
        overview_data = await analytics_engine.get_dashboard_data("overview")
        
        assert overview_data is not None
        assert isinstance(overview_data, dict)
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, analytics_engine):
        """Test de traitement par batch"""
        # Envoyer beaucoup d'événements rapidement
        events = []
        for i in range(20):
            event = AnalyticsEvent(
                event_id=f"batch_test_{i}",
                category=EventCategory.USER_BEHAVIOR,
                event_type="action",
                user_id=f"batch_user_{i % 5}",  # 5 utilisateurs différents
                metrics={"actions": 1}
            )
            events.append(event)
            await analytics_engine.track_event(event)
        
        # Attendre le traitement des batches
        await asyncio.sleep(2.0)
        
        # Vérifier que les événements ont été traités
        assert analytics_engine.stream_processor.processed_events >= 20
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_high_throughput_analytics(self, analytics_engine):
        """Test de throughput élevé"""
        num_events = 100
        start_time = time.time()
        
        # Générer beaucoup d'événements
        tasks = []
        for i in range(num_events):
            event = AnalyticsEvent(
                event_id=f"throughput_test_{i}",
                category=EventCategory.MUSIC_INTERACTION,
                event_type="track_play",
                user_id=f"throughput_user_{i % 10}",
                properties={"track_id": f"track_{i % 20}"},
                metrics={"track_plays": 1}
            )
            tasks.append(analytics_engine.track_event(event))
        
        await asyncio.gather(*tasks)
        ingestion_time = time.time() - start_time
        
        # Attendre le traitement
        await asyncio.sleep(3.0)
        total_time = time.time() - start_time
        
        # Vérifier les performances
        assert ingestion_time < 5.0  # Ingestion rapide
        assert total_time < 10.0  # Traitement dans un délai raisonnable
        
        # Vérifier le throughput
        events_per_second = num_events / ingestion_time
        assert events_per_second > 20  # Au moins 20 événements/seconde


class TestAnalyticsUtilityFunctions:
    """Tests pour les fonctions utilitaires d'analytics"""
    
    @pytest.mark.asyncio
    async def test_track_user_event_function(self):
        """Test de la fonction track_user_event"""
        # Mock du moteur global
        with patch('app.realtime.analytics.analytics_engine') as mock_engine:
            mock_engine.track_event = AsyncMock(return_value={"event_id": "util_test_1"})
            
            event_id = await track_user_event(
                user_id="util_user_1",
                event_type="button_click",
                properties={"button": "play"},
                session_id="util_session_1"
            )
            
            assert event_id == "util_test_1"
            mock_engine.track_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_track_music_event_function(self):
        """Test de la fonction track_music_event"""
        with patch('app.realtime.analytics.analytics_engine') as mock_engine:
            mock_engine.track_event = AsyncMock(return_value={"event_id": "music_util_1"})
            
            event_id = await track_music_event(
                user_id="music_util_user",
                event_type="track_play",
                track_id="util_track_123",
                properties={"playlist_id": "util_playlist_456"}
            )
            
            assert event_id == "music_util_1"
            mock_engine.track_event.assert_called_once()


@pytest.mark.integration
class TestAnalyticsIntegration:
    """Tests d'intégration pour l'analytics"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_analytics_flow(self):
        """Test de flux analytics complet"""
        engine = RealTimeAnalyticsEngine(redis_url=REDIS_TEST_URL)
        await engine.initialize()
        
        try:
            user_id = TestUtils.generate_test_user_id()
            track_id = TestUtils.generate_test_track_id()
            
            # 1. Événement de session
            session_event = AnalyticsEvent(
                event_id="e2e_session",
                category=EventCategory.USER_BEHAVIOR,
                event_type="session_start",
                user_id=user_id,
                session_id="e2e_session_123",
                metrics={"session_duration": 0}
            )
            
            await engine.track_event(session_event)
            
            # 2. Événements musicaux
            music_events = [
                AnalyticsEvent(
                    event_id="e2e_play",
                    category=EventCategory.MUSIC_INTERACTION,
                    event_type="track_play",
                    user_id=user_id,
                    properties={"track_id": track_id, "genre": "rock"},
                    metrics={"track_plays": 1}
                ),
                AnalyticsEvent(
                    event_id="e2e_complete",
                    category=EventCategory.MUSIC_INTERACTION,
                    event_type="track_complete",
                    user_id=user_id,
                    properties={"track_id": track_id, "completion_rate": 0.95},
                    metrics={"track_completes": 1}
                )
            ]
            
            for event in music_events:
                await engine.track_event(event)
            
            # 3. Événement de performance
            perf_event = AnalyticsEvent(
                event_id="e2e_perf",
                category=EventCategory.SYSTEM_PERFORMANCE,
                event_type="api_response",
                user_id=user_id,
                properties={"endpoint": "/api/tracks"},
                metrics={"response_time": 0.5}
            )
            
            await engine.track_event(perf_event)
            
            # Attendre le traitement
            await asyncio.sleep(2.0)
            
            # 4. Vérifier les résultats
            
            # Dashboard overview
            overview = await engine.get_dashboard_data("overview")
            assert overview is not None
            
            # Dashboard musical
            music_dashboard = await engine.get_dashboard_data("music")
            assert music_dashboard is not None
            
            # Dashboard performance
            perf_dashboard = await engine.get_dashboard_data("performance")
            assert perf_dashboard is not None
            
        finally:
            await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_real_time_user_journey_tracking(self):
        """Test de suivi de parcours utilisateur temps réel"""
        engine = RealTimeAnalyticsEngine(redis_url=REDIS_TEST_URL)
        await engine.initialize()
        
        try:
            user_id = TestUtils.generate_test_user_id()
            session_id = f"journey_{uuid.uuid4()}"
            
            # Parcours utilisateur simulé
            journey_events = [
                ("page_view", {"page": "home"}),
                ("search", {"query": "rock music"}),
                ("track_play", {"track_id": "rock_track_1"}),
                ("like", {"track_id": "rock_track_1"}),
                ("playlist_create", {"playlist_name": "My Rock"}),
                ("social_share", {"content": "playlist", "platform": "twitter"})
            ]
            
            for i, (event_type, props) in enumerate(journey_events):
                category = (EventCategory.MUSIC_INTERACTION 
                           if event_type in ["track_play", "like", "playlist_create"] 
                           else EventCategory.USER_BEHAVIOR)
                
                event = AnalyticsEvent(
                    event_id=f"journey_{i}",
                    category=category,
                    event_type=event_type,
                    user_id=user_id,
                    session_id=session_id,
                    properties=props,
                    metrics={"engagement_points": i + 1}
                )
                
                await engine.track_event(event)
                
                # Petit délai pour simuler le temps réel
                await asyncio.sleep(0.1)
            
            # Attendre le traitement
            await asyncio.sleep(1.0)
            
            # Analyser le parcours
            session_data = engine.user_analyzer.user_sessions.get(session_id)
            if session_data:
                assert len(session_data["events"]) == len(journey_events)
                assert "home" in session_data["pages_visited"]
                assert len(session_data["actions_performed"]) > 0
            
        finally:
            await engine.shutdown()


# Utilitaires pour les tests d'analytics
class AnalyticsTestUtils:
    """Utilitaires pour les tests d'analytics"""
    
    @staticmethod
    def generate_user_behavior_events(user_id, count=10):
        """Génère des événements de comportement utilisateur"""
        events = []
        for i in range(count):
            event = AnalyticsEvent(
                event_id=f"behavior_{user_id}_{i}",
                category=EventCategory.USER_BEHAVIOR,
                event_type=["page_view", "click", "scroll", "search"][i % 4],
                user_id=user_id,
                properties={
                    "page": ["home", "discover", "library", "profile"][i % 4],
                    "session_id": f"session_{user_id}"
                },
                metrics={"engagement": 0.1 * (i + 1)}
            )
            events.append(event)
        return events
    
    @staticmethod
    def generate_music_events(user_id, track_ids, count=5):
        """Génère des événements musicaux"""
        events = []
        for i in range(count):
            track_id = track_ids[i % len(track_ids)]
            event = AnalyticsEvent(
                event_id=f"music_{user_id}_{i}",
                category=EventCategory.MUSIC_INTERACTION,
                event_type=["track_play", "track_skip", "track_complete", "like"][i % 4],
                user_id=user_id,
                properties={
                    "track_id": track_id,
                    "genre": ["rock", "pop", "jazz"][i % 3]
                },
                metrics={"track_plays": 1 if i % 4 == 0 else 0}
            )
            events.append(event)
        return events
    
    @staticmethod
    async def simulate_user_activity(engine, user_count=5, events_per_user=10):
        """Simule l'activité de plusieurs utilisateurs"""
        users = [TestUtils.generate_test_user_id() for _ in range(user_count)]
        tracks = [TestUtils.generate_test_track_id() for _ in range(10)]
        
        all_events = []
        for user_id in users:
            # Événements de comportement
            behavior_events = AnalyticsTestUtils.generate_user_behavior_events(
                user_id, events_per_user // 2
            )
            
            # Événements musicaux
            music_events = AnalyticsTestUtils.generate_music_events(
                user_id, tracks, events_per_user // 2
            )
            
            all_events.extend(behavior_events + music_events)
        
        # Envoyer tous les événements
        for event in all_events:
            await engine.track_event(event)
        
        return all_events


# Export des classes de test
__all__ = [
    "TestAnalyticsEvent",
    "TestMetricDefinition", 
    "TestTimeSeries",
    "TestStreamProcessor",
    "TestUserBehaviorAnalyzer",
    "TestMusicAnalytics",
    "TestPerformanceMonitor",
    "TestRealTimeAnalyticsEngine",
    "TestAnalyticsUtilityFunctions",
    "TestAnalyticsIntegration",
    "AnalyticsTestUtils"
]
