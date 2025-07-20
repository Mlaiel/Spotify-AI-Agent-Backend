# 🧪 Tests pour Push Notifications
# =================================
# 
# Tests complets pour le système de notifications push
# avec tests multi-plateformes, ML et personnalisation.
#
# 🎖️ Expert: Push Notification Specialist + ML Test Engineer
#
# 👨‍💻 Développé par: Fahed Mlaiel
# =================================

"""
📱 Push Notifications Tests
===========================

Comprehensive test suite for the Push Notification System:
- Multi-platform delivery tests (iOS, Android, Web)
- ML personalization and optimization tests
- Template engine and i18n tests
- A/B testing framework validation
- Delivery tracking and analytics tests
- Rate limiting and batch processing tests
- Error handling and retry mechanisms
- Performance and scalability tests
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
from app.realtime.push_notifications import (
    AdvancedPushNotificationManager,
    NotificationTemplate,
    PersonalizationEngine,
    TemplateEngine,
    PlatformDeliveryService,
    DeliveryOptimizer,
    PushNotification,
    DeliveryPlatform,
    NotificationType,
    DeliveryStatus,
    ScheduledNotification
)

from . import TestUtils, REDIS_TEST_URL


class TestPushNotification:
    """Tests pour PushNotification"""
    
    def test_notification_creation(self):
        """Test de création de notification"""
        notification = PushNotification(
            notification_id="test_notif_1",
            user_id="user_123",
            title="Test Title",
            body="Test Body",
            platform=DeliveryPlatform.ALL,
            notification_type=NotificationType.MUSIC_RECOMMENDATION
        )
        
        assert notification.notification_id == "test_notif_1"
        assert notification.user_id == "user_123"
        assert notification.title == "Test Title"
        assert notification.body == "Test Body"
        assert notification.platform == DeliveryPlatform.ALL
        assert notification.notification_type == NotificationType.MUSIC_RECOMMENDATION
        assert notification.created_at is not None
        assert notification.delivery_status == DeliveryStatus.PENDING
    
    def test_notification_serialization(self):
        """Test de sérialisation de notification"""
        notification = PushNotification(
            notification_id="test_notif_1",
            user_id="user_123",
            title="Test Title",
            body="Test Body",
            platform=DeliveryPlatform.IOS
        )
        
        serialized = notification.to_dict()
        
        assert "notification_id" in serialized
        assert "user_id" in serialized
        assert "title" in serialized
        assert "body" in serialized
        assert "platform" in serialized
        assert "created_at" in serialized
        assert "delivery_status" in serialized
    
    def test_notification_with_custom_data(self):
        """Test de notification avec données personnalisées"""
        custom_data = {
            "track_id": "track_123",
            "action": "play",
            "deep_link": "spotify://track/123"
        }
        
        notification = PushNotification(
            notification_id="custom_test",
            user_id="user_456",
            title="New Track",
            body="Check out this track!",
            custom_data=custom_data
        )
        
        assert notification.custom_data == custom_data
        assert notification.custom_data["track_id"] == "track_123"


class TestNotificationTemplate:
    """Tests pour NotificationTemplate"""
    
    def test_template_creation(self):
        """Test de création de template"""
        template = NotificationTemplate(
            template_id="music_rec_template",
            name="Music Recommendation",
            title_template="🎵 New music for {user_name}",
            body_template="We found {track_count} songs you might like based on your taste in {genre}!",
            supported_languages=["en", "fr", "de"]
        )
        
        assert template.template_id == "music_rec_template"
        assert template.name == "Music Recommendation"
        assert "{user_name}" in template.title_template
        assert "{track_count}" in template.body_template
        assert "en" in template.supported_languages
    
    def test_template_rendering(self):
        """Test de rendu de template"""
        template = NotificationTemplate(
            template_id="test_template",
            title_template="Hello {user_name}!",
            body_template="You have {count} new messages from {sender}."
        )
        
        variables = {
            "user_name": "John",
            "count": 5,
            "sender": "Alice"
        }
        
        rendered = template.render(variables)
        
        assert rendered["title"] == "Hello John!"
        assert rendered["body"] == "You have 5 new messages from Alice."
    
    def test_template_localization(self):
        """Test de localisation de template"""
        template = NotificationTemplate(
            template_id="localized_template",
            title_template="Welcome {user_name}!",
            body_template="Thanks for joining us!",
            localizations={
                "fr": {
                    "title_template": "Bienvenue {user_name}!",
                    "body_template": "Merci de nous avoir rejoint!"
                },
                "de": {
                    "title_template": "Willkommen {user_name}!",
                    "body_template": "Danke, dass Sie sich uns angeschlossen haben!"
                }
            }
        )
        
        variables = {"user_name": "Marie"}
        
        # Test en français
        rendered_fr = template.render(variables, language="fr")
        assert rendered_fr["title"] == "Bienvenue Marie!"
        assert rendered_fr["body"] == "Merci de nous avoir rejoint!"
        
        # Test en allemand
        rendered_de = template.render(variables, language="de")
        assert rendered_de["title"] == "Willkommen Marie!"
        assert rendered_de["body"] == "Danke, dass Sie sich uns angeschlossen haben!"
    
    def test_template_with_missing_variables(self):
        """Test de template avec variables manquantes"""
        template = NotificationTemplate(
            template_id="missing_vars",
            title_template="Hello {user_name}!",
            body_template="You have {count} items."
        )
        
        # Variables partielles
        variables = {"user_name": "John"}
        
        rendered = template.render(variables)
        
        # Les variables manquantes devraient être remplacées par une chaîne vide ou conservées
        assert "John" in rendered["title"]
        assert rendered["body"] is not None


class TestPersonalizationEngine:
    """Tests pour PersonalizationEngine"""
    
    @pytest.fixture
    async def personalization_engine(self):
        """Engine de personnalisation de test"""
        config = {
            "redis_url": REDIS_TEST_URL,
            "ml_enabled": True,
            "min_user_data": 5
        }
        
        engine = PersonalizationEngine(config)
        await engine.initialize()
        
        # Mock du modèle ML
        engine.ml_model = Mock()
        engine.ml_model.predict = Mock(return_value=0.85)
        
        yield engine
        
        await engine.cleanup()
    
    @pytest.mark.asyncio
    async def test_user_preference_analysis(self, personalization_engine):
        """Test d'analyse des préférences utilisateur"""
        user_id = TestUtils.generate_test_user_id()
        
        # Données utilisateur simulées
        user_data = {
            "listening_history": [
                {"track_id": "track_1", "genre": "rock", "played_at": "2024-01-01"},
                {"track_id": "track_2", "genre": "rock", "played_at": "2024-01-02"},
                {"track_id": "track_3", "genre": "pop", "played_at": "2024-01-03"}
            ],
            "interactions": [
                {"type": "like", "track_id": "track_1"},
                {"type": "share", "track_id": "track_2"}
            ]
        }
        
        await personalization_engine.update_user_data(user_id, user_data)
        preferences = await personalization_engine.analyze_preferences(user_id)
        
        assert "genres" in preferences
        assert "rock" in preferences["genres"]
        assert preferences["engagement_level"] > 0
    
    @pytest.mark.asyncio
    async def test_notification_personalization(self, personalization_engine):
        """Test de personnalisation de notification"""
        user_id = TestUtils.generate_test_user_id()
        
        # Configurer les préférences utilisateur
        preferences = {
            "favorite_genres": ["rock", "indie"],
            "listening_time": "evening",
            "engagement_level": 0.8,
            "language": "en"
        }
        
        await personalization_engine.set_user_preferences(user_id, preferences)
        
        notification = PushNotification(
            notification_id="personalized_test",
            user_id=user_id,
            title="Music Recommendation",
            body="New tracks for you",
            notification_type=NotificationType.MUSIC_RECOMMENDATION
        )
        
        personalized = await personalization_engine.personalize_notification(notification)
        
        assert personalized.user_id == user_id
        assert personalized.personalization_score is not None
        assert personalized.personalization_score > 0
    
    @pytest.mark.asyncio
    async def test_send_time_optimization(self, personalization_engine):
        """Test d'optimisation du moment d'envoi"""
        user_id = TestUtils.generate_test_user_id()
        
        # Historique d'activité
        activity_data = {
            "hourly_activity": {
                "09": 0.2, "10": 0.3, "11": 0.4,
                "18": 0.8, "19": 0.9, "20": 0.7  # Soirée plus active
            },
            "timezone": "Europe/Paris"
        }
        
        await personalization_engine.update_activity_pattern(user_id, activity_data)
        
        optimal_time = await personalization_engine.get_optimal_send_time(user_id)
        
        # Devrait recommander la soirée (18-20h)
        assert optimal_time.hour >= 18
        assert optimal_time.hour <= 20
    
    @pytest.mark.asyncio
    async def test_content_recommendation(self, personalization_engine):
        """Test de recommandation de contenu"""
        user_id = TestUtils.generate_test_user_id()
        
        # Préférences musicales
        music_preferences = {
            "genres": ["electronic", "ambient"],
            "artists": ["artist_1", "artist_2"],
            "mood": "relaxed",
            "tempo": "medium"
        }
        
        await personalization_engine.set_music_preferences(user_id, music_preferences)
        
        recommendations = await personalization_engine.get_content_recommendations(
            user_id, 
            content_type="music_discovery"
        )
        
        assert len(recommendations) > 0
        assert any("electronic" in rec.get("genre", "") for rec in recommendations)


class TestTemplateEngine:
    """Tests pour TemplateEngine"""
    
    @pytest.fixture
    async def template_engine(self):
        """Engine de templates de test"""
        config = {
            "redis_url": REDIS_TEST_URL,
            "default_language": "en",
            "cache_ttl": 3600
        }
        
        engine = TemplateEngine(config)
        await engine.initialize()
        
        yield engine
        
        await engine.cleanup()
    
    @pytest.mark.asyncio
    async def test_template_registration(self, template_engine):
        """Test d'enregistrement de template"""
        template = NotificationTemplate(
            template_id="test_registration",
            name="Test Template",
            title_template="Test {title}",
            body_template="Test {body}",
            supported_languages=["en", "fr"]
        )
        
        await template_engine.register_template(template)
        
        # Vérifier que le template est enregistré
        retrieved = await template_engine.get_template("test_registration")
        assert retrieved is not None
        assert retrieved.template_id == "test_registration"
    
    @pytest.mark.asyncio
    async def test_template_rendering_with_engine(self, template_engine):
        """Test de rendu via l'engine"""
        # Enregistrer un template
        template = NotificationTemplate(
            template_id="render_test",
            title_template="Welcome {user_name}!",
            body_template="You have {new_tracks} new tracks in {genre}.",
            localizations={
                "fr": {
                    "title_template": "Bienvenue {user_name}!",
                    "body_template": "Vous avez {new_tracks} nouvelles pistes en {genre}."
                }
            }
        )
        
        await template_engine.register_template(template)
        
        variables = {
            "user_name": "Alice",
            "new_tracks": 5,
            "genre": "jazz"
        }
        
        # Rendu en anglais
        rendered_en = await template_engine.render_template(
            "render_test", variables, language="en"
        )
        assert rendered_en["title"] == "Welcome Alice!"
        assert "5 new tracks" in rendered_en["body"]
        
        # Rendu en français
        rendered_fr = await template_engine.render_template(
            "render_test", variables, language="fr"
        )
        assert rendered_fr["title"] == "Bienvenue Alice!"
        assert "5 nouvelles pistes" in rendered_fr["body"]
    
    @pytest.mark.asyncio
    async def test_template_caching(self, template_engine):
        """Test de mise en cache des templates"""
        template = NotificationTemplate(
            template_id="cache_test",
            title_template="Cached {title}",
            body_template="Cached {body}"
        )
        
        await template_engine.register_template(template)
        
        # Premier accès (mise en cache)
        start_time = time.time()
        first_access = await template_engine.get_template("cache_test")
        first_time = time.time() - start_time
        
        # Deuxième accès (depuis le cache)
        start_time = time.time()
        second_access = await template_engine.get_template("cache_test")
        second_time = time.time() - start_time
        
        assert first_access.template_id == second_access.template_id
        assert second_time < first_time  # Le cache devrait être plus rapide
    
    @pytest.mark.asyncio
    async def test_a_b_testing_templates(self, template_engine):
        """Test de templates A/B testing"""
        # Template A
        template_a = NotificationTemplate(
            template_id="ab_test_a",
            title_template="Version A: {title}",
            body_template="This is version A of {content}",
            ab_test_group="A",
            ab_test_weight=50
        )
        
        # Template B
        template_b = NotificationTemplate(
            template_id="ab_test_b",
            title_template="Version B: {title}",
            body_template="This is version B of {content}",
            ab_test_group="B",
            ab_test_weight=50
        )
        
        await template_engine.register_template(template_a)
        await template_engine.register_template(template_b)
        
        # Configurer le test A/B
        await template_engine.setup_ab_test("music_promo", ["ab_test_a", "ab_test_b"])
        
        # Obtenir un template pour le test
        selected_template = await template_engine.get_ab_test_template(
            "music_promo", 
            user_id="test_user_123"
        )
        
        assert selected_template is not None
        assert selected_template.template_id in ["ab_test_a", "ab_test_b"]


class TestPlatformDeliveryService:
    """Tests pour PlatformDeliveryService"""
    
    @pytest.fixture
    def delivery_service(self):
        """Service de livraison de test"""
        config = {
            "ios": {
                "apns_key_id": "test_key_id",
                "apns_team_id": "test_team_id",
                "apns_bundle_id": "com.spotify.test"
            },
            "android": {
                "fcm_server_key": "test_fcm_key",
                "fcm_sender_id": "test_sender_id"
            },
            "web": {
                "vapid_public_key": "test_vapid_public",
                "vapid_private_key": "test_vapid_private"
            }
        }
        
        return PlatformDeliveryService(config)
    
    @pytest.mark.asyncio
    async def test_ios_delivery_format(self, delivery_service):
        """Test de formatage pour iOS"""
        notification = PushNotification(
            notification_id="ios_test",
            user_id="user_123",
            title="iOS Test",
            body="Test notification for iOS",
            platform=DeliveryPlatform.IOS,
            custom_data={"track_id": "track_123"}
        )
        
        device_token = "ios_device_token_123"
        
        formatted = delivery_service.format_for_ios(notification, device_token)
        
        assert "aps" in formatted
        assert formatted["aps"]["alert"]["title"] == "iOS Test"
        assert formatted["aps"]["alert"]["body"] == "Test notification for iOS"
        assert formatted["custom_data"]["track_id"] == "track_123"
    
    @pytest.mark.asyncio
    async def test_android_delivery_format(self, delivery_service):
        """Test de formatage pour Android"""
        notification = PushNotification(
            notification_id="android_test",
            user_id="user_456",
            title="Android Test",
            body="Test notification for Android",
            platform=DeliveryPlatform.ANDROID
        )
        
        device_token = "android_device_token_456"
        
        formatted = delivery_service.format_for_android(notification, device_token)
        
        assert "to" in formatted
        assert formatted["to"] == device_token
        assert formatted["notification"]["title"] == "Android Test"
        assert formatted["notification"]["body"] == "Test notification for Android"
    
    @pytest.mark.asyncio
    async def test_web_delivery_format(self, delivery_service):
        """Test de formatage pour Web"""
        notification = PushNotification(
            notification_id="web_test",
            user_id="user_789",
            title="Web Test",
            body="Test notification for Web",
            platform=DeliveryPlatform.WEB
        )
        
        subscription = {
            "endpoint": "https://fcm.googleapis.com/fcm/send/test",
            "keys": {
                "p256dh": "test_p256dh_key",
                "auth": "test_auth_key"
            }
        }
        
        formatted = delivery_service.format_for_web(notification, subscription)
        
        assert "title" in formatted
        assert "body" in formatted
        assert formatted["title"] == "Web Test"
        assert formatted["body"] == "Test notification for Web"
    
    @pytest.mark.asyncio
    async def test_delivery_with_retry(self, delivery_service):
        """Test de livraison avec retry"""
        notification = PushNotification(
            notification_id="retry_test",
            user_id="user_retry",
            title="Retry Test",
            body="Test retry mechanism",
            platform=DeliveryPlatform.IOS
        )
        
        # Mock delivery qui échoue puis réussit
        delivery_attempts = 0
        
        async def mock_send(formatted_notification):
            nonlocal delivery_attempts
            delivery_attempts += 1
            if delivery_attempts < 3:
                raise Exception("Delivery failed")
            return {"success": True, "message_id": "test_message_123"}
        
        delivery_service._send_ios_notification = mock_send
        
        result = await delivery_service.deliver_with_retry(
            notification, 
            "test_device_token",
            max_retries=3
        )
        
        assert result["success"] is True
        assert delivery_attempts == 3


class TestAdvancedPushNotificationManager:
    """Tests pour AdvancedPushNotificationManager complet"""
    
    @pytest.fixture
    async def notification_manager(self):
        """Manager de notifications de test"""
        config = {
            "redis_url": REDIS_TEST_URL,
            "platforms": {
                "ios": {"enabled": True},
                "android": {"enabled": True},
                "web": {"enabled": True}
            },
            "personalization": {"enabled": True},
            "analytics": {"enabled": True},
            "rate_limiting": {
                "max_per_user_per_hour": 10,
                "max_per_user_per_day": 50
            }
        }
        
        manager = AdvancedPushNotificationManager(config)
        await manager.initialize()
        
        yield manager
        
        await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, notification_manager):
        """Test d'initialisation du manager"""
        assert notification_manager.redis_client is not None
        assert notification_manager.personalization_engine is not None
        assert notification_manager.template_engine is not None
        assert notification_manager.delivery_service is not None
    
    @pytest.mark.asyncio
    async def test_send_notification_success(self, notification_manager):
        """Test d'envoi de notification réussi"""
        user_id = TestUtils.generate_test_user_id()
        
        # Enregistrer un device token
        await notification_manager.register_device_token(
            user_id, 
            "test_device_token_123",
            DeliveryPlatform.IOS
        )
        
        notification = PushNotification(
            notification_id="send_test",
            user_id=user_id,
            title="Send Test",
            body="Testing notification sending",
            platform=DeliveryPlatform.IOS
        )
        
        # Mock de la livraison
        with patch.object(notification_manager.delivery_service, 'deliver_with_retry') as mock_deliver:
            mock_deliver.return_value = {"success": True, "message_id": "msg_123"}
            
            result = await notification_manager.send_notification(notification)
            
            assert result["success"] is True
            assert "message_id" in result
    
    @pytest.mark.asyncio
    async def test_send_templated_notification(self, notification_manager):
        """Test d'envoi de notification avec template"""
        user_id = TestUtils.generate_test_user_id()
        
        # Enregistrer device token
        await notification_manager.register_device_token(
            user_id, "test_token", DeliveryPlatform.ANDROID
        )
        
        # Enregistrer un template
        template = NotificationTemplate(
            template_id="templated_test",
            title_template="Hello {user_name}!",
            body_template="You have {count} new {item_type}s."
        )
        
        await notification_manager.template_engine.register_template(template)
        
        # Variables pour le template
        variables = {
            "user_name": "John",
            "count": 3,
            "item_type": "playlist"
        }
        
        # Mock de la livraison
        with patch.object(notification_manager.delivery_service, 'deliver_with_retry') as mock_deliver:
            mock_deliver.return_value = {"success": True}
            
            result = await notification_manager.send_templated_notification(
                user_id=user_id,
                template_id="templated_test",
                variables=variables,
                platform=DeliveryPlatform.ANDROID
            )
            
            assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_scheduled_notification(self, notification_manager):
        """Test de notification programmée"""
        user_id = TestUtils.generate_test_user_id()
        
        # Programmer une notification
        scheduled_time = datetime.utcnow() + timedelta(seconds=2)
        
        scheduled_notification = ScheduledNotification(
            notification_id="scheduled_test",
            user_id=user_id,
            title="Scheduled Test",
            body="This is a scheduled notification",
            scheduled_time=scheduled_time,
            platform=DeliveryPlatform.ALL
        )
        
        await notification_manager.schedule_notification(scheduled_notification)
        
        # Vérifier que la notification est programmée
        scheduled_list = await notification_manager.get_scheduled_notifications(user_id)
        assert len(scheduled_list) == 1
        assert scheduled_list[0]["notification_id"] == "scheduled_test"
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, notification_manager):
        """Test de limitation de taux"""
        user_id = TestUtils.generate_test_user_id()
        
        # Enregistrer device token
        await notification_manager.register_device_token(
            user_id, "rate_limit_token", DeliveryPlatform.IOS
        )
        
        # Envoyer beaucoup de notifications rapidement
        sent_count = 0
        blocked_count = 0
        
        with patch.object(notification_manager.delivery_service, 'deliver_with_retry') as mock_deliver:
            mock_deliver.return_value = {"success": True}
            
            for i in range(15):  # Plus que la limite horaire
                notification = PushNotification(
                    notification_id=f"rate_test_{i}",
                    user_id=user_id,
                    title=f"Rate Test {i}",
                    body="Testing rate limiting",
                    platform=DeliveryPlatform.IOS
                )
                
                result = await notification_manager.send_notification(notification)
                
                if result.get("success"):
                    sent_count += 1
                else:
                    blocked_count += 1
        
        # Certaines notifications devraient être bloquées
        assert blocked_count > 0
        assert sent_count <= 10  # Limite horaire
    
    @pytest.mark.asyncio
    async def test_bulk_notification_sending(self, notification_manager):
        """Test d'envoi en masse"""
        user_ids = [TestUtils.generate_test_user_id() for _ in range(5)]
        
        # Enregistrer des device tokens
        for user_id in user_ids:
            await notification_manager.register_device_token(
                user_id, f"bulk_token_{user_id}", DeliveryPlatform.ANDROID
            )
        
        # Notification à envoyer à tous
        notification_data = {
            "title": "Bulk Notification",
            "body": "This is sent to multiple users",
            "platform": DeliveryPlatform.ANDROID
        }
        
        with patch.object(notification_manager.delivery_service, 'deliver_with_retry') as mock_deliver:
            mock_deliver.return_value = {"success": True}
            
            results = await notification_manager.send_bulk_notification(
                user_ids, notification_data
            )
            
            assert len(results) == len(user_ids)
            assert all(result.get("success") for result in results.values())
    
    @pytest.mark.asyncio
    async def test_notification_analytics(self, notification_manager):
        """Test d'analytics de notifications"""
        user_id = TestUtils.generate_test_user_id()
        notification_id = "analytics_test"
        
        # Enregistrer device token
        await notification_manager.register_device_token(
            user_id, "analytics_token", DeliveryPlatform.WEB
        )
        
        # Envoyer une notification
        notification = PushNotification(
            notification_id=notification_id,
            user_id=user_id,
            title="Analytics Test",
            body="Testing analytics tracking",
            platform=DeliveryPlatform.WEB
        )
        
        with patch.object(notification_manager.delivery_service, 'deliver_with_retry') as mock_deliver:
            mock_deliver.return_value = {"success": True, "message_id": "analytics_msg"}
            
            await notification_manager.send_notification(notification)
        
        # Simuler des événements d'analytics
        await notification_manager.track_notification_event(
            notification_id, "delivered", {"timestamp": datetime.utcnow()}
        )
        
        await notification_manager.track_notification_event(
            notification_id, "opened", {"timestamp": datetime.utcnow()}
        )
        
        # Récupérer les analytics
        analytics = await notification_manager.get_notification_analytics(notification_id)
        
        assert analytics["delivered"] is True
        assert analytics["opened"] is True
        assert "delivery_time" in analytics
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_high_volume_notifications(self, notification_manager):
        """Test de volume élevé de notifications"""
        num_users = 100
        user_ids = [f"perf_user_{i}" for i in range(num_users)]
        
        # Enregistrer des device tokens
        for user_id in user_ids:
            await notification_manager.register_device_token(
                user_id, f"perf_token_{user_id}", DeliveryPlatform.ANDROID
            )
        
        start_time = time.time()
        
        # Mock de livraison rapide
        with patch.object(notification_manager.delivery_service, 'deliver_with_retry') as mock_deliver:
            mock_deliver.return_value = {"success": True}
            
            # Envoyer à tous les utilisateurs
            notification_data = {
                "title": "Performance Test",
                "body": "High volume test notification",
                "platform": DeliveryPlatform.ANDROID
            }
            
            results = await notification_manager.send_bulk_notification(
                user_ids, notification_data
            )
        
        processing_time = time.time() - start_time
        
        # Vérifier les performances
        assert len(results) == num_users
        assert processing_time < 10.0  # Moins de 10 secondes pour 100 notifications
        assert all(result.get("success") for result in results.values())


@pytest.mark.integration
class TestPushNotificationIntegration:
    """Tests d'intégration pour les notifications push"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_notification_flow(self):
        """Test de flux de notification complet"""
        config = {
            "redis_url": REDIS_TEST_URL,
            "platforms": {
                "ios": {"enabled": True},
                "android": {"enabled": True}
            },
            "personalization": {"enabled": True}
        }
        
        manager = AdvancedPushNotificationManager(config)
        await manager.initialize()
        
        try:
            user_id = TestUtils.generate_test_user_id()
            
            # 1. Enregistrer device token
            await manager.register_device_token(
                user_id, "e2e_device_token", DeliveryPlatform.IOS
            )
            
            # 2. Configurer les préférences utilisateur
            await manager.personalization_engine.set_user_preferences(user_id, {
                "language": "fr",
                "music_genres": ["rock", "pop"],
                "notification_frequency": "daily"
            })
            
            # 3. Enregistrer un template localisé
            template = NotificationTemplate(
                template_id="e2e_template",
                title_template="New music for you!",
                body_template="We found {count} {genre} tracks you might like.",
                localizations={
                    "fr": {
                        "title_template": "Nouvelle musique pour vous!",
                        "body_template": "Nous avons trouvé {count} pistes {genre} que vous pourriez aimer."
                    }
                }
            )
            
            await manager.template_engine.register_template(template)
            
            # 4. Envoyer notification personnalisée et localisée
            variables = {"count": 5, "genre": "rock"}
            
            with patch.object(manager.delivery_service, 'deliver_with_retry') as mock_deliver:
                mock_deliver.return_value = {"success": True, "message_id": "e2e_msg"}
                
                result = await manager.send_templated_notification(
                    user_id=user_id,
                    template_id="e2e_template",
                    variables=variables,
                    platform=DeliveryPlatform.IOS
                )
            
            # 5. Vérifier le résultat
            assert result["success"] is True
            
            # 6. Vérifier les analytics
            notifications = await manager.get_user_notification_history(user_id)
            assert len(notifications) == 1
            
        finally:
            await manager.shutdown()


# Utilitaires pour les tests de notifications
class NotificationTestUtils:
    """Utilitaires pour les tests de notifications"""
    
    @staticmethod
    def create_test_notification(user_id=None, platform=DeliveryPlatform.ALL):
        """Crée une notification de test"""
        return PushNotification(
            notification_id=f"test_{uuid.uuid4()}",
            user_id=user_id or TestUtils.generate_test_user_id(),
            title="Test Notification",
            body="This is a test notification",
            platform=platform
        )
    
    @staticmethod
    def create_test_template(template_id=None):
        """Crée un template de test"""
        return NotificationTemplate(
            template_id=template_id or f"template_{uuid.uuid4()}",
            title_template="Test {title}",
            body_template="Test {body} with {variable}",
            supported_languages=["en", "fr"]
        )
    
    @staticmethod
    async def simulate_device_registration(manager, count=10):
        """Simule l'enregistrement de devices"""
        registrations = []
        for i in range(count):
            user_id = f"sim_user_{i}"
            token = f"sim_token_{i}"
            platform = [DeliveryPlatform.IOS, DeliveryPlatform.ANDROID][i % 2]
            
            await manager.register_device_token(user_id, token, platform)
            registrations.append({"user_id": user_id, "token": token, "platform": platform})
        
        return registrations


# Export des classes de test
__all__ = [
    "TestPushNotification",
    "TestNotificationTemplate",
    "TestPersonalizationEngine",
    "TestTemplateEngine",
    "TestPlatformDeliveryService",
    "TestAdvancedPushNotificationManager",
    "TestPushNotificationIntegration",
    "NotificationTestUtils"
]
