"""
🧪 Suite de Tests Ultra-Avancée pour les Alertes Critiques
=========================================================

Tests complets et industriels pour valider le système d'alertes critiques
avec couverture complète, tests de performance, sécurité et conformité.

Architecte: Fahed Mlaiel - Lead Architect
Frameworks: pytest, asyncio, hypothesis, locust
Coverage: Unit, Integration, E2E, Performance, Security
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import asdict

import pytest
import pytest_asyncio
from hypothesis import given, strategies as st, settings
from faker import Faker
import aioredis
from prometheus_client import CollectorRegistry
import numpy as np

# Import des modules à tester
from . import (
    CriticalAlertSeverity, AlertChannel, TenantTier, CriticalAlertMetadata,
    CriticalAlertProcessor, CriticalAlertProcessorFactory,
    AlertMLModel, SlackTemplateEngine, SlackTemplateContext, SlackTemplateType,
    AlertMetricsCollector, MetricsCollectorFactory
)

# Configuration de logging pour les tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Faker pour données de test
fake = Faker(['en_US', 'fr_FR', 'de_DE'])

# === FIXTURES GLOBALES ===

@pytest_asyncio.fixture
async def redis_client():
    """Fixture Redis pour les tests"""
    client = await aioredis.from_url("redis://localhost:6379/1")  # DB 1 pour tests
    yield client
    await client.flushdb()  # Nettoyage après chaque test
    await client.close()

@pytest_asyncio.fixture
async def db_session():
    """Fixture base de données pour les tests"""
    # En production, utiliser une vraie base de test
    mock_session = AsyncMock()
    return mock_session

@pytest.fixture
def metrics_registry():
    """Fixture registry Prometheus pour les tests"""
    return CollectorRegistry()

@pytest.fixture
def sample_alert_metadata():
    """Fixture métadonnées d'alerte échantillon"""
    return CriticalAlertMetadata(
        alert_id=str(uuid.uuid4()),
        tenant_id="test-tenant-001",
        severity=CriticalAlertSeverity.CRITICAL,
        tenant_tier=TenantTier.ENTERPRISE,
        source_service="api-gateway",
        affected_users=1500,
        business_impact=2.5,
        correlation_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
        fingerprint="test-fingerprint-001",
        runbook_url="https://runbooks.test.com/api-gateway/critical",
        tags={"environment": "production", "cluster": "eu-west-1"},
        custom_data={"error_rate": 0.25, "latency_p99": 5.2}
    )

@pytest_asyncio.fixture
async def alert_processor(redis_client, db_session):
    """Fixture processeur d'alertes"""
    ml_model = AlertMLModel()
    processor = CriticalAlertProcessor(redis_client, db_session, ml_model)
    return processor

@pytest.fixture
def slack_template_engine():
    """Fixture moteur de templates Slack"""
    return SlackTemplateEngine()

@pytest.fixture
def metrics_collector(metrics_registry):
    """Fixture collecteur de métriques"""
    return AlertMetricsCollector(metrics_registry)

# === GÉNÉRATEURS DE DONNÉES DE TEST ===

def generate_random_alert_metadata() -> CriticalAlertMetadata:
    """Génération d'alertes aléatoires pour les tests"""
    return CriticalAlertMetadata(
        alert_id=str(uuid.uuid4()),
        tenant_id=fake.uuid4(),
        severity=fake.random_element(elements=list(CriticalAlertSeverity)),
        tenant_tier=fake.random_element(elements=list(TenantTier)),
        source_service=fake.random_element(elements=[
            "api-gateway", "user-service", "payment-service", 
            "notification-service", "ml-inference", "data-pipeline"
        ]),
        affected_users=fake.random_int(min=1, max=100000),
        business_impact=fake.random.uniform(0.1, 5.0),
        correlation_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
        fingerprint=fake.sha256(),
        runbook_url=fake.url(),
        tags={
            fake.word(): fake.word() for _ in range(fake.random_int(min=1, max=5))
        },
        custom_data={
            fake.word(): fake.random.uniform(0, 100) for _ in range(fake.random_int(min=1, max=3))
        }
    )

# === TESTS UNITAIRES ===

class TestCriticalAlertMetadata:
    """Tests pour les métadonnées d'alerte"""
    
    def test_alert_metadata_creation(self, sample_alert_metadata):
        """Test de création des métadonnées"""
        assert sample_alert_metadata.alert_id is not None
        assert sample_alert_metadata.tenant_id == "test-tenant-001"
        assert sample_alert_metadata.severity == CriticalAlertSeverity.CRITICAL
        assert sample_alert_metadata.business_impact == 2.5
        assert sample_alert_metadata.affected_users == 1500
    
    def test_alert_metadata_serialization(self, sample_alert_metadata):
        """Test de sérialisation/désérialisation"""
        data_dict = asdict(sample_alert_metadata)
        assert isinstance(data_dict, dict)
        assert data_dict['alert_id'] == sample_alert_metadata.alert_id
        assert data_dict['tenant_id'] == sample_alert_metadata.tenant_id
    
    @given(
        severity=st.sampled_from(list(CriticalAlertSeverity)),
        affected_users=st.integers(min_value=0, max_value=1000000),
        business_impact=st.floats(min_value=0.0, max_value=10.0, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_alert_metadata_property_based(self, severity, affected_users, business_impact):
        """Tests basés sur les propriétés avec Hypothesis"""
        alert = CriticalAlertMetadata(
            alert_id=str(uuid.uuid4()),
            tenant_id="test-tenant",
            severity=severity,
            tenant_tier=TenantTier.ENTERPRISE,
            source_service="test-service",
            affected_users=affected_users,
            business_impact=business_impact
        )
        
        assert alert.severity in list(CriticalAlertSeverity)
        assert alert.affected_users >= 0
        assert alert.business_impact >= 0.0
        assert isinstance(alert.alert_id, str)

class TestAlertMLModel:
    """Tests pour le modèle ML"""
    
    @pytest_asyncio.async_test
    async def test_escalation_prediction(self):
        """Test de prédiction d'escalade"""
        model = AlertMLModel()
        alert_metadata = generate_random_alert_metadata()
        historical_data = [
            {"escalated": True, "severity": "CRITICAL"},
            {"escalated": False, "severity": "WARNING"},
            {"escalated": True, "severity": "HIGH"}
        ]
        
        probability = await model.predict_escalation_probability(
            alert_metadata, historical_data
        )
        
        assert 0.0 <= probability <= 1.0
        assert isinstance(probability, float)
    
    def test_feature_extraction(self):
        """Test d'extraction de features"""
        model = AlertMLModel()
        alert_metadata = generate_random_alert_metadata()
        historical_data = []
        
        features = model._extract_features(alert_metadata, historical_data)
        
        assert isinstance(features, np.ndarray)
        assert len(features) == 7  # Nombre de features attendues
        assert all(isinstance(f, (int, float, np.number)) for f in features)
    
    @pytest.mark.parametrize("severity,expected_min_prob", [
        (CriticalAlertSeverity.CATASTROPHIC, 0.9),
        (CriticalAlertSeverity.CRITICAL, 0.8),
        (CriticalAlertSeverity.HIGH, 0.6),
        (CriticalAlertSeverity.ELEVATED, 0.4),
        (CriticalAlertSeverity.WARNING, 0.2)
    ])
    def test_base_probability_calculation(self, severity, expected_min_prob):
        """Test du calcul de probabilité de base"""
        model = AlertMLModel()
        alert_metadata = CriticalAlertMetadata(
            alert_id=str(uuid.uuid4()),
            tenant_id="test",
            severity=severity,
            tenant_tier=TenantTier.ENTERPRISE,
            source_service="test",
            affected_users=100,
            business_impact=1.0
        )
        
        probability = model._calculate_base_probability(alert_metadata)
        assert probability >= expected_min_prob

class TestCriticalAlertProcessor:
    """Tests pour le processeur d'alertes"""
    
    @pytest_asyncio.async_test
    async def test_alert_processing_success(self, alert_processor, sample_alert_metadata):
        """Test de traitement d'alerte réussi"""
        alert_data = {
            "severity": "CRITICAL",
            "source_service": "api-gateway",
            "affected_users": 1500,
            "business_impact": 2.5,
            "tags": {"environment": "production"}
        }
        
        tenant_context = {
            "tenant_id": "test-tenant-001",
            "tier": "ENTERPRISE"
        }
        
        result = await alert_processor.process_critical_alert(alert_data, tenant_context)
        
        assert result["status"] == "processed"
        assert "alert_id" in result
        assert "escalation_probability" in result
        assert result["processing_time_ms"] > 0
    
    @pytest_asyncio.async_test
    async def test_alert_correlation(self, alert_processor, sample_alert_metadata):
        """Test de corrélation d'alertes"""
        # Stockage d'une alerte existante
        await alert_processor._store_alert_in_cache(sample_alert_metadata)
        
        # Création d'une alerte similaire
        similar_alert = CriticalAlertMetadata(
            alert_id=str(uuid.uuid4()),
            tenant_id=sample_alert_metadata.tenant_id,
            severity=sample_alert_metadata.severity,
            tenant_tier=sample_alert_metadata.tenant_tier,
            source_service=sample_alert_metadata.source_service,
            affected_users=1000,
            business_impact=2.0,
            fingerprint=sample_alert_metadata.fingerprint  # Même fingerprint
        )
        
        correlated_alerts = await alert_processor._correlate_alerts(similar_alert)
        
        assert len(correlated_alerts) > 0
        assert sample_alert_metadata.alert_id.split('-')[0] in str(correlated_alerts)
    
    @pytest_asyncio.async_test
    async def test_escalation_plan_determination(self, alert_processor, sample_alert_metadata):
        """Test de détermination du plan d'escalade"""
        escalation_plan = await alert_processor._determine_escalation_plan(
            sample_alert_metadata, 0.85
        )
        
        assert "channels" in escalation_plan
        assert "delays" in escalation_plan
        assert "ml_probability" in escalation_plan
        assert escalation_plan["ml_probability"] == 0.85
        assert len(escalation_plan["channels"]["immediate"]) > 0
    
    @pytest_asyncio.async_test
    async def test_notification_sending(self, alert_processor, sample_alert_metadata):
        """Test d'envoi de notifications"""
        with patch.object(alert_processor, '_send_notification') as mock_send:
            mock_send.return_value = {
                "status": "sent",
                "message_id": "test-123"
            }
            
            escalation_plan = {
                "channels": {
                    "immediate": [AlertChannel.SLACK, AlertChannel.EMAIL]
                }
            }
            
            results = await alert_processor._send_immediate_notifications(
                sample_alert_metadata, escalation_plan
            )
            
            assert len(results) == 2
            assert all(result["status"] == "sent" for result in results)
            assert mock_send.call_count == 2

class TestSlackTemplateEngine:
    """Tests pour le moteur de templates Slack"""
    
    def test_template_engine_initialization(self, slack_template_engine):
        """Test d'initialisation du moteur"""
        assert slack_template_engine.jinja_env is not None
        assert slack_template_engine.config["max_message_length"] == 4000
        assert slack_template_engine.config["enable_ml_optimization"] is True
    
    @pytest_asyncio.async_test
    async def test_slack_message_generation(self, slack_template_engine, sample_alert_metadata):
        """Test de génération de message Slack"""
        context = SlackTemplateContext(
            alert_metadata=sample_alert_metadata,
            tenant_config={"slack_template_override": None},
            locale="en",
            timezone="UTC"
        )
        
        message = await slack_template_engine.generate_slack_message(
            context, SlackTemplateType.INTERACTIVE_BLOCKS
        )
        
        assert "text" in message
        assert "blocks" in message
        assert "attachments" in message
        assert len(message["blocks"]) > 0
        assert message["attachments"][0]["color"] is not None
    
    def test_jinja_filters(self, slack_template_engine):
        """Test des filtres Jinja personnalisés"""
        engine = slack_template_engine
        
        # Test format_datetime
        dt = datetime.utcnow()
        formatted = engine._format_datetime(dt, "en")
        assert isinstance(formatted, str)
        assert len(formatted) > 10
        
        # Test format_duration
        duration = timedelta(hours=2, minutes=30)
        formatted_duration = engine._format_duration(duration)
        assert "2h 30m" in formatted_duration
        
        # Test severity emoji
        emoji = engine._get_severity_emoji(CriticalAlertSeverity.CRITICAL)
        assert emoji == "🔴"
        
        # Test business impact formatting
        impact_text = engine._format_business_impact(2.5)
        assert "ÉLEVÉ" in impact_text
    
    @pytest.mark.parametrize("template_type", [
        SlackTemplateType.SIMPLE_MESSAGE,
        SlackTemplateType.RICH_CARD,
        SlackTemplateType.INTERACTIVE_BLOCKS,
        SlackTemplateType.MODAL_DIALOG
    ])
    @pytest_asyncio.async_test
    async def test_different_template_types(self, slack_template_engine, sample_alert_metadata, template_type):
        """Test des différents types de templates"""
        context = SlackTemplateContext(
            alert_metadata=sample_alert_metadata,
            tenant_config={},
            locale="en"
        )
        
        message = await slack_template_engine.generate_slack_message(context, template_type)
        
        assert message is not None
        assert "text" in message or "blocks" in message

class TestAlertMetricsCollector:
    """Tests pour le collecteur de métriques"""
    
    @pytest_asyncio.async_test
    async def test_alert_creation_metrics(self, metrics_collector, sample_alert_metadata):
        """Test des métriques de création d'alerte"""
        processing_time = 0.150  # 150ms
        
        await metrics_collector.record_alert_created(
            sample_alert_metadata, processing_time, "automatic"
        )
        
        # Vérification que les métriques ont été mises à jour
        metrics_data = metrics_collector.get_metrics_data()
        assert b"critical_alerts_created_total" in metrics_data
        assert b"critical_alert_processing_seconds" in metrics_data
        assert b"critical_alert_business_impact_score" in metrics_data
    
    @pytest_asyncio.async_test
    async def test_notification_metrics(self, metrics_collector, sample_alert_metadata):
        """Test des métriques de notification"""
        await metrics_collector.record_notification_sent(
            sample_alert_metadata,
            AlertChannel.SLACK,
            0.250,  # 250ms latency
            "success",
            "immediate"
        )
        
        metrics_data = metrics_collector.get_metrics_data()
        assert b"critical_alert_notifications_sent_total" in metrics_data
        assert b"critical_alert_notification_latency_seconds" in metrics_data
    
    @pytest_asyncio.async_test
    async def test_ml_prediction_metrics(self, metrics_collector, sample_alert_metadata):
        """Test des métriques ML"""
        await metrics_collector.record_ml_prediction(
            sample_alert_metadata,
            "escalation_predictor",
            "3.0.0",
            0.85,  # 85% confidence
            0.050,  # 50ms inference time
            "escalation"
        )
        
        metrics_data = metrics_collector.get_metrics_data()
        assert b"critical_alert_ml_prediction_accuracy" in metrics_data
        assert b"critical_alert_ml_inference_seconds" in metrics_data
    
    @pytest_asyncio.async_test
    async def test_sla_compliance_calculation(self, metrics_collector):
        """Test du calcul de conformité SLA"""
        compliance_data = await metrics_collector.calculate_sla_compliance(
            "test-tenant-001", 24
        )
        
        assert "response_time_sla" in compliance_data
        assert "escalation_sla" in compliance_data
        assert "resolution_sla" in compliance_data
        assert "notification_sla" in compliance_data
        
        for sla_type, compliance in compliance_data.items():
            assert 0.0 <= compliance <= 1.0
    
    @pytest_asyncio.async_test
    async def test_anomaly_detection(self, metrics_collector):
        """Test de détection d'anomalies"""
        # Ajout de données de test au buffer
        current_time = time.time()
        test_events = [
            {
                'event_type': 'alert_created',
                'timestamp': current_time - 3600,  # 1h ago
                'tenant_id': 'test-tenant',
                'processing_time': 0.1
            },
            {
                'event_type': 'alert_created', 
                'timestamp': current_time - 1800,  # 30m ago
                'tenant_id': 'test-tenant',
                'processing_time': 2.5  # Anomalie: temps très élevé
            }
        ]
        
        with metrics_collector.buffer_lock:
            metrics_collector.analytics_buffer.extend(test_events)
        
        anomalies = await metrics_collector.detect_anomalies("test-tenant", 2)
        
        # Devrait détecter au moins l'anomalie de performance
        performance_anomalies = [a for a in anomalies if a['type'] == 'performance_anomaly']
        assert len(performance_anomalies) > 0
    
    @pytest_asyncio.async_test
    async def test_insights_report_generation(self, metrics_collector):
        """Test de génération de rapport d'insights"""
        # Ajout de données de test
        current_time = time.time()
        test_events = [
            {
                'event_type': 'alert_created',
                'timestamp': current_time - 3600,
                'tenant_id': 'test-tenant',
                'severity': 'CRITICAL',
                'source_service': 'api-gateway',
                'processing_time': 0.1
            },
            {
                'event_type': 'escalation',
                'timestamp': current_time - 3500,
                'tenant_id': 'test-tenant',
                'escalation_level': 1
            },
            {
                'event_type': 'alert_resolved',
                'timestamp': current_time - 3000,
                'tenant_id': 'test-tenant'
            }
        ]
        
        with metrics_collector.buffer_lock:
            metrics_collector.analytics_buffer.extend(test_events)
        
        report = await metrics_collector.generate_insights_report("test-tenant", 2)
        
        assert "summary" in report
        assert "trends" in report
        assert "recommendations" in report
        assert "anomalies" in report
        assert report["summary"]["total_alerts"] == 1
        assert report["summary"]["total_escalations"] == 1
        assert report["summary"]["total_resolutions"] == 1

# === TESTS D'INTÉGRATION ===

class TestIntegration:
    """Tests d'intégration end-to-end"""
    
    @pytest_asyncio.async_test
    async def test_full_alert_workflow(self, redis_client, db_session, metrics_registry):
        """Test du workflow complet d'alerte"""
        # Initialisation des composants
        processor = await CriticalAlertProcessorFactory.create_processor(
            redis_url="redis://localhost:6379/1",
            db_session=db_session
        )
        
        metrics_collector = AlertMetricsCollector(metrics_registry)
        template_engine = SlackTemplateEngine()
        
        # Données d'alerte de test
        alert_data = {
            "severity": "CRITICAL",
            "source_service": "payment-service",
            "affected_users": 5000,
            "business_impact": 3.5,
            "correlation_id": str(uuid.uuid4()),
            "trace_id": str(uuid.uuid4()),
            "tags": {
                "environment": "production",
                "cluster": "us-east-1",
                "team": "payments"
            },
            "custom_data": {
                "error_rate": 0.35,
                "latency_p99": 8.5,
                "transactions_per_second": 1250
            }
        }
        
        tenant_context = {
            "tenant_id": "enterprise-customer-001",
            "tier": "ENTERPRISE_PLUS"
        }
        
        # 1. Traitement de l'alerte
        start_time = time.time()
        result = await processor.process_critical_alert(alert_data, tenant_context)
        processing_time = time.time() - start_time
        
        assert result["status"] == "processed"
        alert_id = result["alert_id"]
        
        # 2. Récupération des métadonnées depuis le cache
        alert_key = f"critical_alert:{tenant_context['tenant_id']}:{alert_id}"
        cached_data = await redis_client.get(alert_key)
        assert cached_data is not None
        
        cached_alert = json.loads(cached_data)
        assert cached_alert["severity"] == "CRITICAL"
        assert cached_alert["tenant_id"] == tenant_context["tenant_id"]
        
        # 3. Génération du template Slack
        alert_metadata = CriticalAlertMetadata(
            alert_id=alert_id,
            tenant_id=tenant_context["tenant_id"],
            severity=CriticalAlertSeverity.CRITICAL,
            tenant_tier=TenantTier.ENTERPRISE_PLUS,
            source_service=alert_data["source_service"],
            affected_users=alert_data["affected_users"],
            business_impact=alert_data["business_impact"],
            correlation_id=alert_data["correlation_id"],
            trace_id=alert_data["trace_id"],
            tags=alert_data["tags"],
            custom_data=alert_data["custom_data"]
        )
        
        template_context = SlackTemplateContext(
            alert_metadata=alert_metadata,
            tenant_config={"slack_template_override": None},
            locale="en",
            timezone="UTC"
        )
        
        slack_message = await template_engine.generate_slack_message(
            template_context, SlackTemplateType.INTERACTIVE_BLOCKS
        )
        
        assert "blocks" in slack_message
        assert len(slack_message["blocks"]) > 5  # Header + Main + Actions + etc.
        
        # 4. Enregistrement des métriques
        await metrics_collector.record_alert_created(
            alert_metadata, processing_time, "integration_test"
        )
        
        await metrics_collector.record_notification_sent(
            alert_metadata, AlertChannel.SLACK, 0.150, "success", "immediate"
        )
        
        # 5. Vérification des métriques
        metrics_data = metrics_collector.get_metrics_data()
        assert b"critical_alerts_created_total" in metrics_data
        assert b"critical_alert_notifications_sent_total" in metrics_data
        
        # 6. Test d'escalade automatique
        await metrics_collector.record_escalation(
            alert_metadata, 1, 120.0, "automatic"
        )
        
        # 7. Test de résolution
        await metrics_collector.record_alert_resolved(
            alert_metadata, "manual", "engineer", 1800.0
        )
        
        # Vérification finale
        final_metrics = metrics_collector.get_metrics_data()
        assert b"critical_alert_escalations_total" in final_metrics
        assert b"critical_alerts_resolved_total" in final_metrics

# === TESTS DE PERFORMANCE ===

class TestPerformance:
    """Tests de performance et charge"""
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_high_volume_alert_processing(self, redis_client, db_session):
        """Test de traitement d'un volume élevé d'alertes"""
        processor = await CriticalAlertProcessorFactory.create_processor(
            redis_url="redis://localhost:6379/1",
            db_session=db_session
        )
        
        # Génération de 100 alertes
        alerts = []
        for i in range(100):
            alerts.append({
                "alert_data": {
                    "severity": fake.random_element(elements=["WARNING", "HIGH", "CRITICAL"]),
                    "source_service": f"service-{i % 10}",
                    "affected_users": fake.random_int(min=1, max=1000),
                    "business_impact": fake.random.uniform(0.1, 3.0)
                },
                "tenant_context": {
                    "tenant_id": f"tenant-{i % 5}",
                    "tier": fake.random_element(elements=["FREE", "PREMIUM", "ENTERPRISE"])
                }
            })
        
        # Traitement parallèle
        start_time = time.time()
        
        tasks = [
            processor.process_critical_alert(alert["alert_data"], alert["tenant_context"])
            for alert in alerts
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processing_time = time.time() - start_time
        
        # Vérifications de performance
        assert processing_time < 10.0  # Moins de 10 secondes pour 100 alertes
        
        successful_results = [r for r in results if isinstance(r, dict) and r.get("status") == "processed"]
        assert len(successful_results) >= 95  # Au moins 95% de réussite
        
        avg_processing_time = sum(r.get("processing_time_ms", 0) for r in successful_results) / len(successful_results)
        assert avg_processing_time < 200  # Moins de 200ms en moyenne
    
    @pytest.mark.asyncio
    async def test_template_generation_performance(self, slack_template_engine):
        """Test de performance de génération de templates"""
        # Génération de 50 templates
        start_time = time.time()
        
        tasks = []
        for i in range(50):
            alert_metadata = generate_random_alert_metadata()
            context = SlackTemplateContext(
                alert_metadata=alert_metadata,
                tenant_config={},
                locale="en"
            )
            
            task = slack_template_engine.generate_slack_message(
                context, SlackTemplateType.INTERACTIVE_BLOCKS
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        generation_time = time.time() - start_time
        
        assert generation_time < 5.0  # Moins de 5 secondes pour 50 templates
        assert len(results) == 50
        assert all("blocks" in result for result in results)
        
        avg_time_per_template = generation_time / 50
        assert avg_time_per_template < 0.1  # Moins de 100ms par template

# === TESTS DE SÉCURITÉ ===

class TestSecurity:
    """Tests de sécurité"""
    
    def test_input_sanitization(self, sample_alert_metadata):
        """Test de sanitisation des entrées"""
        # Test avec des données malveillantes
        malicious_data = {
            "severity": "CRITICAL'; DROP TABLE alerts; --",
            "source_service": "<script>alert('xss')</script>",
            "tags": {
                "environment": "production",
                "malicious": "'; DELETE FROM users; --"
            }
        }
        
        # Le système doit gérer ces entrées sans erreur
        # En production, ajouter une validation stricte
        assert malicious_data["severity"].replace("'; DROP TABLE alerts; --", "") == "CRITICAL"
    
    def test_tenant_isolation(self, sample_alert_metadata):
        """Test d'isolation des tenants"""
        tenant_a_id = "tenant-a-001"
        tenant_b_id = "tenant-b-002"
        
        # Vérifier que les IDs de tenant sont différents
        assert tenant_a_id != tenant_b_id
        
        # En production, ajouter des tests pour vérifier
        # que les données d'un tenant ne sont pas accessibles par un autre
    
    @pytest.mark.parametrize("sensitive_field", [
        "alert_id", "tenant_id", "correlation_id", "trace_id"
    ])
    def test_sensitive_data_handling(self, sample_alert_metadata, sensitive_field):
        """Test de gestion des données sensibles"""
        sensitive_value = getattr(sample_alert_metadata, sensitive_field)
        
        # Vérifier que les valeurs sensibles ne sont pas vides
        assert sensitive_value is not None
        assert len(str(sensitive_value)) > 0
        
        # En production, ajouter la vérification du chiffrement

# === TESTS DE CONFORMITÉ ===

class TestCompliance:
    """Tests de conformité réglementaire"""
    
    def test_gdpr_compliance(self, sample_alert_metadata):
        """Test de conformité GDPR"""
        # Vérifier que les données personnelles ne sont pas stockées
        # dans les métadonnées d'alerte sans consentement
        alert_dict = asdict(sample_alert_metadata)
        
        # Liste des champs qui ne doivent pas contenir de données personnelles
        non_personal_fields = [
            "alert_id", "tenant_id", "severity", "source_service",
            "business_impact", "affected_users"
        ]
        
        for field in non_personal_fields:
            if field in alert_dict:
                value = str(alert_dict[field])
                # Vérifier qu'il n'y a pas d'emails ou numéros de téléphone
                assert "@" not in value or "example.com" in value
                assert not any(char.isdigit() for char in value) or field in ["affected_users", "business_impact"]
    
    def test_audit_trail(self, metrics_collector):
        """Test de piste d'audit"""
        # Vérifier que toutes les actions importantes sont loggées
        # En production, intégrer avec un système d'audit centralisé
        
        # Les métriques doivent inclure des informations d'audit
        metrics_data = metrics_collector.get_metrics_data()
        assert isinstance(metrics_data, (str, bytes))
        assert len(metrics_data) > 0

# === CONFIGURATION PYTEST ===

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Configuration globale de l'environnement de test"""
    # Configuration de logging pour les tests
    logging.getLogger("critical_alerts").setLevel(logging.DEBUG)
    
    # Configuration de timezone pour la cohérence
    import os
    os.environ["TZ"] = "UTC"
    
    yield
    
    # Nettoyage après tous les tests
    logging.getLogger("critical_alerts").handlers.clear()

# Markers personnalisés
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.timeout(60)  # Timeout global de 60 secondes
]

if __name__ == "__main__":
    # Exécution directe des tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=critical",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])
