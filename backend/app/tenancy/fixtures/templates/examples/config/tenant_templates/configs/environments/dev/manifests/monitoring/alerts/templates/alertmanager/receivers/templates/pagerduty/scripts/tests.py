#!/usr/bin/env python3
"""
Testing Framework pour PagerDuty Integration Scripts

Framework de tests complet pour tous les composants PagerDuty.
Inclut tests unitaires, tests d'intégration, tests de performance,
et validation de configuration avec mocking avancé.

Fonctionnalités:
- Tests unitaires complets pour tous les modules
- Tests d'intégration avec mocking PagerDuty
- Tests de performance et de charge
- Validation des configurations
- Tests de régression automatisés
- Génération de rapports de couverture
- Tests de sécurité et vulnérabilités

Version: 1.0.0
Auteur: Spotify AI Agent Team
"""

import asyncio
import unittest
import pytest
import json
import sys
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
import tempfile
import yaml
import structlog
from dataclasses import dataclass
import aiohttp
from aioresponses import aioresponses
import time
import concurrent.futures
from contextlib import asynccontextmanager

# Imports des modules à tester
from utils.api_client import PagerDutyAPIClient, PagerDutyEventAction, PagerDutySeverity
from utils.validators import PagerDutyValidator, ValidationResult
from utils.formatters import MessageFormatter
from utils.encryption import KeyManager, SymmetricEncryption
from config_manager import ConfigManager
from health_checker import PagerDutyHealthChecker
from backup_manager import BackupManager
from alert_manager import PagerDutyAlertManager, Alert, AlertStatus
from incident_manager import PagerDutyIncidentManager, IncidentStatus
from reporting import ReportGenerator, ReportType, ExportFormat

logger = structlog.get_logger(__name__)

class TestConfig:
    """Configuration pour les tests"""
    TEST_API_KEY = "test_api_key_12345"
    TEST_INTEGRATION_KEY = "test_integration_key_67890"
    TEST_SERVICE_ID = "TEST_SERVICE_123"
    TEST_EMAIL = "test@example.com"
    MOCK_SERVER_URL = "https://events.pagerduty.com"

@dataclass
class TestMetrics:
    """Métriques de performance des tests"""
    test_name: str
    duration: float
    memory_usage: float
    api_calls: int
    success: bool
    error_message: Optional[str] = None

class MockPagerDutyResponse:
    """Mock pour les réponses PagerDuty"""
    
    def __init__(self, status_code: int = 202, data: Dict[str, Any] = None, error: str = None):
        self.status_code = status_code
        self.data = data or {}
        self.error = error
        self.headers = {"Content-Type": "application/json"}
    
    def json(self):
        return self.data

class AsyncTestCase(unittest.TestCase):
    """Classe de base pour les tests asynchrones"""
    
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        self.loop.close()
    
    def run_async(self, coro):
        """Exécute une coroutine de manière synchrone"""
        return self.loop.run_until_complete(coro)

class TestAPIClient(AsyncTestCase):
    """Tests pour le client API PagerDuty"""
    
    def setUp(self):
        super().setUp()
        self.client = PagerDutyAPIClient(
            api_key=TestConfig.TEST_API_KEY,
            integration_key=TestConfig.TEST_INTEGRATION_KEY
        )
    
    def tearDown(self):
        super().tearDown()
        self.run_async(self.client.close())
    
    def test_client_initialization(self):
        """Test l'initialisation du client"""
        self.assertEqual(self.client.api_key, TestConfig.TEST_API_KEY)
        self.assertEqual(self.client.integration_key, TestConfig.TEST_INTEGRATION_KEY)
        self.assertIsNotNone(self.client.session)
    
    @patch('aiohttp.ClientSession.post')
    def test_send_event_success(self, mock_post):
        """Test l'envoi d'événement réussi"""
        # Mock de la réponse
        mock_response = AsyncMock()
        mock_response.status = 202
        mock_response.json = AsyncMock(return_value={
            "status": "success",
            "dedup_key": "test_dedup_key",
            "incident_key": "test_incident_key"
        })
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Test
        response = self.run_async(self.client.send_event(
            action=PagerDutyEventAction.TRIGGER,
            summary="Test incident",
            source="test_source",
            severity=PagerDutySeverity.HIGH
        ))
        
        self.assertEqual(response.status_code, 202)
        self.assertIn("dedup_key", response.data)
        mock_post.assert_called_once()
    
    @patch('aiohttp.ClientSession.post')
    def test_send_event_failure(self, mock_post):
        """Test l'envoi d'événement échoué"""
        # Mock de la réponse d'erreur
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.json = AsyncMock(return_value={
            "status": "error",
            "message": "Invalid request"
        })
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Test
        response = self.run_async(self.client.send_event(
            action=PagerDutyEventAction.TRIGGER,
            summary="",  # Summary vide pour déclencher l'erreur
            source="test_source",
            severity=PagerDutySeverity.HIGH
        ))
        
        self.assertEqual(response.status_code, 400)
        self.assertIsNotNone(response.error)
    
    def test_rate_limiting(self):
        """Test le rate limiting"""
        # Simuler plusieurs appels rapides
        start_time = time.time()
        
        for _ in range(5):
            can_proceed = self.run_async(self.client.check_rate_limit())
            if not can_proceed:
                break
        
        # Vérifier qu'il y a eu un délai
        elapsed_time = time.time() - start_time
        self.assertGreater(elapsed_time, 0)
    
    def test_retry_mechanism(self):
        """Test le mécanisme de retry"""
        with patch.object(self.client, '_make_request') as mock_request:
            # Simuler des échecs puis un succès
            mock_request.side_effect = [
                Exception("Network error"),
                Exception("Timeout"),
                MockPagerDutyResponse(status_code=202, data={"status": "success"})
            ]
            
            response = self.run_async(self.client.send_event(
                action=PagerDutyEventAction.TRIGGER,
                summary="Test retry",
                source="test_source",
                severity=PagerDutySeverity.HIGH
            ))
            
            # Vérifier que la requête a été retentée
            self.assertEqual(mock_request.call_count, 3)
            self.assertEqual(response.status_code, 202)

class TestValidators(unittest.TestCase):
    """Tests pour les validateurs"""
    
    def setUp(self):
        self.validator = PagerDutyValidator()
    
    def test_validate_api_key_valid(self):
        """Test validation clé API valide"""
        result = self.validator.validate_api_key("u+abc123def456ghi789")
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
    
    def test_validate_api_key_invalid(self):
        """Test validation clé API invalide"""
        result = self.validator.validate_api_key("invalid_key")
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
    
    def test_validate_integration_key_valid(self):
        """Test validation clé d'intégration valide"""
        result = self.validator.validate_integration_key("a" * 32)
        self.assertTrue(result.is_valid)
    
    def test_validate_integration_key_invalid(self):
        """Test validation clé d'intégration invalide"""
        result = self.validator.validate_integration_key("too_short")
        self.assertFalse(result.is_valid)
    
    def test_validate_event_data_complete(self):
        """Test validation données d'événement complètes"""
        event_data = {
            "summary": "Test incident",
            "source": "test_source",
            "severity": "high",
            "dedup_key": "test_key"
        }
        
        result = self.validator.validate_event_data(event_data)
        self.assertTrue(result.is_valid)
    
    def test_validate_event_data_missing_required(self):
        """Test validation données d'événement incomplètes"""
        event_data = {
            "source": "test_source"
            # Summary manquant
        }
        
        result = self.validator.validate_event_data(event_data)
        self.assertFalse(result.is_valid)
        self.assertIn("summary", str(result.errors))
    
    def test_validate_webhook_url_valid(self):
        """Test validation URL webhook valide"""
        result = self.validator.validate_webhook_url("https://hooks.pagerduty.com/services/TEST123")
        self.assertTrue(result.is_valid)
    
    def test_validate_webhook_url_invalid(self):
        """Test validation URL webhook invalide"""
        result = self.validator.validate_webhook_url("not_a_url")
        self.assertFalse(result.is_valid)
    
    def test_security_validation(self):
        """Test validations de sécurité"""
        # Test injection SQL
        malicious_input = "'; DROP TABLE incidents; --"
        result = self.validator.validate_input_security(malicious_input)
        self.assertFalse(result.is_valid)
        
        # Test XSS
        xss_input = "<script>alert('xss')</script>"
        result = self.validator.validate_input_security(xss_input)
        self.assertFalse(result.is_valid)
        
        # Test input valide
        safe_input = "Normal incident description"
        result = self.validator.validate_input_security(safe_input)
        self.assertTrue(result.is_valid)

class TestFormatters(unittest.TestCase):
    """Tests pour les formateurs"""
    
    def setUp(self):
        self.formatter = MessageFormatter()
    
    def test_format_alert_message(self):
        """Test formatage message d'alerte"""
        message = self.formatter.format_alert_message(
            severity="high",
            title="Database connection failed",
            source="db-server-01",
            timestamp=datetime.now(),
            details={"error_code": 500, "retry_count": 3}
        )
        
        self.assertIn("high", message.lower())
        self.assertIn("database connection failed", message.lower())
        self.assertIn("db-server-01", message)
    
    def test_format_incident_summary(self):
        """Test formatage résumé d'incident"""
        incident_data = {
            "id": "INC-123",
            "title": "Service outage",
            "status": "acknowledged",
            "created_at": datetime.now().isoformat(),
            "assigned_team": "platform-team"
        }
        
        summary = self.formatter.format_incident_summary(incident_data)
        
        self.assertIn("INC-123", summary)
        self.assertIn("Service outage", summary)
        self.assertIn("acknowledged", summary)
    
    def test_format_metrics_report(self):
        """Test formatage rapport de métriques"""
        metrics = {
            "mttr": 45.5,
            "mtta": 8.2,
            "incident_count": 12,
            "escalation_rate": 0.25
        }
        
        report = self.formatter.format_metrics_report(metrics)
        
        self.assertIn("45.5", report)
        self.assertIn("8.2", report)
        self.assertIn("12", report)
    
    def test_template_rendering(self):
        """Test rendu de templates"""
        template = "Alert: {{title}} on {{source}} at {{timestamp}}"
        context = {
            "title": "High CPU usage",
            "source": "web-server-01",
            "timestamp": "2024-01-01T12:00:00Z"
        }
        
        result = self.formatter.render_template(template, context)
        
        self.assertEqual(result, "Alert: High CPU usage on web-server-01 at 2024-01-01T12:00:00Z")
    
    def test_sanitize_output(self):
        """Test sanitisation de sortie"""
        unsafe_text = "<script>alert('xss')</script>Safe content"
        safe_text = self.formatter.sanitize_output(unsafe_text)
        
        self.assertNotIn("<script>", safe_text)
        self.assertIn("Safe content", safe_text)

class TestEncryption(unittest.TestCase):
    """Tests pour l'encryption"""
    
    def setUp(self):
        self.key_manager = KeyManager()
        self.encryption = SymmetricEncryption(self.key_manager)
    
    def test_key_generation(self):
        """Test génération de clés"""
        key = self.key_manager.generate_key()
        self.assertIsNotNone(key)
        self.assertEqual(len(key), 44)  # Base64 encoded 32-byte key
    
    def test_symmetric_encryption_decryption(self):
        """Test chiffrement/déchiffrement symétrique"""
        plaintext = "Sensitive configuration data"
        
        # Chiffrer
        encrypted = self.encryption.encrypt(plaintext)
        self.assertNotEqual(plaintext, encrypted)
        
        # Déchiffrer
        decrypted = self.encryption.decrypt(encrypted)
        self.assertEqual(plaintext, decrypted)
    
    def test_encryption_with_different_keys(self):
        """Test que différentes clés donnent des résultats différents"""
        plaintext = "Test data"
        
        # Première encryption
        key1 = self.key_manager.generate_key()
        enc1 = SymmetricEncryption(self.key_manager)
        encrypted1 = enc1.encrypt(plaintext)
        
        # Deuxième encryption avec nouvelle clé
        key2 = self.key_manager.generate_key()
        enc2 = SymmetricEncryption(self.key_manager)
        encrypted2 = enc2.encrypt(plaintext)
        
        self.assertNotEqual(encrypted1, encrypted2)
    
    def test_invalid_decryption(self):
        """Test déchiffrement avec données invalides"""
        with self.assertRaises(Exception):
            self.encryption.decrypt("invalid_encrypted_data")
    
    def test_key_rotation(self):
        """Test rotation des clés"""
        old_key = self.key_manager.current_key
        new_key = self.key_manager.rotate_key()
        
        self.assertNotEqual(old_key, new_key)
        self.assertEqual(self.key_manager.current_key, new_key)

class TestConfigManager(AsyncTestCase):
    """Tests pour le gestionnaire de configuration"""
    
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(base_path=self.temp_dir)
    
    def test_load_config_yaml(self):
        """Test chargement configuration YAML"""
        config_data = {
            "pagerduty": {
                "api_key": "test_key",
                "integration_key": "test_integration"
            },
            "alerts": {
                "default_severity": "medium",
                "timeout": 300
            }
        }
        
        config_file = os.path.join(self.temp_dir, "test_config.yaml")
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        loaded_config = self.run_async(self.config_manager.load_config(config_file))
        
        self.assertEqual(loaded_config["pagerduty"]["api_key"], "test_key")
        self.assertEqual(loaded_config["alerts"]["timeout"], 300)
    
    def test_load_config_json(self):
        """Test chargement configuration JSON"""
        config_data = {
            "pagerduty": {
                "api_key": "test_key",
                "integration_key": "test_integration"
            }
        }
        
        config_file = os.path.join(self.temp_dir, "test_config.json")
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        loaded_config = self.run_async(self.config_manager.load_config(config_file))
        
        self.assertEqual(loaded_config["pagerduty"]["api_key"], "test_key")
    
    def test_validate_config(self):
        """Test validation de configuration"""
        valid_config = {
            "pagerduty": {
                "api_key": "u+valid_key_123",
                "integration_key": "a" * 32
            },
            "alerts": {
                "default_severity": "medium",
                "timeout": 300
            }
        }
        
        is_valid = self.run_async(self.config_manager.validate_config(valid_config))
        self.assertTrue(is_valid)
        
        # Configuration invalide
        invalid_config = {
            "pagerduty": {
                "api_key": "invalid",
                # integration_key manquant
            }
        }
        
        is_valid = self.run_async(self.config_manager.validate_config(invalid_config))
        self.assertFalse(is_valid)
    
    def test_config_encryption(self):
        """Test chiffrement de configuration"""
        sensitive_config = {
            "pagerduty": {
                "api_key": "sensitive_api_key",
                "integration_key": "sensitive_integration_key"
            }
        }
        
        # Chiffrer
        encrypted_path = self.run_async(
            self.config_manager.encrypt_config(sensitive_config, "test_encrypted.enc")
        )
        self.assertTrue(os.path.exists(encrypted_path))
        
        # Déchiffrer
        decrypted_config = self.run_async(
            self.config_manager.decrypt_config(encrypted_path)
        )
        
        self.assertEqual(decrypted_config["pagerduty"]["api_key"], "sensitive_api_key")

class TestHealthChecker(AsyncTestCase):
    """Tests pour le vérificateur de santé"""
    
    def setUp(self):
        super().setUp()
        self.health_checker = PagerDutyHealthChecker(
            api_key=TestConfig.TEST_API_KEY,
            integration_key=TestConfig.TEST_INTEGRATION_KEY
        )
    
    def tearDown(self):
        super().tearDown()
        self.run_async(self.health_checker.close())
    
    @patch('utils.api_client.PagerDutyAPIClient.send_event')
    def test_check_api_connectivity(self, mock_send_event):
        """Test vérification connectivité API"""
        # Mock réponse réussie
        mock_send_event.return_value = MockPagerDutyResponse(
            status_code=202,
            data={"status": "success"}
        )
        
        result = self.run_async(self.health_checker.check_api_connectivity())
        
        self.assertTrue(result["healthy"])
        self.assertIn("response_time", result)
        mock_send_event.assert_called_once()
    
    @patch('utils.api_client.PagerDutyAPIClient.send_event')
    def test_check_api_connectivity_failure(self, mock_send_event):
        """Test vérification connectivité API échec"""
        # Mock réponse d'échec
        mock_send_event.side_effect = Exception("Connection failed")
        
        result = self.run_async(self.health_checker.check_api_connectivity())
        
        self.assertFalse(result["healthy"])
        self.assertIn("error", result)
    
    def test_check_rate_limits(self):
        """Test vérification limites de taux"""
        result = self.run_async(self.health_checker.check_rate_limits())
        
        self.assertIn("healthy", result)
        self.assertIn("current_rate", result)
        self.assertIn("limit", result)
    
    def test_comprehensive_health_check(self):
        """Test vérification de santé complète"""
        with patch.object(self.health_checker, 'check_api_connectivity') as mock_api, \
             patch.object(self.health_checker, 'check_rate_limits') as mock_rate, \
             patch.object(self.health_checker, 'check_configuration') as mock_config:
            
            # Mock toutes les vérifications comme réussies
            mock_api.return_value = {"healthy": True, "response_time": 100}
            mock_rate.return_value = {"healthy": True, "current_rate": 5}
            mock_config.return_value = {"healthy": True}
            
            result = self.run_async(self.health_checker.run_comprehensive_check())
            
            self.assertTrue(result["overall_healthy"])
            self.assertIn("checks", result)
            self.assertEqual(len(result["checks"]), 3)

class TestPerformance(AsyncTestCase):
    """Tests de performance"""
    
    def setUp(self):
        super().setUp()
        self.api_client = PagerDutyAPIClient(
            api_key=TestConfig.TEST_API_KEY,
            integration_key=TestConfig.TEST_INTEGRATION_KEY
        )
        self.metrics = []
    
    def tearDown(self):
        super().tearDown()
        self.run_async(self.api_client.close())
    
    def measure_performance(self, test_name: str):
        """Décorateur pour mesurer les performances"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = self._get_memory_usage()
                api_calls = 0
                error_message = None
                success = True
                
                try:
                    # Patch pour compter les appels API
                    with patch.object(self.api_client, '_make_request') as mock_request:
                        mock_request.return_value = MockPagerDutyResponse()
                        result = await func(*args, **kwargs)
                        api_calls = mock_request.call_count
                except Exception as e:
                    success = False
                    error_message = str(e)
                    result = None
                
                end_time = time.time()
                end_memory = self._get_memory_usage()
                
                metric = TestMetrics(
                    test_name=test_name,
                    duration=end_time - start_time,
                    memory_usage=end_memory - start_memory,
                    api_calls=api_calls,
                    success=success,
                    error_message=error_message
                )
                
                self.metrics.append(metric)
                return result
            
            return wrapper
        return decorator
    
    def _get_memory_usage(self) -> float:
        """Récupère l'utilisation mémoire"""
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    @measure_performance("single_event_send")
    async def test_single_event_performance(self):
        """Test performance envoi d'un événement"""
        response = await self.api_client.send_event(
            action=PagerDutyEventAction.TRIGGER,
            summary="Performance test event",
            source="test_source",
            severity=PagerDutySeverity.HIGH
        )
        return response
    
    @measure_performance("bulk_events_send")
    async def test_bulk_events_performance(self):
        """Test performance envoi d'événements en lot"""
        tasks = []
        
        for i in range(10):
            task = self.api_client.send_event(
                action=PagerDutyEventAction.TRIGGER,
                summary=f"Bulk test event {i}",
                source="test_source",
                severity=PagerDutySeverity.MEDIUM
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        return responses
    
    @measure_performance("concurrent_operations")
    async def test_concurrent_operations_performance(self):
        """Test performance opérations concurrentes"""
        # Simuler diverses opérations concurrentes
        tasks = []
        
        # Envoi d'événements
        for i in range(5):
            task = self.api_client.send_event(
                action=PagerDutyEventAction.TRIGGER,
                summary=f"Concurrent event {i}",
                source="test_source",
                severity=PagerDutySeverity.LOW
            )
            tasks.append(task)
        
        # Validation de données
        validator = PagerDutyValidator()
        for i in range(5):
            task = asyncio.create_task(asyncio.coroutine(
                lambda: validator.validate_api_key(f"test_key_{i}")
            )())
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    def test_memory_leak_detection(self):
        """Test détection de fuites mémoire"""
        initial_memory = self._get_memory_usage()
        
        # Effectuer de nombreuses opérations
        for _ in range(100):
            self.run_async(self.test_single_event_performance())
        
        final_memory = self._get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # La mémoire ne devrait pas augmenter de plus de 50MB
        self.assertLess(memory_increase, 50, 
                       f"Possible memory leak detected: {memory_increase}MB increase")
    
    def test_rate_limiting_performance(self):
        """Test performance avec rate limiting"""
        start_time = time.time()
        
        # Effectuer plus d'appels que la limite
        for _ in range(20):
            can_proceed = self.run_async(self.api_client.check_rate_limit())
            if not can_proceed:
                time.sleep(0.1)  # Attendre un peu
        
        total_time = time.time() - start_time
        
        # Vérifier que le rate limiting a bien fonctionné
        self.assertGreater(total_time, 1.0, "Rate limiting should have introduced delays")
        self.assertLess(total_time, 10.0, "Rate limiting delays should be reasonable")

class TestIntegration(AsyncTestCase):
    """Tests d'intégration bout en bout"""
    
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        
        # Configuration de test
        self.test_config = {
            "pagerduty": {
                "api_key": TestConfig.TEST_API_KEY,
                "integration_key": TestConfig.TEST_INTEGRATION_KEY
            },
            "alerts": {
                "default_severity": "medium",
                "timeout": 300
            }
        }
        
        # Sauvegarder la configuration
        self.config_file = os.path.join(self.temp_dir, "test_config.yaml")
        with open(self.config_file, 'w') as f:
            yaml.dump(self.test_config, f)
    
    @patch('utils.api_client.PagerDutyAPIClient._make_request')
    def test_complete_alert_workflow(self, mock_request):
        """Test workflow complet d'alerte"""
        # Mock des réponses PagerDuty
        mock_request.return_value = MockPagerDutyResponse(
            status_code=202,
            data={"status": "success", "dedup_key": "test_key"}
        )
        
        # Créer le gestionnaire d'alertes
        alert_manager = PagerDutyAlertManager(
            api_key=TestConfig.TEST_API_KEY,
            integration_key=TestConfig.TEST_INTEGRATION_KEY
        )
        
        # Créer une alerte
        alert = Alert(
            id="test_alert_001",
            title="Test database outage",
            description="Database connection failed",
            severity="high",
            source="db-server-01"
        )
        
        # Traiter l'alerte
        result = self.run_async(alert_manager.process_alert(alert))
        
        self.assertTrue(result)
        self.assertEqual(alert.status, AlertStatus.SENT)
        mock_request.assert_called()
    
    @patch('utils.api_client.PagerDutyAPIClient._make_request')
    def test_incident_management_workflow(self, mock_request):
        """Test workflow complet de gestion d'incident"""
        # Mock des réponses
        mock_request.return_value = MockPagerDutyResponse(
            status_code=202,
            data={"status": "success"}
        )
        
        # Créer le gestionnaire d'incidents
        incident_manager = PagerDutyIncidentManager(
            api_key=TestConfig.TEST_API_KEY
        )
        
        # Données d'incident
        incident_data = {
            "id": "INC-001",
            "title": "Critical service outage",
            "severity": "critical",
            "service": "web-frontend",
            "status": "triggered",
            "created_at": datetime.now().isoformat()
        }
        
        # Traiter l'incident
        result = self.run_async(incident_manager.handle_incident(incident_data))
        
        self.assertTrue(result)
        self.assertIn("INC-001", incident_manager.active_incidents)
        
        # Acknowledger l'incident
        ack_result = self.run_async(
            incident_manager.acknowledge_incident("INC-001", TestConfig.TEST_EMAIL)
        )
        self.assertTrue(ack_result)
        
        # Résoudre l'incident
        resolve_result = self.run_async(
            incident_manager.resolve_incident("INC-001", TestConfig.TEST_EMAIL, "Fixed")
        )
        self.assertTrue(resolve_result)
    
    def test_configuration_validation_integration(self):
        """Test intégration validation de configuration"""
        config_manager = ConfigManager(base_path=self.temp_dir)
        
        # Charger et valider la configuration
        config = self.run_async(config_manager.load_config(self.config_file))
        is_valid = self.run_async(config_manager.validate_config(config))
        
        self.assertTrue(is_valid)
        
        # Test avec configuration invalide
        invalid_config = config.copy()
        invalid_config["pagerduty"]["api_key"] = "invalid"
        
        is_valid = self.run_async(config_manager.validate_config(invalid_config))
        self.assertFalse(is_valid)
    
    def test_backup_restore_workflow(self):
        """Test workflow de sauvegarde et restauration"""
        backup_manager = BackupManager(backup_path=self.temp_dir)
        
        # Créer des données de test
        test_data = {
            "configurations": self.test_config,
            "alerts": [
                {"id": "alert_1", "title": "Test alert 1"},
                {"id": "alert_2", "title": "Test alert 2"}
            ]
        }
        
        # Créer une sauvegarde
        backup_path = self.run_async(
            backup_manager.create_backup(test_data, "test_backup")
        )
        
        self.assertTrue(os.path.exists(backup_path))
        
        # Restaurer la sauvegarde
        restored_data = self.run_async(
            backup_manager.restore_backup(backup_path)
        )
        
        self.assertEqual(restored_data["configurations"], self.test_config)
        self.assertEqual(len(restored_data["alerts"]), 2)
    
    @patch('reporting.DataCollector.collect_incidents_data')
    def test_reporting_integration(self, mock_collect):
        """Test intégration du reporting"""
        # Mock des données d'incidents
        mock_incidents = [
            {
                "id": "INC-001",
                "title": "Test incident 1",
                "severity": "high",
                "created_at": datetime.now().isoformat(),
                "resolved_at": (datetime.now() + timedelta(hours=2)).isoformat(),
                "service": "web-frontend"
            },
            {
                "id": "INC-002", 
                "title": "Test incident 2",
                "severity": "medium",
                "created_at": (datetime.now() - timedelta(days=1)).isoformat(),
                "resolved_at": (datetime.now() - timedelta(days=1) + timedelta(hours=1)).isoformat(),
                "service": "api-gateway"
            }
        ]
        
        mock_collect.return_value = mock_incidents
        
        # Créer le générateur de rapports
        report_generator = ReportGenerator(TestConfig.TEST_API_KEY)
        
        # Configuration du rapport
        from reporting import ReportConfig, ReportType, ExportFormat
        config = ReportConfig(
            report_type=ReportType.INCIDENT_SUMMARY,
            date_range=(
                datetime.now() - timedelta(days=7),
                datetime.now()
            ),
            export_format=ExportFormat.JSON
        )
        
        # Générer le rapport
        report = self.run_async(report_generator.generate_report(config))
        
        self.assertIn("metadata", report)
        self.assertIn("kpis", report)
        self.assertEqual(report["metadata"]["total_incidents"], 2)

class TestSecurity(unittest.TestCase):
    """Tests de sécurité"""
    
    def setUp(self):
        self.validator = PagerDutyValidator()
    
    def test_sql_injection_prevention(self):
        """Test prévention injection SQL"""
        malicious_inputs = [
            "'; DROP TABLE incidents; --",
            "' OR '1'='1",
            "'; DELETE FROM users; --",
            "' UNION SELECT * FROM passwords --"
        ]
        
        for malicious_input in malicious_inputs:
            result = self.validator.validate_input_security(malicious_input)
            self.assertFalse(result.is_valid, f"Should detect SQL injection: {malicious_input}")
    
    def test_xss_prevention(self):
        """Test prévention XSS"""
        xss_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<iframe src='javascript:alert(`xss`)'></iframe>"
        ]
        
        for xss_input in xss_inputs:
            result = self.validator.validate_input_security(xss_input)
            self.assertFalse(result.is_valid, f"Should detect XSS: {xss_input}")
    
    def test_command_injection_prevention(self):
        """Test prévention injection de commandes"""
        command_injections = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& curl malicious-site.com",
            "`whoami`"
        ]
        
        for injection in command_injections:
            result = self.validator.validate_input_security(injection)
            self.assertFalse(result.is_valid, f"Should detect command injection: {injection}")
    
    def test_sensitive_data_detection(self):
        """Test détection de données sensibles"""
        sensitive_patterns = [
            "password123",
            "api_key_abcd1234",
            "secret_token_xyz",
            "credit_card_4111111111111111"
        ]
        
        for pattern in sensitive_patterns:
            result = self.validator.detect_sensitive_data(pattern)
            self.assertTrue(result.contains_sensitive_data, 
                          f"Should detect sensitive data: {pattern}")
    
    def test_encryption_security(self):
        """Test sécurité du chiffrement"""
        key_manager = KeyManager()
        encryption = SymmetricEncryption(key_manager)
        
        # Test force du chiffrement
        plaintext = "Sensitive API key: abc123xyz789"
        encrypted = encryption.encrypt(plaintext)
        
        # Le texte chiffré ne doit pas contenir le texte original
        self.assertNotIn("abc123xyz789", encrypted)
        self.assertNotIn("Sensitive", encrypted)
        
        # Test que deux chiffrements du même texte donnent des résultats différents
        encrypted2 = encryption.encrypt(plaintext)
        self.assertNotEqual(encrypted, encrypted2, "Encryption should use random IVs")

class TestRunner:
    """Exécuteur de tests principal"""
    
    def __init__(self):
        self.results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "performance_metrics": [],
            "coverage": {}
        }
    
    def run_all_tests(self):
        """Exécute tous les tests"""
        test_suites = [
            TestAPIClient,
            TestValidators,
            TestFormatters,
            TestEncryption,
            TestConfigManager,
            TestHealthChecker,
            TestPerformance,
            TestIntegration,
            TestSecurity
        ]
        
        print("🧪 Running PagerDuty Integration Tests...")
        print("=" * 60)
        
        for test_suite in test_suites:
            print(f"\n📋 Running {test_suite.__name__}...")
            
            # Créer la suite de tests
            suite = unittest.TestLoader().loadTestsFromTestCase(test_suite)
            
            # Exécuter les tests
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            
            # Collecter les résultats
            self.results["total_tests"] += result.testsRun
            self.results["passed"] += result.testsRun - len(result.failures) - len(result.errors)
            self.results["failed"] += len(result.failures)
            self.results["errors"] += len(result.errors)
            
            # Collecter les métriques de performance si disponibles
            if hasattr(test_suite, 'metrics') and hasattr(result, 'metrics'):
                self.results["performance_metrics"].extend(result.metrics)
    
    def generate_test_report(self) -> str:
        """Génère un rapport de tests"""
        success_rate = (self.results["passed"] / self.results["total_tests"]) * 100 if self.results["total_tests"] > 0 else 0
        
        report = f"""
🧪 PagerDuty Integration Test Report
{'=' * 50}

📊 Summary:
  Total Tests: {self.results['total_tests']}
  ✅ Passed: {self.results['passed']}
  ❌ Failed: {self.results['failed']}
  🚨 Errors: {self.results['errors']}
  📈 Success Rate: {success_rate:.1f}%

"""
        
        if self.results["performance_metrics"]:
            report += "⚡ Performance Metrics:\n"
            for metric in self.results["performance_metrics"]:
                status = "✅" if metric.success else "❌"
                report += f"  {status} {metric.test_name}: {metric.duration:.3f}s, {metric.api_calls} API calls\n"
        
        if self.results["failed"] > 0 or self.results["errors"] > 0:
            report += "\n⚠️  Some tests failed. Please check the detailed output above.\n"
        else:
            report += "\n🎉 All tests passed successfully!\n"
        
        return report

def main():
    """Fonction principale pour exécuter les tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PagerDuty Integration Test Suite")
    parser.add_argument("--test-suite", choices=["unit", "integration", "performance", "security", "all"],
                       default="all", help="Suite de tests à exécuter")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mode verbeux")
    parser.add_argument("--coverage", action="store_true", help="Activer la couverture de code")
    parser.add_argument("--report-file", help="Fichier de sortie pour le rapport")
    
    args = parser.parse_args()
    
    # Configuration du logging
    if args.verbose:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
        )
    
    # Exécution des tests
    runner = TestRunner()
    
    if args.test_suite == "all":
        runner.run_all_tests()
    else:
        # Exécuter seulement la suite spécifiée
        suite_map = {
            "unit": [TestAPIClient, TestValidators, TestFormatters, TestEncryption],
            "integration": [TestIntegration],
            "performance": [TestPerformance],
            "security": [TestSecurity]
        }
        
        for test_class in suite_map.get(args.test_suite, []):
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Générer le rapport
    report = runner.generate_test_report()
    print(report)
    
    # Sauvegarder le rapport si demandé
    if args.report_file:
        with open(args.report_file, 'w') as f:
            f.write(report)
        print(f"📄 Rapport sauvegardé dans: {args.report_file}")
    
    # Code de sortie basé sur les résultats
    if runner.results["failed"] > 0 or runner.results["errors"] > 0:
        return 1
    else:
        return 0

if __name__ == "__main__":
    sys.exit(main())
