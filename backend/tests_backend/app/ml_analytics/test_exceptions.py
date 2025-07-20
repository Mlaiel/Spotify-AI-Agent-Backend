# 🧪 ML Analytics Exceptions Tests
# =================================
# 
# Tests ultra-avancés pour le système d'exceptions
# Enterprise error handling testing
#
# 🎖️ Implementation par l'équipe d'experts:
# ✅ Développeur Backend Senior + Spécialiste Sécurité + Lead Dev
#
# 👨‍💻 Développé par: Fahed Mlaiel
# =================================

"""
🚨 Exception System Test Suite
==============================

Comprehensive testing for exception handling:
- Custom exception hierarchy
- Error categorization and severity
- Logging and alerting integration
- Error recovery mechanisms
- Exception propagation patterns
"""

import pytest
import logging
import json
import asyncio
from unittest.mock import patch, MagicMock, call
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import traceback
import sys

# Import modules to test
from app.ml_analytics.exceptions import (
    ErrorSeverity, ErrorCategory,
    MLAnalyticsError, ModelError, PipelineError,
    DataError, ConfigurationError, ResourceError,
    NetworkError, AuthenticationError, ValidationError,
    TimeoutError, AudioProcessingError, RecommendationError,
    ErrorHandler, ErrorReporter, ExceptionLogger,
    format_error_message, handle_async_exception,
    create_error_response, get_error_context
)


class TestErrorEnums:
    """Tests pour les énumérations d'erreur"""
    
    def test_error_severity_values(self):
        """Test des valeurs de sévérité"""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"
    
    def test_error_category_values(self):
        """Test des catégories d'erreur"""
        assert ErrorCategory.MODEL.value == "model"
        assert ErrorCategory.PIPELINE.value == "pipeline"
        assert ErrorCategory.DATA.value == "data"
        assert ErrorCategory.CONFIGURATION.value == "configuration"
        assert ErrorCategory.RESOURCE.value == "resource"
        assert ErrorCategory.NETWORK.value == "network"
        assert ErrorCategory.AUTHENTICATION.value == "authentication"
        assert ErrorCategory.VALIDATION.value == "validation"
        assert ErrorCategory.TIMEOUT.value == "timeout"
        assert ErrorCategory.UNKNOWN.value == "unknown"


class TestMLAnalyticsError:
    """Tests pour l'exception de base MLAnalyticsError"""
    
    def test_basic_error_creation(self):
        """Test de création d'erreur basique"""
        error = MLAnalyticsError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.category == ErrorCategory.UNKNOWN
        assert error.details == {}
        assert error.original_exception is None
        assert isinstance(error.timestamp, datetime)
        assert error.error_code.startswith("MLA-")
    
    def test_error_with_all_parameters(self):
        """Test d'erreur avec tous les paramètres"""
        original_exception = ValueError("Original error")
        details = {"context": "test", "user_id": 123}
        
        error = MLAnalyticsError(
            message="Complex error",
            error_code="CUSTOM-001",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.VALIDATION,
            details=details,
            original_exception=original_exception
        )
        
        assert error.message == "Complex error"
        assert error.error_code == "CUSTOM-001"
        assert error.severity == ErrorSeverity.HIGH
        assert error.category == ErrorCategory.VALIDATION
        assert error.details == details
        assert error.original_exception == original_exception
    
    def test_error_code_generation(self):
        """Test de génération automatique de code d'erreur"""
        error1 = MLAnalyticsError("Error 1")
        error2 = MLAnalyticsError("Error 2")
        
        # Les codes doivent être uniques
        assert error1.error_code != error2.error_code
        assert error1.error_code.startswith("MLA-")
        assert error2.error_code.startswith("MLA-")
    
    def test_traceback_capture(self):
        """Test de capture de stack trace"""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            error = MLAnalyticsError("Wrapped error", original_exception=e)
            
            assert error.traceback_info is not None
            assert "ValueError: Original error" in error.traceback_info
    
    @patch('app.ml_analytics.exceptions.logging.getLogger')
    def test_automatic_logging(self, mock_get_logger):
        """Test de logging automatique"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        error = MLAnalyticsError(
            "Test error",
            severity=ErrorSeverity.CRITICAL
        )
        
        # Vérifier que le logging a été appelé
        mock_get_logger.assert_called()
        mock_logger.error.assert_called()
    
    def test_error_serialization(self):
        """Test de sérialisation d'erreur"""
        error = MLAnalyticsError(
            "Test error",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MODEL,
            details={"model_name": "test_model"}
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["message"] == "Test error"
        assert error_dict["severity"] == "high"
        assert error_dict["category"] == "model"
        assert error_dict["details"]["model_name"] == "test_model"
        assert "timestamp" in error_dict
        assert "error_code" in error_dict


class TestSpecificExceptions:
    """Tests pour les exceptions spécifiques"""
    
    def test_model_error(self):
        """Test de ModelError"""
        error = ModelError(
            "Model training failed",
            model_name="recommendation_model",
            model_version="1.0"
        )
        
        assert error.category == ErrorCategory.MODEL
        assert error.details["model_name"] == "recommendation_model"
        assert error.details["model_version"] == "1.0"
    
    def test_pipeline_error(self):
        """Test de PipelineError"""
        error = PipelineError(
            "Pipeline execution failed",
            pipeline_id="ml_pipeline_001",
            stage="preprocessing"
        )
        
        assert error.category == ErrorCategory.PIPELINE
        assert error.details["pipeline_id"] == "ml_pipeline_001"
        assert error.details["stage"] == "preprocessing"
    
    def test_data_error(self):
        """Test de DataError"""
        error = DataError(
            "Invalid data format",
            data_source="spotify_tracks",
            expected_format="JSON",
            actual_format="XML"
        )
        
        assert error.category == ErrorCategory.DATA
        assert error.details["data_source"] == "spotify_tracks"
        assert error.details["expected_format"] == "JSON"
        assert error.details["actual_format"] == "XML"
    
    def test_configuration_error(self):
        """Test de ConfigurationError"""
        error = ConfigurationError(
            "Invalid configuration",
            config_section="database",
            invalid_field="url"
        )
        
        assert error.category == ErrorCategory.CONFIGURATION
        assert error.details["config_section"] == "database"
        assert error.details["invalid_field"] == "url"
    
    def test_resource_error(self):
        """Test de ResourceError"""
        error = ResourceError(
            "Insufficient memory",
            resource_type="memory",
            requested=1000000000,
            available=500000000
        )
        
        assert error.category == ErrorCategory.RESOURCE
        assert error.details["resource_type"] == "memory"
        assert error.details["requested"] == 1000000000
        assert error.details["available"] == 500000000
    
    def test_network_error(self):
        """Test de NetworkError"""
        error = NetworkError(
            "Connection timeout",
            endpoint="https://api.spotify.com",
            timeout=30,
            retry_count=3
        )
        
        assert error.category == ErrorCategory.NETWORK
        assert error.details["endpoint"] == "https://api.spotify.com"
        assert error.details["timeout"] == 30
        assert error.details["retry_count"] == 3
    
    def test_authentication_error(self):
        """Test de AuthenticationError"""
        error = AuthenticationError(
            "Invalid token",
            token_type="JWT",
            reason="expired"
        )
        
        assert error.category == ErrorCategory.AUTHENTICATION
        assert error.severity == ErrorSeverity.HIGH
        assert error.details["token_type"] == "JWT"
        assert error.details["reason"] == "expired"
    
    def test_validation_error(self):
        """Test de ValidationError"""
        error = ValidationError(
            "Invalid input data",
            field_name="user_id",
            expected_type="int",
            actual_value="invalid"
        )
        
        assert error.category == ErrorCategory.VALIDATION
        assert error.details["field_name"] == "user_id"
        assert error.details["expected_type"] == "int"
        assert error.details["actual_value"] == "invalid"
    
    def test_timeout_error(self):
        """Test de TimeoutError"""
        error = TimeoutError(
            "Operation timed out",
            operation="model_training",
            timeout_seconds=300,
            elapsed_seconds=350
        )
        
        assert error.category == ErrorCategory.TIMEOUT
        assert error.details["operation"] == "model_training"
        assert error.details["timeout_seconds"] == 300
        assert error.details["elapsed_seconds"] == 350
    
    def test_audio_processing_error(self):
        """Test de AudioProcessingError"""
        error = AudioProcessingError(
            "MFCC extraction failed",
            audio_file="test.mp3",
            sample_rate=22050,
            duration=180.5
        )
        
        assert error.category == ErrorCategory.DATA
        assert error.details["audio_file"] == "test.mp3"
        assert error.details["sample_rate"] == 22050
        assert error.details["duration"] == 180.5
    
    def test_recommendation_error(self):
        """Test de RecommendationError"""
        error = RecommendationError(
            "No recommendations available",
            user_id="user_123",
            algorithm="collaborative_filtering",
            reason="insufficient_data"
        )
        
        assert error.category == ErrorCategory.MODEL
        assert error.details["user_id"] == "user_123"
        assert error.details["algorithm"] == "collaborative_filtering"
        assert error.details["reason"] == "insufficient_data"


class TestErrorHandler:
    """Tests pour le gestionnaire d'erreurs"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.error_handler = ErrorHandler()
    
    def test_error_handler_creation(self):
        """Test de création du gestionnaire"""
        assert isinstance(self.error_handler, ErrorHandler)
        assert self.error_handler.error_count == 0
        assert len(self.error_handler.recent_errors) == 0
    
    def test_handle_error(self):
        """Test de gestion d'erreur"""
        error = MLAnalyticsError("Test error")
        
        handled = self.error_handler.handle_error(error)
        
        assert handled is True
        assert self.error_handler.error_count == 1
        assert len(self.error_handler.recent_errors) == 1
        assert self.error_handler.recent_errors[0] == error
    
    def test_handle_critical_error(self):
        """Test de gestion d'erreur critique"""
        critical_error = MLAnalyticsError(
            "Critical system failure",
            severity=ErrorSeverity.CRITICAL
        )
        
        with patch.object(self.error_handler, '_send_alert') as mock_alert:
            self.error_handler.handle_error(critical_error)
            
            # Vérifier qu'une alerte a été envoyée
            mock_alert.assert_called_once_with(critical_error)
    
    def test_error_recovery(self):
        """Test de récupération d'erreur"""
        def recovery_func():
            return "recovered"
        
        error = MLAnalyticsError("Recoverable error")
        
        result = self.error_handler.handle_error_with_recovery(
            error,
            recovery_func
        )
        
        assert result == "recovered"
    
    def test_error_rate_limiting(self):
        """Test de limitation du taux d'erreurs"""
        # Générer beaucoup d'erreurs rapidement
        for i in range(100):
            error = MLAnalyticsError(f"Error {i}")
            self.error_handler.handle_error(error)
        
        # Vérifier que le rate limiting est activé
        assert self.error_handler.is_rate_limited()
    
    @pytest.mark.asyncio
    async def test_async_error_handling(self):
        """Test de gestion d'erreur asynchrone"""
        async def async_operation():
            raise MLAnalyticsError("Async error")
        
        with pytest.raises(MLAnalyticsError):
            await self.error_handler.handle_async_operation(async_operation)
        
        assert self.error_handler.error_count == 1
    
    def test_error_context_capture(self):
        """Test de capture de contexte d'erreur"""
        context = {
            "user_id": 123,
            "request_id": "req_456",
            "endpoint": "/recommendations"
        }
        
        error = MLAnalyticsError("Test error")
        
        self.error_handler.add_context(context)
        self.error_handler.handle_error(error)
        
        handled_error = self.error_handler.recent_errors[0]
        assert handled_error.details["user_id"] == 123
        assert handled_error.details["request_id"] == "req_456"
        assert handled_error.details["endpoint"] == "/recommendations"


class TestErrorReporter:
    """Tests pour le rapporteur d'erreurs"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.error_reporter = ErrorReporter()
    
    def test_error_reporter_creation(self):
        """Test de création du rapporteur"""
        assert isinstance(self.error_reporter, ErrorReporter)
    
    @patch('app.ml_analytics.exceptions.requests.post')
    def test_send_error_report(self, mock_post):
        """Test d'envoi de rapport d'erreur"""
        mock_post.return_value.status_code = 200
        
        error = MLAnalyticsError("Test error")
        
        success = self.error_reporter.send_error_report(error)
        
        assert success is True
        mock_post.assert_called_once()
    
    @patch('smtplib.SMTP')
    def test_send_email_alert(self, mock_smtp):
        """Test d'envoi d'alerte email"""
        mock_smtp_instance = MagicMock()
        mock_smtp.return_value = mock_smtp_instance
        
        critical_error = MLAnalyticsError(
            "Critical error",
            severity=ErrorSeverity.CRITICAL
        )
        
        self.error_reporter.send_email_alert(critical_error)
        
        mock_smtp.assert_called()
        mock_smtp_instance.send_message.assert_called()
    
    def test_error_aggregation(self):
        """Test d'agrégation d'erreurs"""
        errors = [
            MLAnalyticsError("Error 1"),
            MLAnalyticsError("Error 2"),
            MLAnalyticsError("Error 3")
        ]
        
        for error in errors:
            self.error_reporter.add_error(error)
        
        report = self.error_reporter.generate_aggregated_report()
        
        assert report["total_errors"] == 3
        assert "error_summary" in report
        assert "severity_breakdown" in report
    
    def test_error_filtering(self):
        """Test de filtrage d'erreurs"""
        errors = [
            MLAnalyticsError("Low error", severity=ErrorSeverity.LOW),
            MLAnalyticsError("High error", severity=ErrorSeverity.HIGH),
            MLAnalyticsError("Critical error", severity=ErrorSeverity.CRITICAL)
        ]
        
        for error in errors:
            self.error_reporter.add_error(error)
        
        # Filtrer seulement les erreurs critiques
        critical_errors = self.error_reporter.get_errors_by_severity(
            ErrorSeverity.CRITICAL
        )
        
        assert len(critical_errors) == 1
        assert critical_errors[0].severity == ErrorSeverity.CRITICAL


class TestExceptionLogger:
    """Tests pour le logger d'exceptions"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.exception_logger = ExceptionLogger()
    
    @patch('app.ml_analytics.exceptions.logging.getLogger')
    def test_log_exception(self, mock_get_logger):
        """Test de logging d'exception"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        error = MLAnalyticsError("Test error")
        
        self.exception_logger.log_exception(error)
        
        mock_logger.error.assert_called()
    
    def test_structured_logging(self):
        """Test de logging structuré"""
        error = MLAnalyticsError(
            "Structured error",
            details={"component": "ml_engine"}
        )
        
        log_entry = self.exception_logger.create_structured_log_entry(error)
        
        assert "timestamp" in log_entry
        assert "error_code" in log_entry
        assert "severity" in log_entry
        assert "category" in log_entry
        assert "message" in log_entry
        assert "details" in log_entry
    
    def test_log_rotation(self):
        """Test de rotation des logs"""
        # Simuler beaucoup d'erreurs pour déclencher la rotation
        for i in range(1000):
            error = MLAnalyticsError(f"Error {i}")
            self.exception_logger.log_exception(error)
        
        # Vérifier que la rotation a eu lieu
        assert self.exception_logger.log_rotation_triggered
    
    @pytest.mark.asyncio
    async def test_async_logging(self):
        """Test de logging asynchrone"""
        error = MLAnalyticsError("Async log test")
        
        await self.exception_logger.log_exception_async(error)
        
        # Vérifier que l'erreur a été loggée
        assert error in self.exception_logger.logged_errors


class TestUtilityFunctions:
    """Tests pour les fonctions utilitaires"""
    
    def test_format_error_message(self):
        """Test de formatage de message d'erreur"""
        error = MLAnalyticsError(
            "Test error",
            details={"user_id": 123}
        )
        
        formatted = format_error_message(error)
        
        assert "Test error" in formatted
        assert "user_id: 123" in formatted
        assert error.error_code in formatted
    
    @pytest.mark.asyncio
    async def test_handle_async_exception(self):
        """Test de gestion d'exception asynchrone"""
        async def failing_operation():
            raise ValueError("Async failure")
        
        with pytest.raises(MLAnalyticsError):
            await handle_async_exception(failing_operation)
    
    def test_create_error_response(self):
        """Test de création de réponse d'erreur"""
        error = MLAnalyticsError("API error")
        
        response = create_error_response(error)
        
        assert response["success"] is False
        assert "error" in response
        assert response["error"]["message"] == "API error"
        assert "error_code" in response["error"]
    
    def test_get_error_context(self):
        """Test de récupération de contexte d'erreur"""
        # Simuler une stack trace
        try:
            1 / 0
        except ZeroDivisionError:
            context = get_error_context()
            
            assert "function_name" in context
            assert "file_name" in context
            assert "line_number" in context
            assert "local_variables" in context


class TestErrorRecovery:
    """Tests pour la récupération d'erreurs"""
    
    def test_retry_mechanism(self):
        """Test de mécanisme de retry"""
        attempt_count = 0
        
        def failing_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise MLAnalyticsError("Temporary failure")
            return "success"
        
        error_handler = ErrorHandler()
        result = error_handler.retry_operation(
            failing_operation,
            max_retries=3,
            delay=0.1
        )
        
        assert result == "success"
        assert attempt_count == 3
    
    def test_circuit_breaker(self):
        """Test de circuit breaker"""
        error_handler = ErrorHandler()
        
        # Déclencher beaucoup d'erreurs pour ouvrir le circuit
        for i in range(10):
            error = MLAnalyticsError("Service failure")
            error_handler.handle_error(error)
        
        # Le circuit devrait être ouvert
        assert error_handler.circuit_breaker.is_open()
        
        # Les nouvelles opérations devraient être rejetées rapidement
        with pytest.raises(CircuitBreakerOpenError):
            error_handler.execute_with_circuit_breaker(
                lambda: "operation"
            )
    
    def test_fallback_mechanism(self):
        """Test de mécanisme de fallback"""
        def primary_operation():
            raise MLAnalyticsError("Primary failure")
        
        def fallback_operation():
            return "fallback_result"
        
        error_handler = ErrorHandler()
        result = error_handler.execute_with_fallback(
            primary_operation,
            fallback_operation
        )
        
        assert result == "fallback_result"


class TestErrorMetrics:
    """Tests pour les métriques d'erreur"""
    
    def test_error_rate_calculation(self):
        """Test de calcul du taux d'erreur"""
        error_handler = ErrorHandler()
        
        # Simuler des opérations réussies et échouées
        for i in range(80):
            error_handler.record_success()
        
        for i in range(20):
            error = MLAnalyticsError(f"Error {i}")
            error_handler.handle_error(error)
        
        error_rate = error_handler.get_error_rate()
        assert error_rate == 0.2  # 20%
    
    def test_error_frequency_analysis(self):
        """Test d'analyse de fréquence d'erreurs"""
        error_handler = ErrorHandler()
        
        # Générer différents types d'erreurs
        for i in range(5):
            error_handler.handle_error(ModelError("Model error"))
        
        for i in range(3):
            error_handler.handle_error(DataError("Data error"))
        
        frequency = error_handler.get_error_frequency_by_category()
        
        assert frequency[ErrorCategory.MODEL] == 5
        assert frequency[ErrorCategory.DATA] == 3
    
    def test_error_trend_analysis(self):
        """Test d'analyse de tendance d'erreurs"""
        error_handler = ErrorHandler()
        
        # Simuler des erreurs sur plusieurs périodes
        base_time = datetime.now()
        
        for hour in range(24):
            timestamp = base_time + timedelta(hours=hour)
            error_count = hour % 3 + 1  # Pattern cyclique
            
            for i in range(error_count):
                error = MLAnalyticsError(f"Error at hour {hour}")
                error.timestamp = timestamp
                error_handler.handle_error(error)
        
        trend = error_handler.analyze_error_trend(
            start_time=base_time,
            end_time=base_time + timedelta(hours=24),
            interval_hours=1
        )
        
        assert len(trend) == 24
        assert all(point['error_count'] > 0 for point in trend)


class TestErrorIntegration:
    """Tests d'intégration pour la gestion d'erreurs"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_error_handling(self):
        """Test de gestion d'erreur de bout en bout"""
        # Configuration complète du système d'erreurs
        error_handler = ErrorHandler()
        error_reporter = ErrorReporter()
        exception_logger = ExceptionLogger()
        
        # Simuler une erreur dans un pipeline ML
        try:
            # Opération qui échoue
            raise ValueError("Model training failed")
        except ValueError as e:
            # Créer une erreur ML Analytics
            ml_error = ModelError(
                "Model training pipeline failed",
                model_name="recommendation_v2",
                original_exception=e
            )
            
            # Gestion complète de l'erreur
            error_handler.handle_error(ml_error)
            await exception_logger.log_exception_async(ml_error)
            error_reporter.add_error(ml_error)
            
            # Vérifications
            assert error_handler.error_count == 1
            assert ml_error in exception_logger.logged_errors
            assert len(error_reporter.errors) == 1
    
    @pytest.mark.integration
    def test_error_monitoring_integration(self):
        """Test d'intégration avec le monitoring"""
        with patch('app.ml_analytics.monitoring.HealthMonitor') as mock_monitor:
            mock_monitor_instance = MagicMock()
            mock_monitor.return_value = mock_monitor_instance
            
            error_handler = ErrorHandler()
            error_handler.setup_monitoring_integration(mock_monitor_instance)
            
            # Générer une erreur critique
            critical_error = MLAnalyticsError(
                "System failure",
                severity=ErrorSeverity.CRITICAL
            )
            
            error_handler.handle_error(critical_error)
            
            # Vérifier que le monitoring a été notifié
            mock_monitor_instance.record_error.assert_called_with(critical_error)


# Fixtures pour les tests
@pytest.fixture
def sample_error():
    """Erreur de test"""
    return MLAnalyticsError(
        "Sample error for testing",
        severity=ErrorSeverity.MEDIUM,
        category=ErrorCategory.MODEL,
        details={"test": True}
    )


@pytest.fixture
def error_handler():
    """Gestionnaire d'erreurs de test"""
    return ErrorHandler()


@pytest.fixture
def error_reporter():
    """Rapporteur d'erreurs de test"""
    return ErrorReporter()


@pytest.fixture
def exception_logger():
    """Logger d'exceptions de test"""
    return ExceptionLogger()


# Tests de performance
@pytest.mark.performance
class TestErrorPerformance:
    """Tests de performance pour la gestion d'erreurs"""
    
    def test_error_handling_performance(self):
        """Test de performance de gestion d'erreurs"""
        import time
        
        error_handler = ErrorHandler()
        
        start_time = time.time()
        
        # Gérer beaucoup d'erreurs rapidement
        for i in range(1000):
            error = MLAnalyticsError(f"Performance test error {i}")
            error_handler.handle_error(error)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Devrait être très rapide
        assert duration < 1.0
        assert error_handler.error_count == 1000
    
    def test_error_logging_performance(self):
        """Test de performance de logging d'erreurs"""
        import time
        
        exception_logger = ExceptionLogger()
        
        start_time = time.time()
        
        # Logger beaucoup d'erreurs
        for i in range(500):
            error = MLAnalyticsError(f"Log performance test {i}")
            exception_logger.log_exception(error)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Logging devrait être efficace
        assert duration < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
