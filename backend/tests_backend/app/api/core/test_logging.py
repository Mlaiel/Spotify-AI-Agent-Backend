"""
🎵 Spotify AI Agent - Tests du Système de Logging
================================================

Tests enterprise pour le système de logging avec validation complète
des formatters, handlers, et fonctionnalités avancées.

Architecture de tests:
- Tests de configuration des loggers
- Validation des formatters structurés
- Tests des handlers multiples
- Vérification du logging contextuel
- Tests de performance
- Validation des rotations de fichiers

Développé par Fahed Mlaiel - Enterprise Logging Test Expert
"""

import json
import logging
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from io import StringIO
from datetime import datetime

import pytest

from app.api.core.logging import (
    StructuredFormatter,
    get_logger,
    log_api_request,
    log_error,
    default_logger
)


class TestStructuredFormatter:
    """Tests pour le formatter structuré JSON"""
    
    def test_basic_formatting(self):
        """Test du formatage basique"""
        formatter = StructuredFormatter()
        
        # Créer un record de log
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        
        with patch('app.api.core.logging.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value.isoformat.return_value = "2025-07-14T22:00:00"
            formatted = formatter.format(record)
            
        # Parser le JSON
        log_data = json.loads(formatted)
        
        # Vérifications
        assert log_data["timestamp"] == "2025-07-14T22:00:00"
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test_logger"
        assert log_data["message"] == "Test message"
        assert log_data["module"] == "test_module"
        assert log_data["function"] == "test_function"
        assert log_data["line"] == 42
    
    def test_formatting_with_context(self):
        """Test du formatage avec contexte personnalisé"""
        formatter = StructuredFormatter()
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="/test/path.py",
            lineno=100,
            msg="Error occurred",
            args=(),
            exc_info=None
        )
        record.module = "error_module"
        record.funcName = "error_function"
        record.request_id = "req-123"
        record.user_id = "user-456"
        record.duration = 1.234
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        # Vérifier les champs personnalisés
        assert log_data["request_id"] == "req-123"
        assert log_data["user_id"] == "user-456"
        assert log_data["duration"] == 1.234
    
    def test_formatting_with_exception(self):
        """Test du formatage avec exception"""
        formatter = StructuredFormatter()
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
            
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="/test/path.py",
                lineno=120,
                msg="Exception occurred",
                args=(),
                exc_info=exc_info  # Passer le tuple complet
            )
            record.module = "exception_module"
            record.funcName = "exception_function"
            
            formatted = formatter.format(record)
            log_data = json.loads(formatted)
            
            # Vérifier que l'exception est incluse
            assert "exception" in log_data
            assert "ValueError: Test exception" in log_data["exception"]
            assert "Traceback" in log_data["exception"]


class TestGetLogger:
    """Tests pour la fonction get_logger"""
    
    def test_default_logger_creation(self):
        """Test de création du logger par défaut"""
        logger = get_logger()
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "spotify_ai_agent"
        assert len(logger.handlers) > 0
    
    def test_named_logger_creation(self):
        """Test de création d'un logger nommé"""
        logger = get_logger("custom_logger")
        
        assert logger.name == "custom_logger"
        assert len(logger.handlers) > 0
    
    def test_logger_caching(self):
        """Test de mise en cache des loggers"""
        logger1 = get_logger("cached_logger")
        logger2 = get_logger("cached_logger")
        
        # Même instance grâce au cache
        assert logger1 is logger2
    
    def test_logger_level_configuration(self):
        """Test de configuration du niveau de log"""
        with patch('app.api.core.logging.get_settings') as mock_settings:
            mock_monitoring = MagicMock()
            mock_monitoring.log_level.value = "DEBUG"
            # Pas de fichier de log pour ce test
            mock_monitoring.log_file = None
            mock_settings.return_value.monitoring = mock_monitoring
            
            # Vider le cache pour forcer la recréation
            get_logger.cache_clear()
            
            logger = get_logger("debug_logger")
            assert logger.level == logging.DEBUG
    
    def test_console_handler_configuration(self):
        """Test de configuration du handler console"""
        logger = get_logger("console_test")
        
        # Vérifier qu'il y a au moins un handler
        assert len(logger.handlers) >= 1
        
        # Vérifier qu'au moins un handler utilise StructuredFormatter
        has_structured_formatter = any(
            isinstance(handler.formatter, StructuredFormatter)
            for handler in logger.handlers
        )
        assert has_structured_formatter
    
    def test_file_handler_configuration(self):
        """Test de configuration du handler fichier"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            with patch('app.api.core.logging.get_settings') as mock_settings:
                mock_monitoring = MagicMock()
                mock_monitoring.log_level.value = "INFO"
                mock_monitoring.log_file = str(log_file)
                mock_settings.return_value.monitoring = mock_monitoring
                
                # Vider le cache pour forcer la recréation
                get_logger.cache_clear()
                
                logger = get_logger("file_test")
                
                # Tester l'écriture dans le fichier
                logger.info("Test file logging")
                
                # Vérifier que le fichier existe et contient des données
                assert log_file.exists()
                content = log_file.read_text()
                assert "Test file logging" in content


class TestLogApiRequest:
    """Tests pour la fonction log_api_request"""
    
    def test_basic_api_request_logging(self, caplog):
        """Test du logging basique d'une requête API"""
        logger = get_logger("api_test")
        
        with caplog.at_level(logging.INFO):
            log_api_request(
                logger=logger,
                method="GET",
                path="/api/test",
                status_code=200,
                duration=0.123
            )
        
        # Vérifier que le log a été créé
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert "GET /api/test - 200 (0.123s)" in record.message
        assert hasattr(record, 'duration')
        assert record.duration == 0.123
    
    def test_api_request_logging_with_request_id(self, caplog):
        """Test du logging avec request_id"""
        logger = get_logger("api_request_id_test")
        
        with caplog.at_level(logging.INFO):
            log_api_request(
                logger=logger,
                method="POST",
                path="/api/users",
                status_code=201,
                duration=0.456,
                request_id="req-789"
            )
        
        record = caplog.records[0]
        assert hasattr(record, 'request_id')
        assert record.request_id == "req-789"
    
    def test_different_http_methods(self, caplog):
        """Test du logging pour différentes méthodes HTTP"""
        logger = get_logger("http_methods_test")
        methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]
        
        with caplog.at_level(logging.INFO):
            for method in methods:
                log_api_request(
                    logger=logger,
                    method=method,
                    path=f"/api/{method.lower()}",
                    status_code=200,
                    duration=0.1
                )
        
        assert len(caplog.records) == len(methods)
        for i, method in enumerate(methods):
            assert method in caplog.records[i].message


class TestLogError:
    """Tests pour la fonction log_error"""
    
    def test_basic_error_logging(self, caplog):
        """Test du logging basique d'erreur"""
        logger = get_logger("error_test")
        error = ValueError("Test error message")
        
        with caplog.at_level(logging.ERROR):
            log_error(logger, error)
        
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert "Error: Test error message" in record.message
        assert record.exc_info is not None
    
    def test_error_logging_with_context(self, caplog):
        """Test du logging d'erreur avec contexte"""
        logger = get_logger("error_context_test")
        error = RuntimeError("Runtime error")
        context = {
            "user_id": "user-123",
            "action": "data_processing",
            "timestamp": "2025-07-14T22:00:00"
        }
        
        with caplog.at_level(logging.ERROR):
            log_error(logger, error, context)
        
        record = caplog.records[0]
        assert hasattr(record, 'user_id')
        assert hasattr(record, 'action')
        assert hasattr(record, 'timestamp')
        assert record.user_id == "user-123"
        assert record.action == "data_processing"
    
    def test_different_exception_types(self, caplog):
        """Test du logging pour différents types d'exceptions"""
        logger = get_logger("exception_types_test")
        exceptions = [
            ValueError("Value error"),
            TypeError("Type error"),
            RuntimeError("Runtime error"),
            KeyError("Key error")
        ]
        
        with caplog.at_level(logging.ERROR):
            for exc in exceptions:
                log_error(logger, exc)
        
        assert len(caplog.records) == len(exceptions)
        for i, exc in enumerate(exceptions):
            assert str(exc) in caplog.records[i].message


class TestDefaultLogger:
    """Tests pour le logger par défaut"""
    
    def test_default_logger_exists(self):
        """Test que le logger par défaut existe"""
        assert default_logger is not None
        assert isinstance(default_logger, logging.Logger)
    
    def test_default_logger_functionality(self, caplog):
        """Test de fonctionnalité du logger par défaut"""
        with caplog.at_level(logging.INFO):
            default_logger.info("Test default logger")
        
        assert len(caplog.records) == 1
        assert "Test default logger" in caplog.records[0].message


class TestLoggingIntegration:
    """Tests d'intégration du système de logging"""
    
    def test_structured_logging_flow(self, caplog):
        """Test du flux complet de logging structuré"""
        logger = get_logger("integration_test")
        
        with caplog.at_level(logging.INFO):
            # Simuler une requête API complète
            log_api_request(
                logger=logger,
                method="GET",
                path="/api/integration/test",
                status_code=200,
                duration=0.789,
                request_id="req-integration-123"
            )
            
            # Simuler une erreur
            try:
                raise ValueError("Integration test error")
            except ValueError as e:
                log_error(logger, e, {
                    "request_id": "req-integration-123",
                    "endpoint": "/api/integration/test"
                })
        
        # Vérifier les deux logs
        assert len(caplog.records) == 2
        
        # Vérifier le log de requête
        api_record = caplog.records[0]
        assert "GET /api/integration/test - 200" in api_record.message
        assert hasattr(api_record, 'request_id')
        
        # Vérifier le log d'erreur
        error_record = caplog.records[1]
        assert "Integration test error" in error_record.message
        assert hasattr(error_record, 'request_id')
        assert error_record.request_id == api_record.request_id
    
    def test_performance_logging(self, caplog):
        """Test du logging de performance"""
        logger = get_logger("performance_test")
        
        # Simuler différentes durées de requête
        durations = [0.001, 0.050, 0.200, 1.000, 5.000]
        
        with caplog.at_level(logging.INFO):
            for i, duration in enumerate(durations):
                log_api_request(
                    logger=logger,
                    method="GET",
                    path=f"/api/performance/{i}",
                    status_code=200,
                    duration=duration,
                    request_id=f"perf-{i}"
                )
        
        # Vérifier que toutes les durées sont loggées
        for i, duration in enumerate(durations):
            record = caplog.records[i]
            assert hasattr(record, 'duration')
            assert record.duration == duration
            assert f"({duration:.3f}s)" in record.message


class TestLoggingErrorHandling:
    """Tests de gestion d'erreurs du système de logging"""
    
    def test_logging_with_none_values(self, caplog):
        """Test du logging avec des valeurs None"""
        logger = get_logger("none_test")
        
        with caplog.at_level(logging.INFO):
            log_api_request(
                logger=logger,
                method="GET",
                path="/api/test",
                status_code=200,
                duration=0.123,
                request_id=None  # None value
            )
        
        # Doit fonctionner sans erreur
        assert len(caplog.records) == 1
    
    def test_logging_with_invalid_data(self, caplog):
        """Test du logging avec des données invalides"""
        logger = get_logger("invalid_test")
        
        # Ne doit pas lever d'exception même avec des données étranges
        with caplog.at_level(logging.ERROR):
            try:
                log_error(logger, ValueError("Test"), {"invalid": object()})
            except Exception:
                # Si une exception est levée, c'est un problème
                pytest.fail("Logging should not raise exceptions")
        
        # Au moins un log doit être créé
        assert len(caplog.records) >= 1


if __name__ == "__main__":
    pytest.main([__file__])
