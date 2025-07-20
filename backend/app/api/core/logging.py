"""
🎵 Spotify AI Agent - API Logging System
========================================

Système de logging enterprise avec configuration avancée,
formatage structuré et intégration avec monitoring.

Architecture:
- Logging structuré avec JSON
- Formatters personnalisés
- Handlers multiples (file, console, remote)
- Context-aware logging
- Performance tracking
- Error tracking

Développé par Fahed Mlaiel - Enterprise Logging Expert
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from functools import lru_cache

from .config import get_settings


class StructuredFormatter(logging.Formatter):
    """Formatter pour logs structurés en JSON"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Ajouter des champs personnalisés si présents
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'duration'):
            log_entry['duration'] = record.duration
            
        # Ajouter exception si présente
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry, ensure_ascii=False)


@lru_cache()
def get_logger(name: str = None) -> logging.Logger:
    """Obtenir un logger configuré"""
    if name is None:
        name = "spotify_ai_agent"
    
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        settings = get_settings()
        
        # Configuration du niveau depuis monitoring config
        log_level = "INFO"  # Valeur par défaut
        if hasattr(settings, 'monitoring') and hasattr(settings.monitoring, 'log_level'):
            log_level = settings.monitoring.log_level.value
        
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Handler console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(StructuredFormatter())
        logger.addHandler(console_handler)
        
        # Handler fichier si configuré
        if (hasattr(settings, 'monitoring') and 
            hasattr(settings.monitoring, 'log_file') and 
            settings.monitoring.log_file):
            file_handler = logging.handlers.RotatingFileHandler(
                settings.monitoring.log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(StructuredFormatter())
            logger.addHandler(file_handler)
    
    return logger


def log_api_request(logger: logging.Logger, method: str, path: str, 
                   status_code: int, duration: float, request_id: str = None):
    """Logger une requête API"""
    extra = {
        'request_id': request_id,
        'duration': duration
    }
    
    logger.info(
        f"{method} {path} - {status_code} ({duration:.3f}s)",
        extra=extra
    )


def log_error(logger: logging.Logger, error: Exception, context: Dict[str, Any] = None):
    """Logger une erreur avec contexte"""
    extra = context or {}
    logger.error(f"Error: {str(error)}", exc_info=True, extra=extra)


# Logger par défaut
default_logger = get_logger()


__all__ = [
    "StructuredFormatter",
    "get_logger", 
    "log_api_request",
    "log_error",
    "default_logger"
]
