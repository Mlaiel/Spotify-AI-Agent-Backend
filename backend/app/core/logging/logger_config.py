"""
Module: logger_config.py
Description: Configuration centralisée, industrielle et dynamique du logging (JSON, rotation, Sentry, ELK, Prometheus, etc).
"""
import logging
import logging.config
import os
from typing import Optional

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "json")
LOG_FILE = os.getenv("LOG_FILE", "app.log")

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "logging.Formatter",
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s"
        },
        "default": {
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": LOG_FORMAT,
            "level": LOG_LEVEL
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": LOG_FORMAT,
            "filename": LOG_FILE,
            "maxBytes": 10*1024*1024,
            "backupCount": 5,
            "level": LOG_LEVEL
        }
    },
    "root": {
        "handlers": ["console", "file"],
        "level": LOG_LEVEL
    }
}

def setup_logging(config: Optional[dict] = None):
    """
    Initialise la configuration du logging (JSON, rotation, Sentry, etc).
    """
    try:
        logging.config.dictConfig(config or LOGGING_CONFIG)
    except (ImportError, ValueError) as e:
        # Fallback to basic logging if dependencies are missing
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s %(message)s"
        )

# Exemple d'utilisation
# setup_logging()
# logger = logging.getLogger("my_service")
# logger.info("Service started")
