"""
Module: er        s    def capture_message(self, message: str, level: str = "error", context: Optional[Dict[str, Any]] = None):
        self.logger.log(getattr(logging, level.upper(), logging.ERROR), f"[ERROR] {message}", extra={"context": context or {}})f.logger.error(f"[ERROR] {exc}", extra={"traceback": tb, "context": context or {}})
        # Intégration Sentry/ELK possible ici
        # sentry_sdk.capture_exception(exc)

    def capture_message(self, message: str, level: str = "error", context: Optional[Dict[str, Any]] = None):
        self.logger.log(getattr(logging, level.upper(), logging.ERROR), f"[ERROR] {message}", extra={"context": context or {}})acker.py
Description: Tracker d'erreurs industriel, compatible Sentry, ELK, alerting, avec enrichissement contextuel et support microservices.
"""
import logging
import traceback
from typing import Optional, Dict, Any

class ErrorTracker:
    def __init__(self, name: str = "error_tracker"):
        self.logger = logging.getLogger(name)

    def capture_exception(self, exc: Exception, context: Optional[Dict[str, Any]] = None):
        tb = traceback.format_exc()
        self.logger.error(f"[ERROR] {exc}", extra={"traceback": tb, "context": context or {}})
        # Intégration Sentry/ELK possible ici
        # sentry_sdk.capture_exception(exc)

    def capture_message(self, message: str, level: str = "error", context: Optional[Dict[str, Any]] = None):
        self.logger.log(getattr(logging, level.upper(), logging.ERROR), f"[ERROR] {message}", extra={"context": context or {}})
        # Intégration Sentry/ELK possible ici
        # sentry_sdk.capture_message(message, level=level)

# Exemple d'utilisation
# error_tracker = ErrorTracker()
# try:
#     1/0
# except Exception as e:
#     error_tracker.capture_exception(e, context={"service": "ai"})
