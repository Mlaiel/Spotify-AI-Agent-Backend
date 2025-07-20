"""
Module: structured_logger.py
Description: Logger structuré industriel (JSON, context, trace, correlation ID), prêt pour ELK, Datadog, Sentry, etc.
"""
import logging
import uuid
import json
from typing import Optional, Dict, Any

class StructuredLogger:
    def __init__(self, name: str = "structured_logger"):
        self.logger = logging.getLogger(name)

    def log(self, level: str, message: str, context: Optional[Dict[str, Any]] = None, **kwargs):
        log_entry = {
            "message": message,
            "context": context or {},
            "correlation_id": kwargs.get("correlation_id", str(uuid.uuid4())),
            "trace_id": kwargs.get("trace_id"),
            "extra": kwargs
        }
        self.logger.log(getattr(logging, level.upper(), logging.INFO), json.dumps(log_entry))

    def info(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs):
        self.log("info", message, context, **kwargs)

    def error(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs):
        self.log("error", message, context, **kwargs)

    def warning(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs):
        self.log("warning", message, context, **kwargs)

    def debug(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs):
        self.log("debug", message, context, **kwargs)

# Exemple d'utilisation
# logger = StructuredLogger()
# logger.info("User login", context={"user_id": 123}, correlation_id="abc-123")
