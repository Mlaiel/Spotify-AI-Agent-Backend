"""
Module: audit_logger.py
Description: Logger d'audit industriel pour la traçabilité, la conformité RGPD/SOX, et la sécurité (actions IA, accès, modifications critiques).
"""
import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any

class AuditLogger:
    def __init__(self, name: str = "audit_logger"):
        self.logger = logging.getLogger(name)

    def log_event(self, user_id: str, action: str, resource: str, status: str = "success", details: Optional[Dict[str, Any]] = None):
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "status": status,
            "details": details or {}
        }
        self.logger.info(json.dumps(event))

# Exemple d'utilisation
# audit_logger = AuditLogger()
# audit_logger.log_event(user_id="42", action="delete", resource="ai_content", status="success", details={"content_id": 99})
