"""
Module: audit_logger.py
Description: Logger d'audit sécurité, traçabilité RGPD/SOX, actions critiques, accès, modifications, alertes compliance.
"""
import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any

class SecurityAuditLogger:
    def __init__(self, name: str = "security_audit_logger"):
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
# audit_logger = SecurityAuditLogger()
# audit_logger.log_event(user_id="42", action="delete", resource="user", status="success", details={"ip": "1.2.3.4"})
