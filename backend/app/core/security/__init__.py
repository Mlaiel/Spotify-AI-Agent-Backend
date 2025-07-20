from .api_key_manager import *
from .compliance_checker import *
from .encryption import *
from .password_manager import *
from .threat_detection import *
from .token_manager import *
from .jwt_manager import *
from .security_utils import SecurityUtils

import functools
import logging
import traceback
from .audit_logger import SecurityAuditLogger

# S'assurer que __all__ existe avant de l'étendre
try:
    __all__
except NameError:
    __all__ = []

__all__ += ["audit_log", "secure_task"]

def audit_log(action: str = None):
    """
    Décorateur industriel pour journaliser chaque appel de fonction/tâche avec contexte sécurité, RGPD, SOX, etc.
    - Trace l'utilisateur, l'action, la ressource, le statut, les détails, les erreurs.
    - Utilise SecurityAuditLogger (stockage SIEM-ready, extensible).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = SecurityAuditLogger()
            user_id = kwargs.get("user_id") or "system"
            resource = func.__module__ + "." + func.__name__
            try:
                result = func(*args, **kwargs)
                logger.log_event(user_id=user_id, action=action or func.__name__, resource=resource, status="success")
                return result
            except Exception as exc:
                logger.log_event(user_id=user_id, action=action or func.__name__, resource=resource, status="error", details={"error": str(exc), "trace": traceback.format_exc()})
                raise
        return wrapper
    return decorator

def secure_task(func):
    """
    Décorateur industriel pour tâches sécurisées :
    - Gestion exceptions, alertes sécurité, conformité, audit, rollback, alertes SIEM.
    - Peut être enrichi (RBAC, SSO, monitoring, etc).
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("secure_task")
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            logger.error(f"[SECURE_TASK] Exception: {exc}", exc_info=True)
            # Ici, on pourrait ajouter rollback, alertes, audit, etc.
            raise
    return wrapper