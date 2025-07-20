from .logger_config import *
from .structured_logger import *
from .performance_logger import *
from .audit_logger import *
from .error_tracker import *
from .log_aggregator import *
from .async_logger import *

import logging
from .logger_config import setup_logging

def get_logger(name: str = None):
    """
    Fournit un logger structuré, prêt pour la production (JSON, rotation, Sentry, etc).
    Initialise la configuration si nécessaire.
    """
    setup_logging()
    return logging.getLogger(name)