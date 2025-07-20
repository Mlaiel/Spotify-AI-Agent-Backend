import logging
import sys
from logging.handlers import RotatingFileHandler

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_FILE = "websocket_app.log"


def setup_logging():
    """
    Configure un logging centralisé, rotation des logs, format enrichi, compatible ELK/Loki.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler avec rotation
    fh = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# Exemple d'utilisation :
# from monitoring.logging_config import setup_logging
# setup_logging()
