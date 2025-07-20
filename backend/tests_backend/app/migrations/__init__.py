from unittest.mock import Mock

import logging
import pytest

def migrations_root_logger():
    logger = logging.getLogger("migrations.root.test")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s][MIGRATIONS-ROOT-TEST] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
