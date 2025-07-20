from unittest.mock import Mock
"""
Tests avancés pour monitoring.py (métriques Prometheus, compteurs, latence).
"""


import sys
import os
import sys
# Ajout du dossier racine backend au PYTHONPATH pour import absolu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..', 'backend')))
from backend.services.spleeter_microservice import monitoring

def test_monitoring_metrics():
    assert hasattr(monitoring, 'REQUEST_COUNT')
    assert hasattr(monitoring, 'REQUEST_LATENCY')
    assert hasattr(monitoring, 'ERROR_COUNT')
    assert hasattr(monitoring, 'FILES_PROCESSED')
    assert hasattr(monitoring, 'CURRENT_JOBS')
    metrics = monitoring.get_metrics()
    assert isinstance(metrics, dict)
