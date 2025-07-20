from unittest.mock import Mock
"""
Tests avancés pour health.py (endpoint /health, intégration FastAPI).
"""

import sys
import os
import pytest
from fastapi import status
from fastapi.testclient import TestClient
# Ajout du dossier racine backend au PYTHONPATH pour import absolu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..', 'backend')))
from backend.services.spleeter_microservice.health import router

def test_health_route():
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "ok"
