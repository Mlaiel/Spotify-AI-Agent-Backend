from unittest.mock import Mock
"""
Tests avancés pour security.py (API key, extension JWT/OAuth2 ready).
"""


import sys
import os
import pytest
from fastapi import Request
from starlette.datastructures import Headers
# Ajout du dossier racine backend au PYTHONPATH pour import absolu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..', 'backend')))
from backend.services.spleeter_microservice import security

class DummyRequest:
    def __init__(self, api_key):
        self.headers = Headers({"X-API-KEY": api_key})
        self.client = type('client', (), {'host': '127.0.0.1'})()

@pytest.mark.asyncio
async def test_verify_api_key_valid():
    req = DummyRequest("changeme")
    await security.verify_api_key(req)

@pytest.mark.asyncio
async def test_verify_api_key_invalid():
    req = DummyRequest("wrong")
    with pytest.raises(Exception):
        await security.verify_api_key(req)

class DummyRequest:
    def __init__(self, api_key):
        self.headers = Headers({"X-API-KEY": api_key})
        self.client = type('client', (), {'host': '127.0.0.1'})()

@pytest.mark.asyncio
async def test_verify_api_key_valid():
    req = DummyRequest("changeme")
    # Par défaut, la clé API est changeme
    await security.verify_api_key(req)

@pytest.mark.asyncio
async def test_verify_api_key_invalid():
    req = DummyRequest("wrong")
    with pytest.raises(Exception):
        await security.verify_api_key(req)
