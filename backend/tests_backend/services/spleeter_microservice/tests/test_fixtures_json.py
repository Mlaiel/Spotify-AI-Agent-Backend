from unittest.mock import Mock
"""
Tests avancés pour fixtures.json (présence, structure, licences).
"""


import sys
import os
import pytest
# Ajout du dossier racine backend au PYTHONPATH pour import absolu
import json

# Utilisation directe du fichier fixtures.json du dossier source

def test_fixtures_json_exists():
    path = os.path.join(os.path.dirname(__file__), "../../../../../../services/spleeter_microservice/tests/fixtures.json")
    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    for k, v in data.items():
        assert "type" in v
        assert "license" in v
