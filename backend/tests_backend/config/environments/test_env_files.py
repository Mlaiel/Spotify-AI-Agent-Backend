from unittest.mock import Mock
# -*- coding: utf-8 -*-
"""
Test industriel avancé pour la validation des fichiers d'environnement du projet Spotify AI Agent.
"""
import os
import re
from pathlib import Path

REQUIRED_VARS = [
    "DATABASE_URL", "REDIS_URL", "MONGODB_URI", "SECRET_KEY", "ENV", "LOG_LEVEL", "SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET"
]

ENV_DIR = Path(__file__).parent

def parse_env_file(env_path):
    env_vars = {}
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            key_value = re.split(r'=(.*)', line, maxsplit=1)
            if len(key_value) >= 2:
                key, value = key_value[0].strip(), key_value[1].strip()
                env_vars[key] = value
    return env_vars

def test_env_files_exist():
    """Vérifie que tous les fichiers d'environnement existent."""
    files = list(ENV_DIR.glob(".env.*"))
    assert files, "Aucun fichier d'environnement trouvé."

def test_required_vars_present():
    """Vérifie la présence des variables requises dans chaque fichier d'environnement."""
    for env_file in ENV_DIR.glob(".env.*"):
        env_vars = parse_env_file(env_file)
        missing = [var for var in REQUIRED_VARS if var not in env_vars]
        assert not missing, f"Variables manquantes dans {env_file.name}: {missing}"

def test_no_plain_secrets():
    """Vérifie qu'aucun secret n'est en clair dans les fichiers d'environnement."""
    for env_file in ENV_DIR.glob(".env.*"):
        with open(env_file, "r") as f:
            content = f.read()
            assert "changeme" not in content.lower(), f"Secret faible détecté dans {env_file.name}"
            assert "password" not in content.lower(), f"Mot de passe en clair détecté dans {env_file.name}"

def test_permissions():
    """Vérifie que les permissions des fichiers d'environnement sont sécurisées (lecture seule pour le propriétaire)."""
    for env_file in ENV_DIR.glob(".env.*"):
        st_mode = os.stat(env_file).st_mode
        assert oct(st_mode)[-3:] in ("600", "400"), f"Permissions non sécurisées sur {env_file.name} : {oct(st_mode)[-3:]}"
