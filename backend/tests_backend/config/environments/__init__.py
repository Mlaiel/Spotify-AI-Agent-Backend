from unittest.mock import Mock

"""
Ultra-advanced helpers and fixtures for environment configuration testing in the Spotify AI Agent project.

These utilities are designed for industrial, production-grade validation of all environment files, supporting CI/CD, security, and compliance.
"""

import os
import pytest
from pathlib import Path
import re

def load_env_file(env_path):
    """Load environment variables from a .env file into a dictionary."""
    env = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                k, v = line.split('=', 1)
                env[k.strip()] = v.strip()
    return env

def validate_env_vars(env, required_vars):
    missing = [k for k in required_vars if k not in env]
    return missing

def check_no_weak_secrets(env):
    for k, v in env.items():
        assert "changeme" not in v.lower(), f"Weak secret in {k}"
        assert "password" not in v.lower() or k == "POSTGRES_PASSWORD", f"Plain password in {k}"

def check_format(env, keys):
    for k in keys:
        assert re.match(r"^[a-zA-Z0-9\-_]+$", env[k]), f"Invalid format for {k}"

@pytest.fixture(scope="module")
def env_files():
    """Return a dict of all environment files in the current directory."""
    env_dir = Path(__file__).parent
    files = {f.name: load_env_file(f) for f in env_dir.glob('.env.*')}
    return files
