from unittest.mock import Mock
"""
Ultra-advanced helpers and fixtures for logging configuration testing in the Spotify AI Agent project.

These utilities are designed for industrial, production-grade validation of all logging configuration files, supporting CI/CD, security, and compliance.
"""

import os
import pytest
from pathlib import Path
import yaml

def load_logging_config(config_path):
    """Load a YAML logging config file into a dictionary."""
    with open(config_path) as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="module")
def logging_configs():
    """Return a dict of all logging config files in the current directory."""
    config_dir = Path(__file__).parent
    files = {f.name: load_logging_config(f) for f in config_dir.glob('logging.*.conf')}
    return files
