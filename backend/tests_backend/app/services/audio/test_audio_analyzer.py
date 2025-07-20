# Mock automatique pour boto3
try:
    import boto3
except ImportError:
    import sys
    from unittest.mock import Mock
    sys.modules['boto3'] = Mock()
    if 'boto3' == 'opentelemetry':
        sys.modules['opentelemetry.exporter'] = Mock()
        sys.modules['opentelemetry.instrumentation'] = Mock()
    elif 'boto3' == 'grpc':
        sys.modules['grpc_tools'] = Mock()

# Mock automatique pour redis
try:
    import redis
except ImportError:
    import sys
    from unittest.mock import Mock
    sys.modules['redis'] = Mock()
    if 'redis' == 'opentelemetry':
        sys.modules['opentelemetry.exporter'] = Mock()
        sys.modules['opentelemetry.instrumentation'] = Mock()
    elif 'redis' == 'grpc':
        sys.modules['grpc_tools'] = Mock()

from unittest.mock import Mock
"""
Tests avancés pour audio_analyzer.py (analyse ML, classification, tagging).
"""


import numpy as np
import pytest
from backend.app.services.audio import audio_analyzer

def test_analyze_basic():
    arr = np.random.randn(44100)
    result = audio_analyzer.analyze_audio(arr, 44100)
    assert isinstance(result, dict)
    assert "tags" in result
    assert "genre" in result
