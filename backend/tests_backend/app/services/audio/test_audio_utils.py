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
Tests avancés pour audio_utils.py (normalisation, conversion, extraction de features).
"""


import numpy as np
import soundfile as sf
import pytest
from backend.app.services.audio import audio_utils

def test_normalize_audio():
    # Génère un signal simple
    arr = np.random.randn(44100)
    norm = audio_utils.normalize_audio(arr)
    assert np.isclose(np.max(np.abs(norm)), 1.0, atol=1e-2)

def test_convert_to_wav(tmp_path):
    # Crée un fichier temporaire
    arr = np.random.randn(44100)
    path = tmp_path / "test.wav"
    sf.write(str(path), arr, 44100)
    out = audio_utils.convert_to_wav(str(path), str(tmp_path/"out.wav"))
    assert os.path.exists(out)
    assert out.endswith(".wav")

def test_extract_features():
    arr = np.random.randn(44100)
    features = audio_utils.extract_features(arr, 44100)
    assert isinstance(features, dict)
    assert "rms" in features
