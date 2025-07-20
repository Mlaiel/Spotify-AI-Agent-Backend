"""
🎵 Spotify AI Agent - Spleeter Module Enterprise
==============================================

Module de séparation audio avancé utilisant l'IA pour séparer
les sources audio (voix, instruments, etc.) avec performance industrielle.

Features:
- Séparation multi-sources (2, 4, 5 stems)
- Support GPU/CPU optimisé
- Cache intelligent
- Processing batch
- Monitoring avancé
- Format audio multiple

🎖️ Développé par l'équipe d'experts enterprise
"""

from .core import SpleeterEngine, SpleeterConfig
from .models import ModelManager, PretrainedModels
from .processor import AudioProcessor, BatchProcessor
from .cache import CacheManager
from .utils import AudioUtils, ValidationUtils
from .monitoring import PerformanceMonitor, MetricsCollector
from .exceptions import SpleeterError, ModelNotFoundError, AudioProcessingError

__version__ = "2.4.0-enterprise"
__author__ = "Enterprise Expert Team"

# Configuration par défaut
DEFAULT_CONFIG = SpleeterConfig(
    model_name="spleeter:2stems-16kHz",
    audio_adapter="tensorflow",
    sample_rate=44100,
    frame_length=4096,
    frame_step=1024,
    enable_gpu=True,
    cache_enabled=True,
    batch_size=8,
    max_duration=600  # 10 minutes max
)

# Export des classes principales
__all__ = [
    'SpleeterEngine',
    'SpleeterConfig', 
    'ModelManager',
    'PretrainedModels',
    'AudioProcessor',
    'BatchProcessor',
    'CacheManager',
    'AudioUtils',
    'ValidationUtils',
    'PerformanceMonitor',
    'MetricsCollector',
    'SpleeterError',
    'ModelNotFoundError',
    'AudioProcessingError',
    'DEFAULT_CONFIG'
]
