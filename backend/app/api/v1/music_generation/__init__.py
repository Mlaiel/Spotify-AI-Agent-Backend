"""
Module de génération musicale IA pour artistes Spotify.
Expose les API : génération, remix, mastering, stems, effets, analyse harmonique.
"""

from .audio_synthesis import AudioSynthesizer
from .beat_generator import BeatGenerator
from .audio_effects import AudioEffects
from .mastering_ai import MasteringAI
from .stem_separation import StemSeparator
from .harmony_analyzer import HarmonyAnalyzer

__all__ = [
    "AudioSynthesizer",
    "BeatGenerator",
    "AudioEffects",
    "MasteringAI",
    "StemSeparator",
    "HarmonyAnalyzer"
]
