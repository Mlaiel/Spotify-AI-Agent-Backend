"""
Module Content Generation pour Spotify AI Agent

Ce package expose tous les services de génération de contenu musical et textuel :
- Génération d’accords, mélodies, paroles, arrangements, style transfer, classification
- IA générative (OpenAI, Hugging Face, ML custom)
- Feedback utilisateur, API multimodale, export
- Sécurité, audit, conformité RGPD

Auteur : Lead Dev, Architecte IA, ML Engineer, Backend Senior, Data Engineer, Sécurité
"""

from .chord_progression import ChordProgressionGenerator
from .style_transfer import StyleTransfer
from .arrangement_suggester import ArrangementSuggester
from .melody_composer import MelodyComposer
from .lyrics_generator import LyricsGenerator
from .genre_classifier import GenreClassifier

__all__ = [
    "ChordProgressionGenerator",
    "StyleTransfer",
    "ArrangementSuggester",
    "MelodyComposer",
    "LyricsGenerator",
    "GenreClassifier",
]
