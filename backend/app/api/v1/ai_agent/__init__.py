"""
Module AI Agent pour Spotify Artists

Ce package expose toutes les fonctionnalités avancées d’agent conversationnel IA,
prêtes à l’emploi pour une intégration FastAPI/Django, incluant :
- Orchestration multi-IA (OpenAI, modèles internes, audio)
- Mémoire conversationnelle et contextuelle
- Gestion de la personnalité et adaptation dynamique
- Reconnaissance d’intention et parsing avancé
- Génération de réponses multimodales
- Sécurité, audit, et conformité RGPD

Auteur : Équipe IA Spotify (Lead Dev, Architecte IA, ML, Backend, Sécurité)
"""

from .intent_recognition import IntentRecognizer
from .ai_orchestrator import AIOrchestrator
from .memory_system import MemorySystem
from .personality_engine import PersonalityEngine
from .conversation_handler import ConversationHandler
from .response_generator import ResponseGenerator
from .context_manager import ContextManager

__all__ = [
    "IntentRecognizer",
    "AIOrchestrator",
    "MemorySystem",
    "PersonalityEngine",
    "ConversationHandler",
    "ResponseGenerator",
    "ContextManager",
]
