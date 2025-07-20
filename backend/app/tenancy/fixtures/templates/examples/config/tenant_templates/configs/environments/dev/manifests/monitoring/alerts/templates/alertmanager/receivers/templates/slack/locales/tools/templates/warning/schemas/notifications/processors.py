"""
Processeurs avancés pour le système de notifications
====================================================

Processeurs ultra-sophistiqués pour transformation, enrichissement,
et traitement intelligent des notifications.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import uuid
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading

import aioredis
import aiofiles
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
import httpx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from transformers import pipeline
import structlog

from .models import *
from .schemas import *
from .config import NotificationSettings
from .validators import NotificationValidator


class ProcessingStage(Enum):
    """Étapes de traitement"""
    PREPROCESSING = "preprocessing"
    ENRICHMENT = "enrichment"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"
    FINALIZATION = "finalization"


@dataclass
class ProcessingContext:
    """Contexte de traitement"""
    request_id: str
    tenant_id: str
    user_id: Optional[str]
    stage: ProcessingStage
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time: Dict[str, float] = field(default_factory=dict)
    
    def add_error(self, error: str):
        """Ajouter une erreur"""
        self.errors.append(error)
    
    def add_warning(self, warning: str):
        """Ajouter un avertissement"""
        self.warnings.append(warning)
    
    def set_processing_time(self, processor: str, duration: float):
        """Définir le temps de traitement"""
        self.processing_time[processor] = duration


class BaseProcessor(ABC):
    """Processeur de base"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.priority = self.config.get('priority', 100)
        self.stage = ProcessingStage(self.config.get('stage', ProcessingStage.TRANSFORMATION.value))
        self.logger = structlog.get_logger(f"Processor.{name}")
    
    @abstractmethod
    async def process(
        self,
        notification: NotificationCreateSchema,
        context: ProcessingContext
    ) -> NotificationCreateSchema:
        """Traiter la notification"""
        pass
    
    async def should_process(
        self,
        notification: NotificationCreateSchema,
        context: ProcessingContext
    ) -> bool:
        """Déterminer si ce processeur doit traiter cette notification"""
        return self.enabled
    
    def get_dependencies(self) -> List[str]:
        """Obtenir les dépendances de ce processeur"""
        return self.config.get('dependencies', [])


class ContentEnrichmentProcessor(BaseProcessor):
    """Processeur d'enrichissement de contenu intelligent"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("ContentEnrichment", config)
        self.stage = ProcessingStage.ENRICHMENT
        
        # Services ML
        self._sentiment_analyzer = None
        self._summarizer = None
        self._translator = None
        self._nlp_model = None
        
        # Configuration
        self.enable_sentiment = config.get('enable_sentiment', True)
        self.enable_summarization = config.get('enable_summarization', True)
        self.enable_translation = config.get('enable_translation', False)
        self.enable_entity_extraction = config.get('enable_entity_extraction', True)
        
        self._setup_ml_models()
    
    def _setup_ml_models(self):
        """Configurer les modèles ML"""
        try:
            if self.enable_sentiment:
                self._sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
            
            if self.enable_summarization:
                self._summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn"
                )
            
            if self.enable_translation:
                self._translator = pipeline(
                    "translation",
                    model="Helsinki-NLP/opus-mt-en-fr"
                )
            
            if self.enable_entity_extraction:
                self._nlp_model = spacy.load("en_core_web_sm")
        
        except Exception as e:
            self.logger.warning(f"Erreur lors du chargement des modèles ML: {e}")
    
    async def process(
        self,
        notification: NotificationCreateSchema,
        context: ProcessingContext
    ) -> NotificationCreateSchema:
        """Enrichir le contenu de la notification"""
        
        start_time = time.time()
        
        try:
            # Analyse de sentiment
            if self.enable_sentiment and self._sentiment_analyzer:
                sentiment = await self._analyze_sentiment(notification.content)
                context.metadata['sentiment'] = sentiment
            
            # Extraction d'entités
            if self.enable_entity_extraction and self._nlp_model:
                entities = await self._extract_entities(notification.content)
                context.metadata['entities'] = entities
            
            # Résumé automatique pour contenu long
            if self.enable_summarization and self._summarizer:
                if len(notification.content) > 500:
                    summary = await self._generate_summary(notification.content)
                    context.metadata['summary'] = summary
            
            # Enrichissement avec des métadonnées contextuelles
            await self._add_contextual_metadata(notification, context)
            
            # Optimisation du titre si nécessaire
            optimized_title = await self._optimize_title(notification.title, context)
            if optimized_title != notification.title:
                notification.title = optimized_title
                context.add_warning("Titre optimisé automatiquement")
            
            return notification
        
        finally:
            context.set_processing_time(self.name, time.time() - start_time)
    
    async def _analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyser le sentiment du contenu"""
        try:
            # Exécuter dans un thread pool pour éviter le blocage
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor,
                    self._sentiment_analyzer,
                    content[:512]  # Limiter la taille
                )
            
            return {
                'label': result[0]['label'],
                'score': result[0]['score'],
                'confidence': 'high' if result[0]['score'] > 0.8 else 'medium' if result[0]['score'] > 0.5 else 'low'
            }
        
        except Exception as e:
            self.logger.warning(f"Erreur analyse sentiment: {e}")
            return {'label': 'NEUTRAL', 'score': 0.5, 'confidence': 'low'}
    
    async def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extraire les entités nommées"""
        try:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                doc = await loop.run_in_executor(
                    executor,
                    self._nlp_model,
                    content
                )
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': ent._.confidence if hasattr(ent._, 'confidence') else 1.0
                })
            
            return entities
        
        except Exception as e:
            self.logger.warning(f"Erreur extraction entités: {e}")
            return []
    
    async def _generate_summary(self, content: str) -> str:
        """Générer un résumé du contenu"""
        try:
            if len(content) < 100:
                return content
            
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor,
                    self._summarizer,
                    content,
                    {"max_length": 150, "min_length": 50, "do_sample": False}
                )
            
            return result[0]['summary_text']
        
        except Exception as e:
            self.logger.warning(f"Erreur génération résumé: {e}")
            return content[:200] + "..." if len(content) > 200 else content
    
    async def _add_contextual_metadata(
        self,
        notification: NotificationCreateSchema,
        context: ProcessingContext
    ):
        """Ajouter des métadonnées contextuelles"""
        
        # Détection de la langue
        detected_language = await self._detect_language(notification.content)
        context.metadata['detected_language'] = detected_language
        
        # Classification du type de contenu
        content_type = await self._classify_content_type(notification.content)
        context.metadata['content_type'] = content_type
        
        # Calcul de la complexité de lecture
        readability_score = self._calculate_readability(notification.content)
        context.metadata['readability_score'] = readability_score
        
        # Détection d'urgence basée sur le contenu
        urgency_score = await self._detect_urgency(notification.content)
        context.metadata['urgency_score'] = urgency_score
    
    async def _detect_language(self, content: str) -> str:
        """Détecter la langue du contenu"""
        # Implémentation simple basée sur des mots-clés
        french_words = {'le', 'la', 'les', 'de', 'du', 'des', 'et', 'ou', 'est', 'sont'}
        english_words = {'the', 'and', 'or', 'is', 'are', 'in', 'on', 'at', 'to', 'for'}
        
        words = set(content.lower().split())
        
        french_score = len(words.intersection(french_words))
        english_score = len(words.intersection(english_words))
        
        if french_score > english_score:
            return 'fr'
        elif english_score > french_score:
            return 'en'
        else:
            return 'unknown'
    
    async def _classify_content_type(self, content: str) -> str:
        """Classifier le type de contenu"""
        content_lower = content.lower()
        
        # Patterns pour différents types
        if any(word in content_lower for word in ['urgent', 'emergency', 'alert', 'critical']):
            return 'alert'
        elif any(word in content_lower for word in ['welcome', 'bienvenue', 'hello', 'bonjour']):
            return 'welcome'
        elif any(word in content_lower for word in ['update', 'mise à jour', 'changelog']):
            return 'update'
        elif any(word in content_lower for word in ['reminder', 'rappel', 'remember']):
            return 'reminder'
        elif any(word in content_lower for word in ['thank', 'merci', 'thanks']):
            return 'gratitude'
        else:
            return 'general'
    
    def _calculate_readability(self, content: str) -> float:
        """Calculer le score de lisibilité"""
        # Implémentation simple du score de Flesch
        sentences = len(re.split(r'[.!?]+', content))
        words = len(content.split())
        syllables = sum(self._count_syllables(word) for word in content.split())
        
        if sentences == 0 or words == 0:
            return 0.0
        
        avg_sentence_length = words / sentences
        avg_syllables_per_word = syllables / words
        
        # Score de Flesch simplifié
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        return max(0, min(100, score))
    
    def _count_syllables(self, word: str) -> int:
        """Compter les syllabes dans un mot"""
        word = word.lower()
        syllables = 0
        vowels = 'aeiouy'
        
        for i, char in enumerate(word):
            if char in vowels:
                if i == 0 or word[i-1] not in vowels:
                    syllables += 1
        
        if word.endswith('e'):
            syllables -= 1
        
        return max(1, syllables)
    
    async def _detect_urgency(self, content: str) -> float:
        """Détecter l'urgence basée sur le contenu"""
        urgent_keywords = {
            'urgent': 1.0, 'emergency': 1.0, 'critical': 0.9, 'important': 0.7,
            'asap': 0.8, 'immediately': 0.9, 'now': 0.6, 'quick': 0.5,
            'urgent': 1.0, 'urgence': 1.0, 'critique': 0.9, 'important': 0.7
        }
        
        content_lower = content.lower()
        max_urgency = 0.0
        
        for keyword, score in urgent_keywords.items():
            if keyword in content_lower:
                max_urgency = max(max_urgency, score)
        
        return max_urgency
    
    async def _optimize_title(self, title: str, context: ProcessingContext) -> str:
        """Optimiser le titre de la notification"""
        # Si le titre est trop long, le raccourcir intelligemment
        if len(title) > 100:
            # Garder les premiers mots importants
            words = title.split()
            optimized = ""
            for word in words:
                if len(optimized + word) < 97:
                    optimized += word + " "
                else:
                    break
            return optimized.strip() + "..."
        
        return title


class PersonalizationProcessor(BaseProcessor):
    """Processeur de personnalisation avancée"""
    
    def __init__(self, config: Dict[str, Any] = None, db_session: AsyncSession = None):
        super().__init__("Personalization", config)
        self.stage = ProcessingStage.ENRICHMENT
        self.db = db_session
        
        # Cache des préférences utilisateur
        self._user_preferences_cache = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Vectorizer pour la similarité de contenu
        self._vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self._content_vectors = None
    
    async def process(
        self,
        notification: NotificationCreateSchema,
        context: ProcessingContext
    ) -> NotificationCreateSchema:
        """Personnaliser la notification"""
        
        start_time = time.time()
        
        try:
            # Personnaliser pour chaque destinataire
            personalized_recipients = []
            
            for recipient in notification.recipients:
                # Obtenir les préférences utilisateur
                preferences = await self._get_user_preferences(
                    context.tenant_id,
                    recipient.user_id
                )
                
                # Personnaliser le contenu
                personalized_recipient = await self._personalize_recipient(
                    recipient,
                    notification,
                    preferences,
                    context
                )
                
                personalized_recipients.append(personalized_recipient)
            
            # Remplacer les destinataires
            notification.recipients = personalized_recipients
            
            # Personnaliser les canaux selon les préférences
            notification.channels = await self._optimize_channels(
                notification.channels,
                personalized_recipients,
                context
            )
            
            return notification
        
        finally:
            context.set_processing_time(self.name, time.time() - start_time)
    
    async def _get_user_preferences(
        self,
        tenant_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Obtenir les préférences utilisateur avec cache"""
        
        cache_key = f"{tenant_id}:{user_id}"
        
        # Vérifier le cache
        if cache_key in self._user_preferences_cache:
            cached_data, timestamp = self._user_preferences_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached_data
        
        # Charger depuis la base
        if self.db:
            try:
                query = select(NotificationPreference).where(
                    and_(
                        NotificationPreference.tenant_id == tenant_id,
                        NotificationPreference.user_id == user_id
                    )
                )
                result = await self.db.execute(query)
                preferences = result.scalars().all()
                
                # Convertir en dictionnaire
                prefs_dict = {}
                for pref in preferences:
                    prefs_dict[pref.preference_key] = pref.preference_value
                
                # Mettre en cache
                self._user_preferences_cache[cache_key] = (prefs_dict, time.time())
                
                return prefs_dict
            
            except Exception as e:
                self.logger.warning(f"Erreur chargement préférences: {e}")
        
        # Préférences par défaut
        return {
            'language': 'en',
            'timezone': 'UTC',
            'notification_frequency': 'immediate',
            'preferred_channels': ['email'],
            'content_style': 'formal'
        }
    
    async def _personalize_recipient(
        self,
        recipient: NotificationRecipientSchema,
        notification: NotificationCreateSchema,
        preferences: Dict[str, Any],
        context: ProcessingContext
    ) -> NotificationRecipientSchema:
        """Personnaliser pour un destinataire spécifique"""
        
        # Créer une copie pour éviter les mutations
        personalized = NotificationRecipientSchema(
            user_id=recipient.user_id,
            email=recipient.email,
            phone=recipient.phone,
            metadata=recipient.metadata.copy() if recipient.metadata else {}
        )
        
        # Ajouter les préférences aux métadonnées
        personalized.metadata.update({
            'language': preferences.get('language', 'en'),
            'timezone': preferences.get('timezone', 'UTC'),
            'content_style': preferences.get('content_style', 'formal'),
            'personalized': True
        })
        
        # Personnaliser l'heure d'envoi si nécessaire
        if preferences.get('notification_frequency') == 'digest':
            # Programmer pour l'envoi en batch
            personalized.metadata['schedule_for_digest'] = True
        
        return personalized
    
    async def _optimize_channels(
        self,
        channels: List[NotificationChannelConfigSchema],
        recipients: List[NotificationRecipientSchema],
        context: ProcessingContext
    ) -> List[NotificationChannelConfigSchema]:
        """Optimiser les canaux selon les préférences"""
        
        # Analyser les préférences de canaux
        preferred_channels = {}
        for recipient in recipients:
            if recipient.metadata and 'preferred_channels' in recipient.metadata:
                for channel in recipient.metadata['preferred_channels']:
                    preferred_channels[channel] = preferred_channels.get(channel, 0) + 1
        
        # Réorganiser les canaux par popularité
        if preferred_channels:
            channels_by_preference = sorted(
                channels,
                key=lambda c: preferred_channels.get(c.type.value, 0),
                reverse=True
            )
            return channels_by_preference
        
        return channels


class DeduplicationProcessor(BaseProcessor):
    """Processeur de déduplication intelligente"""
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        redis_client: aioredis.Redis = None
    ):
        super().__init__("Deduplication", config)
        self.stage = ProcessingStage.PREPROCESSING
        self.redis = redis_client
        
        # Configuration
        self.similarity_threshold = config.get('similarity_threshold', 0.8)
        self.time_window_minutes = config.get('time_window_minutes', 60)
        self.use_ml_similarity = config.get('use_ml_similarity', True)
        
        # Vectorizer pour la similarité
        if self.use_ml_similarity:
            self._vectorizer = TfidfVectorizer(
                max_features=500,
                stop_words='english',
                ngram_range=(1, 2)
            )
    
    async def process(
        self,
        notification: NotificationCreateSchema,
        context: ProcessingContext
    ) -> NotificationCreateSchema:
        """Vérifier et éviter les doublons"""
        
        start_time = time.time()
        
        try:
            # Générer une signature de la notification
            signature = self._generate_signature(notification)
            
            # Vérifier les doublons récents
            is_duplicate = await self._check_for_duplicates(
                signature,
                notification,
                context
            )
            
            if is_duplicate:
                context.add_warning("Notification dupliquée détectée")
                # Marquer comme dupliquée dans les métadonnées
                if not hasattr(notification, 'metadata'):
                    notification.metadata = {}
                notification.metadata['is_duplicate'] = True
                notification.metadata['original_signature'] = signature
            else:
                # Enregistrer cette notification pour future déduplication
                await self._store_signature(signature, notification, context)
            
            return notification
        
        finally:
            context.set_processing_time(self.name, time.time() - start_time)
    
    def _generate_signature(self, notification: NotificationCreateSchema) -> str:
        """Générer une signature unique pour la notification"""
        
        # Combiner les éléments clés
        elements = [
            notification.title.lower().strip(),
            notification.content[:500].lower().strip(),  # Premier 500 chars
            str(notification.priority.value),
            ','.join(sorted([c.type.value for c in notification.channels])),
            ','.join(sorted([r.user_id for r in notification.recipients if r.user_id]))
        ]
        
        # Créer un hash
        combined = '|'.join(elements)
        return hashlib.sha256(combined.encode()).hexdigest()
    
    async def _check_for_duplicates(
        self,
        signature: str,
        notification: NotificationCreateSchema,
        context: ProcessingContext
    ) -> bool:
        """Vérifier s'il y a des doublons récents"""
        
        if not self.redis:
            return False
        
        # Clé pour les signatures récentes
        recent_key = f"dedup:recent:{context.tenant_id}"
        
        # Vérifier signature exacte
        exact_match = await self.redis.hexists(recent_key, signature)
        if exact_match:
            return True
        
        # Vérifier similarité si activée
        if self.use_ml_similarity:
            return await self._check_content_similarity(
                notification,
                context.tenant_id
            )
        
        return False
    
    async def _check_content_similarity(
        self,
        notification: NotificationCreateSchema,
        tenant_id: str
    ) -> bool:
        """Vérifier la similarité de contenu avec ML"""
        
        if not self.redis:
            return False
        
        try:
            # Obtenir les notifications récentes
            content_key = f"dedup:content:{tenant_id}"
            recent_contents = await self.redis.lrange(content_key, 0, 50)
            
            if not recent_contents:
                return False
            
            # Préparer les textes pour la comparaison
            current_text = f"{notification.title} {notification.content}"
            recent_texts = [content.decode() for content in recent_contents]
            all_texts = recent_texts + [current_text]
            
            # Calculer la similarité
            tfidf_matrix = self._vectorizer.fit_transform(all_texts)
            current_vector = tfidf_matrix[-1]
            similarities = cosine_similarity(current_vector, tfidf_matrix[:-1])
            
            # Vérifier si une similarité dépasse le seuil
            max_similarity = np.max(similarities)
            
            return max_similarity >= self.similarity_threshold
        
        except Exception as e:
            self.logger.warning(f"Erreur vérification similarité: {e}")
            return False
    
    async def _store_signature(
        self,
        signature: str,
        notification: NotificationCreateSchema,
        context: ProcessingContext
    ):
        """Stocker la signature pour future déduplication"""
        
        if not self.redis:
            return
        
        # Stocker la signature avec timestamp
        recent_key = f"dedup:recent:{context.tenant_id}"
        await self.redis.hset(recent_key, signature, time.time())
        
        # Expirer les anciennes entrées
        await self.redis.expire(recent_key, self.time_window_minutes * 60)
        
        # Stocker le contenu pour vérification de similarité
        if self.use_ml_similarity:
            content_key = f"dedup:content:{context.tenant_id}"
            content_text = f"{notification.title} {notification.content}"
            await self.redis.lpush(content_key, content_text)
            await self.redis.ltrim(content_key, 0, 50)  # Garder 50 dernières
            await self.redis.expire(content_key, self.time_window_minutes * 60)


class OptimizationProcessor(BaseProcessor):
    """Processeur d'optimisation des performances"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Optimization", config)
        self.stage = ProcessingStage.OPTIMIZATION
        
        # Configuration
        self.max_recipients_per_batch = config.get('max_recipients_per_batch', 1000)
        self.max_content_length = config.get('max_content_length', 10000)
        self.enable_compression = config.get('enable_compression', True)
        self.optimize_images = config.get('optimize_images', True)
    
    async def process(
        self,
        notification: NotificationCreateSchema,
        context: ProcessingContext
    ) -> NotificationCreateSchema:
        """Optimiser la notification pour les performances"""
        
        start_time = time.time()
        
        try:
            # Optimiser le contenu
            notification = await self._optimize_content(notification, context)
            
            # Optimiser les destinataires (batching)
            notification = await self._optimize_recipients(notification, context)
            
            # Optimiser les canaux
            notification = await self._optimize_channels(notification, context)
            
            # Optimiser les métadonnées
            notification = await self._optimize_metadata(notification, context)
            
            return notification
        
        finally:
            context.set_processing_time(self.name, time.time() - start_time)
    
    async def _optimize_content(
        self,
        notification: NotificationCreateSchema,
        context: ProcessingContext
    ) -> NotificationCreateSchema:
        """Optimiser le contenu"""
        
        # Tronquer le contenu si trop long
        if len(notification.content) > self.max_content_length:
            notification.content = notification.content[:self.max_content_length-3] + "..."
            context.add_warning(f"Contenu tronqué à {self.max_content_length} caractères")
        
        # Nettoyer le HTML/Markdown inutile
        notification.content = self._clean_content(notification.content)
        
        # Compresser si activé
        if self.enable_compression:
            notification = await self._compress_content(notification, context)
        
        return notification
    
    async def _optimize_recipients(
        self,
        notification: NotificationCreateSchema,
        context: ProcessingContext
    ) -> NotificationCreateSchema:
        """Optimiser les destinataires"""
        
        # Si trop de destinataires, marquer pour traitement en batch
        if len(notification.recipients) > self.max_recipients_per_batch:
            if not hasattr(notification, 'metadata'):
                notification.metadata = {}
            
            notification.metadata['batch_processing'] = True
            notification.metadata['batch_size'] = self.max_recipients_per_batch
            notification.metadata['total_recipients'] = len(notification.recipients)
            
            context.add_warning(
                f"Traitement en batch activé pour {len(notification.recipients)} destinataires"
            )
        
        # Déduplication des destinataires
        unique_recipients = []
        seen_users = set()
        seen_emails = set()
        
        for recipient in notification.recipients:
            # Déduplication par user_id
            if recipient.user_id and recipient.user_id in seen_users:
                continue
            
            # Déduplication par email
            if recipient.email and recipient.email in seen_emails:
                continue
            
            if recipient.user_id:
                seen_users.add(recipient.user_id)
            if recipient.email:
                seen_emails.add(recipient.email)
            
            unique_recipients.append(recipient)
        
        if len(unique_recipients) < len(notification.recipients):
            duplicates_removed = len(notification.recipients) - len(unique_recipients)
            notification.recipients = unique_recipients
            context.add_warning(f"{duplicates_removed} destinataires dupliqués supprimés")
        
        return notification
    
    async def _optimize_channels(
        self,
        notification: NotificationCreateSchema,
        context: ProcessingContext
    ) -> NotificationCreateSchema:
        """Optimiser les canaux"""
        
        # Réorganiser les canaux par efficacité
        channel_efficiency = {
            NotificationChannelType.PUSH: 1,    # Plus rapide
            NotificationChannelType.EMAIL: 2,
            NotificationChannelType.SMS: 3,
            NotificationChannelType.SLACK: 4,
            NotificationChannelType.WEBHOOK: 5
        }
        
        notification.channels.sort(
            key=lambda c: channel_efficiency.get(c.type, 999)
        )
        
        # Supprimer les canaux redondants
        unique_channels = []
        seen_types = set()
        
        for channel in notification.channels:
            if channel.type not in seen_types:
                unique_channels.append(channel)
                seen_types.add(channel.type)
        
        if len(unique_channels) < len(notification.channels):
            notification.channels = unique_channels
            context.add_warning("Canaux redondants supprimés")
        
        return notification
    
    async def _optimize_metadata(
        self,
        notification: NotificationCreateSchema,
        context: ProcessingContext
    ) -> NotificationCreateSchema:
        """Optimiser les métadonnées"""
        
        if not hasattr(notification, 'metadata') or not notification.metadata:
            return notification
        
        # Supprimer les métadonnées vides ou inutiles
        optimized_metadata = {}
        
        for key, value in notification.metadata.items():
            # Garder seulement les valeurs non-nulles et utiles
            if value is not None and value != "" and value != {}:
                optimized_metadata[key] = value
        
        notification.metadata = optimized_metadata
        
        return notification
    
    def _clean_content(self, content: str) -> str:
        """Nettoyer le contenu"""
        
        # Supprimer les espaces multiples
        content = re.sub(r'\s+', ' ', content)
        
        # Supprimer les balises HTML vides
        content = re.sub(r'<(\w+)[^>]*>\s*</\1>', '', content)
        
        # Nettoyer les caractères de contrôle
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
        
        return content.strip()
    
    async def _compress_content(
        self,
        notification: NotificationCreateSchema,
        context: ProcessingContext
    ) -> NotificationCreateSchema:
        """Compresser le contenu si nécessaire"""
        
        import gzip
        import base64
        
        # Compresser seulement si le contenu est assez long
        if len(notification.content) > 1000:
            try:
                compressed = gzip.compress(notification.content.encode('utf-8'))
                
                # Vérifier si la compression est bénéfique
                if len(compressed) < len(notification.content.encode('utf-8')) * 0.8:
                    if not hasattr(notification, 'metadata'):
                        notification.metadata = {}
                    
                    notification.metadata['content_compressed'] = True
                    notification.metadata['original_size'] = len(notification.content)
                    notification.content = base64.b64encode(compressed).decode('ascii')
                    
                    context.add_warning("Contenu compressé")
            
            except Exception as e:
                self.logger.warning(f"Erreur compression: {e}")
        
        return notification


class ProcessingPipeline:
    """Pipeline de traitement des notifications"""
    
    def __init__(self, settings: NotificationSettings):
        self.settings = settings
        self.processors: List[BaseProcessor] = []
        self.logger = structlog.get_logger("ProcessingPipeline")
        
        # Organiser par étapes
        self.stages: Dict[ProcessingStage, List[BaseProcessor]] = {
            stage: [] for stage in ProcessingStage
        }
    
    def add_processor(self, processor: BaseProcessor):
        """Ajouter un processeur au pipeline"""
        self.processors.append(processor)
        self.stages[processor.stage].append(processor)
        
        # Trier par priorité dans chaque étape
        self.stages[processor.stage].sort(key=lambda p: p.priority)
    
    def remove_processor(self, processor_name: str):
        """Supprimer un processeur"""
        self.processors = [p for p in self.processors if p.name != processor_name]
        
        for stage in self.stages:
            self.stages[stage] = [p for p in self.stages[stage] if p.name != processor_name]
    
    async def process(
        self,
        notification: NotificationCreateSchema,
        tenant_id: str,
        user_id: Optional[str] = None
    ) -> tuple[NotificationCreateSchema, ProcessingContext]:
        """Traiter une notification à travers tout le pipeline"""
        
        # Créer le contexte de traitement
        context = ProcessingContext(
            request_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            stage=ProcessingStage.PREPROCESSING
        )
        
        # Traiter par étapes
        for stage in ProcessingStage:
            context.stage = stage
            
            stage_processors = [p for p in self.stages[stage] if p.enabled]
            
            for processor in stage_processors:
                try:
                    # Vérifier si le processeur doit traiter cette notification
                    if await processor.should_process(notification, context):
                        self.logger.debug(
                            f"Traitement avec {processor.name}",
                            stage=stage.value,
                            request_id=context.request_id
                        )
                        
                        notification = await processor.process(notification, context)
                
                except Exception as e:
                    error_msg = f"Erreur dans {processor.name}: {str(e)}"
                    context.add_error(error_msg)
                    
                    self.logger.error(
                        error_msg,
                        processor=processor.name,
                        stage=stage.value,
                        request_id=context.request_id,
                        exc_info=True
                    )
                    
                    # Continuer avec les autres processeurs
                    continue
        
        return notification, context


def create_default_pipeline(
    settings: NotificationSettings,
    db_session: AsyncSession,
    redis_client: aioredis.Redis,
    validator: NotificationValidator
) -> ProcessingPipeline:
    """Créer un pipeline par défaut avec tous les processeurs"""
    
    pipeline = ProcessingPipeline(settings)
    
    # Ajouter les processeurs dans l'ordre
    
    # 1. Déduplication (preprocessing)
    if settings.is_feature_enabled('deduplication'):
        pipeline.add_processor(DeduplicationProcessor(
            {'priority': 10, 'enabled': True},
            redis_client
        ))
    
    # 2. Enrichissement de contenu
    if settings.is_feature_enabled('content_enrichment'):
        pipeline.add_processor(ContentEnrichmentProcessor({
            'priority': 20,
            'enabled': True,
            'enable_sentiment': True,
            'enable_entity_extraction': True
        }))
    
    # 3. Personnalisation
    if settings.is_feature_enabled('personalization'):
        pipeline.add_processor(PersonalizationProcessor(
            {'priority': 30, 'enabled': True},
            db_session
        ))
    
    # 4. Optimisation
    pipeline.add_processor(OptimizationProcessor({
        'priority': 40,
        'enabled': True,
        'max_recipients_per_batch': 1000
    }))
    
    return pipeline
