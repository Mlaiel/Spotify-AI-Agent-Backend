"""
🎨 Advanced Info Template Generators - Production-Ready System
============================================================

Générateurs ultra-avancés de templates d'information avec ML et personnalisation.
Architecture industrielle pour génération contextuelle et optimisation automatique.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import uuid

import numpy as np
import pandas as pd
from jinja2 import Template, Environment, FileSystemLoader
from pydantic import BaseModel, Field, validator
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class TemplateType(Enum):
    """Types de templates supportés"""
    WELCOME = "welcome"
    ALERT = "alert"
    BILLING = "billing"
    SECURITY = "security"
    PERFORMANCE = "performance"
    RECOMMENDATION = "recommendation"
    INSIGHT = "insight"
    OPTIMIZATION = "optimization"
    ANNOUNCEMENT = "announcement"
    MAINTENANCE = "maintenance"


class MessagePriority(Enum):
    """Priorités de message"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    URGENT = "urgent"


class ContentFormat(Enum):
    """Formats de contenu supportés"""
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    SLACK_BLOCKS = "slack_blocks"
    TEAMS_CARD = "teams_card"
    DISCORD_EMBED = "discord_embed"
    EMAIL = "email"


@dataclass
class GenerationContext:
    """Contexte de génération de template"""
    tenant_id: str
    user_id: Optional[str] = None
    template_type: TemplateType = TemplateType.WELCOME
    priority: MessagePriority = MessagePriority.NORMAL
    language: str = "en"
    format: ContentFormat = ContentFormat.TEXT
    channel: str = "default"
    
    # Données contextuelles
    user_data: Dict[str, Any] = field(default_factory=dict)
    tenant_data: Dict[str, Any] = field(default_factory=dict)
    system_data: Dict[str, Any] = field(default_factory=dict)
    event_data: Dict[str, Any] = field(default_factory=dict)
    
    # Configuration avancée
    personalization_enabled: bool = True
    ml_optimization: bool = True
    a_b_testing: bool = False
    analytics_tracking: bool = True
    
    # Métadonnées
    generated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Post-initialisation avec validation"""
        if self.expires_at is None:
            self.expires_at = self.generated_at + timedelta(hours=24)


@dataclass
class GeneratedContent:
    """Contenu généré avec métadonnées"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    format: ContentFormat = ContentFormat.TEXT
    language: str = "en"
    template_type: TemplateType = TemplateType.WELCOME
    
    # Métriques de qualité
    engagement_score: float = 0.0
    personalization_score: float = 0.0
    sentiment_score: float = 0.0
    readability_score: float = 0.0
    
    # Métadonnées
    generated_at: datetime = field(default_factory=datetime.utcnow)
    generation_time_ms: float = 0.0
    template_version: str = "1.0.0"
    model_version: str = "1.0.0"
    
    # A/B Testing
    variant: Optional[str] = None
    test_group: Optional[str] = None
    
    # Analytics
    tracking_data: Dict[str, Any] = field(default_factory=dict)


class BaseTemplateGenerator(ABC):
    """Générateur de base pour templates d'information"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialisation ML
        self._init_ml_components()
        
        # Cache des templates
        self.template_cache: Dict[str, Template] = {}
        
        # Métriques
        self.generation_metrics = {
            'total_generated': 0,
            'avg_generation_time': 0.0,
            'success_rate': 1.0
        }
    
    def _init_ml_components(self):
        """Initialisation des composants ML"""
        try:
            # Modèle de sentiment
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            
            # Tokenizer pour embeddings
            self.tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Vectorizer pour similarité
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english'
            )
            
            self.logger.info("ML components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML components: {str(e)}")
            # Fallback sans ML
            self.sentiment_analyzer = None
            self.tokenizer = None
            self.vectorizer = None
    
    @abstractmethod
    async def generate_template(
        self, 
        context: GenerationContext
    ) -> GeneratedContent:
        """Génération de template (méthode abstraite)"""
        pass
    
    async def _calculate_quality_scores(
        self, 
        content: str, 
        context: GenerationContext
    ) -> Tuple[float, float, float, float]:
        """Calcul des scores de qualité"""
        
        # Score d'engagement (basé sur longueur, structure, etc.)
        engagement_score = self._calculate_engagement_score(content, context)
        
        # Score de personnalisation
        personalization_score = self._calculate_personalization_score(content, context)
        
        # Score de sentiment
        sentiment_score = await self._calculate_sentiment_score(content)
        
        # Score de lisibilité
        readability_score = self._calculate_readability_score(content)
        
        return engagement_score, personalization_score, sentiment_score, readability_score
    
    def _calculate_engagement_score(self, content: str, context: GenerationContext) -> float:
        """Calcul du score d'engagement"""
        score = 0.5  # Base score
        
        # Longueur optimale
        optimal_length = 150  # caractères
        length_ratio = min(len(content) / optimal_length, 2.0)
        if 0.5 <= length_ratio <= 1.5:
            score += 0.2
        
        # Présence d'éléments d'engagement
        engagement_keywords = ['vous', 'votre', 'personnalisé', 'recommandé', 'optimisé']
        for keyword in engagement_keywords:
            if keyword.lower() in content.lower():
                score += 0.05
        
        # Call-to-action
        cta_patterns = ['cliquez', 'découvrez', 'essayez', 'contactez']
        for pattern in cta_patterns:
            if pattern.lower() in content.lower():
                score += 0.1
                break
        
        # Emoji et formatage
        if any(char in content for char in ['📊', '🚀', '✅', '💡', '🔔']):
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_personalization_score(self, content: str, context: GenerationContext) -> float:
        """Calcul du score de personnalisation"""
        score = 0.0
        
        # Utilisation de données utilisateur
        if context.user_data:
            for key, value in context.user_data.items():
                if str(value).lower() in content.lower():
                    score += 0.2
        
        # Données tenant spécifiques
        if context.tenant_data:
            for key, value in context.tenant_data.items():
                if str(value).lower() in content.lower():
                    score += 0.15
        
        # Adaptation à la langue
        if context.language != "en":
            score += 0.2
        
        # Adaptation au canal
        channel_specific_elements = {
            'slack': ['@channel', ':point_right:', 'thread'],
            'teams': ['@team', 'card', 'action'],
            'email': ['subject', 'signature', 'unsubscribe']
        }
        
        if context.channel in channel_specific_elements:
            for element in channel_specific_elements[context.channel]:
                if element in content.lower():
                    score += 0.1
        
        return min(score, 1.0)
    
    async def _calculate_sentiment_score(self, content: str) -> float:
        """Calcul du score de sentiment"""
        if not self.sentiment_analyzer:
            return 0.5  # Neutre par défaut
        
        try:
            result = self.sentiment_analyzer(content)
            
            # Conversion du score sentiment en valeur 0-1
            if result[0]['label'] == 'POSITIVE':
                return result[0]['score']
            else:
                return 1.0 - result[0]['score']
        
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {str(e)}")
            return 0.5
    
    def _calculate_readability_score(self, content: str) -> float:
        """Calcul du score de lisibilité (Flesch Reading Ease simplifié)"""
        
        # Comptage des mots et phrases
        words = len(content.split())
        sentences = content.count('.') + content.count('!') + content.count('?')
        
        if sentences == 0:
            sentences = 1
        
        # Calcul simplifié de lisibilité
        avg_sentence_length = words / sentences
        
        # Score basé sur la longueur moyenne des phrases
        if avg_sentence_length <= 15:
            score = 0.9  # Très lisible
        elif avg_sentence_length <= 20:
            score = 0.7  # Lisible
        elif avg_sentence_length <= 25:
            score = 0.5  # Moyennement lisible
        else:
            score = 0.3  # Difficile à lire
        
        return score


class InfoTemplateGenerator(BaseTemplateGenerator):
    """Générateur principal de templates d'information"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Templates Jinja2
        self.template_env = Environment(
            loader=FileSystemLoader(config.get('template_dir', './templates')),
            autoescape=True
        )
        
        # Cache avancé
        self.template_cache_ttl = config.get('cache_ttl', 3600)
        self.template_cache_size = config.get('cache_size', 1000)
        
        # Configuration ML
        self.ml_enabled = config.get('ml_enabled', True)
        self.personalization_weight = config.get('personalization_weight', 0.3)
        
        # Templates par défaut
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Chargement des templates par défaut"""
        
        self.default_templates = {
            TemplateType.WELCOME: {
                'en': "Welcome {{ user_name }}! Your {{ tier }} account is ready.",
                'fr': "Bienvenue {{ user_name }} ! Votre compte {{ tier }} est prêt.",
                'de': "Willkommen {{ user_name }}! Ihr {{ tier }}-Konto ist bereit."
            },
            TemplateType.ALERT: {
                'en': "🚨 Alert: {{ alert_type }} - {{ message }}",
                'fr': "🚨 Alerte: {{ alert_type }} - {{ message }}",
                'de': "🚨 Warnung: {{ alert_type }} - {{ message }}"
            },
            TemplateType.RECOMMENDATION: {
                'en': "💡 Recommendation: {{ recommendation }} based on your usage",
                'fr': "💡 Recommandation: {{ recommendation }} basée sur votre utilisation",
                'de': "💡 Empfehlung: {{ recommendation }} basierend auf Ihrer Nutzung"
            }
        }
    
    async def generate_template(
        self, 
        context: GenerationContext
    ) -> GeneratedContent:
        """Génération principale de template"""
        
        start_time = datetime.utcnow()
        
        try:
            # Sélection du template optimal
            template_content = await self._select_optimal_template(context)
            
            # Préparation des variables de template
            template_vars = await self._prepare_template_variables(context)
            
            # Rendu du template
            rendered_content = await self._render_template(
                template_content, 
                template_vars, 
                context
            )
            
            # Post-processing et optimisation
            optimized_content = await self._optimize_content(rendered_content, context)
            
            # Calcul des scores de qualité
            scores = await self._calculate_quality_scores(optimized_content, context)
            
            # Calcul du temps de génération
            generation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Création du contenu généré
            generated_content = GeneratedContent(
                content=optimized_content,
                format=context.format,
                language=context.language,
                template_type=context.template_type,
                engagement_score=scores[0],
                personalization_score=scores[1],
                sentiment_score=scores[2],
                readability_score=scores[3],
                generation_time_ms=generation_time,
                tracking_data={
                    'tenant_id': context.tenant_id,
                    'user_id': context.user_id,
                    'channel': context.channel,
                    'template_source': 'advanced_generator'
                }
            )
            
            # Mise à jour des métriques
            self._update_metrics(generation_time, True)
            
            self.logger.info(
                f"Template generated successfully: {context.template_type.value} "
                f"for tenant {context.tenant_id} in {generation_time:.2f}ms"
            )
            
            return generated_content
            
        except Exception as e:
            self.logger.error(f"Template generation failed: {str(e)}")
            self._update_metrics(0, False)
            
            # Fallback template
            return await self._generate_fallback_template(context)
    
    async def _select_optimal_template(self, context: GenerationContext) -> str:
        """Sélection du template optimal basée sur ML"""
        
        # Template de base
        base_template = self.default_templates.get(
            context.template_type, {}
        ).get(context.language, "Template not found")
        
        if not self.ml_enabled:
            return base_template
        
        # Logique ML pour sélection avancée
        # (Ici on utiliserait un modèle entraîné pour sélectionner le meilleur template)
        
        return base_template
    
    async def _prepare_template_variables(self, context: GenerationContext) -> Dict[str, Any]:
        """Préparation des variables de template"""
        
        variables = {
            'tenant_id': context.tenant_id,
            'user_id': context.user_id,
            'current_date': datetime.utcnow().strftime('%Y-%m-%d'),
            'current_time': datetime.utcnow().strftime('%H:%M:%S'),
            'language': context.language,
            'channel': context.channel
        }
        
        # Ajout des données contextuelles
        variables.update(context.user_data)
        variables.update(context.tenant_data)
        variables.update(context.system_data)
        variables.update(context.event_data)
        
        # Variables personnalisées basées sur ML
        if context.personalization_enabled and self.ml_enabled:
            ml_variables = await self._generate_ml_variables(context)
            variables.update(ml_variables)
        
        return variables
    
    async def _generate_ml_variables(self, context: GenerationContext) -> Dict[str, Any]:
        """Génération de variables basées sur ML"""
        
        ml_vars = {}
        
        # Recommandations personnalisées
        if 'usage_data' in context.user_data:
            recommendations = await self._generate_recommendations(
                context.user_data['usage_data']
            )
            ml_vars['ml_recommendations'] = recommendations
        
        # Insights comportementaux
        if 'behavior_data' in context.user_data:
            insights = await self._generate_insights(
                context.user_data['behavior_data']
            )
            ml_vars['ml_insights'] = insights
        
        # Optimisations suggérées
        if 'performance_data' in context.system_data:
            optimizations = await self._generate_optimizations(
                context.system_data['performance_data']
            )
            ml_vars['ml_optimizations'] = optimizations
        
        return ml_vars
    
    async def _render_template(
        self, 
        template_content: str, 
        variables: Dict[str, Any], 
        context: GenerationContext
    ) -> str:
        """Rendu du template avec Jinja2"""
        
        try:
            template = Template(template_content)
            rendered = template.render(**variables)
            
            return rendered.strip()
            
        except Exception as e:
            self.logger.error(f"Template rendering failed: {str(e)}")
            return f"Error rendering template: {template_content}"
    
    async def _optimize_content(self, content: str, context: GenerationContext) -> str:
        """Optimisation du contenu généré"""
        
        optimized = content
        
        # Optimisation du format
        if context.format == ContentFormat.SLACK_BLOCKS:
            optimized = await self._optimize_for_slack(optimized)
        elif context.format == ContentFormat.MARKDOWN:
            optimized = await self._optimize_for_markdown(optimized)
        elif context.format == ContentFormat.HTML:
            optimized = await self._optimize_for_html(optimized)
        
        # Optimisation de la longueur
        max_length = self.config.get('max_message_length', 4096)
        if len(optimized) > max_length:
            optimized = optimized[:max_length-3] + "..."
        
        # Optimisation ML
        if self.ml_enabled and context.ml_optimization:
            optimized = await self._ml_optimize_content(optimized, context)
        
        return optimized
    
    async def _ml_optimize_content(self, content: str, context: GenerationContext) -> str:
        """Optimisation basée sur ML"""
        
        # Ici on implémenterait des optimisations ML avancées
        # comme l'amélioration du sentiment, de l'engagement, etc.
        
        return content
    
    async def _optimize_for_slack(self, content: str) -> str:
        """Optimisation pour Slack"""
        
        # Conversion en blocs Slack si nécessaire
        if not content.startswith('{'):
            # Format texte simple pour Slack
            return content.replace('\n', '\n> ')
        
        return content
    
    async def _optimize_for_markdown(self, content: str) -> str:
        """Optimisation pour Markdown"""
        
        # Ajout d'éléments Markdown si nécessaire
        if not any(marker in content for marker in ['**', '*', '`', '#']):
            # Mise en forme basique
            return f"**{content.split('.')[0]}**\n\n{'.'.join(content.split('.')[1:])}"
        
        return content
    
    async def _optimize_for_html(self, content: str) -> str:
        """Optimisation pour HTML"""
        
        # Conversion en HTML si nécessaire
        if not content.strip().startswith('<'):
            # Conversion basique text -> HTML
            paragraphs = content.split('\n\n')
            html_content = '\n'.join(f'<p>{p}</p>' for p in paragraphs if p.strip())
            return html_content
        
        return content
    
    async def _generate_recommendations(self, usage_data: Dict[str, Any]) -> List[str]:
        """Génération de recommandations basées sur l'usage"""
        
        recommendations = []
        
        # Analyse des patterns d'usage
        if usage_data.get('api_calls', 0) > 1000:
            recommendations.append("Consider upgrading to premium for better rate limits")
        
        if usage_data.get('storage_usage', 0) > 0.8:
            recommendations.append("Your storage is 80% full, consider cleaning up old files")
        
        return recommendations
    
    async def _generate_insights(self, behavior_data: Dict[str, Any]) -> List[str]:
        """Génération d'insights comportementaux"""
        
        insights = []
        
        # Analyse des comportements
        if behavior_data.get('login_frequency', 0) < 3:
            insights.append("You've been less active recently")
        
        if behavior_data.get('feature_usage', {}).get('advanced_features', 0) < 0.1:
            insights.append("Explore advanced features to maximize value")
        
        return insights
    
    async def _generate_optimizations(self, performance_data: Dict[str, Any]) -> List[str]:
        """Génération d'optimisations suggérées"""
        
        optimizations = []
        
        # Analyse des performances
        if performance_data.get('avg_response_time', 0) > 500:
            optimizations.append("Consider optimizing your queries for better performance")
        
        if performance_data.get('error_rate', 0) > 0.05:
            optimizations.append("Review error logs to improve reliability")
        
        return optimizations
    
    async def _generate_fallback_template(self, context: GenerationContext) -> GeneratedContent:
        """Génération d'un template de fallback"""
        
        fallback_content = f"System message for tenant {context.tenant_id}"
        
        return GeneratedContent(
            content=fallback_content,
            format=context.format,
            language=context.language,
            template_type=context.template_type,
            engagement_score=0.3,
            personalization_score=0.0,
            sentiment_score=0.5,
            readability_score=0.8,
            generation_time_ms=1.0,
            tracking_data={
                'tenant_id': context.tenant_id,
                'template_source': 'fallback'
            }
        )
    
    def _update_metrics(self, generation_time: float, success: bool):
        """Mise à jour des métriques de génération"""
        
        self.generation_metrics['total_generated'] += 1
        
        if success:
            # Mise à jour du temps moyen
            current_avg = self.generation_metrics['avg_generation_time']
            total = self.generation_metrics['total_generated']
            
            new_avg = (current_avg * (total - 1) + generation_time) / total
            self.generation_metrics['avg_generation_time'] = new_avg
        
        # Mise à jour du taux de succès
        total_success = self.generation_metrics['success_rate'] * (
            self.generation_metrics['total_generated'] - 1
        )
        if success:
            total_success += 1
        
        self.generation_metrics['success_rate'] = total_success / self.generation_metrics['total_generated']


class DynamicContentEngine:
    """Moteur de contenu dynamique avec adaptation en temps réel"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Générateurs spécialisés
        self.generators: Dict[TemplateType, BaseTemplateGenerator] = {}
        
        # Cache adaptatif
        self.adaptive_cache = {}
        self.cache_hit_rate = 0.0
        
        # Modèles ML pour adaptation
        self._init_adaptive_models()
    
    def _init_adaptive_models(self):
        """Initialisation des modèles adaptatifs"""
        
        # Modèle de prédiction d'engagement
        self.engagement_predictor = None  # Serait un modèle ML entraîné
        
        # Modèle d'optimisation de contenu
        self.content_optimizer = None  # Serait un modèle ML entraîné
        
        # Modèle de détection d'anomalies
        self.anomaly_detector = None  # Serait un modèle ML entraîné
        
        self.logger.info("Adaptive models initialized")
    
    async def generate_dynamic_content(
        self, 
        context: GenerationContext,
        adaptation_level: float = 1.0
    ) -> GeneratedContent:
        """Génération de contenu avec adaptation dynamique"""
        
        # Sélection du générateur optimal
        generator = await self._select_optimal_generator(context)
        
        # Génération initiale
        initial_content = await generator.generate_template(context)
        
        # Adaptation dynamique basée sur le niveau d'adaptation
        if adaptation_level > 0.5:
            adapted_content = await self._adapt_content_dynamically(
                initial_content, 
                context, 
                adaptation_level
            )
            return adapted_content
        
        return initial_content
    
    async def _select_optimal_generator(
        self, 
        context: GenerationContext
    ) -> BaseTemplateGenerator:
        """Sélection du générateur optimal"""
        
        # Pour cette démo, on utilise le générateur principal
        if context.template_type not in self.generators:
            self.generators[context.template_type] = InfoTemplateGenerator(self.config)
        
        return self.generators[context.template_type]
    
    async def _adapt_content_dynamically(
        self, 
        content: GeneratedContent, 
        context: GenerationContext,
        adaptation_level: float
    ) -> GeneratedContent:
        """Adaptation dynamique du contenu"""
        
        adapted_content = content.content
        
        # Adaptation basée sur les métriques historiques
        if self.engagement_predictor and adaptation_level > 0.7:
            adapted_content = await self._optimize_for_engagement(
                adapted_content, 
                context
            )
        
        # Adaptation basée sur le contexte temps réel
        if adaptation_level > 0.8:
            adapted_content = await self._adapt_for_realtime_context(
                adapted_content, 
                context
            )
        
        # Mise à jour du contenu adapté
        content.content = adapted_content
        content.personalization_score *= (1 + adaptation_level * 0.2)
        
        return content
    
    async def _optimize_for_engagement(self, content: str, context: GenerationContext) -> str:
        """Optimisation pour l'engagement"""
        
        # Optimisations basées sur l'historique d'engagement
        optimizations = [
            lambda c: c.replace('.', '!') if context.priority == MessagePriority.HIGH else c,
            lambda c: f"🚀 {c}" if context.template_type == TemplateType.ANNOUNCEMENT else c,
            lambda c: c + " 💡" if context.template_type == TemplateType.RECOMMENDATION else c
        ]
        
        for optimization in optimizations:
            content = optimization(content)
        
        return content
    
    async def _adapt_for_realtime_context(self, content: str, context: GenerationContext) -> str:
        """Adaptation pour le contexte temps réel"""
        
        # Ajout d'éléments contextuels temps réel
        time_of_day = datetime.utcnow().hour
        
        if time_of_day < 12:
            content = f"Good morning! {content}"
        elif time_of_day < 18:
            content = f"Good afternoon! {content}"
        else:
            content = f"Good evening! {content}"
        
        return content
