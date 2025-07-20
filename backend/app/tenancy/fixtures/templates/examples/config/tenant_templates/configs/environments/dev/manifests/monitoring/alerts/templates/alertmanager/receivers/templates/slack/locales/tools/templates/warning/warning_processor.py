"""
Warning Processor - Traitement Intelligent des Alertes pour Spotify AI Agent
Analyse contextuelle, enrichissement et classification automatique des warnings
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import re
import hashlib

import aioredis
import numpy as np
from prometheus_client import Counter, Histogram, Gauge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import spacy

from .schemas import AlertLevel, ProcessedWarning, WarningPattern
from .utils import SecurityUtils, MLUtils, TextAnalyzer


class WarningCategory(Enum):
    """Catégories de warnings détectées automatiquement"""
    PERFORMANCE = "performance"
    SECURITY = "security"
    ML_MODEL = "ml_model"
    API = "api"
    DATABASE = "database"
    INFRASTRUCTURE = "infrastructure"
    USER_BEHAVIOR = "user_behavior"
    DATA_QUALITY = "data_quality"
    SYSTEM_HEALTH = "system_health"
    UNKNOWN = "unknown"


class SeverityLevel(Enum):
    """Niveaux de sévérité calculés"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


@dataclass
class WarningContext:
    """Contexte enrichi d'un warning"""
    original_message: str
    processed_message: str
    category: WarningCategory
    severity_score: float
    confidence: float
    keywords: List[str]
    patterns_matched: List[str]
    recommendations: List[str]
    similar_alerts: List[str]
    metadata: Dict[str, Any]


class WarningProcessor:
    """
    Processeur intelligent de warnings avec fonctionnalités :
    - Classification automatique par catégorie
    - Calcul de score de sévérité basé sur ML
    - Détection de patterns et anomalies
    - Recommandations automatiques
    - Clustering d'alertes similaires
    - Analyse de sentiment et intention
    - Enrichissement contextuel
    - Déduplication intelligente
    """
    
    def __init__(
        self,
        redis_client: aioredis.Redis,
        config: Dict[str, Any],
        tenant_id: str = ""
    ):
        self.redis_client = redis_client
        self.config = config
        self.tenant_id = tenant_id
        
        # Logger avec contexte
        self.logger = logging.getLogger(f"warning_processor.{tenant_id}")
        
        # Métriques Prometheus
        self.processing_counter = Counter(
            'warnings_processed_total',
            'Total warnings processed',
            ['tenant_id', 'category', 'severity']
        )
        
        self.processing_duration = Histogram(
            'warning_processing_duration_seconds',
            'Time spent processing warnings',
            ['tenant_id', 'category']
        )
        
        self.pattern_matches = Counter(
            'warning_patterns_matched_total',
            'Warning patterns matched',
            ['tenant_id', 'pattern_name']
        )
        
        # Modèles ML pour classification
        self.nlp_model = None
        self.tfidf_vectorizer = None
        self.clustering_model = None
        
        # Cache des patterns et règles
        self.warning_patterns = {}
        self.severity_rules = {}
        self.recommendation_templates = {}
        
        # Analyseur de texte
        self.text_analyzer = TextAnalyzer()
        
        # Initialisation asynchrone
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialisation asynchrone du processeur"""
        if self._initialized:
            return
        
        try:
            # Chargement du modèle NLP
            await self._load_nlp_model()
            
            # Chargement des patterns de warning
            await self._load_warning_patterns()
            
            # Chargement des règles de sévérité
            await self._load_severity_rules()
            
            # Chargement des templates de recommandations
            await self._load_recommendation_templates()
            
            # Initialisation du vectorizer TF-IDF
            await self._initialize_tfidf()
            
            # Initialisation du clustering
            await self._initialize_clustering()
            
            self._initialized = True
            self.logger.info("Warning processor initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize warning processor: {str(e)}")
            raise
    
    async def process_warning(
        self,
        level: AlertLevel,
        message: str,
        context: Dict[str, Any],
        tags: Dict[str, str]
    ) -> ProcessedWarning:
        """
        Traite un warning avec analyse complète et enrichissement
        
        Args:
            level: Niveau d'alerte original
            message: Message d'alerte
            context: Contexte de l'alerte
            tags: Tags additionnels
            
        Returns:
            ProcessedWarning: Warning enrichi et analysé
        """
        start_time = time.time()
        
        if not self._initialized:
            await self.initialize()
        
        try:
            # Nettoyage et normalisation du message
            cleaned_message = await self._clean_message(message)
            
            # Classification automatique
            category = await self._classify_warning(cleaned_message, context, tags)
            
            # Calcul du score de sévérité
            severity_score = await self._calculate_severity_score(
                cleaned_message, context, category, level
            )
            
            # Détection de patterns
            matched_patterns = await self._detect_patterns(cleaned_message, context)
            
            # Extraction de mots-clés
            keywords = await self._extract_keywords(cleaned_message, category)
            
            # Génération de recommandations
            recommendations = await self._generate_recommendations(
                category, severity_score, matched_patterns, context
            )
            
            # Recherche d'alertes similaires
            similar_alerts = await self._find_similar_alerts(cleaned_message, category)
            
            # Analyse de sentiment
            sentiment_score = await self._analyze_sentiment(cleaned_message)
            
            # Enrichissement des métadonnées
            enriched_metadata = await self._enrich_metadata(
                context, category, severity_score, sentiment_score
            )
            
            # Construction du résultat
            processed_warning = ProcessedWarning(
                original_message=message,
                processed_message=cleaned_message,
                category=category,
                severity_score=severity_score,
                confidence=self._calculate_confidence(matched_patterns, keywords),
                keywords=keywords,
                patterns_matched=[p.name for p in matched_patterns],
                recommendations=recommendations,
                similar_alerts=similar_alerts,
                metadata=enriched_metadata,
                processing_timestamp=datetime.utcnow(),
                tenant_id=self.tenant_id
            )
            
            # Stockage pour future analyse
            await self._store_processed_warning(processed_warning)
            
            # Mise à jour des métriques
            processing_time = time.time() - start_time
            
            self.processing_counter.labels(
                tenant_id=self.tenant_id,
                category=category.value,
                severity=self._score_to_severity_level(severity_score).name
            ).inc()
            
            self.processing_duration.labels(
                tenant_id=self.tenant_id,
                category=category.value
            ).observe(processing_time)
            
            for pattern in matched_patterns:
                self.pattern_matches.labels(
                    tenant_id=self.tenant_id,
                    pattern_name=pattern.name
                ).inc()
            
            self.logger.info(
                f"Warning processed successfully",
                extra={
                    "category": category.value,
                    "severity_score": severity_score,
                    "patterns_matched": len(matched_patterns),
                    "processing_time": processing_time,
                    "tenant_id": self.tenant_id
                }
            )
            
            return processed_warning
            
        except Exception as e:
            self.logger.error(f"Error processing warning: {str(e)}", exc_info=True)
            
            # Fallback avec traitement minimal
            return ProcessedWarning(
                original_message=message,
                processed_message=message,
                category=WarningCategory.UNKNOWN,
                severity_score=self._alert_level_to_score(level),
                confidence=0.0,
                keywords=[],
                patterns_matched=[],
                recommendations=["Manual review required"],
                similar_alerts=[],
                metadata=context,
                processing_timestamp=datetime.utcnow(),
                tenant_id=self.tenant_id
            )
    
    async def analyze_warning_trends(
        self,
        time_window: timedelta = timedelta(hours=24),
        min_occurrences: int = 5
    ) -> Dict[str, Any]:
        """
        Analyse les tendances des warnings sur une période donnée
        
        Args:
            time_window: Fenêtre de temps pour l'analyse
            min_occurrences: Nombre minimum d'occurrences pour considérer une tendance
            
        Returns:
            Dict: Rapport d'analyse des tendances
        """
        try:
            # Récupération des warnings de la période
            warnings = await self._get_warnings_in_timeframe(time_window)
            
            if len(warnings) < min_occurrences:
                return {
                    'total_warnings': len(warnings),
                    'trends': [],
                    'insights': ['Insufficient data for trend analysis']
                }
            
            # Analyse par catégorie
            category_trends = self._analyze_category_trends(warnings)
            
            # Analyse temporelle
            temporal_patterns = self._analyze_temporal_patterns(warnings)
            
            # Détection d'anomalies
            anomalies = await self._detect_trend_anomalies(warnings)
            
            # Clustering des messages similaires
            clusters = await self._cluster_similar_warnings(warnings)
            
            # Génération d'insights
            insights = self._generate_trend_insights(
                category_trends, temporal_patterns, anomalies, clusters
            )
            
            return {
                'analysis_period': time_window.total_seconds(),
                'total_warnings': len(warnings),
                'category_trends': category_trends,
                'temporal_patterns': temporal_patterns,
                'anomalies': anomalies,
                'clusters': clusters,
                'insights': insights,
                'recommendations': self._generate_trend_recommendations(insights)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing warning trends: {str(e)}")
            return {'error': str(e)}
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de traitement du processeur
        
        Returns:
            Dict: Statistiques détaillées
        """
        try:
            stats_key = f"warning_processor_stats:{self.tenant_id}"
            cached_stats = await self.redis_client.get(stats_key)
            
            if cached_stats:
                return json.loads(cached_stats)
            
            # Calcul des statistiques
            total_processed = await self._count_processed_warnings()
            category_distribution = await self._get_category_distribution()
            avg_processing_time = await self._get_avg_processing_time()
            pattern_effectiveness = await self._get_pattern_effectiveness()
            
            stats = {
                'total_processed': total_processed,
                'category_distribution': category_distribution,
                'avg_processing_time_ms': avg_processing_time * 1000,
                'pattern_effectiveness': pattern_effectiveness,
                'last_updated': datetime.utcnow().isoformat(),
                'tenant_id': self.tenant_id
            }
            
            # Cache pour 5 minutes
            await self.redis_client.setex(stats_key, 300, json.dumps(stats))
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting processing stats: {str(e)}")
            return {'error': str(e)}
    
    # Méthodes privées
    
    async def _load_nlp_model(self) -> None:
        """Charge le modèle NLP pour l'analyse de texte"""
        try:
            # Chargement de spaCy (peut être remplacé par un modèle custom)
            model_name = self.config.get('nlp_model', 'en_core_web_sm')
            self.nlp_model = spacy.load(model_name)
            
        except OSError:
            # Fallback vers un modèle plus simple si pas disponible
            self.logger.warning("spaCy model not found, using fallback")
            self.nlp_model = None
    
    async def _load_warning_patterns(self) -> None:
        """Charge les patterns de warning configurés"""
        patterns_config = self.config.get('warning_patterns', {})
        
        self.warning_patterns = {
            'performance': [
                WarningPattern(
                    name='high_latency',
                    regex=r'latency|response.*time|slow|timeout',
                    category=WarningCategory.PERFORMANCE,
                    severity_multiplier=1.2,
                    keywords=['latency', 'slow', 'timeout', 'response']
                ),
                WarningPattern(
                    name='high_cpu',
                    regex=r'cpu.*high|processor.*load|cpu.*\d+%',
                    category=WarningCategory.PERFORMANCE,
                    severity_multiplier=1.3,
                    keywords=['cpu', 'processor', 'load']
                ),
                WarningPattern(
                    name='memory_leak',
                    regex=r'memory.*leak|out.*of.*memory|oom',
                    category=WarningCategory.PERFORMANCE,
                    severity_multiplier=1.5,
                    keywords=['memory', 'leak', 'oom']
                )
            ],
            'security': [
                WarningPattern(
                    name='failed_auth',
                    regex=r'authentication.*failed|login.*failed|unauthorized',
                    category=WarningCategory.SECURITY,
                    severity_multiplier=1.4,
                    keywords=['auth', 'login', 'unauthorized']
                ),
                WarningPattern(
                    name='suspicious_ip',
                    regex=r'suspicious.*ip|blocked.*ip|malicious',
                    category=WarningCategory.SECURITY,
                    severity_multiplier=1.6,
                    keywords=['suspicious', 'blocked', 'malicious']
                )
            ],
            'ml_model': [
                WarningPattern(
                    name='model_drift',
                    regex=r'drift|model.*performance|accuracy.*drop',
                    category=WarningCategory.ML_MODEL,
                    severity_multiplier=1.3,
                    keywords=['drift', 'performance', 'accuracy']
                ),
                WarningPattern(
                    name='prediction_error',
                    regex=r'prediction.*error|inference.*failed',
                    category=WarningCategory.ML_MODEL,
                    severity_multiplier=1.2,
                    keywords=['prediction', 'inference', 'error']
                )
            ]
        }
    
    async def _load_severity_rules(self) -> None:
        """Charge les règles de calcul de sévérité"""
        self.severity_rules = {
            'keywords': {
                'critical': ['critical', 'fatal', 'emergency', 'disaster'],
                'high': ['error', 'failed', 'exception', 'crash'],
                'medium': ['warning', 'alert', 'issue', 'problem'],
                'low': ['info', 'notice', 'debug']
            },
            'multipliers': {
                WarningCategory.SECURITY: 1.5,
                WarningCategory.ML_MODEL: 1.3,
                WarningCategory.PERFORMANCE: 1.2,
                WarningCategory.DATABASE: 1.4,
                WarningCategory.API: 1.1
            }
        }
    
    async def _load_recommendation_templates(self) -> None:
        """Charge les templates de recommandations"""
        self.recommendation_templates = {
            WarningCategory.PERFORMANCE: {
                'high_latency': [
                    "Check database query performance",
                    "Review API endpoint optimization",
                    "Consider implementing caching",
                    "Monitor network connectivity"
                ],
                'high_cpu': [
                    "Analyze CPU-intensive processes",
                    "Consider horizontal scaling",
                    "Review algorithm efficiency",
                    "Check for infinite loops"
                ],
                'memory_leak': [
                    "Review memory allocation patterns",
                    "Check for unclosed connections",
                    "Analyze garbage collection logs",
                    "Consider memory profiling"
                ]
            },
            WarningCategory.SECURITY: {
                'failed_auth': [
                    "Review authentication logs",
                    "Check for brute force attacks",
                    "Consider implementing rate limiting",
                    "Update security policies"
                ],
                'suspicious_ip': [
                    "Block suspicious IP addresses",
                    "Review firewall rules",
                    "Analyze traffic patterns",
                    "Consider geo-blocking"
                ]
            },
            WarningCategory.ML_MODEL: {
                'model_drift': [
                    "Retrain model with recent data",
                    "Review feature importance",
                    "Check data quality",
                    "Consider A/B testing new model"
                ],
                'prediction_error': [
                    "Review input data validation",
                    "Check model inference pipeline",
                    "Analyze prediction confidence scores",
                    "Consider fallback predictions"
                ]
            }
        }
    
    async def _initialize_tfidf(self) -> None:
        """Initialise le vectorizer TF-IDF pour l'analyse de texte"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    async def _initialize_clustering(self) -> None:
        """Initialise le modèle de clustering pour regrouper les alertes similaires"""
        self.clustering_model = DBSCAN(
            eps=0.3,
            min_samples=2,
            metric='cosine'
        )
    
    async def _clean_message(self, message: str) -> str:
        """Nettoie et normalise un message d'alerte"""
        # Suppression des caractères spéciaux
        cleaned = re.sub(r'[^\w\s]', ' ', message)
        
        # Normalisation des espaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Conversion en minuscules
        cleaned = cleaned.lower()
        
        return cleaned
    
    async def _classify_warning(
        self,
        message: str,
        context: Dict[str, Any],
        tags: Dict[str, str]
    ) -> WarningCategory:
        """Classifie automatiquement un warning par catégorie"""
        
        # Classification basée sur les tags
        if 'category' in tags:
            try:
                return WarningCategory(tags['category'])
            except ValueError:
                pass
        
        # Classification basée sur le service
        service = context.get('service_name', '').lower()
        if 'ml' in service or 'model' in service:
            return WarningCategory.ML_MODEL
        elif 'api' in service or 'rest' in service:
            return WarningCategory.API
        elif 'db' in service or 'database' in service:
            return WarningCategory.DATABASE
        
        # Classification basée sur les patterns
        for category, patterns in self.warning_patterns.items():
            for pattern in patterns:
                if re.search(pattern.regex, message, re.IGNORECASE):
                    return pattern.category
        
        # Classification basée sur les mots-clés
        performance_keywords = ['latency', 'slow', 'timeout', 'cpu', 'memory']
        security_keywords = ['auth', 'login', 'unauthorized', 'attack', 'breach']
        ml_keywords = ['model', 'prediction', 'inference', 'drift', 'accuracy']
        
        if any(keyword in message for keyword in security_keywords):
            return WarningCategory.SECURITY
        elif any(keyword in message for keyword in performance_keywords):
            return WarningCategory.PERFORMANCE
        elif any(keyword in message for keyword in ml_keywords):
            return WarningCategory.ML_MODEL
        
        return WarningCategory.UNKNOWN
    
    async def _calculate_severity_score(
        self,
        message: str,
        context: Dict[str, Any],
        category: WarningCategory,
        level: AlertLevel
    ) -> float:
        """Calcule un score de sévérité basé sur multiple facteurs"""
        
        # Score de base basé sur le niveau d'alerte
        base_score = self._alert_level_to_score(level)
        
        # Multiplicateur basé sur la catégorie
        category_multiplier = self.severity_rules['multipliers'].get(category, 1.0)
        
        # Score basé sur les mots-clés
        keyword_score = 0
        for severity, keywords in self.severity_rules['keywords'].items():
            if any(keyword in message for keyword in keywords):
                if severity == 'critical':
                    keyword_score = max(keyword_score, 0.9)
                elif severity == 'high':
                    keyword_score = max(keyword_score, 0.7)
                elif severity == 'medium':
                    keyword_score = max(keyword_score, 0.5)
                elif severity == 'low':
                    keyword_score = max(keyword_score, 0.3)
        
        # Score basé sur le contexte
        context_score = 0
        if context.get('error_count', 0) > 10:
            context_score += 0.2
        if context.get('affected_users', 0) > 100:
            context_score += 0.3
        if context.get('downtime_minutes', 0) > 5:
            context_score += 0.4
        
        # Calcul final
        final_score = (base_score + keyword_score + context_score) * category_multiplier
        
        # Normalisation entre 0 et 1
        return min(max(final_score, 0.0), 1.0)
    
    async def _detect_patterns(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> List[WarningPattern]:
        """Détecte les patterns correspondants dans le message"""
        matched_patterns = []
        
        for category, patterns in self.warning_patterns.items():
            for pattern in patterns:
                if re.search(pattern.regex, message, re.IGNORECASE):
                    matched_patterns.append(pattern)
        
        return matched_patterns
    
    async def _extract_keywords(self, message: str, category: WarningCategory) -> List[str]:
        """Extrait les mots-clés importants du message"""
        if self.nlp_model:
            doc = self.nlp_model(message)
            keywords = [token.lemma_ for token in doc if not token.is_stop and len(token.text) > 2]
        else:
            # Fallback simple
            words = message.split()
            keywords = [word for word in words if len(word) > 2]
        
        # Ajout des mots-clés spécifiques à la catégorie
        category_keywords = {
            WarningCategory.PERFORMANCE: ['performance', 'latency', 'cpu', 'memory'],
            WarningCategory.SECURITY: ['security', 'auth', 'unauthorized', 'attack'],
            WarningCategory.ML_MODEL: ['model', 'prediction', 'accuracy', 'drift']
        }
        
        if category in category_keywords:
            keywords.extend([kw for kw in category_keywords[category] if kw in message])
        
        return list(set(keywords))[:10]  # Limite à 10 mots-clés
    
    async def _generate_recommendations(
        self,
        category: WarningCategory,
        severity_score: float,
        patterns: List[WarningPattern],
        context: Dict[str, Any]
    ) -> List[str]:
        """Génère des recommandations basées sur l'analyse"""
        recommendations = []
        
        # Recommandations basées sur les patterns
        for pattern in patterns:
            pattern_recommendations = self.recommendation_templates.get(category, {}).get(pattern.name, [])
            recommendations.extend(pattern_recommendations)
        
        # Recommandations basées sur la sévérité
        if severity_score > 0.8:
            recommendations.append("Immediate attention required - escalate to on-call team")
        elif severity_score > 0.6:
            recommendations.append("High priority - review within 30 minutes")
        elif severity_score > 0.4:
            recommendations.append("Medium priority - review within 2 hours")
        
        # Recommandations basées sur le contexte
        if context.get('repeat_count', 0) > 5:
            recommendations.append("Recurring issue detected - investigate root cause")
        
        if context.get('affected_services'):
            recommendations.append("Multiple services affected - check for system-wide issues")
        
        return list(set(recommendations))[:5]  # Limite à 5 recommandations
    
    async def _find_similar_alerts(self, message: str, category: WarningCategory) -> List[str]:
        """Trouve des alertes similaires récentes"""
        try:
            # Recherche dans le cache Redis
            search_key = f"similar_alerts:{self.tenant_id}:{category.value}"
            recent_alerts = await self.redis_client.lrange(search_key, 0, 100)
            
            if not recent_alerts:
                return []
            
            # Vectorisation du message actuel
            if self.tfidf_vectorizer and len(recent_alerts) > 0:
                messages = [message] + [json.loads(alert)['message'] for alert in recent_alerts]
                
                try:
                    vectors = self.tfidf_vectorizer.fit_transform(messages)
                    similarities = (vectors[0] * vectors[1:].T).toarray()[0]
                    
                    # Seuil de similarité
                    threshold = 0.3
                    similar_indices = [i for i, sim in enumerate(similarities) if sim > threshold]
                    
                    return [json.loads(recent_alerts[i])['id'] for i in similar_indices][:3]
                except:
                    pass
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error finding similar alerts: {str(e)}")
            return []
    
    async def _analyze_sentiment(self, message: str) -> float:
        """Analyse le sentiment du message (négatif = plus sévère)"""
        # Mots indicateurs de sévérité
        severe_words = ['critical', 'fatal', 'emergency', 'disaster', 'failed', 'error']
        moderate_words = ['warning', 'issue', 'problem', 'alert']
        mild_words = ['info', 'notice', 'debug']
        
        message_lower = message.lower()
        
        severe_count = sum(1 for word in severe_words if word in message_lower)
        moderate_count = sum(1 for word in moderate_words if word in message_lower)
        mild_count = sum(1 for word in mild_words if word in message_lower)
        
        # Score de sévérité basé sur les mots
        if severe_count > 0:
            return 0.8 + min(severe_count * 0.05, 0.2)
        elif moderate_count > 0:
            return 0.5 + min(moderate_count * 0.1, 0.3)
        elif mild_count > 0:
            return 0.2 + min(mild_count * 0.1, 0.3)
        
        return 0.5  # Score neutre par défaut
    
    async def _enrich_metadata(
        self,
        context: Dict[str, Any],
        category: WarningCategory,
        severity_score: float,
        sentiment_score: float
    ) -> Dict[str, Any]:
        """Enrichit les métadonnées avec les données d'analyse"""
        enriched = context.copy()
        
        enriched.update({
            'analysis': {
                'category': category.value,
                'severity_score': severity_score,
                'sentiment_score': sentiment_score,
                'severity_level': self._score_to_severity_level(severity_score).name,
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'processor_version': '1.0.0'
            },
            'tenant_context': {
                'tenant_id': self.tenant_id,
                'environment': context.get('environment', 'unknown')
            }
        })
        
        return enriched
    
    def _calculate_confidence(self, patterns: List[WarningPattern], keywords: List[str]) -> float:
        """Calcule le niveau de confiance de l'analyse"""
        confidence = 0.5  # Base
        
        # Bonus pour les patterns détectés
        confidence += len(patterns) * 0.1
        
        # Bonus pour les mots-clés pertinents
        confidence += min(len(keywords) * 0.05, 0.3)
        
        return min(confidence, 1.0)
    
    def _alert_level_to_score(self, level: AlertLevel) -> float:
        """Convertit un niveau d'alerte en score numérique"""
        mapping = {
            AlertLevel.CRITICAL: 0.9,
            AlertLevel.HIGH: 0.7,
            AlertLevel.WARNING: 0.5,
            AlertLevel.INFO: 0.3,
            AlertLevel.DEBUG: 0.1
        }
        return mapping.get(level, 0.5)
    
    def _score_to_severity_level(self, score: float) -> SeverityLevel:
        """Convertit un score en niveau de sévérité"""
        if score >= 0.8:
            return SeverityLevel.CRITICAL
        elif score >= 0.6:
            return SeverityLevel.HIGH
        elif score >= 0.4:
            return SeverityLevel.MEDIUM
        elif score >= 0.2:
            return SeverityLevel.LOW
        else:
            return SeverityLevel.VERY_LOW
    
    async def _store_processed_warning(self, warning: ProcessedWarning) -> None:
        """Stocke un warning traité pour analyse future"""
        try:
            # Stockage dans Redis avec TTL
            warning_key = f"processed_warning:{self.tenant_id}:{warning.processing_timestamp.timestamp()}"
            warning_data = asdict(warning)
            
            await self.redis_client.setex(
                warning_key,
                86400,  # 24 heures
                json.dumps(warning_data, default=str)
            )
            
            # Ajout à l'index par catégorie
            category_key = f"category_index:{self.tenant_id}:{warning.category.value}"
            await self.redis_client.lpush(category_key, warning_key)
            await self.redis_client.ltrim(category_key, 0, 1000)  # Garde les 1000 derniers
            
        except Exception as e:
            self.logger.error(f"Error storing processed warning: {str(e)}")
