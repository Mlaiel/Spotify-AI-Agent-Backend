"""
Moteur de Corrélation d'Événements - Spotify AI Agent
====================================================

Système avancé de corrélation d'événements utilisant l'apprentissage automatique
pour détecter les patterns, grouper les alertes liées et prédire les incidents.

Fonctionnalités:
- Corrélation en temps réel avec algorithmes ML
- Détection de patterns complexes
- Groupement intelligent des alertes
- Prédiction d'incidents en cascade
- Analyse de causalité
- Corrélation spatiale et temporelle
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import networkx as nx
import redis.asyncio as redis


class CorrelationType(Enum):
    """Types de corrélation supportés"""
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    CAUSAL = "causal"
    SEMANTIC = "semantic"
    METRIC_BASED = "metric_based"
    SERVICE_DEPENDENCY = "service_dependency"


class CorrelationStrength(Enum):
    """Force de la corrélation"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class CorrelationResult:
    """Résultat de corrélation entre alertes"""
    primary_alert_id: str
    correlated_alert_ids: List[str]
    correlation_type: CorrelationType
    strength: CorrelationStrength
    confidence: float
    correlation_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IncidentPrediction:
    """Prédiction d'incident basée sur les corrélations"""
    incident_id: str
    predicted_severity: str
    probability: float
    affected_services: List[str]
    estimated_impact: Dict[str, Any]
    timeline_prediction: Dict[str, datetime]
    recommended_actions: List[str]
    confidence_factors: Dict[str, float]


class CorrelationEngine:
    """Moteur de corrélation intelligent avec ML"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.correlation_window = timedelta(
            minutes=config.get('correlation_window_minutes', 30)
        )
        self.max_correlations_per_alert = config.get('max_correlations_per_alert', 50)
        self.min_confidence_threshold = config.get('min_confidence_threshold', 0.6)
        
        # Stockage des corrélations
        self.active_correlations: Dict[str, List[CorrelationResult]] = defaultdict(list)
        self.correlation_graph = nx.MultiGraph()
        self.correlation_history: deque = deque(maxlen=100000)
        
        # Cache Redis
        self.redis_client = None
        
        # Modèles ML
        self.clustering_models: Dict[str, Any] = {}
        self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.dimensionality_reducer = PCA(n_components=10)
        
        # Patterns et règles apprises
        self.learned_patterns: Dict[str, Any] = {}
        self.service_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.temporal_patterns: Dict[str, List[Dict]] = defaultdict(list)
        
        # Cache de similarité
        self.similarity_cache: Dict[str, float] = {}
        self.cache_ttl = timedelta(hours=1)
        
        # Métriques de performance
        self.correlation_stats = {
            'total_correlations': 0,
            'successful_predictions': 0,
            'false_positives': 0,
            'processing_times': deque(maxlen=1000)
        }
    
    async def initialize(self):
        """Initialisation asynchrone du moteur"""
        try:
            # Connexion Redis
            self.redis_client = redis.from_url(
                self.config.get('redis_url', 'redis://localhost:6379'),
                decode_responses=True
            )
            
            # Chargement des patterns existants
            await self._load_learned_patterns()
            
            # Initialisation des modèles ML
            await self._initialize_ml_models()
            
            # Chargement des dépendances de services
            await self._load_service_dependencies()
            
            # Démarrage des tâches de fond
            asyncio.create_task(self._continuous_pattern_learning())
            asyncio.create_task(self._correlation_graph_maintenance())
            asyncio.create_task(self._performance_optimization())
            
            self.logger.info("CorrelationEngine initialisé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation: {e}")
            raise
    
    async def correlate_alert(self, alert: Any) -> Optional[str]:
        """Corrélation d'une nouvelle alerte avec les alertes existantes"""
        try:
            start_time = datetime.utcnow()
            correlations = []
            
            # Recherche de corrélations de différents types
            temporal_corr = await self._find_temporal_correlations(alert)
            spatial_corr = await self._find_spatial_correlations(alert)
            semantic_corr = await self._find_semantic_correlations(alert)
            metric_corr = await self._find_metric_correlations(alert)
            dependency_corr = await self._find_dependency_correlations(alert)
            
            correlations.extend(temporal_corr)
            correlations.extend(spatial_corr)
            correlations.extend(semantic_corr)
            correlations.extend(metric_corr)
            correlations.extend(dependency_corr)
            
            # Filtrage et classement des corrélations
            valid_correlations = [
                corr for corr in correlations 
                if corr.confidence >= self.min_confidence_threshold
            ]
            
            if not valid_correlations:
                return None
            
            # Création ou mise à jour d'un groupe de corrélation
            correlation_id = await self._create_or_update_correlation_group(
                alert, valid_correlations
            )
            
            # Mise à jour du graphe de corrélation
            await self._update_correlation_graph(alert, valid_correlations)
            
            # Prédiction d'incident si applicable
            await self._check_incident_prediction(correlation_id, alert, valid_correlations)
            
            # Mise à jour des statistiques
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.correlation_stats['processing_times'].append(processing_time)
            self.correlation_stats['total_correlations'] += len(valid_correlations)
            
            self.logger.debug(
                f"Alerte {alert.id} corrélée avec {len(valid_correlations)} alertes "
                f"(groupe: {correlation_id})"
            )
            
            return correlation_id
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la corrélation: {e}")
            return None
    
    async def analyze_active_alerts(self, alerts: List[Any]) -> Dict[str, Any]:
        """Analyse complète des alertes actives pour détecter des patterns"""
        try:
            analysis_result = {
                'total_alerts': len(alerts),
                'correlation_groups': [],
                'incident_predictions': [],
                'patterns_detected': [],
                'recommendations': []
            }
            
            if len(alerts) < 2:
                return analysis_result
            
            # Clustering des alertes par similarité
            clusters = await self._cluster_alerts_by_similarity(alerts)
            
            for cluster_id, cluster_alerts in clusters.items():
                if len(cluster_alerts) < 2:
                    continue
                
                # Analyse du cluster
                cluster_analysis = await self._analyze_alert_cluster(cluster_alerts)
                analysis_result['correlation_groups'].append(cluster_analysis)
                
                # Détection de patterns
                patterns = await self._detect_cluster_patterns(cluster_alerts)
                analysis_result['patterns_detected'].extend(patterns)
                
                # Prédiction d'incident pour ce cluster
                prediction = await self._predict_cluster_incident(cluster_alerts)
                if prediction:
                    analysis_result['incident_predictions'].append(prediction)
            
            # Génération de recommandations
            recommendations = await self._generate_correlation_recommendations(
                analysis_result
            )
            analysis_result['recommendations'] = recommendations
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse des alertes: {e}")
            return {}
    
    async def predict_incident_cascade(self, initial_alert: Any) -> Optional[IncidentPrediction]:
        """Prédiction d'un incident en cascade à partir d'une alerte initiale"""
        try:
            # Recherche des services dépendants
            affected_services = await self._find_dependent_services(
                initial_alert.service, initial_alert.tenant_id
            )
            
            if not affected_services:
                return None
            
            # Calcul de la probabilité de propagation
            propagation_probs = {}
            for service in affected_services:
                prob = await self._calculate_propagation_probability(
                    initial_alert, service
                )
                propagation_probs[service] = prob
            
            # Estimation de la sévérité de l'incident
            estimated_severity = await self._estimate_incident_severity(
                initial_alert, affected_services, propagation_probs
            )
            
            # Prédiction de la timeline
            timeline = await self._predict_incident_timeline(
                initial_alert, affected_services
            )
            
            # Génération de recommandations d'actions
            actions = await self._generate_incident_actions(
                initial_alert, affected_services, estimated_severity
            )
            
            # Calcul des facteurs de confiance
            confidence_factors = {
                'historical_pattern_match': await self._get_pattern_match_confidence(initial_alert),
                'service_dependency_strength': self._calculate_dependency_strength(affected_services),
                'metric_correlation_strength': await self._get_metric_correlation_strength(initial_alert),
                'temporal_pattern_match': await self._get_temporal_pattern_confidence(initial_alert)
            }
            
            # Calcul de la probabilité globale
            overall_probability = np.mean(list(confidence_factors.values()))
            
            prediction = IncidentPrediction(
                incident_id=f"incident_pred_{int(datetime.utcnow().timestamp())}",
                predicted_severity=estimated_severity,
                probability=overall_probability,
                affected_services=list(affected_services),
                estimated_impact=await self._estimate_business_impact(affected_services),
                timeline_prediction=timeline,
                recommended_actions=actions,
                confidence_factors=confidence_factors
            )
            
            self.logger.info(
                f"Prédiction d'incident générée pour l'alerte {initial_alert.id}: "
                f"probabilité {overall_probability:.2f}, sévérité {estimated_severity}"
            )
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction d'incident: {e}")
            return None
    
    async def learn_from_incident_resolution(self, incident_data: Dict[str, Any]):
        """Apprentissage à partir de la résolution d'un incident"""
        try:
            # Extraction des informations pertinentes
            root_cause = incident_data.get('root_cause', '')
            affected_services = incident_data.get('affected_services', [])
            duration = incident_data.get('duration_minutes', 0)
            resolution_actions = incident_data.get('resolution_actions', [])
            
            # Mise à jour des patterns temporels
            await self._update_temporal_patterns(incident_data)
            
            # Mise à jour des dépendances de services
            await self._update_service_dependencies(affected_services, incident_data)
            
            # Apprentissage des corrélations causales
            await self._learn_causal_correlations(incident_data)
            
            # Mise à jour des modèles de prédiction
            await self._update_prediction_models(incident_data)
            
            self.logger.info(f"Apprentissage effectué à partir de l'incident {incident_data.get('id', 'unknown')}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'apprentissage: {e}")
    
    async def _find_temporal_correlations(self, alert: Any) -> List[CorrelationResult]:
        """Recherche de corrélations temporelles"""
        correlations = []
        
        try:
            # Recherche d'alertes dans la fenêtre temporelle
            time_window_start = alert.timestamp - self.correlation_window
            time_window_end = alert.timestamp + self.correlation_window
            
            # Requête Redis pour les alertes récentes
            recent_alerts_key = f"recent_alerts:{alert.tenant_id}"
            recent_alerts_data = await self.redis_client.lrange(recent_alerts_key, 0, -1)
            
            for alert_data in recent_alerts_data:
                try:
                    other_alert = json.loads(alert_data)
                    other_timestamp = datetime.fromisoformat(other_alert['timestamp'])
                    
                    if (time_window_start <= other_timestamp <= time_window_end and
                        other_alert['id'] != alert.id):
                        
                        # Calcul de la similarité temporelle
                        time_diff = abs((alert.timestamp - other_timestamp).total_seconds())
                        temporal_similarity = max(0, 1 - (time_diff / self.correlation_window.total_seconds()))
                        
                        if temporal_similarity > 0.3:  # Seuil de similarité temporelle
                            correlation = CorrelationResult(
                                primary_alert_id=alert.id,
                                correlated_alert_ids=[other_alert['id']],
                                correlation_type=CorrelationType.TEMPORAL,
                                strength=self._classify_correlation_strength(temporal_similarity),
                                confidence=temporal_similarity,
                                correlation_id=f"temporal_{alert.id}_{other_alert['id']}",
                                explanation=f"Alertes survenues à {time_diff:.0f}s d'intervalle"
                            )
                            correlations.append(correlation)
                
                except (json.JSONDecodeError, KeyError) as e:
                    continue
        
        except Exception as e:
            self.logger.error(f"Erreur dans la corrélation temporelle: {e}")
        
        return correlations
    
    async def _find_semantic_correlations(self, alert: Any) -> List[CorrelationResult]:
        """Recherche de corrélations sémantiques basées sur le texte"""
        correlations = []
        
        try:
            # Vectorisation du texte de l'alerte courante
            alert_text = f"{alert.title} {alert.description}"
            
            # Recherche d'alertes similaires par le contenu
            similar_alerts_key = f"alert_texts:{alert.tenant_id}"
            similar_texts_data = await self.redis_client.hgetall(similar_alerts_key)
            
            if not similar_texts_data:
                return correlations
            
            # Création du corpus pour la vectorisation
            texts = [alert_text]
            alert_ids = [alert.id]
            
            for other_id, other_text in similar_texts_data.items():
                if other_id != alert.id:
                    texts.append(other_text)
                    alert_ids.append(other_id)
            
            if len(texts) < 2:
                return correlations
            
            # Vectorisation TF-IDF
            tfidf_matrix = self.text_vectorizer.fit_transform(texts)
            
            # Calcul de similarité cosinus
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            for i, similarity in enumerate(similarities):
                if similarity > 0.5:  # Seuil de similarité sémantique
                    other_alert_id = alert_ids[i + 1]
                    
                    correlation = CorrelationResult(
                        primary_alert_id=alert.id,
                        correlated_alert_ids=[other_alert_id],
                        correlation_type=CorrelationType.SEMANTIC,
                        strength=self._classify_correlation_strength(similarity),
                        confidence=similarity,
                        correlation_id=f"semantic_{alert.id}_{other_alert_id}",
                        explanation=f"Similarité textuelle: {similarity:.2f}"
                    )
                    correlations.append(correlation)
        
        except Exception as e:
            self.logger.error(f"Erreur dans la corrélation sémantique: {e}")
        
        return correlations
    
    async def _find_spatial_correlations(self, alert: Any) -> List[CorrelationResult]:
        """Recherche de corrélations spatiales (même service, environnement, etc.)"""
        correlations = []
        
        try:
            # Critères de corrélation spatiale
            spatial_criteria = [
                ('service', alert.service),
                ('environment', alert.environment),
                ('tenant_id', alert.tenant_id)
            ]
            
            for criterion, value in spatial_criteria:
                if not value:
                    continue
                
                # Recherche d'alertes avec le même critère
                alerts_key = f"alerts_by_{criterion}:{value}"
                related_alerts = await self.redis_client.smembers(alerts_key)
                
                for other_alert_id in related_alerts:
                    if other_alert_id != alert.id:
                        # Vérification de la proximité temporelle
                        other_alert_data = await self.redis_client.hget(
                            f"alert:{other_alert_id}", 'data'
                        )
                        
                        if other_alert_data:
                            try:
                                other_alert = json.loads(other_alert_data)
                                other_timestamp = datetime.fromisoformat(other_alert['timestamp'])
                                
                                time_diff = abs((alert.timestamp - other_timestamp).total_seconds())
                                if time_diff <= self.correlation_window.total_seconds():
                                    
                                    # Calcul de la force de corrélation spatiale
                                    spatial_strength = self._calculate_spatial_strength(
                                        alert, other_alert, criterion
                                    )
                                    
                                    correlation = CorrelationResult(
                                        primary_alert_id=alert.id,
                                        correlated_alert_ids=[other_alert_id],
                                        correlation_type=CorrelationType.SPATIAL,
                                        strength=self._classify_correlation_strength(spatial_strength),
                                        confidence=spatial_strength,
                                        correlation_id=f"spatial_{criterion}_{alert.id}_{other_alert_id}",
                                        explanation=f"Corrélation spatiale par {criterion}: {value}"
                                    )
                                    correlations.append(correlation)
                            
                            except (json.JSONDecodeError, KeyError):
                                continue
        
        except Exception as e:
            self.logger.error(f"Erreur dans la corrélation spatiale: {e}")
        
        return correlations
    
    def _classify_correlation_strength(self, confidence: float) -> CorrelationStrength:
        """Classification de la force de corrélation"""
        if confidence >= 0.9:
            return CorrelationStrength.VERY_STRONG
        elif confidence >= 0.75:
            return CorrelationStrength.STRONG
        elif confidence >= 0.6:
            return CorrelationStrength.MODERATE
        else:
            return CorrelationStrength.WEAK
    
    def _calculate_spatial_strength(self, alert1: Any, alert2: Dict, criterion: str) -> float:
        """Calcul de la force de corrélation spatiale"""
        base_strength = 0.7  # Force de base pour la correspondance du critère
        
        # Bonus pour d'autres correspondances
        bonus = 0.0
        
        if alert1.service == alert2.get('service'):
            bonus += 0.1
        if alert1.environment == alert2.get('environment'):
            bonus += 0.1
        if alert1.severity.value == alert2.get('severity'):
            bonus += 0.1
        
        return min(1.0, base_strength + bonus)
