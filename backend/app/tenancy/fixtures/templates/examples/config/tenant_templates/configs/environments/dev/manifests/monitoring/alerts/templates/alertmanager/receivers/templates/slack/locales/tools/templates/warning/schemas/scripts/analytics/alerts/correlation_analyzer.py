"""
Analyseur de corrélation intelligent pour Spotify AI Agent
Module avancé d'analyse de corrélation entre événements et alertes
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import aioredis
import asyncpg
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

class CorrelationType(Enum):
    """Types de corrélation détectés"""
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    SEMANTIC = "semantic"
    SPATIAL = "spatial"
    FUNCTIONAL = "functional"

class CorrelationStrength(Enum):
    """Force de la corrélation"""
    VERY_STRONG = "very_strong"  # > 0.9
    STRONG = "strong"           # 0.7 - 0.9
    MODERATE = "moderate"       # 0.5 - 0.7
    WEAK = "weak"              # 0.3 - 0.5
    VERY_WEAK = "very_weak"    # < 0.3

@dataclass
class CorrelationEvent:
    """Événement pour analyse de corrélation"""
    id: str
    timestamp: datetime
    service: str
    component: str
    severity: str
    message: str
    labels: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CorrelationRule:
    """Règle de corrélation découverte"""
    id: str
    pattern_name: str
    condition: Dict[str, Any]
    correlation_type: CorrelationType
    strength: float
    confidence: float
    frequency: int
    last_seen: datetime
    examples: List[str] = field(default_factory=list)

@dataclass
class CorrelationResult:
    """Résultat d'analyse de corrélation"""
    source_event_id: str
    correlated_events: List[CorrelationEvent]
    correlation_rules: List[CorrelationRule]
    correlation_graph: Dict[str, List[str]]
    correlation_scores: Dict[str, float]
    pattern_analysis: Dict[str, Any]
    temporal_analysis: Dict[str, Any]
    impact_prediction: Dict[str, Any]

class CorrelationAnalyzer:
    """
    Analyseur de corrélation intelligent
    
    Fonctionnalités:
    - Détection de patterns de corrélation temporelle
    - Analyse causale automatique
    - Clustering d'événements similaires
    - Graphe de dépendances dynamique
    - Prédiction d'événements corrélés
    - Apprentissage automatique de règles
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client: Optional[aioredis.Redis] = None
        self.db_pool: Optional[asyncpg.Pool] = None
        
        # Structures de données pour l'analyse
        self.correlation_graph = nx.DiGraph()
        self.temporal_windows = defaultdict(deque)
        self.learned_rules = {}
        self.service_dependencies = defaultdict(set)
        
        # Modèles ML
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.clustering_model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.5,
            linkage='ward'
        )
        
        # Cache et optimisation
        self.correlation_cache = {}
        self.pattern_cache = {}
        self.last_analysis_time = {}
        
        # Métriques Prometheus
        self.correlations_found = Counter(
            'correlations_found_total',
            'Total des corrélations trouvées',
            ['type', 'strength']
        )
        self.analysis_duration = Histogram(
            'correlation_analysis_duration_seconds',
            'Durée d\'analyse de corrélation'
        )
        self.correlation_strength_gauge = Gauge(
            'correlation_strength',
            'Force de corrélation',
            ['source_service', 'target_service']
        )
        
    async def initialize(self):
        """Initialisation de l'analyseur"""
        try:
            # Connexion Redis
            self.redis_client = await aioredis.from_url(
                self.config.get('redis_url', 'redis://localhost:6379'),
                encoding='utf-8',
                decode_responses=True
            )
            
            # Pool PostgreSQL
            self.db_pool = await asyncpg.create_pool(
                self.config.get('database_url'),
                min_size=3,
                max_size=10
            )
            
            # Chargement des règles existantes
            await self._load_learned_rules()
            
            # Construction du graphe de dépendances
            await self._build_dependency_graph()
            
            logger.info("Analyseur de corrélation initialisé")
            
        except Exception as e:
            logger.error(f"Erreur initialisation analyseur: {e}")
            raise
    
    async def analyze_correlations(
        self, 
        source_event: CorrelationEvent,
        time_window: timedelta = timedelta(minutes=30)
    ) -> CorrelationResult:
        """
        Analyse de corrélation pour un événement source
        
        Args:
            source_event: Événement source pour l'analyse
            time_window: Fenêtre temporelle d'analyse
            
        Returns:
            CorrelationResult: Résultat complet de l'analyse
        """
        start_time = datetime.now()
        
        try:
            # Recherche d'événements dans la fenêtre temporelle
            related_events = await self._find_related_events(source_event, time_window)
            
            # Analyse de corrélation temporelle
            temporal_correlations = await self._analyze_temporal_correlations(
                source_event, related_events
            )
            
            # Analyse de corrélation sémantique
            semantic_correlations = await self._analyze_semantic_correlations(
                source_event, related_events
            )
            
            # Analyse de corrélation causale
            causal_correlations = await self._analyze_causal_correlations(
                source_event, related_events
            )
            
            # Fusion des résultats
            all_correlations = {
                **temporal_correlations,
                **semantic_correlations,
                **causal_correlations
            }
            
            # Filtrage par seuil de confiance
            min_confidence = self.config.get('min_correlation_confidence', 0.5)
            filtered_correlations = {
                k: v for k, v in all_correlations.items() 
                if v.get('confidence', 0) >= min_confidence
            }
            
            # Extraction des événements corrélés
            correlated_events = await self._extract_correlated_events(
                filtered_correlations, related_events
            )
            
            # Recherche de règles applicables
            applicable_rules = await self._find_applicable_rules(
                source_event, correlated_events
            )
            
            # Construction du graphe de corrélation
            correlation_graph = await self._build_correlation_graph(
                source_event, correlated_events, filtered_correlations
            )
            
            # Analyse de patterns
            pattern_analysis = await self._analyze_patterns(
                source_event, correlated_events
            )
            
            # Analyse temporelle avancée
            temporal_analysis = await self._advanced_temporal_analysis(
                source_event, correlated_events
            )
            
            # Prédiction d'impact
            impact_prediction = await self._predict_correlation_impact(
                source_event, correlated_events, filtered_correlations
            )
            
            result = CorrelationResult(
                source_event_id=source_event.id,
                correlated_events=correlated_events,
                correlation_rules=applicable_rules,
                correlation_graph=correlation_graph,
                correlation_scores=filtered_correlations,
                pattern_analysis=pattern_analysis,
                temporal_analysis=temporal_analysis,
                impact_prediction=impact_prediction
            )
            
            # Mise à jour des métriques
            duration = (datetime.now() - start_time).total_seconds()
            self.analysis_duration.observe(duration)
            
            for correlation_type, data in filtered_correlations.items():
                strength = self._score_to_strength(data.get('score', 0))
                self.correlations_found.labels(
                    type=correlation_type,
                    strength=strength.value
                ).inc()
            
            # Apprentissage automatique de nouvelles règles
            await self._learn_new_rules(source_event, result)
            
            # Cache du résultat
            await self._cache_correlation_result(source_event.id, result)
            
            logger.info(f"Analyse de corrélation terminée en {duration:.3f}s, "
                       f"{len(correlated_events)} événements corrélés trouvés")
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur analyse corrélation: {e}")
            raise
    
    async def _find_related_events(
        self, 
        source_event: CorrelationEvent, 
        time_window: timedelta
    ) -> List[CorrelationEvent]:
        """Recherche d'événements liés dans la fenêtre temporelle"""
        try:
            window_start = source_event.timestamp - time_window
            window_end = source_event.timestamp + time_window
            
            # Requête optimisée avec index sur timestamp
            query = """
                SELECT DISTINCT e.*
                FROM correlation_events e
                WHERE e.timestamp BETWEEN $1 AND $2
                AND e.id != $3
                AND (
                    e.service = $4
                    OR e.component = $5
                    OR EXISTS (
                        SELECT 1 FROM service_dependencies sd
                        WHERE (sd.source_service = $4 AND sd.target_service = e.service)
                        OR (sd.source_service = e.service AND sd.target_service = $4)
                    )
                )
                ORDER BY e.timestamp
                LIMIT 500
            """
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    query, window_start, window_end,
                    source_event.id, source_event.service, source_event.component
                )
            
            events = []
            for row in rows:
                event = CorrelationEvent(
                    id=row['id'],
                    timestamp=row['timestamp'],
                    service=row['service'],
                    component=row['component'],
                    severity=row['severity'],
                    message=row['message'],
                    labels=row.get('labels', {}),
                    metrics=row.get('metrics', {}),
                    features=row.get('features', {})
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Erreur recherche événements liés: {e}")
            return []
    
    async def _analyze_temporal_correlations(
        self, 
        source_event: CorrelationEvent, 
        related_events: List[CorrelationEvent]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyse de corrélations temporelles"""
        try:
            temporal_correlations = {}
            
            for event in related_events:
                time_diff = abs((event.timestamp - source_event.timestamp).total_seconds())
                
                # Calcul du score temporel (plus proche = score plus élevé)
                max_window = self.config.get('max_temporal_window', 1800)  # 30 min
                temporal_score = max(0, 1 - (time_diff / max_window))
                
                if temporal_score > 0.1:  # Seuil minimal
                    # Analyse de patterns temporels
                    pattern_strength = await self._calculate_temporal_pattern_strength(
                        source_event, event
                    )
                    
                    # Calcul de la confiance basée sur l'historique
                    confidence = await self._calculate_temporal_confidence(
                        source_event, event, time_diff
                    )
                    
                    correlation_key = f"temporal_{event.id}"
                    temporal_correlations[correlation_key] = {
                        'event_id': event.id,
                        'type': CorrelationType.TEMPORAL.value,
                        'score': temporal_score * pattern_strength,
                        'confidence': confidence,
                        'time_difference': time_diff,
                        'pattern_strength': pattern_strength,
                        'details': {
                            'lag_seconds': time_diff,
                            'direction': 'before' if event.timestamp < source_event.timestamp else 'after',
                            'service_correlation': source_event.service == event.service
                        }
                    }
            
            return temporal_correlations
            
        except Exception as e:
            logger.error(f"Erreur analyse temporelle: {e}")
            return {}
    
    async def _analyze_semantic_correlations(
        self, 
        source_event: CorrelationEvent, 
        related_events: List[CorrelationEvent]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyse de corrélations sémantiques"""
        try:
            if not related_events:
                return {}
            
            # Préparation des textes pour analyse TF-IDF
            documents = [source_event.message]
            documents.extend([event.message for event in related_events])
            
            # Vectorisation TF-IDF
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            except ValueError:
                # Fallback si pas assez de données
                return {}
            
            # Calcul de similarité cosine
            source_vector = tfidf_matrix[0:1]
            other_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(source_vector, other_vectors)[0]
            
            semantic_correlations = {}
            
            for i, (event, similarity) in enumerate(zip(related_events, similarities)):
                if similarity > self.config.get('min_semantic_similarity', 0.3):
                    # Analyse des labels communs
                    label_similarity = self._calculate_label_similarity(
                        source_event.labels, event.labels
                    )
                    
                    # Analyse des métriques
                    metric_similarity = self._calculate_metric_similarity(
                        source_event.metrics, event.metrics
                    )
                    
                    # Score combiné
                    combined_score = (
                        similarity * 0.5 +
                        label_similarity * 0.3 +
                        metric_similarity * 0.2
                    )
                    
                    # Confiance basée sur la consistance historique
                    confidence = await self._calculate_semantic_confidence(
                        source_event, event, similarity
                    )
                    
                    correlation_key = f"semantic_{event.id}"
                    semantic_correlations[correlation_key] = {
                        'event_id': event.id,
                        'type': CorrelationType.SEMANTIC.value,
                        'score': combined_score,
                        'confidence': confidence,
                        'text_similarity': similarity,
                        'label_similarity': label_similarity,
                        'metric_similarity': metric_similarity,
                        'details': {
                            'common_keywords': self._extract_common_keywords(
                                source_event.message, event.message
                            ),
                            'shared_labels': set(source_event.labels.keys()) & 
                                           set(event.labels.keys()),
                            'severity_match': source_event.severity == event.severity
                        }
                    }
            
            return semantic_correlations
            
        except Exception as e:
            logger.error(f"Erreur analyse sémantique: {e}")
            return {}
    
    async def _analyze_causal_correlations(
        self, 
        source_event: CorrelationEvent, 
        related_events: List[CorrelationEvent]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyse de corrélations causales"""
        try:
            causal_correlations = {}
            
            # Analyse des dépendances de services connues
            dependencies = await self._get_service_dependencies(source_event.service)
            
            for event in related_events:
                causal_score = 0.0
                causal_type = None
                
                # Corrélation basée sur les dépendances de services
                if event.service in dependencies.get('downstream', []):
                    causal_score += 0.7
                    causal_type = 'downstream_dependency'
                elif event.service in dependencies.get('upstream', []):
                    causal_score += 0.8
                    causal_type = 'upstream_dependency'
                
                # Analyse de la séquence temporelle
                if event.timestamp < source_event.timestamp:
                    # Événement antérieur = cause potentielle
                    time_diff = (source_event.timestamp - event.timestamp).total_seconds()
                    if time_diff < 300:  # 5 minutes
                        causal_score += 0.6
                        causal_type = 'temporal_cause'
                
                # Analyse des patterns de propagation d'erreur
                propagation_score = await self._analyze_error_propagation(
                    source_event, event
                )
                causal_score += propagation_score * 0.5
                
                # Analyse de corrélation de métriques
                metric_correlation = await self._analyze_metric_correlation(
                    source_event, event
                )
                causal_score += metric_correlation * 0.3
                
                if causal_score > 0.3:  # Seuil minimal pour causalité
                    # Calcul de confiance causale
                    confidence = await self._calculate_causal_confidence(
                        source_event, event, causal_score
                    )
                    
                    correlation_key = f"causal_{event.id}"
                    causal_correlations[correlation_key] = {
                        'event_id': event.id,
                        'type': CorrelationType.CAUSAL.value,
                        'score': min(1.0, causal_score),
                        'confidence': confidence,
                        'causal_type': causal_type,
                        'propagation_score': propagation_score,
                        'metric_correlation': metric_correlation,
                        'details': {
                            'dependency_relation': causal_type,
                            'temporal_order': 'cause' if event.timestamp < source_event.timestamp else 'effect',
                            'service_path': await self._find_service_path(event.service, source_event.service)
                        }
                    }
            
            return causal_correlations
            
        except Exception as e:
            logger.error(f"Erreur analyse causale: {e}")
            return {}
    
    async def _calculate_temporal_pattern_strength(
        self, 
        source_event: CorrelationEvent, 
        target_event: CorrelationEvent
    ) -> float:
        """Calcul de la force du pattern temporel"""
        try:
            # Recherche de patterns historiques similaires
            pattern_key = f"{source_event.service}:{target_event.service}"
            
            # Cache lookup
            cached_strength = self.pattern_cache.get(pattern_key)
            if cached_strength is not None:
                return cached_strength
            
            # Requête historique
            query = """
                SELECT COUNT(*) as pattern_count
                FROM correlation_patterns
                WHERE source_service = $1 
                AND target_service = $2
                AND pattern_type = 'temporal'
                AND confidence > 0.6
            """
            
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval(query, source_event.service, target_event.service)
            
            pattern_count = result or 0
            
            # Normalisation (plus de patterns = plus forte)
            strength = min(1.0, pattern_count / 10.0)
            
            # Cache pour optimisation
            self.pattern_cache[pattern_key] = strength
            
            return strength
            
        except Exception as e:
            logger.error(f"Erreur calcul pattern temporel: {e}")
            return 0.5
    
    async def _calculate_temporal_confidence(
        self, 
        source_event: CorrelationEvent, 
        target_event: CorrelationEvent,
        time_diff: float
    ) -> float:
        """Calcul de confiance temporelle"""
        try:
            # Analyse de la distribution temporelle historique
            query = """
                SELECT AVG(time_difference) as avg_diff,
                       STDDEV(time_difference) as std_diff,
                       COUNT(*) as occurrence_count
                FROM temporal_correlations
                WHERE source_service = $1 
                AND target_service = $2
                AND ABS(time_difference - $3) < 300
            """
            
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow(
                    query, source_event.service, target_event.service, time_diff
                )
            
            if result and result['occurrence_count'] > 5:
                avg_diff = result['avg_diff'] or time_diff
                std_diff = result['std_diff'] or 60  # 1 minute par défaut
                
                # Calcul de probabilité basée sur distribution normale
                z_score = abs(time_diff - avg_diff) / max(std_diff, 1)
                confidence = max(0.1, 1 - (z_score / 3))  # 3-sigma rule
                
                # Bonus pour fréquence élevée
                frequency_bonus = min(0.2, result['occurrence_count'] / 100)
                confidence += frequency_bonus
                
                return min(1.0, confidence)
            
            return 0.5  # Confiance par défaut
            
        except Exception as e:
            logger.error(f"Erreur calcul confiance temporelle: {e}")
            return 0.5
    
    def _calculate_label_similarity(
        self, 
        labels1: Dict[str, str], 
        labels2: Dict[str, str]
    ) -> float:
        """Calcul de similarité des labels"""
        if not labels1 or not labels2:
            return 0.0
        
        keys1 = set(labels1.keys())
        keys2 = set(labels2.keys())
        
        # Similarité Jaccard des clés
        key_similarity = len(keys1 & keys2) / len(keys1 | keys2) if keys1 | keys2 else 0
        
        # Similarité des valeurs pour les clés communes
        common_keys = keys1 & keys2
        value_similarity = 0.0
        
        if common_keys:
            matches = sum(1 for key in common_keys if labels1[key] == labels2[key])
            value_similarity = matches / len(common_keys)
        
        return (key_similarity + value_similarity) / 2
    
    def _calculate_metric_similarity(
        self, 
        metrics1: Dict[str, float], 
        metrics2: Dict[str, float]
    ) -> float:
        """Calcul de similarité des métriques"""
        if not metrics1 or not metrics2:
            return 0.0
        
        common_metrics = set(metrics1.keys()) & set(metrics2.keys())
        if not common_metrics:
            return 0.0
        
        # Corrélation de Pearson simplifiée
        values1 = [metrics1[key] for key in common_metrics]
        values2 = [metrics2[key] for key in common_metrics]
        
        try:
            correlation = np.corrcoef(values1, values2)[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _extract_common_keywords(self, text1: str, text2: str) -> List[str]:
        """Extraction de mots-clés communs"""
        try:
            # Tokenisation simple
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            # Mots communs, filtrage des mots vides
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            common_words = (words1 & words2) - stop_words
            
            return list(common_words)[:10]  # Top 10
            
        except Exception as e:
            logger.error(f"Erreur extraction mots-clés: {e}")
            return []
    
    async def _get_service_dependencies(self, service: str) -> Dict[str, List[str]]:
        """Récupération des dépendances de service"""
        try:
            cache_key = f"dependencies:{service}"
            cached_deps = await self.redis_client.get(cache_key)
            
            if cached_deps:
                return eval(cached_deps)  # En production, utiliser JSON
            
            # Requête des dépendances
            query = """
                SELECT 
                    array_agg(DISTINCT target_service) FILTER (WHERE direction = 'downstream') as downstream,
                    array_agg(DISTINCT source_service) FILTER (WHERE direction = 'upstream') as upstream
                FROM (
                    SELECT target_service, 'downstream' as direction
                    FROM service_dependencies WHERE source_service = $1
                    UNION ALL
                    SELECT source_service, 'upstream' as direction
                    FROM service_dependencies WHERE target_service = $1
                ) deps
            """
            
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow(query, service)
            
            dependencies = {
                'downstream': result['downstream'] or [],
                'upstream': result['upstream'] or []
            }
            
            # Cache pour 1 heure
            await self.redis_client.setex(cache_key, 3600, str(dependencies))
            
            return dependencies
            
        except Exception as e:
            logger.error(f"Erreur récupération dépendances: {e}")
            return {'downstream': [], 'upstream': []}
    
    async def _analyze_error_propagation(
        self, 
        source_event: CorrelationEvent, 
        target_event: CorrelationEvent
    ) -> float:
        """Analyse de propagation d'erreur"""
        try:
            propagation_score = 0.0
            
            # Patterns de propagation connus
            error_keywords = ['error', 'exception', 'failure', 'timeout', 'unavailable']
            
            source_has_error = any(keyword in source_event.message.lower() for keyword in error_keywords)
            target_has_error = any(keyword in target_event.message.lower() for keyword in error_keywords)
            
            if source_has_error and target_has_error:
                propagation_score += 0.5
            
            # Analyse de la gravité (escalade d'erreur)
            severity_order = {'info': 1, 'low': 2, 'medium': 3, 'high': 4, 'critical': 5}
            source_severity = severity_order.get(source_event.severity.lower(), 3)
            target_severity = severity_order.get(target_event.severity.lower(), 3)
            
            if target_severity > source_severity:
                propagation_score += 0.3  # Escalade de gravité
            
            # Analyse des codes d'erreur
            if 'error_code' in source_event.labels and 'error_code' in target_event.labels:
                if source_event.labels['error_code'] == target_event.labels['error_code']:
                    propagation_score += 0.4
            
            return min(1.0, propagation_score)
            
        except Exception as e:
            logger.error(f"Erreur analyse propagation: {e}")
            return 0.0
    
    async def _analyze_metric_correlation(
        self, 
        source_event: CorrelationEvent, 
        target_event: CorrelationEvent
    ) -> float:
        """Analyse de corrélation des métriques"""
        try:
            if not source_event.metrics or not target_event.metrics:
                return 0.0
            
            # Métriques communes
            common_metrics = set(source_event.metrics.keys()) & set(target_event.metrics.keys())
            
            if not common_metrics:
                return 0.0
            
            correlations = []
            
            for metric in common_metrics:
                # Récupération de l'historique de cette métrique
                historical_correlation = await self._get_historical_metric_correlation(
                    source_event.service, target_event.service, metric
                )
                
                # Différence actuelle
                source_value = source_event.metrics[metric]
                target_value = target_event.metrics[metric]
                
                # Score basé sur corrélation historique et valeurs actuelles
                if historical_correlation > 0.5:
                    # Vérification si les valeurs suivent le pattern historique
                    value_consistency = self._check_value_consistency(
                        source_value, target_value, historical_correlation
                    )
                    correlations.append(historical_correlation * value_consistency)
                else:
                    correlations.append(0.0)
            
            return np.mean(correlations) if correlations else 0.0
            
        except Exception as e:
            logger.error(f"Erreur analyse corrélation métriques: {e}")
            return 0.0
    
    def _check_value_consistency(
        self, 
        source_value: float, 
        target_value: float, 
        expected_correlation: float
    ) -> float:
        """Vérification de cohérence des valeurs"""
        try:
            # Logique simplifiée de cohérence
            if expected_correlation > 0:
                # Corrélation positive attendue
                if (source_value > 0 and target_value > 0) or (source_value < 0 and target_value < 0):
                    return 1.0
                else:
                    return 0.3
            else:
                # Corrélation négative attendue
                if (source_value > 0 and target_value < 0) or (source_value < 0 and target_value > 0):
                    return 1.0
                else:
                    return 0.3
        except:
            return 0.5
    
    async def _get_historical_metric_correlation(
        self, 
        source_service: str, 
        target_service: str, 
        metric: str
    ) -> float:
        """Récupération de corrélation métrique historique"""
        try:
            cache_key = f"metric_corr:{source_service}:{target_service}:{metric}"
            cached_corr = await self.redis_client.get(cache_key)
            
            if cached_corr:
                return float(cached_corr)
            
            # Simulation de corrélation historique
            # En production, calculer depuis les données historiques
            correlation = np.random.uniform(0.3, 0.9)
            
            # Cache pour 2 heures
            await self.redis_client.setex(cache_key, 7200, str(correlation))
            
            return correlation
            
        except Exception as e:
            logger.error(f"Erreur récupération corrélation métrique: {e}")
            return 0.5
    
    def _score_to_strength(self, score: float) -> CorrelationStrength:
        """Conversion score vers force de corrélation"""
        if score >= 0.9:
            return CorrelationStrength.VERY_STRONG
        elif score >= 0.7:
            return CorrelationStrength.STRONG
        elif score >= 0.5:
            return CorrelationStrength.MODERATE
        elif score >= 0.3:
            return CorrelationStrength.WEAK
        else:
            return CorrelationStrength.VERY_WEAK
    
    # Continuation des méthodes dans la partie 2...
