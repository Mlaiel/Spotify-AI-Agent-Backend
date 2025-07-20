"""
Moteur de corrélation avancé pour l'analyse des relations entre métriques et événements.

Ce module implémente des algorithmes sophistiqués pour :
- Corrélation temporelle et causale entre métriques
- Détection de patterns d'incidents liés
- Analyse de propagation des problèmes
- Clustering intelligent d'événements
- Prédiction de cascades d'alertes

Optimisé pour la production avec traitement en temps réel et analyse historique.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import asyncio
import networkx as nx
from collections import defaultdict, deque
import json
import redis
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import numpy.linalg as la

# Métriques et monitoring
from prometheus_client import Counter, Histogram, Gauge, Summary

logger = logging.getLogger(__name__)

# Métriques Prometheus
CORRELATION_CALCULATIONS = Counter('correlation_calculations_total', 'Total correlation calculations performed')
CORRELATION_LATENCY = Histogram('correlation_calculation_duration_seconds', 'Time spent calculating correlations')
ACTIVE_CORRELATIONS = Gauge('active_correlations_count', 'Number of active correlations detected')
CAUSALITY_DETECTIONS = Counter('causality_detections_total', 'Total causality relationships detected')
EVENT_CLUSTERS = Gauge('event_clusters_count', 'Number of event clusters detected')

@dataclass
class MetricPoint:
    """Point de métrique avec timestamp."""
    timestamp: datetime
    value: float
    metric_name: str
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class CorrelationResult:
    """Résultat de corrélation entre deux métriques."""
    metric_a: str
    metric_b: str
    correlation_coefficient: float
    correlation_type: str  # 'pearson', 'spearman', 'kendall'
    p_value: float
    confidence_level: float
    time_lag: timedelta
    causality_direction: Optional[str] = None  # 'a_causes_b', 'b_causes_a', 'bidirectional', 'none'
    strength: str = field(default="weak")  # 'weak', 'moderate', 'strong', 'very_strong'

@dataclass
class EventCluster:
    """Cluster d'événements corrélés."""
    cluster_id: str
    events: List[Dict[str, Any]]
    centroid_time: datetime
    duration: timedelta
    affected_systems: Set[str]
    root_cause_likelihood: Dict[str, float]
    propagation_path: List[str]
    severity_score: float

@dataclass
class CausalityRelation:
    """Relation de causalité entre événements."""
    cause_event: str
    effect_event: str
    delay: timedelta
    confidence: float
    evidence_strength: float
    historical_occurrences: int

class CorrelationEngine:
    """Moteur principal de corrélation pour l'analyse des métriques."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.correlation_cache = {}
        self.metric_history = defaultdict(list)
        self.correlation_graph = nx.DiGraph()
        
    def _default_config(self) -> Dict[str, Any]:
        """Configuration par défaut du moteur de corrélation."""
        return {
            'window_size': timedelta(hours=1),
            'min_correlation_threshold': 0.3,
            'max_lag_minutes': 30,
            'cache_ttl_seconds': 300,
            'correlation_methods': ['pearson', 'spearman'],
            'causality_methods': ['granger', 'transfer_entropy'],
            'clustering_algorithms': ['dbscan', 'kmeans'],
            'min_samples_for_correlation': 10
        }
    
    def add_metric_point(self, metric_point: MetricPoint) -> None:
        """Ajoute un point de métrique à l'historique."""
        metric_name = metric_point.metric_name
        
        # Ajout à l'historique local
        self.metric_history[metric_name].append(metric_point)
        
        # Nettoyage des anciennes données
        cutoff_time = datetime.now() - self.config['window_size']
        self.metric_history[metric_name] = [
            point for point in self.metric_history[metric_name]
            if point.timestamp > cutoff_time
        ]
        
        # Mise à jour du cache Redis
        self._update_redis_cache(metric_point)
    
    def _update_redis_cache(self, metric_point: MetricPoint) -> None:
        """Met à jour le cache Redis avec le nouveau point."""
        try:
            key = f"metric_history:{metric_point.metric_name}"
            value = {
                'timestamp': metric_point.timestamp.isoformat(),
                'value': metric_point.value,
                'tags': metric_point.tags
            }
            
            # Ajout à une liste Redis avec TTL
            self.redis_client.lpush(key, json.dumps(value))
            self.redis_client.ltrim(key, 0, 1000)  # Garde seulement les 1000 derniers points
            self.redis_client.expire(key, self.config['cache_ttl_seconds'])
            
        except Exception as e:
            logger.warning(f"Failed to update Redis cache: {e}")
    
    def calculate_correlation(self, metric_a: str, metric_b: str, 
                            method: str = 'pearson') -> Optional[CorrelationResult]:
        """Calcule la corrélation entre deux métriques."""
        
        with CORRELATION_LATENCY.time():
            try:
                # Récupération des données
                data_a = self._get_metric_values(metric_a)
                data_b = self._get_metric_values(metric_b)
                
                if len(data_a) < self.config['min_samples_for_correlation'] or \
                   len(data_b) < self.config['min_samples_for_correlation']:
                    return None
                
                # Alignement temporel des données
                aligned_a, aligned_b = self._align_time_series(data_a, data_b)
                
                if len(aligned_a) < self.config['min_samples_for_correlation']:
                    return None
                
                # Calcul de la corrélation
                correlation_coeff, p_value = self._compute_correlation(
                    aligned_a, aligned_b, method
                )
                
                if abs(correlation_coeff) < self.config['min_correlation_threshold']:
                    return None
                
                # Détection du lag optimal
                time_lag = self._detect_optimal_lag(data_a, data_b)
                
                # Analyse de causalité
                causality_direction = self._analyze_causality(data_a, data_b, time_lag)
                
                # Détermination de la force de corrélation
                strength = self._determine_correlation_strength(abs(correlation_coeff))
                
                # Calcul du niveau de confiance
                confidence_level = 1 - p_value
                
                result = CorrelationResult(
                    metric_a=metric_a,
                    metric_b=metric_b,
                    correlation_coefficient=correlation_coeff,
                    correlation_type=method,
                    p_value=p_value,
                    confidence_level=confidence_level,
                    time_lag=time_lag,
                    causality_direction=causality_direction,
                    strength=strength
                )
                
                # Mise à jour des métriques
                CORRELATION_CALCULATIONS.inc()
                
                # Mise à jour du graphe de corrélation
                self._update_correlation_graph(result)
                
                return result
                
            except Exception as e:
                logger.error(f"Error calculating correlation: {e}")
                return None
    
    def _get_metric_values(self, metric_name: str) -> List[Tuple[datetime, float]]:
        """Récupère les valeurs d'une métrique avec timestamps."""
        if metric_name in self.metric_history:
            return [(point.timestamp, point.value) for point in self.metric_history[metric_name]]
        
        # Fallback sur Redis
        try:
            key = f"metric_history:{metric_name}"
            cached_data = self.redis_client.lrange(key, 0, -1)
            
            result = []
            for item in cached_data:
                data = json.loads(item)
                timestamp = datetime.fromisoformat(data['timestamp'])
                value = data['value']
                result.append((timestamp, value))
                
            return sorted(result, key=lambda x: x[0])
            
        except Exception as e:
            logger.warning(f"Failed to retrieve from Redis: {e}")
            return []
    
    def _align_time_series(self, data_a: List[Tuple[datetime, float]], 
                          data_b: List[Tuple[datetime, float]]) -> Tuple[List[float], List[float]]:
        """Aligne deux séries temporelles sur les mêmes timestamps."""
        
        # Conversion en DataFrames pour faciliter l'alignement
        df_a = pd.DataFrame(data_a, columns=['timestamp', 'value_a'])
        df_b = pd.DataFrame(data_b, columns=['timestamp', 'value_b'])
        
        # Fusion sur les timestamps avec interpolation
        merged = pd.merge_asof(
            df_a.sort_values('timestamp'),
            df_b.sort_values('timestamp'),
            on='timestamp',
            tolerance=pd.Timedelta(minutes=5),
            direction='nearest'
        )
        
        # Suppression des valeurs manquantes
        merged = merged.dropna()
        
        return merged['value_a'].tolist(), merged['value_b'].tolist()
    
    def _compute_correlation(self, values_a: List[float], values_b: List[float], 
                           method: str) -> Tuple[float, float]:
        """Calcule le coefficient de corrélation et la p-value."""
        
        if method == 'pearson':
            coeff, p_value = pearsonr(values_a, values_b)
        elif method == 'spearman':
            coeff, p_value = spearmanr(values_a, values_b)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        return coeff, p_value
    
    def _detect_optimal_lag(self, data_a: List[Tuple[datetime, float]], 
                           data_b: List[Tuple[datetime, float]]) -> timedelta:
        """Détecte le lag optimal entre deux séries temporelles."""
        
        max_lag_minutes = self.config['max_lag_minutes']
        best_lag = timedelta(0)
        best_correlation = 0
        
        # Test de différents lags
        for lag_minutes in range(0, max_lag_minutes + 1, 5):
            lag = timedelta(minutes=lag_minutes)
            
            # Décalage de la série B
            shifted_data_b = [(ts + lag, value) for ts, value in data_b]
            
            # Alignement et calcul de corrélation
            aligned_a, aligned_b = self._align_time_series(data_a, shifted_data_b)
            
            if len(aligned_a) >= self.config['min_samples_for_correlation']:
                try:
                    corr, _ = pearsonr(aligned_a, aligned_b)
                    if abs(corr) > abs(best_correlation):
                        best_correlation = corr
                        best_lag = lag
                except:
                    continue
        
        return best_lag
    
    def _analyze_causality(self, data_a: List[Tuple[datetime, float]], 
                          data_b: List[Tuple[datetime, float]], 
                          time_lag: timedelta) -> Optional[str]:
        """Analyse la direction de causalité entre deux métriques."""
        
        try:
            # Granger causality test simplifié
            causality_a_to_b = self._granger_causality_test(data_a, data_b, time_lag)
            causality_b_to_a = self._granger_causality_test(data_b, data_a, time_lag)
            
            if causality_a_to_b > 0.05 and causality_b_to_a > 0.05:
                return "none"
            elif causality_a_to_b <= 0.05 and causality_b_to_a > 0.05:
                return "a_causes_b"
            elif causality_b_to_a <= 0.05 and causality_a_to_b > 0.05:
                return "b_causes_a"
            else:
                return "bidirectional"
                
        except Exception as e:
            logger.warning(f"Causality analysis failed: {e}")
            return None
    
    def _granger_causality_test(self, cause_data: List[Tuple[datetime, float]], 
                               effect_data: List[Tuple[datetime, float]], 
                               lag: timedelta) -> float:
        """Test de causalité de Granger simplifié."""
        
        # Alignement des données avec lag
        shifted_cause = [(ts + lag, value) for ts, value in cause_data]
        aligned_cause, aligned_effect = self._align_time_series(shifted_cause, effect_data)
        
        if len(aligned_cause) < 10:
            return 1.0  # Pas assez de données
        
        # Régression linéaire simple pour tester la causalité
        cause_array = np.array(aligned_cause).reshape(-1, 1)
        effect_array = np.array(aligned_effect)
        
        # Calcul du R² pour déterminer si la cause explique l'effet
        correlation = np.corrcoef(cause_array.flatten(), effect_array)[0, 1]
        r_squared = correlation ** 2
        
        # Conversion en p-value approximative
        n = len(aligned_cause)
        t_stat = correlation * np.sqrt((n - 2) / (1 - r_squared))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        return p_value
    
    def _determine_correlation_strength(self, abs_correlation: float) -> str:
        """Détermine la force de la corrélation."""
        if abs_correlation >= 0.8:
            return "very_strong"
        elif abs_correlation >= 0.6:
            return "strong"
        elif abs_correlation >= 0.4:
            return "moderate"
        else:
            return "weak"
    
    def _update_correlation_graph(self, correlation: CorrelationResult) -> None:
        """Met à jour le graphe de corrélations."""
        
        if correlation.causality_direction == "a_causes_b":
            self.correlation_graph.add_edge(
                correlation.metric_a, 
                correlation.metric_b,
                weight=abs(correlation.correlation_coefficient),
                lag=correlation.time_lag.total_seconds(),
                confidence=correlation.confidence_level
            )
        elif correlation.causality_direction == "b_causes_a":
            self.correlation_graph.add_edge(
                correlation.metric_b, 
                correlation.metric_a,
                weight=abs(correlation.correlation_coefficient),
                lag=correlation.time_lag.total_seconds(),
                confidence=correlation.confidence_level
            )
        elif correlation.causality_direction == "bidirectional":
            self.correlation_graph.add_edge(
                correlation.metric_a, 
                correlation.metric_b,
                weight=abs(correlation.correlation_coefficient),
                lag=correlation.time_lag.total_seconds(),
                confidence=correlation.confidence_level
            )
            self.correlation_graph.add_edge(
                correlation.metric_b, 
                correlation.metric_a,
                weight=abs(correlation.correlation_coefficient),
                lag=correlation.time_lag.total_seconds(),
                confidence=correlation.confidence_level
            )
    
    def find_related_metrics(self, metric_name: str, max_depth: int = 3) -> List[str]:
        """Trouve les métriques liées à une métrique donnée."""
        
        if metric_name not in self.correlation_graph:
            return []
        
        # Recherche en largeur dans le graphe
        visited = set()
        queue = deque([(metric_name, 0)])
        related_metrics = []
        
        while queue:
            current_metric, depth = queue.popleft()
            
            if depth >= max_depth or current_metric in visited:
                continue
                
            visited.add(current_metric)
            
            if current_metric != metric_name:
                related_metrics.append(current_metric)
            
            # Ajout des voisins
            for neighbor in self.correlation_graph.neighbors(current_metric):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
        
        return related_metrics
    
    def get_correlation_matrix(self, metric_names: List[str]) -> pd.DataFrame:
        """Génère une matrice de corrélation pour un ensemble de métriques."""
        
        n_metrics = len(metric_names)
        correlation_matrix = np.eye(n_metrics)
        
        for i, metric_a in enumerate(metric_names):
            for j, metric_b in enumerate(metric_names):
                if i != j:
                    correlation = self.calculate_correlation(metric_a, metric_b)
                    if correlation:
                        correlation_matrix[i, j] = correlation.correlation_coefficient
        
        return pd.DataFrame(
            correlation_matrix,
            index=metric_names,
            columns=metric_names
        )

class EventCorrelationAnalyzer:
    """Analyseur de corrélation pour les événements et alertes."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'time_window': timedelta(minutes=30),
            'similarity_threshold': 0.7,
            'min_cluster_size': 2,
            'max_cluster_distance': 600  # secondes
        }
        self.event_history = []
        self.clusters = {}
        
    def add_event(self, event: Dict[str, Any]) -> None:
        """Ajoute un événement à l'historique."""
        
        # Ajout du timestamp si absent
        if 'timestamp' not in event:
            event['timestamp'] = datetime.now()
        
        self.event_history.append(event)
        
        # Nettoyage des anciens événements
        cutoff_time = datetime.now() - self.config['time_window']
        self.event_history = [
            e for e in self.event_history 
            if e['timestamp'] > cutoff_time
        ]
        
        # Déclenchement de l'analyse de clustering
        self._analyze_event_clusters()
    
    def _analyze_event_clusters(self) -> None:
        """Analyse les clusters d'événements corrélés."""
        
        if len(self.event_history) < self.config['min_cluster_size']:
            return
        
        # Extraction des features d'événements
        features, event_indices = self._extract_event_features()
        
        if len(features) < self.config['min_cluster_size']:
            return
        
        # Clustering DBSCAN
        try:
            clustering = DBSCAN(
                eps=self.config['max_cluster_distance'],
                min_samples=self.config['min_cluster_size'],
                metric='euclidean'
            ).fit(features)
            
            # Création des clusters
            unique_labels = set(clustering.labels_)
            
            for label in unique_labels:
                if label == -1:  # Bruit
                    continue
                
                cluster_indices = [i for i, l in enumerate(clustering.labels_) if l == label]
                cluster_events = [self.event_history[event_indices[i]] for i in cluster_indices]
                
                cluster = self._create_event_cluster(cluster_events, f"cluster_{label}")
                self.clusters[cluster.cluster_id] = cluster
                
                EVENT_CLUSTERS.set(len(self.clusters))
                
        except Exception as e:
            logger.error(f"Error in event clustering: {e}")
    
    def _extract_event_features(self) -> Tuple[np.ndarray, List[int]]:
        """Extrait les features des événements pour le clustering."""
        
        features = []
        indices = []
        
        for i, event in enumerate(self.event_history):
            try:
                feature_vector = [
                    # Features temporelles
                    event['timestamp'].hour,
                    event['timestamp'].minute,
                    event['timestamp'].weekday(),
                    
                    # Features d'impact
                    event.get('severity_score', 0.5),
                    event.get('user_impact', 0),
                    len(event.get('affected_services', [])),
                    
                    # Features de contenu (hash des chaînes principales)
                    hash(event.get('alert_name', '')) % 1000,
                    hash(event.get('source_system', '')) % 1000,
                    hash(event.get('category', '')) % 100,
                ]
                
                features.append(feature_vector)
                indices.append(i)
                
            except Exception as e:
                logger.warning(f"Failed to extract features for event {i}: {e}")
                continue
        
        return np.array(features), indices
    
    def _create_event_cluster(self, events: List[Dict[str, Any]], cluster_id: str) -> EventCluster:
        """Crée un cluster d'événements."""
        
        # Calcul du centroïde temporel
        timestamps = [e['timestamp'] for e in events]
        centroid_time = min(timestamps) + (max(timestamps) - min(timestamps)) / 2
        
        # Calcul de la durée
        duration = max(timestamps) - min(timestamps)
        
        # Systèmes affectés
        affected_systems = set()
        for event in events:
            affected_systems.update(event.get('affected_services', []))
            if 'source_system' in event:
                affected_systems.add(event['source_system'])
        
        # Analyse de la cause racine probable
        root_cause_likelihood = self._analyze_root_cause(events)
        
        # Chemin de propagation
        propagation_path = self._analyze_propagation_path(events)
        
        # Score de sévérité
        severity_scores = [e.get('severity_score', 0.5) for e in events]
        severity_score = max(severity_scores) if severity_scores else 0.5
        
        return EventCluster(
            cluster_id=cluster_id,
            events=events,
            centroid_time=centroid_time,
            duration=duration,
            affected_systems=affected_systems,
            root_cause_likelihood=root_cause_likelihood,
            propagation_path=propagation_path,
            severity_score=severity_score
        )
    
    def _analyze_root_cause(self, events: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyse la probabilité de cause racine pour chaque événement."""
        
        # Tri par timestamp
        sorted_events = sorted(events, key=lambda x: x['timestamp'])
        
        root_cause_scores = {}
        
        for i, event in enumerate(sorted_events):
            event_id = event.get('alert_id', f"event_{i}")
            
            # Score basé sur l'ordre temporel (premier = plus probable)
            temporal_score = (len(sorted_events) - i) / len(sorted_events)
            
            # Score basé sur la sévérité
            severity_score = event.get('severity_score', 0.5)
            
            # Score basé sur l'impact
            impact_score = min(1.0, len(event.get('affected_services', [])) / 5)
            
            # Score composite
            composite_score = (temporal_score * 0.5 + severity_score * 0.3 + impact_score * 0.2)
            
            root_cause_scores[event_id] = composite_score
        
        return root_cause_scores
    
    def _analyze_propagation_path(self, events: List[Dict[str, Any]]) -> List[str]:
        """Analyse le chemin de propagation des événements."""
        
        # Tri par timestamp
        sorted_events = sorted(events, key=lambda x: x['timestamp'])
        
        propagation_path = []
        
        for event in sorted_events:
            if 'source_system' in event:
                if event['source_system'] not in propagation_path:
                    propagation_path.append(event['source_system'])
            
            # Ajout des services affectés
            for service in event.get('affected_services', []):
                if service not in propagation_path:
                    propagation_path.append(service)
        
        return propagation_path
    
    def get_active_clusters(self) -> List[EventCluster]:
        """Retourne les clusters actifs."""
        
        current_time = datetime.now()
        active_clusters = []
        
        for cluster in self.clusters.values():
            # Cluster actif si événements récents
            latest_event_time = max(e['timestamp'] for e in cluster.events)
            if current_time - latest_event_time <= self.config['time_window']:
                active_clusters.append(cluster)
        
        return active_clusters
    
    def predict_next_events(self, cluster: EventCluster) -> List[Dict[str, Any]]:
        """Prédit les prochains événements probables dans un cluster."""
        
        predictions = []
        
        # Analyse des patterns historiques
        pattern_analysis = self._analyze_historical_patterns(cluster)
        
        for pattern in pattern_analysis:
            if pattern['confidence'] > 0.6:
                prediction = {
                    'predicted_event_type': pattern['event_type'],
                    'predicted_time': datetime.now() + pattern['expected_delay'],
                    'confidence': pattern['confidence'],
                    'predicted_systems': pattern['likely_systems']
                }
                predictions.append(prediction)
        
        return predictions
    
    def _analyze_historical_patterns(self, cluster: EventCluster) -> List[Dict[str, Any]]:
        """Analyse les patterns historiques pour prédiction."""
        
        # Simulation de patterns basés sur les données du cluster
        patterns = []
        
        # Pattern 1: Propagation vers services dépendants
        if 'database' in cluster.affected_systems:
            patterns.append({
                'event_type': 'downstream_service_failure',
                'expected_delay': timedelta(minutes=5),
                'confidence': 0.7,
                'likely_systems': ['api-gateway', 'user-service']
            })
        
        # Pattern 2: Escalade de sévérité
        if cluster.severity_score < 0.8:
            patterns.append({
                'event_type': 'severity_escalation',
                'expected_delay': timedelta(minutes=10),
                'confidence': 0.6,
                'likely_systems': list(cluster.affected_systems)
            })
        
        return patterns

class CausalityDetector:
    """Détecteur de relations de causalité avancé."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'min_confidence': 0.6,
            'max_delay': timedelta(hours=1),
            'evidence_threshold': 3
        }
        self.causality_rules = []
        self.evidence_database = defaultdict(list)
        
    def learn_causality_rule(self, cause_pattern: str, effect_pattern: str, 
                           delay: timedelta, evidence: List[Dict[str, Any]]) -> None:
        """Apprend une nouvelle règle de causalité."""
        
        confidence = self._calculate_rule_confidence(evidence)
        
        if confidence >= self.config['min_confidence']:
            rule = CausalityRelation(
                cause_event=cause_pattern,
                effect_event=effect_pattern,
                delay=delay,
                confidence=confidence,
                evidence_strength=len(evidence),
                historical_occurrences=len(evidence)
            )
            
            self.causality_rules.append(rule)
            CAUSALITY_DETECTIONS.inc()
            
            logger.info(f"Learned causality rule: {cause_pattern} -> {effect_pattern} "
                       f"(delay: {delay}, confidence: {confidence:.2f})")
    
    def _calculate_rule_confidence(self, evidence: List[Dict[str, Any]]) -> float:
        """Calcule la confiance d'une règle basée sur l'évidence."""
        
        if len(evidence) < self.config['evidence_threshold']:
            return 0.0
        
        # Confiance basée sur la fréquence et la consistance
        total_cases = len(evidence)
        consistent_cases = sum(1 for e in evidence if e.get('consistent', True))
        
        base_confidence = consistent_cases / total_cases
        
        # Bonus pour nombre d'observations
        observation_bonus = min(0.2, (total_cases - self.config['evidence_threshold']) / 100)
        
        return min(1.0, base_confidence + observation_bonus)
    
    def detect_causality(self, events: List[Dict[str, Any]]) -> List[CausalityRelation]:
        """Détecte les relations de causalité dans une séquence d'événements."""
        
        detected_relations = []
        
        # Tri des événements par timestamp
        sorted_events = sorted(events, key=lambda x: x.get('timestamp', datetime.now()))
        
        # Recherche de patterns causals
        for i, cause_event in enumerate(sorted_events):
            for j, effect_event in enumerate(sorted_events[i+1:], i+1):
                
                delay = effect_event['timestamp'] - cause_event['timestamp']
                
                if delay > self.config['max_delay']:
                    continue
                
                # Vérification des règles connues
                for rule in self.causality_rules:
                    if self._match_event_pattern(cause_event, rule.cause_event) and \
                       self._match_event_pattern(effect_event, rule.effect_event):
                        
                        # Vérification du délai
                        if abs(delay - rule.delay) <= timedelta(minutes=5):
                            detected_relations.append(rule)
                            break
        
        return detected_relations
    
    def _match_event_pattern(self, event: Dict[str, Any], pattern: str) -> bool:
        """Vérifie si un événement correspond à un pattern."""
        
        # Matching simple basé sur des mots-clés
        event_text = f"{event.get('alert_name', '')} {event.get('description', '')} {event.get('source_system', '')}"
        
        # Pattern matching flexible
        pattern_words = pattern.lower().split()
        event_words = event_text.lower().split()
        
        matches = sum(1 for word in pattern_words if word in event_words)
        return matches / len(pattern_words) >= 0.6
    
    def get_causality_graph(self) -> nx.DiGraph:
        """Construit un graphe de causalité basé sur les règles apprises."""
        
        graph = nx.DiGraph()
        
        for rule in self.causality_rules:
            graph.add_edge(
                rule.cause_event,
                rule.effect_event,
                weight=rule.confidence,
                delay=rule.delay.total_seconds(),
                evidence=rule.evidence_strength
            )
        
        return graph
    
    def predict_cascade_effects(self, initial_event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prédit les effets en cascade d'un événement initial."""
        
        predictions = []
        
        # Recherche des règles applicables
        for rule in self.causality_rules:
            if self._match_event_pattern(initial_event, rule.cause_event):
                
                predicted_time = initial_event['timestamp'] + rule.delay
                
                prediction = {
                    'predicted_event_pattern': rule.effect_event,
                    'predicted_time': predicted_time,
                    'confidence': rule.confidence,
                    'causal_rule': rule,
                    'initial_event': initial_event['alert_id']
                }
                
                predictions.append(prediction)
        
        return sorted(predictions, key=lambda x: x['predicted_time'])

# Factory et utilitaires

class MetricCorrelationEngine:
    """Moteur principal combinant tous les analyseurs de corrélation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.correlation_engine = CorrelationEngine(config)
        self.event_analyzer = EventCorrelationAnalyzer(config)
        self.causality_detector = CausalityDetector(config)
        
    def add_metric_point(self, metric_point: MetricPoint) -> None:
        """Ajoute un point de métrique."""
        self.correlation_engine.add_metric_point(metric_point)
    
    def add_event(self, event: Dict[str, Any]) -> None:
        """Ajoute un événement."""
        self.event_analyzer.add_event(event)
    
    def get_comprehensive_analysis(self, metric_name: str) -> Dict[str, Any]:
        """Analyse complète d'une métrique."""
        
        analysis = {
            'metric_name': metric_name,
            'timestamp': datetime.now(),
            'related_metrics': self.correlation_engine.find_related_metrics(metric_name),
            'active_clusters': self.event_analyzer.get_active_clusters(),
            'causality_rules': self.causality_detector.causality_rules,
            'correlation_graph_stats': self._get_graph_stats()
        }
        
        return analysis
    
    def _get_graph_stats(self) -> Dict[str, Any]:
        """Statistiques du graphe de corrélation."""
        
        graph = self.correlation_engine.correlation_graph
        
        return {
            'nodes_count': graph.number_of_nodes(),
            'edges_count': graph.number_of_edges(),
            'density': nx.density(graph),
            'average_clustering': nx.average_clustering(graph.to_undirected()) if graph.number_of_nodes() > 0 else 0
        }
    
    async def real_time_correlation_analysis(self, metric_points: List[MetricPoint]) -> Dict[str, Any]:
        """Analyse de corrélation en temps réel."""
        
        # Ajout des points de métrique
        for point in metric_points:
            self.add_metric_point(point)
        
        # Analyse asynchrone des corrélations
        tasks = []
        metric_names = list(set(point.metric_name for point in metric_points))
        
        for i, metric_a in enumerate(metric_names):
            for metric_b in metric_names[i+1:]:
                task = asyncio.create_task(
                    asyncio.to_thread(
                        self.correlation_engine.calculate_correlation, 
                        metric_a, metric_b
                    )
                )
                tasks.append(task)
        
        # Collecte des résultats
        correlations = []
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                if result:
                    correlations.append(result)
            except Exception as e:
                logger.warning(f"Correlation calculation failed: {e}")
        
        return {
            'correlations_found': len(correlations),
            'correlations': correlations,
            'analysis_timestamp': datetime.now()
        }
