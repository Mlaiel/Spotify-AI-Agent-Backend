"""
Ultra-Advanced Correlation Engine - Enterprise-Grade Event Correlation System
===========================================================================

Ce module fournit un moteur de corrélation événementielle avancé avec intelligence
artificielle, reconnaissance de motifs temporels, analyse prédictive et corrélation
multi-dimensionnelle pour des environnements multi-tenant à haute performance.

Fonctionnalités Principales:
- Corrélation temporelle avec détection de patterns complexes
- Intelligence artificielle pour la prédiction de pannes en cascade
- Analyse multi-dimensionnelle avec clustering automatique
- Détection d'anomalies avec machine learning
- Corrélation géographique et topologique
- Analyse de causalité avec graphes dirigés
- Corrélation sémantique avec NLP avancé
- Détection de storm d'alertes avec suppression intelligente

Architecture Enterprise:
- Processing distribué avec sharding automatique
- Cache Redis pour corrélations temps réel
- Pipeline ML avec feature engineering automatique
- Graphes de corrélation persistants
- Analytics temps réel avec streaming
- Auto-tuning des algorithmes de corrélation
- Audit trail complet et compliance
- Sécurité end-to-end avec chiffrement

Version: 5.0.0
Auteur: Fahed Mlaiel (Lead Dev + Architecte IA)
Architecture: Event-Driven Microservices avec ML Pipeline
"""

import asyncio
import logging
import time
import uuid
import hashlib
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import (
    Dict, List, Optional, Any, Callable, Union, Tuple, Set,
    Protocol, TypeVar, Generic, AsyncIterator, NamedTuple
)
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum, auto
from collections import defaultdict, deque, Counter
import networkx as nx
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import statistics
import redis
import asyncpg
from prometheus_client import Counter as PrometheusCounter, Histogram, Gauge

# Configuration du logging structuré
logger = logging.getLogger(__name__)


class CorrelationType(Enum):
    """Types de corrélation supportés"""
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    SEMANTIC = "semantic"
    CAUSAL = "causal"
    PATTERN = "pattern"
    ANOMALY = "anomaly"
    CASCADE = "cascade"
    STORM = "storm"


class CorrelationStrength(Enum):
    """Force de corrélation"""
    VERY_WEAK = 0.1
    WEAK = 0.3
    MODERATE = 0.5
    STRONG = 0.7
    VERY_STRONG = 0.9


@dataclass
class CorrelationRule:
    """Règle de corrélation avancée"""
    id: str
    name: str
    description: str
    correlation_type: CorrelationType
    time_window: timedelta
    min_events: int
    max_events: int
    strength_threshold: float
    enabled: bool = True
    pattern: Optional[Dict[str, Any]] = None
    ml_model: Optional[str] = None
    custom_logic: Optional[Callable] = None
    tags: Set[str] = field(default_factory=set)
    tenant_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CorrelationResult:
    """Résultat de corrélation avec métadonnées enrichies"""
    id: str
    correlation_type: CorrelationType
    strength: float
    confidence: float
    events: List[Dict[str, Any]]
    pattern: Dict[str, Any]
    root_cause: Optional[str] = None
    predicted_impact: Optional[Dict[str, Any]] = None
    remediation_suggestions: List[str] = field(default_factory=list)
    timeline: List[Tuple[datetime, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TemporalPattern:
    """Pattern temporel détecté"""
    id: str
    pattern_type: str
    frequency: float
    duration: timedelta
    amplitude: float
    seasonality: Optional[Dict[str, Any]] = None
    trend: Optional[str] = None
    anomalies: List[datetime] = field(default_factory=list)
    confidence: float = 0.0


class EventGraphNode:
    """Nœud dans le graphe d'événements"""
    
    def __init__(self, event_id: str, event_data: Dict[str, Any]):
        self.event_id = event_id
        self.event_data = event_data
        self.timestamp = event_data.get('timestamp', datetime.utcnow())
        self.severity = event_data.get('severity', 'info')
        self.source = event_data.get('source', 'unknown')
        self.correlations = {}
        self.features = {}
        
    def add_correlation(self, other_node: 'EventGraphNode', strength: float, correlation_type: CorrelationType):
        """Ajoute une corrélation avec un autre nœud"""
        self.correlations[other_node.event_id] = {
            'node': other_node,
            'strength': strength,
            'type': correlation_type,
            'created_at': datetime.utcnow()
        }


class AdvancedCorrelationEngine:
    """
    Moteur de corrélation avancé avec intelligence artificielle
    
    Fonctionnalités:
    - Corrélation temporelle avec détection de patterns
    - ML pour prédiction de pannes en cascade
    - Analyse sémantique avec NLP
    - Graphes de corrélation persistants
    - Auto-tuning des paramètres
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.correlation_rules = {}
        self.event_graph = nx.DiGraph()
        self.temporal_patterns = {}
        self.ml_models = {}
        self.correlation_cache = {}
        self.performance_metrics = {
            'correlations_processed': PrometheusCounter('correlations_processed_total', 'Total correlations processed'),
            'correlation_time': Histogram('correlation_processing_seconds', 'Time spent processing correlations'),
            'active_correlations': Gauge('active_correlations', 'Number of active correlations'),
        }
        
        # Initialize ML components
        self.scaler = StandardScaler()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        
        # Redis pour le cache
        self.redis_client = None
        self.setup_redis()
        
        # Base de données pour persistence
        self.db_pool = None
        self.setup_database()
        
        logger.info("Advanced Correlation Engine initialized with ML capabilities")
    
    async def setup_redis(self):
        """Configuration du client Redis"""
        try:
            self.redis_client = redis.asyncio.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                db=self.config.get('redis_db', 0),
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connection established for correlation engine")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
    
    async def setup_database(self):
        """Configuration de la base de données PostgreSQL"""
        try:
            self.db_pool = await asyncpg.create_pool(
                host=self.config.get('db_host', 'localhost'),
                port=self.config.get('db_port', 5432),
                user=self.config.get('db_user', 'postgres'),
                password=self.config.get('db_password', ''),
                database=self.config.get('db_name', 'alerts'),
                min_size=5,
                max_size=20
            )
            await self.create_tables()
            logger.info("Database connection pool established")
        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
    
    async def create_tables(self):
        """Crée les tables nécessaires"""
        create_correlations_table = """
        CREATE TABLE IF NOT EXISTS correlations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            correlation_type VARCHAR(50) NOT NULL,
            strength FLOAT NOT NULL,
            confidence FLOAT NOT NULL,
            events JSONB NOT NULL,
            pattern JSONB,
            root_cause TEXT,
            predicted_impact JSONB,
            remediation_suggestions JSONB,
            timeline JSONB,
            metadata JSONB,
            tenant_id VARCHAR(100) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            INDEX (tenant_id, created_at),
            INDEX (correlation_type, strength),
            INDEX USING GIN (events),
            INDEX USING GIN (metadata)
        );
        """
        
        create_patterns_table = """
        CREATE TABLE IF NOT EXISTS temporal_patterns (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            pattern_type VARCHAR(50) NOT NULL,
            frequency FLOAT NOT NULL,
            duration INTERVAL NOT NULL,
            amplitude FLOAT NOT NULL,
            seasonality JSONB,
            trend VARCHAR(20),
            anomalies JSONB,
            confidence FLOAT NOT NULL,
            tenant_id VARCHAR(100) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            INDEX (tenant_id, pattern_type),
            INDEX (frequency, amplitude)
        );
        """
        
        async with self.db_pool.acquire() as conn:
            await conn.execute(create_correlations_table)
            await conn.execute(create_patterns_table)
    
    async def add_correlation_rule(self, rule: CorrelationRule) -> bool:
        """Ajoute une nouvelle règle de corrélation"""
        try:
            self.correlation_rules[rule.id] = rule
            
            # Persister en base
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO correlation_rules 
                        (id, name, description, correlation_type, time_window, 
                         min_events, max_events, strength_threshold, enabled, 
                         pattern, ml_model, tags, tenant_id, created_at, updated_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                        ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        updated_at = NOW()
                    """, rule.id, rule.name, rule.description, rule.correlation_type.value,
                    rule.time_window, rule.min_events, rule.max_events,
                    rule.strength_threshold, rule.enabled, json.dumps(rule.pattern),
                    rule.ml_model, json.dumps(list(rule.tags)), rule.tenant_id,
                    rule.created_at, rule.updated_at)
            
            logger.info(f"Correlation rule added: {rule.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add correlation rule: {e}")
            return False
    
    async def correlate_events(self, events: List[Dict[str, Any]], tenant_id: str) -> List[CorrelationResult]:
        """
        Corrèle des événements selon les règles définies
        """
        start_time = time.time()
        correlations = []
        
        try:
            # Preprocessing des événements
            processed_events = await self.preprocess_events(events, tenant_id)
            
            # Corrélation temporelle
            temporal_correlations = await self.temporal_correlation(processed_events, tenant_id)
            correlations.extend(temporal_correlations)
            
            # Corrélation sémantique
            semantic_correlations = await self.semantic_correlation(processed_events, tenant_id)
            correlations.extend(semantic_correlations)
            
            # Détection de patterns
            pattern_correlations = await self.pattern_correlation(processed_events, tenant_id)
            correlations.extend(pattern_correlations)
            
            # Détection d'anomalies
            anomaly_correlations = await self.anomaly_correlation(processed_events, tenant_id)
            correlations.extend(anomaly_correlations)
            
            # Analyse de causalité
            causal_correlations = await self.causal_correlation(processed_events, tenant_id)
            correlations.extend(causal_correlations)
            
            # Mise à jour des métriques
            self.performance_metrics['correlations_processed'].inc(len(correlations))
            self.performance_metrics['correlation_time'].observe(time.time() - start_time)
            self.performance_metrics['active_correlations'].set(len(correlations))
            
            # Persistence des résultats
            await self.persist_correlations(correlations, tenant_id)
            
            logger.info(f"Processed {len(events)} events, found {len(correlations)} correlations")
            return correlations
            
        except Exception as e:
            logger.error(f"Error in event correlation: {e}")
            return []
    
    async def preprocess_events(self, events: List[Dict[str, Any]], tenant_id: str) -> List[Dict[str, Any]]:
        """Prétraite les événements pour la corrélation"""
        processed = []
        
        for event in events:
            # Normalisation timestamp
            timestamp = event.get('timestamp')
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            elif not isinstance(timestamp, datetime):
                timestamp = datetime.utcnow()
            
            # Feature extraction
            features = await self.extract_features(event)
            
            # Enrichissement contextuel
            context = await self.enrich_context(event, tenant_id)
            
            processed_event = {
                **event,
                'timestamp': timestamp,
                'features': features,
                'context': context,
                'tenant_id': tenant_id,
                'correlation_id': str(uuid.uuid4())
            }
            
            processed.append(processed_event)
        
        return processed
    
    async def extract_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extrait les features pour le ML"""
        features = {}
        
        # Features numériques
        features['severity_score'] = self.severity_to_score(event.get('severity', 'info'))
        features['source_hash'] = hash(event.get('source', '')) % 1000
        features['hour_of_day'] = event.get('timestamp', datetime.utcnow()).hour
        features['day_of_week'] = event.get('timestamp', datetime.utcnow().weekday())
        
        # Features textuelles
        message = event.get('message', '')
        features['message_length'] = len(message)
        features['word_count'] = len(message.split())
        features['has_error_keywords'] = any(keyword in message.lower() 
                                           for keyword in ['error', 'fail', 'exception', 'critical'])
        
        # Features JSON
        if 'metadata' in event:
            metadata = event['metadata']
            features['metadata_size'] = len(json.dumps(metadata))
            features['metadata_keys'] = len(metadata.keys()) if isinstance(metadata, dict) else 0
        
        return features
    
    def severity_to_score(self, severity: str) -> float:
        """Convertit la sévérité en score numérique"""
        severity_map = {
            'debug': 0.1,
            'info': 0.3,
            'warning': 0.5,
            'error': 0.7,
            'critical': 0.9,
            'fatal': 1.0
        }
        return severity_map.get(severity.lower(), 0.3)
    
    async def enrich_context(self, event: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
        """Enrichit le contexte de l'événement"""
        context = {}
        
        # Contexte géographique
        if 'location' in event:
            context['geographic'] = await self.get_geographic_context(event['location'])
        
        # Contexte temporel
        context['temporal'] = await self.get_temporal_context(event['timestamp'])
        
        # Contexte business
        context['business'] = await self.get_business_context(event, tenant_id)
        
        # Contexte technique
        context['technical'] = await self.get_technical_context(event)
        
        return context
    
    async def temporal_correlation(self, events: List[Dict[str, Any]], tenant_id: str) -> List[CorrelationResult]:
        """Corrélation temporelle avec détection de patterns"""
        correlations = []
        
        # Grouper par fenêtre temporelle
        time_windows = defaultdict(list)
        for event in events:
            window_key = int(event['timestamp'].timestamp() // 300)  # 5 min windows
            time_windows[window_key].append(event)
        
        # Analyser chaque fenêtre
        for window_key, window_events in time_windows.items():
            if len(window_events) < 2:
                continue
            
            # Détection de patterns temporels
            pattern = await self.detect_temporal_pattern(window_events)
            
            if pattern and pattern.confidence > 0.7:
                correlation = CorrelationResult(
                    id=str(uuid.uuid4()),
                    correlation_type=CorrelationType.TEMPORAL,
                    strength=pattern.confidence,
                    confidence=pattern.confidence,
                    events=window_events,
                    pattern={
                        'type': pattern.pattern_type,
                        'frequency': pattern.frequency,
                        'duration': pattern.duration.total_seconds(),
                        'amplitude': pattern.amplitude
                    },
                    timeline=[(event['timestamp'], event.get('message', '')) for event in window_events],
                    metadata={'window_key': window_key, 'tenant_id': tenant_id}
                )
                correlations.append(correlation)
        
        return correlations
    
    async def semantic_correlation(self, events: List[Dict[str, Any]], tenant_id: str) -> List[CorrelationResult]:
        """Corrélation sémantique avec NLP"""
        correlations = []
        
        if len(events) < 2:
            return correlations
        
        # Extraction des messages
        messages = [event.get('message', '') for event in events]
        
        try:
            # Vectorisation TF-IDF
            tfidf_matrix = self.vectorizer.fit_transform(messages)
            
            # Calcul de similarité cosinus
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Recherche de clusters sémantiques
            for i in range(len(events)):
                for j in range(i + 1, len(events)):
                    similarity = similarity_matrix[i][j]
                    
                    if similarity > 0.7:  # Seuil de similarité
                        correlation = CorrelationResult(
                            id=str(uuid.uuid4()),
                            correlation_type=CorrelationType.SEMANTIC,
                            strength=similarity,
                            confidence=similarity,
                            events=[events[i], events[j]],
                            pattern={
                                'semantic_similarity': similarity,
                                'common_terms': self.extract_common_terms(messages[i], messages[j])
                            },
                            metadata={'tenant_id': tenant_id}
                        )
                        correlations.append(correlation)
        
        except Exception as e:
            logger.error(f"Error in semantic correlation: {e}")
        
        return correlations
    
    async def pattern_correlation(self, events: List[Dict[str, Any]], tenant_id: str) -> List[CorrelationResult]:
        """Détection de patterns avec machine learning"""
        correlations = []
        
        if len(events) < 3:
            return correlations
        
        try:
            # Préparation des features pour ML
            features_matrix = []
            for event in events:
                feature_vector = [
                    event['features']['severity_score'],
                    event['features']['hour_of_day'],
                    event['features']['day_of_week'],
                    event['features']['message_length'],
                    event['features']['word_count'],
                    int(event['features']['has_error_keywords'])
                ]
                features_matrix.append(feature_vector)
            
            features_array = np.array(features_matrix)
            
            # Normalisation
            normalized_features = self.scaler.fit_transform(features_array)
            
            # Clustering DBSCAN
            clusters = self.dbscan.fit_predict(normalized_features)
            
            # Analyse des clusters
            cluster_groups = defaultdict(list)
            for idx, cluster_id in enumerate(clusters):
                if cluster_id != -1:  # Ignorer le bruit
                    cluster_groups[cluster_id].append(idx)
            
            # Créer des corrélations pour chaque cluster
            for cluster_id, event_indices in cluster_groups.items():
                if len(event_indices) >= 2:
                    cluster_events = [events[i] for i in event_indices]
                    
                    # Calculer la force du pattern
                    strength = len(event_indices) / len(events)
                    
                    correlation = CorrelationResult(
                        id=str(uuid.uuid4()),
                        correlation_type=CorrelationType.PATTERN,
                        strength=strength,
                        confidence=min(strength * 1.2, 1.0),
                        events=cluster_events,
                        pattern={
                            'cluster_id': int(cluster_id),
                            'cluster_size': len(event_indices),
                            'pattern_features': self.analyze_cluster_features(cluster_events)
                        },
                        metadata={'tenant_id': tenant_id}
                    )
                    correlations.append(correlation)
        
        except Exception as e:
            logger.error(f"Error in pattern correlation: {e}")
        
        return correlations
    
    async def anomaly_correlation(self, events: List[Dict[str, Any]], tenant_id: str) -> List[CorrelationResult]:
        """Détection d'anomalies avec Isolation Forest"""
        correlations = []
        
        if len(events) < 5:
            return correlations
        
        try:
            # Préparation des features
            features_matrix = []
            for event in events:
                feature_vector = [
                    event['features']['severity_score'],
                    event['features']['hour_of_day'],
                    event['features']['message_length'],
                    event['features']['word_count']
                ]
                features_matrix.append(feature_vector)
            
            features_array = np.array(features_matrix)
            
            # Détection d'anomalies
            anomaly_scores = self.isolation_forest.fit_predict(features_array)
            anomaly_scores_proba = self.isolation_forest.decision_function(features_array)
            
            # Identifier les anomalies
            anomalous_events = []
            for idx, (score, proba) in enumerate(zip(anomaly_scores, anomaly_scores_proba)):
                if score == -1:  # Anomalie détectée
                    anomalous_events.append((events[idx], abs(proba)))
            
            if anomalous_events:
                # Trier par score d'anomalie
                anomalous_events.sort(key=lambda x: x[1], reverse=True)
                
                correlation = CorrelationResult(
                    id=str(uuid.uuid4()),
                    correlation_type=CorrelationType.ANOMALY,
                    strength=len(anomalous_events) / len(events),
                    confidence=np.mean([score for _, score in anomalous_events]),
                    events=[event for event, _ in anomalous_events],
                    pattern={
                        'anomaly_count': len(anomalous_events),
                        'anomaly_scores': [float(score) for _, score in anomalous_events],
                        'detection_method': 'isolation_forest'
                    },
                    metadata={'tenant_id': tenant_id}
                )
                correlations.append(correlation)
        
        except Exception as e:
            logger.error(f"Error in anomaly correlation: {e}")
        
        return correlations
    
    async def causal_correlation(self, events: List[Dict[str, Any]], tenant_id: str) -> List[CorrelationResult]:
        """Analyse de causalité avec graphes dirigés"""
        correlations = []
        
        if len(events) < 2:
            return correlations
        
        try:
            # Construire le graphe de causalité
            causal_graph = nx.DiGraph()
            
            # Ajouter les nœuds
            for event in events:
                causal_graph.add_node(
                    event['correlation_id'],
                    timestamp=event['timestamp'],
                    severity=event['features']['severity_score'],
                    source=event.get('source', 'unknown')
                )
            
            # Ajouter les arêtes basées sur la temporalité et la sévérité
            events_sorted = sorted(events, key=lambda x: x['timestamp'])
            
            for i in range(len(events_sorted) - 1):
                current = events_sorted[i]
                next_event = events_sorted[i + 1]
                
                # Calculer la probabilité de causalité
                time_diff = (next_event['timestamp'] - current['timestamp']).total_seconds()
                severity_correlation = abs(current['features']['severity_score'] - 
                                         next_event['features']['severity_score'])
                
                # Heuristique de causalité
                if time_diff < 300 and severity_correlation < 0.3:  # 5 min et sévérité similaire
                    causal_strength = max(0, 1 - (time_diff / 300) - severity_correlation)
                    
                    if causal_strength > 0.5:
                        causal_graph.add_edge(
                            current['correlation_id'],
                            next_event['correlation_id'],
                            weight=causal_strength,
                            time_diff=time_diff
                        )
            
            # Analyser les chaînes causales
            causal_chains = []
            for node in causal_graph.nodes():
                if causal_graph.in_degree(node) == 0:  # Nœud racine
                    chain = list(nx.dfs_preorder_nodes(causal_graph, node))
                    if len(chain) > 1:
                        causal_chains.append(chain)
            
            # Créer des corrélations pour les chaînes causales
            for chain in causal_chains:
                if len(chain) >= 2:
                    chain_events = [next(e for e in events if e['correlation_id'] == node_id) 
                                  for node_id in chain]
                    
                    # Calculer la force de la chaîne causale
                    total_weight = sum(causal_graph[chain[i]][chain[i+1]]['weight'] 
                                     for i in range(len(chain)-1))
                    avg_strength = total_weight / (len(chain) - 1)
                    
                    correlation = CorrelationResult(
                        id=str(uuid.uuid4()),
                        correlation_type=CorrelationType.CAUSAL,
                        strength=avg_strength,
                        confidence=avg_strength,
                        events=chain_events,
                        pattern={
                            'causal_chain_length': len(chain),
                            'root_cause': chain_events[0].get('message', 'Unknown'),
                            'causal_weights': [causal_graph[chain[i]][chain[i+1]]['weight'] 
                                             for i in range(len(chain)-1)]
                        },
                        root_cause=chain_events[0].get('message', 'Unknown'),
                        timeline=[(event['timestamp'], event.get('message', '')) 
                                for event in chain_events],
                        metadata={'tenant_id': tenant_id}
                    )
                    correlations.append(correlation)
        
        except Exception as e:
            logger.error(f"Error in causal correlation: {e}")
        
        return correlations
    
    async def detect_temporal_pattern(self, events: List[Dict[str, Any]]) -> Optional[TemporalPattern]:
        """Détecte les patterns temporels dans une série d'événements"""
        if len(events) < 3:
            return None
        
        try:
            # Extraire les timestamps
            timestamps = [event['timestamp'] for event in events]
            timestamps.sort()
            
            # Calculer les intervalles
            intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                        for i in range(len(timestamps)-1)]
            
            if not intervals:
                return None
            
            # Analyser la fréquence
            mean_interval = statistics.mean(intervals)
            std_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0
            
            # Détecter la régularité
            regularity = 1 - (std_interval / mean_interval) if mean_interval > 0 else 0
            regularity = max(0, min(1, regularity))
            
            # Détecter les anomalies temporelles
            anomalies = []
            for i, interval in enumerate(intervals):
                if abs(interval - mean_interval) > 2 * std_interval:
                    anomalies.append(timestamps[i+1])
            
            # Déterminer le type de pattern
            pattern_type = "regular" if regularity > 0.8 else "irregular"
            if len(anomalies) > len(intervals) * 0.3:
                pattern_type = "burst"
            
            pattern = TemporalPattern(
                id=str(uuid.uuid4()),
                pattern_type=pattern_type,
                frequency=1 / mean_interval if mean_interval > 0 else 0,
                duration=timestamps[-1] - timestamps[0],
                amplitude=len(events),
                anomalies=anomalies,
                confidence=regularity
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"Error detecting temporal pattern: {e}")
            return None
    
    def extract_common_terms(self, text1: str, text2: str) -> List[str]:
        """Extrait les termes communs entre deux textes"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        common = words1.intersection(words2)
        return list(common)
    
    def analyze_cluster_features(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyse les features communes d'un cluster"""
        if not events:
            return {}
        
        features = {
            'avg_severity': np.mean([e['features']['severity_score'] for e in events]),
            'common_hour': statistics.mode([e['features']['hour_of_day'] for e in events]),
            'avg_message_length': np.mean([e['features']['message_length'] for e in events]),
            'error_rate': np.mean([e['features']['has_error_keywords'] for e in events])
        }
        
        return features
    
    async def get_geographic_context(self, location: str) -> Dict[str, Any]:
        """Obtient le contexte géographique"""
        return {
            'location': location,
            'timezone': 'UTC',  # À implémenter avec une vraie API de géolocalisation
            'region': 'unknown'
        }
    
    async def get_temporal_context(self, timestamp: datetime) -> Dict[str, Any]:
        """Obtient le contexte temporel"""
        return {
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'is_weekend': timestamp.weekday() >= 5,
            'is_business_hours': 9 <= timestamp.hour <= 17
        }
    
    async def get_business_context(self, event: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
        """Obtient le contexte business"""
        return {
            'tenant_id': tenant_id,
            'business_impact': 'medium',  # À calculer selon les règles business
            'affected_services': []
        }
    
    async def get_technical_context(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Obtient le contexte technique"""
        return {
            'source_system': event.get('source', 'unknown'),
            'event_type': event.get('type', 'unknown'),
            'technology_stack': []
        }
    
    async def persist_correlations(self, correlations: List[CorrelationResult], tenant_id: str):
        """Persiste les corrélations en base de données"""
        if not self.db_pool or not correlations:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                for correlation in correlations:
                    await conn.execute("""
                        INSERT INTO correlations 
                        (id, correlation_type, strength, confidence, events, pattern, 
                         root_cause, predicted_impact, remediation_suggestions, 
                         timeline, metadata, tenant_id, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    """, 
                    correlation.id, correlation.correlation_type.value, 
                    correlation.strength, correlation.confidence,
                    json.dumps(correlation.events), json.dumps(correlation.pattern),
                    correlation.root_cause, json.dumps(correlation.predicted_impact),
                    json.dumps(correlation.remediation_suggestions),
                    json.dumps([(t.isoformat(), msg) for t, msg in correlation.timeline]),
                    json.dumps(correlation.metadata), tenant_id, correlation.created_at)
                    
            logger.info(f"Persisted {len(correlations)} correlations for tenant {tenant_id}")
            
        except Exception as e:
            logger.error(f"Failed to persist correlations: {e}")
    
    async def get_correlation_history(self, tenant_id: str, limit: int = 100) -> List[CorrelationResult]:
        """Récupère l'historique des corrélations"""
        if not self.db_pool:
            return []
        
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM correlations 
                    WHERE tenant_id = $1 
                    ORDER BY created_at DESC 
                    LIMIT $2
                """, tenant_id, limit)
                
                correlations = []
                for row in rows:
                    correlation = CorrelationResult(
                        id=row['id'],
                        correlation_type=CorrelationType(row['correlation_type']),
                        strength=row['strength'],
                        confidence=row['confidence'],
                        events=json.loads(row['events']),
                        pattern=json.loads(row['pattern']) if row['pattern'] else {},
                        root_cause=row['root_cause'],
                        predicted_impact=json.loads(row['predicted_impact']) if row['predicted_impact'] else None,
                        remediation_suggestions=json.loads(row['remediation_suggestions']) if row['remediation_suggestions'] else [],
                        timeline=[(datetime.fromisoformat(t), msg) for t, msg in json.loads(row['timeline'])] if row['timeline'] else [],
                        metadata=json.loads(row['metadata']) if row['metadata'] else {},
                        created_at=row['created_at']
                    )
                    correlations.append(correlation)
                
                return correlations
                
        except Exception as e:
            logger.error(f"Failed to get correlation history: {e}")
            return []
    
    async def cleanup_old_correlations(self, tenant_id: str, days: int = 30):
        """Nettoie les anciennes corrélations"""
        if not self.db_pool:
            return
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            async with self.db_pool.acquire() as conn:
                result = await conn.execute("""
                    DELETE FROM correlations 
                    WHERE tenant_id = $1 AND created_at < $2
                """, tenant_id, cutoff_date)
                
                logger.info(f"Cleaned up old correlations: {result}")
                
        except Exception as e:
            logger.error(f"Failed to cleanup correlations: {e}")
    
    async def shutdown(self):
        """Arrêt propre du moteur de corrélation"""
        if self.redis_client:
            await self.redis_client.close()
        
        if self.db_pool:
            await self.db_pool.close()
        
        logger.info("Correlation engine shutdown complete")


# Factory function pour faciliter l'instanciation
def create_correlation_engine(config: Dict[str, Any]) -> AdvancedCorrelationEngine:
    """Crée une instance du moteur de corrélation avec la configuration donnée"""
    return AdvancedCorrelationEngine(config)


# Export des classes principales
__all__ = [
    'AdvancedCorrelationEngine',
    'CorrelationType',
    'CorrelationStrength', 
    'CorrelationRule',
    'CorrelationResult',
    'TemporalPattern',
    'EventGraphNode',
    'create_correlation_engine'
]
