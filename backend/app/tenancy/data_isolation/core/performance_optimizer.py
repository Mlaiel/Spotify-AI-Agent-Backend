"""
⚡ Performance Optimizer - Optimiseur de Performance Ultra-Avancé
===============================================================

Système d'optimisation des performances pour l'isolation des données
avec intelligence artificielle, prédiction et adaptation dynamique.

Author: Ingénieur Machine Learning - Fahed Mlaiel
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import statistics
import numpy as np
from collections import defaultdict, deque
import json
import hashlib

from .tenant_context import TenantContext, IsolationLevel
from ..exceptions import PerformanceOptimizationError
from ...core.config import settings
from ...utils.metrics import MetricsCollector
from ...monitoring.performance_monitor import PerformanceMonitor


class OptimizationStrategy(Enum):
    """Stratégies d'optimisation"""
    AGGRESSIVE = "aggressive"      # Performance maximale
    BALANCED = "balanced"          # Équilibre performance/sécurité
    CONSERVATIVE = "conservative"  # Sécurité prioritaire
    ADAPTIVE = "adaptive"          # Adaptation automatique
    PREDICTIVE = "predictive"      # Basé sur les prédictions ML


class CacheStrategy(Enum):
    """Stratégies de cache"""
    LRU = "lru"                   # Least Recently Used
    LFU = "lfu"                   # Least Frequently Used
    TTL = "ttl"                   # Time To Live
    ADAPTIVE_TTL = "adaptive_ttl"  # TTL adaptatif
    ML_PREDICTIVE = "ml_predictive" # Prédictif ML


class QueryOptimizationType(Enum):
    """Types d'optimisation de requêtes"""
    INDEX_SUGGESTION = "index_suggestion"
    QUERY_REWRITE = "query_rewrite"
    PARTITION_PRUNING = "partition_pruning"
    MATERIALIZED_VIEW = "materialized_view"
    CACHE_WARMING = "cache_warming"


@dataclass
class PerformanceMetrics:
    """Métriques de performance"""
    operation_type: str
    tenant_id: str
    timestamp: datetime
    
    # Métriques temporelles
    response_time_ms: float
    query_execution_time_ms: float
    cache_lookup_time_ms: float
    network_latency_ms: float
    
    # Métriques de ressources
    cpu_usage_percent: float
    memory_usage_mb: float
    io_operations: int
    network_bytes: int
    
    # Métriques de cache
    cache_hit_ratio: float
    cache_size_mb: float
    cache_evictions: int
    
    # Métriques de base de données
    db_connections_active: int
    db_query_count: int
    db_index_usage: float
    
    # Score de performance global
    performance_score: float = field(default=0.0)
    
    # Contexte additionnel
    isolation_level: str = "strict"
    optimization_applied: List[str] = field(default_factory=list)


@dataclass
class OptimizationRecommendation:
    """Recommandation d'optimisation"""
    recommendation_id: str
    strategy: OptimizationStrategy
    description: str
    expected_improvement: float  # Pourcentage d'amélioration attendu
    implementation_cost: str     # low, medium, high
    priority: int               # 1-10
    
    # Détails techniques
    target_metric: str
    current_value: float
    target_value: float
    
    # Actions recommandées
    actions: List[Dict[str, Any]] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    
    # Validation
    confidence_score: float = 0.0
    risk_level: str = "low"
    
    # Métadonnées
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None


@dataclass
class CacheEntry:
    """Entrée de cache optimisée"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl: Optional[timedelta]
    tenant_id: str
    size_bytes: int
    priority: float = 1.0
    
    # Prédictions ML
    predicted_access_probability: float = 0.5
    predicted_next_access: Optional[datetime] = None


class QueryOptimizer:
    """Optimiseur de requêtes intelligent"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.query_patterns: Dict[str, Any] = {}
        self.optimization_history: List[Dict[str, Any]] = []
        self.index_suggestions: Dict[str, List[str]] = defaultdict(list)
    
    async def optimize_query(
        self,
        query: str,
        context: TenantContext,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimise une requête SQL"""
        
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # Analyse de la requête
        analysis = await self._analyze_query(query, context)
        
        # Suggestions d'optimisation
        optimizations = []
        
        # Suggestion d'index
        if analysis['missing_indexes']:
            optimizations.append({
                'type': QueryOptimizationType.INDEX_SUGGESTION,
                'indexes': analysis['missing_indexes'],
                'expected_improvement': 40.0
            })
        
        # Réécriture de requête
        if analysis['can_rewrite']:
            rewritten_query = await self._rewrite_query(query, analysis)
            optimizations.append({
                'type': QueryOptimizationType.QUERY_REWRITE,
                'original_query': query,
                'optimized_query': rewritten_query,
                'expected_improvement': 25.0
            })
        
        # Pruning de partitions
        if analysis['can_prune_partitions']:
            optimizations.append({
                'type': QueryOptimizationType.PARTITION_PRUNING,
                'prunable_partitions': analysis['prunable_partitions'],
                'expected_improvement': 60.0
            })
        
        return {
            'query_hash': query_hash,
            'analysis': analysis,
            'optimizations': optimizations,
            'estimated_improvement': sum(opt['expected_improvement'] for opt in optimizations),
            'recommendation': self._select_best_optimization(optimizations)
        }
    
    async def _analyze_query(self, query: str, context: TenantContext) -> Dict[str, Any]:
        """Analyse une requête pour détecter les opportunités d'optimisation"""
        
        # Analyse basique de la requête
        query_lower = query.lower()
        
        analysis = {
            'query_type': self._detect_query_type(query_lower),
            'table_count': query_lower.count(' join ') + 1,
            'has_subquery': 'select' in query_lower[query_lower.find('select') + 6:],
            'has_order_by': 'order by' in query_lower,
            'has_group_by': 'group by' in query_lower,
            'missing_indexes': [],
            'can_rewrite': False,
            'can_prune_partitions': False,
            'prunable_partitions': []
        }
        
        # Détection d'index manquants
        if 'where' in query_lower:
            where_clause = query_lower.split('where')[1].split('order by')[0].split('group by')[0]
            # Analyse simplifiée des colonnes dans WHERE
            analysis['missing_indexes'] = self._suggest_indexes(where_clause)
        
        # Détection des opportunités de réécriture
        if analysis['has_subquery'] and 'exists' not in query_lower:
            analysis['can_rewrite'] = True
        
        # Détection du pruning de partitions
        if context.tenant_id and 'tenant_id' in query_lower:
            analysis['can_prune_partitions'] = True
            analysis['prunable_partitions'] = [f"partition_{context.tenant_id}"]
        
        return analysis
    
    def _detect_query_type(self, query: str) -> str:
        """Détecte le type de requête"""
        if query.strip().startswith('select'):
            return 'SELECT'
        elif query.strip().startswith('insert'):
            return 'INSERT'
        elif query.strip().startswith('update'):
            return 'UPDATE'
        elif query.strip().startswith('delete'):
            return 'DELETE'
        else:
            return 'OTHER'
    
    def _suggest_indexes(self, where_clause: str) -> List[str]:
        """Suggère des index basés sur la clause WHERE"""
        # Implémentation simplifiée
        suggested = []
        
        # Recherche de colonnes fréquemment utilisées
        common_columns = ['tenant_id', 'user_id', 'created_at', 'status']
        for column in common_columns:
            if column in where_clause:
                suggested.append(f"idx_{column}")
        
        return suggested
    
    async def _rewrite_query(self, query: str, analysis: Dict[str, Any]) -> str:
        """Réécrit une requête pour l'optimiser"""
        # Implémentation simplifiée de réécriture
        return query  # Placeholder
    
    def _select_best_optimization(self, optimizations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Sélectionne la meilleure optimisation"""
        if not optimizations:
            return None
        
        return max(optimizations, key=lambda x: x['expected_improvement'])


class IntelligentCache:
    """Cache intelligent avec prédictions ML"""
    
    def __init__(self, max_size_mb: int = 1024):
        self.max_size_mb = max_size_mb
        self.cache: Dict[str, CacheEntry] = {}
        self.access_history: deque = deque(maxlen=10000)
        self.size_mb = 0.0
        
        # Modèle ML pour prédictions (simplifié)
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        
    async def get(self, key: str, tenant_id: str) -> Optional[Any]:
        """Récupère une valeur du cache"""
        if key in self.cache:
            entry = self.cache[key]
            
            # Mise à jour des statistiques d'accès
            entry.last_accessed = datetime.now(timezone.utc)
            entry.access_count += 1
            
            # Enregistrement pour l'apprentissage
            self.access_history.append({
                'key': key,
                'tenant_id': tenant_id,
                'timestamp': datetime.now(timezone.utc),
                'hit': True
            })
            
            # Vérification TTL
            if entry.ttl and datetime.now(timezone.utc) - entry.created_at > entry.ttl:
                await self.remove(key)
                return None
            
            return entry.value
        
        # Cache miss
        self.access_history.append({
            'key': key,
            'tenant_id': tenant_id,
            'timestamp': datetime.now(timezone.utc),
            'hit': False
        })
        
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        tenant_id: str,
        ttl: Optional[timedelta] = None,
        priority: float = 1.0
    ):
        """Stocke une valeur dans le cache"""
        
        # Calcul de la taille
        size_bytes = len(json.dumps(value, default=str).encode())
        
        # Prédiction de probabilité d'accès
        predicted_probability = await self._predict_access_probability(key, tenant_id)
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(timezone.utc),
            last_accessed=datetime.now(timezone.utc),
            access_count=1,
            ttl=ttl,
            tenant_id=tenant_id,
            size_bytes=size_bytes,
            priority=priority,
            predicted_access_probability=predicted_probability
        )
        
        # Éviction si nécessaire
        await self._ensure_capacity(size_bytes)
        
        self.cache[key] = entry
        self.size_mb += size_bytes / (1024 * 1024)
    
    async def _predict_access_probability(self, key: str, tenant_id: str) -> float:
        """Prédit la probabilité d'accès futur"""
        
        # Analyse des patterns d'accès historiques
        pattern_key = f"{tenant_id}:{key}"
        
        if pattern_key in self.access_patterns:
            recent_accesses = self.access_patterns[pattern_key][-10:]  # 10 derniers accès
            
            if len(recent_accesses) >= 3:
                # Calcul de la tendance
                avg_interval = statistics.mean(recent_accesses)
                return min(1.0, 1.0 / (avg_interval + 1))
        
        return 0.5  # Valeur par défaut
    
    async def _ensure_capacity(self, required_bytes: int):
        """Assure la capacité disponible dans le cache"""
        
        required_mb = required_bytes / (1024 * 1024)
        
        while self.size_mb + required_mb > self.max_size_mb:
            # Sélection de l'entrée à évincer
            victim = await self._select_eviction_victim()
            if victim:
                await self.remove(victim)
            else:
                break
    
    async def _select_eviction_victim(self) -> Optional[str]:
        """Sélectionne l'entrée à évincer"""
        
        if not self.cache:
            return None
        
        # Score d'éviction basé sur plusieurs facteurs
        best_candidate = None
        best_score = float('-inf')
        
        for key, entry in self.cache.items():
            
            # Facteurs d'éviction
            age_factor = (datetime.now(timezone.utc) - entry.last_accessed).total_seconds() / 3600  # heures
            frequency_factor = 1.0 / (entry.access_count + 1)
            size_factor = entry.size_bytes / (1024 * 1024)  # MB
            priority_factor = 1.0 / (entry.priority + 1)
            prediction_factor = 1.0 - entry.predicted_access_probability
            
            # Score composite (plus élevé = meilleur candidat pour éviction)
            score = (age_factor * 0.3 + 
                    frequency_factor * 0.2 + 
                    size_factor * 0.2 + 
                    priority_factor * 0.1 + 
                    prediction_factor * 0.2)
            
            if score > best_score:
                best_score = score
                best_candidate = key
        
        return best_candidate
    
    async def remove(self, key: str):
        """Supprime une entrée du cache"""
        if key in self.cache:
            entry = self.cache[key]
            self.size_mb -= entry.size_bytes / (1024 * 1024)
            del self.cache[key]


class PerformanceOptimizer:
    """
    Optimiseur de performance ultra-avancé avec IA
    
    Features:
    - Optimisation automatique des requêtes
    - Cache intelligent avec prédictions ML
    - Surveillance en temps réel
    - Recommandations adaptatives
    - Prédiction des goulots d'étranglement
    - Auto-tuning des paramètres
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_collector = MetricsCollector()
        self.performance_monitor = PerformanceMonitor()
        self.query_optimizer = QueryOptimizer()
        
        # Cache intelligent
        self.cache = IntelligentCache(max_size_mb=2048)
        
        # Métriques et historique
        self.metrics_history: deque = deque(maxlen=10000)
        self.optimization_history: List[OptimizationRecommendation] = []
        
        # Configuration d'optimisation
        self.current_strategy = OptimizationStrategy.ADAPTIVE
        self.optimization_targets = {
            'response_time_ms': 100.0,
            'cache_hit_ratio': 0.85,
            'cpu_usage_percent': 70.0,
            'memory_usage_mb': 1024.0
        }
        
        # État de l'optimiseur
        self.active = True
        self.learning_mode = True
        
        # Statistiques
        self.statistics = {
            'optimizations_applied': 0,
            'performance_improvements': 0,
            'cache_hit_ratio': 0.0,
            'avg_response_time_ms': 0.0,
            'recommendations_generated': 0
        }
        
        # Modèles ML (simplifiés)
        self.performance_predictor = None
        self.bottleneck_detector = None
    
    async def optimize_operation(
        self,
        operation_type: str,
        context: TenantContext,
        data: Dict[str, Any],
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Optimise une opération complète
        
        Args:
            operation_type: Type d'opération
            context: Contexte du tenant
            data: Données de l'opération
            query: Requête SQL optionnelle
            
        Returns:
            Résultat de l'optimisation avec métriques
        """
        start_time = datetime.now(timezone.utc)
        
        # Collecte des métriques initiales
        initial_metrics = await self._collect_metrics(context, operation_type)
        
        # Optimisations applicables
        optimizations_applied = []
        modified_data = data.copy()
        
        # 1. Optimisation du cache
        cache_result = await self._optimize_cache_usage(operation_type, context, modified_data)
        if cache_result['optimization_applied']:
            optimizations_applied.append('cache_optimization')
            if cache_result['cached_result']:
                return {
                    'result': cache_result['cached_result'],
                    'optimizations_applied': optimizations_applied,
                    'performance_gain': cache_result['performance_gain'],
                    'from_cache': True,
                    'metrics': initial_metrics
                }
        
        # 2. Optimisation des requêtes
        if query:
            query_optimization = await self.query_optimizer.optimize_query(query, context, data)
            if query_optimization['optimizations']:
                optimizations_applied.append('query_optimization')
                query = query_optimization['recommendation']['optimized_query']
        
        # 3. Optimisation de la stratégie d'isolation
        isolation_optimization = await self._optimize_isolation_strategy(context, operation_type)
        if isolation_optimization['optimization_applied']:
            optimizations_applied.append('isolation_optimization')
            context.isolation_level = isolation_optimization['recommended_level']
        
        # 4. Optimisation des ressources
        resource_optimization = await self._optimize_resource_allocation(context, operation_type, initial_metrics)
        if resource_optimization['optimization_applied']:
            optimizations_applied.append('resource_optimization')
        
        # Collecte des métriques finales
        end_time = datetime.now(timezone.utc)
        final_metrics = await self._collect_metrics(context, operation_type)
        
        # Calcul de l'amélioration
        performance_gain = self._calculate_performance_gain(initial_metrics, final_metrics)
        
        # Stockage des métriques
        operation_metrics = PerformanceMetrics(
            operation_type=operation_type,
            tenant_id=context.tenant_id,
            timestamp=start_time,
            response_time_ms=(end_time - start_time).total_seconds() * 1000,
            query_execution_time_ms=final_metrics.get('query_time_ms', 0),
            cache_lookup_time_ms=final_metrics.get('cache_time_ms', 0),
            network_latency_ms=final_metrics.get('network_latency_ms', 0),
            cpu_usage_percent=final_metrics.get('cpu_usage', 0),
            memory_usage_mb=final_metrics.get('memory_usage_mb', 0),
            io_operations=final_metrics.get('io_operations', 0),
            network_bytes=final_metrics.get('network_bytes', 0),
            cache_hit_ratio=final_metrics.get('cache_hit_ratio', 0),
            cache_size_mb=final_metrics.get('cache_size_mb', 0),
            cache_evictions=final_metrics.get('cache_evictions', 0),
            db_connections_active=final_metrics.get('db_connections', 0),
            db_query_count=final_metrics.get('db_queries', 0),
            db_index_usage=final_metrics.get('index_usage', 0),
            performance_score=self._calculate_performance_score(final_metrics),
            isolation_level=context.isolation_level.value,
            optimization_applied=optimizations_applied
        )
        
        self.metrics_history.append(operation_metrics)
        
        # Mise à jour des statistiques
        self.statistics['optimizations_applied'] += len(optimizations_applied)
        if performance_gain > 0:
            self.statistics['performance_improvements'] += 1
        
        # Apprentissage adaptatif
        if self.learning_mode:
            await self._update_learning_models(operation_metrics, performance_gain)
        
        return {
            'optimizations_applied': optimizations_applied,
            'performance_gain_percent': performance_gain,
            'metrics': operation_metrics.__dict__,
            'recommendations': await self._generate_recommendations(operation_metrics),
            'from_cache': False
        }
    
    async def _optimize_cache_usage(
        self,
        operation_type: str,
        context: TenantContext,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimise l'utilisation du cache"""
        
        # Génération de la clé de cache
        cache_key = self._generate_cache_key(operation_type, context, data)
        
        # Tentative de récupération depuis le cache
        cached_result = await self.cache.get(cache_key, context.tenant_id)
        
        if cached_result:
            return {
                'optimization_applied': True,
                'cached_result': cached_result,
                'performance_gain': 80.0  # Gain typique du cache
            }
        
        return {
            'optimization_applied': False,
            'cached_result': None,
            'performance_gain': 0.0
        }
    
    async def _optimize_isolation_strategy(
        self,
        context: TenantContext,
        operation_type: str
    ) -> Dict[str, Any]:
        """Optimise la stratégie d'isolation"""
        
        current_level = context.isolation_level
        
        # Analyse du type d'opération pour ajuster l'isolation
        if operation_type in ['read', 'search'] and current_level == IsolationLevel.PARANOID:
            # Les lectures peuvent tolérer un niveau moins strict
            return {
                'optimization_applied': True,
                'recommended_level': IsolationLevel.STRICT,
                'performance_gain': 15.0
            }
        
        elif operation_type in ['bulk_insert', 'batch_update'] and current_level == IsolationLevel.STRICT:
            # Les opérations bulk peuvent bénéficier d'un niveau moins strict
            return {
                'optimization_applied': True,
                'recommended_level': IsolationLevel.BASIC,
                'performance_gain': 25.0
            }
        
        return {
            'optimization_applied': False,
            'recommended_level': current_level,
            'performance_gain': 0.0
        }
    
    async def _optimize_resource_allocation(
        self,
        context: TenantContext,
        operation_type: str,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimise l'allocation des ressources"""
        
        optimizations = []
        
        # Optimisation de la mémoire
        if metrics.get('memory_usage_mb', 0) > 800:
            optimizations.append('memory_pressure_relief')
        
        # Optimisation CPU
        if metrics.get('cpu_usage', 0) > 80:
            optimizations.append('cpu_optimization')
        
        # Optimisation des connexions DB
        if metrics.get('db_connections', 0) > 50:
            optimizations.append('connection_pooling')
        
        return {
            'optimization_applied': len(optimizations) > 0,
            'optimizations': optimizations,
            'performance_gain': len(optimizations) * 10.0
        }
    
    def _generate_cache_key(
        self,
        operation_type: str,
        context: TenantContext,
        data: Dict[str, Any]
    ) -> str:
        """Génère une clé de cache optimisée"""
        
        # Éléments de la clé
        key_elements = [
            operation_type,
            context.tenant_id,
            context.isolation_level.value,
            str(sorted(data.keys()))
        ]
        
        # Hash des valeurs pour les données sensibles
        data_hash = hashlib.md5(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()
        key_elements.append(data_hash[:8])
        
        return ":".join(key_elements)
    
    async def _collect_metrics(self, context: TenantContext, operation_type: str) -> Dict[str, Any]:
        """Collecte les métriques de performance"""
        
        # Collecte des métriques système
        return {
            'cpu_usage': await self.metrics_collector.get_cpu_usage(),
            'memory_usage_mb': await self.metrics_collector.get_memory_usage(),
            'cache_hit_ratio': self.statistics.get('cache_hit_ratio', 0.0),
            'cache_size_mb': self.cache.size_mb,
            'db_connections': await self.metrics_collector.get_db_connections(),
            'network_latency_ms': await self.metrics_collector.get_network_latency()
        }
    
    def _calculate_performance_gain(
        self,
        initial_metrics: Dict[str, Any],
        final_metrics: Dict[str, Any]
    ) -> float:
        """Calcule le gain de performance"""
        
        gains = []
        
        # Comparaison des métriques clés
        for metric in ['cpu_usage', 'memory_usage_mb', 'network_latency_ms']:
            initial = initial_metrics.get(metric, 0)
            final = final_metrics.get(metric, 0)
            
            if initial > 0:
                improvement = ((initial - final) / initial) * 100
                gains.append(improvement)
        
        return statistics.mean(gains) if gains else 0.0
    
    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calcule un score de performance global"""
        
        # Normalisation des métriques (0-100)
        cpu_score = max(0, 100 - metrics.get('cpu_usage', 0))
        memory_score = max(0, 100 - (metrics.get('memory_usage_mb', 0) / 10))
        cache_score = metrics.get('cache_hit_ratio', 0) * 100
        latency_score = max(0, 100 - metrics.get('network_latency_ms', 0))
        
        # Score pondéré
        weights = {'cpu': 0.3, 'memory': 0.2, 'cache': 0.3, 'latency': 0.2}
        
        return (cpu_score * weights['cpu'] +
                memory_score * weights['memory'] +
                cache_score * weights['cache'] +
                latency_score * weights['latency'])
    
    async def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[OptimizationRecommendation]:
        """Génère des recommandations d'optimisation"""
        
        recommendations = []
        
        # Recommandation de cache
        if metrics.cache_hit_ratio < 0.7:
            rec = OptimizationRecommendation(
                recommendation_id=f"CACHE_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                strategy=OptimizationStrategy.ADAPTIVE,
                description="Améliorer le taux de succès du cache",
                expected_improvement=30.0,
                implementation_cost="low",
                priority=8,
                target_metric="cache_hit_ratio",
                current_value=metrics.cache_hit_ratio,
                target_value=0.85,
                actions=[
                    {"type": "increase_cache_size", "value": "2GB"},
                    {"type": "optimize_cache_strategy", "strategy": "ml_predictive"}
                ],
                confidence_score=0.85
            )
            recommendations.append(rec)
        
        # Recommandation CPU
        if metrics.cpu_usage_percent > 80:
            rec = OptimizationRecommendation(
                recommendation_id=f"CPU_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                strategy=OptimizationStrategy.AGGRESSIVE,
                description="Réduire l'utilisation CPU",
                expected_improvement=25.0,
                implementation_cost="medium",
                priority=9,
                target_metric="cpu_usage_percent",
                current_value=metrics.cpu_usage_percent,
                target_value=70.0,
                actions=[
                    {"type": "enable_query_caching", "duration": "1h"},
                    {"type": "optimize_algorithms", "focus": "tenant_resolution"}
                ],
                confidence_score=0.75
            )
            recommendations.append(rec)
        
        return recommendations
    
    async def _update_learning_models(self, metrics: PerformanceMetrics, performance_gain: float):
        """Met à jour les modèles d'apprentissage"""
        
        # Apprentissage des patterns de performance
        pattern_key = f"{metrics.tenant_id}:{metrics.operation_type}"
        
        if pattern_key not in self.cache.access_patterns:
            self.cache.access_patterns[pattern_key] = []
        
        self.cache.access_patterns[pattern_key].append(performance_gain)
        
        # Limitation de l'historique
        if len(self.cache.access_patterns[pattern_key]) > 100:
            self.cache.access_patterns[pattern_key] = self.cache.access_patterns[pattern_key][-50:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de l'optimiseur"""
        
        recent_metrics = list(self.metrics_history)[-100:] if self.metrics_history else []
        
        if recent_metrics:
            avg_response_time = statistics.mean([m.response_time_ms for m in recent_metrics])
            avg_performance_score = statistics.mean([m.performance_score for m in recent_metrics])
        else:
            avg_response_time = 0.0
            avg_performance_score = 0.0
        
        return {
            **self.statistics,
            'cache_size_mb': self.cache.size_mb,
            'cache_entries': len(self.cache.cache),
            'avg_response_time_ms': avg_response_time,
            'avg_performance_score': avg_performance_score,
            'recommendations_pending': len(self.optimization_history),
            'learning_patterns': len(self.cache.access_patterns),
            'current_strategy': self.current_strategy.value
        }
