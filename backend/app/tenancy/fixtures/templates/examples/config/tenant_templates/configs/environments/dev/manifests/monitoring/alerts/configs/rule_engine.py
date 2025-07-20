"""
Moteur de Règles d'Alertes Intelligent - Spotify AI Agent
=========================================================

Système de règles d'alertes avancé avec apprentissage automatique,
logique floue et adaptation dynamique des seuils.

Fonctionnalités:
- Règles complexes avec opérateurs logiques
- Seuils adaptatifs basés sur ML
- Logique floue pour la gestion de l'incertitude
- Règles contextuelles par tenant/service
- Auto-apprentissage des patterns d'alertes
- Prédiction proactive d'incidents
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import yaml
import re
import ast
import operator
from collections import defaultdict, deque
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import redis.asyncio as redis


class RuleOperator(Enum):
    """Opérateurs pour les règles d'alertes"""
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    CONTAINS = "contains"
    REGEX = "regex"
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"


class LogicalOperator(Enum):
    """Opérateurs logiques pour combiner les conditions"""
    AND = "and"
    OR = "or"
    NOT = "not"
    XOR = "xor"


class RuleType(Enum):
    """Types de règles d'alertes"""
    THRESHOLD = "threshold"
    RATE_OF_CHANGE = "rate_of_change"
    ANOMALY = "anomaly"
    PATTERN = "pattern"
    COMPOSITE = "composite"
    PREDICTIVE = "predictive"
    BUSINESS = "business"


@dataclass
class RuleCondition:
    """Condition individuelle d'une règle"""
    metric_name: str
    operator: RuleOperator
    value: Union[float, str, List]
    labels: Dict[str, str] = field(default_factory=dict)
    time_window: timedelta = field(default=timedelta(minutes=5))
    aggregation: str = "avg"  # avg, max, min, sum, count
    weight: float = 1.0  # Poids pour la logique floue


@dataclass
class AlertRule:
    """Règle d'alerte complète"""
    id: str
    name: str
    description: str
    tenant_id: str = ""
    service: str = ""
    environment: str = ""
    rule_type: RuleType = RuleType.THRESHOLD
    conditions: List[RuleCondition] = field(default_factory=list)
    logical_operator: LogicalOperator = LogicalOperator.AND
    severity: str = "medium"
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Configuration avancée
    cooldown_period: timedelta = field(default=timedelta(minutes=10))
    auto_resolve: bool = True
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)
    notification_channels: List[str] = field(default_factory=list)
    
    # ML et adaptation
    adaptive_thresholds: bool = False
    learning_enabled: bool = False
    confidence_threshold: float = 0.8
    historical_data_window: timedelta = field(default=timedelta(days=7))
    
    # Métadonnées d'exécution
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    false_positive_count: int = 0
    accuracy_score: float = 0.0


@dataclass
class RuleEvaluationResult:
    """Résultat de l'évaluation d'une règle"""
    rule_id: str
    triggered: bool
    confidence: float
    evaluation_time: datetime
    metric_values: Dict[str, float]
    condition_results: List[Dict[str, Any]]
    anomaly_score: float = 0.0
    predicted_severity: str = ""
    explanation: str = ""


class RuleEngine:
    """Moteur de règles intelligent avec ML"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Stockage des règles
        self.rules: Dict[str, AlertRule] = {}
        self.rule_cache: Dict[str, Any] = {}
        
        # Cache Redis
        self.redis_client = None
        
        # Modèles ML
        self.threshold_predictors: Dict[str, RandomForestClassifier] = {}
        self.pattern_detectors: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Historique des évaluations
        self.evaluation_history: deque = deque(maxlen=100000)
        self.rule_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Seuils adaptatifs
        self.adaptive_thresholds: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.threshold_learning_data: Dict[str, List[float]] = defaultdict(list)
        
        # Configuration de performance
        self.evaluation_interval = config.get('evaluation_interval', 30)
        self.max_rules_per_evaluation = config.get('max_rules_per_evaluation', 1000)
        
        # Opérateurs personnalisés
        self.custom_operators: Dict[str, Callable] = {}
        
    async def initialize(self):
        """Initialisation asynchrone du moteur de règles"""
        try:
            # Connexion Redis
            self.redis_client = redis.from_url(
                self.config.get('redis_url', 'redis://localhost:6379'),
                decode_responses=True
            )
            
            # Chargement des règles existantes
            await self._load_rules_from_storage()
            
            # Initialisation des modèles ML
            await self._initialize_ml_models()
            
            # Chargement des seuils adaptatifs
            await self._load_adaptive_thresholds()
            
            # Démarrage des tâches de fond
            asyncio.create_task(self._continuous_evaluation())
            asyncio.create_task(self._adaptive_learning_task())
            asyncio.create_task(self._rule_performance_analysis())
            asyncio.create_task(self._threshold_optimization())
            
            self.logger.info("RuleEngine initialisé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation: {e}")
            raise
    
    async def create_rule(self, rule_data: Dict[str, Any]) -> AlertRule:
        """Création d'une nouvelle règle d'alerte"""
        try:
            # Validation des données
            await self._validate_rule_data(rule_data)
            
            # Création de la règle
            rule = AlertRule(
                id=rule_data.get('id', f"rule_{len(self.rules)}_{int(datetime.utcnow().timestamp())}"),
                name=rule_data['name'],
                description=rule_data.get('description', ''),
                tenant_id=rule_data.get('tenant_id', ''),
                service=rule_data.get('service', ''),
                environment=rule_data.get('environment', ''),
                rule_type=RuleType(rule_data.get('rule_type', 'threshold')),
                severity=rule_data.get('severity', 'medium'),
                enabled=rule_data.get('enabled', True),
                created_by=rule_data.get('created_by', ''),
                tags=rule_data.get('tags', []),
                cooldown_period=timedelta(seconds=rule_data.get('cooldown_seconds', 600)),
                auto_resolve=rule_data.get('auto_resolve', True),
                adaptive_thresholds=rule_data.get('adaptive_thresholds', False),
                learning_enabled=rule_data.get('learning_enabled', False)
            )
            
            # Création des conditions
            for condition_data in rule_data.get('conditions', []):
                condition = RuleCondition(
                    metric_name=condition_data['metric_name'],
                    operator=RuleOperator(condition_data['operator']),
                    value=condition_data['value'],
                    labels=condition_data.get('labels', {}),
                    time_window=timedelta(seconds=condition_data.get('time_window_seconds', 300)),
                    aggregation=condition_data.get('aggregation', 'avg'),
                    weight=condition_data.get('weight', 1.0)
                )
                rule.conditions.append(condition)
            
            # Opérateur logique
            if 'logical_operator' in rule_data:
                rule.logical_operator = LogicalOperator(rule_data['logical_operator'])
            
            # Stockage
            self.rules[rule.id] = rule
            await self._persist_rule(rule)
            
            # Initialisation ML si nécessaire
            if rule.learning_enabled:
                await self._initialize_rule_ml_model(rule)
            
            self.logger.info(f"Règle créée: {rule.id} - {rule.name}")
            return rule
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la création de règle: {e}")
            raise
    
    async def evaluate_rules(self, metrics_data: List[Dict[str, Any]]) -> List[RuleEvaluationResult]:
        """Évaluation de toutes les règles actives"""
        results = []
        
        try:
            # Préparation des données de métriques
            metrics_by_name = defaultdict(list)
            for metric in metrics_data:
                metrics_by_name[metric['name']].append(metric)
            
            # Évaluation de chaque règle
            for rule in self.rules.values():
                if not rule.enabled:
                    continue
                
                # Vérification du cooldown
                if self._is_in_cooldown(rule):
                    continue
                
                # Évaluation de la règle
                result = await self._evaluate_single_rule(rule, metrics_by_name)
                results.append(result)
                
                # Mise à jour des statistiques
                await self._update_rule_statistics(rule, result)
            
            # Sauvegarde de l'historique
            self.evaluation_history.extend(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'évaluation des règles: {e}")
            return []
    
    async def evaluate_single_rule(self, rule_id: str, metrics_data: List[Dict[str, Any]]) -> Optional[RuleEvaluationResult]:
        """Évaluation d'une règle spécifique"""
        if rule_id not in self.rules:
            return None
        
        rule = self.rules[rule_id]
        metrics_by_name = defaultdict(list)
        for metric in metrics_data:
            metrics_by_name[metric['name']].append(metric)
        
        return await self._evaluate_single_rule(rule, metrics_by_name)
    
    async def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Mise à jour d'une règle existante"""
        if rule_id not in self.rules:
            return False
        
        try:
            rule = self.rules[rule_id]
            
            # Mise à jour des champs autorisés
            for field, value in updates.items():
                if hasattr(rule, field) and field not in ['id', 'created_at']:
                    setattr(rule, field, value)
            
            rule.updated_at = datetime.utcnow()
            
            # Persistance
            await self._persist_rule(rule)
            
            self.logger.info(f"Règle mise à jour: {rule_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour de règle: {e}")
            return False
    
    async def delete_rule(self, rule_id: str) -> bool:
        """Suppression d'une règle"""
        if rule_id not in self.rules:
            return False
        
        try:
            del self.rules[rule_id]
            await self.redis_client.hdel('alert_rules', rule_id)
            
            self.logger.info(f"Règle supprimée: {rule_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la suppression de règle: {e}")
            return False
    
    async def get_rule_performance(self, rule_id: str) -> Dict[str, Any]:
        """Analyse de performance d'une règle"""
        if rule_id not in self.rules:
            return {}
        
        rule = self.rules[rule_id]
        performance = self.rule_performance.get(rule_id, {})
        
        return {
            'rule_id': rule_id,
            'name': rule.name,
            'trigger_count': rule.trigger_count,
            'false_positive_count': rule.false_positive_count,
            'accuracy_score': rule.accuracy_score,
            'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None,
            'avg_confidence': performance.get('avg_confidence', 0.0),
            'trigger_rate': performance.get('trigger_rate', 0.0),
            'effectiveness_score': performance.get('effectiveness_score', 0.0)
        }
    
    async def optimize_rule_thresholds(self, rule_id: str) -> Dict[str, Any]:
        """Optimisation automatique des seuils d'une règle"""
        if rule_id not in self.rules or not self.rules[rule_id].adaptive_thresholds:
            return {}
        
        try:
            rule = self.rules[rule_id]
            optimizations = {}
            
            for condition in rule.conditions:
                metric_key = f"{rule.tenant_id}:{condition.metric_name}"
                
                if metric_key in self.threshold_learning_data:
                    data = self.threshold_learning_data[metric_key]
                    
                    if len(data) >= 100:  # Minimum de données
                        # Calcul des seuils optimaux
                        optimized_threshold = await self._calculate_optimal_threshold(
                            data, condition.operator, condition.value
                        )
                        
                        if optimized_threshold != condition.value:
                            optimizations[condition.metric_name] = {
                                'old_threshold': condition.value,
                                'new_threshold': optimized_threshold,
                                'improvement_expected': True
                            }
                            
                            # Mise à jour du seuil
                            condition.value = optimized_threshold
            
            if optimizations:
                rule.updated_at = datetime.utcnow()
                await self._persist_rule(rule)
                
                self.logger.info(f"Seuils optimisés pour la règle {rule_id}: {optimizations}")
            
            return optimizations
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'optimisation des seuils: {e}")
            return {}
    
    def register_custom_operator(self, name: str, operator_func: Callable):
        """Enregistrement d'un opérateur personnalisé"""
        self.custom_operators[name] = operator_func
        self.logger.info(f"Opérateur personnalisé enregistré: {name}")
    
    async def _evaluate_single_rule(self, rule: AlertRule, metrics_by_name: Dict[str, List]) -> RuleEvaluationResult:
        """Évaluation d'une règle individuelle"""
        try:
            evaluation_time = datetime.utcnow()
            condition_results = []
            metric_values = {}
            
            # Évaluation de chaque condition
            for condition in rule.conditions:
                condition_result = await self._evaluate_condition(condition, metrics_by_name)
                condition_results.append(condition_result)
                
                if 'aggregated_value' in condition_result:
                    metric_values[condition.metric_name] = condition_result['aggregated_value']
            
            # Application de la logique
            triggered = self._apply_logical_operator(
                rule.logical_operator,
                [result['satisfied'] for result in condition_results]
            )
            
            # Calcul de la confiance avec logique floue
            confidence = self._calculate_fuzzy_confidence(rule, condition_results)
            
            # Prédiction ML si disponible
            predicted_severity = ""
            anomaly_score = 0.0
            
            if rule.learning_enabled and rule.id in self.threshold_predictors:
                ml_result = await self._apply_ml_prediction(rule, metric_values)
                predicted_severity = ml_result.get('severity', '')
                anomaly_score = ml_result.get('anomaly_score', 0.0)
                
                # Ajustement de la confiance avec ML
                confidence = 0.7 * confidence + 0.3 * ml_result.get('confidence', 0.0)
            
            # Génération d'explication
            explanation = self._generate_explanation(rule, condition_results, triggered)
            
            return RuleEvaluationResult(
                rule_id=rule.id,
                triggered=triggered and confidence >= rule.confidence_threshold,
                confidence=confidence,
                evaluation_time=evaluation_time,
                metric_values=metric_values,
                condition_results=condition_results,
                anomaly_score=anomaly_score,
                predicted_severity=predicted_severity,
                explanation=explanation
            )
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'évaluation de la règle {rule.id}: {e}")
            return RuleEvaluationResult(
                rule_id=rule.id,
                triggered=False,
                confidence=0.0,
                evaluation_time=datetime.utcnow(),
                metric_values={},
                condition_results=[],
                explanation=f"Erreur d'évaluation: {e}"
            )
    
    async def _evaluate_condition(self, condition: RuleCondition, metrics_by_name: Dict[str, List]) -> Dict[str, Any]:
        """Évaluation d'une condition individuelle"""
        try:
            metrics = metrics_by_name.get(condition.metric_name, [])
            
            if not metrics:
                return {
                    'satisfied': False,
                    'reason': f"Aucune métrique trouvée pour {condition.metric_name}",
                    'aggregated_value': None
                }
            
            # Filtrage par labels
            filtered_metrics = self._filter_metrics_by_labels(metrics, condition.labels)
            
            if not filtered_metrics:
                return {
                    'satisfied': False,
                    'reason': f"Aucune métrique correspondant aux labels",
                    'aggregated_value': None
                }
            
            # Filtrage par fenêtre temporelle
            now = datetime.utcnow()
            start_time = now - condition.time_window
            
            recent_metrics = [
                m for m in filtered_metrics
                if start_time <= datetime.fromisoformat(m['timestamp'].replace('Z', '+00:00')) <= now
            ]
            
            if not recent_metrics:
                return {
                    'satisfied': False,
                    'reason': f"Aucune métrique récente dans la fenêtre {condition.time_window}",
                    'aggregated_value': None
                }
            
            # Agrégation
            values = [m['value'] for m in recent_metrics]
            aggregated_value = self._aggregate_values(values, condition.aggregation)
            
            # Évaluation de l'opérateur
            satisfied = self._evaluate_operator(
                aggregated_value,
                condition.operator,
                condition.value
            )
            
            return {
                'satisfied': satisfied,
                'aggregated_value': aggregated_value,
                'sample_count': len(values),
                'operator': condition.operator.value,
                'threshold': condition.value,
                'reason': f"{condition.aggregation}({condition.metric_name}) = {aggregated_value} {condition.operator.value} {condition.value}"
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'évaluation de condition: {e}")
            return {
                'satisfied': False,
                'reason': f"Erreur: {e}",
                'aggregated_value': None
            }
    
    def _calculate_fuzzy_confidence(self, rule: AlertRule, condition_results: List[Dict[str, Any]]) -> float:
        """Calcul de confiance avec logique floue"""
        if not condition_results:
            return 0.0
        
        weighted_scores = []
        total_weight = 0.0
        
        for i, condition in enumerate(rule.conditions):
            if i < len(condition_results):
                result = condition_results[i]
                
                # Score basé sur la satisfaction de la condition
                base_score = 1.0 if result['satisfied'] else 0.0
                
                # Ajustement basé sur la proximité du seuil
                if result['aggregated_value'] is not None and isinstance(condition.value, (int, float)):
                    proximity = self._calculate_threshold_proximity(
                        result['aggregated_value'],
                        condition.value,
                        condition.operator
                    )
                    base_score *= proximity
                
                weighted_scores.append(base_score * condition.weight)
                total_weight += condition.weight
        
        if total_weight == 0:
            return 0.0
        
        return sum(weighted_scores) / total_weight
    
    def _calculate_threshold_proximity(self, value: float, threshold: float, operator: RuleOperator) -> float:
        """Calcul de la proximité d'une valeur par rapport à un seuil"""
        if operator in [RuleOperator.GREATER_THAN, RuleOperator.GREATER_EQUAL]:
            if value >= threshold:
                return 1.0
            else:
                return max(0.0, value / threshold)
        
        elif operator in [RuleOperator.LESS_THAN, RuleOperator.LESS_EQUAL]:
            if value <= threshold:
                return 1.0
            else:
                return max(0.0, threshold / value)
        
        elif operator == RuleOperator.EQUALS:
            diff = abs(value - threshold)
            max_diff = max(abs(threshold * 0.1), 1.0)  # 10% de tolérance
            return max(0.0, 1.0 - (diff / max_diff))
        
        return 1.0 if value else 0.0
