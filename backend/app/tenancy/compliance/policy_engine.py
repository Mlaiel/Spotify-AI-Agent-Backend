"""
Spotify AI Agent - PolicyEngine Ultra-Avancé
===========================================

Moteur de politiques intelligent avec automatisation complète, IA prédictive
et gestion dynamique des règles de conformité multi-framework.

Développé par l'équipe d'experts Policy Management & AI
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from uuid import uuid4
from collections import defaultdict
import numpy as np

class PolicyType(Enum):
    """Types de politiques"""
    ACCESS_CONTROL = "access_control"
    DATA_PROTECTION = "data_protection"
    CONTENT_MANAGEMENT = "content_management"
    PRIVACY_ENFORCEMENT = "privacy_enforcement"
    COMPLIANCE_REQUIREMENT = "compliance_requirement"
    SECURITY_POLICY = "security_policy"
    MUSIC_LICENSING = "music_licensing"
    GEOGRAPHIC_RESTRICTION = "geographic_restriction"
    USER_BEHAVIOR = "user_behavior"
    AUTOMATED_DECISION = "automated_decision"

class PolicyStatus(Enum):
    """États des politiques"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    DEPRECATED = "deprecated"
    TESTING = "testing"

class PolicyAction(Enum):
    """Actions de politique"""
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    LOG_AND_ALLOW = "log_and_allow"
    LOG_AND_DENY = "log_and_deny"
    CONDITIONAL = "conditional"
    ESCALATE = "escalate"

class PolicyPriority(Enum):
    """Priorités des politiques"""
    CRITICAL = 100
    HIGH = 80
    MEDIUM = 60
    LOW = 40
    INFORMATIONAL = 20

class ComplianceFramework(Enum):
    """Frameworks de conformité"""
    GDPR = "gdpr"
    SOX = "sox"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"
    MUSIC_INDUSTRY = "music_industry"
    CCPA = "ccpa"

@dataclass
class PolicyCondition:
    """Condition d'application d'une politique"""
    field: str
    operator: str  # eq, ne, gt, lt, in, not_in, contains, regex, etc.
    value: Any
    description: str = ""
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Évaluation de la condition"""
        
        field_value = self._get_field_value(context, self.field)
        
        if field_value is None:
            return False
        
        try:
            if self.operator == "eq":
                return field_value == self.value
            elif self.operator == "ne":
                return field_value != self.value
            elif self.operator == "gt":
                return field_value > self.value
            elif self.operator == "lt":
                return field_value < self.value
            elif self.operator == "gte":
                return field_value >= self.value
            elif self.operator == "lte":
                return field_value <= self.value
            elif self.operator == "in":
                return field_value in self.value
            elif self.operator == "not_in":
                return field_value not in self.value
            elif self.operator == "contains":
                return self.value in str(field_value)
            elif self.operator == "not_contains":
                return self.value not in str(field_value)
            elif self.operator == "regex":
                return bool(re.search(self.value, str(field_value)))
            elif self.operator == "exists":
                return True  # Si on arrive ici, le champ existe
            elif self.operator == "not_exists":
                return False  # Si on arrive ici, le champ existe
            else:
                return False
                
        except Exception:
            return False
    
    def _get_field_value(self, context: Dict[str, Any], field_path: str) -> Any:
        """Récupération de la valeur d'un champ avec support des chemins imbriqués"""
        
        try:
            value = context
            for part in field_path.split('.'):
                if isinstance(value, dict):
                    value = value.get(part)
                elif isinstance(value, list) and part.isdigit():
                    value = value[int(part)] if int(part) < len(value) else None
                else:
                    return None
            return value
        except Exception:
            return None

@dataclass
class PolicyRule:
    """Règle de politique complète"""
    rule_id: str
    name: str
    description: str
    policy_type: PolicyType
    priority: PolicyPriority
    status: PolicyStatus
    
    # Conditions d'application
    conditions: List[PolicyCondition] = field(default_factory=list)
    condition_logic: str = "AND"  # AND, OR
    
    # Actions et conséquences
    action: PolicyAction = PolicyAction.ALLOW
    action_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Métadonnées
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    # Gestion temporelle
    effective_date: datetime = field(default_factory=datetime.utcnow)
    expiry_date: Optional[datetime] = None
    
    # Métriques et monitoring
    execution_count: int = 0
    last_executed: Optional[datetime] = None
    
    # IA et apprentissage
    confidence_score: float = 1.0
    auto_generated: bool = False
    learning_enabled: bool = False
    
    def is_applicable(self, context: Dict[str, Any], current_time: datetime = None) -> bool:
        """Vérification si la règle s'applique au contexte donné"""
        
        current_time = current_time or datetime.utcnow()
        
        # Vérifications temporelles
        if current_time < self.effective_date:
            return False
        
        if self.expiry_date and current_time > self.expiry_date:
            return False
        
        # Vérification du statut
        if self.status != PolicyStatus.ACTIVE:
            return False
        
        # Évaluation des conditions
        if not self.conditions:
            return True
        
        if self.condition_logic == "AND":
            return all(condition.evaluate(context) for condition in self.conditions)
        elif self.condition_logic == "OR":
            return any(condition.evaluate(context) for condition in self.conditions)
        else:
            return False
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Exécution de la règle de politique"""
        
        self.execution_count += 1
        self.last_executed = datetime.utcnow()
        
        result = {
            'rule_id': self.rule_id,
            'rule_name': self.name,
            'action': self.action.value,
            'applied': True,
            'priority': self.priority.value,
            'confidence': self.confidence_score,
            'execution_timestamp': self.last_executed.isoformat(),
            'context_matched': True
        }
        
        # Ajout des paramètres d'action
        if self.action_parameters:
            result['action_parameters'] = self.action_parameters.copy()
        
        # Logging spécialisé selon l'action
        if self.action in [PolicyAction.LOG_AND_ALLOW, PolicyAction.LOG_AND_DENY]:
            result['requires_logging'] = True
            result['log_message'] = f"Politique {self.name} appliquée: {self.action.value}"
        
        return result

class PolicyValidator:
    """
    Validateur de politiques avec vérifications avancées
    """
    
    def __init__(self):
        self.logger = logging.getLogger("policy.validator")
    
    async def validate_rule(self, rule: PolicyRule) -> Dict[str, Any]:
        """Validation complète d'une règle de politique"""
        
        errors = []
        warnings = []
        suggestions = []
        
        # Validation de base
        if not rule.name or not rule.name.strip():
            errors.append("Le nom de la règle est requis")
        
        if not rule.description or not rule.description.strip():
            warnings.append("Description de la règle manquante")
        
        # Validation des conditions
        condition_validation = await self._validate_conditions(rule.conditions)
        errors.extend(condition_validation.get('errors', []))
        warnings.extend(condition_validation.get('warnings', []))
        
        # Validation de la logique
        if rule.condition_logic not in ["AND", "OR"] and rule.conditions:
            errors.append("Logique de condition invalide (doit être AND ou OR)")
        
        # Validation temporelle
        if rule.expiry_date and rule.expiry_date <= rule.effective_date:
            errors.append("La date d'expiration doit être postérieure à la date d'effet")
        
        # Validation des frameworks de conformité
        framework_validation = await self._validate_compliance_frameworks(rule)
        warnings.extend(framework_validation.get('warnings', []))
        suggestions.extend(framework_validation.get('suggestions', []))
        
        # Validation de priorité et conflits
        conflict_validation = await self._validate_conflicts(rule)
        warnings.extend(conflict_validation.get('warnings', []))
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'suggestions': suggestions,
            'validation_score': self._calculate_validation_score(errors, warnings),
            'validated_at': datetime.utcnow().isoformat()
        }
    
    async def _validate_conditions(self, conditions: List[PolicyCondition]) -> Dict[str, List[str]]:
        """Validation des conditions"""
        
        errors = []
        warnings = []
        
        for i, condition in enumerate(conditions):
            # Validation de l'opérateur
            valid_operators = [
                "eq", "ne", "gt", "lt", "gte", "lte",
                "in", "not_in", "contains", "not_contains",
                "regex", "exists", "not_exists"
            ]
            
            if condition.operator not in valid_operators:
                errors.append(f"Condition {i}: Opérateur '{condition.operator}' invalide")
            
            # Validation des champs
            if not condition.field:
                errors.append(f"Condition {i}: Champ requis")
            
            # Validation des regex
            if condition.operator == "regex":
                try:
                    re.compile(condition.value)
                except re.error:
                    errors.append(f"Condition {i}: Expression régulière invalide")
            
            # Validation des valeurs pour opérateurs in/not_in
            if condition.operator in ["in", "not_in"] and not isinstance(condition.value, (list, tuple, set)):
                warnings.append(f"Condition {i}: Valeur devrait être une liste pour l'opérateur '{condition.operator}'")
        
        return {'errors': errors, 'warnings': warnings}
    
    async def _validate_compliance_frameworks(self, rule: PolicyRule) -> Dict[str, List[str]]:
        """Validation des frameworks de conformité"""
        
        warnings = []
        suggestions = []
        
        # Suggestions basées sur le type de politique
        if rule.policy_type == PolicyType.DATA_PROTECTION:
            if ComplianceFramework.GDPR not in rule.compliance_frameworks:
                suggestions.append("Considérer l'ajout du framework GDPR pour la protection des données")
        
        elif rule.policy_type == PolicyType.MUSIC_LICENSING:
            if ComplianceFramework.MUSIC_INDUSTRY not in rule.compliance_frameworks:
                suggestions.append("Considérer l'ajout du framework MUSIC_INDUSTRY")
        
        elif rule.policy_type == PolicyType.SECURITY_POLICY:
            if ComplianceFramework.ISO27001 not in rule.compliance_frameworks:
                suggestions.append("Considérer l'ajout du framework ISO27001 pour les politiques de sécurité")
        
        return {'warnings': warnings, 'suggestions': suggestions}
    
    async def _validate_conflicts(self, rule: PolicyRule) -> Dict[str, List[str]]:
        """Validation des conflits potentiels"""
        
        warnings = []
        
        # Cette méthode pourrait être étendue pour vérifier les conflits
        # avec d'autres règles existantes
        
        return {'warnings': warnings}
    
    def _calculate_validation_score(self, errors: List[str], warnings: List[str]) -> float:
        """Calcul du score de validation"""
        
        if errors:
            return 0.0
        
        warning_penalty = len(warnings) * 0.1
        return max(1.0 - warning_penalty, 0.0)

class PolicyAI:
    """
    Intelligence artificielle pour gestion prédictive des politiques
    """
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.logger = logging.getLogger(f"policy.ai.{tenant_id}")
        
        # Modèles d'apprentissage
        self._pattern_detection_model = {}
        self._recommendation_engine = {}
        self._anomaly_detection_model = {}
        
        # Historique et données d'apprentissage
        self._execution_history = defaultdict(list)
        self._context_patterns = defaultdict(list)
        self._effectiveness_metrics = {}
    
    async def suggest_policy_improvements(self, rule: PolicyRule) -> List[Dict[str, Any]]:
        """Suggestions d'amélioration basées sur l'IA"""
        
        suggestions = []
        
        # Analyse de l'efficacité
        if rule.execution_count > 10:
            effectiveness = await self._analyze_rule_effectiveness(rule)
            
            if effectiveness < 0.7:
                suggestions.append({
                    'type': 'effectiveness_improvement',
                    'priority': 'high',
                    'description': 'Cette règle semble peu efficace. Considérer une révision des conditions.',
                    'confidence': 0.8,
                    'data': {'current_effectiveness': effectiveness}
                })
        
        # Analyse des patterns d'exécution
        patterns = await self._detect_execution_patterns(rule)
        for pattern in patterns:
            suggestions.append({
                'type': 'pattern_optimization',
                'priority': 'medium',
                'description': f"Pattern détecté: {pattern['description']}",
                'confidence': pattern['confidence'],
                'recommendation': pattern['recommendation']
            })
        
        # Suggestions de conditions manquantes
        missing_conditions = await self._suggest_missing_conditions(rule)
        for condition in missing_conditions:
            suggestions.append({
                'type': 'missing_condition',
                'priority': 'medium',
                'description': f"Condition suggérée: {condition['description']}",
                'confidence': condition['confidence'],
                'suggested_condition': condition
            })
        
        return suggestions
    
    async def predict_policy_impact(self, rule: PolicyRule, context_sample: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prédiction de l'impact d'une politique"""
        
        predictions = {
            'estimated_application_rate': 0.0,
            'expected_allow_rate': 0.0,
            'expected_deny_rate': 0.0,
            'risk_level': 'low',
            'confidence': 0.5
        }
        
        if not context_sample:
            return predictions
        
        # Simulation de l'application de la règle
        applicable_count = 0
        allow_count = 0
        deny_count = 0
        
        for context in context_sample:
            if rule.is_applicable(context):
                applicable_count += 1
                
                if rule.action in [PolicyAction.ALLOW, PolicyAction.LOG_AND_ALLOW]:
                    allow_count += 1
                elif rule.action in [PolicyAction.DENY, PolicyAction.LOG_AND_DENY]:
                    deny_count += 1
        
        total_contexts = len(context_sample)
        
        predictions.update({
            'estimated_application_rate': applicable_count / total_contexts if total_contexts > 0 else 0,
            'expected_allow_rate': allow_count / applicable_count if applicable_count > 0 else 0,
            'expected_deny_rate': deny_count / applicable_count if applicable_count > 0 else 0,
            'confidence': min(0.7 + (total_contexts / 1000), 0.95)
        })
        
        # Évaluation du niveau de risque
        if predictions['expected_deny_rate'] > 0.5:
            predictions['risk_level'] = 'high'
        elif predictions['expected_deny_rate'] > 0.2:
            predictions['risk_level'] = 'medium'
        
        return predictions
    
    async def auto_generate_policy(self, context_patterns: List[Dict[str, Any]], policy_goal: str) -> Optional[PolicyRule]:
        """Génération automatique de politique basée sur les patterns"""
        
        if len(context_patterns) < 10:  # Pas assez de données
            return None
        
        # Analyse des patterns pour identifier les conditions communes
        common_fields = self._identify_common_fields(context_patterns)
        suggested_conditions = self._generate_conditions_from_patterns(common_fields, context_patterns)
        
        if not suggested_conditions:
            return None
        
        # Génération de la règle
        rule_id = f"auto_{uuid4().hex[:8]}"
        
        rule = PolicyRule(
            rule_id=rule_id,
            name=f"Règle auto-générée - {policy_goal}",
            description=f"Règle générée automatiquement basée sur l'analyse de {len(context_patterns)} patterns",
            policy_type=PolicyType.USER_BEHAVIOR,  # Type par défaut
            priority=PolicyPriority.MEDIUM,
            status=PolicyStatus.TESTING,  # Commencer en mode test
            conditions=suggested_conditions,
            action=PolicyAction.LOG_AND_ALLOW,  # Action conservatrice par défaut
            auto_generated=True,
            learning_enabled=True,
            confidence_score=0.6  # Score initial conservateur
        )
        
        return rule
    
    def _identify_common_fields(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identification des champs communs dans les patterns"""
        
        field_frequency = defaultdict(int)
        field_values = defaultdict(set)
        
        for pattern in patterns:
            for field, value in pattern.items():
                field_frequency[field] += 1
                if isinstance(value, (str, int, float, bool)):
                    field_values[field].add(value)
        
        # Garder seulement les champs présents dans au moins 70% des patterns
        threshold = len(patterns) * 0.7
        common_fields = {
            field: {
                'frequency': count,
                'values': list(field_values[field]),
                'frequency_ratio': count / len(patterns)
            }
            for field, count in field_frequency.items()
            if count >= threshold
        }
        
        return common_fields
    
    def _generate_conditions_from_patterns(
        self,
        common_fields: Dict[str, Any],
        patterns: List[Dict[str, Any]]
    ) -> List[PolicyCondition]:
        """Génération de conditions basées sur les champs communs"""
        
        conditions = []
        
        for field, field_info in common_fields.items():
            values = field_info['values']
            frequency_ratio = field_info['frequency_ratio']
            
            # Si le champ a peu de valeurs distinctes, utiliser "in"
            if len(values) <= 5 and frequency_ratio > 0.8:
                condition = PolicyCondition(
                    field=field,
                    operator="in",
                    value=values,
                    description=f"Auto-généré: {field} dans les valeurs communes"
                )
                conditions.append(condition)
            
            # Si le champ est numérique, chercher des seuils
            elif all(isinstance(v, (int, float)) for v in values):
                avg_value = sum(values) / len(values)
                condition = PolicyCondition(
                    field=field,
                    operator="gte",
                    value=avg_value,
                    description=f"Auto-généré: {field} supérieur à la moyenne ({avg_value:.2f})"
                )
                conditions.append(condition)
        
        return conditions[:5]  # Limiter à 5 conditions maximum
    
    async def _analyze_rule_effectiveness(self, rule: PolicyRule) -> float:
        """Analyse de l'efficacité d'une règle"""
        
        if rule.execution_count == 0:
            return 0.5  # Score neutre sans données
        
        # Simulation d'analyse d'efficacité
        # Dans un vrai système, cela analyserait les résultats réels
        
        base_effectiveness = 0.8
        
        # Pénalité pour les règles très fréquemment exécutées
        if rule.execution_count > 1000:
            base_effectiveness -= 0.1
        
        # Bonus pour les règles avec confiance élevée
        confidence_bonus = (rule.confidence_score - 0.5) * 0.2
        
        return max(0.0, min(1.0, base_effectiveness + confidence_bonus))
    
    async def _detect_execution_patterns(self, rule: PolicyRule) -> List[Dict[str, Any]]:
        """Détection de patterns d'exécution"""
        
        patterns = []
        
        # Pattern basé sur la fréquence d'exécution
        if rule.execution_count > 100:
            patterns.append({
                'type': 'high_frequency',
                'description': 'Cette règle est exécutée très fréquemment',
                'confidence': 0.9,
                'recommendation': 'Considérer l\'optimisation des conditions pour réduire la charge'
            })
        
        # Pattern basé sur l\'âge de la règle
        if rule.last_executed:
            age_days = (datetime.utcnow() - rule.effective_date).days
            if age_days > 365 and rule.execution_count < 10:
                patterns.append({
                    'type': 'low_usage',
                    'description': 'Cette règle ancienne est rarement utilisée',
                    'confidence': 0.8,
                    'recommendation': 'Considérer la révision ou la suppression de cette règle'
                })
        
        return patterns
    
    async def _suggest_missing_conditions(self, rule: PolicyRule) -> List[Dict[str, Any]]:
        """Suggestion de conditions manquantes"""
        
        suggestions = []
        
        # Suggestions basées sur le type de politique
        if rule.policy_type == PolicyType.DATA_PROTECTION:
            # Vérifier si les conditions incluent la géolocalisation
            has_geo_condition = any(
                'country' in condition.field or 'region' in condition.field
                for condition in rule.conditions
            )
            
            if not has_geo_condition:
                suggestions.append({
                    'field': 'user.country_code',
                    'operator': 'in',
                    'value': ['EU', 'US', 'CA'],
                    'description': 'Restriction géographique pour la protection des données',
                    'confidence': 0.7
                })
        
        elif rule.policy_type == PolicyType.MUSIC_LICENSING:
            # Vérifier les conditions de licence
            has_license_condition = any(
                'license' in condition.field.lower()
                for condition in rule.conditions
            )
            
            if not has_license_condition:
                suggestions.append({
                    'field': 'content.license_status',
                    'operator': 'eq',
                    'value': 'valid',
                    'description': 'Vérification du statut de licence musicale',
                    'confidence': 0.9
                })
        
        return suggestions

class PolicyEngine:
    """
    Moteur de politiques central ultra-avancé
    
    Fonctionnalités principales:
    - Gestion dynamique des règles de politique
    - Évaluation en temps réel avec IA
    - Apprentissage automatique et optimisation
    - Support multi-framework
    - Génération automatique de politiques
    """
    
    def __init__(self, tenant_id: str = "default"):
        self.tenant_id = tenant_id
        self.logger = logging.getLogger(f"policy.engine.{tenant_id}")
        
        # Composants spécialisés
        self.validator = PolicyValidator()
        self.ai_engine = PolicyAI(tenant_id)
        
        # Stockage des règles
        self._rules: Dict[str, PolicyRule] = {}
        self._rules_by_type: Dict[PolicyType, List[str]] = defaultdict(list)
        self._rules_by_framework: Dict[ComplianceFramework, List[str]] = defaultdict(list)
        
        # Cache et optimisation
        self._evaluation_cache = {}
        self._cache_ttl = timedelta(minutes=5)
        
        # Métriques et monitoring
        self._metrics = {
            'total_rules': 0,
            'active_rules': 0,
            'evaluations_performed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'ai_suggestions_generated': 0,
            'auto_generated_rules': 0
        }
        
        # Configuration
        self._config = {
            'enable_caching': True,
            'enable_ai_optimization': True,
            'auto_rule_generation': True,
            'max_rules_per_type': 100,
            'evaluation_timeout': 5.0,  # secondes
            'learning_mode': True
        }
        
        self.logger.info(f"PolicyEngine initialisé pour tenant {tenant_id}")
    
    async def add_rule(self, rule: PolicyRule, validate: bool = True) -> Dict[str, Any]:
        """Ajout d'une nouvelle règle de politique"""
        
        if validate:
            validation_result = await self.validator.validate_rule(rule)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': 'Validation échouée',
                    'validation_result': validation_result
                }
        
        # Vérification des limites
        rules_of_type = len(self._rules_by_type[rule.policy_type])
        if rules_of_type >= self._config['max_rules_per_type']:
            return {
                'success': False,
                'error': f'Limite de règles atteinte pour le type {rule.policy_type.value}'
            }
        
        # Ajout de la règle
        self._rules[rule.rule_id] = rule
        self._rules_by_type[rule.policy_type].append(rule.rule_id)
        
        for framework in rule.compliance_frameworks:
            self._rules_by_framework[framework].append(rule.rule_id)
        
        # Mise à jour des métriques
        self._update_metrics()
        
        # Invalidation du cache
        self._evaluation_cache.clear()
        
        # Suggestions d'amélioration IA
        suggestions = []
        if self._config['enable_ai_optimization'] and not rule.auto_generated:
            suggestions = await self.ai_engine.suggest_policy_improvements(rule)
        
        self.logger.info(f"Règle ajoutée: {rule.rule_id} - {rule.name}")
        
        return {
            'success': True,
            'rule_id': rule.rule_id,
            'ai_suggestions': suggestions,
            'validation_result': validation_result if validate else None
        }
    
    async def evaluate_policies(
        self,
        context: Dict[str, Any],
        policy_types: List[PolicyType] = None,
        frameworks: List[ComplianceFramework] = None
    ) -> Dict[str, Any]:
        """Évaluation des politiques pour un contexte donné"""
        
        # Génération de clé de cache
        cache_key = self._generate_cache_key(context, policy_types, frameworks)
        
        # Vérification du cache
        if self._config['enable_caching'] and cache_key in self._evaluation_cache:
            cached_result = self._evaluation_cache[cache_key]
            if datetime.utcnow() - cached_result['timestamp'] < self._cache_ttl:
                self._metrics['cache_hits'] += 1
                return cached_result['result']
        
        self._metrics['cache_misses'] += 1
        
        # Sélection des règles à évaluer
        rules_to_evaluate = self._select_rules_for_evaluation(policy_types, frameworks)
        
        # Évaluation des règles
        evaluation_results = []
        applicable_rules = []
        
        start_time = datetime.utcnow()
        
        for rule_id in rules_to_evaluate:
            rule = self._rules[rule_id]
            
            try:
                # Vérification du timeout
                if (datetime.utcnow() - start_time).total_seconds() > self._config['evaluation_timeout']:
                    self.logger.warning(f"Timeout d'évaluation atteint, arrêt à la règle {rule_id}")
                    break
                
                if rule.is_applicable(context):
                    applicable_rules.append(rule)
                    result = rule.execute(context)
                    evaluation_results.append(result)
            
            except Exception as e:
                self.logger.error(f"Erreur lors de l'évaluation de la règle {rule_id}: {e}")
                evaluation_results.append({
                    'rule_id': rule_id,
                    'error': str(e),
                    'applied': False
                })
        
        # Résolution des conflits et priorisation
        final_decision = self._resolve_policy_conflicts(evaluation_results)
        
        # Résultat final
        result = {
            'decision': final_decision,
            'applicable_rules_count': len(applicable_rules),
            'total_rules_evaluated': len(rules_to_evaluate),
            'evaluation_results': evaluation_results,
            'evaluation_time_ms': (datetime.utcnow() - start_time).total_seconds() * 1000,
            'cache_key': cache_key,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Mise en cache
        if self._config['enable_caching']:
            self._evaluation_cache[cache_key] = {
                'result': result,
                'timestamp': datetime.utcnow()
            }
        
        # Mise à jour des métriques
        self._metrics['evaluations_performed'] += 1
        
        # Apprentissage automatique
        if self._config['learning_mode']:
            await self._learn_from_evaluation(context, applicable_rules, result)
        
        return result
    
    def _select_rules_for_evaluation(
        self,
        policy_types: List[PolicyType] = None,
        frameworks: List[ComplianceFramework] = None
    ) -> List[str]:
        """Sélection des règles à évaluer"""
        
        selected_rules = set()
        
        # Sélection par type de politique
        if policy_types:
            for policy_type in policy_types:
                selected_rules.update(self._rules_by_type[policy_type])
        
        # Sélection par framework
        if frameworks:
            for framework in frameworks:
                selected_rules.update(self._rules_by_framework[framework])
        
        # Si aucun filtre, évaluer toutes les règles actives
        if not policy_types and not frameworks:
            selected_rules = {
                rule_id for rule_id, rule in self._rules.items()
                if rule.status == PolicyStatus.ACTIVE
            }
        
        # Tri par priorité
        return sorted(selected_rules, key=lambda rid: self._rules[rid].priority.value, reverse=True)
    
    def _resolve_policy_conflicts(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Résolution des conflits entre politiques"""
        
        # Filtrer les résultats appliqués avec succès
        applied_results = [r for r in evaluation_results if r.get('applied', False)]
        
        if not applied_results:
            return {
                'action': PolicyAction.ALLOW.value,
                'reason': 'Aucune politique applicable',
                'confidence': 1.0
            }
        
        # Tri par priorité
        applied_results.sort(key=lambda r: r.get('priority', 0), reverse=True)
        
        # La règle de plus haute priorité gagne
        winning_rule = applied_results[0]
        
        # Vérification des conflits
        conflicts = []
        for result in applied_results[1:]:
            if result['action'] != winning_rule['action']:
                conflicts.append({
                    'rule_id': result['rule_id'],
                    'conflicting_action': result['action'],
                    'priority': result.get('priority', 0)
                })
        
        decision = {
            'action': winning_rule['action'],
            'winning_rule': {
                'rule_id': winning_rule['rule_id'],
                'rule_name': winning_rule.get('rule_name', 'Unknown'),
                'priority': winning_rule.get('priority', 0)
            },
            'confidence': winning_rule.get('confidence', 1.0),
            'conflicts_detected': len(conflicts),
            'conflicts': conflicts
        }
        
        if conflicts:
            decision['resolution_method'] = 'highest_priority'
            self.logger.warning(f"Conflits de politique résolus par priorité: {len(conflicts)} conflits")
        
        return decision
    
    def _generate_cache_key(
        self,
        context: Dict[str, Any],
        policy_types: List[PolicyType] = None,
        frameworks: List[ComplianceFramework] = None
    ) -> str:
        """Génération de clé de cache pour l'évaluation"""
        
        import hashlib
        
        # Créer une représentation stable du contexte
        context_str = json.dumps(context, sort_keys=True, default=str)
        types_str = str(sorted([pt.value for pt in (policy_types or [])]))
        frameworks_str = str(sorted([fw.value for fw in (frameworks or [])]))
        
        combined = f"{context_str}_{types_str}_{frameworks_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _update_metrics(self):
        """Mise à jour des métriques"""
        
        self._metrics['total_rules'] = len(self._rules)
        self._metrics['active_rules'] = len([
            r for r in self._rules.values()
            if r.status == PolicyStatus.ACTIVE
        ])
    
    async def _learn_from_evaluation(
        self,
        context: Dict[str, Any],
        applicable_rules: List[PolicyRule],
        result: Dict[str, Any]
    ):
        """Apprentissage automatique à partir des évaluations"""
        
        # Mise à jour des patterns contextuels
        for rule in applicable_rules:
            if rule.learning_enabled:
                self.ai_engine._context_patterns[rule.rule_id].append({
                    'context': context.copy(),
                    'timestamp': datetime.utcnow(),
                    'result': result['decision'].copy()
                })
                
                # Limiter l'historique
                if len(self.ai_engine._context_patterns[rule.rule_id]) > 1000:
                    self.ai_engine._context_patterns[rule.rule_id].pop(0)
        
        # Génération automatique de nouvelles règles si activée
        if self._config['auto_rule_generation']:
            await self._auto_generate_rules_from_patterns()
    
    async def _auto_generate_rules_from_patterns(self):
        """Génération automatique de règles basée sur les patterns"""
        
        # Analyser les patterns tous les 100 évaluations
        if self._metrics['evaluations_performed'] % 100 == 0:
            
            # Collecter tous les patterns contextuels
            all_patterns = []
            for patterns_list in self.ai_engine._context_patterns.values():
                all_patterns.extend([p['context'] for p in patterns_list[-50:]])  # 50 derniers
            
            if len(all_patterns) >= 50:
                # Tentative de génération d'une nouvelle règle
                new_rule = await self.ai_engine.auto_generate_policy(
                    all_patterns,
                    "Optimisation comportementale"
                )
                
                if new_rule:
                    result = await self.add_rule(new_rule, validate=True)
                    if result['success']:
                        self._metrics['auto_generated_rules'] += 1
                        self.logger.info(f"Règle auto-générée: {new_rule.rule_id}")
    
    async def optimize_rules(self) -> Dict[str, Any]:
        """Optimisation automatique des règles existantes"""
        
        optimization_results = {
            'rules_analyzed': 0,
            'suggestions_generated': 0,
            'rules_optimized': 0,
            'recommendations': []
        }
        
        for rule_id, rule in self._rules.items():
            optimization_results['rules_analyzed'] += 1
            
            # Suggestions d'amélioration IA
            if self._config['enable_ai_optimization']:
                suggestions = await self.ai_engine.suggest_policy_improvements(rule)
                
                if suggestions:
                    optimization_results['suggestions_generated'] += len(suggestions)
                    optimization_results['recommendations'].append({
                        'rule_id': rule_id,
                        'rule_name': rule.name,
                        'suggestions': suggestions
                    })
                    
                    # Application automatique des optimisations à haute confiance
                    high_confidence_suggestions = [
                        s for s in suggestions
                        if s.get('confidence', 0) > 0.85 and s.get('priority') == 'high'
                    ]
                    
                    if high_confidence_suggestions:
                        # Ici, on pourrait appliquer automatiquement certaines optimisations
                        optimization_results['rules_optimized'] += 1
        
        return optimization_results
    
    async def get_policy_metrics(self) -> Dict[str, Any]:
        """Récupération des métriques du moteur de politiques"""
        
        # Métriques par type de politique
        type_metrics = {}
        for policy_type in PolicyType:
            rules = [self._rules[rid] for rid in self._rules_by_type[policy_type]]
            type_metrics[policy_type.value] = {
                'total_rules': len(rules),
                'active_rules': len([r for r in rules if r.status == PolicyStatus.ACTIVE]),
                'total_executions': sum(r.execution_count for r in rules),
                'avg_confidence': sum(r.confidence_score for r in rules) / len(rules) if rules else 0
            }
        
        # Métriques par framework
        framework_metrics = {}
        for framework in ComplianceFramework:
            rules = [self._rules[rid] for rid in self._rules_by_framework[framework]]
            framework_metrics[framework.value] = {
                'total_rules': len(rules),
                'active_rules': len([r for r in rules if r.status == PolicyStatus.ACTIVE]),
                'total_executions': sum(r.execution_count for r in rules)
            }
        
        return {
            'tenant_id': self.tenant_id,
            'overall_metrics': self._metrics.copy(),
            'type_metrics': type_metrics,
            'framework_metrics': framework_metrics,
            'cache_stats': {
                'cache_size': len(self._evaluation_cache),
                'hit_rate': (self._metrics['cache_hits'] / 
                           (self._metrics['cache_hits'] + self._metrics['cache_misses']) 
                           if (self._metrics['cache_hits'] + self._metrics['cache_misses']) > 0 else 0)
            },
            'configuration': self._config.copy(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def get_rule(self, rule_id: str) -> Optional[PolicyRule]:
        """Récupération d'une règle par ID"""
        return self._rules.get(rule_id)
    
    async def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Mise à jour d'une règle existante"""
        
        if rule_id not in self._rules:
            return {'success': False, 'error': 'Règle non trouvée'}
        
        rule = self._rules[rule_id]
        
        # Application des mises à jour
        for field, value in updates.items():
            if hasattr(rule, field):
                setattr(rule, field, value)
        
        # Validation de la règle mise à jour
        validation_result = await self.validator.validate_rule(rule)
        
        if not validation_result['valid']:
            return {
                'success': False,
                'error': 'Validation échouée après mise à jour',
                'validation_result': validation_result
            }
        
        # Invalidation du cache
        self._evaluation_cache.clear()
        
        self.logger.info(f"Règle mise à jour: {rule_id}")
        
        return {
            'success': True,
            'rule_id': rule_id,
            'validation_result': validation_result
        }
    
    async def delete_rule(self, rule_id: str) -> Dict[str, Any]:
        """Suppression d'une règle"""
        
        if rule_id not in self._rules:
            return {'success': False, 'error': 'Règle non trouvée'}
        
        rule = self._rules[rule_id]
        
        # Suppression des index
        if rule.policy_type in self._rules_by_type:
            if rule_id in self._rules_by_type[rule.policy_type]:
                self._rules_by_type[rule.policy_type].remove(rule_id)
        
        for framework in rule.compliance_frameworks:
            if framework in self._rules_by_framework:
                if rule_id in self._rules_by_framework[framework]:
                    self._rules_by_framework[framework].remove(rule_id)
        
        # Suppression de la règle
        del self._rules[rule_id]
        
        # Mise à jour des métriques
        self._update_metrics()
        
        # Invalidation du cache
        self._evaluation_cache.clear()
        
        self.logger.info(f"Règle supprimée: {rule_id}")
        
        return {'success': True, 'rule_id': rule_id}
    
    async def export_rules(self, format: str = "json") -> Dict[str, Any]:
        """Export des règles de politique"""
        
        if format.lower() == "json":
            # Export JSON complet
            export_data = {
                'tenant_id': self.tenant_id,
                'export_timestamp': datetime.utcnow().isoformat(),
                'total_rules': len(self._rules),
                'rules': []
            }
            
            for rule in self._rules.values():
                rule_data = {
                    'rule_id': rule.rule_id,
                    'name': rule.name,
                    'description': rule.description,
                    'policy_type': rule.policy_type.value,
                    'priority': rule.priority.value,
                    'status': rule.status.value,
                    'action': rule.action.value,
                    'conditions': [
                        {
                            'field': c.field,
                            'operator': c.operator,
                            'value': c.value,
                            'description': c.description
                        }
                        for c in rule.conditions
                    ],
                    'compliance_frameworks': [f.value for f in rule.compliance_frameworks],
                    'effective_date': rule.effective_date.isoformat(),
                    'expiry_date': rule.expiry_date.isoformat() if rule.expiry_date else None,
                    'execution_count': rule.execution_count,
                    'confidence_score': rule.confidence_score,
                    'auto_generated': rule.auto_generated
                }
                export_data['rules'].append(rule_data)
            
            return {'success': True, 'data': export_data, 'format': 'json'}
        
        else:
            return {'success': False, 'error': f'Format non supporté: {format}'}
