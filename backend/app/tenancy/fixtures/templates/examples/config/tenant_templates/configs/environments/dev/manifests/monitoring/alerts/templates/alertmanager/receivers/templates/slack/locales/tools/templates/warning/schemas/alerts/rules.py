"""
Schémas de règles d'alertes - Spotify AI Agent
Définition et gestion des règles de déclenchement d'alertes
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Literal
from uuid import UUID, uuid4
from decimal import Decimal
from enum import Enum
import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field, validator, computed_field, ConfigDict

from . import (
    BaseSchema, TimestampMixin, TenantMixin, MetadataMixin,
    AlertLevel, AlertStatus, WarningCategory, NotificationChannel,
    Priority, Environment
)


class RuleType(str, Enum):
    """Types de règles d'alertes"""
    THRESHOLD = "threshold"          # Règle basée sur un seuil
    ANOMALY = "anomaly"             # Détection d'anomalie ML
    PATTERN = "pattern"             # Reconnaissance de motifs
    CORRELATION = "correlation"      # Corrélation entre métriques
    COMPOSITE = "composite"         # Règle composite
    PREDICTIVE = "predictive"       # Règle prédictive
    BASELINE = "baseline"           # Comparaison avec une baseline
    RATE_OF_CHANGE = "rate_of_change"  # Taux de changement


class ComparisonOperator(str, Enum):
    """Opérateurs de comparaison"""
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="
    NEQ = "!="
    REGEX = "regex"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"


class AggregationFunction(str, Enum):
    """Fonctions d'agrégation"""
    AVG = "avg"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    MEDIAN = "median"
    PERCENTILE = "percentile"
    STDDEV = "stddev"
    RATE = "rate"
    INCREASE = "increase"


class TimeUnit(str, Enum):
    """Unités de temps"""
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"


@dataclass
class RuleCondition:
    """Condition de déclenchement d'une règle"""
    metric_name: str
    operator: ComparisonOperator
    threshold_value: Union[float, int, str]
    aggregation: Optional[AggregationFunction] = None
    time_window: Optional[int] = None
    time_unit: TimeUnit = TimeUnit.MINUTES
    percentile: Optional[float] = None  # Pour PERCENTILE aggregation


class AlertRule(BaseSchema, TimestampMixin, TenantMixin, MetadataMixin):
    """Règle de déclenchement d'alerte avancée"""
    
    # Informations de base
    name: StrictStr = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    rule_type: RuleType = Field(default=RuleType.THRESHOLD)
    category: WarningCategory = Field(...)
    environment: Environment = Field(...)
    
    # Configuration de priorité et sévérité
    severity: AlertLevel = Field(...)
    priority: Priority = Field(...)
    
    # Conditions de déclenchement
    conditions: List[RuleCondition] = Field(..., min_items=1, max_items=10)
    logic_operator: Literal["AND", "OR"] = Field("AND")
    
    # Configuration temporelle
    evaluation_interval_seconds: int = Field(60, ge=1, le=3600)
    evaluation_window_minutes: int = Field(5, ge=1, le=1440)
    min_data_points: int = Field(1, ge=1, le=1000)
    
    # Gestion des états
    auto_resolve_timeout_minutes: Optional[int] = Field(None, ge=1, le=10080)
    resolve_condition: Optional[str] = Field(None)
    
    # Configuration d'escalade
    escalation_enabled: bool = Field(True)
    escalation_policy_id: Optional[UUID] = Field(None)
    escalation_delay_minutes: int = Field(15, ge=1, le=1440)
    max_escalation_level: int = Field(3, ge=1, le=10)
    
    # Suppression et dé-duplication
    suppression_enabled: bool = Field(False)
    suppression_duration_minutes: int = Field(60, ge=1, le=10080)
    suppression_conditions: List[str] = Field(default_factory=list)
    deduplication_enabled: bool = Field(True)
    deduplication_key_template: Optional[str] = Field(None)
    deduplication_window_minutes: int = Field(5, ge=1, le=60)
    
    # Configuration de notification
    notification_channels: List[NotificationChannel] = Field(default_factory=list)
    notification_template_id: Optional[UUID] = Field(None)
    notification_delay_seconds: int = Field(0, ge=0, le=3600)
    notification_rate_limit: int = Field(10, ge=1, le=1000)  # Max par heure
    
    # Maintenance et disponibilité
    maintenance_windows: List[Dict[str, Any]] = Field(default_factory=list)
    business_hours_only: bool = Field(False)
    timezone: str = Field("UTC")
    
    # Validité temporelle
    effective_from: Optional[datetime] = Field(None)
    effective_until: Optional[datetime] = Field(None)
    
    # Configuration ML (pour règles d'anomalie)
    ml_model_id: Optional[UUID] = Field(None)
    anomaly_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    sensitivity: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # État et contrôle
    enabled: bool = Field(True)
    paused: bool = Field(False)
    paused_until: Optional[datetime] = Field(None)
    paused_reason: Optional[str] = Field(None, max_length=500)
    
    # Métriques et performance
    fire_count: int = Field(0, ge=0)
    last_fired: Optional[datetime] = Field(None)
    last_evaluation: Optional[datetime] = Field(None)
    evaluation_duration_ms: Optional[float] = Field(None, ge=0)
    
    # Tags et labels
    tags: Set[str] = Field(default_factory=set)
    labels: Dict[str, str] = Field(default_factory=dict)
    
    # Audit et compliance
    created_by: Optional[UUID] = Field(None)
    last_modified_by: Optional[UUID] = Field(None)
    approval_required: bool = Field(False)
    approved: bool = Field(False)
    approved_by: Optional[UUID] = Field(None)
    approved_at: Optional[datetime] = Field(None)
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )

    @validator('conditions')
    def validate_conditions(cls, v):
        """Valide les conditions de la règle"""
        if not v:
            raise ValueError('At least one condition is required')
        
        # Vérifier la cohérence des conditions
        for condition in v:
            if condition.aggregation == AggregationFunction.PERCENTILE:
                if not condition.percentile or not (0 <= condition.percentile <= 100):
                    raise ValueError('Percentile must be between 0 and 100')
        
        return v

    @validator('timezone')
    def validate_timezone(cls, v):
        """Valide le fuseau horaire"""
        try:
            import pytz
            pytz.timezone(v)
        except Exception:
            raise ValueError(f'Invalid timezone: {v}')
        return v

    @validator('deduplication_key_template')
    def validate_deduplication_template(cls, v):
        """Valide le template de clé de déduplication"""
        if v:
            # Vérifier que le template contient des variables valides
            variables = re.findall(r'\{(\w+)\}', v)
            allowed_vars = {
                'tenant_id', 'rule_id', 'metric_name', 'instance',
                'severity', 'category', 'environment'
            }
            invalid_vars = set(variables) - allowed_vars
            if invalid_vars:
                raise ValueError(f'Invalid template variables: {invalid_vars}')
        return v

    @validator('effective_until')
    def validate_effective_dates(cls, v, values):
        """Valide les dates d'efficacité"""
        effective_from = values.get('effective_from')
        if effective_from and v and v <= effective_from:
            raise ValueError('effective_until must be after effective_from')
        return v

    @computed_field
    @property
    def is_active(self) -> bool:
        """Indique si la règle est actuellement active"""
        now = datetime.now(timezone.utc)
        
        # Vérifier si la règle est activée
        if not self.enabled or self.paused:
            return False
        
        # Vérifier la pause temporaire
        if self.paused_until and now < self.paused_until:
            return False
        
        # Vérifier la période d'efficacité
        if self.effective_from and now < self.effective_from:
            return False
        
        if self.effective_until and now > self.effective_until:
            return False
        
        return True

    @computed_field
    @property
    def next_evaluation(self) -> Optional[datetime]:
        """Prochaine évaluation programmée"""
        if not self.last_evaluation:
            return datetime.now(timezone.utc)
        
        return self.last_evaluation + timedelta(seconds=self.evaluation_interval_seconds)

    def evaluate_condition(self, condition: RuleCondition, metric_values: List[float]) -> bool:
        """Évalue une condition spécifique"""
        if not metric_values:
            return False
        
        # Appliquer l'agrégation
        if condition.aggregation == AggregationFunction.AVG:
            value = sum(metric_values) / len(metric_values)
        elif condition.aggregation == AggregationFunction.SUM:
            value = sum(metric_values)
        elif condition.aggregation == AggregationFunction.MIN:
            value = min(metric_values)
        elif condition.aggregation == AggregationFunction.MAX:
            value = max(metric_values)
        elif condition.aggregation == AggregationFunction.COUNT:
            value = len(metric_values)
        elif condition.aggregation == AggregationFunction.PERCENTILE:
            import numpy as np
            value = np.percentile(metric_values, condition.percentile or 95)
        else:
            value = metric_values[-1]  # Dernière valeur par défaut
        
        # Appliquer l'opérateur de comparaison
        threshold = float(condition.threshold_value)
        
        if condition.operator == ComparisonOperator.GT:
            return value > threshold
        elif condition.operator == ComparisonOperator.GTE:
            return value >= threshold
        elif condition.operator == ComparisonOperator.LT:
            return value < threshold
        elif condition.operator == ComparisonOperator.LTE:
            return value <= threshold
        elif condition.operator == ComparisonOperator.EQ:
            return abs(value - threshold) < 1e-9
        elif condition.operator == ComparisonOperator.NEQ:
            return abs(value - threshold) >= 1e-9
        
        return False

    def should_fire(self, metrics_data: Dict[str, List[float]]) -> bool:
        """Détermine si la règle doit se déclencher"""
        if not self.is_active:
            return False
        
        results = []
        for condition in self.conditions:
            metric_values = metrics_data.get(condition.metric_name, [])
            
            # Vérifier le nombre minimum de points de données
            if len(metric_values) < self.min_data_points:
                results.append(False)
                continue
            
            result = self.evaluate_condition(condition, metric_values)
            results.append(result)
        
        # Appliquer l'opérateur logique
        if self.logic_operator == "AND":
            return all(results)
        else:  # OR
            return any(results)

    def get_deduplication_key(self, context: Dict[str, Any]) -> str:
        """Génère la clé de déduplication"""
        if not self.deduplication_key_template:
            return f"{self.tenant_id}:{self.id}"
        
        # Remplacer les variables dans le template
        template_vars = {
            'tenant_id': str(self.tenant_id),
            'rule_id': str(self.id),
            'severity': self.severity.value,
            'category': self.category.value,
            'environment': self.environment.value,
            **context
        }
        
        try:
            return self.deduplication_key_template.format(**template_vars)
        except KeyError as e:
            # Fallback si une variable n'est pas disponible
            return f"{self.tenant_id}:{self.id}:{hash(str(context))}"


class RuleGroup(BaseSchema, TimestampMixin, TenantMixin, MetadataMixin):
    """Groupe de règles d'alertes"""
    
    name: StrictStr = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    
    # Règles du groupe
    rule_ids: List[UUID] = Field(default_factory=list)
    
    # Configuration du groupe
    evaluation_interval_seconds: int = Field(60, ge=1, le=3600)
    parallel_evaluation: bool = Field(True)
    max_concurrent_evaluations: int = Field(10, ge=1, le=100)
    
    # État du groupe
    enabled: bool = Field(True)
    last_evaluation: Optional[datetime] = Field(None)
    evaluation_count: int = Field(0, ge=0)
    error_count: int = Field(0, ge=0)
    
    # Tags et organisation
    tags: Set[str] = Field(default_factory=set)
    environment: Environment = Field(...)
    
    @computed_field
    @property
    def next_evaluation(self) -> Optional[datetime]:
        """Prochaine évaluation du groupe"""
        if not self.last_evaluation:
            return datetime.now(timezone.utc)
        
        return self.last_evaluation + timedelta(seconds=self.evaluation_interval_seconds)


class RuleTemplate(BaseSchema, TimestampMixin, TenantMixin):
    """Template de règle d'alerte réutilisable"""
    
    name: StrictStr = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    category: WarningCategory = Field(...)
    
    # Template de la règle
    rule_template: Dict[str, Any] = Field(...)
    template_variables: Dict[str, Any] = Field(default_factory=dict)
    
    # Métadonnées du template
    version: str = Field("1.0.0")
    author: Optional[UUID] = Field(None)
    tags: Set[str] = Field(default_factory=set)
    
    # Usage et adoption
    usage_count: int = Field(0, ge=0)
    last_used: Optional[datetime] = Field(None)
    
    def instantiate(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Instancie une règle à partir du template"""
        import copy
        
        rule_data = copy.deepcopy(self.rule_template)
        
        # Remplacer les variables dans le template
        def replace_variables(obj):
            if isinstance(obj, dict):
                return {k: replace_variables(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_variables(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                var_name = obj[2:-1]
                return variables.get(var_name, self.template_variables.get(var_name, obj))
            else:
                return obj
        
        return replace_variables(rule_data)


__all__ = [
    'RuleType', 'ComparisonOperator', 'AggregationFunction', 'TimeUnit',
    'RuleCondition', 'AlertRule', 'RuleGroup', 'RuleTemplate'
]
