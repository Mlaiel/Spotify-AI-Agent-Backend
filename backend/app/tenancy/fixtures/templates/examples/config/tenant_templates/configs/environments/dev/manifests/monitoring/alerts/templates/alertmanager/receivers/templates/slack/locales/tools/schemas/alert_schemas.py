"""
Schémas Pydantic avancés pour la configuration des alertes.

Ce module définit tous les schémas de validation pour les règles d'alertes,
configurations AlertManager, routage, seuils et escalation.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator, root_validator
from decimal import Decimal


class AlertSeverity(str, Enum):
    """Niveaux de sévérité des alertes."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(str, Enum):
    """États possibles des alertes."""
    FIRING = "firing"
    PENDING = "pending"
    RESOLVED = "resolved"
    SILENCED = "silenced"
    INHIBITED = "inhibited"


class ComparisonOperator(str, Enum):
    """Opérateurs de comparaison pour les seuils."""
    GREATER_THAN = "gt"
    GREATER_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_EQUAL = "lte"
    EQUAL = "eq"
    NOT_EQUAL = "ne"


class AggregationFunction(str, Enum):
    """Fonctions d'agrégation pour les métriques."""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    STDDEV = "stddev"
    PERCENTILE = "percentile"


class AlertTimeWindowSchema(BaseModel):
    """Schéma pour les fenêtres temporelles d'évaluation."""
    duration: str = Field(..., description="Durée de la fenêtre (ex: 5m, 1h)")
    step: Optional[str] = Field(None, description="Pas d'évaluation")
    offset: Optional[str] = Field(None, description="Décalage temporel")
    
    @validator('duration')
    def validate_duration(cls, v):
        """Valide le format de durée."""
        import re
        if not re.match(r'^\d+[smhdw]$', v):
            raise ValueError('Format de durée invalide. Utilisez: 30s, 5m, 1h, 1d, 1w')
        return v


class AlertThresholdSchema(BaseModel):
    """Schéma pour les seuils d'alerte."""
    value: Union[float, int, Decimal] = Field(..., description="Valeur seuil")
    operator: ComparisonOperator = Field(..., description="Opérateur de comparaison")
    unit: Optional[str] = Field(None, description="Unité de mesure")
    percentage: bool = Field(False, description="Seuil en pourcentage")
    
    class Config:
        use_enum_values = True


class AlertConditionSchema(BaseModel):
    """Schéma pour les conditions d'alerte."""
    metric_name: str = Field(..., description="Nom de la métrique")
    threshold: AlertThresholdSchema = Field(..., description="Seuil d'alerte")
    time_window: AlertTimeWindowSchema = Field(..., description="Fenêtre temporelle")
    aggregation: AggregationFunction = Field(AggregationFunction.AVG, description="Fonction d'agrégation")
    filters: Dict[str, str] = Field(default_factory=dict, description="Filtres sur les labels")
    
    class Config:
        use_enum_values = True


class AlertMetricSchema(BaseModel):
    """Schéma pour les métriques d'alerte."""
    name: str = Field(..., description="Nom de la métrique")
    query: str = Field(..., description="Requête PromQL")
    datasource: str = Field("prometheus", description="Source de données")
    interval: str = Field("30s", description="Intervalle d'évaluation")
    labels: Dict[str, str] = Field(default_factory=dict, description="Labels de la métrique")
    annotations: Dict[str, str] = Field(default_factory=dict, description="Annotations")


class AlertEscalationSchema(BaseModel):
    """Schéma pour l'escalation des alertes."""
    level: int = Field(..., ge=1, le=5, description="Niveau d'escalation")
    delay: str = Field(..., description="Délai avant escalation")
    channels: List[str] = Field(..., description="Canaux de notification")
    conditions: List[str] = Field(default_factory=list, description="Conditions d'escalation")
    auto_resolve: bool = Field(True, description="Résolution automatique")


class AlertChannelSchema(BaseModel):
    """Schéma pour les canaux de notification."""
    name: str = Field(..., description="Nom du canal")
    type: str = Field(..., description="Type de canal (slack, email, webhook)")
    config: Dict[str, Any] = Field(..., description="Configuration du canal")
    enabled: bool = Field(True, description="Canal activé")
    filters: Dict[str, str] = Field(default_factory=dict, description="Filtres de notification")


class AlertRuleSchema(BaseModel):
    """Schéma principal pour les règles d'alerte."""
    name: str = Field(..., description="Nom unique de la règle")
    description: Optional[str] = Field(None, description="Description de la règle")
    severity: AlertSeverity = Field(AlertSeverity.MEDIUM, description="Sévérité de l'alerte")
    conditions: List[AlertConditionSchema] = Field(..., description="Conditions d'alerte")
    metrics: List[AlertMetricSchema] = Field(..., description="Métriques surveillées")
    escalation: Optional[AlertEscalationSchema] = Field(None, description="Règles d'escalation")
    channels: List[AlertChannelSchema] = Field(..., description="Canaux de notification")
    enabled: bool = Field(True, description="Règle activée")
    tenant_id: Optional[str] = Field(None, description="ID du tenant")
    environment: str = Field("dev", description="Environnement cible")
    tags: List[str] = Field(default_factory=list, description="Tags de classification")
    created_at: Optional[datetime] = Field(None, description="Date de création")
    updated_at: Optional[datetime] = Field(None, description="Date de mise à jour")
    
    class Config:
        use_enum_values = True
        allow_population_by_field_name = True


class AlertGroupSchema(BaseModel):
    """Schéma pour les groupes d'alertes."""
    name: str = Field(..., description="Nom du groupe")
    rules: List[AlertRuleSchema] = Field(..., description="Règles du groupe")
    interval: str = Field("30s", description="Intervalle d'évaluation du groupe")
    partial_response_strategy: str = Field("warn", description="Stratégie de réponse partielle")


class AlertReceiverSchema(BaseModel):
    """Schéma pour les destinataires d'alertes."""
    name: str = Field(..., description="Nom du destinataire")
    slack_configs: List[Dict[str, Any]] = Field(default_factory=list, description="Configurations Slack")
    email_configs: List[Dict[str, Any]] = Field(default_factory=list, description="Configurations email")
    webhook_configs: List[Dict[str, Any]] = Field(default_factory=list, description="Configurations webhook")
    pagerduty_configs: List[Dict[str, Any]] = Field(default_factory=list, description="Configurations PagerDuty")


class AlertRoutingSchema(BaseModel):
    """Schéma pour le routage des alertes."""
    receiver: str = Field(..., description="Destinataire par défaut")
    group_by: List[str] = Field(default_factory=list, description="Groupage des alertes")
    group_wait: str = Field("10s", description="Attente de groupage")
    group_interval: str = Field("10s", description="Intervalle de groupage")
    repeat_interval: str = Field("1h", description="Intervalle de répétition")
    routes: List["AlertRoutingSchema"] = Field(default_factory=list, description="Routes filles")
    matchers: List[str] = Field(default_factory=list, description="Matchers pour le routage")
    continue_routing: bool = Field(False, description="Continuer le routage")


class AlertInhibitRuleSchema(BaseModel):
    """Schéma pour les règles d'inhibition."""
    source_matchers: List[str] = Field(..., description="Matchers source")
    target_matchers: List[str] = Field(..., description="Matchers cible")
    equal: List[str] = Field(default_factory=list, description="Labels égaux requis")


class AlertSilenceSchema(BaseModel):
    """Schéma pour les silences d'alertes."""
    id: Optional[str] = Field(None, description="ID du silence")
    matchers: List[str] = Field(..., description="Matchers du silence")
    starts_at: datetime = Field(..., description="Début du silence")
    ends_at: datetime = Field(..., description="Fin du silence")
    created_by: str = Field(..., description="Créateur du silence")
    comment: str = Field(..., description="Commentaire du silence")


class AlertTemplateSchema(BaseModel):
    """Schéma pour les templates d'alertes."""
    name: str = Field(..., description="Nom du template")
    title: str = Field(..., description="Titre de l'alerte")
    text: str = Field(..., description="Corps du message")
    title_link: Optional[str] = Field(None, description="Lien du titre")
    color: Optional[str] = Field(None, description="Couleur du message")
    fields: List[Dict[str, str]] = Field(default_factory=list, description="Champs du message")
    actions: List[Dict[str, Any]] = Field(default_factory=list, description="Actions disponibles")


class AlertManagerConfigSchema(BaseModel):
    """Schéma principal pour la configuration AlertManager."""
    global_config: Dict[str, Any] = Field(default_factory=dict, description="Configuration globale")
    route: AlertRoutingSchema = Field(..., description="Routage principal")
    receivers: List[AlertReceiverSchema] = Field(..., description="Destinataires")
    inhibit_rules: List[AlertInhibitRuleSchema] = Field(default_factory=list, description="Règles d'inhibition")
    templates: List[str] = Field(default_factory=list, description="Templates disponibles")
    
    @root_validator
    def validate_config(cls, values):
        """Validation globale de la configuration."""
        route = values.get('route')
        receivers = values.get('receivers', [])
        
        if route and receivers:
            receiver_names = {r.name for r in receivers}
            if route.receiver not in receiver_names:
                raise ValueError(f"Receiver '{route.receiver}' non trouvé dans la liste des receivers")
        
        return values


# Mise à jour des références forward
AlertRoutingSchema.model_rebuild()
