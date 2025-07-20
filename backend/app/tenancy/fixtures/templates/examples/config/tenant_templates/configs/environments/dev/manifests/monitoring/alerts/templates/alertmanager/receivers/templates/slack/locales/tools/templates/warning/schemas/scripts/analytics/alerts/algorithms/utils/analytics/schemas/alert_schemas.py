"""
Alert Management Schemas - Ultra-Advanced Edition
===============================================

Schémas ultra-avancés pour la gestion d'alertes avec ML, workflows automatisés
et escalade intelligente.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Literal, Set, Tuple
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import UUID4, PositiveInt, NonNegativeFloat, PositiveFloat


class AlertType(str, Enum):
    """Types d'alertes."""
    THRESHOLD = "threshold"
    ANOMALY = "anomaly"
    TREND = "trend"
    COMPOSITE = "composite"
    PREDICTIVE = "predictive"
    CORRELATION = "correlation"


class AlertChannel(str, Enum):
    """Canaux de notification."""
    EMAIL = "email"
    SLACK = "slack"
    SMS = "sms"
    WEBHOOK = "webhook"
    PHONE = "phone"
    MOBILE_PUSH = "mobile_push"


class SmartAlert(BaseModel):
    """Alerte intelligente avec ML et workflows."""
    
    alert_id: UUID4 = Field(default_factory=lambda: UUID4())
    title: str = Field(..., description="Titre de l'alerte")
    description: str = Field(..., description="Description")
    
    # Classification
    alert_type: AlertType = Field(..., description="Type d'alerte")
    severity: str = Field(..., description="Sévérité")
    
    # ML et prédiction
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Score de confiance")
    predicted_impact: Optional[str] = Field(None, description="Impact prédit")
    similar_alerts: List[UUID4] = Field(default_factory=list, description="Alertes similaires")
    
    # Workflow
    auto_actions: List[Dict[str, Any]] = Field(default_factory=list, description="Actions automatiques")
    escalation_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Règles d'escalade")
    
    # Notification
    notification_channels: List[AlertChannel] = Field(..., description="Canaux de notification")
    notification_sent: bool = Field(default=False, description="Notification envoyée")


# Export
__all__ = ["AlertType", "AlertChannel", "SmartAlert"]
