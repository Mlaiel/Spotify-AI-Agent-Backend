"""
Event Management Schemas - Ultra-Advanced Edition
===============================================

Schémas ultra-avancés pour la gestion d'événements business, système et utilisateur
avec event sourcing, CQRS et orchestration complexe.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Literal, Set, Tuple
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import UUID4, PositiveInt, NonNegativeFloat, PositiveFloat


class EventCategory(str, Enum):
    """Catégories d'événements."""
    USER_INTERACTION = "user_interaction"
    BUSINESS_PROCESS = "business_process"
    SYSTEM_OPERATION = "system_operation"
    DATA_CHANGE = "data_change"
    SECURITY_EVENT = "security_event"
    INTEGRATION_EVENT = "integration_event"
    WORKFLOW_EVENT = "workflow_event"
    NOTIFICATION_EVENT = "notification_event"


class EventSeverity(str, Enum):
    """Niveaux de sévérité."""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class EventState(str, Enum):
    """États des événements."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


class BusinessEvent(BaseModel):
    """Événement business avec contexte métier complet."""
    
    event_id: UUID4 = Field(default_factory=lambda: UUID4())
    tenant_id: UUID4 = Field(..., description="ID du tenant")
    
    # Classification
    category: EventCategory = Field(..., description="Catégorie d'événement")
    event_type: str = Field(..., description="Type spécifique")
    severity: EventSeverity = Field(default=EventSeverity.INFO)
    
    # Contexte temporel
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    business_date: Optional[datetime] = Field(None, description="Date business")
    
    # Contexte business
    business_unit: Optional[str] = Field(None, description="Unité business")
    process_name: str = Field(..., description="Nom du processus")
    workflow_id: Optional[UUID4] = Field(None, description="ID du workflow")
    
    # Acteurs
    initiated_by: Optional[UUID4] = Field(None, description="Initié par utilisateur")
    affected_entities: List[Dict[str, Any]] = Field(default_factory=list, description="Entités affectées")
    
    # Contenu
    event_data: Dict[str, Any] = Field(..., description="Données de l'événement")
    before_state: Optional[Dict[str, Any]] = Field(None, description="État avant")
    after_state: Optional[Dict[str, Any]] = Field(None, description="État après")
    
    # Impact
    impact_level: str = Field(default="low", description="Niveau d'impact")
    revenue_impact: Optional[Decimal] = Field(None, description="Impact revenus")
    user_impact_count: Optional[int] = Field(None, description="Utilisateurs impactés")


# Export
__all__ = ["EventCategory", "EventSeverity", "EventState", "BusinessEvent"]
