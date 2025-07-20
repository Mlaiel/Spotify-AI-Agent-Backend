"""
Advanced Metrics Schemas - Ultra-Advanced Edition
===============================================

Schémas ultra-avancés pour les métriques avec calculs statistiques,
agrégations temporelles et analyses prédictives.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Literal, Set, Tuple
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import UUID4, PositiveInt, NonNegativeFloat, PositiveFloat


class MetricAggregationType(str, Enum):
    """Types d'agrégation des métriques."""
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    MEDIAN = "median"
    PERCENTILE = "percentile"
    RATE = "rate"
    HISTOGRAM = "histogram"


class MetricValue(BaseModel):
    """Valeur de métrique avec métadonnées complètes."""
    
    metric_id: UUID4 = Field(default_factory=lambda: UUID4())
    name: str = Field(..., description="Nom de la métrique")
    value: float = Field(..., description="Valeur")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Métadonnées
    unit: str = Field(..., description="Unité de mesure")
    tags: Dict[str, str] = Field(default_factory=dict)
    dimensions: Dict[str, str] = Field(default_factory=dict)
    
    # Qualité
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Niveau de confiance")
    is_estimated: bool = Field(default=False, description="Valeur estimée")


class AggregatedMetric(BaseModel):
    """Métrique agrégée avec statistiques avancées."""
    
    metric_name: str = Field(..., description="Nom de la métrique")
    aggregation_type: MetricAggregationType = Field(..., description="Type d'agrégation")
    
    # Période
    period_start: datetime = Field(..., description="Début période")
    period_end: datetime = Field(..., description="Fin période")
    
    # Valeurs agrégées
    value: float = Field(..., description="Valeur agrégée")
    sample_count: int = Field(..., ge=1, description="Nombre d'échantillons")
    
    # Statistiques
    min_value: Optional[float] = Field(None, description="Valeur minimale")
    max_value: Optional[float] = Field(None, description="Valeur maximale")
    std_deviation: Optional[float] = Field(None, description="Écart-type")
    variance: Optional[float] = Field(None, description="Variance")


# Export
__all__ = ["MetricAggregationType", "MetricValue", "AggregatedMetric"]
