"""
Schémas de corrélation d'alertes - Spotify AI Agent
Analyse intelligente des corrélations et regroupements d'alertes
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Tuple
from uuid import UUID, uuid4
from enum import Enum
import json
import math
from statistics import mean, median, stdev

from pydantic import BaseModel, Field, validator, computed_field, ConfigDict

from . import (
    BaseSchema, TimestampMixin, TenantMixin, MetadataMixin,
    AlertLevel, AlertStatus, WarningCategory, Priority, Environment
)


class CorrelationType(str, Enum):
    """Types de corrélation"""
    TEMPORAL = "temporal"           # Corrélation temporelle
    SPATIAL = "spatial"             # Corrélation spatiale (infrastructure)
    CAUSAL = "causal"              # Corrélation causale
    PATTERN = "pattern"            # Corrélation de motif
    SIMILARITY = "similarity"       # Similarité de contenu
    DEPENDENCY = "dependency"       # Dépendance de service
    THRESHOLD = "threshold"         # Seuil de corrélation
    STATISTICAL = "statistical"    # Corrélation statistique


class CorrelationMethod(str, Enum):
    """Méthodes de corrélation"""
    PEARSON = "pearson"            # Corrélation de Pearson
    SPEARMAN = "spearman"          # Corrélation de Spearman
    KENDALL = "kendall"            # Tau de Kendall
    COSINE = "cosine"              # Similarité cosinus
    JACCARD = "jaccard"            # Index de Jaccard
    EUCLIDEAN = "euclidean"        # Distance euclidienne
    MANHATTAN = "manhattan"        # Distance de Manhattan
    HAMMING = "hamming"            # Distance de Hamming
    EDIT_DISTANCE = "edit_distance" # Distance d'édition


class CorrelationStrength(str, Enum):
    """Force de corrélation"""
    VERY_WEAK = "very_weak"        # 0.0 - 0.2
    WEAK = "weak"                  # 0.2 - 0.4
    MODERATE = "moderate"          # 0.4 - 0.6
    STRONG = "strong"              # 0.6 - 0.8
    VERY_STRONG = "very_strong"    # 0.8 - 1.0


class ClusteringStatus(str, Enum):
    """États de clustering"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class CorrelationRule(BaseSchema, TimestampMixin, TenantMixin):
    """Règle de corrélation d'alertes"""
    
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    
    # Configuration de la règle
    correlation_type: CorrelationType = Field(...)
    method: CorrelationMethod = Field(...)
    threshold: float = Field(0.7, ge=0.0, le=1.0)
    
    # Fenêtre de corrélation
    time_window_minutes: int = Field(15, ge=1, le=1440)
    max_alerts_per_window: int = Field(1000, ge=1, le=10000)
    
    # Filtres d'inclusion
    alert_level_filter: List[AlertLevel] = Field(default_factory=list)
    category_filter: List[WarningCategory] = Field(default_factory=list)
    environment_filter: List[Environment] = Field(default_factory=list)
    
    # Attributs de corrélation
    correlation_attributes: List[str] = Field(default_factory=list)
    required_attributes: Set[str] = Field(default_factory=set)
    
    # Configuration avancée
    min_correlation_size: int = Field(2, ge=2, le=100)
    max_correlation_size: int = Field(50, ge=2, le=1000)
    auto_merge_similar: bool = Field(True)
    merge_threshold: float = Field(0.85, ge=0.0, le=1.0)
    
    # État et performance
    enabled: bool = Field(True)
    total_correlations: int = Field(0, ge=0)
    successful_correlations: int = Field(0, ge=0)
    last_execution: Optional[datetime] = Field(None)
    avg_execution_time_ms: Optional[float] = Field(None, ge=0)
    
    # Tags et métadonnées
    tags: Set[str] = Field(default_factory=set)
    
    @validator('correlation_attributes')
    def validate_correlation_attributes(cls, v):
        """Valide les attributs de corrélation"""
        allowed_attributes = {
            'message', 'source', 'instance', 'service', 'component',
            'severity', 'category', 'tags', 'labels', 'metrics',
            'fingerprint', 'description'
        }
        
        invalid_attrs = set(v) - allowed_attributes
        if invalid_attrs:
            raise ValueError(f'Invalid correlation attributes: {invalid_attrs}')
        
        return v

    def calculate_correlation_score(self, alert1: Dict[str, Any], 
                                  alert2: Dict[str, Any]) -> float:
        """Calcule le score de corrélation entre deux alertes"""
        if self.method == CorrelationMethod.JACCARD:
            return self._jaccard_similarity(alert1, alert2)
        elif self.method == CorrelationMethod.COSINE:
            return self._cosine_similarity(alert1, alert2)
        elif self.method == CorrelationMethod.EUCLIDEAN:
            return self._euclidean_distance(alert1, alert2)
        elif self.method == CorrelationMethod.EDIT_DISTANCE:
            return self._edit_distance_similarity(alert1, alert2)
        else:
            return self._default_similarity(alert1, alert2)

    def _jaccard_similarity(self, alert1: Dict[str, Any], alert2: Dict[str, Any]) -> float:
        """Calcule la similarité de Jaccard"""
        set1 = set()
        set2 = set()
        
        for attr in self.correlation_attributes:
            val1 = alert1.get(attr)
            val2 = alert2.get(attr)
            
            if val1:
                if isinstance(val1, (list, set)):
                    set1.update(str(v) for v in val1)
                else:
                    set1.add(str(val1))
            
            if val2:
                if isinstance(val2, (list, set)):
                    set2.update(str(v) for v in val2)
                else:
                    set2.add(str(val2))
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0

    def _cosine_similarity(self, alert1: Dict[str, Any], alert2: Dict[str, Any]) -> float:
        """Calcule la similarité cosinus"""
        # Implémentation simplifiée pour les attributs textuels
        import re
        
        text1 = " ".join(str(alert1.get(attr, "")) for attr in self.correlation_attributes)
        text2 = " ".join(str(alert2.get(attr, "")) for attr in self.correlation_attributes)
        
        # Tokenisation simple
        tokens1 = set(re.findall(r'\w+', text1.lower()))
        tokens2 = set(re.findall(r'\w+', text2.lower()))
        
        intersection = len(tokens1 & tokens2)
        magnitude1 = math.sqrt(len(tokens1))
        magnitude2 = math.sqrt(len(tokens2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return intersection / (magnitude1 * magnitude2)

    def _euclidean_distance(self, alert1: Dict[str, Any], alert2: Dict[str, Any]) -> float:
        """Calcule la distance euclidienne normalisée"""
        # Conversion en score de similarité (1 - distance normalisée)
        differences = []
        
        for attr in self.correlation_attributes:
            val1 = alert1.get(attr)
            val2 = alert2.get(attr)
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Différence numérique normalisée
                max_val = max(abs(val1), abs(val2), 1)
                diff = abs(val1 - val2) / max_val
                differences.append(diff)
            elif val1 == val2:
                differences.append(0.0)
            else:
                differences.append(1.0)
        
        if not differences:
            return 0.0
        
        distance = math.sqrt(sum(d**2 for d in differences)) / len(differences)
        return max(0.0, 1.0 - distance)

    def _edit_distance_similarity(self, alert1: Dict[str, Any], alert2: Dict[str, Any]) -> float:
        """Calcule la similarité basée sur la distance d'édition"""
        def edit_distance(s1: str, s2: str) -> int:
            """Distance d'édition de Levenshtein"""
            if len(s1) < len(s2):
                return edit_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        similarities = []
        for attr in self.correlation_attributes:
            str1 = str(alert1.get(attr, ""))
            str2 = str(alert2.get(attr, ""))
            
            max_len = max(len(str1), len(str2), 1)
            distance = edit_distance(str1, str2)
            similarity = 1.0 - (distance / max_len)
            similarities.append(similarity)
        
        return mean(similarities) if similarities else 0.0

    def _default_similarity(self, alert1: Dict[str, Any], alert2: Dict[str, Any]) -> float:
        """Calcule une similarité par défaut"""
        matches = 0
        total = 0
        
        for attr in self.correlation_attributes:
            val1 = alert1.get(attr)
            val2 = alert2.get(attr)
            
            total += 1
            if val1 == val2:
                matches += 1
        
        return matches / total if total > 0 else 0.0


class AlertCorrelation(BaseSchema, TimestampMixin, TenantMixin, MetadataMixin):
    """Corrélation d'alertes détectée"""
    
    correlation_id: UUID = Field(default_factory=uuid4)
    rule_id: UUID = Field(...)
    
    # Alertes corrélées
    alert_ids: List[UUID] = Field(..., min_items=2)
    primary_alert_id: UUID = Field(...)  # Alerte principale
    
    # Métadonnées de corrélation
    correlation_type: CorrelationType = Field(...)
    correlation_score: float = Field(..., ge=0.0, le=1.0)
    correlation_strength: CorrelationStrength = Field(...)
    
    # Fenêtre temporelle
    start_time: datetime = Field(...)
    end_time: datetime = Field(...)
    duration_minutes: float = Field(..., ge=0)
    
    # Attributs communs
    common_attributes: Dict[str, Any] = Field(default_factory=dict)
    correlation_summary: str = Field(..., min_length=1)
    
    # Classification
    root_cause_alert_id: Optional[UUID] = Field(None)
    category: WarningCategory = Field(...)
    severity: AlertLevel = Field(...)
    
    # État et actions
    status: AlertStatus = Field(AlertStatus.ACTIVE)
    acknowledged: bool = Field(False)
    acknowledged_by: Optional[UUID] = Field(None)
    acknowledged_at: Optional[datetime] = Field(None)
    
    # Résolution
    resolved: bool = Field(False)
    resolved_by: Optional[UUID] = Field(None)
    resolved_at: Optional[datetime] = Field(None)
    resolution_notes: Optional[str] = Field(None, max_length=2000)
    
    # Métriques et insights
    alert_frequency: float = Field(..., ge=0)  # Alertes par minute
    geographical_spread: Optional[Dict[str, int]] = Field(None)
    service_impact: Optional[Dict[str, Any]] = Field(None)
    
    # Tags et labels
    tags: Set[str] = Field(default_factory=set)
    labels: Dict[str, str] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )

    @computed_field
    @property
    def alert_count(self) -> int:
        """Nombre d'alertes dans la corrélation"""
        return len(self.alert_ids)

    @computed_field
    @property
    def correlation_age_minutes(self) -> float:
        """Âge de la corrélation en minutes"""
        now = datetime.now(timezone.utc)
        age = now - self.created_at
        return age.total_seconds() / 60

    @validator('correlation_strength', pre=True, always=True)
    def determine_correlation_strength(cls, v, values):
        """Détermine automatiquement la force de corrélation"""
        score = values.get('correlation_score', 0.0)
        
        if score >= 0.8:
            return CorrelationStrength.VERY_STRONG
        elif score >= 0.6:
            return CorrelationStrength.STRONG
        elif score >= 0.4:
            return CorrelationStrength.MODERATE
        elif score >= 0.2:
            return CorrelationStrength.WEAK
        else:
            return CorrelationStrength.VERY_WEAK

    def add_alert(self, alert_id: UUID, recalculate_score: bool = True):
        """Ajoute une alerte à la corrélation"""
        if alert_id not in self.alert_ids:
            self.alert_ids.append(alert_id)
            self.updated_at = datetime.now(timezone.utc)
            
            if recalculate_score:
                # Logique de recalcul du score si nécessaire
                pass

    def remove_alert(self, alert_id: UUID):
        """Retire une alerte de la corrélation"""
        if alert_id in self.alert_ids:
            self.alert_ids.remove(alert_id)
            self.updated_at = datetime.now(timezone.utc)
            
            # Si l'alerte principale est retirée, choisir une nouvelle
            if alert_id == self.primary_alert_id and self.alert_ids:
                self.primary_alert_id = self.alert_ids[0]


class AlertCluster(BaseSchema, TimestampMixin, TenantMixin, MetadataMixin):
    """Cluster d'alertes similaires"""
    
    cluster_id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    
    # Configuration du cluster
    clustering_method: str = Field("hierarchical")
    similarity_threshold: float = Field(0.8, ge=0.0, le=1.0)
    
    # Corrélations incluses
    correlation_ids: List[UUID] = Field(default_factory=list)
    total_alerts: int = Field(0, ge=0)
    
    # Centroïde du cluster
    centroid_attributes: Dict[str, Any] = Field(default_factory=dict)
    representative_alert_id: Optional[UUID] = Field(None)
    
    # Métadonnées du cluster
    category: WarningCategory = Field(...)
    dominant_severity: AlertLevel = Field(...)
    environments: Set[Environment] = Field(default_factory=set)
    
    # Statistiques temporelles
    first_alert_time: Optional[datetime] = Field(None)
    last_alert_time: Optional[datetime] = Field(None)
    peak_alert_rate: Optional[float] = Field(None)  # Alertes par minute
    
    # État et gestion
    status: ClusteringStatus = Field(ClusteringStatus.PENDING)
    auto_created: bool = Field(True)
    
    # Insights et analytics
    pattern_signature: Optional[str] = Field(None)
    recurrence_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    impact_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    
    # Tags et classification
    tags: Set[str] = Field(default_factory=set)
    labels: Dict[str, str] = Field(default_factory=dict)
    
    @computed_field
    @property
    def cluster_lifespan_hours(self) -> Optional[float]:
        """Durée de vie du cluster en heures"""
        if not self.first_alert_time or not self.last_alert_time:
            return None
        
        duration = self.last_alert_time - self.first_alert_time
        return duration.total_seconds() / 3600

    @computed_field
    @property
    def average_alert_rate(self) -> Optional[float]:
        """Taux moyen d'alertes par heure"""
        lifespan = self.cluster_lifespan_hours
        if lifespan and lifespan > 0:
            return self.total_alerts / lifespan
        return None

    def merge_with(self, other_cluster: 'AlertCluster') -> 'AlertCluster':
        """Fusionne avec un autre cluster"""
        # Créer un nouveau cluster fusionné
        merged = AlertCluster(
            tenant_id=self.tenant_id,
            name=f"Merged: {self.name} + {other_cluster.name}",
            clustering_method=self.clustering_method,
            similarity_threshold=min(self.similarity_threshold, other_cluster.similarity_threshold),
            correlation_ids=self.correlation_ids + other_cluster.correlation_ids,
            total_alerts=self.total_alerts + other_cluster.total_alerts,
            category=self.category,  # Garder la catégorie du cluster principal
            dominant_severity=max(self.dominant_severity, other_cluster.dominant_severity),
            environments=self.environments | other_cluster.environments,
            first_alert_time=min(
                self.first_alert_time or datetime.max.replace(tzinfo=timezone.utc),
                other_cluster.first_alert_time or datetime.max.replace(tzinfo=timezone.utc)
            ),
            last_alert_time=max(
                self.last_alert_time or datetime.min.replace(tzinfo=timezone.utc),
                other_cluster.last_alert_time or datetime.min.replace(tzinfo=timezone.utc)
            ),
            tags=self.tags | other_cluster.tags,
            auto_created=True
        )
        
        return merged


class CorrelationEngine(BaseSchema, TimestampMixin, TenantMixin):
    """Moteur de corrélation d'alertes"""
    
    engine_id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=255)
    version: str = Field("1.0.0")
    
    # Configuration
    enabled: bool = Field(True)
    processing_interval_seconds: int = Field(60, ge=1, le=3600)
    max_alerts_per_batch: int = Field(1000, ge=1, le=10000)
    
    # Règles actives
    active_rules: List[UUID] = Field(default_factory=list)
    
    # Performance et métriques
    total_processed: int = Field(0, ge=0)
    total_correlations_found: int = Field(0, ge=0)
    total_clusters_created: int = Field(0, ge=0)
    
    # État de santé
    last_execution: Optional[datetime] = Field(None)
    avg_processing_time_ms: Optional[float] = Field(None, ge=0)
    error_count: int = Field(0, ge=0)
    last_error: Optional[str] = Field(None)
    
    # Configuration avancée
    machine_learning_enabled: bool = Field(False)
    ml_model_version: Optional[str] = Field(None)
    real_time_processing: bool = Field(True)
    
    @computed_field
    @property
    def correlation_rate(self) -> float:
        """Taux de corrélation (pourcentage d'alertes corrélées)"""
        if self.total_processed == 0:
            return 0.0
        return (self.total_correlations_found / self.total_processed) * 100

    @computed_field
    @property
    def health_status(self) -> str:
        """État de santé du moteur"""
        if not self.enabled:
            return "disabled"
        
        if self.error_count > 10:
            return "unhealthy"
        
        if self.last_execution:
            time_since_last = datetime.now(timezone.utc) - self.last_execution
            if time_since_last > timedelta(minutes=self.processing_interval_seconds / 60 * 2):
                return "stale"
        
        return "healthy"


__all__ = [
    'CorrelationType', 'CorrelationMethod', 'CorrelationStrength', 'ClusteringStatus',
    'CorrelationRule', 'AlertCorrelation', 'AlertCluster', 'CorrelationEngine'
]
