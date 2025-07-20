"""
Mixins réutilisables - Spotify AI Agent
Composants modulaires pour l'extension des modèles Pydantic
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Callable, Type
from uuid import UUID, uuid4
from decimal import Decimal
import json
import hashlib
import threading
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, validator, root_validator, computed_field, ConfigDict
from pydantic.types import StrictStr, PositiveInt, NonNegativeInt

from .enums import AlertLevel, Priority, Environment, SecurityLevel, PermissionFlag


class VersionedMixin(BaseModel):
    """Mixin pour le versioning et la gestion des conflits"""
    version: PositiveInt = Field(1, description="Version de l'entité")
    schema_version: str = Field("1.0.0", description="Version du schéma")
    etag: Optional[str] = Field(None, description="ETag pour la concurrence optimiste")
    lock_token: Optional[str] = Field(None, description="Token de verrouillage")
    locked_by: Optional[UUID] = Field(None, description="Utilisateur ayant verrouillé")
    locked_at: Optional[datetime] = Field(None, description="Date de verrouillage")
    lock_timeout_seconds: int = Field(300, description="Timeout du verrouillage en secondes")
    
    def increment_version(self, user_id: Optional[UUID] = None) -> int:
        """Incrémente la version et met à jour l'ETag"""
        self.version += 1
        self.etag = self._generate_etag()
        if hasattr(self, 'updated_by'):
            self.updated_by = user_id
        if hasattr(self, 'mark_updated'):
            self.mark_updated()
        return self.version
    
    def _generate_etag(self) -> str:
        """Génère un ETag basé sur le contenu"""
        content = self.dict(exclude={'etag', 'lock_token', 'locked_by', 'locked_at'})
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def acquire_lock(self, user_id: UUID, timeout_seconds: Optional[int] = None) -> bool:
        """Acquiert un verrou sur l'entité"""
        now = datetime.now(timezone.utc)
        
        # Vérifier si déjà verrouillé
        if self.is_locked() and self.locked_by != user_id:
            return False
        
        self.lock_token = str(uuid4())
        self.locked_by = user_id
        self.locked_at = now
        if timeout_seconds:
            self.lock_timeout_seconds = timeout_seconds
        
        return True
    
    def release_lock(self, user_id: UUID, lock_token: Optional[str] = None) -> bool:
        """Libère le verrou"""
        if not self.is_locked():
            return True
        
        if self.locked_by != user_id:
            return False
        
        if lock_token and self.lock_token != lock_token:
            return False
        
        self.lock_token = None
        self.locked_by = None
        self.locked_at = None
        
        return True
    
    def is_locked(self) -> bool:
        """Vérifie si l'entité est verrouillée"""
        if not self.locked_at or not self.lock_token:
            return False
        
        # Vérifier le timeout
        timeout_delta = timedelta(seconds=self.lock_timeout_seconds)
        if datetime.now(timezone.utc) > self.locked_at + timeout_delta:
            # Auto-libération du verrou expiré
            self.lock_token = None
            self.locked_by = None
            self.locked_at = None
            return False
        
        return True
    
    def can_edit(self, user_id: UUID) -> bool:
        """Vérifie si l'utilisateur peut éditer l'entité"""
        if not self.is_locked():
            return True
        return self.locked_by == user_id


class CacheableMixin(BaseModel):
    """Mixin pour la gestion du cache avec invalidation intelligente"""
    cache_key: Optional[str] = Field(None, description="Clé de cache")
    cache_expires_at: Optional[datetime] = Field(None, description="Expiration du cache")
    cache_version: int = Field(1, description="Version du cache")
    cache_tags: Set[str] = Field(default_factory=set, description="Tags pour l'invalidation")
    cache_dependencies: Set[str] = Field(default_factory=set, description="Dépendances de cache")
    
    def generate_cache_key(self, prefix: str = "") -> str:
        """Génère une clé de cache unique"""
        entity_id = getattr(self, 'id', 'unknown')
        entity_type = self.__class__.__name__.lower()
        version = getattr(self, 'version', 1)
        
        base_key = f"{prefix}{entity_type}:{entity_id}:v{version}"
        
        # Ajouter un hash du contenu pour la sensibilité aux changements
        content_hash = hashlib.sha256(
            json.dumps(self.dict(), sort_keys=True, default=str).encode()
        ).hexdigest()[:8]
        
        self.cache_key = f"{base_key}:{content_hash}"
        return self.cache_key
    
    def set_cache_expiration(self, ttl_seconds: int):
        """Définit l'expiration du cache"""
        self.cache_expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
    
    def is_cache_valid(self) -> bool:
        """Vérifie si le cache est encore valide"""
        if not self.cache_expires_at:
            return True
        return datetime.now(timezone.utc) < self.cache_expires_at
    
    def invalidate_cache(self, cascade: bool = False):
        """Invalide le cache"""
        self.cache_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        self.cache_version += 1
        
        if cascade and self.cache_dependencies:
            # Logique d'invalidation en cascade (à implémenter selon le contexte)
            pass
    
    def add_cache_tag(self, tag: str):
        """Ajoute un tag de cache"""
        self.cache_tags.add(tag)
    
    def remove_cache_tag(self, tag: str):
        """Supprime un tag de cache"""
        self.cache_tags.discard(tag)
    
    def has_cache_tag(self, tag: str) -> bool:
        """Vérifie la présence d'un tag"""
        return tag in self.cache_tags


class ObservableMixin(BaseModel):
    """Mixin pour l'observabilité avec événements et métriques"""
    observers: List[Callable] = Field(default_factory=list, exclude=True)
    event_history: List[Dict[str, Any]] = Field(default_factory=list, description="Historique des événements")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Métriques personnalisées")
    telemetry_enabled: bool = Field(True, description="Activation de la télémétrie")
    
    def add_observer(self, observer: Callable):
        """Ajoute un observateur"""
        if observer not in self.observers:
            self.observers.append(observer)
    
    def remove_observer(self, observer: Callable):
        """Supprime un observateur"""
        if observer in self.observers:
            self.observers.remove(observer)
    
    def notify_observers(self, event_type: str, data: Optional[Dict[str, Any]] = None):
        """Notifie tous les observateurs"""
        if not self.telemetry_enabled:
            return
        
        event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_type': event_type,
            'entity_id': str(getattr(self, 'id', 'unknown')),
            'entity_type': self.__class__.__name__,
            'data': data or {}
        }
        
        # Ajouter à l'historique
        self.event_history.append(event)
        
        # Limiter la taille de l'historique
        if len(self.event_history) > 100:
            self.event_history = self.event_history[-50:]
        
        # Notifier les observateurs
        for observer in self.observers:
            try:
                observer(event)
            except Exception as e:
                # Log l'erreur mais ne pas interrompre
                print(f"Observer error: {e}")
    
    def record_metric(self, name: str, value: Any, tags: Optional[Dict[str, str]] = None):
        """Enregistre une métrique"""
        if not self.telemetry_enabled:
            return
        
        metric = {
            'value': value,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'tags': tags or {}
        }
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(metric)
        
        # Limiter le nombre de métriques
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-500:]
    
    def get_metric_history(self, name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Récupère l'historique d'une métrique"""
        return self.metrics.get(name, [])[-limit:]
    
    def get_recent_events(self, event_type: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Récupère les événements récents"""
        events = self.event_history[-limit:]
        if event_type:
            events = [e for e in events if e.get('event_type') == event_type]
        return events


class WorkflowMixin(BaseModel):
    """Mixin pour la gestion des workflows et états"""
    workflow_state: str = Field("initial", description="État actuel du workflow")
    workflow_context: Dict[str, Any] = Field(default_factory=dict, description="Contexte du workflow")
    workflow_history: List[Dict[str, Any]] = Field(default_factory=list, description="Historique des transitions")
    workflow_locked: bool = Field(False, description="Workflow verrouillé")
    workflow_error: Optional[str] = Field(None, description="Dernière erreur de workflow")
    
    def transition_to(self, new_state: str, context: Optional[Dict[str, Any]] = None, 
                     user_id: Optional[UUID] = None) -> bool:
        """Effectue une transition d'état"""
        if self.workflow_locked:
            return False
        
        if not self._can_transition_to(new_state):
            return False
        
        old_state = self.workflow_state
        self.workflow_state = new_state
        
        # Mettre à jour le contexte
        if context:
            self.workflow_context.update(context)
        
        # Enregistrer la transition
        transition = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'from_state': old_state,
            'to_state': new_state,
            'user_id': str(user_id) if user_id else None,
            'context': context or {}
        }
        self.workflow_history.append(transition)
        
        # Limiter l'historique
        if len(self.workflow_history) > 1000:
            self.workflow_history = self.workflow_history[-500:]
        
        # Notifier si observable
        if hasattr(self, 'notify_observers'):
            self.notify_observers('state_transition', transition)
        
        # Exécuter les actions post-transition
        self._on_state_entered(new_state, old_state)
        
        return True
    
    def _can_transition_to(self, new_state: str) -> bool:
        """Vérifie si la transition est autorisée (à surcharger)"""
        return True
    
    def _on_state_entered(self, new_state: str, old_state: str):
        """Actions à exécuter lors de l'entrée dans un nouvel état (à surcharger)"""
        pass
    
    def lock_workflow(self, reason: Optional[str] = None):
        """Verrouille le workflow"""
        self.workflow_locked = True
        if reason:
            self.workflow_context['lock_reason'] = reason
    
    def unlock_workflow(self):
        """Déverrouille le workflow"""
        self.workflow_locked = False
        self.workflow_context.pop('lock_reason', None)
    
    def set_workflow_error(self, error: str):
        """Définit une erreur de workflow"""
        self.workflow_error = error
        self.workflow_context['last_error'] = {
            'message': error,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def clear_workflow_error(self):
        """Efface l'erreur de workflow"""
        self.workflow_error = None
        self.workflow_context.pop('last_error', None)
    
    def get_state_duration(self) -> Optional[timedelta]:
        """Retourne la durée dans l'état actuel"""
        if not self.workflow_history:
            return None
        
        last_transition = self.workflow_history[-1]
        transition_time = datetime.fromisoformat(last_transition['timestamp'].replace('Z', '+00:00'))
        return datetime.now(timezone.utc) - transition_time


class SecurityMixin(BaseModel):
    """Mixin pour la sécurité et l'autorisation"""
    security_level: SecurityLevel = Field(SecurityLevel.STANDARD, description="Niveau de sécurité")
    permissions: Set[str] = Field(default_factory=set, description="Permissions explicites")
    access_control_list: Dict[str, Set[str]] = Field(default_factory=dict, description="Liste de contrôle d'accès")
    security_tags: Set[str] = Field(default_factory=set, description="Tags de sécurité")
    encryption_enabled: bool = Field(False, description="Chiffrement activé")
    audit_required: bool = Field(False, description="Audit requis")
    data_classification: str = Field("internal", description="Classification des données")
    
    @validator('data_classification')
    def validate_data_classification(cls, v):
        """Valide la classification des données"""
        valid_classifications = ['public', 'internal', 'confidential', 'restricted', 'top_secret']
        if v.lower() not in valid_classifications:
            raise ValueError(f'Invalid data classification. Must be one of: {valid_classifications}')
        return v.lower()
    
    def has_permission(self, permission: str, user_permissions: Optional[Set[str]] = None) -> bool:
        """Vérifie si une permission est accordée"""
        # Vérifier les permissions explicites
        if permission in self.permissions:
            return True
        
        # Vérifier les permissions utilisateur si fournies
        if user_permissions and permission in user_permissions:
            return True
        
        return False
    
    def grant_permission(self, permission: str, user_or_role: str):
        """Accorde une permission"""
        self.permissions.add(permission)
        
        if user_or_role not in self.access_control_list:
            self.access_control_list[user_or_role] = set()
        self.access_control_list[user_or_role].add(permission)
    
    def revoke_permission(self, permission: str, user_or_role: Optional[str] = None):
        """Révoque une permission"""
        self.permissions.discard(permission)
        
        if user_or_role and user_or_role in self.access_control_list:
            self.access_control_list[user_or_role].discard(permission)
            if not self.access_control_list[user_or_role]:
                del self.access_control_list[user_or_role]
    
    def check_access(self, user_id: str, action: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Vérifie l'accès pour une action"""
        # Vérifier le niveau de sécurité
        required_permissions = self.security_level.required_controls
        
        # Vérifier les permissions utilisateur
        user_permissions = self.access_control_list.get(user_id, set())
        
        # Logique d'autorisation basée sur l'action
        if action == 'read':
            return True  # Lecture généralement autorisée
        elif action in ['write', 'update']:
            return 'write' in user_permissions or 'admin' in user_permissions
        elif action == 'delete':
            return 'delete' in user_permissions or 'admin' in user_permissions
        elif action == 'admin':
            return 'admin' in user_permissions
        
        return False
    
    def add_security_tag(self, tag: str):
        """Ajoute un tag de sécurité"""
        self.security_tags.add(tag)
    
    def remove_security_tag(self, tag: str):
        """Supprime un tag de sécurité"""
        self.security_tags.discard(tag)
    
    def requires_encryption(self) -> bool:
        """Détermine si le chiffrement est requis"""
        sensitive_classifications = ['confidential', 'restricted', 'top_secret']
        return (
            self.encryption_enabled or 
            self.data_classification in sensitive_classifications or
            self.security_level.security_score >= 75
        )
    
    def get_required_controls(self) -> List[str]:
        """Retourne les contrôles de sécurité requis"""
        base_controls = self.security_level.required_controls.copy()
        
        # Ajouter des contrôles basés sur la classification
        if self.data_classification in ['restricted', 'top_secret']:
            base_controls.extend(['data_loss_prevention', 'access_monitoring'])
        
        if self.audit_required:
            base_controls.append('comprehensive_audit')
        
        return list(set(base_controls))  # Dédoublonner


class PerformanceMixin(BaseModel):
    """Mixin pour le monitoring de performance"""
    performance_metrics: Dict[str, List[float]] = Field(default_factory=dict, description="Métriques de performance")
    performance_thresholds: Dict[str, float] = Field(default_factory=dict, description="Seuils de performance")
    performance_alerts_enabled: bool = Field(True, description="Alertes de performance activées")
    last_performance_check: Optional[datetime] = Field(None, description="Dernière vérification de performance")
    
    def record_performance_metric(self, metric_name: str, value: float):
        """Enregistre une métrique de performance"""
        if metric_name not in self.performance_metrics:
            self.performance_metrics[metric_name] = []
        
        self.performance_metrics[metric_name].append(value)
        
        # Limiter à 1000 points de données
        if len(self.performance_metrics[metric_name]) > 1000:
            self.performance_metrics[metric_name] = self.performance_metrics[metric_name][-500:]
        
        # Vérifier les seuils
        if self.performance_alerts_enabled:
            self._check_performance_threshold(metric_name, value)
    
    def _check_performance_threshold(self, metric_name: str, value: float):
        """Vérifie les seuils de performance"""
        threshold = self.performance_thresholds.get(metric_name)
        if threshold and value > threshold:
            # Logique d'alerte (à intégrer avec le système d'alertes)
            if hasattr(self, 'notify_observers'):
                self.notify_observers('performance_threshold_exceeded', {
                    'metric': metric_name,
                    'value': value,
                    'threshold': threshold
                })
    
    def set_performance_threshold(self, metric_name: str, threshold: float):
        """Définit un seuil de performance"""
        self.performance_thresholds[metric_name] = threshold
    
    def get_performance_stats(self, metric_name: str) -> Optional[Dict[str, float]]:
        """Calcule les statistiques de performance"""
        values = self.performance_metrics.get(metric_name)
        if not values:
            return None
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'latest': values[-1] if values else 0
        }
    
    def check_performance_health(self) -> Dict[str, bool]:
        """Vérifie la santé des performances"""
        health = {}
        
        for metric_name, threshold in self.performance_thresholds.items():
            values = self.performance_metrics.get(metric_name, [])
            if values:
                latest_value = values[-1]
                health[metric_name] = latest_value <= threshold
            else:
                health[metric_name] = True  # Pas de données = OK
        
        self.last_performance_check = datetime.now(timezone.utc)
        return health


__all__ = [
    'VersionedMixin', 'CacheableMixin', 'ObservableMixin', 'WorkflowMixin',
    'SecurityMixin', 'PerformanceMixin'
]
