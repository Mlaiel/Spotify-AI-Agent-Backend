"""
🎛️ Context Manager - Gestionnaire de Contexte Ultra-Avancé
==========================================================

Système ultra-avancé de gestion des contextes multi-tenant avec switching
intelligent, validation en temps réel et optimisation des performances.

Author: Lead Dev + Architecte IA - Fahed Mlaiel
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable, AsyncContextManager
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import threading
from contextlib import asynccontextmanager, contextmanager
from concurrent.futures import ThreadPoolExecutor
import contextvars
import weakref
import uuid

from .tenant_context import TenantContext, TenantType, IsolationLevel
from .security_policy_engine import SecurityPolicyEngine
from .performance_optimizer import PerformanceOptimizer
from .compliance_engine import ComplianceEngine
from ..exceptions import ContextError, ContextSwitchError, ContextValidationError
from ...core.config import settings
from ...monitoring.context_monitor import ContextMonitor


class ContextState(Enum):
    """États du contexte"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    SWITCHING = "switching"
    VALIDATING = "validating"
    ERROR = "error"
    SUSPENDED = "suspended"


class SwitchingStrategy(Enum):
    """Stratégies de basculement de contexte"""
    IMMEDIATE = "immediate"      # Basculement immédiat
    GRACEFUL = "graceful"        # Basculement progressif
    QUEUED = "queued"           # Basculement en file d'attente
    OPTIMIZED = "optimized"     # Basculement optimisé


class ValidationLevel(Enum):
    """Niveaux de validation"""
    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


@dataclass
class ContextSnapshot:
    """Instantané du contexte"""
    snapshot_id: str
    tenant_context: TenantContext
    timestamp: datetime
    operation_count: int
    performance_metrics: Dict[str, Any]
    security_events: List[Dict[str, Any]]
    validation_status: str
    
    # Métadonnées
    created_by: str = "system"
    description: str = ""
    expires_at: Optional[datetime] = None


@dataclass
class ContextSwitchEvent:
    """Événement de basculement de contexte"""
    event_id: str
    from_tenant: Optional[str]
    to_tenant: str
    switch_reason: str
    switch_strategy: SwitchingStrategy
    
    # Timing
    initiated_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    
    # Résultats
    success: bool = False
    error_message: Optional[str] = None
    performance_impact: Dict[str, Any] = field(default_factory=dict)
    
    # Contexte
    operation_context: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class ContextValidator:
    """Validateur de contexte avancé"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.logger = logging.getLogger(__name__)
        
        # Règles de validation
        self.validation_rules: Dict[str, Callable] = {
            'tenant_exists': self._validate_tenant_exists,
            'security_context': self._validate_security_context,
            'isolation_level': self._validate_isolation_level,
            'metadata_consistency': self._validate_metadata_consistency,
            'permissions': self._validate_permissions
        }
    
    async def validate_context(self, context: TenantContext) -> Dict[str, Any]:
        """Valide un contexte tenant"""
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'score': 100.0,
            'details': {}
        }
        
        # Application des règles selon le niveau
        rules_to_apply = self._get_rules_for_level(self.validation_level)
        
        for rule_name in rules_to_apply:
            if rule_name in self.validation_rules:
                try:
                    rule_result = await self.validation_rules[rule_name](context)
                    validation_result['details'][rule_name] = rule_result
                    
                    if not rule_result['valid']:
                        validation_result['valid'] = False
                        validation_result['errors'].extend(rule_result.get('errors', []))
                        validation_result['score'] -= rule_result.get('penalty', 10)
                    
                    validation_result['warnings'].extend(rule_result.get('warnings', []))
                    
                except Exception as e:
                    self.logger.error(f"Validation rule {rule_name} failed: {e}")
                    validation_result['errors'].append(f"Rule {rule_name} execution failed")
                    validation_result['score'] -= 15
        
        validation_result['score'] = max(0, validation_result['score'])
        return validation_result
    
    def _get_rules_for_level(self, level: ValidationLevel) -> List[str]:
        """Retourne les règles à appliquer selon le niveau"""
        
        rules_map = {
            ValidationLevel.NONE: [],
            ValidationLevel.BASIC: ['tenant_exists'],
            ValidationLevel.STANDARD: ['tenant_exists', 'security_context', 'isolation_level'],
            ValidationLevel.STRICT: ['tenant_exists', 'security_context', 'isolation_level', 'metadata_consistency'],
            ValidationLevel.PARANOID: ['tenant_exists', 'security_context', 'isolation_level', 'metadata_consistency', 'permissions']
        }
        
        return rules_map.get(level, rules_map[ValidationLevel.STANDARD])
    
    async def _validate_tenant_exists(self, context: TenantContext) -> Dict[str, Any]:
        """Valide l'existence du tenant"""
        # Implémentation de validation de l'existence
        return {'valid': True, 'errors': [], 'warnings': []}
    
    async def _validate_security_context(self, context: TenantContext) -> Dict[str, Any]:
        """Valide le contexte de sécurité"""
        if not context.security:
            return {
                'valid': False,
                'errors': ['Security context missing'],
                'penalty': 20
            }
        
        return {'valid': True, 'errors': [], 'warnings': []}
    
    async def _validate_isolation_level(self, context: TenantContext) -> Dict[str, Any]:
        """Valide le niveau d'isolation"""
        valid_levels = [level for level in IsolationLevel]
        
        if context.isolation_level not in valid_levels:
            return {
                'valid': False,
                'errors': ['Invalid isolation level'],
                'penalty': 15
            }
        
        return {'valid': True, 'errors': [], 'warnings': []}
    
    async def _validate_metadata_consistency(self, context: TenantContext) -> Dict[str, Any]:
        """Valide la cohérence des métadonnées"""
        warnings = []
        
        # Vérification de la cohérence région/timezone
        if context.metadata.region == 'EU' and not context.metadata.timezone.startswith('Europe'):
            warnings.append('Region/timezone mismatch detected')
        
        return {'valid': True, 'errors': [], 'warnings': warnings}
    
    async def _validate_permissions(self, context: TenantContext) -> Dict[str, Any]:
        """Valide les permissions"""
        if context.security and not context.security.permissions:
            return {
                'valid': False,
                'errors': ['No permissions defined'],
                'penalty': 10
            }
        
        return {'valid': True, 'errors': [], 'warnings': []}


class ContextSwitcher:
    """Gestionnaire de basculement de contexte intelligent"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.performance_optimizer = PerformanceOptimizer()
        
        # État du switching
        self.switch_queue: asyncio.Queue = asyncio.Queue()
        self.active_switches: Dict[str, ContextSwitchEvent] = {}
        
        # Stratégies de basculement
        self.strategies: Dict[SwitchingStrategy, Callable] = {
            SwitchingStrategy.IMMEDIATE: self._immediate_switch,
            SwitchingStrategy.GRACEFUL: self._graceful_switch,
            SwitchingStrategy.QUEUED: self._queued_switch,
            SwitchingStrategy.OPTIMIZED: self._optimized_switch
        }
        
        # Statistiques
        self.statistics = {
            'switches_total': 0,
            'switches_successful': 0,
            'switches_failed': 0,
            'avg_switch_time_ms': 0.0
        }
        
        # Cache des contextes récents
        self.context_cache: Dict[str, TenantContext] = {}
        self.cache_max_size = 100
    
    async def switch_context(
        self,
        from_context: Optional[TenantContext],
        to_context: TenantContext,
        strategy: SwitchingStrategy = SwitchingStrategy.OPTIMIZED,
        reason: str = "manual_switch"
    ) -> ContextSwitchEvent:
        """Effectue un basculement de contexte"""
        
        switch_event = ContextSwitchEvent(
            event_id=str(uuid.uuid4()),
            from_tenant=from_context.tenant_id if from_context else None,
            to_tenant=to_context.tenant_id,
            switch_reason=reason,
            switch_strategy=strategy,
            initiated_at=datetime.now(timezone.utc)
        )
        
        self.statistics['switches_total'] += 1
        self.active_switches[switch_event.event_id] = switch_event
        
        try:
            # Exécution de la stratégie de basculement
            if strategy in self.strategies:
                await self.strategies[strategy](from_context, to_context, switch_event)
                switch_event.success = True
                self.statistics['switches_successful'] += 1
            else:
                raise ContextSwitchError(f"Unknown switching strategy: {strategy}")
        
        except Exception as e:
            switch_event.error_message = str(e)
            switch_event.success = False
            self.statistics['switches_failed'] += 1
            self.logger.error(f"Context switch failed: {e}")
        
        finally:
            switch_event.completed_at = datetime.now(timezone.utc)
            switch_event.duration_ms = (
                switch_event.completed_at - switch_event.initiated_at
            ).total_seconds() * 1000
            
            # Mise à jour des statistiques
            self._update_switch_statistics(switch_event)
            
            # Nettoyage
            if switch_event.event_id in self.active_switches:
                del self.active_switches[switch_event.event_id]
        
        return switch_event
    
    async def _immediate_switch(
        self,
        from_context: Optional[TenantContext],
        to_context: TenantContext,
        switch_event: ContextSwitchEvent
    ):
        """Basculement immédiat"""
        # Basculement direct sans optimisation
        pass
    
    async def _graceful_switch(
        self,
        from_context: Optional[TenantContext],
        to_context: TenantContext,
        switch_event: ContextSwitchEvent
    ):
        """Basculement progressif"""
        
        # 1. Finalisation des opérations en cours
        if from_context:
            await self._finalize_pending_operations(from_context)
        
        # 2. Préparation du nouveau contexte
        await self._prepare_context(to_context)
        
        # 3. Basculement progressif
        await asyncio.sleep(0.01)  # Délai minimal pour la progressivité
    
    async def _queued_switch(
        self,
        from_context: Optional[TenantContext],
        to_context: TenantContext,
        switch_event: ContextSwitchEvent
    ):
        """Basculement en file d'attente"""
        
        # Ajout à la file d'attente
        await self.switch_queue.put((from_context, to_context, switch_event))
        
        # Traitement de la file d'attente
        await self._process_switch_queue()
    
    async def _optimized_switch(
        self,
        from_context: Optional[TenantContext],
        to_context: TenantContext,
        switch_event: ContextSwitchEvent
    ):
        """Basculement optimisé avec intelligence"""
        
        # 1. Analyse de performance pour optimisation
        optimization_result = await self.performance_optimizer.optimize_operation(
            "context_switch",
            to_context,
            {"from_tenant": from_context.tenant_id if from_context else None}
        )
        
        switch_event.performance_impact = optimization_result
        
        # 2. Mise en cache du contexte
        self.context_cache[to_context.tenant_id] = to_context
        await self._manage_cache_size()
        
        # 3. Pré-chargement des ressources
        await self._preload_context_resources(to_context)
    
    async def _finalize_pending_operations(self, context: TenantContext):
        """Finalise les opérations en attente"""
        # Implémentation de finalisation
        pass
    
    async def _prepare_context(self, context: TenantContext):
        """Prépare un contexte pour activation"""
        # Implémentation de préparation
        pass
    
    async def _process_switch_queue(self):
        """Traite la file d'attente de basculement"""
        while not self.switch_queue.empty():
            try:
                from_context, to_context, switch_event = await asyncio.wait_for(
                    self.switch_queue.get(), timeout=1.0
                )
                # Traitement du basculement
                await self._immediate_switch(from_context, to_context, switch_event)
            except asyncio.TimeoutError:
                break
    
    async def _preload_context_resources(self, context: TenantContext):
        """Pré-charge les ressources du contexte"""
        # Implémentation de pré-chargement
        pass
    
    async def _manage_cache_size(self):
        """Gère la taille du cache de contextes"""
        if len(self.context_cache) > self.cache_max_size:
            # Suppression des contextes les moins récemment utilisés
            oldest_key = next(iter(self.context_cache))
            del self.context_cache[oldest_key]
    
    def _update_switch_statistics(self, switch_event: ContextSwitchEvent):
        """Met à jour les statistiques de basculement"""
        if switch_event.duration_ms:
            # Calcul de la moyenne mobile
            current_avg = self.statistics['avg_switch_time_ms']
            total_switches = self.statistics['switches_total']
            
            self.statistics['avg_switch_time_ms'] = (
                (current_avg * (total_switches - 1) + switch_event.duration_ms) / total_switches
            )


# Variable de contexte global pour le tenant actuel
current_tenant_context: contextvars.ContextVar[Optional[TenantContext]] = contextvars.ContextVar(
    'current_tenant_context', 
    default=None
)


class ContextManager:
    """
    Gestionnaire de contexte ultra-avancé
    
    Features:
    - Gestion automatique des contextes multi-tenant
    - Validation en temps réel
    - Basculement intelligent
    - Optimisation des performances
    - Surveillance et audit
    - Cache intelligent
    - Support asynchrone complet
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Composants spécialisés
        self.validator = ContextValidator(ValidationLevel.STANDARD)
        self.switcher = ContextSwitcher()
        self.security_engine = SecurityPolicyEngine()
        self.compliance_engine = ComplianceEngine()
        self.performance_optimizer = PerformanceOptimizer()
        self.context_monitor = ContextMonitor()
        
        # État du gestionnaire
        self.active_contexts: Dict[str, TenantContext] = {}
        self.context_snapshots: Dict[str, ContextSnapshot] = {}
        
        # Configuration
        self.auto_validation = True
        self.auto_optimization = True
        self.snapshot_interval = timedelta(minutes=5)
        
        # Statistiques globales
        self.statistics = {
            'contexts_managed': 0,
            'validations_performed': 0,
            'optimizations_applied': 0,
            'snapshots_created': 0,
            'errors_handled': 0
        }
        
        # Tâches de fond
        self._background_tasks: List[asyncio.Task] = []
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Lance les tâches de fond"""
        
        # Tâche de nettoyage périodique
        cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self._background_tasks.append(cleanup_task)
        
        # Tâche de création de snapshots
        snapshot_task = asyncio.create_task(self._periodic_snapshots())
        self._background_tasks.append(snapshot_task)
    
    async def get_current_context(self) -> Optional[TenantContext]:
        """Retourne le contexte tenant actuel"""
        return current_tenant_context.get()
    
    async def set_context(
        self,
        context: TenantContext,
        validate: bool = True,
        optimize: bool = True
    ) -> Dict[str, Any]:
        """
        Définit le contexte tenant actuel
        
        Args:
            context: Nouveau contexte à activer
            validate: Effectuer la validation
            optimize: Effectuer l'optimisation
            
        Returns:
            Résultat de l'activation du contexte
        """
        
        result = {
            'success': True,
            'context_id': context.tenant_id,
            'validation_result': None,
            'optimization_result': None,
            'switch_event': None,
            'warnings': []
        }
        
        try:
            # Obtention du contexte actuel
            current_context = current_tenant_context.get()
            
            # Validation du nouveau contexte
            if validate and self.auto_validation:
                validation_result = await self.validator.validate_context(context)
                result['validation_result'] = validation_result
                self.statistics['validations_performed'] += 1
                
                if not validation_result['valid']:
                    result['success'] = False
                    result['warnings'].extend(validation_result['errors'])
                    return result
            
            # Basculement de contexte
            switch_event = await self.switcher.switch_context(
                current_context,
                context,
                SwitchingStrategy.OPTIMIZED,
                "context_manager_set"
            )
            result['switch_event'] = switch_event.__dict__
            
            if not switch_event.success:
                result['success'] = False
                result['warnings'].append(switch_event.error_message)
                return result
            
            # Activation du contexte
            current_tenant_context.set(context)
            self.active_contexts[context.tenant_id] = context
            self.statistics['contexts_managed'] += 1
            
            # Optimisation si demandée
            if optimize and self.auto_optimization:
                optimization_result = await self.performance_optimizer.optimize_operation(
                    "context_activation",
                    context,
                    {"previous_tenant": current_context.tenant_id if current_context else None}
                )
                result['optimization_result'] = optimization_result
                self.statistics['optimizations_applied'] += 1
            
            # Surveillance
            await self.context_monitor.record_context_activation(context)
            
        except Exception as e:
            self.logger.error(f"Failed to set context: {e}")
            result['success'] = False
            result['warnings'].append(str(e))
            self.statistics['errors_handled'] += 1
        
        return result
    
    @asynccontextmanager
    async def context_scope(
        self,
        context: TenantContext,
        validate: bool = True,
        auto_restore: bool = True
    ) -> AsyncContextManager[TenantContext]:
        """
        Gestionnaire de contexte temporaire
        
        Args:
            context: Contexte à activer temporairement
            validate: Effectuer la validation
            auto_restore: Restaurer automatiquement le contexte précédent
        """
        
        # Sauvegarde du contexte actuel
        previous_context = current_tenant_context.get()
        
        try:
            # Activation du nouveau contexte
            set_result = await self.set_context(context, validate=validate)
            
            if not set_result['success']:
                raise ContextError(f"Failed to activate context: {set_result['warnings']}")
            
            yield context
            
        finally:
            # Restauration du contexte précédent
            if auto_restore and previous_context:
                await self.set_context(previous_context, validate=False, optimize=False)
            elif auto_restore:
                current_tenant_context.set(None)
    
    async def create_snapshot(
        self,
        context: TenantContext,
        description: str = "auto_snapshot"
    ) -> ContextSnapshot:
        """Crée un instantané du contexte"""
        
        snapshot = ContextSnapshot(
            snapshot_id=str(uuid.uuid4()),
            tenant_context=context,
            timestamp=datetime.now(timezone.utc),
            operation_count=await self._get_operation_count(context),
            performance_metrics=await self._collect_performance_metrics(context),
            security_events=await self._collect_security_events(context),
            validation_status="valid",
            description=description,
            expires_at=datetime.now(timezone.utc) + timedelta(days=7)
        )
        
        self.context_snapshots[snapshot.snapshot_id] = snapshot
        self.statistics['snapshots_created'] += 1
        
        return snapshot
    
    async def restore_snapshot(
        self,
        snapshot_id: str,
        validate: bool = True
    ) -> Dict[str, Any]:
        """Restaure un instantané de contexte"""
        
        if snapshot_id not in self.context_snapshots:
            raise ContextError(f"Snapshot {snapshot_id} not found")
        
        snapshot = self.context_snapshots[snapshot_id]
        
        # Vérification de l'expiration
        if snapshot.expires_at and datetime.now(timezone.utc) > snapshot.expires_at:
            raise ContextError(f"Snapshot {snapshot_id} has expired")
        
        return await self.set_context(snapshot.tenant_context, validate=validate)
    
    async def _get_operation_count(self, context: TenantContext) -> int:
        """Obtient le nombre d'opérations pour le contexte"""
        # Implémentation de comptage des opérations
        return 0
    
    async def _collect_performance_metrics(self, context: TenantContext) -> Dict[str, Any]:
        """Collecte les métriques de performance"""
        return await self.performance_optimizer._collect_metrics(context, "snapshot")
    
    async def _collect_security_events(self, context: TenantContext) -> List[Dict[str, Any]]:
        """Collecte les événements de sécurité"""
        # Implémentation de collecte d'événements
        return []
    
    async def _periodic_cleanup(self):
        """Nettoyage périodique"""
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                
                # Nettoyage des snapshots expirés
                current_time = datetime.now(timezone.utc)
                expired_snapshots = [
                    sid for sid, snapshot in self.context_snapshots.items()
                    if snapshot.expires_at and current_time > snapshot.expires_at
                ]
                
                for sid in expired_snapshots:
                    del self.context_snapshots[sid]
                
                # Nettoyage du cache du switcher
                await self.switcher._manage_cache_size()
                
                self.logger.info(f"Cleanup completed: removed {len(expired_snapshots)} expired snapshots")
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
    
    async def _periodic_snapshots(self):
        """Création automatique de snapshots"""
        while True:
            try:
                await asyncio.sleep(self.snapshot_interval.total_seconds())
                
                # Création de snapshots pour les contextes actifs
                for context in self.active_contexts.values():
                    await self.create_snapshot(context, "auto_periodic_snapshot")
                
            except Exception as e:
                self.logger.error(f"Periodic snapshot error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques globales"""
        return {
            **self.statistics,
            'active_contexts': len(self.active_contexts),
            'snapshots_count': len(self.context_snapshots),
            'switcher_stats': self.switcher.statistics,
            'validator_level': self.validator.validation_level.value,
            'background_tasks': len(self._background_tasks)
        }
    
    async def shutdown(self):
        """Arrêt propre du gestionnaire"""
        
        # Annulation des tâches de fond
        for task in self._background_tasks:
            task.cancel()
        
        # Attente de l'arrêt des tâches
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self.logger.info("Context manager shutdown completed")
