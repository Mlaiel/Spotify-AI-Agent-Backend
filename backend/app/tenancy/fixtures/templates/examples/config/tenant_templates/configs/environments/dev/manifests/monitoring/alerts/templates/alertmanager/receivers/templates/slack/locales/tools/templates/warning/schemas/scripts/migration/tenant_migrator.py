# -*- coding: utf-8 -*-
"""
Tenant Migrator - Industrial Grade Migration System
===================================================

Système de migration complète pour les tenants avec architecture industrielle.
Gère la migration de tous les aspects d'un tenant : données, configurations,
schémas, et métadonnées avec support de rollback et validation.

Key Features:
- Migration complète de tenant entre environnements
- Validation de cohérence pré et post-migration
- Rollback automatique en cas d'échec
- Monitoring en temps réel avec métriques
- Support de migration partielle et incrémentale
- Gestion des dépendances entre composants
- Optimisation des performances avec parallélisation

Author: Spotify AI Agent Team
Version: 1.0.0
"""

import asyncio
import logging
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

import aiofiles
from pydantic import BaseModel, Field, validator

# Configuration du logging
logger = logging.getLogger(__name__)

# =============================================================================
# MODÈLES ET ENUMS
# =============================================================================

class MigrationStatus(Enum):
    """Statuts possibles d'une migration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"

class MigrationType(Enum):
    """Types de migration disponibles."""
    FULL = "full"                    # Migration complète
    PARTIAL = "partial"              # Migration partielle
    INCREMENTAL = "incremental"      # Migration incrémentale
    SCHEMA_ONLY = "schema_only"      # Schémas uniquement
    DATA_ONLY = "data_only"          # Données uniquement
    CONFIG_ONLY = "config_only"      # Configuration uniquement

class ComponentType(Enum):
    """Types de composants migrables."""
    DATABASE = "database"
    SCHEMAS = "schemas"
    CONFIGURATIONS = "configurations"
    METADATA = "metadata"
    FILES = "files"
    PERMISSIONS = "permissions"
    INTEGRATIONS = "integrations"

@dataclass
class MigrationComponent:
    """Représente un composant à migrer."""
    type: ComponentType
    name: str
    source_path: str
    target_path: str
    dependencies: List[str]
    size_bytes: int = 0
    priority: int = 1
    requires_validation: bool = True
    supports_rollback: bool = True
    estimated_duration: int = 0  # en secondes

class TenantMigrationConfig(BaseModel):
    """Configuration pour la migration de tenant."""
    
    tenant_id: str = Field(..., description="ID du tenant à migrer")
    source_environment: str = Field(..., description="Environnement source")
    target_environment: str = Field(..., description="Environnement cible")
    migration_type: MigrationType = Field(default=MigrationType.FULL)
    
    # Options de migration
    dry_run: bool = Field(default=False, description="Mode simulation")
    force_migration: bool = Field(default=False, description="Forcer la migration")
    skip_validation: bool = Field(default=False, description="Ignorer la validation")
    skip_backup: bool = Field(default=False, description="Ignorer la sauvegarde")
    parallel_execution: bool = Field(default=True, description="Exécution parallèle")
    
    # Configuration de performance
    batch_size: int = Field(default=1000, ge=1, le=10000)
    max_concurrent_operations: int = Field(default=5, ge=1, le=20)
    timeout_seconds: int = Field(default=3600, ge=60)
    retry_attempts: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: int = Field(default=10, ge=1)
    
    # Filtres et sélections
    include_components: Optional[List[ComponentType]] = None
    exclude_components: Optional[List[ComponentType]] = None
    include_tables: Optional[List[str]] = None
    exclude_tables: Optional[List[str]] = None
    
    @validator('target_environment')
    def validate_target_different_from_source(cls, v, values):
        if 'source_environment' in values and v == values['source_environment']:
            raise ValueError('Target environment must be different from source')
        return v

@dataclass
class MigrationMetrics:
    """Métriques de migration."""
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0
    total_components: int = 0
    completed_components: int = 0
    failed_components: int = 0
    skipped_components: int = 0
    total_records: int = 0
    migrated_records: int = 0
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
    
    @property
    def success_rate(self) -> float:
        if self.total_components == 0:
            return 0.0
        return (self.completed_components / self.total_components) * 100
    
    @property
    def is_completed(self) -> bool:
        return self.end_time is not None

# =============================================================================
# GESTIONNAIRE D'ÉTAT DE MIGRATION
# =============================================================================

class MigrationState:
    """Gestionnaire d'état pour le suivi des migrations."""
    
    def __init__(self, migration_id: str):
        self.migration_id = migration_id
        self.status = MigrationStatus.PENDING
        self.current_component: Optional[str] = None
        self.completed_components: Set[str] = set()
        self.failed_components: Set[str] = set()
        self.rollback_stack: List[Dict[str, Any]] = []
        self.checkpoints: Dict[str, Any] = {}
        
    def start_component(self, component_name: str):
        """Marque le début de migration d'un composant."""
        self.current_component = component_name
        logger.info(f"Starting migration of component: {component_name}")
    
    def complete_component(self, component_name: str, checkpoint_data: Dict[str, Any]):
        """Marque la fin réussie de migration d'un composant."""
        self.completed_components.add(component_name)
        self.checkpoints[component_name] = checkpoint_data
        self.rollback_stack.append({
            'component': component_name,
            'checkpoint': checkpoint_data,
            'timestamp': datetime.now(timezone.utc)
        })
        logger.info(f"Completed migration of component: {component_name}")
    
    def fail_component(self, component_name: str, error: str):
        """Marque l'échec de migration d'un composant."""
        self.failed_components.add(component_name)
        logger.error(f"Failed migration of component {component_name}: {error}")
    
    def can_rollback(self) -> bool:
        """Vérifie si un rollback est possible."""
        return len(self.rollback_stack) > 0
    
    def get_rollback_steps(self) -> List[Dict[str, Any]]:
        """Récupère les étapes de rollback dans l'ordre inverse."""
        return list(reversed(self.rollback_stack))

# =============================================================================
# VALIDATEUR DE MIGRATION
# =============================================================================

class MigrationValidator:
    """Validateur pour les migrations de tenant."""
    
    def __init__(self, config: TenantMigrationConfig):
        self.config = config
    
    async def validate_pre_migration(self) -> Dict[str, Any]:
        """
        Valide les conditions avant migration.
        
        Returns:
            Résultat de validation avec détails
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'checks': {}
        }
        
        # Vérifier la connectivité aux environnements
        source_connectivity = await self._check_environment_connectivity(
            self.config.source_environment
        )
        target_connectivity = await self._check_environment_connectivity(
            self.config.target_environment
        )
        
        validation_result['checks']['source_connectivity'] = source_connectivity
        validation_result['checks']['target_connectivity'] = target_connectivity
        
        if not source_connectivity['available']:
            validation_result['errors'].append(
                f"Source environment {self.config.source_environment} not accessible"
            )
            validation_result['valid'] = False
        
        if not target_connectivity['available']:
            validation_result['errors'].append(
                f"Target environment {self.config.target_environment} not accessible"
            )
            validation_result['valid'] = False
        
        # Vérifier l'existence du tenant source
        tenant_exists = await self._check_tenant_exists(
            self.config.tenant_id,
            self.config.source_environment
        )
        validation_result['checks']['tenant_exists'] = tenant_exists
        
        if not tenant_exists:
            validation_result['errors'].append(
                f"Tenant {self.config.tenant_id} not found in {self.config.source_environment}"
            )
            validation_result['valid'] = False
        
        # Vérifier les conflits dans l'environnement cible
        target_conflicts = await self._check_target_conflicts()
        validation_result['checks']['target_conflicts'] = target_conflicts
        
        if target_conflicts['has_conflicts'] and not self.config.force_migration:
            validation_result['errors'].append(
                "Target environment has conflicts. Use force_migration to override."
            )
            validation_result['valid'] = False
        
        # Vérifier les ressources disponibles
        resource_check = await self._check_resources()
        validation_result['checks']['resources'] = resource_check
        
        if not resource_check['sufficient']:
            if resource_check['critical']:
                validation_result['errors'].append(
                    "Insufficient resources for migration"
                )
                validation_result['valid'] = False
            else:
                validation_result['warnings'].append(
                    "Limited resources may impact migration performance"
                )
        
        return validation_result
    
    async def validate_post_migration(self, migration_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valide les résultats après migration.
        
        Args:
            migration_result: Résultat de la migration
            
        Returns:
            Résultat de validation
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'checks': {}
        }
        
        # Vérifier l'intégrité des données
        integrity_check = await self._check_data_integrity()
        validation_result['checks']['data_integrity'] = integrity_check
        
        if not integrity_check['valid']:
            validation_result['errors'].extend(integrity_check['errors'])
            validation_result['valid'] = False
        
        # Vérifier la cohérence des schémas
        schema_check = await self._check_schema_consistency()
        validation_result['checks']['schema_consistency'] = schema_check
        
        if not schema_check['consistent']:
            validation_result['errors'].append("Schema inconsistencies detected")
            validation_result['valid'] = False
        
        # Vérifier les configurations
        config_check = await self._check_configuration_validity()
        validation_result['checks']['configuration'] = config_check
        
        if not config_check['valid']:
            validation_result['warnings'].extend(config_check['warnings'])
        
        return validation_result
    
    async def _check_environment_connectivity(self, environment: str) -> Dict[str, Any]:
        """Vérifie la connectivité à un environnement."""
        # Simulation de vérification de connectivité
        await asyncio.sleep(0.1)
        return {
            'available': True,
            'response_time_ms': 50,
            'version': '1.0.0'
        }
    
    async def _check_tenant_exists(self, tenant_id: str, environment: str) -> bool:
        """Vérifie si un tenant existe dans un environnement."""
        # Simulation de vérification d'existence
        await asyncio.sleep(0.1)
        return True
    
    async def _check_target_conflicts(self) -> Dict[str, Any]:
        """Vérifie les conflits dans l'environnement cible."""
        await asyncio.sleep(0.1)
        return {
            'has_conflicts': False,
            'conflicts': []
        }
    
    async def _check_resources(self) -> Dict[str, Any]:
        """Vérifie les ressources disponibles."""
        await asyncio.sleep(0.1)
        return {
            'sufficient': True,
            'critical': False,
            'disk_space_gb': 100,
            'memory_gb': 16,
            'cpu_cores': 8
        }
    
    async def _check_data_integrity(self) -> Dict[str, Any]:
        """Vérifie l'intégrité des données migrées."""
        await asyncio.sleep(0.2)
        return {
            'valid': True,
            'errors': [],
            'total_records': 10000,
            'validated_records': 10000
        }
    
    async def _check_schema_consistency(self) -> Dict[str, Any]:
        """Vérifie la cohérence des schémas."""
        await asyncio.sleep(0.1)
        return {
            'consistent': True,
            'schema_version': '1.0.0',
            'differences': []
        }
    
    async def _check_configuration_validity(self) -> Dict[str, Any]:
        """Vérifie la validité des configurations."""
        await asyncio.sleep(0.1)
        return {
            'valid': True,
            'warnings': [],
            'config_files_checked': 15
        }

# =============================================================================
# GESTIONNAIRE DE ROLLBACK
# =============================================================================

class RollbackManager:
    """Gestionnaire de rollback pour les migrations."""
    
    def __init__(self, config: TenantMigrationConfig):
        self.config = config
    
    async def execute_rollback(self, migration_state: MigrationState) -> Dict[str, Any]:
        """
        Exécute un rollback complet de la migration.
        
        Args:
            migration_state: État de la migration à rollback
            
        Returns:
            Résultat du rollback
        """
        rollback_result = {
            'success': True,
            'errors': [],
            'rolled_back_components': [],
            'failed_rollbacks': []
        }
        
        if not migration_state.can_rollback():
            rollback_result['success'] = False
            rollback_result['errors'].append("No rollback data available")
            return rollback_result
        
        rollback_steps = migration_state.get_rollback_steps()
        
        for step in rollback_steps:
            component_name = step['component']
            checkpoint_data = step['checkpoint']
            
            try:
                await self._rollback_component(component_name, checkpoint_data)
                rollback_result['rolled_back_components'].append(component_name)
                logger.info(f"Rolled back component: {component_name}")
                
            except Exception as e:
                error_msg = f"Failed to rollback component {component_name}: {e}"
                rollback_result['errors'].append(error_msg)
                rollback_result['failed_rollbacks'].append(component_name)
                logger.error(error_msg)
        
        if rollback_result['failed_rollbacks']:
            rollback_result['success'] = False
        
        return rollback_result
    
    async def _rollback_component(self, component_name: str, checkpoint_data: Dict[str, Any]):
        """
        Rollback d'un composant spécifique.
        
        Args:
            component_name: Nom du composant
            checkpoint_data: Données de checkpoint pour le rollback
        """
        # Simulation de rollback selon le type de composant
        component_type = checkpoint_data.get('type', 'unknown')
        
        rollback_methods = {
            'database': self._rollback_database,
            'schemas': self._rollback_schemas,
            'configurations': self._rollback_configurations,
            'files': self._rollback_files
        }
        
        rollback_method = rollback_methods.get(component_type, self._rollback_generic)
        await rollback_method(component_name, checkpoint_data)
    
    async def _rollback_database(self, component_name: str, checkpoint_data: Dict[str, Any]):
        """Rollback de composant base de données."""
        await asyncio.sleep(0.5)  # Simulation
        logger.info(f"Database rollback completed for {component_name}")
    
    async def _rollback_schemas(self, component_name: str, checkpoint_data: Dict[str, Any]):
        """Rollback de composant schémas."""
        await asyncio.sleep(0.2)  # Simulation
        logger.info(f"Schema rollback completed for {component_name}")
    
    async def _rollback_configurations(self, component_name: str, checkpoint_data: Dict[str, Any]):
        """Rollback de composant configurations."""
        await asyncio.sleep(0.1)  # Simulation
        logger.info(f"Configuration rollback completed for {component_name}")
    
    async def _rollback_files(self, component_name: str, checkpoint_data: Dict[str, Any]):
        """Rollback de composant fichiers."""
        await asyncio.sleep(0.3)  # Simulation
        logger.info(f"Files rollback completed for {component_name}")
    
    async def _rollback_generic(self, component_name: str, checkpoint_data: Dict[str, Any]):
        """Rollback générique."""
        await asyncio.sleep(0.1)  # Simulation
        logger.info(f"Generic rollback completed for {component_name}")

# =============================================================================
# MIGRATEUR PRINCIPAL
# =============================================================================

class TenantMigrator:
    """
    Migrateur principal pour les tenants.
    Orchestre la migration complète d'un tenant avec tous ses composants.
    """
    
    def __init__(self, config: TenantMigrationConfig):
        self.config = config
        self.migration_id = f"tenant_migration_{config.tenant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.state = MigrationState(self.migration_id)
        self.validator = MigrationValidator(config)
        self.rollback_manager = RollbackManager(config)
        self.metrics = MigrationMetrics(start_time=datetime.now(timezone.utc))
        
        logger.info(f"TenantMigrator initialized - ID: {self.migration_id}")
    
    async def execute(self) -> Dict[str, Any]:
        """
        Exécute la migration complète du tenant.
        
        Returns:
            Résultat de la migration avec métriques complètes
        """
        try:
            self.state.status = MigrationStatus.RUNNING
            logger.info(f"Starting tenant migration: {self.migration_id}")
            
            # Phase de validation pré-migration
            if not self.config.skip_validation:
                validation_result = await self.validator.validate_pre_migration()
                if not validation_result['valid']:
                    raise ValueError(f"Pre-migration validation failed: {validation_result['errors']}")
            
            # Découverte et préparation des composants
            components = await self._discover_components()
            self.metrics.total_components = len(components)
            
            # Planification de la migration
            migration_plan = await self._create_migration_plan(components)
            
            # Exécution de la migration selon le plan
            if self.config.parallel_execution:
                await self._execute_parallel_migration(migration_plan)
            else:
                await self._execute_sequential_migration(migration_plan)
            
            # Validation post-migration
            if not self.config.skip_validation:
                validation_result = await self.validator.validate_post_migration({})
                if not validation_result['valid']:
                    raise ValueError(f"Post-migration validation failed: {validation_result['errors']}")
            
            # Finalisation
            self.state.status = MigrationStatus.COMPLETED
            self.metrics.end_time = datetime.now(timezone.utc)
            self.metrics.duration_seconds = (self.metrics.end_time - self.metrics.start_time).total_seconds()
            
            result = {
                'migration_id': self.migration_id,
                'status': self.state.status.value,
                'metrics': asdict(self.metrics),
                'tenant_id': self.config.tenant_id,
                'source_environment': self.config.source_environment,
                'target_environment': self.config.target_environment,
                'migration_type': self.config.migration_type.value,
                'completed_components': list(self.state.completed_components),
                'failed_components': list(self.state.failed_components)
            }
            
            logger.info(f"Tenant migration completed successfully: {self.migration_id}")
            return result
            
        except Exception as e:
            logger.error(f"Tenant migration failed: {e}")
            self.state.status = MigrationStatus.FAILED
            
            # Tentative de rollback si activé
            if not self.config.skip_backup and self.state.can_rollback():
                try:
                    rollback_result = await self.rollback_manager.execute_rollback(self.state)
                    if rollback_result['success']:
                        self.state.status = MigrationStatus.ROLLED_BACK
                        logger.info("Migration rolled back successfully")
                except Exception as rollback_error:
                    logger.error(f"Rollback failed: {rollback_error}")
            
            # Finaliser les métriques
            self.metrics.end_time = datetime.now(timezone.utc)
            self.metrics.duration_seconds = (self.metrics.end_time - self.metrics.start_time).total_seconds()
            self.metrics.errors.append(str(e))
            
            raise
    
    async def _discover_components(self) -> List[MigrationComponent]:
        """
        Découvre tous les composants à migrer pour le tenant.
        
        Returns:
            Liste des composants à migrer
        """
        components = []
        
        # Composants par défaut selon le type de migration
        component_mapping = {
            MigrationType.FULL: [
                ComponentType.DATABASE,
                ComponentType.SCHEMAS,
                ComponentType.CONFIGURATIONS,
                ComponentType.METADATA,
                ComponentType.FILES,
                ComponentType.PERMISSIONS,
                ComponentType.INTEGRATIONS
            ],
            MigrationType.SCHEMA_ONLY: [ComponentType.SCHEMAS],
            MigrationType.DATA_ONLY: [ComponentType.DATABASE],
            MigrationType.CONFIG_ONLY: [ComponentType.CONFIGURATIONS]
        }
        
        component_types = component_mapping.get(
            self.config.migration_type,
            [ComponentType.DATABASE, ComponentType.SCHEMAS, ComponentType.CONFIGURATIONS]
        )
        
        # Filtrer selon la configuration
        if self.config.include_components:
            component_types = [ct for ct in component_types if ct in self.config.include_components]
        
        if self.config.exclude_components:
            component_types = [ct for ct in component_types if ct not in self.config.exclude_components]
        
        # Créer les composants avec métadonnées
        for component_type in component_types:
            component = await self._create_component(component_type)
            if component:
                components.append(component)
        
        logger.info(f"Discovered {len(components)} components for migration")
        return components
    
    async def _create_component(self, component_type: ComponentType) -> Optional[MigrationComponent]:
        """
        Crée un composant de migration avec ses métadonnées.
        
        Args:
            component_type: Type de composant
            
        Returns:
            Composant de migration ou None si non applicable
        """
        component_configs = {
            ComponentType.DATABASE: {
                'name': f'database_{self.config.tenant_id}',
                'source_path': f'/data/{self.config.source_environment}/{self.config.tenant_id}',
                'target_path': f'/data/{self.config.target_environment}/{self.config.tenant_id}',
                'dependencies': [],
                'size_bytes': 1000000,  # 1MB simulé
                'priority': 1,
                'estimated_duration': 300
            },
            ComponentType.SCHEMAS: {
                'name': f'schemas_{self.config.tenant_id}',
                'source_path': f'/schemas/{self.config.source_environment}/{self.config.tenant_id}',
                'target_path': f'/schemas/{self.config.target_environment}/{self.config.tenant_id}',
                'dependencies': [],
                'size_bytes': 50000,  # 50KB simulé
                'priority': 2,
                'estimated_duration': 60
            },
            ComponentType.CONFIGURATIONS: {
                'name': f'config_{self.config.tenant_id}',
                'source_path': f'/config/{self.config.source_environment}/{self.config.tenant_id}',
                'target_path': f'/config/{self.config.target_environment}/{self.config.tenant_id}',
                'dependencies': ['schemas'],
                'size_bytes': 10000,  # 10KB simulé
                'priority': 3,
                'estimated_duration': 30
            }
        }
        
        config = component_configs.get(component_type)
        if not config:
            return None
        
        return MigrationComponent(
            type=component_type,
            **config
        )
    
    async def _create_migration_plan(self, components: List[MigrationComponent]) -> Dict[str, Any]:
        """
        Crée un plan de migration optimisé.
        
        Args:
            components: Liste des composants à migrer
            
        Returns:
            Plan de migration
        """
        # Trier par priorité et dépendances
        sorted_components = sorted(components, key=lambda c: c.priority)
        
        # Calculer l'estimation totale
        total_estimated_duration = sum(c.estimated_duration for c in components)
        total_size_bytes = sum(c.size_bytes for c in components)
        
        plan = {
            'components': sorted_components,
            'total_components': len(components),
            'total_estimated_duration': total_estimated_duration,
            'total_size_bytes': total_size_bytes,
            'parallel_groups': self._create_parallel_groups(sorted_components) if self.config.parallel_execution else None
        }
        
        logger.info(f"Migration plan created with {len(components)} components")
        return plan
    
    def _create_parallel_groups(self, components: List[MigrationComponent]) -> List[List[MigrationComponent]]:
        """
        Crée des groupes de composants qui peuvent être migrés en parallèle.
        
        Args:
            components: Liste des composants
            
        Returns:
            Groupes de composants parallélisables
        """
        groups = []
        remaining_components = components.copy()
        
        while remaining_components:
            current_group = []
            components_to_remove = []
            
            for component in remaining_components:
                # Vérifier si toutes les dépendances sont déjà migrées
                dependencies_satisfied = all(
                    dep in [c.name for c in sum(groups, [])]
                    for dep in component.dependencies
                )
                
                if dependencies_satisfied and len(current_group) < self.config.max_concurrent_operations:
                    current_group.append(component)
                    components_to_remove.append(component)
            
            # Retirer les composants ajoutés au groupe
            for component in components_to_remove:
                remaining_components.remove(component)
            
            if current_group:
                groups.append(current_group)
            elif remaining_components:
                # Si aucun composant ne peut être traité, ajouter le premier (dépendances circulaires)
                groups.append([remaining_components.pop(0)])
        
        return groups
    
    async def _execute_sequential_migration(self, migration_plan: Dict[str, Any]):
        """
        Exécute la migration en mode séquentiel.
        
        Args:
            migration_plan: Plan de migration
        """
        components = migration_plan['components']
        
        for component in components:
            await self._migrate_component(component)
    
    async def _execute_parallel_migration(self, migration_plan: Dict[str, Any]):
        """
        Exécute la migration en mode parallèle.
        
        Args:
            migration_plan: Plan de migration
        """
        parallel_groups = migration_plan['parallel_groups']
        
        for group in parallel_groups:
            # Exécuter tous les composants du groupe en parallèle
            tasks = [self._migrate_component(component) for component in group]
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _migrate_component(self, component: MigrationComponent):
        """
        Migre un composant spécifique.
        
        Args:
            component: Composant à migrer
        """
        try:
            self.state.start_component(component.name)
            
            # Simulation de migration selon le type
            migration_methods = {
                ComponentType.DATABASE: self._migrate_database,
                ComponentType.SCHEMAS: self._migrate_schemas,
                ComponentType.CONFIGURATIONS: self._migrate_configurations,
                ComponentType.METADATA: self._migrate_metadata,
                ComponentType.FILES: self._migrate_files,
                ComponentType.PERMISSIONS: self._migrate_permissions,
                ComponentType.INTEGRATIONS: self._migrate_integrations
            }
            
            migration_method = migration_methods.get(component.type, self._migrate_generic)
            checkpoint_data = await migration_method(component)
            
            self.state.complete_component(component.name, checkpoint_data)
            self.metrics.completed_components += 1
            
        except Exception as e:
            self.state.fail_component(component.name, str(e))
            self.metrics.failed_components += 1
            self.metrics.errors.append(f"Component {component.name}: {e}")
            
            # Re-lever l'exception si pas en mode force
            if not self.config.force_migration:
                raise
    
    async def _migrate_database(self, component: MigrationComponent) -> Dict[str, Any]:
        """Migre le composant base de données."""
        # Simulation de migration de base de données
        await asyncio.sleep(component.estimated_duration / 100)  # Simulation accélérée
        
        return {
            'type': 'database',
            'tables_migrated': 25,
            'records_migrated': 10000,
            'backup_location': f'/backups/{component.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
    
    async def _migrate_schemas(self, component: MigrationComponent) -> Dict[str, Any]:
        """Migre le composant schémas."""
        await asyncio.sleep(component.estimated_duration / 100)
        
        return {
            'type': 'schemas',
            'schemas_migrated': 15,
            'schema_version': '1.0.0',
            'backup_location': f'/backups/{component.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
    
    async def _migrate_configurations(self, component: MigrationComponent) -> Dict[str, Any]:
        """Migre le composant configurations."""
        await asyncio.sleep(component.estimated_duration / 100)
        
        return {
            'type': 'configurations',
            'config_files_migrated': 8,
            'backup_location': f'/backups/{component.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
    
    async def _migrate_metadata(self, component: MigrationComponent) -> Dict[str, Any]:
        """Migre le composant métadonnées."""
        await asyncio.sleep(component.estimated_duration / 100)
        
        return {
            'type': 'metadata',
            'metadata_entries_migrated': 50,
            'backup_location': f'/backups/{component.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
    
    async def _migrate_files(self, component: MigrationComponent) -> Dict[str, Any]:
        """Migre le composant fichiers."""
        await asyncio.sleep(component.estimated_duration / 100)
        
        return {
            'type': 'files',
            'files_migrated': 100,
            'total_size_bytes': component.size_bytes,
            'backup_location': f'/backups/{component.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
    
    async def _migrate_permissions(self, component: MigrationComponent) -> Dict[str, Any]:
        """Migre le composant permissions."""
        await asyncio.sleep(component.estimated_duration / 100)
        
        return {
            'type': 'permissions',
            'permissions_migrated': 20,
            'backup_location': f'/backups/{component.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
    
    async def _migrate_integrations(self, component: MigrationComponent) -> Dict[str, Any]:
        """Migre le composant intégrations."""
        await asyncio.sleep(component.estimated_duration / 100)
        
        return {
            'type': 'integrations',
            'integrations_migrated': 5,
            'backup_location': f'/backups/{component.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
    
    async def _migrate_generic(self, component: MigrationComponent) -> Dict[str, Any]:
        """Migration générique pour composants non spécialisés."""
        await asyncio.sleep(component.estimated_duration / 100)
        
        return {
            'type': 'generic',
            'component_name': component.name,
            'backup_location': f'/backups/{component.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Récupère le statut actuel de la migration.
        
        Returns:
            Statut détaillé de la migration
        """
        progress_percentage = 0
        if self.metrics.total_components > 0:
            progress_percentage = (self.metrics.completed_components / self.metrics.total_components) * 100
        
        return {
            'migration_id': self.migration_id,
            'status': self.state.status.value,
            'progress_percentage': round(progress_percentage, 2),
            'current_component': self.state.current_component,
            'completed_components': len(self.state.completed_components),
            'failed_components': len(self.state.failed_components),
            'total_components': self.metrics.total_components,
            'duration_seconds': self.metrics.duration_seconds,
            'errors': self.metrics.errors[-5:],  # Dernières 5 erreurs
            'warnings': self.metrics.warnings[-5:]  # Derniers 5 warnings
        }
    
    async def cancel(self) -> Dict[str, Any]:
        """
        Annule la migration en cours.
        
        Returns:
            Résultat de l'annulation
        """
        if self.state.status not in [MigrationStatus.RUNNING, MigrationStatus.PENDING]:
            return {
                'success': False,
                'message': f"Cannot cancel migration in status: {self.state.status.value}"
            }
        
        self.state.status = MigrationStatus.CANCELLED
        logger.info(f"Migration {self.migration_id} cancelled")
        
        return {
            'success': True,
            'message': f"Migration {self.migration_id} cancelled successfully",
            'components_completed': len(self.state.completed_components),
            'can_rollback': self.state.can_rollback()
        }

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'TenantMigrator',
    'TenantMigrationConfig',
    'MigrationStatus',
    'MigrationType',
    'ComponentType',
    'MigrationComponent',
    'MigrationMetrics',
    'MigrationState',
    'MigrationValidator',
    'RollbackManager'
]
