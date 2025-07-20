"""
Migration de Locales Avancée pour Spotify AI Agent
Système de migration et synchronisation pour les données de localisation
"""

import asyncio
import json
import hashlib
import logging
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import os
import shutil
import tempfile
from pathlib import Path
import aiofiles

logger = logging.getLogger(__name__)


class MigrationType(Enum):
    """Types de migration"""
    SCHEMA = "schema"
    DATA = "data"
    CONFIG = "config"
    TENANT = "tenant"
    LOCALE = "locale"


class MigrationStatus(Enum):
    """Statuts de migration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ConflictResolution(Enum):
    """Stratégies de résolution de conflits"""
    MERGE = "merge"
    OVERWRITE = "overwrite"
    SKIP = "skip"
    MANUAL = "manual"


@dataclass
class MigrationPlan:
    """Plan de migration"""
    id: str
    name: str
    description: str
    migration_type: MigrationType
    version_from: str
    version_to: str
    tenant_ids: List[str] = field(default_factory=list)
    locale_codes: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    rollback_plan: Optional[str] = None
    estimated_duration: Optional[int] = None  # en secondes
    risk_level: str = "medium"  # low, medium, high
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MigrationResult:
    """Résultat de migration"""
    migration_id: str
    status: MigrationStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    records_processed: int = 0
    records_migrated: int = 0
    records_failed: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rollback_info: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConflictInfo:
    """Information de conflit"""
    id: str
    type: str
    description: str
    source_value: Any
    target_value: Any
    suggested_resolution: ConflictResolution
    metadata: Dict[str, Any] = field(default_factory=dict)


class MigrationValidator(ABC):
    """Interface pour les validateurs de migration"""
    
    @abstractmethod
    async def validate_plan(self, plan: MigrationPlan) -> List[str]:
        """Valide un plan de migration"""
        pass
    
    @abstractmethod
    async def validate_data(self, data: Dict[str, Any]) -> List[str]:
        """Valide les données à migrer"""
        pass


class SchemaMigrationValidator(MigrationValidator):
    """Validateur pour les migrations de schéma"""
    
    async def validate_plan(self, plan: MigrationPlan) -> List[str]:
        """Valide un plan de migration de schéma"""
        errors = []
        
        try:
            if plan.migration_type != MigrationType.SCHEMA:
                errors.append("Migration type must be SCHEMA")
            
            if not plan.version_from or not plan.version_to:
                errors.append("Source and target versions must be specified")
            
            # Vérifier la compatibilité des versions
            if not self._is_version_compatible(plan.version_from, plan.version_to):
                errors.append(f"Version migration from {plan.version_from} to {plan.version_to} is not supported")
            
            return errors
            
        except Exception as e:
            logger.error(f"Schema migration validation error: {e}")
            return [f"Validation error: {e}"]
    
    async def validate_data(self, data: Dict[str, Any]) -> List[str]:
        """Valide les données de schéma"""
        errors = []
        
        try:
            required_fields = ['schema_version', 'tables', 'indexes']
            for field in required_fields:
                if field not in data:
                    errors.append(f"Missing required field: {field}")
            
            # Valider la structure des tables
            if 'tables' in data:
                for table_name, table_def in data['tables'].items():
                    if not isinstance(table_def, dict):
                        errors.append(f"Invalid table definition for {table_name}")
                        continue
                    
                    if 'columns' not in table_def:
                        errors.append(f"Missing columns definition for table {table_name}")
            
            return errors
            
        except Exception as e:
            logger.error(f"Schema data validation error: {e}")
            return [f"Data validation error: {e}"]
    
    def _is_version_compatible(self, from_version: str, to_version: str) -> bool:
        """Vérifie la compatibilité des versions"""
        try:
            # Logique simplifiée - dans la réalité, plus complexe
            from_parts = from_version.split('.')
            to_parts = to_version.split('.')
            
            # Permettre seulement les migrations vers des versions supérieures
            for i in range(min(len(from_parts), len(to_parts))):
                from_num = int(from_parts[i])
                to_num = int(to_parts[i])
                
                if to_num > from_num:
                    return True
                elif to_num < from_num:
                    return False
            
            return len(to_parts) >= len(from_parts)
            
        except Exception:
            return False


class DataMigrationValidator(MigrationValidator):
    """Validateur pour les migrations de données"""
    
    async def validate_plan(self, plan: MigrationPlan) -> List[str]:
        """Valide un plan de migration de données"""
        errors = []
        
        try:
            if plan.migration_type != MigrationType.DATA:
                errors.append("Migration type must be DATA")
            
            if not plan.tenant_ids and not plan.locale_codes:
                errors.append("Either tenant_ids or locale_codes must be specified for data migration")
            
            # Vérifier que les tenants/locales existent
            # (implémentation dépendante du système)
            
            return errors
            
        except Exception as e:
            logger.error(f"Data migration validation error: {e}")
            return [f"Validation error: {e}"]
    
    async def validate_data(self, data: Dict[str, Any]) -> List[str]:
        """Valide les données à migrer"""
        errors = []
        
        try:
            if not isinstance(data, dict):
                errors.append("Data must be a dictionary")
                return errors
            
            # Valider la structure des données de locale
            for locale_code, locale_data in data.items():
                if not isinstance(locale_data, dict):
                    errors.append(f"Invalid locale data for {locale_code}")
                    continue
                
                # Vérifier les champs requis
                required_fields = ['messages', 'metadata']
                for field in required_fields:
                    if field not in locale_data:
                        errors.append(f"Missing {field} in locale {locale_code}")
                
                # Valider les messages
                if 'messages' in locale_data:
                    messages = locale_data['messages']
                    if not isinstance(messages, dict):
                        errors.append(f"Messages must be a dictionary for locale {locale_code}")
                    else:
                        for key, value in messages.items():
                            if not isinstance(key, str):
                                errors.append(f"Message key must be string in locale {locale_code}")
                            if not isinstance(value, (str, dict)):
                                errors.append(f"Message value must be string or dict in locale {locale_code}")
            
            return errors
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return [f"Data validation error: {e}"]


class BackupManager:
    """Gestionnaire de sauvegardes pour les migrations"""
    
    def __init__(self, backup_dir: str):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    async def create_backup(
        self,
        migration_id: str,
        data: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> str:
        """Crée une sauvegarde"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_id = f"{migration_id}_{timestamp}"
            backup_path = self.backup_dir / f"{backup_id}.json"
            
            backup_data = {
                'migration_id': migration_id,
                'backup_id': backup_id,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {},
                'data': data
            }
            
            async with aiofiles.open(backup_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(backup_data, indent=2, ensure_ascii=False))
            
            logger.info(f"Backup created: {backup_id}")
            return backup_id
            
        except Exception as e:
            logger.error(f"Backup creation error: {e}")
            raise
    
    async def restore_backup(self, backup_id: str) -> Dict[str, Any]:
        """Restaure une sauvegarde"""
        try:
            backup_path = self.backup_dir / f"{backup_id}.json"
            
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup not found: {backup_id}")
            
            async with aiofiles.open(backup_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                backup_data = json.loads(content)
            
            logger.info(f"Backup restored: {backup_id}")
            return backup_data['data']
            
        except Exception as e:
            logger.error(f"Backup restoration error: {e}")
            raise
    
    async def list_backups(self, migration_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Liste les sauvegardes"""
        try:
            backups = []
            
            for backup_file in self.backup_dir.glob("*.json"):
                try:
                    async with aiofiles.open(backup_file, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        backup_data = json.loads(content)
                    
                    if migration_id and backup_data.get('migration_id') != migration_id:
                        continue
                    
                    backup_info = {
                        'backup_id': backup_data['backup_id'],
                        'migration_id': backup_data['migration_id'],
                        'timestamp': backup_data['timestamp'],
                        'size_bytes': backup_file.stat().st_size,
                        'metadata': backup_data.get('metadata', {})
                    }
                    
                    backups.append(backup_info)
                    
                except Exception as e:
                    logger.warning(f"Error reading backup file {backup_file}: {e}")
            
            return sorted(backups, key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            logger.error(f"Backup listing error: {e}")
            return []
    
    async def cleanup_old_backups(self, retention_days: int = 30):
        """Nettoie les sauvegardes anciennes"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            for backup_file in self.backup_dir.glob("*.json"):
                try:
                    # Vérifier la date de création
                    creation_time = datetime.fromtimestamp(backup_file.stat().st_ctime)
                    
                    if creation_time < cutoff_date:
                        backup_file.unlink()
                        logger.info(f"Deleted old backup: {backup_file.name}")
                        
                except Exception as e:
                    logger.warning(f"Error deleting backup {backup_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Backup cleanup error: {e}")


class ConflictResolver:
    """Résolveur de conflits pour les migrations"""
    
    def __init__(self):
        self._resolution_strategies = {
            ConflictResolution.MERGE: self._merge_strategy,
            ConflictResolution.OVERWRITE: self._overwrite_strategy,
            ConflictResolution.SKIP: self._skip_strategy
        }
    
    async def detect_conflicts(
        self,
        source_data: Dict[str, Any],
        target_data: Dict[str, Any]
    ) -> List[ConflictInfo]:
        """Détecte les conflits entre les données source et cible"""
        conflicts = []
        
        try:
            # Détecter les conflits de clés
            for key in source_data:
                if key in target_data:
                    source_value = source_data[key]
                    target_value = target_data[key]
                    
                    if source_value != target_value:
                        conflict = ConflictInfo(
                            id=f"key_conflict_{key}",
                            type="value_mismatch",
                            description=f"Different values for key '{key}'",
                            source_value=source_value,
                            target_value=target_value,
                            suggested_resolution=self._suggest_resolution(source_value, target_value)
                        )
                        conflicts.append(conflict)
            
            return conflicts
            
        except Exception as e:
            logger.error(f"Conflict detection error: {e}")
            return []
    
    async def resolve_conflicts(
        self,
        conflicts: List[ConflictInfo],
        resolution_strategy: ConflictResolution,
        custom_resolutions: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Résout les conflits selon la stratégie donnée"""
        try:
            resolved_data = {}
            
            for conflict in conflicts:
                if custom_resolutions and conflict.id in custom_resolutions:
                    # Résolution personnalisée
                    resolved_data[conflict.id] = custom_resolutions[conflict.id]
                else:
                    # Résolution automatique
                    strategy_func = self._resolution_strategies.get(resolution_strategy)
                    if strategy_func:
                        resolved_value = await strategy_func(conflict)
                        resolved_data[conflict.id] = resolved_value
                    else:
                        logger.warning(f"Unknown resolution strategy: {resolution_strategy}")
            
            return resolved_data
            
        except Exception as e:
            logger.error(f"Conflict resolution error: {e}")
            return {}
    
    def _suggest_resolution(self, source_value: Any, target_value: Any) -> ConflictResolution:
        """Suggère une stratégie de résolution"""
        try:
            # Logique de suggestion basée sur les types et valeurs
            if isinstance(source_value, dict) and isinstance(target_value, dict):
                return ConflictResolution.MERGE
            
            if isinstance(source_value, str) and isinstance(target_value, str):
                if len(source_value) > len(target_value):
                    return ConflictResolution.OVERWRITE
                else:
                    return ConflictResolution.SKIP
            
            # Par défaut, suggérer la fusion
            return ConflictResolution.MERGE
            
        except Exception:
            return ConflictResolution.MANUAL
    
    async def _merge_strategy(self, conflict: ConflictInfo) -> Any:
        """Stratégie de fusion"""
        try:
            source = conflict.source_value
            target = conflict.target_value
            
            if isinstance(source, dict) and isinstance(target, dict):
                # Fusion de dictionnaires
                merged = target.copy()
                merged.update(source)
                return merged
            
            if isinstance(source, list) and isinstance(target, list):
                # Fusion de listes (union)
                return list(set(target + source))
            
            # Pour les autres types, prendre la valeur source
            return source
            
        except Exception as e:
            logger.error(f"Merge strategy error: {e}")
            return conflict.source_value
    
    async def _overwrite_strategy(self, conflict: ConflictInfo) -> Any:
        """Stratégie d'écrasement"""
        return conflict.source_value
    
    async def _skip_strategy(self, conflict: ConflictInfo) -> Any:
        """Stratégie d'ignore"""
        return conflict.target_value


class MigrationEngine:
    """Moteur de migration principal"""
    
    def __init__(
        self,
        backup_manager: BackupManager,
        conflict_resolver: ConflictResolver,
        validators: Dict[MigrationType, MigrationValidator]
    ):
        self.backup_manager = backup_manager
        self.conflict_resolver = conflict_resolver
        self.validators = validators
        self._active_migrations = {}
        self._migration_history = []
        self._callbacks = {
            'before_migration': [],
            'after_migration': [],
            'on_conflict': [],
            'on_error': []
        }
    
    async def plan_migration(
        self,
        migration_type: MigrationType,
        source_data: Dict[str, Any],
        target_data: Dict[str, Any],
        plan: MigrationPlan
    ) -> Tuple[List[ConflictInfo], List[str]]:
        """Planifie une migration"""
        try:
            # Valider le plan
            validator = self.validators.get(migration_type)
            validation_errors = []
            
            if validator:
                validation_errors = await validator.validate_plan(plan)
                if validation_errors:
                    return [], validation_errors
                
                # Valider les données
                data_errors = await validator.validate_data(source_data)
                validation_errors.extend(data_errors)
            
            # Détecter les conflits
            conflicts = await self.conflict_resolver.detect_conflicts(source_data, target_data)
            
            return conflicts, validation_errors
            
        except Exception as e:
            logger.error(f"Migration planning error: {e}")
            return [], [f"Planning error: {e}"]
    
    async def execute_migration(
        self,
        plan: MigrationPlan,
        source_data: Dict[str, Any],
        target_data: Dict[str, Any],
        conflict_resolution: ConflictResolution = ConflictResolution.MERGE,
        custom_resolutions: Dict[str, Any] = None,
        dry_run: bool = False
    ) -> MigrationResult:
        """Exécute une migration"""
        migration_id = plan.id
        result = MigrationResult(
            migration_id=migration_id,
            status=MigrationStatus.RUNNING,
            started_at=datetime.now()
        )
        
        try:
            # Marquer comme migration active
            self._active_migrations[migration_id] = result
            
            # Callbacks avant migration
            await self._execute_callbacks('before_migration', plan, source_data)
            
            # Planifier la migration
            conflicts, validation_errors = await self.plan_migration(
                plan.migration_type,
                source_data,
                target_data,
                plan
            )
            
            if validation_errors:
                result.status = MigrationStatus.FAILED
                result.errors.extend(validation_errors)
                return result
            
            # Créer une sauvegarde
            if not dry_run:
                backup_id = await self.backup_manager.create_backup(
                    migration_id,
                    target_data,
                    {
                        'plan': plan.__dict__,
                        'conflicts_count': len(conflicts)
                    }
                )
                result.rollback_info = {'backup_id': backup_id}
            
            # Résoudre les conflits
            if conflicts:
                await self._execute_callbacks('on_conflict', conflicts)
                
                resolved_conflicts = await self.conflict_resolver.resolve_conflicts(
                    conflicts,
                    conflict_resolution,
                    custom_resolutions
                )
                
                result.metadata['resolved_conflicts'] = len(resolved_conflicts)
            
            # Effectuer la migration
            migrated_data = await self._perform_migration(
                plan,
                source_data,
                target_data,
                conflicts,
                dry_run
            )
            
            # Mettre à jour les statistiques
            result.records_processed = len(source_data)
            result.records_migrated = len(migrated_data)
            result.status = MigrationStatus.COMPLETED
            result.completed_at = datetime.now()
            result.duration_seconds = int(
                (result.completed_at - result.started_at).total_seconds()
            )
            
            # Callbacks après migration
            await self._execute_callbacks('after_migration', result, migrated_data)
            
            logger.info(f"Migration {migration_id} completed successfully")
            
        except Exception as e:
            result.status = MigrationStatus.FAILED
            result.errors.append(str(e))
            result.completed_at = datetime.now()
            
            await self._execute_callbacks('on_error', result, e)
            
            logger.error(f"Migration {migration_id} failed: {e}")
            
        finally:
            # Nettoyer les migrations actives
            if migration_id in self._active_migrations:
                del self._active_migrations[migration_id]
            
            # Ajouter à l'historique
            self._migration_history.append(result)
        
        return result
    
    async def rollback_migration(self, migration_id: str) -> MigrationResult:
        """Annule une migration"""
        try:
            # Trouver la migration dans l'historique
            migration_result = None
            for result in self._migration_history:
                if result.migration_id == migration_id:
                    migration_result = result
                    break
            
            if not migration_result:
                raise ValueError(f"Migration {migration_id} not found in history")
            
            if migration_result.status != MigrationStatus.COMPLETED:
                raise ValueError(f"Cannot rollback migration with status {migration_result.status}")
            
            if not migration_result.rollback_info or 'backup_id' not in migration_result.rollback_info:
                raise ValueError(f"No rollback information available for migration {migration_id}")
            
            # Restaurer la sauvegarde
            backup_id = migration_result.rollback_info['backup_id']
            backup_data = await self.backup_manager.restore_backup(backup_id)
            
            # Créer un résultat de rollback
            rollback_result = MigrationResult(
                migration_id=f"{migration_id}_rollback",
                status=MigrationStatus.COMPLETED,
                started_at=datetime.now(),
                completed_at=datetime.now(),
                records_processed=len(backup_data),
                records_migrated=len(backup_data),
                metadata={'original_migration': migration_id}
            )
            
            # Mettre à jour le statut de la migration originale
            migration_result.status = MigrationStatus.ROLLED_BACK
            
            self._migration_history.append(rollback_result)
            
            logger.info(f"Migration {migration_id} rolled back successfully")
            return rollback_result
            
        except Exception as e:
            logger.error(f"Migration rollback error: {e}")
            raise
    
    async def get_migration_status(self, migration_id: str) -> Optional[MigrationResult]:
        """Récupère le statut d'une migration"""
        # Vérifier les migrations actives
        if migration_id in self._active_migrations:
            return self._active_migrations[migration_id]
        
        # Vérifier l'historique
        for result in self._migration_history:
            if result.migration_id == migration_id:
                return result
        
        return None
    
    async def list_migrations(
        self,
        status_filter: Optional[MigrationStatus] = None,
        type_filter: Optional[MigrationType] = None
    ) -> List[MigrationResult]:
        """Liste les migrations"""
        try:
            results = self._migration_history.copy()
            
            # Ajouter les migrations actives
            results.extend(self._active_migrations.values())
            
            # Filtrer par statut
            if status_filter:
                results = [r for r in results if r.status == status_filter]
            
            # Filtrer par type (nécessite d'avoir le plan dans les métadonnées)
            if type_filter:
                filtered_results = []
                for result in results:
                    # Logique de filtrage basée sur les métadonnées
                    if 'migration_type' in result.metadata:
                        if result.metadata['migration_type'] == type_filter.value:
                            filtered_results.append(result)
                results = filtered_results
            
            # Trier par date de début (plus récent en premier)
            results.sort(key=lambda x: x.started_at, reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Migration listing error: {e}")
            return []
    
    def add_callback(self, event: str, callback: Callable):
        """Ajoute un callback d'événement"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def remove_callback(self, event: str, callback: Callable):
        """Retire un callback d'événement"""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)
    
    async def _perform_migration(
        self,
        plan: MigrationPlan,
        source_data: Dict[str, Any],
        target_data: Dict[str, Any],
        conflicts: List[ConflictInfo],
        dry_run: bool
    ) -> Dict[str, Any]:
        """Effectue la migration réelle"""
        try:
            if dry_run:
                # En mode dry-run, retourner les données source
                logger.info(f"Dry-run migration for plan {plan.id}")
                return source_data
            
            # Logique de migration spécifique au type
            if plan.migration_type == MigrationType.DATA:
                return await self._migrate_data(source_data, target_data, plan)
            elif plan.migration_type == MigrationType.SCHEMA:
                return await self._migrate_schema(source_data, target_data, plan)
            elif plan.migration_type == MigrationType.CONFIG:
                return await self._migrate_config(source_data, target_data, plan)
            elif plan.migration_type == MigrationType.TENANT:
                return await self._migrate_tenant(source_data, target_data, plan)
            elif plan.migration_type == MigrationType.LOCALE:
                return await self._migrate_locale(source_data, target_data, plan)
            else:
                raise ValueError(f"Unsupported migration type: {plan.migration_type}")
                
        except Exception as e:
            logger.error(f"Migration execution error: {e}")
            raise
    
    async def _migrate_data(
        self,
        source_data: Dict[str, Any],
        target_data: Dict[str, Any],
        plan: MigrationPlan
    ) -> Dict[str, Any]:
        """Migre les données de locale"""
        try:
            migrated_data = target_data.copy()
            
            # Fusionner les données source dans les données cible
            for locale_code, locale_data in source_data.items():
                if locale_code in migrated_data:
                    # Fusionner les messages
                    if 'messages' in locale_data and 'messages' in migrated_data[locale_code]:
                        migrated_data[locale_code]['messages'].update(locale_data['messages'])
                    
                    # Mettre à jour les métadonnées
                    if 'metadata' in locale_data:
                        if 'metadata' not in migrated_data[locale_code]:
                            migrated_data[locale_code]['metadata'] = {}
                        migrated_data[locale_code]['metadata'].update(locale_data['metadata'])
                else:
                    # Nouvelle locale
                    migrated_data[locale_code] = locale_data
            
            return migrated_data
            
        except Exception as e:
            logger.error(f"Data migration error: {e}")
            raise
    
    async def _migrate_schema(
        self,
        source_data: Dict[str, Any],
        target_data: Dict[str, Any],
        plan: MigrationPlan
    ) -> Dict[str, Any]:
        """Migre le schéma"""
        # Implémentation spécifique au schéma
        return source_data
    
    async def _migrate_config(
        self,
        source_data: Dict[str, Any],
        target_data: Dict[str, Any],
        plan: MigrationPlan
    ) -> Dict[str, Any]:
        """Migre la configuration"""
        # Implémentation spécifique à la configuration
        return source_data
    
    async def _migrate_tenant(
        self,
        source_data: Dict[str, Any],
        target_data: Dict[str, Any],
        plan: MigrationPlan
    ) -> Dict[str, Any]:
        """Migre les données de tenant"""
        # Implémentation spécifique au tenant
        return source_data
    
    async def _migrate_locale(
        self,
        source_data: Dict[str, Any],
        target_data: Dict[str, Any],
        plan: MigrationPlan
    ) -> Dict[str, Any]:
        """Migre les données de locale spécifiques"""
        # Implémentation spécifique aux locales
        return source_data
    
    async def _execute_callbacks(self, event: str, *args, **kwargs):
        """Exécute les callbacks pour un événement"""
        try:
            for callback in self._callbacks.get(event, []):
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(*args, **kwargs)
                    else:
                        callback(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Callback error for event {event}: {e}")
                    
        except Exception as e:
            logger.error(f"Callback execution error: {e}")


class LocaleMigrationManager:
    """Gestionnaire de migration spécialisé pour les locales"""
    
    def __init__(self, backup_dir: str = "/tmp/locale_backups"):
        self.backup_manager = BackupManager(backup_dir)
        self.conflict_resolver = ConflictResolver()
        
        # Initialiser les validateurs
        self.validators = {
            MigrationType.SCHEMA: SchemaMigrationValidator(),
            MigrationType.DATA: DataMigrationValidator(),
            MigrationType.CONFIG: DataMigrationValidator(),  # Réutiliser pour la config
            MigrationType.TENANT: DataMigrationValidator(),
            MigrationType.LOCALE: DataMigrationValidator()
        }
        
        self.migration_engine = MigrationEngine(
            self.backup_manager,
            self.conflict_resolver,
            self.validators
        )
        
        # Configurer les callbacks par défaut
        self._setup_default_callbacks()
    
    async def migrate_locale_data(
        self,
        tenant_id: str,
        locale_code: str,
        source_data: Dict[str, Any],
        target_data: Dict[str, Any],
        conflict_resolution: ConflictResolution = ConflictResolution.MERGE
    ) -> MigrationResult:
        """Migre les données d'une locale spécifique"""
        try:
            plan = MigrationPlan(
                id=f"locale_migration_{tenant_id}_{locale_code}_{int(datetime.now().timestamp())}",
                name=f"Migrate locale {locale_code} for tenant {tenant_id}",
                description=f"Migration of locale data for {locale_code} in tenant {tenant_id}",
                migration_type=MigrationType.LOCALE,
                version_from="current",
                version_to="new",
                tenant_ids=[tenant_id],
                locale_codes=[locale_code]
            )
            
            return await self.migration_engine.execute_migration(
                plan,
                source_data,
                target_data,
                conflict_resolution
            )
            
        except Exception as e:
            logger.error(f"Locale migration error: {e}")
            raise
    
    async def bulk_migrate_locales(
        self,
        tenant_id: str,
        locale_data_map: Dict[str, Dict[str, Any]],
        conflict_resolution: ConflictResolution = ConflictResolution.MERGE
    ) -> List[MigrationResult]:
        """Migre plusieurs locales en lot"""
        try:
            results = []
            
            for locale_code, data_pair in locale_data_map.items():
                source_data = data_pair.get('source', {})
                target_data = data_pair.get('target', {})
                
                result = await self.migrate_locale_data(
                    tenant_id,
                    locale_code,
                    source_data,
                    target_data,
                    conflict_resolution
                )
                
                results.append(result)
                
                # Délai entre les migrations pour éviter la surcharge
                await asyncio.sleep(0.1)
            
            return results
            
        except Exception as e:
            logger.error(f"Bulk locale migration error: {e}")
            raise
    
    async def migrate_tenant_locales(
        self,
        tenant_id: str,
        source_tenant_data: Dict[str, Any],
        target_tenant_data: Dict[str, Any] = None,
        conflict_resolution: ConflictResolution = ConflictResolution.MERGE
    ) -> MigrationResult:
        """Migre toutes les locales d'un tenant"""
        try:
            plan = MigrationPlan(
                id=f"tenant_locale_migration_{tenant_id}_{int(datetime.now().timestamp())}",
                name=f"Migrate all locales for tenant {tenant_id}",
                description=f"Migration of all locale data for tenant {tenant_id}",
                migration_type=MigrationType.TENANT,
                version_from="current",
                version_to="new",
                tenant_ids=[tenant_id]
            )
            
            return await self.migration_engine.execute_migration(
                plan,
                source_tenant_data,
                target_tenant_data or {},
                conflict_resolution
            )
            
        except Exception as e:
            logger.error(f"Tenant locale migration error: {e}")
            raise
    
    def _setup_default_callbacks(self):
        """Configure les callbacks par défaut"""
        
        def log_migration_start(plan, source_data):
            logger.info(f"Starting migration: {plan.name} (ID: {plan.id})")
        
        def log_migration_complete(result, migrated_data):
            logger.info(f"Migration completed: {result.migration_id} - Status: {result.status.value}")
        
        def log_conflicts(conflicts):
            if conflicts:
                logger.warning(f"Migration conflicts detected: {len(conflicts)} conflicts")
        
        def log_errors(result, error):
            logger.error(f"Migration error in {result.migration_id}: {error}")
        
        self.migration_engine.add_callback('before_migration', log_migration_start)
        self.migration_engine.add_callback('after_migration', log_migration_complete)
        self.migration_engine.add_callback('on_conflict', log_conflicts)
        self.migration_engine.add_callback('on_error', log_errors)
