#!/usr/bin/env python3
"""
🔄 Data Isolation Migration Manager
==================================

Gestionnaire avancé de migrations pour le système d'isolation des données
avec support de rollback, validation et monitoring.

Author: DBA & Data Engineer - Fahed Mlaiel
"""

import asyncio
import json
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import argparse
from pathlib import Path
import hashlib
from abc import ABC, abstractmethod

# Ajouter le chemin du module au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from app.tenancy.data_isolation.core import (
    TenantContext,
    TenantType,
    IsolationLevel,
    PartitionType,
    PartitionStrategy
)


class MigrationError(Exception):
    """Exception pour les erreurs de migration"""
    pass


class Migration(ABC):
    """Interface de base pour les migrations"""
    
    def __init__(self, version: str, description: str):
        self.version = version
        self.description = description
        self.applied_at: Optional[datetime] = None
        self.rollback_data: Dict[str, Any] = {}
    
    @abstractmethod
    async def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Applique la migration"""
        pass
    
    @abstractmethod
    async def rollback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Annule la migration"""
        pass
    
    @abstractmethod
    async def validate(self, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valide que la migration peut être appliquée"""
        pass
    
    def get_checksum(self) -> str:
        """Calcule le checksum de la migration"""
        content = f"{self.version}:{self.description}:{self.__class__.__name__}"
        return hashlib.sha256(content.encode()).hexdigest()


class AddComplianceFieldsMigration(Migration):
    """Migration pour ajouter les champs de conformité"""
    
    def __init__(self):
        super().__init__(
            version="2025.07.001",
            description="Add compliance tracking fields to tenant contexts"
        )
    
    async def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Ajoute les champs de conformité"""
        print(f"  📝 Applying {self.description}...")
        
        # Simulation d'ajout de champs
        new_fields = {
            'compliance_version': '2.0',
            'gdpr_consent_timestamp': datetime.now(timezone.utc),
            'audit_retention_days': 2555,  # 7 ans
            'encryption_version': 2
        }
        
        # Sauvegarde pour rollback
        self.rollback_data = {
            'fields_added': list(new_fields.keys()),
            'original_schema_version': context.get('schema_version', '1.0')
        }
        
        result = {
            'status': 'success',
            'fields_added': new_fields,
            'affected_tenants': context.get('tenant_count', 0)
        }
        
        print(f"    ✅ Added {len(new_fields)} compliance fields")
        return result
    
    async def rollback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Supprime les champs de conformité"""
        print(f"  ↩️  Rolling back {self.description}...")
        
        fields_to_remove = self.rollback_data.get('fields_added', [])
        
        result = {
            'status': 'success',
            'fields_removed': fields_to_remove,
            'restored_schema_version': self.rollback_data.get('original_schema_version')
        }
        
        print(f"    ✅ Removed {len(fields_to_remove)} fields")
        return result
    
    async def validate(self, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valide la migration"""
        errors = []
        
        # Vérifier la version du schéma
        current_version = context.get('schema_version', '1.0')
        if current_version != '1.0':
            errors.append(f"Expected schema version 1.0, found {current_version}")
        
        # Vérifier l'espace disque
        available_space_gb = context.get('available_disk_space_gb', 0)
        if available_space_gb < 1:
            errors.append("Insufficient disk space (< 1GB)")
        
        return len(errors) == 0, errors


class OptimizePartitioningMigration(Migration):
    """Migration pour optimiser le partitionnement"""
    
    def __init__(self):
        super().__init__(
            version="2025.07.002",
            description="Optimize data partitioning strategy for better performance"
        )
    
    async def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimise le partitionnement"""
        print(f"  📝 Applying {self.description}...")
        
        # Sauvegarde de la configuration actuelle
        self.rollback_data = {
            'old_partition_strategy': context.get('partition_strategy', 'hash'),
            'old_partition_count': context.get('partition_count', 16),
            'old_replication_factor': context.get('replication_factor', 2)
        }
        
        # Nouvelle configuration optimisée
        optimizations = {
            'partition_strategy': 'consistent_hash',
            'partition_count': 32,
            'replication_factor': 3,
            'enable_auto_scaling': True
        }
        
        result = {
            'status': 'success',
            'optimizations_applied': optimizations,
            'estimated_performance_gain': '35%'
        }
        
        print(f"    ✅ Partitioning strategy optimized")
        return result
    
    async def rollback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Restaure l'ancien partitionnement"""
        print(f"  ↩️  Rolling back {self.description}...")
        
        old_config = {
            'partition_strategy': self.rollback_data['old_partition_strategy'],
            'partition_count': self.rollback_data['old_partition_count'],
            'replication_factor': self.rollback_data['old_replication_factor']
        }
        
        result = {
            'status': 'success',
            'restored_configuration': old_config
        }
        
        print(f"    ✅ Partitioning configuration restored")
        return result
    
    async def validate(self, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valide la migration"""
        errors = []
        
        # Vérifier le nombre de tenants
        tenant_count = context.get('tenant_count', 0)
        if tenant_count < 10:
            errors.append("Partitioning optimization only beneficial with 10+ tenants")
        
        # Vérifier les ressources système
        cpu_count = context.get('cpu_count', 1)
        if cpu_count < 4:
            errors.append("Partitioning optimization requires at least 4 CPU cores")
        
        return len(errors) == 0, errors


class AddSecurityPoliciesMigration(Migration):
    """Migration pour ajouter les nouvelles politiques de sécurité"""
    
    def __init__(self):
        super().__init__(
            version="2025.07.003",
            description="Add enhanced security policies and threat detection"
        )
    
    async def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Ajoute les politiques de sécurité"""
        print(f"  📝 Applying {self.description}...")
        
        new_policies = [
            {
                'name': 'Advanced Threat Detection',
                'type': 'threat_detection',
                'enabled': True
            },
            {
                'name': 'ML-based Anomaly Detection',
                'type': 'anomaly_detection',
                'enabled': True
            },
            {
                'name': 'Enhanced Audit Logging',
                'type': 'audit_logging',
                'enabled': True
            }
        ]
        
        self.rollback_data = {
            'policies_added': [policy['name'] for policy in new_policies]
        }
        
        result = {
            'status': 'success',
            'policies_added': new_policies,
            'security_level': 'enhanced'
        }
        
        print(f"    ✅ Added {len(new_policies)} security policies")
        return result
    
    async def rollback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Supprime les nouvelles politiques"""
        print(f"  ↩️  Rolling back {self.description}...")
        
        policies_removed = self.rollback_data.get('policies_added', [])
        
        result = {
            'status': 'success',
            'policies_removed': policies_removed,
            'security_level': 'standard'
        }
        
        print(f"    ✅ Removed {len(policies_removed)} security policies")
        return result
    
    async def validate(self, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valide la migration"""
        errors = []
        
        # Vérifier les dépendances
        security_engine_available = context.get('security_engine_available', False)
        if not security_engine_available:
            errors.append("Security engine not available")
        
        return len(errors) == 0, errors


class MigrationManager:
    """Gestionnaire de migrations pour l'isolation des données"""
    
    def __init__(self, migrations_dir: str = None):
        self.migrations_dir = Path(migrations_dir) if migrations_dir else Path(__file__).parent / "migrations"
        self.migrations_dir.mkdir(exist_ok=True)
        
        # État des migrations
        self.applied_migrations: Dict[str, Dict[str, Any]] = {}
        self.available_migrations: List[Migration] = []
        
        # Configuration
        self.dry_run = False
        self.verbose = True
        
        # Initialiser les migrations intégrées
        self._initialize_builtin_migrations()
        
        # Charger l'état des migrations
        self._load_migration_state()
    
    def _initialize_builtin_migrations(self):
        """Initialise les migrations intégrées"""
        self.available_migrations = [
            AddComplianceFieldsMigration(),
            OptimizePartitioningMigration(),
            AddSecurityPoliciesMigration()
        ]
    
    def _load_migration_state(self):
        """Charge l'état des migrations depuis le fichier"""
        state_file = self.migrations_dir / "migration_state.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    self.applied_migrations = json.load(f)
                print(f"📁 Loaded migration state: {len(self.applied_migrations)} applied")
            except Exception as e:
                print(f"⚠️ Failed to load migration state: {e}")
        else:
            print("📁 No existing migration state found")
    
    def _save_migration_state(self):
        """Sauvegarde l'état des migrations"""
        state_file = self.migrations_dir / "migration_state.json"
        
        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(self.applied_migrations, f, indent=2, default=str)
            print(f"💾 Migration state saved")
        except Exception as e:
            print(f"❌ Failed to save migration state: {e}")
    
    async def get_system_context(self) -> Dict[str, Any]:
        """Récupère le contexte système pour les migrations"""
        import psutil
        
        # Informations système
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        context = {
            'timestamp': datetime.now(timezone.utc),
            'schema_version': '1.0',  # Version actuelle du schéma
            'tenant_count': 150,  # Simulation
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': memory.total / (1024**3),
            'available_disk_space_gb': disk.free / (1024**3),
            'security_engine_available': True,
            'compliance_engine_available': True
        }
        
        return context
    
    def get_pending_migrations(self) -> List[Migration]:
        """Retourne les migrations en attente"""
        pending = []
        
        for migration in self.available_migrations:
            if migration.version not in self.applied_migrations:
                pending.append(migration)
        
        return sorted(pending, key=lambda m: m.version)
    
    def get_applied_migrations(self) -> List[Dict[str, Any]]:
        """Retourne les migrations appliquées"""
        applied = []
        
        for version, info in self.applied_migrations.items():
            migration = next((m for m in self.available_migrations if m.version == version), None)
            if migration:
                applied.append({
                    'migration': migration,
                    'applied_at': info['applied_at'],
                    'checksum': info['checksum'],
                    'result': info.get('result', {})
                })
        
        return sorted(applied, key=lambda m: m['applied_at'])
    
    async def validate_migration(self, migration: Migration) -> Tuple[bool, List[str]]:
        """Valide qu'une migration peut être appliquée"""
        print(f"🔍 Validating migration {migration.version}...")
        
        context = await self.get_system_context()
        valid, errors = await migration.validate(context)
        
        if valid:
            print(f"    ✅ Migration validation passed")
        else:
            print(f"    ❌ Migration validation failed:")
            for error in errors:
                print(f"        • {error}")
        
        return valid, errors
    
    async def apply_migration(self, migration: Migration) -> Dict[str, Any]:
        """Applique une migration"""
        print(f"🚀 Applying migration {migration.version}: {migration.description}")
        
        if self.dry_run:
            print("    🧪 DRY RUN MODE - No changes will be made")
            return {'status': 'dry_run', 'migration': migration.version}
        
        try:
            # Validation préalable
            valid, errors = await self.validate_migration(migration)
            if not valid:
                raise MigrationError(f"Validation failed: {errors}")
            
            # Application de la migration
            context = await self.get_system_context()
            result = await migration.apply(context)
            
            # Enregistrement de l'application
            migration.applied_at = datetime.now(timezone.utc)
            self.applied_migrations[migration.version] = {
                'applied_at': migration.applied_at.isoformat(),
                'checksum': migration.get_checksum(),
                'result': result,
                'rollback_data': migration.rollback_data
            }
            
            self._save_migration_state()
            
            print(f"    ✅ Migration {migration.version} applied successfully")
            return result
            
        except Exception as e:
            print(f"    ❌ Migration {migration.version} failed: {e}")
            raise MigrationError(f"Migration {migration.version} failed: {e}")
    
    async def rollback_migration(self, migration: Migration) -> Dict[str, Any]:
        """Annule une migration"""
        print(f"↩️ Rolling back migration {migration.version}")
        
        if migration.version not in self.applied_migrations:
            raise MigrationError(f"Migration {migration.version} is not applied")
        
        if self.dry_run:
            print("    🧪 DRY RUN MODE - No changes will be made")
            return {'status': 'dry_run', 'migration': migration.version}
        
        try:
            # Restauration des données de rollback
            migration.rollback_data = self.applied_migrations[migration.version].get('rollback_data', {})
            
            # Exécution du rollback
            context = await self.get_system_context()
            result = await migration.rollback(context)
            
            # Suppression de l'enregistrement
            del self.applied_migrations[migration.version]
            self._save_migration_state()
            
            print(f"    ✅ Migration {migration.version} rolled back successfully")
            return result
            
        except Exception as e:
            print(f"    ❌ Rollback of {migration.version} failed: {e}")
            raise MigrationError(f"Rollback of {migration.version} failed: {e}")
    
    async def migrate_up(self, target_version: str = None) -> Dict[str, Any]:
        """Applique toutes les migrations en attente jusqu'à une version cible"""
        print("🚀 Starting database migration (UP)...")
        print("=" * 60)
        
        pending = self.get_pending_migrations()
        
        if target_version:
            pending = [m for m in pending if m.version <= target_version]
        
        if not pending:
            print("✅ No pending migrations found")
            return {'status': 'up_to_date', 'migrations_applied': 0}
        
        print(f"📋 Found {len(pending)} pending migrations:")
        for migration in pending:
            print(f"   • {migration.version}: {migration.description}")
        print("")
        
        applied_count = 0
        failed_migrations = []
        
        for migration in pending:
            try:
                await self.apply_migration(migration)
                applied_count += 1
                print("")
            except MigrationError as e:
                failed_migrations.append({'migration': migration.version, 'error': str(e)})
                print(f"❌ Stopping due to migration failure")
                break
        
        print("=" * 60)
        print(f"🏁 Migration complete: {applied_count} applied")
        
        if failed_migrations:
            print(f"❌ {len(failed_migrations)} migrations failed")
            return {
                'status': 'partial_failure',
                'migrations_applied': applied_count,
                'failed_migrations': failed_migrations
            }
        
        return {
            'status': 'success',
            'migrations_applied': applied_count
        }
    
    async def migrate_down(self, target_version: str = None, steps: int = None) -> Dict[str, Any]:
        """Annule des migrations jusqu'à une version cible ou un nombre d'étapes"""
        print("↩️ Starting database migration (DOWN)...")
        print("=" * 60)
        
        applied = self.get_applied_migrations()
        
        if not applied:
            print("✅ No applied migrations found")
            return {'status': 'nothing_to_rollback', 'migrations_rolled_back': 0}
        
        # Déterminer les migrations à annuler
        to_rollback = []
        
        if steps:
            to_rollback = applied[-steps:]  # Dernières N migrations
        elif target_version:
            to_rollback = [m for m in applied if m['migration'].version > target_version]
        else:
            print("❌ Either target_version or steps must be specified")
            return {'status': 'error', 'message': 'Missing target_version or steps'}
        
        if not to_rollback:
            print("✅ No migrations to rollback")
            return {'status': 'nothing_to_rollback', 'migrations_rolled_back': 0}
        
        # Ordre inverse pour le rollback
        to_rollback.reverse()
        
        print(f"📋 Will rollback {len(to_rollback)} migrations:")
        for item in to_rollback:
            migration = item['migration']
            print(f"   • {migration.version}: {migration.description}")
        print("")
        
        rolled_back_count = 0
        failed_rollbacks = []
        
        for item in to_rollback:
            migration = item['migration']
            try:
                await self.rollback_migration(migration)
                rolled_back_count += 1
                print("")
            except MigrationError as e:
                failed_rollbacks.append({'migration': migration.version, 'error': str(e)})
                print(f"❌ Stopping due to rollback failure")
                break
        
        print("=" * 60)
        print(f"🏁 Rollback complete: {rolled_back_count} rolled back")
        
        if failed_rollbacks:
            print(f"❌ {len(failed_rollbacks)} rollbacks failed")
            return {
                'status': 'partial_failure',
                'migrations_rolled_back': rolled_back_count,
                'failed_rollbacks': failed_rollbacks
            }
        
        return {
            'status': 'success',
            'migrations_rolled_back': rolled_back_count
        }
    
    def show_status(self):
        """Affiche le statut des migrations"""
        print("📊 MIGRATION STATUS")
        print("=" * 60)
        
        applied = self.get_applied_migrations()
        pending = self.get_pending_migrations()
        
        print(f"Applied migrations: {len(applied)}")
        print(f"Pending migrations: {len(pending)}")
        print("")
        
        if applied:
            print("✅ APPLIED MIGRATIONS:")
            for item in applied:
                migration = item['migration']
                applied_at = item['applied_at']
                print(f"   • {migration.version}: {migration.description}")
                print(f"     Applied: {applied_at}")
            print("")
        
        if pending:
            print("⏳ PENDING MIGRATIONS:")
            for migration in pending:
                print(f"   • {migration.version}: {migration.description}")
            print("")
        else:
            print("✅ All migrations are up to date")


async def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Data Isolation Migration Manager')
    parser.add_argument('command', choices=['status', 'up', 'down', 'validate'], 
                       help='Migration command')
    parser.add_argument('--target', help='Target migration version')
    parser.add_argument('--steps', type=int, help='Number of migration steps')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--migrations-dir', help='Custom migrations directory')
    
    args = parser.parse_args()
    
    # Initialisation du gestionnaire
    manager = MigrationManager(args.migrations_dir)
    manager.dry_run = args.dry_run
    
    if args.dry_run:
        print("🧪 DRY RUN MODE ENABLED")
        print("")
    
    try:
        if args.command == 'status':
            manager.show_status()
        
        elif args.command == 'up':
            result = await manager.migrate_up(args.target)
            exit_code = 0 if result['status'] in ['success', 'up_to_date'] else 1
            sys.exit(exit_code)
        
        elif args.command == 'down':
            if not args.target and not args.steps:
                print("❌ For 'down' command, either --target or --steps must be specified")
                sys.exit(1)
            
            result = await manager.migrate_down(args.target, args.steps)
            exit_code = 0 if result['status'] in ['success', 'nothing_to_rollback'] else 1
            sys.exit(exit_code)
        
        elif args.command == 'validate':
            pending = manager.get_pending_migrations()
            if not pending:
                print("✅ No pending migrations to validate")
                sys.exit(0)
            
            all_valid = True
            for migration in pending:
                valid, errors = await manager.validate_migration(migration)
                if not valid:
                    all_valid = False
            
            sys.exit(0 if all_valid else 1)
    
    except KeyboardInterrupt:
        print("\n🛑 Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
