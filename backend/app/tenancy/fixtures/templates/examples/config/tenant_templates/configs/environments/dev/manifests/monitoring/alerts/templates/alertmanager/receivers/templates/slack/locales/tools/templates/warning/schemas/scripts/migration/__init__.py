# -*- coding: utf-8 -*-
"""
Migration Tools - Spotify AI Agent Tenancy Scripts
==================================================

Module avancé de gestion des migrations pour l'architecture multi-tenant.
Fournit des outils industriels pour la migration de données, schémas et configurations
avec support de rollback, validation et monitoring en temps réel.

Key Features:
- Migration de schémas de base de données avec versioning
- Migration de données entre environnements
- Migration de configurations tenant
- Validation de cohérence post-migration
- Rollback automatique en cas d'échec
- Monitoring et alertes en temps réel
- Support multi-base de données
- Parallélisation et optimisation des performances

Author: Spotify AI Agent Team
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Type
from pathlib import Path

from .tenant_migrator import TenantMigrator
from .schema_migrator import SchemaMigrator
from .data_migrator import DataMigrator
from .config_migrator import ConfigMigrator
from .migration_validator import MigrationValidator
from .migration_monitor import MigrationMonitor
from .rollback_manager import RollbackManager

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION DU MODULE
# =============================================================================

MIGRATION_CONFIG = {
    "version": "1.0.0",
    "name": "migration_tools",
    "description": "Industrial grade migration tools for multi-tenant architecture",
    "author": "Spotify AI Agent Team",
    
    # Configuration des migrations
    "migration_settings": {
        "batch_size": 1000,
        "max_concurrent_migrations": 5,
        "timeout_seconds": 3600,
        "retry_attempts": 3,
        "retry_delay_seconds": 10,
        "validation_enabled": True,
        "rollback_enabled": True,
        "monitoring_enabled": True,
        "backup_enabled": True
    },
    
    # Configuration des bases de données
    "database_config": {
        "connection_pool_size": 10,
        "connection_timeout": 30,
        "query_timeout": 300,
        "isolation_level": "READ_COMMITTED",
        "autocommit": False
    },
    
    # Configuration du monitoring
    "monitoring_config": {
        "metrics_enabled": True,
        "alerts_enabled": True,
        "log_level": "INFO",
        "retention_days": 30,
        "export_format": "prometheus"
    },
    
    # Configuration des chemins
    "paths": {
        "migrations_dir": "migrations",
        "backups_dir": "backups",
        "logs_dir": "logs",
        "temp_dir": "temp"
    }
}

# Types de migration supportés
MIGRATION_TYPES = {
    "schema": {
        "class": "SchemaMigrator",
        "description": "Database schema migrations",
        "supports_rollback": True,
        "requires_validation": True
    },
    "data": {
        "class": "DataMigrator", 
        "description": "Data migrations between environments",
        "supports_rollback": True,
        "requires_validation": True
    },
    "config": {
        "class": "ConfigMigrator",
        "description": "Configuration migrations",
        "supports_rollback": True,
        "requires_validation": False
    },
    "tenant": {
        "class": "TenantMigrator",
        "description": "Complete tenant migrations",
        "supports_rollback": True,
        "requires_validation": True
    }
}

# Stratégies de migration
MIGRATION_STRATEGIES = {
    "sequential": {
        "description": "Execute migrations one by one",
        "parallel": False,
        "safety_level": "high"
    },
    "parallel": {
        "description": "Execute compatible migrations in parallel",
        "parallel": True,
        "safety_level": "medium"
    },
    "bulk": {
        "description": "Bulk migration with optimizations",
        "parallel": True,
        "safety_level": "low"
    }
}

# =============================================================================
# REGISTRY DES MIGRATEURS
# =============================================================================

class MigrationRegistry:
    """Registry centralisé pour tous les types de migrateurs."""
    
    def __init__(self):
        self._migrators: Dict[str, Type] = {}
        self._initialize_registry()
    
    def _initialize_registry(self):
        """Initialise le registry avec les migrateurs disponibles."""
        self._migrators = {
            "tenant": TenantMigrator,
            "schema": SchemaMigrator,
            "data": DataMigrator,
            "config": ConfigMigrator
        }
    
    def register_migrator(self, name: str, migrator_class: Type):
        """Enregistre un nouveau type de migrateur."""
        self._migrators[name] = migrator_class
        logger.info(f"Registered migrator: {name}")
    
    def get_migrator(self, name: str) -> Optional[Type]:
        """Récupère un migrateur par son nom."""
        return self._migrators.get(name)
    
    def list_migrators(self) -> List[str]:
        """Liste tous les migrateurs disponibles."""
        return list(self._migrators.keys())

# Instance globale du registry
migration_registry = MigrationRegistry()

# =============================================================================
# FACTORY POUR LES MIGRATIONS
# =============================================================================

class MigrationFactory:
    """Factory pour créer des instances de migrateurs."""
    
    @staticmethod
    def create_migrator(migration_type: str, **kwargs) -> Any:
        """
        Crée une instance de migrateur selon le type.
        
        Args:
            migration_type: Type de migration (schema, data, config, tenant)
            **kwargs: Arguments de configuration
            
        Returns:
            Instance du migrateur approprié
            
        Raises:
            ValueError: Si le type de migration n'est pas supporté
        """
        migrator_class = migration_registry.get_migrator(migration_type)
        
        if not migrator_class:
            raise ValueError(f"Unsupported migration type: {migration_type}")
        
        # Configuration par défaut
        config = MIGRATION_CONFIG.copy()
        config.update(kwargs.get('config', {}))
        
        return migrator_class(config=config, **kwargs)

# =============================================================================
# GESTIONNAIRE PRINCIPAL DES MIGRATIONS
# =============================================================================

class MigrationManager:
    """
    Gestionnaire principal pour orchestrer toutes les migrations.
    Fournit une interface unifiée pour l'exécution, le monitoring et le rollback.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or MIGRATION_CONFIG
        self.monitor = MigrationMonitor(self.config)
        self.validator = MigrationValidator(self.config)
        self.rollback_manager = RollbackManager(self.config)
        self.active_migrations: Dict[str, Any] = {}
        
        logger.info("MigrationManager initialized")
    
    async def execute_migration(
        self,
        migration_type: str,
        source: str,
        target: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Exécute une migration avec monitoring et validation.
        
        Args:
            migration_type: Type de migration
            source: Environnement source
            target: Environnement cible
            **kwargs: Options de migration
            
        Returns:
            Résultat de la migration avec métriques
        """
        migration_id = f"{migration_type}_{source}_to_{target}_{asyncio.get_event_loop().time()}"
        
        try:
            # Créer le migrateur
            migrator = MigrationFactory.create_migrator(
                migration_type,
                source=source,
                target=target,
                **kwargs
            )
            
            # Enregistrer la migration
            self.active_migrations[migration_id] = migrator
            
            # Démarrer le monitoring
            await self.monitor.start_monitoring(migration_id, migrator)
            
            # Exécuter la migration
            result = await migrator.execute()
            
            # Valider le résultat
            if self.config["migration_settings"]["validation_enabled"]:
                validation_result = await self.validator.validate_migration(
                    migration_id, result
                )
                result["validation"] = validation_result
            
            # Arrêter le monitoring
            await self.monitor.stop_monitoring(migration_id)
            
            # Nettoyer
            del self.active_migrations[migration_id]
            
            logger.info(f"Migration {migration_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Migration {migration_id} failed: {e}")
            
            # Rollback si activé
            if self.config["migration_settings"]["rollback_enabled"]:
                await self.rollback_migration(migration_id)
            
            # Nettoyer
            if migration_id in self.active_migrations:
                del self.active_migrations[migration_id]
            
            raise
    
    async def rollback_migration(self, migration_id: str) -> Dict[str, Any]:
        """
        Effectue un rollback d'une migration.
        
        Args:
            migration_id: ID de la migration à rollback
            
        Returns:
            Résultat du rollback
        """
        return await self.rollback_manager.rollback(migration_id)
    
    async def get_migration_status(self, migration_id: str) -> Dict[str, Any]:
        """
        Récupère le statut d'une migration.
        
        Args:
            migration_id: ID de la migration
            
        Returns:
            Statut de la migration
        """
        if migration_id in self.active_migrations:
            migrator = self.active_migrations[migration_id]
            return await migrator.get_status()
        
        return await self.monitor.get_migration_history(migration_id)
    
    def list_active_migrations(self) -> List[str]:
        """
        Liste les migrations actives.
        
        Returns:
            Liste des IDs de migrations actives
        """
        return list(self.active_migrations.keys())

# =============================================================================
# UTILITAIRES ET HELPERS
# =============================================================================

def validate_migration_config(config: Dict[str, Any]) -> bool:
    """
    Valide la configuration de migration.
    
    Args:
        config: Configuration à valider
        
    Returns:
        True si valide, False sinon
    """
    required_keys = ["migration_settings", "database_config", "paths"]
    
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required config key: {key}")
            return False
    
    return True

def get_migration_metrics() -> Dict[str, Any]:
    """
    Récupère les métriques globales des migrations.
    
    Returns:
        Dictionnaire des métriques
    """
    # Cette fonction serait normalement connectée à un système de métriques
    return {
        "total_migrations": 0,
        "successful_migrations": 0,
        "failed_migrations": 0,
        "average_duration": 0,
        "last_migration": None
    }

async def cleanup_old_migrations(retention_days: int = 30) -> int:
    """
    Nettoie les anciennes données de migration.
    
    Args:
        retention_days: Nombre de jours de rétention
        
    Returns:
        Nombre d'éléments nettoyés
    """
    # Implémentation du nettoyage
    cleaned_count = 0
    logger.info(f"Cleaned {cleaned_count} old migration records")
    return cleaned_count

# =============================================================================
# EXPORTS PUBLICS
# =============================================================================

__all__ = [
    # Classes principales
    "MigrationManager",
    "MigrationFactory",
    "MigrationRegistry",
    
    # Migrateurs spécialisés
    "TenantMigrator",
    "SchemaMigrator", 
    "DataMigrator",
    "ConfigMigrator",
    
    # Utilitaires
    "MigrationValidator",
    "MigrationMonitor",
    "RollbackManager",
    
    # Configuration
    "MIGRATION_CONFIG",
    "MIGRATION_TYPES",
    "MIGRATION_STRATEGIES",
    
    # Registry global
    "migration_registry",
    
    # Fonctions utilitaires
    "validate_migration_config",
    "get_migration_metrics",
    "cleanup_old_migrations"
]

# Initialisation du module
logger.info(f"Migration module initialized - Version {MIGRATION_CONFIG['version']}")
