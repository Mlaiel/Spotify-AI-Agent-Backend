#!/usr/bin/env python3
"""
Script de migration et mise à jour du système d'analytics
Migration automatisée pour le système d'analytics d'alertes Spotify AI Agent
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import asyncpg
import aioredis
from pydantic import BaseModel

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MigrationConfig(BaseModel):
    """Configuration pour les migrations"""
    database_url: str
    redis_url: str
    backup_path: str = "/tmp/spotify_ai_agent_backup"
    migration_timeout: int = 3600  # 1 heure
    verify_integrity: bool = True
    rollback_on_failure: bool = True

class MigrationStep(BaseModel):
    """Étape de migration"""
    version: str
    name: str
    description: str
    sql_up: Optional[str] = None
    sql_down: Optional[str] = None
    python_up: Optional[str] = None
    python_down: Optional[str] = None
    requires: List[str] = []
    timestamp: datetime = datetime.now()

class DatabaseMigrator:
    """Gestionnaire de migrations de base de données"""
    
    def __init__(self, config: MigrationConfig):
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None
        
    async def connect(self):
        """Connexion à la base de données"""
        try:
            self.pool = await asyncpg.create_pool(self.config.database_url)
            logger.info("Connexion à la base de données établie")
        except Exception as e:
            logger.error(f"Erreur de connexion DB: {e}")
            raise
    
    async def close(self):
        """Fermeture des connexions"""
        if self.pool:
            await self.pool.close()
    
    async def ensure_migration_table(self):
        """Création de la table de migrations"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(500) NOT NULL,
                    description TEXT,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    execution_time_ms INTEGER,
                    checksum VARCHAR(64),
                    success BOOLEAN DEFAULT TRUE
                )
            """)
            logger.info("Table schema_migrations vérifiée")
    
    async def get_applied_migrations(self) -> List[str]:
        """Récupération des migrations appliquées"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT version FROM schema_migrations WHERE success = TRUE ORDER BY applied_at"
            )
            return [row['version'] for row in rows]
    
    async def apply_migration(self, migration: MigrationStep) -> bool:
        """Application d'une migration"""
        start_time = time.time()
        
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # Vérification des prérequis
                    applied = await self.get_applied_migrations()
                    for req in migration.requires:
                        if req not in applied:
                            raise ValueError(f"Migration prérequise manquante: {req}")
                    
                    # Exécution SQL
                    if migration.sql_up:
                        await conn.execute(migration.sql_up)
                        logger.info(f"SQL exécuté pour {migration.version}")
                    
                    # Exécution Python
                    if migration.python_up:
                        exec(migration.python_up)
                        logger.info(f"Code Python exécuté pour {migration.version}")
                    
                    # Enregistrement
                    execution_time = int((time.time() - start_time) * 1000)
                    await conn.execute("""
                        INSERT INTO schema_migrations 
                        (version, name, description, execution_time_ms, success)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (version) DO UPDATE SET
                        applied_at = CURRENT_TIMESTAMP,
                        execution_time_ms = $4,
                        success = $5
                    """, migration.version, migration.name, migration.description, 
                         execution_time, True)
                    
                    logger.info(f"Migration {migration.version} appliquée avec succès")
                    return True
                    
        except Exception as e:
            logger.error(f"Erreur lors de la migration {migration.version}: {e}")
            # Enregistrement de l'échec
            try:
                async with self.pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO schema_migrations 
                        (version, name, description, success)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (version) DO UPDATE SET
                        applied_at = CURRENT_TIMESTAMP,
                        success = $4
                    """, migration.version, migration.name, migration.description, False)
            except:
                pass
            return False
    
    async def rollback_migration(self, migration: MigrationStep) -> bool:
        """Rollback d'une migration"""
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # Exécution SQL de rollback
                    if migration.sql_down:
                        await conn.execute(migration.sql_down)
                        logger.info(f"Rollback SQL pour {migration.version}")
                    
                    # Exécution Python de rollback
                    if migration.python_down:
                        exec(migration.python_down)
                        logger.info(f"Rollback Python pour {migration.version}")
                    
                    # Suppression de l'enregistrement
                    await conn.execute(
                        "DELETE FROM schema_migrations WHERE version = $1",
                        migration.version
                    )
                    
                    logger.info(f"Rollback {migration.version} effectué")
                    return True
                    
        except Exception as e:
            logger.error(f"Erreur lors du rollback {migration.version}: {e}")
            return False

class RedisMigrator:
    """Gestionnaire de migrations Redis"""
    
    def __init__(self, config: MigrationConfig):
        self.config = config
        self.redis: Optional[aioredis.Redis] = None
    
    async def connect(self):
        """Connexion à Redis"""
        try:
            self.redis = await aioredis.from_url(self.config.redis_url)
            await self.redis.ping()
            logger.info("Connexion Redis établie")
        except Exception as e:
            logger.error(f"Erreur de connexion Redis: {e}")
            raise
    
    async def close(self):
        """Fermeture de la connexion Redis"""
        if self.redis:
            await self.redis.close()
    
    async def backup_keys(self, pattern: str = "*") -> Dict[str, Any]:
        """Sauvegarde des clés Redis"""
        backup = {}
        try:
            keys = await self.redis.keys(pattern)
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                key_type = await self.redis.type(key)
                
                if key_type == b'string':
                    backup[key_str] = await self.redis.get(key)
                elif key_type == b'hash':
                    backup[key_str] = await self.redis.hgetall(key)
                elif key_type == b'list':
                    backup[key_str] = await self.redis.lrange(key, 0, -1)
                elif key_type == b'set':
                    backup[key_str] = list(await self.redis.smembers(key))
                elif key_type == b'zset':
                    backup[key_str] = await self.redis.zrange(key, 0, -1, withscores=True)
                
            logger.info(f"Sauvegarde Redis: {len(backup)} clés")
            return backup
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde Redis: {e}")
            return {}
    
    async def restore_keys(self, backup: Dict[str, Any]):
        """Restauration des clés Redis"""
        try:
            for key, value in backup.items():
                # Détermination du type et restauration
                if isinstance(value, (str, bytes)):
                    await self.redis.set(key, value)
                elif isinstance(value, dict):
                    await self.redis.hset(key, mapping=value)
                elif isinstance(value, list):
                    if all(isinstance(item, tuple) and len(item) == 2 for item in value):
                        # ZSET
                        for item, score in value:
                            await self.redis.zadd(key, {item: score})
                    else:
                        # LIST
                        await self.redis.lpush(key, *value)
                        
            logger.info(f"Restauration Redis: {len(backup)} clés")
            
        except Exception as e:
            logger.error(f"Erreur lors de la restauration Redis: {e}")
            raise

class BackupManager:
    """Gestionnaire de sauvegardes"""
    
    def __init__(self, config: MigrationConfig):
        self.config = config
        self.backup_dir = Path(config.backup_path)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    async def create_backup(self, db_migrator: DatabaseMigrator, redis_migrator: RedisMigrator) -> str:
        """Création d'une sauvegarde complète"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}"
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        try:
            # Sauvegarde PostgreSQL
            db_backup_file = backup_path / "database.sql"
            os.system(f"pg_dump {self.config.database_url} > {db_backup_file}")
            logger.info(f"Sauvegarde DB: {db_backup_file}")
            
            # Sauvegarde Redis
            redis_backup = await redis_migrator.backup_keys()
            redis_backup_file = backup_path / "redis.json"
            with open(redis_backup_file, 'w') as f:
                json.dump(redis_backup, f, default=str, indent=2)
            logger.info(f"Sauvegarde Redis: {redis_backup_file}")
            
            # Métadonnées
            metadata = {
                "timestamp": timestamp,
                "database_url": self.config.database_url,
                "redis_url": self.config.redis_url,
                "creation_time": datetime.now().isoformat()
            }
            
            metadata_file = backup_path / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Sauvegarde créée: {backup_name}")
            return backup_name
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
            raise
    
    async def restore_backup(self, backup_name: str, redis_migrator: RedisMigrator):
        """Restauration d'une sauvegarde"""
        backup_path = self.backup_dir / backup_name
        
        if not backup_path.exists():
            raise ValueError(f"Sauvegarde introuvable: {backup_name}")
        
        try:
            # Restauration PostgreSQL
            db_backup_file = backup_path / "database.sql"
            if db_backup_file.exists():
                os.system(f"psql {self.config.database_url} < {db_backup_file}")
                logger.info("Base de données restaurée")
            
            # Restauration Redis
            redis_backup_file = backup_path / "redis.json"
            if redis_backup_file.exists():
                with open(redis_backup_file, 'r') as f:
                    redis_backup = json.load(f)
                await redis_migrator.restore_keys(redis_backup)
                logger.info("Redis restauré")
            
            logger.info(f"Restauration terminée: {backup_name}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la restauration: {e}")
            raise

class MigrationManager:
    """Gestionnaire principal des migrations"""
    
    def __init__(self, config: MigrationConfig):
        self.config = config
        self.db_migrator = DatabaseMigrator(config)
        self.redis_migrator = RedisMigrator(config)
        self.backup_manager = BackupManager(config)
        
        # Migrations définies
        self.migrations = self._load_migrations()
    
    def _load_migrations(self) -> List[MigrationStep]:
        """Chargement des migrations"""
        return [
            MigrationStep(
                version="001_initial_schema",
                name="Schéma initial",
                description="Création des tables de base pour le système d'analytics",
                sql_up="""
                    -- Table des événements d'alerte
                    CREATE TABLE IF NOT EXISTS alert_events (
                        id SERIAL PRIMARY KEY,
                        event_id VARCHAR(255) UNIQUE NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        source VARCHAR(255) NOT NULL,
                        severity VARCHAR(50) NOT NULL,
                        message TEXT,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    -- Index pour les performances
                    CREATE INDEX IF NOT EXISTS idx_alert_events_timestamp ON alert_events(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_alert_events_source ON alert_events(source);
                    CREATE INDEX IF NOT EXISTS idx_alert_events_severity ON alert_events(severity);
                    
                    -- Table des métriques d'analyse
                    CREATE TABLE IF NOT EXISTS alert_metrics (
                        id SERIAL PRIMARY KEY,
                        event_id VARCHAR(255) REFERENCES alert_events(event_id),
                        metric_type VARCHAR(100) NOT NULL,
                        metric_value FLOAT NOT NULL,
                        calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    -- Table des corrélations
                    CREATE TABLE IF NOT EXISTS alert_correlations (
                        id SERIAL PRIMARY KEY,
                        source_event_id VARCHAR(255),
                        target_event_id VARCHAR(255),
                        correlation_type VARCHAR(100) NOT NULL,
                        correlation_score FLOAT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """,
                sql_down="""
                    DROP TABLE IF EXISTS alert_correlations;
                    DROP TABLE IF EXISTS alert_metrics;
                    DROP TABLE IF EXISTS alert_events;
                """
            ),
            
            MigrationStep(
                version="002_ml_models",
                name="Tables ML",
                description="Ajout des tables pour le machine learning",
                sql_up="""
                    -- Table des modèles ML
                    CREATE TABLE IF NOT EXISTS ml_models (
                        id SERIAL PRIMARY KEY,
                        model_name VARCHAR(255) UNIQUE NOT NULL,
                        model_type VARCHAR(100) NOT NULL,
                        model_version VARCHAR(50) NOT NULL,
                        model_data BYTEA,
                        parameters JSONB,
                        performance_metrics JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    -- Table des prédictions
                    CREATE TABLE IF NOT EXISTS ml_predictions (
                        id SERIAL PRIMARY KEY,
                        model_id INTEGER REFERENCES ml_models(id),
                        event_id VARCHAR(255),
                        prediction_type VARCHAR(100) NOT NULL,
                        prediction_value FLOAT NOT NULL,
                        confidence_score FLOAT,
                        predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    -- Index pour les performances
                    CREATE INDEX IF NOT EXISTS idx_ml_predictions_event_id ON ml_predictions(event_id);
                    CREATE INDEX IF NOT EXISTS idx_ml_predictions_predicted_at ON ml_predictions(predicted_at);
                """,
                sql_down="""
                    DROP TABLE IF EXISTS ml_predictions;
                    DROP TABLE IF EXISTS ml_models;
                """,
                requires=["001_initial_schema"]
            ),
            
            MigrationStep(
                version="003_performance_optimization",
                name="Optimisations performance",
                description="Ajout d'index et optimisations pour les performances",
                sql_up="""
                    -- Partitioning par date pour alert_events
                    CREATE TABLE IF NOT EXISTS alert_events_partitioned (
                        LIKE alert_events INCLUDING ALL
                    ) PARTITION BY RANGE (timestamp);
                    
                    -- Partition pour le mois courant
                    CREATE TABLE IF NOT EXISTS alert_events_current PARTITION OF alert_events_partitioned
                    FOR VALUES FROM (DATE_TRUNC('month', CURRENT_DATE)) 
                    TO (DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month');
                    
                    -- Index composites pour les requêtes fréquentes
                    CREATE INDEX IF NOT EXISTS idx_alert_events_timestamp_severity 
                    ON alert_events(timestamp, severity);
                    
                    CREATE INDEX IF NOT EXISTS idx_alert_events_source_timestamp 
                    ON alert_events(source, timestamp);
                    
                    -- Index GIN pour les métadonnées JSONB
                    CREATE INDEX IF NOT EXISTS idx_alert_events_metadata_gin 
                    ON alert_events USING GIN(metadata);
                    
                    -- Matérialized view pour les statistiques rapides
                    CREATE MATERIALIZED VIEW IF NOT EXISTS alert_stats_hourly AS
                    SELECT 
                        DATE_TRUNC('hour', timestamp) as hour,
                        source,
                        severity,
                        COUNT(*) as event_count,
                        AVG(CASE WHEN metadata->>'response_time' IS NOT NULL 
                            THEN (metadata->>'response_time')::float END) as avg_response_time
                    FROM alert_events
                    WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days'
                    GROUP BY DATE_TRUNC('hour', timestamp), source, severity;
                    
                    -- Index sur la vue matérialisée
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_alert_stats_hourly_unique
                    ON alert_stats_hourly(hour, source, severity);
                """,
                sql_down="""
                    DROP MATERIALIZED VIEW IF EXISTS alert_stats_hourly;
                    DROP INDEX IF EXISTS idx_alert_events_metadata_gin;
                    DROP INDEX IF EXISTS idx_alert_events_source_timestamp;
                    DROP INDEX IF EXISTS idx_alert_events_timestamp_severity;
                    DROP TABLE IF EXISTS alert_events_current;
                    DROP TABLE IF EXISTS alert_events_partitioned;
                """,
                requires=["001_initial_schema", "002_ml_models"]
            ),
            
            MigrationStep(
                version="004_redis_optimization",
                name="Optimisation Redis",
                description="Configuration et optimisation de Redis",
                python_up="""
import asyncio
import aioredis

async def optimize_redis():
    redis = await aioredis.from_url('{redis_url}')
    
    # Configuration optimale pour analytics
    await redis.config_set('maxmemory-policy', 'allkeys-lru')
    await redis.config_set('maxmemory-samples', '10')
    await redis.config_set('timeout', '300')
    
    # Structure pour les caches
    await redis.hset('analytics:config', mapping={
        'cache_ttl': '3600',
        'max_cache_size': '1000000',
        'compression_enabled': 'true'
    })
    
    # Initialisation des structures de données
    await redis.zadd('analytics:events:timeline', {})
    await redis.sadd('analytics:sources', 'system')
    
    await redis.close()
    print("Redis optimisé pour analytics")

asyncio.run(optimize_redis())
""".format(redis_url=config.redis_url if 'config' in locals() else "redis://localhost:6379"),
                requires=["001_initial_schema"]
            )
        ]
    
    async def connect(self):
        """Connexion aux services"""
        await self.db_migrator.connect()
        await self.redis_migrator.connect()
        await self.db_migrator.ensure_migration_table()
    
    async def close(self):
        """Fermeture des connexions"""
        await self.db_migrator.close()
        await self.redis_migrator.close()
    
    async def run_migrations(self, target_version: Optional[str] = None) -> bool:
        """Exécution des migrations"""
        try:
            # Sauvegarde avant migration
            if self.config.verify_integrity:
                backup_name = await self.backup_manager.create_backup(
                    self.db_migrator, self.redis_migrator
                )
                logger.info(f"Sauvegarde créée: {backup_name}")
            
            # Récupération des migrations appliquées
            applied = await self.db_migrator.get_applied_migrations()
            pending = [m for m in self.migrations if m.version not in applied]
            
            if target_version:
                pending = [m for m in pending if m.version <= target_version]
            
            if not pending:
                logger.info("Aucune migration en attente")
                return True
            
            logger.info(f"Migrations à appliquer: {[m.version for m in pending]}")
            
            # Application des migrations
            success = True
            for migration in pending:
                logger.info(f"Application de {migration.version}: {migration.name}")
                
                if not await self.db_migrator.apply_migration(migration):
                    success = False
                    if self.config.rollback_on_failure:
                        logger.warning("Rollback des migrations appliquées...")
                        await self._rollback_migrations(applied)
                    break
            
            if success:
                logger.info("Toutes les migrations appliquées avec succès")
            
            return success
            
        except Exception as e:
            logger.error(f"Erreur lors des migrations: {e}")
            return False
    
    async def _rollback_migrations(self, target_migrations: List[str]):
        """Rollback des migrations"""
        applied = await self.db_migrator.get_applied_migrations()
        to_rollback = [m for m in reversed(self.migrations) 
                      if m.version in applied and m.version not in target_migrations]
        
        for migration in to_rollback:
            logger.info(f"Rollback de {migration.version}")
            await self.db_migrator.rollback_migration(migration)
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Statut des migrations"""
        applied = await self.db_migrator.get_applied_migrations()
        pending = [m.version for m in self.migrations if m.version not in applied]
        
        return {
            "total_migrations": len(self.migrations),
            "applied_count": len(applied),
            "pending_count": len(pending),
            "applied_migrations": applied,
            "pending_migrations": pending,
            "last_applied": applied[-1] if applied else None
        }

async def main():
    """Point d'entrée principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migration Spotify AI Agent Analytics")
    parser.add_argument("command", choices=["migrate", "rollback", "status", "backup", "restore"])
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://localhost/spotify_ai_agent"))
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379"))
    parser.add_argument("--target-version", help="Version cible pour la migration")
    parser.add_argument("--backup-name", help="Nom de la sauvegarde pour restore")
    parser.add_argument("--backup-path", default="/tmp/spotify_ai_agent_backup")
    parser.add_argument("--no-backup", action="store_true", help="Pas de sauvegarde avant migration")
    
    args = parser.parse_args()
    
    config = MigrationConfig(
        database_url=args.database_url,
        redis_url=args.redis_url,
        backup_path=args.backup_path,
        verify_integrity=not args.no_backup
    )
    
    manager = MigrationManager(config)
    
    try:
        await manager.connect()
        
        if args.command == "migrate":
            success = await manager.run_migrations(args.target_version)
            sys.exit(0 if success else 1)
            
        elif args.command == "status":
            status = await manager.get_migration_status()
            print(json.dumps(status, indent=2))
            
        elif args.command == "backup":
            backup_name = await manager.backup_manager.create_backup(
                manager.db_migrator, manager.redis_migrator
            )
            print(f"Sauvegarde créée: {backup_name}")
            
        elif args.command == "restore":
            if not args.backup_name:
                print("--backup-name requis pour restore")
                sys.exit(1)
            await manager.backup_manager.restore_backup(
                args.backup_name, manager.redis_migrator
            )
            print(f"Restauration terminée: {args.backup_name}")
            
        elif args.command == "rollback":
            if not args.target_version:
                print("--target-version requis pour rollback")
                sys.exit(1)
            await manager._rollback_migrations([args.target_version])
            print(f"Rollback vers {args.target_version} terminé")
            
    except Exception as e:
        logger.error(f"Erreur: {e}")
        sys.exit(1)
    finally:
        await manager.close()

if __name__ == "__main__":
    asyncio.run(main())
