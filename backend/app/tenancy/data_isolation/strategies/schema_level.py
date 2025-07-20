"""
🏗️ Schema Level Strategy - Isolation par Schéma
==============================================

Stratégie d'isolation au niveau schéma avec un schéma dédié par tenant
dans une base de données partagée.

Author: DBA & Data Engineer - Fahed Mlaiel
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass
from datetime import datetime, timezone
import re

from ..core.tenant_context import TenantContext
from ..core.isolation_engine import IsolationStrategy, EngineConfig
from ..managers.connection_manager import DatabaseConnection
from ..exceptions import DataIsolationError, TenantNotFoundError


@dataclass
class SchemaConfig:
    """Configuration pour l'isolation au niveau schéma"""
    schema_prefix: str = "tenant_"
    default_schema: str = "public"
    auto_create_schema: bool = True
    auto_migrate_tables: bool = True
    schema_search_path_isolation: bool = True
    enforce_schema_permissions: bool = True
    allow_cross_schema_queries: bool = False
    schema_naming_pattern: str = r"^tenant_[a-zA-Z0-9_-]+$"


class SchemaLevelStrategy(IsolationStrategy):
    """
    Stratégie d'isolation au niveau schéma
    
    Features:
    - Schéma séparé par tenant dans une DB partagée
    - Search path automatique par tenant
    - Permissions de schéma strictes
    - Création automatique des schémas
    - Migration automatique des tables
    - Query rewriting pour isolation
    - Performance monitoring per-schema
    """
    
    def __init__(self, schema_config: Optional[SchemaConfig] = None):
        self.schema_config = schema_config or SchemaConfig()
        self.logger = logging.getLogger("isolation.schema_level")
        
        # Schema management
        self.created_schemas: Set[str] = set()
        self.schema_tables: Dict[str, List[str]] = {}
        self.schema_permissions: Dict[str, List[str]] = {}
        
        # Connection management (shared across tenants)
        self.shared_connection: Optional[DatabaseConnection] = None
        
        # Query rewriting
        self.query_rewriter = SchemaQueryRewriter(self.schema_config)
        
        # Stats per schema
        self.schema_stats: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("Schema level isolation strategy initialized")
    
    async def initialize(self, config: EngineConfig):
        """Initialise la stratégie"""
        try:
            # Initialize shared connection
            await self._init_shared_connection()
            
            # Discover existing tenant schemas
            await self._discover_tenant_schemas()
            
            # Initialize query rewriter
            await self.query_rewriter.initialize()
            
            self.logger.info("Schema level strategy ready")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize schema level strategy: {e}")
            raise DataIsolationError(f"Schema initialization failed: {e}")
    
    async def _init_shared_connection(self):
        """Initialise la connexion partagée"""
        # This would use the main database connection
        self.shared_connection = DatabaseConnection(
            host="localhost",  # From config
            port=5432,
            database="spotify_ai_agent",
            username="postgres",
            password=""
        )
        
        await self.shared_connection.connect()
        self.logger.info("Shared database connection established")
    
    async def _discover_tenant_schemas(self):
        """Découvre les schémas tenant existants"""
        try:
            query = """
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name LIKE %s 
            AND schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
            """
            
            schemas = await self.shared_connection.fetch_all(
                query, 
                (f"{self.schema_config.schema_prefix}%",)
            )
            
            for schema_record in schemas:
                schema_name = schema_record["schema_name"]
                self.created_schemas.add(schema_name)
                
                # Discover tables in this schema
                await self._discover_schema_tables(schema_name)
                
                self.logger.debug(f"Discovered tenant schema: {schema_name}")
            
            self.logger.info(f"Discovered {len(self.created_schemas)} tenant schemas")
            
        except Exception as e:
            self.logger.error(f"Failed to discover tenant schemas: {e}")
    
    async def _discover_schema_tables(self, schema_name: str):
        """Découvre les tables d'un schéma"""
        try:
            query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = %s
            """
            
            tables = await self.shared_connection.fetch_all(query, (schema_name,))
            self.schema_tables[schema_name] = [
                table["table_name"] for table in tables
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to discover tables for schema {schema_name}: {e}")
    
    async def apply_isolation(
        self, 
        context: TenantContext, 
        target: Any,
        **kwargs
    ) -> Any:
        """Applique l'isolation au niveau schéma"""
        try:
            # Get tenant schema
            schema_name = self._get_tenant_schema_name(context.tenant_id)
            
            # Ensure schema exists
            await self._ensure_tenant_schema(context.tenant_id, schema_name)
            
            # Set search path for isolation
            await self._set_search_path(schema_name)
            
            # Rewrite query if necessary
            if isinstance(target, str):  # SQL query
                rewritten_query = await self.query_rewriter.rewrite_query(
                    target, schema_name, **kwargs
                )
                result = await self._execute_isolated_query(
                    rewritten_query, schema_name, **kwargs
                )
            else:
                # Handle ORM objects
                result = await self._handle_orm_target(
                    target, schema_name, **kwargs
                )
            
            # Update stats
            self._update_schema_stats(schema_name, "query_executed")
            
            return result
            
        except Exception as e:
            schema_name = self._get_tenant_schema_name(context.tenant_id)
            self.logger.error(f"Schema isolation failed for tenant {context.tenant_id}: {e}")
            self._update_schema_stats(schema_name, "query_failed")
            raise DataIsolationError(f"Schema isolation failed: {e}")
    
    def _get_tenant_schema_name(self, tenant_id: str) -> str:
        """Génère le nom du schéma pour un tenant"""
        return f"{self.schema_config.schema_prefix}{tenant_id}"
    
    async def _ensure_tenant_schema(self, tenant_id: str, schema_name: str):
        """S'assure que le schéma tenant existe"""
        if schema_name in self.created_schemas:
            return
        
        if not self.schema_config.auto_create_schema:
            raise TenantNotFoundError(f"Schema for tenant {tenant_id} does not exist")
        
        try:
            # Create schema
            create_schema_query = f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"'
            await self.shared_connection.execute(create_schema_query)
            
            # Set schema permissions
            await self._set_schema_permissions(schema_name)
            
            self.created_schemas.add(schema_name)
            self.logger.info(f"Created schema for tenant {tenant_id}: {schema_name}")
            
            # Initialize tables if auto-migrate is enabled
            if self.schema_config.auto_migrate_tables:
                await self._initialize_schema_tables(schema_name)
            
        except Exception as e:
            self.logger.error(f"Failed to create schema for tenant {tenant_id}: {e}")
            raise DataIsolationError(f"Schema creation failed: {e}")
    
    async def _set_schema_permissions(self, schema_name: str):
        """Définit les permissions du schéma"""
        if not self.schema_config.enforce_schema_permissions:
            return
        
        try:
            # Grant usage to the schema owner/role
            grant_query = f'GRANT USAGE ON SCHEMA "{schema_name}" TO current_user'
            await self.shared_connection.execute(grant_query)
            
            # Grant create on schema for table creation
            create_grant_query = f'GRANT CREATE ON SCHEMA "{schema_name}" TO current_user'
            await self.shared_connection.execute(create_grant_query)
            
            self.logger.debug(f"Set permissions for schema: {schema_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to set permissions for schema {schema_name}: {e}")
    
    async def _set_search_path(self, schema_name: str):
        """Définit le search path pour l'isolation"""
        if not self.schema_config.schema_search_path_isolation:
            return
        
        try:
            # Set search path to tenant schema first, then default
            search_path = f'"{schema_name}", {self.schema_config.default_schema}'
            set_path_query = f"SET search_path TO {search_path}"
            await self.shared_connection.execute(set_path_query)
            
            self.logger.debug(f"Set search path: {search_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to set search path for schema {schema_name}: {e}")
    
    async def _initialize_schema_tables(self, schema_name: str):
        """Initialise les tables dans le schéma tenant"""
        try:
            # Base tables for Spotify AI Agent (schema-qualified)
            table_queries = [
                # Users table
                f"""
                CREATE TABLE IF NOT EXISTS "{schema_name}".users (
                    id SERIAL PRIMARY KEY,
                    spotify_id VARCHAR(255) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    display_name VARCHAR(255),
                    profile_image_url TEXT,
                    subscription_type VARCHAR(50) DEFAULT 'free',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """,
                
                # Artists table
                f"""
                CREATE TABLE IF NOT EXISTS "{schema_name}".artists (
                    id SERIAL PRIMARY KEY,
                    spotify_id VARCHAR(255) UNIQUE NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    genres TEXT[],
                    popularity INTEGER DEFAULT 0,
                    followers_count INTEGER DEFAULT 0,
                    image_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """,
                
                # Tracks table
                f"""
                CREATE TABLE IF NOT EXISTS "{schema_name}".tracks (
                    id SERIAL PRIMARY KEY,
                    spotify_id VARCHAR(255) UNIQUE NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    artist_id INTEGER REFERENCES "{schema_name}".artists(id),
                    album_name VARCHAR(255),
                    duration_ms INTEGER,
                    popularity INTEGER DEFAULT 0,
                    preview_url TEXT,
                    analysis_data JSONB,
                    ai_recommendations JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """,
                
                # AI Sessions table
                f"""
                CREATE TABLE IF NOT EXISTS "{schema_name}".ai_sessions (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES "{schema_name}".users(id),
                    session_type VARCHAR(100) NOT NULL,
                    session_data JSONB,
                    results JSONB,
                    status VARCHAR(50) DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
                """,
                
                # Analytics table
                f"""
                CREATE TABLE IF NOT EXISTS "{schema_name}".analytics (
                    id SERIAL PRIMARY KEY,
                    entity_type VARCHAR(100) NOT NULL,
                    entity_id VARCHAR(255) NOT NULL,
                    event_type VARCHAR(100) NOT NULL,
                    event_data JSONB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed BOOLEAN DEFAULT FALSE
                )
                """,
                
                # Indexes
                f'CREATE INDEX IF NOT EXISTS idx_users_spotify_id ON "{schema_name}".users(spotify_id)',
                f'CREATE INDEX IF NOT EXISTS idx_artists_spotify_id ON "{schema_name}".artists(spotify_id)',
                f'CREATE INDEX IF NOT EXISTS idx_tracks_spotify_id ON "{schema_name}".tracks(spotify_id)',
                f'CREATE INDEX IF NOT EXISTS idx_tracks_artist_id ON "{schema_name}".tracks(artist_id)',
                f'CREATE INDEX IF NOT EXISTS idx_ai_sessions_user_id ON "{schema_name}".ai_sessions(user_id)',
                f'CREATE INDEX IF NOT EXISTS idx_analytics_entity ON "{schema_name}".analytics(entity_type, entity_id)',
                f'CREATE INDEX IF NOT EXISTS idx_analytics_timestamp ON "{schema_name}".analytics(timestamp)'
            ]
            
            for query in table_queries:
                await self.shared_connection.execute(query)
            
            self.schema_tables[schema_name] = [
                "users", "artists", "tracks", "ai_sessions", "analytics"
            ]
            
            self.logger.info(f"Initialized tables for schema {schema_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tables for schema {schema_name}: {e}")
            raise DataIsolationError(f"Table initialization failed: {e}")
    
    async def _execute_isolated_query(
        self, 
        query: str, 
        schema_name: str,
        **kwargs
    ) -> Any:
        """Exécute une requête isolée dans le schéma"""
        params = kwargs.get("params", [])
        query_type = kwargs.get("query_type", "select")
        
        # Ensure search path is set
        await self._set_search_path(schema_name)
        
        if query_type.lower() == "select":
            return await self.shared_connection.fetch_all(query, params)
        elif query_type.lower() in ["insert", "update", "delete"]:
            return await self.shared_connection.execute(query, params)
        else:
            return await self.shared_connection.fetch_one(query, params)
    
    async def _handle_orm_target(
        self, 
        target: Any, 
        schema_name: str,
        **kwargs
    ) -> Any:
        """Gère les cibles ORM avec isolation de schéma"""
        # This would handle ORM models by setting the schema
        # For SQLAlchemy, this might involve setting __table_args__
        # For Django, this might involve setting db_table
        self.logger.debug(f"Handling ORM target: {type(target).__name__} in schema: {schema_name}")
        return target
    
    def _update_schema_stats(self, schema_name: str, stat_type: str):
        """Met à jour les statistiques du schéma"""
        if schema_name not in self.schema_stats:
            self.schema_stats[schema_name] = {
                "queries_executed": 0,
                "queries_failed": 0,
                "last_activity": datetime.now(timezone.utc),
                "created_at": datetime.now(timezone.utc)
            }
        
        if stat_type == "query_executed":
            self.schema_stats[schema_name]["queries_executed"] += 1
        elif stat_type == "query_failed":
            self.schema_stats[schema_name]["queries_failed"] += 1
        
        self.schema_stats[schema_name]["last_activity"] = datetime.now(timezone.utc)
    
    async def validate_access(self, context: TenantContext, resource: str) -> bool:
        """Valide l'accès à une ressource"""
        try:
            schema_name = self._get_tenant_schema_name(context.tenant_id)
            await self._ensure_tenant_schema(context.tenant_id, schema_name)
            
            # Validate schema name pattern
            if not re.match(self.schema_config.schema_naming_pattern, schema_name):
                self.logger.warning(f"Invalid schema name pattern: {schema_name}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Access validation failed for tenant {context.tenant_id}: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Vérification de santé de la stratégie"""
        try:
            # Check shared connection
            if not self.shared_connection or not await self.shared_connection.is_healthy():
                self.logger.warning("Shared connection unhealthy")
                return False
            
            # Check if we can access schemas
            test_query = "SELECT schema_name FROM information_schema.schemata LIMIT 1"
            await self.shared_connection.fetch_one(test_query)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    async def get_schema_statistics(self, tenant_id: str) -> Dict[str, Any]:
        """Obtient les statistiques pour un schéma tenant"""
        schema_name = self._get_tenant_schema_name(tenant_id)
        stats = self.schema_stats.get(schema_name, {})
        
        if schema_name in self.created_schemas:
            try:
                # Get schema size
                size_query = """
                SELECT 
                    schemaname,
                    pg_size_pretty(SUM(pg_total_relation_size(schemaname||'.'||tablename))::bigint) as size,
                    SUM(pg_total_relation_size(schemaname||'.'||tablename))::bigint as size_bytes
                FROM pg_tables 
                WHERE schemaname = %s
                GROUP BY schemaname
                """
                size_result = await self.shared_connection.fetch_one(size_query, (schema_name,))
                
                # Get table counts
                table_counts = {}
                for table_name in self.schema_tables.get(schema_name, []):
                    count_query = f'SELECT COUNT(*) as count FROM "{schema_name}".{table_name}'
                    count_result = await self.shared_connection.fetch_one(count_query)
                    table_counts[table_name] = count_result["count"]
                
                if size_result:
                    stats.update({
                        "schema_size": size_result["size"],
                        "schema_size_bytes": size_result["size_bytes"],
                    })
                
                stats.update({
                    "table_counts": table_counts,
                    "schema_tables": self.schema_tables.get(schema_name, [])
                })
                
            except Exception as e:
                self.logger.error(f"Failed to get schema statistics for {schema_name}: {e}")
        
        return stats
    
    async def cleanup(self):
        """Nettoie les ressources de la stratégie"""
        self.logger.info("Cleaning up schema level strategy...")
        
        # Close shared connection
        if self.shared_connection:
            try:
                await self.shared_connection.disconnect()
                self.logger.debug("Disconnected shared connection")
            except Exception as e:
                self.logger.error(f"Error disconnecting shared connection: {e}")
        
        self.logger.info("Schema level strategy cleanup completed")
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de la stratégie"""
        return {
            "strategy_type": "schema_level",
            "total_tenant_schemas": len(self.created_schemas),
            "schema_stats": dict(self.schema_stats),
            "config": {
                "auto_create_schema": self.schema_config.auto_create_schema,
                "auto_migrate_tables": self.schema_config.auto_migrate_tables,
                "schema_search_path_isolation": self.schema_config.schema_search_path_isolation,
                "enforce_schema_permissions": self.schema_config.enforce_schema_permissions,
                "allow_cross_schema_queries": self.schema_config.allow_cross_schema_queries
            }
        }


class SchemaQueryRewriter:
    """Réécriture de requêtes pour l'isolation de schéma"""
    
    def __init__(self, config: SchemaConfig):
        self.config = config
        self.logger = logging.getLogger("isolation.query_rewriter")
        
        # Common table patterns to rewrite
        self.table_patterns = [
            r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
            r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
            r'\bINTO\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
            r'\bUPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
            r'\bINSERT\s+INTO\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
            r'\bDELETE\s+FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)\b'
        ]
    
    async def initialize(self):
        """Initialise le réécriture de requêtes"""
        self.logger.debug("Query rewriter initialized")
    
    async def rewrite_query(
        self, 
        query: str, 
        schema_name: str,
        **kwargs
    ) -> str:
        """Réécrit une requête pour l'isolation de schéma"""
        if not query.strip():
            return query
        
        # If query already contains schema-qualified tables, don't rewrite
        if f'"{schema_name}".' in query or f'{schema_name}.' in query:
            return query
        
        # If cross-schema queries are not allowed, we need to rewrite
        if not self.config.allow_cross_schema_queries:
            rewritten_query = self._qualify_table_names(query, schema_name)
            
            self.logger.debug(f"Rewritten query for schema {schema_name}")
            return rewritten_query
        
        return query
    
    def _qualify_table_names(self, query: str, schema_name: str) -> str:
        """Qualifie les noms de table avec le schéma"""
        import re
        
        rewritten = query
        
        for pattern in self.table_patterns:
            def replace_table(match):
                table_name = match.group(1)
                # Don't rewrite if already qualified or if it's a system table
                if '.' in table_name or table_name.startswith('pg_') or table_name.startswith('information_schema'):
                    return match.group(0)
                
                # Replace with schema-qualified name
                return match.group(0).replace(table_name, f'"{schema_name}".{table_name}')
            
            rewritten = re.sub(pattern, replace_table, rewritten, flags=re.IGNORECASE)
        
        return rewritten
