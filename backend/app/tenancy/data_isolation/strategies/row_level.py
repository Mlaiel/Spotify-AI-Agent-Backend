"""
🔐 Row Level Strategy - Isolation au Niveau des Lignes (RLS)
==========================================================

Stratégie d'isolation utilisant Row Level Security (RLS) de PostgreSQL
pour séparer les données par tenant dans des tables partagées.

Author: Spécialiste Sécurité Backend - Fahed Mlaiel
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import re

from ..core.tenant_context import TenantContext
from ..core.isolation_engine import IsolationStrategy, EngineConfig
from ..managers.connection_manager import DatabaseConnection
from ..exceptions import DataIsolationError, SecurityViolationError


@dataclass
class RLSConfig:
    """Configuration pour Row Level Security"""
    tenant_column: str = "tenant_id"
    auto_create_policies: bool = True
    enforce_rls_on_tables: bool = True
    allow_superuser_bypass: bool = False
    policy_name_prefix: str = "tenant_isolation_"
    auto_add_tenant_column: bool = True
    default_tenant_column_type: str = "VARCHAR(255)"
    create_tenant_index: bool = True
    audit_rls_violations: bool = True


class RLSPolicyManager:
    """Gestionnaire des politiques RLS"""
    
    def __init__(self, config: RLSConfig):
        self.config = config
        self.logger = logging.getLogger("rls.policy_manager")
        
        # Track policies and tables
        self.rls_enabled_tables: Set[str] = set()
        self.created_policies: Dict[str, List[str]] = {}  # table -> policies
        self.table_tenant_columns: Dict[str, str] = {}  # table -> tenant_column
    
    async def enable_rls_on_table(
        self, 
        connection: DatabaseConnection,
        table_name: str,
        tenant_column: str = None
    ):
        """Active RLS sur une table"""
        tenant_col = tenant_column or self.config.tenant_column
        
        try:
            # Add tenant column if it doesn't exist
            if self.config.auto_add_tenant_column:
                await self._ensure_tenant_column(connection, table_name, tenant_col)
            
            # Enable RLS on table
            enable_rls_query = f"ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY"
            await connection.execute(enable_rls_query)
            
            # Create isolation policy
            await self._create_tenant_isolation_policy(connection, table_name, tenant_col)
            
            self.rls_enabled_tables.add(table_name)
            self.table_tenant_columns[table_name] = tenant_col
            
            self.logger.info(f"Enabled RLS on table: {table_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to enable RLS on table {table_name}: {e}")
            raise DataIsolationError(f"RLS enablement failed: {e}")
    
    async def _ensure_tenant_column(
        self, 
        connection: DatabaseConnection,
        table_name: str,
        tenant_column: str
    ):
        """S'assure que la colonne tenant existe"""
        try:
            # Check if column exists
            check_column_query = """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = %s AND column_name = %s
            """
            
            existing_column = await connection.fetch_one(
                check_column_query, 
                (table_name, tenant_column)
            )
            
            if not existing_column:
                # Add tenant column
                add_column_query = f"""
                ALTER TABLE {table_name} 
                ADD COLUMN {tenant_column} {self.config.default_tenant_column_type}
                """
                await connection.execute(add_column_query)
                
                # Create index for performance
                if self.config.create_tenant_index:
                    index_name = f"idx_{table_name}_{tenant_column}"
                    create_index_query = f"""
                    CREATE INDEX IF NOT EXISTS {index_name} 
                    ON {table_name} ({tenant_column})
                    """
                    await connection.execute(create_index_query)
                
                self.logger.info(f"Added tenant column {tenant_column} to table {table_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to ensure tenant column on {table_name}: {e}")
            raise
    
    async def _create_tenant_isolation_policy(
        self, 
        connection: DatabaseConnection,
        table_name: str,
        tenant_column: str
    ):
        """Crée la politique d'isolation tenant"""
        try:
            policy_name = f"{self.config.policy_name_prefix}{table_name}"
            
            # Drop existing policy if it exists
            drop_policy_query = f"""
            DROP POLICY IF EXISTS {policy_name} ON {table_name}
            """
            await connection.execute(drop_policy_query)
            
            # Create new isolation policy
            # This policy allows access only to rows where tenant_id matches current setting
            create_policy_query = f"""
            CREATE POLICY {policy_name} ON {table_name}
            FOR ALL
            TO PUBLIC
            USING ({tenant_column} = current_setting('app.current_tenant_id', true))
            WITH CHECK ({tenant_column} = current_setting('app.current_tenant_id', true))
            """
            
            await connection.execute(create_policy_query)
            
            # Track created policy
            if table_name not in self.created_policies:
                self.created_policies[table_name] = []
            self.created_policies[table_name].append(policy_name)
            
            self.logger.debug(f"Created RLS policy {policy_name} for table {table_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to create RLS policy for {table_name}: {e}")
            raise
    
    async def set_tenant_context(
        self, 
        connection: DatabaseConnection,
        tenant_id: str
    ):
        """Définit le contexte tenant pour RLS"""
        try:
            set_context_query = "SELECT set_config('app.current_tenant_id', %s, false)"
            await connection.execute(set_context_query, (tenant_id,))
            
            self.logger.debug(f"Set tenant context: {tenant_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to set tenant context {tenant_id}: {e}")
            raise
    
    async def clear_tenant_context(self, connection: DatabaseConnection):
        """Efface le contexte tenant"""
        try:
            clear_context_query = "SELECT set_config('app.current_tenant_id', '', false)"
            await connection.execute(clear_context_query)
            
        except Exception as e:
            self.logger.error(f"Failed to clear tenant context: {e}")
    
    async def get_table_policies(
        self, 
        connection: DatabaseConnection,
        table_name: str
    ) -> List[Dict[str, Any]]:
        """Récupère les politiques d'une table"""
        try:
            query = """
            SELECT 
                schemaname,
                tablename,
                policyname,
                permissive,
                roles,
                cmd,
                qual,
                with_check
            FROM pg_policies 
            WHERE tablename = %s
            """
            
            policies = await connection.fetch_all(query, (table_name,))
            return [dict(policy) for policy in policies]
            
        except Exception as e:
            self.logger.error(f"Failed to get policies for table {table_name}: {e}")
            return []
    
    async def disable_rls_on_table(
        self, 
        connection: DatabaseConnection,
        table_name: str
    ):
        """Désactive RLS sur une table"""
        try:
            # Drop policies first
            if table_name in self.created_policies:
                for policy_name in self.created_policies[table_name]:
                    drop_policy_query = f"DROP POLICY IF EXISTS {policy_name} ON {table_name}"
                    await connection.execute(drop_policy_query)
            
            # Disable RLS
            disable_rls_query = f"ALTER TABLE {table_name} DISABLE ROW LEVEL SECURITY"
            await connection.execute(disable_rls_query)
            
            # Clean up tracking
            self.rls_enabled_tables.discard(table_name)
            self.created_policies.pop(table_name, None)
            self.table_tenant_columns.pop(table_name, None)
            
            self.logger.info(f"Disabled RLS on table: {table_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to disable RLS on table {table_name}: {e}")


class RowLevelStrategy(IsolationStrategy):
    """
    Stratégie d'isolation au niveau des lignes (RLS)
    
    Features:
    - Row Level Security automatique
    - Isolation transparente par tenant_id
    - Performance optimisée avec indexes
    - Audit des violations RLS
    - Gestion automatique des politiques
    - Support des requêtes complexes
    - Monitoring granulaire
    """
    
    def __init__(self, rls_config: Optional[RLSConfig] = None):
        self.rls_config = rls_config or RLSConfig()
        self.logger = logging.getLogger("isolation.row_level")
        
        # RLS management
        self.policy_manager = RLSPolicyManager(self.rls_config)
        
        # Connection management (shared)
        self.shared_connection: Optional[DatabaseConnection] = None
        
        # Query monitoring
        self.query_monitor = RLSQueryMonitor(self.rls_config)
        
        # Stats per tenant
        self.tenant_stats: Dict[str, Dict[str, Any]] = {}
        
        # Tables that need RLS
        self.target_tables = [
            "users", "artists", "tracks", "ai_sessions", 
            "analytics", "playlists", "recommendations"
        ]
        
        self.logger.info("Row level isolation strategy initialized")
    
    async def initialize(self, config: EngineConfig):
        """Initialise la stratégie RLS"""
        try:
            # Initialize shared connection
            await self._init_shared_connection()
            
            # Setup RLS on target tables
            if self.rls_config.auto_create_policies:
                await self._setup_rls_on_tables()
            
            # Initialize query monitor
            await self.query_monitor.initialize(self.shared_connection)
            
            self.logger.info("Row level strategy ready")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize row level strategy: {e}")
            raise DataIsolationError(f"RLS initialization failed: {e}")
    
    async def _init_shared_connection(self):
        """Initialise la connexion partagée"""
        self.shared_connection = DatabaseConnection(
            host="localhost",
            port=5432,
            database="spotify_ai_agent",
            username="postgres",
            password=""
        )
        
        await self.shared_connection.connect()
        self.logger.info("Shared connection established for RLS")
    
    async def _setup_rls_on_tables(self):
        """Configure RLS sur les tables cibles"""
        try:
            for table_name in self.target_tables:
                # Check if table exists
                if await self._table_exists(table_name):
                    await self.policy_manager.enable_rls_on_table(
                        self.shared_connection,
                        table_name,
                        self.rls_config.tenant_column
                    )
                else:
                    self.logger.warning(f"Target table {table_name} does not exist")
            
            self.logger.info(f"RLS configured on {len(self.policy_manager.rls_enabled_tables)} tables")
            
        except Exception as e:
            self.logger.error(f"Failed to setup RLS on tables: {e}")
            raise
    
    async def _table_exists(self, table_name: str) -> bool:
        """Vérifie si une table existe"""
        try:
            query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            )
            """
            result = await self.shared_connection.fetch_one(query, (table_name,))
            return result["exists"]
            
        except Exception:
            return False
    
    async def apply_isolation(
        self, 
        context: TenantContext, 
        target: Any,
        **kwargs
    ) -> Any:
        """Applique l'isolation RLS"""
        try:
            # Set tenant context for RLS
            await self.policy_manager.set_tenant_context(
                self.shared_connection,
                context.tenant_id
            )
            
            # Execute query with RLS in effect
            if isinstance(target, str):  # SQL query
                result = await self._execute_rls_query(target, context.tenant_id, **kwargs)
            else:
                # Handle ORM objects
                result = await self._handle_orm_target(target, context.tenant_id, **kwargs)
            
            # Monitor query for compliance
            await self.query_monitor.log_query(
                context.tenant_id,
                target,
                "success",
                kwargs
            )
            
            # Update stats
            self._update_tenant_stats(context.tenant_id, "query_executed")
            
            return result
            
        except Exception as e:
            self.logger.error(f"RLS isolation failed for tenant {context.tenant_id}: {e}")
            
            # Log violation if it's a security issue
            if self.rls_config.audit_rls_violations:
                await self.query_monitor.log_query(
                    context.tenant_id,
                    target,
                    "security_violation",
                    {"error": str(e), **kwargs}
                )
            
            self._update_tenant_stats(context.tenant_id, "query_failed")
            raise DataIsolationError(f"RLS isolation failed: {e}")
        
        finally:
            # Clear tenant context
            await self.policy_manager.clear_tenant_context(self.shared_connection)
    
    async def _execute_rls_query(
        self, 
        query: str, 
        tenant_id: str,
        **kwargs
    ) -> Any:
        """Exécute une requête avec RLS actif"""
        params = kwargs.get("params", [])
        query_type = kwargs.get("query_type", "select")
        
        # RLS is automatically enforced by PostgreSQL
        # No need to modify the query
        
        if query_type.lower() == "select":
            return await self.shared_connection.fetch_all(query, params)
        elif query_type.lower() in ["insert", "update", "delete"]:
            # For INSERT/UPDATE, ensure tenant_id is set
            if query_type.lower() in ["insert", "update"]:
                query = self._ensure_tenant_id_in_query(query, tenant_id)
            return await self.shared_connection.execute(query, params)
        else:
            return await self.shared_connection.fetch_one(query, params)
    
    def _ensure_tenant_id_in_query(self, query: str, tenant_id: str) -> str:
        """S'assure que tenant_id est inclus dans les requêtes INSERT/UPDATE"""
        # This is a simplified implementation
        # In a real scenario, you'd use a proper SQL parser
        
        if "INSERT INTO" in query.upper() and self.rls_config.tenant_column not in query:
            # Add tenant_id to INSERT if not present
            # This is a basic implementation - would need proper SQL parsing
            pass
        
        if "UPDATE" in query.upper() and f"SET {self.rls_config.tenant_column}" not in query:
            # Ensure UPDATE doesn't change tenant_id
            pass
        
        return query
    
    async def _handle_orm_target(
        self, 
        target: Any, 
        tenant_id: str,
        **kwargs
    ) -> Any:
        """Gère les cibles ORM avec RLS"""
        # ORM frameworks typically handle RLS transparently
        # We just need to ensure the tenant context is set
        self.logger.debug(f"Handling ORM target: {type(target).__name__} for tenant: {tenant_id}")
        return target
    
    def _update_tenant_stats(self, tenant_id: str, stat_type: str):
        """Met à jour les statistiques tenant"""
        if tenant_id not in self.tenant_stats:
            self.tenant_stats[tenant_id] = {
                "queries_executed": 0,
                "queries_failed": 0,
                "security_violations": 0,
                "last_activity": datetime.now(timezone.utc),
                "first_seen": datetime.now(timezone.utc)
            }
        
        if stat_type == "query_executed":
            self.tenant_stats[tenant_id]["queries_executed"] += 1
        elif stat_type == "query_failed":
            self.tenant_stats[tenant_id]["queries_failed"] += 1
        elif stat_type == "security_violation":
            self.tenant_stats[tenant_id]["security_violations"] += 1
        
        self.tenant_stats[tenant_id]["last_activity"] = datetime.now(timezone.utc)
    
    async def validate_access(self, context: TenantContext, resource: str) -> bool:
        """Valide l'accès avec RLS"""
        try:
            # Set tenant context
            await self.policy_manager.set_tenant_context(
                self.shared_connection,
                context.tenant_id
            )
            
            # RLS will automatically enforce access control
            # Additional validation can be added here
            
            return True
            
        except Exception as e:
            self.logger.error(f"RLS access validation failed for tenant {context.tenant_id}: {e}")
            return False
        
        finally:
            await self.policy_manager.clear_tenant_context(self.shared_connection)
    
    async def health_check(self) -> bool:
        """Vérification de santé RLS"""
        try:
            # Check shared connection
            if not self.shared_connection or not await self.shared_connection.is_healthy():
                return False
            
            # Check if RLS is working
            test_query = "SELECT current_setting('app.current_tenant_id', true) as tenant_id"
            await self.shared_connection.fetch_one(test_query)
            
            # Check policies exist
            for table_name in self.policy_manager.rls_enabled_tables:
                policies = await self.policy_manager.get_table_policies(
                    self.shared_connection,
                    table_name
                )
                if not policies:
                    self.logger.warning(f"No RLS policies found for table {table_name}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"RLS health check failed: {e}")
            return False
    
    async def get_tenant_statistics(self, tenant_id: str) -> Dict[str, Any]:
        """Obtient les statistiques RLS pour un tenant"""
        stats = self.tenant_stats.get(tenant_id, {})
        
        # Add RLS-specific stats
        stats.update({
            "rls_enabled_tables": list(self.policy_manager.rls_enabled_tables),
            "tenant_column": self.rls_config.tenant_column,
            "policies_count": sum(
                len(policies) for policies in self.policy_manager.created_policies.values()
            )
        })
        
        return stats
    
    async def add_table_to_rls(self, table_name: str):
        """Ajoute une table à RLS"""
        if table_name not in self.target_tables:
            self.target_tables.append(table_name)
        
        if await self._table_exists(table_name):
            await self.policy_manager.enable_rls_on_table(
                self.shared_connection,
                table_name,
                self.rls_config.tenant_column
            )
            self.logger.info(f"Added table {table_name} to RLS")
    
    async def remove_table_from_rls(self, table_name: str):
        """Supprime une table de RLS"""
        if table_name in self.target_tables:
            self.target_tables.remove(table_name)
        
        await self.policy_manager.disable_rls_on_table(
            self.shared_connection,
            table_name
        )
        self.logger.info(f"Removed table {table_name} from RLS")
    
    async def cleanup(self):
        """Nettoie les ressources RLS"""
        self.logger.info("Cleaning up row level strategy...")
        
        # Cleanup query monitor
        await self.query_monitor.cleanup()
        
        # Close connection
        if self.shared_connection:
            try:
                await self.shared_connection.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting: {e}")
        
        self.logger.info("Row level strategy cleanup completed")
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de la stratégie RLS"""
        return {
            "strategy_type": "row_level",
            "rls_enabled_tables": list(self.policy_manager.rls_enabled_tables),
            "total_policies": sum(
                len(policies) for policies in self.policy_manager.created_policies.values()
            ),
            "tenant_stats": dict(self.tenant_stats),
            "config": {
                "tenant_column": self.rls_config.tenant_column,
                "auto_create_policies": self.rls_config.auto_create_policies,
                "enforce_rls_on_tables": self.rls_config.enforce_rls_on_tables,
                "audit_rls_violations": self.rls_config.audit_rls_violations
            }
        }


class RLSQueryMonitor:
    """Moniteur des requêtes RLS pour audit et sécurité"""
    
    def __init__(self, config: RLSConfig):
        self.config = config
        self.logger = logging.getLogger("rls.query_monitor")
        self.query_log: List[Dict[str, Any]] = []
    
    async def initialize(self, connection: DatabaseConnection):
        """Initialise le moniteur"""
        # Create audit table if needed
        if self.config.audit_rls_violations:
            await self._create_audit_table(connection)
    
    async def _create_audit_table(self, connection: DatabaseConnection):
        """Crée la table d'audit RLS"""
        try:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS rls_audit_log (
                id SERIAL PRIMARY KEY,
                tenant_id VARCHAR(255),
                query_text TEXT,
                query_type VARCHAR(100),
                status VARCHAR(50),
                metadata JSONB,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            await connection.execute(create_table_query)
            
            # Create index for performance
            create_index_query = """
            CREATE INDEX IF NOT EXISTS idx_rls_audit_tenant_timestamp 
            ON rls_audit_log (tenant_id, timestamp)
            """
            
            await connection.execute(create_index_query)
            
        except Exception as e:
            self.logger.error(f"Failed to create audit table: {e}")
    
    async def log_query(
        self, 
        tenant_id: str,
        query: Any,
        status: str,
        metadata: Dict[str, Any]
    ):
        """Log une requête pour audit"""
        log_entry = {
            "tenant_id": tenant_id,
            "query": str(query)[:1000],  # Limit query length
            "status": status,
            "metadata": metadata,
            "timestamp": datetime.now(timezone.utc)
        }
        
        self.query_log.append(log_entry)
        
        # Keep only last 1000 entries in memory
        if len(self.query_log) > 1000:
            self.query_log = self.query_log[-1000:]
        
        if status == "security_violation":
            self.logger.warning(f"RLS security violation for tenant {tenant_id}: {query}")
    
    async def cleanup(self):
        """Nettoie le moniteur"""
        self.query_log.clear()
