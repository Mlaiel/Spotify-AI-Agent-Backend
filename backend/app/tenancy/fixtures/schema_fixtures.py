"""
Spotify AI Agent - Schema Fixtures
=================================

Enterprise database schema management and initialization
for multi-tenant Spotify AI Agent architecture.
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

from sqlalchemy import text, MetaData, Table, inspect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.engine import Connection
from alembic import command
from alembic.config import Config

from app.tenancy.fixtures.base import BaseFixture, FixtureMetadata, FixtureType
from app.tenancy.fixtures.exceptions import (
    FixtureSchemaError,
    FixtureValidationError,
    FixtureDataError
)
from app.tenancy.fixtures.constants import FIXTURE_SCHEMAS

logger = logging.getLogger(__name__)


class SchemaType(Enum):
    """Database schema types."""
    TENANT = "tenant"
    SHARED = "shared"
    SYSTEM = "system"
    ANALYTICS = "analytics"
    AUDIT = "audit"


class MigrationDirection(Enum):
    """Migration direction enumeration."""
    UP = "up"
    DOWN = "down"


@dataclass
class SchemaVersion:
    """Schema version information."""
    version: str
    applied_at: datetime
    description: str
    checksum: str
    migration_file: Optional[str] = None


class SchemaConfiguration:
    """Schema configuration and metadata."""
    
    def __init__(
        self,
        schema_name: str,
        schema_type: SchemaType,
        tenant_id: Optional[str] = None,
        version: str = "1.0.0"
    ):
        self.schema_name = schema_name
        self.schema_type = schema_type
        self.tenant_id = tenant_id
        self.version = version
        self.tables: List[str] = []
        self.indexes: List[str] = []
        self.functions: List[str] = []
        self.triggers: List[str] = []
        self.constraints: List[str] = []
        self.permissions: Dict[str, List[str]] = {}
        
    def add_table(self, table_name: str) -> None:
        """Add a table to the schema configuration."""
        if table_name not in self.tables:
            self.tables.append(table_name)
    
    def add_index(self, index_name: str) -> None:
        """Add an index to the schema configuration."""
        if index_name not in self.indexes:
            self.indexes.append(index_name)
    
    def add_permission(self, role: str, permissions: List[str]) -> None:
        """Add permissions for a role."""
        if role not in self.permissions:
            self.permissions[role] = []
        self.permissions[role].extend(permissions)


class SchemaFixture(BaseFixture[SchemaConfiguration]):
    """
    Fixture for database schema management and initialization.
    
    Handles:
    - Schema creation and deletion
    - Table creation with proper constraints
    - Index management
    - Permission setup
    - Migration execution
    - Version tracking
    """
    
    def __init__(
        self,
        schema_config: SchemaConfiguration,
        sql_files: Optional[List[str]] = None,
        migration_path: Optional[str] = None,
        **kwargs
    ):
        metadata = FixtureMetadata(
            fixture_type=FixtureType.SCHEMA,
            tenant_id=schema_config.tenant_id,
            name=f"schema_{schema_config.schema_name}",
            description=f"Schema setup for {schema_config.schema_name}",
            tags={"schema", schema_config.schema_type.value}
        )
        super().__init__(metadata, **kwargs)
        
        self.schema_config = schema_config
        self.sql_files = sql_files or []
        self.migration_path = migration_path
        self.created_objects: List[Tuple[str, str]] = []  # (object_type, object_name)
        
        logger.info(f"Initialized schema fixture: {schema_config.schema_name}")
    
    async def validate(self) -> bool:
        """
        Validate schema configuration and SQL files.
        
        Returns:
            bool: True if validation passes
            
        Raises:
            FixtureValidationError: If validation fails
        """
        errors = []
        
        try:
            # Validate schema name
            if not self._is_valid_schema_name(self.schema_config.schema_name):
                errors.append(f"Invalid schema name: {self.schema_config.schema_name}")
            
            # Check if schema already exists
            session = await self.get_session()
            exists = await self._schema_exists(session, self.schema_config.schema_name)
            if exists:
                errors.append(f"Schema already exists: {self.schema_config.schema_name}")
            
            # Validate SQL files
            for sql_file in self.sql_files:
                if not os.path.exists(sql_file):
                    errors.append(f"SQL file not found: {sql_file}")
                else:
                    file_errors = await self._validate_sql_file(sql_file)
                    errors.extend(file_errors)
            
            # Validate migration path if provided
            if self.migration_path and not os.path.exists(self.migration_path):
                errors.append(f"Migration path not found: {self.migration_path}")
            
            if errors:
                raise FixtureValidationError(
                    f"Schema validation failed for {self.schema_config.schema_name}",
                    validation_errors=errors,
                    fixture_id=self.metadata.fixture_id,
                    tenant_id=self.metadata.tenant_id
                )
            
            logger.info(f"Schema validation passed: {self.schema_config.schema_name}")
            return True
            
        except Exception as e:
            if isinstance(e, FixtureValidationError):
                raise
            raise FixtureValidationError(
                f"Schema validation error: {str(e)}",
                fixture_id=self.metadata.fixture_id,
                tenant_id=self.metadata.tenant_id
            )
    
    async def apply(self) -> SchemaConfiguration:
        """
        Apply schema configuration and create database objects.
        
        Returns:
            SchemaConfiguration: Applied configuration
            
        Raises:
            FixtureSchemaError: If schema creation fails
        """
        try:
            session = await self.get_session()
            
            # Create schema
            await self._create_schema(session)
            self.increment_processed()
            
            # Execute SQL files
            for sql_file in self.sql_files:
                await self._execute_sql_file(session, sql_file)
                self.increment_processed()
            
            # Create core tables if not provided in SQL files
            if not self.sql_files:
                await self._create_core_tables(session)
                self.increment_processed(5)
            
            # Create indexes
            await self._create_indexes(session)
            self.increment_processed()
            
            # Setup permissions
            await self._setup_permissions(session)
            self.increment_processed()
            
            # Run migrations if migration path provided
            if self.migration_path:
                await self._run_migrations(session)
                self.increment_processed()
            
            # Record schema version
            await self._record_schema_version(session)
            self.increment_processed()
            
            await session.commit()
            
            logger.info(f"Schema creation completed: {self.schema_config.schema_name}")
            return self.schema_config
            
        except Exception as e:
            logger.error(f"Schema creation failed: {self.schema_config.schema_name} - {e}")
            await session.rollback()
            raise FixtureSchemaError(
                f"Failed to create schema: {str(e)}",
                schema_name=self.schema_config.schema_name,
                fixture_id=self.metadata.fixture_id,
                tenant_id=self.metadata.tenant_id
            )
    
    async def rollback(self) -> bool:
        """
        Rollback schema creation.
        
        Returns:
            bool: True if rollback successful
        """
        try:
            session = await self.get_session()
            
            # Drop schema with cascade to remove all objects
            await session.execute(
                text(f"DROP SCHEMA IF EXISTS {self.schema_config.schema_name} CASCADE")
            )
            await session.commit()
            
            logger.info(f"Schema rollback completed: {self.schema_config.schema_name}")
            return True
            
        except Exception as e:
            logger.error(f"Schema rollback failed: {self.schema_config.schema_name} - {e}")
            return False
    
    def _is_valid_schema_name(self, schema_name: str) -> bool:
        """Validate schema name format."""
        # PostgreSQL identifier rules
        pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
        return bool(re.match(pattern, schema_name)) and len(schema_name) <= 63
    
    async def _schema_exists(self, session: AsyncSession, schema_name: str) -> bool:
        """Check if schema already exists."""
        result = await session.execute(
            text(
                "SELECT 1 FROM information_schema.schemata "
                "WHERE schema_name = :schema_name"
            ),
            {"schema_name": schema_name}
        )
        return result.first() is not None
    
    async def _validate_sql_file(self, sql_file: str) -> List[str]:
        """Validate SQL file syntax and content."""
        errors = []
        
        try:
            with open(sql_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic SQL validation
            if not content.strip():
                errors.append(f"SQL file is empty: {sql_file}")
                return errors
            
            # Check for dangerous operations
            dangerous_patterns = [
                r'\bDROP\s+DATABASE\b',
                r'\bTRUNCATE\s+TABLE\b',
                r'\bDELETE\s+FROM\s+\w+\s*;?\s*$'  # DELETE without WHERE
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    self.add_warning(f"Potentially dangerous SQL found in {sql_file}")
            
            # Check for required schema prefix
            if self.schema_config.schema_type == SchemaType.TENANT:
                if not re.search(rf'\b{self.schema_config.schema_name}\.', content):
                    errors.append(f"SQL file should reference schema {self.schema_config.schema_name}")
            
        except Exception as e:
            errors.append(f"Error reading SQL file {sql_file}: {str(e)}")
        
        return errors
    
    async def _create_schema(self, session: AsyncSession) -> None:
        """Create the database schema."""
        await session.execute(
            text(f"CREATE SCHEMA {self.schema_config.schema_name}")
        )
        self.created_objects.append(("schema", self.schema_config.schema_name))
        logger.info(f"Created schema: {self.schema_config.schema_name}")
    
    async def _execute_sql_file(self, session: AsyncSession, sql_file: str) -> None:
        """Execute SQL statements from file."""
        try:
            with open(sql_file, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # Split by semicolon and execute each statement
            statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
            
            for statement in statements:
                if statement:
                    await session.execute(text(statement))
            
            logger.info(f"Executed SQL file: {sql_file}")
            
        except Exception as e:
            logger.error(f"Failed to execute SQL file {sql_file}: {e}")
            raise FixtureSchemaError(f"SQL execution failed: {str(e)}")
    
    async def _create_core_tables(self, session: AsyncSession) -> None:
        """Create core tables for the schema."""
        schema_name = self.schema_config.schema_name
        
        if self.schema_config.schema_type == SchemaType.TENANT:
            await self._create_tenant_tables(session, schema_name)
        elif self.schema_config.schema_type == SchemaType.ANALYTICS:
            await self._create_analytics_tables(session, schema_name)
        elif self.schema_config.schema_type == SchemaType.AUDIT:
            await self._create_audit_tables(session, schema_name)
        
        logger.info(f"Created core tables for schema: {schema_name}")
    
    async def _create_tenant_tables(self, session: AsyncSession, schema_name: str) -> None:
        """Create tables for tenant schema."""
        tables_sql = [
            f"""
            CREATE TABLE {schema_name}.users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                tenant_id VARCHAR(50) NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                name VARCHAR(100) NOT NULL,
                role VARCHAR(50) DEFAULT 'user',
                spotify_user_id VARCHAR(100),
                preferences JSONB DEFAULT '{{}}',
                is_active BOOLEAN DEFAULT true,
                last_login_at TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            """,
            f"""
            CREATE TABLE {schema_name}.spotify_connections (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES {schema_name}.users(id) ON DELETE CASCADE,
                spotify_user_id VARCHAR(100) UNIQUE NOT NULL,
                access_token TEXT,
                refresh_token TEXT,
                token_expires_at TIMESTAMP WITH TIME ZONE,
                scopes TEXT[],
                profile_data JSONB,
                is_active BOOLEAN DEFAULT true,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            """,
            f"""
            CREATE TABLE {schema_name}.ai_sessions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES {schema_name}.users(id) ON DELETE CASCADE,
                session_type VARCHAR(50) NOT NULL,
                status VARCHAR(20) DEFAULT 'active',
                context JSONB DEFAULT '{{}}',
                metadata JSONB DEFAULT '{{}}',
                expires_at TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            """,
            f"""
            CREATE TABLE {schema_name}.content_generated (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id UUID REFERENCES {schema_name}.ai_sessions(id) ON DELETE CASCADE,
                user_id UUID REFERENCES {schema_name}.users(id) ON DELETE CASCADE,
                content_type VARCHAR(50) NOT NULL,
                content_data JSONB NOT NULL,
                metadata JSONB DEFAULT '{{}}',
                quality_score DECIMAL(3,2),
                feedback_score INTEGER,
                is_published BOOLEAN DEFAULT false,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            """,
            f"""
            CREATE TABLE {schema_name}.collaborations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                creator_id UUID REFERENCES {schema_name}.users(id) ON DELETE CASCADE,
                title VARCHAR(200) NOT NULL,
                description TEXT,
                collaboration_type VARCHAR(50) NOT NULL,
                status VARCHAR(20) DEFAULT 'active',
                settings JSONB DEFAULT '{{}}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            """,
            f"""
            CREATE TABLE {schema_name}.collaboration_participants (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                collaboration_id UUID REFERENCES {schema_name}.collaborations(id) ON DELETE CASCADE,
                user_id UUID REFERENCES {schema_name}.users(id) ON DELETE CASCADE,
                role VARCHAR(50) DEFAULT 'participant',
                permissions JSONB DEFAULT '{{}}',
                joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(collaboration_id, user_id)
            )
            """
        ]
        
        for sql in tables_sql:
            await session.execute(text(sql))
            self.schema_config.add_table(sql.split('.')[1].split('(')[0].strip())
    
    async def _create_analytics_tables(self, session: AsyncSession, schema_name: str) -> None:
        """Create tables for analytics schema."""
        tables_sql = [
            f"""
            CREATE TABLE {schema_name}.events (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                tenant_id VARCHAR(50) NOT NULL,
                user_id UUID,
                event_type VARCHAR(100) NOT NULL,
                event_name VARCHAR(100) NOT NULL,
                properties JSONB DEFAULT '{{}}',
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                session_id VARCHAR(100),
                ip_address INET,
                user_agent TEXT
            )
            """,
            f"""
            CREATE TABLE {schema_name}.metrics (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                tenant_id VARCHAR(50) NOT NULL,
                metric_name VARCHAR(100) NOT NULL,
                metric_value DECIMAL(15,4) NOT NULL,
                dimensions JSONB DEFAULT '{{}}',
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                aggregation_period VARCHAR(20)
            )
            """
        ]
        
        for sql in tables_sql:
            await session.execute(text(sql))
    
    async def _create_audit_tables(self, session: AsyncSession, schema_name: str) -> None:
        """Create tables for audit schema."""
        tables_sql = [
            f"""
            CREATE TABLE {schema_name}.audit_log (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                tenant_id VARCHAR(50) NOT NULL,
                user_id UUID,
                action VARCHAR(100) NOT NULL,
                resource_type VARCHAR(100) NOT NULL,
                resource_id VARCHAR(100),
                old_values JSONB,
                new_values JSONB,
                ip_address INET,
                user_agent TEXT,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            """
        ]
        
        for sql in tables_sql:
            await session.execute(text(sql))
    
    async def _create_indexes(self, session: AsyncSession) -> None:
        """Create performance indexes."""
        schema_name = self.schema_config.schema_name
        
        if self.schema_config.schema_type == SchemaType.TENANT:
            indexes_sql = [
                f"CREATE INDEX idx_{schema_name}_users_tenant_id ON {schema_name}.users(tenant_id)",
                f"CREATE INDEX idx_{schema_name}_users_email ON {schema_name}.users(email)",
                f"CREATE INDEX idx_{schema_name}_spotify_connections_user_id ON {schema_name}.spotify_connections(user_id)",
                f"CREATE INDEX idx_{schema_name}_ai_sessions_user_id ON {schema_name}.ai_sessions(user_id)",
                f"CREATE INDEX idx_{schema_name}_ai_sessions_status ON {schema_name}.ai_sessions(status)",
                f"CREATE INDEX idx_{schema_name}_content_generated_session_id ON {schema_name}.content_generated(session_id)",
                f"CREATE INDEX idx_{schema_name}_content_generated_user_id ON {schema_name}.content_generated(user_id)",
                f"CREATE INDEX idx_{schema_name}_content_generated_type ON {schema_name}.content_generated(content_type)",
                f"CREATE INDEX idx_{schema_name}_collaborations_creator_id ON {schema_name}.collaborations(creator_id)",
                f"CREATE INDEX idx_{schema_name}_collaboration_participants_collab_id ON {schema_name}.collaboration_participants(collaboration_id)"
            ]
        elif self.schema_config.schema_type == SchemaType.ANALYTICS:
            indexes_sql = [
                f"CREATE INDEX idx_{schema_name}_events_tenant_id ON {schema_name}.events(tenant_id)",
                f"CREATE INDEX idx_{schema_name}_events_timestamp ON {schema_name}.events(timestamp)",
                f"CREATE INDEX idx_{schema_name}_events_type ON {schema_name}.events(event_type)",
                f"CREATE INDEX idx_{schema_name}_metrics_tenant_id ON {schema_name}.metrics(tenant_id)",
                f"CREATE INDEX idx_{schema_name}_metrics_name ON {schema_name}.metrics(metric_name)",
                f"CREATE INDEX idx_{schema_name}_metrics_timestamp ON {schema_name}.metrics(timestamp)"
            ]
        elif self.schema_config.schema_type == SchemaType.AUDIT:
            indexes_sql = [
                f"CREATE INDEX idx_{schema_name}_audit_log_tenant_id ON {schema_name}.audit_log(tenant_id)",
                f"CREATE INDEX idx_{schema_name}_audit_log_user_id ON {schema_name}.audit_log(user_id)",
                f"CREATE INDEX idx_{schema_name}_audit_log_timestamp ON {schema_name}.audit_log(timestamp)",
                f"CREATE INDEX idx_{schema_name}_audit_log_action ON {schema_name}.audit_log(action)"
            ]
        else:
            indexes_sql = []
        
        for sql in indexes_sql:
            await session.execute(text(sql))
            index_name = sql.split(' ')[2]  # Extract index name
            self.schema_config.add_index(index_name)
        
        logger.info(f"Created {len(indexes_sql)} indexes for schema: {schema_name}")
    
    async def _setup_permissions(self, session: AsyncSession) -> None:
        """Setup database permissions for the schema."""
        schema_name = self.schema_config.schema_name
        
        # Grant permissions to application roles
        permissions_sql = [
            f"GRANT USAGE ON SCHEMA {schema_name} TO app_user",
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {schema_name} TO app_user",
            f"GRANT USAGE ON ALL SEQUENCES IN SCHEMA {schema_name} TO app_user",
            f"GRANT USAGE ON SCHEMA {schema_name} TO app_readonly",
            f"GRANT SELECT ON ALL TABLES IN SCHEMA {schema_name} TO app_readonly"
        ]
        
        for sql in permissions_sql:
            try:
                await session.execute(text(sql))
            except Exception as e:
                # Log warning if role doesn't exist but continue
                self.add_warning(f"Permission setup warning: {e}")
        
        logger.info(f"Setup permissions for schema: {schema_name}")
    
    async def _run_migrations(self, session: AsyncSession) -> None:
        """Run database migrations if migration path provided."""
        # This would integrate with Alembic or custom migration system
        logger.info(f"Running migrations for schema: {self.schema_config.schema_name}")
    
    async def _record_schema_version(self, session: AsyncSession) -> None:
        """Record schema version information."""
        # Create version tracking if it doesn't exist
        version_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.schema_config.schema_name}.schema_versions (
            version VARCHAR(50) PRIMARY KEY,
            applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            description TEXT,
            checksum VARCHAR(64)
        )
        """
        
        await session.execute(text(version_table_sql))
        
        # Insert current version
        insert_version_sql = f"""
        INSERT INTO {self.schema_config.schema_name}.schema_versions 
        (version, description, checksum) 
        VALUES (:version, :description, :checksum)
        """
        
        await session.execute(
            text(insert_version_sql),
            {
                "version": self.schema_config.version,
                "description": f"Initial schema creation for {self.schema_config.schema_name}",
                "checksum": "initial"
            }
        )
        
        logger.info(f"Recorded schema version {self.schema_config.version}")


class SchemaInitializer:
    """
    Utility class for initializing database schemas
    from templates and configuration files.
    """
    
    def __init__(self, base_path: str = "/app/schemas"):
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(f"{__name__}.SchemaInitializer")
    
    async def initialize_tenant_schema(
        self,
        tenant_id: str,
        schema_template: str = "tenant_template.sql"
    ) -> SchemaConfiguration:
        """Initialize a new tenant schema from template."""
        schema_name = f"tenant_{tenant_id}"
        config = SchemaConfiguration(
            schema_name=schema_name,
            schema_type=SchemaType.TENANT,
            tenant_id=tenant_id
        )
        
        template_path = self.base_path / "templates" / schema_template
        if template_path.exists():
            sql_content = await self._load_template(template_path, {"tenant_id": tenant_id})
            # Process template and create fixture
            # This would be expanded based on template system
        
        self.logger.info(f"Initialized tenant schema configuration: {schema_name}")
        return config
    
    async def _load_template(self, template_path: Path, variables: Dict[str, str]) -> str:
        """Load and process SQL template with variable substitution."""
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple variable substitution
        for key, value in variables.items():
            content = content.replace(f"{{{key}}}", value)
        
        return content
    
    async def validate_schema_integrity(
        self,
        session: AsyncSession,
        schema_name: str
    ) -> List[str]:
        """Validate schema integrity and return any issues found."""
        issues = []
        
        try:
            # Check if schema exists
            result = await session.execute(
                text(
                    "SELECT 1 FROM information_schema.schemata "
                    "WHERE schema_name = :schema_name"
                ),
                {"schema_name": schema_name}
            )
            
            if not result.first():
                issues.append(f"Schema does not exist: {schema_name}")
                return issues
            
            # Check for required tables
            required_tables = ["users", "spotify_connections", "ai_sessions"]
            for table in required_tables:
                result = await session.execute(
                    text(
                        "SELECT 1 FROM information_schema.tables "
                        "WHERE table_schema = :schema_name AND table_name = :table_name"
                    ),
                    {"schema_name": schema_name, "table_name": table}
                )
                
                if not result.first():
                    issues.append(f"Missing required table: {schema_name}.{table}")
            
            # Check for foreign key constraints
            # This would be expanded with more comprehensive checks
            
        except Exception as e:
            issues.append(f"Error validating schema integrity: {str(e)}")
        
        return issues
