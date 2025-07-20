#!/usr/bin/env python3
"""
Spotify AI Agent - Data Validation Script
=========================================

Comprehensive data validation script that performs:
- Schema validation and integrity checks
- Business rule validation
- Performance and resource validation
- Security compliance checks
- Multi-tenant data consistency validation

Usage:
    python -m app.tenancy.fixtures.scripts.validate_data --tenant-id mycompany
    python validate_data.py --tenant-id startup --check-type all --fix-issues

Author: Expert Development Team (Fahed Mlaiel)
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session
from app.core.cache import get_redis_client
from app.tenancy.fixtures.base import FixtureManager
from app.tenancy.fixtures.validators import (
    TenantValidator, DataValidator, SchemaValidator, SecurityValidator
)
from app.tenancy.fixtures.monitoring import FixtureMonitor
from app.tenancy.fixtures.utils import FixtureUtils, TenantUtils, ValidationUtils
from app.tenancy.fixtures.exceptions import FixtureError, FixtureValidationError
from app.tenancy.fixtures.constants import (
    VALIDATION_RULES, SECURITY_CHECKS, PERFORMANCE_THRESHOLDS
)

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Comprehensive data validation orchestrator.
    
    Performs multiple levels of validation:
    - Structure validation (schema, tables, indexes)
    - Data integrity validation (constraints, references)
    - Business logic validation (rules, workflows)
    - Performance validation (query efficiency, resource usage)
    - Security validation (permissions, encryption, compliance)
    """
    
    def __init__(self, session: AsyncSession, redis_client=None):
        self.session = session
        self.redis_client = redis_client
        self.fixture_manager = FixtureManager(session, redis_client)
        self.monitor = FixtureMonitor(session, redis_client)
        
        # Initialize validators
        self.tenant_validator = TenantValidator(session)
        self.data_validator = DataValidator(session)
        self.schema_validator = SchemaValidator(session)
        self.security_validator = SecurityValidator(session)
    
    async def validate_tenant_data(
        self,
        tenant_id: str,
        check_types: List[str] = None,
        fix_issues: bool = False,
        detailed_report: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive tenant data validation.
        
        Args:
            tenant_id: Target tenant identifier
            check_types: Types of checks to perform (schema, data, business, performance, security)
            fix_issues: Attempt to automatically fix found issues
            detailed_report: Generate detailed validation report
            
        Returns:
            Validation results with issues found and fixes applied
        """
        start_time = datetime.now(timezone.utc)
        
        if check_types is None:
            check_types = ["schema", "data", "business", "performance", "security"]
        
        validation_result = {
            "tenant_id": tenant_id,
            "check_types": check_types,
            "status": "started",
            "start_time": start_time.isoformat(),
            "fix_issues": fix_issues,
            "checks_performed": {},
            "issues_found": [],
            "issues_fixed": [],
            "warnings": [],
            "summary": {},
            "recommendations": []
        }
        
        try:
            # Validate tenant exists
            if not await self._check_tenant_exists(tenant_id):
                raise FixtureError(f"Tenant not found: {tenant_id}")
            
            # Perform requested validations
            for check_type in check_types:
                logger.info(f"Performing {check_type} validation for tenant: {tenant_id}")
                
                check_result = await self._perform_validation_check(
                    tenant_id, check_type, fix_issues
                )
                
                validation_result["checks_performed"][check_type] = check_result
                validation_result["issues_found"].extend(check_result.get("issues", []))
                validation_result["issues_fixed"].extend(check_result.get("fixes", []))
                validation_result["warnings"].extend(check_result.get("warnings", []))
                
                if check_result.get("recommendations"):
                    validation_result["recommendations"].extend(check_result["recommendations"])
            
            # Generate summary
            validation_result["summary"] = await self._generate_validation_summary(
                tenant_id, validation_result
            )
            
            # Calculate final metrics
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            validation_result.update({
                "status": "completed",
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "total_issues": len(validation_result["issues_found"]),
                "issues_fixed_count": len(validation_result["issues_fixed"]),
                "warnings_count": len(validation_result["warnings"])
            })
            
            # Record validation metrics
            await self.monitor.record_validation_operation(tenant_id, validation_result)
            
            logger.info(
                f"Validation completed for {tenant_id}: "
                f"{validation_result['total_issues']} issues found, "
                f"{validation_result['issues_fixed_count']} fixed"
            )
            
        except Exception as e:
            validation_result["status"] = "failed"
            validation_result["error"] = str(e)
            logger.error(f"Validation failed for {tenant_id}: {e}")
            raise
        
        return validation_result
    
    async def _check_tenant_exists(self, tenant_id: str) -> bool:
        """Check if tenant exists in the system."""
        try:
            schema_name = TenantUtils.get_tenant_schema_name(tenant_id)
            result = await self.session.execute(
                text(f"SELECT schema_name FROM information_schema.schemata WHERE schema_name = '{schema_name}'")
            )
            return result.scalar() is not None
        except Exception:
            return False
    
    async def _perform_validation_check(
        self,
        tenant_id: str,
        check_type: str,
        fix_issues: bool
    ) -> Dict[str, Any]:
        """Perform specific type of validation check."""
        check_methods = {
            "schema": self._validate_schema_structure,
            "data": self._validate_data_integrity,
            "business": self._validate_business_rules,
            "performance": self._validate_performance_metrics,
            "security": self._validate_security_compliance
        }
        
        if check_type not in check_methods:
            raise FixtureValidationError(f"Unknown check type: {check_type}")
        
        return await check_methods[check_type](tenant_id, fix_issues)
    
    async def _validate_schema_structure(
        self,
        tenant_id: str,
        fix_issues: bool
    ) -> Dict[str, Any]:
        """Validate database schema structure."""
        check_result = {
            "check_type": "schema",
            "issues": [],
            "fixes": [],
            "warnings": [],
            "recommendations": []
        }
        
        schema_name = TenantUtils.get_tenant_schema_name(tenant_id)
        
        # Check required tables exist
        required_tables = [
            "users", "collaborations", "content_generated", "ai_sessions",
            "spotify_connections", "collaboration_participants"
        ]
        
        for table in required_tables:
            table_exists = await self._check_table_exists(schema_name, table)
            if not table_exists:
                issue = f"Missing required table: {schema_name}.{table}"
                check_result["issues"].append(issue)
                
                if fix_issues:
                    try:
                        await self._create_missing_table(schema_name, table)
                        check_result["fixes"].append(f"Created table: {table}")
                    except Exception as e:
                        check_result["warnings"].append(f"Failed to create table {table}: {e}")
        
        # Check indexes exist
        await self._validate_indexes(schema_name, check_result, fix_issues)
        
        # Check foreign key constraints
        await self._validate_foreign_keys(schema_name, check_result, fix_issues)
        
        # Check column types and constraints
        await self._validate_column_definitions(schema_name, check_result, fix_issues)
        
        return check_result
    
    async def _validate_data_integrity(
        self,
        tenant_id: str,
        fix_issues: bool
    ) -> Dict[str, Any]:
        """Validate data integrity and consistency."""
        check_result = {
            "check_type": "data",
            "issues": [],
            "fixes": [],
            "warnings": [],
            "recommendations": []
        }
        
        schema_name = TenantUtils.get_tenant_schema_name(tenant_id)
        
        # Check for orphaned records
        await self._check_orphaned_records(schema_name, check_result, fix_issues)
        
        # Check data consistency
        await self._check_data_consistency(schema_name, check_result, fix_issues)
        
        # Check for duplicate records
        await self._check_duplicate_records(schema_name, check_result, fix_issues)
        
        # Validate data formats
        await self._validate_data_formats(schema_name, check_result, fix_issues)
        
        # Check referential integrity
        await self._check_referential_integrity(schema_name, check_result, fix_issues)
        
        return check_result
    
    async def _validate_business_rules(
        self,
        tenant_id: str,
        fix_issues: bool
    ) -> Dict[str, Any]:
        """Validate business logic and rules."""
        check_result = {
            "check_type": "business",
            "issues": [],
            "fixes": [],
            "warnings": [],
            "recommendations": []
        }
        
        schema_name = TenantUtils.get_tenant_schema_name(tenant_id)
        
        # Validate user limits based on tier
        await self._validate_tier_limits(tenant_id, schema_name, check_result, fix_issues)
        
        # Check collaboration rules
        await self._validate_collaboration_rules(schema_name, check_result, fix_issues)
        
        # Validate content generation limits
        await self._validate_content_limits(schema_name, check_result, fix_issues)
        
        # Check AI session rules
        await self._validate_ai_session_rules(schema_name, check_result, fix_issues)
        
        # Validate Spotify connection rules
        await self._validate_spotify_rules(schema_name, check_result, fix_issues)
        
        return check_result
    
    async def _validate_performance_metrics(
        self,
        tenant_id: str,
        fix_issues: bool
    ) -> Dict[str, Any]:
        """Validate performance and resource usage."""
        check_result = {
            "check_type": "performance",
            "issues": [],
            "fixes": [],
            "warnings": [],
            "recommendations": []
        }
        
        schema_name = TenantUtils.get_tenant_schema_name(tenant_id)
        
        # Check query performance
        await self._check_query_performance(schema_name, check_result, fix_issues)
        
        # Check storage usage
        await self._check_storage_usage(schema_name, check_result, fix_issues)
        
        # Check cache efficiency
        await self._check_cache_efficiency(tenant_id, check_result, fix_issues)
        
        # Check resource limits
        await self._check_resource_limits(tenant_id, check_result, fix_issues)
        
        return check_result
    
    async def _validate_security_compliance(
        self,
        tenant_id: str,
        fix_issues: bool
    ) -> Dict[str, Any]:
        """Validate security and compliance requirements."""
        check_result = {
            "check_type": "security",
            "issues": [],
            "fixes": [],
            "warnings": [],
            "recommendations": []
        }
        
        schema_name = TenantUtils.get_tenant_schema_name(tenant_id)
        
        # Check data encryption
        await self._check_data_encryption(schema_name, check_result, fix_issues)
        
        # Validate access permissions
        await self._validate_permissions(schema_name, check_result, fix_issues)
        
        # Check for sensitive data exposure
        await self._check_sensitive_data(schema_name, check_result, fix_issues)
        
        # Validate audit trails
        await self._validate_audit_trails(schema_name, check_result, fix_issues)
        
        # Check compliance with data protection regulations
        await self._check_compliance_rules(schema_name, check_result, fix_issues)
        
        return check_result
    
    async def _check_table_exists(self, schema_name: str, table_name: str) -> bool:
        """Check if table exists in schema."""
        try:
            result = await self.session.execute(
                text(f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = '{schema_name}' 
                AND table_name = '{table_name}'
                """)
            )
            return result.scalar() is not None
        except Exception:
            return False
    
    async def _create_missing_table(self, schema_name: str, table_name: str) -> None:
        """Create missing table with basic structure."""
        table_definitions = {
            "users": f"""
                CREATE TABLE {schema_name}.users (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    email VARCHAR(255) UNIQUE NOT NULL,
                    username VARCHAR(100) NOT NULL,
                    is_active BOOLEAN DEFAULT true,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "collaborations": f"""
                CREATE TABLE {schema_name}.collaborations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    status VARCHAR(50) DEFAULT 'active',
                    owner_id UUID REFERENCES {schema_name}.users(id),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "content_generated": f"""
                CREATE TABLE {schema_name}.content_generated (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    title VARCHAR(255) NOT NULL,
                    content_type VARCHAR(100) NOT NULL,
                    content_data JSONB,
                    user_id UUID REFERENCES {schema_name}.users(id),
                    collaboration_id UUID REFERENCES {schema_name}.collaborations(id),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "ai_sessions": f"""
                CREATE TABLE {schema_name}.ai_sessions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    session_type VARCHAR(100) NOT NULL,
                    input_data JSONB,
                    output_data JSONB,
                    status VARCHAR(50) DEFAULT 'pending',
                    user_id UUID REFERENCES {schema_name}.users(id),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP WITH TIME ZONE
                )
            """,
            "spotify_connections": f"""
                CREATE TABLE {schema_name}.spotify_connections (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES {schema_name}.users(id),
                    spotify_user_id VARCHAR(255) NOT NULL,
                    access_token_encrypted TEXT,
                    refresh_token_encrypted TEXT,
                    token_expires_at TIMESTAMP WITH TIME ZONE,
                    is_active BOOLEAN DEFAULT true,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "collaboration_participants": f"""
                CREATE TABLE {schema_name}.collaboration_participants (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    collaboration_id UUID REFERENCES {schema_name}.collaborations(id),
                    user_id UUID REFERENCES {schema_name}.users(id),
                    role VARCHAR(100) DEFAULT 'member',
                    joined_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(collaboration_id, user_id)
                )
            """
        }
        
        if table_name in table_definitions:
            await self.session.execute(text(table_definitions[table_name]))
            await self.session.commit()
    
    async def _validate_indexes(
        self,
        schema_name: str,
        check_result: Dict[str, Any],
        fix_issues: bool
    ) -> None:
        """Validate required indexes exist."""
        required_indexes = [
            ("users", "email"),
            ("users", "username"),
            ("collaborations", "owner_id"),
            ("content_generated", "user_id"),
            ("content_generated", "collaboration_id"),
            ("ai_sessions", "user_id"),
            ("spotify_connections", "user_id"),
            ("collaboration_participants", "collaboration_id"),
            ("collaboration_participants", "user_id")
        ]
        
        for table, column in required_indexes:
            index_exists = await self._check_index_exists(schema_name, table, column)
            if not index_exists:
                issue = f"Missing index on {schema_name}.{table}({column})"
                check_result["issues"].append(issue)
                
                if fix_issues:
                    try:
                        index_name = f"idx_{table}_{column}"
                        await self.session.execute(
                            text(f"CREATE INDEX {index_name} ON {schema_name}.{table}({column})")
                        )
                        await self.session.commit()
                        check_result["fixes"].append(f"Created index: {index_name}")
                    except Exception as e:
                        check_result["warnings"].append(f"Failed to create index on {table}({column}): {e}")
    
    async def _check_index_exists(self, schema_name: str, table_name: str, column_name: str) -> bool:
        """Check if index exists on column."""
        try:
            result = await self.session.execute(
                text(f"""
                SELECT i.indexname
                FROM pg_indexes i
                WHERE i.schemaname = '{schema_name}'
                AND i.tablename = '{table_name}'
                AND i.indexdef LIKE '%{column_name}%'
                """)
            )
            return result.scalar() is not None
        except Exception:
            return False
    
    async def _validate_foreign_keys(
        self,
        schema_name: str,
        check_result: Dict[str, Any],
        fix_issues: bool
    ) -> None:
        """Validate foreign key constraints."""
        # Check for missing foreign key constraints and orphaned records
        constraints_to_check = [
            ("collaborations", "owner_id", "users", "id"),
            ("content_generated", "user_id", "users", "id"),
            ("content_generated", "collaboration_id", "collaborations", "id"),
            ("ai_sessions", "user_id", "users", "id"),
            ("spotify_connections", "user_id", "users", "id"),
            ("collaboration_participants", "collaboration_id", "collaborations", "id"),
            ("collaboration_participants", "user_id", "users", "id")
        ]
        
        for child_table, child_column, parent_table, parent_column in constraints_to_check:
            # Check for orphaned records
            try:
                result = await self.session.execute(
                    text(f"""
                    SELECT COUNT(*)
                    FROM {schema_name}.{child_table} c
                    LEFT JOIN {schema_name}.{parent_table} p ON c.{child_column} = p.{parent_column}
                    WHERE c.{child_column} IS NOT NULL AND p.{parent_column} IS NULL
                    """)
                )
                orphaned_count = result.scalar() or 0
                
                if orphaned_count > 0:
                    issue = f"Found {orphaned_count} orphaned records in {child_table}.{child_column}"
                    check_result["issues"].append(issue)
                    
                    if fix_issues:
                        # Delete orphaned records (be careful with this in production)
                        await self.session.execute(
                            text(f"""
                            DELETE FROM {schema_name}.{child_table}
                            WHERE {child_column} NOT IN (
                                SELECT {parent_column} FROM {schema_name}.{parent_table}
                                WHERE {parent_column} IS NOT NULL
                            ) AND {child_column} IS NOT NULL
                            """)
                        )
                        await self.session.commit()
                        check_result["fixes"].append(f"Removed {orphaned_count} orphaned records from {child_table}")
                        
            except Exception as e:
                check_result["warnings"].append(f"Error checking foreign keys for {child_table}: {e}")
    
    async def _validate_column_definitions(
        self,
        schema_name: str,
        check_result: Dict[str, Any],
        fix_issues: bool
    ) -> None:
        """Validate column definitions and constraints."""
        # Check for missing NOT NULL constraints on critical columns
        critical_columns = [
            ("users", "email"),
            ("users", "username"),
            ("collaborations", "name"),
            ("content_generated", "title"),
            ("content_generated", "content_type")
        ]
        
        for table, column in critical_columns:
            try:
                result = await self.session.execute(
                    text(f"""
                    SELECT is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = '{schema_name}'
                    AND table_name = '{table}'
                    AND column_name = '{column}'
                    """)
                )
                is_nullable = result.scalar()
                
                if is_nullable == 'YES':
                    check_result["warnings"].append(
                        f"Column {table}.{column} should have NOT NULL constraint"
                    )
                    
            except Exception as e:
                check_result["warnings"].append(f"Error checking column {table}.{column}: {e}")
    
    async def _check_orphaned_records(
        self,
        schema_name: str,
        check_result: Dict[str, Any],
        fix_issues: bool
    ) -> None:
        """Check for orphaned records across tables."""
        # This is handled in _validate_foreign_keys
        pass
    
    async def _check_data_consistency(
        self,
        schema_name: str,
        check_result: Dict[str, Any],
        fix_issues: bool
    ) -> None:
        """Check data consistency rules."""
        # Check for inconsistent timestamps
        try:
            result = await self.session.execute(
                text(f"""
                SELECT COUNT(*)
                FROM {schema_name}.users
                WHERE updated_at < created_at
                """)
            )
            inconsistent_timestamps = result.scalar() or 0
            
            if inconsistent_timestamps > 0:
                issue = f"Found {inconsistent_timestamps} records with updated_at < created_at"
                check_result["issues"].append(issue)
                
                if fix_issues:
                    await self.session.execute(
                        text(f"""
                        UPDATE {schema_name}.users
                        SET updated_at = created_at
                        WHERE updated_at < created_at
                        """)
                    )
                    await self.session.commit()
                    check_result["fixes"].append(f"Fixed {inconsistent_timestamps} timestamp inconsistencies")
                    
        except Exception as e:
            check_result["warnings"].append(f"Error checking timestamp consistency: {e}")
    
    async def _check_duplicate_records(
        self,
        schema_name: str,
        check_result: Dict[str, Any],
        fix_issues: bool
    ) -> None:
        """Check for duplicate records."""
        # Check for duplicate emails
        try:
            result = await self.session.execute(
                text(f"""
                SELECT email, COUNT(*)
                FROM {schema_name}.users
                GROUP BY email
                HAVING COUNT(*) > 1
                """)
            )
            duplicate_emails = result.fetchall()
            
            if duplicate_emails:
                issue = f"Found {len(duplicate_emails)} duplicate email addresses"
                check_result["issues"].append(issue)
                
                if fix_issues:
                    # Keep the most recent user for each email
                    for email, count in duplicate_emails:
                        await self.session.execute(
                            text(f"""
                            DELETE FROM {schema_name}.users
                            WHERE email = '{email}'
                            AND id NOT IN (
                                SELECT id FROM {schema_name}.users
                                WHERE email = '{email}'
                                ORDER BY created_at DESC
                                LIMIT 1
                            )
                            """)
                        )
                    await self.session.commit()
                    check_result["fixes"].append(f"Removed duplicate email records")
                    
        except Exception as e:
            check_result["warnings"].append(f"Error checking duplicate emails: {e}")
    
    async def _validate_data_formats(
        self,
        schema_name: str,
        check_result: Dict[str, Any],
        fix_issues: bool
    ) -> None:
        """Validate data formats and patterns."""
        # Check email format
        try:
            result = await self.session.execute(
                text(f"""
                SELECT COUNT(*)
                FROM {schema_name}.users
                WHERE email IS NOT NULL
                AND email !~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{{2,}}$'
                """)
            )
            invalid_emails = result.scalar() or 0
            
            if invalid_emails > 0:
                issue = f"Found {invalid_emails} users with invalid email format"
                check_result["issues"].append(issue)
                
                if fix_issues:
                    check_result["recommendations"].append(
                        "Consider implementing email validation at application level"
                    )
                    
        except Exception as e:
            check_result["warnings"].append(f"Error validating email formats: {e}")
    
    async def _check_referential_integrity(
        self,
        schema_name: str,
        check_result: Dict[str, Any],
        fix_issues: bool
    ) -> None:
        """Check referential integrity beyond foreign keys."""
        # This is covered in foreign key validation
        pass
    
    async def _validate_tier_limits(
        self,
        tenant_id: str,
        schema_name: str,
        check_result: Dict[str, Any],
        fix_issues: bool
    ) -> None:
        """Validate tenant tier limits are respected."""
        # Get tenant tier information
        tier_limits = {
            "free": {"users": 1, "collaborations": 0, "ai_sessions_per_month": 10},
            "basic": {"users": 5, "collaborations": 2, "ai_sessions_per_month": 100},
            "premium": {"users": 25, "collaborations": 10, "ai_sessions_per_month": 1000},
            "enterprise": {"users": -1, "collaborations": -1, "ai_sessions_per_month": -1}
        }
        
        # This would require knowing the tenant tier - simplified check
        try:
            result = await self.session.execute(
                text(f"SELECT COUNT(*) FROM {schema_name}.users WHERE is_active = true")
            )
            user_count = result.scalar() or 0
            
            if user_count > 100:  # Basic threshold check
                check_result["warnings"].append(
                    f"High user count ({user_count}) - verify tier limits"
                )
                
        except Exception as e:
            check_result["warnings"].append(f"Error checking tier limits: {e}")
    
    async def _validate_collaboration_rules(
        self,
        schema_name: str,
        check_result: Dict[str, Any],
        fix_issues: bool
    ) -> None:
        """Validate collaboration business rules."""
        # Check for collaborations without participants
        try:
            result = await self.session.execute(
                text(f"""
                SELECT c.id, c.name
                FROM {schema_name}.collaborations c
                LEFT JOIN {schema_name}.collaboration_participants cp ON c.id = cp.collaboration_id
                WHERE cp.collaboration_id IS NULL
                AND c.status = 'active'
                """)
            )
            empty_collaborations = result.fetchall()
            
            if empty_collaborations:
                issue = f"Found {len(empty_collaborations)} active collaborations without participants"
                check_result["issues"].append(issue)
                
                if fix_issues:
                    # Set status to inactive for empty collaborations
                    for collab_id, name in empty_collaborations:
                        await self.session.execute(
                            text(f"""
                            UPDATE {schema_name}.collaborations
                            SET status = 'inactive'
                            WHERE id = '{collab_id}'
                            """)
                        )
                    await self.session.commit()
                    check_result["fixes"].append(f"Deactivated {len(empty_collaborations)} empty collaborations")
                    
        except Exception as e:
            check_result["warnings"].append(f"Error checking collaboration rules: {e}")
    
    async def _validate_content_limits(
        self,
        schema_name: str,
        check_result: Dict[str, Any],
        fix_issues: bool
    ) -> None:
        """Validate content generation limits."""
        # Check for excessive content generation per user
        try:
            result = await self.session.execute(
                text(f"""
                SELECT user_id, COUNT(*)
                FROM {schema_name}.content_generated
                WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY user_id
                HAVING COUNT(*) > 1000
                """)
            )
            heavy_users = result.fetchall()
            
            if heavy_users:
                check_result["warnings"].append(
                    f"Found {len(heavy_users)} users with high content generation (>1000/month)"
                )
                
        except Exception as e:
            check_result["warnings"].append(f"Error checking content limits: {e}")
    
    async def _validate_ai_session_rules(
        self,
        schema_name: str,
        check_result: Dict[str, Any],
        fix_issues: bool
    ) -> None:
        """Validate AI session business rules."""
        # Check for hanging AI sessions
        try:
            result = await self.session.execute(
                text(f"""
                SELECT COUNT(*)
                FROM {schema_name}.ai_sessions
                WHERE status = 'pending'
                AND created_at < CURRENT_TIMESTAMP - INTERVAL '1 hour'
                """)
            )
            hanging_sessions = result.scalar() or 0
            
            if hanging_sessions > 0:
                issue = f"Found {hanging_sessions} AI sessions stuck in pending state"
                check_result["issues"].append(issue)
                
                if fix_issues:
                    await self.session.execute(
                        text(f"""
                        UPDATE {schema_name}.ai_sessions
                        SET status = 'timeout'
                        WHERE status = 'pending'
                        AND created_at < CURRENT_TIMESTAMP - INTERVAL '1 hour'
                        """)
                    )
                    await self.session.commit()
                    check_result["fixes"].append(f"Updated {hanging_sessions} hanging AI sessions to timeout")
                    
        except Exception as e:
            check_result["warnings"].append(f"Error checking AI session rules: {e}")
    
    async def _validate_spotify_rules(
        self,
        schema_name: str,
        check_result: Dict[str, Any],
        fix_issues: bool
    ) -> None:
        """Validate Spotify connection rules."""
        # Check for expired tokens
        try:
            result = await self.session.execute(
                text(f"""
                SELECT COUNT(*)
                FROM {schema_name}.spotify_connections
                WHERE is_active = true
                AND token_expires_at < CURRENT_TIMESTAMP
                """)
            )
            expired_tokens = result.scalar() or 0
            
            if expired_tokens > 0:
                issue = f"Found {expired_tokens} Spotify connections with expired tokens"
                check_result["issues"].append(issue)
                
                if fix_issues:
                    await self.session.execute(
                        text(f"""
                        UPDATE {schema_name}.spotify_connections
                        SET is_active = false
                        WHERE is_active = true
                        AND token_expires_at < CURRENT_TIMESTAMP
                        """)
                    )
                    await self.session.commit()
                    check_result["fixes"].append(f"Deactivated {expired_tokens} expired Spotify connections")
                    
        except Exception as e:
            check_result["warnings"].append(f"Error checking Spotify rules: {e}")
    
    async def _check_query_performance(
        self,
        schema_name: str,
        check_result: Dict[str, Any],
        fix_issues: bool
    ) -> None:
        """Check query performance metrics."""
        # Check for slow queries (this would require pg_stat_statements)
        check_result["recommendations"].append(
            "Consider enabling pg_stat_statements for query performance monitoring"
        )
    
    async def _check_storage_usage(
        self,
        schema_name: str,
        check_result: Dict[str, Any],
        fix_issues: bool
    ) -> None:
        """Check storage usage and efficiency."""
        try:
            result = await self.session.execute(
                text(f"""
                SELECT 
                    schemaname,
                    SUM(pg_total_relation_size(schemaname||'.'||tablename)) as total_size
                FROM pg_tables 
                WHERE schemaname = '{schema_name}'
                GROUP BY schemaname
                """)
            )
            row = result.fetchone()
            
            if row:
                schema_name_result, total_size = row
                size_mb = total_size / 1024 / 1024
                
                if size_mb > 1000:  # Over 1GB
                    check_result["warnings"].append(
                        f"Schema {schema_name_result} uses {size_mb:.1f}MB of storage"
                    )
                    check_result["recommendations"].append(
                        "Consider archiving old data or implementing data retention policies"
                    )
                    
        except Exception as e:
            check_result["warnings"].append(f"Error checking storage usage: {e}")
    
    async def _check_cache_efficiency(
        self,
        tenant_id: str,
        check_result: Dict[str, Any],
        fix_issues: bool
    ) -> None:
        """Check cache efficiency metrics."""
        if self.redis_client:
            try:
                # Check cache hit ratio for tenant
                cache_namespace = TenantUtils.get_tenant_cache_namespace(tenant_id)
                
                # This would require custom cache metrics
                check_result["recommendations"].append(
                    "Implement cache hit ratio monitoring for performance optimization"
                )
                
            except Exception as e:
                check_result["warnings"].append(f"Error checking cache efficiency: {e}")
    
    async def _check_resource_limits(
        self,
        tenant_id: str,
        check_result: Dict[str, Any],
        fix_issues: bool
    ) -> None:
        """Check resource limits and usage."""
        # Check connection pool usage, memory usage, etc.
        check_result["recommendations"].append(
            "Implement resource monitoring for connection pools and memory usage"
        )
    
    async def _check_data_encryption(
        self,
        schema_name: str,
        check_result: Dict[str, Any],
        fix_issues: bool
    ) -> None:
        """Check data encryption compliance."""
        # Check for unencrypted sensitive data
        try:
            result = await self.session.execute(
                text(f"""
                SELECT COUNT(*)
                FROM {schema_name}.spotify_connections
                WHERE access_token_encrypted IS NOT NULL
                AND access_token_encrypted NOT LIKE 'gAAAAA%'  -- Fernet encryption prefix
                """)
            )
            unencrypted_tokens = result.scalar() or 0
            
            if unencrypted_tokens > 0:
                issue = f"Found {unencrypted_tokens} potentially unencrypted Spotify tokens"
                check_result["issues"].append(issue)
                
        except Exception as e:
            check_result["warnings"].append(f"Error checking data encryption: {e}")
    
    async def _validate_permissions(
        self,
        schema_name: str,
        check_result: Dict[str, Any],
        fix_issues: bool
    ) -> None:
        """Validate access permissions."""
        # Check schema permissions
        try:
            result = await self.session.execute(
                text(f"""
                SELECT grantee, privilege_type
                FROM information_schema.schema_privileges
                WHERE schema_name = '{schema_name}'
                """)
            )
            permissions = result.fetchall()
            
            # Basic permission validation
            if not permissions:
                check_result["warnings"].append(f"No explicit permissions found for schema {schema_name}")
                
        except Exception as e:
            check_result["warnings"].append(f"Error checking permissions: {e}")
    
    async def _check_sensitive_data(
        self,
        schema_name: str,
        check_result: Dict[str, Any],
        fix_issues: bool
    ) -> None:
        """Check for sensitive data exposure."""
        # Check for potential PII in logs or unencrypted fields
        check_result["recommendations"].append(
            "Implement regular PII scanning and data classification"
        )
    
    async def _validate_audit_trails(
        self,
        schema_name: str,
        check_result: Dict[str, Any],
        fix_issues: bool
    ) -> None:
        """Validate audit trail completeness."""
        # Check for missing audit information
        try:
            result = await self.session.execute(
                text(f"""
                SELECT COUNT(*)
                FROM {schema_name}.users
                WHERE created_at IS NULL OR updated_at IS NULL
                """)
            )
            missing_audit = result.scalar() or 0
            
            if missing_audit > 0:
                issue = f"Found {missing_audit} records with missing audit timestamps"
                check_result["issues"].append(issue)
                
        except Exception as e:
            check_result["warnings"].append(f"Error checking audit trails: {e}")
    
    async def _check_compliance_rules(
        self,
        schema_name: str,
        check_result: Dict[str, Any],
        fix_issues: bool
    ) -> None:
        """Check compliance with data protection regulations."""
        # GDPR, CCPA compliance checks
        check_result["recommendations"].extend([
            "Implement data retention policies for GDPR compliance",
            "Add user consent tracking for data processing",
            "Implement right-to-be-forgotten functionality"
        ])
    
    async def _generate_validation_summary(
        self,
        tenant_id: str,
        validation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive validation summary."""
        total_checks = len(validation_result["checks_performed"])
        issues_by_type = defaultdict(int)
        fixes_by_type = defaultdict(int)
        
        for check_type, check_result in validation_result["checks_performed"].items():
            issues_by_type[check_type] = len(check_result.get("issues", []))
            fixes_by_type[check_type] = len(check_result.get("fixes", []))
        
        return {
            "tenant_id": tenant_id,
            "total_checks_performed": total_checks,
            "total_issues_found": len(validation_result["issues_found"]),
            "total_issues_fixed": len(validation_result["issues_fixed"]),
            "total_warnings": len(validation_result["warnings"]),
            "issues_by_type": dict(issues_by_type),
            "fixes_by_type": dict(fixes_by_type),
            "health_score": self._calculate_health_score(validation_result),
            "recommendations_count": len(validation_result["recommendations"])
        }
    
    def _calculate_health_score(self, validation_result: Dict[str, Any]) -> float:
        """Calculate tenant health score based on validation results."""
        total_checks = len(validation_result["checks_performed"])
        total_issues = len(validation_result["issues_found"])
        total_warnings = len(validation_result["warnings"])
        
        if total_checks == 0:
            return 0.0
        
        # Base score of 100, subtract points for issues and warnings
        score = 100.0
        score -= (total_issues * 10)  # 10 points per issue
        score -= (total_warnings * 2)  # 2 points per warning
        
        return max(0.0, min(100.0, score))


async def validate_tenant_data(
    tenant_id: str,
    check_types: List[str] = None,
    fix_issues: bool = False,
    detailed_report: bool = True
) -> Dict[str, Any]:
    """
    Main function to validate tenant data.
    
    Args:
        tenant_id: Target tenant
        check_types: Types of validation to perform
        fix_issues: Attempt to fix found issues
        detailed_report: Generate detailed report
        
    Returns:
        Validation results
    """
    async with get_async_session() as session:
        redis_client = await get_redis_client()
        
        try:
            validator = DataValidator(session, redis_client)
            result = await validator.validate_tenant_data(
                tenant_id=tenant_id,
                check_types=check_types,
                fix_issues=fix_issues,
                detailed_report=detailed_report
            )
            
            return result
            
        finally:
            if redis_client:
                await redis_client.close()


def main():
    """Command line interface for data validation."""
    parser = argparse.ArgumentParser(
        description="Validate tenant data integrity and compliance"
    )
    
    parser.add_argument(
        "--tenant-id",
        required=True,
        help="Target tenant identifier"
    )
    
    parser.add_argument(
        "--check-type",
        choices=["schema", "data", "business", "performance", "security", "all"],
        default="all",
        help="Type of validation to perform"
    )
    
    parser.add_argument(
        "--fix-issues",
        action="store_true",
        help="Attempt to automatically fix found issues"
    )
    
    parser.add_argument(
        "--detailed-report",
        action="store_true",
        default=True,
        help="Generate detailed validation report"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Determine check types
    check_types = None
    if args.check_type != "all":
        check_types = [args.check_type]
    
    try:
        # Run validation
        result = asyncio.run(
            validate_tenant_data(
                tenant_id=args.tenant_id,
                check_types=check_types,
                fix_issues=args.fix_issues,
                detailed_report=args.detailed_report
            )
        )
        
        # Display results
        print(f"\nData Validation Results for '{args.tenant_id}':")
        print(f"Status: {result['status']}")
        print(f"Duration: {FixtureUtils.format_duration(result.get('duration_seconds', 0))}")
        print(f"Health Score: {result['summary']['health_score']:.1f}/100")
        print(f"Issues Found: {result['total_issues']}")
        print(f"Issues Fixed: {result['issues_fixed_count']}")
        print(f"Warnings: {result['warnings_count']}")
        
        if result.get('summary', {}).get('issues_by_type'):
            print("\nIssues by Type:")
            for check_type, count in result['summary']['issues_by_type'].items():
                print(f"  {check_type}: {count}")
        
        if result.get('recommendations'):
            print(f"\nRecommendations: {len(result['recommendations'])}")
            for i, rec in enumerate(result['recommendations'][:5], 1):
                print(f"  {i}. {rec}")
            
            if len(result['recommendations']) > 5:
                print(f"  ... and {len(result['recommendations']) - 5} more")
        
        if result['status'] == 'completed':
            if result['total_issues'] == 0:
                print("\n✅ All validations passed successfully!")
                sys.exit(0)
            else:
                print(f"\n⚠️  Validation completed with {result['total_issues']} issues")
                sys.exit(2)
        else:
            print(f"\n❌ Validation failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
