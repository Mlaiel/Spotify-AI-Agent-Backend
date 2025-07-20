"""
Spotify AI Agent - Base Fixture Infrastructure
=============================================

Provides foundational classes and interfaces for all fixture operations
in the multi-tenant Spotify AI Agent system.

This module implements the core architecture for:
- Base fixture classes and interfaces
- Common fixture operations
- Data validation frameworks
- Performance monitoring hooks
- Error handling and recovery
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Generic, TypeVar
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field, validator

from app.core.database import get_async_session
from app.core.exceptions import BaseApplicationException
from app.tenancy.fixtures.exceptions import (
    FixtureError,
    FixtureValidationError,
    FixtureTimeoutError,
    FixtureConflictError
)
from app.tenancy.fixtures.constants import (
    DEFAULT_BATCH_SIZE,
    MAX_CONCURRENT_OPERATIONS,
    FIXTURE_CACHE_TTL,
    VALIDATION_TIMEOUT
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class FixtureStatus(Enum):
    """Fixture operation status enumeration."""
    PENDING = "pending"
    RUNNING = "running" 
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class FixtureType(Enum):
    """Types of fixtures supported by the system."""
    TENANT = "tenant"
    SCHEMA = "schema"
    CONFIG = "config"
    DATA = "data"
    VALIDATION = "validation"
    MONITORING = "monitoring"


@dataclass
class FixtureMetadata:
    """Metadata for fixture operations."""
    fixture_id: UUID = field(default_factory=uuid4)
    fixture_type: FixtureType
    tenant_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    dependencies: Set[UUID] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.updated_at is None:
            self.updated_at = self.created_at


@dataclass 
class FixtureResult:
    """Result of a fixture operation."""
    fixture_id: UUID
    status: FixtureStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    records_processed: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate duration if end_time is set."""
        if self.end_time and self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()


class FixtureConfig(BaseModel):
    """Configuration for fixture operations."""
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=1, le=10000)
    max_concurrent: int = Field(default=MAX_CONCURRENT_OPERATIONS, ge=1, le=50)
    timeout: int = Field(default=VALIDATION_TIMEOUT, ge=30, le=3600)
    enable_validation: bool = Field(default=True)
    enable_monitoring: bool = Field(default=True)
    enable_rollback: bool = Field(default=True)
    dry_run: bool = Field(default=False)
    
    @validator('timeout')
    def validate_timeout(cls, v):
        """Validate timeout value."""
        if v < 30:
            raise ValueError("Timeout must be at least 30 seconds")
        return v


class BaseFixture(ABC, Generic[T]):
    """
    Abstract base class for all fixture implementations.
    
    Provides common functionality for:
    - Lifecycle management
    - Error handling
    - Performance monitoring
    - Validation
    """
    
    def __init__(
        self,
        metadata: FixtureMetadata,
        config: Optional[FixtureConfig] = None,
        session: Optional[AsyncSession] = None
    ):
        self.metadata = metadata
        self.config = config or FixtureConfig()
        self.session = session
        self._start_time: Optional[datetime] = None
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        # Performance tracking
        self._records_processed = 0
        self._errors: List[str] = []
        self._warnings: List[str] = []
        
        logger.info(
            f"Initialized fixture {self.metadata.name} "
            f"(ID: {self.metadata.fixture_id})"
        )
    
    @abstractmethod
    async def validate(self) -> bool:
        """
        Validate fixture data and dependencies.
        
        Returns:
            bool: True if validation passes
            
        Raises:
            FixtureValidationError: If validation fails
        """
        pass
    
    @abstractmethod
    async def apply(self) -> T:
        """
        Apply the fixture operation.
        
        Returns:
            T: Result of the fixture operation
            
        Raises:
            FixtureError: If operation fails
        """
        pass
    
    @abstractmethod
    async def rollback(self) -> bool:
        """
        Rollback the fixture operation.
        
        Returns:
            bool: True if rollback successful
            
        Raises:
            FixtureError: If rollback fails
        """
        pass
    
    async def execute(self) -> FixtureResult:
        """
        Execute the complete fixture workflow.
        
        Returns:
            FixtureResult: Execution result with metrics
        """
        self._start_time = datetime.now(timezone.utc)
        result = FixtureResult(
            fixture_id=self.metadata.fixture_id,
            status=FixtureStatus.RUNNING,
            start_time=self._start_time
        )
        
        try:
            logger.info(f"Starting fixture execution: {self.metadata.name}")
            
            # Validation phase
            if self.config.enable_validation:
                logger.debug(f"Validating fixture: {self.metadata.name}")
                validation_success = await asyncio.wait_for(
                    self.validate(),
                    timeout=self.config.timeout
                )
                if not validation_success:
                    raise FixtureValidationError(
                        f"Validation failed for fixture: {self.metadata.name}"
                    )
            
            # Application phase
            if not self.config.dry_run:
                logger.debug(f"Applying fixture: {self.metadata.name}")
                await asyncio.wait_for(
                    self.apply(),
                    timeout=self.config.timeout
                )
            else:
                logger.info(f"Dry run mode - skipping application: {self.metadata.name}")
            
            # Success
            result.status = FixtureStatus.SUCCESS
            result.records_processed = self._records_processed
            logger.info(
                f"Fixture execution completed successfully: {self.metadata.name} "
                f"({result.records_processed} records processed)"
            )
            
        except asyncio.TimeoutError:
            result.status = FixtureStatus.TIMEOUT
            result.errors.append(f"Fixture execution timed out after {self.config.timeout}s")
            logger.error(f"Fixture execution timeout: {self.metadata.name}")
            
            # Attempt rollback on timeout
            if self.config.enable_rollback:
                await self._safe_rollback()
                
        except FixtureValidationError as e:
            result.status = FixtureStatus.FAILED
            result.errors.append(str(e))
            logger.error(f"Fixture validation error: {self.metadata.name} - {e}")
            
        except Exception as e:
            result.status = FixtureStatus.FAILED
            result.errors.append(str(e))
            logger.error(f"Fixture execution error: {self.metadata.name} - {e}")
            
            # Attempt rollback on error
            if self.config.enable_rollback:
                await self._safe_rollback()
                
        finally:
            result.end_time = datetime.now(timezone.utc)
            result.duration = (result.end_time - result.start_time).total_seconds()
            result.errors.extend(self._errors)
            result.warnings.extend(self._warnings)
            
            # Update metadata
            self.metadata.updated_at = result.end_time
            
        return result
    
    async def _safe_rollback(self) -> None:
        """Safely attempt rollback without raising exceptions."""
        try:
            logger.info(f"Attempting rollback for fixture: {self.metadata.name}")
            await self.rollback()
            logger.info(f"Rollback successful for fixture: {self.metadata.name}")
        except Exception as e:
            logger.error(f"Rollback failed for fixture: {self.metadata.name} - {e}")
            self._errors.append(f"Rollback failed: {e}")
    
    def add_error(self, error: str) -> None:
        """Add an error to the fixture execution."""
        self._errors.append(error)
        logger.error(f"Fixture error ({self.metadata.name}): {error}")
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the fixture execution."""
        self._warnings.append(warning)
        logger.warning(f"Fixture warning ({self.metadata.name}): {warning}")
    
    def increment_processed(self, count: int = 1) -> None:
        """Increment the count of processed records."""
        self._records_processed += count
    
    async def get_session(self) -> AsyncSession:
        """Get or create a database session."""
        if self.session is None:
            self.session = await get_async_session()
        return self.session


class FixtureManager:
    """
    Central manager for all fixture operations.
    
    Provides:
    - Fixture registration and discovery
    - Dependency resolution
    - Parallel execution
    - Progress tracking
    - Error handling
    """
    
    def __init__(self):
        self._fixtures: Dict[UUID, BaseFixture] = {}
        self._results: Dict[UUID, FixtureResult] = {}
        self._execution_order: List[UUID] = []
        
        logger.info("FixtureManager initialized")
    
    def register_fixture(self, fixture: BaseFixture) -> UUID:
        """
        Register a fixture for execution.
        
        Args:
            fixture: Fixture instance to register
            
        Returns:
            UUID: Fixture ID
        """
        fixture_id = fixture.metadata.fixture_id
        self._fixtures[fixture_id] = fixture
        
        logger.info(
            f"Registered fixture: {fixture.metadata.name} "
            f"(ID: {fixture_id})"
        )
        
        return fixture_id
    
    def get_fixture(self, fixture_id: UUID) -> Optional[BaseFixture]:
        """Get a registered fixture by ID."""
        return self._fixtures.get(fixture_id)
    
    def get_result(self, fixture_id: UUID) -> Optional[FixtureResult]:
        """Get execution result for a fixture."""
        return self._results.get(fixture_id)
    
    async def execute_fixture(self, fixture_id: UUID) -> FixtureResult:
        """
        Execute a single fixture.
        
        Args:
            fixture_id: ID of fixture to execute
            
        Returns:
            FixtureResult: Execution result
            
        Raises:
            FixtureError: If fixture not found or execution fails
        """
        fixture = self.get_fixture(fixture_id)
        if not fixture:
            raise FixtureError(f"Fixture not found: {fixture_id}")
        
        result = await fixture.execute()
        self._results[fixture_id] = result
        
        return result
    
    async def execute_all(
        self,
        parallel: bool = True,
        stop_on_error: bool = False
    ) -> Dict[UUID, FixtureResult]:
        """
        Execute all registered fixtures.
        
        Args:
            parallel: Execute fixtures in parallel
            stop_on_error: Stop execution on first error
            
        Returns:
            Dict[UUID, FixtureResult]: Results by fixture ID
        """
        if not self._fixtures:
            logger.warning("No fixtures registered for execution")
            return {}
        
        # Resolve execution order based on dependencies
        execution_order = self._resolve_dependencies()
        
        logger.info(f"Executing {len(execution_order)} fixtures")
        
        if parallel:
            return await self._execute_parallel(execution_order, stop_on_error)
        else:
            return await self._execute_sequential(execution_order, stop_on_error)
    
    def _resolve_dependencies(self) -> List[UUID]:
        """
        Resolve fixture dependencies and return execution order.
        
        Returns:
            List[UUID]: Fixture IDs in execution order
            
        Raises:
            FixtureError: If circular dependencies detected
        """
        # Topological sort for dependency resolution
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(fixture_id: UUID):
            if fixture_id in temp_visited:
                raise FixtureError(f"Circular dependency detected: {fixture_id}")
            
            if fixture_id not in visited:
                temp_visited.add(fixture_id)
                fixture = self._fixtures[fixture_id]
                
                for dep_id in fixture.metadata.dependencies:
                    if dep_id in self._fixtures:
                        visit(dep_id)
                
                temp_visited.remove(fixture_id)
                visited.add(fixture_id)
                order.append(fixture_id)
        
        for fixture_id in self._fixtures:
            if fixture_id not in visited:
                visit(fixture_id)
        
        logger.info(f"Resolved dependency order for {len(order)} fixtures")
        return order
    
    async def _execute_sequential(
        self,
        execution_order: List[UUID],
        stop_on_error: bool
    ) -> Dict[UUID, FixtureResult]:
        """Execute fixtures sequentially."""
        results = {}
        
        for fixture_id in execution_order:
            try:
                result = await self.execute_fixture(fixture_id)
                results[fixture_id] = result
                
                if stop_on_error and result.status == FixtureStatus.FAILED:
                    logger.error(f"Stopping execution due to error in fixture: {fixture_id}")
                    break
                    
            except Exception as e:
                logger.error(f"Failed to execute fixture {fixture_id}: {e}")
                if stop_on_error:
                    break
        
        return results
    
    async def _execute_parallel(
        self,
        execution_order: List[UUID],
        stop_on_error: bool
    ) -> Dict[UUID, FixtureResult]:
        """Execute fixtures in parallel respecting dependencies."""
        results = {}
        completed = set()
        
        # Group fixtures by dependency level
        levels = self._group_by_dependency_level(execution_order)
        
        for level, fixture_ids in levels.items():
            logger.info(f"Executing dependency level {level} ({len(fixture_ids)} fixtures)")
            
            # Execute all fixtures at this level in parallel
            tasks = [
                self.execute_fixture(fixture_id)
                for fixture_id in fixture_ids
            ]
            
            level_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for fixture_id, result in zip(fixture_ids, level_results):
                if isinstance(result, Exception):
                    logger.error(f"Fixture {fixture_id} failed: {result}")
                    if stop_on_error:
                        return results
                else:
                    results[fixture_id] = result
                    completed.add(fixture_id)
        
        return results
    
    def _group_by_dependency_level(
        self,
        execution_order: List[UUID]
    ) -> Dict[int, List[UUID]]:
        """Group fixtures by dependency level for parallel execution."""
        levels = {}
        fixture_levels = {}
        
        # Calculate dependency level for each fixture
        for fixture_id in execution_order:
            fixture = self._fixtures[fixture_id]
            level = 0
            
            for dep_id in fixture.metadata.dependencies:
                if dep_id in fixture_levels:
                    level = max(level, fixture_levels[dep_id] + 1)
            
            fixture_levels[fixture_id] = level
            
            if level not in levels:
                levels[level] = []
            levels[level].append(fixture_id)
        
        return levels
    
    def clear_fixtures(self) -> None:
        """Clear all registered fixtures and results."""
        self._fixtures.clear()
        self._results.clear()
        self._execution_order.clear()
        logger.info("Cleared all fixtures and results")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary statistics."""
        total = len(self._results)
        if total == 0:
            return {"total": 0, "status": "no_executions"}
        
        status_counts = {}
        total_duration = 0
        total_records = 0
        
        for result in self._results.values():
            status = result.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
            if result.duration:
                total_duration += result.duration
            total_records += result.records_processed
        
        return {
            "total": total,
            "status_breakdown": status_counts,
            "total_duration": total_duration,
            "total_records_processed": total_records,
            "average_duration": total_duration / total if total > 0 else 0
        }
