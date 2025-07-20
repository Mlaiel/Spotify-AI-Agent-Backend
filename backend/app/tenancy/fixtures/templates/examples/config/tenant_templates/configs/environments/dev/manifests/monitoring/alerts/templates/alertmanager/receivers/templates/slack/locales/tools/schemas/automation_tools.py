"""
Advanced automation and tooling schemas for enterprise-grade operations.

This module provides comprehensive schemas for managing automation workflows,
deployment pipelines, monitoring tools, and operational procedures.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import SecretStr, EmailStr


class ToolCategory(str, Enum):
    """Categories of automation tools."""
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    SECURITY = "security"
    BACKUP = "backup"
    MAINTENANCE = "maintenance"
    ANALYTICS = "analytics"
    TESTING = "testing"
    NOTIFICATION = "notification"
    WORKFLOW = "workflow"
    INTEGRATION = "integration"


class ExecutionStrategy(str, Enum):
    """Tool execution strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    SCHEDULED = "scheduled"
    EVENT_DRIVEN = "event_driven"
    MANUAL = "manual"


class ToolStatus(str, Enum):
    """Tool execution status."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    RETRYING = "retrying"


class Priority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    URGENT = "urgent"


class ToolConfigurationSchema(BaseModel):
    """Base schema for tool configuration."""
    tool_id: UUID = Field(..., description="Unique tool identifier")
    name: str = Field(..., min_length=3, max_length=100, description="Tool name")
    description: str = Field(..., max_length=500, description="Tool description")
    category: ToolCategory = Field(..., description="Tool category")
    version: str = Field(..., description="Tool version")
    
    # Execution configuration
    executable_path: str = Field(..., description="Path to executable or script")
    working_directory: str = Field(".", description="Working directory for execution")
    environment_variables: Dict[str, str] = Field({}, description="Environment variables")
    command_arguments: List[str] = Field([], description="Command line arguments")
    
    # Resource requirements
    cpu_limit: Optional[str] = Field(None, description="CPU limit (e.g., '1.0', '500m')")
    memory_limit: Optional[str] = Field(None, description="Memory limit (e.g., '1Gi', '512Mi')")
    timeout_seconds: int = Field(3600, ge=1, description="Execution timeout in seconds")
    
    # Dependencies
    dependencies: List[str] = Field([], description="Tool dependencies")
    prerequisites: List[str] = Field([], description="Prerequisites for execution")
    
    # Security settings
    run_as_user: Optional[str] = Field(None, description="User to run tool as")
    required_permissions: List[str] = Field([], description="Required permissions")
    sandbox_enabled: bool = Field(True, description="Enable sandboxed execution")
    
    # Metadata
    tags: Dict[str, str] = Field({}, description="Tool tags for categorization")
    owner: EmailStr = Field(..., description="Tool owner")
    maintainers: List[EmailStr] = Field([], description="Tool maintainers")
    
    # Status
    enabled: bool = Field(True, description="Tool is enabled")
    deprecated: bool = Field(False, description="Tool is deprecated")
    
    class Config:
        schema_extra = {
            "example": {
                "tool_id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "Database Backup Tool",
                "description": "Automated database backup with compression and encryption",
                "category": "backup",
                "version": "2.1.0",
                "executable_path": "/usr/local/bin/db-backup.sh",
                "working_directory": "/opt/backup",
                "environment_variables": {
                    "DB_HOST": "localhost",
                    "BACKUP_DIR": "/backups"
                },
                "command_arguments": ["--compress", "--encrypt"],
                "cpu_limit": "500m",
                "memory_limit": "1Gi",
                "timeout_seconds": 7200,
                "dependencies": ["postgresql-client", "gpg"],
                "owner": "dba-team@spotify.com",
                "tags": {
                    "criticality": "high",
                    "frequency": "daily"
                }
            }
        }


class WorkflowStepSchema(BaseModel):
    """Schema for individual workflow steps."""
    step_id: UUID = Field(..., description="Unique step identifier")
    name: str = Field(..., description="Step name")
    description: str = Field(..., description="Step description")
    tool_id: UUID = Field(..., description="Tool to execute for this step")
    
    # Execution settings
    retry_count: int = Field(3, ge=0, le=10, description="Number of retries on failure")
    retry_delay_seconds: int = Field(60, ge=1, description="Delay between retries")
    continue_on_failure: bool = Field(False, description="Continue workflow on step failure")
    
    # Conditional execution
    conditions: List[Dict[str, Any]] = Field([], description="Conditions for step execution")
    skip_if: Optional[str] = Field(None, description="Skip condition expression")
    
    # Input/Output
    input_mapping: Dict[str, str] = Field({}, description="Input parameter mapping")
    output_mapping: Dict[str, str] = Field({}, description="Output parameter mapping")
    
    # Timing
    max_duration_seconds: Optional[int] = Field(None, description="Maximum step duration")
    
    class Config:
        schema_extra = {
            "example": {
                "step_id": "660e8400-e29b-41d4-a716-446655440001",
                "name": "Backup Database",
                "description": "Create encrypted backup of production database",
                "tool_id": "550e8400-e29b-41d4-a716-446655440000",
                "retry_count": 3,
                "retry_delay_seconds": 60,
                "continue_on_failure": False,
                "conditions": [
                    {"type": "schedule", "value": "daily"},
                    {"type": "health_check", "value": "database_healthy"}
                ],
                "input_mapping": {
                    "database_name": "production_db",
                    "backup_location": "/backups/daily"
                },
                "output_mapping": {
                    "backup_file": "backup_path",
                    "backup_size": "file_size"
                }
            }
        }


class WorkflowSchema(BaseModel):
    """Schema for automation workflows."""
    workflow_id: UUID = Field(..., description="Unique workflow identifier")
    name: str = Field(..., min_length=3, max_length=100, description="Workflow name")
    description: str = Field(..., max_length=1000, description="Workflow description")
    category: ToolCategory = Field(..., description="Workflow category")
    
    # Execution configuration
    execution_strategy: ExecutionStrategy = Field(..., description="Execution strategy")
    priority: Priority = Field(Priority.MEDIUM, description="Workflow priority")
    
    # Steps
    steps: List[WorkflowStepSchema] = Field(..., description="Workflow steps")
    
    # Scheduling
    schedule_enabled: bool = Field(False, description="Enable scheduled execution")
    schedule_cron: Optional[str] = Field(None, description="Cron expression for scheduling")
    schedule_timezone: str = Field("UTC", description="Timezone for scheduling")
    
    # Triggers
    event_triggers: List[str] = Field([], description="Event triggers for workflow")
    webhook_triggers: List[str] = Field([], description="Webhook triggers")
    
    # Notifications
    notify_on_start: bool = Field(False, description="Notify on workflow start")
    notify_on_completion: bool = Field(True, description="Notify on workflow completion")
    notify_on_failure: bool = Field(True, description="Notify on workflow failure")
    notification_channels: List[str] = Field([], description="Notification channels")
    
    # Resource limits
    max_concurrent_executions: int = Field(1, ge=1, description="Maximum concurrent executions")
    execution_timeout_seconds: int = Field(7200, ge=60, description="Total execution timeout")
    
    # Metadata
    tags: Dict[str, str] = Field({}, description="Workflow tags")
    owner: EmailStr = Field(..., description="Workflow owner")
    approvers: List[EmailStr] = Field([], description="Required approvers for execution")
    
    # Status
    enabled: bool = Field(True, description="Workflow is enabled")
    approval_required: bool = Field(False, description="Require approval before execution")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_executed_at: Optional[datetime] = Field(None, description="Last execution timestamp")
    
    @validator('schedule_cron')
    def validate_cron_expression(cls, v):
        """Validate cron expression format."""
        if v is not None:
            # Basic cron validation (simplified)
            parts = v.split()
            if len(parts) not in [5, 6]:  # Standard or with seconds
                raise ValueError("Invalid cron expression format")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "workflow_id": "770e8400-e29b-41d4-a716-446655440002",
                "name": "Daily Backup Workflow",
                "description": "Automated daily backup with health checks and notifications",
                "category": "backup",
                "execution_strategy": "sequential",
                "priority": "high",
                "schedule_enabled": True,
                "schedule_cron": "0 2 * * *",
                "schedule_timezone": "UTC",
                "notify_on_completion": True,
                "notify_on_failure": True,
                "notification_channels": ["#ops-alerts", "ops@spotify.com"],
                "owner": "platform-team@spotify.com",
                "enabled": True
            }
        }


class ExecutionContextSchema(BaseModel):
    """Schema for workflow execution context."""
    execution_id: UUID = Field(..., description="Unique execution identifier")
    workflow_id: UUID = Field(..., description="Workflow identifier")
    trigger_type: str = Field(..., description="Type of trigger that started execution")
    trigger_data: Dict[str, Any] = Field({}, description="Trigger data payload")
    
    # Execution metadata
    started_by: EmailStr = Field(..., description="User who started execution")
    approved_by: Optional[EmailStr] = Field(None, description="User who approved execution")
    environment: str = Field(..., description="Execution environment")
    tenant_id: Optional[str] = Field(None, description="Associated tenant")
    
    # Parameters
    input_parameters: Dict[str, Any] = Field({}, description="Input parameters")
    runtime_variables: Dict[str, Any] = Field({}, description="Runtime variables")
    
    # Resource allocation
    allocated_resources: Dict[str, str] = Field({}, description="Allocated resources")
    resource_constraints: Dict[str, str] = Field({}, description="Resource constraints")
    
    class Config:
        schema_extra = {
            "example": {
                "execution_id": "880e8400-e29b-41d4-a716-446655440003",
                "workflow_id": "770e8400-e29b-41d4-a716-446655440002",
                "trigger_type": "scheduled",
                "trigger_data": {"cron_expression": "0 2 * * *"},
                "started_by": "system@spotify.com",
                "environment": "production",
                "tenant_id": "spotify-premium-001",
                "input_parameters": {
                    "backup_type": "full",
                    "compress": True,
                    "encrypt": True
                },
                "allocated_resources": {
                    "cpu": "1.0",
                    "memory": "2Gi"
                }
            }
        }


class ExecutionResultSchema(BaseModel):
    """Schema for workflow execution results."""
    execution_id: UUID = Field(..., description="Execution identifier")
    status: ToolStatus = Field(..., description="Execution status")
    
    # Timing information
    started_at: datetime = Field(..., description="Execution start time")
    completed_at: Optional[datetime] = Field(None, description="Execution completion time")
    duration_seconds: Optional[float] = Field(None, description="Execution duration")
    
    # Results
    success: bool = Field(False, description="Execution was successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    step_results: List[Dict[str, Any]] = Field([], description="Individual step results")
    
    # Output
    output_data: Dict[str, Any] = Field({}, description="Execution output data")
    artifacts: List[str] = Field([], description="Generated artifacts")
    logs: List[str] = Field([], description="Execution logs")
    
    # Resource usage
    resource_usage: Dict[str, Any] = Field({}, description="Resource usage statistics")
    performance_metrics: Dict[str, float] = Field({}, description="Performance metrics")
    
    # Notifications
    notifications_sent: List[Dict[str, Any]] = Field([], description="Sent notifications")
    
    class Config:
        schema_extra = {
            "example": {
                "execution_id": "880e8400-e29b-41d4-a716-446655440003",
                "status": "completed",
                "started_at": "2025-01-19T02:00:00Z",
                "completed_at": "2025-01-19T02:15:30Z",
                "duration_seconds": 930.5,
                "success": True,
                "step_results": [
                    {
                        "step_id": "660e8400-e29b-41d4-a716-446655440001",
                        "status": "completed",
                        "duration": 925.2,
                        "output": {"backup_file": "/backups/prod_20250119.sql.gz"}
                    }
                ],
                "output_data": {
                    "backup_file": "/backups/prod_20250119.sql.gz",
                    "backup_size_mb": 1024.5,
                    "checksum": "sha256:abc123..."
                },
                "resource_usage": {
                    "cpu_avg": 0.45,
                    "memory_max_mb": 1536,
                    "disk_io_mb": 2048
                }
            }
        }


class MaintenanceWindowSchema(BaseModel):
    """Schema for maintenance window configuration."""
    window_id: UUID = Field(..., description="Unique window identifier")
    name: str = Field(..., description="Maintenance window name")
    description: str = Field(..., description="Maintenance window description")
    
    # Timing
    start_time: datetime = Field(..., description="Window start time")
    end_time: datetime = Field(..., description="Window end time")
    timezone: str = Field("UTC", description="Timezone for window")
    recurrence_pattern: Optional[str] = Field(None, description="Recurrence pattern")
    
    # Scope
    affected_services: List[str] = Field([], description="Affected services")
    affected_tenants: List[str] = Field([], description="Affected tenants")
    environment: str = Field(..., description="Target environment")
    
    # Workflows
    pre_maintenance_workflows: List[UUID] = Field([], description="Pre-maintenance workflows")
    maintenance_workflows: List[UUID] = Field([], description="Maintenance workflows")
    post_maintenance_workflows: List[UUID] = Field([], description="Post-maintenance workflows")
    
    # Approval and notifications
    approval_required: bool = Field(True, description="Require approval")
    approvers: List[EmailStr] = Field([], description="Required approvers")
    notification_advance_hours: int = Field(24, description="Notification advance time")
    
    # Emergency handling
    emergency_contact: EmailStr = Field(..., description="Emergency contact")
    rollback_plan: Optional[str] = Field(None, description="Rollback plan")
    
    class Config:
        schema_extra = {
            "example": {
                "window_id": "990e8400-e29b-41d4-a716-446655440004",
                "name": "Monthly Security Patches",
                "description": "Apply security patches to all production systems",
                "start_time": "2025-02-01T02:00:00Z",
                "end_time": "2025-02-01T06:00:00Z",
                "timezone": "UTC",
                "recurrence_pattern": "0 2 1 * *",
                "affected_services": ["api-gateway", "auth-service"],
                "environment": "production",
                "approval_required": True,
                "approvers": ["cto@spotify.com", "ops-lead@spotify.com"],
                "notification_advance_hours": 48,
                "emergency_contact": "oncall@spotify.com"
            }
        }


class AutomationToolsConfigSchema(BaseModel):
    """Master schema for automation tools configuration."""
    config_id: UUID = Field(..., description="Configuration identifier")
    name: str = Field(..., description="Configuration name")
    description: str = Field(..., description="Configuration description")
    
    # Tools and workflows
    tools: List[ToolConfigurationSchema] = Field([], description="Available tools")
    workflows: List[WorkflowSchema] = Field([], description="Configured workflows")
    maintenance_windows: List[MaintenanceWindowSchema] = Field([], description="Maintenance windows")
    
    # Global settings
    default_timeout_seconds: int = Field(3600, description="Default execution timeout")
    max_concurrent_workflows: int = Field(10, description="Maximum concurrent workflows")
    retention_days: int = Field(90, description="Execution history retention")
    
    # Security settings
    execution_isolation: bool = Field(True, description="Enable execution isolation")
    audit_logging: bool = Field(True, description="Enable audit logging")
    encryption_at_rest: bool = Field(True, description="Enable encryption at rest")
    
    # Monitoring
    health_check_interval: int = Field(300, description="Health check interval in seconds")
    metrics_collection: bool = Field(True, description="Enable metrics collection")
    alerting_enabled: bool = Field(True, description="Enable alerting")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "config_id": "aa0e8400-e29b-41d4-a716-446655440005",
                "name": "Production Automation Suite",
                "description": "Complete automation configuration for production environment",
                "default_timeout_seconds": 3600,
                "max_concurrent_workflows": 10,
                "retention_days": 90,
                "execution_isolation": True,
                "audit_logging": True,
                "health_check_interval": 300,
                "metrics_collection": True
            }
        }


# Export all schemas
__all__ = [
    "ToolCategory",
    "ExecutionStrategy",
    "ToolStatus",
    "Priority",
    "ToolConfigurationSchema",
    "WorkflowStepSchema",
    "WorkflowSchema",
    "ExecutionContextSchema",
    "ExecutionResultSchema",
    "MaintenanceWindowSchema",
    "AutomationToolsConfigSchema"
]
