"""
Enterprise Automation and DevOps Schemas
Advanced automation, CI/CD, and infrastructure-as-code configurations.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator, HttpUrl
import uuid


class DeploymentStrategy(str, Enum):
    """Deployment strategies."""
    BLUE_GREEN = "blue_green"
    ROLLING = "rolling"
    CANARY = "canary"
    A_B_TESTING = "a_b_testing"
    RECREATE = "recreate"
    FEATURE_TOGGLE = "feature_toggle"


class EnvironmentType(str, Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DISASTER_RECOVERY = "disaster_recovery"


class AutomationLevel(str, Enum):
    """Automation maturity levels."""
    MANUAL = "manual"
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    FULLY_AUTOMATED = "fully_automated"


class PipelineStage(str, Enum):
    """CI/CD pipeline stages."""
    BUILD = "build"
    TEST = "test"
    SECURITY_SCAN = "security_scan"
    QUALITY_GATE = "quality_gate"
    DEPLOY = "deploy"
    VERIFY = "verify"
    PROMOTE = "promote"
    ROLLBACK = "rollback"


class InfrastructureProvider(str, Enum):
    """Infrastructure providers."""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    KUBERNETES = "kubernetes"
    DOCKER = "docker"
    ON_PREMISE = "on_premise"
    HYBRID = "hybrid"


class AutomationTriggerSchema(BaseModel):
    """Automation trigger configuration."""
    
    trigger_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., min_length=2, max_length=100)
    description: str = Field(..., max_length=500)
    
    # Trigger types
    trigger_type: str = Field(..., regex=r"^(webhook|schedule|event|manual|api)$")
    
    # Webhook configuration
    webhook_url: Optional[HttpUrl] = None
    webhook_secret: Optional[str] = None
    webhook_events: List[str] = Field(default_factory=list)
    
    # Schedule configuration
    cron_expression: Optional[str] = None
    timezone: str = "UTC"
    
    # Event-based triggers
    event_source: Optional[str] = None
    event_filters: Dict[str, Any] = Field(default_factory=dict)
    
    # Conditions
    conditions: List[Dict[str, Any]] = Field(default_factory=list)
    approval_required: bool = False
    approvers: List[str] = Field(default_factory=list)
    
    # Execution settings
    max_concurrent_executions: int = Field(1, ge=1, le=10)
    timeout_minutes: int = Field(60, ge=1, le=480)
    retry_attempts: int = Field(3, ge=0, le=10)
    retry_delay_seconds: int = Field(30, ge=1, le=300)
    
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('cron_expression')
    def validate_cron(cls, v):
        if v:
            # Basic cron validation (simplified)
            parts = v.split()
            if len(parts) != 5:
                raise ValueError("Cron expression must have 5 parts")
        return v


class PipelineStageSchema(BaseModel):
    """CI/CD pipeline stage configuration."""
    
    stage_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., min_length=2, max_length=50)
    stage_type: PipelineStage
    order: int = Field(..., ge=1, le=100)
    
    # Dependencies
    depends_on: List[str] = Field(default_factory=list)
    parallel_execution: bool = False
    
    # Stage configuration
    commands: List[str] = Field(default_factory=list)
    scripts: List[str] = Field(default_factory=list)
    environment_variables: Dict[str, str] = Field(default_factory=dict)
    secrets: List[str] = Field(default_factory=list)
    
    # Resource requirements
    cpu_limit: str = "1000m"
    memory_limit: str = "2Gi"
    timeout_minutes: int = Field(30, ge=1, le=480)
    
    # Quality gates
    quality_gates: List[Dict[str, Any]] = Field(default_factory=list)
    success_criteria: Dict[str, Any] = Field(default_factory=dict)
    failure_conditions: List[str] = Field(default_factory=list)
    
    # Artifacts
    input_artifacts: List[str] = Field(default_factory=list)
    output_artifacts: List[str] = Field(default_factory=list)
    artifact_retention_days: int = Field(30, ge=1, le=365)
    
    # Notifications
    on_success_notify: List[str] = Field(default_factory=list)
    on_failure_notify: List[str] = Field(default_factory=list)
    
    enabled: bool = True


class DeploymentConfigSchema(BaseModel):
    """Advanced deployment configuration."""
    
    strategy: DeploymentStrategy = DeploymentStrategy.BLUE_GREEN
    environment: EnvironmentType
    
    # Blue-green deployment
    blue_green_config: Dict[str, Any] = Field(default_factory=lambda: {
        "warm_up_time_seconds": 300,
        "validation_time_seconds": 600,
        "auto_promote": False,
        "rollback_on_failure": True
    })
    
    # Rolling deployment
    rolling_config: Dict[str, Any] = Field(default_factory=lambda: {
        "max_unavailable": "25%",
        "max_surge": "25%",
        "batch_size": 2,
        "pause_between_batches_seconds": 30
    })
    
    # Canary deployment
    canary_config: Dict[str, Any] = Field(default_factory=lambda: {
        "initial_traffic_percentage": 5,
        "traffic_increment_percentage": 10,
        "evaluation_duration_minutes": 15,
        "success_threshold_percentage": 99.5,
        "auto_promote_thresholds": {
            "error_rate": 0.01,
            "response_time_p95": 1000
        }
    })
    
    # Health checks
    health_check_enabled: bool = True
    health_check_path: str = "/health"
    health_check_timeout_seconds: int = Field(10, ge=1, le=60)
    health_check_interval_seconds: int = Field(30, ge=5, le=300)
    healthy_threshold: int = Field(2, ge=1, le=10)
    unhealthy_threshold: int = Field(3, ge=1, le=10)
    
    # Rollback configuration
    auto_rollback_enabled: bool = True
    rollback_triggers: List[str] = Field(default_factory=lambda: [
        "health_check_failure",
        "error_rate_threshold",
        "response_time_threshold"
    ])
    rollback_timeout_minutes: int = Field(15, ge=1, le=60)
    
    # Pre and post deployment hooks
    pre_deployment_hooks: List[str] = Field(default_factory=list)
    post_deployment_hooks: List[str] = Field(default_factory=list)
    cleanup_hooks: List[str] = Field(default_factory=list)


class InfrastructureConfigSchema(BaseModel):
    """Infrastructure-as-Code configuration."""
    
    provider: InfrastructureProvider
    region: str = Field(..., description="Primary deployment region")
    availability_zones: List[str] = Field(..., min_items=1)
    
    # Compute resources
    compute_config: Dict[str, Any] = Field(default_factory=lambda: {
        "instance_types": ["t3.medium", "t3.large"],
        "min_instances": 2,
        "max_instances": 50,
        "auto_scaling_enabled": True,
        "spot_instances_enabled": False
    })
    
    # Networking
    network_config: Dict[str, Any] = Field(default_factory=lambda: {
        "vpc_cidr": "10.0.0.0/16",
        "public_subnets": ["10.0.1.0/24", "10.0.2.0/24"],
        "private_subnets": ["10.0.10.0/24", "10.0.20.0/24"],
        "enable_nat_gateway": True,
        "enable_vpn_gateway": False
    })
    
    # Storage
    storage_config: Dict[str, Any] = Field(default_factory=lambda: {
        "volume_type": "gp3",
        "volume_size_gb": 100,
        "encrypted": True,
        "backup_enabled": True,
        "snapshot_retention_days": 30
    })
    
    # Database
    database_config: Dict[str, Any] = Field(default_factory=lambda: {
        "engine": "postgresql",
        "version": "13.7",
        "instance_class": "db.t3.medium",
        "multi_az": True,
        "backup_retention_days": 7,
        "encryption_enabled": True
    })
    
    # Monitoring and logging
    monitoring_config: Dict[str, Any] = Field(default_factory=lambda: {
        "cloudwatch_enabled": True,
        "custom_metrics_enabled": True,
        "log_retention_days": 90,
        "tracing_enabled": True
    })
    
    # Security
    security_config: Dict[str, Any] = Field(default_factory=lambda: {
        "waf_enabled": True,
        "ddos_protection": True,
        "security_groups_strict": True,
        "encryption_in_transit": True,
        "encryption_at_rest": True
    })
    
    # Tags and metadata
    resource_tags: Dict[str, str] = Field(default_factory=lambda: {
        "Environment": "production",
        "Project": "spotify-ai-agent",
        "Owner": "platform-team",
        "CostCenter": "engineering"
    })


class AutomationWorkflowSchema(BaseModel):
    """Complete automation workflow configuration."""
    
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., min_length=2, max_length=100)
    description: str = Field(..., max_length=1000)
    version: str = Field("1.0.0", regex=r"^\d+\.\d+\.\d+$")
    
    # Workflow metadata
    owner: str = Field(..., description="Workflow owner")
    team: str = Field(..., description="Responsible team")
    automation_level: AutomationLevel = AutomationLevel.ADVANCED
    
    # Triggers
    triggers: List[AutomationTriggerSchema] = Field(..., min_items=1)
    
    # Pipeline stages
    pipeline_stages: List[PipelineStageSchema] = Field(..., min_items=1)
    
    # Deployment configuration
    deployment_config: DeploymentConfigSchema
    
    # Infrastructure configuration
    infrastructure_config: InfrastructureConfigSchema
    
    # Global settings
    global_environment_variables: Dict[str, str] = Field(default_factory=dict)
    global_secrets: List[str] = Field(default_factory=list)
    
    # Concurrency and limits
    max_concurrent_workflows: int = Field(3, ge=1, le=10)
    workflow_timeout_hours: int = Field(4, ge=1, le=24)
    
    # Notifications and reporting
    notification_channels: List[str] = Field(default_factory=list)
    report_generation_enabled: bool = True
    metrics_collection_enabled: bool = True
    
    # Compliance and governance
    approval_gates: List[Dict[str, Any]] = Field(default_factory=list)
    compliance_checks: List[str] = Field(default_factory=list)
    audit_logging_enabled: bool = True
    
    # Error handling and recovery
    error_handling_strategy: str = Field("fail_fast", regex=r"^(fail_fast|continue_on_error|retry)$")
    recovery_procedures: List[str] = Field(default_factory=list)
    
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('pipeline_stages')
    def validate_stage_order(cls, v):
        orders = [stage.order for stage in v]
        if len(orders) != len(set(orders)):
            raise ValueError("Stage orders must be unique")
        return sorted(v, key=lambda x: x.order)
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Production Deployment Workflow",
                "description": "Automated CI/CD pipeline for production deployments",
                "owner": "platform-team",
                "team": "DevOps",
                "automation_level": "fully_automated"
            }
        }


class AutomationExecutionSchema(BaseModel):
    """Automation execution tracking and results."""
    
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = Field(..., description="Associated workflow ID")
    tenant_id: str = Field(..., description="Tenant identifier")
    
    # Execution metadata
    triggered_by: str = Field(..., description="User or system that triggered execution")
    trigger_type: str = Field(..., description="Type of trigger")
    execution_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Timeline
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    
    # Status tracking
    status: str = Field("running", regex=r"^(pending|running|succeeded|failed|cancelled|timeout)$")
    current_stage: Optional[str] = None
    completed_stages: List[str] = Field(default_factory=list)
    failed_stage: Optional[str] = None
    
    # Results and metrics
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    artifacts_generated: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    
    # Stage execution details
    stage_executions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Resource usage
    resources_used: Dict[str, Any] = Field(default_factory=dict)
    cost_estimate: Optional[float] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "workflow_id": "wf_123456",
                "tenant_id": "tenant_001",
                "triggered_by": "github_webhook",
                "trigger_type": "webhook",
                "status": "succeeded"
            }
        }
