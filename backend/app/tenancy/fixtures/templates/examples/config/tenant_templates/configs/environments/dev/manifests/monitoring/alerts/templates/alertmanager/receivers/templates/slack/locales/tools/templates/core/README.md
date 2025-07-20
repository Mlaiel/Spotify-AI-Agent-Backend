# Core Module - Advanced Tenancy System

**Author**: Fahed Mlaiel  
**Role**: Lead Dev & AI Architect  
**Version**: 1.0.0  

## Overview

This Core module provides the central infrastructure for the Spotify AI Agent's multi-tenant tenancy system. It integrates advanced management, security, monitoring, and orchestration features for a complete industrial solution.

## Architecture

### Main Components

```
┌─────────────────────────────────────────────────────────────┐
│                    CORE MODULE                              │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│ │   Configuration │ │    Security     │ │     Cache       │ │
│ │    Manager      │ │    Manager      │ │    Manager      │ │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘ │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│ │    Alerts       │ │   Templates     │ │   Metrics       │ │
│ │    Manager      │ │    Engine       │ │   Collector     │ │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘ │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│ │   Validation    │ │    Workflow     │ │    Events       │ │
│ │   Framework     │ │    Engine       │ │    Bus          │ │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 1. Configuration Manager (`config.py`)
- **Function**: Centralized configuration management
- **Features**:
  - Hierarchical configuration by environment
  - Hot reloading
  - Configuration validation
  - Secrets management
  - Per-tenant configuration

### 2. Alert System (`alerts.py`)
- **Function**: Intelligent alert management
- **Features**:
  - Configurable alert rules
  - Multiple channels (Email, Slack, Webhook)
  - Aggregation and deduplication
  - Automatic escalation
  - Alert templates

### 3. Template Engine (`templates.py`)
- **Function**: Template rendering with localization
- **Features**:
  - Advanced Jinja2 support
  - Internationalization (i18n)
  - Template caching
  - Dynamic templates
  - Template validation

### 4. Security Manager (`security.py`)
- **Function**: Multi-layer security
- **Features**:
  - AES-256 encryption
  - Permission management
  - Complete audit trail
  - Security policies
  - Granular access control

### 5. Cache System (`cache.py`)
- **Function**: High-performance distributed cache
- **Features**:
  - Redis cluster support
  - Multi-level caching
  - Smart invalidation
  - Automatic compression
  - Cache metrics

### 6. Metrics Collector (`metrics.py`)
- **Function**: Monitoring and observability
- **Features**:
  - Prometheus metrics
  - Real-time aggregation
  - System and business metrics
  - Automatic dashboards
  - Thresholds and alerts

### 7. Validation Framework (`validation.py`)
- **Function**: Advanced data validation
- **Features**:
  - Flexible validation rules
  - JSON/YAML schema validation
  - Custom validators
  - Detailed error reports
  - Asynchronous validation

### 8. Workflow Engine (`workflow.py`)
- **Function**: Process orchestration
- **Features**:
  - Configurable workflows
  - Parallel and sequential tasks
  - Error handling and retry
  - Conditions and loops
  - Workflow monitoring

### 9. Event Bus (`events.py`)
- **Function**: Event-driven architecture
- **Features**:
  - Asynchronous event bus
  - Configurable handlers
  - Priorities and filtering
  - Dead letter queue
  - Event metrics

## Usage

### System Initialization

```python
from core import initialize_core_system, shutdown_core_system

# Initialization
await initialize_core_system()

# Using components
from core import config_manager, alert_manager, template_engine

# Configuration
config = await config_manager.get_tenant_config("tenant_123")

# Alerts
await alert_manager.send_alert("system.high_cpu", {"value": 95})

# Templates
html = await template_engine.render("welcome_email", {"user": "John"}, locale="en")

# Graceful shutdown
await shutdown_core_system()
```

### Tenant Configuration

```python
from core import tenant_validator, workflow_engine

# Validation
tenant_data = {
    "tenant_id": "acme_corp",
    "name": "ACME Corporation", 
    "email": "admin@acme.com",
    "api_quota_per_hour": 5000,
    "storage_quota_gb": 100.0,
    "features": ["audio_processing", "analytics"]
}

result = tenant_validator.validate(tenant_data)
if result.is_valid:
    # Launch provisioning workflow
    workflow_id = await workflow_engine.create_workflow_from_template(
        "tenant_provisioning", 
        tenant_data["tenant_id"],
        {"tenant_config": tenant_data}
    )
    
    # Execute workflow
    workflow_result = await workflow_engine.execute_workflow(
        workflow_id, 
        tenant_data["tenant_id"], 
        {"tenant_config": tenant_data}
    )
```

### Event Management

```python
from core import event_bus, publish_tenant_created

# Event publishing
await publish_tenant_created("tenant_123", {
    "name": "Test Tenant",
    "plan": "premium"
})

# Custom handler
class CustomHandler(EventHandler):
    async def handle(self, event):
        print(f"Processing {event.event_type} for {event.tenant_id}")
        return True

# Registration
custom_handler = CustomHandler()
event_bus.register_handler(custom_handler)
```

## Metrics and Monitoring

### Available Metrics

- **Tenant Metrics**:
  - `tenant_requests_total`: Number of requests per tenant
  - `tenant_response_time_seconds`: Response time
  - `tenant_storage_usage_bytes`: Storage usage
  - `tenant_api_quota_usage`: API quota usage

- **System Metrics**:
  - `system_cpu_usage_percent`: CPU usage
  - `system_memory_usage_bytes`: Memory usage
  - `system_disk_usage_percent`: Disk usage

### Dashboards

The module automatically generates Grafana dashboards for:
- Tenant overview
- System performance
- Business metrics
- Alerts and incidents

## Security

### Security Features

1. **Encryption**:
   - AES-256 for sensitive data
   - Encryption in transit (TLS)
   - Automatic key rotation

2. **Access Control**:
   - RBAC (Role-Based Access Control)
   - Granular permissions
   - Tenant isolation

3. **Audit**:
   - Complete audit logs
   - Action traceability
   - Automatic compliance

4. **Policies**:
   - Configurable security policies
   - Automatic validation
   - Compliance reports

## Configuration

### Main Configuration File

```yaml
# config/environments/dev/core.yaml
core:
  security:
    encryption_key: "${ENCRYPTION_KEY}"
    audit_enabled: true
    
  cache:
    redis_url: "redis://localhost:6379"
    default_ttl: 3600
    
  metrics:
    prometheus_enabled: true
    collection_interval: 30
    
  alerts:
    channels:
      email:
        smtp_host: "smtp.example.com"
        smtp_port: 587
      slack:
        webhook_url: "${SLACK_WEBHOOK}"
```

## Administration Scripts

### Available Scripts

1. **Initialization**: `scripts/init_core_system.py`
2. **Backup**: `scripts/backup_core_data.py`
3. **Migration**: `scripts/migrate_core_schema.py`
4. **Monitoring**: `scripts/health_check.py`

### Usage

```bash
# Initialization
python scripts/init_core_system.py --env dev

# Health check
python scripts/health_check.py --detailed

# Backup
python scripts/backup_core_data.py --output /backup/core_$(date +%Y%m%d).tar.gz
```

## Testing and Validation

### Test Types

1. **Unit Tests**: Each component individually
2. **Integration Tests**: Component interactions
3. **Performance Tests**: Load and stress testing
4. **Security Tests**: Vulnerability and penetration testing

### Running Tests

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Performance tests
pytest tests/performance/ --benchmark-only

# Code coverage
pytest --cov=core tests/
```

## Deployment

### Environments

- **Development**: Basic configuration
- **Staging**: Near-production configuration
- **Production**: Optimized configuration

### Containers

```dockerfile
# Dockerfile for core module
FROM python:3.11-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY core/ /app/core/
WORKDIR /app

CMD ["python", "-m", "core"]
```

## Support and Maintenance

### Logging

Logs are structured in JSON for easy analysis:

```json
{
  "timestamp": "2025-01-19T10:30:00Z",
  "level": "INFO",
  "component": "core.cache",
  "tenant_id": "tenant_123",
  "message": "Cache hit for key tenant_config",
  "latency_ms": 2.5
}
```

### Troubleshooting

1. **Performance Issues**: Check cache metrics
2. **Configuration Errors**: Validate with schema
3. **Security Issues**: Check audit logs
4. **Workflow Failures**: Analyze task results

### Contact

**Lead Developer**: Fahed Mlaiel  
**Email**: fahed.mlaiel@spotify-ai.com  
**Role**: Lead Developer & AI Architect  

---

*This module is part of the Spotify AI Agent project and follows industry standards for security, performance, and maintainability.*
