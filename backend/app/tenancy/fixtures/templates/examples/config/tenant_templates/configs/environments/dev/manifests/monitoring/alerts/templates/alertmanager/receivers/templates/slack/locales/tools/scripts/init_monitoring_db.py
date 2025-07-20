#!/bin/bash
"""
Spotify AI Agent - Advanced Monitoring Database Initialization Script
=====================================================================

Enterprise-grade database schema initialization for monitoring system.
Creates all necessary tables, indexes, and constraints for:
- Multi-tenant monitoring data
- Security events and audit logs
- Alert management
- Performance metrics
- Compliance tracking

Author: Fahed Mlaiel (Lead Developer + AI Architect)
Team: Expert Development Team
"""

set -euo pipefail

# Configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-spotify_ai_monitoring}"
DB_USER="${DB_USER:-monitoring_user}"
DB_SCHEMA="${DB_SCHEMA:-monitoring}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v psql &> /dev/null; then
        log_error "psql (PostgreSQL client) is not installed"
        exit 1
    fi
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    log_success "All dependencies are available"
}

# Create database if it doesn't exist
create_database() {
    log_info "Creating database if not exists..."
    
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "
        SELECT 'CREATE DATABASE $DB_NAME'
        WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$DB_NAME')\gexec
    " || {
        log_error "Failed to create database"
        exit 1
    }
    
    log_success "Database $DB_NAME is ready"
}

# Create schema and extensions
create_schema() {
    log_info "Creating schema and extensions..."
    
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" << EOF
-- Create schema
CREATE SCHEMA IF NOT EXISTS $DB_SCHEMA;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Set search path
SET search_path TO $DB_SCHEMA, public;
EOF

    log_success "Schema and extensions created"
}

# Create monitoring tables
create_monitoring_tables() {
    log_info "Creating monitoring tables..."
    
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" << EOF
SET search_path TO $DB_SCHEMA, public;

-- Tenant metrics table
CREATE TABLE IF NOT EXISTS tenant_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id VARCHAR(255) NOT NULL,
    cpu_usage DECIMAL(5,2) NOT NULL,
    memory_usage DECIMAL(5,2) NOT NULL,
    disk_usage DECIMAL(5,2) NOT NULL,
    api_response_time DECIMAL(10,3) NOT NULL,
    active_users INTEGER NOT NULL DEFAULT 0,
    database_connections INTEGER NOT NULL DEFAULT 0,
    error_rate DECIMAL(5,2) NOT NULL DEFAULT 0.0,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Create indexes for tenant_metrics
CREATE INDEX IF NOT EXISTS idx_tenant_metrics_tenant_id ON tenant_metrics(tenant_id);
CREATE INDEX IF NOT EXISTS idx_tenant_metrics_timestamp ON tenant_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_tenant_metrics_tenant_timestamp ON tenant_metrics(tenant_id, timestamp);

-- Alert rules table
CREATE TABLE IF NOT EXISTS alert_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id VARCHAR(255) NOT NULL,
    rule_name VARCHAR(255) NOT NULL,
    expression TEXT NOT NULL,
    severity VARCHAR(50) NOT NULL CHECK (severity IN ('critical', 'warning', 'info', 'debug')),
    threshold DECIMAL(10,2) NOT NULL,
    duration_seconds INTEGER NOT NULL DEFAULT 300,
    labels JSONB DEFAULT '{}',
    annotations JSONB DEFAULT '{}',
    enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE(tenant_id, rule_name)
);

-- Create indexes for alert_rules
CREATE INDEX IF NOT EXISTS idx_alert_rules_tenant_id ON alert_rules(tenant_id);
CREATE INDEX IF NOT EXISTS idx_alert_rules_enabled ON alert_rules(enabled);

-- Alert instances table
CREATE TABLE IF NOT EXISTS alert_instances (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    rule_id UUID NOT NULL REFERENCES alert_rules(id) ON DELETE CASCADE,
    tenant_id VARCHAR(255) NOT NULL,
    rule_name VARCHAR(255) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL CHECK (status IN ('firing', 'resolved', 'silenced')),
    threshold DECIMAL(10,2) NOT NULL,
    current_value DECIMAL(10,2) NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE,
    labels JSONB DEFAULT '{}',
    annotations JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);

-- Create indexes for alert_instances
CREATE INDEX IF NOT EXISTS idx_alert_instances_rule_id ON alert_instances(rule_id);
CREATE INDEX IF NOT EXISTS idx_alert_instances_tenant_id ON alert_instances(tenant_id);
CREATE INDEX IF NOT EXISTS idx_alert_instances_status ON alert_instances(status);
CREATE INDEX IF NOT EXISTS idx_alert_instances_started_at ON alert_instances(started_at);

-- Notification channels table
CREATE TABLE IF NOT EXISTS notification_channels (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id VARCHAR(255) NOT NULL,
    channel_name VARCHAR(255) NOT NULL,
    channel_type VARCHAR(50) NOT NULL CHECK (channel_type IN ('slack', 'email', 'webhook', 'sms')),
    configuration JSONB NOT NULL DEFAULT '{}',
    enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE(tenant_id, channel_name)
);

-- Create indexes for notification_channels
CREATE INDEX IF NOT EXISTS idx_notification_channels_tenant_id ON notification_channels(tenant_id);
CREATE INDEX IF NOT EXISTS idx_notification_channels_type ON notification_channels(channel_type);
EOF

    log_success "Monitoring tables created"
}

# Create security tables
create_security_tables() {
    log_info "Creating security tables..."
    
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" << EOF
SET search_path TO $DB_SCHEMA, public;

-- Users table
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    roles TEXT[] NOT NULL DEFAULT '{}',
    is_active BOOLEAN NOT NULL DEFAULT true,
    last_login TIMESTAMP WITH TIME ZONE,
    last_password_change TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    failed_login_attempts INTEGER NOT NULL DEFAULT 0,
    account_locked_until TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Create indexes for users
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_tenant_id ON users(tenant_id);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);

-- Roles table
CREATE TABLE IF NOT EXISTS roles (
    role_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    role_name VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    permissions JSONB NOT NULL DEFAULT '[]',
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE(tenant_id, role_name)
);

-- User permissions table
CREATE TABLE IF NOT EXISTS user_permissions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    role_id UUID NOT NULL REFERENCES roles(role_id) ON DELETE CASCADE,
    tenant_id VARCHAR(255) NOT NULL,
    granted_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    granted_by UUID REFERENCES users(user_id),
    UNIQUE(user_id, role_id)
);

-- Security events table
CREATE TABLE IF NOT EXISTS security_events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(255) NOT NULL,
    severity VARCHAR(50) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    source_ip INET NOT NULL,
    user_id UUID REFERENCES users(user_id),
    tenant_id VARCHAR(255) NOT NULL,
    resource VARCHAR(255) NOT NULL DEFAULT '',
    action VARCHAR(255) NOT NULL DEFAULT '',
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    details JSONB DEFAULT '{}',
    risk_score DECIMAL(4,2) NOT NULL DEFAULT 0.0,
    processed BOOLEAN NOT NULL DEFAULT false
);

-- Create indexes for security_events
CREATE INDEX IF NOT EXISTS idx_security_events_timestamp ON security_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_security_events_severity ON security_events(severity);
CREATE INDEX IF NOT EXISTS idx_security_events_source_ip ON security_events(source_ip);
CREATE INDEX IF NOT EXISTS idx_security_events_user_id ON security_events(user_id);
CREATE INDEX IF NOT EXISTS idx_security_events_tenant_id ON security_events(tenant_id);
CREATE INDEX IF NOT EXISTS idx_security_events_processed ON security_events(processed);

-- Audit logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(255) NOT NULL,
    user_id UUID REFERENCES users(user_id),
    tenant_id VARCHAR(255) NOT NULL,
    resource VARCHAR(255) NOT NULL DEFAULT '',
    action VARCHAR(255) NOT NULL,
    result VARCHAR(50) NOT NULL CHECK (result IN ('success', 'failure', 'denied')),
    ip_address INET NOT NULL,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Create indexes for audit_logs
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON audit_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_tenant_id ON audit_logs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_result ON audit_logs(result);
EOF

    log_success "Security tables created"
}

# Create automation tables
create_automation_tables() {
    log_info "Creating automation tables..."
    
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" << EOF
SET search_path TO $DB_SCHEMA, public;

-- Script tasks table
CREATE TABLE IF NOT EXISTS script_tasks (
    task_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id VARCHAR(255) NOT NULL,
    task_name VARCHAR(255) NOT NULL,
    script_type VARCHAR(50) NOT NULL CHECK (script_type IN ('bash', 'python', 'javascript', 'sql', 'yaml')),
    script_content TEXT,
    script_path VARCHAR(500),
    arguments TEXT[] DEFAULT '{}',
    environment JSONB DEFAULT '{}',
    working_directory VARCHAR(500) NOT NULL DEFAULT '/tmp',
    timeout_seconds INTEGER NOT NULL DEFAULT 300,
    priority VARCHAR(50) NOT NULL DEFAULT 'normal' CHECK (priority IN ('low', 'normal', 'high', 'critical')),
    max_retries INTEGER NOT NULL DEFAULT 3,
    schedule_cron VARCHAR(255),
    enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Create indexes for script_tasks
CREATE INDEX IF NOT EXISTS idx_script_tasks_tenant_id ON script_tasks(tenant_id);
CREATE INDEX IF NOT EXISTS idx_script_tasks_enabled ON script_tasks(enabled);
CREATE INDEX IF NOT EXISTS idx_script_tasks_priority ON script_tasks(priority);

-- Script executions table
CREATE TABLE IF NOT EXISTS script_executions (
    execution_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id UUID NOT NULL REFERENCES script_tasks(task_id) ON DELETE CASCADE,
    tenant_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled', 'timeout')),
    exit_code INTEGER,
    stdout TEXT,
    stderr TEXT,
    execution_time_seconds DECIMAL(10,3),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    resource_usage JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Create indexes for script_executions
CREATE INDEX IF NOT EXISTS idx_script_executions_task_id ON script_executions(task_id);
CREATE INDEX IF NOT EXISTS idx_script_executions_tenant_id ON script_executions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_script_executions_status ON script_executions(status);
CREATE INDEX IF NOT EXISTS idx_script_executions_started_at ON script_executions(started_at);
EOF

    log_success "Automation tables created"
}

# Create views for reporting
create_views() {
    log_info "Creating reporting views..."
    
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" << EOF
SET search_path TO $DB_SCHEMA, public;

-- Tenant metrics summary view
CREATE OR REPLACE VIEW tenant_metrics_summary AS
SELECT 
    tenant_id,
    DATE_TRUNC('hour', timestamp) as hour,
    AVG(cpu_usage) as avg_cpu_usage,
    MAX(cpu_usage) as max_cpu_usage,
    AVG(memory_usage) as avg_memory_usage,
    MAX(memory_usage) as max_memory_usage,
    AVG(disk_usage) as avg_disk_usage,
    MAX(disk_usage) as max_disk_usage,
    AVG(api_response_time) as avg_api_response_time,
    MAX(api_response_time) as max_api_response_time,
    AVG(active_users) as avg_active_users,
    MAX(active_users) as max_active_users,
    AVG(error_rate) as avg_error_rate,
    MAX(error_rate) as max_error_rate,
    COUNT(*) as sample_count
FROM tenant_metrics
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY tenant_id, DATE_TRUNC('hour', timestamp)
ORDER BY tenant_id, hour DESC;

-- Active alerts view
CREATE OR REPLACE VIEW active_alerts AS
SELECT 
    ai.id,
    ai.tenant_id,
    ai.rule_name,
    ai.severity,
    ai.current_value,
    ai.threshold,
    ai.started_at,
    ai.labels,
    ai.annotations,
    ar.expression,
    EXTRACT(EPOCH FROM (NOW() - ai.started_at)) as duration_seconds
FROM alert_instances ai
JOIN alert_rules ar ON ai.rule_id = ar.id
WHERE ai.status = 'firing'
ORDER BY ai.severity, ai.started_at;

-- Security events summary view
CREATE OR REPLACE VIEW security_events_summary AS
SELECT 
    tenant_id,
    DATE_TRUNC('day', timestamp) as day,
    event_type,
    severity,
    COUNT(*) as event_count,
    AVG(risk_score) as avg_risk_score,
    MAX(risk_score) as max_risk_score
FROM security_events
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY tenant_id, DATE_TRUNC('day', timestamp), event_type, severity
ORDER BY tenant_id, day DESC, event_count DESC;
EOF

    log_success "Reporting views created"
}

# Create stored procedures
create_procedures() {
    log_info "Creating stored procedures..."
    
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" << EOF
SET search_path TO $DB_SCHEMA, public;

-- Function to clean old metrics data
CREATE OR REPLACE FUNCTION cleanup_old_metrics(retention_days INTEGER DEFAULT 30)
RETURNS INTEGER AS \$\$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM tenant_metrics 
    WHERE timestamp < NOW() - (retention_days || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN deleted_count;
END;
\$\$ LANGUAGE plpgsql;

-- Function to calculate tenant health score
CREATE OR REPLACE FUNCTION calculate_tenant_health_score(p_tenant_id VARCHAR)
RETURNS DECIMAL(4,2) AS \$\$
DECLARE
    health_score DECIMAL(4,2) := 100.0;
    cpu_score DECIMAL(4,2);
    memory_score DECIMAL(4,2);
    error_score DECIMAL(4,2);
BEGIN
    -- Get recent metrics (last hour)
    SELECT 
        AVG(cpu_usage),
        AVG(memory_usage),
        AVG(error_rate)
    INTO cpu_score, memory_score, error_score
    FROM tenant_metrics 
    WHERE tenant_id = p_tenant_id 
    AND timestamp >= NOW() - INTERVAL '1 hour';
    
    -- Adjust health score based on metrics
    IF cpu_score > 80 THEN
        health_score := health_score - (cpu_score - 80) * 2;
    END IF;
    
    IF memory_score > 85 THEN
        health_score := health_score - (memory_score - 85) * 2;
    END IF;
    
    IF error_score > 5 THEN
        health_score := health_score - error_score * 10;
    END IF;
    
    -- Ensure score is between 0 and 100
    health_score := GREATEST(0, LEAST(100, health_score));
    
    RETURN health_score;
END;
\$\$ LANGUAGE plpgsql;

-- Function to trigger alert
CREATE OR REPLACE FUNCTION trigger_alert(
    p_rule_id UUID,
    p_tenant_id VARCHAR,
    p_current_value DECIMAL
)
RETURNS UUID AS \$\$
DECLARE
    alert_id UUID;
    rule_rec RECORD;
BEGIN
    -- Get rule details
    SELECT * INTO rule_rec FROM alert_rules WHERE id = p_rule_id;
    
    -- Check if alert is already firing
    SELECT id INTO alert_id FROM alert_instances 
    WHERE rule_id = p_rule_id AND status = 'firing';
    
    IF alert_id IS NULL THEN
        -- Create new alert instance
        INSERT INTO alert_instances (
            rule_id, tenant_id, rule_name, severity, status,
            threshold, current_value, labels, annotations
        ) VALUES (
            p_rule_id, p_tenant_id, rule_rec.rule_name, rule_rec.severity, 'firing',
            rule_rec.threshold, p_current_value, rule_rec.labels, rule_rec.annotations
        ) RETURNING id INTO alert_id;
    ELSE
        -- Update existing alert
        UPDATE alert_instances 
        SET current_value = p_current_value,
            annotations = rule_rec.annotations || jsonb_build_object('updated_at', NOW())
        WHERE id = alert_id;
    END IF;
    
    RETURN alert_id;
END;
\$\$ LANGUAGE plpgsql;
EOF

    log_success "Stored procedures created"
}

# Create triggers
create_triggers() {
    log_info "Creating triggers..."
    
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" << EOF
SET search_path TO $DB_SCHEMA, public;

-- Trigger function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS \$\$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
\$\$ LANGUAGE plpgsql;

-- Apply trigger to tables with updated_at column
CREATE TRIGGER trigger_alert_rules_updated_at
    BEFORE UPDATE ON alert_rules
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_notification_channels_updated_at
    BEFORE UPDATE ON notification_channels
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_roles_updated_at
    BEFORE UPDATE ON roles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_script_tasks_updated_at
    BEFORE UPDATE ON script_tasks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
EOF

    log_success "Triggers created"
}

# Insert default data
insert_default_data() {
    log_info "Inserting default data..."
    
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" << EOF
SET search_path TO $DB_SCHEMA, public;

-- Insert default admin user (password: admin123!)
INSERT INTO users (username, email, password_hash, tenant_id, roles, is_active)
VALUES (
    'admin',
    'admin@spotify-ai-agent.com',
    '\$2b\$12\$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNiVyYnCKKG5q',  -- admin123!
    'system',
    ARRAY['admin', 'super_admin'],
    true
) ON CONFLICT (username) DO NOTHING;

-- Insert default system role
INSERT INTO roles (role_name, tenant_id, permissions, description)
VALUES (
    'admin',
    'system',
    '["*:*"]'::jsonb,
    'System administrator with full access'
),
(
    'monitoring_user',
    'system', 
    '["metrics:read", "alerts:read", "dashboards:read"]'::jsonb,
    'Read-only monitoring access'
) ON CONFLICT (tenant_id, role_name) DO NOTHING;

-- Insert default alert rules
INSERT INTO alert_rules (tenant_id, rule_name, expression, severity, threshold, duration_seconds, labels, annotations)
VALUES 
(
    'system',
    'high_cpu_usage',
    'cpu_usage > threshold',
    'warning',
    80.0,
    300,
    '{"team": "infrastructure", "service": "monitoring"}'::jsonb,
    '{"description": "High CPU usage detected", "runbook": "https://docs.example.com/runbooks/high-cpu"}'::jsonb
),
(
    'system',
    'high_memory_usage',
    'memory_usage > threshold',
    'warning',
    85.0,
    300,
    '{"team": "infrastructure", "service": "monitoring"}'::jsonb,
    '{"description": "High memory usage detected", "runbook": "https://docs.example.com/runbooks/high-memory"}'::jsonb
),
(
    'system',
    'high_error_rate',
    'error_rate > threshold',
    'critical',
    10.0,
    120,
    '{"team": "sre", "service": "api"}'::jsonb,
    '{"description": "High error rate detected", "runbook": "https://docs.example.com/runbooks/high-errors"}'::jsonb
) ON CONFLICT (tenant_id, rule_name) DO NOTHING;

-- Insert default notification channel
INSERT INTO notification_channels (tenant_id, channel_name, channel_type, configuration)
VALUES (
    'system',
    'default_slack',
    'slack',
    '{"webhook_url": "https://hooks.slack.com/services/CHANGEME", "channel": "#alerts", "username": "AlertBot"}'::jsonb
) ON CONFLICT (tenant_id, channel_name) DO NOTHING;
EOF

    log_success "Default data inserted"
}

# Grant permissions
grant_permissions() {
    log_info "Granting permissions..."
    
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" << EOF
-- Grant schema permissions
GRANT USAGE ON SCHEMA $DB_SCHEMA TO $DB_USER;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA $DB_SCHEMA TO $DB_USER;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA $DB_SCHEMA TO $DB_USER;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA $DB_SCHEMA TO $DB_USER;

-- Grant default permissions for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA $DB_SCHEMA GRANT ALL ON TABLES TO $DB_USER;
ALTER DEFAULT PRIVILEGES IN SCHEMA $DB_SCHEMA GRANT ALL ON SEQUENCES TO $DB_USER;
ALTER DEFAULT PRIVILEGES IN SCHEMA $DB_SCHEMA GRANT ALL ON FUNCTIONS TO $DB_USER;
EOF

    log_success "Permissions granted"
}

# Main execution
main() {
    log_info "Starting monitoring database initialization..."
    log_info "Target: $DB_NAME on $DB_HOST:$DB_PORT"
    
    check_dependencies
    create_database
    create_schema
    create_monitoring_tables
    create_security_tables
    create_automation_tables
    create_views
    create_procedures
    create_triggers
    insert_default_data
    grant_permissions
    
    log_success "Monitoring database initialization completed successfully!"
    log_info "You can now connect to the database using:"
    log_info "psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME"
}

# Run main function
main "$@"
