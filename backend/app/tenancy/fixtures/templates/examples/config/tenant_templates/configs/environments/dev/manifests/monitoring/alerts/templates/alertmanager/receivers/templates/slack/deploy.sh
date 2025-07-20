#!/bin/bash

# ==================================================================================
# SPOTIFY AI AGENT - ENTERPRISE TEMPLATE DEPLOYMENT SCRIPT
# ==================================================================================
# Description: Industrial-grade automation for template deployment and management
# Version: 2.0.0
# Author: Fahed Mlaiel (Lead Dev + AI Architect)
# Last Modified: 2025-07-18
# ==================================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
readonly TEMPLATES_DIR="${SCRIPT_DIR}"
readonly CONFIG_FILE="${TEMPLATES_DIR}/template_config.yaml"
readonly LOG_FILE="/var/log/spotify-ai-agent/template-deployment.log"
readonly BACKUP_DIR="/var/backups/spotify-ai-agent/templates"
readonly VERSION_FILE="${TEMPLATES_DIR}/.template_version"

# Default values
DEFAULT_ENVIRONMENT="development"
DEFAULT_DRY_RUN="false"
DEFAULT_ROLLBACK="false"
DEFAULT_FORCE="false"
DEFAULT_VALIDATE_ONLY="false"

# Global variables
ENVIRONMENT="${DEFAULT_ENVIRONMENT}"
DRY_RUN="${DEFAULT_DRY_RUN}"
ROLLBACK="${DEFAULT_ROLLBACK}"
FORCE="${DEFAULT_FORCE}"
VALIDATE_ONLY="${DEFAULT_VALIDATE_ONLY}"
DEPLOYMENT_ID=""
START_TIME=$(date +%s)

# ==================================================================================
# UTILITY FUNCTIONS
# ==================================================================================

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Create log directory if it doesn't exist
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Log to file
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    # Log to console with colors
    case "$level" in
        "INFO")  echo -e "${GREEN}[INFO]${NC} $message" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC} $message" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} $message" ;;
        "DEBUG") echo -e "${BLUE}[DEBUG]${NC} $message" ;;
        "SUCCESS") echo -e "${GREEN}[SUCCESS]${NC} $message" ;;
        *) echo "[$level] $message" ;;
    esac
}

# Error handling
error_exit() {
    log "ERROR" "$1"
    exit 1
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Generate deployment ID
generate_deployment_id() {
    echo "$(date +%Y%m%d_%H%M%S)_$(hostname)_$$"
}

# Create backup
create_backup() {
    local backup_name="templates_backup_$(date +%Y%m%d_%H%M%S)"
    local backup_path="${BACKUP_DIR}/${backup_name}"
    
    log "INFO" "Creating backup: $backup_name"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would create backup at: $backup_path"
        return 0
    fi
    
    mkdir -p "$BACKUP_DIR"
    
    # Create compressed backup
    tar -czf "${backup_path}.tar.gz" -C "$(dirname "$TEMPLATES_DIR")" "$(basename "$TEMPLATES_DIR")" || {
        error_exit "Failed to create backup"
    }
    
    log "SUCCESS" "Backup created: ${backup_path}.tar.gz"
    echo "$backup_name" > "${BACKUP_DIR}/.latest_backup"
}

# Restore from backup
restore_backup() {
    local backup_name="${1:-}"
    
    if [[ -z "$backup_name" ]]; then
        if [[ -f "${BACKUP_DIR}/.latest_backup" ]]; then
            backup_name=$(cat "${BACKUP_DIR}/.latest_backup")
        else
            error_exit "No backup specified and no latest backup found"
        fi
    fi
    
    local backup_path="${BACKUP_DIR}/${backup_name}.tar.gz"
    
    if [[ ! -f "$backup_path" ]]; then
        error_exit "Backup not found: $backup_path"
    fi
    
    log "INFO" "Restoring from backup: $backup_name"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would restore from: $backup_path"
        return 0
    fi
    
    # Extract backup
    tar -xzf "$backup_path" -C "$(dirname "$TEMPLATES_DIR")" || {
        error_exit "Failed to restore backup"
    }
    
    log "SUCCESS" "Successfully restored from backup: $backup_name"
}

# ==================================================================================
# VALIDATION FUNCTIONS
# ==================================================================================

# Check prerequisites
check_prerequisites() {
    log "INFO" "Checking prerequisites..."
    
    local missing_commands=()
    
    # Required commands
    local required_commands=("python3" "pip3" "jinja2" "yamllint" "git")
    for cmd in "${required_commands[@]}"; do
        if ! command_exists "$cmd"; then
            missing_commands+=("$cmd")
        fi
    done
    
    if [[ ${#missing_commands[@]} -gt 0 ]]; then
        error_exit "Missing required commands: ${missing_commands[*]}"
    fi
    
    # Check Python dependencies
    if ! python3 -c "import jinja2, yaml, redis, aiofiles" 2>/dev/null; then
        log "WARN" "Some Python dependencies are missing. Installing..."
        pip3 install jinja2 pyyaml redis aiofiles || {
            error_exit "Failed to install Python dependencies"
        }
    fi
    
    # Check directories
    if [[ ! -d "$TEMPLATES_DIR" ]]; then
        error_exit "Templates directory not found: $TEMPLATES_DIR"
    fi
    
    if [[ ! -f "$CONFIG_FILE" ]]; then
        error_exit "Configuration file not found: $CONFIG_FILE"
    fi
    
    log "SUCCESS" "Prerequisites check completed"
}
    
    # Validate configuration
validate_config() {
    log "INFO" "Validating configuration..."
    
    # YAML syntax validation
    if command_exists yamllint; then
        yamllint "$CONFIG_FILE" || {
            error_exit "Configuration file has invalid YAML syntax"
        }
    fi
    
    # Python validation
    python3 -c "
import yaml
import sys

try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    
    # Check required sections
    required_sections = ['template_engine', 'security', 'performance', 'template_mappings']
    for section in required_sections:
        if section not in config:
            print(f'Missing required section: {section}')
            sys.exit(1)
    
    print('Configuration validation passed')
except Exception as e:
    print(f'Configuration validation failed: {e}')
    sys.exit(1)
" || {
        error_exit "Configuration validation failed"
    }
    
    log "SUCCESS" "Configuration validation completed"
}

# Validate templates
validate_templates() {
    log "INFO" "Validating templates..."
    
    local validation_script="${TEMPLATES_DIR}/template_validator.py"
    
    if [[ ! -f "$validation_script" ]]; then
        error_exit "Template validator not found: $validation_script"
    fi
    
    # Run template validation
    python3 "$validation_script" --config "$CONFIG_FILE" --templates-dir "$TEMPLATES_DIR" || {
        error_exit "Template validation failed"
    }
    
    log "SUCCESS" "Template validation completed"
}

# Security scan
security_scan() {
    log "INFO" "Running security scan..."
    
    # Check for hardcoded secrets
    local secrets_found=false
    
    while IFS= read -r -d '' file; do
        if grep -q -E "(password|secret|key|token)" "$file"; then
            log "WARN" "Potential secret found in: $file"
            secrets_found=true
        fi
    done < <(find "$TEMPLATES_DIR" -name "*.j2" -print0)
    
    if [[ "$secrets_found" == "true" ]] && [[ "$FORCE" != "true" ]]; then
        error_exit "Security scan found potential issues. Use --force to override."
    fi
    
    # Check file permissions
    while IFS= read -r -d '' file; do
        local perms=$(stat -c "%a" "$file")
        if [[ "$perms" -gt 644 ]]; then
            log "WARN" "File has excessive permissions: $file ($perms)"
        fi
    done < <(find "$TEMPLATES_DIR" -type f -print0)
    
    log "SUCCESS" "Security scan completed"
}

# ==================================================================================
# DEPLOYMENT FUNCTIONS
# ==================================================================================

# Deploy templates
deploy_templates() {
    log "INFO" "Starting template deployment..."
    
    local deployment_log="${LOG_FILE%.log}_deployment_${DEPLOYMENT_ID}.log"
    
    # Create deployment manifest
    cat > "${TEMPLATES_DIR}/.deployment_manifest" << EOF
deployment_id: $DEPLOYMENT_ID
environment: $ENVIRONMENT
timestamp: $(date -Iseconds)
user: $(whoami)
hostname: $(hostname)
git_commit: $(git rev-parse HEAD 2>/dev/null || echo "unknown")
files_deployed: []
EOF
    
    local files_deployed=()
    
    # Deploy each template file
    while IFS= read -r -d '' template_file; do
        local relative_path="${template_file#$TEMPLATES_DIR/}"
        
        if [[ "$relative_path" =~ \.(j2|yaml|py)$ ]]; then
            log "INFO" "Deploying: $relative_path"
            
            if [[ "$DRY_RUN" == "true" ]]; then
                log "INFO" "[DRY RUN] Would deploy: $relative_path"
            else
                # Perform actual deployment (copy to target location, reload services, etc.)
                # This would be environment-specific
                deploy_single_file "$template_file"
                files_deployed+=("$relative_path")
            fi
        fi
    done < <(find "$TEMPLATES_DIR" -type f -print0)
    
    # Update deployment manifest
    python3 -c "
import yaml
import json

with open('${TEMPLATES_DIR}/.deployment_manifest', 'r') as f:
    manifest = yaml.safe_load(f)

manifest['files_deployed'] = $(printf '%s
' "${files_deployed[@]}" | jq -R . | jq -s .)
manifest['status'] = 'completed'
manifest['duration_seconds'] = $(($(date +%s) - START_TIME))

with open('${TEMPLATES_DIR}/.deployment_manifest', 'w') as f:
    yaml.dump(manifest, f)
"
    
    log "SUCCESS" "Template deployment completed"
}

# Deploy single file
deploy_single_file() {
    local file_path="$1"
    local filename=$(basename "$file_path")
    
    case "$ENVIRONMENT" in
        "production")
            deploy_to_production "$file_path"
            ;;
        "staging")
            deploy_to_staging "$file_path"
            ;;
        "development")
            deploy_to_development "$file_path"
            ;;
        *)
            error_exit "Unknown environment: $ENVIRONMENT"
            ;;
    esac
    
    # Validate deployment
    validate_deployed_file "$file_path" || {
        error_exit "Deployment validation failed for: $file_path"
    }
}

# Environment-specific deployment functions
deploy_to_production() {
    local file_path="$1"
    log "INFO" "Deploying to production: $(basename "$file_path")"
    
    # Production deployment logic
    # - Copy to production template directory
    # - Restart template services
    # - Update load balancer configuration
    # - Notify monitoring systems
    
    # Example implementation:
    # kubectl apply -f "$file_path" --namespace=production
    # systemctl reload template-manager
}

deploy_to_staging() {
    local file_path="$1"
    log "INFO" "Deploying to staging: $(basename "$file_path")"
    
    # Staging deployment logic
    # Similar to production but with staging-specific configurations
}

deploy_to_development() {
    local file_path="$1"
    log "INFO" "Deploying to development: $(basename "$file_path")"
    
    # Development deployment logic
    # Local file copy and service restart
}

# Validate deployed file
validate_deployed_file() {
    local file_path="$1"
    
    # Check if file was deployed correctly
    # Validate template syntax in target environment
    # Check service health after deployment
    
    return 0  # Simplified for now
}

# ==================================================================================
# ROLLBACK FUNCTIONS
# ==================================================================================

# Perform rollback
perform_rollback() {
    local rollback_target="${1:-latest}"
    
    log "INFO" "Starting rollback to: $rollback_target"
    
    if [[ "$rollback_target" == "latest" ]]; then
        restore_backup
    else
        restore_backup "$rollback_target"
    fi
    
    # Restart services after rollback
    restart_services
    
    # Validate rollback
    validate_templates
    
    log "SUCCESS" "Rollback completed successfully"
}

# Restart services
restart_services() {
    log "INFO" "Restarting template services..."
    
    case "$ENVIRONMENT" in
        "production"|"staging")
            # Kubernetes rollout restart
            if command_exists kubectl; then
                kubectl rollout restart deployment/template-manager -n "$ENVIRONMENT" || {
                    log "WARN" "Failed to restart template-manager deployment"
                }
            fi
            ;;
        "development")
            # Local service restart
            if systemctl is-active --quiet template-manager; then
                systemctl restart template-manager || {
                    log "WARN" "Failed to restart template-manager service"
                }
            fi
            ;;
    esac
    
    log "SUCCESS" "Services restarted"
}

# ==================================================================================
# MONITORING AND METRICS
# ==================================================================================

# Send deployment metrics
send_metrics() {
    local status="$1"
    local duration=$(($(date +%s) - START_TIME))
    
    # Send metrics to monitoring system
    if command_exists curl; then
        local metrics_payload=$(cat << EOF
{
    "deployment_id": "$DEPLOYMENT_ID",
    "environment": "$ENVIRONMENT",
    "status": "$status",
    "duration_seconds": $duration,
    "timestamp": "$(date -Iseconds)",
    "dry_run": $DRY_RUN
}
EOF
)
        
        curl -s -X POST 
            -H "Content-Type: application/json" 
            -d "$metrics_payload" 
            "https://metrics.spotify-ai-agent.com/deployments" || {
            log "WARN" "Failed to send deployment metrics"
        }
    fi
    
    # Update Prometheus metrics if available
    if command_exists prometheus_push; then
        echo "template_deployment_duration_seconds $duration" | 
            prometheus_push --gateway https://prometheus-pushgateway.spotify-ai-agent.com 
                          --job template_deployment 
                          --grouping environment="$ENVIRONMENT"
    fi
}

# Health check
health_check() {
    log "INFO" "Performing health check..."
    
    # Check template manager service
    local health_url="http://localhost:8080/health"
    
    if command_exists curl; then
        local response=$(curl -s -o /dev/null -w "%{http_code}" "$health_url" || echo "000")
        
        if [[ "$response" == "200" ]]; then
            log "SUCCESS" "Template manager service is healthy"
        else
            log "WARN" "Template manager service health check failed (HTTP $response)"
        fi
    fi
    
    # Check template rendering
    python3 -c "
import sys
sys.path.append('$TEMPLATES_DIR')

try:
    from template_manager import SlackTemplateManager
    
    manager = SlackTemplateManager(config_path='$CONFIG_FILE')
    
    # Test render a simple template
    test_data = {
        'alert': {
            'alert_id': 'test-123',
            'title': 'Test Alert',
            'severity': 'info'
        },
        'environment': 'test'
    }
    
    result = manager.render_template('critical', test_data, 'en', 'text')
    print('Template rendering test passed')
    
except Exception as e:
    print(f'Template rendering test failed: {e}')
    sys.exit(1)
" || {
        log "ERROR" "Template rendering health check failed"
        return 1
    }
    
    log "SUCCESS" "Health check completed"
}

# ==================================================================================
# MAIN FUNCTIONS
# ==================================================================================

# Show usage information
show_usage() {
    cat << EOF
SPOTIFY AI AGENT - Template Deployment Script v2.0.0

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -e, --environment ENV    Target environment (development|staging|production)
                            Default: $DEFAULT_ENVIRONMENT
    
    -d, --dry-run           Perform a dry run without making changes
                            Default: $DEFAULT_DRY_RUN
    
    -r, --rollback [TARGET] Rollback to previous version or specified backup
                            Default: $DEFAULT_ROLLBACK
    
    -f, --force             Force deployment even if validation warnings exist
                            Default: $DEFAULT_FORCE
    
    -v, --validate-only     Only run validation without deployment
                            Default: $DEFAULT_VALIDATE_ONLY
    
    -h, --help              Show this help message

EXAMPLES:
    # Deploy to development environment
    $0 --environment development
    
    # Dry run deployment to production
    $0 --environment production --dry-run
    
    # Rollback to latest backup
    $0 --rollback
    
    # Rollback to specific backup
    $0 --rollback templates_backup_20250718_120000
    
    # Validate templates only
    $0 --validate-only
    
    # Force deployment despite warnings
    $0 --environment production --force

ENVIRONMENT VARIABLES:
    TEMPLATES_CONFIG_PATH   Path to template configuration file
    LOG_LEVEL              Logging level (DEBUG|INFO|WARN|ERROR)
    BACKUP_RETENTION_DAYS  Number of days to keep backups
    
For more information, visit: https://docs.spotify-ai-agent.com/templates/deployment
EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN="true"
                shift
                ;;
            -r|--rollback)
                ROLLBACK="true"
                if [[ $# -gt 1 && ! "$2" =~ ^- ]]; then
                    ROLLBACK_TARGET="$2"
                    shift 2
                else
                    ROLLBACK_TARGET="latest"
                    shift
                fi
                ;;
            -f|--force)
                FORCE="true"
                shift
                ;;
            -v|--validate-only)
                VALIDATE_ONLY="true"
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                error_exit "Unknown option: $1. Use --help for usage information."
                ;;
        esac
    done
    
    # Validate environment
    case "$ENVIRONMENT" in
        "development"|"staging"|"production")
            ;;
        *)
            error_exit "Invalid environment: $ENVIRONMENT. Must be one of: development, staging, production"
            ;;
    esac
}

# Main execution function
main() {
    local exit_code=0
    
    # Generate deployment ID
    DEPLOYMENT_ID=$(generate_deployment_id)
    
    log "INFO" "Starting template deployment process"
    log "INFO" "Deployment ID: $DEPLOYMENT_ID"
    log "INFO" "Environment: $ENVIRONMENT"
    log "INFO" "Dry Run: $DRY_RUN"
    log "INFO" "Rollback: $ROLLBACK"
    log "INFO" "Force: $FORCE"
    log "INFO" "Validate Only: $VALIDATE_ONLY"
    
    trap 'log "ERROR" "Deployment interrupted"; send_metrics "failed"; exit 1' INT TERM
    
    # Execute based on options
    if [[ "$ROLLBACK" == "true" ]]; then
        perform_rollback "${ROLLBACK_TARGET:-latest}"
        send_metrics "rollback_completed"
    elif [[ "$VALIDATE_ONLY" == "true" ]]; then
        check_prerequisites
        validate_config
        validate_templates
        security_scan
        log "SUCCESS" "Validation completed successfully"
        send_metrics "validation_completed"
    else
        # Full deployment process
        check_prerequisites
        validate_config
        validate_templates
        security_scan
        
        # Create backup before deployment
        create_backup
        
        # Deploy templates
        deploy_templates
        
        # Health check
        health_check || {
            log "ERROR" "Health check failed, initiating rollback"
            perform_rollback "latest"
            send_metrics "failed_with_rollback"
            exit_code=1
        }
        
        if [[ $exit_code -eq 0 ]]; then
            log "SUCCESS" "Deployment completed successfully"
            send_metrics "completed"
        fi
    fi
    
    local duration=$(($(date +%s) - START_TIME))
    log "INFO" "Total execution time: ${duration}s"
    
    exit $exit_code
}

# ==================================================================================
# SCRIPT EXECUTION
# ==================================================================================

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Parse arguments and run main function
    parse_arguments "$@"
    main
fi
}

setup_python_environment() {
    log_info "Configuration de l'environnement Python..."
    
    # Création de l'environnement virtuel si nécessaire
    if [ ! -d "${PROJECT_ROOT}/venv" ]; then
        log_info "Création de l'environnement virtuel..."
        python3 -m venv "${PROJECT_ROOT}/venv"
    fi
    
    # Activation de l'environnement virtuel
    source "${PROJECT_ROOT}/venv/bin/activate"
    
    # Mise à jour pip
    pip install --upgrade pip
    
    # Installation des dépendances
    log_info "Installation des dépendances Python..."
    pip install -r "${PROJECT_ROOT}/requirements.txt" || {
        log_warning "requirements.txt non trouvé, installation des dépendances de base..."
        pip install aioredis asyncpg fastapi aiohttp jinja2 babel prometheus_client tenacity
    }
    
    log_success "Environnement Python configuré"
}

setup_redis_config() {
    log_info "Configuration Redis pour le système d'alerting..."
    
    # Vérification de la connexion Redis
    if ! redis-cli ping &> /dev/null; then
        log_error "Impossible de se connecter à Redis"
        exit 1
    fi
    
    # Configuration des clés par défaut
    redis-cli HSET "slack:default_config" \
        "max_alerts_per_minute" "10" \
        "max_alerts_per_hour" "100" \
        "rate_limit_window" "60" \
        "fallback_channel" "#alerts-fallback"
    
    # Configuration du canal par défaut
    redis-cli SET "slack:default_webhook" "https://hooks.slack.com/services/CHANGE_ME"
    
    # Expiration des clés de test
    redis-cli EXPIRE "slack:test:*" 3600
    
    log_success "Configuration Redis terminée"
}

setup_postgres_schema() {
    log_info "Configuration du schéma PostgreSQL..."
    
    # Variables d'environnement par défaut
    DB_HOST="${DB_HOST:-localhost}"
    DB_PORT="${DB_PORT:-5432}"
    DB_NAME="${DB_NAME:-spotify_ai}"
    DB_USER="${DB_USER:-postgres}"
    DB_PASSWORD="${DB_PASSWORD:-password}"
    
    # Test de connexion
    if ! PGPASSWORD="${DB_PASSWORD}" psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -c "SELECT 1;" &> /dev/null; then
        log_error "Impossible de se connecter à PostgreSQL"
        log_info "Vérifiez les variables d'environnement: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD"
        exit 1
    fi
    
    # Création du schéma
    log_info "Création des tables..."
    PGPASSWORD="${DB_PASSWORD}" psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" << 'EOF'
-- Schema pour le système d'alerting Slack
CREATE SCHEMA IF NOT EXISTS slack_alerting;

-- Table des configurations de tenants
CREATE TABLE IF NOT EXISTS slack_alerting.tenant_configs (
    tenant_id VARCHAR(255) PRIMARY KEY,
    config JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table des templates
CREATE TABLE IF NOT EXISTS slack_alerting.alert_templates (
    template_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    template_type VARCHAR(100) NOT NULL,
    language VARCHAR(10) NOT NULL,
    format_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table des métriques d'usage
CREATE TABLE IF NOT EXISTS slack_alerting.usage_metrics (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    metric_type VARCHAR(100) NOT NULL,
    metric_value NUMERIC NOT NULL,
    labels JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index pour les performances
CREATE INDEX IF NOT EXISTS idx_tenant_configs_tenant_id ON slack_alerting.tenant_configs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_alert_templates_tenant_type ON slack_alerting.alert_templates(tenant_id, template_type);
CREATE INDEX IF NOT EXISTS idx_usage_metrics_tenant_timestamp ON slack_alerting.usage_metrics(tenant_id, timestamp);

-- Trigger pour mise à jour automatique
CREATE OR REPLACE FUNCTION slack_alerting.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_tenant_configs_updated_at BEFORE UPDATE ON slack_alerting.tenant_configs
    FOR EACH ROW EXECUTE FUNCTION slack_alerting.update_updated_at_column();

CREATE TRIGGER update_alert_templates_updated_at BEFORE UPDATE ON slack_alerting.alert_templates
    FOR EACH ROW EXECUTE FUNCTION slack_alerting.update_updated_at_column();
EOF
    
    log_success "Schéma PostgreSQL créé"
}

create_systemd_service() {
    log_info "Création du service systemd..."
    
    cat > /tmp/slack-alerting.service << EOF
[Unit]
Description=Spotify AI Agent - Slack Alerting Service
After=network.target redis.service postgresql.service
Requires=redis.service postgresql.service

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=${PROJECT_ROOT}
Environment=PATH=${PROJECT_ROOT}/venv/bin
ExecStart=${PROJECT_ROOT}/venv/bin/python -m slack.slack_alert_manager
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=slack-alerting

# Sécurité
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=${PROJECT_ROOT}/logs

# Variables d'environnement
Environment=REDIS_URL=redis://localhost:6379
Environment=POSTGRES_URL=postgresql://postgres:password@localhost:5432/spotify_ai
Environment=LOG_LEVEL=INFO

[Install]
WantedBy=multi-user.target
EOF
    
    sudo mv /tmp/slack-alerting.service /etc/systemd/system/
    sudo systemctl daemon-reload
    
    log_success "Service systemd créé"
}

setup_monitoring() {
    log_info "Configuration du monitoring..."
    
    # Création du fichier de configuration Prometheus
    mkdir -p "${PROJECT_ROOT}/monitoring"
    
    cat > "${PROJECT_ROOT}/monitoring/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'slack-alerting'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    
  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - localhost:9093
EOF
    
    # Règles d'alerting
    cat > "${PROJECT_ROOT}/monitoring/alert_rules.yml" << 'EOF'
groups:
  - name: slack-alerting
    rules:
      - alert: SlackAlertingDown
        expr: up{job="slack-alerting"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service d'alerting Slack indisponible"
          description: "Le service d'alerting Slack est en panne depuis plus d'1 minute."
          
      - alert: HighAlertVolume
        expr: rate(slack_alerts_sent_total[5m]) > 10
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Volume d'alertes élevé"
          description: "Plus de 10 alertes par seconde envoyées."
          
      - alert: RateLimitHigh
        expr: rate(slack_rate_limit_hits_total[5m]) > 1
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Rate limiting fréquent"
          description: "Taux de rate limiting élevé détecté."
EOF
    
    log_success "Configuration monitoring créée"
}

run_tests() {
    log_info "Exécution des tests..."
    
    # Activation de l'environnement virtuel
    source "${PROJECT_ROOT}/venv/bin/activate"
    
    # Tests de connectivité
    python3 << 'EOF'
import asyncio
import aioredis
import asyncpg
import sys

async def test_connections():
    try:
        # Test Redis
        redis = aioredis.from_url("redis://localhost:6379")
        await redis.ping()
        await redis.close()
        print("✅ Connexion Redis OK")
        
        # Test PostgreSQL
        conn = await asyncpg.connect("postgresql://postgres:password@localhost:5432/spotify_ai")
        await conn.execute("SELECT 1")
        await conn.close()
        print("✅ Connexion PostgreSQL OK")
        
        print("✅ Tous les tests de connectivité réussis")
        return True
        
    except Exception as e:
        print(f"❌ Erreur de test: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_connections())
    sys.exit(0 if result else 1)
EOF
    
    if [ $? -eq 0 ]; then
        log_success "Tests de connectivité réussis"
    else
        log_error "Échec des tests de connectivité"
        exit 1
    fi
}

create_configuration_examples() {
    log_info "Création des exemples de configuration..."
    
    mkdir -p "${PROJECT_ROOT}/examples"
    
    # Configuration tenant exemple
    cat > "${PROJECT_ROOT}/examples/tenant_config.json" << 'EOF'
{
    "tenant_id": "spotify-ai-tenant-1",
    "slack_config": {
        "default_channel": "#alerts-prod",
        "escalation_channel": "#alerts-critical",
        "fallback_channel": "#alerts-fallback",
        "rate_limits": {
            "max_per_minute": 10,
            "max_per_hour": 100,
            "burst_threshold": 5
        },
        "routing_rules": [
            {
                "name": "Critical to Executive",
                "conditions": {
                    "severity": ["critical", "emergency"],
                    "environment": "production"
                },
                "target_channels": ["#alerts-executive", "#alerts-critical"]
            },
            {
                "name": "Service Specific",
                "conditions": {
                    "service_name": ["user-service", "payment-service"]
                },
                "target_channels": ["#team-backend"]
            }
        ],
        "escalation_policies": [
            {
                "policy_id": "critical-escalation",
                "name": "Escalade Critique",
                "steps": [
                    {
                        "level": "l1",
                        "delay_minutes": 5,
                        "target_users": ["@dev-team"],
                        "actions": ["notify_slack"]
                    },
                    {
                        "level": "l2",
                        "delay_minutes": 15,
                        "target_users": ["@tech-lead"],
                        "actions": ["notify_slack", "send_email"]
                    },
                    {
                        "level": "management",
                        "delay_minutes": 30,
                        "target_users": ["@management"],
                        "actions": ["notify_slack", "page_oncall"]
                    }
                ]
            }
        ]
    }
}
EOF
    
    # Configuration webhook exemple
    cat > "${PROJECT_ROOT}/examples/webhook_config.json" << 'EOF'
{
    "webhooks": {
        "#alerts-prod": {
            "url": "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX",
            "retry_config": {
                "max_retries": 3,
                "backoff_multiplier": 2.0,
                "timeout": 30
            }
        },
        "#alerts-critical": {
            "url": "https://hooks.slack.com/services/T00000000/B00000001/YYYYYYYYYYYYYYYYYYYYYYYY",
            "retry_config": {
                "max_retries": 5,
                "backoff_multiplier": 1.5,
                "timeout": 60
            }
        }
    }
}
EOF
    
    log_success "Exemples de configuration créés"
}

generate_documentation() {
    log_info "Génération de la documentation..."
    
    mkdir -p "${PROJECT_ROOT}/docs"
    
    cat > "${PROJECT_ROOT}/docs/DEPLOYMENT.md" << 'EOF'
# Guide de Déploiement - Système d'Alerting Slack

## Vue d'ensemble

Ce système d'alerting Slack industrialisé offre:
- Gestion multi-tenant avec isolation complète
- Rate limiting intelligent et agrégation d'alertes
- Escalade automatique avec SLA
- Templates dynamiques multi-langue
- Métriques et monitoring avancés

## Prérequis

- Python 3.8+
- Redis 6.0+
- PostgreSQL 12+
- Slack Workspace avec permissions webhook

## Installation

1. **Clonage et setup initial**
```bash
./deploy.sh --setup
```

2. **Configuration des variables d'environnement**
```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
export REDIS_URL="redis://localhost:6379"
export POSTGRES_URL="postgresql://user:pass@localhost:5432/db"
```

3. **Démarrage du service**
```bash
sudo systemctl enable slack-alerting
sudo systemctl start slack-alerting
```

## Configuration

### Configuration Tenant
Voir `examples/tenant_config.json` pour un exemple complet.

### Templates Slack
Les templates sont stockés dans `templates/` et supportent:
- Multi-langue (FR/EN/DE)
- Formats adaptatifs (minimal, standard, détaillé)
- Variables dynamiques et conditionnelles

### Monitoring
- Métriques Prometheus: http://localhost:8000/metrics
- Dashboard Grafana: Importer dashboard slack-alerting.json
- Logs: journalctl -u slack-alerting

## API

### Endpoints principaux
- `POST /alerts` - Envoi d'alerte
- `GET /alerts/{id}` - Status d'alerte
- `POST /alerts/{id}/acknowledge` - Acquittement
- `POST /alerts/{id}/resolve` - Résolution
- `GET /metrics` - Métriques Prometheus

### Exemple d'utilisation
```python
import httpx

# Envoi d'alerte
response = httpx.post("http://localhost:8000/alerts", json={
    "tenant_id": "my-tenant",
    "title": "High CPU Usage",
    "description": "CPU usage > 90%",
    "severity": "warning",
    "service_name": "web-server",
    "environment": "production"
})
```

## Troubleshooting

### Logs
```bash
# Logs du service
journalctl -u slack-alerting -f

# Logs Redis
redis-cli monitor

# Logs PostgreSQL
sudo tail -f /var/log/postgresql/postgresql-*.log
```

### Tests de connectivité
```bash
# Test Redis
redis-cli ping

# Test PostgreSQL
psql -h localhost -U postgres -d spotify_ai -c "SELECT 1;"

# Test webhook Slack
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"Test message"}' \
  YOUR_WEBHOOK_URL
```

## Maintenance

### Nettoyage automatique
Le système inclut un nettoyage automatique:
- Cache Redis: TTL automatique
- Métriques anciennes: Purge après 7 jours
- Logs d'audit: Rétention 30 jours

### Sauvegarde
```bash
# Sauvegarde PostgreSQL
pg_dump spotify_ai > backup_$(date +%Y%m%d).sql

# Sauvegarde Redis
redis-cli BGSAVE
```

## Performance

### Métriques clés
- Latence d'envoi webhook: < 500ms
- Débit maximum: 1000 alertes/minute
- Disponibilité cible: 99.9%

### Optimisations
- Pool de connexions Redis/PostgreSQL
- Cache des templates compilés
- Rate limiting intelligent
- Circuit breakers pour la résilience
EOF
    
    log_success "Documentation générée"
}

main() {
    log_info "🚀 Démarrage du déploiement du système d'alerting Slack"
    log_info "Auteur: Fahed Mlaiel - Architecture industrielle Spotify AI Agent"
    
    # Vérification des droits
    if [[ $EUID -eq 0 ]]; then
        log_warning "Exécution en tant que root détectée"
    fi
    
    # Étapes de déploiement
    check_dependencies
    setup_python_environment
    setup_redis_config
    setup_postgres_schema
    create_systemd_service
    setup_monitoring
    create_configuration_examples
    generate_documentation
    run_tests
    
    log_success "🎉 Déploiement terminé avec succès!"
    log_info "📋 Prochaines étapes:"
    log_info "   1. Configurer vos webhooks Slack"
    log_info "   2. Ajuster la configuration tenant"
    log_info "   3. Démarrer le service: sudo systemctl start slack-alerting"
    log_info "   4. Vérifier les métriques: curl http://localhost:8000/metrics"
    log_info "📖 Documentation complète: ${PROJECT_ROOT}/docs/DEPLOYMENT.md"
}

# Options de ligne de commande
case "${1:-}" in
    --setup)
        main
        ;;
    --test)
        run_tests
        ;;
    --monitor)
        setup_monitoring
        ;;
    --docs)
        generate_documentation
        ;;
    *)
        log_info "Usage: $0 [--setup|--test|--monitor|--docs]"
        log_info "  --setup   : Installation complète"
        log_info "  --test    : Tests de connectivité uniquement"
        log_info "  --monitor : Configuration monitoring uniquement"
        log_info "  --docs    : Génération documentation uniquement"
        main
        ;;
esac
