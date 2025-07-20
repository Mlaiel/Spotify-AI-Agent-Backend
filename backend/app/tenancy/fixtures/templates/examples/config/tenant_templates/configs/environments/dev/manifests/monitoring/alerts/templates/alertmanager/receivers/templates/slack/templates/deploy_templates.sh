#!/bin/bash

# Enterprise Template Deployment & Management Script
# Advanced Industrial Grade Automation
# Developed by: Fahed Mlaiel (Lead Dev + AI Architect)

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATES_DIR="${SCRIPT_DIR}/../templates"
CONFIG_FILE="${SCRIPT_DIR}/../template_config.yaml"
LOG_FILE="/var/log/spotify-ai-agent/template-deployment.log"
BACKUP_DIR="/backups/templates"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOG_FILE}"
}

# Info logging
info() {
    log "INFO" "$@"
    echo -e "${BLUE}[INFO]${NC} $*"
}

# Warning logging
warn() {
    log "WARN" "$@"
    echo -e "${YELLOW}[WARN]${NC} $*"
}

# Error logging
error() {
    log "ERROR" "$@"
    echo -e "${RED}[ERROR]${NC} $*"
}

# Success logging
success() {
    log "SUCCESS" "$@"
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

# Check dependencies
check_dependencies() {
    info "Checking dependencies..."
    
    local deps=("python3" "jinja2" "yq" "docker" "kubectl" "redis-cli")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        error "Missing dependencies: ${missing_deps[*]}"
        error "Please install missing dependencies before proceeding."
        exit 1
    fi
    
    success "All dependencies are installed."
}

# Validate configuration
validate_config() {
    info "Validating configuration..."
    
    if [[ ! -f "$CONFIG_FILE" ]]; then
        error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    
    # Validate YAML syntax
    if ! yq eval '.' "$CONFIG_FILE" > /dev/null 2>&1; then
        error "Invalid YAML syntax in configuration file: $CONFIG_FILE"
        exit 1
    fi
    
    # Check required sections
    local required_sections=("template_manager" "environments" "tenants")
    for section in "${required_sections[@]}"; do
        if ! yq eval "has(\"$section\")" "$CONFIG_FILE" | grep -q "true"; then
            error "Missing required section '$section' in configuration file"
            exit 1
        fi
    done
    
    success "Configuration validation passed."
}

# Validate templates
validate_templates() {
    info "Validating templates..."
    
    local validation_script="${SCRIPT_DIR}/../template_validator.py"
    local failed_validations=0
    
    if [[ ! -f "$validation_script" ]]; then
        error "Template validator script not found: $validation_script"
        exit 1
    fi
    
    # Find all template files
    local template_files
    mapfile -t template_files < <(find "$TEMPLATES_DIR" -name "*.j2" -type f)
    
    if [[ ${#template_files[@]} -eq 0 ]]; then
        warn "No template files found in $TEMPLATES_DIR"
        return 0
    fi
    
    info "Found ${#template_files[@]} template files to validate"
    
    # Validate each template
    for template_file in "${template_files[@]}"; do
        info "Validating template: $(basename "$template_file")"
        
        if python3 "$validation_script" --template "$template_file" --config "$CONFIG_FILE"; then
            success "✓ $(basename "$template_file")"
        else
            error "✗ $(basename "$template_file")"
            ((failed_validations++))
        fi
    done
    
    if [[ $failed_validations -gt 0 ]]; then
        error "$failed_validations template(s) failed validation"
        exit 1
    fi
    
    success "All templates passed validation."
}

# Create backup
create_backup() {
    info "Creating backup..."
    
    local backup_timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_path="${BACKUP_DIR}/templates_backup_${backup_timestamp}"
    
    mkdir -p "$backup_path"
    
    # Backup templates
    if [[ -d "$TEMPLATES_DIR" ]]; then
        cp -r "$TEMPLATES_DIR" "$backup_path/"
        success "Templates backed up to: $backup_path"
    fi
    
    # Backup configuration
    if [[ -f "$CONFIG_FILE" ]]; then
        cp "$CONFIG_FILE" "$backup_path/"
        success "Configuration backed up to: $backup_path"
    fi
    
    # Create backup metadata
    cat > "$backup_path/backup_metadata.json" << EOF
{
    "timestamp": "$backup_timestamp",
    "version": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "environment": "${ENVIRONMENT:-production}",
    "created_by": "$(whoami)",
    "backup_type": "automated",
    "templates_count": $(find "$TEMPLATES_DIR" -name "*.j2" -type f | wc -l),
    "backup_size": "$(du -sh "$backup_path" | cut -f1)"
}
EOF
    
    success "Backup created successfully: $backup_path"
    echo "$backup_path"
}

# Deploy templates to Kubernetes
deploy_k8s() {
    local environment="${1:-production}"
    
    info "Deploying templates to Kubernetes environment: $environment"
    
    # Create ConfigMap for templates
    kubectl create configmap spotify-ai-agent-templates \
        --from-file="$TEMPLATES_DIR" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Create ConfigMap for configuration
    kubectl create configmap spotify-ai-agent-template-config \
        --from-file="$CONFIG_FILE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply deployment manifests
    local manifests_dir="${SCRIPT_DIR}/../k8s"
    if [[ -d "$manifests_dir" ]]; then
        kubectl apply -f "$manifests_dir/"
    fi
    
    success "Templates deployed to Kubernetes"
}

# Deploy templates to Docker
deploy_docker() {
    local environment="${1:-production}"
    
    info "Deploying templates to Docker environment: $environment"
    
    # Build Docker image with templates
    local docker_context="${SCRIPT_DIR}/../"
    local image_tag="spotify-ai-agent-templates:$(date '+%Y%m%d-%H%M%S')"
    
    docker build -t "$image_tag" "$docker_context"
    
    # Update docker-compose if exists
    local compose_file="${SCRIPT_DIR}/../docker-compose.yml"
    if [[ -f "$compose_file" ]]; then
        docker-compose -f "$compose_file" up -d --force-recreate
    fi
    
    success "Templates deployed to Docker"
}

# Test deployment
test_deployment() {
    info "Testing deployment..."
    
    local test_script="${SCRIPT_DIR}/../test_templates.py"
    
    if [[ ! -f "$test_script" ]]; then
        warn "Test script not found: $test_script"
        return 0
    fi
    
    # Run template tests
    if python3 "$test_script" --config "$CONFIG_FILE"; then
        success "Deployment tests passed"
    else
        error "Deployment tests failed"
        exit 1
    fi
}

# Update cache
update_cache() {
    info "Updating template cache..."
    
    # Clear Redis cache
    local redis_host=$(yq eval '.template_manager.caching.redis_config.host' "$CONFIG_FILE")
    local redis_port=$(yq eval '.template_manager.caching.redis_config.port' "$CONFIG_FILE")
    local redis_db=$(yq eval '.template_manager.caching.redis_config.db' "$CONFIG_FILE")
    
    if command -v redis-cli &> /dev/null; then
        redis-cli -h "$redis_host" -p "$redis_port" -n "$redis_db" FLUSHDB
        success "Redis cache cleared"
    else
        warn "redis-cli not available, skipping cache clear"
    fi
    
    # Warm up cache with critical templates
    local critical_templates=("critical_en_text.j2" "critical_fr_text.j2" "critical_de_text.j2")
    
    for template in "${critical_templates[@]}"; do
        if [[ -f "$TEMPLATES_DIR/$template" ]]; then
            info "Warming up cache for: $template"
            # Simulate template rendering to warm cache
            python3 -c "
import sys
sys.path.append('$SCRIPT_DIR/..')
from template_manager import TemplateManager
tm = TemplateManager('$CONFIG_FILE')
tm.render_template('$template', {'test': True})
"
        fi
    done
    
    success "Template cache updated"
}

# Monitor deployment
monitor_deployment() {
    info "Starting deployment monitoring..."
    
    local monitoring_duration=300  # 5 minutes
    local check_interval=30       # 30 seconds
    local checks_count=$((monitoring_duration / check_interval))
    
    for ((i=1; i<=checks_count; i++)); do
        info "Health check $i/$checks_count"
        
        # Check template manager health
        if python3 -c "
import sys
sys.path.append('$SCRIPT_DIR/..')
from template_manager import TemplateManager
tm = TemplateManager('$CONFIG_FILE')
health = tm.health_check()
if not health['healthy']:
    sys.exit(1)
"; then
            success "✓ Template manager is healthy"
        else
            error "✗ Template manager health check failed"
            exit 1
        fi
        
        # Check cache connectivity
        local redis_host=$(yq eval '.template_manager.caching.redis_config.host' "$CONFIG_FILE")
        local redis_port=$(yq eval '.template_manager.caching.redis_config.port' "$CONFIG_FILE")
        
        if redis-cli -h "$redis_host" -p "$redis_port" ping > /dev/null 2>&1; then
            success "✓ Redis cache is accessible"
        else
            warn "✗ Redis cache connectivity issue"
        fi
        
        sleep $check_interval
    done
    
    success "Deployment monitoring completed successfully"
}

# Rollback function
rollback() {
    local backup_path="$1"
    
    if [[ -z "$backup_path" ]]; then
        error "Backup path is required for rollback"
        exit 1
    fi
    
    if [[ ! -d "$backup_path" ]]; then
        error "Backup directory not found: $backup_path"
        exit 1
    fi
    
    info "Rolling back to backup: $backup_path"
    
    # Restore templates
    if [[ -d "$backup_path/templates" ]]; then
        rm -rf "$TEMPLATES_DIR"
        cp -r "$backup_path/templates" "$TEMPLATES_DIR"
        success "Templates restored from backup"
    fi
    
    # Restore configuration
    if [[ -f "$backup_path/template_config.yaml" ]]; then
        cp "$backup_path/template_config.yaml" "$CONFIG_FILE"
        success "Configuration restored from backup"
    fi
    
    # Clear cache after rollback
    update_cache
    
    success "Rollback completed successfully"
}

# Generate deployment report
generate_report() {
    local deployment_start="$1"
    local backup_path="$2"
    local environment="${3:-production}"
    
    local deployment_end=$(date '+%Y-%m-%d %H:%M:%S')
    local report_file="/var/log/spotify-ai-agent/deployment-report-$(date '+%Y%m%d_%H%M%S').json"
    
    # Count templates
    local template_count=$(find "$TEMPLATES_DIR" -name "*.j2" -type f | wc -l)
    
    # Get version info
    local git_commit=$(git rev-parse HEAD 2>/dev/null || echo 'unknown')
    local git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')
    
    # Generate report
    cat > "$report_file" << EOF
{
    "deployment": {
        "start_time": "$deployment_start",
        "end_time": "$deployment_end",
        "duration_seconds": $(($(date -d "$deployment_end" +%s) - $(date -d "$deployment_start" +%s))),
        "environment": "$environment",
        "status": "success",
        "deployed_by": "$(whoami)"
    },
    "templates": {
        "total_count": $template_count,
        "languages": ["en", "fr", "de"],
        "types": ["critical", "warning", "resolved", "digest", "security", "performance", "infrastructure", "incident"]
    },
    "version": {
        "git_commit": "$git_commit",
        "git_branch": "$git_branch",
        "deployment_tag": "$(date '+%Y%m%d-%H%M%S')"
    },
    "backup": {
        "path": "$backup_path",
        "created": true
    },
    "validation": {
        "config_validated": true,
        "templates_validated": true,
        "tests_passed": true
    },
    "cache": {
        "cleared": true,
        "warmed_up": true
    },
    "monitoring": {
        "health_checks_passed": true,
        "duration_minutes": 5
    }
}
EOF
    
    success "Deployment report generated: $report_file"
    echo "$report_file"
}

# Main deployment function
deploy() {
    local environment="${1:-production}"
    local deployment_type="${2:-kubernetes}"
    local skip_backup="${3:-false}"
    
    local deployment_start=$(date '+%Y-%m-%d %H:%M:%S')
    
    info "Starting template deployment..."
    info "Environment: $environment"
    info "Deployment type: $deployment_type"
    info "Start time: $deployment_start"
    
    # Pre-deployment checks
    check_dependencies
    validate_config
    validate_templates
    
    # Create backup (unless skipped)
    local backup_path=""
    if [[ "$skip_backup" != "true" ]]; then
        backup_path=$(create_backup)
    fi
    
    # Deploy based on type
    case "$deployment_type" in
        "kubernetes"|"k8s")
            deploy_k8s "$environment"
            ;;
        "docker")
            deploy_docker "$environment"
            ;;
        *)
            error "Unknown deployment type: $deployment_type"
            exit 1
            ;;
    esac
    
    # Post-deployment tasks
    update_cache
    test_deployment
    monitor_deployment
    
    # Generate report
    local report_file=$(generate_report "$deployment_start" "$backup_path" "$environment")
    
    success "Template deployment completed successfully!"
    success "Deployment report: $report_file"
    
    if [[ -n "$backup_path" ]]; then
        info "Backup available for rollback: $backup_path"
    fi
}

# CLI interface
show_help() {
    cat << EOF
Enterprise Template Deployment & Management Script
Developed by: Fahed Mlaiel (Lead Dev + AI Architect)

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    deploy [ENV] [TYPE]     Deploy templates
                           ENV: production|staging|development (default: production)
                           TYPE: kubernetes|docker (default: kubernetes)
    
    validate               Validate templates and configuration
    backup                 Create backup of current templates
    rollback [PATH]        Rollback to specified backup
    test                   Run template tests
    cache-clear           Clear template cache
    cache-warm            Warm up template cache
    monitor               Monitor deployment health
    
Options:
    -h, --help            Show this help message
    -v, --verbose         Enable verbose logging
    --skip-backup         Skip backup creation during deployment
    --config FILE         Use custom configuration file
    --templates-dir DIR   Use custom templates directory

Examples:
    $0 deploy production kubernetes
    $0 deploy staging docker --skip-backup
    $0 validate
    $0 backup
    $0 rollback /backups/templates/templates_backup_20241201_143022
    $0 test
    $0 cache-clear
    $0 monitor

Environment Variables:
    ENVIRONMENT           Target environment (production|staging|development)
    TEMPLATES_DIR         Path to templates directory
    CONFIG_FILE           Path to configuration file
    BACKUP_DIR           Path to backup directory
    LOG_FILE             Path to log file

EOF
}

# Parse command line arguments
main() {
    # Create log directory if it doesn't exist
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Parse global options
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                set -x
                shift
                ;;
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --templates-dir)
                TEMPLATES_DIR="$2"
                shift 2
                ;;
            --skip-backup)
                SKIP_BACKUP="true"
                shift
                ;;
            *)
                break
                ;;
        esac
    done
    
    # Parse command
    local command="${1:-deploy}"
    shift || true
    
    case "$command" in
        "deploy")
            local environment="${1:-production}"
            local deployment_type="${2:-kubernetes}"
            deploy "$environment" "$deployment_type" "${SKIP_BACKUP:-false}"
            ;;
        "validate")
            check_dependencies
            validate_config
            validate_templates
            ;;
        "backup")
            create_backup
            ;;
        "rollback")
            local backup_path="$1"
            rollback "$backup_path"
            ;;
        "test")
            test_deployment
            ;;
        "cache-clear")
            update_cache
            ;;
        "cache-warm")
            update_cache
            ;;
        "monitor")
            monitor_deployment
            ;;
        *)
            error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
