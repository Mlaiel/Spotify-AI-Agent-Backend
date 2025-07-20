#!/bin/bash

# ==================================================================================
# SPOTIFY AI AGENT - ENTERPRISE TEMPLATE ROLLBACK SCRIPT
# ==================================================================================
# Description: Advanced rollback automation for template deployments
# Version: 2.0.0
# Author: Fahed Mlaiel (Lead Dev + AI Architect)
# Last Modified: 2025-07-18
# ==================================================================================

set -euo pipefail

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
readonly BACKUP_DIR="/var/backups/spotify-ai-agent/templates"
readonly LOG_FILE="/var/log/spotify-ai-agent/template-rollback.log"
readonly DEPLOYMENT_MANIFEST="${SCRIPT_DIR}/.deployment_manifest"
readonly ROLLBACK_HISTORY="${BACKUP_DIR}/.rollback_history"

# Default values
DEFAULT_ENVIRONMENT="development"
DEFAULT_DRY_RUN="false"
DEFAULT_FORCE="false"
DEFAULT_LIST_BACKUPS="false"

# Global variables
ENVIRONMENT="${DEFAULT_ENVIRONMENT}"
DRY_RUN="${DEFAULT_DRY_RUN}"
FORCE="${DEFAULT_FORCE}"
LIST_BACKUPS="${DEFAULT_LIST_BACKUPS}"
ROLLBACK_TARGET=""
ROLLBACK_ID=""
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

# Generate rollback ID
generate_rollback_id() {
    echo "rollback_$(date +%Y%m%d_%H%M%S)_$(hostname)_$$"
}

# ==================================================================================
# BACKUP MANAGEMENT
# ==================================================================================

# List available backups
list_available_backups() {
    log "INFO" "Listing available backups..."
    
    if [[ ! -d "$BACKUP_DIR" ]]; then
        log "WARN" "No backup directory found: $BACKUP_DIR"
        return 0
    fi
    
    local backups=()
    while IFS= read -r -d '' backup_file; do
        local backup_name=$(basename "$backup_file" .tar.gz)
        backups+=("$backup_name")
    done < <(find "$BACKUP_DIR" -name "templates_backup_*.tar.gz" -print0 | sort -z)
    
    if [[ ${#backups[@]} -eq 0 ]]; then
        log "WARN" "No backups found"
        return 0
    fi
    
    echo ""
    echo -e "${CYAN}Available Backups:${NC}"
    echo "=================="
    
    for backup in "${backups[@]}"; do
        local backup_file="${BACKUP_DIR}/${backup}.tar.gz"
        local backup_date=$(echo "$backup" | sed 's/templates_backup_\([0-9]\{8\}_[0-9]\{6\}\).*/\1/' | sed 's/_/ /')
        local backup_size=$(du -h "$backup_file" | cut -f1)
        
        # Check if this is the latest backup
        local latest_marker=""
        if [[ -f "${BACKUP_DIR}/.latest_backup" ]]; then
            local latest_backup=$(cat "${BACKUP_DIR}/.latest_backup")
            if [[ "$backup" == "$latest_backup" ]]; then
                latest_marker=" ${GREEN}(latest)${NC}"
            fi
        fi
        
        echo -e "  ${BLUE}•${NC} $backup - $backup_date - $backup_size$latest_marker"
        
        # Show backup metadata if available
        local metadata_file="${BACKUP_DIR}/${backup}_metadata.json"
        if [[ -f "$metadata_file" ]]; then
            local environment=$(jq -r '.environment // "unknown"' "$metadata_file" 2>/dev/null)
            local git_commit=$(jq -r '.git_commit // "unknown"' "$metadata_file" 2>/dev/null | cut -c1-8)
            echo -e "    Environment: $environment, Git: $git_commit"
        fi
    done
    
    echo ""
}

# Validate backup exists
validate_backup() {
    local backup_name="$1"
    local backup_file="${BACKUP_DIR}/${backup_name}.tar.gz"
    
    if [[ ! -f "$backup_file" ]]; then
        error_exit "Backup not found: $backup_file"
    fi
    
    # Validate backup integrity
    if ! tar -tzf "$backup_file" >/dev/null 2>&1; then
        error_exit "Backup file is corrupted: $backup_file"
    fi
    
    log "SUCCESS" "Backup validation passed: $backup_name"
}

# Get backup metadata
get_backup_metadata() {
    local backup_name="$1"
    local metadata_file="${BACKUP_DIR}/${backup_name}_metadata.json"
    
    if [[ -f "$metadata_file" ]]; then
        cat "$metadata_file"
    else
        echo '{"environment": "unknown", "git_commit": "unknown", "created_at": "unknown"}'
    fi
}

# ==================================================================================
# ROLLBACK FUNCTIONS
# ==================================================================================

# Create pre-rollback backup
create_pre_rollback_backup() {
    log "INFO" "Creating pre-rollback backup..."
    
    local pre_rollback_name="pre_rollback_$(date +%Y%m%d_%H%M%S)"
    local backup_path="${BACKUP_DIR}/${pre_rollback_name}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would create pre-rollback backup: $pre_rollback_name"
        return 0
    fi
    
    mkdir -p "$BACKUP_DIR"
    
    # Create compressed backup
    tar -czf "${backup_path}.tar.gz" -C "$(dirname "$SCRIPT_DIR")" "$(basename "$SCRIPT_DIR")" || {
        error_exit "Failed to create pre-rollback backup"
    }
    
    # Create metadata
    cat > "${backup_path}_metadata.json" << EOF
{
    "backup_type": "pre_rollback",
    "environment": "$ENVIRONMENT",
    "created_at": "$(date -Iseconds)",
    "rollback_id": "$ROLLBACK_ID",
    "target_backup": "$ROLLBACK_TARGET",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo "unknown")"
}
EOF
    
    log "SUCCESS" "Pre-rollback backup created: $pre_rollback_name"
    echo "$pre_rollback_name" > "${BACKUP_DIR}/.pre_rollback_backup"
}

# Perform rollback
perform_rollback() {
    local backup_name="$1"
    
    log "INFO" "Starting rollback to: $backup_name"
    
    # Validate backup
    validate_backup "$backup_name"
    
    # Get backup metadata
    local metadata=$(get_backup_metadata "$backup_name")
    local backup_env=$(echo "$metadata" | jq -r '.environment // "unknown"')
    local backup_commit=$(echo "$metadata" | jq -r '.git_commit // "unknown"')
    
    log "INFO" "Backup environment: $backup_env"
    log "INFO" "Backup commit: $backup_commit"
    
    # Environment validation
    if [[ "$backup_env" != "unknown" && "$backup_env" != "$ENVIRONMENT" ]] && [[ "$FORCE" != "true" ]]; then
        error_exit "Backup environment ($backup_env) doesn't match target environment ($ENVIRONMENT). Use --force to override."
    fi
    
    # Create pre-rollback backup
    create_pre_rollback_backup
    
    # Perform the actual rollback
    execute_rollback "$backup_name"
    
    # Validate rollback
    validate_rollback "$backup_name"
    
    # Record rollback history
    record_rollback_history "$backup_name"
    
    log "SUCCESS" "Rollback completed successfully"
}

# Execute rollback
execute_rollback() {
    local backup_name="$1"
    local backup_file="${BACKUP_DIR}/${backup_name}.tar.gz"
    
    log "INFO" "Executing rollback from: $backup_file"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would extract backup to: $(dirname "$SCRIPT_DIR")"
        log "INFO" "[DRY RUN] Would restart services"
        return 0
    fi
    
    # Stop services before rollback
    stop_services
    
    # Extract backup
    log "INFO" "Extracting backup..."
    tar -xzf "$backup_file" -C "$(dirname "$SCRIPT_DIR")" || {
        error_exit "Failed to extract backup"
    }
    
    # Set correct permissions
    chmod -R 644 "${SCRIPT_DIR}"/*.j2 2>/dev/null || true
    chmod -R 644 "${SCRIPT_DIR}"/*.py 2>/dev/null || true
    chmod +x "${SCRIPT_DIR}"/deploy.sh 2>/dev/null || true
    chmod +x "${SCRIPT_DIR}"/rollback.sh 2>/dev/null || true
    
    # Start services after rollback
    start_services
    
    log "SUCCESS" "Rollback extraction completed"
}

# Validate rollback
validate_rollback() {
    local backup_name="$1"
    
    log "INFO" "Validating rollback..."
    
    # Check template files exist
    local template_files=("${SCRIPT_DIR}"/*.j2)
    if [[ ${#template_files[@]} -eq 0 ]]; then
        error_exit "No template files found after rollback"
    fi
    
    # Validate template syntax
    if command_exists python3; then
        python3 -c "
import os
import sys
from jinja2 import Environment, FileSystemLoader, TemplateSyntaxError

template_dir = '$SCRIPT_DIR'
env = Environment(loader=FileSystemLoader(template_dir))

try:
    for filename in os.listdir(template_dir):
        if filename.endswith('.j2'):
            template = env.get_template(filename)
            print(f'✅ Template syntax valid: {filename}')
    print('✅ All templates validated successfully')
except TemplateSyntaxError as e:
    print(f'❌ Template syntax error: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Validation error: {e}')
    sys.exit(1)
" || {
            error_exit "Template validation failed after rollback"
        }
    fi
    
    # Test service health
    test_service_health
    
    log "SUCCESS" "Rollback validation completed"
}

# Record rollback history
record_rollback_history() {
    local backup_name="$1"
    
    log "INFO" "Recording rollback history..."
    
    mkdir -p "$(dirname "$ROLLBACK_HISTORY")"
    
    local rollback_entry=$(cat << EOF
{
    "rollback_id": "$ROLLBACK_ID",
    "timestamp": "$(date -Iseconds)",
    "environment": "$ENVIRONMENT",
    "backup_name": "$backup_name",
    "user": "$(whoami)",
    "hostname": "$(hostname)",
    "reason": "Manual rollback",
    "duration_seconds": $(($(date +%s) - START_TIME))
}
EOF
)
    
    # Append to rollback history
    if [[ -f "$ROLLBACK_HISTORY" ]]; then
        local temp_file=$(mktemp)
        jq ". + [$rollback_entry]" "$ROLLBACK_HISTORY" > "$temp_file" && mv "$temp_file" "$ROLLBACK_HISTORY"
    else
        echo "[$rollback_entry]" > "$ROLLBACK_HISTORY"
    fi
    
    log "SUCCESS" "Rollback history recorded"
}

# ==================================================================================
# SERVICE MANAGEMENT
# ==================================================================================

# Stop services
stop_services() {
    log "INFO" "Stopping services for rollback..."
    
    case "$ENVIRONMENT" in
        "production"|"staging")
            # Kubernetes
            if command_exists kubectl; then
                kubectl scale deployment template-manager --replicas=0 -n "$ENVIRONMENT" || {
                    log "WARN" "Failed to scale down template-manager deployment"
                }
            fi
            ;;
        "development")
            # Local services
            if systemctl is-active --quiet template-manager; then
                systemctl stop template-manager || {
                    log "WARN" "Failed to stop template-manager service"
                }
            fi
            ;;
    esac
    
    # Wait for services to stop
    sleep 5
    
    log "SUCCESS" "Services stopped"
}

# Start services
start_services() {
    log "INFO" "Starting services after rollback..."
    
    case "$ENVIRONMENT" in
        "production"|"staging")
            # Kubernetes
            if command_exists kubectl; then
                kubectl scale deployment template-manager --replicas=2 -n "$ENVIRONMENT" || {
                    log "WARN" "Failed to scale up template-manager deployment"
                }
                
                # Wait for rollout
                kubectl rollout status deployment/template-manager -n "$ENVIRONMENT" --timeout=300s || {
                    log "WARN" "Deployment rollout timeout"
                }
            fi
            ;;
        "development")
            # Local services
            systemctl start template-manager || {
                log "WARN" "Failed to start template-manager service"
            }
            ;;
    esac
    
    log "SUCCESS" "Services started"
}

# Test service health
test_service_health() {
    log "INFO" "Testing service health..."
    
    local health_url="http://localhost:8080/health"
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if command_exists curl; then
            local response=$(curl -s -o /dev/null -w "%{http_code}" "$health_url" 2>/dev/null || echo "000")
            
            if [[ "$response" == "200" ]]; then
                log "SUCCESS" "Service health check passed (attempt $attempt)"
                return 0
            fi
        fi
        
        log "INFO" "Health check attempt $attempt/$max_attempts failed, waiting..."
        sleep 10
        ((attempt++))
    done
    
    log "WARN" "Service health check failed after $max_attempts attempts"
    return 1
}

# ==================================================================================
# MAIN FUNCTIONS
# ==================================================================================

# Show usage information
show_usage() {
    cat << EOF
SPOTIFY AI AGENT - Template Rollback Script v2.0.0

USAGE:
    $0 [OPTIONS] [BACKUP_NAME]

OPTIONS:
    -e, --environment ENV    Target environment (development|staging|production)
                            Default: $DEFAULT_ENVIRONMENT
    
    -d, --dry-run           Perform a dry run without making changes
                            Default: $DEFAULT_DRY_RUN
    
    -f, --force             Force rollback even if environment mismatch
                            Default: $DEFAULT_FORCE
    
    -l, --list              List available backups
                            Default: $DEFAULT_LIST_BACKUPS
    
    -h, --help              Show this help message

EXAMPLES:
    # List available backups
    $0 --list
    
    # Rollback to latest backup
    $0
    
    # Rollback to specific backup
    $0 templates_backup_20250718_120000
    
    # Dry run rollback
    $0 --dry-run templates_backup_20250718_120000
    
    # Force rollback despite environment mismatch
    $0 --force --environment production templates_backup_20250718_120000

ENVIRONMENT VARIABLES:
    BACKUP_RETENTION_DAYS   Number of days to keep backups
    LOG_LEVEL              Logging level (DEBUG|INFO|WARN|ERROR)
    
For more information, visit: https://docs.spotify-ai-agent.com/templates/rollback
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
            -f|--force)
                FORCE="true"
                shift
                ;;
            -l|--list)
                LIST_BACKUPS="true"
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            -*)
                error_exit "Unknown option: $1. Use --help for usage information."
                ;;
            *)
                if [[ -z "$ROLLBACK_TARGET" ]]; then
                    ROLLBACK_TARGET="$1"
                else
                    error_exit "Multiple backup names specified. Use --help for usage information."
                fi
                shift
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
    
    # Set default rollback target if not specified
    if [[ -z "$ROLLBACK_TARGET" && "$LIST_BACKUPS" != "true" ]]; then
        if [[ -f "${BACKUP_DIR}/.latest_backup" ]]; then
            ROLLBACK_TARGET=$(cat "${BACKUP_DIR}/.latest_backup")
        else
            error_exit "No backup specified and no latest backup found. Use --list to see available backups."
        fi
    fi
}

# Main execution function
main() {
    local exit_code=0
    
    # Generate rollback ID
    ROLLBACK_ID=$(generate_rollback_id)
    
    log "INFO" "Starting template rollback process"
    log "INFO" "Rollback ID: $ROLLBACK_ID"
    log "INFO" "Environment: $ENVIRONMENT"
    log "INFO" "Dry Run: $DRY_RUN"
    log "INFO" "Force: $FORCE"
    log "INFO" "Target: $ROLLBACK_TARGET"
    
    trap 'log "ERROR" "Rollback interrupted"; exit 1' INT TERM
    
    # Execute based on options
    if [[ "$LIST_BACKUPS" == "true" ]]; then
        list_available_backups
    else
        # Perform rollback
        perform_rollback "$ROLLBACK_TARGET" || {
            exit_code=1
        }
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
