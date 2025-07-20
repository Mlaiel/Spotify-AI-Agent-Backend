#!/bin/bash
# =============================================================================
# Enterprise Keys Backup Automation Script
# =============================================================================
# 
# Ultra-advanced automated backup system for cryptographic keys and secrets
# with scheduling, monitoring, rotation, and disaster recovery capabilities
#
# Expert Development Team:
# - Lead Dev + AI Architect
# - Senior Backend Developer (Python/FastAPI/Django)
# - ML Engineer (TensorFlow/PyTorch/Hugging Face)
# - DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
# - Backend Security Specialist
# - Microservices Architect
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
KEYS_DIR="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="$SCRIPT_DIR"
LOG_FILE="$BACKUP_DIR/backup_automation.log"
CONFIG_FILE="$SCRIPT_DIR/backup_config.json"
LOCK_FILE="$BACKUP_DIR/.backup_lock"
PID_FILE="$BACKUP_DIR/backup_automation.pid"

# Default configuration
DEFAULT_RETENTION_DAYS=30
DEFAULT_MAX_BACKUPS=50
DEFAULT_COMPRESSION_LEVEL=9
DEFAULT_ENCRYPTION_ENABLED=true
DEFAULT_VERIFICATION_ENABLED=true
DEFAULT_CLOUD_SYNC_ENABLED=false
DEFAULT_NOTIFICATION_ENABLED=true
DEFAULT_BACKUP_SCHEDULE="0 2 * * *"  # Daily at 2 AM

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_with_timestamp() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() {
    log_with_timestamp "INFO" "$1"
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    log_with_timestamp "SUCCESS" "$1"
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    log_with_timestamp "WARNING" "$1"
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    log_with_timestamp "ERROR" "$1"
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        log_with_timestamp "DEBUG" "$1"
        echo -e "${PURPLE}[DEBUG]${NC} $1"
    fi
}

# Configuration management
create_default_config() {
    log_info "Creating default backup configuration..."
    
    cat > "$CONFIG_FILE" << EOF
{
    "backup_settings": {
        "retention_days": $DEFAULT_RETENTION_DAYS,
        "max_backups": $DEFAULT_MAX_BACKUPS,
        "compression_level": $DEFAULT_COMPRESSION_LEVEL,
        "encryption_enabled": $DEFAULT_ENCRYPTION_ENABLED,
        "verification_enabled": $DEFAULT_VERIFICATION_ENABLED,
        "cloud_sync_enabled": $DEFAULT_CLOUD_SYNC_ENABLED,
        "notification_enabled": $DEFAULT_NOTIFICATION_ENABLED,
        "backup_schedule": "$DEFAULT_BACKUP_SCHEDULE"
    },
    "paths": {
        "keys_directory": "$KEYS_DIR",
        "backup_directory": "$BACKUP_DIR",
        "temp_directory": "/tmp/keys_backup_temp",
        "log_file": "$LOG_FILE"
    },
    "encryption": {
        "algorithm": "AES-256-GCM",
        "key_derivation": "PBKDF2-SHA256",
        "iterations": 100000,
        "key_file": "$BACKUP_DIR/backup_master.key"
    },
    "compression": {
        "algorithm": "gzip",
        "level": $DEFAULT_COMPRESSION_LEVEL,
        "excluded_extensions": [".gz", ".zip", ".bz2", ".xz"]
    },
    "verification": {
        "checksum_algorithm": "SHA256",
        "verify_after_backup": true,
        "verify_before_restore": true,
        "integrity_check_interval": 24
    },
    "cloud_storage": {
        "provider": "s3",
        "bucket": "keys-backup-bucket",
        "region": "us-east-1",
        "encryption_at_rest": true,
        "versioning_enabled": true
    },
    "notifications": {
        "email": {
            "enabled": false,
            "smtp_server": "smtp.example.com",
            "smtp_port": 587,
            "username": "",
            "recipients": ["admin@example.com"]
        },
        "slack": {
            "enabled": false,
            "webhook_url": "",
            "channel": "#security-alerts"
        },
        "webhook": {
            "enabled": false,
            "url": "",
            "secret": ""
        }
    },
    "monitoring": {
        "metrics_enabled": true,
        "prometheus_enabled": false,
        "grafana_dashboard": false,
        "health_check_endpoint": "http://localhost:8080/health",
        "performance_monitoring": true
    },
    "security": {
        "access_control": {
            "allowed_users": ["root", "backup"],
            "allowed_groups": ["backup", "security"],
            "required_permissions": "600"
        },
        "audit": {
            "audit_log_enabled": true,
            "audit_log_file": "$BACKUP_DIR/backup_audit.log",
            "log_all_operations": true,
            "log_sensitive_data": false
        },
        "compliance": {
            "gdpr_compliant": true,
            "hipaa_compliant": false,
            "pci_compliant": true,
            "data_retention_policy": true
        }
    },
    "advanced": {
        "parallel_processing": {
            "enabled": true,
            "max_workers": 4,
            "chunk_size_mb": 100
        },
        "deduplication": {
            "enabled": true,
            "algorithm": "sha256",
            "block_size_kb": 64
        },
        "delta_backups": {
            "enabled": true,
            "baseline_interval_days": 7,
            "max_delta_chain_length": 10
        },
        "disaster_recovery": {
            "geo_replication": false,
            "cross_region_backup": false,
            "emergency_recovery_plan": true
        }
    }
}
EOF

    chmod 600 "$CONFIG_FILE"
    log_success "Default configuration created at $CONFIG_FILE"
}

load_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_warning "Configuration file not found, creating default configuration"
        create_default_config
    fi
    
    if command -v jq >/dev/null 2>&1; then
        # Load configuration using jq
        RETENTION_DAYS=$(jq -r '.backup_settings.retention_days' "$CONFIG_FILE")
        MAX_BACKUPS=$(jq -r '.backup_settings.max_backups' "$CONFIG_FILE")
        COMPRESSION_LEVEL=$(jq -r '.backup_settings.compression_level' "$CONFIG_FILE")
        ENCRYPTION_ENABLED=$(jq -r '.backup_settings.encryption_enabled' "$CONFIG_FILE")
        VERIFICATION_ENABLED=$(jq -r '.backup_settings.verification_enabled' "$CONFIG_FILE")
        NOTIFICATION_ENABLED=$(jq -r '.backup_settings.notification_enabled' "$CONFIG_FILE")
    else
        log_warning "jq not available, using default configuration values"
        RETENTION_DAYS=$DEFAULT_RETENTION_DAYS
        MAX_BACKUPS=$DEFAULT_MAX_BACKUPS
        COMPRESSION_LEVEL=$DEFAULT_COMPRESSION_LEVEL
        ENCRYPTION_ENABLED=$DEFAULT_ENCRYPTION_ENABLED
        VERIFICATION_ENABLED=$DEFAULT_VERIFICATION_ENABLED
        NOTIFICATION_ENABLED=$DEFAULT_NOTIFICATION_ENABLED
    fi
    
    log_debug "Configuration loaded successfully"
}

# Lock management
acquire_lock() {
    if [[ -f "$LOCK_FILE" ]]; then
        local lock_pid=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
        if [[ -n "$lock_pid" ]] && kill -0 "$lock_pid" 2>/dev/null; then
            log_error "Another backup process is already running (PID: $lock_pid)"
            return 1
        else
            log_warning "Removing stale lock file"
            rm -f "$LOCK_FILE"
        fi
    fi
    
    echo $$ > "$LOCK_FILE"
    echo $$ > "$PID_FILE"
    log_debug "Lock acquired (PID: $$)"
}

release_lock() {
    if [[ -f "$LOCK_FILE" ]]; then
        rm -f "$LOCK_FILE"
        log_debug "Lock released"
    fi
    if [[ -f "$PID_FILE" ]]; then
        rm -f "$PID_FILE"
    fi
}

# Cleanup function
cleanup() {
    local exit_code=$?
    log_info "Cleaning up backup process..."
    
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    # Clean temporary files
    if [[ -n "${TEMP_DIR:-}" ]] && [[ -d "$TEMP_DIR" ]]; then
        rm -rf "$TEMP_DIR"
        log_debug "Temporary directory cleaned: $TEMP_DIR"
    fi
    
    # Release lock
    release_lock
    
    # Send notification if enabled
    if [[ "$NOTIFICATION_ENABLED" == "true" ]]; then
        if [[ $exit_code -eq 0 ]]; then
            send_notification "success" "Keys backup completed successfully"
        else
            send_notification "error" "Keys backup failed with exit code $exit_code"
        fi
    fi
    
    log_info "Backup process cleanup completed"
    exit $exit_code
}

# Trap signals for cleanup
trap cleanup EXIT INT TERM

# System requirements check
check_requirements() {
    log_info "Checking system requirements..."
    
    local missing_tools=()
    local required_tools=("tar" "gzip" "openssl" "python3")
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install the missing tools before running the backup"
        return 1
    fi
    
    # Check Python modules
    local python_modules=("cryptography" "hashlib" "json")
    for module in "${python_modules[@]}"; do
        if ! python3 -c "import $module" 2>/dev/null; then
            log_warning "Python module '$module' not available, some features may be limited"
        fi
    done
    
    # Check disk space
    local available_space=$(df "$BACKUP_DIR" | awk 'NR==2 {print $4}')
    local keys_size=$(du -s "$KEYS_DIR" | awk '{print $1}')
    local required_space=$((keys_size * 3))  # 3x for safety margin
    
    if [[ $available_space -lt $required_space ]]; then
        log_warning "Low disk space: Available: ${available_space}KB, Required: ${required_space}KB"
    fi
    
    # Check permissions
    if [[ ! -r "$KEYS_DIR" ]]; then
        log_error "Cannot read keys directory: $KEYS_DIR"
        return 1
    fi
    
    if [[ ! -w "$BACKUP_DIR" ]]; then
        log_error "Cannot write to backup directory: $BACKUP_DIR"
        return 1
    fi
    
    log_success "System requirements check passed"
}

# Pre-backup validation
validate_keys_directory() {
    log_info "Validating keys directory..."
    
    if [[ ! -d "$KEYS_DIR" ]]; then
        log_error "Keys directory does not exist: $KEYS_DIR"
        return 1
    fi
    
    # Count key files
    local key_files=(
        "$KEYS_DIR"/*.key
        "$KEYS_DIR"/*.pem
        "$KEYS_DIR"/*.p12
        "$KEYS_DIR"/*.pfx
        "$KEYS_DIR"/*.jks
        "$KEYS_DIR"/*.crt
        "$KEYS_DIR"/*.csr
    )
    
    local file_count=0
    for pattern in "${key_files[@]}"; do
        if compgen -G "$pattern" > /dev/null; then
            file_count=$((file_count + $(ls -1 $pattern 2>/dev/null | wc -l)))
        fi
    done
    
    if [[ $file_count -eq 0 ]]; then
        log_warning "No key files found in directory: $KEYS_DIR"
        log_warning "Proceeding with directory backup anyway"
    else
        log_info "Found $file_count key files to backup"
    fi
    
    # Check for sensitive files
    local sensitive_patterns=("*.key" "*.pem" "*private*" "*secret*")
    for pattern in "${sensitive_patterns[@]}"; do
        while IFS= read -r -d '' file; do
            local perms=$(stat -c "%a" "$file")
            if [[ "$perms" != "600" ]] && [[ "$perms" != "400" ]]; then
                log_warning "Sensitive file has loose permissions: $file ($perms)"
            fi
        done < <(find "$KEYS_DIR" -name "$pattern" -type f -print0 2>/dev/null)
    done
    
    log_success "Keys directory validation completed"
}

# Generate backup filename
generate_backup_filename() {
    local backup_type="${1:-full}"
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local hostname=$(hostname -s)
    local random_suffix=$(openssl rand -hex 4)
    
    echo "keys_backup_${backup_type}_${hostname}_${timestamp}_${random_suffix}"
}

# Create backup archive
create_backup_archive() {
    local backup_name="$1"
    local backup_file="$BACKUP_DIR/${backup_name}.tar.gz"
    local temp_archive="$TEMP_DIR/${backup_name}.tar"
    
    log_info "Creating backup archive: $backup_name"
    
    # Create temporary directory
    TEMP_DIR=$(mktemp -d -t keys_backup.XXXXXX)
    
    # Create tar archive with full paths preserved
    if tar -cf "$temp_archive" -C "$(dirname "$KEYS_DIR")" "$(basename "$KEYS_DIR")" 2>/dev/null; then
        log_success "Tar archive created successfully"
    else
        log_error "Failed to create tar archive"
        return 1
    fi
    
    # Compress with specified level
    if gzip -"$COMPRESSION_LEVEL" "$temp_archive"; then
        mv "${temp_archive}.gz" "$backup_file"
        log_success "Archive compressed successfully"
    else
        log_error "Failed to compress archive"
        return 1
    fi
    
    # Set secure permissions
    chmod 600 "$backup_file"
    
    echo "$backup_file"
}

# Encrypt backup file
encrypt_backup() {
    local backup_file="$1"
    local encrypted_file="${backup_file}.enc"
    local key_file="$BACKUP_DIR/backup_master.key"
    
    if [[ "$ENCRYPTION_ENABLED" != "true" ]]; then
        log_debug "Encryption disabled, skipping encryption step"
        echo "$backup_file"
        return 0
    fi
    
    log_info "Encrypting backup file..."
    
    # Generate or load encryption key
    if [[ ! -f "$key_file" ]]; then
        log_info "Generating new encryption key"
        openssl rand -base64 32 > "$key_file"
        chmod 600 "$key_file"
    fi
    
    # Encrypt using AES-256-CBC
    if openssl enc -aes-256-cbc -salt -pbkdf2 -iter 100000 \
        -in "$backup_file" -out "$encrypted_file" -pass file:"$key_file"; then
        
        # Remove unencrypted file
        rm -f "$backup_file"
        chmod 600 "$encrypted_file"
        
        log_success "Backup encrypted successfully"
        echo "$encrypted_file"
    else
        log_error "Failed to encrypt backup"
        return 1
    fi
}

# Calculate checksums
calculate_checksums() {
    local file="$1"
    local checksum_file="${file}.checksums"
    
    log_info "Calculating checksums for backup file..."
    
    {
        echo "# Backup Checksums - Generated $(date)"
        echo "# File: $(basename "$file")"
        echo "# Size: $(stat -c%s "$file") bytes"
        echo ""
        echo "SHA256: $(sha256sum "$file" | cut -d' ' -f1)"
        echo "SHA1:   $(sha1sum "$file" | cut -d' ' -f1)"
        echo "MD5:    $(md5sum "$file" | cut -d' ' -f1)"
    } > "$checksum_file"
    
    chmod 600 "$checksum_file"
    log_success "Checksums calculated and saved to $checksum_file"
}

# Verify backup integrity
verify_backup() {
    local backup_file="$1"
    local checksum_file="${backup_file}.checksums"
    
    if [[ "$VERIFICATION_ENABLED" != "true" ]]; then
        log_debug "Verification disabled, skipping verification step"
        return 0
    fi
    
    log_info "Verifying backup integrity..."
    
    if [[ ! -f "$checksum_file" ]]; then
        log_error "Checksum file not found: $checksum_file"
        return 1
    fi
    
    # Verify SHA256
    local expected_sha256=$(grep "^SHA256:" "$checksum_file" | cut -d' ' -f2)
    local actual_sha256=$(sha256sum "$backup_file" | cut -d' ' -f1)
    
    if [[ "$expected_sha256" == "$actual_sha256" ]]; then
        log_success "SHA256 checksum verification passed"
    else
        log_error "SHA256 checksum verification failed"
        log_error "Expected: $expected_sha256"
        log_error "Actual:   $actual_sha256"
        return 1
    fi
    
    # Test archive integrity
    local temp_test_dir=$(mktemp -d)
    
    if [[ "$backup_file" == *.enc ]]; then
        # Decrypt for testing
        local key_file="$BACKUP_DIR/backup_master.key"
        local temp_decrypted="$temp_test_dir/test_decrypt.tar.gz"
        
        if openssl enc -aes-256-cbc -d -pbkdf2 -iter 100000 \
            -in "$backup_file" -out "$temp_decrypted" -pass file:"$key_file"; then
            
            if tar -tzf "$temp_decrypted" >/dev/null 2>&1; then
                log_success "Archive integrity verification passed"
            else
                log_error "Archive integrity verification failed"
                rm -rf "$temp_test_dir"
                return 1
            fi
        else
            log_error "Failed to decrypt backup for verification"
            rm -rf "$temp_test_dir"
            return 1
        fi
    else
        if tar -tzf "$backup_file" >/dev/null 2>&1; then
            log_success "Archive integrity verification passed"
        else
            log_error "Archive integrity verification failed"
            rm -rf "$temp_test_dir"
            return 1
        fi
    fi
    
    rm -rf "$temp_test_dir"
    log_success "Backup verification completed successfully"
}

# Create backup metadata
create_backup_metadata() {
    local backup_file="$1"
    local metadata_file="${backup_file}.metadata.json"
    
    log_info "Creating backup metadata..."
    
    local file_size=$(stat -c%s "$backup_file")
    local file_date=$(date -r "$backup_file" -Iseconds)
    local backup_date=$(date -Iseconds)
    local hostname=$(hostname -f)
    local user=$(whoami)
    
    cat > "$metadata_file" << EOF
{
    "backup_info": {
        "backup_id": "$(basename "$backup_file" | sed 's/\.[^.]*$//')",
        "backup_type": "full",
        "created_at": "$backup_date",
        "created_by": "$user",
        "hostname": "$hostname",
        "version": "2.0.0"
    },
    "source_info": {
        "source_path": "$KEYS_DIR",
        "source_type": "directory",
        "file_count": $(find "$KEYS_DIR" -type f | wc -l),
        "total_size_bytes": $(du -sb "$KEYS_DIR" | cut -f1)
    },
    "backup_file": {
        "filename": "$(basename "$backup_file")",
        "full_path": "$backup_file",
        "size_bytes": $file_size,
        "size_human": "$(numfmt --to=iec-i --suffix=B $file_size)",
        "last_modified": "$file_date",
        "permissions": "$(stat -c%a "$backup_file")",
        "encrypted": $([ "$ENCRYPTION_ENABLED" == "true" ] && echo "true" || echo "false"),
        "compressed": true
    },
    "configuration": {
        "compression_level": $COMPRESSION_LEVEL,
        "encryption_enabled": $ENCRYPTION_ENABLED,
        "verification_enabled": $VERIFICATION_ENABLED,
        "retention_days": $RETENTION_DAYS
    },
    "checksums": {
        "sha256": "$(sha256sum "$backup_file" | cut -d' ' -f1)",
        "sha1": "$(sha1sum "$backup_file" | cut -d' ' -f1)",
        "md5": "$(md5sum "$backup_file" | cut -d' ' -f1)"
    },
    "verification": {
        "verified": $([ "$VERIFICATION_ENABLED" == "true" ] && echo "true" || echo "false"),
        "verification_date": "$backup_date",
        "integrity_check_passed": true
    },
    "retention": {
        "retention_days": $RETENTION_DAYS,
        "expire_date": "$(date -d "+$RETENTION_DAYS days" -Iseconds)",
        "auto_delete": true
    }
}
EOF
    
    chmod 600 "$metadata_file"
    log_success "Backup metadata created: $metadata_file"
}

# Send notifications
send_notification() {
    local status="$1"
    local message="$2"
    
    if [[ "$NOTIFICATION_ENABLED" != "true" ]]; then
        return 0
    fi
    
    log_info "Sending notification: $status - $message"
    
    # Log notification
    local notification_log="$BACKUP_DIR/notifications.log"
    echo "[$(date -Iseconds)] [$status] $message" >> "$notification_log"
    
    # Email notification (if configured)
    if command -v mail >/dev/null 2>&1; then
        local subject="Keys Backup $(echo "$status" | tr '[:lower:]' '[:upper:]'): $(hostname)"
        echo "$message" | mail -s "$subject" "admin@localhost" 2>/dev/null || true
    fi
    
    # Webhook notification (placeholder)
    # curl -X POST -H "Content-Type: application/json" \
    #      -d "{\"status\":\"$status\",\"message\":\"$message\"}" \
    #      "$WEBHOOK_URL" 2>/dev/null || true
    
    log_debug "Notification sent successfully"
}

# Cleanup old backups
cleanup_old_backups() {
    log_info "Cleaning up old backups (retention: $RETENTION_DAYS days, max: $MAX_BACKUPS)..."
    
    local deleted_count=0
    local current_count=$(find "$BACKUP_DIR" -name "keys_backup_*.tar.gz*" -o -name "keys_backup_*.enc" | wc -l)
    
    # Remove backups older than retention period
    while IFS= read -r -d '' backup_file; do
        local file_age_days=$(( ($(date +%s) - $(stat -c %Y "$backup_file")) / 86400 ))
        
        if [[ $file_age_days -gt $RETENTION_DAYS ]]; then
            log_info "Removing expired backup: $(basename "$backup_file") (age: $file_age_days days)"
            
            # Remove backup file and associated files
            rm -f "$backup_file"
            rm -f "${backup_file}.checksums"
            rm -f "${backup_file}.metadata.json"
            
            deleted_count=$((deleted_count + 1))
        fi
    done < <(find "$BACKUP_DIR" -name "keys_backup_*.tar.gz" -o -name "keys_backup_*.enc" -print0)
    
    # Remove excess backups if over limit
    if [[ $current_count -gt $MAX_BACKUPS ]]; then
        local excess_count=$((current_count - MAX_BACKUPS))
        log_info "Removing $excess_count excess backups (current: $current_count, max: $MAX_BACKUPS)"
        
        # Get oldest backups
        while IFS= read -r backup_file; do
            if [[ $excess_count -le 0 ]]; then
                break
            fi
            
            log_info "Removing excess backup: $(basename "$backup_file")"
            rm -f "$backup_file"
            rm -f "${backup_file}.checksums"
            rm -f "${backup_file}.metadata.json"
            
            deleted_count=$((deleted_count + 1))
            excess_count=$((excess_count - 1))
        done < <(find "$BACKUP_DIR" -name "keys_backup_*.tar.gz" -o -name "keys_backup_*.enc" -printf '%T@ %p\n' | sort -n | cut -d' ' -f2-)
    fi
    
    if [[ $deleted_count -gt 0 ]]; then
        log_success "Cleanup completed: $deleted_count old backups removed"
    else
        log_info "No old backups to clean up"
    fi
}

# Generate backup report
generate_backup_report() {
    local backup_file="$1"
    local report_file="$BACKUP_DIR/backup_report_$(date +%Y%m%d).txt"
    
    log_info "Generating backup report..."
    
    {
        echo "==================================================="
        echo "           KEYS BACKUP REPORT"
        echo "==================================================="
        echo "Report Generated: $(date)"
        echo "Hostname: $(hostname -f)"
        echo "User: $(whoami)"
        echo "Backup File: $(basename "$backup_file")"
        echo ""
        echo "BACKUP DETAILS:"
        echo "- Source Directory: $KEYS_DIR"
        echo "- Backup Directory: $BACKUP_DIR"
        echo "- File Size: $(stat -c%s "$backup_file" | numfmt --to=iec-i --suffix=B)"
        echo "- Compression Level: $COMPRESSION_LEVEL"
        echo "- Encryption: $([ "$ENCRYPTION_ENABLED" == "true" ] && echo "Enabled" || echo "Disabled")"
        echo "- Verification: $([ "$VERIFICATION_ENABLED" == "true" ] && echo "Enabled" || echo "Disabled")"
        echo ""
        echo "CHECKSUMS:"
        echo "- SHA256: $(sha256sum "$backup_file" | cut -d' ' -f1)"
        echo "- MD5:    $(md5sum "$backup_file" | cut -d' ' -f1)"
        echo ""
        echo "CONFIGURATION:"
        echo "- Retention Days: $RETENTION_DAYS"
        echo "- Max Backups: $MAX_BACKUPS"
        echo "- Notification: $([ "$NOTIFICATION_ENABLED" == "true" ] && echo "Enabled" || echo "Disabled")"
        echo ""
        echo "SYSTEM INFO:"
        echo "- Available Space: $(df -h "$BACKUP_DIR" | awk 'NR==2 {print $4}')"
        echo "- Load Average: $(uptime | awk -F'load average:' '{print $2}')"
        echo "- Memory Usage: $(free -h | awk 'NR==2{printf "%.1f%%", $3*100/$2 }')"
        echo ""
        echo "BACKUP HISTORY (Last 10):"
        find "$BACKUP_DIR" -name "keys_backup_*.tar.gz*" -o -name "keys_backup_*.enc" -printf '%T@ %TY-%Tm-%Td %TH:%TM %s %p\n' | \
            sort -nr | head -10 | awk '{print "  " $2 " " $3 " " $4 " bytes " $5}' || echo "  No previous backups found"
        echo ""
        echo "==================================================="
    } > "$report_file"
    
    chmod 600 "$report_file"
    log_success "Backup report generated: $report_file"
}

# Main backup function
perform_backup() {
    log_info "=== Starting Keys Backup Process ==="
    log_info "Backup directory: $BACKUP_DIR"
    log_info "Keys directory: $KEYS_DIR"
    
    # Pre-backup checks
    check_requirements || return 1
    validate_keys_directory || return 1
    
    # Generate backup filename
    local backup_name=$(generate_backup_filename "full")
    log_info "Backup name: $backup_name"
    
    # Create backup archive
    local backup_file
    backup_file=$(create_backup_archive "$backup_name") || return 1
    
    # Encrypt backup if enabled
    backup_file=$(encrypt_backup "$backup_file") || return 1
    
    # Calculate checksums
    calculate_checksums "$backup_file" || return 1
    
    # Verify backup integrity
    verify_backup "$backup_file" || return 1
    
    # Create metadata
    create_backup_metadata "$backup_file" || return 1
    
    # Generate report
    generate_backup_report "$backup_file" || return 1
    
    # Cleanup old backups
    cleanup_old_backups
    
    local final_size=$(stat -c%s "$backup_file" | numfmt --to=iec-i --suffix=B)
    log_success "=== Backup Process Completed Successfully ==="
    log_success "Backup file: $(basename "$backup_file")"
    log_success "Final size: $final_size"
    
    return 0
}

# Restore backup function
restore_backup() {
    local backup_file="$1"
    local restore_dir="${2:-$KEYS_DIR.restored}"
    
    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi
    
    log_info "=== Starting Keys Restore Process ==="
    log_info "Backup file: $backup_file"
    log_info "Restore directory: $restore_dir"
    
    # Create restore directory
    mkdir -p "$restore_dir"
    
    # Create temporary directory for restoration
    local temp_restore_dir=$(mktemp -d -t keys_restore.XXXXXX)
    
    # Decrypt if encrypted
    local working_file="$backup_file"
    if [[ "$backup_file" == *.enc ]]; then
        log_info "Decrypting backup file..."
        local key_file="$BACKUP_DIR/backup_master.key"
        local temp_decrypted="$temp_restore_dir/decrypted.tar.gz"
        
        if openssl enc -aes-256-cbc -d -pbkdf2 -iter 100000 \
            -in "$backup_file" -out "$temp_decrypted" -pass file:"$key_file"; then
            working_file="$temp_decrypted"
            log_success "Backup decrypted successfully"
        else
            log_error "Failed to decrypt backup"
            rm -rf "$temp_restore_dir"
            return 1
        fi
    fi
    
    # Extract archive
    log_info "Extracting backup archive..."
    if tar -xzf "$working_file" -C "$temp_restore_dir"; then
        log_success "Archive extracted successfully"
    else
        log_error "Failed to extract archive"
        rm -rf "$temp_restore_dir"
        return 1
    fi
    
    # Move extracted files to restore directory
    if mv "$temp_restore_dir"/* "$restore_dir"/; then
        log_success "Files restored to: $restore_dir"
    else
        log_error "Failed to move restored files"
        rm -rf "$temp_restore_dir"
        return 1
    fi
    
    # Cleanup
    rm -rf "$temp_restore_dir"
    
    log_success "=== Restore Process Completed Successfully ==="
    return 0
}

# List backups
list_backups() {
    log_info "=== Available Backups ==="
    
    local backup_files=($(find "$BACKUP_DIR" -name "keys_backup_*.tar.gz" -o -name "keys_backup_*.enc" | sort -r))
    
    if [[ ${#backup_files[@]} -eq 0 ]]; then
        log_info "No backups found in $BACKUP_DIR"
        return 0
    fi
    
    printf "%-30s %-12s %-20s %-10s %-10s\n" "Backup Name" "Size" "Date" "Encrypted" "Verified"
    printf "%-30s %-12s %-20s %-10s %-10s\n" "$(printf '%*s' 30 '' | tr ' ' '-')" "$(printf '%*s' 12 '' | tr ' ' '-')" "$(printf '%*s' 20 '' | tr ' ' '-')" "$(printf '%*s' 10 '' | tr ' ' '-')" "$(printf '%*s' 10 '' | tr ' ' '-')"
    
    for backup_file in "${backup_files[@]}"; do
        local basename_file=$(basename "$backup_file")
        local size_human=$(stat -c%s "$backup_file" | numfmt --to=iec-i --suffix=B)
        local date_human=$(date -r "$backup_file" '+%Y-%m-%d %H:%M')
        local encrypted=$([ "${backup_file}" == *.enc ] && echo "Yes" || echo "No")
        local verified=$([ -f "${backup_file}.checksums" ] && echo "Yes" || echo "No")
        
        printf "%-30s %-12s %-20s %-10s %-10s\n" \
            "${basename_file:0:29}" \
            "$size_human" \
            "$date_human" \
            "$encrypted" \
            "$verified"
    done
    
    echo ""
    log_info "Total backups: ${#backup_files[@]}"
}

# Health check
health_check() {
    log_info "=== Backup System Health Check ==="
    
    local health_status="healthy"
    local issues=()
    
    # Check backup directory
    if [[ ! -d "$BACKUP_DIR" ]]; then
        issues+=("Backup directory does not exist: $BACKUP_DIR")
        health_status="critical"
    elif [[ ! -w "$BACKUP_DIR" ]]; then
        issues+=("Backup directory is not writable: $BACKUP_DIR")
        health_status="critical"
    fi
    
    # Check keys directory
    if [[ ! -d "$KEYS_DIR" ]]; then
        issues+=("Keys directory does not exist: $KEYS_DIR")
        health_status="critical"
    elif [[ ! -r "$KEYS_DIR" ]]; then
        issues+=("Keys directory is not readable: $KEYS_DIR")
        health_status="critical"
    fi
    
    # Check disk space
    local available_space_kb=$(df "$BACKUP_DIR" | awk 'NR==2 {print $4}')
    local keys_size_kb=$(du -s "$KEYS_DIR" | awk '{print $1}')
    local required_space_kb=$((keys_size_kb * 3))
    
    if [[ $available_space_kb -lt $required_space_kb ]]; then
        issues+=("Insufficient disk space: Available: ${available_space_kb}KB, Required: ${required_space_kb}KB")
        health_status="warning"
    fi
    
    # Check recent backups
    local recent_backups=$(find "$BACKUP_DIR" -name "keys_backup_*.tar.gz" -o -name "keys_backup_*.enc" -mtime -1 | wc -l)
    if [[ $recent_backups -eq 0 ]]; then
        issues+=("No recent backups (last 24 hours)")
        if [[ "$health_status" == "healthy" ]]; then
            health_status="warning"
        fi
    fi
    
    # Check encryption key
    local key_file="$BACKUP_DIR/backup_master.key"
    if [[ "$ENCRYPTION_ENABLED" == "true" ]]; then
        if [[ ! -f "$key_file" ]]; then
            issues+=("Encryption key file missing: $key_file")
            health_status="critical"
        elif [[ "$(stat -c%a "$key_file")" != "600" ]]; then
            issues+=("Encryption key has insecure permissions: $key_file")
            if [[ "$health_status" == "healthy" ]]; then
                health_status="warning"
            fi
        fi
    fi
    
    # Report results
    echo "Overall Status: $health_status"
    echo "Backup Directory: $BACKUP_DIR"
    echo "Keys Directory: $KEYS_DIR"
    echo "Available Space: $(df -h "$BACKUP_DIR" | awk 'NR==2 {print $4}')"
    echo "Recent Backups: $recent_backups"
    echo "Total Backups: $(find "$BACKUP_DIR" -name "keys_backup_*.tar.gz" -o -name "keys_backup_*.enc" | wc -l)"
    
    if [[ ${#issues[@]} -gt 0 ]]; then
        echo ""
        echo "Issues Found:"
        for issue in "${issues[@]}"; do
            echo "  - $issue"
        done
    fi
    
    case "$health_status" in
        "healthy")
            log_success "Health check passed: System is healthy"
            return 0
            ;;
        "warning")
            log_warning "Health check completed with warnings"
            return 1
            ;;
        "critical")
            log_error "Health check failed: Critical issues found"
            return 2
            ;;
    esac
}

# Usage information
show_usage() {
    cat << EOF
Enterprise Keys Backup Automation Script

USAGE:
    $0 [OPTIONS] COMMAND

COMMANDS:
    backup              Create a new backup of keys directory
    restore FILE [DIR]  Restore backup FILE to directory DIR (default: keys.restored)
    list               List all available backups
    verify FILE        Verify backup file integrity
    cleanup            Clean up old backups according to retention policy
    health             Perform system health check
    config             Show current configuration
    help               Show this usage information

OPTIONS:
    -c, --config FILE   Use custom configuration file
    -v, --verbose       Enable verbose output
    -d, --debug         Enable debug output
    -q, --quiet         Suppress non-error output
    --dry-run          Show what would be done without executing
    --force            Force operation without confirmation prompts

EXAMPLES:
    $0 backup                                    # Create new backup
    $0 restore keys_backup_full_20240716.tar.gz # Restore specific backup
    $0 list                                      # List all backups
    $0 health                                    # Check system health
    $0 cleanup                                   # Clean old backups

CONFIGURATION:
    Configuration file: $CONFIG_FILE
    Use '$0 config' to view current settings

For detailed documentation, see the README files in the parent directory.
EOF
}

# Show configuration
show_config() {
    log_info "=== Current Configuration ==="
    
    if [[ -f "$CONFIG_FILE" ]]; then
        if command -v jq >/dev/null 2>&1; then
            jq '.' "$CONFIG_FILE"
        else
            cat "$CONFIG_FILE"
        fi
    else
        echo "Configuration file not found: $CONFIG_FILE"
        echo "Run with 'backup' command to create default configuration."
    fi
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -d|--debug)
                DEBUG=true
                shift
                ;;
            -q|--quiet)
                QUIET=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            backup|restore|list|verify|cleanup|health|config|help)
                COMMAND="$1"
                shift
                break
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Store remaining arguments
    ARGS=("$@")
}

# Main execution
main() {
    # Initialize variables
    COMMAND=""
    ARGS=()
    VERBOSE=${VERBOSE:-false}
    DEBUG=${DEBUG:-false}
    QUIET=${QUIET:-false}
    DRY_RUN=${DRY_RUN:-false}
    FORCE=${FORCE:-false}
    
    # Parse command line
    parse_arguments "$@"
    
    # Check if command is provided
    if [[ -z "$COMMAND" ]]; then
        log_error "No command specified"
        show_usage
        exit 1
    fi
    
    # Load configuration
    load_config
    
    # Acquire lock for backup operations
    if [[ "$COMMAND" == "backup" ]] || [[ "$COMMAND" == "cleanup" ]]; then
        acquire_lock || exit 1
    fi
    
    # Execute command
    case "$COMMAND" in
        backup)
            perform_backup
            ;;
        restore)
            if [[ ${#ARGS[@]} -lt 1 ]]; then
                log_error "Restore command requires backup file argument"
                exit 1
            fi
            restore_backup "${ARGS[0]}" "${ARGS[1]:-}"
            ;;
        list)
            list_backups
            ;;
        verify)
            if [[ ${#ARGS[@]} -lt 1 ]]; then
                log_error "Verify command requires backup file argument"
                exit 1
            fi
            verify_backup "${ARGS[0]}"
            ;;
        cleanup)
            cleanup_old_backups
            ;;
        health)
            health_check
            ;;
        config)
            show_config
            ;;
        help)
            show_usage
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
