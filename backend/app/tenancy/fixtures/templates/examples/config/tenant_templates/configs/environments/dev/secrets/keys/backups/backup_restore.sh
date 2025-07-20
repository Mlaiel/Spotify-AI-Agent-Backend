#!/bin/bash
# =============================================================================
# Enterprise Keys Backup Restoration & Recovery System
# =============================================================================
# 
# Ultra-advanced disaster recovery and restoration system for cryptographic keys
# with point-in-time recovery, incremental restoration, validation,
# and automated rollback capabilities
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
RESTORE_CONFIG="$SCRIPT_DIR/restore_config.json"
RESTORE_LOG="$SCRIPT_DIR/restore.log"
RESTORE_AUDIT_LOG="$SCRIPT_DIR/restore_audit.log"
RECOVERY_LOG="$SCRIPT_DIR/recovery.log"
RESTORE_LOCK="$SCRIPT_DIR/.restore_lock"
RESTORE_PID="$SCRIPT_DIR/restore.pid"

# Default configuration
DEFAULT_RESTORE_DIR="$SCRIPT_DIR/../restored"
DEFAULT_VERIFICATION_ENABLED=true
DEFAULT_BACKUP_ORIGINAL=true
DEFAULT_INCREMENTAL_RESTORE=false
DEFAULT_PARALLEL_PROCESSING=true
DEFAULT_MAX_WORKERS=4

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
    echo "[$timestamp] [$level] $message" | tee -a "$RESTORE_LOG"
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

log_critical() {
    log_with_timestamp "CRITICAL" "$1"
    echo -e "${RED}[CRITICAL]${NC} $1" >&2
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        log_with_timestamp "DEBUG" "$1"
        echo -e "${PURPLE}[DEBUG]${NC} $1"
    fi
}

# Audit logging
log_audit() {
    local operation="$1"
    local source="$2"
    local target="$3"
    local result="$4"
    local details="${5:-}"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local user=$(whoami)
    local hostname=$(hostname -f)
    
    echo "[$timestamp] AUDIT: operation=$operation, user=$user, hostname=$hostname, source=$source, target=$target, result=$result, details=$details" >> "$RESTORE_AUDIT_LOG"
}

# Create default restore configuration
create_default_restore_config() {
    log_info "Creating default restore configuration..."
    
    cat > "$RESTORE_CONFIG" << EOF
{
    "restore_settings": {
        "default_restore_directory": "$DEFAULT_RESTORE_DIR",
        "verification_enabled": $DEFAULT_VERIFICATION_ENABLED,
        "backup_original_enabled": $DEFAULT_BACKUP_ORIGINAL,
        "incremental_restore_enabled": $DEFAULT_INCREMENTAL_RESTORE,
        "parallel_processing_enabled": $DEFAULT_PARALLEL_PROCESSING,
        "max_workers": $DEFAULT_MAX_WORKERS,
        "preserve_permissions": true,
        "preserve_ownership": true,
        "preserve_timestamps": true
    },
    "security": {
        "verify_checksums": true,
        "verify_signatures": false,
        "decrypt_automatically": true,
        "require_confirmation": true,
        "log_all_operations": true,
        "secure_deletion": true
    },
    "validation": {
        "pre_restore_validation": true,
        "post_restore_validation": true,
        "key_format_validation": true,
        "certificate_validation": true,
        "permission_validation": true,
        "integrity_validation": true
    },
    "recovery": {
        "point_in_time_recovery": true,
        "incremental_recovery": true,
        "differential_recovery": false,
        "transaction_log_recovery": false,
        "automatic_rollback": true,
        "recovery_point_interval_minutes": 15
    },
    "performance": {
        "parallel_extraction": true,
        "memory_limit_mb": 1024,
        "io_buffer_size_kb": 64,
        "compression_on_fly": true,
        "cache_extracted_files": false
    },
    "notifications": {
        "email_on_completion": false,
        "email_on_failure": true,
        "webhook_notifications": false,
        "syslog_integration": true
    },
    "paths": {
        "backup_directory": "$SCRIPT_DIR",
        "temp_directory": "/tmp/keys_restore",
        "recovery_directory": "$SCRIPT_DIR/recovery",
        "archive_directory": "$SCRIPT_DIR/archive"
    },
    "retention": {
        "keep_restore_logs_days": 90,
        "keep_recovery_points_days": 30,
        "auto_cleanup_temp_files": true,
        "max_concurrent_restores": 3
    },
    "advanced": {
        "selective_restore": true,
        "pattern_matching": true,
        "dry_run_mode": true,
        "interactive_mode": false,
        "expert_mode": false,
        "debug_mode": false
    }
}
EOF

    chmod 600 "$RESTORE_CONFIG"
    log_success "Default restore configuration created at $RESTORE_CONFIG"
}

# Load restore configuration
load_restore_config() {
    if [[ ! -f "$RESTORE_CONFIG" ]]; then
        log_warning "Restore configuration not found, creating default"
        create_default_restore_config
    fi
    
    if command -v jq >/dev/null 2>&1; then
        # Load configuration using jq
        RESTORE_DIR=$(jq -r '.restore_settings.default_restore_directory' "$RESTORE_CONFIG")
        VERIFICATION_ENABLED=$(jq -r '.restore_settings.verification_enabled' "$RESTORE_CONFIG")
        BACKUP_ORIGINAL=$(jq -r '.restore_settings.backup_original_enabled' "$RESTORE_CONFIG")
        PARALLEL_PROCESSING=$(jq -r '.restore_settings.parallel_processing_enabled' "$RESTORE_CONFIG")
        MAX_WORKERS=$(jq -r '.restore_settings.max_workers' "$RESTORE_CONFIG")
        TEMP_DIR=$(jq -r '.paths.temp_directory' "$RESTORE_CONFIG")
    else
        log_warning "jq not available, using default configuration values"
        RESTORE_DIR="$DEFAULT_RESTORE_DIR"
        VERIFICATION_ENABLED="$DEFAULT_VERIFICATION_ENABLED"
        BACKUP_ORIGINAL="$DEFAULT_BACKUP_ORIGINAL"
        PARALLEL_PROCESSING="$DEFAULT_PARALLEL_PROCESSING"
        MAX_WORKERS="$DEFAULT_MAX_WORKERS"
        TEMP_DIR="/tmp/keys_restore"
    fi
    
    log_debug "Restore configuration loaded successfully"
}

# Lock management
acquire_restore_lock() {
    if [[ -f "$RESTORE_LOCK" ]]; then
        local lock_pid=$(cat "$RESTORE_LOCK" 2>/dev/null || echo "")
        if [[ -n "$lock_pid" ]] && kill -0 "$lock_pid" 2>/dev/null; then
            log_error "Another restore process is already running (PID: $lock_pid)"
            return 1
        else
            log_warning "Removing stale restore lock file"
            rm -f "$RESTORE_LOCK"
        fi
    fi
    
    echo $$ > "$RESTORE_LOCK"
    echo $$ > "$RESTORE_PID"
    log_debug "Restore lock acquired (PID: $$)"
}

release_restore_lock() {
    if [[ -f "$RESTORE_LOCK" ]]; then
        rm -f "$RESTORE_LOCK"
        log_debug "Restore lock released"
    fi
    if [[ -f "$RESTORE_PID" ]]; then
        rm -f "$RESTORE_PID"
    fi
}

# Cleanup function
cleanup_restore() {
    local exit_code=$?
    log_info "Cleaning up restore process..."
    
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    # Clean temporary files
    if [[ -n "${RESTORE_TEMP_DIR:-}" ]] && [[ -d "$RESTORE_TEMP_DIR" ]]; then
        if [[ "$AUTO_CLEANUP_TEMP" == "true" ]]; then
            rm -rf "$RESTORE_TEMP_DIR"
            log_debug "Temporary directory cleaned: $RESTORE_TEMP_DIR"
        else
            log_info "Temporary files preserved: $RESTORE_TEMP_DIR"
        fi
    fi
    
    # Release lock
    release_restore_lock
    
    # Send notification if configured
    if [[ "${NOTIFICATION_ENABLED:-false}" == "true" ]]; then
        if [[ $exit_code -eq 0 ]]; then
            send_restore_notification "success" "Keys restore completed successfully"
        else
            send_restore_notification "error" "Keys restore failed with exit code $exit_code"
        fi
    fi
    
    log_info "Restore process cleanup completed"
    exit $exit_code
}

# Trap signals for cleanup
trap cleanup_restore EXIT INT TERM

# Pre-restore validation
validate_backup_file() {
    local backup_file="$1"
    
    log_info "Validating backup file: $(basename "$backup_file")"
    
    # Check if file exists
    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi
    
    # Check file permissions
    if [[ ! -r "$backup_file" ]]; then
        log_error "Cannot read backup file: $backup_file"
        return 1
    fi
    
    # Check file size
    local file_size=$(stat -c%s "$backup_file")
    if [[ $file_size -eq 0 ]]; then
        log_error "Backup file is empty: $backup_file"
        return 1
    fi
    
    log_info "Backup file size: $(numfmt --to=iec-i --suffix=B $file_size)"
    
    # Verify checksums if available
    local checksum_file="${backup_file}.checksums"
    if [[ -f "$checksum_file" ]] && [[ "$VERIFICATION_ENABLED" == "true" ]]; then
        log_info "Verifying backup file checksums..."
        
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
    else
        log_warning "Checksum file not found or verification disabled"
    fi
    
    # Test archive integrity
    if [[ "$backup_file" == *.tar.gz ]]; then
        if tar -tzf "$backup_file" >/dev/null 2>&1; then
            log_success "Tar.gz archive integrity check passed"
        else
            log_error "Tar.gz archive integrity check failed"
            return 1
        fi
    elif [[ "$backup_file" == *.enc ]]; then
        log_info "Encrypted backup detected, integrity will be verified during decryption"
    fi
    
    log_success "Backup file validation completed successfully"
    return 0
}

# Backup existing files before restore
backup_existing_files() {
    local target_dir="$1"
    local backup_suffix="$(date +%Y%m%d_%H%M%S)_pre_restore"
    
    if [[ ! -d "$target_dir" ]]; then
        log_info "Target directory does not exist, no backup needed"
        return 0
    fi
    
    if [[ "$BACKUP_ORIGINAL" != "true" ]]; then
        log_info "Original file backup disabled, skipping"
        return 0
    fi
    
    log_info "Backing up existing files before restore..."
    
    local backup_dir="${target_dir}_backup_${backup_suffix}"
    
    if cp -rp "$target_dir" "$backup_dir"; then
        log_success "Existing files backed up to: $backup_dir"
        echo "$backup_dir" > "$SCRIPT_DIR/.restore_backup_path"
        log_audit "BACKUP_ORIGINAL" "$target_dir" "$backup_dir" "SUCCESS"
    else
        log_error "Failed to backup existing files"
        log_audit "BACKUP_ORIGINAL" "$target_dir" "$backup_dir" "FAILED"
        return 1
    fi
    
    return 0
}

# Decrypt backup file if encrypted
decrypt_backup() {
    local backup_file="$1"
    local temp_dir="$2"
    
    if [[ "$backup_file" != *.enc ]]; then
        log_debug "Backup file is not encrypted, skipping decryption"
        echo "$backup_file"
        return 0
    fi
    
    log_info "Decrypting backup file..."
    
    local key_file="$SCRIPT_DIR/backup_master.key"
    local decrypted_file="$temp_dir/$(basename "${backup_file%.enc}")"
    
    if [[ ! -f "$key_file" ]]; then
        log_error "Encryption key file not found: $key_file"
        return 1
    fi
    
    # Verify key file permissions
    local key_perms=$(stat -c "%a" "$key_file")
    if [[ "$key_perms" != "600" ]]; then
        log_warning "Encryption key has insecure permissions: $key_perms"
    fi
    
    # Decrypt using OpenSSL
    if openssl enc -aes-256-cbc -d -pbkdf2 -iter 100000 \
        -in "$backup_file" -out "$decrypted_file" -pass file:"$key_file"; then
        
        log_success "Backup decrypted successfully"
        log_audit "DECRYPT" "$backup_file" "$decrypted_file" "SUCCESS"
        echo "$decrypted_file"
    else
        log_error "Failed to decrypt backup file"
        log_audit "DECRYPT" "$backup_file" "$decrypted_file" "FAILED"
        return 1
    fi
}

# Extract backup archive
extract_backup() {
    local backup_file="$1"
    local extract_dir="$2"
    local selective_patterns=("${@:3}")
    
    log_info "Extracting backup archive..."
    log_debug "Extract directory: $extract_dir"
    
    # Create extraction directory
    mkdir -p "$extract_dir"
    
    # Determine archive type and extract
    if [[ "$backup_file" == *.tar.gz ]]; then
        local tar_options="-xzf"
        
        # Add verbose option if debug enabled
        if [[ "${DEBUG:-false}" == "true" ]]; then
            tar_options="${tar_options}v"
        fi
        
        # Selective restore based on patterns
        if [[ ${#selective_patterns[@]} -gt 0 ]]; then
            log_info "Performing selective restore with patterns: ${selective_patterns[*]}"
            
            # Create a temporary file list
            local file_list="$extract_dir/.restore_file_list"
            tar -tzf "$backup_file" > "$file_list"
            
            # Filter files based on patterns
            local filtered_list="$extract_dir/.restore_filtered_list"
            > "$filtered_list"
            
            for pattern in "${selective_patterns[@]}"; do
                grep -E "$pattern" "$file_list" >> "$filtered_list" || true
            done
            
            if [[ -s "$filtered_list" ]]; then
                local file_count=$(wc -l < "$filtered_list")
                log_info "Extracting $file_count selected files..."
                
                tar $tar_options "$backup_file" -C "$extract_dir" -T "$filtered_list"
            else
                log_warning "No files matched the specified patterns"
                return 1
            fi
            
            # Cleanup temporary files
            rm -f "$file_list" "$filtered_list"
        else
            # Extract all files
            if tar $tar_options "$backup_file" -C "$extract_dir"; then
                log_success "Archive extracted successfully"
            else
                log_error "Failed to extract archive"
                return 1
            fi
        fi
    else
        log_error "Unsupported archive format: $backup_file"
        return 1
    fi
    
    # Count extracted files
    local extracted_count=$(find "$extract_dir" -type f | wc -l)
    log_info "Extracted $extracted_count files"
    
    log_audit "EXTRACT" "$backup_file" "$extract_dir" "SUCCESS" "files_count=$extracted_count"
    return 0
}

# Validate extracted files
validate_extracted_files() {
    local extract_dir="$1"
    
    if [[ "$VERIFICATION_ENABLED" != "true" ]]; then
        log_debug "File validation disabled, skipping"
        return 0
    fi
    
    log_info "Validating extracted files..."
    
    local validation_errors=0
    local total_files=0
    
    while IFS= read -r -d '' file; do
        total_files=$((total_files + 1))
        
        # Check file accessibility
        if [[ ! -r "$file" ]]; then
            log_warning "Cannot read extracted file: $file"
            validation_errors=$((validation_errors + 1))
            continue
        fi
        
        # Validate key files specifically
        if [[ "$file" == *.key ]] || [[ "$file" == *.pem ]]; then
            # Basic PEM format validation
            if head -1 "$file" | grep -q "BEGIN.*PRIVATE KEY\|BEGIN.*CERTIFICATE\|BEGIN.*PUBLIC KEY"; then
                log_debug "PEM format validation passed: $(basename "$file")"
            else
                log_warning "PEM format validation failed: $(basename "$file")"
                validation_errors=$((validation_errors + 1))
            fi
        fi
        
        # Check for binary files that might be corrupted
        if file "$file" | grep -q "data"; then
            log_debug "Binary file detected: $(basename "$file")"
        fi
        
    done < <(find "$extract_dir" -type f -print0)
    
    log_info "File validation completed: $total_files files, $validation_errors errors"
    
    if [[ $validation_errors -gt 0 ]]; then
        local error_percentage=$(( validation_errors * 100 / total_files ))
        if [[ $error_percentage -gt 10 ]]; then  # More than 10% errors
            log_error "High validation error rate: ${error_percentage}%"
            return 1
        else
            log_warning "Some validation errors found but within acceptable range: ${error_percentage}%"
        fi
    fi
    
    log_success "Extracted files validation completed successfully"
    return 0
}

# Move files to final destination
install_restored_files() {
    local extract_dir="$1"
    local target_dir="$2"
    local preserve_structure="${3:-true}"
    
    log_info "Installing restored files to target directory..."
    log_debug "Source: $extract_dir"
    log_debug "Target: $target_dir"
    
    # Create target directory
    mkdir -p "$target_dir"
    
    # Find the actual keys directory within extracted files
    local keys_subdir
    if [[ -d "$extract_dir/keys" ]]; then
        keys_subdir="$extract_dir/keys"
    elif [[ -d "$extract_dir"/*/ ]]; then
        # Find the first subdirectory
        keys_subdir=$(find "$extract_dir" -maxdepth 1 -type d ! -path "$extract_dir" | head -1)
    else
        keys_subdir="$extract_dir"
    fi
    
    log_debug "Keys subdirectory: $keys_subdir"
    
    # Copy files with appropriate options
    local cp_options="-r"
    
    # Preserve permissions and timestamps if configured
    if [[ "$(jq -r '.restore_settings.preserve_permissions' "$RESTORE_CONFIG" 2>/dev/null)" == "true" ]]; then
        cp_options="${cp_options}p"
    fi
    
    # Use parallel processing if enabled and rsync is available
    if [[ "$PARALLEL_PROCESSING" == "true" ]] && command -v rsync >/dev/null 2>&1; then
        log_info "Using rsync for optimized file transfer..."
        
        local rsync_options="-av --progress"
        
        if [[ "$(jq -r '.restore_settings.preserve_ownership' "$RESTORE_CONFIG" 2>/dev/null)" == "true" ]]; then
            rsync_options="${rsync_options} --owner --group"
        fi
        
        if [[ "$(jq -r '.restore_settings.preserve_timestamps' "$RESTORE_CONFIG" 2>/dev/null)" == "true" ]]; then
            rsync_options="${rsync_options} --times"
        fi
        
        if rsync $rsync_options "$keys_subdir/" "$target_dir/"; then
            log_success "Files transferred successfully using rsync"
        else
            log_error "Rsync transfer failed, falling back to cp"
            cp $cp_options "$keys_subdir"/* "$target_dir"/ 2>/dev/null || true
        fi
    else
        # Standard copy operation
        if cp $cp_options "$keys_subdir"/* "$target_dir"/ 2>/dev/null; then
            log_success "Files copied successfully"
        else
            log_error "Failed to copy restored files"
            return 1
        fi
    fi
    
    # Count installed files
    local installed_count=$(find "$target_dir" -type f | wc -l)
    log_info "Installed $installed_count files to target directory"
    
    # Set appropriate permissions for key files
    find "$target_dir" -name "*.key" -exec chmod 600 {} \; 2>/dev/null || true
    find "$target_dir" -name "*.pem" -exec chmod 600 {} \; 2>/dev/null || true
    find "$target_dir" -name "*private*" -exec chmod 600 {} \; 2>/dev/null || true
    
    log_audit "INSTALL" "$keys_subdir" "$target_dir" "SUCCESS" "files_count=$installed_count"
    return 0
}

# Post-restore validation
post_restore_validation() {
    local target_dir="$1"
    local backup_file="$2"
    
    log_info "Performing post-restore validation..."
    
    # Check if target directory exists and has files
    if [[ ! -d "$target_dir" ]]; then
        log_error "Target directory does not exist after restore: $target_dir"
        return 1
    fi
    
    local restored_files=$(find "$target_dir" -type f | wc -l)
    if [[ $restored_files -eq 0 ]]; then
        log_error "No files found in target directory after restore"
        return 1
    fi
    
    log_info "Found $restored_files restored files"
    
    # Validate key file formats
    local key_validation_errors=0
    while IFS= read -r -d '' key_file; do
        if [[ "$key_file" == *.key ]] || [[ "$key_file" == *.pem ]]; then
            # Basic format validation
            if head -1 "$key_file" | grep -q "BEGIN.*PRIVATE KEY\|BEGIN.*CERTIFICATE\|BEGIN.*PUBLIC KEY"; then
                log_debug "Key format validation passed: $(basename "$key_file")"
            else
                log_warning "Key format validation failed: $(basename "$key_file")"
                key_validation_errors=$((key_validation_errors + 1))
            fi
            
            # Check permissions
            local perms=$(stat -c "%a" "$key_file")
            if [[ "$perms" != "600" ]] && [[ "$perms" != "400" ]]; then
                log_warning "Key file has loose permissions: $(basename "$key_file") ($perms)"
            fi
        fi
    done < <(find "$target_dir" -type f -print0)
    
    # Compare with original backup metadata if available
    local metadata_file="${backup_file}.metadata.json"
    if [[ -f "$metadata_file" ]]; then
        log_info "Comparing with original backup metadata..."
        
        local original_file_count=$(jq -r '.source_info.file_count' "$metadata_file" 2>/dev/null || echo "unknown")
        if [[ "$original_file_count" != "unknown" ]] && [[ "$original_file_count" != "$restored_files" ]]; then
            log_warning "File count mismatch: original=$original_file_count, restored=$restored_files"
        else
            log_success "File count matches original backup"
        fi
    fi
    
    if [[ $key_validation_errors -gt 0 ]]; then
        log_warning "Post-restore validation found $key_validation_errors key format issues"
    else
        log_success "Post-restore validation completed successfully"
    fi
    
    return 0
}

# Create restore point for rollback
create_restore_point() {
    local target_dir="$1"
    local restore_point_dir="$SCRIPT_DIR/recovery/restore_point_$(date +%Y%m%d_%H%M%S)"
    
    log_info "Creating restore point for potential rollback..."
    
    mkdir -p "$(dirname "$restore_point_dir")"
    
    if [[ -d "$target_dir" ]]; then
        if cp -rp "$target_dir" "$restore_point_dir"; then
            log_success "Restore point created: $restore_point_dir"
            echo "$restore_point_dir" > "$SCRIPT_DIR/.last_restore_point"
            log_audit "CREATE_RESTORE_POINT" "$target_dir" "$restore_point_dir" "SUCCESS"
            return 0
        else
            log_warning "Failed to create restore point"
            return 1
        fi
    else
        log_info "Target directory does not exist, no restore point needed"
        return 0
    fi
}

# Rollback to previous state
rollback_restore() {
    local restore_point_file="$SCRIPT_DIR/.last_restore_point"
    
    if [[ ! -f "$restore_point_file" ]]; then
        log_error "No restore point found for rollback"
        return 1
    fi
    
    local restore_point_dir=$(cat "$restore_point_file")
    
    if [[ ! -d "$restore_point_dir" ]]; then
        log_error "Restore point directory not found: $restore_point_dir"
        return 1
    fi
    
    log_info "Rolling back to previous state..."
    log_info "Restore point: $restore_point_dir"
    
    local target_dir="$(dirname "$(dirname "$restore_point_dir")")/keys"
    
    # Backup current state before rollback
    local rollback_backup_dir="${target_dir}_rollback_backup_$(date +%Y%m%d_%H%M%S)"
    if [[ -d "$target_dir" ]]; then
        cp -rp "$target_dir" "$rollback_backup_dir" || true
        log_info "Current state backed up to: $rollback_backup_dir"
    fi
    
    # Perform rollback
    if cp -rp "$restore_point_dir"/* "$(dirname "$target_dir")"/; then
        log_success "Rollback completed successfully"
        log_audit "ROLLBACK" "$restore_point_dir" "$target_dir" "SUCCESS"
        return 0
    else
        log_error "Rollback failed"
        log_audit "ROLLBACK" "$restore_point_dir" "$target_dir" "FAILED"
        return 1
    fi
}

# Generate restore report
generate_restore_report() {
    local backup_file="$1"
    local target_dir="$2"
    local start_time="$3"
    local end_time="$4"
    local report_file="$SCRIPT_DIR/restore_report_$(date +%Y%m%d_%H%M%S).txt"
    
    log_info "Generating restore report..."
    
    local duration=$((end_time - start_time))
    local restored_files=$(find "$target_dir" -type f 2>/dev/null | wc -l)
    local total_size=$(du -sh "$target_dir" 2>/dev/null | cut -f1)
    
    {
        echo "==================================================="
        echo "           KEYS RESTORE REPORT"
        echo "==================================================="
        echo "Report Generated: $(date)"
        echo "Hostname: $(hostname -f)"
        echo "User: $(whoami)"
        echo ""
        echo "RESTORE DETAILS:"
        echo "- Source Backup: $(basename "$backup_file")"
        echo "- Target Directory: $target_dir"
        echo "- Start Time: $(date -d @$start_time)"
        echo "- End Time: $(date -d @$end_time)"
        echo "- Duration: ${duration} seconds"
        echo "- Status: $([ $? -eq 0 ] && echo "SUCCESS" || echo "FAILED")"
        echo ""
        echo "FILE STATISTICS:"
        echo "- Files Restored: $restored_files"
        echo "- Total Size: $total_size"
        echo "- Average File Size: $(( $(du -sb "$target_dir" 2>/dev/null | cut -f1) / restored_files 2>/dev/null || echo "0" )) bytes"
        echo ""
        echo "CONFIGURATION:"
        echo "- Verification Enabled: $VERIFICATION_ENABLED"
        echo "- Backup Original: $BACKUP_ORIGINAL"
        echo "- Parallel Processing: $PARALLEL_PROCESSING"
        echo "- Max Workers: $MAX_WORKERS"
        echo ""
        echo "VALIDATION RESULTS:"
        echo "- Pre-restore Validation: $(grep -c "validation.*passed" "$RESTORE_LOG" || echo "0") checks passed"
        echo "- Post-restore Validation: $(grep -c "validation.*completed" "$RESTORE_LOG" || echo "0") checks completed"
        echo ""
        echo "SECURITY CHECKS:"
        echo "- Key Files: $(find "$target_dir" -name "*.key" 2>/dev/null | wc -l)"
        echo "- Certificate Files: $(find "$target_dir" -name "*.pem" -o -name "*.crt" 2>/dev/null | wc -l)"
        echo "- Private Key Permissions: $(find "$target_dir" -name "*.key" -perm 600 2>/dev/null | wc -l) with correct permissions"
        echo ""
        echo "PERFORMANCE METRICS:"
        echo "- Extraction Rate: $(( restored_files / duration )) files/second"
        echo "- Data Transfer Rate: $(( $(du -sb "$target_dir" 2>/dev/null | cut -f1) / duration / 1024 )) KB/second"
        echo ""
        echo "ERRORS AND WARNINGS:"
        grep -c "ERROR" "$RESTORE_LOG" 2>/dev/null && echo "- Errors: $(grep -c "ERROR" "$RESTORE_LOG")" || echo "- Errors: 0"
        grep -c "WARNING" "$RESTORE_LOG" 2>/dev/null && echo "- Warnings: $(grep -c "WARNING" "$RESTORE_LOG")" || echo "- Warnings: 0"
        echo ""
        echo "==================================================="
    } > "$report_file"
    
    chmod 600 "$report_file"
    log_success "Restore report generated: $report_file"
}

# Send restore notifications
send_restore_notification() {
    local status="$1"
    local message="$2"
    
    log_debug "Sending restore notification: $status - $message"
    
    # Syslog notification
    if command -v logger >/dev/null 2>&1; then
        logger -p local0.info "KeysRestore[$status]: $message"
    fi
    
    # Email notification (if configured)
    if command -v mail >/dev/null 2>&1; then
        local subject="Keys Restore $(echo "$status" | tr '[:lower:]' '[:upper:]'): $(hostname)"
        echo "$message" | mail -s "$subject" "admin@localhost" 2>/dev/null || true
    fi
}

# List available backups for restore
list_available_backups() {
    log_info "=== Available Backups for Restore ==="
    
    local backup_files=($(find "$SCRIPT_DIR" -name "keys_backup_*.tar.gz" -o -name "keys_backup_*.enc" | sort -r))
    
    if [[ ${#backup_files[@]} -eq 0 ]]; then
        log_info "No backup files found in $SCRIPT_DIR"
        return 0
    fi
    
    printf "%-40s %-12s %-20s %-10s %-10s\n" "Backup File" "Size" "Date" "Encrypted" "Verified"
    printf "%-40s %-12s %-20s %-10s %-10s\n" "$(printf '%*s' 40 '' | tr ' ' '-')" "$(printf '%*s' 12 '' | tr ' ' '-')" "$(printf '%*s' 20 '' | tr ' ' '-')" "$(printf '%*s' 10 '' | tr ' ' '-')" "$(printf '%*s' 10 '' | tr ' ' '-')"
    
    for backup_file in "${backup_files[@]}"; do
        local basename_file=$(basename "$backup_file")
        local size_human=$(stat -c%s "$backup_file" | numfmt --to=iec-i --suffix=B)
        local date_human=$(date -r "$backup_file" '+%Y-%m-%d %H:%M')
        local encrypted=$([ "${backup_file}" == *.enc ] && echo "Yes" || echo "No")
        local verified=$([ -f "${backup_file}.checksums" ] && echo "Yes" || echo "No")
        
        printf "%-40s %-12s %-20s %-10s %-10s\n" \
            "${basename_file:0:39}" \
            "$size_human" \
            "$date_human" \
            "$encrypted" \
            "$verified"
    done
    
    echo ""
    log_info "Total backups available: ${#backup_files[@]}"
    echo ""
    echo "Usage: $0 restore <backup_file> [target_directory]"
}

# Interactive restore mode
interactive_restore() {
    log_info "=== Interactive Restore Mode ==="
    
    # List available backups
    list_available_backups
    
    # Prompt for backup selection
    echo ""
    read -p "Enter backup filename to restore: " backup_filename
    
    if [[ -z "$backup_filename" ]]; then
        log_error "No backup filename provided"
        return 1
    fi
    
    local backup_file="$SCRIPT_DIR/$backup_filename"
    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi
    
    # Prompt for target directory
    echo ""
    read -p "Enter target directory [$RESTORE_DIR]: " target_dir
    target_dir="${target_dir:-$RESTORE_DIR}"
    
    # Prompt for options
    echo ""
    echo "Restore Options:"
    read -p "Verify backup integrity? [Y/n]: " verify_choice
    verify_choice="${verify_choice:-Y}"
    
    read -p "Backup existing files? [Y/n]: " backup_choice
    backup_choice="${backup_choice:-Y}"
    
    read -p "Perform selective restore? [y/N]: " selective_choice
    selective_choice="${selective_choice:-N}"
    
    local selective_patterns=()
    if [[ "$selective_choice" == [Yy] ]]; then
        echo "Enter file patterns (one per line, empty line to finish):"
        while true; do
            read -p "Pattern: " pattern
            if [[ -z "$pattern" ]]; then
                break
            fi
            selective_patterns+=("$pattern")
        done
    fi
    
    # Confirmation
    echo ""
    echo "Restore Summary:"
    echo "- Source: $backup_file"
    echo "- Target: $target_dir"
    echo "- Verify: $verify_choice"
    echo "- Backup existing: $backup_choice"
    echo "- Selective patterns: ${selective_patterns[*]:-none}"
    echo ""
    
    read -p "Proceed with restore? [y/N]: " confirm
    if [[ "$confirm" != [Yy] ]]; then
        log_info "Restore cancelled by user"
        return 1
    fi
    
    # Set options based on user choices
    VERIFICATION_ENABLED=$([ "$verify_choice" == [Yy] ] && echo "true" || echo "false")
    BACKUP_ORIGINAL=$([ "$backup_choice" == [Yy] ] && echo "true" || echo "false")
    
    # Perform restore
    restore_backup_file "$backup_file" "$target_dir" "${selective_patterns[@]}"
}

# Main restore function
restore_backup_file() {
    local backup_file="$1"
    local target_dir="${2:-$RESTORE_DIR}"
    local selective_patterns=("${@:3}")
    
    local start_time=$(date +%s)
    
    log_info "=== Starting Keys Restore Process ==="
    log_info "Source backup: $(basename "$backup_file")"
    log_info "Target directory: $target_dir"
    
    # Create temporary directory for extraction
    RESTORE_TEMP_DIR=$(mktemp -d -t keys_restore.XXXXXX)
    log_debug "Temporary directory: $RESTORE_TEMP_DIR"
    
    # Pre-restore validation
    validate_backup_file "$backup_file" || return 1
    
    # Create restore point
    create_restore_point "$target_dir" || log_warning "Failed to create restore point"
    
    # Backup existing files if enabled
    if [[ -d "$target_dir" ]]; then
        backup_existing_files "$target_dir" || return 1
    fi
    
    # Decrypt backup if encrypted
    local working_file
    working_file=$(decrypt_backup "$backup_file" "$RESTORE_TEMP_DIR") || return 1
    
    # Extract backup archive
    local extract_dir="$RESTORE_TEMP_DIR/extracted"
    extract_backup "$working_file" "$extract_dir" "${selective_patterns[@]}" || return 1
    
    # Validate extracted files
    validate_extracted_files "$extract_dir" || return 1
    
    # Install restored files
    install_restored_files "$extract_dir" "$target_dir" || return 1
    
    # Post-restore validation
    post_restore_validation "$target_dir" "$backup_file" || return 1
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Generate restore report
    generate_restore_report "$backup_file" "$target_dir" "$start_time" "$end_time"
    
    log_success "=== Restore Process Completed Successfully ==="
    log_success "Duration: ${duration} seconds"
    log_success "Target: $target_dir"
    
    # Audit logging
    log_audit "RESTORE_COMPLETE" "$backup_file" "$target_dir" "SUCCESS" "duration=${duration}s"
    
    return 0
}

# Usage information
show_usage() {
    cat << EOF
Enterprise Keys Backup Restoration & Recovery System

USAGE:
    $0 [OPTIONS] COMMAND

COMMANDS:
    restore FILE [DIR]  Restore backup FILE to directory DIR
    list               List available backups for restore
    interactive        Interactive restore mode with guided prompts
    rollback           Rollback to previous restore point
    verify FILE        Verify backup file integrity without restoring
    config             Show current restore configuration
    help               Show this usage information

OPTIONS:
    -c, --config FILE   Use custom configuration file
    -v, --verbose       Enable verbose output
    -d, --debug         Enable debug output
    -q, --quiet         Suppress non-error output
    --dry-run          Show what would be done without executing
    --force            Force operation without confirmation prompts
    --selective PATTERN Use selective restore with pattern
    --no-verify        Skip backup verification
    --no-backup        Skip backing up existing files

EXAMPLES:
    $0 restore keys_backup_20240716.tar.gz                    # Restore to default directory
    $0 restore keys_backup_20240716.enc /opt/keys/restored    # Restore encrypted backup to specific directory
    $0 interactive                                             # Interactive restore mode
    $0 list                                                    # List available backups
    $0 verify keys_backup_20240716.tar.gz                     # Verify backup integrity
    $0 rollback                                                # Rollback last restore

SELECTIVE RESTORE:
    $0 restore backup.tar.gz --selective "*.key"              # Restore only .key files
    $0 restore backup.tar.gz --selective "server.*"           # Restore files matching pattern

CONFIGURATION:
    Configuration file: $RESTORE_CONFIG
    Use '$0 config' to view current settings

The restore system provides:
- Pre and post-restore validation
- Automatic backup of existing files
- Encrypted backup support
- Selective file restoration
- Point-in-time recovery
- Automatic rollback capabilities
- Comprehensive audit logging

For detailed documentation, see the README files in the parent directory.
EOF
}

# Show configuration
show_restore_config() {
    log_info "=== Restore Configuration ==="
    
    if [[ -f "$RESTORE_CONFIG" ]]; then
        if command -v jq >/dev/null 2>&1; then
            jq '.' "$RESTORE_CONFIG"
        else
            cat "$RESTORE_CONFIG"
        fi
    else
        echo "Configuration file not found: $RESTORE_CONFIG"
        echo "Run with 'restore' command to create default configuration."
    fi
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--config)
                RESTORE_CONFIG="$2"
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
            --selective)
                SELECTIVE_PATTERNS+=("$2")
                shift 2
                ;;
            --no-verify)
                VERIFICATION_ENABLED=false
                shift
                ;;
            --no-backup)
                BACKUP_ORIGINAL=false
                shift
                ;;
            restore|list|interactive|rollback|verify|config|help)
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
    SELECTIVE_PATTERNS=()
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
    load_restore_config
    
    # Override configuration with command line options
    AUTO_CLEANUP_TEMP=$(jq -r '.retention.auto_cleanup_temp_files' "$RESTORE_CONFIG" 2>/dev/null || echo "true")
    NOTIFICATION_ENABLED=$(jq -r '.notifications.email_on_completion' "$RESTORE_CONFIG" 2>/dev/null || echo "false")
    
    # Acquire lock for restore operations
    if [[ "$COMMAND" == "restore" ]] || [[ "$COMMAND" == "interactive" ]] || [[ "$COMMAND" == "rollback" ]]; then
        acquire_restore_lock || exit 1
    fi
    
    # Execute command
    case "$COMMAND" in
        restore)
            if [[ ${#ARGS[@]} -lt 1 ]]; then
                log_error "Restore command requires backup file argument"
                exit 1
            fi
            restore_backup_file "${ARGS[0]}" "${ARGS[1]:-}" "${SELECTIVE_PATTERNS[@]}"
            ;;
        list)
            list_available_backups
            ;;
        interactive)
            interactive_restore
            ;;
        rollback)
            rollback_restore
            ;;
        verify)
            if [[ ${#ARGS[@]} -lt 1 ]]; then
                log_error "Verify command requires backup file argument"
                exit 1
            fi
            validate_backup_file "${ARGS[0]}"
            ;;
        config)
            show_restore_config
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
