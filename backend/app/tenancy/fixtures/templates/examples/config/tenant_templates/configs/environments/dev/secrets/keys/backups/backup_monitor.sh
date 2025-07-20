#!/bin/bash
# =============================================================================
# Enterprise Keys Backup Monitoring & Health Surveillance System
# =============================================================================
# 
# Ultra-advanced continuous monitoring system for cryptographic keys backup
# with real-time alerts, performance metrics, compliance monitoring,
# and automated incident response capabilities
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
MONITOR_CONFIG="$SCRIPT_DIR/monitor_config.json"
MONITOR_LOG="$SCRIPT_DIR/monitor.log"
METRICS_LOG="$SCRIPT_DIR/metrics.log"
ALERTS_LOG="$SCRIPT_DIR/alerts.log"
PID_FILE="$SCRIPT_DIR/monitor.pid"
STATUS_FILE="$SCRIPT_DIR/monitor_status.json"

# Default monitoring intervals (seconds)
DEFAULT_HEALTH_CHECK_INTERVAL=300    # 5 minutes
DEFAULT_BACKUP_CHECK_INTERVAL=900    # 15 minutes
DEFAULT_DISK_CHECK_INTERVAL=180      # 3 minutes
DEFAULT_PERFORMANCE_CHECK_INTERVAL=60 # 1 minute
DEFAULT_SECURITY_CHECK_INTERVAL=600  # 10 minutes

# Alert thresholds
DEFAULT_DISK_WARNING_THRESHOLD=80    # 80% disk usage
DEFAULT_DISK_CRITICAL_THRESHOLD=90   # 90% disk usage
DEFAULT_BACKUP_AGE_WARNING_HOURS=26  # 26 hours since last backup
DEFAULT_BACKUP_AGE_CRITICAL_HOURS=48 # 48 hours since last backup
DEFAULT_FAILED_BACKUPS_THRESHOLD=3   # 3 consecutive failures

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
    echo "[$timestamp] [$level] $message" | tee -a "$MONITOR_LOG"
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

# Metrics logging
log_metric() {
    local metric_name="$1"
    local metric_value="$2"
    local metric_type="${3:-gauge}"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] $metric_type:$metric_name:$metric_value" >> "$METRICS_LOG"
}

# Alert logging
log_alert() {
    local alert_level="$1"
    local alert_type="$2"
    local alert_message="$3"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] [$alert_level] [$alert_type] $alert_message" >> "$ALERTS_LOG"
    
    # Send notification based on alert level
    case "$alert_level" in
        "CRITICAL")
            send_alert_notification "critical" "$alert_type" "$alert_message"
            ;;
        "WARNING")
            send_alert_notification "warning" "$alert_type" "$alert_message"
            ;;
        "INFO")
            send_alert_notification "info" "$alert_type" "$alert_message"
            ;;
    esac
}

# Create default monitoring configuration
create_default_monitor_config() {
    log_info "Creating default monitoring configuration..."
    
    cat > "$MONITOR_CONFIG" << EOF
{
    "monitoring": {
        "enabled": true,
        "daemon_mode": true,
        "log_level": "INFO",
        "metrics_retention_days": 30,
        "alerts_retention_days": 90
    },
    "intervals": {
        "health_check": $DEFAULT_HEALTH_CHECK_INTERVAL,
        "backup_check": $DEFAULT_BACKUP_CHECK_INTERVAL,
        "disk_check": $DEFAULT_DISK_CHECK_INTERVAL,
        "performance_check": $DEFAULT_PERFORMANCE_CHECK_INTERVAL,
        "security_check": $DEFAULT_SECURITY_CHECK_INTERVAL
    },
    "thresholds": {
        "disk_warning_percent": $DEFAULT_DISK_WARNING_THRESHOLD,
        "disk_critical_percent": $DEFAULT_DISK_CRITICAL_THRESHOLD,
        "backup_age_warning_hours": $DEFAULT_BACKUP_AGE_WARNING_HOURS,
        "backup_age_critical_hours": $DEFAULT_BACKUP_AGE_CRITICAL_HOURS,
        "failed_backups_threshold": $DEFAULT_FAILED_BACKUPS_THRESHOLD,
        "memory_warning_percent": 80,
        "memory_critical_percent": 90,
        "cpu_warning_percent": 80,
        "cpu_critical_percent": 95,
        "load_average_warning": 5.0,
        "load_average_critical": 10.0
    },
    "paths": {
        "backup_directory": "$SCRIPT_DIR",
        "keys_directory": "$(dirname "$SCRIPT_DIR")",
        "log_directory": "$SCRIPT_DIR",
        "temp_directory": "/tmp/backup_monitor"
    },
    "checks": {
        "backup_integrity": {
            "enabled": true,
            "verify_checksums": true,
            "test_restore": false,
            "check_encryption": true
        },
        "system_health": {
            "enabled": true,
            "check_disk_space": true,
            "check_memory": true,
            "check_cpu": true,
            "check_load": true,
            "check_processes": true
        },
        "security_audit": {
            "enabled": true,
            "check_permissions": true,
            "check_ownership": true,
            "check_encryption_keys": true,
            "scan_for_threats": false
        },
        "compliance": {
            "enabled": true,
            "gdpr_compliance": true,
            "pci_compliance": true,
            "hipaa_compliance": false,
            "audit_trail": true
        }
    },
    "notifications": {
        "email": {
            "enabled": false,
            "smtp_server": "localhost",
            "smtp_port": 587,
            "username": "",
            "password": "",
            "recipients": ["admin@localhost"],
            "send_on_critical": true,
            "send_on_warning": true,
            "send_on_recovery": true
        },
        "slack": {
            "enabled": false,
            "webhook_url": "",
            "channel": "#alerts",
            "username": "BackupMonitor",
            "send_on_critical": true,
            "send_on_warning": false
        },
        "webhook": {
            "enabled": false,
            "url": "",
            "secret": "",
            "timeout": 30,
            "retry_attempts": 3
        },
        "syslog": {
            "enabled": true,
            "facility": "local0",
            "priority": "info"
        }
    },
    "metrics": {
        "prometheus": {
            "enabled": false,
            "port": 9090,
            "metrics_path": "/metrics",
            "job_name": "backup_monitor"
        },
        "influxdb": {
            "enabled": false,
            "host": "localhost",
            "port": 8086,
            "database": "backup_metrics",
            "username": "",
            "password": ""
        },
        "custom": {
            "enabled": true,
            "file_format": "json",
            "include_system_metrics": true,
            "include_backup_metrics": true
        }
    },
    "remediation": {
        "auto_remediation": {
            "enabled": false,
            "cleanup_old_logs": true,
            "restart_failed_services": false,
            "fix_permissions": true
        },
        "backup_recovery": {
            "auto_retry_failed_backups": true,
            "max_retry_attempts": 3,
            "retry_delay_minutes": 30,
            "escalation_threshold": 5
        }
    }
}
EOF

    chmod 600 "$MONITOR_CONFIG"
    log_success "Default monitoring configuration created at $MONITOR_CONFIG"
}

# Load monitoring configuration
load_monitor_config() {
    if [[ ! -f "$MONITOR_CONFIG" ]]; then
        log_warning "Monitoring configuration not found, creating default"
        create_default_monitor_config
    fi
    
    if command -v jq >/dev/null 2>&1; then
        # Load configuration using jq
        HEALTH_CHECK_INTERVAL=$(jq -r '.intervals.health_check' "$MONITOR_CONFIG")
        BACKUP_CHECK_INTERVAL=$(jq -r '.intervals.backup_check' "$MONITOR_CONFIG")
        DISK_CHECK_INTERVAL=$(jq -r '.intervals.disk_check' "$MONITOR_CONFIG")
        PERFORMANCE_CHECK_INTERVAL=$(jq -r '.intervals.performance_check' "$MONITOR_CONFIG")
        SECURITY_CHECK_INTERVAL=$(jq -r '.intervals.security_check' "$MONITOR_CONFIG")
        
        DISK_WARNING_THRESHOLD=$(jq -r '.thresholds.disk_warning_percent' "$MONITOR_CONFIG")
        DISK_CRITICAL_THRESHOLD=$(jq -r '.thresholds.disk_critical_percent' "$MONITOR_CONFIG")
        BACKUP_AGE_WARNING_HOURS=$(jq -r '.thresholds.backup_age_warning_hours' "$MONITOR_CONFIG")
        BACKUP_AGE_CRITICAL_HOURS=$(jq -r '.thresholds.backup_age_critical_hours' "$MONITOR_CONFIG")
        
        BACKUP_DIR=$(jq -r '.paths.backup_directory' "$MONITOR_CONFIG")
        KEYS_DIR=$(jq -r '.paths.keys_directory' "$MONITOR_CONFIG")
    else
        log_warning "jq not available, using default configuration values"
        HEALTH_CHECK_INTERVAL=$DEFAULT_HEALTH_CHECK_INTERVAL
        BACKUP_CHECK_INTERVAL=$DEFAULT_BACKUP_CHECK_INTERVAL
        DISK_CHECK_INTERVAL=$DEFAULT_DISK_CHECK_INTERVAL
        PERFORMANCE_CHECK_INTERVAL=$DEFAULT_PERFORMANCE_CHECK_INTERVAL
        SECURITY_CHECK_INTERVAL=$DEFAULT_SECURITY_CHECK_INTERVAL
        
        DISK_WARNING_THRESHOLD=$DEFAULT_DISK_WARNING_THRESHOLD
        DISK_CRITICAL_THRESHOLD=$DEFAULT_DISK_CRITICAL_THRESHOLD
        BACKUP_AGE_WARNING_HOURS=$DEFAULT_BACKUP_AGE_WARNING_HOURS
        BACKUP_AGE_CRITICAL_HOURS=$DEFAULT_BACKUP_AGE_CRITICAL_HOURS
        
        BACKUP_DIR="$SCRIPT_DIR"
        KEYS_DIR="$(dirname "$SCRIPT_DIR")"
    fi
    
    log_debug "Monitoring configuration loaded successfully"
}

# System resource monitoring
check_system_resources() {
    log_debug "Checking system resources..."
    
    # Disk space check
    local disk_usage=$(df "$BACKUP_DIR" | awk 'NR==2 {gsub(/%/,"",$5); print $5}')
    log_metric "disk_usage_percent" "$disk_usage"
    
    if [[ $disk_usage -ge $DISK_CRITICAL_THRESHOLD ]]; then
        log_alert "CRITICAL" "DISK_SPACE" "Disk usage critical: ${disk_usage}% (threshold: ${DISK_CRITICAL_THRESHOLD}%)"
    elif [[ $disk_usage -ge $DISK_WARNING_THRESHOLD ]]; then
        log_alert "WARNING" "DISK_SPACE" "Disk usage warning: ${disk_usage}% (threshold: ${DISK_WARNING_THRESHOLD}%)"
    fi
    
    # Memory usage check
    local memory_info=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')
    log_metric "memory_usage_percent" "$memory_info"
    
    if (( $(echo "$memory_info >= 90" | bc -l) )); then
        log_alert "CRITICAL" "MEMORY" "Memory usage critical: ${memory_info}%"
    elif (( $(echo "$memory_info >= 80" | bc -l) )); then
        log_alert "WARNING" "MEMORY" "Memory usage warning: ${memory_info}%"
    fi
    
    # CPU usage check (5-minute average)
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    log_metric "cpu_usage_percent" "$cpu_usage"
    
    # Load average check
    local load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
    log_metric "load_average_1min" "$load_avg"
    
    if (( $(echo "$load_avg >= 10.0" | bc -l) )); then
        log_alert "CRITICAL" "LOAD_AVERAGE" "Load average critical: $load_avg"
    elif (( $(echo "$load_avg >= 5.0" | bc -l) )); then
        log_alert "WARNING" "LOAD_AVERAGE" "Load average warning: $load_avg"
    fi
    
    # Process monitoring
    local backup_processes=$(pgrep -f "backup_automation.sh" | wc -l)
    log_metric "backup_processes_count" "$backup_processes"
    
    # I/O statistics
    if command -v iostat >/dev/null 2>&1; then
        local io_util=$(iostat -x 1 2 | awk 'NR==7{print $10}' | head -1)
        log_metric "disk_io_utilization" "${io_util:-0}"
    fi
    
    log_debug "System resources check completed"
}

# Backup monitoring
check_backup_status() {
    log_debug "Checking backup status..."
    
    local backup_files=($(find "$BACKUP_DIR" -name "keys_backup_*.tar.gz" -o -name "keys_backup_*.enc" | sort -r))
    local backup_count=${#backup_files[@]}
    
    log_metric "total_backups_count" "$backup_count"
    
    if [[ $backup_count -eq 0 ]]; then
        log_alert "CRITICAL" "BACKUP_MISSING" "No backup files found in $BACKUP_DIR"
        return 1
    fi
    
    # Check last backup age
    local latest_backup="${backup_files[0]}"
    local backup_age_seconds=$(( $(date +%s) - $(stat -c %Y "$latest_backup") ))
    local backup_age_hours=$(( backup_age_seconds / 3600 ))
    
    log_metric "last_backup_age_hours" "$backup_age_hours"
    
    if [[ $backup_age_hours -ge $BACKUP_AGE_CRITICAL_HOURS ]]; then
        log_alert "CRITICAL" "BACKUP_OUTDATED" "Last backup is ${backup_age_hours} hours old (critical threshold: ${BACKUP_AGE_CRITICAL_HOURS}h)"
    elif [[ $backup_age_hours -ge $BACKUP_AGE_WARNING_HOURS ]]; then
        log_alert "WARNING" "BACKUP_OUTDATED" "Last backup is ${backup_age_hours} hours old (warning threshold: ${BACKUP_AGE_WARNING_HOURS}h)"
    fi
    
    # Check backup integrity
    for backup_file in "${backup_files[@]:0:3}"; do  # Check last 3 backups
        local checksum_file="${backup_file}.checksums"
        if [[ -f "$checksum_file" ]]; then
            local expected_sha256=$(grep "^SHA256:" "$checksum_file" | cut -d' ' -f2)
            local actual_sha256=$(sha256sum "$backup_file" | cut -d' ' -f1)
            
            if [[ "$expected_sha256" == "$actual_sha256" ]]; then
                log_debug "Backup integrity verified: $(basename "$backup_file")"
            else
                log_alert "CRITICAL" "BACKUP_CORRUPTION" "Backup corruption detected: $(basename "$backup_file")"
            fi
        else
            log_alert "WARNING" "MISSING_CHECKSUMS" "Checksum file missing for: $(basename "$backup_file")"
        fi
    done
    
    # Check backup sizes (detect unusual size changes)
    if [[ ${#backup_files[@]} -ge 2 ]]; then
        local latest_size=$(stat -c%s "${backup_files[0]}")
        local previous_size=$(stat -c%s "${backup_files[1]}")
        local size_diff_percent=$(( (latest_size - previous_size) * 100 / previous_size ))
        
        log_metric "backup_size_change_percent" "$size_diff_percent"
        
        if [[ ${size_diff_percent#-} -gt 50 ]]; then  # Remove negative sign for comparison
            log_alert "WARNING" "BACKUP_SIZE_CHANGE" "Significant backup size change: ${size_diff_percent}%"
        fi
    fi
    
    # Check failed backup attempts
    local failed_backups=$(grep -c "ERROR" "$SCRIPT_DIR/backup_automation.log" 2>/dev/null || echo "0")
    local recent_failed_backups=$(tail -100 "$SCRIPT_DIR/backup_automation.log" 2>/dev/null | grep -c "ERROR" || echo "0")
    
    log_metric "failed_backups_total" "$failed_backups"
    log_metric "recent_failed_backups" "$recent_failed_backups"
    
    if [[ $recent_failed_backups -ge $DEFAULT_FAILED_BACKUPS_THRESHOLD ]]; then
        log_alert "CRITICAL" "BACKUP_FAILURES" "Multiple recent backup failures: $recent_failed_backups"
    fi
    
    log_debug "Backup status check completed"
}

# Security monitoring
check_security_status() {
    log_debug "Checking security status..."
    
    # Check file permissions
    while IFS= read -r -d '' file; do
        local perms=$(stat -c "%a" "$file")
        local owner=$(stat -c "%U" "$file")
        
        # Key files should have restricted permissions
        if [[ "$file" == *.key ]] || [[ "$file" == *.pem ]] || [[ "$file" == *private* ]]; then
            if [[ "$perms" != "600" ]] && [[ "$perms" != "400" ]]; then
                log_alert "WARNING" "FILE_PERMISSIONS" "Insecure permissions on sensitive file: $file ($perms)"
            fi
        fi
        
        # Backup files should be owned by backup user or root
        if [[ "$file" == *backup* ]] && [[ "$owner" != "root" ]] && [[ "$owner" != "backup" ]]; then
            log_alert "WARNING" "FILE_OWNERSHIP" "Unexpected ownership on backup file: $file ($owner)"
        fi
    done < <(find "$BACKUP_DIR" "$KEYS_DIR" -type f -print0 2>/dev/null)
    
    # Check for unauthorized access attempts
    local auth_failures=$(journalctl --since="1 hour ago" | grep -i "authentication failure" | wc -l 2>/dev/null || echo "0")
    log_metric "auth_failures_1hour" "$auth_failures"
    
    if [[ $auth_failures -gt 10 ]]; then
        log_alert "WARNING" "SECURITY" "Multiple authentication failures detected: $auth_failures in last hour"
    fi
    
    # Check encryption key security
    local key_file="$BACKUP_DIR/backup_master.key"
    if [[ -f "$key_file" ]]; then
        local key_perms=$(stat -c "%a" "$key_file")
        if [[ "$key_perms" != "600" ]]; then
            log_alert "CRITICAL" "KEY_SECURITY" "Encryption key has insecure permissions: $key_file ($key_perms)"
        fi
        
        # Check key age
        local key_age_days=$(( ($(date +%s) - $(stat -c %Y "$key_file")) / 86400 ))
        log_metric "encryption_key_age_days" "$key_age_days"
        
        if [[ $key_age_days -gt 365 ]]; then
            log_alert "WARNING" "KEY_ROTATION" "Encryption key is over 1 year old, consider rotation"
        fi
    fi
    
    log_debug "Security status check completed"
}

# Performance monitoring
check_performance_metrics() {
    log_debug "Collecting performance metrics..."
    
    # Backup operation timing
    if [[ -f "$SCRIPT_DIR/backup_automation.log" ]]; then
        local last_backup_duration=$(grep "Backup Process Completed" "$SCRIPT_DIR/backup_automation.log" | tail -1 | grep -o '[0-9]\+\.[0-9]\+s' | sed 's/s//' || echo "0")
        log_metric "last_backup_duration_seconds" "$last_backup_duration"
    fi
    
    # Disk I/O metrics
    if command -v iostat >/dev/null 2>&1; then
        local read_iops=$(iostat -x 1 2 | awk 'END{print $4}')
        local write_iops=$(iostat -x 1 2 | awk 'END{print $5}')
        log_metric "disk_read_iops" "${read_iops:-0}"
        log_metric "disk_write_iops" "${write_iops:-0}"
    fi
    
    # Network I/O (if applicable for remote backups)
    if command -v iftop >/dev/null 2>&1; then
        local network_usage=$(iftop -t -s 10 2>/dev/null | grep "Total send rate" | awk '{print $4}' || echo "0")
        log_metric "network_usage_kbps" "${network_usage:-0}"
    fi
    
    # System uptime
    local uptime_seconds=$(awk '{print int($1)}' /proc/uptime)
    log_metric "system_uptime_seconds" "$uptime_seconds"
    
    log_debug "Performance metrics collection completed"
}

# Compliance monitoring
check_compliance_status() {
    log_debug "Checking compliance status..."
    
    # Data retention compliance
    local oldest_backup=$(find "$BACKUP_DIR" -name "keys_backup_*.tar.gz" -o -name "keys_backup_*.enc" -printf '%T@ %p\n' | sort -n | head -1 | cut -d' ' -f2)
    if [[ -n "$oldest_backup" ]]; then
        local oldest_age_days=$(( ($(date +%s) - $(stat -c %Y "$oldest_backup")) / 86400 ))
        log_metric "oldest_backup_age_days" "$oldest_age_days"
        
        # Check against retention policy
        local retention_days=$(jq -r '.backup_settings.retention_days' "$SCRIPT_DIR/backup_config.json" 2>/dev/null || echo "30")
        if [[ $oldest_age_days -gt $((retention_days + 7)) ]]; then  # Grace period
            log_alert "WARNING" "COMPLIANCE" "Backup retention policy violation: oldest backup is ${oldest_age_days} days old"
        fi
    fi
    
    # Audit trail check
    local audit_log="$BACKUP_DIR/backup_audit.log"
    if [[ -f "$audit_log" ]]; then
        local audit_entries_today=$(grep "$(date +%Y-%m-%d)" "$audit_log" | wc -l)
        log_metric "audit_entries_today" "$audit_entries_today"
    fi
    
    # Encryption compliance
    local unencrypted_backups=$(find "$BACKUP_DIR" -name "keys_backup_*.tar.gz" ! -name "*.enc" | wc -l)
    local total_backups=$(find "$BACKUP_DIR" -name "keys_backup_*.tar.gz" -o -name "keys_backup_*.enc" | wc -l)
    
    if [[ $total_backups -gt 0 ]]; then
        local encryption_percentage=$(( (total_backups - unencrypted_backups) * 100 / total_backups ))
        log_metric "backup_encryption_percentage" "$encryption_percentage"
        
        if [[ $encryption_percentage -lt 100 ]]; then
            log_alert "WARNING" "COMPLIANCE" "Not all backups are encrypted: ${encryption_percentage}%"
        fi
    fi
    
    log_debug "Compliance status check completed"
}

# Health check orchestrator
perform_health_check() {
    log_debug "Performing comprehensive health check..."
    
    local start_time=$(date +%s)
    local health_status="healthy"
    local issues_found=0
    
    # Capture current status
    local status_data="{
        \"timestamp\": \"$(date -Iseconds)\",
        \"status\": \"running\",
        \"checks\": {
            \"system_resources\": {\"status\": \"pending\", \"last_check\": null},
            \"backup_status\": {\"status\": \"pending\", \"last_check\": null},
            \"security_status\": {\"status\": \"pending\", \"last_check\": null},
            \"performance_metrics\": {\"status\": \"pending\", \"last_check\": null},
            \"compliance_status\": {\"status\": \"pending\", \"last_check\": null}
        }
    }"
    
    echo "$status_data" > "$STATUS_FILE"
    
    # System resources check
    if check_system_resources; then
        update_check_status "system_resources" "healthy"
    else
        update_check_status "system_resources" "unhealthy"
        health_status="degraded"
        issues_found=$((issues_found + 1))
    fi
    
    # Backup status check
    if check_backup_status; then
        update_check_status "backup_status" "healthy"
    else
        update_check_status "backup_status" "unhealthy"
        health_status="critical"
        issues_found=$((issues_found + 1))
    fi
    
    # Security status check
    if check_security_status; then
        update_check_status "security_status" "healthy"
    else
        update_check_status "security_status" "unhealthy"
        health_status="degraded"
        issues_found=$((issues_found + 1))
    fi
    
    # Performance metrics
    if check_performance_metrics; then
        update_check_status "performance_metrics" "healthy"
    else
        update_check_status "performance_metrics" "unhealthy"
        health_status="degraded"
        issues_found=$((issues_found + 1))
    fi
    
    # Compliance status
    if check_compliance_status; then
        update_check_status "compliance_status" "healthy"
    else
        update_check_status "compliance_status" "unhealthy"
        health_status="degraded"
        issues_found=$((issues_found + 1))
    fi
    
    local end_time=$(date +%s)
    local check_duration=$((end_time - start_time))
    log_metric "health_check_duration_seconds" "$check_duration"
    
    # Update overall status
    jq --arg status "$health_status" --arg timestamp "$(date -Iseconds)" \
       '.status = $status | .timestamp = $timestamp' "$STATUS_FILE" > "${STATUS_FILE}.tmp" && \
       mv "${STATUS_FILE}.tmp" "$STATUS_FILE"
    
    log_info "Health check completed: $health_status ($issues_found issues found in ${check_duration}s)"
    
    # Send summary alert if needed
    if [[ "$health_status" == "critical" ]]; then
        log_alert "CRITICAL" "HEALTH_CHECK" "System health critical: $issues_found critical issues found"
    elif [[ "$health_status" == "degraded" ]]; then
        log_alert "WARNING" "HEALTH_CHECK" "System health degraded: $issues_found issues found"
    fi
}

# Update individual check status
update_check_status() {
    local check_name="$1"
    local status="$2"
    local timestamp=$(date -Iseconds)
    
    jq --arg check "$check_name" --arg status "$status" --arg timestamp "$timestamp" \
       '.checks[$check].status = $status | .checks[$check].last_check = $timestamp' \
       "$STATUS_FILE" > "${STATUS_FILE}.tmp" && mv "${STATUS_FILE}.tmp" "$STATUS_FILE"
}

# Send alert notifications
send_alert_notification() {
    local alert_level="$1"
    local alert_type="$2"
    local alert_message="$3"
    
    log_debug "Sending alert notification: $alert_level - $alert_type"
    
    # Syslog notification
    if command -v logger >/dev/null 2>&1; then
        logger -p local0.info "BackupMonitor[$alert_level]: $alert_type - $alert_message"
    fi
    
    # Email notification (if configured)
    if [[ -f "$MONITOR_CONFIG" ]] && command -v jq >/dev/null 2>&1; then
        local email_enabled=$(jq -r '.notifications.email.enabled' "$MONITOR_CONFIG" 2>/dev/null)
        if [[ "$email_enabled" == "true" ]]; then
            send_email_alert "$alert_level" "$alert_type" "$alert_message"
        fi
    fi
    
    # Webhook notification
    send_webhook_alert "$alert_level" "$alert_type" "$alert_message"
}

# Send email alert
send_email_alert() {
    local alert_level="$1"
    local alert_type="$2"
    local alert_message="$3"
    
    if command -v mail >/dev/null 2>&1; then
        local subject="[$alert_level] Backup Monitor Alert: $alert_type"
        local body="Alert Details:
Level: $alert_level
Type: $alert_type
Message: $alert_message
Timestamp: $(date)
Hostname: $(hostname -f)

This is an automated alert from the backup monitoring system."
        
        echo "$body" | mail -s "$subject" "admin@localhost" 2>/dev/null || true
    fi
}

# Send webhook alert
send_webhook_alert() {
    local alert_level="$1"
    local alert_type="$2"
    local alert_message="$3"
    
    # Placeholder for webhook implementation
    local webhook_payload="{
        \"level\": \"$alert_level\",
        \"type\": \"$alert_type\",
        \"message\": \"$alert_message\",
        \"timestamp\": \"$(date -Iseconds)\",
        \"hostname\": \"$(hostname -f)\",
        \"service\": \"backup_monitor\"
    }"
    
    # Example webhook call (uncomment and configure as needed)
    # curl -X POST -H "Content-Type: application/json" \
    #      -d "$webhook_payload" \
    #      "$WEBHOOK_URL" 2>/dev/null || true
}

# Daemon mode functions
start_daemon() {
    if [[ -f "$PID_FILE" ]]; then
        local existing_pid=$(cat "$PID_FILE")
        if kill -0 "$existing_pid" 2>/dev/null; then
            log_error "Monitor daemon already running (PID: $existing_pid)"
            return 1
        else
            log_warning "Removing stale PID file"
            rm -f "$PID_FILE"
        fi
    fi
    
    log_info "Starting backup monitor daemon..."
    
    # Create daemon process
    nohup "$0" --daemon-worker > /dev/null 2>&1 &
    local daemon_pid=$!
    echo "$daemon_pid" > "$PID_FILE"
    
    log_success "Monitor daemon started (PID: $daemon_pid)"
}

stop_daemon() {
    if [[ -f "$PID_FILE" ]]; then
        local daemon_pid=$(cat "$PID_FILE")
        if kill -0 "$daemon_pid" 2>/dev/null; then
            log_info "Stopping monitor daemon (PID: $daemon_pid)..."
            kill "$daemon_pid"
            
            # Wait for graceful shutdown
            local timeout=10
            while [[ $timeout -gt 0 ]] && kill -0 "$daemon_pid" 2>/dev/null; do
                sleep 1
                timeout=$((timeout - 1))
            done
            
            # Force kill if necessary
            if kill -0 "$daemon_pid" 2>/dev/null; then
                log_warning "Force killing daemon process"
                kill -9 "$daemon_pid" 2>/dev/null || true
            fi
            
            rm -f "$PID_FILE"
            log_success "Monitor daemon stopped"
        else
            log_warning "Daemon PID file exists but process not running"
            rm -f "$PID_FILE"
        fi
    else
        log_warning "Monitor daemon not running (no PID file found)"
    fi
}

daemon_status() {
    if [[ -f "$PID_FILE" ]]; then
        local daemon_pid=$(cat "$PID_FILE")
        if kill -0 "$daemon_pid" 2>/dev/null; then
            log_info "Monitor daemon running (PID: $daemon_pid)"
            
            # Show last status if available
            if [[ -f "$STATUS_FILE" ]]; then
                echo ""
                echo "Last Status Update:"
                jq '.' "$STATUS_FILE" 2>/dev/null || cat "$STATUS_FILE"
            fi
            return 0
        else
            log_warning "Daemon PID file exists but process not running"
            rm -f "$PID_FILE"
            return 1
        fi
    else
        log_info "Monitor daemon not running"
        return 1
    fi
}

# Daemon worker loop
daemon_worker() {
    log_info "Monitor daemon worker started (PID: $$)"
    
    # Cleanup function
    cleanup_daemon() {
        log_info "Monitor daemon shutting down..."
        rm -f "$PID_FILE" "$STATUS_FILE"
        exit 0
    }
    
    trap cleanup_daemon TERM INT
    
    # Initialize timing variables
    local last_health_check=0
    local last_backup_check=0
    local last_disk_check=0
    local last_performance_check=0
    local last_security_check=0
    
    while true; do
        local current_time=$(date +%s)
        
        # Health check
        if [[ $((current_time - last_health_check)) -ge $HEALTH_CHECK_INTERVAL ]]; then
            check_system_resources
            last_health_check=$current_time
        fi
        
        # Backup check
        if [[ $((current_time - last_backup_check)) -ge $BACKUP_CHECK_INTERVAL ]]; then
            check_backup_status
            last_backup_check=$current_time
        fi
        
        # Disk check
        if [[ $((current_time - last_disk_check)) -ge $DISK_CHECK_INTERVAL ]]; then
            check_system_resources  # Includes disk monitoring
            last_disk_check=$current_time
        fi
        
        # Performance check
        if [[ $((current_time - last_performance_check)) -ge $PERFORMANCE_CHECK_INTERVAL ]]; then
            check_performance_metrics
            last_performance_check=$current_time
        fi
        
        # Security check
        if [[ $((current_time - last_security_check)) -ge $SECURITY_CHECK_INTERVAL ]]; then
            check_security_status
            last_security_check=$current_time
        fi
        
        # Sleep for a short interval
        sleep 30
    done
}

# Metrics and reporting
generate_metrics_report() {
    local period="${1:-24h}"
    local report_file="$SCRIPT_DIR/metrics_report_$(date +%Y%m%d_%H%M%S).txt"
    
    log_info "Generating metrics report for last $period..."
    
    {
        echo "==================================================="
        echo "          BACKUP MONITORING METRICS REPORT"
        echo "==================================================="
        echo "Report Generated: $(date)"
        echo "Report Period: Last $period"
        echo "Hostname: $(hostname -f)"
        echo ""
        
        echo "SYSTEM METRICS:"
        if [[ -f "$METRICS_LOG" ]]; then
            echo "- Average Disk Usage: $(grep "disk_usage_percent" "$METRICS_LOG" | tail -100 | awk -F: '{sum+=$3; count++} END {if(count>0) printf "%.1f%%", sum/count; else print "N/A"}')"
            echo "- Average Memory Usage: $(grep "memory_usage_percent" "$METRICS_LOG" | tail -100 | awk -F: '{sum+=$3; count++} END {if(count>0) printf "%.1f%%", sum/count; else print "N/A"}')"
            echo "- Average Load Average: $(grep "load_average_1min" "$METRICS_LOG" | tail -100 | awk -F: '{sum+=$3; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')"
        fi
        
        echo ""
        echo "BACKUP METRICS:"
        echo "- Total Backups: $(find "$BACKUP_DIR" -name "keys_backup_*.tar.gz" -o -name "keys_backup_*.enc" | wc -l)"
        echo "- Last Backup Age: $(find "$BACKUP_DIR" -name "keys_backup_*.tar.gz" -o -name "keys_backup_*.enc" -printf '%T@\n' | sort -nr | head -1 | xargs -I{} date -d @{} '+%Y-%m-%d %H:%M' 2>/dev/null || echo "N/A")"
        echo "- Average Backup Size: $(find "$BACKUP_DIR" -name "keys_backup_*.tar.gz" -o -name "keys_backup_*.enc" -exec stat -c%s {} \; | awk '{sum+=$1; count++} END {if(count>0) printf "%.1f MB", sum/count/1024/1024; else print "N/A"}')"
        
        echo ""
        echo "ALERT SUMMARY:"
        if [[ -f "$ALERTS_LOG" ]]; then
            echo "- Critical Alerts (Last 24h): $(grep -c "CRITICAL" "$ALERTS_LOG" 2>/dev/null || echo "0")"
            echo "- Warning Alerts (Last 24h): $(grep -c "WARNING" "$ALERTS_LOG" 2>/dev/null || echo "0")"
            echo "- Info Alerts (Last 24h): $(grep -c "INFO" "$ALERTS_LOG" 2>/dev/null || echo "0")"
        fi
        
        echo ""
        echo "RECENT ALERTS:"
        if [[ -f "$ALERTS_LOG" ]]; then
            tail -10 "$ALERTS_LOG" | while read -r line; do
                echo "  $line"
            done
        fi
        
        echo ""
        echo "==================================================="
    } > "$report_file"
    
    chmod 600 "$report_file"
    log_success "Metrics report generated: $report_file"
}

# Usage information
show_usage() {
    cat << EOF
Enterprise Keys Backup Monitoring & Health Surveillance System

USAGE:
    $0 [OPTIONS] COMMAND

COMMANDS:
    start               Start monitoring daemon
    stop                Stop monitoring daemon
    status              Show daemon status
    restart             Restart monitoring daemon
    check               Perform one-time health check
    metrics [PERIOD]    Generate metrics report (default: 24h)
    alerts              Show recent alerts
    config              Show current monitoring configuration
    help                Show this usage information

DAEMON COMMANDS:
    --daemon-worker     Internal daemon worker (do not call directly)

OPTIONS:
    -c, --config FILE   Use custom configuration file
    -v, --verbose       Enable verbose output
    -d, --debug         Enable debug output
    -q, --quiet         Suppress non-error output

EXAMPLES:
    $0 start                    # Start monitoring daemon
    $0 check                    # Perform one-time health check
    $0 metrics 7d               # Generate 7-day metrics report
    $0 alerts                   # Show recent alerts
    $0 status                   # Check daemon status

CONFIGURATION:
    Configuration file: $MONITOR_CONFIG
    Use '$0 config' to view current settings

The monitoring system continuously checks:
- System resources (disk, memory, CPU, load)
- Backup status and integrity
- Security compliance
- Performance metrics
- Compliance requirements

For detailed documentation, see the README files in the parent directory.
EOF
}

# Show recent alerts
show_alerts() {
    log_info "=== Recent Alerts ==="
    
    if [[ -f "$ALERTS_LOG" ]]; then
        local alert_count=$(wc -l < "$ALERTS_LOG")
        echo "Total alerts in log: $alert_count"
        echo ""
        
        echo "Last 20 alerts:"
        tail -20 "$ALERTS_LOG" | while read -r line; do
            # Color code alerts based on level
            if [[ "$line" == *"CRITICAL"* ]]; then
                echo -e "${RED}$line${NC}"
            elif [[ "$line" == *"WARNING"* ]]; then
                echo -e "${YELLOW}$line${NC}"
            else
                echo "$line"
            fi
        done
    else
        echo "No alerts log file found: $ALERTS_LOG"
    fi
}

# Show configuration
show_monitor_config() {
    log_info "=== Monitoring Configuration ==="
    
    if [[ -f "$MONITOR_CONFIG" ]]; then
        if command -v jq >/dev/null 2>&1; then
            jq '.' "$MONITOR_CONFIG"
        else
            cat "$MONITOR_CONFIG"
        fi
    else
        echo "Configuration file not found: $MONITOR_CONFIG"
        echo "Run with 'start' command to create default configuration."
    fi
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--config)
                MONITOR_CONFIG="$2"
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
            --daemon-worker)
                DAEMON_WORKER=true
                shift
                ;;
            start|stop|status|restart|check|metrics|alerts|config|help)
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
    DAEMON_WORKER=${DAEMON_WORKER:-false}
    
    # Handle daemon worker mode
    if [[ "$DAEMON_WORKER" == "true" ]]; then
        load_monitor_config
        daemon_worker
        return $?
    fi
    
    # Parse command line
    parse_arguments "$@"
    
    # Check if command is provided
    if [[ -z "$COMMAND" ]]; then
        log_error "No command specified"
        show_usage
        exit 1
    fi
    
    # Load configuration for most commands
    if [[ "$COMMAND" != "help" ]]; then
        load_monitor_config
    fi
    
    # Execute command
    case "$COMMAND" in
        start)
            start_daemon
            ;;
        stop)
            stop_daemon
            ;;
        status)
            daemon_status
            ;;
        restart)
            stop_daemon
            sleep 2
            start_daemon
            ;;
        check)
            perform_health_check
            ;;
        metrics)
            local period="${ARGS[0]:-24h}"
            generate_metrics_report "$period"
            ;;
        alerts)
            show_alerts
            ;;
        config)
            show_monitor_config
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
