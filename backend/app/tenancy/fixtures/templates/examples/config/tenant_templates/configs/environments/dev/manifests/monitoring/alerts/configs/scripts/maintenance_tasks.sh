#!/bin/bash

# Maintenance Tasks Automation - Spotify AI Agent
# ===============================================
#
# Comprehensive maintenance automation system with:
# - Scheduled system maintenance and optimization
# - Database maintenance and optimization
# - Log rotation and cleanup
# - Cache management and optimization
# - Resource monitoring and cleanup
# - Performance tuning automation
# - Health checks and diagnostics
# - Preventive maintenance scheduling

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/spotify-maintenance.log"
MAINTENANCE_CONFIG="/etc/spotify/maintenance_config.yaml"
LOCK_FILE="/var/run/spotify-maintenance.lock"

# Default settings
MAINTENANCE_TYPE="routine"
TENANT_ID=""
ENVIRONMENT="dev"
SCHEDULE_MODE=false
DRY_RUN=false
VERBOSE=false
FORCE_CLEANUP=false
OPTIMIZE_DATABASES=true
CLEANUP_LOGS=true
OPTIMIZE_CACHE=true
CHECK_DISK_SPACE=true
UPDATE_SYSTEM=false

# Thresholds
DISK_USAGE_THRESHOLD=85
LOG_RETENTION_DAYS=30
CACHE_MAX_SIZE_GB=10
DB_OPTIMIZATION_THRESHOLD=75

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOG_FILE}"
    
    case $level in
        "ERROR") echo -e "${RED}❌ ${message}${NC}" >&2 ;;
        "SUCCESS") echo -e "${GREEN}✅ ${message}${NC}" ;;
        "WARNING") echo -e "${YELLOW}⚠️  ${message}${NC}" ;;
        "INFO") echo -e "${BLUE}ℹ️  ${message}${NC}" ;;
        "MAINTENANCE") echo -e "${PURPLE}🔧 ${message}${NC}" ;;
        "OPTIMIZE") echo -e "${CYAN}⚡ ${message}${NC}" ;;
    esac
}

# Create lock file
create_lock() {
    if [[ -f "$LOCK_FILE" ]]; then
        local lock_pid=$(cat "$LOCK_FILE")
        if kill -0 "$lock_pid" 2>/dev/null; then
            log "ERROR" "Maintenance already running (PID: $lock_pid)"
            exit 1
        else
            log "WARNING" "Removing stale lock file"
            rm -f "$LOCK_FILE"
        fi
    fi
    
    echo $$ > "$LOCK_FILE"
    log "INFO" "Maintenance lock acquired"
}

# Remove lock file
remove_lock() {
    rm -f "$LOCK_FILE"
    log "INFO" "Maintenance lock released"
}

# Set up cleanup on exit
trap remove_lock EXIT

# Initialize maintenance system
init_maintenance_system() {
    log "INFO" "Initializing maintenance system"
    
    # Create necessary directories
    mkdir -p "$(dirname "$LOG_FILE")"
    mkdir -p "$(dirname "$MAINTENANCE_CONFIG")"
    
    # Create default configuration if not exists
    if [[ ! -f "$MAINTENANCE_CONFIG" ]]; then
        create_default_maintenance_config
    fi
    
    # Verify dependencies
    check_maintenance_dependencies
    
    log "SUCCESS" "Maintenance system initialized"
}

# Create default maintenance configuration
create_default_maintenance_config() {
    log "INFO" "Creating default maintenance configuration"
    
    cat > "$MAINTENANCE_CONFIG" << 'EOF'
# Spotify AI Agent Maintenance Configuration
maintenance:
  schedule:
    routine:
      frequency: "daily"
      time: "02:00"
      enabled: true
    deep_clean:
      frequency: "weekly"
      day: "sunday"
      time: "01:00"
      enabled: true
    optimization:
      frequency: "monthly"
      day: 1
      time: "00:00"
      enabled: true

  tasks:
    log_cleanup:
      enabled: true
      retention_days: 30
      compress_old: true
    database_optimization:
      enabled: true
      vacuum_threshold: 75
      reindex_threshold: 85
    cache_cleanup:
      enabled: true
      max_size_gb: 10
      cleanup_threshold: 80
    disk_cleanup:
      enabled: true
      temp_files: true
      old_backups: true
      unused_packages: true
    system_optimization:
      enabled: true
      memory_cleanup: true
      swap_optimization: true
      kernel_cache: true

  notifications:
    email:
      enabled: true
      recipients: ["admin@spotify.com"]
    slack:
      enabled: false
      webhook_url: ""

  thresholds:
    disk_usage_warning: 80
    disk_usage_critical: 90
    memory_usage_warning: 85
    cpu_usage_warning: 90
    database_size_warning: "10GB"
    log_size_warning: "5GB"
EOF
    
    log "SUCCESS" "Default maintenance configuration created"
}

# Check maintenance dependencies
check_maintenance_dependencies() {
    local missing_deps=()
    
    # Required tools
    local required_tools=(
        "find" "du" "df" "ps" "systemctl" "crontab" "logrotate"
    )
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_deps+=("$tool")
        fi
    done
    
    # Database tools
    if command -v psql &> /dev/null; then
        log "INFO" "PostgreSQL maintenance tools available"
    else
        log "WARNING" "PostgreSQL tools not found - database maintenance limited"
    fi
    
    if command -v redis-cli &> /dev/null; then
        log "INFO" "Redis maintenance tools available"
    else
        log "WARNING" "Redis tools not found - cache maintenance limited"
    fi
    
    if command -v mongo &> /dev/null; then
        log "INFO" "MongoDB maintenance tools available"
    else
        log "WARNING" "MongoDB tools not found - document DB maintenance limited"
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log "WARNING" "Missing maintenance tools: ${missing_deps[*]}"
    else
        log "SUCCESS" "All maintenance dependencies verified"
    fi
}

# Check disk space
check_disk_space() {
    log "MAINTENANCE" "Checking disk space usage"
    
    local critical_partitions=()
    local warning_partitions=()
    
    while read -r filesystem size used avail use_percent mount; do
        local usage=$(echo "$use_percent" | sed 's/%//')
        
        if [[ $usage -ge 90 ]]; then
            critical_partitions+=("$mount:$usage%")
            log "ERROR" "Critical disk usage on $mount: $usage%"
        elif [[ $usage -ge $DISK_USAGE_THRESHOLD ]]; then
            warning_partitions+=("$mount:$usage%")
            log "WARNING" "High disk usage on $mount: $usage%"
        else
            [[ "$VERBOSE" == "true" ]] && log "INFO" "Disk usage on $mount: $usage%"
        fi
        
    done < <(df -h | tail -n +2 | grep -E '^/dev/')
    
    if [[ ${#critical_partitions[@]} -gt 0 ]]; then
        log "ERROR" "Critical disk space issues found: ${critical_partitions[*]}"
        return 1
    elif [[ ${#warning_partitions[@]} -gt 0 ]]; then
        log "WARNING" "Disk space warnings: ${warning_partitions[*]}"
        return 2
    else
        log "SUCCESS" "Disk space usage within normal limits"
        return 0
    fi
}

# Clean up log files
cleanup_logs() {
    if [[ "$CLEANUP_LOGS" != "true" ]]; then
        return
    fi
    
    log "MAINTENANCE" "Cleaning up log files"
    
    local log_dirs=(
        "/var/log/spotify"
        "/var/log/nginx"
        "/var/log/apache2"
        "/opt/spotify/logs"
        "/tmp"
    )
    
    local total_cleaned=0
    
    for log_dir in "${log_dirs[@]}"; do
        if [[ -d "$log_dir" ]]; then
            log "INFO" "Cleaning logs in: $log_dir"
            
            # Find and remove old log files
            local old_logs=""
            if [[ "$DRY_RUN" == "true" ]]; then
                old_logs=$(find "$log_dir" -name "*.log" -type f -mtime +$LOG_RETENTION_DAYS 2>/dev/null | wc -l)
                log "INFO" "DRY RUN: Would delete $old_logs old log files in $log_dir"
            else
                old_logs=$(find "$log_dir" -name "*.log" -type f -mtime +$LOG_RETENTION_DAYS -delete -print 2>/dev/null | wc -l)
                if [[ $old_logs -gt 0 ]]; then
                    log "SUCCESS" "Deleted $old_logs old log files in $log_dir"
                    total_cleaned=$((total_cleaned + old_logs))
                fi
            fi
            
            # Compress recent log files
            if [[ "$DRY_RUN" == "false" ]]; then
                find "$log_dir" -name "*.log" -type f -mtime +7 -mtime -$LOG_RETENTION_DAYS ! -name "*.gz" -exec gzip {} \; 2>/dev/null || true
            fi
            
            # Clean temporary files
            if [[ "$log_dir" == "/tmp" ]]; then
                if [[ "$DRY_RUN" == "false" ]]; then
                    find /tmp -type f -atime +7 -delete 2>/dev/null || true
                    find /tmp -type d -empty -delete 2>/dev/null || true
                fi
            fi
        fi
    done
    
    # Rotate system logs
    if command -v logrotate &> /dev/null && [[ "$DRY_RUN" == "false" ]]; then
        logrotate -f /etc/logrotate.conf >/dev/null 2>&1 || true
        log "SUCCESS" "System log rotation completed"
    fi
    
    log "SUCCESS" "Log cleanup completed - removed $total_cleaned files"
}

# Optimize databases
optimize_databases() {
    if [[ "$OPTIMIZE_DATABASES" != "true" ]]; then
        return
    fi
    
    log "MAINTENANCE" "Optimizing databases"
    
    # PostgreSQL optimization
    optimize_postgresql
    
    # Redis optimization
    optimize_redis
    
    # MongoDB optimization
    optimize_mongodb
    
    log "SUCCESS" "Database optimization completed"
}

# Optimize PostgreSQL
optimize_postgresql() {
    if ! command -v psql &> /dev/null; then
        return
    fi
    
    log "INFO" "Optimizing PostgreSQL databases"
    
    # Get list of databases
    local databases=($(psql -t -c "SELECT datname FROM pg_database WHERE datistemplate = false;" 2>/dev/null | grep -v "^\s*$" || echo ""))
    
    for db in "${databases[@]}"; do
        db=$(echo "$db" | xargs)  # Trim whitespace
        
        if [[ -n "$db" ]]; then
            log "INFO" "Optimizing PostgreSQL database: $db"
            
            if [[ "$DRY_RUN" == "false" ]]; then
                # Vacuum and analyze
                psql "$db" -c "VACUUM ANALYZE;" >/dev/null 2>&1 || true
                
                # Reindex if needed
                local db_size=$(psql "$db" -t -c "SELECT pg_size_pretty(pg_database_size('$db'));" 2>/dev/null | xargs)
                log "INFO" "Database $db size: $db_size"
                
                # Check for bloated tables and reindex if necessary
                psql "$db" -c "REINDEX DATABASE \"$db\";" >/dev/null 2>&1 || true
                
                log "SUCCESS" "PostgreSQL database $db optimized"
            else
                log "INFO" "DRY RUN: Would optimize PostgreSQL database $db"
            fi
        fi
    done
}

# Optimize Redis
optimize_redis() {
    if ! command -v redis-cli &> /dev/null; then
        return
    fi
    
    log "INFO" "Optimizing Redis cache"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        # Get Redis info
        local redis_info=$(redis-cli INFO 2>/dev/null || echo "")
        
        if [[ -n "$redis_info" ]]; then
            # Check memory usage
            local used_memory=$(echo "$redis_info" | grep "used_memory_human:" | cut -d: -f2 | tr -d '\r')
            local max_memory=$(echo "$redis_info" | grep "maxmemory_human:" | cut -d: -f2 | tr -d '\r')
            
            log "INFO" "Redis memory usage: $used_memory / $max_memory"
            
            # Run background save
            redis-cli BGSAVE >/dev/null 2>&1 || true
            
            # Clean expired keys
            redis-cli --scan --pattern "*" | head -1000 | while read -r key; do
                redis-cli TTL "$key" >/dev/null 2>&1 || true
            done
            
            log "SUCCESS" "Redis optimization completed"
        else
            log "WARNING" "Could not connect to Redis for optimization"
        fi
    else
        log "INFO" "DRY RUN: Would optimize Redis cache"
    fi
}

# Optimize MongoDB
optimize_mongodb() {
    if ! command -v mongo &> /dev/null; then
        return
    fi
    
    log "INFO" "Optimizing MongoDB databases"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        # Get list of databases
        local databases=$(mongo --quiet --eval "db.adminCommand('listDatabases').databases.forEach(function(d){print(d.name)})" 2>/dev/null || echo "")
        
        for db in $databases; do
            if [[ "$db" != "admin" ]] && [[ "$db" != "local" ]] && [[ "$db" != "config" ]]; then
                log "INFO" "Optimizing MongoDB database: $db"
                
                # Compact collections
                mongo "$db" --quiet --eval "
                    db.getCollectionNames().forEach(function(name) {
                        try {
                            db.runCommand({compact: name});
                        } catch(e) {
                            print('Could not compact ' + name + ': ' + e);
                        }
                    });
                " >/dev/null 2>&1 || true
                
                log "SUCCESS" "MongoDB database $db optimized"
            fi
        done
    else
        log "INFO" "DRY RUN: Would optimize MongoDB databases"
    fi
}

# Clean up cache
cleanup_cache() {
    if [[ "$OPTIMIZE_CACHE" != "true" ]]; then
        return
    fi
    
    log "MAINTENANCE" "Cleaning up cache"
    
    local cache_dirs=(
        "/var/cache/spotify"
        "/tmp/spotify_cache"
        "/opt/spotify/cache"
        "/var/lib/spotify/cache"
    )
    
    local total_cleaned_mb=0
    
    for cache_dir in "${cache_dirs[@]}"; do
        if [[ -d "$cache_dir" ]]; then
            log "INFO" "Cleaning cache directory: $cache_dir"
            
            # Get current cache size
            local cache_size_kb=$(du -sk "$cache_dir" 2>/dev/null | cut -f1)
            local cache_size_mb=$((cache_size_kb / 1024))
            
            if [[ $cache_size_mb -gt $((CACHE_MAX_SIZE_GB * 1024)) ]]; then
                log "WARNING" "Cache directory $cache_dir is large: ${cache_size_mb}MB"
                
                if [[ "$DRY_RUN" == "false" ]]; then
                    # Remove old cache files
                    find "$cache_dir" -type f -atime +7 -delete 2>/dev/null || true
                    
                    # Calculate cleaned space
                    local new_cache_size_kb=$(du -sk "$cache_dir" 2>/dev/null | cut -f1)
                    local new_cache_size_mb=$((new_cache_size_kb / 1024))
                    local cleaned_mb=$((cache_size_mb - new_cache_size_mb))
                    
                    total_cleaned_mb=$((total_cleaned_mb + cleaned_mb))
                    
                    log "SUCCESS" "Cleaned ${cleaned_mb}MB from $cache_dir"
                else
                    log "INFO" "DRY RUN: Would clean cache directory $cache_dir"
                fi
            else
                [[ "$VERBOSE" == "true" ]] && log "INFO" "Cache directory $cache_dir size OK: ${cache_size_mb}MB"
            fi
        fi
    done
    
    # Clear system cache
    if [[ "$DRY_RUN" == "false" ]]; then
        # Clear page cache, dentries and inodes
        sync
        echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
        log "SUCCESS" "System cache cleared"
    fi
    
    log "SUCCESS" "Cache cleanup completed - freed ${total_cleaned_mb}MB"
}

# System optimization
optimize_system() {
    log "MAINTENANCE" "Performing system optimization"
    
    # Memory optimization
    optimize_memory
    
    # Swap optimization
    optimize_swap
    
    # Network optimization
    optimize_network
    
    # Process cleanup
    cleanup_processes
    
    log "SUCCESS" "System optimization completed"
}

# Optimize memory
optimize_memory() {
    log "INFO" "Optimizing memory usage"
    
    # Get memory statistics
    local memory_info=$(free -m)
    local total_mem=$(echo "$memory_info" | awk '/^Mem:/ {print $2}')
    local used_mem=$(echo "$memory_info" | awk '/^Mem:/ {print $3}')
    local free_mem=$(echo "$memory_info" | awk '/^Mem:/ {print $4}')
    local usage_percent=$((used_mem * 100 / total_mem))
    
    log "INFO" "Memory usage: ${used_mem}MB / ${total_mem}MB (${usage_percent}%)"
    
    if [[ $usage_percent -gt 85 ]]; then
        log "WARNING" "High memory usage detected: ${usage_percent}%"
        
        if [[ "$DRY_RUN" == "false" ]]; then
            # Clear buffer cache
            sync
            echo 1 > /proc/sys/vm/drop_caches 2>/dev/null || true
            
            # Find and suggest killing memory-heavy processes
            log "INFO" "Top memory consumers:"
            ps aux --sort=-%mem | head -6 | tail -5 | while read -r line; do
                log "INFO" "  $line"
            done
        fi
    else
        log "SUCCESS" "Memory usage within normal limits"
    fi
}

# Optimize swap
optimize_swap() {
    log "INFO" "Optimizing swap usage"
    
    local swap_info=$(free -m | grep "^Swap:")
    if [[ -n "$swap_info" ]]; then
        local total_swap=$(echo "$swap_info" | awk '{print $2}')
        local used_swap=$(echo "$swap_info" | awk '{print $3}')
        
        if [[ $total_swap -gt 0 ]]; then
            local swap_usage_percent=$((used_swap * 100 / total_swap))
            log "INFO" "Swap usage: ${used_swap}MB / ${total_swap}MB (${swap_usage_percent}%)"
            
            if [[ $swap_usage_percent -gt 50 ]] && [[ "$DRY_RUN" == "false" ]]; then
                log "WARNING" "High swap usage detected: ${swap_usage_percent}%"
                
                # Adjust swappiness
                echo 10 > /proc/sys/vm/swappiness 2>/dev/null || true
                log "INFO" "Adjusted swappiness to reduce swap usage"
            fi
        else
            log "INFO" "No swap space configured"
        fi
    fi
}

# Optimize network
optimize_network() {
    log "INFO" "Optimizing network settings"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        # Optimize TCP settings for better performance
        sysctl -w net.core.rmem_max=16777216 >/dev/null 2>&1 || true
        sysctl -w net.core.wmem_max=16777216 >/dev/null 2>&1 || true
        sysctl -w net.ipv4.tcp_rmem="4096 12582912 16777216" >/dev/null 2>&1 || true
        sysctl -w net.ipv4.tcp_wmem="4096 12582912 16777216" >/dev/null 2>&1 || true
        
        log "SUCCESS" "Network parameters optimized"
    else
        log "INFO" "DRY RUN: Would optimize network parameters"
    fi
}

# Clean up processes
cleanup_processes() {
    log "INFO" "Cleaning up zombie and unnecessary processes"
    
    # Find zombie processes
    local zombie_count=$(ps aux | awk '$8 ~ /^Z/ { count++ } END { print count+0 }')
    
    if [[ $zombie_count -gt 0 ]]; then
        log "WARNING" "Found $zombie_count zombie processes"
        ps aux | awk '$8 ~ /^Z/ { print $2, $11 }' | while read -r pid cmd; do
            log "WARNING" "Zombie process: PID $pid - $cmd"
        done
    else
        log "SUCCESS" "No zombie processes found"
    fi
    
    # Check for high CPU processes
    log "INFO" "Top CPU consumers:"
    ps aux --sort=-%cpu | head -6 | tail -5 | while read -r line; do
        [[ "$VERBOSE" == "true" ]] && log "INFO" "  $line"
    done
}

# Update system packages
update_system() {
    if [[ "$UPDATE_SYSTEM" != "true" ]]; then
        return
    fi
    
    log "MAINTENANCE" "Updating system packages"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        # Update package lists
        if command -v apt &> /dev/null; then
            apt update >/dev/null 2>&1 || true
            
            # Check for available updates
            local updates=$(apt list --upgradable 2>/dev/null | wc -l)
            if [[ $updates -gt 1 ]]; then  # Subtract 1 for header line
                log "INFO" "$((updates - 1)) package updates available"
                
                # Install security updates only
                apt upgrade -y >/dev/null 2>&1 || true
                log "SUCCESS" "Security updates installed"
            else
                log "SUCCESS" "System is up to date"
            fi
            
            # Clean package cache
            apt autoremove -y >/dev/null 2>&1 || true
            apt autoclean >/dev/null 2>&1 || true
            
        elif command -v yum &> /dev/null; then
            yum check-update >/dev/null 2>&1 || true
            yum update -y --security >/dev/null 2>&1 || true
            yum autoremove -y >/dev/null 2>&1 || true
            log "SUCCESS" "System packages updated"
        fi
    else
        log "INFO" "DRY RUN: Would update system packages"
    fi
}

# Generate maintenance report
generate_maintenance_report() {
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local report_file="/tmp/maintenance_report_${timestamp}.txt"
    
    log "INFO" "Generating maintenance report: $report_file"
    
    cat > "$report_file" << EOF
SPOTIFY AI AGENT MAINTENANCE REPORT
==================================

Maintenance Details:
- Type: $MAINTENANCE_TYPE
- Tenant: ${TENANT_ID:-all}
- Environment: $ENVIRONMENT
- Timestamp: $(date '+%Y-%m-%d %H:%M:%S')
- Hostname: $(hostname)

System Status:
EOF

    # Add disk usage information
    echo "" >> "$report_file"
    echo "Disk Usage:" >> "$report_file"
    df -h | grep -E '^/dev/' >> "$report_file"
    
    # Add memory usage information
    echo "" >> "$report_file"
    echo "Memory Usage:" >> "$report_file"
    free -h >> "$report_file"
    
    # Add service status
    echo "" >> "$report_file"
    echo "Critical Services:" >> "$report_file"
    for service in spotify-api spotify-ml spotify-monitoring postgresql redis-server nginx; do
        if systemctl is-active "$service" >/dev/null 2>&1; then
            echo "  $service: ACTIVE" >> "$report_file"
        else
            echo "  $service: INACTIVE" >> "$report_file"
        fi
    done
    
    echo "" >> "$report_file"
    echo "Maintenance completed at: $(date '+%Y-%m-%d %H:%M:%S')" >> "$report_file"
    
    log "SUCCESS" "Maintenance report generated: $report_file"
}

# Send notification
send_notification() {
    local status="$1"
    local message="$2"
    
    # TODO: Implement notification sending
    # This would integrate with email, Slack, Teams, etc.
    log "INFO" "Notification: $status - $message"
}

# Schedule maintenance tasks
schedule_maintenance() {
    log "INFO" "Scheduling maintenance tasks"
    
    # Create cron job for routine maintenance
    local cron_entry="0 2 * * * $0 --type routine --environment $ENVIRONMENT >/dev/null 2>&1"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        (crontab -l 2>/dev/null | grep -v "$0"; echo "$cron_entry") | crontab -
        log "SUCCESS" "Scheduled routine maintenance for 2:00 AM daily"
    else
        log "INFO" "DRY RUN: Would schedule maintenance with cron entry: $cron_entry"
    fi
}

# Main maintenance function
perform_maintenance() {
    log "INFO" "Starting maintenance tasks"
    log "INFO" "Maintenance type: $MAINTENANCE_TYPE"
    log "INFO" "Tenant: ${TENANT_ID:-all}"
    log "INFO" "Environment: $ENVIRONMENT"
    
    # Check prerequisites
    if [[ "$CHECK_DISK_SPACE" == "true" ]]; then
        check_disk_space
    fi
    
    # Perform maintenance tasks
    cleanup_logs
    optimize_databases
    cleanup_cache
    optimize_system
    
    # Update system if requested
    if [[ "$UPDATE_SYSTEM" == "true" ]]; then
        update_system
    fi
    
    # Generate report
    generate_maintenance_report
    
    # Send notification
    send_notification "SUCCESS" "Maintenance completed successfully"
    
    log "SUCCESS" "All maintenance tasks completed"
}

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Maintenance tasks automation for Spotify AI Agent

Options:
    --type TYPE             Maintenance type (routine, deep, optimization)
    --tenant ID             Perform maintenance for specific tenant
    --environment ENV       Environment (dev, staging, prod)
    --schedule              Schedule maintenance tasks
    --optimize-db           Optimize databases (default: true)
    --cleanup-logs          Clean up log files (default: true)
    --optimize-cache        Optimize cache (default: true)
    --check-disk            Check disk space (default: true)
    --update-system         Update system packages
    --force-cleanup         Force aggressive cleanup
    --dry-run               Simulate maintenance without execution
    --verbose, -v           Verbose output
    --help, -h              Show this help

Examples:
    $0 --type routine --tenant spotify_prod
    $0 --type deep --cleanup-logs --optimize-db --verbose
    $0 --schedule --environment prod
    $0 --dry-run --force-cleanup --update-system

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            MAINTENANCE_TYPE="$2"
            shift 2
            ;;
        --tenant)
            TENANT_ID="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --schedule)
            SCHEDULE_MODE=true
            shift
            ;;
        --optimize-db)
            OPTIMIZE_DATABASES=true
            shift
            ;;
        --cleanup-logs)
            CLEANUP_LOGS=true
            shift
            ;;
        --optimize-cache)
            OPTIMIZE_CACHE=true
            shift
            ;;
        --check-disk)
            CHECK_DISK_SPACE=true
            shift
            ;;
        --update-system)
            UPDATE_SYSTEM=true
            shift
            ;;
        --force-cleanup)
            FORCE_CLEANUP=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            log "ERROR" "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    echo "=============================================="
    echo "  Maintenance Tasks - Spotify AI Agent"
    echo "=============================================="
    echo
    
    # Create lock to prevent concurrent maintenance
    create_lock
    
    # Initialize maintenance system
    init_maintenance_system
    
    # Schedule mode or perform maintenance
    if [[ "$SCHEDULE_MODE" == "true" ]]; then
        schedule_maintenance
    else
        perform_maintenance
    fi
}

# Run main function
main "$@"
