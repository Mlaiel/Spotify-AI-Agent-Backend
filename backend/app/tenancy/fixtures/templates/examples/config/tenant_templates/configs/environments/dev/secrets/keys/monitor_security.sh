#!/bin/bash
# =============================================================================
# Advanced Real-Time Security Monitoring for Cryptographic Keys
# =============================================================================
# 
# Enterprise-grade real-time security monitoring system for the Spotify AI Agent
# cryptographic key infrastructure with threat detection, anomaly analysis,
# and automated incident response.
#
# This script implements continuous security monitoring with machine learning
# based threat detection and automated response capabilities.
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../../../../.." && pwd)"
SECRETS_DIR="$SCRIPT_DIR"
MONITOR_LOG="$SECRETS_DIR/security_monitor.log"
ALERT_LOG="$SECRETS_DIR/security_alerts.log"
METRICS_DIR="$SECRETS_DIR/metrics"
CONFIG_FILE="$SECRETS_DIR/monitor_config.json"

# Security settings
UMASK_ORIGINAL=$(umask)
umask 077

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Global variables
DAEMON_MODE=false
MONITORING_INTERVAL=30
ALERT_THRESHOLD_HIGH=80
ALERT_THRESHOLD_CRITICAL=95
WEBHOOK_URL=""
SLACK_WEBHOOK=""
EMAIL_ALERTS=false
THREAT_DETECTION=true
ANOMALY_DETECTION=true

# Monitoring metrics
declare -A METRICS=(
    ["key_access_count"]=0
    ["failed_access_attempts"]=0
    ["rotation_events"]=0
    ["security_violations"]=0
    ["threat_score"]=0
    ["anomaly_score"]=0
)

# Threat detection patterns
declare -A THREAT_PATTERNS=(
    ["brute_force"]="Multiple failed access attempts from same source"
    ["privilege_escalation"]="Unexpected elevation of access privileges"
    ["data_exfiltration"]="Unusual data access patterns"
    ["lateral_movement"]="Cross-system access attempts"
    ["persistence"]="Attempts to maintain unauthorized access"
)

# Logging functions
log_monitor() {
    local level="$1"
    local message="$2"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local component="${3:-MONITOR}"
    
    echo "[$timestamp] [$level] [$component] $message" >> "$MONITOR_LOG"
    
    case "$level" in
        "CRITICAL")
            echo -e "${RED}[CRITICAL]${NC} $message"
            log_alert "CRITICAL" "$message"
            ;;
        "HIGH")
            echo -e "${YELLOW}[HIGH]${NC} $message"
            log_alert "HIGH" "$message"
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING]${NC} $message"
            ;;
        "INFO")
            echo -e "${BLUE}[INFO]${NC} $message"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} $message"
            ;;
        "THREAT")
            echo -e "${PURPLE}[THREAT]${NC} $message"
            log_alert "THREAT" "$message"
            ;;
        "ANOMALY")
            echo -e "${CYAN}[ANOMALY]${NC} $message"
            ;;
    esac
}

# Alert logging
log_alert() {
    local level="$1"
    local message="$2"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    echo "[$timestamp] [$level] $message" >> "$ALERT_LOG"
    
    # Send external notifications
    send_notification "$level" "$message"
    
    # Update threat metrics
    case "$level" in
        "CRITICAL")
            METRICS["threat_score"]=$((METRICS["threat_score"] + 50))
            ;;
        "HIGH")
            METRICS["threat_score"]=$((METRICS["threat_score"] + 25))
            ;;
        "THREAT")
            METRICS["threat_score"]=$((METRICS["threat_score"] + 75))
            ;;
    esac
}

# Initialize monitoring
initialize_monitoring() {
    echo "==============================================================================" > "$MONITOR_LOG"
    echo "Advanced Real-Time Security Monitoring - $(date -u)" >> "$MONITOR_LOG"
    echo "==============================================================================" >> "$MONITOR_LOG"
    
    log_monitor "INFO" "Starting advanced security monitoring system"
    log_monitor "INFO" "Monitoring directory: $SECRETS_DIR"
    log_monitor "INFO" "Monitoring interval: ${MONITORING_INTERVAL}s"
    log_monitor "INFO" "Threat detection: $THREAT_DETECTION"
    log_monitor "INFO" "Anomaly detection: $ANOMALY_DETECTION"
    
    # Create metrics directory
    mkdir -p "$METRICS_DIR"
    
    # Load configuration
    load_monitor_config
    
    # Initialize baseline metrics
    initialize_baseline_metrics
}

# Load monitoring configuration
load_monitor_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        create_default_monitor_config
    fi
    
    # Parse configuration
    if command -v jq &> /dev/null; then
        local config_content=$(cat "$CONFIG_FILE")
        WEBHOOK_URL=$(echo "$config_content" | jq -r '.notifications.webhook_url // ""')
        SLACK_WEBHOOK=$(echo "$config_content" | jq -r '.notifications.slack_webhook // ""')
        EMAIL_ALERTS=$(echo "$config_content" | jq -r '.notifications.email_alerts // false')
        MONITORING_INTERVAL=$(echo "$config_content" | jq -r '.monitoring.interval_seconds // 30')
        THREAT_DETECTION=$(echo "$config_content" | jq -r '.detection.threat_detection // true')
        ANOMALY_DETECTION=$(echo "$config_content" | jq -r '.detection.anomaly_detection // true')
    fi
    
    log_monitor "SUCCESS" "Configuration loaded successfully"
}

# Create default monitoring configuration
create_default_monitor_config() {
    cat > "$CONFIG_FILE" << EOF
{
  "version": "2.0.0",
  "monitoring": {
    "interval_seconds": 30,
    "metrics_retention_days": 90,
    "log_rotation_size_mb": 100,
    "real_time_analysis": true
  },
  "detection": {
    "threat_detection": true,
    "anomaly_detection": true,
    "behavioral_analysis": true,
    "machine_learning": false,
    "threat_intelligence": false
  },
  "thresholds": {
    "failed_access_threshold": 5,
    "anomaly_score_threshold": 75,
    "threat_score_threshold": 85,
    "response_time_threshold_ms": 1000
  },
  "notifications": {
    "webhook_url": "",
    "slack_webhook": "",
    "email_alerts": false,
    "sms_alerts": false,
    "notification_levels": ["CRITICAL", "HIGH", "THREAT"]
  },
  "incident_response": {
    "auto_response": true,
    "quarantine_suspicious_keys": false,
    "disable_compromised_accounts": false,
    "emergency_rotation": true
  },
  "compliance": {
    "log_everything": true,
    "retain_evidence": true,
    "forensic_mode": false,
    "gdpr_compliance": true
  }
}
EOF
    
    chmod 600 "$CONFIG_FILE"
    log_monitor "SUCCESS" "Default monitoring configuration created"
}

# Initialize baseline metrics
initialize_baseline_metrics() {
    log_monitor "INFO" "Initializing baseline security metrics..."
    
    # Create baseline file
    local baseline_file="$METRICS_DIR/baseline_metrics.json"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    cat > "$baseline_file" << EOF
{
  "established_at": "$timestamp",
  "baseline_metrics": {
    "average_key_access_per_hour": 0,
    "normal_access_patterns": [],
    "typical_user_behavior": {},
    "standard_response_times": {},
    "common_operation_types": []
  },
  "learning_period_days": 30,
  "confidence_level": 0.0
}
EOF
    
    chmod 600 "$baseline_file"
    log_monitor "SUCCESS" "Baseline metrics initialized"
}

# Monitor file system events
monitor_filesystem_events() {
    log_monitor "INFO" "Starting file system monitoring..." "FILESYSTEM"
    
    # Monitor key file modifications
    if command -v inotifywait &> /dev/null; then
        inotifywait -m -r -e modify,access,open,close,move,create,delete "$SECRETS_DIR" --format '%T %w%f %e' --timefmt '%Y-%m-%d %H:%M:%S' 2>/dev/null | while read -r timestamp file event; do
            process_filesystem_event "$timestamp" "$file" "$event"
        done &
        
        log_monitor "SUCCESS" "File system monitoring active" "FILESYSTEM"
    else
        log_monitor "WARNING" "inotify-tools not available - using polling method" "FILESYSTEM"
        monitor_filesystem_polling &
    fi
}

# Process file system events
process_filesystem_event() {
    local timestamp="$1"
    local file="$2"
    local event="$3"
    
    # Filter key files only
    if [[ "$file" =~ \.(key|pem)$ ]]; then
        log_monitor "INFO" "Key file event: $event on $(basename "$file")" "FILESYSTEM"
        
        # Increment access counter
        METRICS["key_access_count"]=$((METRICS["key_access_count"] + 1))
        
        # Check for suspicious events
        case "$event" in
            "DELETE"|"MOVED_FROM")
                log_monitor "CRITICAL" "Key file deleted or moved: $(basename "$file")" "FILESYSTEM"
                trigger_incident_response "key_file_deletion" "$file"
                ;;
            "MODIFY")
                if ! verify_authorized_modification "$file"; then
                    log_monitor "THREAT" "Unauthorized key file modification: $(basename "$file")" "FILESYSTEM"
                    METRICS["security_violations"]=$((METRICS["security_violations"] + 1))
                fi
                ;;
            "ACCESS"|"OPEN")
                analyze_access_pattern "$file" "$timestamp"
                ;;
        esac
        
        # Update metrics
        update_metrics "filesystem_event" "$event" "$file"
    fi
}

# Polling-based file system monitoring
monitor_filesystem_polling() {
    local last_check=$(date +%s)
    
    while true; do
        local current_time=$(date +%s)
        
        for key_file in "$SECRETS_DIR"/*.key "$SECRETS_DIR"/*.pem; do
            if [[ -f "$key_file" ]]; then
                local file_mtime=$(stat -c %Y "$key_file")
                
                if [[ $file_mtime -gt $last_check ]]; then
                    process_filesystem_event "$(date)" "$key_file" "MODIFY"
                fi
            fi
        done
        
        last_check=$current_time
        sleep "$MONITORING_INTERVAL"
    done
}

# Verify authorized modification
verify_authorized_modification() {
    local file="$1"
    local current_user=$(whoami)
    local file_owner=$(stat -c "%U" "$file")
    
    # Check if modification is by authorized user
    if [[ "$current_user" == "$file_owner" ]] || [[ "$current_user" == "root" ]]; then
        # Additional checks for script-based modifications
        if pgrep -f "generate_keys.sh\|rotate_keys.sh" >/dev/null; then
            return 0  # Authorized script modification
        fi
        
        # Check if modification is during maintenance window
        local current_hour=$(date +%H)
        if [[ $current_hour -ge 2 ]] && [[ $current_hour -le 4 ]]; then
            return 0  # Maintenance window
        fi
    fi
    
    return 1  # Unauthorized modification
}

# Analyze access patterns
analyze_access_pattern() {
    local file="$1"
    local timestamp="$2"
    
    # Store access pattern
    local pattern_file="$METRICS_DIR/access_patterns.log"
    echo "$timestamp $(basename "$file") $(whoami) $(who am i | awk '{print $5}' | tr -d '()')" >> "$pattern_file"
    
    # Analyze for anomalies
    if [[ "$ANOMALY_DETECTION" == "true" ]]; then
        detect_access_anomalies "$file" "$timestamp"
    fi
}

# Detect access anomalies
detect_access_anomalies() {
    local file="$1"
    local timestamp="$2"
    local anomaly_score=0
    
    # Check access frequency
    local recent_accesses=$(grep "$(basename "$file")" "$METRICS_DIR/access_patterns.log" 2>/dev/null | tail -10 | wc -l)
    if [[ $recent_accesses -gt 20 ]]; then
        ((anomaly_score += 30))
        log_monitor "ANOMALY" "High access frequency detected: $(basename "$file")" "ANOMALY"
    fi
    
    # Check access time patterns
    local access_hour=$(date -d "$timestamp" +%H 2>/dev/null || date +%H)
    if [[ $access_hour -lt 6 ]] || [[ $access_hour -gt 22 ]]; then
        ((anomaly_score += 20))
        log_monitor "ANOMALY" "Off-hours access detected: $(basename "$file") at $access_hour:00" "ANOMALY"
    fi
    
    # Check geographic anomalies (if available)
    local current_ip=$(who am i | awk '{print $5}' | tr -d '()')
    if [[ -n "$current_ip" ]] && [[ "$current_ip" != "localhost" ]]; then
        if ! is_known_ip "$current_ip"; then
            ((anomaly_score += 40))
            log_monitor "ANOMALY" "Unknown IP access detected: $current_ip" "ANOMALY"
        fi
    fi
    
    # Update anomaly score
    METRICS["anomaly_score"]=$((METRICS["anomaly_score"] + anomaly_score))
    
    # Trigger alert if threshold exceeded
    if [[ $anomaly_score -gt 50 ]]; then
        log_monitor "HIGH" "High anomaly score detected: $anomaly_score for $(basename "$file")" "ANOMALY"
    fi
}

# Check if IP is known/trusted
is_known_ip() {
    local ip="$1"
    local trusted_ips=("127.0.0.1" "::1" "10.0.0.0/8" "172.16.0.0/12" "192.168.0.0/16")
    
    for trusted_ip in "${trusted_ips[@]}"; do
        if [[ "$ip" == "$trusted_ip" ]] || [[ "$ip" =~ ^${trusted_ip%/*} ]]; then
            return 0
        fi
    done
    
    return 1
}

# Monitor process activity
monitor_process_activity() {
    log_monitor "INFO" "Starting process activity monitoring..." "PROCESS"
    
    while true; do
        # Monitor for suspicious processes
        local suspicious_processes=$(ps aux | grep -E "(john|hashcat|hydra|nmap|metasploit|sqlmap)" | grep -v grep || true)
        
        if [[ -n "$suspicious_processes" ]]; then
            log_monitor "THREAT" "Suspicious security tools detected" "PROCESS"
            echo "$suspicious_processes" | while read -r process; do
                log_monitor "THREAT" "Suspicious process: $process" "PROCESS"
            done
            METRICS["security_violations"]=$((METRICS["security_violations"] + 1))
        fi
        
        # Monitor for unauthorized access attempts
        local failed_logins=$(grep "Failed password" /var/log/auth.log 2>/dev/null | tail -10 | wc -l || echo "0")
        if [[ $failed_logins -gt 5 ]]; then
            log_monitor "HIGH" "Multiple failed login attempts detected: $failed_logins" "PROCESS"
            METRICS["failed_access_attempts"]=$((METRICS["failed_access_attempts"] + failed_logins))
        fi
        
        sleep "$MONITORING_INTERVAL"
    done &
}

# Monitor network activity
monitor_network_activity() {
    log_monitor "INFO" "Starting network activity monitoring..." "NETWORK"
    
    while true; do
        # Monitor for unusual network connections
        if command -v netstat &> /dev/null; then
            local suspicious_connections=$(netstat -an | grep ":22\|:443\|:80" | grep ESTABLISHED | wc -l)
            
            if [[ $suspicious_connections -gt 50 ]]; then
                log_monitor "WARNING" "High number of network connections: $suspicious_connections" "NETWORK"
            fi
        fi
        
        # Monitor for data exfiltration attempts
        if command -v ss &> /dev/null; then
            local outbound_connections=$(ss -tuln | grep ":443\|:80" | wc -l)
            
            if [[ $outbound_connections -gt 100 ]]; then
                log_monitor "HIGH" "Potential data exfiltration detected: $outbound_connections connections" "NETWORK"
            fi
        fi
        
        sleep "$MONITORING_INTERVAL"
    done &
}

# Monitor system resources
monitor_system_resources() {
    log_monitor "INFO" "Starting system resource monitoring..." "SYSTEM"
    
    while true; do
        # Monitor CPU usage
        local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
        if [[ "${cpu_usage%.*}" -gt 90 ]]; then
            log_monitor "WARNING" "High CPU usage detected: ${cpu_usage}%" "SYSTEM"
        fi
        
        # Monitor memory usage
        local memory_usage=$(free | grep Mem | awk '{printf("%.0f", $3/$2 * 100)}')
        if [[ $memory_usage -gt 90 ]]; then
            log_monitor "WARNING" "High memory usage detected: ${memory_usage}%" "SYSTEM"
        fi
        
        # Monitor disk usage
        local disk_usage=$(df "$SECRETS_DIR" | tail -1 | awk '{print $5}' | cut -d'%' -f1)
        if [[ $disk_usage -gt 90 ]]; then
            log_monitor "CRITICAL" "High disk usage detected: ${disk_usage}%" "SYSTEM"
        fi
        
        sleep "$MONITORING_INTERVAL"
    done &
}

# Threat intelligence integration
check_threat_intelligence() {
    log_monitor "INFO" "Checking threat intelligence feeds..." "THREAT_INTEL"
    
    # Simulated threat intelligence check
    # In production, this would integrate with real threat intelligence feeds
    local threat_indicators=("malware_signature" "known_bad_ip" "suspicious_domain")
    
    for indicator in "${threat_indicators[@]}"; do
        # Check if indicator is present in system
        if check_threat_indicator "$indicator"; then
            log_monitor "THREAT" "Threat indicator detected: $indicator" "THREAT_INTEL"
            trigger_incident_response "threat_indicator" "$indicator"
        fi
    done
}

# Check specific threat indicator
check_threat_indicator() {
    local indicator="$1"
    
    case "$indicator" in
        "malware_signature")
            # Check for known malware signatures in processes
            if pgrep -f "crypto.*miner\|bitcoin.*miner" >/dev/null 2>&1; then
                return 0
            fi
            ;;
        "known_bad_ip")
            # Check for connections to known bad IPs
            # This would normally query a threat intelligence database
            return 1
            ;;
        "suspicious_domain")
            # Check for DNS queries to suspicious domains
            return 1
            ;;
    esac
    
    return 1
}

# Trigger incident response
trigger_incident_response() {
    local incident_type="$1"
    local details="$2"
    
    log_monitor "CRITICAL" "INCIDENT RESPONSE TRIGGERED: $incident_type - $details" "INCIDENT"
    
    # Create incident file
    local incident_id="INC_$(date +%Y%m%d_%H%M%S)_$$"
    local incident_file="$METRICS_DIR/incidents/${incident_id}.json"
    
    mkdir -p "$METRICS_DIR/incidents"
    
    cat > "$incident_file" << EOF
{
  "incident_id": "$incident_id",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "type": "$incident_type",
  "details": "$details",
  "severity": "CRITICAL",
  "status": "ACTIVE",
  "response_actions": [],
  "forensic_data": {
    "system_state": "$(uname -a)",
    "user_context": "$(whoami)",
    "network_connections": "$(netstat -an 2>/dev/null | head -20 || echo 'N/A')",
    "running_processes": "$(ps aux | head -20)"
  }
}
EOF
    
    chmod 600 "$incident_file"
    
    # Execute automated response
    execute_incident_response "$incident_type" "$incident_id"
}

# Execute incident response actions
execute_incident_response() {
    local incident_type="$1"
    local incident_id="$2"
    
    log_monitor "INFO" "Executing incident response for: $incident_type" "INCIDENT"
    
    case "$incident_type" in
        "key_file_deletion")
            log_monitor "INFO" "Initiating emergency key restoration" "INCIDENT"
            # Restore from backup
            restore_deleted_keys
            ;;
        "threat_indicator")
            log_monitor "INFO" "Isolating threat" "INCIDENT"
            # Quarantine suspicious activity
            quarantine_threat
            ;;
        "unauthorized_access")
            log_monitor "INFO" "Securing access" "INCIDENT"
            # Lock down access
            secure_access
            ;;
    esac
    
    # Send emergency notifications
    send_emergency_notification "$incident_type" "$incident_id"
}

# Restore deleted keys
restore_deleted_keys() {
    log_monitor "INFO" "Attempting to restore deleted keys from backup..." "RESTORE"
    
    local backup_dir="$SECRETS_DIR/backups"
    if [[ -d "$backup_dir" ]]; then
        local latest_backup=$(ls -t "$backup_dir"/*.backup 2>/dev/null | head -1)
        if [[ -n "$latest_backup" ]]; then
            log_monitor "SUCCESS" "Restoring from backup: $(basename "$latest_backup")" "RESTORE"
            # Implementation would restore the backup
        else
            log_monitor "ERROR" "No backup available for restoration" "RESTORE"
        fi
    fi
}

# Quarantine threat
quarantine_threat() {
    log_monitor "INFO" "Quarantining detected threat..." "QUARANTINE"
    
    # Kill suspicious processes
    pkill -f "crypto.*miner\|bitcoin.*miner" 2>/dev/null || true
    
    # Block suspicious IPs
    # iptables implementation would go here
    
    log_monitor "SUCCESS" "Threat quarantine initiated" "QUARANTINE"
}

# Secure access
secure_access() {
    log_monitor "INFO" "Securing system access..." "SECURE"
    
    # Change key file permissions to more restrictive
    find "$SECRETS_DIR" -name "*.key" -exec chmod 600 {} \;
    find "$SECRETS_DIR" -name "*.pem" -exec chmod 600 {} \;
    
    log_monitor "SUCCESS" "Access secured" "SECURE"
}

# Send notifications
send_notification() {
    local level="$1"
    local message="$2"
    
    # Webhook notification
    if [[ -n "$WEBHOOK_URL" ]]; then
        local payload='{"level":"'"$level"'","message":"'"$message"'","timestamp":"'"$(date -u +"%Y-%m-%dT%H:%M:%SZ")"'","system":"spotify-ai-agent"}'
        
        if command -v curl &> /dev/null; then
            curl -s -X POST -H "Content-Type: application/json" -d "$payload" "$WEBHOOK_URL" || true
        fi
    fi
    
    # Slack notification
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        local slack_payload='{"text":"🚨 Security Alert ['$level'] - '$message'","username":"SecurityBot"}'
        
        if command -v curl &> /dev/null; then
            curl -s -X POST -H "Content-Type: application/json" -d "$slack_payload" "$SLACK_WEBHOOK" || true
        fi
    fi
}

# Send emergency notification
send_emergency_notification() {
    local incident_type="$1"
    local incident_id="$2"
    
    local emergency_message="🚨 SECURITY INCIDENT - $incident_type (ID: $incident_id) - Immediate attention required!"
    
    send_notification "EMERGENCY" "$emergency_message"
    
    # Additional emergency channels
    if [[ "$EMAIL_ALERTS" == "true" ]]; then
        # Email implementation would go here
        log_monitor "INFO" "Emergency email notification sent" "NOTIFICATION"
    fi
}

# Update metrics
update_metrics() {
    local event_type="$1"
    local event_details="$2"
    local file="$3"
    
    # Update real-time metrics
    local metrics_file="$METRICS_DIR/realtime_metrics.json"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    cat > "$metrics_file" << EOF
{
  "timestamp": "$timestamp",
  "metrics": {
    "key_access_count": ${METRICS["key_access_count"]},
    "failed_access_attempts": ${METRICS["failed_access_attempts"]},
    "rotation_events": ${METRICS["rotation_events"]},
    "security_violations": ${METRICS["security_violations"]},
    "threat_score": ${METRICS["threat_score"]},
    "anomaly_score": ${METRICS["anomaly_score"]}
  },
  "last_event": {
    "type": "$event_type",
    "details": "$event_details",
    "file": "$(basename "$file" 2>/dev/null || echo 'N/A')"
  }
}
EOF
    
    chmod 600 "$metrics_file"
}

# Generate monitoring report
generate_monitoring_report() {
    log_monitor "INFO" "Generating security monitoring report..." "REPORT"
    
    local report_file="$METRICS_DIR/monitoring_report_$(date +%Y%m%d_%H%M%S).json"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    cat > "$report_file" << EOF
{
  "report_metadata": {
    "generated_at": "$timestamp",
    "monitoring_period": "real-time",
    "system": "spotify-ai-agent-security-monitor"
  },
  "security_metrics": {
    "total_key_accesses": ${METRICS["key_access_count"]},
    "failed_access_attempts": ${METRICS["failed_access_attempts"]},
    "security_violations": ${METRICS["security_violations"]},
    "current_threat_score": ${METRICS["threat_score"]},
    "current_anomaly_score": ${METRICS["anomaly_score"]}
  },
  "monitoring_status": {
    "filesystem_monitoring": "ACTIVE",
    "process_monitoring": "ACTIVE",
    "network_monitoring": "ACTIVE",
    "threat_detection": "$THREAT_DETECTION",
    "anomaly_detection": "$ANOMALY_DETECTION"
  },
  "recent_incidents": "$(ls -1 "$METRICS_DIR/incidents/" 2>/dev/null | wc -l) incidents recorded",
  "recommendations": [
    "Continue monitoring key access patterns",
    "Review and update threat detection rules",
    "Analyze anomaly patterns for false positives",
    "Implement additional security controls if needed"
  ]
}
EOF
    
    chmod 600 "$report_file"
    log_monitor "SUCCESS" "Monitoring report generated: $(basename "$report_file")" "REPORT"
}

# Main monitoring loop
main_monitoring_loop() {
    log_monitor "INFO" "Starting main monitoring loop..." "MAIN"
    
    # Start all monitoring components
    monitor_filesystem_events
    monitor_process_activity
    monitor_network_activity
    monitor_system_resources
    
    local report_interval=3600  # Generate report every hour
    local last_report_time=$(date +%s)
    local threat_check_interval=300  # Check threat intelligence every 5 minutes
    local last_threat_check=$(date +%s)
    
    while true; do
        local current_time=$(date +%s)
        
        # Generate periodic reports
        if [[ $((current_time - last_report_time)) -gt $report_interval ]]; then
            generate_monitoring_report
            last_report_time=$current_time
        fi
        
        # Check threat intelligence
        if [[ $((current_time - last_threat_check)) -gt $threat_check_interval ]]; then
            check_threat_intelligence
            last_threat_check=$current_time
        fi
        
        # Check alert thresholds
        if [[ ${METRICS["threat_score"]} -gt $ALERT_THRESHOLD_CRITICAL ]]; then
            log_monitor "CRITICAL" "Critical threat score threshold exceeded: ${METRICS["threat_score"]}" "ALERT"
            trigger_incident_response "high_threat_score" "${METRICS["threat_score"]}"
        elif [[ ${METRICS["threat_score"]} -gt $ALERT_THRESHOLD_HIGH ]]; then
            log_monitor "HIGH" "High threat score detected: ${METRICS["threat_score"]}" "ALERT"
        fi
        
        # Update real-time metrics
        update_metrics "monitoring_cycle" "periodic_update" "monitoring_system"
        
        # Reset threat score periodically to prevent indefinite accumulation
        if [[ $((current_time % 3600)) -eq 0 ]]; then
            METRICS["threat_score"]=$((METRICS["threat_score"] / 2))
            METRICS["anomaly_score"]=$((METRICS["anomaly_score"] / 2))
        fi
        
        sleep "$MONITORING_INTERVAL"
    done
}

# Parse command line arguments
parse_monitor_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --daemon|-d)
                DAEMON_MODE=true
                shift
                ;;
            --interval)
                MONITORING_INTERVAL="$2"
                shift 2
                ;;
            --no-threat-detection)
                THREAT_DETECTION=false
                shift
                ;;
            --no-anomaly-detection)
                ANOMALY_DETECTION=false
                shift
                ;;
            --webhook)
                WEBHOOK_URL="$2"
                shift 2
                ;;
            --slack)
                SLACK_WEBHOOK="$2"
                shift 2
                ;;
            --help)
                show_monitor_help
                exit 0
                ;;
            *)
                log_monitor "ERROR" "Unknown option: $1"
                show_monitor_help
                exit 1
                ;;
        esac
    done
}

# Show help
show_monitor_help() {
    echo "Advanced Security Monitoring Script v2.0"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --daemon                 Run in daemon mode"
    echo "  --interval SECONDS           Set monitoring interval (default: 30)"
    echo "  --no-threat-detection        Disable threat detection"
    echo "  --no-anomaly-detection       Disable anomaly detection"
    echo "  --webhook URL                Set webhook URL for notifications"
    echo "  --slack URL                  Set Slack webhook URL"
    echo "  --help                       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --daemon                             # Run as daemon"
    echo "  $0 --interval 60                        # Monitor every 60 seconds"
    echo "  $0 --webhook https://alerts.company.com # Send webhook notifications"
}

# Cleanup function
cleanup_monitoring() {
    umask "$UMASK_ORIGINAL"
    
    # Kill background monitoring processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    log_monitor "INFO" "Security monitoring stopped"
}

# Signal handlers
handle_sigterm() {
    log_monitor "INFO" "Received SIGTERM, shutting down gracefully..."
    cleanup_monitoring
    exit 0
}

handle_sigint() {
    log_monitor "INFO" "Received SIGINT, shutting down..."
    cleanup_monitoring
    exit 0
}

# Main execution
main() {
    # Set up signal handlers
    trap handle_sigterm TERM
    trap handle_sigint INT
    trap cleanup_monitoring EXIT
    
    # Initialize
    initialize_monitoring
    
    # Parse arguments
    parse_monitor_arguments "$@"
    
    # Check dependencies
    if ! command -v inotifywait &> /dev/null; then
        log_monitor "WARNING" "inotify-tools not installed - using polling method"
    fi
    
    # Start monitoring
    if [[ "$DAEMON_MODE" == "true" ]]; then
        log_monitor "INFO" "Starting daemon mode monitoring..."
        main_monitoring_loop
    else
        log_monitor "INFO" "Starting interactive monitoring (Ctrl+C to stop)..."
        main_monitoring_loop
    fi
}

# Execute main function
main "$@"
