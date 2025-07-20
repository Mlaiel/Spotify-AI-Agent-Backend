#!/bin/bash

# Advanced Kubernetes Jobs Management Script for Spotify AI Agent
# Ultra-Advanced Enterprise-Grade Job Orchestration and Management
# 
# This script provides comprehensive job management capabilities:
# - Job creation, monitoring, and lifecycle management
# - Advanced scheduling and dependency management
# - Resource optimization and auto-scaling
# - Security and compliance enforcement
# - Real-time monitoring and alerting
# - Multi-tenant job isolation and resource quotas

set -euo pipefail

# Script Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "$0")"
SCRIPT_VERSION="2.1.0"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
DRY_RUN="${DRY_RUN:-false}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"

# Kubernetes Configuration
NAMESPACE="${NAMESPACE:-spotify-ai-agent-dev}"
KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
KUBECTL_TIMEOUT="${KUBECTL_TIMEOUT:-300}"

# Job Management Configuration
JOB_TIMEOUT_DEFAULT=3600
JOB_RETRY_LIMIT=3
JOB_CLEANUP_POLICY="OnCompletion"
JOB_TTL_AFTER_FINISHED=86400

# Monitoring and Alerting
PROMETHEUS_ENABLED="${PROMETHEUS_ENABLED:-true}"
JAEGER_ENABLED="${JAEGER_ENABLED:-true}"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
PAGERDUTY_INTEGRATION_KEY="${PAGERDUTY_INTEGRATION_KEY:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" >&1
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
}

log_debug() {
    if [[ "${LOG_LEVEL}" == "DEBUG" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
    fi
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" >&1
}

# Utility functions
check_dependencies() {
    local deps=("kubectl" "jq" "yq" "curl" "openssl")
    
    log_info "Checking dependencies..."
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log_error "Required dependency not found: $dep"
            exit 1
        fi
    done
    
    log_success "All dependencies satisfied"
}

check_kubernetes_connection() {
    log_info "Checking Kubernetes connection..."
    
    if ! kubectl cluster-info --request-timeout=10s &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warn "Namespace '$NAMESPACE' does not exist, creating..."
        kubectl create namespace "$NAMESPACE"
    fi
    
    log_success "Kubernetes connection verified"
}

validate_job_manifest() {
    local manifest_file="$1"
    
    log_info "Validating job manifest: $manifest_file"
    
    if [[ ! -f "$manifest_file" ]]; then
        log_error "Job manifest file not found: $manifest_file"
        return 1
    fi
    
    # Validate YAML syntax
    if ! yq eval '.' "$manifest_file" &> /dev/null; then
        log_error "Invalid YAML syntax in: $manifest_file"
        return 1
    fi
    
    # Validate Kubernetes job schema
    if ! kubectl apply --dry-run=client -f "$manifest_file" &> /dev/null; then
        log_error "Invalid Kubernetes job manifest: $manifest_file"
        return 1
    fi
    
    # Security validation
    validate_job_security "$manifest_file"
    
    log_success "Job manifest validation passed: $manifest_file"
}

validate_job_security() {
    local manifest_file="$1"
    
    log_info "Performing security validation for: $manifest_file"
    
    # Check for privileged containers
    if yq eval '.spec.template.spec.containers[].securityContext.privileged // false' "$manifest_file" | grep -q "true"; then
        log_error "Privileged containers detected in: $manifest_file"
        return 1
    fi
    
    # Check for hostNetwork
    if yq eval '.spec.template.spec.hostNetwork // false' "$manifest_file" | grep -q "true"; then
        log_error "hostNetwork enabled in: $manifest_file"
        return 1
    fi
    
    # Check for runAsRoot
    if yq eval '.spec.template.spec.securityContext.runAsUser // 0' "$manifest_file" | grep -q "^0$"; then
        log_error "Container running as root detected in: $manifest_file"
        return 1
    fi
    
    # Check for resource limits
    if ! yq eval '.spec.template.spec.containers[].resources.limits' "$manifest_file" | grep -q "cpu\|memory"; then
        log_warn "No resource limits specified in: $manifest_file"
    fi
    
    log_success "Security validation passed for: $manifest_file"
}

create_job() {
    local job_type="$1"
    local tenant_id="${2:-enterprise-client-001}"
    local priority="${3:-normal}"
    local resource_tier="${4:-medium}"
    
    log_info "Creating job: type=$job_type, tenant=$tenant_id, priority=$priority, tier=$resource_tier"
    
    local job_manifest_file=""
    
    case "$job_type" in
        "ml-training")
            job_manifest_file="$SCRIPT_DIR/ml-training-job.yaml"
            ;;
        "data-etl")
            job_manifest_file="$SCRIPT_DIR/data-etl-job.yaml"
            ;;
        "tenant-backup")
            job_manifest_file="$SCRIPT_DIR/tenant-backup-job.yaml"
            ;;
        "security-scan")
            job_manifest_file="$SCRIPT_DIR/security-scan-job.yaml"
            ;;
        "billing-reporting")
            job_manifest_file="$SCRIPT_DIR/billing-reporting-job.yaml"
            ;;
        *)
            log_error "Unknown job type: $job_type"
            return 1
            ;;
    esac
    
    if [[ ! -f "$job_manifest_file" ]]; then
        log_error "Job manifest file not found: $job_manifest_file"
        return 1
    fi
    
    # Validate manifest
    validate_job_manifest "$job_manifest_file"
    
    # Generate unique job name
    local timestamp=$(date '+%Y%m%d-%H%M%S')
    local job_name="${job_type}-${tenant_id}-${timestamp}"
    
    # Create temporary manifest with substitutions
    local temp_manifest="/tmp/${job_name}.yaml"
    cp "$job_manifest_file" "$temp_manifest"
    
    # Substitute variables
    sed -i "s/enterprise-client-001/${tenant_id}/g" "$temp_manifest"
    sed -i "s/name: .*-advanced/name: ${job_name}/g" "$temp_manifest"
    
    # Apply resource tier adjustments
    apply_resource_tier "$temp_manifest" "$resource_tier"
    
    # Apply priority adjustments
    apply_priority_settings "$temp_manifest" "$priority"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would create job with manifest:"
        cat "$temp_manifest"
        rm -f "$temp_manifest"
        return 0
    fi
    
    # Create the job
    if kubectl apply -f "$temp_manifest" -n "$NAMESPACE"; then
        log_success "Job created successfully: $job_name"
        
        # Start monitoring the job
        monitor_job "$job_name" &
        
        # Send notification
        send_notification "Job Created" "Job $job_name has been created successfully" "info"
        
        # Store job metadata
        store_job_metadata "$job_name" "$job_type" "$tenant_id" "$priority" "$resource_tier"
        
        echo "$job_name"
    else
        log_error "Failed to create job: $job_name"
        rm -f "$temp_manifest"
        return 1
    fi
    
    rm -f "$temp_manifest"
}

apply_resource_tier() {
    local manifest_file="$1"
    local tier="$2"
    
    log_debug "Applying resource tier: $tier to $manifest_file"
    
    case "$tier" in
        "micro")
            yq eval -i '.spec.template.spec.containers[0].resources.requests.cpu = "100m"' "$manifest_file"
            yq eval -i '.spec.template.spec.containers[0].resources.requests.memory = "128Mi"' "$manifest_file"
            yq eval -i '.spec.template.spec.containers[0].resources.limits.cpu = "200m"' "$manifest_file"
            yq eval -i '.spec.template.spec.containers[0].resources.limits.memory = "256Mi"' "$manifest_file"
            ;;
        "small")
            yq eval -i '.spec.template.spec.containers[0].resources.requests.cpu = "250m"' "$manifest_file"
            yq eval -i '.spec.template.spec.containers[0].resources.requests.memory = "512Mi"' "$manifest_file"
            yq eval -i '.spec.template.spec.containers[0].resources.limits.cpu = "500m"' "$manifest_file"
            yq eval -i '.spec.template.spec.containers[0].resources.limits.memory = "1Gi"' "$manifest_file"
            ;;
        "medium")
            # Default values, no changes needed
            ;;
        "large")
            yq eval -i '.spec.template.spec.containers[0].resources.requests.cpu = "2000m"' "$manifest_file"
            yq eval -i '.spec.template.spec.containers[0].resources.requests.memory = "4Gi"' "$manifest_file"
            yq eval -i '.spec.template.spec.containers[0].resources.limits.cpu = "4000m"' "$manifest_file"
            yq eval -i '.spec.template.spec.containers[0].resources.limits.memory = "8Gi"' "$manifest_file"
            ;;
        "xlarge")
            yq eval -i '.spec.template.spec.containers[0].resources.requests.cpu = "8000m"' "$manifest_file"
            yq eval -i '.spec.template.spec.containers[0].resources.requests.memory = "16Gi"' "$manifest_file"
            yq eval -i '.spec.template.spec.containers[0].resources.limits.cpu = "16000m"' "$manifest_file"
            yq eval -i '.spec.template.spec.containers[0].resources.limits.memory = "32Gi"' "$manifest_file"
            ;;
        "enterprise")
            yq eval -i '.spec.template.spec.containers[0].resources.requests.cpu = "16000m"' "$manifest_file"
            yq eval -i '.spec.template.spec.containers[0].resources.requests.memory = "32Gi"' "$manifest_file"
            yq eval -i '.spec.template.spec.containers[0].resources.limits.cpu = "32000m"' "$manifest_file"
            yq eval -i '.spec.template.spec.containers[0].resources.limits.memory = "64Gi"' "$manifest_file"
            ;;
    esac
}

apply_priority_settings() {
    local manifest_file="$1"
    local priority="$2"
    
    log_debug "Applying priority settings: $priority to $manifest_file"
    
    case "$priority" in
        "critical")
            yq eval -i '.spec.template.spec.priorityClassName = "critical-priority"' "$manifest_file"
            yq eval -i '.spec.activeDeadlineSeconds = 7200' "$manifest_file"  # 2 hours
            yq eval -i '.spec.backoffLimit = 1' "$manifest_file"
            ;;
        "high")
            yq eval -i '.spec.template.spec.priorityClassName = "high-priority"' "$manifest_file"
            yq eval -i '.spec.activeDeadlineSeconds = 10800' "$manifest_file"  # 3 hours
            yq eval -i '.spec.backoffLimit = 2' "$manifest_file"
            ;;
        "normal")
            yq eval -i '.spec.template.spec.priorityClassName = "normal-priority"' "$manifest_file"
            # Default values
            ;;
        "low")
            yq eval -i '.spec.template.spec.priorityClassName = "low-priority"' "$manifest_file"
            yq eval -i '.spec.activeDeadlineSeconds = 21600' "$manifest_file"  # 6 hours
            yq eval -i '.spec.backoffLimit = 5' "$manifest_file"
            ;;
        "batch")
            yq eval -i '.spec.template.spec.priorityClassName = "batch-priority"' "$manifest_file"
            yq eval -i '.spec.activeDeadlineSeconds = 43200' "$manifest_file"  # 12 hours
            yq eval -i '.spec.backoffLimit = 3' "$manifest_file"
            ;;
    esac
}

monitor_job() {
    local job_name="$1"
    local max_wait_time=7200  # 2 hours
    local check_interval=30   # 30 seconds
    local elapsed_time=0
    
    log_info "Starting monitoring for job: $job_name"
    
    while [[ $elapsed_time -lt $max_wait_time ]]; do
        local job_status
        job_status=$(kubectl get job "$job_name" -n "$NAMESPACE" -o jsonpath='{.status}' 2>/dev/null || echo '{}')
        
        if [[ -z "$job_status" || "$job_status" == "{}" ]]; then
            log_warn "Job not found or status unavailable: $job_name"
            sleep $check_interval
            elapsed_time=$((elapsed_time + check_interval))
            continue
        fi
        
        local conditions
        conditions=$(echo "$job_status" | jq -r '.conditions // []')
        
        # Check for completion
        if echo "$conditions" | jq -e '.[] | select(.type=="Complete" and .status=="True")' &> /dev/null; then
            log_success "Job completed successfully: $job_name"
            send_notification "Job Completed" "Job $job_name has completed successfully" "success"
            update_job_status "$job_name" "completed"
            return 0
        fi
        
        # Check for failure
        if echo "$conditions" | jq -e '.[] | select(.type=="Failed" and .status=="True")' &> /dev/null; then
            log_error "Job failed: $job_name"
            
            # Get failure reason
            local failure_reason
            failure_reason=$(echo "$conditions" | jq -r '.[] | select(.type=="Failed") | .reason // "Unknown"')
            
            send_notification "Job Failed" "Job $job_name has failed: $failure_reason" "error"
            update_job_status "$job_name" "failed"
            
            # Get pod logs for debugging
            get_job_logs "$job_name"
            return 1
        fi
        
        # Log progress
        local active_pods
        active_pods=$(echo "$job_status" | jq -r '.active // 0')
        local succeeded_pods
        succeeded_pods=$(echo "$job_status" | jq -r '.succeeded // 0')
        local failed_pods
        failed_pods=$(echo "$job_status" | jq -r '.failed // 0')
        
        log_debug "Job $job_name status: active=$active_pods, succeeded=$succeeded_pods, failed=$failed_pods"
        
        # Update metrics if Prometheus is enabled
        if [[ "$PROMETHEUS_ENABLED" == "true" ]]; then
            update_prometheus_metrics "$job_name" "$active_pods" "$succeeded_pods" "$failed_pods"
        fi
        
        sleep $check_interval
        elapsed_time=$((elapsed_time + check_interval))
    done
    
    log_warn "Job monitoring timeout reached for: $job_name"
    send_notification "Job Timeout" "Job $job_name monitoring timeout after ${max_wait_time}s" "warning"
    update_job_status "$job_name" "timeout"
    return 2
}

get_job_logs() {
    local job_name="$1"
    local log_file="/tmp/${job_name}-logs.txt"
    
    log_info "Collecting logs for job: $job_name"
    
    # Get pods for this job
    local pods
    pods=$(kubectl get pods -n "$NAMESPACE" -l "job-name=$job_name" -o jsonpath='{.items[*].metadata.name}')
    
    if [[ -z "$pods" ]]; then
        log_warn "No pods found for job: $job_name"
        return
    fi
    
    echo "=== Job Logs for $job_name ===" > "$log_file"
    echo "Timestamp: $(date)" >> "$log_file"
    echo "" >> "$log_file"
    
    for pod in $pods; do
        echo "=== Pod: $pod ===" >> "$log_file"
        kubectl logs "$pod" -n "$NAMESPACE" --all-containers=true >> "$log_file" 2>&1 || true
        echo "" >> "$log_file"
        
        # Get pod events
        echo "=== Pod Events: $pod ===" >> "$log_file"
        kubectl describe pod "$pod" -n "$NAMESPACE" | grep -A 20 "Events:" >> "$log_file" 2>&1 || true
        echo "" >> "$log_file"
    done
    
    log_info "Job logs saved to: $log_file"
}

list_jobs() {
    local filter="${1:-all}"
    local tenant_id="${2:-}"
    
    log_info "Listing jobs with filter: $filter"
    
    local label_selector=""
    if [[ -n "$tenant_id" ]]; then
        label_selector="-l tenant-id=$tenant_id"
    fi
    
    case "$filter" in
        "all")
            kubectl get jobs -n "$NAMESPACE" $label_selector -o wide
            ;;
        "running")
            kubectl get jobs -n "$NAMESPACE" $label_selector -o wide | grep -E "(Running|Active)"
            ;;
        "completed")
            kubectl get jobs -n "$NAMESPACE" $label_selector -o wide | grep "Complete"
            ;;
        "failed")
            kubectl get jobs -n "$NAMESPACE" $label_selector -o wide | grep "Failed"
            ;;
        *)
            log_error "Invalid filter: $filter. Use: all, running, completed, failed"
            return 1
            ;;
    esac
}

delete_job() {
    local job_name="$1"
    local force="${2:-false}"
    
    log_info "Deleting job: $job_name (force=$force)"
    
    if [[ "$force" != "true" ]]; then
        read -p "Are you sure you want to delete job '$job_name'? (y/N): " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Job deletion cancelled"
            return 0
        fi
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would delete job: $job_name"
        return 0
    fi
    
    # Delete the job and associated pods
    if kubectl delete job "$job_name" -n "$NAMESPACE" --cascade=foreground; then
        log_success "Job deleted successfully: $job_name"
        send_notification "Job Deleted" "Job $job_name has been deleted" "info"
        remove_job_metadata "$job_name"
    else
        log_error "Failed to delete job: $job_name"
        return 1
    fi
}

cleanup_completed_jobs() {
    local max_age_hours="${1:-24}"
    
    log_info "Cleaning up completed jobs older than $max_age_hours hours"
    
    local cutoff_time
    cutoff_time=$(date -d "$max_age_hours hours ago" --iso-8601)
    
    # Get completed jobs
    local completed_jobs
    completed_jobs=$(kubectl get jobs -n "$NAMESPACE" -o json | jq -r '.items[] | select(.status.conditions[]? | select(.type=="Complete" and .status=="True")) | select(.status.completionTime < "'"$cutoff_time"'") | .metadata.name')
    
    if [[ -z "$completed_jobs" ]]; then
        log_info "No completed jobs found for cleanup"
        return 0
    fi
    
    log_info "Found completed jobs for cleanup: $(echo "$completed_jobs" | wc -l)"
    
    for job in $completed_jobs; do
        log_info "Cleaning up completed job: $job"
        delete_job "$job" "true"
    done
    
    log_success "Completed jobs cleanup finished"
}

get_job_status() {
    local job_name="$1"
    
    if ! kubectl get job "$job_name" -n "$NAMESPACE" &> /dev/null; then
        echo "NOT_FOUND"
        return 1
    fi
    
    local job_status
    job_status=$(kubectl get job "$job_name" -n "$NAMESPACE" -o jsonpath='{.status}')
    
    local conditions
    conditions=$(echo "$job_status" | jq -r '.conditions // []')
    
    if echo "$conditions" | jq -e '.[] | select(.type=="Complete" and .status=="True")' &> /dev/null; then
        echo "COMPLETED"
    elif echo "$conditions" | jq -e '.[] | select(.type=="Failed" and .status=="True")' &> /dev/null; then
        echo "FAILED"
    elif [[ $(echo "$job_status" | jq -r '.active // 0') -gt 0 ]]; then
        echo "RUNNING"
    else
        echo "PENDING"
    fi
}

store_job_metadata() {
    local job_name="$1"
    local job_type="$2"
    local tenant_id="$3"
    local priority="$4"
    local resource_tier="$5"
    
    local metadata_file="/tmp/job-metadata-${job_name}.json"
    
    cat > "$metadata_file" << EOF
{
    "job_name": "$job_name",
    "job_type": "$job_type",
    "tenant_id": "$tenant_id",
    "priority": "$priority",
    "resource_tier": "$resource_tier",
    "created_at": "$(date --iso-8601)",
    "namespace": "$NAMESPACE"
}
EOF
    
    log_debug "Job metadata stored: $metadata_file"
}

update_job_status() {
    local job_name="$1"
    local status="$2"
    
    local metadata_file="/tmp/job-metadata-${job_name}.json"
    
    if [[ -f "$metadata_file" ]]; then
        jq --arg status "$status" --arg updated_at "$(date --iso-8601)" '. + {status: $status, updated_at: $updated_at}' "$metadata_file" > "${metadata_file}.tmp"
        mv "${metadata_file}.tmp" "$metadata_file"
        log_debug "Job status updated: $job_name -> $status"
    fi
}

remove_job_metadata() {
    local job_name="$1"
    local metadata_file="/tmp/job-metadata-${job_name}.json"
    
    if [[ -f "$metadata_file" ]]; then
        rm -f "$metadata_file"
        log_debug "Job metadata removed: $metadata_file"
    fi
}

update_prometheus_metrics() {
    local job_name="$1"
    local active_pods="$2"
    local succeeded_pods="$3"
    local failed_pods="$4"
    
    if [[ "$PROMETHEUS_ENABLED" != "true" ]] || [[ -z "$PROMETHEUS_GATEWAY" ]]; then
        return 0
    fi
    
    local metrics_data="# TYPE spotify_ai_job_active_pods gauge
spotify_ai_job_active_pods{job_name=\"$job_name\",namespace=\"$NAMESPACE\"} $active_pods
# TYPE spotify_ai_job_succeeded_pods gauge
spotify_ai_job_succeeded_pods{job_name=\"$job_name\",namespace=\"$NAMESPACE\"} $succeeded_pods
# TYPE spotify_ai_job_failed_pods gauge
spotify_ai_job_failed_pods{job_name=\"$job_name\",namespace=\"$NAMESPACE\"} $failed_pods"
    
    if ! curl -s -X POST "$PROMETHEUS_GATEWAY/metrics/job/spotify-ai-jobs/instance/$job_name" \
         --data-binary "$metrics_data" &> /dev/null; then
        log_debug "Failed to push metrics to Prometheus for job: $job_name"
    fi
}

send_notification() {
    local title="$1"
    local message="$2"
    local level="${3:-info}"
    
    case "$level" in
        "success")
            local color="good"
            local emoji="✅"
            ;;
        "warning")
            local color="warning"
            local emoji="⚠️"
            ;;
        "error")
            local color="danger"
            local emoji="❌"
            ;;
        *)
            local color="#36a64f"
            local emoji="ℹ️"
            ;;
    esac
    
    # Send Slack notification
    if [[ -n "$SLACK_WEBHOOK_URL" ]]; then
        local slack_payload
        slack_payload=$(cat << EOF
{
    "attachments": [
        {
            "color": "$color",
            "title": "$emoji $title",
            "text": "$message",
            "fields": [
                {
                    "title": "Namespace",
                    "value": "$NAMESPACE",
                    "short": true
                },
                {
                    "title": "Timestamp",
                    "value": "$(date --iso-8601)",
                    "short": true
                }
            ]
        }
    ]
}
EOF
        )
        
        curl -s -X POST -H 'Content-type: application/json' \
             --data "$slack_payload" \
             "$SLACK_WEBHOOK_URL" &> /dev/null || true
    fi
    
    # Send PagerDuty alert for critical issues
    if [[ "$level" == "error" && -n "$PAGERDUTY_INTEGRATION_KEY" ]]; then
        local pagerduty_payload
        pagerduty_payload=$(cat << EOF
{
    "routing_key": "$PAGERDUTY_INTEGRATION_KEY",
    "event_action": "trigger",
    "payload": {
        "summary": "$title",
        "source": "spotify-ai-jobs",
        "severity": "error",
        "custom_details": {
            "message": "$message",
            "namespace": "$NAMESPACE"
        }
    }
}
EOF
        )
        
        curl -s -X POST -H 'Content-Type: application/json' \
             --data "$pagerduty_payload" \
             "https://events.pagerduty.com/v2/enqueue" &> /dev/null || true
    fi
}

show_help() {
    cat << EOF
Advanced Kubernetes Jobs Management Script v$SCRIPT_VERSION

USAGE:
    $SCRIPT_NAME <command> [options]

COMMANDS:
    create <type> [tenant_id] [priority] [tier]  Create a new job
    list [filter] [tenant_id]                   List jobs with optional filter
    status <job_name>                           Get job status
    monitor <job_name>                          Monitor job execution
    delete <job_name> [force]                   Delete a job
    cleanup [max_age_hours]                     Cleanup old completed jobs
    logs <job_name>                             Get job logs
    validate <manifest_file>                    Validate job manifest
    help                                        Show this help message

JOB TYPES:
    ml-training        Machine Learning model training
    data-etl          Data ETL pipeline processing
    tenant-backup     Tenant backup and migration
    security-scan     Security and compliance scanning
    billing-reporting Billing and revenue reporting

PRIORITIES:
    critical          Critical priority (highest)
    high             High priority
    normal           Normal priority (default)
    low              Low priority
    batch            Batch priority (lowest)

RESOURCE TIERS:
    micro            Minimal resources (100m CPU, 128Mi RAM)
    small            Small resources (250m CPU, 512Mi RAM)
    medium           Medium resources (500m CPU, 1Gi RAM) [default]
    large            Large resources (2 CPU, 4Gi RAM)
    xlarge           Extra large resources (8 CPU, 16Gi RAM)
    enterprise       Enterprise resources (16 CPU, 32Gi RAM)

FILTERS:
    all              Show all jobs (default)
    running          Show only running jobs
    completed        Show only completed jobs
    failed           Show only failed jobs

ENVIRONMENT VARIABLES:
    NAMESPACE        Kubernetes namespace (default: spotify-ai-agent-dev)
    LOG_LEVEL        Logging level: INFO, DEBUG (default: INFO)
    DRY_RUN          Dry run mode: true, false (default: false)
    PARALLEL_JOBS    Number of parallel jobs (default: 4)

EXAMPLES:
    # Create ML training job for premium tenant
    $SCRIPT_NAME create ml-training premium-client-001 high xlarge

    # List all running jobs
    $SCRIPT_NAME list running

    # Monitor specific job
    $SCRIPT_NAME monitor ml-training-premium-client-001-20250717-143022

    # Cleanup jobs older than 48 hours
    $SCRIPT_NAME cleanup 48

    # Delete failed job
    $SCRIPT_NAME delete security-scan-enterprise-20250717-120000 true

EOF
}

# Main execution
main() {
    case "${1:-help}" in
        "create")
            check_dependencies
            check_kubernetes_connection
            create_job "${2:-}" "${3:-}" "${4:-}" "${5:-}"
            ;;
        "list")
            check_dependencies
            check_kubernetes_connection
            list_jobs "${2:-all}" "${3:-}"
            ;;
        "status")
            if [[ -z "${2:-}" ]]; then
                log_error "Job name required for status command"
                exit 1
            fi
            check_dependencies
            check_kubernetes_connection
            get_job_status "$2"
            ;;
        "monitor")
            if [[ -z "${2:-}" ]]; then
                log_error "Job name required for monitor command"
                exit 1
            fi
            check_dependencies
            check_kubernetes_connection
            monitor_job "$2"
            ;;
        "delete")
            if [[ -z "${2:-}" ]]; then
                log_error "Job name required for delete command"
                exit 1
            fi
            check_dependencies
            check_kubernetes_connection
            delete_job "$2" "${3:-false}"
            ;;
        "cleanup")
            check_dependencies
            check_kubernetes_connection
            cleanup_completed_jobs "${2:-24}"
            ;;
        "logs")
            if [[ -z "${2:-}" ]]; then
                log_error "Job name required for logs command"
                exit 1
            fi
            check_dependencies
            check_kubernetes_connection
            get_job_logs "$2"
            ;;
        "validate")
            if [[ -z "${2:-}" ]]; then
                log_error "Manifest file required for validate command"
                exit 1
            fi
            check_dependencies
            validate_job_manifest "$2"
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "Unknown command: ${1:-}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"
