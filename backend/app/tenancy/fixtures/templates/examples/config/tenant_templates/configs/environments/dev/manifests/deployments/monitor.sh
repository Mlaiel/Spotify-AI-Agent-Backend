#!/bin/bash
#
# Advanced Deployment Monitoring and Validation Script
# Developed by Fahed Mlaiel - DevOps & Monitoring Specialist
# 
# This script provides comprehensive monitoring and validation for deployments:
# - Real-time health monitoring
# - Performance metrics collection
# - Security compliance validation
# - Resource utilization analysis
# - Automated alerting and notifications
# - SLA monitoring and reporting
#

set -euo pipefail

# Script metadata
readonly SCRIPT_VERSION="2.0.0"
readonly SCRIPT_AUTHOR="Fahed Mlaiel - DevOps & Monitoring Specialist"
readonly SCRIPT_DATE="2025-07-17"

# Configuration
readonly NAMESPACE="${NAMESPACE:-spotify-ai-agent-dev}"
readonly ENVIRONMENT="${ENVIRONMENT:-development}"
readonly MONITORING_INTERVAL="${MONITORING_INTERVAL:-30}"
readonly ALERT_THRESHOLD_CPU="${ALERT_THRESHOLD_CPU:-80}"
readonly ALERT_THRESHOLD_MEMORY="${ALERT_THRESHOLD_MEMORY:-85}"
readonly ALERT_THRESHOLD_ERROR_RATE="${ALERT_THRESHOLD_ERROR_RATE:-5}"
readonly PROMETHEUS_URL="${PROMETHEUS_URL:-http://prometheus:9090}"
readonly GRAFANA_URL="${GRAFANA_URL:-http://grafana:3000}"
readonly ELASTICSEARCH_URL="${ELASTICSEARCH_URL:-http://elasticsearch:9200}"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Log file
readonly LOG_FILE="/tmp/monitoring-$(date +%Y%m%d-%H%M%S).log"

# Services to monitor
readonly SERVICES=(
    "spotify-ai-agent-backend"
    "spotify-ai-ml-service"
    "spotify-ai-analytics-service"
    "spotify-ai-notification-service"
    "spotify-ai-auth-service"
    "spotify-ai-billing-service"
    "spotify-ai-tenant-service"
)

#######################################
# Logging functions
#######################################
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        ERROR)
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message" >&2
            echo "[ERROR] ${timestamp} - $message" >> "$LOG_FILE"
            send_alert "ERROR" "$message"
            ;;
        WARN)
            echo -e "${YELLOW}[WARN]${NC} ${timestamp} - $message"
            echo "[WARN] ${timestamp} - $message" >> "$LOG_FILE"
            ;;
        INFO)
            echo -e "${GREEN}[INFO]${NC} ${timestamp} - $message"
            echo "[INFO] ${timestamp} - $message" >> "$LOG_FILE"
            ;;
        SUCCESS)
            echo -e "${GREEN}[SUCCESS]${NC} ${timestamp} - $message"
            echo "[SUCCESS] ${timestamp} - $message" >> "$LOG_FILE"
            ;;
        DEBUG)
            echo -e "${BLUE}[DEBUG]${NC} ${timestamp} - $message"
            echo "[DEBUG] ${timestamp} - $message" >> "$LOG_FILE"
            ;;
    esac
}

#######################################
# Monitoring functions
#######################################
check_deployment_health() {
    local service=$1
    
    log DEBUG "Checking health of deployment: $service"
    
    # Check if deployment exists
    if ! kubectl get deployment "$service" -n "$NAMESPACE" &> /dev/null; then
        log ERROR "Deployment $service not found in namespace $NAMESPACE"
        return 1
    fi
    
    # Get deployment status
    local desired_replicas=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
    local ready_replicas=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    local available_replicas=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.status.availableReplicas}' 2>/dev/null || echo "0")
    local updated_replicas=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.status.updatedReplicas}' 2>/dev/null || echo "0")
    
    # Calculate health percentage
    local health_percentage=0
    if [[ "$desired_replicas" -gt 0 ]]; then
        health_percentage=$((ready_replicas * 100 / desired_replicas))
    fi
    
    # Determine health status
    local status="UNHEALTHY"
    if [[ "$ready_replicas" == "$desired_replicas" && "$available_replicas" == "$desired_replicas" ]]; then
        status="HEALTHY"
        log SUCCESS "$service is healthy ($ready_replicas/$desired_replicas replicas ready)"
    elif [[ "$ready_replicas" -gt 0 ]]; then
        status="DEGRADED"
        log WARN "$service is degraded ($ready_replicas/$desired_replicas replicas ready)"
    else
        status="UNHEALTHY"
        log ERROR "$service is unhealthy (0/$desired_replicas replicas ready)"
    fi
    
    # Check pod status
    local pod_status=$(kubectl get pods -n "$NAMESPACE" -l app="$service" -o jsonpath='{.items[*].status.phase}')
    local failing_pods=$(kubectl get pods -n "$NAMESPACE" -l app="$service" --field-selector=status.phase!=Running -o name 2>/dev/null | wc -l)
    
    if [[ "$failing_pods" -gt 0 ]]; then
        log WARN "$service has $failing_pods failing pods"
        kubectl get pods -n "$NAMESPACE" -l app="$service" --field-selector=status.phase!=Running
    fi
    
    # Store metrics for reporting
    echo "$service,$status,$health_percentage,$ready_replicas,$desired_replicas,$failing_pods,$(date '+%Y-%m-%d %H:%M:%S')" >> "/tmp/health_metrics.csv"
    
    return $([[ "$status" == "HEALTHY" ]] && echo 0 || echo 1)
}

monitor_resource_usage() {
    local service=$1
    
    log DEBUG "Monitoring resource usage for: $service"
    
    # Get resource metrics from Kubernetes
    local cpu_usage=$(kubectl top pods -n "$NAMESPACE" -l app="$service" --no-headers 2>/dev/null | awk '{sum += $2} END {print sum}' | sed 's/m//')
    local memory_usage=$(kubectl top pods -n "$NAMESPACE" -l app="$service" --no-headers 2>/dev/null | awk '{sum += $3} END {print sum}' | sed 's/Mi//')
    
    # Get resource limits
    local cpu_limit=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].resources.limits.cpu}' | sed 's/m//')
    local memory_limit=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].resources.limits.memory}' | sed 's/Mi//')
    
    # Calculate usage percentages
    local cpu_percentage=0
    local memory_percentage=0
    
    if [[ -n "$cpu_limit" && "$cpu_limit" != "null" && "$cpu_limit" -gt 0 ]]; then
        cpu_percentage=$(echo "scale=2; $cpu_usage * 100 / $cpu_limit" | bc 2>/dev/null || echo "0")
    fi
    
    if [[ -n "$memory_limit" && "$memory_limit" != "null" && "$memory_limit" -gt 0 ]]; then
        memory_percentage=$(echo "scale=2; $memory_usage * 100 / $memory_limit" | bc 2>/dev/null || echo "0")
    fi
    
    # Check thresholds
    if (( $(echo "$cpu_percentage > $ALERT_THRESHOLD_CPU" | bc -l) )); then
        log WARN "$service CPU usage is high: ${cpu_percentage}% (threshold: ${ALERT_THRESHOLD_CPU}%)"
    fi
    
    if (( $(echo "$memory_percentage > $ALERT_THRESHOLD_MEMORY" | bc -l) )); then
        log WARN "$service Memory usage is high: ${memory_percentage}% (threshold: ${ALERT_THRESHOLD_MEMORY}%)"
    fi
    
    # Store metrics
    echo "$service,CPU,$cpu_usage,$cpu_limit,$cpu_percentage,$(date '+%Y-%m-%d %H:%M:%S')" >> "/tmp/resource_metrics.csv"
    echo "$service,Memory,$memory_usage,$memory_limit,$memory_percentage,$(date '+%Y-%m-%d %H:%M:%S')" >> "/tmp/resource_metrics.csv"
    
    log INFO "$service resources - CPU: ${cpu_percentage}%, Memory: ${memory_percentage}%"
}

check_application_metrics() {
    local service=$1
    
    log DEBUG "Checking application metrics for: $service"
    
    # Get service endpoint
    local service_port=$(kubectl get service "$service" -n "$NAMESPACE" -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "8080")
    local service_ip=$(kubectl get service "$service" -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}' 2>/dev/null)
    
    if [[ -z "$service_ip" || "$service_ip" == "null" ]]; then
        log WARN "Cannot get service IP for $service, skipping application metrics"
        return 1
    fi
    
    # Check health endpoint
    local health_status="UNKNOWN"
    if curl -s --max-time 5 "http://${service_ip}:${service_port}/health" > /dev/null 2>&1; then
        health_status="UP"
        log SUCCESS "$service health endpoint is responding"
    else
        health_status="DOWN"
        log ERROR "$service health endpoint is not responding"
    fi
    
    # Check metrics endpoint
    local metrics_available="false"
    if curl -s --max-time 5 "http://${service_ip}:9000/metrics" > /dev/null 2>&1; then
        metrics_available="true"
        log INFO "$service metrics endpoint is available"
        
        # Get basic metrics
        local response_time=$(curl -s --max-time 5 -w "%{time_total}" "http://${service_ip}:${service_port}/health" -o /dev/null 2>/dev/null || echo "0")
        local http_requests_total=$(curl -s --max-time 5 "http://${service_ip}:9000/metrics" | grep "http_requests_total" | head -1 | awk '{print $2}' || echo "0")
        
        echo "$service,$health_status,$metrics_available,$response_time,$http_requests_total,$(date '+%Y-%m-%d %H:%M:%S')" >> "/tmp/app_metrics.csv"
    else
        log WARN "$service metrics endpoint is not available"
        echo "$service,$health_status,$metrics_available,0,0,$(date '+%Y-%m-%d %H:%M:%S')" >> "/tmp/app_metrics.csv"
    fi
}

validate_security_compliance() {
    local service=$1
    
    log DEBUG "Validating security compliance for: $service"
    
    local compliance_issues=0
    
    # Check security context
    local run_as_non_root=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.securityContext.runAsNonRoot}' 2>/dev/null)
    if [[ "$run_as_non_root" != "true" ]]; then
        log WARN "$service: Security issue - not running as non-root user"
        ((compliance_issues++))
    fi
    
    # Check read-only root filesystem
    local read_only_root=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].securityContext.readOnlyRootFilesystem}' 2>/dev/null)
    if [[ "$read_only_root" != "true" ]]; then
        log WARN "$service: Security issue - root filesystem is not read-only"
        ((compliance_issues++))
    fi
    
    # Check privileged escalation
    local allow_privilege_escalation=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].securityContext.allowPrivilegeEscalation}' 2>/dev/null)
    if [[ "$allow_privilege_escalation" == "true" ]]; then
        log WARN "$service: Security issue - privilege escalation is allowed"
        ((compliance_issues++))
    fi
    
    # Check resource limits
    local cpu_limits=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].resources.limits.cpu}' 2>/dev/null)
    local memory_limits=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].resources.limits.memory}' 2>/dev/null)
    
    if [[ -z "$cpu_limits" || "$cpu_limits" == "null" ]]; then
        log WARN "$service: Security issue - no CPU limits defined"
        ((compliance_issues++))
    fi
    
    if [[ -z "$memory_limits" || "$memory_limits" == "null" ]]; then
        log WARN "$service: Security issue - no memory limits defined"
        ((compliance_issues++))
    fi
    
    # Check network policies
    local network_policies=$(kubectl get networkpolicies -n "$NAMESPACE" -o name | wc -l)
    if [[ "$network_policies" -eq 0 ]]; then
        log WARN "$service: Security issue - no network policies found in namespace"
        ((compliance_issues++))
    fi
    
    local compliance_status="COMPLIANT"
    if [[ "$compliance_issues" -gt 0 ]]; then
        compliance_status="NON_COMPLIANT"
        log ERROR "$service has $compliance_issues security compliance issues"
    else
        log SUCCESS "$service is security compliant"
    fi
    
    echo "$service,$compliance_status,$compliance_issues,$(date '+%Y-%m-%d %H:%M:%S')" >> "/tmp/security_metrics.csv"
    
    return $compliance_issues
}

check_service_dependencies() {
    local service=$1
    
    log DEBUG "Checking service dependencies for: $service"
    
    local dependency_issues=0
    
    # Check database connections
    case $service in
        *auth*|*billing*|*tenant*)
            # Check PostgreSQL connectivity
            if ! kubectl run dependency-check-db --rm -i --restart=Never --image=postgres:15-alpine -n "$NAMESPACE" -- pg_isready -h postgres-service -p 5432 &> /dev/null; then
                log WARN "$service: Database dependency issue - PostgreSQL not reachable"
                ((dependency_issues++))
            fi
            ;;
    esac
    
    # Check Redis connectivity
    if ! kubectl run dependency-check-redis --rm -i --restart=Never --image=redis:7-alpine -n "$NAMESPACE" -- redis-cli -h redis-cluster -p 6379 ping &> /dev/null; then
        log WARN "$service: Cache dependency issue - Redis not reachable"
        ((dependency_issues++))
    fi
    
    # Check Kafka connectivity for analytics and notification services
    case $service in
        *analytics*|*notification*)
            if ! kubectl run dependency-check-kafka --rm -i --restart=Never --image=confluentinc/cp-kafka:latest -n "$NAMESPACE" -- kafka-topics --bootstrap-server kafka-cluster:9092 --list &> /dev/null; then
                log WARN "$service: Message queue dependency issue - Kafka not reachable"
                ((dependency_issues++))
            fi
            ;;
    esac
    
    local dependency_status="HEALTHY"
    if [[ "$dependency_issues" -gt 0 ]]; then
        dependency_status="DEGRADED"
        log WARN "$service has $dependency_issues dependency issues"
    else
        log SUCCESS "$service dependencies are healthy"
    fi
    
    echo "$service,$dependency_status,$dependency_issues,$(date '+%Y-%m-%d %H:%M:%S')" >> "/tmp/dependency_metrics.csv"
    
    return $dependency_issues
}

#######################################
# Alerting functions
#######################################
send_alert() {
    local level=$1
    local message=$2
    
    # Send alert to webhook (Slack, Teams, etc.)
    if [[ -n "${WEBHOOK_URL:-}" ]]; then
        local payload="{\"level\": \"$level\", \"message\": \"$message\", \"timestamp\": \"$(date -Iseconds)\", \"environment\": \"$ENVIRONMENT\", \"namespace\": \"$NAMESPACE\"}"
        curl -s -X POST -H "Content-Type: application/json" -d "$payload" "$WEBHOOK_URL" || true
    fi
    
    # Send email alert
    if [[ -n "${SMTP_SERVER:-}" && "$level" == "ERROR" ]]; then
        echo "Subject: [${ENVIRONMENT}] Deployment Alert - $level
        
$message

Timestamp: $(date)
Environment: $ENVIRONMENT
Namespace: $NAMESPACE" | sendmail "${ALERT_EMAIL:-devops@spotify-ai-agent.com}" || true
    fi
}

generate_health_report() {
    log INFO "Generating comprehensive health report"
    
    local report_file="/tmp/health-report-$(date +%Y%m%d-%H%M%S).html"
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Spotify AI Agent - Deployment Health Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #1db954; color: white; padding: 20px; text-align: center; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .healthy { background-color: #d4edda; }
        .warning { background-color: #fff3cd; }
        .error { background-color: #f8d7da; }
        .metric { display: inline-block; margin: 5px 10px; padding: 5px; background-color: #f8f9fa; border-radius: 3px; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Spotify AI Agent - Deployment Health Report</h1>
        <p>Generated on $(date) by ${SCRIPT_AUTHOR}</p>
        <p>Environment: $ENVIRONMENT | Namespace: $NAMESPACE</p>
    </div>
    
    <div class="section">
        <h2>Overall System Health</h2>
        <div class="metric">Services Monitored: ${#SERVICES[@]}</div>
        <div class="metric">Environment: $ENVIRONMENT</div>
        <div class="metric">Namespace: $NAMESPACE</div>
        <div class="metric">Report Time: $(date)</div>
    </div>
    
    <div class="section">
        <h2>Service Health Status</h2>
        <table>
            <tr>
                <th>Service</th>
                <th>Status</th>
                <th>Health %</th>
                <th>Ready Replicas</th>
                <th>Desired Replicas</th>
                <th>Failing Pods</th>
            </tr>
EOF
    
    # Add health metrics to report
    if [[ -f "/tmp/health_metrics.csv" ]]; then
        while IFS=',' read -r service status health_pct ready_replicas desired_replicas failing_pods timestamp; do
            local row_class=""
            case $status in
                HEALTHY) row_class="healthy" ;;
                DEGRADED) row_class="warning" ;;
                UNHEALTHY) row_class="error" ;;
            esac
            
            echo "            <tr class=\"$row_class\">" >> "$report_file"
            echo "                <td>$service</td>" >> "$report_file"
            echo "                <td>$status</td>" >> "$report_file"
            echo "                <td>${health_pct}%</td>" >> "$report_file"
            echo "                <td>$ready_replicas</td>" >> "$report_file"
            echo "                <td>$desired_replicas</td>" >> "$report_file"
            echo "                <td>$failing_pods</td>" >> "$report_file"
            echo "            </tr>" >> "$report_file"
        done < "/tmp/health_metrics.csv"
    fi
    
    cat >> "$report_file" << EOF
        </table>
    </div>
    
    <div class="section">
        <h2>Resource Utilization</h2>
        <p>Current resource usage across all monitored services:</p>
        <!-- Resource metrics would be added here -->
    </div>
    
    <div class="section">
        <h2>Security Compliance</h2>
        <p>Security compliance status for all services:</p>
        <!-- Security metrics would be added here -->
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        <ul>
            <li>Monitor services with high resource utilization</li>
            <li>Address security compliance issues</li>
            <li>Review and optimize unhealthy services</li>
            <li>Ensure all dependencies are available</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Generated by</h2>
        <p>${SCRIPT_AUTHOR}</p>
        <p>Advanced Deployment Monitoring Script v${SCRIPT_VERSION}</p>
    </div>
</body>
</html>
EOF
    
    log INFO "Health report generated: $report_file"
    echo "$report_file"
}

#######################################
# Main monitoring loop
#######################################
monitor_all_services() {
    log INFO "Starting comprehensive monitoring of all services"
    
    # Initialize metric files
    echo "Service,Status,HealthPercentage,ReadyReplicas,DesiredReplicas,FailingPods,Timestamp" > "/tmp/health_metrics.csv"
    echo "Service,ResourceType,Usage,Limit,Percentage,Timestamp" > "/tmp/resource_metrics.csv"
    echo "Service,HealthStatus,MetricsAvailable,ResponseTime,RequestsTotal,Timestamp" > "/tmp/app_metrics.csv"
    echo "Service,ComplianceStatus,Issues,Timestamp" > "/tmp/security_metrics.csv"
    echo "Service,DependencyStatus,Issues,Timestamp" > "/tmp/dependency_metrics.csv"
    
    local total_issues=0
    local healthy_services=0
    
    for service in "${SERVICES[@]}"; do
        log INFO "Monitoring service: $service"
        
        # Check deployment health
        if check_deployment_health "$service"; then
            ((healthy_services++))
        else
            ((total_issues++))
        fi
        
        # Monitor resource usage
        monitor_resource_usage "$service"
        
        # Check application metrics
        check_application_metrics "$service"
        
        # Validate security compliance
        local security_issues=0
        validate_security_compliance "$service" || security_issues=$?
        total_issues=$((total_issues + security_issues))
        
        # Check service dependencies
        local dependency_issues=0
        check_service_dependencies "$service" || dependency_issues=$?
        total_issues=$((total_issues + dependency_issues))
        
        log INFO "Service $service monitoring completed"
        echo "---"
    done
    
    # Summary
    log INFO "Monitoring summary:"
    log INFO "  Total services: ${#SERVICES[@]}"
    log INFO "  Healthy services: $healthy_services"
    log INFO "  Total issues found: $total_issues"
    
    # Generate and send report
    local report_file=$(generate_health_report)
    
    if [[ "$total_issues" -gt 0 ]]; then
        send_alert "WARN" "Deployment monitoring found $total_issues issues across ${#SERVICES[@]} services. Healthy services: $healthy_services"
    else
        log SUCCESS "All services are healthy!"
    fi
    
    return $total_issues
}

continuous_monitoring() {
    log INFO "Starting continuous monitoring (interval: ${MONITORING_INTERVAL}s)"
    
    while true; do
        monitor_all_services
        log INFO "Waiting ${MONITORING_INTERVAL}s before next monitoring cycle..."
        sleep "$MONITORING_INTERVAL"
    done
}

#######################################
# Main execution
#######################################
usage() {
    cat << EOF
Advanced Deployment Monitoring Script v${SCRIPT_VERSION}
Developed by ${SCRIPT_AUTHOR}

Usage: $0 [OPTIONS] COMMAND [ARGS]

Commands:
    monitor                    Run single monitoring cycle
    continuous                 Run continuous monitoring
    health-check <service>     Check specific service health
    resource-check <service>   Check specific service resources
    security-check <service>   Check specific service security
    dependency-check <service> Check specific service dependencies
    generate-report           Generate comprehensive health report

Options:
    -n, --namespace NAMESPACE          Kubernetes namespace
    -e, --environment ENV              Environment
    -i, --interval SECONDS             Monitoring interval for continuous mode
    --cpu-threshold PERCENTAGE         CPU usage alert threshold
    --memory-threshold PERCENTAGE      Memory usage alert threshold
    --webhook-url URL                  Webhook URL for alerts
    --smtp-server SERVER               SMTP server for email alerts
    --alert-email EMAIL                Email address for alerts
    -h, --help                         Show this help

Environment Variables:
    NAMESPACE                  Kubernetes namespace
    ENVIRONMENT               Deployment environment
    MONITORING_INTERVAL       Monitoring interval in seconds
    ALERT_THRESHOLD_CPU       CPU usage alert threshold (%)
    ALERT_THRESHOLD_MEMORY    Memory usage alert threshold (%)
    PROMETHEUS_URL            Prometheus server URL
    GRAFANA_URL              Grafana server URL
    WEBHOOK_URL              Webhook URL for alerts
    SMTP_SERVER              SMTP server for email alerts
    ALERT_EMAIL              Email address for alerts

Examples:
    $0 monitor
    $0 continuous --interval 60
    $0 health-check spotify-ai-auth-service
    $0 security-check spotify-ai-billing-service
    $0 generate-report

EOF
}

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -i|--interval)
                MONITORING_INTERVAL="$2"
                shift 2
                ;;
            --cpu-threshold)
                ALERT_THRESHOLD_CPU="$2"
                shift 2
                ;;
            --memory-threshold)
                ALERT_THRESHOLD_MEMORY="$2"
                shift 2
                ;;
            --webhook-url)
                WEBHOOK_URL="$2"
                shift 2
                ;;
            --smtp-server)
                SMTP_SERVER="$2"
                shift 2
                ;;
            --alert-email)
                ALERT_EMAIL="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                break
                ;;
        esac
    done
    
    # Ensure we have a command
    if [[ $# -eq 0 ]]; then
        log ERROR "No command specified"
        usage
        exit 1
    fi
    
    local command=$1
    shift
    
    # Initialize
    log INFO "Starting Advanced Deployment Monitoring Script v${SCRIPT_VERSION}"
    log INFO "Author: ${SCRIPT_AUTHOR}"
    log INFO "Log file: ${LOG_FILE}"
    
    # Check prerequisites
    if ! command -v kubectl &> /dev/null; then
        log ERROR "kubectl is required but not installed"
        exit 1
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        log ERROR "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Execute command
    case $command in
        monitor)
            monitor_all_services
            ;;
        continuous)
            continuous_monitoring
            ;;
        health-check)
            if [[ $# -lt 1 ]]; then
                log ERROR "Service name required for health-check command"
                exit 1
            fi
            check_deployment_health "$1"
            ;;
        resource-check)
            if [[ $# -lt 1 ]]; then
                log ERROR "Service name required for resource-check command"
                exit 1
            fi
            monitor_resource_usage "$1"
            ;;
        security-check)
            if [[ $# -lt 1 ]]; then
                log ERROR "Service name required for security-check command"
                exit 1
            fi
            validate_security_compliance "$1"
            ;;
        dependency-check)
            if [[ $# -lt 1 ]]; then
                log ERROR "Service name required for dependency-check command"
                exit 1
            fi
            check_service_dependencies "$1"
            ;;
        generate-report)
            generate_health_report
            ;;
        *)
            log ERROR "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
    
    log INFO "Monitoring script completed"
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
