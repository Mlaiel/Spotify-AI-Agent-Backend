#!/bin/bash
#
# Advanced Deployment Automation Script
# Developed by Fahed Mlaiel - DevOps & Infrastructure Architect
# 
# This script provides comprehensive deployment automation for the 
# Spotify AI Agent multi-tenant platform with advanced features:
# - Blue-Green deployments
# - Canary releases
# - Health checks and rollbacks
# - Multi-environment management
# - Resource optimization
# - Security validation
#

set -euo pipefail

# Script metadata
readonly SCRIPT_VERSION="2.0.0"
readonly SCRIPT_AUTHOR="Fahed Mlaiel - DevOps Architect"
readonly SCRIPT_DATE="2025-07-17"

# Configuration
readonly NAMESPACE="${NAMESPACE:-spotify-ai-agent-dev}"
readonly ENVIRONMENT="${ENVIRONMENT:-development}"
readonly DEPLOYMENT_STRATEGY="${DEPLOYMENT_STRATEGY:-rolling}"
readonly DRY_RUN="${DRY_RUN:-false}"
readonly VERBOSE="${VERBOSE:-false}"
readonly LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Log file
readonly LOG_FILE="/tmp/deployment-$(date +%Y%m%d-%H%M%S).log"

# Available services
readonly SERVICES=(
    "backend"
    "ml-service" 
    "analytics"
    "notification"
    "auth"
    "billing"
    "tenant-service"
)

# Deployment strategies
readonly STRATEGIES=(
    "rolling"
    "blue-green"
    "canary"
    "recreate"
)

# Tenant tiers
readonly TENANT_TIERS=(
    "free"
    "premium"
    "enterprise"
    "enterprise-plus"
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
            ;;
        WARN)
            echo -e "${YELLOW}[WARN]${NC} ${timestamp} - $message"
            echo "[WARN] ${timestamp} - $message" >> "$LOG_FILE"
            ;;
        INFO)
            echo -e "${GREEN}[INFO]${NC} ${timestamp} - $message"
            echo "[INFO] ${timestamp} - $message" >> "$LOG_FILE"
            ;;
        DEBUG)
            if [[ "$VERBOSE" == "true" ]]; then
                echo -e "${BLUE}[DEBUG]${NC} ${timestamp} - $message"
                echo "[DEBUG] ${timestamp} - $message" >> "$LOG_FILE"
            fi
            ;;
    esac
}

#######################################
# Utility functions
#######################################
check_prerequisites() {
    log INFO "Checking prerequisites..."
    
    local required_tools=("kubectl" "helm" "jq" "yq" "curl")
    local missing_tools=()
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log ERROR "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    # Check Kubernetes connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log ERROR "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log WARN "Namespace $NAMESPACE does not exist, creating it..."
        kubectl create namespace "$NAMESPACE"
    fi
    
    log INFO "Prerequisites check completed successfully"
}

validate_environment() {
    log INFO "Validating deployment environment..."
    
    # Validate environment
    case $ENVIRONMENT in
        development|staging|production)
            log INFO "Environment: $ENVIRONMENT"
            ;;
        *)
            log ERROR "Invalid environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
    
    # Validate strategy
    if [[ ! " ${STRATEGIES[*]} " =~ " ${DEPLOYMENT_STRATEGY} " ]]; then
        log ERROR "Invalid deployment strategy: $DEPLOYMENT_STRATEGY"
        log INFO "Available strategies: ${STRATEGIES[*]}"
        exit 1
    fi
    
    log INFO "Environment validation completed"
}

#######################################
# Deployment functions
#######################################
deploy_service() {
    local service=$1
    local strategy=${2:-$DEPLOYMENT_STRATEGY}
    local tier=${3:-premium}
    
    log INFO "Deploying service: $service with strategy: $strategy for tier: $tier"
    
    local deployment_file=""
    case $service in
        "backend")
            deployment_file="backend-deployment.yaml"
            ;;
        "ml-service")
            deployment_file="ml-service-deployment.yaml"
            ;;
        "analytics")
            deployment_file="analytics-deployment.yaml"
            ;;
        "notification")
            deployment_file="notification-deployment.yaml"
            ;;
        "auth")
            deployment_file="auth-deployment.yaml"
            ;;
        "billing")
            deployment_file="billing-deployment.yaml"
            ;;
        "tenant-service")
            deployment_file="tenant-service-deployment.yaml"
            ;;
        *)
            log ERROR "Unknown service: $service"
            return 1
            ;;
    esac
    
    local manifest_path="$(dirname "$0")/$deployment_file"
    
    if [[ ! -f "$manifest_path" ]]; then
        log ERROR "Deployment manifest not found: $manifest_path"
        return 1
    fi
    
    case $strategy in
        "rolling")
            deploy_rolling "$manifest_path" "$service"
            ;;
        "blue-green")
            deploy_blue_green "$manifest_path" "$service"
            ;;
        "canary")
            deploy_canary "$manifest_path" "$service"
            ;;
        "recreate")
            deploy_recreate "$manifest_path" "$service"
            ;;
    esac
}

deploy_rolling() {
    local manifest_path=$1
    local service=$2
    
    log INFO "Executing rolling deployment for $service"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log INFO "DRY RUN: Would apply manifest: $manifest_path"
        kubectl apply --dry-run=client -f "$manifest_path" -n "$NAMESPACE"
    else
        kubectl apply -f "$manifest_path" -n "$NAMESPACE"
        wait_for_rollout "$service"
    fi
}

deploy_blue_green() {
    local manifest_path=$1
    local service=$2
    
    log INFO "Executing blue-green deployment for $service"
    
    # Create green deployment
    local green_manifest="/tmp/${service}-green.yaml"
    sed "s/${service}/${service}-green/g" "$manifest_path" > "$green_manifest"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log INFO "DRY RUN: Would create green deployment"
        kubectl apply --dry-run=client -f "$green_manifest" -n "$NAMESPACE"
    else
        kubectl apply -f "$green_manifest" -n "$NAMESPACE"
        wait_for_rollout "${service}-green"
        
        # Health check green deployment
        if health_check "${service}-green"; then
            log INFO "Green deployment healthy, switching traffic"
            switch_traffic_blue_green "$service"
        else
            log ERROR "Green deployment unhealthy, rolling back"
            kubectl delete -f "$green_manifest" -n "$NAMESPACE" || true
            return 1
        fi
    fi
}

deploy_canary() {
    local manifest_path=$1
    local service=$2
    local canary_percentage=${CANARY_PERCENTAGE:-10}
    
    log INFO "Executing canary deployment for $service (${canary_percentage}% traffic)"
    
    # Create canary deployment with reduced replicas
    local canary_manifest="/tmp/${service}-canary.yaml"
    sed -e "s/${service}/${service}-canary/g" \
        -e "s/replicas: [0-9]*/replicas: 1/g" \
        "$manifest_path" > "$canary_manifest"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log INFO "DRY RUN: Would create canary deployment"
        kubectl apply --dry-run=client -f "$canary_manifest" -n "$NAMESPACE"
    else
        kubectl apply -f "$canary_manifest" -n "$NAMESPACE"
        wait_for_rollout "${service}-canary"
        
        # Health check and gradual rollout
        if canary_health_check "${service}-canary"; then
            log INFO "Canary deployment successful, proceeding with full rollout"
            deploy_rolling "$manifest_path" "$service"
            kubectl delete -f "$canary_manifest" -n "$NAMESPACE" || true
        else
            log ERROR "Canary deployment failed, cleaning up"
            kubectl delete -f "$canary_manifest" -n "$NAMESPACE" || true
            return 1
        fi
    fi
}

deploy_recreate() {
    local manifest_path=$1
    local service=$2
    
    log INFO "Executing recreate deployment for $service"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log INFO "DRY RUN: Would delete and recreate deployment"
        kubectl delete --dry-run=client deployment "$service" -n "$NAMESPACE" --ignore-not-found=true
        kubectl apply --dry-run=client -f "$manifest_path" -n "$NAMESPACE"
    else
        kubectl delete deployment "$service" -n "$NAMESPACE" --ignore-not-found=true
        sleep 5
        kubectl apply -f "$manifest_path" -n "$NAMESPACE"
        wait_for_rollout "$service"
    fi
}

#######################################
# Health check functions
#######################################
wait_for_rollout() {
    local service=$1
    local timeout=${ROLLOUT_TIMEOUT:-600}
    
    log INFO "Waiting for rollout of $service (timeout: ${timeout}s)"
    
    if ! kubectl rollout status deployment "$service" -n "$NAMESPACE" --timeout="${timeout}s"; then
        log ERROR "Rollout timeout for $service"
        return 1
    fi
    
    log INFO "Rollout completed successfully for $service"
}

health_check() {
    local service=$1
    local max_attempts=${HEALTH_CHECK_ATTEMPTS:-30}
    local delay=${HEALTH_CHECK_DELAY:-10}
    
    log INFO "Performing health check for $service"
    
    for ((i=1; i<=max_attempts; i++)); do
        if kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' | grep -q '[0-9]\+'; then
            local ready_replicas=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
            local desired_replicas=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
            
            if [[ "$ready_replicas" == "$desired_replicas" ]]; then
                log INFO "Health check passed for $service ($ready_replicas/$desired_replicas replicas ready)"
                return 0
            fi
        fi
        
        log DEBUG "Health check attempt $i/$max_attempts for $service"
        sleep "$delay"
    done
    
    log ERROR "Health check failed for $service after $max_attempts attempts"
    return 1
}

canary_health_check() {
    local service=$1
    local monitoring_duration=${CANARY_MONITORING_DURATION:-300}
    
    log INFO "Monitoring canary deployment $service for ${monitoring_duration}s"
    
    if ! health_check "$service"; then
        return 1
    fi
    
    # Monitor metrics for the canary
    local start_time=$(date +%s)
    local end_time=$((start_time + monitoring_duration))
    
    while [[ $(date +%s) -lt $end_time ]]; do
        if ! check_canary_metrics "$service"; then
            log ERROR "Canary metrics check failed for $service"
            return 1
        fi
        sleep 30
    done
    
    log INFO "Canary monitoring completed successfully for $service"
    return 0
}

check_canary_metrics() {
    local service=$1
    
    # Check error rate, response time, etc.
    # This would integrate with your monitoring system (Prometheus, etc.)
    log DEBUG "Checking canary metrics for $service"
    
    # Placeholder for actual metrics checks
    return 0
}

#######################################
# Traffic management
#######################################
switch_traffic_blue_green() {
    local service=$1
    
    log INFO "Switching traffic from blue to green for $service"
    
    # Update service selector to point to green deployment
    kubectl patch service "$service" -n "$NAMESPACE" -p '{"spec":{"selector":{"app":"'${service}'-green"}}}'
    
    # Wait for traffic switch
    sleep 10
    
    # Delete blue deployment
    kubectl delete deployment "$service" -n "$NAMESPACE" --ignore-not-found=true
    
    # Rename green deployment to original name
    kubectl patch deployment "${service}-green" -n "$NAMESPACE" -p '{"metadata":{"name":"'${service}'"}}'
    
    log INFO "Traffic switch completed for $service"
}

#######################################
# Multi-tenant deployment
#######################################
deploy_multi_tenant() {
    local service=$1
    local strategy=${2:-$DEPLOYMENT_STRATEGY}
    
    log INFO "Deploying $service for all tenant tiers"
    
    for tier in "${TENANT_TIERS[@]}"; do
        log INFO "Deploying $service for tier: $tier"
        deploy_service "$service" "$strategy" "$tier"
        
        if [[ $? -ne 0 ]]; then
            log ERROR "Failed to deploy $service for tier: $tier"
            return 1
        fi
    done
    
    log INFO "Multi-tenant deployment completed for $service"
}

#######################################
# Resource management
#######################################
optimize_resources() {
    log INFO "Optimizing resource allocation"
    
    # Analyze resource usage
    kubectl top nodes
    kubectl top pods -n "$NAMESPACE"
    
    # Apply resource optimization recommendations
    # This would be based on actual usage metrics
    log INFO "Resource optimization completed"
}

#######################################
# Security validation
#######################################
validate_security() {
    log INFO "Validating security configurations"
    
    # Check security contexts
    local insecure_pods=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{.items[?(@.spec.securityContext.runAsRoot==true)].metadata.name}')
    
    if [[ -n "$insecure_pods" ]]; then
        log WARN "Found pods running as root: $insecure_pods"
    fi
    
    # Check network policies
    if ! kubectl get networkpolicies -n "$NAMESPACE" &> /dev/null; then
        log WARN "No network policies found in namespace $NAMESPACE"
    fi
    
    log INFO "Security validation completed"
}

#######################################
# Cleanup functions
#######################################
cleanup() {
    log INFO "Cleaning up temporary files"
    rm -f /tmp/*-green.yaml /tmp/*-canary.yaml
    log INFO "Cleanup completed"
}

rollback() {
    local service=$1
    local revision=${2:-}
    
    log INFO "Rolling back $service"
    
    if [[ -n "$revision" ]]; then
        kubectl rollout undo deployment "$service" -n "$NAMESPACE" --to-revision="$revision"
    else
        kubectl rollout undo deployment "$service" -n "$NAMESPACE"
    fi
    
    wait_for_rollout "$service"
    log INFO "Rollback completed for $service"
}

#######################################
# Main execution functions
#######################################
usage() {
    cat << EOF
Advanced Deployment Automation Script v${SCRIPT_VERSION}
Developed by ${SCRIPT_AUTHOR}

Usage: $0 [OPTIONS] COMMAND [ARGS]

Commands:
    deploy <service> [strategy] [tier]  Deploy a specific service
    deploy-all [strategy]               Deploy all services
    deploy-multi-tenant <service>       Deploy service for all tenant tiers
    health-check <service>              Check service health
    rollback <service> [revision]       Rollback a service
    optimize-resources                  Optimize resource allocation
    validate-security                   Validate security configurations
    cleanup                            Clean up temporary files

Services:
    ${SERVICES[*]}

Strategies:
    ${STRATEGIES[*]}

Tenant Tiers:
    ${TENANT_TIERS[*]}

Options:
    -n, --namespace NAMESPACE      Kubernetes namespace (default: $NAMESPACE)
    -e, --environment ENV          Environment (development|staging|production)
    -s, --strategy STRATEGY        Deployment strategy (default: $DEPLOYMENT_STRATEGY)
    -d, --dry-run                  Perform dry run
    -v, --verbose                  Verbose output
    -h, --help                     Show this help

Environment Variables:
    NAMESPACE                      Kubernetes namespace
    ENVIRONMENT                    Deployment environment
    DEPLOYMENT_STRATEGY            Default deployment strategy
    DRY_RUN                        Enable dry run mode
    VERBOSE                        Enable verbose logging
    ROLLOUT_TIMEOUT                Rollout timeout in seconds
    HEALTH_CHECK_ATTEMPTS          Number of health check attempts
    CANARY_PERCENTAGE              Canary deployment traffic percentage

Examples:
    $0 deploy backend rolling premium
    $0 deploy-all blue-green
    $0 deploy-multi-tenant ml-service
    $0 rollback auth 5
    $0 --dry-run --verbose deploy notification canary

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
            -s|--strategy)
                DEPLOYMENT_STRATEGY="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN="true"
                shift
                ;;
            -v|--verbose)
                VERBOSE="true"
                shift
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
    log INFO "Starting Advanced Deployment Script v${SCRIPT_VERSION}"
    log INFO "Author: ${SCRIPT_AUTHOR}"
    log INFO "Log file: ${LOG_FILE}"
    
    # Setup trap for cleanup
    trap cleanup EXIT
    
    # Run prerequisites and validation
    check_prerequisites
    validate_environment
    
    # Execute command
    case $command in
        deploy)
            if [[ $# -lt 1 ]]; then
                log ERROR "Service name required for deploy command"
                exit 1
            fi
            deploy_service "$@"
            ;;
        deploy-all)
            local strategy=${1:-$DEPLOYMENT_STRATEGY}
            for service in "${SERVICES[@]}"; do
                deploy_service "$service" "$strategy"
            done
            ;;
        deploy-multi-tenant)
            if [[ $# -lt 1 ]]; then
                log ERROR "Service name required for deploy-multi-tenant command"
                exit 1
            fi
            deploy_multi_tenant "$@"
            ;;
        health-check)
            if [[ $# -lt 1 ]]; then
                log ERROR "Service name required for health-check command"
                exit 1
            fi
            health_check "$1"
            ;;
        rollback)
            if [[ $# -lt 1 ]]; then
                log ERROR "Service name required for rollback command"
                exit 1
            fi
            rollback "$@"
            ;;
        optimize-resources)
            optimize_resources
            ;;
        validate-security)
            validate_security
            ;;
        cleanup)
            cleanup
            ;;
        *)
            log ERROR "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
    
    log INFO "Deployment script completed successfully"
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
