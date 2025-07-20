#!/bin/bash

# Advanced Scripts System Startup Script
# Version: 3.0.0
# Developed by Spotify AI Agent Team
# Lead Developer & AI Architect: Fahed Mlaiel

set -euo pipefail

# ============================================================================
# Configuration Variables
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="${SCRIPT_DIR}/logs/startup.log"
PID_FILE="${SCRIPT_DIR}/scripts-system.pid"

# Default values
ENVIRONMENT="${ENV:-development}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
WORKERS="${WORKERS:-4}"
TIMEOUT="${TIMEOUT:-30}"
BIND_HOST="${BIND_HOST:-0.0.0.0}"
BIND_PORT="${BIND_PORT:-8000}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Utility Functions
# ============================================================================

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
        "ERROR")
            echo -e "${RED}[$timestamp] [ERROR] $message${NC}" >&2
            ;;
        "WARN")
            echo -e "${YELLOW}[$timestamp] [WARN] $message${NC}"
            ;;
        "INFO")
            echo -e "${GREEN}[$timestamp] [INFO] $message${NC}"
            ;;
        "DEBUG")
            if [[ "$LOG_LEVEL" == "DEBUG" ]]; then
                echo -e "${BLUE}[$timestamp] [DEBUG] $message${NC}"
            fi
            ;;
        *)
            echo "[$timestamp] [$level] $message"
            ;;
    esac
}

check_command() {
    local cmd="$1"
    if ! command -v "$cmd" &> /dev/null; then
        log "ERROR" "Required command '$cmd' not found"
        return 1
    fi
    return 0
}

check_port() {
    local port="$1"
    if netstat -tuln 2>/dev/null | grep -q ":$port "; then
        log "WARN" "Port $port is already in use"
        return 1
    fi
    return 0
}

wait_for_service() {
    local host="$1"
    local port="$2"
    local timeout="${3:-30}"
    local count=0
    
    log "INFO" "Waiting for service at $host:$port (timeout: ${timeout}s)"
    
    while ! nc -z "$host" "$port" 2>/dev/null; do
        if [ $count -ge $timeout ]; then
            log "ERROR" "Timeout waiting for service at $host:$port"
            return 1
        fi
        sleep 1
        ((count++))
    done
    
    log "INFO" "Service at $host:$port is ready"
    return 0
}

# ============================================================================
# System Checks
# ============================================================================

check_system() {
    log "INFO" "Performing system checks..."
    
    # Check required commands
    local required_commands=(
        "python3"
        "pip3"
        "docker"
        "docker-compose"
    )
    
    for cmd in "${required_commands[@]}"; do
        if ! check_command "$cmd"; then
            log "ERROR" "System check failed: missing $cmd"
            return 1
        fi
    done
    
    # Check Python version
    local python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    local min_version="3.9"
    
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
        log "ERROR" "Python version $python_version is too old. Minimum required: $min_version"
        return 1
    fi
    
    log "INFO" "Python version: $python_version"
    
    # Check available memory
    local available_mem=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    if [ "$available_mem" -lt 1024 ]; then
        log "WARN" "Low available memory: ${available_mem}MB"
    fi
    
    # Check disk space
    local available_disk=$(df . | awk 'NR==2{print $4}')
    if [ "$available_disk" -lt 1048576 ]; then  # 1GB in KB
        log "WARN" "Low available disk space: ${available_disk}KB"
    fi
    
    log "INFO" "System checks completed successfully"
    return 0
}

# ============================================================================
# Environment Setup
# ============================================================================

setup_environment() {
    log "INFO" "Setting up environment: $ENVIRONMENT"
    
    # Create necessary directories
    local dirs=(
        "logs"
        "data"
        "workspace"
        "configs"
        "templates"
        "artifacts"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "${SCRIPT_DIR}/$dir"
        log "DEBUG" "Created directory: $dir"
    done
    
    # Set environment variables
    export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
    export SCRIPTS_ROOT="$SCRIPT_DIR"
    export SCRIPTS_ENV="$ENVIRONMENT"
    export SCRIPTS_LOG_LEVEL="$LOG_LEVEL"
    
    # Load environment-specific configuration
    local env_file="${SCRIPT_DIR}/configs/${ENVIRONMENT}.env"
    if [[ -f "$env_file" ]]; then
        log "INFO" "Loading environment configuration: $env_file"
        # shellcheck source=/dev/null
        source "$env_file"
    else
        log "WARN" "Environment configuration file not found: $env_file"
    fi
    
    log "INFO" "Environment setup completed"
}

# ============================================================================
# Dependency Management
# ============================================================================

install_dependencies() {
    log "INFO" "Installing Python dependencies..."
    
    local requirements_file="${SCRIPT_DIR}/requirements.txt"
    if [[ ! -f "$requirements_file" ]]; then
        log "ERROR" "Requirements file not found: $requirements_file"
        return 1
    fi
    
    # Check if virtual environment exists
    local venv_dir="${SCRIPT_DIR}/venv"
    if [[ ! -d "$venv_dir" ]]; then
        log "INFO" "Creating virtual environment..."
        python3 -m venv "$venv_dir"
    fi
    
    # Activate virtual environment
    # shellcheck source=/dev/null
    source "${venv_dir}/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install requirements
    pip install -r "$requirements_file"
    
    log "INFO" "Dependencies installed successfully"
}

# ============================================================================
# Database Setup
# ============================================================================

setup_database() {
    local db_url="${DATABASE_URL:-}"
    
    if [[ -z "$db_url" ]]; then
        log "INFO" "No database URL configured, skipping database setup"
        return 0
    fi
    
    log "INFO" "Setting up database..."
    
    # Wait for database to be ready
    local db_host=$(echo "$db_url" | sed -n 's/.*@\([^:]*\):.*/\1/p')
    local db_port=$(echo "$db_url" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
    
    if [[ -n "$db_host" && -n "$db_port" ]]; then
        wait_for_service "$db_host" "$db_port" 60
    fi
    
    # Run database migrations
    if [[ "$RUN_MIGRATIONS" == "true" ]]; then
        log "INFO" "Running database migrations..."
        python3 -m alembic upgrade head
    fi
    
    log "INFO" "Database setup completed"
}

# ============================================================================
# Service Management
# ============================================================================

start_services() {
    log "INFO" "Starting Advanced Scripts System services..."
    
    # Check if port is available
    if ! check_port "$BIND_PORT"; then
        log "ERROR" "Cannot start service: port $BIND_PORT is already in use"
        return 1
    fi
    
    # Start main application
    local app_cmd="python3 -m uvicorn main:app"
    app_cmd="$app_cmd --host $BIND_HOST"
    app_cmd="$app_cmd --port $BIND_PORT"
    app_cmd="$app_cmd --workers $WORKERS"
    app_cmd="$app_cmd --log-level ${LOG_LEVEL,,}"
    
    if [[ "$ENVIRONMENT" == "development" ]]; then
        app_cmd="$app_cmd --reload"
    fi
    
    log "INFO" "Starting application: $app_cmd"
    
    # Start in background and capture PID
    nohup $app_cmd >> "$LOG_FILE" 2>&1 &
    local app_pid=$!
    
    # Save PID
    echo "$app_pid" > "$PID_FILE"
    
    # Wait for service to be ready
    sleep 5
    if ! wait_for_service "$BIND_HOST" "$BIND_PORT" 30; then
        log "ERROR" "Failed to start application service"
        return 1
    fi
    
    log "INFO" "Application started successfully (PID: $app_pid)"
    
    # Start additional services based on environment
    if [[ "$ENVIRONMENT" == "production" ]]; then
        start_worker_services
        start_monitoring_services
    fi
    
    return 0
}

start_worker_services() {
    log "INFO" "Starting worker services..."
    
    local celery_broker="${CELERY_BROKER_URL:-redis://localhost:6379/1}"
    
    # Start Celery worker
    local worker_cmd="celery -A main.celery worker"
    worker_cmd="$worker_cmd --loglevel=${LOG_LEVEL,,}"
    worker_cmd="$worker_cmd --concurrency=$WORKERS"
    
    nohup $worker_cmd >> "${LOG_FILE}.worker" 2>&1 &
    local worker_pid=$!
    echo "$worker_pid" > "${PID_FILE}.worker"
    
    # Start Celery beat scheduler
    local beat_cmd="celery -A main.celery beat"
    beat_cmd="$beat_cmd --loglevel=${LOG_LEVEL,,}"
    
    nohup $beat_cmd >> "${LOG_FILE}.beat" 2>&1 &
    local beat_pid=$!
    echo "$beat_pid" > "${PID_FILE}.beat"
    
    log "INFO" "Worker services started (Worker PID: $worker_pid, Beat PID: $beat_pid)"
}

start_monitoring_services() {
    log "INFO" "Starting monitoring services..."
    
    # Start Flower monitoring
    if [[ "$FLOWER_ENABLED" == "true" ]]; then
        local flower_cmd="celery -A main.celery flower"
        flower_cmd="$flower_cmd --port=${FLOWER_PORT:-5555}"
        
        nohup $flower_cmd >> "${LOG_FILE}.flower" 2>&1 &
        local flower_pid=$!
        echo "$flower_pid" > "${PID_FILE}.flower"
        
        log "INFO" "Flower monitoring started (PID: $flower_pid)"
    fi
}

# ============================================================================
# Health Checks
# ============================================================================

health_check() {
    log "INFO" "Performing health check..."
    
    local health_url="http://${BIND_HOST}:${BIND_PORT}/health"
    local response=$(curl -s -o /dev/null -w "%{http_code}" "$health_url" 2>/dev/null || echo "000")
    
    if [[ "$response" == "200" ]]; then
        log "INFO" "Health check passed"
        return 0
    else
        log "ERROR" "Health check failed (HTTP $response)"
        return 1
    fi
}

# ============================================================================
# Cleanup Functions
# ============================================================================

cleanup() {
    log "INFO" "Cleaning up..."
    
    # Kill processes
    local pid_files=("$PID_FILE" "${PID_FILE}.worker" "${PID_FILE}.beat" "${PID_FILE}.flower")
    
    for pid_file in "${pid_files[@]}"; do
        if [[ -f "$pid_file" ]]; then
            local pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                log "INFO" "Stopping process: $pid"
                kill "$pid"
                sleep 2
                if kill -0 "$pid" 2>/dev/null; then
                    log "WARN" "Force killing process: $pid"
                    kill -9 "$pid"
                fi
            fi
            rm -f "$pid_file"
        fi
    done
    
    log "INFO" "Cleanup completed"
}

# ============================================================================
# Signal Handlers
# ============================================================================

handle_signal() {
    local signal="$1"
    log "INFO" "Received signal: $signal"
    cleanup
    exit 0
}

# Set signal handlers
trap 'handle_signal SIGTERM' SIGTERM
trap 'handle_signal SIGINT' SIGINT
trap 'handle_signal SIGQUIT' SIGQUIT

# ============================================================================
# Main Function
# ============================================================================

main() {
    local action="${1:-start}"
    
    case "$action" in
        "start")
            log "INFO" "Starting Advanced Scripts System..."
            
            if ! check_system; then
                log "ERROR" "System checks failed"
                exit 1
            fi
            
            setup_environment
            
            if [[ "$SKIP_DEPS" != "true" ]]; then
                install_dependencies
            fi
            
            setup_database
            
            if ! start_services; then
                log "ERROR" "Failed to start services"
                cleanup
                exit 1
            fi
            
            # Wait a bit and perform health check
            sleep 10
            if ! health_check; then
                log "ERROR" "Health check failed after startup"
                cleanup
                exit 1
            fi
            
            log "INFO" "Advanced Scripts System started successfully"
            log "INFO" "Access the application at: http://${BIND_HOST}:${BIND_PORT}"
            
            # Keep the script running
            wait
            ;;
            
        "stop")
            log "INFO" "Stopping Advanced Scripts System..."
            cleanup
            ;;
            
        "restart")
            log "INFO" "Restarting Advanced Scripts System..."
            "$0" stop
            sleep 5
            "$0" start
            ;;
            
        "status")
            if [[ -f "$PID_FILE" ]]; then
                local pid=$(cat "$PID_FILE")
                if kill -0 "$pid" 2>/dev/null; then
                    log "INFO" "Advanced Scripts System is running (PID: $pid)"
                    health_check
                else
                    log "WARN" "PID file exists but process is not running"
                    exit 1
                fi
            else
                log "INFO" "Advanced Scripts System is not running"
                exit 1
            fi
            ;;
            
        "health")
            health_check
            ;;
            
        "logs")
            if [[ -f "$LOG_FILE" ]]; then
                tail -f "$LOG_FILE"
            else
                log "ERROR" "Log file not found: $LOG_FILE"
                exit 1
            fi
            ;;
            
        "help"|"-h"|"--help")
            cat << EOF
Advanced Scripts System Startup Script

Usage: $0 [COMMAND]

Commands:
    start     Start the application (default)
    stop      Stop the application
    restart   Restart the application
    status    Show application status
    health    Perform health check
    logs      Show application logs
    help      Show this help message

Environment Variables:
    ENV               Environment (development|testing|production)
    LOG_LEVEL         Log level (DEBUG|INFO|WARN|ERROR)
    BIND_HOST         Bind host (default: 0.0.0.0)
    BIND_PORT         Bind port (default: 8000)
    WORKERS           Number of workers (default: 4)
    SKIP_DEPS         Skip dependency installation (true|false)
    RUN_MIGRATIONS    Run database migrations (true|false)

Examples:
    $0 start
    ENV=production $0 start
    LOG_LEVEL=DEBUG $0 start
    $0 stop
    $0 status

EOF
            ;;
            
        *)
            log "ERROR" "Unknown command: $action"
            log "INFO" "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# ============================================================================
# Script Entry Point
# ============================================================================

# Change to script directory
cd "$SCRIPT_DIR"

# Run main function with all arguments
main "$@"
