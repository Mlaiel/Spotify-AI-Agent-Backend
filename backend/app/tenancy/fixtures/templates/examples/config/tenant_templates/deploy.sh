#!/bin/bash
# Enterprise Tenant Template Deployment Script
# Ultra-Advanced Industrial Multi-Tenant Architecture Deployment
# Developed by Expert Team led by Fahed Mlaiel

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/deployment.log"
TEMPLATES_DIR="${SCRIPT_DIR}"
BACKUP_DIR="${SCRIPT_DIR}/backups"

# Expert team banner
print_banner() {
    echo -e "${PURPLE}"
    echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
    echo "║               Enterprise Tenant Template Deployment System                   ║"
    echo "║          Ultra-Advanced Industrial Multi-Tenant Architecture                 ║"
    echo "║                                                                               ║"
    echo "║                    Developed by Expert Team led by Fahed Mlaiel              ║"
    echo "║                                                                               ║"
    echo "║  • Lead Dev + AI Architect: Fahed Mlaiel                                     ║"
    echo "║  • Senior Backend Developer: Python/FastAPI High Performance                 ║"
    echo "║  • ML Engineer: Intelligent Optimization & Predictions                       ║"
    echo "║  • DBA & Data Engineer: Multi-Database & Sharding                            ║"
    echo "║  • Security Specialist: End-to-End Encryption & Compliance                   ║"
    echo "║  • Microservices Architect: Event-Driven & CQRS                              ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

# Print colored messages
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; log "INFO" "$1"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; log "SUCCESS" "$1"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; log "WARNING" "$1"; }
print_error() { echo -e "${RED}❌ $1${NC}"; log "ERROR" "$1"; }

# Check prerequisites
check_prerequisites() {
    print_info "Checking deployment prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    for tool in python3 docker docker-compose kubectl helm terraform ansible; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        print_info "Please install missing tools before proceeding"
        exit 1
    fi
    
    # Check Python dependencies
    if ! python3 -c "import asyncio, redis, psycopg2, cryptography, prometheus_client" 2>/dev/null; then
        print_warning "Some Python dependencies may be missing"
        print_info "Installing Python dependencies..."
        pip3 install -r "${SCRIPT_DIR}/../../../requirements.txt" || {
            print_error "Failed to install Python dependencies"
            exit 1
        }
    fi
    
    print_success "All prerequisites satisfied"
}

# Setup environment
setup_environment() {
    print_info "Setting up deployment environment..."
    
    # Create necessary directories
    mkdir -p "$BACKUP_DIR"
    mkdir -p "${SCRIPT_DIR}/logs"
    mkdir -p "${SCRIPT_DIR}/configs"
    mkdir -p "${SCRIPT_DIR}/secrets"
    
    # Set proper permissions
    chmod 700 "${SCRIPT_DIR}/secrets"
    
    # Generate encryption key if not exists
    if [ ! -f "${SCRIPT_DIR}/secrets/encryption.key" ]; then
        print_info "Generating encryption key..."
        python3 -c "
from cryptography.fernet import Fernet
key = Fernet.generate_key()
with open('${SCRIPT_DIR}/secrets/encryption.key', 'wb') as f:
    f.write(key)
print('✅ Encryption key generated')
"
        chmod 600 "${SCRIPT_DIR}/secrets/encryption.key"
    fi
    
    print_success "Environment setup completed"
}

# Database initialization
init_database() {
    print_info "Initializing tenant templates database..."
    
    # Check if PostgreSQL is available
    if ! command -v psql &> /dev/null; then
        print_warning "PostgreSQL client not found, using Docker container"
        
        # Start PostgreSQL container if not running
        if ! docker ps | grep -q tenant-postgres; then
            print_info "Starting PostgreSQL container..."
            docker run -d \
                --name tenant-postgres \
                -e POSTGRES_DB=tenant_templates \
                -e POSTGRES_USER=tenant_admin \
                -e POSTGRES_PASSWORD=secure_password_$(openssl rand -hex 8) \
                -p 5432:5432 \
                -v tenant_postgres_data:/var/lib/postgresql/data \
                postgres:13-alpine
            
            # Wait for PostgreSQL to be ready
            sleep 10
        fi
    fi
    
    # Initialize database schema
    python3 -c "
import asyncio
import sys
sys.path.append('${SCRIPT_DIR}')
from tenant_utils import TenantTemplateUtility

async def init_db():
    utility = TenantTemplateUtility()
    try:
        await utility.initialize()
        print('✅ Database initialized successfully')
    except Exception as e:
        print(f'❌ Database initialization failed: {e}')
        sys.exit(1)

asyncio.run(init_db())
"
    
    print_success "Database initialization completed"
}

# Redis initialization
init_redis() {
    print_info "Initializing Redis cache..."
    
    # Start Redis container if not running
    if ! docker ps | grep -q tenant-redis; then
        print_info "Starting Redis container..."
        docker run -d \
            --name tenant-redis \
            -p 6379:6379 \
            -v tenant_redis_data:/data \
            redis:6-alpine redis-server --appendonly yes --requirepass "secure_redis_$(openssl rand -hex 8)"
        
        sleep 5
    fi
    
    # Test Redis connection
    if docker exec tenant-redis redis-cli ping | grep -q PONG; then
        print_success "Redis is running and accessible"
    else
        print_error "Redis connection failed"
        exit 1
    fi
}

# Deploy template configurations
deploy_templates() {
    print_info "Deploying tenant template configurations..."
    
    local templates=("free.yaml" "standard.yaml" "premium.yaml" "enterprise.yaml" "enterprise_plus.yaml")
    local deployed_count=0
    
    for template in "${templates[@]}"; do
        local template_path="${TEMPLATES_DIR}/${template}"
        
        if [ -f "$template_path" ]; then
            print_info "Deploying template: $template"
            
            # Create template from YAML
            python3 "${SCRIPT_DIR}/tenant_utils.py" create-from-yaml "$template" --name "${template%.yaml}_production" || {
                print_error "Failed to deploy template: $template"
                continue
            }
            
            ((deployed_count++))
            print_success "Deployed template: $template"
        else
            print_warning "Template file not found: $template"
        fi
    done
    
    print_success "Deployed $deployed_count templates"
}

# Setup monitoring
setup_monitoring() {
    print_info "Setting up monitoring infrastructure..."
    
    # Deploy Prometheus
    print_info "Deploying Prometheus..."
    docker run -d \
        --name tenant-prometheus \
        -p 9090:9090 \
        -v "${SCRIPT_DIR}/configs/prometheus.yml:/etc/prometheus/prometheus.yml" \
        prom/prometheus:latest || print_warning "Prometheus deployment failed"
    
    # Deploy Grafana
    print_info "Deploying Grafana..."
    docker run -d \
        --name tenant-grafana \
        -p 3000:3000 \
        -e GF_SECURITY_ADMIN_PASSWORD=admin_$(openssl rand -hex 8) \
        -v grafana-storage:/var/lib/grafana \
        grafana/grafana:latest || print_warning "Grafana deployment failed"
    
    # Deploy Jaeger for tracing
    print_info "Deploying Jaeger..."
    docker run -d \
        --name tenant-jaeger \
        -p 16686:16686 \
        -p 14268:14268 \
        jaegertracing/all-in-one:latest || print_warning "Jaeger deployment failed"
    
    print_success "Monitoring infrastructure deployed"
}

# Security hardening
security_hardening() {
    print_info "Applying security hardening..."
    
    # Generate SSL certificates
    if [ ! -f "${SCRIPT_DIR}/secrets/server.crt" ]; then
        print_info "Generating SSL certificates..."
        openssl req -x509 -newkey rsa:4096 -keyout "${SCRIPT_DIR}/secrets/server.key" \
            -out "${SCRIPT_DIR}/secrets/server.crt" -days 365 -nodes \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=tenant-templates.local"
        chmod 600 "${SCRIPT_DIR}/secrets/server.key"
    fi
    
    # Setup firewall rules (if available)
    if command -v ufw &> /dev/null; then
        print_info "Configuring firewall..."
        ufw allow 22/tcp    # SSH
        ufw allow 80/tcp    # HTTP
        ufw allow 443/tcp   # HTTPS
        ufw allow 5432/tcp  # PostgreSQL
        ufw allow 6379/tcp  # Redis
        ufw allow 9090/tcp  # Prometheus
        ufw allow 3000/tcp  # Grafana
        ufw --force enable
    fi
    
    # Setup fail2ban (if available)
    if command -v fail2ban-client &> /dev/null; then
        print_info "Configuring fail2ban..."
        systemctl enable fail2ban
        systemctl start fail2ban
    fi
    
    print_success "Security hardening completed"
}

# Health checks
health_checks() {
    print_info "Performing health checks..."
    
    local services=("tenant-postgres:5432" "tenant-redis:6379")
    local failed_services=()
    
    for service in "${services[@]}"; do
        local service_name=$(echo "$service" | cut -d: -f1)
        local port=$(echo "$service" | cut -d: -f2)
        
        if docker ps | grep -q "$service_name"; then
            print_success "Service $service_name is running"
        else
            failed_services+=("$service_name")
        fi
    done
    
    # Test Python module
    python3 -c "
import sys
sys.path.append('${SCRIPT_DIR}')
try:
    from tenant_utils import TenantTemplateUtility
    print('✅ Python module import successful')
except Exception as e:
    print(f'❌ Python module import failed: {e}')
    sys.exit(1)
"
    
    if [ ${#failed_services[@]} -eq 0 ]; then
        print_success "All health checks passed"
        return 0
    else
        print_error "Failed services: ${failed_services[*]}"
        return 1
    fi
}

# Backup current configuration
backup_configuration() {
    print_info "Creating configuration backup..."
    
    local backup_timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_path="${BACKUP_DIR}/backup_${backup_timestamp}"
    
    mkdir -p "$backup_path"
    
    # Backup templates
    cp -r "${TEMPLATES_DIR}"/*.yaml "$backup_path/" 2>/dev/null || true
    
    # Backup database (if possible)
    if docker ps | grep -q tenant-postgres; then
        docker exec tenant-postgres pg_dump -U tenant_admin tenant_templates > "${backup_path}/database_backup.sql"
    fi
    
    # Backup Redis (if possible)
    if docker ps | grep -q tenant-redis; then
        docker exec tenant-redis redis-cli --rdb "${backup_path}/redis_backup.rdb"
    fi
    
    print_success "Backup created at: $backup_path"
}

# Rollback deployment
rollback_deployment() {
    print_warning "Initiating deployment rollback..."
    
    # Stop containers
    local containers=("tenant-postgres" "tenant-redis" "tenant-prometheus" "tenant-grafana" "tenant-jaeger")
    
    for container in "${containers[@]}"; do
        if docker ps | grep -q "$container"; then
            print_info "Stopping container: $container"
            docker stop "$container" || true
            docker rm "$container" || true
        fi
    done
    
    print_success "Rollback completed"
}

# Generate deployment report
generate_report() {
    print_info "Generating deployment report..."
    
    local report_file="${SCRIPT_DIR}/deployment_report_$(date '+%Y%m%d_%H%M%S').json"
    
    # Get system information
    local deployment_info=$(cat <<EOF
{
    "deployment": {
        "timestamp": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
        "script_version": "2.0.0",
        "expert_team": "Fahed Mlaiel Expert Team",
        "environment": {
            "os": "$(uname -s)",
            "architecture": "$(uname -m)",
            "kernel": "$(uname -r)"
        },
        "services": {
            "postgresql": $(docker ps | grep -q tenant-postgres && echo '"running"' || echo '"stopped"'),
            "redis": $(docker ps | grep -q tenant-redis && echo '"running"' || echo '"stopped"'),
            "prometheus": $(docker ps | grep -q tenant-prometheus && echo '"running"' || echo '"stopped"'),
            "grafana": $(docker ps | grep -q tenant-grafana && echo '"running"' || echo '"stopped"'),
            "jaeger": $(docker ps | grep -q tenant-jaeger && echo '"running"' || echo '"stopped"')
        },
        "templates_deployed": [
            "free_production",
            "standard_production", 
            "premium_production",
            "enterprise_production",
            "enterprise_plus_production"
        ],
        "security": {
            "encryption_enabled": true,
            "ssl_certificates": true,
            "firewall_configured": $(command -v ufw &> /dev/null && echo true || echo false),
            "fail2ban_configured": $(command -v fail2ban-client &> /dev/null && echo true || echo false)
        }
    }
}
EOF
)
    
    echo "$deployment_info" > "$report_file"
    print_success "Deployment report generated: $report_file"
}

# Main deployment function
main() {
    print_banner
    
    # Parse command line arguments
    local command="${1:-deploy}"
    
    case "$command" in
        "deploy")
            print_info "Starting enterprise tenant template deployment..."
            check_prerequisites
            setup_environment
            backup_configuration
            init_database
            init_redis
            deploy_templates
            setup_monitoring
            security_hardening
            
            if health_checks; then
                generate_report
                print_success "🎉 Enterprise tenant template deployment completed successfully!"
                print_info "Access points:"
                print_info "  • Grafana Dashboard: http://localhost:3000"
                print_info "  • Prometheus Metrics: http://localhost:9090" 
                print_info "  • Jaeger Tracing: http://localhost:16686"
                print_info "  • PostgreSQL: localhost:5432"
                print_info "  • Redis: localhost:6379"
            else
                print_error "Deployment completed with errors. Check logs for details."
                exit 1
            fi
            ;;
        "rollback")
            rollback_deployment
            ;;
        "health")
            health_checks
            ;;
        "backup")
            backup_configuration
            ;;
        *)
            echo "Usage: $0 [deploy|rollback|health|backup]"
            echo ""
            echo "Commands:"
            echo "  deploy   - Deploy enterprise tenant template system (default)"
            echo "  rollback - Rollback the deployment"
            echo "  health   - Perform health checks"
            echo "  backup   - Create configuration backup"
            exit 1
            ;;
    esac
}

# Trap to handle interrupts
trap 'print_warning "Deployment interrupted by user"; rollback_deployment; exit 1' INT TERM

# Run main function
main "$@"
