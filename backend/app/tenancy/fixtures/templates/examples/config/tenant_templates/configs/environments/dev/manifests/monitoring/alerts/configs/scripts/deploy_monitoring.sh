#!/bin/bash

# Script de Déploiement du Système de Monitoring - Spotify AI Agent
# =================================================================
# 
# Script complet d'installation et configuration du système de monitoring
# avancé avec alertes intelligentes et tableaux de bord automatiques
#
# Auteur: Fahed Mlaiel
# Version: 2.0.0

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/.."
LOG_FILE="/var/log/spotify-monitoring-deploy.log"
BACKUP_DIR="/var/backups/spotify-monitoring"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Variables par défaut
ENVIRONMENT="${1:-dev}"
TENANT_ID="${2:-default}"
DRY_RUN="${3:-false}"
FORCE_REINSTALL="${4:-false}"

# Fonctions utilitaires
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOG_FILE}"
    
    case $level in
        "ERROR")
            echo -e "${RED}❌ ${message}${NC}" >&2
            ;;
        "SUCCESS")
            echo -e "${GREEN}✅ ${message}${NC}"
            ;;
        "WARNING")
            echo -e "${YELLOW}⚠️  ${message}${NC}"
            ;;
        "INFO")
            echo -e "${BLUE}ℹ️  ${message}${NC}"
            ;;
    esac
}

check_prerequisites() {
    log "INFO" "Vérification des prérequis..."
    
    # Vérification des commandes requises
    local required_commands=("docker" "docker-compose" "python3" "pip3" "redis-cli" "curl" "jq")
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log "ERROR" "Commande requise manquante: $cmd"
            exit 1
        fi
    done
    
    # Vérification de Docker
    if ! docker info &> /dev/null; then
        log "ERROR" "Docker n'est pas en cours d'exécution"
        exit 1
    fi
    
    # Vérification des permissions
    if [[ $EUID -eq 0 ]] && [[ "$FORCE_REINSTALL" != "true" ]]; then
        log "WARNING" "Exécution en tant que root détectée. Utiliser --force pour continuer."
        read -p "Continuer quand même ? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    log "SUCCESS" "Tous les prérequis sont satisfaits"
}

create_directories() {
    log "INFO" "Création des répertoires nécessaires..."
    
    local directories=(
        "/opt/spotify-monitoring"
        "/opt/spotify-monitoring/config"
        "/opt/spotify-monitoring/data"
        "/opt/spotify-monitoring/logs"
        "/opt/spotify-monitoring/prometheus"
        "/opt/spotify-monitoring/grafana"
        "/opt/spotify-monitoring/redis"
        "${BACKUP_DIR}"
    )
    
    for dir in "${directories[@]}"; do
        if [[ "$DRY_RUN" == "true" ]]; then
            log "INFO" "[DRY RUN] Créerait le répertoire: $dir"
        else
            sudo mkdir -p "$dir"
            sudo chown -R $USER:$USER "$dir"
            log "SUCCESS" "Répertoire créé: $dir"
        fi
    done
}

install_python_dependencies() {
    log "INFO" "Installation des dépendances Python..."
    
    local requirements_file="${CONFIG_DIR}/requirements.txt"
    
    if [[ ! -f "$requirements_file" ]]; then
        log "INFO" "Création du fichier requirements.txt..."
        cat > "$requirements_file" << 'EOF'
# Monitoring System Dependencies
asyncio>=3.4.3
aiohttp>=3.8.0
aioredis>=2.0.0
fastapi>=0.95.0
uvicorn>=0.20.0
pydantic>=1.10.0
prometheus-client>=0.16.0
redis>=4.5.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.2.0
tensorflow>=2.12.0
torch>=2.0.0
transformers>=4.28.0
pyyaml>=6.0
jinja2>=3.1.0
watchdog>=3.0.0
jsonschema>=4.17.0
cryptography>=40.0.0
psutil>=5.9.0
networkx>=3.1.0
scipy>=1.10.0
toml>=0.10.2
requests>=2.28.0
sqlalchemy>=2.0.0
alembic>=1.10.0
celery>=5.2.0
kombu>=5.2.0
flower>=1.2.0
gunicorn>=20.1.0
slack-sdk>=3.20.0
twilio>=8.2.0
email-validator>=2.0.0
python-multipart>=0.0.6
python-jose>=3.3.0
passlib>=1.7.4
bcrypt>=4.0.0
httpx>=0.24.0
websockets>=11.0.0
apscheduler>=3.10.0
EOF
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Installerait les dépendances Python"
    else
        # Création d'un environnement virtuel
        python3 -m venv /opt/spotify-monitoring/venv
        source /opt/spotify-monitoring/venv/bin/activate
        
        # Mise à jour de pip
        pip install --upgrade pip setuptools wheel
        
        # Installation des dépendances
        pip install -r "$requirements_file"
        
        log "SUCCESS" "Dépendances Python installées"
    fi
}

setup_docker_services() {
    log "INFO" "Configuration des services Docker..."
    
    local docker_compose_file="/opt/spotify-monitoring/docker-compose.yml"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Créerait le fichier docker-compose.yml"
    else
        cat > "$docker_compose_file" << 'EOF'
version: '3.8'

services:
  # Redis pour le cache et les sessions
  redis:
    image: redis:7-alpine
    container_name: spotify-monitoring-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - ./redis/data:/data
      - ./redis/redis.conf:/usr/local/etc/redis/redis.conf
    command: redis-server /usr/local/etc/redis/redis.conf
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - monitoring

  # Prometheus pour les métriques
  prometheus:
    image: prom/prometheus:latest
    container_name: spotify-monitoring-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./prometheus/rules:/etc/prometheus/rules
      - ./prometheus/data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9090/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - monitoring

  # Grafana pour les dashboards
  grafana:
    image: grafana/grafana:latest
    container_name: spotify-monitoring-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    volumes:
      - ./grafana/data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=spotify_admin_2024
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-piechart-panel,grafana-worldmap-panel
      - GF_FEATURE_TOGGLES_ENABLE=ngalert
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:3000/api/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - monitoring
    depends_on:
      - prometheus

  # AlertManager pour la gestion des alertes
  alertmanager:
    image: prom/alertmanager:latest
    container_name: spotify-monitoring-alertmanager
    restart: unless-stopped
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - ./alertmanager/data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
      - '--web.external-url=http://localhost:9093'
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9093/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - monitoring

  # Application de monitoring principal
  monitoring-app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: spotify-monitoring-app
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    environment:
      - ENVIRONMENT=${ENVIRONMENT:-dev}
      - TENANT_ID=${TENANT_ID:-default}
      - REDIS_URL=redis://redis:6379
      - PROMETHEUS_URL=http://prometheus:9090
      - GRAFANA_URL=http://grafana:3000
    depends_on:
      - redis
      - prometheus
      - grafana
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - monitoring

networks:
  monitoring:
    driver: bridge

volumes:
  prometheus_data:
  grafana_data:
  redis_data:
EOF

        log "SUCCESS" "Fichier docker-compose.yml créé"
    fi
}

create_configuration_files() {
    log "INFO" "Création des fichiers de configuration..."
    
    # Configuration Prometheus
    create_prometheus_config
    
    # Configuration Grafana
    create_grafana_config
    
    # Configuration AlertManager
    create_alertmanager_config
    
    # Configuration Redis
    create_redis_config
    
    # Configuration de l'application
    create_app_config
}

create_prometheus_config() {
    local prometheus_dir="/opt/spotify-monitoring/prometheus"
    mkdir -p "$prometheus_dir/rules"
    
    cat > "$prometheus_dir/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'spotify-monitoring'
    static_configs:
      - targets: ['monitoring-app:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

remote_write:
  - url: "http://localhost:8000/api/v1/metrics/write"
    queue_config:
      max_samples_per_send: 1000
      batch_send_deadline: 5s
EOF

    # Règles d'alertes
    cat > "$prometheus_dir/rules/spotify-alerts.yml" << 'EOF'
groups:
  - name: spotify.alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} for {{ $labels.service }}"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "95th percentile latency is {{ $value }}s"

      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service is down"
          description: "{{ $labels.job }} has been down for more than 1 minute"
EOF
}

create_grafana_config() {
    local grafana_dir="/opt/spotify-monitoring/grafana"
    mkdir -p "$grafana_dir/provisioning/datasources"
    mkdir -p "$grafana_dir/provisioning/dashboards"
    mkdir -p "$grafana_dir/dashboards"
    
    # Configuration des datasources
    cat > "$grafana_dir/provisioning/datasources/prometheus.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

    # Configuration des dashboards
    cat > "$grafana_dir/provisioning/dashboards/dashboard.yml" << 'EOF'
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF
}

create_alertmanager_config() {
    local alertmanager_dir="/opt/spotify-monitoring/alertmanager"
    mkdir -p "$alertmanager_dir"
    
    cat > "$alertmanager_dir/alertmanager.yml" << 'EOF'
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alertmanager@spotify-ai-agent.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
  - name: 'web.hook'
    webhook_configs:
      - url: 'http://monitoring-app:8000/api/v1/alerts/webhook'
        send_resolved: true

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'dev', 'instance']
EOF
}

create_redis_config() {
    local redis_dir="/opt/spotify-monitoring/redis"
    mkdir -p "$redis_dir"
    
    cat > "$redis_dir/redis.conf" << 'EOF'
# Configuration Redis pour Spotify Monitoring
port 6379
bind 0.0.0.0
protected-mode yes
timeout 0
keepalive 300
databases 16
save 900 1
save 300 10
save 60 10000
rdbcompression yes
rdbchecksum yes
maxmemory 2gb
maxmemory-policy allkeys-lru
appendonly yes
appendfsync everysec
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
EOF
}

create_app_config() {
    local config_dir="/opt/spotify-monitoring/config"
    mkdir -p "$config_dir"
    
    cat > "$config_dir/monitoring.yaml" << EOF
# Configuration du système de monitoring Spotify AI Agent
environment: ${ENVIRONMENT}
tenant_id: ${TENANT_ID}

# Configuration Redis
redis:
  url: redis://localhost:6379
  db: 0
  max_connections: 100

# Configuration Prometheus
prometheus:
  url: http://localhost:9090
  scrape_interval: 30s
  evaluation_interval: 30s

# Configuration Grafana
grafana:
  url: http://localhost:3000
  api_key: ""
  org_id: 1

# Configuration des alertes
alerts:
  correlation_window_minutes: 30
  max_correlations_per_alert: 50
  min_confidence_threshold: 0.6
  evaluation_interval: 30
  
# Configuration des notifications
notifications:
  channels:
    email:
      smtp_server: localhost
      smtp_port: 587
      from_address: noreply@spotify-ai-agent.com
    slack:
      webhook_url: ""
    teams:
      webhook_url: ""
  
  rate_limits:
    email: 100  # par heure
    slack: 1000  # par heure
    sms: 10  # par heure

# Configuration ML
machine_learning:
  models_path: /opt/spotify-monitoring/models
  prediction_threshold: 0.8
  learning_enabled: true
  
# Configuration de performance
performance:
  max_memory_mb: 2048
  cache_ttl_hours: 1
  batch_size: 1000
  
# Configuration de sécurité
security:
  encryption_key: "$(openssl rand -base64 32)"
  jwt_secret: "$(openssl rand -base64 64)"
  
# Configuration des tenants
tenants:
  default:
    name: "Default Tenant"
    services:
      api:
        sla_multiplier: 1.0
      ml:
        sla_multiplier: 1.5
      auth:
        sla_multiplier: 0.8
EOF
}

start_services() {
    log "INFO" "Démarrage des services..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Démarrerait les services Docker"
        return
    fi
    
    cd /opt/spotify-monitoring
    
    # Construction et démarrage
    docker-compose up -d --build
    
    # Attente que les services soient prêts
    local services=("redis" "prometheus" "grafana" "alertmanager")
    
    for service in "${services[@]}"; do
        log "INFO" "Attente que $service soit prêt..."
        local attempts=0
        local max_attempts=30
        
        while [[ $attempts -lt $max_attempts ]]; do
            if docker-compose ps | grep -q "$service.*Up"; then
                log "SUCCESS" "$service est prêt"
                break
            fi
            
            attempts=$((attempts + 1))
            sleep 2
        done
        
        if [[ $attempts -eq $max_attempts ]]; then
            log "ERROR" "$service n'a pas démarré correctement"
            exit 1
        fi
    done
}

verify_installation() {
    log "INFO" "Vérification de l'installation..."
    
    local services=(
        "http://localhost:9090/-/healthy:Prometheus"
        "http://localhost:3000/api/health:Grafana"
        "http://localhost:9093/-/healthy:AlertManager"
        "http://localhost:8000/health:Monitoring App"
    )
    
    local all_healthy=true
    
    for service_info in "${services[@]}"; do
        local url="${service_info%%:*}"
        local name="${service_info##*:}"
        
        if curl -f -s "$url" > /dev/null; then
            log "SUCCESS" "$name est accessible"
        else
            log "ERROR" "$name n'est pas accessible"
            all_healthy=false
        fi
    done
    
    # Test Redis
    if redis-cli ping | grep -q "PONG"; then
        log "SUCCESS" "Redis répond correctement"
    else
        log "ERROR" "Redis ne répond pas"
        all_healthy=false
    fi
    
    if [[ "$all_healthy" == "true" ]]; then
        log "SUCCESS" "Installation vérifiée avec succès!"
        print_access_info
    else
        log "ERROR" "Certains services ne fonctionnent pas correctement"
        exit 1
    fi
}

print_access_info() {
    echo
    echo "=========================================="
    echo "  Installation Terminée avec Succès! 🎉"
    echo "=========================================="
    echo
    echo "Services disponibles:"
    echo "  • Grafana:      http://localhost:3000 (admin/spotify_admin_2024)"
    echo "  • Prometheus:   http://localhost:9090"
    echo "  • AlertManager: http://localhost:9093"
    echo "  • Monitoring:   http://localhost:8000"
    echo "  • Redis:        localhost:6379"
    echo
    echo "Commandes utiles:"
    echo "  • Voir les logs:        docker-compose logs -f"
    echo "  • Redémarrer:          docker-compose restart"
    echo "  • Arrêter:             docker-compose down"
    echo "  • Mise à jour:         ./deploy_monitoring.sh $ENVIRONMENT $TENANT_ID"
    echo
    echo "Configuration:"
    echo "  • Dossier:             /opt/spotify-monitoring"
    echo "  • Logs:                /var/log/spotify-monitoring-deploy.log"
    echo "  • Sauvegardes:         $BACKUP_DIR"
    echo
}

create_systemd_service() {
    log "INFO" "Création du service systemd..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Créerait le service systemd"
        return
    fi
    
    sudo tee /etc/systemd/system/spotify-monitoring.service > /dev/null << 'EOF'
[Unit]
Description=Spotify AI Agent Monitoring System
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/spotify-monitoring
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable spotify-monitoring.service
    
    log "SUCCESS" "Service systemd créé et activé"
}

cleanup_old_installation() {
    if [[ "$FORCE_REINSTALL" == "true" ]]; then
        log "INFO" "Nettoyage de l'installation précédente..."
        
        # Arrêt des services
        cd /opt/spotify-monitoring 2>/dev/null && docker-compose down 2>/dev/null || true
        
        # Sauvegarde
        if [[ -d "/opt/spotify-monitoring" ]]; then
            sudo mv /opt/spotify-monitoring "${BACKUP_DIR}/spotify-monitoring_${TIMESTAMP}" 2>/dev/null || true
        fi
        
        log "SUCCESS" "Nettoyage terminé"
    fi
}

main() {
    echo "=========================================="
    echo "  Déploiement Spotify AI Agent Monitoring"
    echo "  Environnement: $ENVIRONMENT"
    echo "  Tenant: $TENANT_ID"
    echo "  Mode: $([[ "$DRY_RUN" == "true" ]] && echo "DRY RUN" || echo "PRODUCTION")"
    echo "=========================================="
    echo
    
    # Création du fichier de log
    sudo touch "$LOG_FILE"
    sudo chown $USER:$USER "$LOG_FILE"
    
    log "INFO" "Début du déploiement..."
    
    # Étapes du déploiement
    check_prerequisites
    cleanup_old_installation
    create_directories
    install_python_dependencies
    setup_docker_services
    create_configuration_files
    start_services
    create_systemd_service
    verify_installation
    
    log "SUCCESS" "Déploiement terminé avec succès!"
}

# Gestion des signaux
trap 'log "ERROR" "Script interrompu"; exit 1' INT TERM

# Aide
if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    echo "Usage: $0 [ENVIRONMENT] [TENANT_ID] [DRY_RUN] [FORCE_REINSTALL]"
    echo
    echo "Arguments:"
    echo "  ENVIRONMENT      Environnement (dev, staging, prod) [default: dev]"
    echo "  TENANT_ID        ID du tenant [default: default]"
    echo "  DRY_RUN          Mode simulation (true/false) [default: false]"
    echo "  FORCE_REINSTALL  Forcer la réinstallation (true/false) [default: false]"
    echo
    echo "Exemples:"
    echo "  $0                           # Installation par défaut"
    echo "  $0 prod spotify_main         # Installation pour production"
    echo "  $0 dev test_tenant true      # Simulation pour test"
    echo "  $0 dev tenant1 false true    # Réinstallation forcée"
    exit 0
fi

# Exécution du script principal
main "$@"
