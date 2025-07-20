#!/bin/bash
"""
Script de déploiement et configuration automatisée
Déploiement complet du système d'analytics d'alertes Spotify AI Agent
"""

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/tmp/spotify_ai_agent_deploy.log"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Variables d'environnement
ENVIRONMENT=${ENVIRONMENT:-development}
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-spotify_ai_agent}
DB_USER=${DB_USER:-postgres}
REDIS_HOST=${REDIS_HOST:-localhost}
REDIS_PORT=${REDIS_PORT:-6379}
KAFKA_BROKERS=${KAFKA_BROKERS:-localhost:9092}

# Fonctions utilitaires
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}" | tee -a "$LOG_FILE"
}

check_dependencies() {
    log "Vérification des dépendances système..."
    
    # Python 3.8+
    if ! command -v python3 &> /dev/null; then
        error "Python 3 requis mais non trouvé"
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
    if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
        error "Python 3.8+ requis, trouvé: $PYTHON_VERSION"
    fi
    info "Python version: $PYTHON_VERSION ✓"
    
    # PostgreSQL
    if ! command -v psql &> /dev/null; then
        warn "PostgreSQL client non trouvé - installation recommandée"
    else
        info "PostgreSQL client disponible ✓"
    fi
    
    # Redis
    if ! command -v redis-cli &> /dev/null; then
        warn "Redis CLI non trouvé - installation recommandée"
    else
        info "Redis CLI disponible ✓"
    fi
    
    # Docker (optionnel)
    if command -v docker &> /dev/null; then
        info "Docker disponible ✓"
    else
        warn "Docker non trouvé - déploiement conteneurisé indisponible"
    fi
    
    log "Vérification des dépendances terminée"
}

setup_python_environment() {
    log "Configuration de l'environnement Python..."
    
    cd "$PROJECT_ROOT"
    
    # Création de l'environnement virtuel
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        info "Environnement virtuel créé"
    fi
    
    # Activation
    source venv/bin/activate
    
    # Mise à jour pip
    pip install --upgrade pip
    
    # Installation des dépendances
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        info "Dépendances Python installées ✓"
    else
        # Installation manuelle des packages principaux
        pip install \
            asyncpg \
            aioredis \
            pydantic \
            fastapi \
            uvicorn \
            pandas \
            numpy \
            scikit-learn \
            tensorflow \
            prometheus-client \
            kafka-python \
            psutil \
            tabulate \
            openpyxl \
            uvloop
        info "Packages Python de base installés ✓"
    fi
    
    log "Environnement Python configuré"
}

setup_database() {
    log "Configuration de la base de données..."
    
    # Test de connectivité
    if ! PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "\q" 2>/dev/null; then
        error "Impossible de se connecter à PostgreSQL sur $DB_HOST:$DB_PORT"
    fi
    info "Connexion PostgreSQL OK ✓"
    
    # Création de la base si nécessaire
    if ! PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
        PGPASSWORD="$DB_PASS" createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME"
        info "Base de données $DB_NAME créée ✓"
    else
        info "Base de données $DB_NAME existe ✓"
    fi
    
    # Initialisation des tables
    source venv/bin/activate
    python3 admin_tools.py db init
    log "Base de données configurée"
}

setup_redis() {
    log "Configuration de Redis..."
    
    # Test de connectivité
    if ! redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping > /dev/null 2>&1; then
        error "Impossible de se connecter à Redis sur $REDIS_HOST:$REDIS_PORT"
    fi
    info "Connexion Redis OK ✓"
    
    # Configuration Redis pour analytics
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" CONFIG SET maxmemory-policy allkeys-lru
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" CONFIG SET save "900 1 300 10 60 10000"
    
    log "Redis configuré"
}

setup_configuration() {
    log "Configuration du système d'analytics..."
    
    # Création du fichier .env
    cat > "$PROJECT_ROOT/.env" << EOF
# Configuration Spotify AI Agent Analytics - $TIMESTAMP
ENVIRONMENT=$ENVIRONMENT
DEBUG=${DEBUG:-false}
LOG_LEVEL=${LOG_LEVEL:-INFO}

# Base de données
DATABASE_URL=postgresql://$DB_USER:$DB_PASS@$DB_HOST:$DB_PORT/$DB_NAME

# Redis
REDIS_URL=redis://$REDIS_HOST:$REDIS_PORT/0

# Machine Learning
ML_ANOMALY_DETECTION_ENABLED=true
ML_CORRELATION_ENABLED=true
ML_PREDICTION_ENABLED=true
ML_BATCH_SIZE=100
ML_ANOMALY_THRESHOLD=0.8

# Streaming
STREAMING_ENABLED=true
STREAMING_PROCESSING_MODE=real_time
STREAMING_BUFFER_SIZE=10000
STREAMING_KAFKA_ENABLED=${KAFKA_ENABLED:-false}
STREAMING_KAFKA_BOOTSTRAP_SERVERS=$KAFKA_BROKERS

# Performance
MAX_CONCURRENT_ANALYSES=100
ANALYSIS_TIMEOUT_SECONDS=300
SCALING_STRATEGY=auto_throughput

# Monitoring
ENABLE_PROMETHEUS_METRICS=true
PROMETHEUS_PORT=8000
HEALTH_CHECK_INTERVAL_SECONDS=60

# Sécurité
ENABLE_AUTHENTICATION=true
API_KEY_REQUIRED=true
RATE_LIMIT_PER_MINUTE=1000

# Rétention
ALERT_RETENTION_DAYS=90
ANALYTICS_RETENTION_DAYS=30
MODEL_RETENTION_DAYS=180
EOF
    
    info "Configuration générée: $PROJECT_ROOT/.env ✓"
    
    # Configuration spécifique par environnement
    case "$ENVIRONMENT" in
        production)
            cat >> "$PROJECT_ROOT/.env" << EOF

# Configuration Production
DEBUG=false
LOG_LEVEL=INFO
ML_BATCH_SIZE=1000
STREAMING_BUFFER_SIZE=100000
STREAMING_KAFKA_ENABLED=true
SCALING_STRATEGY=predictive
EOF
            ;;
        staging)
            cat >> "$PROJECT_ROOT/.env" << EOF

# Configuration Staging
DEBUG=false
LOG_LEVEL=INFO
ML_BATCH_SIZE=500
STREAMING_BUFFER_SIZE=50000
EOF
            ;;
        development)
            cat >> "$PROJECT_ROOT/.env" << EOF

# Configuration Development
DEBUG=true
LOG_LEVEL=DEBUG
ML_BATCH_SIZE=10
STREAMING_BUFFER_SIZE=1000
MAX_CONCURRENT_ANALYSES=10
EOF
            ;;
    esac
    
    log "Configuration système completée"
}

setup_systemd_service() {
    if [ "$ENVIRONMENT" != "production" ]; then
        warn "Service systemd configuré uniquement en production"
        return
    fi
    
    log "Configuration du service systemd..."
    
    # Création du fichier service
    sudo tee /etc/systemd/system/spotify-ai-agent-analytics.service > /dev/null << EOF
[Unit]
Description=Spotify AI Agent Analytics System
After=network.target postgresql.service redis.service
Requires=postgresql.service redis.service

[Service]
Type=simple
User=spotify-analytics
Group=spotify-analytics
WorkingDirectory=$PROJECT_ROOT
Environment=PATH=$PROJECT_ROOT/venv/bin
ExecStart=$PROJECT_ROOT/venv/bin/python run_analytics_system.py --environment production
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=spotify-ai-agent-analytics

# Sécurité
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$PROJECT_ROOT /tmp /var/log

# Ressources
LimitNOFILE=65536
MemoryMax=4G
CPUQuota=200%

[Install]
WantedBy=multi-user.target
EOF
    
    # Rechargement systemd
    sudo systemctl daemon-reload
    sudo systemctl enable spotify-ai-agent-analytics
    
    info "Service systemd configuré ✓"
    log "Service systemd installé"
}

setup_monitoring() {
    log "Configuration du monitoring..."
    
    # Configuration Prometheus (si disponible)
    if command -v prometheus &> /dev/null; then
        PROMETHEUS_CONFIG="/etc/prometheus/prometheus.yml"
        if [ -f "$PROMETHEUS_CONFIG" ]; then
            # Ajout du job analytics
            cat >> "$PROMETHEUS_CONFIG" << EOF

  - job_name: 'spotify-ai-agent-analytics'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 30s
    metrics_path: /metrics
    scheme: http
EOF
            info "Configuration Prometheus mise à jour ✓"
        fi
    fi
    
    # Configuration Grafana (dashboards)
    GRAFANA_DIR="/etc/grafana/provisioning/dashboards"
    if [ -d "$GRAFANA_DIR" ]; then
        cat > "$GRAFANA_DIR/spotify-ai-agent-analytics.json" << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "Spotify AI Agent - Analytics",
    "version": 1,
    "schemaVersion": 30,
    "panels": [
      {
        "id": 1,
        "title": "Alertes traitées/sec",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(stream_events_processed_total[5m])",
            "legendFormat": "Events/sec"
          }
        ]
      },
      {
        "id": 2,
        "title": "Score d'anomalie moyen",
        "type": "stat",
        "targets": [
          {
            "expr": "avg(alerts_anomaly_score)",
            "legendFormat": "Score moyen"
          }
        ]
      }
    ]
  }
}
EOF
        info "Dashboard Grafana configuré ✓"
    fi
    
    log "Monitoring configuré"
}

setup_log_rotation() {
    log "Configuration de la rotation des logs..."
    
    # Configuration logrotate
    sudo tee /etc/logrotate.d/spotify-ai-agent-analytics > /dev/null << EOF
$PROJECT_ROOT/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 spotify-analytics spotify-analytics
    postrotate
        systemctl reload spotify-ai-agent-analytics || true
    endscript
}
EOF
    
    info "Rotation des logs configurée ✓"
    log "Logs configurés"
}

run_health_checks() {
    log "Exécution des tests de santé..."
    
    source venv/bin/activate
    
    # Tests de base
    python3 admin_tools.py health
    
    # Test de performance
    info "Test de performance..."
    python3 -c "
import asyncio
import time
from config.analytics_config import get_analytics_config

async def perf_test():
    config = get_analytics_config()
    start = time.time()
    # Test basique de configuration
    assert config.environment is not None
    duration = time.time() - start
    print(f'Configuration chargée en {duration:.3f}s')

asyncio.run(perf_test())
"
    
    log "Tests de santé terminés"
}

deploy_docker() {
    if [ "$DEPLOYMENT_MODE" != "docker" ]; then
        return
    fi
    
    log "Déploiement Docker..."
    
    # Création du Dockerfile
    cat > "$PROJECT_ROOT/Dockerfile" << 'EOF'
FROM python:3.11-slim

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Dépendances système
RUN apt-get update && apt-get install -y \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Utilisateur non-root
RUN useradd --create-home --shell /bin/bash analytics

# Répertoire de travail
WORKDIR /app

# Copie des requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code
COPY . .
RUN chown -R analytics:analytics /app

# Utilisateur
USER analytics

# Port d'exposition
EXPOSE 8000

# Santé
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 admin_tools.py health || exit 1

# Commande par défaut
CMD ["python3", "run_analytics_system.py", "--environment", "production"]
EOF
    
    # Création docker-compose
    cat > "$PROJECT_ROOT/docker-compose.yml" << EOF
version: '3.8'

services:
  analytics:
    build: .
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/spotify_ai_agent
      - REDIS_URL=redis://redis:6379/0
      - ENVIRONMENT=production
    depends_on:
      - postgres
      - redis
    ports:
      - "8000:8000"
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=spotify_ai_agent
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

volumes:
  postgres_data:
  redis_data:
EOF
    
    # Construction et démarrage
    docker-compose build
    docker-compose up -d
    
    info "Déploiement Docker terminé ✓"
    log "Conteneurs démarrés"
}

show_deployment_summary() {
    log "Résumé du déploiement"
    
    echo ""
    echo "🎉 Déploiement terminé avec succès!"
    echo ""
    echo "📋 Résumé de la configuration:"
    echo "   Environnement: $ENVIRONMENT"
    echo "   Base de données: $DB_HOST:$DB_PORT/$DB_NAME"
    echo "   Redis: $REDIS_HOST:$REDIS_PORT"
    echo "   Répertoire: $PROJECT_ROOT"
    echo ""
    echo "🚀 Commandes utiles:"
    echo "   Démarrage: cd $PROJECT_ROOT && source venv/bin/activate && python3 run_analytics_system.py"
    echo "   Statut: python3 admin_tools.py status"
    echo "   Santé: python3 admin_tools.py health"
    echo "   Logs: tail -f $LOG_FILE"
    echo ""
    
    if [ "$ENVIRONMENT" = "production" ]; then
        echo "🔧 Service systemd:"
        echo "   Démarrage: sudo systemctl start spotify-ai-agent-analytics"
        echo "   Statut: sudo systemctl status spotify-ai-agent-analytics"
        echo "   Logs: sudo journalctl -u spotify-ai-agent-analytics -f"
        echo ""
    fi
    
    echo "📊 Monitoring:"
    echo "   Métriques Prometheus: http://localhost:8000/metrics"
    echo "   Health check: http://localhost:8000/health"
    echo ""
    
    echo "📝 Configuration: $PROJECT_ROOT/.env"
    echo "📋 Documentation: $PROJECT_ROOT/README.md"
    echo ""
}

# === MAIN ===

main() {
    echo "🚀 Déploiement Spotify AI Agent - Système d'Analytics"
    echo "================================================="
    
    # Vérification des variables requises
    if [ -z "${DB_PASS:-}" ]; then
        error "Variable DB_PASS requise"
    fi
    
    log "Début du déploiement pour l'environnement: $ENVIRONMENT"
    
    # Étapes de déploiement
    check_dependencies
    setup_python_environment
    setup_database
    setup_redis
    setup_configuration
    
    # Spécifique à l'environnement
    case "$ENVIRONMENT" in
        production)
            setup_systemd_service
            setup_monitoring
            setup_log_rotation
            ;;
        docker)
            deploy_docker
            ;;
    esac
    
    # Tests finaux
    run_health_checks
    
    # Résumé
    show_deployment_summary
    
    log "Déploiement terminé avec succès"
}

# Gestion des signaux
trap 'error "Déploiement interrompu"' INT TERM

# Point d'entrée
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
