#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Docker pour le module i18n Slack ultra-avancé

Ce module fournit une configuration Docker complète pour déployer
le système d'internationalisation en production avec:
- Support multi-stage pour optimisation
- Configuration Redis intégrée  
- Monitoring et métriques
- Secrets management
- Health checks avancés
- Auto-scaling et load balancing

Auteur: Expert Team
Version: 2.0.0
"""

# Dockerfile pour l'application principale
DOCKERFILE_CONTENT = '''
# Multi-stage build pour optimisation de taille
FROM python:3.11-slim as builder

# Métadonnées
LABEL maintainer="Fahed Mlaiel <fahed@example.com>"
LABEL version="2.0.0"
LABEL description="Ultra-advanced i18n Slack alerts system"

# Variables d'environnement pour le build
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Installation des dépendances système
RUN apt-get update && apt-get install -y \\
    build-essential \\
    libffi-dev \\
    libssl-dev \\
    libxml2-dev \\
    libxslt1-dev \\
    libjpeg-dev \\
    libfreetype6-dev \\
    zlib1g-dev \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Création du répertoire de travail
WORKDIR /app

# Copie des requirements
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --user --no-warn-script-location -r requirements.txt

# Stage de production
FROM python:3.11-slim as production

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO

# Installation des dépendances runtime uniquement
RUN apt-get update && apt-get install -y \\
    curl \\
    ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

# Création utilisateur non-root pour sécurité
RUN groupadd -r i18nuser && useradd -r -g i18nuser i18nuser

# Copie des dépendances depuis le builder
COPY --from=builder /root/.local /home/i18nuser/.local

# Mise à jour du PATH
ENV PATH=/home/i18nuser/.local/bin:$PATH

# Création des répertoires
WORKDIR /app
RUN mkdir -p /app/logs /app/cache /app/translations && \\
    chown -R i18nuser:i18nuser /app

# Copie du code source
COPY --chown=i18nuser:i18nuser . .

# Configuration des permissions
RUN chmod +x manage_i18n.sh

# Utilisateur non-root
USER i18nuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD python -c "import asyncio; from config import get_config; asyncio.run(get_config())" || exit 1

# Exposition du port
EXPOSE 8000

# Point d'entrée
ENTRYPOINT ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
'''

# Docker Compose pour l'environnement complet
DOCKER_COMPOSE_CONTENT = '''
version: '3.8'

services:
  # Service principal i18n
  i18n-slack:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    container_name: i18n-slack-alerts
    restart: unless-stopped
    environment:
      - ENVIRONMENT=production
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - REDIS_DB=2
      - MONITORING_ENABLED=true
      - AI_ENABLED=true
      - AI_API_KEY=${AI_API_KEY}
      - LOG_LEVEL=INFO
    ports:
      - "8000:8000"
    depends_on:
      redis:
        condition: service_healthy
      prometheus:
        condition: service_started
    networks:
      - i18n-network
    volumes:
      - ./logs:/app/logs
      - ./cache:/app/cache
      - ./translations:/app/translations
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.i18n.rule=Host(`i18n.localhost`)"
      - "traefik.http.services.i18n.loadbalancer.server.port=8000"

  # Redis pour le cache
  redis:
    image: redis:7.2-alpine
    container_name: i18n-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
      - ./redis.conf:/usr/local/etc/redis/redis.conf
    command: redis-server /usr/local/etc/redis/redis.conf
    networks:
      - i18n-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 3s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.25'
          memory: 256M

  # Prometheus pour les métriques
  prometheus:
    image: prom/prometheus:v2.48.0
    container_name: i18n-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    networks:
      - i18n-network
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G

  # Grafana pour la visualisation
  grafana:
    image: grafana/grafana:10.2.0
    container_name: i18n-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    networks:
      - i18n-network
    depends_on:
      - prometheus
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M

  # Jaeger pour le tracing distribué
  jaeger:
    image: jaegertracing/all-in-one:1.51
    container_name: i18n-jaeger
    restart: unless-stopped
    ports:
      - "16686:16686"
      - "14268:14268"
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    networks:
      - i18n-network
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M

  # Load balancer (HAProxy)
  loadbalancer:
    image: haproxy:2.8-alpine
    container_name: i18n-lb
    restart: unless-stopped
    ports:
      - "80:80"
      - "8080:8080"  # Stats
    volumes:
      - ./haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg
    depends_on:
      - i18n-slack
    networks:
      - i18n-network
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 256M

  # Elasticsearch pour les logs (optionnel)
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    container_name: i18n-elasticsearch
    restart: unless-stopped
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data
    networks:
      - i18n-network
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
    profiles:
      - logging

  # Kibana pour l'analyse des logs
  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    container_name: i18n-kibana
    restart: unless-stopped
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch
    networks:
      - i18n-network
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
    profiles:
      - logging

  # Backup service
  backup:
    image: alpine:3.18
    container_name: i18n-backup
    restart: "no"
    volumes:
      - redis-data:/backup/redis
      - ./logs:/backup/logs
      - ./cache:/backup/cache
      - ./backups:/backups
    command: |
      sh -c "
        apk add --no-cache tar gzip &&
        tar -czf /backups/i18n-backup-$$(date +%Y%m%d_%H%M%S).tar.gz /backup/ &&
        find /backups -name '*.tar.gz' -mtime +7 -delete
      "
    networks:
      - i18n-network
    profiles:
      - backup

# Volumes persistants
volumes:
  redis-data:
    driver: local
  prometheus-data:
    driver: local
  grafana-data:
    driver: local
  elasticsearch-data:
    driver: local

# Réseau dédié
networks:
  i18n-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
'''

# Configuration Redis optimisée
REDIS_CONFIG = '''
# Configuration Redis pour i18n Slack Alerts
# Optimisée pour performance et sécurité

# Réseau
bind 0.0.0.0
port 6379
timeout 300
tcp-keepalive 60

# Sécurité
protected-mode yes
requirepass ${REDIS_PASSWORD}

# Mémoire
maxmemory 1gb
maxmemory-policy allkeys-lru
maxmemory-samples 5

# Persistance
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /data

# Logs
loglevel notice
logfile /data/redis.log
syslog-enabled yes
syslog-ident redis

# Performance
databases 16
tcp-backlog 511
hz 10
dynamic-hz yes

# Clustering (pour le futur)
# cluster-enabled yes
# cluster-config-file nodes.conf
# cluster-node-timeout 15000

# Modules
loadmodule /usr/lib/redis/modules/redisearch.so
loadmodule /usr/lib/redis/modules/redisjson.so
'''

# Configuration Prometheus
PROMETHEUS_CONFIG = '''
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
  - job_name: 'i18n-slack'
    static_configs:
      - targets: ['i18n-slack:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
'''

# Configuration HAProxy
HAPROXY_CONFIG = '''
global
    daemon
    log stdout local0
    chroot /var/lib/haproxy
    stats socket /run/haproxy/admin.sock mode 660 level admin
    stats timeout 30s
    user haproxy
    group haproxy

defaults
    mode http
    log global
    option httplog
    option dontlognull
    option log-health-checks
    option forwardfor
    option httpchk GET /health
    timeout connect 5000
    timeout client 50000
    timeout server 50000
    retries 3

frontend i18n_frontend
    bind *:80
    capture request header Host len 32
    capture request header User-Agent len 64
    default_backend i18n_backend

backend i18n_backend
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200
    server i18n1 i18n-slack:8000 check inter 2000 rise 2 fall 3
    # server i18n2 i18n-slack-2:8000 check inter 2000 rise 2 fall 3

listen stats
    bind *:8080
    stats enable
    stats uri /stats
    stats refresh 30s
    stats admin if TRUE
'''

# Script de déploiement
DEPLOY_SCRIPT = '''#!/bin/bash
# Script de déploiement pour i18n Slack Alerts
set -euo pipefail

# Configuration
PROJECT_NAME="i18n-slack-alerts"
DOCKER_COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

# Couleurs pour les logs
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
NC='\\033[0m' # No Color

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Vérification des prérequis
check_prerequisites() {
    log "Vérification des prérequis..."
    
    command -v docker >/dev/null 2>&1 || error "Docker n'est pas installé"
    command -v docker-compose >/dev/null 2>&1 || error "Docker Compose n'est pas installé"
    
    # Vérification de la version Docker
    DOCKER_VERSION=$(docker --version | grep -oE '[0-9]+\\.[0-9]+\\.[0-9]+' | head -1)
    log "Version Docker: $DOCKER_VERSION"
    
    # Vérification de l'espace disque
    AVAILABLE_SPACE=$(df / | awk 'NR==2 {print $4}')
    if [ "$AVAILABLE_SPACE" -lt 5000000 ]; then
        warn "Espace disque faible: ${AVAILABLE_SPACE}KB disponible"
    fi
}

# Création du fichier .env si inexistant
create_env_file() {
    if [ ! -f "$ENV_FILE" ]; then
        log "Création du fichier .env..."
        cat > "$ENV_FILE" << EOF
# Configuration i18n Slack Alerts
ENVIRONMENT=production
AI_API_KEY=your_ai_api_key_here
REDIS_PASSWORD=$(openssl rand -base64 32)
GRAFANA_PASSWORD=$(openssl rand -base64 16)

# Monitoring
MONITORING_ENABLED=true
METRICS_ENABLED=true

# Sécurité
SECURITY_ENABLED=true
RATE_LIMITING_ENABLED=true

# Performance
PERFORMANCE_MODE=fast
MAX_CONCURRENT_REQUESTS=500
EOF
        warn "Fichier .env créé. Veuillez configurer les variables nécessaires."
    fi
}

# Build des images
build_images() {
    log "Construction des images Docker..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" build --no-cache
}

# Déploiement
deploy() {
    log "Déploiement du stack i18n..."
    
    # Arrêt des conteneurs existants
    docker-compose -f "$DOCKER_COMPOSE_FILE" down
    
    # Démarrage des services
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    # Attente de la disponibilité
    log "Attente de la disponibilité des services..."
    sleep 30
    
    # Vérification du health check
    check_health
}

# Vérification de santé
check_health() {
    log "Vérification de la santé des services..."
    
    local services=("i18n-slack" "redis" "prometheus" "grafana")
    
    for service in "${services[@]}"; do
        if docker-compose -f "$DOCKER_COMPOSE_FILE" ps "$service" | grep -q "healthy\\|Up"; then
            log "✅ $service est opérationnel"
        else
            error "❌ $service n'est pas disponible"
        fi
    done
}

# Backup
backup() {
    log "Création d'un backup..."
    
    local backup_dir="./backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup Redis
    docker exec i18n-redis redis-cli BGSAVE
    docker cp i18n-redis:/data/dump.rdb "$backup_dir/"
    
    # Backup logs
    cp -r ./logs "$backup_dir/"
    
    # Backup config
    cp -r ./cache "$backup_dir/"
    
    log "Backup créé dans: $backup_dir"
}

# Restore
restore() {
    local backup_path="$1"
    
    if [ ! -d "$backup_path" ]; then
        error "Répertoire de backup introuvable: $backup_path"
    fi
    
    log "Restauration depuis: $backup_path"
    
    # Arrêt des services
    docker-compose -f "$DOCKER_COMPOSE_FILE" stop
    
    # Restore Redis
    if [ -f "$backup_path/dump.rdb" ]; then
        docker cp "$backup_path/dump.rdb" i18n-redis:/data/
    fi
    
    # Restore logs et cache
    if [ -d "$backup_path/logs" ]; then
        cp -r "$backup_path/logs" ./
    fi
    
    if [ -d "$backup_path/cache" ]; then
        cp -r "$backup_path/cache" ./
    fi
    
    # Redémarrage
    docker-compose -f "$DOCKER_COMPOSE_FILE" start
    
    log "Restauration terminée"
}

# Monitoring
monitor() {
    log "Affichage du monitoring..."
    
    echo "📊 URLs de monitoring:"
    echo "  - Application: http://localhost:8000"
    echo "  - Métriques: http://localhost:8000/metrics"
    echo "  - Grafana: http://localhost:3000 (admin/admin)"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - HAProxy Stats: http://localhost:8080/stats"
    echo "  - Jaeger: http://localhost:16686"
    
    # Logs en temps réel
    docker-compose -f "$DOCKER_COMPOSE_FILE" logs -f i18n-slack
}

# Mise à jour
update() {
    log "Mise à jour du système..."
    
    # Pull des nouvelles images
    docker-compose -f "$DOCKER_COMPOSE_FILE" pull
    
    # Redéploiement
    deploy
}

# Nettoyage
cleanup() {
    log "Nettoyage du système..."
    
    # Arrêt et suppression des conteneurs
    docker-compose -f "$DOCKER_COMPOSE_FILE" down -v
    
    # Suppression des images non utilisées
    docker image prune -f
    docker volume prune -f
    
    log "Nettoyage terminé"
}

# Menu principal
show_help() {
    echo "Usage: $0 {deploy|backup|restore|monitor|update|cleanup|help}"
    echo
    echo "Commandes:"
    echo "  deploy   - Déploie le stack complet"
    echo "  backup   - Crée un backup"
    echo "  restore  - Restaure un backup"
    echo "  monitor  - Affiche les URLs de monitoring"
    echo "  update   - Met à jour le système"
    echo "  cleanup  - Nettoie le système"
    echo "  help     - Affiche cette aide"
}

# Point d'entrée principal
main() {
    case "${1:-help}" in
        deploy)
            check_prerequisites
            create_env_file
            build_images
            deploy
            monitor
            ;;
        backup)
            backup
            ;;
        restore)
            restore "${2:-}"
            ;;
        monitor)
            monitor
            ;;
        update)
            update
            ;;
        cleanup)
            cleanup
            ;;
        help|*)
            show_help
            ;;
    esac
}

main "$@"
'''

def create_docker_files(base_path: str = "."):
    """
    Crée tous les fichiers Docker nécessaires
    
    Args:
        base_path: Répertoire de base où créer les fichiers
    """
    import os
    from pathlib import Path
    
    base = Path(base_path)
    
    # Dockerfile principal
    (base / "Dockerfile").write_text(DOCKERFILE_CONTENT)
    
    # Docker Compose
    (base / "docker-compose.yml").write_text(DOCKER_COMPOSE_CONTENT)
    
    # Configuration Redis
    (base / "redis.conf").write_text(REDIS_CONFIG)
    
    # Configuration Prometheus
    (base / "prometheus.yml").write_text(PROMETHEUS_CONFIG)
    
    # Configuration HAProxy
    (base / "haproxy.cfg").write_text(HAPROXY_CONFIG)
    
    # Script de déploiement
    deploy_script_path = base / "deploy.sh"
    deploy_script_path.write_text(DEPLOY_SCRIPT)
    deploy_script_path.chmod(0o755)
    
    # Création des répertoires
    for directory in ["logs", "cache", "backups", "grafana/dashboards", "grafana/datasources"]:
        (base / directory).mkdir(parents=True, exist_ok=True)
    
    # .dockerignore
    (base / ".dockerignore").write_text("""
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git/
.mypy_cache/
.pytest_cache/
.hypothesis/
.DS_Store
*.swp
*.swo
*~
""")
    
    print("✅ Fichiers Docker créés avec succès!")
    print("📁 Structure créée:")
    print("  - Dockerfile")
    print("  - docker-compose.yml") 
    print("  - redis.conf")
    print("  - prometheus.yml")
    print("  - haproxy.cfg")
    print("  - deploy.sh")
    print("  - .dockerignore")
    print("\n🚀 Pour déployer: ./deploy.sh deploy")

if __name__ == "__main__":
    create_docker_files()
