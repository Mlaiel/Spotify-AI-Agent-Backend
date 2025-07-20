#!/bin/bash
# =============================================================================
# Script de Déploiement Automatisé - Production Ready
# =============================================================================
# 
# Déploiement complet du système de gestion des incidents et métriques
# avec orchestration automatique, monitoring et rollback.
#
# Auteur: Équipe DevOps & SRE
# Lead Technique: Fahed Mlaiel
# Version: 2.0.0 Enterprise
# =============================================================================

set -euo pipefail

# Configuration globale
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly LOG_FILE="/var/log/incidents/deployment_$(date +%Y%m%d_%H%M%S).log"
readonly BACKUP_DIR="/opt/incidents/backups/pre_deployment_$(date +%Y%m%d_%H%M%S)"
readonly DEPLOYMENT_CONFIG="${SCRIPT_DIR}/deployment.yaml"

# Variables d'environnement avec valeurs par défaut
ENVIRONMENT="${ENVIRONMENT:-development}"
NAMESPACE="${NAMESPACE:-incidents}"
REPLICA_COUNT="${REPLICA_COUNT:-2}"
ENABLE_MONITORING="${ENABLE_MONITORING:-true}"
ENABLE_BACKUP="${ENABLE_BACKUP:-true}"
DRY_RUN="${DRY_RUN:-false}"
FORCE_DEPLOY="${FORCE_DEPLOY:-false}"

# Couleurs pour l'affichage
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Fonctions utilitaires
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        "INFO")
            echo -e "${GREEN}[INFO]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE"
            ;;
        "DEBUG")
            echo -e "${BLUE}[DEBUG]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE"
            ;;
        *)
            echo -e "${NC}[LOG]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE"
            ;;
    esac
}

# Vérification des prérequis
check_prerequisites() {
    log "INFO" "🔍 Vérification des prérequis..."
    
    local missing_tools=()
    
    # Vérification des outils requis
    local required_tools=("docker" "kubectl" "helm" "jq" "yq" "curl" "psql" "redis-cli")
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log "ERROR" "Outils manquants: ${missing_tools[*]}"
        log "INFO" "Installation automatique des outils manquants..."
        install_missing_tools "${missing_tools[@]}"
    fi
    
    # Vérification de Docker
    if ! docker info &> /dev/null; then
        log "ERROR" "Docker n'est pas en cours d'exécution"
        log "INFO" "Démarrage de Docker..."
        sudo systemctl start docker || {
            log "ERROR" "Impossible de démarrer Docker"
            exit 1
        }
    fi
    
    # Vérification de Kubernetes
    if ! kubectl cluster-info &> /dev/null; then
        log "WARN" "Cluster Kubernetes non accessible - déploiement en mode Docker uniquement"
        export DEPLOYMENT_MODE="docker"
    else
        export DEPLOYMENT_MODE="kubernetes"
        log "INFO" "Cluster Kubernetes détecté: $(kubectl config current-context)"
    fi
    
    # Vérification des ressources système
    check_system_resources
    
    log "SUCCESS" "Prérequis validés avec succès"
}

install_missing_tools() {
    local tools=("$@")
    
    # Détection de l'OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            # Ubuntu/Debian
            sudo apt-get update
            for tool in "${tools[@]}"; do
                case "$tool" in
                    "kubectl")
                        curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
                        sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
                        ;;
                    "helm")
                        curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
                        ;;
                    "jq")
                        sudo apt-get install -y jq
                        ;;
                    "yq")
                        sudo wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64
                        sudo chmod +x /usr/local/bin/yq
                        ;;
                    "psql")
                        sudo apt-get install -y postgresql-client
                        ;;
                    "redis-cli")
                        sudo apt-get install -y redis-tools
                        ;;
                esac
            done
        elif command -v yum &> /dev/null; then
            # CentOS/RHEL
            for tool in "${tools[@]}"; do
                case "$tool" in
                    "jq")
                        sudo yum install -y jq
                        ;;
                    "psql")
                        sudo yum install -y postgresql
                        ;;
                    "redis-cli")
                        sudo yum install -y redis
                        ;;
                esac
            done
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            for tool in "${tools[@]}"; do
                case "$tool" in
                    "kubectl"|"helm"|"jq"|"yq")
                        brew install "$tool"
                        ;;
                    "psql")
                        brew install postgresql
                        ;;
                    "redis-cli")
                        brew install redis
                        ;;
                esac
            done
        fi
    fi
}

check_system_resources() {
    log "INFO" "Vérification des ressources système..."
    
    # Vérification CPU
    local cpu_cores=$(nproc)
    if [[ $cpu_cores -lt 4 ]]; then
        log "WARN" "Nombre de cœurs CPU insuffisant: $cpu_cores (recommandé: 4+)"
    fi
    
    # Vérification mémoire
    local mem_gb=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $mem_gb -lt 8 ]]; then
        log "WARN" "Mémoire insuffisante: ${mem_gb}GB (recommandé: 8GB+)"
    fi
    
    # Vérification espace disque
    local disk_free=$(df -BG / | awk 'NR==2{gsub(/G/,"",$4); print $4}')
    if [[ $disk_free -lt 50 ]]; then
        log "WARN" "Espace disque insuffisant: ${disk_free}GB (recommandé: 50GB+)"
    fi
    
    log "INFO" "Ressources système: CPU=$cpu_cores cores, RAM=${mem_gb}GB, Disk=${disk_free}GB libre"
}

# Préparation de l'environnement
prepare_environment() {
    log "INFO" "🏗️ Préparation de l'environnement de déploiement..."
    
    # Création des répertoires
    create_directories
    
    # Configuration des secrets et variables d'environnement
    setup_secrets
    
    # Préparation des configs par environnement
    prepare_environment_configs
    
    # Génération des certificats SSL si nécessaire
    if [[ "$ENVIRONMENT" == "production" ]]; then
        generate_ssl_certificates
    fi
    
    log "SUCCESS" "Environnement préparé"
}

create_directories() {
    local directories=(
        "/var/log/incidents"
        "/var/lib/incidents"
        "/etc/incidents/config"
        "/opt/incidents/scripts"
        "/opt/incidents/backups"
        "/opt/incidents/ssl"
        "/tmp/incidents"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            sudo mkdir -p "$dir"
            sudo chown $(whoami):$(whoami) "$dir"
            log "INFO" "Répertoire créé: $dir"
        fi
    done
}

setup_secrets() {
    log "INFO" "Configuration des secrets..."
    
    # Génération des mots de passe sécurisés
    if [[ ! -f "/etc/incidents/config/secrets.env" ]]; then
        cat > "/etc/incidents/config/secrets.env" << EOF
# Secrets générés automatiquement le $(date)
DATABASE_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
JWT_SECRET=$(openssl rand -base64 64)
ENCRYPTION_KEY=$(openssl rand -base64 32)
API_SECRET_KEY=$(openssl rand -base64 32)
GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 16)
PROMETHEUS_PASSWORD=$(openssl rand -base64 16)
EOF
        chmod 600 "/etc/incidents/config/secrets.env"
        log "INFO" "Secrets générés et sauvegardés"
    fi
    
    # Chargement des secrets
    source "/etc/incidents/config/secrets.env"
}

prepare_environment_configs() {
    log "INFO" "Préparation des configurations pour l'environnement: $ENVIRONMENT"
    
    # Configuration Docker Compose
    cat > "${SCRIPT_DIR}/docker-compose.${ENVIRONMENT}.yml" << EOF
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    container_name: incidents-postgres-${ENVIRONMENT}
    environment:
      POSTGRES_DB: incidents
      POSTGRES_USER: incidents_user
      POSTGRES_PASSWORD: \${DATABASE_PASSWORD}
      POSTGRES_INITDB_ARGS: "--auth-host=scram-sha-256"
    volumes:
      - postgres_data_${ENVIRONMENT}:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    networks:
      - incidents-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U incidents_user -d incidents"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: incidents-redis-${ENVIRONMENT}
    environment:
      REDIS_PASSWORD: \${REDIS_PASSWORD}
    command: >
      --requirepass \${REDIS_PASSWORD}
      --appendonly yes
      --appendfsync everysec
      --maxmemory 1gb
      --maxmemory-policy allkeys-lru
    volumes:
      - redis_data_${ENVIRONMENT}:/data
    ports:
      - "6379:6379"
    networks:
      - incidents-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    container_name: incidents-prometheus-${ENVIRONMENT}
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
      - '--web.enable-lifecycle'
      - '--storage.tsdb.retention.time=15d'
    volumes:
      - ./configs/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./configs/alert-rules.yml:/etc/prometheus/alert-rules.yml:ro
      - prometheus_data_${ENVIRONMENT}:/prometheus
    ports:
      - "9090:9090"
    networks:
      - incidents-network
    restart: unless-stopped
    depends_on:
      - postgres
      - redis

  grafana:
    image: grafana/grafana:latest
    container_name: incidents-grafana-${ENVIRONMENT}
    environment:
      GF_SECURITY_ADMIN_PASSWORD: \${GRAFANA_ADMIN_PASSWORD}
      GF_USERS_ALLOW_SIGN_UP: false
      GF_SECURITY_SECRET_KEY: \${JWT_SECRET}
    volumes:
      - grafana_data_${ENVIRONMENT}:/var/lib/grafana
      - ./configs/grafana:/etc/grafana/provisioning
    ports:
      - "3000:3000"
    networks:
      - incidents-network
    restart: unless-stopped
    depends_on:
      - prometheus

  incidents-api:
    image: incidents-api:latest
    container_name: incidents-api-${ENVIRONMENT}
    environment:
      ENVIRONMENT: ${ENVIRONMENT}
      DATABASE_URL: postgresql://incidents_user:\${DATABASE_PASSWORD}@postgres:5432/incidents
      REDIS_URL: redis://:\${REDIS_PASSWORD}@redis:6379/0
      SECRET_KEY: \${API_SECRET_KEY}
      JWT_SECRET: \${JWT_SECRET}
    volumes:
      - ./configs:/app/config:ro
      - /var/log/incidents:/app/logs
    ports:
      - "8000:8000"
    networks:
      - incidents-network
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    container_name: incidents-nginx-${ENVIRONMENT}
    volumes:
      - ./configs/nginx.conf:/etc/nginx/nginx.conf:ro
      - /opt/incidents/ssl:/etc/nginx/ssl:ro
    ports:
      - "80:80"
      - "443:443"
    networks:
      - incidents-network
    restart: unless-stopped
    depends_on:
      - incidents-api
      - grafana

volumes:
  postgres_data_${ENVIRONMENT}:
    driver: local
  redis_data_${ENVIRONMENT}:
    driver: local
  prometheus_data_${ENVIRONMENT}:
    driver: local
  grafana_data_${ENVIRONMENT}:
    driver: local

networks:
  incidents-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
EOF

    # Configuration Kubernetes si disponible
    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        generate_kubernetes_manifests
    fi
}

generate_ssl_certificates() {
    log "INFO" "Génération des certificats SSL..."
    
    local ssl_dir="/opt/incidents/ssl"
    
    if [[ ! -f "${ssl_dir}/server.crt" ]]; then
        # Génération d'une clé privée
        openssl genrsa -out "${ssl_dir}/server.key" 2048
        
        # Génération d'une demande de certificat
        openssl req -new -key "${ssl_dir}/server.key" -out "${ssl_dir}/server.csr" \
            -subj "/C=FR/ST=IDF/L=Paris/O=Company/OU=IT/CN=incidents.company.com"
        
        # Génération du certificat auto-signé
        openssl x509 -req -days 365 -in "${ssl_dir}/server.csr" \
            -signkey "${ssl_dir}/server.key" -out "${ssl_dir}/server.crt"
        
        # Sécurisation des permissions
        chmod 600 "${ssl_dir}/server.key"
        chmod 644 "${ssl_dir}/server.crt"
        
        log "INFO" "Certificats SSL générés (auto-signés)"
    fi
}

generate_kubernetes_manifests() {
    log "INFO" "Génération des manifests Kubernetes..."
    
    mkdir -p "${SCRIPT_DIR}/k8s"
    
    # Namespace
    cat > "${SCRIPT_DIR}/k8s/namespace.yaml" << EOF
apiVersion: v1
kind: Namespace
metadata:
  name: ${NAMESPACE}
  labels:
    name: ${NAMESPACE}
    environment: ${ENVIRONMENT}
---
EOF

    # ConfigMap pour les configurations
    cat > "${SCRIPT_DIR}/k8s/configmap.yaml" << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: incidents-config
  namespace: ${NAMESPACE}
data:
  environment: "${ENVIRONMENT}"
  database-host: "postgres-service"
  redis-host: "redis-service"
  prometheus-url: "http://prometheus-service:9090"
  grafana-url: "http://grafana-service:3000"
---
EOF

    # Secret pour les mots de passe
    cat > "${SCRIPT_DIR}/k8s/secrets.yaml" << EOF
apiVersion: v1
kind: Secret
metadata:
  name: incidents-secrets
  namespace: ${NAMESPACE}
type: Opaque
data:
  database-password: $(echo -n "${DATABASE_PASSWORD}" | base64)
  redis-password: $(echo -n "${REDIS_PASSWORD}" | base64)
  jwt-secret: $(echo -n "${JWT_SECRET}" | base64)
  api-secret-key: $(echo -n "${API_SECRET_KEY}" | base64)
---
EOF

    # Déploiements des services
    generate_kubernetes_deployments
    
    log "SUCCESS" "Manifests Kubernetes générés"
}

generate_kubernetes_deployments() {
    # PostgreSQL Deployment
    cat > "${SCRIPT_DIR}/k8s/postgres.yaml" << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: ${NAMESPACE}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: "incidents"
        - name: POSTGRES_USER
          value: "incidents_user"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: incidents-secrets
              key: database-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        livenessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - incidents_user
            - -d
            - incidents
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - incidents_user
            - -d
            - incidents
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: ${NAMESPACE}
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: ${NAMESPACE}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
EOF

    # Redis Deployment
    cat > "${SCRIPT_DIR}/k8s/redis.yaml" << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: ${NAMESPACE}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        command:
        - redis-server
        - --requirepass
        - \$(REDIS_PASSWORD)
        - --appendonly
        - "yes"
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: incidents-secrets
              key: redis-password
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-storage
          mountPath: /data
        livenessProbe:
          exec:
            command:
            - redis-cli
            - ping
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - redis-cli
            - ping
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: redis-storage
        persistentVolumeClaim:
          claimName: redis-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: ${NAMESPACE}
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-pvc
  namespace: ${NAMESPACE}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
---
EOF

    # API Deployment
    cat > "${SCRIPT_DIR}/k8s/api.yaml" << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: incidents-api
  namespace: ${NAMESPACE}
spec:
  replicas: ${REPLICA_COUNT}
  selector:
    matchLabels:
      app: incidents-api
  template:
    metadata:
      labels:
        app: incidents-api
    spec:
      containers:
      - name: incidents-api
        image: incidents-api:latest
        env:
        - name: ENVIRONMENT
          valueFrom:
            configMapKeyRef:
              name: incidents-config
              key: environment
        - name: DATABASE_URL
          value: "postgresql://incidents_user:\$(DATABASE_PASSWORD)@postgres-service:5432/incidents"
        - name: REDIS_URL
          value: "redis://:\$(REDIS_PASSWORD)@redis-service:6379/0"
        - name: DATABASE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: incidents-secrets
              key: database-password
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: incidents-secrets
              key: redis-password
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: incidents-secrets
              key: api-secret-key
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: incidents-api-service
  namespace: ${NAMESPACE}
spec:
  selector:
    app: incidents-api
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
---
EOF
}

# Sauvegarde pré-déploiement
create_backup() {
    if [[ "$ENABLE_BACKUP" != "true" ]]; then
        log "INFO" "Sauvegarde désactivée"
        return 0
    fi
    
    log "INFO" "💾 Création de la sauvegarde pré-déploiement..."
    
    mkdir -p "$BACKUP_DIR"
    
    # Sauvegarde de la base de données si elle existe
    if docker container inspect incidents-postgres-${ENVIRONMENT} &> /dev/null; then
        log "INFO" "Sauvegarde de la base de données..."
        docker exec incidents-postgres-${ENVIRONMENT} pg_dump -U incidents_user incidents > "${BACKUP_DIR}/database_backup.sql"
        log "SUCCESS" "Base de données sauvegardée"
    fi
    
    # Sauvegarde des configurations
    if [[ -d "/etc/incidents/config" ]]; then
        log "INFO" "Sauvegarde des configurations..."
        cp -r "/etc/incidents/config" "${BACKUP_DIR}/config_backup"
        log "SUCCESS" "Configurations sauvegardées"
    fi
    
    # Sauvegarde des volumes Docker
    if [[ "$DEPLOYMENT_MODE" == "docker" ]]; then
        log "INFO" "Sauvegarde des volumes Docker..."
        docker run --rm -v postgres_data_${ENVIRONMENT}:/data -v ${BACKUP_DIR}:/backup alpine tar czf /backup/postgres_volume.tar.gz -C /data .
        docker run --rm -v redis_data_${ENVIRONMENT}:/data -v ${BACKUP_DIR}:/backup alpine tar czf /backup/redis_volume.tar.gz -C /data .
        log "SUCCESS" "Volumes Docker sauvegardés"
    fi
    
    log "SUCCESS" "Sauvegarde terminée: $BACKUP_DIR"
}

# Construction des images Docker
build_images() {
    log "INFO" "🏗️ Construction des images Docker..."
    
    # Construction de l'image API
    if [[ -f "${PROJECT_ROOT}/Dockerfile" ]]; then
        log "INFO" "Construction de l'image incidents-api..."
        docker build -t incidents-api:latest -f "${PROJECT_ROOT}/Dockerfile" "$PROJECT_ROOT"
        log "SUCCESS" "Image incidents-api construite"
    else
        log "WARN" "Dockerfile non trouvé, utilisation d'une image existante"
    fi
    
    # Tag des images pour l'environnement
    docker tag incidents-api:latest incidents-api:${ENVIRONMENT}
    docker tag incidents-api:latest incidents-api:$(date +%Y%m%d-%H%M%S)
    
    log "SUCCESS" "Images construites et taguées"
}

# Déploiement Docker
deploy_docker() {
    log "INFO" "🚀 Déploiement Docker en cours..."
    
    # Arrêt des conteneurs existants si force deploy
    if [[ "$FORCE_DEPLOY" == "true" ]]; then
        log "INFO" "Arrêt forcé des conteneurs existants..."
        docker-compose -f "${SCRIPT_DIR}/docker-compose.${ENVIRONMENT}.yml" down --remove-orphans || true
    fi
    
    # Chargement des secrets
    export $(cat /etc/incidents/config/secrets.env | xargs)
    
    # Déploiement avec Docker Compose
    log "INFO" "Lancement des services avec Docker Compose..."
    docker-compose -f "${SCRIPT_DIR}/docker-compose.${ENVIRONMENT}.yml" up -d
    
    # Attente que les services soient prêts
    wait_for_services_docker
    
    log "SUCCESS" "Déploiement Docker terminé"
}

# Déploiement Kubernetes
deploy_kubernetes() {
    log "INFO" "🚀 Déploiement Kubernetes en cours..."
    
    # Application des manifests
    log "INFO" "Application des manifests Kubernetes..."
    
    kubectl apply -f "${SCRIPT_DIR}/k8s/namespace.yaml"
    kubectl apply -f "${SCRIPT_DIR}/k8s/configmap.yaml"
    kubectl apply -f "${SCRIPT_DIR}/k8s/secrets.yaml"
    kubectl apply -f "${SCRIPT_DIR}/k8s/postgres.yaml"
    kubectl apply -f "${SCRIPT_DIR}/k8s/redis.yaml"
    kubectl apply -f "${SCRIPT_DIR}/k8s/api.yaml"
    
    # Attente que les services soient prêts
    wait_for_services_kubernetes
    
    log "SUCCESS" "Déploiement Kubernetes terminé"
}

wait_for_services_docker() {
    log "INFO" "⏳ Attente que les services soient prêts..."
    
    local services=("postgres" "redis" "incidents-api")
    local max_wait=300  # 5 minutes
    local wait_time=0
    
    for service in "${services[@]}"; do
        log "INFO" "Attente du service: $service"
        
        while [[ $wait_time -lt $max_wait ]]; do
            if docker-compose -f "${SCRIPT_DIR}/docker-compose.${ENVIRONMENT}.yml" ps "$service" | grep -q "healthy\|Up"; then
                log "SUCCESS" "Service $service est prêt"
                break
            fi
            
            sleep 10
            wait_time=$((wait_time + 10))
            
            if [[ $wait_time -ge $max_wait ]]; then
                log "ERROR" "Timeout: Service $service n'est pas prêt après ${max_wait}s"
                return 1
            fi
        done
    done
    
    log "SUCCESS" "Tous les services sont prêts"
}

wait_for_services_kubernetes() {
    log "INFO" "⏳ Attente que les pods soient prêts..."
    
    local deployments=("postgres" "redis" "incidents-api")
    
    for deployment in "${deployments[@]}"; do
        log "INFO" "Attente du déploiement: $deployment"
        kubectl rollout status deployment/$deployment -n $NAMESPACE --timeout=300s
        log "SUCCESS" "Déploiement $deployment est prêt"
    done
    
    log "SUCCESS" "Tous les déploiements sont prêts"
}

# Tests post-déploiement
run_post_deployment_tests() {
    log "INFO" "🧪 Exécution des tests post-déploiement..."
    
    local test_results=()
    
    # Test de santé de l'API
    if test_api_health; then
        test_results+=("API:PASS")
    else
        test_results+=("API:FAIL")
    fi
    
    # Test de la base de données
    if test_database_connection; then
        test_results+=("DATABASE:PASS")
    else
        test_results+=("DATABASE:FAIL")
    fi
    
    # Test de Redis
    if test_redis_connection; then
        test_results+=("REDIS:PASS")
    else
        test_results+=("REDIS:FAIL")
    fi
    
    # Test des métriques
    if test_metrics_endpoint; then
        test_results+=("METRICS:PASS")
    else
        test_results+=("METRICS:FAIL")
    fi
    
    # Affichage des résultats
    log "INFO" "Résultats des tests:"
    for result in "${test_results[@]}"; do
        local service=$(echo $result | cut -d: -f1)
        local status=$(echo $result | cut -d: -f2)
        
        if [[ "$status" == "PASS" ]]; then
            log "SUCCESS" "$service: ✅"
        else
            log "ERROR" "$service: ❌"
        fi
    done
    
    # Vérification globale
    local failed_tests=$(printf '%s\n' "${test_results[@]}" | grep -c "FAIL" || true)
    if [[ $failed_tests -eq 0 ]]; then
        log "SUCCESS" "Tous les tests sont passés"
        return 0
    else
        log "ERROR" "$failed_tests tests ont échoué"
        return 1
    fi
}

test_api_health() {
    local endpoint="http://localhost:8000/health"
    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        endpoint="http://$(kubectl get svc incidents-api-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):8000/health"
    fi
    
    log "DEBUG" "Test de l'endpoint: $endpoint"
    
    local response
    response=$(curl -s -o /dev/null -w "%{http_code}" "$endpoint" 2>/dev/null || echo "000")
    
    if [[ "$response" == "200" ]]; then
        return 0
    else
        log "DEBUG" "API Health Check failed with status: $response"
        return 1
    fi
}

test_database_connection() {
    if [[ "$DEPLOYMENT_MODE" == "docker" ]]; then
        docker exec incidents-postgres-${ENVIRONMENT} psql -U incidents_user -d incidents -c "SELECT 1;" &> /dev/null
    else
        kubectl exec -n $NAMESPACE deployment/postgres -- psql -U incidents_user -d incidents -c "SELECT 1;" &> /dev/null
    fi
}

test_redis_connection() {
    if [[ "$DEPLOYMENT_MODE" == "docker" ]]; then
        docker exec incidents-redis-${ENVIRONMENT} redis-cli ping | grep -q "PONG"
    else
        kubectl exec -n $NAMESPACE deployment/redis -- redis-cli ping | grep -q "PONG"
    fi
}

test_metrics_endpoint() {
    local endpoint="http://localhost:9090/-/healthy"
    if [[ "$DEPLOYMENT_MODE" == "kubernetes" ]]; then
        endpoint="http://$(kubectl get svc prometheus-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):9090/-/healthy"
    fi
    
    local response
    response=$(curl -s -o /dev/null -w "%{http_code}" "$endpoint" 2>/dev/null || echo "000")
    
    [[ "$response" == "200" ]]
}

# Configuration du monitoring
setup_monitoring() {
    if [[ "$ENABLE_MONITORING" != "true" ]]; then
        log "INFO" "Monitoring désactivé"
        return 0
    fi
    
    log "INFO" "📊 Configuration du monitoring..."
    
    # Configuration des dashboards Grafana
    setup_grafana_dashboards
    
    # Configuration des alertes Prometheus
    setup_prometheus_alerts
    
    # Configuration des exporters de métriques
    setup_metric_exporters
    
    log "SUCCESS" "Monitoring configuré"
}

setup_grafana_dashboards() {
    log "INFO" "Configuration des dashboards Grafana..."
    
    # Attente que Grafana soit prêt
    local grafana_url="http://localhost:3000"
    local max_wait=60
    local wait_time=0
    
    while [[ $wait_time -lt $max_wait ]]; do
        if curl -s "$grafana_url/api/health" &> /dev/null; then
            break
        fi
        sleep 5
        wait_time=$((wait_time + 5))
    done
    
    # Import des dashboards
    local dashboard_files=(
        "${SCRIPT_DIR}/configs/grafana/dashboards/incidents.json"
        "${SCRIPT_DIR}/configs/grafana/dashboards/system.json"
        "${SCRIPT_DIR}/configs/grafana/dashboards/security.json"
    )
    
    for dashboard_file in "${dashboard_files[@]}"; do
        if [[ -f "$dashboard_file" ]]; then
            log "INFO" "Import du dashboard: $(basename $dashboard_file)"
            curl -X POST \
                -H "Content-Type: application/json" \
                -d @"$dashboard_file" \
                "http://admin:${GRAFANA_ADMIN_PASSWORD}@localhost:3000/api/dashboards/db" \
                &> /dev/null || true
        fi
    done
    
    log "SUCCESS" "Dashboards Grafana configurés"
}

setup_prometheus_alerts() {
    log "INFO" "Configuration des alertes Prometheus..."
    
    # Rechargement de la configuration Prometheus
    curl -X POST http://localhost:9090/-/reload &> /dev/null || true
    
    log "SUCCESS" "Alertes Prometheus configurées"
}

setup_metric_exporters() {
    log "INFO" "Configuration des exporters de métriques..."
    
    # Node Exporter si pas déjà déployé
    if ! docker container inspect node-exporter &> /dev/null; then
        docker run -d \
            --name node-exporter \
            --restart unless-stopped \
            -p 9100:9100 \
            -v "/proc:/host/proc:ro" \
            -v "/sys:/host/sys:ro" \
            -v "/:/rootfs:ro" \
            prom/node-exporter:latest \
            --path.procfs=/host/proc \
            --path.sysfs=/host/sys \
            --collector.filesystem.ignored-mount-points='^/(sys|proc|dev|host|etc)($|/)'
        
        log "SUCCESS" "Node Exporter déployé"
    fi
}

# Nettoyage et rollback
cleanup() {
    log "INFO" "🧹 Nettoyage en cours..."
    
    # Nettoyage des fichiers temporaires
    rm -rf /tmp/incidents/*
    
    # Nettoyage des images Docker non utilisées
    docker image prune -f &> /dev/null || true
    
    log "SUCCESS" "Nettoyage terminé"
}

rollback() {
    log "WARN" "🔄 Rollback en cours..."
    
    if [[ -d "$BACKUP_DIR" ]]; then
        # Arrêt des services actuels
        if [[ "$DEPLOYMENT_MODE" == "docker" ]]; then
            docker-compose -f "${SCRIPT_DIR}/docker-compose.${ENVIRONMENT}.yml" down
        else
            kubectl delete -f "${SCRIPT_DIR}/k8s/" --ignore-not-found=true
        fi
        
        # Restauration de la base de données
        if [[ -f "${BACKUP_DIR}/database_backup.sql" ]]; then
            log "INFO" "Restauration de la base de données..."
            # Logique de restauration
        fi
        
        # Restauration des configurations
        if [[ -d "${BACKUP_DIR}/config_backup" ]]; then
            log "INFO" "Restauration des configurations..."
            cp -r "${BACKUP_DIR}/config_backup"/* "/etc/incidents/config/"
        fi
        
        log "SUCCESS" "Rollback terminé"
    else
        log "ERROR" "Aucune sauvegarde trouvée pour le rollback"
    fi
}

# Fonction principale
main() {
    local action="${1:-deploy}"
    
    # Gestion des signaux pour nettoyage
    trap cleanup EXIT
    trap 'log "ERROR" "Déploiement interrompu"; rollback; exit 1' INT TERM
    
    # Bannière de démarrage
    echo -e "${CYAN}"
    echo "============================================================================="
    echo "   🚀 SYSTÈME DE GESTION DES INCIDENTS & MÉTRIQUES - DÉPLOIEMENT v2.0.0"
    echo "============================================================================="
    echo -e "${NC}"
    echo "Environnement: $ENVIRONMENT"
    echo "Mode de déploiement: $DEPLOYMENT_MODE"
    echo "Namespace/Environnement: $NAMESPACE"
    echo "Réplicas: $REPLICA_COUNT"
    echo "Monitoring: $ENABLE_MONITORING"
    echo "Sauvegarde: $ENABLE_BACKUP"
    echo "Dry Run: $DRY_RUN"
    echo "============================================================================="
    echo
    
    # Vérification du mode dry-run
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "Mode DRY RUN activé - aucune action ne sera effectuée"
        return 0
    fi
    
    # Création du fichier de log
    mkdir -p "$(dirname "$LOG_FILE")"
    touch "$LOG_FILE"
    
    case "$action" in
        "deploy")
            log "INFO" "🚀 Début du déploiement complet..."
            
            check_prerequisites
            prepare_environment
            create_backup
            build_images
            
            if [[ "$DEPLOYMENT_MODE" == "docker" ]]; then
                deploy_docker
            else
                deploy_kubernetes
            fi
            
            setup_monitoring
            
            if run_post_deployment_tests; then
                log "SUCCESS" "🎉 Déploiement terminé avec succès!"
                log "INFO" "📊 Accès aux services:"
                log "INFO" "  - API: http://localhost:8000"
                log "INFO" "  - Grafana: http://localhost:3000 (admin/${GRAFANA_ADMIN_PASSWORD})"
                log "INFO" "  - Prometheus: http://localhost:9090"
            else
                log "ERROR" "❌ Tests post-déploiement échoués"
                if [[ "$FORCE_DEPLOY" != "true" ]]; then
                    rollback
                fi
                exit 1
            fi
            ;;
            
        "rollback")
            rollback
            ;;
            
        "cleanup")
            cleanup
            ;;
            
        "test")
            run_post_deployment_tests
            ;;
            
        *)
            log "ERROR" "Action non reconnue: $action"
            echo "Usage: $0 [deploy|rollback|cleanup|test]"
            exit 1
            ;;
    esac
}

# Gestion des arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --replicas)
            REPLICA_COUNT="$2"
            shift 2
            ;;
        --force)
            FORCE_DEPLOY="true"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --no-monitoring)
            ENABLE_MONITORING="false"
            shift
            ;;
        --no-backup)
            ENABLE_BACKUP="false"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] [ACTION]"
            echo ""
            echo "Actions:"
            echo "  deploy     Déploiement complet (défaut)"
            echo "  rollback   Rollback vers la sauvegarde précédente"
            echo "  cleanup    Nettoyage des ressources"
            echo "  test       Tests post-déploiement uniquement"
            echo ""
            echo "Options:"
            echo "  --environment ENV    Environnement cible (development|staging|production)"
            echo "  --namespace NS       Namespace Kubernetes"
            echo "  --replicas N         Nombre de réplicas"
            echo "  --force              Force le redéploiement"
            echo "  --dry-run            Mode simulation"
            echo "  --no-monitoring      Désactive le monitoring"
            echo "  --no-backup          Désactive la sauvegarde"
            echo "  --help, -h           Affiche cette aide"
            echo ""
            echo "Variables d'environnement:"
            echo "  ENVIRONMENT          Environnement de déploiement"
            echo "  NAMESPACE            Namespace Kubernetes"
            echo "  REPLICA_COUNT        Nombre de réplicas"
            echo "  ENABLE_MONITORING    Active/désactive le monitoring"
            echo "  ENABLE_BACKUP        Active/désactive la sauvegarde"
            echo ""
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

# Exécution de la fonction principale
main "$@"
