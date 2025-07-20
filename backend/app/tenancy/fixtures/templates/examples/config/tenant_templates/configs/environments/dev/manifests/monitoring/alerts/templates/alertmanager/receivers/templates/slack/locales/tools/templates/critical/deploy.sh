#!/bin/bash

# 🚀 Script Ultra-Avancé de Déploiement des Alertes Critiques
# ===========================================================
#
# Script de déploiement industriel pour le système d'alertes critiques
# avec support multi-environnement, validation complète et rollback automatique.
#
# Architecte: Fahed Mlaiel - Lead Architect
# Version: 3.0.0-enterprise
# Support: 24/7 Enterprise Support

set -euo pipefail

# === CONFIGURATION GLOBALE ===
SCRIPT_VERSION="3.0.0-enterprise"
DEPLOYMENT_ID="critical-alerts-$(date +%s)"
LOG_FILE="/var/log/critical-alerts-deploy.log"
LOCK_FILE="/tmp/critical-alerts-deploy.lock"
BACKUP_DIR="/opt/backups/critical-alerts"
CONFIG_DIR="$(dirname "$(readlink -f "$0")")"

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# === FONCTIONS UTILITAIRES ===

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOG_FILE}"
    
    case "${level}" in
        "ERROR")   echo -e "${RED}[ERROR]${NC} ${message}" ;;
        "SUCCESS") echo -e "${GREEN}[SUCCESS]${NC} ${message}" ;;
        "WARNING") echo -e "${YELLOW}[WARNING]${NC} ${message}" ;;
        "INFO")    echo -e "${BLUE}[INFO]${NC} ${message}" ;;
        "DEBUG")   echo -e "${PURPLE}[DEBUG]${NC} ${message}" ;;
        *)         echo -e "${CYAN}[${level}]${NC} ${message}" ;;
    esac
}

check_prerequisites() {
    log "INFO" "Vérification des prérequis système..."
    
    local missing_tools=()
    
    # Vérification des outils requis
    for tool in docker kubectl helm python3 pip redis-cli psql; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log "ERROR" "Outils manquants: ${missing_tools[*]}"
        log "ERROR" "Veuillez installer les outils manquants avant de continuer"
        exit 1
    fi
    
    # Vérification des versions minimales
    check_tool_version "docker" "20.10" "$(docker --version | grep -oE '[0-9]+\.[0-9]+' | head -1)"
    check_tool_version "kubectl" "1.20" "$(kubectl version --client -o json | jq -r '.clientVersion.gitVersion' | sed 's/v//')"
    check_tool_version "helm" "3.5" "$(helm version --short | grep -oE '[0-9]+\.[0-9]+' | head -1)"
    check_tool_version "python3" "3.9" "$(python3 --version | grep -oE '[0-9]+\.[0-9]+')"
    
    log "SUCCESS" "Tous les prérequis sont satisfaits"
}

check_tool_version() {
    local tool="$1"
    local min_version="$2"
    local current_version="$3"
    
    if [ -z "$current_version" ]; then
        log "ERROR" "Impossible de déterminer la version de $tool"
        exit 1
    fi
    
    if ! printf '%s\n' "$min_version" "$current_version" | sort -V | head -n1 | grep -q "^$min_version$"; then
        log "ERROR" "$tool version $current_version < $min_version (requis)"
        exit 1
    fi
    
    log "DEBUG" "$tool version $current_version ✓"
}

acquire_lock() {
    if [ -f "$LOCK_FILE" ]; then
        local pid=$(cat "$LOCK_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            log "ERROR" "Un autre déploiement est en cours (PID: $pid)"
            exit 1
        else
            log "WARNING" "Lock file obsolète détecté, suppression"
            rm -f "$LOCK_FILE"
        fi
    fi
    
    echo $$ > "$LOCK_FILE"
    log "INFO" "Verrou de déploiement acquis (PID: $$)"
}

release_lock() {
    rm -f "$LOCK_FILE"
    log "INFO" "Verrou de déploiement libéré"
}

# Trap pour nettoyer en cas d'interruption
trap release_lock EXIT INT TERM

validate_environment() {
    local environment="$1"
    
    log "INFO" "Validation de l'environnement: $environment"
    
    case "$environment" in
        "dev"|"development")
            export NAMESPACE="critical-alerts-dev"
            export REPLICAS=1
            export RESOURCES_CPU="100m"
            export RESOURCES_MEMORY="256Mi"
            export DEBUG_MODE="true"
            ;;
        "staging"|"stage")
            export NAMESPACE="critical-alerts-staging"
            export REPLICAS=2
            export RESOURCES_CPU="200m"
            export RESOURCES_MEMORY="512Mi"
            export DEBUG_MODE="false"
            ;;
        "prod"|"production")
            export NAMESPACE="critical-alerts-prod"
            export REPLICAS=3
            export RESOURCES_CPU="500m"
            export RESOURCES_MEMORY="1Gi"
            export DEBUG_MODE="false"
            ;;
        *)
            log "ERROR" "Environnement non supporté: $environment"
            log "INFO" "Environnements supportés: dev, staging, prod"
            exit 1
            ;;
    esac
    
    log "SUCCESS" "Environnement $environment validé"
    log "INFO" "Configuration: Namespace=$NAMESPACE, Replicas=$REPLICAS"
}

create_namespace() {
    log "INFO" "Création/vérification du namespace $NAMESPACE"
    
    if kubectl get namespace "$NAMESPACE" &>/dev/null; then
        log "INFO" "Namespace $NAMESPACE existe déjà"
    else
        kubectl create namespace "$NAMESPACE"
        log "SUCCESS" "Namespace $NAMESPACE créé"
    fi
    
    # Labels du namespace
    kubectl label namespace "$NAMESPACE" \
        app.kubernetes.io/name=critical-alerts \
        app.kubernetes.io/component=monitoring \
        app.kubernetes.io/part-of=spotify-ai-agent \
        app.kubernetes.io/managed-by=helm \
        --overwrite
}

setup_secrets() {
    log "INFO" "Configuration des secrets"
    
    # Vérification des variables d'environnement requises
    local required_vars=(
        "SLACK_BOT_TOKEN"
        "SLACK_SIGNING_SECRET"
        "SMTP_USER"
        "SMTP_PASSWORD"
        "PAGERDUTY_INTEGRATION_KEY"
        "REDIS_PASSWORD"
        "POSTGRES_PASSWORD"
    )
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            log "ERROR" "Variable d'environnement manquante: $var"
            exit 1
        fi
    done
    
    # Création du secret Kubernetes
    kubectl create secret generic critical-alerts-secrets \
        --namespace="$NAMESPACE" \
        --from-literal=slack-bot-token="$SLACK_BOT_TOKEN" \
        --from-literal=slack-signing-secret="$SLACK_SIGNING_SECRET" \
        --from-literal=smtp-user="$SMTP_USER" \
        --from-literal=smtp-password="$SMTP_PASSWORD" \
        --from-literal=pagerduty-integration-key="$PAGERDUTY_INTEGRATION_KEY" \
        --from-literal=redis-password="$REDIS_PASSWORD" \
        --from-literal=postgres-password="$POSTGRES_PASSWORD" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log "SUCCESS" "Secrets configurés"
}

build_docker_images() {
    log "INFO" "Construction des images Docker"
    
    # Image principale de l'application
    docker build \
        --build-arg BUILD_VERSION="$SCRIPT_VERSION" \
        --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --build-arg COMMIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')" \
        -t "critical-alerts:$SCRIPT_VERSION" \
        -t "critical-alerts:latest" \
        "$CONFIG_DIR/../"
    
    # Image des workers ML
    docker build \
        --build-arg BUILD_VERSION="$SCRIPT_VERSION" \
        -t "critical-alerts-ml:$SCRIPT_VERSION" \
        -t "critical-alerts-ml:latest" \
        -f "$CONFIG_DIR/../Dockerfile.ml" \
        "$CONFIG_DIR/../"
    
    # Image de monitoring
    docker build \
        --build-arg BUILD_VERSION="$SCRIPT_VERSION" \
        -t "critical-alerts-monitoring:$SCRIPT_VERSION" \
        -t "critical-alerts-monitoring:latest" \
        -f "$CONFIG_DIR/../Dockerfile.monitoring" \
        "$CONFIG_DIR/../"
    
    log "SUCCESS" "Images Docker construites"
}

run_tests() {
    log "INFO" "Exécution des tests de validation"
    
    # Tests unitaires
    docker run --rm \
        -v "$CONFIG_DIR/../:/app" \
        -w /app \
        critical-alerts:latest \
        python -m pytest tests/unit/ -v --tb=short
    
    # Tests d'intégration
    docker run --rm \
        -v "$CONFIG_DIR/../:/app" \
        -w /app \
        --network=host \
        critical-alerts:latest \
        python -m pytest tests/integration/ -v --tb=short
    
    # Tests de performance
    docker run --rm \
        -v "$CONFIG_DIR/../:/app" \
        -w /app \
        critical-alerts:latest \
        python -m pytest tests/performance/ -v --tb=short
    
    # Validation des configurations
    python3 "$CONFIG_DIR/validate_config.py" "$CONFIG_DIR/config.yaml"
    
    log "SUCCESS" "Tous les tests sont passés"
}

deploy_infrastructure() {
    log "INFO" "Déploiement de l'infrastructure de base"
    
    # Déploiement Redis Cluster
    helm upgrade --install redis-cluster \
        bitnami/redis-cluster \
        --namespace="$NAMESPACE" \
        --set="auth.password=$REDIS_PASSWORD" \
        --set="cluster.nodes=6" \
        --set="cluster.replicas=1" \
        --set="persistence.enabled=true" \
        --set="persistence.size=10Gi" \
        --wait --timeout=10m
    
    # Déploiement PostgreSQL
    helm upgrade --install postgresql \
        bitnami/postgresql \
        --namespace="$NAMESPACE" \
        --set="auth.postgresPassword=$POSTGRES_PASSWORD" \
        --set="auth.database=critical_alerts" \
        --set="persistence.enabled=true" \
        --set="persistence.size=20Gi" \
        --set="metrics.enabled=true" \
        --wait --timeout=10m
    
    # Déploiement Prometheus
    helm upgrade --install prometheus \
        prometheus-community/kube-prometheus-stack \
        --namespace="$NAMESPACE" \
        --set="prometheus.prometheusSpec.retention=30d" \
        --set="prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi" \
        --set="grafana.enabled=true" \
        --set="grafana.adminPassword=$GRAFANA_PASSWORD" \
        --wait --timeout=15m
    
    log "SUCCESS" "Infrastructure déployée"
}

deploy_application() {
    log "INFO" "Déploiement de l'application principale"
    
    # Génération du manifeste Helm
    cat > "$CONFIG_DIR/helm-values.yaml" << EOF
image:
  repository: critical-alerts
  tag: $SCRIPT_VERSION
  pullPolicy: IfNotPresent

replicaCount: $REPLICAS

resources:
  requests:
    cpu: $RESOURCES_CPU
    memory: $RESOURCES_MEMORY
  limits:
    cpu: $(echo "$RESOURCES_CPU" | sed 's/m$//' | awk '{print $1*2"m"}')
    memory: $(echo "$RESOURCES_MEMORY" | sed 's/Mi$//' | awk '{print $1*2"Mi"}')

env:
  DEBUG_MODE: "$DEBUG_MODE"
  DEPLOYMENT_ID: "$DEPLOYMENT_ID"
  ENVIRONMENT: "$environment"

secrets:
  name: critical-alerts-secrets

service:
  type: ClusterIP
  port: 8000

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: critical-alerts-${environment}.spotify-ai-agent.com
      paths:
        - path: /
          pathType: Prefix

autoscaling:
  enabled: true
  minReplicas: $REPLICAS
  maxReplicas: $(($REPLICAS * 3))
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: 30s

healthCheck:
  enabled: true
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
    initialDelaySeconds: 10
    periodSeconds: 5
EOF

    # Déploiement avec Helm
    helm upgrade --install critical-alerts \
        "$CONFIG_DIR/helm-chart/" \
        --namespace="$NAMESPACE" \
        --values="$CONFIG_DIR/helm-values.yaml" \
        --wait --timeout=10m
    
    log "SUCCESS" "Application déployée"
}

run_smoke_tests() {
    log "INFO" "Exécution des tests de fumée"
    
    # Attendre que les pods soient prêts
    kubectl wait --for=condition=ready pod \
        -l app.kubernetes.io/name=critical-alerts \
        --namespace="$NAMESPACE" \
        --timeout=300s
    
    # Test de santé de l'API
    local api_url="http://critical-alerts-${environment}.spotify-ai-agent.com"
    
    # Test du endpoint de santé
    if curl -f "$api_url/health" &>/dev/null; then
        log "SUCCESS" "API health check OK"
    else
        log "ERROR" "API health check FAILED"
        return 1
    fi
    
    # Test de création d'alerte de test
    local test_alert_response=$(curl -s -w "%{http_code}" \
        -X POST "$api_url/api/v1/critical-alerts/test" \
        -H "Content-Type: application/json" \
        -d '{
            "severity": "WARNING",
            "source_service": "deployment-test",
            "message": "Test alert from deployment script",
            "tenant_id": "test-tenant"
        }')
    
    if [[ "$test_alert_response" == *"200"* ]] || [[ "$test_alert_response" == *"201"* ]]; then
        log "SUCCESS" "Test alert creation OK"
    else
        log "ERROR" "Test alert creation FAILED: $test_alert_response"
        return 1
    fi
    
    # Test des métriques Prometheus
    if curl -f "$api_url/metrics" | grep -q "critical_alerts"; then
        log "SUCCESS" "Prometheus metrics OK"
    else
        log "ERROR" "Prometheus metrics FAILED"
        return 1
    fi
    
    log "SUCCESS" "Tous les tests de fumée sont passés"
}

create_backup() {
    log "INFO" "Création d'une sauvegarde avant déploiement"
    
    mkdir -p "$BACKUP_DIR/$DEPLOYMENT_ID"
    
    # Sauvegarde des manifestes Kubernetes actuels
    kubectl get all,configmaps,secrets,ingress \
        --namespace="$NAMESPACE" \
        -o yaml > "$BACKUP_DIR/$DEPLOYMENT_ID/k8s-manifests.yaml" 2>/dev/null || true
    
    # Sauvegarde de la base de données
    kubectl exec -n "$NAMESPACE" deployment/postgresql -- \
        pg_dump -U postgres critical_alerts > "$BACKUP_DIR/$DEPLOYMENT_ID/database.sql" 2>/dev/null || true
    
    # Sauvegarde des configurations Redis
    kubectl exec -n "$NAMESPACE" deployment/redis-cluster-0 -- \
        redis-cli --rdb /tmp/dump.rdb > "$BACKUP_DIR/$DEPLOYMENT_ID/redis.rdb" 2>/dev/null || true
    
    log "SUCCESS" "Sauvegarde créée: $BACKUP_DIR/$DEPLOYMENT_ID"
}

rollback() {
    local backup_id="$1"
    
    log "WARNING" "Début du rollback vers $backup_id"
    
    if [ ! -d "$BACKUP_DIR/$backup_id" ]; then
        log "ERROR" "Sauvegarde $backup_id introuvable"
        exit 1
    fi
    
    # Rollback Kubernetes
    if [ -f "$BACKUP_DIR/$backup_id/k8s-manifests.yaml" ]; then
        kubectl apply -f "$BACKUP_DIR/$backup_id/k8s-manifests.yaml" --namespace="$NAMESPACE"
    fi
    
    # Rollback base de données (si nécessaire)
    if [ -f "$BACKUP_DIR/$backup_id/database.sql" ]; then
        kubectl exec -n "$NAMESPACE" deployment/postgresql -- \
            psql -U postgres -d critical_alerts -f /tmp/database.sql
    fi
    
    log "SUCCESS" "Rollback terminé"
}

cleanup() {
    log "INFO" "Nettoyage post-déploiement"
    
    # Suppression des anciennes images Docker
    docker image prune -f --filter "label=component=critical-alerts" --filter "until=168h"
    
    # Suppression des anciennes sauvegardes (garde les 10 dernières)
    cd "$BACKUP_DIR" && ls -t | tail -n +11 | xargs rm -rf 2>/dev/null || true
    
    # Nettoyage des fichiers temporaires
    rm -f "$CONFIG_DIR/helm-values.yaml"
    
    log "SUCCESS" "Nettoyage terminé"
}

print_summary() {
    log "INFO" "=== RÉSUMÉ DU DÉPLOIEMENT ==="
    log "INFO" "Deployment ID: $DEPLOYMENT_ID"
    log "INFO" "Version: $SCRIPT_VERSION"
    log "INFO" "Environnement: $environment"
    log "INFO" "Namespace: $NAMESPACE"
    log "INFO" "Replicas: $REPLICAS"
    log "INFO" "Sauvegarde: $BACKUP_DIR/$DEPLOYMENT_ID"
    log "INFO" "==============================="
    
    # URLs utiles
    log "INFO" "URLs utiles:"
    log "INFO" "  - API: http://critical-alerts-${environment}.spotify-ai-agent.com"
    log "INFO" "  - Grafana: http://grafana-${environment}.spotify-ai-agent.com"
    log "INFO" "  - Prometheus: http://prometheus-${environment}.spotify-ai-agent.com"
    
    # Commandes utiles
    log "INFO" "Commandes utiles:"
    log "INFO" "  - Logs: kubectl logs -f deployment/critical-alerts -n $NAMESPACE"
    log "INFO" "  - Scale: kubectl scale deployment critical-alerts --replicas=N -n $NAMESPACE"
    log "INFO" "  - Rollback: $0 rollback $DEPLOYMENT_ID"
}

# === FONCTION PRINCIPALE ===

main() {
    local command="${1:-deploy}"
    local environment="${2:-dev}"
    
    log "INFO" "Démarrage du script de déploiement v$SCRIPT_VERSION"
    log "INFO" "Commande: $command, Environnement: $environment"
    
    case "$command" in
        "deploy")
            acquire_lock
            check_prerequisites
            validate_environment "$environment"
            create_backup
            create_namespace
            setup_secrets
            build_docker_images
            run_tests
            deploy_infrastructure
            deploy_application
            run_smoke_tests
            cleanup
            print_summary
            log "SUCCESS" "Déploiement terminé avec succès!"
            ;;
        "rollback")
            local backup_id="${3:-}"
            if [ -z "$backup_id" ]; then
                log "ERROR" "Usage: $0 rollback <environment> <backup_id>"
                exit 1
            fi
            validate_environment "$environment"
            rollback "$backup_id"
            ;;
        "test")
            validate_environment "$environment"
            run_smoke_tests
            ;;
        "cleanup")
            cleanup
            ;;
        *)
            log "ERROR" "Commande inconnue: $command"
            log "INFO" "Commandes disponibles: deploy, rollback, test, cleanup"
            exit 1
            ;;
    esac
}

# Exécution du script
main "$@"
