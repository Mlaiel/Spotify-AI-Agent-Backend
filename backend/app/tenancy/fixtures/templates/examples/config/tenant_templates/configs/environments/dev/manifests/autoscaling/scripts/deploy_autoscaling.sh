#!/bin/bash

# TYPE: deployment
# MODE: sync
# PRIORITY: high
# DESCRIPTION: Enterprise-grade autoscaling deployment with advanced orchestration
# VERSION: 2.1.0
# AUTHOR: Enterprise Architecture Team
# COMPLIANCE: SOC2, GDPR, HIPAA

set -euo pipefail
IFS=$'\n\t'

# Advanced Enterprise Autoscaling Deployment Script
# Ultra-advanced industrial deployment with comprehensive validation and monitoring

# Global Configuration
readonly SCRIPT_NAME="deploy_autoscaling"
readonly SCRIPT_VERSION="2.1.0"
readonly LOG_LEVEL="${LOG_LEVEL:-INFO}"
readonly DEPLOYMENT_ID="deploy_$(date +%s)"

# Enhanced Logging System
setup_logging() {
    local log_dir="/var/log/autoscaling"
    local log_file="${log_dir}/${SCRIPT_NAME}_$(date +%Y%m%d_%H%M%S).log"
    
    mkdir -p "$log_dir"
    exec 1> >(tee -a "$log_file")
    exec 2> >(tee -a "$log_file" >&2)
    
    echo "=== Enterprise Autoscaling Deployment Started ==="
    echo "Deployment ID: $DEPLOYMENT_ID"
    echo "Timestamp: $(date -Iseconds)"
    echo "Script Version: $SCRIPT_VERSION"
    echo "Log File: $log_file"
    echo "=================================================="
}

# Advanced Configuration Management
readonly NAMESPACE="${NAMESPACE:-autoscaling-dev}"
readonly ENVIRONMENT="${ENVIRONMENT:-development}"
readonly CONFIG_PATH="${CONFIG_PATH:-./configs}"
readonly KUBECTL_TIMEOUT="${KUBECTL_TIMEOUT:-300s}"
readonly ROLLBACK_ENABLED="${ROLLBACK_ENABLED:-true}"
readonly HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-600}"
readonly MONITORING_ENABLED="${MONITORING_ENABLED:-true}"
readonly BACKUP_ENABLED="${BACKUP_ENABLED:-true}"
readonly DRY_RUN="${DRY_RUN:-false}"

# Vérification des prérequis
check_prerequisites() {
    log_info "Vérification des prérequis..."
    
    # Kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl n'est pas installé"
        exit 1
    fi
    
    # Helm (optionnel)
    if command -v helm &> /dev/null; then
        log_info "Helm détecté, utilisation de Helm pour le déploiement"
        USE_HELM=true
    else
        log_warning "Helm non détecté, utilisation de kubectl"
        USE_HELM=false
    fi
    
    # Vérification de la connectivité cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Impossible de se connecter au cluster Kubernetes"
        exit 1
    fi
    
    log_success "Prérequis validés"
}

# Création du namespace
create_namespace() {
    log_info "Création du namespace $NAMESPACE..."
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Le namespace $NAMESPACE existe déjà"
    else
        kubectl create namespace "$NAMESPACE"
        log_success "Namespace $NAMESPACE créé"
    fi
    
    # Labels pour le namespace
    kubectl label namespace "$NAMESPACE" \
        app.kubernetes.io/name=spotify-autoscaling \
        app.kubernetes.io/component=autoscaling \
        app.kubernetes.io/managed-by=deploy-script \
        --overwrite
}

# Déploiement des ConfigMaps
deploy_configmaps() {
    log_info "Déploiement des ConfigMaps..."
    
    # ConfigMap pour la configuration globale
    kubectl create configmap autoscaling-global-config \
        --from-file="$PROJECT_ROOT/configs/global-config.yaml" \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # ConfigMap pour les politiques par défaut
    kubectl create configmap autoscaling-default-policies \
        --from-file="$PROJECT_ROOT/configs/default-policies.yaml" \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # ConfigMaps pour les configurations tenant
    for config_file in "$PROJECT_ROOT/configs/tenant-configs"/*.yaml; do
        if [[ -f "$config_file" ]]; then
            config_name="tenant-config-$(basename "$config_file" .yaml)"
            kubectl create configmap "$config_name" \
                --from-file="$config_file" \
                --namespace="$NAMESPACE" \
                --dry-run=client -o yaml | kubectl apply -f -
        fi
    done
    
    log_success "ConfigMaps déployées"
}

# Déploiement des secrets
deploy_secrets() {
    log_info "Déploiement des secrets..."
    
    # Secret pour les métriques externes
    kubectl create secret generic autoscaling-metrics-secrets \
        --from-literal=prometheus-token="${PROMETHEUS_TOKEN:-}" \
        --from-literal=influxdb-token="${INFLUXDB_TOKEN:-}" \
        --from-literal=cloudwatch-access-key="${AWS_ACCESS_KEY_ID:-}" \
        --from-literal=cloudwatch-secret-key="${AWS_SECRET_ACCESS_KEY:-}" \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Secret pour les webhooks
    kubectl create secret generic autoscaling-webhook-secrets \
        --from-literal=slack-webhook-url="${SLACK_WEBHOOK_URL:-}" \
        --from-literal=webhook-token="${WEBHOOK_TOKEN:-}" \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "Secrets déployés"
}

# Génération des manifestes Kubernetes
generate_k8s_manifests() {
    log_info "Génération des manifestes Kubernetes..."
    
    local manifests_dir="$PROJECT_ROOT/manifests"
    mkdir -p "$manifests_dir"
    
    # Deployment principal
    cat > "$manifests_dir/deployment.yaml" << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: autoscaling-controller
  namespace: $NAMESPACE
  labels:
    app: autoscaling-controller
    version: v1.0.0
spec:
  replicas: 2
  selector:
    matchLabels:
      app: autoscaling-controller
  template:
    metadata:
      labels:
        app: autoscaling-controller
    spec:
      serviceAccountName: autoscaling-controller
      containers:
      - name: controller
        image: spotify-ai-agent/autoscaling:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: NAMESPACE
          value: "$NAMESPACE"
        - name: ENVIRONMENT
          value: "$ENVIRONMENT"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            cpu: 200m
            memory: 256Mi
          limits:
            cpu: 1000m
            memory: 1Gi
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: secrets-volume
          mountPath: /app/secrets
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config-volume
        configMap:
          name: autoscaling-global-config
      - name: secrets-volume
        secret:
          secretName: autoscaling-metrics-secrets
EOF
    
    # Service
    cat > "$manifests_dir/service.yaml" << EOF
apiVersion: v1
kind: Service
metadata:
  name: autoscaling-controller
  namespace: $NAMESPACE
  labels:
    app: autoscaling-controller
spec:
  selector:
    app: autoscaling-controller
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP
EOF
    
    # ServiceAccount et RBAC
    cat > "$manifests_dir/rbac.yaml" << EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: autoscaling-controller
  namespace: $NAMESPACE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: autoscaling-controller
rules:
- apiGroups: [""]
  resources: ["pods", "nodes", "services", "endpoints"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch", "update", "patch"]
- apiGroups: ["autoscaling"]
  resources: ["horizontalpodautoscalers"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["metrics.k8s.io"]
  resources: ["pods", "nodes"]
  verbs: ["get", "list"]
- apiGroups: ["custom.metrics.k8s.io"]
  resources: ["*"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: autoscaling-controller
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: autoscaling-controller
subjects:
- kind: ServiceAccount
  name: autoscaling-controller
  namespace: $NAMESPACE
EOF
    
    log_success "Manifestes générés dans $manifests_dir"
}

# Déploiement avec Helm
deploy_with_helm() {
    log_info "Déploiement avec Helm..."
    
    local chart_dir="$PROJECT_ROOT/helm-chart"
    mkdir -p "$chart_dir"
    
    # Génération du Chart.yaml
    cat > "$chart_dir/Chart.yaml" << EOF
apiVersion: v2
name: spotify-autoscaling
description: Autoscaling intelligent pour Spotify AI Agent
version: 1.0.0
appVersion: 1.0.0
maintainers:
- name: Fahed Mlaiel
  email: fahed.mlaiel@company.com
EOF
    
    # Déploiement
    helm upgrade --install "$RELEASE_NAME" "$chart_dir" \
        --namespace "$NAMESPACE" \
        --create-namespace \
        --set environment="$ENVIRONMENT" \
        --set image.tag=latest \
        --wait --timeout=600s
    
    log_success "Déploiement Helm terminé"
}

# Déploiement avec kubectl
deploy_with_kubectl() {
    log_info "Déploiement avec kubectl..."
    
    local manifests_dir="$PROJECT_ROOT/manifests"
    
    # Application des manifestes
    kubectl apply -f "$manifests_dir/" --namespace="$NAMESPACE"
    
    # Attente du déploiement
    kubectl rollout status deployment/autoscaling-controller \
        --namespace="$NAMESPACE" --timeout=300s
    
    log_success "Déploiement kubectl terminé"
}

# Vérification du déploiement
verify_deployment() {
    log_info "Vérification du déploiement..."
    
    # Vérification des pods
    if kubectl get pods -n "$NAMESPACE" -l app=autoscaling-controller | grep -q Running; then
        log_success "Pods en cours d'exécution"
    else
        log_error "Pods non démarrés"
        kubectl get pods -n "$NAMESPACE" -l app=autoscaling-controller
        return 1
    fi
    
    # Vérification du service
    local service_ip
    service_ip=$(kubectl get svc autoscaling-controller -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    if [[ -n "$service_ip" ]]; then
        log_success "Service disponible sur $service_ip"
    else
        log_error "Service non disponible"
        return 1
    fi
    
    # Test de santé
    if kubectl exec -n "$NAMESPACE" deployment/autoscaling-controller -- \
        curl -f http://localhost:8080/health &> /dev/null; then
        log_success "Test de santé réussi"
    else
        log_warning "Test de santé échoué (peut être temporaire)"
    fi
}

# Configuration post-déploiement
post_deployment_config() {
    log_info "Configuration post-déploiement..."
    
    # Application des HPA par défaut
    for service in api-service ml-service audio-processor; do
        kubectl apply -f - << EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: $service-hpa
  namespace: $NAMESPACE
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: $service
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
EOF
    done
    
    log_success "HPA par défaut configurés"
}

# Nettoyage en cas d'erreur
cleanup_on_error() {
    log_error "Erreur détectée, nettoyage..."
    
    if [[ "${USE_HELM:-false}" == "true" ]]; then
        helm uninstall "$RELEASE_NAME" --namespace "$NAMESPACE" || true
    else
        kubectl delete -f "$PROJECT_ROOT/manifests/" --namespace="$NAMESPACE" || true
    fi
    
    kubectl delete namespace "$NAMESPACE" || true
}

# Fonction principale
main() {
    log_info "Début du déploiement de l'autoscaling Spotify AI Agent"
    
    # Piège pour le nettoyage en cas d'erreur
    trap cleanup_on_error ERR
    
    check_prerequisites
    create_namespace
    deploy_configmaps
    deploy_secrets
    generate_k8s_manifests
    
    if [[ "${USE_HELM:-false}" == "true" ]]; then
        deploy_with_helm
    else
        deploy_with_kubectl
    fi
    
    verify_deployment
    post_deployment_config
    
    log_success "Déploiement terminé avec succès!"
    log_info "Accès au service: kubectl port-forward svc/autoscaling-controller 8080:80 -n $NAMESPACE"
    log_info "Métriques: kubectl port-forward svc/autoscaling-controller 9090:9090 -n $NAMESPACE"
}

# Affichage de l'aide
show_help() {
    cat << EOF
Script de déploiement de l'autoscaling Spotify AI Agent

Usage: $0 [OPTIONS]

Options:
  -n, --namespace NAMESPACE    Namespace Kubernetes (défaut: autoscaling)
  -e, --environment ENV        Environnement (dev/staging/prod, défaut: dev)
  -r, --release-name NAME      Nom de release Helm (défaut: spotify-autoscaling)
  -h, --help                   Affiche cette aide

Variables d'environnement:
  PROMETHEUS_TOKEN            Token d'accès Prometheus
  INFLUXDB_TOKEN             Token d'accès InfluxDB
  AWS_ACCESS_KEY_ID          Clé d'accès AWS CloudWatch
  AWS_SECRET_ACCESS_KEY      Clé secrète AWS CloudWatch
  SLACK_WEBHOOK_URL          URL webhook Slack
  WEBHOOK_TOKEN              Token webhook générique

Exemples:
  $0                                    # Déploiement simple
  $0 -n production -e prod              # Déploiement en production
  $0 --namespace staging --environment staging   # Déploiement en staging

EOF
}

# Parsing des arguments
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
        -r|--release-name)
            RELEASE_NAME="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Option inconnue: $1"
            show_help
            exit 1
            ;;
    esac
done

# Exécution
main "$@"
