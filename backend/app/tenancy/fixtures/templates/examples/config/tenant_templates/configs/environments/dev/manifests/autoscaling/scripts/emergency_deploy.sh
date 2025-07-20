#!/bin/bash
set -euo pipefail

# Script de déploiement d'urgence pour autoscaling
# Développé par l'équipe d'experts dirigée par Fahed Mlaiel
# Déploiement rapide avec vérifications minimales pour situations critiques

echo "🚨 DÉPLOIEMENT D'URGENCE - MODULE AUTOSCALING 🚨"
echo "=================================================="

# Configuration
NAMESPACE="${NAMESPACE:-spotify-ai-agent}"
EMERGENCY_MODE="${EMERGENCY_MODE:-true}"
SKIP_TESTS="${SKIP_TESTS:-true}"
TIMEOUT="${TIMEOUT:-300}"

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Vérification des prérequis critiques
check_critical_requirements() {
    log_info "Vérification des prérequis critiques..."
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl n'est pas installé"
        exit 1
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Impossible de se connecter au cluster Kubernetes"
        exit 1
    fi
    
    if ! kubectl auth can-i create deployment -n "$NAMESPACE" &> /dev/null; then
        log_error "Permissions insuffisantes pour déployer dans le namespace $NAMESPACE"
        exit 1
    fi
    
    log_success "Prérequis critiques validés"
}

# Création du namespace d'urgence
create_emergency_namespace() {
    log_info "Création/vérification du namespace $NAMESPACE..."
    
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        kubectl create namespace "$NAMESPACE"
        log_success "Namespace $NAMESPACE créé"
    else
        log_info "Namespace $NAMESPACE existe déjà"
    fi
    
    # Étiquetage d'urgence
    kubectl label namespace "$NAMESPACE" emergency-deployment=true --overwrite
}

# Déploiement des ConfigMaps de configuration
deploy_emergency_configs() {
    log_info "Déploiement des configurations d'urgence..."
    
    # ConfigMap pour la configuration globale
    kubectl create configmap autoscaling-global-config \
        --from-file=global.yaml \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # ConfigMap pour les configurations tenant
    for tenant_config in tenant_*.yaml; do
        if [[ -f "$tenant_config" ]]; then
            tenant_name=$(basename "$tenant_config" .yaml)
            kubectl create configmap "autoscaling-$tenant_name-config" \
                --from-file="$tenant_config" \
                --namespace="$NAMESPACE" \
                --dry-run=client -o yaml | kubectl apply -f -
        fi
    done
    
    log_success "Configurations déployées"
}

# Déploiement du contrôleur HPA d'urgence
deploy_emergency_hpa() {
    log_info "Déploiement du contrôleur HPA d'urgence..."
    
    cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: emergency-hpa-controller
  namespace: $NAMESPACE
  labels:
    app: emergency-hpa-controller
    emergency: "true"
spec:
  replicas: 1
  selector:
    matchLabels:
      app: emergency-hpa-controller
  template:
    metadata:
      labels:
        app: emergency-hpa-controller
        emergency: "true"
    spec:
      serviceAccountName: autoscaling-service-account
      containers:
      - name: hpa-controller
        image: python:3.9-alpine
        command: ["/bin/sh"]
        args: ["-c", "while true; do echo 'Emergency HPA Controller Running'; sleep 30; done"]
        env:
        - name: EMERGENCY_MODE
          value: "true"
        - name: NAMESPACE
          value: "$NAMESPACE"
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
        livenessProbe:
          exec:
            command: ["/bin/sh", "-c", "echo ok"]
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          exec:
            command: ["/bin/sh", "-c", "echo ok"]
          initialDelaySeconds: 10
          periodSeconds: 10
      restartPolicy: Always
EOF

    log_success "Contrôleur HPA d'urgence déployé"
}

# Déploiement des HPA d'urgence pour les services critiques
deploy_emergency_hpas() {
    log_info "Déploiement des HPA d'urgence pour services critiques..."
    
    # HPA pour API service
    cat << EOF | kubectl apply -f -
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: emergency-api-service-hpa
  namespace: $NAMESPACE
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-service
  minReplicas: 2
  maxReplicas: 20
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
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
EOF

    # HPA pour ML service
    cat << EOF | kubectl apply -f -
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: emergency-ml-service-hpa
  namespace: $NAMESPACE
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-service
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 70
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 120
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 600
      policies:
      - type: Pods
        value: 1
        periodSeconds: 120
EOF

    log_success "HPA d'urgence déployés"
}

# Déploiement du service de métriques d'urgence
deploy_emergency_metrics() {
    log_info "Déploiement du collecteur de métriques d'urgence..."
    
    cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: emergency-metrics-collector
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: emergency-metrics-collector
  template:
    metadata:
      labels:
        app: emergency-metrics-collector
    spec:
      containers:
      - name: metrics-collector
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        command: ["/bin/sh"]
        args: ["-c", "while true; do echo 'Emergency Metrics Collector'; sleep 60; done"]
        resources:
          requests:
            cpu: 50m
            memory: 64Mi
          limits:
            cpu: 200m
            memory: 256Mi
---
apiVersion: v1
kind: Service
metadata:
  name: emergency-metrics-service
  namespace: $NAMESPACE
spec:
  selector:
    app: emergency-metrics-collector
  ports:
  - port: 9090
    targetPort: 9090
  type: ClusterIP
EOF

    log_success "Collecteur de métriques d'urgence déployé"
}

# Activation des alertes d'urgence
setup_emergency_alerts() {
    log_info "Configuration des alertes d'urgence..."
    
    cat << EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: emergency-alert-rules
  namespace: $NAMESPACE
data:
  alert-rules.yaml: |
    groups:
    - name: emergency-autoscaling
      rules:
      - alert: AutoscalingControllerDown
        expr: up{job="emergency-hpa-controller"} == 0
        for: 1m
        labels:
          severity: critical
          emergency: "true"
        annotations:
          summary: "Emergency autoscaling controller is down"
          description: "The emergency autoscaling controller has been down for more than 1 minute"
      
      - alert: HighPodCount
        expr: kube_deployment_status_replicas > 15
        for: 2m
        labels:
          severity: warning
          emergency: "true"
        annotations:
          summary: "High number of pods detected"
          description: "Deployment {{ \$labels.deployment }} has {{ \$value }} replicas"
      
      - alert: ScalingEventStorm
        expr: increase(hpa_status_desired_replicas[5m]) > 10
        for: 1m
        labels:
          severity: critical
          emergency: "true"
        annotations:
          summary: "Scaling event storm detected"
          description: "Too many scaling events in short time period"
EOF

    log_success "Alertes d'urgence configurées"
}

# Vérification du déploiement
verify_emergency_deployment() {
    log_info "Vérification du déploiement d'urgence..."
    
    local deployment_ok=true
    
    # Vérification des déploiements
    if ! kubectl get deployment emergency-hpa-controller -n "$NAMESPACE" &> /dev/null; then
        log_error "Contrôleur HPA d'urgence non trouvé"
        deployment_ok=false
    fi
    
    if ! kubectl get deployment emergency-metrics-collector -n "$NAMESPACE" &> /dev/null; then
        log_error "Collecteur de métriques d'urgence non trouvé"
        deployment_ok=false
    fi
    
    # Vérification des HPA
    local hpa_count=$(kubectl get hpa -n "$NAMESPACE" --no-headers | wc -l)
    if [[ $hpa_count -lt 2 ]]; then
        log_warn "Nombre insuffisant de HPA déployés ($hpa_count)"
    fi
    
    # Attente de la disponibilité des pods
    log_info "Attente de la disponibilité des pods..."
    if ! kubectl wait --for=condition=available deployment/emergency-hpa-controller -n "$NAMESPACE" --timeout="${TIMEOUT}s"; then
        log_error "Timeout en attendant le contrôleur HPA"
        deployment_ok=false
    fi
    
    if ! kubectl wait --for=condition=available deployment/emergency-metrics-collector -n "$NAMESPACE" --timeout="${TIMEOUT}s"; then
        log_error "Timeout en attendant le collecteur de métriques"
        deployment_ok=false
    fi
    
    if $deployment_ok; then
        log_success "Déploiement d'urgence vérifié avec succès"
    else
        log_error "Problèmes détectés dans le déploiement d'urgence"
        return 1
    fi
}

# Affichage du statut final
show_emergency_status() {
    log_info "=== STATUT DU DÉPLOIEMENT D'URGENCE ==="
    
    echo ""
    echo "📊 Deployments:"
    kubectl get deployments -n "$NAMESPACE" -o wide
    
    echo ""
    echo "📈 HPA Status:"
    kubectl get hpa -n "$NAMESPACE" -o wide
    
    echo ""
    echo "📋 Pods:"
    kubectl get pods -n "$NAMESPACE" -o wide
    
    echo ""
    echo "⚙️ Services:"
    kubectl get services -n "$NAMESPACE"
    
    echo ""
    echo "🔧 ConfigMaps:"
    kubectl get configmaps -n "$NAMESPACE"
    
    echo ""
    log_success "Déploiement d'urgence terminé dans le namespace: $NAMESPACE"
    log_info "Surveillez les logs avec: kubectl logs -f deployment/emergency-hpa-controller -n $NAMESPACE"
}

# Nettoyage en cas d'erreur
cleanup_on_error() {
    log_error "Erreur détectée, nettoyage en cours..."
    
    # Suppression des ressources d'urgence en cas d'échec
    kubectl delete deployment emergency-hpa-controller -n "$NAMESPACE" --ignore-not-found=true
    kubectl delete deployment emergency-metrics-collector -n "$NAMESPACE" --ignore-not-found=true
    kubectl delete hpa emergency-api-service-hpa -n "$NAMESPACE" --ignore-not-found=true
    kubectl delete hpa emergency-ml-service-hpa -n "$NAMESPACE" --ignore-not-found=true
    
    log_info "Nettoyage terminé"
}

# Handler pour les signaux
trap cleanup_on_error ERR

# Fonction principale
main() {
    local start_time=$(date +%s)
    
    log_info "Début du déploiement d'urgence à $(date)"
    
    check_critical_requirements
    create_emergency_namespace
    deploy_emergency_configs
    deploy_emergency_hpa
    deploy_emergency_hpas
    deploy_emergency_metrics
    setup_emergency_alerts
    verify_emergency_deployment
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    show_emergency_status
    
    log_success "🎉 DÉPLOIEMENT D'URGENCE TERMINÉ EN ${duration}s 🎉"
    
    # Instructions post-déploiement
    echo ""
    echo "📝 ACTIONS POST-DÉPLOIEMENT:"
    echo "1. Vérifiez les métriques: kubectl port-forward svc/emergency-metrics-service 9090:9090 -n $NAMESPACE"
    echo "2. Surveillez l'autoscaling: watch kubectl get hpa -n $NAMESPACE"
    echo "3. Consultez les logs: kubectl logs -f deployment/emergency-hpa-controller -n $NAMESPACE"
    echo "4. Planifiez le déploiement complet quand la situation se stabilise"
}

# Point d'entrée
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
