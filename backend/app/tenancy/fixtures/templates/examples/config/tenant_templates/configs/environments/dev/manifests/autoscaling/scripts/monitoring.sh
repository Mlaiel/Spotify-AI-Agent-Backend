#!/bin/bash
set -euo pipefail

# Script de surveillance et observabilité pour l'autoscaling
# Développé par l'équipe d'experts dirigée par Fahed Mlaiel
# Monitoring complet avec alertes et dashboards

echo "📊 SYSTÈME DE SURVEILLANCE AUTOSCALING 📊"
echo "=========================================="

# Configuration
NAMESPACE="${NAMESPACE:-spotify-ai-agent}"
MONITORING_NAMESPACE="${MONITORING_NAMESPACE:-monitoring}"
GRAFANA_PORT="${GRAFANA_PORT:-3000}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-9090}"
ALERT_WEBHOOK="${ALERT_WEBHOOK:-}"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_monitor() { echo -e "${PURPLE}[MONITOR]${NC} $1"; }

# Vérification des prérequis
check_monitoring_requirements() {
    log_info "Vérification des prérequis de monitoring..."
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl requis mais non installé"
        exit 1
    fi
    
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_error "Namespace $NAMESPACE non trouvé"
        exit 1
    fi
    
    log_success "Prérequis validés"
}

# Création du namespace de monitoring
setup_monitoring_namespace() {
    log_info "Configuration du namespace de monitoring..."
    
    if ! kubectl get namespace "$MONITORING_NAMESPACE" &> /dev/null; then
        kubectl create namespace "$MONITORING_NAMESPACE"
        log_success "Namespace $MONITORING_NAMESPACE créé"
    fi
    
    # Labels pour la surveillance
    kubectl label namespace "$MONITORING_NAMESPACE" monitoring=true --overwrite
    kubectl label namespace "$NAMESPACE" monitored=true --overwrite
}

# Déploiement de Prometheus pour l'autoscaling
deploy_prometheus() {
    log_info "Déploiement de Prometheus pour l'autoscaling..."
    
    # Configuration Prometheus spécialisée
    cat << 'EOF' | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-autoscaling-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    rule_files:
    - "/etc/prometheus/rules/*.yml"
    
    alerting:
      alertmanagers:
      - static_configs:
        - targets: ["alertmanager:9093"]
    
    scrape_configs:
    # Kubernetes API Server
    - job_name: 'kubernetes-apiserver'
      kubernetes_sd_configs:
      - role: endpoints
      scheme: https
      tls_config:
        ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
      bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
      relabel_configs:
      - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
        action: keep
        regex: default;kubernetes;https
    
    # HPA Metrics
    - job_name: 'hpa-metrics'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: [spotify-ai-agent]
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: .*hpa.*
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
    
    # Node Exporter
    - job_name: 'node-exporter'
      kubernetes_sd_configs:
      - role: node
      relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)
      - target_label: __address__
        replacement: kubernetes.default.svc:443
      - source_labels: [__meta_kubernetes_node_name]
        regex: (.+)
        target_label: __metrics_path__
        replacement: /api/v1/nodes/${1}/proxy/metrics
    
    # Kubelet cAdvisor
    - job_name: 'kubernetes-cadvisor'
      kubernetes_sd_configs:
      - role: node
      scheme: https
      tls_config:
        ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
      bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
      relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)
      - target_label: __address__
        replacement: kubernetes.default.svc:443
      - source_labels: [__meta_kubernetes_node_name]
        regex: (.+)
        target_label: __metrics_path__
        replacement: /api/v1/nodes/${1}/proxy/metrics/cadvisor
    
    # Pods autoscaling
    - job_name: 'autoscaling-pods'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: [spotify-ai-agent]
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_component]
        action: keep
        regex: autoscaling
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus-autoscaling
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus-autoscaling
  template:
    metadata:
      labels:
        app: prometheus-autoscaling
    spec:
      serviceAccountName: prometheus
      containers:
      - name: prometheus
        image: prom/prometheus:v2.40.0
        args:
        - '--config.file=/etc/prometheus/prometheus.yml'
        - '--storage.tsdb.path=/prometheus/'
        - '--web.console.libraries=/etc/prometheus/console_libraries'
        - '--web.console.templates=/etc/prometheus/consoles'
        - '--storage.tsdb.retention.time=7d'
        - '--web.enable-lifecycle'
        - '--storage.tsdb.no-lockfile'
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: prometheus-config
          mountPath: /etc/prometheus/
        - name: prometheus-storage
          mountPath: /prometheus/
        resources:
          requests:
            cpu: 200m
            memory: 512Mi
          limits:
            cpu: 1000m
            memory: 2Gi
      volumes:
      - name: prometheus-config
        configMap:
          name: prometheus-autoscaling-config
      - name: prometheus-storage
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus-autoscaling
  namespace: monitoring
spec:
  selector:
    app: prometheus-autoscaling
  ports:
  - port: 9090
    targetPort: 9090
  type: ClusterIP
EOF

    log_success "Prometheus déployé"
}

# Déploiement des règles d'alerte
deploy_alert_rules() {
    log_info "Déploiement des règles d'alerte autoscaling..."
    
    cat << 'EOF' | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: autoscaling-alert-rules
  namespace: monitoring
data:
  autoscaling.yml: |
    groups:
    - name: autoscaling.rules
      interval: 30s
      rules:
      
      # Alerte HPA non fonctionnel
      - alert: HPANotWorking
        expr: kube_horizontalpodautoscaler_status_condition{condition="ScalingActive", status="false"} == 1
        for: 5m
        labels:
          severity: critical
          component: autoscaling
        annotations:
          summary: "HPA {{ $labels.horizontalpodautoscaler }} not scaling"
          description: "HPA {{ $labels.horizontalpodautoscaler }} in namespace {{ $labels.namespace }} has not been scaling for 5 minutes"
      
      # Alerte scaling trop fréquent
      - alert: FrequentScaling
        expr: increase(kube_horizontalpodautoscaler_status_desired_replicas[10m]) > 5
        for: 2m
        labels:
          severity: warning
          component: autoscaling
        annotations:
          summary: "Frequent scaling detected for {{ $labels.horizontalpodautoscaler }}"
          description: "HPA {{ $labels.horizontalpodautoscaler }} has scaled more than 5 times in 10 minutes"
      
      # Alerte limite max atteinte
      - alert: HPAMaxReplicasReached
        expr: kube_horizontalpodautoscaler_status_current_replicas == on(namespace, horizontalpodautoscaler) kube_horizontalpodautoscaler_spec_max_replicas
        for: 3m
        labels:
          severity: warning
          component: autoscaling
        annotations:
          summary: "HPA {{ $labels.horizontalpodautoscaler }} at maximum replicas"
          description: "HPA {{ $labels.horizontalpodautoscaler }} has reached its maximum replica count of {{ $value }}"
      
      # Alerte CPU élevé persistant
      - alert: HighCPUUsageNonScaling
        expr: (sum by (namespace, pod) (rate(container_cpu_usage_seconds_total[5m])) * 100) > 90
        for: 10m
        labels:
          severity: critical
          component: autoscaling
        annotations:
          summary: "High CPU usage without scaling"
          description: "Pod {{ $labels.pod }} in namespace {{ $labels.namespace }} has high CPU usage ({{ $value }}%) for 10 minutes"
      
      # Alerte mémoire élevée persistante
      - alert: HighMemoryUsageNonScaling
        expr: (container_memory_working_set_bytes / container_spec_memory_limit_bytes) * 100 > 85
        for: 10m
        labels:
          severity: warning
          component: autoscaling
        annotations:
          summary: "High memory usage without scaling"
          description: "Pod {{ $labels.pod }} has high memory usage ({{ $value }}%) for 10 minutes"
      
      # Alerte métriques manquantes
      - alert: MissingMetrics
        expr: up{job="autoscaling-pods"} == 0
        for: 2m
        labels:
          severity: critical
          component: monitoring
        annotations:
          summary: "Autoscaling metrics missing"
          description: "No metrics received from autoscaling components for 2 minutes"
      
      # Alerte thrashing (oscillation)
      - alert: ScalingThrashing
        expr: changes(kube_horizontalpodautoscaler_status_desired_replicas[30m]) > 10
        for: 5m
        labels:
          severity: warning
          component: autoscaling
        annotations:
          summary: "Scaling thrashing detected"
          description: "HPA {{ $labels.horizontalpodautoscaler }} is oscillating ({{ $value }} changes in 30 minutes)"
      
      # Alerte coût élevé
      - alert: HighAutoscalingCost
        expr: sum by (namespace) (kube_horizontalpodautoscaler_status_current_replicas) > 50
        for: 15m
        labels:
          severity: warning
          component: cost
        annotations:
          summary: "High replica count detected"
          description: "Namespace {{ $labels.namespace }} has {{ $value }} total replicas across all HPAs"
      
      # Alerte pod OOMKilled
      - alert: PodOOMKilled
        expr: increase(kube_pod_container_status_restarts_total[1h]) > 3
        for: 0s
        labels:
          severity: critical
          component: autoscaling
        annotations:
          summary: "Pod frequently restarting"
          description: "Pod {{ $labels.pod }} in namespace {{ $labels.namespace }} has restarted {{ $value }} times in the last hour"
EOF

    log_success "Règles d'alerte déployées"
}

# Déploiement de Grafana avec dashboards
deploy_grafana() {
    log_info "Déploiement de Grafana avec dashboards autoscaling..."
    
    cat << 'EOF' | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-autoscaling-dashboard
  namespace: monitoring
data:
  autoscaling-dashboard.json: |
    {
      "dashboard": {
        "id": null,
        "title": "Autoscaling Monitoring",
        "tags": ["autoscaling", "kubernetes"],
        "timezone": "browser",
        "panels": [
          {
            "id": 1,
            "title": "Current Replicas vs Desired",
            "type": "graph",
            "targets": [
              {
                "expr": "kube_horizontalpodautoscaler_status_current_replicas",
                "legendFormat": "Current - {{horizontalpodautoscaler}}"
              },
              {
                "expr": "kube_horizontalpodautoscaler_status_desired_replicas",
                "legendFormat": "Desired - {{horizontalpodautoscaler}}"
              }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
          },
          {
            "id": 2,
            "title": "CPU Utilization",
            "type": "graph",
            "targets": [
              {
                "expr": "sum by (namespace, pod) (rate(container_cpu_usage_seconds_total[5m])) * 100",
                "legendFormat": "{{namespace}}/{{pod}}"
              }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
          },
          {
            "id": 3,
            "title": "Memory Utilization",
            "type": "graph",
            "targets": [
              {
                "expr": "(container_memory_working_set_bytes / container_spec_memory_limit_bytes) * 100",
                "legendFormat": "{{namespace}}/{{pod}}"
              }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
          },
          {
            "id": 4,
            "title": "Scaling Events",
            "type": "graph",
            "targets": [
              {
                "expr": "increase(kube_horizontalpodautoscaler_status_desired_replicas[5m])",
                "legendFormat": "{{horizontalpodautoscaler}}"
              }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
          }
        ],
        "time": {"from": "now-1h", "to": "now"},
        "refresh": "30s"
      }
    }
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana-autoscaling
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana-autoscaling
  template:
    metadata:
      labels:
        app: grafana-autoscaling
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:9.0.0
        ports:
        - containerPort: 3000
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          value: "admin123"
        - name: GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH
          value: "/var/lib/grafana/dashboards/autoscaling-dashboard.json"
        volumeMounts:
        - name: grafana-dashboard
          mountPath: /var/lib/grafana/dashboards/
        resources:
          requests:
            cpu: 100m
            memory: 256Mi
          limits:
            cpu: 500m
            memory: 1Gi
      volumes:
      - name: grafana-dashboard
        configMap:
          name: grafana-autoscaling-dashboard
---
apiVersion: v1
kind: Service
metadata:
  name: grafana-autoscaling
  namespace: monitoring
spec:
  selector:
    app: grafana-autoscaling
  ports:
  - port: 3000
    targetPort: 3000
  type: ClusterIP
EOF

    log_success "Grafana déployé"
}

# Service de notification d'alertes
deploy_alertmanager() {
    log_info "Déploiement d'Alertmanager..."
    
    cat << 'EOF' | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: alertmanager-config
  namespace: monitoring
data:
  alertmanager.yml: |
    global:
      smtp_smarthost: 'localhost:587'
      smtp_from: 'autoscaling-alerts@company.com'
    
    route:
      group_by: ['alertname', 'component']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 1h
      receiver: 'default'
      routes:
      - match:
          severity: critical
        receiver: 'critical-alerts'
      - match:
          component: autoscaling
        receiver: 'autoscaling-team'
    
    receivers:
    - name: 'default'
      webhook_configs:
      - url: 'http://webhook-receiver:8080/alerts'
        send_resolved: true
    
    - name: 'critical-alerts'
      email_configs:
      - to: 'ops-team@company.com'
        subject: 'CRITICAL: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
      webhook_configs:
      - url: 'http://webhook-receiver:8080/critical'
        send_resolved: true
    
    - name: 'autoscaling-team'
      email_configs:
      - to: 'autoscaling-team@company.com'
        subject: 'Autoscaling Alert: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Namespace: {{ .Labels.namespace }}
          {{ end }}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: alertmanager
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: alertmanager
  template:
    metadata:
      labels:
        app: alertmanager
    spec:
      containers:
      - name: alertmanager
        image: prom/alertmanager:v0.25.0
        args:
        - '--config.file=/etc/alertmanager/alertmanager.yml'
        - '--storage.path=/alertmanager'
        - '--web.external-url=http://localhost:9093'
        ports:
        - containerPort: 9093
        volumeMounts:
        - name: alertmanager-config
          mountPath: /etc/alertmanager/
        resources:
          requests:
            cpu: 50m
            memory: 128Mi
          limits:
            cpu: 200m
            memory: 512Mi
      volumes:
      - name: alertmanager-config
        configMap:
          name: alertmanager-config
---
apiVersion: v1
kind: Service
metadata:
  name: alertmanager
  namespace: monitoring
spec:
  selector:
    app: alertmanager
  ports:
  - port: 9093
    targetPort: 9093
  type: ClusterIP
EOF

    log_success "Alertmanager déployé"
}

# Surveillance en temps réel
start_realtime_monitoring() {
    log_monitor "Démarrage de la surveillance en temps réel..."
    
    # Function pour afficher les métriques
    show_autoscaling_metrics() {
        clear
        echo "=== SURVEILLANCE AUTOSCALING EN TEMPS RÉEL ==="
        echo "Namespace: $NAMESPACE | $(date)"
        echo "=============================================="
        
        echo ""
        echo "📊 HPA Status:"
        kubectl get hpa -n "$NAMESPACE" -o custom-columns="NAME:.metadata.name,CURRENT:.status.currentReplicas,DESIRED:.status.desiredReplicas,MIN:.spec.minReplicas,MAX:.spec.maxReplicas,CPU:.status.currentCPUUtilizationPercentage" 2>/dev/null || echo "Aucun HPA trouvé"
        
        echo ""
        echo "🔄 Recent Scaling Events:"
        kubectl get events -n "$NAMESPACE" --field-selector reason=SuccessfulRescale --sort-by='.lastTimestamp' | tail -5 2>/dev/null || echo "Aucun événement récent"
        
        echo ""
        echo "📈 Pods par Deployment:"
        kubectl get deployments -n "$NAMESPACE" -o custom-columns="NAME:.metadata.name,READY:.status.readyReplicas,AVAILABLE:.status.availableReplicas,UP-TO-DATE:.status.updatedReplicas" 2>/dev/null || echo "Aucun deployment trouvé"
        
        echo ""
        echo "💾 Resource Usage (Top Pods):"
        kubectl top pods -n "$NAMESPACE" --sort-by=cpu 2>/dev/null | head -10 || echo "Métriques non disponibles"
        
        echo ""
        echo "⚠️  Recent Alerts:"
        if command -v curl &> /dev/null; then
            curl -s "http://localhost:$PROMETHEUS_PORT/api/v1/alerts" 2>/dev/null | jq -r '.data.alerts[] | select(.labels.component=="autoscaling") | .annotations.summary' 2>/dev/null | head -3 || echo "Aucune alerte récente"
        else
            echo "curl non disponible pour vérifier les alertes"
        fi
        
        echo ""
        echo "🔧 Appuyez sur Ctrl+C pour arrêter la surveillance"
    }
    
    # Boucle de surveillance
    while true; do
        show_autoscaling_metrics
        sleep 10
    done
}

# Tests de charge pour tester l'autoscaling
run_load_test() {
    log_info "Lancement du test de charge pour autoscaling..."
    
    cat << 'EOF' | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: load-generator
  namespace: spotify-ai-agent
spec:
  replicas: 1
  selector:
    matchLabels:
      app: load-generator
  template:
    metadata:
      labels:
        app: load-generator
    spec:
      containers:
      - name: load-generator
        image: busybox
        command: ["/bin/sh"]
        args: ["-c", "while true; do wget -q --spider http://api-service:8080/health || true; sleep 0.1; done"]
        resources:
          requests:
            cpu: 10m
            memory: 32Mi
          limits:
            cpu: 100m
            memory: 128Mi
EOF

    log_success "Générateur de charge déployé"
    log_info "Surveillez l'autoscaling avec: watch kubectl get hpa -n $NAMESPACE"
}

# Création des ServiceAccounts nécessaires
create_service_accounts() {
    log_info "Création des ServiceAccounts pour le monitoring..."
    
    cat << 'EOF' | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: prometheus
  namespace: monitoring
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: prometheus
rules:
- apiGroups: [""]
  resources: ["nodes", "services", "endpoints", "pods"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["configmaps"]
  verbs: ["get"]
- apiGroups: ["networking.k8s.io"]
  resources: ["ingresses"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["autoscaling"]
  resources: ["horizontalpodautoscalers"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: prometheus
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: prometheus
subjects:
- kind: ServiceAccount
  name: prometheus
  namespace: monitoring
EOF

    log_success "ServiceAccounts créés"
}

# Affichage des informations d'accès
show_access_info() {
    log_info "=== INFORMATIONS D'ACCÈS ==="
    
    echo ""
    echo "📊 Prometheus:"
    echo "   Port-forward: kubectl port-forward svc/prometheus-autoscaling $PROMETHEUS_PORT:9090 -n $MONITORING_NAMESPACE"
    echo "   URL: http://localhost:$PROMETHEUS_PORT"
    
    echo ""
    echo "📈 Grafana:"
    echo "   Port-forward: kubectl port-forward svc/grafana-autoscaling $GRAFANA_PORT:3000 -n $MONITORING_NAMESPACE"
    echo "   URL: http://localhost:$GRAFANA_PORT"
    echo "   Login: admin / admin123"
    
    echo ""
    echo "🚨 Alertmanager:"
    echo "   Port-forward: kubectl port-forward svc/alertmanager 9093:9093 -n $MONITORING_NAMESPACE"
    echo "   URL: http://localhost:9093"
    
    echo ""
    echo "⚡ Tests rapides:"
    echo "   HPA Status: kubectl get hpa -n $NAMESPACE"
    echo "   Events: kubectl get events -n $NAMESPACE | grep -i scale"
    echo "   Top Pods: kubectl top pods -n $NAMESPACE"
    
    echo ""
    echo "🔧 Commandes utiles:"
    echo "   Surveillance: $0 --monitor"
    echo "   Test charge: $0 --load-test"
    echo "   Cleanup: $0 --cleanup"
}

# Nettoyage des ressources de monitoring
cleanup_monitoring() {
    log_warn "Nettoyage des ressources de monitoring..."
    
    kubectl delete deployment prometheus-autoscaling -n "$MONITORING_NAMESPACE" --ignore-not-found=true
    kubectl delete deployment grafana-autoscaling -n "$MONITORING_NAMESPACE" --ignore-not-found=true
    kubectl delete deployment alertmanager -n "$MONITORING_NAMESPACE" --ignore-not-found=true
    kubectl delete deployment load-generator -n "$NAMESPACE" --ignore-not-found=true
    
    kubectl delete service prometheus-autoscaling -n "$MONITORING_NAMESPACE" --ignore-not-found=true
    kubectl delete service grafana-autoscaling -n "$MONITORING_NAMESPACE" --ignore-not-found=true
    kubectl delete service alertmanager -n "$MONITORING_NAMESPACE" --ignore-not-found=true
    
    kubectl delete configmap prometheus-autoscaling-config -n "$MONITORING_NAMESPACE" --ignore-not-found=true
    kubectl delete configmap autoscaling-alert-rules -n "$MONITORING_NAMESPACE" --ignore-not-found=true
    kubectl delete configmap grafana-autoscaling-dashboard -n "$MONITORING_NAMESPACE" --ignore-not-found=true
    kubectl delete configmap alertmanager-config -n "$MONITORING_NAMESPACE" --ignore-not-found=true
    
    log_success "Nettoyage terminé"
}

# Fonction principale
main() {
    case "${1:-setup}" in
        "setup")
            check_monitoring_requirements
            setup_monitoring_namespace
            create_service_accounts
            deploy_prometheus
            deploy_alert_rules
            deploy_grafana
            deploy_alertmanager
            show_access_info
            log_success "🎉 Système de surveillance déployé avec succès!"
            ;;
        "--monitor"|"monitor")
            start_realtime_monitoring
            ;;
        "--load-test"|"load-test")
            run_load_test
            ;;
        "--cleanup"|"cleanup")
            cleanup_monitoring
            ;;
        "--help"|"help")
            echo "Usage: $0 [setup|monitor|load-test|cleanup|help]"
            echo ""
            echo "Commands:"
            echo "  setup      - Deploy monitoring stack (default)"
            echo "  monitor    - Start real-time monitoring"
            echo "  load-test  - Deploy load generator"
            echo "  cleanup    - Clean up monitoring resources"
            echo "  help       - Show this help"
            ;;
        *)
            log_error "Option inconnue: $1"
            log_info "Utilisez '$0 help' pour voir les options disponibles"
            exit 1
            ;;
    esac
}

# Gestion des signaux
trap 'log_info "Arrêt en cours..."; exit 0' INT TERM

# Point d'entrée
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
