#!/bin/bash
# Script de monitoring et surveillance de l'autoscaling - Spotify AI Agent
# Développé par l'équipe d'experts dirigée par Fahed Mlaiel

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-autoscaling}"
REFRESH_INTERVAL="${REFRESH_INTERVAL:-10}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-table}"  # table, json, prometheus

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Fonctions utilitaires
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Fonction de nettoyage du terminal
clear_screen() {
    if [[ "$OUTPUT_FORMAT" == "table" ]]; then
        clear
    fi
}

# Collecte des métriques de scaling
collect_scaling_metrics() {
    local metrics=()
    
    # HPA Status
    local hpa_data
    hpa_data=$(kubectl get hpa -n "$NAMESPACE" -o json 2>/dev/null || echo '{"items":[]}')
    
    # Deployments
    local deployment_data
    deployment_data=$(kubectl get deployments -n "$NAMESPACE" -o json 2>/dev/null || echo '{"items":[]}')
    
    # Pods metrics
    local pods_data
    pods_data=$(kubectl top pods -n "$NAMESPACE" --no-headers 2>/dev/null || echo "")
    
    # Nodes metrics
    local nodes_data
    nodes_data=$(kubectl top nodes --no-headers 2>/dev/null || echo "")
    
    echo "$hpa_data|$deployment_data|$pods_data|$nodes_data"
}

# Affichage du tableau de bord
display_dashboard() {
    local metrics_data="$1"
    IFS='|' read -r hpa_data deployment_data pods_data nodes_data <<< "$metrics_data"
    
    clear_screen
    
    echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║                    SPOTIFY AI AGENT - AUTOSCALING DASHBOARD                 ║${NC}"
    echo -e "${BOLD}${CYAN}║                     Développé par l'équipe Fahed Mlaiel                     ║${NC}"
    echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo
    
    # Timestamp
    echo -e "${BOLD}Dernière mise à jour:${NC} $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${BOLD}Namespace:${NC} $NAMESPACE"
    echo -e "${BOLD}Intervalle de rafraîchissement:${NC} ${REFRESH_INTERVAL}s"
    echo
    
    # Section HPA
    display_hpa_status "$hpa_data"
    echo
    
    # Section Deployments
    display_deployment_status "$deployment_data"
    echo
    
    # Section Pods
    display_pods_metrics "$pods_data"
    echo
    
    # Section Nodes
    display_nodes_metrics "$nodes_data"
    echo
    
    # Section Recommandations
    display_recommendations
    echo
    
    # Légende
    echo -e "${BOLD}Légende:${NC}"
    echo -e "  ${GREEN}●${NC} Optimal    ${YELLOW}●${NC} Attention    ${RED}●${NC} Critique"
    echo
    echo -e "${CYAN}Raccourcis: q=quitter, r=rafraîchir, h=aide${NC}"
}

# Affichage du statut HPA
display_hpa_status() {
    local hpa_data="$1"
    
    echo -e "${BOLD}${BLUE}┌─ HORIZONTAL POD AUTOSCALERS ─────────────────────────────────────────────────┐${NC}"
    
    if [[ "$hpa_data" == '{"items":[]}' ]]; then
        echo -e "${YELLOW}│ Aucun HPA configuré${NC}"
        echo -e "${BLUE}└──────────────────────────────────────────────────────────────────────────────┘${NC}"
        return
    fi
    
    printf "│ %-20s %-10s %-10s %-10s %-15s %-10s │\n" "SERVICE" "MIN" "MAX" "CURRENT" "CPU TARGET" "STATUS"
    echo -e "${BLUE}├──────────────────────────────────────────────────────────────────────────────┤${NC}"
    
    local hpa_items
    hpa_items=$(echo "$hpa_data" | jq -r '.items[] | "\(.metadata.name)|\(.spec.minReplicas)|\(.spec.maxReplicas)|\(.status.currentReplicas // "N/A")|\(.spec.metrics[0].resource.target.averageUtilization // "N/A")|\(.status.conditions[-1].type // "Unknown")"')
    
    while IFS='|' read -r name min_replicas max_replicas current_replicas cpu_target status; do
        local status_icon
        case "$status" in
            "ScalingActive") status_icon="${GREEN}●${NC}" ;;
            "AbleToScale") status_icon="${GREEN}●${NC}" ;;
            "ScalingLimited") status_icon="${YELLOW}●${NC}" ;;
            *) status_icon="${RED}●${NC}" ;;
        esac
        
        printf "│ %s %-20s %-10s %-10s %-10s %-15s %-10s │\n" \
            "$status_icon" "$name" "$min_replicas" "$max_replicas" "$current_replicas" "${cpu_target}%" "$status"
    done <<< "$hpa_items"
    
    echo -e "${BLUE}└──────────────────────────────────────────────────────────────────────────────┘${NC}"
}

# Affichage du statut des déploiements
display_deployment_status() {
    local deployment_data="$1"
    
    echo -e "${BOLD}${BLUE}┌─ DEPLOYMENTS STATUS ─────────────────────────────────────────────────────────┐${NC}"
    
    if [[ "$deployment_data" == '{"items":[]}' ]]; then
        echo -e "${YELLOW}│ Aucun déploiement trouvé${NC}"
        echo -e "${BLUE}└──────────────────────────────────────────────────────────────────────────────┘${NC}"
        return
    fi
    
    printf "│ %-25s %-8s %-8s %-8s %-20s %-10s │\n" "DEPLOYMENT" "DESIRED" "CURRENT" "READY" "AGE" "STATUS"
    echo -e "${BLUE}├──────────────────────────────────────────────────────────────────────────────┤${NC}"
    
    local deployments
    deployments=$(echo "$deployment_data" | jq -r '.items[] | "\(.metadata.name)|\(.spec.replicas // 0)|\(.status.replicas // 0)|\(.status.readyReplicas // 0)|\(.metadata.creationTimestamp)"')
    
    while IFS='|' read -r name desired current ready created; do
        local age
        age=$(date -d "$created" +%s 2>/dev/null || echo "0")
        local now
        now=$(date +%s)
        local age_diff=$((now - age))
        local age_str
        
        if [[ $age_diff -gt 86400 ]]; then
            age_str="$((age_diff / 86400))d"
        elif [[ $age_diff -gt 3600 ]]; then
            age_str="$((age_diff / 3600))h"
        else
            age_str="$((age_diff / 60))m"
        fi
        
        local status_icon
        if [[ "$ready" == "$desired" ]] && [[ "$desired" -gt 0 ]]; then
            status_icon="${GREEN}●${NC}"
            status="Ready"
        elif [[ "$ready" -lt "$desired" ]]; then
            status_icon="${YELLOW}●${NC}"
            status="Scaling"
        else
            status_icon="${RED}●${NC}"
            status="Error"
        fi
        
        printf "│ %s %-25s %-8s %-8s %-8s %-20s %-10s │\n" \
            "$status_icon" "$name" "$desired" "$current" "$ready" "$age_str" "$status"
    done <<< "$deployments"
    
    echo -e "${BLUE}└──────────────────────────────────────────────────────────────────────────────┘${NC}"
}

# Affichage des métriques pods
display_pods_metrics() {
    local pods_data="$1"
    
    echo -e "${BOLD}${BLUE}┌─ PODS RESOURCES USAGE ───────────────────────────────────────────────────────┐${NC}"
    
    if [[ -z "$pods_data" ]]; then
        echo -e "${YELLOW}│ Métriques pods non disponibles (metrics-server requis)${NC}"
        echo -e "${BLUE}└──────────────────────────────────────────────────────────────────────────────┘${NC}"
        return
    fi
    
    printf "│ %-30s %-15s %-15s %-10s │\n" "POD" "CPU" "MEMORY" "STATUS"
    echo -e "${BLUE}├──────────────────────────────────────────────────────────────────────────────┤${NC}"
    
    while read -r line; do
        if [[ -n "$line" ]]; then
            read -r name cpu memory <<< "$line"
            
            # Détermination du statut basé sur l'utilisation
            local cpu_value
            cpu_value=$(echo "$cpu" | sed 's/m$//')
            local status_icon
            
            if [[ "$cpu_value" -gt 800 ]]; then
                status_icon="${RED}●${NC}"
                status="High"
            elif [[ "$cpu_value" -gt 500 ]]; then
                status_icon="${YELLOW}●${NC}"
                status="Medium"
            else
                status_icon="${GREEN}●${NC}"
                status="Low"
            fi
            
            printf "│ %s %-30s %-15s %-15s %-10s │\n" \
                "$status_icon" "$name" "$cpu" "$memory" "$status"
        fi
    done <<< "$pods_data"
    
    echo -e "${BLUE}└──────────────────────────────────────────────────────────────────────────────┘${NC}"
}

# Affichage des métriques nodes
display_nodes_metrics() {
    local nodes_data="$1"
    
    echo -e "${BOLD}${BLUE}┌─ CLUSTER NODES USAGE ────────────────────────────────────────────────────────┐${NC}"
    
    if [[ -z "$nodes_data" ]]; then
        echo -e "${YELLOW}│ Métriques nodes non disponibles${NC}"
        echo -e "${BLUE}└──────────────────────────────────────────────────────────────────────────────┘${NC}"
        return
    fi
    
    printf "│ %-25s %-15s %-15s %-20s %-10s │\n" "NODE" "CPU" "MEMORY" "CPU%" "STATUS"
    echo -e "${BLUE}├──────────────────────────────────────────────────────────────────────────────┤${NC}"
    
    while read -r line; do
        if [[ -n "$line" ]]; then
            read -r name cpu memory cpu_percent memory_percent <<< "$line"
            
            # Calcul du pourcentage CPU si disponible
            local cpu_usage_percent="N/A"
            if [[ "$cpu" =~ ([0-9]+)m ]]; then
                local cpu_millis="${BASH_REMATCH[1]}"
                # Estimation basée sur une capacité moyenne de 2000m par core
                cpu_usage_percent="$((cpu_millis / 20))%"
            fi
            
            local status_icon
            local cpu_val
            cpu_val=$(echo "$cpu_usage_percent" | sed 's/%$//')
            
            if [[ "$cpu_val" != "N/A" ]] && [[ "$cpu_val" -gt 80 ]]; then
                status_icon="${RED}●${NC}"
                status="Critical"
            elif [[ "$cpu_val" != "N/A" ]] && [[ "$cpu_val" -gt 60 ]]; then
                status_icon="${YELLOW}●${NC}"
                status="Warning"
            else
                status_icon="${GREEN}●${NC}"
                status="OK"
            fi
            
            printf "│ %s %-25s %-15s %-15s %-20s %-10s │\n" \
                "$status_icon" "$name" "$cpu" "$memory" "$cpu_usage_percent" "$status"
        fi
    done <<< "$nodes_data"
    
    echo -e "${BLUE}└──────────────────────────────────────────────────────────────────────────────┘${NC}"
}

# Affichage des recommandations
display_recommendations() {
    echo -e "${BOLD}${BLUE}┌─ AUTOSCALING RECOMMENDATIONS ───────────────────────────────────────────────┐${NC}"
    
    # Analyse automatique et recommandations
    local recommendations=()
    
    # Vérification des HPA sans métriques
    local hpa_without_metrics
    hpa_without_metrics=$(kubectl get hpa -n "$NAMESPACE" -o json 2>/dev/null | jq -r '.items[] | select(.status.currentMetrics == null) | .metadata.name' 2>/dev/null || echo "")
    
    if [[ -n "$hpa_without_metrics" ]]; then
        while read -r hpa_name; do
            if [[ -n "$hpa_name" ]]; then
                recommendations+=("${YELLOW}⚠${NC} HPA '$hpa_name' sans métriques - vérifier metrics-server")
            fi
        done <<< "$hpa_without_metrics"
    fi
    
    # Recommandations de ressources
    local high_cpu_pods
    high_cpu_pods=$(kubectl top pods -n "$NAMESPACE" --no-headers 2>/dev/null | awk '$2 ~ /^[0-9]+m$/ && $2+0 > 800 {print $1}' || echo "")
    
    if [[ -n "$high_cpu_pods" ]]; then
        while read -r pod_name; do
            if [[ -n "$pod_name" ]]; then
                recommendations+=("${RED}●${NC} Pod '$pod_name' haute utilisation CPU - considérer scaling up")
            fi
        done <<< "$high_cpu_pods"
    fi
    
    # Recommandations de coût
    local over_provisioned
    over_provisioned=$(kubectl top pods -n "$NAMESPACE" --no-headers 2>/dev/null | awk '$2 ~ /^[0-9]+m$/ && $2+0 < 50 {print $1}' | head -3 || echo "")
    
    if [[ -n "$over_provisioned" ]]; then
        recommendations+=("${GREEN}💰${NC} Pods sous-utilisés détectés - optimisation des coûts possible")
    fi
    
    # Affichage des recommandations
    if [[ ${#recommendations[@]} -eq 0 ]]; then
        echo -e "${GREEN}│ ✓ Système optimal - aucune recommandation${NC}"
    else
        for rec in "${recommendations[@]}"; do
            echo -e "│ $rec"
        done
    fi
    
    echo -e "${BLUE}└──────────────────────────────────────────────────────────────────────────────┘${NC}"
}

# Affichage au format JSON
display_json() {
    local metrics_data="$1"
    IFS='|' read -r hpa_data deployment_data pods_data nodes_data <<< "$metrics_data"
    
    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    cat << EOF
{
  "timestamp": "$timestamp",
  "namespace": "$NAMESPACE",
  "hpa": $hpa_data,
  "deployments": $deployment_data,
  "pods_metrics": $(echo "$pods_data" | jq -R . | jq -s .),
  "nodes_metrics": $(echo "$nodes_data" | jq -R . | jq -s .)
}
EOF
}

# Affichage au format Prometheus
display_prometheus() {
    local metrics_data="$1"
    IFS='|' read -r hpa_data deployment_data pods_data nodes_data <<< "$metrics_data"
    
    echo "# HELP autoscaling_hpa_count Number of HPA resources"
    echo "# TYPE autoscaling_hpa_count gauge"
    local hpa_count
    hpa_count=$(echo "$hpa_data" | jq '.items | length')
    echo "autoscaling_hpa_count{namespace=\"$NAMESPACE\"} $hpa_count"
    
    echo "# HELP autoscaling_deployment_replicas Number of deployment replicas"
    echo "# TYPE autoscaling_deployment_replicas gauge"
    echo "$deployment_data" | jq -r '.items[] | "autoscaling_deployment_replicas{namespace=\"'$NAMESPACE'\",deployment=\"\(.metadata.name)\"} \(.spec.replicas // 0)"'
    
    echo "# HELP autoscaling_deployment_ready_replicas Number of ready deployment replicas"
    echo "# TYPE autoscaling_deployment_ready_replicas gauge"
    echo "$deployment_data" | jq -r '.items[] | "autoscaling_deployment_ready_replicas{namespace=\"'$NAMESPACE'\",deployment=\"\(.metadata.name)\"} \(.status.readyReplicas // 0)"'
}

# Mode interactif
interactive_mode() {
    log_info "Démarrage du mode interactif (Ctrl+C pour quitter)"
    
    while true; do
        local metrics_data
        metrics_data=$(collect_scaling_metrics)
        
        case "$OUTPUT_FORMAT" in
            "table")
                display_dashboard "$metrics_data"
                ;;
            "json")
                display_json "$metrics_data"
                ;;
            "prometheus")
                display_prometheus "$metrics_data"
                ;;
        esac
        
        # Lecture non-bloquante
        if read -t "$REFRESH_INTERVAL" -n 1 key 2>/dev/null; then
            case "$key" in
                q|Q)
                    log_info "Arrêt du monitoring"
                    exit 0
                    ;;
                r|R)
                    continue
                    ;;
                h|H)
                    show_help_interactive
                    ;;
            esac
        fi
    done
}

# Mode single shot
single_shot_mode() {
    local metrics_data
    metrics_data=$(collect_scaling_metrics)
    
    case "$OUTPUT_FORMAT" in
        "table")
            display_dashboard "$metrics_data"
            ;;
        "json")
            display_json "$metrics_data"
            ;;
        "prometheus")
            display_prometheus "$metrics_data"
            ;;
    esac
}

# Aide interactive
show_help_interactive() {
    clear_screen
    cat << EOF
${BOLD}${CYAN}AIDE - MONITORING AUTOSCALING${NC}

${BOLD}Raccourcis clavier:${NC}
  q, Q    - Quitter le monitoring
  r, R    - Rafraîchir immédiatement
  h, H    - Afficher cette aide
  
${BOLD}Indicateurs de statut:${NC}
  ${GREEN}●${NC}      - Statut optimal
  ${YELLOW}●${NC}      - Attention requise
  ${RED}●${NC}      - Situation critique

${BOLD}Métriques surveillées:${NC}
  - HPA (Horizontal Pod Autoscalers)
  - Statut des déploiements
  - Utilisation des ressources pods
  - Utilisation des nodes
  - Recommandations automatiques

${BOLD}Configuration:${NC}
  Namespace: $NAMESPACE
  Rafraîchissement: ${REFRESH_INTERVAL}s

Appuyez sur une touche pour revenir au tableau de bord...
EOF
    read -n 1
}

# Aide générale
show_help() {
    cat << EOF
Script de monitoring de l'autoscaling Spotify AI Agent

Usage: $0 [OPTIONS]

Options:
  -n, --namespace NAMESPACE    Namespace à surveiller (défaut: autoscaling)
  -i, --interval SECONDS       Intervalle de rafraîchissement (défaut: 10)
  -f, --format FORMAT          Format de sortie: table, json, prometheus (défaut: table)
  -o, --once                   Mode single shot (pas de rafraîchissement)
  -h, --help                   Affiche cette aide

Exemples:
  $0                           # Mode interactif table
  $0 -n production -i 5        # Production avec rafraîchissement 5s
  $0 -f json -o               # Single shot JSON
  $0 -f prometheus            # Format Prometheus pour scraping

Mode interactif (format table):
  q - Quitter
  r - Rafraîchir
  h - Aide

EOF
}

# Variables par défaut
INTERACTIVE=true

# Parsing des arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -i|--interval)
            REFRESH_INTERVAL="$2"
            shift 2
            ;;
        -f|--format)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        -o|--once)
            INTERACTIVE=false
            shift
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

# Validation du format
if [[ ! "$OUTPUT_FORMAT" =~ ^(table|json|prometheus)$ ]]; then
    log_error "Format invalide: $OUTPUT_FORMAT (autorisés: table, json, prometheus)"
    exit 1
fi

# Vérification des prérequis
if ! command -v kubectl &> /dev/null; then
    log_error "kubectl n'est pas installé"
    exit 1
fi

if ! kubectl cluster-info &> /dev/null; then
    log_error "Impossible de se connecter au cluster Kubernetes"
    exit 1
fi

# Vérification du namespace
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    log_error "Le namespace '$NAMESPACE' n'existe pas"
    exit 1
fi

# Exécution
if [[ "$INTERACTIVE" == "true" ]] && [[ "$OUTPUT_FORMAT" == "table" ]]; then
    interactive_mode
else
    single_shot_mode
fi
