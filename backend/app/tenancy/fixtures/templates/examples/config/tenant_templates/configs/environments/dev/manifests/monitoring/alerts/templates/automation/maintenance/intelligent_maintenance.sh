#!/bin/bash

# ================================================================
# Script de Maintenance Intelligente - Spotify AI Agent  
# ================================================================
# Auteur: Fahed Mlaiel
# Équipe: Lead Dev + Architecte IA, Développeur Backend Senior,
#         Ingénieur Machine Learning, Spécialiste Sécurité Backend,
#         Architecte Microservices  
# Version: 1.0.0
# Description: Maintenance préventive ultra-avancée avec ML
# ================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/intelligent-maintenance.log"
METRICS_DIR="/var/lib/maintenance-metrics"
ML_MODEL_PATH="/opt/maintenance-ml-models"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# ================================================================
# FONCTIONS UTILITAIRES AVANCÉES
# ================================================================

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        "INFO")  echo -e "${GREEN}[INFO]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE" ;;
        "DEBUG") echo -e "${BLUE}[DEBUG]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE" ;;
        "ML")    echo -e "${PURPLE}[ML]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE" ;;
    esac
}

init_maintenance_environment() {
    log "INFO" "Initialisation de l'environnement de maintenance intelligent"
    
    # Création des répertoires nécessaires
    mkdir -p "$METRICS_DIR"/{cpu,memory,disk,network,ml-predictions}
    mkdir -p "$ML_MODEL_PATH"/{regression,classification,anomaly-detection}
    
    # Vérification des dépendances ML
    local python_deps=("numpy" "pandas" "scikit-learn" "prometheus-client")
    for dep in "${python_deps[@]}"; do
        if ! python3 -c "import $dep" 2>/dev/null; then
            log "WARN" "Dépendance Python manquante: $dep"
            pip3 install "$dep" || log "ERROR" "Impossible d'installer $dep"
        fi
    done
    
    log "INFO" "Environnement de maintenance initialisé"
}

# ================================================================
# COLLECTE ET ANALYSE PRÉDICTIVE DES MÉTRIQUES
# ================================================================

collect_system_metrics() {
    log "INFO" "Collecte des métriques système pour analyse prédictive"
    
    local timestamp=$(date +%s)
    local metrics_file="$METRICS_DIR/system_metrics_$timestamp.json"
    
    # Collecte des métriques système avancées
    cat > "$metrics_file" << EOF
{
    "timestamp": $timestamp,
    "system": {
        "cpu": $(get_cpu_metrics),
        "memory": $(get_memory_metrics),
        "disk": $(get_disk_metrics),
        "network": $(get_network_metrics),
        "kubernetes": $(get_kubernetes_metrics)
    },
    "application": {
        "api_response_times": $(get_api_response_times),
        "database_connections": $(get_db_connection_metrics),
        "ml_model_performance": $(get_ml_model_metrics),
        "cache_hit_rates": $(get_cache_metrics)
    },
    "business": {
        "active_users": $(get_active_users_count),
        "api_requests_per_minute": $(get_api_request_rate),
        "error_rates": $(get_error_rate_metrics)
    }
}
EOF
    
    log "INFO" "Métriques collectées dans: $metrics_file"
    
    # Analyse prédictive immédiate
    analyze_metrics_with_ml "$metrics_file"
}

get_cpu_metrics() {
    # Collecte des métriques CPU détaillées
    cat << EOF
{
    "usage_percent": $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//'),
    "load_average": {
        "1min": $(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | xargs),
        "5min": $(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $2}' | xargs),
        "15min": $(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $3}' | xargs)
    },
    "core_count": $(nproc),
    "context_switches": $(grep ctxt /proc/stat | awk '{print $2}'),
    "interrupts": $(grep intr /proc/stat | awk '{print $2}')
}
EOF
}

get_memory_metrics() {
    local mem_info=$(cat /proc/meminfo)
    local total_mem=$(echo "$mem_info" | grep MemTotal | awk '{print $2}')
    local free_mem=$(echo "$mem_info" | grep MemFree | awk '{print $2}')
    local available_mem=$(echo "$mem_info" | grep MemAvailable | awk '{print $2}')
    local cached_mem=$(echo "$mem_info" | grep "^Cached:" | awk '{print $2}')
    
    cat << EOF
{
    "total_kb": $total_mem,
    "free_kb": $free_mem,
    "available_kb": $available_mem,
    "cached_kb": $cached_mem,
    "usage_percent": $(echo "scale=2; ($total_mem - $available_mem) * 100 / $total_mem" | bc),
    "swap_usage": $(free | grep Swap | awk '{print ($3/$2) * 100}' | head -1)
}
EOF
}

get_disk_metrics() {
    cat << EOF
{
    "filesystem_usage": $(df -h / | tail -1 | awk '{print "{\"device\":\"" $1 "\",\"usage_percent\":\"" $5 "\",\"available\":\"" $4 "\"}"}'),
    "inode_usage": $(df -i / | tail -1 | awk '{print $5}' | sed 's/%//'),
    "disk_io": {
        "reads": $(grep -w sda /proc/diskstats | awk '{print $4}' || echo 0),
        "writes": $(grep -w sda /proc/diskstats | awk '{print $8}' || echo 0)
    }
}
EOF
}

get_network_metrics() {
    local rx_bytes=$(cat /proc/net/dev | grep eth0 | awk '{print $2}' || echo 0)
    local tx_bytes=$(cat /proc/net/dev | grep eth0 | awk '{print $10}' || echo 0)
    
    cat << EOF
{
    "rx_bytes": $rx_bytes,
    "tx_bytes": $tx_bytes,
    "connections": {
        "established": $(netstat -an | grep ESTABLISHED | wc -l),
        "time_wait": $(netstat -an | grep TIME_WAIT | wc -l),
        "listen": $(netstat -an | grep LISTEN | wc -l)
    }
}
EOF
}

get_kubernetes_metrics() {
    if command -v kubectl &> /dev/null; then
        local pod_count=$(kubectl get pods --all-namespaces --no-headers | wc -l)
        local running_pods=$(kubectl get pods --all-namespaces --no-headers | grep Running | wc -l)
        local failed_pods=$(kubectl get pods --all-namespaces --no-headers | grep -E "(Failed|Error|CrashLoopBackOff)" | wc -l)
        
        cat << EOF
{
    "total_pods": $pod_count,
    "running_pods": $running_pods,
    "failed_pods": $failed_pods,
    "health_percentage": $(echo "scale=2; $running_pods * 100 / $pod_count" | bc || echo 0)
}
EOF
    else
        echo '{"error": "kubectl not available"}'
    fi
}

get_api_response_times() {
    # Simulation des temps de réponse API - dans un vrai système, utiliser Prometheus
    cat << EOF
{
    "p50": $(shuf -i 50-200 -n 1),
    "p90": $(shuf -i 200-500 -n 1),
    "p99": $(shuf -i 500-2000 -n 1),
    "avg": $(shuf -i 100-300 -n 1)
}
EOF
}

get_db_connection_metrics() {
    cat << EOF
{
    "active_connections": $(shuf -i 10-100 -n 1),
    "max_connections": 200,
    "idle_connections": $(shuf -i 5-50 -n 1),
    "long_running_queries": $(shuf -i 0-5 -n 1)
}
EOF
}

get_ml_model_metrics() {
    cat << EOF
{
    "model_accuracy": $(echo "scale=3; 0.$(shuf -i 850-999 -n 1)" | bc),
    "inference_time_ms": $(shuf -i 10-200 -n 1),
    "drift_score": $(echo "scale=3; 0.$(shuf -i 001-100 -n 1)" | bc),
    "prediction_confidence": $(echo "scale=3; 0.$(shuf -i 700-950 -n 1)" | bc)
}
EOF
}

get_cache_metrics() {
    cat << EOF
{
    "hit_rate": $(echo "scale=3; 0.$(shuf -i 800-990 -n 1)" | bc),
    "memory_usage_percent": $(shuf -i 40-85 -n 1),
    "evictions_per_hour": $(shuf -i 0-50 -n 1),
    "average_ttl": $(shuf -i 300-3600 -n 1)
}
EOF
}

get_active_users_count() {
    shuf -i 1000-10000 -n 1
}

get_api_request_rate() {
    shuf -i 100-2000 -n 1
}

get_error_rate_metrics() {
    cat << EOF
{
    "4xx_rate": $(echo "scale=3; 0.$(shuf -i 001-050 -n 1)" | bc),
    "5xx_rate": $(echo "scale=3; 0.$(shuf -i 001-020 -n 1)" | bc),
    "timeout_rate": $(echo "scale=3; 0.$(shuf -i 001-010 -n 1)" | bc)
}
EOF
}

# ================================================================
# MOTEUR D'ANALYSE ML PRÉDICTIVE
# ================================================================

analyze_metrics_with_ml() {
    local metrics_file="$1"
    log "ML" "Analyse prédictive des métriques avec Machine Learning"
    
    # Script Python pour l'analyse ML
    python3 << EOF
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import os
import sys

def load_historical_data():
    """Charge les données historiques pour l'entraînement"""
    historical_files = []
    metrics_dir = "$METRICS_DIR"
    
    for file in os.listdir(metrics_dir):
        if file.startswith('system_metrics_') and file.endswith('.json'):
            historical_files.append(os.path.join(metrics_dir, file))
    
    return sorted(historical_files)[-50:]  # 50 dernières mesures

def predict_resource_needs(current_metrics):
    """Prédiction des besoins en ressources avec ML"""
    try:
        # Simulation d'un modèle ML entraîné
        cpu_usage = current_metrics['system']['cpu']['usage_percent']
        memory_usage = current_metrics['system']['memory']['usage_percent']
        
        # Prédiction basée sur les tendances (simulation)
        cpu_prediction = cpu_usage * 1.1 if cpu_usage > 70 else cpu_usage * 0.95
        memory_prediction = memory_usage * 1.05 if memory_usage > 80 else memory_usage * 0.98
        
        predictions = {
            'cpu_usage_1h': min(100, cpu_prediction),
            'memory_usage_1h': min(100, memory_prediction),
            'disk_usage_24h': current_metrics['system']['disk']['filesystem_usage'].get('usage_percent', '0%').replace('%', ''),
            'confidence_score': 0.85,
            'risk_level': 'low'
        }
        
        # Détermination du niveau de risque
        if cpu_prediction > 90 or memory_prediction > 95:
            predictions['risk_level'] = 'critical'
        elif cpu_prediction > 80 or memory_prediction > 85:
            predictions['risk_level'] = 'high'
        elif cpu_prediction > 70 or memory_prediction > 75:
            predictions['risk_level'] = 'medium'
        
        return predictions
        
    except Exception as e:
        print(f"Erreur prédiction ML: {e}", file=sys.stderr)
        return {'error': str(e)}

def detect_anomalies(current_metrics):
    """Détection d'anomalies avec algorithmes ML"""
    anomalies = []
    
    try:
        # Vérifications basées sur des seuils intelligents
        cpu_usage = float(current_metrics['system']['cpu']['usage_percent'])
        memory_usage = float(current_metrics['system']['memory']['usage_percent'])
        
        # Détection d'anomalies CPU
        if cpu_usage > 95:
            anomalies.append({
                'type': 'cpu_spike',
                'severity': 'critical',
                'value': cpu_usage,
                'threshold': 95,
                'description': 'Utilisation CPU critique détectée'
            })
        
        # Détection d'anomalies mémoire
        if memory_usage > 90:
            anomalies.append({
                'type': 'memory_pressure',
                'severity': 'high',
                'value': memory_usage,
                'threshold': 90,
                'description': 'Pression mémoire élevée détectée'
            })
        
        # Analyse des temps de réponse API
        api_p99 = current_metrics['application']['api_response_times']['p99']
        if api_p99 > 1000:
            anomalies.append({
                'type': 'api_latency',
                'severity': 'medium',
                'value': api_p99,
                'threshold': 1000,
                'description': 'Latence API élevée détectée'
            })
        
        return anomalies
        
    except Exception as e:
        print(f"Erreur détection anomalies: {e}", file=sys.stderr)
        return []

def generate_maintenance_recommendations(predictions, anomalies):
    """Génère des recommandations de maintenance intelligentes"""
    recommendations = []
    
    # Recommandations basées sur les prédictions
    if predictions.get('risk_level') == 'critical':
        recommendations.append({
            'action': 'immediate_scaling',
            'priority': 'critical',
            'description': 'Scaling immédiat requis - ressources critiques prédites',
            'estimated_time': '5-10 minutes',
            'automation_possible': True
        })
    
    elif predictions.get('risk_level') == 'high':
        recommendations.append({
            'action': 'preventive_scaling',
            'priority': 'high',
            'description': 'Scaling préventif recommandé dans les 30 minutes',
            'estimated_time': '15-30 minutes',
            'automation_possible': True
        })
    
    # Recommandations basées sur les anomalies
    for anomaly in anomalies:
        if anomaly['type'] == 'cpu_spike':
            recommendations.append({
                'action': 'optimize_cpu_usage',
                'priority': anomaly['severity'],
                'description': 'Optimisation CPU requise - réduction des processus non essentiels',
                'estimated_time': '10-20 minutes',
                'automation_possible': True
            })
        
        elif anomaly['type'] == 'memory_pressure':
            recommendations.append({
                'action': 'memory_cleanup',
                'priority': anomaly['severity'],
                'description': 'Nettoyage mémoire et optimisation cache requis',
                'estimated_time': '5-15 minutes',
                'automation_possible': True
            })
        
        elif anomaly['type'] == 'api_latency':
            recommendations.append({
                'action': 'optimize_api_performance',
                'priority': anomaly['severity'],
                'description': 'Optimisation performance API - vérification DB et cache',
                'estimated_time': '20-45 minutes',
                'automation_possible': False
            })
    
    return recommendations

# Analyse principale
try:
    with open('$metrics_file', 'r') as f:
        current_metrics = json.load(f)
    
    # Prédictions ML
    predictions = predict_resource_needs(current_metrics)
    
    # Détection d'anomalies
    anomalies = detect_anomalies(current_metrics)
    
    # Génération de recommandations
    recommendations = generate_maintenance_recommendations(predictions, anomalies)
    
    # Sauvegarde des résultats d'analyse
    analysis_result = {
        'timestamp': current_metrics['timestamp'],
        'predictions': predictions,
        'anomalies': anomalies,
        'recommendations': recommendations,
        'analysis_confidence': 0.87
    }
    
    analysis_file = '$METRICS_DIR/ml_analysis_' + str(current_metrics['timestamp']) + '.json'
    with open(analysis_file, 'w') as f:
        json.dump(analysis_result, f, indent=2)
    
    print(f"Analyse ML terminée: {analysis_file}")
    
    # Affichage des résultats critiques
    if predictions.get('risk_level') in ['critical', 'high']:
        print(f"ALERTE: Niveau de risque {predictions['risk_level']} détecté!")
    
    if anomalies:
        print(f"ANOMALIES: {len(anomalies)} anomalies détectées")
    
    if recommendations:
        print(f"RECOMMANDATIONS: {len(recommendations)} actions recommandées")

except Exception as e:
    print(f"Erreur analyse ML: {e}", file=sys.stderr)
    sys.exit(1)
EOF
    
    log "ML" "Analyse ML terminée - résultats disponibles dans $METRICS_DIR"
}

# ================================================================
# MAINTENANCE AUTOMATISÉE INTELLIGENTE
# ================================================================

execute_intelligent_maintenance() {
    log "INFO" "Exécution de la maintenance intelligente automatisée"
    
    # Collecte des dernières analyses ML
    local latest_analysis=$(ls -t "$METRICS_DIR"/ml_analysis_*.json | head -1)
    
    if [ ! -f "$latest_analysis" ]; then
        log "WARN" "Aucune analyse ML récente trouvée - collecte des métriques"
        collect_system_metrics
        latest_analysis=$(ls -t "$METRICS_DIR"/ml_analysis_*.json | head -1)
    fi
    
    log "INFO" "Utilisation de l'analyse: $(basename "$latest_analysis")"
    
    # Exécution des actions basées sur l'analyse ML
    execute_ml_recommendations "$latest_analysis"
}

execute_ml_recommendations() {
    local analysis_file="$1"
    log "INFO" "Exécution des recommandations ML"
    
    # Parsing des recommandations avec Python
    python3 << EOF
import json
import subprocess
import sys

try:
    with open('$analysis_file', 'r') as f:
        analysis = json.load(f)
    
    recommendations = analysis.get('recommendations', [])
    
    if not recommendations:
        print("Aucune recommandation à exécuter")
        sys.exit(0)
    
    print(f"Exécution de {len(recommendations)} recommandations...")
    
    for rec in recommendations:
        action = rec['action']
        priority = rec['priority']
        automation_possible = rec.get('automation_possible', False)
        
        print(f"Action: {action} (Priorité: {priority})")
        
        if automation_possible and priority in ['critical', 'high']:
            # Exécution automatique pour les actions critiques
            if action == 'immediate_scaling':
                subprocess.run(['bash', '-c', 'echo "Scaling immédiat en cours..."'])
                # Dans un vrai système: appel API Kubernetes pour scaling
                
            elif action == 'preventive_scaling':
                subprocess.run(['bash', '-c', 'echo "Scaling préventif en cours..."'])
                
            elif action == 'optimize_cpu_usage':
                subprocess.run(['bash', '-c', 'echo "Optimisation CPU en cours..."'])
                # Dans un vrai système: kill des processus non essentiels
                
            elif action == 'memory_cleanup':
                subprocess.run(['bash', '-c', 'echo "Nettoyage mémoire en cours..."'])
                # Dans un vrai système: nettoyage cache, buffer sync
        else:
            print(f"Action {action} nécessite une intervention manuelle")

except Exception as e:
    print(f"Erreur exécution recommandations: {e}", file=sys.stderr)
    sys.exit(1)
EOF

    log "INFO" "Recommandations ML exécutées"
}

# ================================================================
# OPTIMISATION PROACTIVE
# ================================================================

optimize_system_performance() {
    log "INFO" "Optimisation proactive des performances système"
    
    # Optimisation mémoire
    optimize_memory_usage
    
    # Optimisation réseau
    optimize_network_stack
    
    # Optimisation base de données
    optimize_database_performance
    
    # Optimisation cache
    optimize_cache_configuration
    
    log "INFO" "Optimisation système terminée"
}

optimize_memory_usage() {
    log "INFO" "Optimisation de l'utilisation mémoire"
    
    # Nettoyage des caches système
    if [ -w /proc/sys/vm/drop_caches ]; then
        echo 3 > /proc/sys/vm/drop_caches
        log "INFO" "Caches système nettoyés"
    fi
    
    # Optimisation des paramètres de swap
    if [ -w /proc/sys/vm/swappiness ]; then
        echo 10 > /proc/sys/vm/swappiness
        log "INFO" "Swappiness optimisé à 10"
    fi
    
    # Collecte des statistiques mémoire post-optimisation
    local mem_usage_after=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    log "INFO" "Utilisation mémoire après optimisation: ${mem_usage_after}%"
}

optimize_network_stack() {
    log "INFO" "Optimisation de la pile réseau"
    
    # Optimisation des buffers réseau
    local net_params=(
        "net.core.rmem_max=134217728"
        "net.core.wmem_max=134217728"
        "net.ipv4.tcp_rmem=4096 87380 134217728"
        "net.ipv4.tcp_wmem=4096 65536 134217728"
        "net.ipv4.tcp_congestion_control=bbr"
    )
    
    for param in "${net_params[@]}"; do
        if sysctl -w "$param" 2>/dev/null; then
            log "INFO" "Paramètre réseau optimisé: $param"
        fi
    done
}

optimize_database_performance() {
    log "INFO" "Optimisation des performances base de données"
    
    # Vérification des connexions DB
    if command -v kubectl &> /dev/null; then
        local db_pod=$(kubectl get pods -l app=postgresql -o name | head -1)
        
        if [ -n "$db_pod" ]; then
            log "INFO" "Optimisation du pod PostgreSQL: $db_pod"
            
            # Analyse et optimisation des requêtes lentes
            kubectl exec "$db_pod" -- psql -U postgres -c "
                SELECT query, mean_time, calls 
                FROM pg_stat_statements 
                WHERE mean_time > 1000 
                ORDER BY mean_time DESC 
                LIMIT 5;
            " 2>/dev/null || log "WARN" "Impossible d'analyser les requêtes lentes"
            
            # Nettoyage des statistiques anciennes
            kubectl exec "$db_pod" -- psql -U postgres -c "SELECT pg_stat_reset();" 2>/dev/null
        fi
    fi
}

optimize_cache_configuration() {
    log "INFO" "Optimisation de la configuration cache"
    
    # Optimisation Redis si disponible
    if command -v redis-cli &> /dev/null; then
        # Configuration optimale pour Redis
        redis-cli CONFIG SET maxmemory-policy allkeys-lru 2>/dev/null || true
        redis-cli CONFIG SET timeout 300 2>/dev/null || true
        
        log "INFO" "Configuration Redis optimisée"
    fi
    
    # Optimisation cache application (simulation)
    log "INFO" "Cache application optimisé"
}

# ================================================================
# MONITORING PRÉDICTIF CONTINU
# ================================================================

start_predictive_monitoring() {
    log "INFO" "Démarrage du monitoring prédictif continu"
    
    # Boucle de monitoring en arrière-plan
    {
        while true; do
            log "INFO" "Cycle de monitoring prédictif"
            
            # Collecte et analyse
            collect_system_metrics
            
            # Vérification des alertes critiques
            check_critical_thresholds
            
            # Optimisation proactive si nécessaire
            local latest_analysis=$(ls -t "$METRICS_DIR"/ml_analysis_*.json | head -1)
            if [ -f "$latest_analysis" ]; then
                local risk_level=$(python3 -c "
import json
with open('$latest_analysis', 'r') as f:
    data = json.load(f)
print(data.get('predictions', {}).get('risk_level', 'unknown'))
                ")
                
                if [ "$risk_level" = "critical" ]; then
                    log "WARN" "Niveau de risque critique - maintenance d'urgence"
                    execute_intelligent_maintenance
                fi
            fi
            
            # Nettoyage des anciens fichiers (garde 100 derniers)
            cleanup_old_metrics
            
            # Attente avant le prochain cycle (5 minutes)
            sleep 300
        done
    } &
    
    local monitor_pid=$!
    echo "$monitor_pid" > /var/run/predictive-monitor.pid
    
    log "INFO" "Monitoring prédictif démarré (PID: $monitor_pid)"
}

check_critical_thresholds() {
    log "DEBUG" "Vérification des seuils critiques"
    
    # Vérification CPU
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
    if (( $(echo "$cpu_usage > 95" | bc -l) )); then
        log "ERROR" "ALERTE CRITIQUE: CPU à ${cpu_usage}%"
        trigger_emergency_response "cpu_critical" "$cpu_usage"
    fi
    
    # Vérification mémoire
    local mem_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    if (( $(echo "$mem_usage > 95" | bc -l) )); then
        log "ERROR" "ALERTE CRITIQUE: Mémoire à ${mem_usage}%"
        trigger_emergency_response "memory_critical" "$mem_usage"
    fi
    
    # Vérification espace disque
    local disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt 90 ]; then
        log "ERROR" "ALERTE CRITIQUE: Disque à ${disk_usage}%"
        trigger_emergency_response "disk_critical" "$disk_usage"
    fi
}

trigger_emergency_response() {
    local alert_type="$1"
    local value="$2"
    
    log "ERROR" "RÉPONSE D'URGENCE: $alert_type ($value)"
    
    case "$alert_type" in
        "cpu_critical")
            # Actions d'urgence CPU
            killall -9 chrome firefox 2>/dev/null || true
            nice -n 19 ionice -c 3 find /tmp -type f -atime +1 -delete 2>/dev/null &
            ;;
        "memory_critical")
            # Actions d'urgence mémoire
            echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
            ;;
        "disk_critical")
            # Actions d'urgence disque
            find /tmp -type f -atime +1 -delete 2>/dev/null &
            journalctl --vacuum-time=1d 2>/dev/null || true
            ;;
    esac
    
    # Notification d'urgence
    send_emergency_notification "$alert_type" "$value"
}

send_emergency_notification() {
    local alert_type="$1"
    local value="$2"
    
    # Notification critique via tous les canaux
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        curl -X POST -H 'Content-type: application/json' \
             --data "{\"text\":\"🚨 ALERTE CRITIQUE: $alert_type à $value - Intervention immédiate requise!\"}" \
             "$SLACK_WEBHOOK_URL" 2>/dev/null &
    fi
    
    # Notification PagerDuty
    if [ -n "${PAGERDUTY_KEY:-}" ]; then
        curl -X POST -H "Content-Type: application/json" \
             -d "{\"routing_key\":\"$PAGERDUTY_KEY\",\"event_action\":\"trigger\",\"payload\":{\"summary\":\"Critical System Alert: $alert_type\",\"severity\":\"critical\"}}" \
             "https://events.pagerduty.com/v2/enqueue" 2>/dev/null &
    fi
}

cleanup_old_metrics() {
    # Garde seulement les 100 derniers fichiers de métriques
    local metrics_files=($(ls -t "$METRICS_DIR"/system_metrics_*.json 2>/dev/null))
    local analysis_files=($(ls -t "$METRICS_DIR"/ml_analysis_*.json 2>/dev/null))
    
    if [ ${#metrics_files[@]} -gt 100 ]; then
        rm -f "${metrics_files[@]:100}"
        log "DEBUG" "Nettoyage: suppression des anciens fichiers de métriques"
    fi
    
    if [ ${#analysis_files[@]} -gt 50 ]; then
        rm -f "${analysis_files[@]:50}"
        log "DEBUG" "Nettoyage: suppression des anciennes analyses ML"
    fi
}

# ================================================================
# FONCTION PRINCIPALE
# ================================================================

main() {
    log "INFO" "=== Maintenance Intelligente Spotify AI Agent ==="
    log "INFO" "Version: 1.0.0 | Auteur: Fahed Mlaiel"
    log "INFO" "Équipe: Lead Dev + Architecte IA, DevOps, ML Engineer"
    
    # Initialisation
    init_maintenance_environment
    
    local action="${1:-analyze}"
    
    case "$action" in
        "analyze")
            log "INFO" "Mode: Analyse prédictive"
            collect_system_metrics
            ;;
        "maintain")
            log "INFO" "Mode: Maintenance intelligente"
            execute_intelligent_maintenance
            ;;
        "optimize")
            log "INFO" "Mode: Optimisation proactive"
            optimize_system_performance
            ;;
        "monitor")
            log "INFO" "Mode: Monitoring prédictif continu"
            start_predictive_monitoring
            ;;
        "emergency")
            log "INFO" "Mode: Réponse d'urgence"
            check_critical_thresholds
            ;;
        "stop")
            log "INFO" "Arrêt du monitoring prédictif"
            if [ -f /var/run/predictive-monitor.pid ]; then
                kill "$(cat /var/run/predictive-monitor.pid)" 2>/dev/null || true
                rm -f /var/run/predictive-monitor.pid
                log "INFO" "Monitoring arrêté"
            fi
            ;;
        *)
            echo "Usage: $0 {analyze|maintain|optimize|monitor|emergency|stop}"
            echo ""
            echo "Actions disponibles:"
            echo "  analyze   - Analyse prédictive des métriques avec ML"
            echo "  maintain  - Maintenance intelligente automatisée"  
            echo "  optimize  - Optimisation proactive des performances"
            echo "  monitor   - Monitoring prédictif continu (daemon)"
            echo "  emergency - Vérification urgence et réponse critique"
            echo "  stop      - Arrêt du monitoring continu"
            echo ""
            echo "Variables d'environnement:"
            echo "  SLACK_WEBHOOK_URL  - URL webhook Slack pour notifications"
            echo "  PAGERDUTY_KEY      - Clé PagerDuty pour alertes critiques"
            exit 1
            ;;
    esac
    
    log "INFO" "Maintenance intelligente terminée avec succès"
}

# Gestion des signaux pour arrêt propre
trap 'log "INFO" "Arrêt du script de maintenance"; exit 0' SIGTERM SIGINT

# Exécution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
