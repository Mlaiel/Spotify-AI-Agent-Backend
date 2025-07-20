#!/bin/bash

# ================================================================
# Script d'Auto-Remédiation Ultra-Avancé - Spotify AI Agent
# ================================================================
# Auteur: Fahed Mlaiel
# Équipe: Lead Dev + Architecte IA, Développeur Backend Senior,
#         Ingénieur Machine Learning, Spécialiste Sécurité Backend,
#         Architecte Microservices
# Version: 1.0.0
# Description: Script d'auto-remédiation intelligent pour incidents critiques
# ================================================================

set -euo pipefail

# Configuration globale
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/auto-remediation.log"
KUBECTL_CONTEXT="${KUBECTL_CONTEXT:-default}"
NAMESPACE="${NAMESPACE:-default}"
DRY_RUN="${DRY_RUN:-false}"

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ================================================================
# FONCTIONS UTILITAIRES
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
    esac
}

check_dependencies() {
    log "INFO" "Vérification des dépendances..."
    
    local deps=("kubectl" "curl" "jq" "docker" "prometheus-cli")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log "ERROR" "Dépendances manquantes: ${missing_deps[*]}"
        exit 1
    fi
    
    log "INFO" "Toutes les dépendances sont présentes"
}

check_cluster_health() {
    log "INFO" "Vérification de la santé du cluster Kubernetes..."
    
    if ! kubectl cluster-info &> /dev/null; then
        log "ERROR" "Impossible de se connecter au cluster Kubernetes"
        return 1
    fi
    
    # Vérification des nœuds
    local ready_nodes=$(kubectl get nodes --no-headers | grep -c "Ready")
    local total_nodes=$(kubectl get nodes --no-headers | wc -l)
    
    if [ "$ready_nodes" -lt "$total_nodes" ]; then
        log "WARN" "Nœuds non prêts détectés: $ready_nodes/$total_nodes"
    else
        log "INFO" "Tous les nœuds sont prêts: $ready_nodes/$total_nodes"
    fi
    
    return 0
}

# ================================================================
# FONCTIONS DE REMÉDIATION API
# ================================================================

restart_api_pods() {
    local alert_data="$1"
    log "INFO" "Démarrage de la remédiation: Redémarrage des pods API"
    
    # Vérification de la santé du cluster
    if ! check_cluster_health; then
        log "ERROR" "Cluster non sain - abandon de la remédiation"
        return 1
    fi
    
    # Récupération des pods API
    local api_pods=$(kubectl get pods -n "$NAMESPACE" -l app=spotify-ai-agent-api --no-headers -o custom-columns=":metadata.name")
    
    if [ -z "$api_pods" ]; then
        log "ERROR" "Aucun pod API trouvé"
        return 1
    fi
    
    log "INFO" "Pods API trouvés: $(echo "$api_pods" | tr '\n' ' ')"
    
    # Redémarrage progressif des pods
    for pod in $api_pods; do
        log "INFO" "Redémarrage du pod: $pod"
        
        if [ "$DRY_RUN" = "true" ]; then
            log "INFO" "[DRY-RUN] kubectl delete pod $pod -n $NAMESPACE"
        else
            if kubectl delete pod "$pod" -n "$NAMESPACE" --grace-period=30; then
                log "INFO" "Pod $pod supprimé avec succès"
                
                # Attente que le nouveau pod soit prêt
                log "INFO" "Attente du redémarrage du pod..."
                sleep 10
                
                # Vérification de la santé après redémarrage
                if check_pod_health "$pod"; then
                    log "INFO" "Pod $pod redémarré avec succès"
                else
                    log "WARN" "Pod $pod redémarré mais pas encore sain"
                fi
            else
                log "ERROR" "Échec de suppression du pod $pod"
                return 1
            fi
        fi
        
        # Délai entre les redémarrages pour éviter l'indisponibilité
        sleep 15
    done
    
    # Validation finale
    if validate_api_health; then
        log "INFO" "Remédiation réussie: API redémarrée et fonctionnelle"
        send_notification "success" "API pods restarted successfully"
        return 0
    else
        log "ERROR" "Remédiation échouée: API non fonctionnelle après redémarrage"
        send_notification "failure" "API restart failed - manual intervention required"
        return 1
    fi
}

scale_up_api_instances() {
    local alert_data="$1"
    log "INFO" "Démarrage de la remédiation: Scaling up des instances API"
    
    # Récupération du deployment API
    local deployment="spotify-ai-agent-api"
    local current_replicas=$(kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
    
    if [ -z "$current_replicas" ]; then
        log "ERROR" "Impossible de récupérer le nombre de replicas actuels"
        return 1
    fi
    
    log "INFO" "Replicas actuels: $current_replicas"
    
    # Calcul du nouveau nombre de replicas (doubler avec un maximum)
    local new_replicas=$((current_replicas * 2))
    local max_replicas=20
    
    if [ "$new_replicas" -gt "$max_replicas" ]; then
        new_replicas=$max_replicas
        log "WARN" "Limitation du scaling à $max_replicas replicas"
    fi
    
    log "INFO" "Scaling vers $new_replicas replicas"
    
    if [ "$DRY_RUN" = "true" ]; then
        log "INFO" "[DRY-RUN] kubectl scale deployment $deployment --replicas=$new_replicas -n $NAMESPACE"
    else
        if kubectl scale deployment "$deployment" --replicas="$new_replicas" -n "$NAMESPACE"; then
            log "INFO" "Scaling initié vers $new_replicas replicas"
            
            # Attente que le scaling soit effectif
            log "INFO" "Attente de la disponibilité des nouveaux pods..."
            kubectl rollout status deployment/"$deployment" -n "$NAMESPACE" --timeout=300s
            
            # Validation du scaling
            local actual_replicas=$(kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
            
            if [ "$actual_replicas" -eq "$new_replicas" ]; then
                log "INFO" "Scaling réussi: $actual_replicas replicas prêts"
                
                # Validation de la distribution de charge
                if validate_load_distribution; then
                    log "INFO" "Remédiation réussie: Scaling effectué avec succès"
                    send_notification "success" "API scaled up to $new_replicas replicas"
                    
                    # Programmation d'un scale-down automatique après 30 minutes
                    schedule_auto_scale_down "$deployment" "$current_replicas" 1800
                    
                    return 0
                else
                    log "WARN" "Scaling effectué mais distribution de charge non optimale"
                    return 1
                fi
            else
                log "ERROR" "Scaling échoué: seulement $actual_replicas/$new_replicas replicas prêts"
                return 1
            fi
        else
            log "ERROR" "Échec de la commande de scaling"
            return 1
        fi
    fi
}

# ================================================================
# FONCTIONS DE REMÉDIATION BASE DE DONNÉES
# ================================================================

optimize_database_connections() {
    local alert_data="$1"
    log "INFO" "Démarrage de la remédiation: Optimisation des connexions DB"
    
    # Récupération des informations de la base de données
    local db_pod=$(kubectl get pods -n "$NAMESPACE" -l app=postgresql --no-headers -o custom-columns=":metadata.name" | head -1)
    
    if [ -z "$db_pod" ]; then
        log "ERROR" "Pod PostgreSQL non trouvé"
        return 1
    fi
    
    log "INFO" "Pod PostgreSQL trouvé: $db_pod"
    
    # Vérification des connexions actives
    local active_connections
    if [ "$DRY_RUN" = "true" ]; then
        active_connections=50  # Simulation
        log "INFO" "[DRY-RUN] Connexions actives simulées: $active_connections"
    else
        active_connections=$(kubectl exec -n "$NAMESPACE" "$db_pod" -- psql -U postgres -d spotify_ai_agent -t -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';" | tr -d ' ')
    fi
    
    log "INFO" "Connexions actives: $active_connections"
    
    # Terminaison des connexions inactives
    local idle_threshold=300  # 5 minutes
    
    if [ "$DRY_RUN" = "true" ]; then
        log "INFO" "[DRY-RUN] Terminaison des connexions inactives > ${idle_threshold}s"
    else
        local terminated_connections=$(kubectl exec -n "$NAMESPACE" "$db_pod" -- psql -U postgres -d spotify_ai_agent -t -c "
            SELECT pg_terminate_backend(pid) 
            FROM pg_stat_activity 
            WHERE state = 'idle' 
            AND state_change < NOW() - INTERVAL '${idle_threshold} seconds'
            AND pid <> pg_backend_pid();
        " | grep -c 't' || echo "0")
        
        log "INFO" "Connexions inactives terminées: $terminated_connections"
    fi
    
    # Optimisation des paramètres de connexion
    if optimize_postgres_config "$db_pod"; then
        log "INFO" "Configuration PostgreSQL optimisée"
    else
        log "WARN" "Échec de l'optimisation de la configuration"
    fi
    
    # Validation de l'optimisation
    if validate_database_health; then
        log "INFO" "Remédiation réussie: Connexions DB optimisées"
        send_notification "success" "Database connections optimized"
        return 0
    else
        log "ERROR" "Remédiation échouée: Problèmes de connexion persistants"
        send_notification "failure" "Database optimization failed"
        return 1
    fi
}

optimize_postgres_config() {
    local db_pod="$1"
    log "INFO" "Optimisation de la configuration PostgreSQL"
    
    # Augmentation du pool de connexions
    local new_max_connections=200
    
    if [ "$DRY_RUN" = "true" ]; then
        log "INFO" "[DRY-RUN] Augmentation max_connections à $new_max_connections"
        return 0
    else
        # Dans un environnement réel, modifier postgresql.conf et redémarrer
        log "INFO" "Configuration des paramètres de connexion optimaux"
        
        # Exemple de requêtes d'optimisation
        kubectl exec -n "$NAMESPACE" "$db_pod" -- psql -U postgres -d spotify_ai_agent -c "
            ALTER SYSTEM SET max_connections = '$new_max_connections';
            ALTER SYSTEM SET shared_buffers = '256MB';
            ALTER SYSTEM SET effective_cache_size = '1GB';
            SELECT pg_reload_conf();
        " || return 1
        
        return 0
    fi
}

# ================================================================
# FONCTIONS DE REMÉDIATION SÉCURITÉ
# ================================================================

block_suspicious_ips() {
    local alert_data="$1"
    log "INFO" "Démarrage de la remédiation: Blocage des IPs suspectes"
    
    # Extraction des IPs suspectes depuis les logs ou l'alerte
    local suspicious_ips
    if [ "$DRY_RUN" = "true" ]; then
        suspicious_ips=("192.168.1.100" "10.0.0.50")  # IPs de test
    else
        # Dans un vrai système, extraire depuis les logs Prometheus/AlertManager
        suspicious_ips=($(echo "$alert_data" | jq -r '.suspicious_ips[]' 2>/dev/null || echo ""))
    fi
    
    if [ ${#suspicious_ips[@]} -eq 0 ]; then
        log "WARN" "Aucune IP suspecte identifiée"
        return 1
    fi
    
    log "INFO" "IPs suspectes identifiées: ${suspicious_ips[*]}"
    
    # Application des règles de firewall
    for ip in "${suspicious_ips[@]}"; do
        if validate_ip_address "$ip"; then
            if [ "$DRY_RUN" = "true" ]; then
                log "INFO" "[DRY-RUN] Blocage de l'IP: $ip"
            else
                if block_ip_address "$ip"; then
                    log "INFO" "IP bloquée avec succès: $ip"
                else
                    log "ERROR" "Échec du blocage de l'IP: $ip"
                fi
            fi
        else
            log "WARN" "IP invalide ignorée: $ip"
        fi
    done
    
    # Validation du blocage
    if validate_ip_blocking "${suspicious_ips[@]}"; then
        log "INFO" "Remédiation réussie: IPs suspectes bloquées"
        send_notification "success" "Suspicious IPs blocked: ${suspicious_ips[*]}"
        
        # Programmation du déblocage automatique après 1 heure
        schedule_ip_unblock "${suspicious_ips[@]}" 3600
        
        return 0
    else
        log "ERROR" "Remédiation échouée: Blocage IP non effectif"
        send_notification "failure" "IP blocking failed"
        return 1
    fi
}

validate_ip_address() {
    local ip="$1"
    # Validation format IPv4
    if [[ $ip =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
        return 0
    else
        return 1
    fi
}

block_ip_address() {
    local ip="$1"
    log "INFO" "Blocage de l'IP: $ip"
    
    # Utilisation d'iptables ou d'un service de firewall cloud
    # Exemple avec iptables (nécessite des privilèges root)
    if command -v iptables &> /dev/null; then
        iptables -A INPUT -s "$ip" -j DROP
        return $?
    else
        # Alternative: utilisation d'un service cloud ou d'un ingress controller
        log "WARN" "iptables non disponible - utilisation alternative"
        
        # Exemple avec Kubernetes Network Policies
        apply_network_policy_block "$ip"
        return $?
    fi
}

apply_network_policy_block() {
    local ip="$1"
    local policy_name="block-ip-$(echo "$ip" | tr '.' '-')"
    
    cat << EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: $policy_name
  namespace: $NAMESPACE
spec:
  podSelector:
    matchLabels:
      app: spotify-ai-agent-api
  policyTypes:
  - Ingress
  ingress:
  - from:
    - ipBlock:
        cidr: 0.0.0.0/0
        except:
        - $ip/32
EOF
}

# ================================================================
# FONCTIONS DE VALIDATION
# ================================================================

validate_api_health() {
    log "INFO" "Validation de la santé de l'API"
    
    # Récupération de l'endpoint de health check
    local api_endpoint="${API_ENDPOINT:-http://localhost:8000/health}"
    
    # Test de santé avec retry
    local max_attempts=5
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        log "INFO" "Tentative de health check $attempt/$max_attempts"
        
        if [ "$DRY_RUN" = "true" ]; then
            log "INFO" "[DRY-RUN] Health check simulé - succès"
            return 0
        fi
        
        if curl -f -s --max-time 10 "$api_endpoint" > /dev/null; then
            log "INFO" "API health check réussi"
            return 0
        else
            log "WARN" "Health check échoué (tentative $attempt)"
            ((attempt++))
            sleep 10
        fi
    done
    
    log "ERROR" "API health check échoué après $max_attempts tentatives"
    return 1
}

check_pod_health() {
    local pod_name="$1"
    log "INFO" "Vérification de la santé du pod: $pod_name"
    
    # Attente que le pod soit en état Running
    local timeout=300  # 5 minutes
    local start_time=$(date +%s)
    
    while true; do
        local pod_status=$(kubectl get pod "$pod_name" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound")
        
        if [ "$pod_status" = "Running" ]; then
            # Vérification que le pod est ready
            local ready_status=$(kubectl get pod "$pod_name" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "False")
            
            if [ "$ready_status" = "True" ]; then
                log "INFO" "Pod $pod_name est sain et prêt"
                return 0
            fi
        fi
        
        local current_time=$(date +%s)
        if [ $((current_time - start_time)) -gt $timeout ]; then
            log "ERROR" "Timeout: Pod $pod_name non prêt après ${timeout}s"
            return 1
        fi
        
        sleep 5
    done
}

validate_load_distribution() {
    log "INFO" "Validation de la distribution de charge"
    
    # Vérification que la charge est bien distribuée entre les pods
    local api_pods=$(kubectl get pods -n "$NAMESPACE" -l app=spotify-ai-agent-api --no-headers -o custom-columns=":metadata.name")
    local total_pods=$(echo "$api_pods" | wc -l)
    
    if [ "$total_pods" -lt 2 ]; then
        log "WARN" "Moins de 2 pods API - distribution limitée"
        return 1
    fi
    
    # Dans un vrai système, vérifier les métriques de charge depuis Prometheus
    if [ "$DRY_RUN" = "true" ]; then
        log "INFO" "[DRY-RUN] Distribution de charge validée"
        return 0
    fi
    
    # Vérification basique de la répartition CPU
    local unbalanced_pods=0
    for pod in $api_pods; do
        local cpu_usage=$(kubectl top pod "$pod" -n "$NAMESPACE" --no-headers | awk '{print $2}' | sed 's/m//' 2>/dev/null || echo "0")
        
        if [ "$cpu_usage" -gt 800 ]; then  # Plus de 80% d'un core
            ((unbalanced_pods++))
        fi
    done
    
    if [ "$unbalanced_pods" -gt $((total_pods / 2)) ]; then
        log "WARN" "Distribution de charge déséquilibrée"
        return 1
    fi
    
    log "INFO" "Distribution de charge acceptable"
    return 0
}

validate_database_health() {
    log "INFO" "Validation de la santé de la base de données"
    
    local db_pod=$(kubectl get pods -n "$NAMESPACE" -l app=postgresql --no-headers -o custom-columns=":metadata.name" | head -1)
    
    if [ -z "$db_pod" ]; then
        log "ERROR" "Pod PostgreSQL non trouvé"
        return 1
    fi
    
    if [ "$DRY_RUN" = "true" ]; then
        log "INFO" "[DRY-RUN] Santé DB validée"
        return 0
    fi
    
    # Test de connectivité
    if kubectl exec -n "$NAMESPACE" "$db_pod" -- pg_isready -U postgres > /dev/null 2>&1; then
        log "INFO" "Base de données accessible"
        
        # Test de requête simple
        if kubectl exec -n "$NAMESPACE" "$db_pod" -- psql -U postgres -d spotify_ai_agent -c "SELECT 1;" > /dev/null 2>&1; then
            log "INFO" "Base de données fonctionnelle"
            return 0
        else
            log "ERROR" "Requête de test échouée"
            return 1
        fi
    else
        log "ERROR" "Base de données non accessible"
        return 1
    fi
}

validate_ip_blocking() {
    local ips=("$@")
    log "INFO" "Validation du blocage des IPs: ${ips[*]}"
    
    if [ "$DRY_RUN" = "true" ]; then
        log "INFO" "[DRY-RUN] Blocage IP validé"
        return 0
    fi
    
    # Vérification des règles de firewall actives
    for ip in "${ips[@]}"; do
        if iptables -L INPUT -n | grep -q "$ip"; then
            log "INFO" "Règle de blocage active pour $ip"
        else
            log "WARN" "Aucune règle de blocage trouvée pour $ip"
            return 1
        fi
    done
    
    return 0
}

# ================================================================
# FONCTIONS DE NOTIFICATION
# ================================================================

send_notification() {
    local status="$1"
    local message="$2"
    
    log "INFO" "Envoi de notification: $status - $message"
    
    # Notification Slack
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        send_slack_notification "$status" "$message"
    fi
    
    # Notification email
    if [ -n "${EMAIL_RECIPIENTS:-}" ]; then
        send_email_notification "$status" "$message"
    fi
    
    # Notification PagerDuty pour les échecs critiques
    if [ "$status" = "failure" ] && [ -n "${PAGERDUTY_KEY:-}" ]; then
        send_pagerduty_notification "$message"
    fi
}

send_slack_notification() {
    local status="$1"
    local message="$2"
    local color="good"
    
    if [ "$status" = "failure" ]; then
        color="danger"
    elif [ "$status" = "warning" ]; then
        color="warning"
    fi
    
    local payload=$(cat << EOF
{
    "channel": "#auto-remediation",
    "username": "Auto-Remediation Bot",
    "icon_emoji": ":robot_face:",
    "attachments": [{
        "color": "$color",
        "title": "Auto-Remediation Report",
        "text": "$message",
        "fields": [
            {
                "title": "Status",
                "value": "$status",
                "short": true
            },
            {
                "title": "Timestamp", 
                "value": "$(date -u +"%Y-%m-%d %H:%M:%S UTC")",
                "short": true
            },
            {
                "title": "Environment",
                "value": "${ENVIRONMENT:-dev}",
                "short": true
            }
        ]
    }]
}
EOF
    )
    
    if [ "$DRY_RUN" = "true" ]; then
        log "INFO" "[DRY-RUN] Notification Slack: $message"
    else
        curl -X POST -H 'Content-type: application/json' \
             --data "$payload" \
             "$SLACK_WEBHOOK_URL" > /dev/null 2>&1 || log "WARN" "Échec envoi notification Slack"
    fi
}

send_email_notification() {
    local status="$1"
    local message="$2"
    
    if [ "$DRY_RUN" = "true" ]; then
        log "INFO" "[DRY-RUN] Notification email: $message"
        return
    fi
    
    # Utilisation de sendmail ou d'un service SMTP
    if command -v sendmail &> /dev/null; then
        local subject="Auto-Remediation Alert - $status"
        
        {
            echo "To: $EMAIL_RECIPIENTS"
            echo "Subject: $subject"
            echo "Content-Type: text/html"
            echo ""
            echo "<h2>Auto-Remediation Report</h2>"
            echo "<p><strong>Status:</strong> $status</p>"
            echo "<p><strong>Message:</strong> $message</p>"
            echo "<p><strong>Timestamp:</strong> $(date -u)</p>"
            echo "<p><strong>Environment:</strong> ${ENVIRONMENT:-dev}</p>"
        } | sendmail "$EMAIL_RECIPIENTS"
    fi
}

send_pagerduty_notification() {
    local message="$1"
    
    if [ "$DRY_RUN" = "true" ]; then
        log "INFO" "[DRY-RUN] Notification PagerDuty: $message"
        return
    fi
    
    local payload=$(cat << EOF
{
    "routing_key": "$PAGERDUTY_KEY",
    "event_action": "trigger",
    "payload": {
        "summary": "Auto-Remediation Failure: $message",
        "source": "spotify-ai-agent-auto-remediation",
        "severity": "critical",
        "component": "auto-remediation-system",
        "group": "platform"
    }
}
EOF
    )
    
    curl -X POST \
         -H "Content-Type: application/json" \
         -d "$payload" \
         "https://events.pagerduty.com/v2/enqueue" > /dev/null 2>&1 || log "WARN" "Échec notification PagerDuty"
}

# ================================================================
# FONCTIONS DE SCHEDULING
# ================================================================

schedule_auto_scale_down() {
    local deployment="$1"
    local original_replicas="$2"
    local delay_seconds="$3"
    
    log "INFO" "Programmation du scale-down automatique dans ${delay_seconds}s"
    
    if [ "$DRY_RUN" = "true" ]; then
        log "INFO" "[DRY-RUN] Scale-down programmé: $deployment vers $original_replicas replicas"
        return
    fi
    
    # Création d'un job en arrière-plan pour le scale-down
    {
        sleep "$delay_seconds"
        log "INFO" "Exécution du scale-down automatique: $deployment"
        
        if kubectl scale deployment "$deployment" --replicas="$original_replicas" -n "$NAMESPACE"; then
            log "INFO" "Scale-down automatique réussi: $deployment vers $original_replicas replicas"
            send_notification "success" "Auto scale-down completed: $deployment"
        else
            log "ERROR" "Échec du scale-down automatique: $deployment"
            send_notification "failure" "Auto scale-down failed: $deployment"
        fi
    } &
}

schedule_ip_unblock() {
    local ips=("$@")
    local delay_seconds="${!#}"  # Dernier argument
    local ip_list=("${@:1:$#-1}")  # Tous sauf le dernier
    
    log "INFO" "Programmation du déblocage IP dans ${delay_seconds}s: ${ip_list[*]}"
    
    if [ "$DRY_RUN" = "true" ]; then
        log "INFO" "[DRY-RUN] Déblocage IP programmé: ${ip_list[*]}"
        return
    fi
    
    # Job en arrière-plan pour le déblocage
    {
        sleep "$delay_seconds"
        log "INFO" "Exécution du déblocage IP automatique"
        
        for ip in "${ip_list[@]}"; do
            if iptables -D INPUT -s "$ip" -j DROP 2>/dev/null; then
                log "INFO" "IP débloquée: $ip"
            else
                log "WARN" "Échec du déblocage IP: $ip"
            fi
        done
        
        send_notification "success" "Auto IP unblock completed: ${ip_list[*]}"
    } &
}

# ================================================================
# FONCTION PRINCIPALE
# ================================================================

main() {
    log "INFO" "Démarrage du script d'auto-remédiation ultra-avancé"
    log "INFO" "Version: 1.0.0 | Auteur: Fahed Mlaiel"
    
    # Vérification des dépendances
    check_dependencies
    
    # Parsing des arguments
    local action="$1"
    local alert_data="${2:-{}}"
    
    case "$action" in
        "restart_api_pods")
            restart_api_pods "$alert_data"
            ;;
        "scale_up_api_instances")
            scale_up_api_instances "$alert_data"
            ;;
        "optimize_database_connections")
            optimize_database_connections "$alert_data"
            ;;
        "block_suspicious_ips")
            block_suspicious_ips "$alert_data"
            ;;
        "health_check")
            check_cluster_health && validate_api_health && validate_database_health
            ;;
        *)
            echo "Usage: $0 {restart_api_pods|scale_up_api_instances|optimize_database_connections|block_suspicious_ips|health_check} [alert_data]"
            echo ""
            echo "Actions disponibles:"
            echo "  restart_api_pods              - Redémarre les pods API"
            echo "  scale_up_api_instances        - Scale up les instances API"
            echo "  optimize_database_connections - Optimise les connexions DB"
            echo "  block_suspicious_ips          - Bloque les IPs suspectes"
            echo "  health_check                  - Vérifie la santé globale"
            echo ""
            echo "Variables d'environnement:"
            echo "  DRY_RUN=true                  - Mode simulation"
            echo "  NAMESPACE=default             - Namespace Kubernetes"
            echo "  SLACK_WEBHOOK_URL             - URL webhook Slack"
            echo "  EMAIL_RECIPIENTS              - Destinataires email"
            echo "  PAGERDUTY_KEY                 - Clé PagerDuty"
            exit 1
            ;;
    esac
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log "INFO" "Remédiation terminée avec succès"
    else
        log "ERROR" "Remédiation échouée (code: $exit_code)"
    fi
    
    exit $exit_code
}

# Exécution du script si appelé directement
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
