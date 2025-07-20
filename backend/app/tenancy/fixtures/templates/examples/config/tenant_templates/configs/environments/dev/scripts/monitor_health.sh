#!/bin/bash
# =============================================================================
# Script de Surveillance et de Monitoring pour l'Environnement de Développement
# =============================================================================
# 
# Ce script surveille la santé des services et génère des rapports
# de performance pour l'environnement de développement
#
# Développé par l'équipe d'experts dirigée par Fahed Mlaiel
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)"
ENV_DIR="$(dirname "$SCRIPT_DIR")"

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration par défaut
MONITORING_INTERVAL="${MONITORING_INTERVAL:-30}"
ALERT_THRESHOLD_CPU="${ALERT_THRESHOLD_CPU:-80}"
ALERT_THRESHOLD_MEMORY="${ALERT_THRESHOLD_MEMORY:-85}"
ALERT_THRESHOLD_DISK="${ALERT_THRESHOLD_DISK:-90}"
ALERT_THRESHOLD_RESPONSE_TIME="${ALERT_THRESHOLD_RESPONSE_TIME:-5000}"
LOG_RETENTION_DAYS="${LOG_RETENTION_DAYS:-7}"

# Fonctions utilitaires
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_metric() {
    echo -e "${PURPLE}[METRIC]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Initialisation des répertoires de logs
init_monitoring() {
    local logs_dir="$PROJECT_ROOT/logs/monitoring"
    local reports_dir="$PROJECT_ROOT/logs/reports"
    
    mkdir -p "$logs_dir" "$reports_dir"
    
    # Fichiers de logs
    export HEALTH_LOG="$logs_dir/health.log"
    export PERFORMANCE_LOG="$logs_dir/performance.log"
    export ALERTS_LOG="$logs_dir/alerts.log"
    export METRICS_LOG="$logs_dir/metrics.log"
    
    # Fichier de rapport quotidien
    export DAILY_REPORT="$reports_dir/daily_$(date +%Y-%m-%d).json"
    
    log_info "Monitoring initialisé - Logs dans $logs_dir"
}

# Chargement des variables d'environnement
load_environment() {
    local env_file="$ENV_DIR/secrets/.env"
    
    if [ -f "$env_file" ]; then
        set -a
        source "$env_file"
        set +a
    fi
    
    # Variables par défaut
    export DEV_API_HOST="${DEV_FASTAPI_HOST:-localhost}"
    export DEV_API_PORT="${DEV_FASTAPI_PORT:-8000}"
    export DEV_DB_HOST="${DEV_DB_HOST:-localhost}"
    export DEV_DB_PORT="${DEV_DB_PORT:-5432}"
    export DEV_REDIS_HOST="${DEV_REDIS_HOST:-localhost}"
    export DEV_REDIS_PORT="${DEV_REDIS_PORT:-6379}"
}

# Fonction de vérification de la santé de l'API
check_api_health() {
    local api_url="http://$DEV_API_HOST:$DEV_API_PORT"
    local health_endpoint="$api_url/health"
    local docs_endpoint="$api_url/docs"
    
    local result=$(cat <<EOF
{
    "timestamp": "$(date -Iseconds)",
    "service": "api",
    "status": "unknown",
    "response_time": 0,
    "endpoints": {},
    "details": {}
}
EOF
)
    
    # Test de connectivité de base
    if ! nc -z "$DEV_API_HOST" "$DEV_API_PORT" 2>/dev/null; then
        result=$(echo "$result" | jq '.status = "down" | .details.error = "Service not reachable"')
        echo "$result" | tee -a "$HEALTH_LOG"
        return 1
    fi
    
    # Test du endpoint de santé
    local start_time=$(date +%s%3N)
    local health_response=$(curl -s -o /dev/null -w "%{http_code}" "$health_endpoint" 2>/dev/null || echo "000")
    local end_time=$(date +%s%3N)
    local response_time=$((end_time - start_time))
    
    result=$(echo "$result" | jq --arg rt "$response_time" '.response_time = ($rt | tonumber)')
    
    # Analyse de la réponse
    case $health_response in
        200)
            result=$(echo "$result" | jq '.status = "healthy" | .endpoints.health = "ok"')
            log_success "API service: Healthy (${response_time}ms)"
            ;;
        404)
            result=$(echo "$result" | jq '.status = "partial" | .endpoints.health = "not_found"')
            log_warning "API service: Health endpoint not found"
            ;;
        000)
            result=$(echo "$result" | jq '.status = "down" | .details.error = "Connection failed"')
            log_error "API service: Connection failed"
            ;;
        *)
            result=$(echo "$result" | jq --arg code "$health_response" '.status = "unhealthy" | .details.http_code = $code')
            log_error "API service: Unhealthy (HTTP $health_response)"
            ;;
    esac
    
    # Test du endpoint de documentation
    local docs_response=$(curl -s -o /dev/null -w "%{http_code}" "$docs_endpoint" 2>/dev/null || echo "000")
    if [ "$docs_response" = "200" ]; then
        result=$(echo "$result" | jq '.endpoints.docs = "ok"')
    else
        result=$(echo "$result" | jq --arg code "$docs_response" '.endpoints.docs = $code')
    fi
    
    # Alerte sur temps de réponse
    if [ "$response_time" -gt "$ALERT_THRESHOLD_RESPONSE_TIME" ]; then
        local alert_msg="API response time high: ${response_time}ms (threshold: ${ALERT_THRESHOLD_RESPONSE_TIME}ms)"
        log_warning "$alert_msg"
        echo "$(date -Iseconds) - $alert_msg" >> "$ALERTS_LOG"
    fi
    
    echo "$result" | tee -a "$HEALTH_LOG"
}

# Fonction de vérification de la santé de PostgreSQL
check_postgresql_health() {
    local result=$(cat <<EOF
{
    "timestamp": "$(date -Iseconds)",
    "service": "postgresql",
    "status": "unknown",
    "response_time": 0,
    "connections": {},
    "details": {}
}
EOF
)
    
    # Test de connectivité
    if ! nc -z "$DEV_DB_HOST" "$DEV_DB_PORT" 2>/dev/null; then
        result=$(echo "$result" | jq '.status = "down" | .details.error = "Service not reachable"')
        echo "$result" | tee -a "$HEALTH_LOG"
        return 1
    fi
    
    # Test de connexion avec psql si disponible
    if command -v psql &> /dev/null; then
        local start_time=$(date +%s%3N)
        local db_name="${DEV_DB_NAME:-spotify_ai_dev}"
        local db_user="${DEV_DB_USER:-postgres}"
        
        # Test de connexion simple
        if PGPASSWORD="${DEV_DB_PASSWORD:-postgres}" psql -h "$DEV_DB_HOST" -p "$DEV_DB_PORT" -U "$db_user" -d "$db_name" -c "SELECT 1;" >/dev/null 2>&1; then
            local end_time=$(date +%s%3N)
            local response_time=$((end_time - start_time))
            
            result=$(echo "$result" | jq --arg rt "$response_time" '.status = "healthy" | .response_time = ($rt | tonumber)')
            
            # Récupération des statistiques de connexion
            local conn_stats=$(PGPASSWORD="${DEV_DB_PASSWORD:-postgres}" psql -h "$DEV_DB_HOST" -p "$DEV_DB_PORT" -U "$db_user" -d "$db_name" -t -c "SELECT count(*) FROM pg_stat_activity;" 2>/dev/null | xargs)
            
            if [ -n "$conn_stats" ] && [ "$conn_stats" != "" ]; then
                result=$(echo "$result" | jq --arg conn "$conn_stats" '.connections.active = ($conn | tonumber)')
            fi
            
            log_success "PostgreSQL: Healthy (${response_time}ms, $conn_stats connections)"
        else
            result=$(echo "$result" | jq '.status = "unhealthy" | .details.error = "Authentication or query failed"')
            log_error "PostgreSQL: Authentication or query failed"
        fi
    else
        result=$(echo "$result" | jq '.status = "partial" | .details.note = "psql not available, only port check performed"')
        log_warning "PostgreSQL: Port accessible but psql not available for full check"
    fi
    
    echo "$result" | tee -a "$HEALTH_LOG"
}

# Fonction de vérification de la santé de Redis
check_redis_health() {
    local result=$(cat <<EOF
{
    "timestamp": "$(date -Iseconds)",
    "service": "redis",
    "status": "unknown",
    "response_time": 0,
    "memory": {},
    "details": {}
}
EOF
)
    
    # Test de connectivité
    if ! nc -z "$DEV_REDIS_HOST" "$DEV_REDIS_PORT" 2>/dev/null; then
        result=$(echo "$result" | jq '.status = "down" | .details.error = "Service not reachable"')
        echo "$result" | tee -a "$HEALTH_LOG"
        return 1
    fi
    
    # Test avec redis-cli si disponible
    if command -v redis-cli &> /dev/null; then
        local start_time=$(date +%s%3N)
        
        # Test PING
        if redis-cli -h "$DEV_REDIS_HOST" -p "$DEV_REDIS_PORT" ping >/dev/null 2>&1; then
            local end_time=$(date +%s%3N)
            local response_time=$((end_time - start_time))
            
            result=$(echo "$result" | jq --arg rt "$response_time" '.status = "healthy" | .response_time = ($rt | tonumber)')
            
            # Récupération des statistiques mémoire
            local memory_info=$(redis-cli -h "$DEV_REDIS_HOST" -p "$DEV_REDIS_PORT" info memory 2>/dev/null | grep "used_memory_human" | cut -d: -f2 | tr -d '\r')
            local memory_peak=$(redis-cli -h "$DEV_REDIS_HOST" -p "$DEV_REDIS_PORT" info memory 2>/dev/null | grep "used_memory_peak_human" | cut -d: -f2 | tr -d '\r')
            
            if [ -n "$memory_info" ]; then
                result=$(echo "$result" | jq --arg mem "$memory_info" --arg peak "$memory_peak" '.memory.used = $mem | .memory.peak = $peak')
            fi
            
            log_success "Redis: Healthy (${response_time}ms, memory: $memory_info)"
        else
            result=$(echo "$result" | jq '.status = "unhealthy" | .details.error = "PING command failed"')
            log_error "Redis: PING command failed"
        fi
    else
        result=$(echo "$result" | jq '.status = "partial" | .details.note = "redis-cli not available, only port check performed"')
        log_warning "Redis: Port accessible but redis-cli not available for full check"
    fi
    
    echo "$result" | tee -a "$HEALTH_LOG"
}

# Fonction de collecte des métriques système
collect_system_metrics() {
    log_info "Collecte des métriques système..."
    
    local metrics=$(cat <<EOF
{
    "timestamp": "$(date -Iseconds)",
    "system": {
        "cpu": {},
        "memory": {},
        "disk": {},
        "network": {}
    },
    "processes": {}
}
EOF
)
    
    # Métriques CPU
    if command -v top &> /dev/null; then
        local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 || echo "0")
        metrics=$(echo "$metrics" | jq --arg cpu "$cpu_usage" '.system.cpu.usage_percent = ($cpu | tonumber)')
        
        if (( $(echo "$cpu_usage > $ALERT_THRESHOLD_CPU" | bc -l) )); then
            local alert_msg="High CPU usage: ${cpu_usage}% (threshold: ${ALERT_THRESHOLD_CPU}%)"
            log_warning "$alert_msg"
            echo "$(date -Iseconds) - $alert_msg" >> "$ALERTS_LOG"
        fi
    fi
    
    # Métriques mémoire
    if command -v free &> /dev/null; then
        local memory_info=$(free -m | awk 'NR==2{printf "%.1f %.1f %.1f", $3*100/$2, $2, $3}')
        local memory_percent=$(echo "$memory_info" | awk '{print $1}')
        local memory_total=$(echo "$memory_info" | awk '{print $2}')
        local memory_used=$(echo "$memory_info" | awk '{print $3}')
        
        metrics=$(echo "$metrics" | jq --arg pct "$memory_percent" --arg total "$memory_total" --arg used "$memory_used" '.system.memory.usage_percent = ($pct | tonumber) | .system.memory.total_mb = ($total | tonumber) | .system.memory.used_mb = ($used | tonumber)')
        
        if (( $(echo "$memory_percent > $ALERT_THRESHOLD_MEMORY" | bc -l) )); then
            local alert_msg="High memory usage: ${memory_percent}% (threshold: ${ALERT_THRESHOLD_MEMORY}%)"
            log_warning "$alert_msg"
            echo "$(date -Iseconds) - $alert_msg" >> "$ALERTS_LOG"
        fi
    fi
    
    # Métriques disque
    if command -v df &> /dev/null; then
        local disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
        metrics=$(echo "$metrics" | jq --arg disk "$disk_usage" '.system.disk.usage_percent = ($disk | tonumber)')
        
        if [ "$disk_usage" -gt "$ALERT_THRESHOLD_DISK" ]; then
            local alert_msg="High disk usage: ${disk_usage}% (threshold: ${ALERT_THRESHOLD_DISK}%)"
            log_warning "$alert_msg"
            echo "$(date -Iseconds) - $alert_msg" >> "$ALERTS_LOG"
        fi
    fi
    
    # Processus spécifiques au projet
    local api_processes=$(pgrep -f "uvicorn.*app.main:app" | wc -l)
    local redis_processes=$(pgrep -f "redis-server" | wc -l)
    local postgres_processes=$(pgrep -f "postgres" | wc -l)
    
    metrics=$(echo "$metrics" | jq --arg api "$api_processes" --arg redis "$redis_processes" --arg postgres "$postgres_processes" '.processes.api = ($api | tonumber) | .processes.redis = ($redis | tonumber) | .processes.postgres = ($postgres | tonumber)')
    
    echo "$metrics" | tee -a "$METRICS_LOG"
    log_metric "CPU: ${cpu_usage:-N/A}%, Memory: ${memory_percent:-N/A}%, Disk: ${disk_usage:-N/A}%"
}

# Fonction de test de performance
run_performance_tests() {
    log_info "Exécution des tests de performance..."
    
    local api_url="http://$DEV_API_HOST:$DEV_API_PORT"
    local results=$(cat <<EOF
{
    "timestamp": "$(date -Iseconds)",
    "api_tests": {
        "health_check": {},
        "docs_endpoint": {},
        "load_test": {}
    }
}
EOF
)
    
    # Test de charge simple avec curl
    if command -v curl &> /dev/null; then
        log_info "Test de performance endpoint /health..."
        
        local total_time=0
        local successful_requests=0
        local failed_requests=0
        local test_iterations=10
        
        for i in $(seq 1 $test_iterations); do
            local start_time=$(date +%s%3N)
            local response_code=$(curl -s -o /dev/null -w "%{http_code}" "$api_url/health" 2>/dev/null || echo "000")
            local end_time=$(date +%s%3N)
            local request_time=$((end_time - start_time))
            
            total_time=$((total_time + request_time))
            
            if [ "$response_code" = "200" ]; then
                successful_requests=$((successful_requests + 1))
            else
                failed_requests=$((failed_requests + 1))
            fi
            
            sleep 0.1  # Petite pause entre les requêtes
        done
        
        local avg_response_time=$((total_time / test_iterations))
        local success_rate=$(echo "scale=2; $successful_requests * 100 / $test_iterations" | bc)
        
        results=$(echo "$results" | jq --arg avg "$avg_response_time" --arg success "$success_rate" --arg total "$test_iterations" '.api_tests.load_test.avg_response_time_ms = ($avg | tonumber) | .api_tests.load_test.success_rate_percent = ($success | tonumber) | .api_tests.load_test.total_requests = ($total | tonumber)')
        
        log_metric "Load test: ${avg_response_time}ms avg, ${success_rate}% success rate"
        
        # Alerte si performance dégradée
        if [ "$avg_response_time" -gt "$ALERT_THRESHOLD_RESPONSE_TIME" ]; then
            local alert_msg="Performance degradation: ${avg_response_time}ms avg response time"
            log_warning "$alert_msg"
            echo "$(date -Iseconds) - $alert_msg" >> "$ALERTS_LOG"
        fi
    fi
    
    echo "$results" | tee -a "$PERFORMANCE_LOG"
}

# Fonction de génération de rapport quotidien
generate_daily_report() {
    log_info "Génération du rapport quotidien..."
    
    local today=$(date +%Y-%m-%d)
    local report_file="$PROJECT_ROOT/logs/reports/daily_$today.json"
    
    # Agrégation des données du jour
    local health_data="[]"
    local metrics_data="[]"
    local alerts_data="[]"
    
    # Lecture des logs du jour si disponibles
    if [ -f "$HEALTH_LOG" ]; then
        health_data=$(grep "$today" "$HEALTH_LOG" | jq -s '.' 2>/dev/null || echo "[]")
    fi
    
    if [ -f "$METRICS_LOG" ]; then
        metrics_data=$(grep "$today" "$METRICS_LOG" | jq -s '.' 2>/dev/null || echo "[]")
    fi
    
    if [ -f "$ALERTS_LOG" ]; then
        alerts_data=$(grep "$today" "$ALERTS_LOG" | while read line; do echo "\"$line\""; done | jq -s '.' 2>/dev/null || echo "[]")
    fi
    
    # Calcul des statistiques
    local total_checks=$(echo "$health_data" | jq 'length')
    local healthy_checks=$(echo "$health_data" | jq '[.[] | select(.status == "healthy")] | length')
    local unhealthy_checks=$(echo "$health_data" | jq '[.[] | select(.status == "unhealthy" or .status == "down")] | length')
    local total_alerts=$(echo "$alerts_data" | jq 'length')
    
    local availability_rate="0"
    if [ "$total_checks" -gt 0 ]; then
        availability_rate=$(echo "scale=2; $healthy_checks * 100 / $total_checks" | bc)
    fi
    
    # Génération du rapport
    local report=$(cat <<EOF
{
    "date": "$today",
    "generated_at": "$(date -Iseconds)",
    "summary": {
        "total_health_checks": $total_checks,
        "healthy_checks": $healthy_checks,
        "unhealthy_checks": $unhealthy_checks,
        "availability_rate_percent": $availability_rate,
        "total_alerts": $total_alerts
    },
    "services": {
        "api": {},
        "postgresql": {},
        "redis": {}
    },
    "system_metrics": $metrics_data,
    "health_checks": $health_data,
    "alerts": $alerts_data
}
EOF
)
    
    # Calcul des moyennes par service
    for service in api postgresql redis; do
        local service_data=$(echo "$health_data" | jq "[.[] | select(.service == \"$service\")]")
        local service_count=$(echo "$service_data" | jq 'length')
        local service_healthy=$(echo "$service_data" | jq '[.[] | select(.status == "healthy")] | length')
        local avg_response_time=$(echo "$service_data" | jq '[.[] | .response_time] | if length > 0 then add / length else 0 end')
        
        local service_availability="0"
        if [ "$service_count" -gt 0 ]; then
            service_availability=$(echo "scale=2; $service_healthy * 100 / $service_count" | bc)
        fi
        
        report=$(echo "$report" | jq --arg svc "$service" --arg avail "$service_availability" --arg resp "$avg_response_time" --arg count "$service_count" '.services[$svc] = {"availability_percent": ($avail | tonumber), "avg_response_time_ms": ($resp | tonumber), "total_checks": ($count | tonumber)}')
    done
    
    echo "$report" | jq '.' > "$report_file"
    
    log_success "Rapport quotidien généré: $report_file"
    log_info "Disponibilité globale: ${availability_rate}%"
    log_info "Total des alertes: $total_alerts"
}

# Fonction de nettoyage des anciens logs
cleanup_old_logs() {
    log_info "Nettoyage des anciens logs (>${LOG_RETENTION_DAYS} jours)..."
    
    local logs_dir="$PROJECT_ROOT/logs"
    
    if [ -d "$logs_dir" ]; then
        find "$logs_dir" -name "*.log" -mtime +$LOG_RETENTION_DAYS -delete 2>/dev/null || true
        find "$logs_dir" -name "daily_*.json" -mtime +$LOG_RETENTION_DAYS -delete 2>/dev/null || true
        
        log_success "Nettoyage des logs terminé"
    fi
}

# Fonction d'affichage du tableau de bord
show_dashboard() {
    clear
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                    MONITORING DASHBOARD - DEVELOPMENT                       ║${NC}"
    echo -e "${BLUE}║                         Spotify AI Agent - $(date '+%Y-%m-%d %H:%M:%S')                     ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo
    
    # Statut des services
    echo -e "${CYAN}┌─ STATUT DES SERVICES ─────────────────────────────────────────────────────────┐${NC}"
    
    # API
    if nc -z "$DEV_API_HOST" "$DEV_API_PORT" 2>/dev/null; then
        echo -e "│ ${GREEN}● API Service${NC}        http://$DEV_API_HOST:$DEV_API_PORT                    │"
    else
        echo -e "│ ${RED}● API Service${NC}        OFFLINE                                              │"
    fi
    
    # PostgreSQL
    if nc -z "$DEV_DB_HOST" "$DEV_DB_PORT" 2>/dev/null; then
        echo -e "│ ${GREEN}● PostgreSQL${NC}         $DEV_DB_HOST:$DEV_DB_PORT                          │"
    else
        echo -e "│ ${RED}● PostgreSQL${NC}         OFFLINE                                              │"
    fi
    
    # Redis
    if nc -z "$DEV_REDIS_HOST" "$DEV_REDIS_PORT" 2>/dev/null; then
        echo -e "│ ${GREEN}● Redis${NC}              $DEV_REDIS_HOST:$DEV_REDIS_PORT                     │"
    else
        echo -e "│ ${RED}● Redis${NC}              OFFLINE                                              │"
    fi
    
    echo -e "${CYAN}└───────────────────────────────────────────────────────────────────────────────┘${NC}"
    
    # Métriques système
    echo -e "${CYAN}┌─ MÉTRIQUES SYSTÈME ───────────────────────────────────────────────────────────┐${NC}"
    
    if command -v top &> /dev/null; then
        local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 || echo "N/A")
        echo -e "│ CPU Usage:           ${cpu_usage}%                                              │"
    fi
    
    if command -v free &> /dev/null; then
        local memory_info=$(free -m | awk 'NR==2{printf "%.1f%%", $3*100/$2}')
        echo -e "│ Memory Usage:        ${memory_info}                                             │"
    fi
    
    if command -v df &> /dev/null; then
        local disk_usage=$(df -h / | awk 'NR==2 {print $5}')
        echo -e "│ Disk Usage:          ${disk_usage}                                              │"
    fi
    
    echo -e "${CYAN}└───────────────────────────────────────────────────────────────────────────────┘${NC}"
    
    # Dernières alertes
    echo -e "${CYAN}┌─ DERNIÈRES ALERTES ───────────────────────────────────────────────────────────┐${NC}"
    
    if [ -f "$ALERTS_LOG" ]; then
        local recent_alerts=$(tail -3 "$ALERTS_LOG" 2>/dev/null)
        if [ -n "$recent_alerts" ]; then
            echo "$recent_alerts" | while IFS= read -r line; do
                if [ -n "$line" ]; then
                    echo -e "│ ${YELLOW}⚠${NC} ${line:20:65}... │"
                fi
            done
        else
            echo -e "│ ${GREEN}✓${NC} Aucune alerte récente                                               │"
        fi
    else
        echo -e "│ ${GREEN}✓${NC} Aucune alerte récente                                               │"
    fi
    
    echo -e "${CYAN}└───────────────────────────────────────────────────────────────────────────────┘${NC}"
    
    echo -e "\nDernière mise à jour: $(date '+%H:%M:%S')"
    echo -e "Appuyez sur ${BLUE}Ctrl+C${NC} pour arrêter le monitoring"
}

# Fonction d'affichage de l'aide
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Surveille la santé et les performances de l'environnement de développement"
    echo
    echo "OPTIONS:"
    echo "  --once               Exécute une seule vérification et quitte"
    echo "  --dashboard          Mode tableau de bord interactif"
    echo "  --health-only        Vérifie seulement la santé des services"
    echo "  --metrics-only       Collecte seulement les métriques système"
    echo "  --performance        Exécute les tests de performance"
    echo "  --report             Génère le rapport quotidien"
    echo "  --cleanup            Nettoie les anciens logs"
    echo "  --interval SECONDS   Intervalle de monitoring (défaut: 30s)"
    echo "  --help               Affiche cette aide"
    echo
    echo "VARIABLES D'ENVIRONNEMENT:"
    echo "  MONITORING_INTERVAL           Intervalle en secondes (défaut: 30)"
    echo "  ALERT_THRESHOLD_CPU           Seuil d'alerte CPU % (défaut: 80)"
    echo "  ALERT_THRESHOLD_MEMORY        Seuil d'alerte mémoire % (défaut: 85)"
    echo "  ALERT_THRESHOLD_DISK          Seuil d'alerte disque % (défaut: 90)"
    echo "  ALERT_THRESHOLD_RESPONSE_TIME Seuil temps réponse ms (défaut: 5000)"
    echo "  LOG_RETENTION_DAYS            Rétention logs en jours (défaut: 7)"
    echo
    echo "EXEMPLES:"
    echo "  $0                      # Monitoring continu"
    echo "  $0 --once              # Une seule vérification"
    echo "  $0 --dashboard          # Mode tableau de bord"
    echo "  $0 --performance        # Tests de performance"
    echo "  $0 --interval 60        # Monitoring toutes les 60s"
    echo
}

# Fonction principale
main() {
    local mode="continuous"
    local health_only=false
    local metrics_only=false
    local performance=false
    local report=false
    local cleanup=false
    local dashboard=false
    
    # Analyse des arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --once)
                mode="once"
                shift
                ;;
            --dashboard)
                dashboard=true
                mode="dashboard"
                shift
                ;;
            --health-only)
                health_only=true
                shift
                ;;
            --metrics-only)
                metrics_only=true
                shift
                ;;
            --performance)
                performance=true
                shift
                ;;
            --report)
                report=true
                shift
                ;;
            --cleanup)
                cleanup=true
                shift
                ;;
            --interval)
                MONITORING_INTERVAL="$2"
                shift 2
                ;;
            --help)
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
    
    # Initialisation
    load_environment
    init_monitoring
    
    # Nettoyage si demandé
    if [ "$cleanup" = true ]; then
        cleanup_old_logs
        exit 0
    fi
    
    # Génération de rapport si demandé
    if [ "$report" = true ]; then
        generate_daily_report
        exit 0
    fi
    
    # Mode tableau de bord
    if [ "$dashboard" = true ]; then
        while true; do
            show_dashboard
            sleep "$MONITORING_INTERVAL"
        done
    fi
    
    # Exécution selon le mode
    case $mode in
        "once")
            log_info "Exécution d'une vérification unique..."
            
            if [ "$metrics_only" != true ]; then
                check_api_health
                check_postgresql_health
                check_redis_health
            fi
            
            if [ "$health_only" != true ]; then
                collect_system_metrics
            fi
            
            if [ "$performance" = true ]; then
                run_performance_tests
            fi
            
            log_success "Vérification terminée"
            ;;
            
        "continuous")
            log_info "Démarrage du monitoring continu (intervalle: ${MONITORING_INTERVAL}s)"
            log_info "Appuyez sur Ctrl+C pour arrêter"
            echo
            
            # Boucle de monitoring
            while true; do
                log_info "--- Cycle de monitoring ---"
                
                if [ "$metrics_only" != true ]; then
                    check_api_health
                    check_postgresql_health
                    check_redis_health
                fi
                
                if [ "$health_only" != true ]; then
                    collect_system_metrics
                fi
                
                if [ "$performance" = true ]; then
                    run_performance_tests
                fi
                
                # Génération du rapport quotidien à minuit
                local current_hour=$(date +%H)
                local current_minute=$(date +%M)
                if [ "$current_hour" = "00" ] && [ "$current_minute" = "00" ]; then
                    generate_daily_report
                fi
                
                log_info "Prochaine vérification dans ${MONITORING_INTERVAL}s..."
                sleep "$MONITORING_INTERVAL"
            done
            ;;
    esac
}

# Gestion des signaux pour un arrêt propre
cleanup_on_exit() {
    log_info "Arrêt du monitoring..."
    exit 0
}

trap cleanup_on_exit SIGINT SIGTERM

# Exécution du script principal
main "$@"
