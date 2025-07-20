#!/bin/bash

# Script de Validation du Système de Monitoring - Spotify AI Agent
# =================================================================
#
# Validation complète de l'installation et du fonctionnement
# du système de monitoring avec tests automatisés
#
# Auteur: Fahed Mlaiel
# Version: 2.0.0

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/spotify-monitoring-validation.log"
REPORT_FILE="/tmp/validation_report_$(date +%Y%m%d_%H%M%S).html"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Variables
MONITORING_URL="http://localhost:8000"
GRAFANA_URL="http://localhost:3000"
PROMETHEUS_URL="http://localhost:9090"
ALERTMANAGER_URL="http://localhost:9093"
REDIS_URL="localhost:6379"
VERBOSE=false
GENERATE_REPORT=false
TENANT_ID="test_validation"

# Compteurs de tests
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
WARNING_TESTS=0

# Résultats des tests
declare -a TEST_RESULTS=()

# Fonctions utilitaires
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOG_FILE}"
    
    case $level in
        "ERROR") echo -e "${RED}❌ ${message}${NC}" >&2 ;;
        "SUCCESS") echo -e "${GREEN}✅ ${message}${NC}" ;;
        "WARNING") echo -e "${YELLOW}⚠️  ${message}${NC}" ;;
        "INFO") echo -e "${BLUE}ℹ️  ${message}${NC}" ;;
    esac
}

run_test() {
    local test_name="$1"
    local test_function="$2"
    local test_description="$3"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    log "INFO" "Test: ${test_name}"
    [[ "$VERBOSE" == "true" ]] && log "INFO" "Description: ${test_description}"
    
    local start_time=$(date +%s)
    local result=""
    local status=""
    local details=""
    
    if $test_function; then
        status="PASS"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        result="✅ PASS"
        log "SUCCESS" "${test_name}: PASS"
    else
        local exit_code=$?
        if [[ $exit_code -eq 2 ]]; then
            status="WARNING"
            WARNING_TESTS=$((WARNING_TESTS + 1))
            result="⚠️ WARNING"
            log "WARNING" "${test_name}: WARNING"
        else
            status="FAIL"
            FAILED_TESTS=$((FAILED_TESTS + 1))
            result="❌ FAIL"
            log "ERROR" "${test_name}: FAIL"
        fi
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    TEST_RESULTS+=("${test_name}|${status}|${duration}|${test_description}")
    
    echo "----------------------------------------"
}

# Tests de connectivité
test_redis_connectivity() {
    if command -v redis-cli &> /dev/null; then
        if redis-cli -h localhost -p 6379 ping | grep -q "PONG"; then
            return 0
        fi
    fi
    return 1
}

test_prometheus_connectivity() {
    local response=$(curl -s -w "%{http_code}" -o /dev/null "${PROMETHEUS_URL}/-/healthy")
    if [[ "$response" -eq 200 ]]; then
        return 0
    fi
    return 1
}

test_grafana_connectivity() {
    local response=$(curl -s -w "%{http_code}" -o /dev/null "${GRAFANA_URL}/api/health")
    if [[ "$response" -eq 200 ]]; then
        return 0
    fi
    return 1
}

test_alertmanager_connectivity() {
    local response=$(curl -s -w "%{http_code}" -o /dev/null "${ALERTMANAGER_URL}/-/healthy")
    if [[ "$response" -eq 200 ]]; then
        return 0
    fi
    return 1
}

test_monitoring_api_connectivity() {
    local response=$(curl -s -w "%{http_code}" -o /dev/null "${MONITORING_URL}/health")
    if [[ "$response" -eq 200 ]]; then
        return 0
    fi
    return 1
}

# Tests fonctionnels
test_metrics_collection() {
    local metrics_response=$(curl -s "${MONITORING_URL}/metrics")
    if echo "$metrics_response" | grep -q "prometheus_"; then
        return 0
    fi
    return 1
}

test_alert_rule_creation() {
    local test_rule='{
        "name": "Test Rule",
        "description": "Test rule for validation",
        "tenant_id": "'${TENANT_ID}'",
        "rule_type": "threshold",
        "severity": "medium",
        "conditions": [
            {
                "metric_name": "test_metric",
                "operator": ">",
                "value": 100,
                "time_window_seconds": 300,
                "aggregation": "avg"
            }
        ]
    }'
    
    local response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$test_rule" \
        -w "%{http_code}" \
        -o /dev/null \
        "${MONITORING_URL}/api/v1/rules")
    
    if [[ "$response" -eq 200 ]] || [[ "$response" -eq 201 ]]; then
        return 0
    fi
    return 1
}

test_dashboard_generation() {
    local dashboard_request='{
        "tenant_id": "'${TENANT_ID}'",
        "dashboard_type": "technical"
    }'
    
    local response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$dashboard_request" \
        -w "%{http_code}" \
        -o /dev/null \
        "${MONITORING_URL}/api/v1/dashboards/generate")
    
    if [[ "$response" -eq 200 ]] || [[ "$response" -eq 201 ]]; then
        return 0
    fi
    return 1
}

test_notification_dispatch() {
    local test_notification='{
        "title": "Test Notification",
        "message": "This is a test notification for validation",
        "priority": "low",
        "tenant_id": "'${TENANT_ID}'",
        "channel": "email",
        "recipient": {
            "name": "Test User",
            "email": "test@example.com"
        }
    }'
    
    local response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$test_notification" \
        -w "%{http_code}" \
        -o /dev/null \
        "${MONITORING_URL}/api/v1/notifications/send")
    
    if [[ "$response" -eq 200 ]] || [[ "$response" -eq 202 ]]; then
        return 0
    fi
    return 1
}

# Tests de performance
test_api_response_time() {
    local start_time=$(date +%s%N)
    curl -s "${MONITORING_URL}/health" > /dev/null
    local end_time=$(date +%s%N)
    
    local response_time=$(( (end_time - start_time) / 1000000 )) # en millisecondes
    
    if [[ $response_time -lt 1000 ]]; then  # moins de 1 seconde
        return 0
    elif [[ $response_time -lt 5000 ]]; then  # moins de 5 secondes
        return 2  # WARNING
    else
        return 1  # FAIL
    fi
}

test_memory_usage() {
    local memory_usage=$(docker stats --no-stream --format "table {{.MemPerc}}" spotify-monitoring-app 2>/dev/null | tail -n +2 | head -1 | sed 's/%//')
    
    if [[ -n "$memory_usage" ]]; then
        local mem_num=$(echo "$memory_usage" | cut -d'.' -f1)
        if [[ $mem_num -lt 80 ]]; then
            return 0
        elif [[ $mem_num -lt 90 ]]; then
            return 2  # WARNING
        else
            return 1  # FAIL
        fi
    fi
    return 2  # WARNING - unable to get memory stats
}

test_cpu_usage() {
    local cpu_usage=$(docker stats --no-stream --format "table {{.CPUPerc}}" spotify-monitoring-app 2>/dev/null | tail -n +2 | head -1 | sed 's/%//')
    
    if [[ -n "$cpu_usage" ]]; then
        local cpu_num=$(echo "$cpu_usage" | cut -d'.' -f1)
        if [[ $cpu_num -lt 80 ]]; then
            return 0
        elif [[ $cpu_num -lt 90 ]]; then
            return 2  # WARNING
        else
            return 1  # FAIL
        fi
    fi
    return 2  # WARNING - unable to get CPU stats
}

# Tests de sécurité
test_unauthorized_access() {
    # Test d'accès non autorisé aux endpoints sensibles
    local response=$(curl -s -w "%{http_code}" -o /dev/null "${MONITORING_URL}/api/v1/admin/config")
    
    # On s'attend à un code d'erreur (401, 403, etc.)
    if [[ "$response" -eq 401 ]] || [[ "$response" -eq 403 ]]; then
        return 0
    elif [[ "$response" -eq 404 ]]; then
        return 2  # WARNING - endpoint might not exist
    else
        return 1  # FAIL - unauthorized access allowed
    fi
}

test_ssl_configuration() {
    # Vérification de la configuration SSL/TLS
    if curl -s -k -I "${MONITORING_URL}" | grep -q "Strict-Transport-Security"; then
        return 0
    else
        return 2  # WARNING - HSTS header not found
    fi
}

# Tests d'intégration
test_prometheus_scraping() {
    local targets_response=$(curl -s "${PROMETHEUS_URL}/api/v1/targets")
    if echo "$targets_response" | jq -e '.data.activeTargets[] | select(.labels.job == "spotify-monitoring")' > /dev/null 2>&1; then
        return 0
    fi
    return 1
}

test_grafana_datasource() {
    # Test de base sans authentification
    local datasources_response=$(curl -s "${GRAFANA_URL}/api/datasources")
    if echo "$datasources_response" | grep -q "prometheus" 2>/dev/null; then
        return 0
    else
        return 2  # WARNING - might need authentication
    fi
}

test_alertmanager_config() {
    local config_response=$(curl -s "${ALERTMANAGER_URL}/api/v1/status")
    if echo "$config_response" | jq -e '.data.configYAML' > /dev/null 2>&1; then
        return 0
    fi
    return 1
}

# Tests de données
test_metrics_data_integrity() {
    local query="up"
    local response=$(curl -s "${PROMETHEUS_URL}/api/v1/query?query=${query}")
    
    if echo "$response" | jq -e '.data.result | length > 0' > /dev/null 2>&1; then
        return 0
    fi
    return 1
}

test_correlation_engine() {
    local correlation_response=$(curl -s "${MONITORING_URL}/api/v1/correlations/status")
    local status_code=$(curl -s -w "%{http_code}" -o /dev/null "${MONITORING_URL}/api/v1/correlations/status")
    
    if [[ "$status_code" -eq 200 ]]; then
        return 0
    elif [[ "$status_code" -eq 404 ]]; then
        return 2  # WARNING - endpoint might not be implemented
    else
        return 1
    fi
}

# Génération de rapport HTML
generate_html_report() {
    if [[ "$GENERATE_REPORT" != "true" ]]; then
        return
    fi
    
    log "INFO" "Génération du rapport HTML: ${REPORT_FILE}"
    
    cat > "$REPORT_FILE" << EOF
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport de Validation - Spotify AI Agent Monitoring</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; padding: 20px; background: linear-gradient(135deg, #1db954, #1ed760); color: white; border-radius: 8px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric-card { padding: 20px; border-radius: 8px; text-align: center; color: white; }
        .metric-card.passed { background-color: #28a745; }
        .metric-card.failed { background-color: #dc3545; }
        .metric-card.warning { background-color: #ffc107; color: #333; }
        .metric-card.total { background-color: #007bff; }
        .test-results { margin-top: 30px; }
        .test-item { display: flex; align-items: center; padding: 15px; margin: 5px 0; border-radius: 5px; border-left: 4px solid; }
        .test-item.pass { background-color: #d4edda; border-color: #28a745; }
        .test-item.fail { background-color: #f8d7da; border-color: #dc3545; }
        .test-item.warning { background-color: #fff3cd; border-color: #ffc107; }
        .test-name { font-weight: bold; margin-right: 15px; min-width: 200px; }
        .test-status { padding: 5px 10px; border-radius: 3px; color: white; font-size: 12px; margin-right: 10px; }
        .test-status.pass { background-color: #28a745; }
        .test-status.fail { background-color: #dc3545; }
        .test-status.warning { background-color: #ffc107; color: #333; }
        .test-duration { font-size: 12px; color: #666; margin-left: auto; }
        .footer { text-align: center; margin-top: 30px; padding: 20px; color: #666; border-top: 1px solid #eee; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎵 Spotify AI Agent Monitoring</h1>
            <h2>Rapport de Validation du Système</h2>
            <p>Généré le: $(date '+%d/%m/%Y à %H:%M:%S')</p>
        </div>
        
        <div class="summary">
            <div class="metric-card total">
                <h3>${TOTAL_TESTS}</h3>
                <p>Tests Exécutés</p>
            </div>
            <div class="metric-card passed">
                <h3>${PASSED_TESTS}</h3>
                <p>Tests Réussis</p>
            </div>
            <div class="metric-card warning">
                <h3>${WARNING_TESTS}</h3>
                <p>Avertissements</p>
            </div>
            <div class="metric-card failed">
                <h3>${FAILED_TESTS}</h3>
                <p>Tests Échoués</p>
            </div>
        </div>
        
        <div class="test-results">
            <h3>Résultats Détaillés</h3>
EOF

    for result in "${TEST_RESULTS[@]}"; do
        IFS='|' read -r name status duration description <<< "$result"
        local css_class=""
        local status_text=""
        
        case $status in
            "PASS") css_class="pass"; status_text="PASS" ;;
            "FAIL") css_class="fail"; status_text="FAIL" ;;
            "WARNING") css_class="warning"; status_text="WARN" ;;
        esac
        
        cat >> "$REPORT_FILE" << EOF
            <div class="test-item ${css_class}">
                <div class="test-name">${name}</div>
                <div class="test-status ${css_class}">${status_text}</div>
                <div class="test-description">${description}</div>
                <div class="test-duration">${duration}s</div>
            </div>
EOF
    done

    cat >> "$REPORT_FILE" << EOF
        </div>
        
        <div class="footer">
            <p>Système développé par <strong>Fahed Mlaiel</strong></p>
            <p>Lead Dev + Architecte IA | Ingénieur ML | Spécialiste Sécurité Backend | Architecte Microservices</p>
        </div>
    </div>
</body>
</html>
EOF

    log "SUCCESS" "Rapport HTML généré: ${REPORT_FILE}"
}

# Fonction principale
main() {
    echo "=========================================="
    echo "  Validation Spotify AI Agent Monitoring"
    echo "=========================================="
    echo
    
    # Création du fichier de log
    touch "$LOG_FILE"
    
    log "INFO" "Début de la validation du système..."
    log "INFO" "Mode verbose: $VERBOSE"
    log "INFO" "Génération de rapport: $GENERATE_REPORT"
    
    # Tests de connectivité
    echo "=== Tests de Connectivité ==="
    run_test "Redis Connectivity" "test_redis_connectivity" "Vérification de la connectivité à Redis"
    run_test "Prometheus Connectivity" "test_prometheus_connectivity" "Vérification de la connectivité à Prometheus"
    run_test "Grafana Connectivity" "test_grafana_connectivity" "Vérification de la connectivité à Grafana"
    run_test "AlertManager Connectivity" "test_alertmanager_connectivity" "Vérification de la connectivité à AlertManager"
    run_test "Monitoring API Connectivity" "test_monitoring_api_connectivity" "Vérification de l'API de monitoring"
    
    # Tests fonctionnels
    echo "=== Tests Fonctionnels ==="
    run_test "Metrics Collection" "test_metrics_collection" "Test de collecte des métriques"
    run_test "Alert Rule Creation" "test_alert_rule_creation" "Test de création de règles d'alerte"
    run_test "Dashboard Generation" "test_dashboard_generation" "Test de génération de dashboard"
    run_test "Notification Dispatch" "test_notification_dispatch" "Test d'envoi de notifications"
    
    # Tests de performance
    echo "=== Tests de Performance ==="
    run_test "API Response Time" "test_api_response_time" "Test du temps de réponse de l'API"
    run_test "Memory Usage" "test_memory_usage" "Vérification de l'utilisation mémoire"
    run_test "CPU Usage" "test_cpu_usage" "Vérification de l'utilisation CPU"
    
    # Tests de sécurité
    echo "=== Tests de Sécurité ==="
    run_test "Unauthorized Access" "test_unauthorized_access" "Test d'accès non autorisé"
    run_test "SSL Configuration" "test_ssl_configuration" "Vérification de la configuration SSL"
    
    # Tests d'intégration
    echo "=== Tests d'Intégration ==="
    run_test "Prometheus Scraping" "test_prometheus_scraping" "Test du scraping Prometheus"
    run_test "Grafana Datasource" "test_grafana_datasource" "Test de la source de données Grafana"
    run_test "AlertManager Config" "test_alertmanager_config" "Test de la configuration AlertManager"
    
    # Tests de données
    echo "=== Tests de Données ==="
    run_test "Metrics Data Integrity" "test_metrics_data_integrity" "Test d'intégrité des données de métriques"
    run_test "Correlation Engine" "test_correlation_engine" "Test du moteur de corrélation"
    
    # Génération du rapport
    generate_html_report
    
    # Résumé final
    echo
    echo "=========================================="
    echo "  RÉSUMÉ DE LA VALIDATION"
    echo "=========================================="
    echo "Total des tests:     $TOTAL_TESTS"
    echo "Tests réussis:       $PASSED_TESTS"
    echo "Avertissements:      $WARNING_TESTS"
    echo "Tests échoués:       $FAILED_TESTS"
    echo
    
    local success_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    echo "Taux de réussite:    ${success_rate}%"
    
    if [[ $FAILED_TESTS -eq 0 ]]; then
        log "SUCCESS" "Validation terminée avec succès! 🎉"
        echo "✅ Le système est opérationnel"
    elif [[ $FAILED_TESTS -le 2 ]]; then
        log "WARNING" "Validation terminée avec des problèmes mineurs"
        echo "⚠️  Le système fonctionne mais nécessite des ajustements"
    else
        log "ERROR" "Validation échouée avec des problèmes critiques"
        echo "❌ Le système nécessite des corrections importantes"
        exit 1
    fi
    
    echo
    echo "Fichiers générés:"
    echo "  • Log de validation: $LOG_FILE"
    [[ "$GENERATE_REPORT" == "true" ]] && echo "  • Rapport HTML: $REPORT_FILE"
    echo
}

# Gestion des arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --report|-r)
            GENERATE_REPORT=true
            shift
            ;;
        --tenant)
            TENANT_ID="$2"
            shift 2
            ;;
        --monitoring-url)
            MONITORING_URL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  --verbose, -v         Mode verbeux"
            echo "  --report, -r          Générer un rapport HTML"
            echo "  --tenant ID           ID du tenant pour les tests"
            echo "  --monitoring-url URL  URL de l'API de monitoring"
            echo "  --help, -h            Afficher cette aide"
            exit 0
            ;;
        *)
            log "ERROR" "Option inconnue: $1"
            exit 1
            ;;
    esac
done

# Exécution
main
