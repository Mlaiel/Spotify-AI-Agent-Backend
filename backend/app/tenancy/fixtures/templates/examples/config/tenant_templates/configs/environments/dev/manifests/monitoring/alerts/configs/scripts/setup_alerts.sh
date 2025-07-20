#!/bin/bash

# Script de Configuration des Alertes - Spotify AI Agent
# ======================================================
#
# Configuration automatique des règles d'alertes et des notifications
# pour un tenant spécifique avec ML et intelligence artificielle
#
# Auteur: Fahed Mlaiel
# Version: 2.0.0

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/.."
LOG_FILE="/var/log/spotify-alerts-setup.log"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Variables
TENANT_ID=""
ENVIRONMENT="dev"
MONITORING_URL="http://localhost:8000"
GRAFANA_URL="http://localhost:3000"
GRAFANA_API_KEY=""
DRY_RUN=false
FORCE_UPDATE=false

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

show_help() {
    cat << EOF
Usage: $0 --tenant TENANT_ID [OPTIONS]

Configuration des alertes pour un tenant spécifique

Options obligatoires:
  --tenant TENANT_ID        ID du tenant à configurer

Options:
  --environment ENV         Environnement (dev/staging/prod) [default: dev]
  --monitoring-url URL      URL de l'API de monitoring [default: http://localhost:8000]
  --grafana-url URL         URL de Grafana [default: http://localhost:3000]  
  --grafana-api-key KEY     Clé API Grafana
  --dry-run                 Mode simulation sans modifications
  --force                   Forcer la mise à jour des règles existantes
  --help, -h                Afficher cette aide

Exemples:
  $0 --tenant spotify_main
  $0 --tenant test_client --environment prod --force
  $0 --tenant demo --dry-run

Configuration avancée:
  Le script configure automatiquement:
  • Règles d'alertes intelligentes avec ML
  • Seuils adaptatifs par service
  • Notifications multi-canal (Email, Slack, Teams)
  • Escalade automatique selon les SLA
  • Dashboards personnalisés
  • Corrélation d'événements
EOF
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --tenant)
                TENANT_ID="$2"
                shift 2
                ;;
            --environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --monitoring-url)
                MONITORING_URL="$2"
                shift 2
                ;;
            --grafana-url)
                GRAFANA_URL="$2"
                shift 2
                ;;
            --grafana-api-key)
                GRAFANA_API_KEY="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --force)
                FORCE_UPDATE=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log "ERROR" "Option inconnue: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    if [[ -z "$TENANT_ID" ]]; then
        log "ERROR" "ID de tenant requis. Utiliser --tenant TENANT_ID"
        show_help
        exit 1
    fi
}

check_services() {
    log "INFO" "Vérification des services..."
    
    # Vérification API de monitoring
    if ! curl -s -f "${MONITORING_URL}/health" > /dev/null; then
        log "ERROR" "Service de monitoring non accessible: ${MONITORING_URL}"
        exit 1
    fi
    
    # Vérification Grafana
    if ! curl -s -f "${GRAFANA_URL}/api/health" > /dev/null; then
        log "ERROR" "Grafana non accessible: ${GRAFANA_URL}"
        exit 1
    fi
    
    log "SUCCESS" "Tous les services sont accessibles"
}

create_tenant_alert_rules() {
    log "INFO" "Création des règles d'alertes pour ${TENANT_ID}..."
    
    local rules_file="/tmp/alert_rules_${TENANT_ID}.json"
    
    cat > "$rules_file" << EOF
{
  "tenant_id": "${TENANT_ID}",
  "environment": "${ENVIRONMENT}",
  "rules": [
    {
      "name": "High Error Rate - ${TENANT_ID}",
      "description": "Détection de taux d'erreur élevé pour ${TENANT_ID}",
      "rule_type": "threshold",
      "severity": "critical",
      "conditions": [
        {
          "metric_name": "http_requests_total",
          "operator": ">",
          "value": 0.05,
          "labels": {
            "tenant_id": "${TENANT_ID}",
            "status": "~5.."
          },
          "time_window_seconds": 300,
          "aggregation": "rate"
        }
      ],
      "logical_operator": "and",
      "auto_resolve": true,
      "adaptive_thresholds": true,
      "learning_enabled": true,
      "cooldown_seconds": 600,
      "notification_channels": ["email", "slack"],
      "escalation_rules": [
        {
          "level": "l2",
          "delay_minutes": 15
        },
        {
          "level": "l3", 
          "delay_minutes": 30
        }
      ]
    },
    {
      "name": "High Latency - ${TENANT_ID}",
      "description": "Latence élevée détectée pour ${TENANT_ID}",
      "rule_type": "threshold",
      "severity": "high",
      "conditions": [
        {
          "metric_name": "http_request_duration_seconds",
          "operator": ">",
          "value": 1.0,
          "labels": {
            "tenant_id": "${TENANT_ID}"
          },
          "time_window_seconds": 300,
          "aggregation": "p95"
        }
      ],
      "adaptive_thresholds": true,
      "learning_enabled": true,
      "notification_channels": ["email", "slack"]
    },
    {
      "name": "Service Down - ${TENANT_ID}",
      "description": "Service indisponible pour ${TENANT_ID}",
      "rule_type": "threshold",
      "severity": "critical",
      "conditions": [
        {
          "metric_name": "up",
          "operator": "==",
          "value": 0,
          "labels": {
            "tenant_id": "${TENANT_ID}"
          },
          "time_window_seconds": 60,
          "aggregation": "avg"
        }
      ],
      "cooldown_seconds": 300,
      "notification_channels": ["email", "slack", "sms"],
      "escalation_rules": [
        {
          "level": "l2",
          "delay_minutes": 5
        },
        {
          "level": "manager",
          "delay_minutes": 15
        }
      ]
    },
    {
      "name": "Memory Usage High - ${TENANT_ID}",
      "description": "Utilisation mémoire élevée pour ${TENANT_ID}",
      "rule_type": "threshold",
      "severity": "warning",
      "conditions": [
        {
          "metric_name": "system_memory_percent",
          "operator": ">",
          "value": 85.0,
          "labels": {
            "tenant_id": "${TENANT_ID}"
          },
          "time_window_seconds": 600,
          "aggregation": "avg"
        }
      ],
      "adaptive_thresholds": true,
      "notification_channels": ["email"]
    },
    {
      "name": "CPU Usage High - ${TENANT_ID}",
      "description": "Utilisation CPU élevée pour ${TENANT_ID}",
      "rule_type": "threshold",
      "severity": "warning",
      "conditions": [
        {
          "metric_name": "system_cpu_percent",
          "operator": ">",
          "value": 80.0,
          "labels": {
            "tenant_id": "${TENANT_ID}"
          },
          "time_window_seconds": 600,
          "aggregation": "avg"
        }
      ],
      "adaptive_thresholds": true,
      "notification_channels": ["email"]
    },
    {
      "name": "Database Connection Pool Exhausted - ${TENANT_ID}",
      "description": "Pool de connexions base de données épuisé",
      "rule_type": "threshold",
      "severity": "high",
      "conditions": [
        {
          "metric_name": "database_connections_active",
          "operator": ">=",
          "value": 95,
          "labels": {
            "tenant_id": "${TENANT_ID}"
          },
          "time_window_seconds": 120,
          "aggregation": "max"
        }
      ],
      "notification_channels": ["email", "slack"]
    },
    {
      "name": "Spotify API Rate Limit Approaching - ${TENANT_ID}",
      "description": "Limite de taux API Spotify approchée",
      "rule_type": "threshold",
      "severity": "medium",
      "conditions": [
        {
          "metric_name": "spotify_api_requests_remaining",
          "operator": "<",
          "value": 100,
          "labels": {
            "tenant_id": "${TENANT_ID}"
          },
          "time_window_seconds": 300,
          "aggregation": "min"
        }
      ],
      "notification_channels": ["email"]
    },
    {
      "name": "ML Model Accuracy Drop - ${TENANT_ID}",
      "description": "Chute de précision du modèle ML",
      "rule_type": "threshold",
      "severity": "medium",
      "conditions": [
        {
          "metric_name": "ml_model_accuracy",
          "operator": "<",
          "value": 0.85,
          "labels": {
            "tenant_id": "${TENANT_ID}"
          },
          "time_window_seconds": 3600,
          "aggregation": "avg"
        }
      ],
      "learning_enabled": true,
      "notification_channels": ["email", "slack"]
    },
    {
      "name": "Security Anomaly Detected - ${TENANT_ID}",
      "description": "Anomalie de sécurité détectée",
      "rule_type": "anomaly",
      "severity": "high",
      "conditions": [
        {
          "metric_name": "security_events_total",
          "operator": "anomaly",
          "value": 3.0,
          "labels": {
            "tenant_id": "${TENANT_ID}"
          },
          "time_window_seconds": 600,
          "aggregation": "rate"
        }
      ],
      "notification_channels": ["email", "slack", "webhook"]
    },
    {
      "name": "Playlist Generation Failure Rate - ${TENANT_ID}",
      "description": "Taux d'échec de génération de playlists élevé",
      "rule_type": "business",
      "severity": "medium",
      "conditions": [
        {
          "metric_name": "spotify_playlist_generation_failures_total",
          "operator": ">",
          "value": 0.1,
          "labels": {
            "tenant_id": "${TENANT_ID}"
          },
          "time_window_seconds": 900,
          "aggregation": "rate"
        }
      ],
      "notification_channels": ["email"]
    }
  ]
}
EOF

    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Règles créées dans: $rules_file"
        cat "$rules_file" | jq '.'
    else
        # Envoi des règles à l'API
        local response=$(curl -s -X POST \
            -H "Content-Type: application/json" \
            -d @"$rules_file" \
            "${MONITORING_URL}/api/v1/rules/batch" \
            --write-out "HTTPSTATUS:%{http_code}")
        
        local body=$(echo "$response" | sed -E 's/HTTPSTATUS:[0-9]{3}$//')
        local status=$(echo "$response" | tr -d '\n' | sed -E 's/.*HTTPSTATUS:([0-9]{3})$/\1/')
        
        if [[ "$status" -eq 200 ]] || [[ "$status" -eq 201 ]]; then
            local created_rules=$(echo "$body" | jq -r '.created // 0')
            local updated_rules=$(echo "$body" | jq -r '.updated // 0')
            log "SUCCESS" "Règles configurées: ${created_rules} créées, ${updated_rules} mises à jour"
        else
            log "ERROR" "Erreur lors de la création des règles: HTTP $status"
            echo "$body" | jq '.' 2>/dev/null || echo "$body"
            exit 1
        fi
    fi
    
    rm -f "$rules_file"
}

setup_notification_channels() {
    log "INFO" "Configuration des canaux de notification..."
    
    local channels_file="/tmp/notification_channels_${TENANT_ID}.json"
    
    cat > "$channels_file" << EOF
{
  "tenant_id": "${TENANT_ID}",
  "channels": [
    {
      "name": "email_primary",
      "type": "email",
      "config": {
        "recipients": [
          {
            "name": "Admin ${TENANT_ID}",
            "email": "admin@${TENANT_ID}.com",
            "preferred_channels": ["email", "slack"],
            "quiet_hours": {
              "start": "22:00",
              "end": "08:00",
              "timezone": "UTC"
            }
          },
          {
            "name": "DevOps ${TENANT_ID}",
            "email": "devops@${TENANT_ID}.com",
            "preferred_channels": ["email", "slack", "sms"]
          }
        ],
        "template_id": "default_alert"
      }
    },
    {
      "name": "slack_alerts",
      "type": "slack",
      "config": {
        "webhook_url": "${SLACK_WEBHOOK_URL:-}",
        "channel": "#alerts-${TENANT_ID}",
        "username": "Spotify AI Monitor",
        "icon_emoji": ":warning:",
        "template_id": "slack_alert"
      }
    },
    {
      "name": "teams_critical",
      "type": "teams", 
      "config": {
        "webhook_url": "${TEAMS_WEBHOOK_URL:-}",
        "template_id": "teams_alert"
      }
    },
    {
      "name": "webhook_integration",
      "type": "webhook",
      "config": {
        "url": "${WEBHOOK_URL:-http://localhost:8080/alerts}",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${WEBHOOK_TOKEN:-}"
        },
        "template_id": "json_webhook"
      }
    }
  ]
}
EOF

    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Canaux de notification configurés dans: $channels_file"
        cat "$channels_file" | jq '.'
    else
        local response=$(curl -s -X POST \
            -H "Content-Type: application/json" \
            -d @"$channels_file" \
            "${MONITORING_URL}/api/v1/notifications/channels" \
            --write-out "HTTPSTATUS:%{http_code}")
        
        local status=$(echo "$response" | tr -d '\n' | sed -E 's/.*HTTPSTATUS:([0-9]{3})$/\1/')
        
        if [[ "$status" -eq 200 ]] || [[ "$status" -eq 201 ]]; then
            log "SUCCESS" "Canaux de notification configurés"
        else
            log "ERROR" "Erreur lors de la configuration des canaux: HTTP $status"
            exit 1
        fi
    fi
    
    rm -f "$channels_file"
}

create_escalation_rules() {
    log "INFO" "Configuration des règles d'escalade..."
    
    local escalation_file="/tmp/escalation_rules_${TENANT_ID}.json"
    
    cat > "$escalation_file" << EOF
{
  "tenant_id": "${TENANT_ID}",
  "escalation_rules": [
    {
      "name": "Critical Alerts Escalation",
      "severity": "critical",
      "escalation_path": ["l1", "l2", "l3", "manager"],
      "sla_thresholds": {
        "l1": 15,
        "l2": 30,
        "l3": 60,
        "manager": 120
      },
      "auto_escalate": true,
      "business_hours_only": false
    },
    {
      "name": "High Priority Escalation",
      "severity": "high",
      "escalation_path": ["l1", "l2", "l3"],
      "sla_thresholds": {
        "l1": 30,
        "l2": 60,
        "l3": 120
      },
      "auto_escalate": true,
      "business_hours_only": false
    },
    {
      "name": "Medium Priority Escalation",
      "severity": "medium",
      "escalation_path": ["l1", "l2"],
      "sla_thresholds": {
        "l1": 60,
        "l2": 240
      },
      "auto_escalate": true,
      "business_hours_only": true
    }
  ],
  "oncall_schedules": [
    {
      "team": "DevOps",
      "level": "l1",
      "primary_contact": "devops@${TENANT_ID}.com",
      "secondary_contact": "devops-backup@${TENANT_ID}.com",
      "max_concurrent_incidents": 5,
      "escalation_delay_minutes": 30
    },
    {
      "team": "Engineering",
      "level": "l2",
      "primary_contact": "engineering@${TENANT_ID}.com",
      "secondary_contact": "engineering-lead@${TENANT_ID}.com",
      "max_concurrent_incidents": 3,
      "escalation_delay_minutes": 15
    },
    {
      "team": "Architecture",
      "level": "l3",
      "primary_contact": "architects@${TENANT_ID}.com",
      "secondary_contact": "cto@${TENANT_ID}.com",
      "max_concurrent_incidents": 2,
      "escalation_delay_minutes": 10
    }
  ]
}
EOF

    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Règles d'escalade configurées dans: $escalation_file"
        cat "$escalation_file" | jq '.'
    else
        local response=$(curl -s -X POST \
            -H "Content-Type: application/json" \
            -d @"$escalation_file" \
            "${MONITORING_URL}/api/v1/escalation/rules" \
            --write-out "HTTPSTATUS:%{http_code}")
        
        local status=$(echo "$response" | tr -d '\n' | sed -E 's/.*HTTPSTATUS:([0-9]{3})$/\1/')
        
        if [[ "$status" -eq 200 ]] || [[ "$status" -eq 201 ]]; then
            log "SUCCESS" "Règles d'escalade configurées"
        else
            log "ERROR" "Erreur lors de la configuration d'escalade: HTTP $status"
            exit 1
        fi
    fi
    
    rm -f "$escalation_file"
}

generate_dashboards() {
    log "INFO" "Génération des dashboards Grafana..."
    
    if [[ -z "$GRAFANA_API_KEY" ]]; then
        log "WARNING" "Clé API Grafana non fournie, génération des dashboards ignorée"
        return
    fi
    
    local dashboard_types=("executive" "technical" "security" "ml_analytics")
    
    for dashboard_type in "${dashboard_types[@]}"; do
        log "INFO" "Génération du dashboard ${dashboard_type} pour ${TENANT_ID}..."
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log "INFO" "[DRY RUN] Générerait le dashboard ${dashboard_type}"
            continue
        fi
        
        local response=$(curl -s -X POST \
            -H "Content-Type: application/json" \
            -d "{\"tenant_id\": \"${TENANT_ID}\", \"dashboard_type\": \"${dashboard_type}\"}" \
            "${MONITORING_URL}/api/v1/dashboards/generate" \
            --write-out "HTTPSTATUS:%{http_code}")
        
        local status=$(echo "$response" | tr -d '\n' | sed -E 's/.*HTTPSTATUS:([0-9]{3})$/\1/')
        
        if [[ "$status" -eq 200 ]] || [[ "$status" -eq 201 ]]; then
            local dashboard_uid=$(echo "$response" | sed -E 's/HTTPSTATUS:[0-9]{3}$//' | jq -r '.uid // "unknown"')
            log "SUCCESS" "Dashboard ${dashboard_type} créé: ${dashboard_uid}"
        else
            log "WARNING" "Erreur lors de la création du dashboard ${dashboard_type}: HTTP $status"
        fi
    done
}

validate_configuration() {
    log "INFO" "Validation de la configuration..."
    
    # Test des règles d'alertes
    local rules_response=$(curl -s "${MONITORING_URL}/api/v1/rules?tenant_id=${TENANT_ID}")
    local rules_count=$(echo "$rules_response" | jq '. | length' 2>/dev/null || echo "0")
    
    if [[ "$rules_count" -gt 0 ]]; then
        log "SUCCESS" "${rules_count} règles d'alertes configurées"
    else
        log "WARNING" "Aucune règle d'alerte trouvée"
    fi
    
    # Test des canaux de notification
    local channels_response=$(curl -s "${MONITORING_URL}/api/v1/notifications/channels?tenant_id=${TENANT_ID}")
    local channels_count=$(echo "$channels_response" | jq '. | length' 2>/dev/null || echo "0")
    
    if [[ "$channels_count" -gt 0 ]]; then
        log "SUCCESS" "${channels_count} canaux de notification configurés"
    else
        log "WARNING" "Aucun canal de notification trouvé"
    fi
    
    # Test de connectivité
    if curl -s -f "${MONITORING_URL}/api/v1/health/tenant/${TENANT_ID}" > /dev/null; then
        log "SUCCESS" "Configuration tenant validée"
    else
        log "WARNING" "Problème de validation de la configuration tenant"
    fi
}

print_configuration_summary() {
    echo
    echo "=========================================="
    echo "  Configuration Terminée - ${TENANT_ID}"
    echo "=========================================="
    echo
    echo "Services configurés:"
    echo "  • Règles d'alertes intelligentes avec ML"
    echo "  • Seuils adaptatifs par service"
    echo "  • Notifications multi-canal"
    echo "  • Escalade automatique"
    echo "  • Dashboards Grafana personnalisés"
    echo "  • Corrélation d'événements"
    echo
    echo "URLs importantes:"
    echo "  • API Monitoring:  ${MONITORING_URL}/api/v1/tenants/${TENANT_ID}"
    echo "  • Dashboards:      ${GRAFANA_URL}/dashboards"
    echo "  • Alertes actives: ${MONITORING_URL}/api/v1/alerts?tenant_id=${TENANT_ID}"
    echo
    echo "Prochaines étapes:"
    echo "  1. Configurer les webhooks Slack/Teams"
    echo "  2. Ajuster les seuils selon les besoins"
    echo "  3. Tester les notifications"
    echo "  4. Former les équipes sur les procédures"
    echo
}

main() {
    echo "=========================================="
    echo "  Configuration Alertes Spotify AI Agent"
    echo "  Tenant: ${TENANT_ID}"
    echo "  Environment: ${ENVIRONMENT}"
    echo "=========================================="
    echo
    
    # Création du fichier de log
    sudo touch "$LOG_FILE" 2>/dev/null || touch "$LOG_FILE"
    
    log "INFO" "Début de la configuration des alertes pour ${TENANT_ID}..."
    
    # Étapes de configuration
    check_services
    create_tenant_alert_rules
    setup_notification_channels
    create_escalation_rules
    generate_dashboards
    validate_configuration
    print_configuration_summary
    
    log "SUCCESS" "Configuration des alertes terminée!"
}

# Gestion des arguments
parse_arguments "$@"

# Exécution
main
