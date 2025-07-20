#!/usr/bin/env bash
set -euo pipefail

# Spotify AI Agent - Tenancy Schema Management Script
# ==================================================
# 
# Script automatisé pour la gestion complète des schémas tenancy
# avec validation, génération, tests et déploiement industriel.
#
# Auteur: Équipe de développement Spotify AI Agent
# Version: 1.0.0

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../../../../../../../../../../" && pwd)"
SCHEMAS_DIR="$SCRIPT_DIR"
TEMP_DIR="/tmp/tenancy_schemas_$$"
LOG_FILE="/var/log/tenancy_schemas.log"

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction de logging
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [$level] $message" | tee -a "$LOG_FILE"
}

info() { log "INFO" "$@"; }
warn() { log "WARN" "${YELLOW}$*${NC}"; }
error() { log "ERROR" "${RED}$*${NC}"; }
success() { log "SUCCESS" "${GREEN}$*${NC}"; }

# Vérification des prérequis
check_prerequisites() {
    info "Vérification des prérequis..."
    
    local missing_deps=()
    
    # Vérifier Python et les dépendances
    if ! command -v python3 >/dev/null 2>&1; then
        missing_deps+=("python3")
    fi
    
    if ! python3 -c "import pydantic" >/dev/null 2>&1; then
        missing_deps+=("pydantic")
    fi
    
    if ! python3 -c "import yaml" >/dev/null 2>&1; then
        missing_deps+=("pyyaml")
    fi
    
    # Vérifier les outils système
    for tool in jq yq; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            missing_deps+=("$tool")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        error "Dépendances manquantes: ${missing_deps[*]}"
        info "Installation automatique..."
        install_dependencies "${missing_deps[@]}"
    fi
    
    success "Tous les prérequis sont satisfaits"
}

# Installation des dépendances
install_dependencies() {
    local deps=("$@")
    
    # Détecter le gestionnaire de paquets
    if command -v apt-get >/dev/null 2>&1; then
        info "Installation via apt-get..."
        sudo apt-get update -qq
        for dep in "${deps[@]}"; do
            case "$dep" in
                "python3") sudo apt-get install -y python3 python3-pip ;;
                "pydantic") pip3 install pydantic ;;
                "pyyaml") pip3 install pyyaml ;;
                "jq") sudo apt-get install -y jq ;;
                "yq") sudo apt-get install -y yq ;;
            esac
        done
    elif command -v yum >/dev/null 2>&1; then
        info "Installation via yum..."
        for dep in "${deps[@]}"; do
            case "$dep" in
                "python3") sudo yum install -y python3 python3-pip ;;
                "pydantic") pip3 install pydantic ;;
                "pyyaml") pip3 install pyyaml ;;
                "jq") sudo yum install -y jq ;;
                "yq") sudo yum install -y yq ;;
            esac
        done
    else
        error "Gestionnaire de paquets non supporté"
        exit 1
    fi
}

# Validation des schémas
validate_schemas() {
    info "Validation des schémas tenancy..."
    
    mkdir -p "$TEMP_DIR"
    local report_file="$TEMP_DIR/validation_report.txt"
    
    # Utiliser le script Python pour la validation
    if python3 "$SCHEMAS_DIR/schema_utils.py" validate "$SCHEMAS_DIR" --recursive --output "$report_file"; then
        success "Validation des schémas réussie"
        
        # Afficher le résumé
        if [ -f "$report_file" ]; then
            info "Résumé de validation:"
            head -20 "$report_file"
        fi
        
        return 0
    else
        error "Validation des schémas échouée"
        if [ -f "$report_file" ]; then
            cat "$report_file"
        fi
        return 1
    fi
}

# Génération de la documentation
generate_docs() {
    info "Génération de la documentation..."
    
    local docs_dir="$PROJECT_ROOT/docs/schemas/tenancy"
    mkdir -p "$docs_dir"
    
    # Générer la documentation des schémas
    python3 "$SCHEMAS_DIR/schema_utils.py" docs "$docs_dir"
    
    # Générer un index
    cat > "$docs_dir/README.md" << EOF
# Tenancy Schemas Documentation

Cette documentation présente les schémas de données pour le système multi-tenant
du Spotify AI Agent.

## Schémas disponibles

$(find "$docs_dir" -name "*_schema.md" -exec basename {} .md \; | sort | sed 's/^/- [/' | sed 's/$/_schema](/' | sed 's/$/_schema.md)/')

## Vue d'ensemble de l'architecture

Le système de tenancy utilise plusieurs types de schémas:

1. **Configuration Tenant** - Définition des tenants et leurs caractéristiques
2. **Alertes** - Gestion des alertes multi-tenant avec escalation
3. **Avertissements** - Système d'avertissement préventif avec ML
4. **Notifications** - Notifications multi-canal intelligentes
5. **Monitoring** - Configuration du monitoring avancé
6. **Compliance** - Gestion de la conformité et audit
7. **Performance** - Métriques et analytics de performance

## Utilisation

Chaque schéma peut être utilisé individuellement ou en combinaison pour créer
une configuration complète de tenant.

### Exemple d'utilisation

\`\`\`python
from tenancy.schemas.tenancy import TenantConfigSchema

config = TenantConfigSchema(
    tenant_id="enterprise_001",
    tenant_name="Acme Corporation",
    tenant_type="enterprise",
    admin_email="admin@acme.com",
    country_code="US"
)
\`\`\`

## Validation

Utilisez l'outil de validation pour vérifier vos configurations:

\`\`\`bash
python3 schema_utils.py validate config.json
\`\`\`

---

Généré automatiquement le $(date)
EOF
    
    success "Documentation générée dans $docs_dir"
}

# Tests automatisés
run_tests() {
    info "Exécution des tests automatisés..."
    
    local test_dir="$TEMP_DIR/tests"
    mkdir -p "$test_dir"
    
    # Générer des configurations de test
    generate_test_configs "$test_dir"
    
    # Valider les configurations de test
    local failed_tests=0
    local total_tests=0
    
    for test_file in "$test_dir"/*.json; do
        if [ -f "$test_file" ]; then
            total_tests=$((total_tests + 1))
            info "Test: $(basename "$test_file")"
            
            if python3 "$SCHEMAS_DIR/schema_utils.py" validate "$test_file" >/dev/null 2>&1; then
                success "✓ $(basename "$test_file")"
            else
                error "✗ $(basename "$test_file")"
                failed_tests=$((failed_tests + 1))
            fi
        fi
    done
    
    if [ $failed_tests -eq 0 ]; then
        success "Tous les tests passés ($total_tests/$total_tests)"
        return 0
    else
        error "$failed_tests tests échoués sur $total_tests"
        return 1
    fi
}

# Génération des configurations de test
generate_test_configs() {
    local test_dir="$1"
    
    # Configuration tenant enterprise
    cat > "$test_dir/tenant_enterprise.json" << 'EOF'
{
    "tenant_id": "test_enterprise_001",
    "tenant_name": "Test Enterprise Corp",
    "tenant_type": "enterprise",
    "admin_email": "admin@testenterprise.com",
    "country_code": "US",
    "features": {
        "advanced_analytics": true,
        "custom_alerts": true,
        "real_time_monitoring": true,
        "custom_branding": true,
        "sso_integration": true,
        "max_users": 1000,
        "max_storage_gb": 1000
    },
    "compliance_levels": ["gdpr", "soc2"]
}
EOF
    
    # Configuration alert
    cat > "$test_dir/alert_performance.json" << 'EOF'
{
    "tenant_id": "test_enterprise_001",
    "name": "high_cpu_usage_test",
    "title": "High CPU Usage Test Alert",
    "description": "Test alert for high CPU usage detection",
    "severity": "high",
    "category": "performance",
    "conditions": [{
        "metric_name": "cpu_usage_percent",
        "operator": "gt",
        "threshold": 80.0,
        "duration_minutes": 5
    }],
    "notification_channels": ["email", "slack"],
    "recipients": ["test@example.com"]
}
EOF
    
    # Configuration warning
    cat > "$test_dir/warning_threshold.json" << 'EOF'
{
    "tenant_id": "test_enterprise_001",
    "title": "CPU Usage Approaching Threshold",
    "description": "Test warning for CPU usage approaching critical threshold",
    "severity": "medium",
    "warning_type": "threshold_approaching",
    "category": "infrastructure",
    "source": "monitoring_system",
    "metrics": [{
        "metric_name": "cpu_usage_percent",
        "current_value": 75.5,
        "threshold_value": 80.0,
        "trend": "increasing"
    }]
}
EOF
    
    # Configuration notification
    cat > "$test_dir/notification_alert.json" << 'EOF'
{
    "tenant_id": "test_enterprise_001",
    "notification_type": "alert",
    "priority": "high",
    "title": "Test Critical System Alert",
    "message": "Test notification for critical system alert",
    "preferred_channels": ["email", "slack"],
    "recipients": [{
        "recipient_id": "admin_001",
        "recipient_type": "user",
        "contact_info": {"email": "admin@test.com"},
        "timezone": "UTC"
    }]
}
EOF
}

# Benchmarks de performance
run_benchmarks() {
    info "Exécution des benchmarks de performance..."
    
    local benchmark_results="$TEMP_DIR/benchmark_results.json"
    
    # Test de validation en masse
    local start_time=$(date +%s.%N)
    
    # Générer 1000 configurations de test
    local test_configs_dir="$TEMP_DIR/benchmark_configs"
    mkdir -p "$test_configs_dir"
    
    for i in $(seq 1 1000); do
        cat > "$test_configs_dir/tenant_$i.json" << EOF
{
    "tenant_id": "benchmark_tenant_$i",
    "tenant_name": "Benchmark Tenant $i",
    "tenant_type": "standard",
    "admin_email": "admin$i@benchmark.com",
    "country_code": "US"
}
EOF
    done
    
    # Valider toutes les configurations
    python3 "$SCHEMAS_DIR/schema_utils.py" validate "$test_configs_dir" --recursive > /dev/null
    
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)
    local configs_per_second=$(echo "scale=2; 1000 / $duration" | bc)
    
    # Sauvegarder les résultats
    cat > "$benchmark_results" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "validation_performance": {
        "total_configs": 1000,
        "duration_seconds": $duration,
        "configs_per_second": $configs_per_second
    }
}
EOF
    
    success "Benchmark terminé: $configs_per_second configurations/seconde"
    info "Résultats sauvegardés dans $benchmark_results"
}

# Génération d'exemples
generate_examples() {
    info "Génération d'exemples de configuration..."
    
    local examples_dir="$PROJECT_ROOT/examples/tenancy_schemas"
    mkdir -p "$examples_dir"
    
    # Exemple complet enterprise
    python3 "$SCHEMAS_DIR/schema_utils.py" generate tenant_config enterprise \
        --variables '{"tenant_id": "example_enterprise", "tenant_name": "Example Enterprise", "admin_email": "admin@example-enterprise.com", "country_code": "US"}' \
        --output "$examples_dir/enterprise_tenant.json"
    
    # Exemple alert critique
    python3 "$SCHEMAS_DIR/schema_utils.py" generate alert performance \
        --variables '{"tenant_id": "example_enterprise", "metric_name": "response_time_ms", "metric_display_name": "Response Time", "threshold": "500", "duration": "5", "admin_email": "ops@example.com"}' \
        --output "$examples_dir/critical_alert.json"
    
    # Créer un README pour les exemples
    cat > "$examples_dir/README.md" << 'EOF'
# Exemples de Configurations Tenancy

Ce répertoire contient des exemples de configurations pour le système multi-tenant.

## Fichiers disponibles

- `enterprise_tenant.json` - Configuration complète pour un tenant enterprise
- `critical_alert.json` - Exemple d'alerte critique de performance

## Utilisation

Ces exemples peuvent être utilisés comme base pour créer vos propres configurations.

### Validation

Pour valider un exemple:

```bash
python3 ../schemas/tenancy/schema_utils.py validate enterprise_tenant.json
```

### Personnalisation

Modifiez les valeurs selon vos besoins et validez la configuration avant déploiement.
EOF
    
    success "Exemples générés dans $examples_dir"
}

# Métriques et monitoring
collect_metrics() {
    info "Collecte des métriques du système..."
    
    local metrics_file="$TEMP_DIR/system_metrics.json"
    
    # Collecter les métriques système
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    local memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    local disk_usage=$(df -h . | awk 'NR==2 {print $5}' | sed 's/%//')
    
    # Métriques des schémas
    local total_schemas=$(find "$SCHEMAS_DIR" -name "*.py" -not -name "__*" | wc -l)
    local total_lines=$(find "$SCHEMAS_DIR" -name "*.py" -not -name "__*" -exec wc -l {} + | tail -1 | awk '{print $1}')
    
    cat > "$metrics_file" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "system_metrics": {
        "cpu_usage_percent": $cpu_usage,
        "memory_usage_percent": $memory_usage,
        "disk_usage_percent": $disk_usage
    },
    "schema_metrics": {
        "total_schema_files": $total_schemas,
        "total_lines_of_code": $total_lines,
        "schemas_per_category": {
            "tenant_config": 1,
            "alerts": 2,
            "warnings": 2,
            "notifications": 1,
            "monitoring": 1,
            "compliance": 1,
            "performance": 1
        }
    }
}
EOF
    
    info "Métriques collectées dans $metrics_file"
}

# Nettoyage
cleanup() {
    info "Nettoyage des fichiers temporaires..."
    
    if [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
        success "Nettoyage terminé"
    fi
}

# Gestion d'erreur
handle_error() {
    local exit_code=$?
    error "Erreur détectée (code: $exit_code)"
    cleanup
    exit $exit_code
}

# Configuration du piège d'erreur
trap handle_error ERR

# Fonction d'aide
show_help() {
    cat << 'EOF'
Script de Gestion des Schémas Tenancy
=====================================

Usage: ./schema_management.sh [COMMAND] [OPTIONS]

Commandes disponibles:
  validate      Valider tous les schémas
  docs          Générer la documentation
  test          Exécuter les tests automatisés
  benchmark     Exécuter les benchmarks de performance
  examples      Générer des exemples de configuration
  metrics       Collecter les métriques du système
  all           Exécuter toutes les opérations
  help          Afficher cette aide

Options:
  --verbose     Mode verbeux
  --dry-run     Simulation sans modification
  --output DIR  Répertoire de sortie personnalisé

Exemples:
  ./schema_management.sh validate
  ./schema_management.sh docs --output /tmp/docs
  ./schema_management.sh all --verbose

EOF
}

# Fonction principale
main() {
    local command="${1:-help}"
    local verbose=false
    local dry_run=false
    local output_dir=""
    
    # Parser les arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --verbose)
                verbose=true
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            --output)
                output_dir="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                if [[ -z "$command" || "$command" == "help" ]]; then
                    command="$1"
                fi
                shift
                ;;
        esac
    done
    
    # Configuration du logging
    if [ "$verbose" = true ]; then
        set -x
    fi
    
    # Créer le répertoire temporaire
    mkdir -p "$TEMP_DIR"
    
    # Vérifier les prérequis
    check_prerequisites
    
    # Exécuter la commande
    case "$command" in
        validate)
            validate_schemas
            ;;
        docs)
            generate_docs
            ;;
        test)
            run_tests
            ;;
        benchmark)
            run_benchmarks
            ;;
        examples)
            generate_examples
            ;;
        metrics)
            collect_metrics
            ;;
        all)
            info "Exécution complète du pipeline..."
            validate_schemas && \
            generate_docs && \
            run_tests && \
            run_benchmarks && \
            generate_examples && \
            collect_metrics
            success "Pipeline complet terminé avec succès"
            ;;
        help)
            show_help
            ;;
        *)
            error "Commande inconnue: $command"
            show_help
            exit 1
            ;;
    esac
    
    # Nettoyage final
    cleanup
}

# Point d'entrée
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
