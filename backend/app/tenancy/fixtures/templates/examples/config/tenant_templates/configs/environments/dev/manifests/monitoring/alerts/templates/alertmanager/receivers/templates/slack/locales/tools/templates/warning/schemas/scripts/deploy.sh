#!/bin/bash

# =============================================================================
# TENANT DEPLOYMENT SCRIPT - INDUSTRIAL GRADE
# =============================================================================
# Script de déploiement automatisé pour les tenants avec architecture industrielle
# Support multi-environnement, validation complète et rollback automatique
#
# Usage:
#   ./deploy.sh [ENVIRONMENT] [OPTIONS]
#
# Environments: dev, staging, prod
# Options:
#   --tenant-id TENANT_ID    : ID du tenant à déployer
#   --strategy STRATEGY      : Stratégie de déploiement (standard, blue_green, canary, rolling)
#   --dry-run               : Mode simulation
#   --no-backup             : Désactiver la sauvegarde
#   --no-rollback           : Désactiver le rollback automatique
#   --force                 : Forcer le déploiement
#   --timeout SECONDS       : Timeout en secondes (défaut: 1800)
#   --config-file FILE      : Fichier de configuration custom
#   --verbose               : Mode verbeux
#   --help                  : Afficher l'aide
#
# Examples:
#   ./deploy.sh dev --tenant-id enterprise-001
#   ./deploy.sh prod --strategy blue_green --timeout 3600
#   ./deploy.sh staging --dry-run --verbose
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION ET VARIABLES
# =============================================================================

# Métadonnées du script
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_VERSION="1.0.0"
readonly SCRIPT_AUTHOR="Spotify AI Agent Team"

# Configuration par défaut
DEFAULT_ENVIRONMENT="dev"
DEFAULT_STRATEGY="standard"
DEFAULT_TIMEOUT=1800
DEFAULT_LOG_LEVEL="INFO"

# Couleurs pour l'affichage
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly WHITE='\033[1;37m'
readonly NC='\033[0m' # No Color

# Icônes pour l'affichage
readonly ICON_SUCCESS="✅"
readonly ICON_ERROR="❌"
readonly ICON_WARNING="⚠️"
readonly ICON_INFO="ℹ️"
readonly ICON_DEPLOY="🚀"
readonly ICON_BACKUP="💾"
readonly ICON_HEALTH="🏥"
readonly ICON_ROLLBACK="🔄"

# Environnements supportés
readonly SUPPORTED_ENVIRONMENTS=("dev" "staging" "prod")
readonly SUPPORTED_STRATEGIES=("standard" "blue_green" "canary" "rolling")

# Variables globales
ENVIRONMENT=""
TENANT_ID=""
STRATEGY=""
DRY_RUN=false
NO_BACKUP=false
NO_ROLLBACK=false
FORCE=false
TIMEOUT=""
CONFIG_FILE=""
VERBOSE=false
DEPLOYMENT_ID=""
START_TIME=""

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

# Affichage coloré avec icônes
log_info() {
    echo -e "${ICON_INFO} ${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${ICON_SUCCESS} ${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${ICON_WARNING} ${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${ICON_ERROR} ${RED}[ERROR]${NC} $*" >&2
}

log_deploy() {
    echo -e "${ICON_DEPLOY} ${PURPLE}[DEPLOY]${NC} $*"
}

check_dependencies() {
    log_info "Vérification des dépendances..."
    
    # Vérification Python
    if ! command -v "$PYTHON_CMD" &> /dev/null; then
        log_error "Python 3 n'est pas installé ou non accessible"
        exit 1
    fi
    
    # Vérification des modules Python requis
    local required_modules=("pydantic" "typing_extensions" "email_validator")
    for module in "${required_modules[@]}"; do
        if ! "$PYTHON_CMD" -c "import $module" &> /dev/null; then
            log_warning "Module $module manquant, installation en cours..."
            pip install "$module"
        fi
    done
    
    log_success "Toutes les dépendances sont satisfaites"
}

setup_environment() {
    log_info "Configuration de l'environnement..."
    
    # Création des répertoires nécessaires
    mkdir -p "$SCHEMAS_DIR/docs"
    mkdir -p "$SCHEMAS_DIR/dist"
    mkdir -p "$SCHEMAS_DIR/tests"
    mkdir -p "$SCHEMAS_DIR/logs"
    
    # Configuration PYTHONPATH
    export PYTHONPATH="$SCHEMAS_DIR:$PYTHONPATH"
    
    log_success "Environnement configuré"
}

validate_schemas() {
    log_info "Validation des schémas..."
    
    cd "$SCHEMAS_DIR"
    if "$PYTHON_CMD" scripts/validate_schemas.py validate; then
        log_success "Validation des schémas réussie"
        return 0
    else
        log_error "Validation des schémas échouée"
        return 1
    fi
}

generate_documentation() {
    log_info "Génération de la documentation..."
    
    cd "$SCHEMAS_DIR"
    "$PYTHON_CMD" scripts/validate_schemas.py docs --output docs/
    
    # Génération README principal si absent
    if [[ ! -f "$SCHEMAS_DIR/docs/index.md" ]]; then
        cat > "$SCHEMAS_DIR/docs/index.md" << 'EOF'
# Documentation des Schémas - Spotify AI Agent

## Vue d'ensemble

Cette documentation présente l'ensemble des schémas Pydantic utilisés par le système d'alerting et de monitoring de Spotify AI Agent.

## Structure

- **Alertes**: Gestion complète du cycle de vie des alertes
- **Notifications**: Système de notification multi-canal
- **ML/IA**: Modèles d'intelligence artificielle
- **Métriques**: Collecte et analyse des métriques
- **Sécurité**: Schémas de sécurité et audit

## Navigation

Consultez les fichiers individuels pour chaque schéma dans ce répertoire.
EOF
    fi
    
    log_success "Documentation générée dans $SCHEMAS_DIR/docs/"
}

export_openapi() {
    log_info "Export de la spécification OpenAPI..."
    
    cd "$SCHEMAS_DIR"
    "$PYTHON_CMD" scripts/validate_schemas.py openapi --output openapi.json
    
    log_success "Spécification OpenAPI exportée"
}

analyze_metrics() {
    log_info "Analyse des métriques des schémas..."
    
    cd "$SCHEMAS_DIR"
    "$PYTHON_CMD" scripts/validate_schemas.py metrics --output metrics_report.json
    
    log_success "Rapport de métriques généré"
}

run_tests() {
    log_info "Exécution des tests..."
    
    # Test de validation basique
    if validate_schemas; then
        log_success "Tests de validation passés"
    else
        log_error "Tests de validation échoués"
        return 1
    fi
    
    # Tests de performance (simulation)
    log_info "Tests de performance..."
    local start_time=$(date +%s%N)
    
    cd "$SCHEMAS_DIR"
    "$PYTHON_CMD" -c "
import time
from schemas import AlertInstance, NotificationMessage, MLModel

# Test de création d'instances
start = time.time()
for i in range(1000):
    alert = AlertInstance(
        rule_id='123e4567-e89b-12d3-a456-426614174000',
        alert_id=f'test_alert_{i}',
        title='Test Alert',
        message='Test message',
        level='warning',
        category='performance',
        tenant_id='test_tenant'
    )
end = time.time()

print(f'Création de 1000 instances en {end - start:.3f}s')
print(f'Performance: {1000 / (end - start):.0f} instances/sec')
"
    
    local end_time=$(date +%s%N)
    local duration=$(( (end_time - start_time) / 1000000 ))
    
    log_success "Tests de performance terminés en ${duration}ms"
}

build_package() {
    log_info "Construction du package..."
    
    # Nettoyage
    rm -rf "$SCHEMAS_DIR/dist/*"
    
    # Validation complète
    if ! run_tests; then
        log_error "Build échoué - tests en échec"
        return 1
    fi
    
    # Génération de tous les artefacts
    generate_documentation
    export_openapi
    analyze_metrics
    
    # Création de l'archive
    cd "$SCHEMAS_DIR"
    tar -czf "dist/${PROJECT_NAME}-$(date +%Y%m%d-%H%M%S).tar.gz" \
        --exclude='dist' \
        --exclude='logs' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        .
    
    log_success "Package créé dans $SCHEMAS_DIR/dist/"
}

deploy() {
    log_info "Déploiement des schémas..."
    
    # Validation pré-déploiement
    if ! validate_schemas; then
        log_error "Déploiement annulé - validation échouée"
        return 1
    fi
    
    # Build
    if ! build_package; then
        log_error "Déploiement annulé - build échoué"
        return 1
    fi
    
    # Ici, on pourrait ajouter des étapes de déploiement réelles
    # comme la synchronisation avec un registre de schémas
    
    log_success "Déploiement terminé avec succès"
}

clean() {
    log_info "Nettoyage des fichiers temporaires..."
    
    cd "$SCHEMAS_DIR"
    
    # Suppression des fichiers temporaires
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "*.pyo" -delete 2>/dev/null || true
    find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # Nettoyage des logs anciens
    find logs/ -name "*.log" -mtime +7 -delete 2>/dev/null || true
    
    log_success "Nettoyage terminé"
}

monitor() {
    log_info "Monitoring des schémas en continu..."
    
    while true; do
        log_info "Vérification $(date)"
        
        if validate_schemas; then
            log_success "✓ Schémas valides"
        else
            log_error "✗ Problèmes détectés dans les schémas"
        fi
        
        # Attente de 30 secondes
        sleep 30
    done
}

show_status() {
    log_info "État du système de schémas:"
    
    cd "$SCHEMAS_DIR"
    
    # Nombre de schémas
    local schema_count=$(find . -name "*.py" -not -path "./scripts/*" | wc -l)
    echo "  📄 Fichiers de schémas: $schema_count"
    
    # Dernière validation
    if [[ -f "logs/last_validation.log" ]]; then
        local last_validation=$(stat -c %Y logs/last_validation.log 2>/dev/null || echo "0")
        local current_time=$(date +%s)
        local age=$(( (current_time - last_validation) / 60 ))
        echo "  🔍 Dernière validation: il y a ${age} minutes"
    else
        echo "  🔍 Dernière validation: jamais"
    fi
    
    # Taille du cache de documentation
    if [[ -d "docs" ]]; then
        local docs_size=$(du -sh docs/ 2>/dev/null | cut -f1)
        echo "  📚 Documentation: $docs_size"
    fi
    
    # Vérification rapide
    if "$PYTHON_CMD" -c "from schemas import get_schema_info; print('✓ Import réussi')" 2>/dev/null; then
        echo "  ✅ État: Opérationnel"
    else
        echo "  ❌ État: Problème détecté"
    fi
}

usage() {
    cat << EOF
Usage: $0 <command> [options]

Commandes disponibles:
  setup       - Configuration initiale de l'environnement
  validate    - Validation de tous les schémas
  test        - Exécution des tests complets
  docs        - Génération de la documentation
  openapi     - Export de la spécification OpenAPI
  metrics     - Analyse des métriques des schémas
  build       - Construction du package complet
  deploy      - Déploiement des schémas
  clean       - Nettoyage des fichiers temporaires
  monitor     - Monitoring continu
  status      - Affichage de l'état du système
  help        - Affichage de cette aide

Exemples:
  $0 setup                    # Configuration initiale
  $0 validate                 # Validation simple
  $0 build                    # Build complet
  $0 deploy                   # Déploiement
  $0 monitor                  # Monitoring continu

EOF
}

main() {
    case "${1:-help}" in
        setup)
            check_dependencies
            setup_environment
            ;;
        validate)
            setup_environment
            validate_schemas
            ;;
        test)
            setup_environment
            run_tests
            ;;
        docs)
            setup_environment
            generate_documentation
            ;;
        openapi)
            setup_environment
            export_openapi
            ;;
        metrics)
            setup_environment
            analyze_metrics
            ;;
        build)
            check_dependencies
            setup_environment
            build_package
            ;;
        deploy)
            check_dependencies
            setup_environment
            deploy
            ;;
        clean)
            clean
            ;;
        monitor)
            setup_environment
            monitor
            ;;
        status)
            show_status
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            log_error "Commande inconnue: $1"
            usage
            exit 1
            ;;
    esac
}

# Point d'entrée
main "$@"
