#!/bin/bash
# Script de gestion des templates Slack pour Alertmanager
# Auteur: Fahed Mlaiel
# Date: 2025-07-19

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
TOOLS_DIR="$SCRIPT_DIR"
TEMPLATES_DIR="$BASE_DIR/templates"
LOCALES_DIR="$BASE_DIR/locales"
GENERATED_DIR="$BASE_DIR/generated"
CONFIG_FILE="$BASE_DIR/config/deployment.yaml"

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

show_usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

COMMANDES:
    generate        Génère les templates Slack
    validate        Valide les templates existants
    deploy          Déploie les templates vers Alertmanager
    rollback        Effectue un rollback
    clean           Nettoie les fichiers générés
    setup           Configure l'environnement initial
    status          Affiche le statut des déploiements
    help            Affiche cette aide

OPTIONS:
    --locale LOCALE         Locale pour l'internationalisation (en, fr)
    --tenant-id TENANT      ID du tenant spécifique
    --dry-run              Mode simulation
    --force                Force l'opération
    --config FILE          Fichier de configuration alternatif
    --output-dir DIR       Répertoire de sortie
    --backup-dir DIR       Répertoire de sauvegarde

EXEMPLES:
    $0 generate --locale fr
    $0 validate --tenant-id prod-tenant-1
    $0 deploy --dry-run
    $0 rollback --backup-dir /path/to/backups

EOF
}

check_dependencies() {
    log_info "Vérification des dépendances..."
    
    local deps=("python3" "kubectl" "yq")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Dépendances manquantes: ${missing_deps[*]}"
        log_info "Installez les dépendances manquantes et relancez le script"
        return 1
    fi
    
    # Vérifier les modules Python
    if ! python3 -c "import yaml, jinja2, requests" 2>/dev/null; then
        log_error "Modules Python manquants (yaml, jinja2, requests)"
        log_info "Installez avec: pip install PyYAML Jinja2 requests"
        return 1
    fi
    
    log_success "Toutes les dépendances sont disponibles"
    return 0
}

setup_environment() {
    log_info "Configuration de l'environnement..."
    
    # Créer les répertoires nécessaires
    mkdir -p "$GENERATED_DIR"
    mkdir -p "$BASE_DIR/config"
    mkdir -p "$BASE_DIR/backups"
    mkdir -p "$BASE_DIR/logs"
    
    # Créer un fichier de configuration par défaut si inexistant
    if [ ! -f "$CONFIG_FILE" ]; then
        log_info "Création du fichier de configuration par défaut..."
        cat > "$CONFIG_FILE" << 'EOF'
# Configuration de déploiement des templates Slack
# Auteur: Fahed Mlaiel

tenants:
  - id: "dev-tenant"
    name: "Development Environment"
    environment: "dev"
    region: "eu-west-1"
    cluster: "dev-cluster"
    
    # Configuration Slack
    slack_channel: "#alerts-dev"
    slack_webhook_url: "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    
    # Configuration Alertmanager
    alertmanager_url: "http://alertmanager.monitoring.svc.cluster.local:9093"
    
    # Configuration de déploiement
    deployment_mode: "kubernetes"  # ou "file"
    kubernetes_namespace: "monitoring"
    
    # Politiques d'escalade
    escalation_policy: "dev-team"
    
  - id: "prod-tenant"
    name: "Production Environment"
    environment: "prod"
    region: "eu-west-1"
    cluster: "prod-cluster"
    
    slack_channel: "#alerts-prod"
    slack_webhook_url: "https://hooks.slack.com/services/YOUR/PROD/WEBHOOK/URL"
    
    alertmanager_url: "https://alertmanager.prod.spotify-ai-agent.com"
    
    deployment_mode: "kubernetes"
    kubernetes_namespace: "monitoring"
    
    escalation_policy: "prod-team"

# Configuration globale
global:
  default_locale: "en"
  backup_retention_days: 30
  validation_strict: false
  
  # Limites Slack
  slack_limits:
    message_max_length: 4000
    attachments_max_count: 20
    
  # Templates par défaut
  default_templates:
    - "critical"
    - "warning"
    - "info"
    - "resolved"
EOF
        log_success "Fichier de configuration créé: $CONFIG_FILE"
        log_warning "Pensez à modifier les URLs et tokens dans $CONFIG_FILE"
    fi
    
    log_success "Environnement configuré"
}

generate_templates() {
    local locale="${1:-en}"
    local tenant_id="${2:-}"
    
    log_info "Génération des templates Slack (locale: $locale)..."
    
    local cmd_args=(
        "$TOOLS_DIR/generator.py"
        "--base-path" "$BASE_DIR"
        "--locale" "$locale"
        "--tenant-config" "$CONFIG_FILE"
    )
    
    if [ -n "$tenant_id" ]; then
        # Filtrer pour un tenant spécifique
        log_info "Génération pour le tenant: $tenant_id"
    fi
    
    if python3 "${cmd_args[@]}"; then
        log_success "Templates générés avec succès"
        
        # Afficher un résumé
        local generated_files
        generated_files=$(find "$GENERATED_DIR" -name "*.yaml" | wc -l)
        log_info "Fichiers générés: $generated_files"
        
        return 0
    else
        log_error "Échec de la génération des templates"
        return 1
    fi
}

validate_templates() {
    local tenant_id="${1:-}"
    
    log_info "Validation des templates..."
    
    local cmd_args=(
        "$TOOLS_DIR/validator.py"
        "--templates-dir" "$BASE_DIR"
        "--format" "text"
    )
    
    # Mode strict en production
    if [[ "$tenant_id" == *"prod"* ]]; then
        cmd_args+=("--strict")
        log_info "Mode strict activé pour la production"
    fi
    
    local report_file="$BASE_DIR/logs/validation_report_$(date +%Y%m%d_%H%M%S).txt"
    cmd_args+=("--output-report" "$report_file")
    
    if python3 "${cmd_args[@]}"; then
        log_success "Validation réussie"
        log_info "Rapport sauvegardé: $report_file"
        return 0
    else
        log_error "Échec de la validation"
        log_info "Consultez le rapport: $report_file"
        return 1
    fi
}

deploy_templates() {
    local dry_run="${1:-false}"
    local tenant_id="${2:-}"
    
    if [ "$dry_run" = "true" ]; then
        log_info "Déploiement en mode simulation..."
    else
        log_info "Déploiement des templates..."
    fi
    
    local cmd_args=(
        "$TOOLS_DIR/deployer.py"
        "--config" "$CONFIG_FILE"
        "--templates-dir" "$GENERATED_DIR"
    )
    
    if [ "$dry_run" = "true" ]; then
        cmd_args+=("--dry-run")
    fi
    
    if python3 "${cmd_args[@]}"; then
        log_success "Déploiement terminé"
        return 0
    else
        log_error "Échec du déploiement"
        return 1
    fi
}

rollback_deployment() {
    local backup_file="${1:-}"
    
    if [ -z "$backup_file" ]; then
        log_error "Fichier de sauvegarde requis pour le rollback"
        return 1
    fi
    
    if [ ! -f "$backup_file" ]; then
        log_error "Fichier de sauvegarde non trouvé: $backup_file"
        return 1
    fi
    
    log_warning "Rollback vers: $backup_file"
    
    local cmd_args=(
        "$TOOLS_DIR/deployer.py"
        "--config" "$CONFIG_FILE"
        "--templates-dir" "$GENERATED_DIR"
        "--rollback" "$backup_file"
    )
    
    if python3 "${cmd_args[@]}"; then
        log_success "Rollback terminé"
        return 0
    else
        log_error "Échec du rollback"
        return 1
    fi
}

clean_generated() {
    log_info "Nettoyage des fichiers générés..."
    
    if [ -d "$GENERATED_DIR" ]; then
        rm -rf "${GENERATED_DIR:?}"/*
        log_success "Fichiers générés supprimés"
    fi
    
    # Nettoyer les anciens logs (> 7 jours)
    if [ -d "$BASE_DIR/logs" ]; then
        find "$BASE_DIR/logs" -name "*.txt" -mtime +7 -delete 2>/dev/null || true
        log_info "Anciens logs nettoyés"
    fi
}

show_deployment_status() {
    log_info "Statut des déploiements..."
    
    local cmd_args=(
        "$TOOLS_DIR/deployer.py"
        "--config" "$CONFIG_FILE"
        "--templates-dir" "$GENERATED_DIR"
        "--validate-only"
    )
    
    if python3 "${cmd_args[@]}"; then
        log_success "Tous les services sont opérationnels"
    else
        log_warning "Certains services sont inaccessibles"
    fi
    
    # Afficher des statistiques
    if [ -d "$GENERATED_DIR" ]; then
        local template_count
        template_count=$(find "$GENERATED_DIR" -name "*.yaml" | wc -l)
        log_info "Templates générés: $template_count"
    fi
    
    if [ -d "$BASE_DIR/backups" ]; then
        local backup_count
        backup_count=$(find "$BASE_DIR/backups" -name "*.yaml" | wc -l)
        log_info "Sauvegardes disponibles: $backup_count"
    fi
}

# Analyse des arguments
COMMAND=""
LOCALE="en"
TENANT_ID=""
DRY_RUN="false"
FORCE="false"
CUSTOM_CONFIG=""
OUTPUT_DIR=""
BACKUP_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        generate|validate|deploy|rollback|clean|setup|status|help)
            COMMAND="$1"
            shift
            ;;
        --locale)
            LOCALE="$2"
            shift 2
            ;;
        --tenant-id)
            TENANT_ID="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --force)
            FORCE="true"
            shift
            ;;
        --config)
            CUSTOM_CONFIG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --backup-dir)
            BACKUP_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            log_error "Option inconnue: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Utiliser la configuration personnalisée si fournie
if [ -n "$CUSTOM_CONFIG" ]; then
    CONFIG_FILE="$CUSTOM_CONFIG"
fi

# Modifier les répertoires si fournis
if [ -n "$OUTPUT_DIR" ]; then
    GENERATED_DIR="$OUTPUT_DIR"
fi

if [ -n "$BACKUP_DIR" ]; then
    mkdir -p "$BACKUP_DIR"
fi

# Validation de la commande
if [ -z "$COMMAND" ]; then
    log_error "Commande requise"
    show_usage
    exit 1
fi

# Vérifier les dépendances (sauf pour help et setup)
if [[ "$COMMAND" != "help" && "$COMMAND" != "setup" ]]; then
    if ! check_dependencies; then
        exit 1
    fi
fi

# Exécuter la commande
case "$COMMAND" in
    setup)
        setup_environment
        ;;
    generate)
        generate_templates "$LOCALE" "$TENANT_ID"
        ;;
    validate)
        validate_templates "$TENANT_ID"
        ;;
    deploy)
        deploy_templates "$DRY_RUN" "$TENANT_ID"
        ;;
    rollback)
        if [ -n "$BACKUP_DIR" ]; then
            # Utiliser la sauvegarde la plus récente
            BACKUP_FILE=$(find "$BACKUP_DIR" -name "*.yaml" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
            rollback_deployment "$BACKUP_FILE"
        else
            log_error "Répertoire de sauvegarde requis pour le rollback"
            exit 1
        fi
        ;;
    clean)
        if [ "$FORCE" = "true" ]; then
            clean_generated
        else
            read -p "Êtes-vous sûr de vouloir supprimer tous les fichiers générés ? (y/N) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                clean_generated
            else
                log_info "Nettoyage annulé"
            fi
        fi
        ;;
    status)
        show_deployment_status
        ;;
    help)
        show_usage
        ;;
    *)
        log_error "Commande inconnue: $COMMAND"
        show_usage
        exit 1
        ;;
esac

log_success "Commande '$COMMAND' terminée"
