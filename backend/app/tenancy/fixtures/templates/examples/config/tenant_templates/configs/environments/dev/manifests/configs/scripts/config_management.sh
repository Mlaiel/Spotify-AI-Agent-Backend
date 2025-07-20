#!/bin/bash
"""
Configuration Management Suite - Makefile equivalent
===================================================

Script bash pour automatiser toutes les opérations de gestion des configurations.
Combine génération, validation, déploiement et surveillance.

Author: Fahed Mlaiel
Team: Spotify AI Agent Development Team
Version: 1.0.0

Usage:
    ./config_management.sh [command] [options]
    
Commands:
    generate    - Génère les configurations
    validate    - Valide les configurations
    deploy      - Déploie les configurations
    monitor     - Surveille les configurations
    full-cycle  - Cycle complet (generate -> validate -> deploy -> monitor)
    status      - Affiche le statut actuel
    cleanup     - Nettoie les ressources
    help        - Affiche l'aide
"""

set -euo pipefail

# Configuration par défaut
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../configs"
NAMESPACE="${NAMESPACE:-spotify-ai-agent-dev}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
DRY_RUN="${DRY_RUN:-false}"

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

check_dependencies() {
    log_info "Vérification des dépendances..."
    
    local deps=("python3" "kubectl" "docker")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Dépendances manquantes: ${missing_deps[*]}"
        exit 1
    fi
    
    log_success "Toutes les dépendances sont présentes"
}

check_python_packages() {
    log_info "Vérification des packages Python..."
    
    if ! python3 -c "import yaml, pydantic, kubernetes" 2>/dev/null; then
        log_warning "Installation des packages Python requis..."
        pip3 install pyyaml pydantic kubernetes
    fi
    
    log_success "Packages Python disponibles"
}

check_cluster_access() {
    log_info "Vérification de l'accès au cluster Kubernetes..."
    
    if ! kubectl cluster-info &>/dev/null; then
        log_error "Impossible d'accéder au cluster Kubernetes"
        log_info "Assurez-vous que kubectl est configuré correctement"
        exit 1
    fi
    
    log_success "Accès au cluster Kubernetes confirmé"
}

generate_configs() {
    log_info "Génération des configurations..."
    
    local args=""
    if [ "$DRY_RUN" = "true" ]; then
        args="$args --dry-run"
    fi
    
    if [ -n "${OUTPUT_DIR:-}" ]; then
        args="$args --output-dir $OUTPUT_DIR"
    fi
    
    python3 "${SCRIPT_DIR}/generate_configs.py" \
        --environment "$ENVIRONMENT" \
        --namespace "$NAMESPACE" \
        $args
    
    if [ $? -eq 0 ]; then
        log_success "Configurations générées avec succès"
    else
        log_error "Échec de la génération des configurations"
        exit 1
    fi
}

validate_configs() {
    log_info "Validation des configurations..."
    
    local args=""
    if [ "$DRY_RUN" = "true" ]; then
        args="$args --dry-run"
    fi
    
    python3 "${SCRIPT_DIR}/validate_configs.py" \
        --config-dir "$CONFIG_DIR" \
        --environment "$ENVIRONMENT" \
        --security-compliance \
        $args
    
    if [ $? -eq 0 ]; then
        log_success "Configurations validées avec succès"
    else
        log_error "Échec de la validation des configurations"
        exit 1
    fi
}

deploy_configs() {
    log_info "Déploiement des configurations..."
    
    local args=""
    if [ "$DRY_RUN" = "true" ]; then
        args="$args --dry-run"
    else
        args="$args --apply"
    fi
    
    python3 "${SCRIPT_DIR}/deploy_configs.py" \
        --config-dir "$CONFIG_DIR" \
        --namespace "$NAMESPACE" \
        --wait-for-rollout \
        --verify \
        $args
    
    if [ $? -eq 0 ]; then
        log_success "Configurations déployées avec succès"
    else
        log_error "Échec du déploiement des configurations"
        exit 1
    fi
}

monitor_configs() {
    log_info "Surveillance des configurations..."
    
    local duration="${MONITOR_DURATION:-300}"  # 5 minutes par défaut
    
    python3 "${SCRIPT_DIR}/monitor_configs.py" \
        --namespace "$NAMESPACE" \
        --watch \
        --duration "$duration" \
        --interval 30
    
    log_success "Surveillance terminée"
}

status_check() {
    log_info "Vérification du statut actuel..."
    
    python3 "${SCRIPT_DIR}/monitor_configs.py" \
        --namespace "$NAMESPACE" \
        --one-shot
}

full_cycle() {
    log_info "Démarrage du cycle complet de gestion des configurations"
    
    # Étape 1: Génération
    log_info "=== ÉTAPE 1: GÉNÉRATION ==="
    generate_configs
    
    # Étape 2: Validation
    log_info "=== ÉTAPE 2: VALIDATION ==="
    validate_configs
    
    # Étape 3: Déploiement (si pas en dry-run)
    if [ "$DRY_RUN" != "true" ]; then
        log_info "=== ÉTAPE 3: DÉPLOIEMENT ==="
        deploy_configs
        
        # Attendre un peu avant la surveillance
        sleep 10
        
        # Étape 4: Surveillance courte pour vérifier
        log_info "=== ÉTAPE 4: VÉRIFICATION ==="
        MONITOR_DURATION=60 monitor_configs
    else
        log_info "=== MODE DRY-RUN: Déploiement et surveillance sautés ==="
    fi
    
    log_success "Cycle complet terminé avec succès!"
}

cleanup_resources() {
    log_info "Nettoyage des ressources..."
    
    # Suppression des ConfigMaps
    if kubectl get configmaps -n "$NAMESPACE" | grep -q "spotify-ai-agent"; then
        log_info "Suppression des ConfigMaps..."
        kubectl delete configmaps -n "$NAMESPACE" -l app=spotify-ai-agent
    fi
    
    # Suppression des Secrets (attention!)
    read -p "Supprimer également les Secrets? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if kubectl get secrets -n "$NAMESPACE" | grep -q "spotify-ai-agent"; then
            log_info "Suppression des Secrets..."
            kubectl delete secrets -n "$NAMESPACE" -l app=spotify-ai-agent
        fi
    fi
    
    log_success "Nettoyage terminé"
}

security_scan() {
    log_info "Lancement du scan de sécurité..."
    
    local args=""
    if [ "$DRY_RUN" = "true" ]; then
        args="$args --dry-run"
    fi
    
    python3 "${SCRIPT_DIR}/security_scanner.py" \
        --namespace "$NAMESPACE" \
        --full-scan \
        $args
    
    if [ $? -eq 0 ]; then
        log_success "Scan de sécurité terminé avec succès"
    else
        log_error "Échec du scan de sécurité"
        exit 1
    fi
}

rollback_configs() {
    log_info "Gestion des rollbacks..."
    
    local backup_id="${ROLLBACK_TARGET:-}"
    
    if [ -z "$backup_id" ]; then
        # Lister les sauvegardes disponibles
        python3 "${SCRIPT_DIR}/rollback_configs.py" \
            --namespace "$NAMESPACE" \
            --list-backups
    else
        # Exécuter le rollback
        python3 "${SCRIPT_DIR}/rollback_configs.py" \
            --namespace "$NAMESPACE" \
            --rollback \
            --target-revision "$backup_id" \
            --confirm
    fi
}

backup_restore_management() {
    log_info "Gestion des sauvegardes et restaurations..."
    
    local action="${BACKUP_ACTION:-list}"
    
    case "$action" in
        create)
            python3 "${SCRIPT_DIR}/backup_restore.py" \
                --namespace "$NAMESPACE" \
                --create-backup \
                --description "Sauvegarde automatique via script de gestion"
            ;;
        restore)
            if [ -n "${BACKUP_ID:-}" ]; then
                python3 "${SCRIPT_DIR}/backup_restore.py" \
                    --namespace "$NAMESPACE" \
                    --restore \
                    --backup-id "$BACKUP_ID"
            else
                log_error "BACKUP_ID requis pour la restauration"
                exit 1
            fi
            ;;
        *)
            python3 "${SCRIPT_DIR}/backup_restore.py" \
                --namespace "$NAMESPACE" \
                --list-backups \
                --limit 10
            ;;
    esac
}

advanced_monitoring() {
    log_info "Surveillance avancée avec métriques détaillées..."
    
    local duration="${MONITOR_DURATION:-600}"  # 10 minutes par défaut
    
    # Lancement du monitoring avec export automatique
    python3 "${SCRIPT_DIR}/monitor_configs.py" \
        --namespace "$NAMESPACE" \
        --watch \
        --duration "$duration" \
        --interval 15 &
    
    local monitor_pid=$!
    
    # Export périodique des métriques
    sleep 60
    while kill -0 $monitor_pid 2>/dev/null; do
        python3 "${SCRIPT_DIR}/monitor_configs.py" \
            --namespace "$NAMESPACE" \
            --export-metrics \
            --format prometheus \
            --output "/tmp/spotify-ai-metrics-$(date +%Y%m%d-%H%M%S).txt"
        
        sleep 120  # Export toutes les 2 minutes
    done
}

complete_security_audit() {
    log_info "Audit de sécurité complet..."
    
    # 1. Scan de sécurité
    log_info "=== ÉTAPE 1: SCAN DE SÉCURITÉ ==="
    python3 "${SCRIPT_DIR}/security_scanner.py" \
        --namespace "$NAMESPACE" \
        --full-scan \
        --export-report \
        --format json \
        --output "/tmp/security-report-$(date +%Y%m%d-%H%M%S).json"
    
    # 2. Vérification des sauvegardes
    log_info "=== ÉTAPE 2: VÉRIFICATION DES SAUVEGARDES ==="
    python3 "${SCRIPT_DIR}/backup_restore.py" \
        --namespace "$NAMESPACE" \
        --list-backups \
        --limit 5
    
    # 3. Test de rollback
    log_info "=== ÉTAPE 3: TEST DE ROLLBACK ==="
    local latest_backup=$(python3 "${SCRIPT_DIR}/backup_restore.py" --list-backups --limit 1 | grep -o "backup-[0-9-]*" | head -1)
    
    if [ -n "$latest_backup" ]; then
        python3 "${SCRIPT_DIR}/backup_restore.py" \
            --test-restore "$latest_backup"
    else
        log_warning "Aucune sauvegarde disponible pour le test"
    fi
    
    log_success "Audit de sécurité terminé"
}

disaster_recovery_test() {
    log_info "Test de reprise après sinistre..."
    
    # 1. Sauvegarde de sécurité
    log_info "Création d'une sauvegarde de sécurité..."
    python3 "${SCRIPT_DIR}/backup_restore.py" \
        --namespace "$NAMESPACE" \
        --create-backup \
        --description "Sauvegarde avant test de disaster recovery"
    
    # 2. Simulation de panne (en mode dry-run)
    log_info "Simulation de restauration complète..."
    local test_backup=$(python3 "${SCRIPT_DIR}/backup_restore.py" --list-backups --limit 1 | grep -o "backup-[0-9-]*" | head -1)
    
    if [ -n "$test_backup" ]; then
        python3 "${SCRIPT_DIR}/backup_restore.py" \
            --restore \
            --backup-id "$test_backup" \
            --dry-run
    fi
    
    # 3. Vérification de la santé après simulation
    log_info "Vérification de la santé du système..."
    python3 "${SCRIPT_DIR}/monitor_configs.py" \
        --namespace "$NAMESPACE" \
        --one-shot
    
    log_success "Test de disaster recovery terminé"
}

export_metrics() {
    log_info "Export des métriques..."
    
    local output_file="${METRICS_OUTPUT:-/tmp/spotify-ai-agent-metrics-$(date +%Y%m%d-%H%M%S).json}"
    local format="${METRICS_FORMAT:-json}"
    
    python3 "${SCRIPT_DIR}/monitor_configs.py" \
        --namespace "$NAMESPACE" \
        --export-metrics \
        --format "$format" \
        --output "$output_file"
    
    log_success "Métriques exportées vers: $output_file"
}

show_help() {
    cat << EOF
Configuration Management Suite - Spotify AI Agent
================================================

USAGE:
    $0 [COMMAND] [OPTIONS]

COMMANDS:
    generate         Génère les configurations pour l'environnement spécifié
    validate         Valide les configurations existantes
    deploy           Déploie les configurations dans Kubernetes
    monitor          Surveille l'état des configurations déployées
    status           Affiche le statut actuel du déploiement
    full-cycle       Exécute le cycle complet (generate -> validate -> deploy -> monitor)
    cleanup          Nettoie les ressources déployées
    backup           Sauvegarde les configurations actuelles (deprecated, utiliser backup-restore)
    export           Exporte les métriques de surveillance
    security-scan    Lance un scan de sécurité complet
    rollback         Gère les rollbacks de configuration
    backup-restore   Système avancé de sauvegarde et restauration
    advanced-monitor Surveillance avancée avec métriques détaillées
    security-audit   Audit de sécurité complet
    disaster-test    Test de reprise après sinistre
    help             Affiche cette aide

VARIABLES D'ENVIRONNEMENT:
    NAMESPACE          Namespace Kubernetes (défaut: spotify-ai-agent-dev)
    ENVIRONMENT        Environnement de déploiement (défaut: dev)
    DRY_RUN           Mode simulation (true/false, défaut: false)
    OUTPUT_DIR        Répertoire de sortie pour les configurations générées
    MONITOR_DURATION  Durée de surveillance en secondes (défaut: 300)
    METRICS_OUTPUT    Fichier de sortie pour les métriques
    METRICS_FORMAT    Format d'export des métriques (json/prometheus/csv)
    ROLLBACK_TARGET   ID de révision pour rollback
    BACKUP_ACTION     Action de sauvegarde (create/restore/list)
    BACKUP_ID         ID de sauvegarde pour restauration

EXEMPLES:
    # Génération et validation en mode dry-run
    DRY_RUN=true $0 generate
    DRY_RUN=true $0 validate
    
    # Cycle complet pour l'environnement de production
    ENVIRONMENT=prod NAMESPACE=spotify-ai-agent-prod $0 full-cycle
    
    # Surveillance continue
    MONITOR_DURATION=3600 $0 monitor
    
    # Export des métriques Prometheus
    METRICS_FORMAT=prometheus $0 export
    
    # Audit de sécurité complet
    $0 security-audit
    
    # Test de disaster recovery
    $0 disaster-test
    
    # Sauvegarde et rollback
    BACKUP_ACTION=create $0 backup-restore
    ROLLBACK_TARGET=5 $0 rollback
    
    # Nettoyage
    $0 cleanup

DOCUMENTATION:
    Consultez le README.md pour une documentation complète.

EOF
}

# Fonction principale
main() {
    local command="${1:-help}"
    shift || true
    
    # Vérification des dépendances de base
    check_dependencies
    
    case "$command" in
        generate)
            check_python_packages
            generate_configs
            ;;
        validate)
            check_python_packages
            validate_configs
            ;;
        deploy)
            check_python_packages
            check_cluster_access
            deploy_configs
            ;;
        monitor)
            check_python_packages
            check_cluster_access
            monitor_configs
            ;;
        status)
            check_python_packages
            check_cluster_access
            status_check
            ;;
        full-cycle)
            check_python_packages
            check_cluster_access
            full_cycle
            ;;
        cleanup)
            check_cluster_access
            cleanup_resources
            ;;
        backup)
            check_cluster_access
            backup_configs
            ;;
        export)
            check_python_packages
            check_cluster_access
            export_metrics
            ;;
        security-scan)
            check_python_packages
            check_cluster_access
            security_scan
            ;;
        rollback)
            check_python_packages
            check_cluster_access
            rollback_configs
            ;;
        backup-restore)
            check_python_packages
            check_cluster_access
            backup_restore_management
            ;;
        advanced-monitor)
            check_python_packages
            check_cluster_access
            advanced_monitoring
            ;;
        security-audit)
            check_python_packages
            check_cluster_access
            complete_security_audit
            ;;
        disaster-test)
            check_python_packages
            check_cluster_access
            disaster_recovery_test
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Commande inconnue: $command"
            show_help
            exit 1
            ;;
    esac
}

# Gestion des signaux pour arrêt propre
trap 'log_warning "Interruption reçue, arrêt..."; exit 130' INT TERM

# Exécution du script
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
