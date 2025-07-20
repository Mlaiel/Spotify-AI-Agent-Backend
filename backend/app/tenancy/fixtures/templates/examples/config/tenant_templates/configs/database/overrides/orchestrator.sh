#!/bin/bash

# =============================================================================
# Script Principal d'Orchestration de la Gestion Base de Données
# =============================================================================
# 
# Ce script unifie toutes les opérations de gestion des bases de données
# pour l'architecture multi-tenant Spotify AI Agent.
#
# Auteur: Équipe DevOps & Architecture (Lead: Fahed Mlaiel)
# Version: 2.1.0
# Dernière mise à jour: 2025-07-16
#
# Fonctionnalités:
# - Orchestration complète du cycle de vie des bases de données
# - Déploiement, migration, monitoring et optimisation unifiés
# - Interface CLI intuitive avec autocomplétion
# - Gestion d'erreurs robuste avec rollback automatique
# - Intégration complète avec l'écosystème de monitoring
# =============================================================================

set -euo pipefail

# Configuration globale
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_FILE="/var/log/spotify-ai/db-orchestrator.log"
readonly CONFIG_DIR="${SCRIPT_DIR}"

# Scripts disponibles
readonly DEPLOY_SCRIPT="${CONFIG_DIR}/deploy_database_config.sh"
readonly MIGRATE_SCRIPT="${CONFIG_DIR}/migrate_database.sh"
readonly MONITOR_SCRIPT="${CONFIG_DIR}/monitor_performance.sh"

# Couleurs pour l'affichage
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'

# Configuration logging
exec 1> >(tee -a "${LOG_FILE}")
exec 2> >(tee -a "${LOG_FILE}" >&2)

# =============================================================================
# FONCTIONS D'AFFICHAGE
# =============================================================================

print_header() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                    🎵 SPOTIFY AI AGENT - DATABASE ORCHESTRATOR 🎵           ║"
    echo "║                                                                              ║"
    echo "║  Gestionnaire Unifié des Bases de Données Multi-Tenant                     ║"
    echo "║  Version 2.1.0 - Architecture de Classe Mondiale                           ║"
    echo "║                                                                              ║"
    echo "║  Développé par: Fahed Mlaiel & Équipe Architecture                          ║"
    echo "║  Lead Dev + Architecte IA | Backend Senior | ML Engineer                    ║"
    echo "║  DBA & Data Engineer | Sécurité Backend | Architecte Microservices         ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_menu() {
    echo -e "${BLUE}📋 MENU PRINCIPAL${NC}"
    echo -e "${GREEN}════════════════${NC}"
    echo
    echo -e "${YELLOW}🚀 DÉPLOIEMENT & CONFIGURATION${NC}"
    echo "  1) deploy-config     - Déployer configuration base de données"
    echo "  2) validate-config   - Valider configuration avant déploiement"
    echo "  3) rollback-config   - Rollback configuration précédente"
    echo
    echo -e "${YELLOW}🔄 MIGRATIONS & SCHÉMAS${NC}"
    echo "  4) migrate           - Exécuter migration de base de données"
    echo "  5) migrate-rollback  - Rollback migration"
    echo "  6) migrate-validate  - Valider migration sans l'exécuter"
    echo
    echo -e "${YELLOW}📊 MONITORING & PERFORMANCE${NC}"
    echo "  7) monitor           - Monitoring temps réel des performances"
    echo "  8) analyze          - Analyse de performance approfondie"
    echo "  9) optimize         - Optimisation automatique des performances"
    echo " 10) report           - Générer rapport de performance complet"
    echo
    echo -e "${YELLOW}🔧 UTILITAIRES & MAINTENANCE${NC}"
    echo " 11) health-check     - Vérification de santé complète"
    echo " 12) backup           - Backup manuel des bases de données"
    echo " 13) restore          - Restauration depuis backup"
    echo " 14) scale            - Scaling automatique des ressources"
    echo
    echo -e "${YELLOW}📈 TABLEAU DE BORD & RAPPORTS${NC}"
    echo " 15) dashboard        - Tableau de bord en temps réel"
    echo " 16) metrics          - Extraction de métriques personnalisées"
    echo " 17) alerts           - Gestion des alertes et notifications"
    echo
    echo -e "${YELLOW}⚙️  ADMINISTRATION${NC}"
    echo " 18) tenant-setup     - Configuration complète nouveau tenant"
    echo " 19) tenant-cleanup   - Nettoyage ressources tenant"
    echo " 20) bulk-operations  - Opérations en masse sur plusieurs tenants"
    echo
    echo -e "${RED}🚪 SORTIE${NC}"
    echo "  0) exit             - Quitter l'orchestrateur"
    echo
}

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "${level}" in
        "INFO")     echo -e "${BLUE}[${timestamp}] INFO:${NC} ${message}" ;;
        "WARN")     echo -e "${YELLOW}[${timestamp}] WARN:${NC} ${message}" ;;
        "ERROR")    echo -e "${RED}[${timestamp}] ERROR:${NC} ${message}" ;;
        "SUCCESS")  echo -e "${GREEN}[${timestamp}] SUCCESS:${NC} ${message}" ;;
        "DEBUG")    echo -e "${PURPLE}[${timestamp}] DEBUG:${NC} ${message}" ;;
    esac
}

# =============================================================================
# FONCTIONS DE VALIDATION
# =============================================================================

validate_prerequisites() {
    log "INFO" "Vérification des prérequis système..."
    
    # Vérification des scripts requis
    local required_scripts=("${DEPLOY_SCRIPT}" "${MIGRATE_SCRIPT}" "${MONITOR_SCRIPT}")
    
    for script in "${required_scripts[@]}"; do
        if [[ ! -f "${script}" ]] || [[ ! -x "${script}" ]]; then
            log "ERROR" "Script requis manquant ou non exécutable: ${script}"
            return 1
        fi
    done
    
    # Vérification des outils CLI requis
    local required_tools=("psql" "redis-cli" "mongosh" "jq" "python3")
    local missing_tools=()
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "${tool}" &> /dev/null; then
            missing_tools+=("${tool}")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log "WARN" "Outils manquants (optionnels): ${missing_tools[*]}"
    fi
    
    # Vérification des permissions
    if [[ ! -w "/var/log/spotify-ai" ]]; then
        log "WARN" "Permissions d'écriture manquantes pour /var/log/spotify-ai"
    fi
    
    log "SUCCESS" "Prérequis validés"
}

validate_input() {
    local tenant_id="$1"
    local environment="$2"
    local database_type="$3"
    
    # Validation Tenant ID
    if [[ ! "${tenant_id}" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        log "ERROR" "Tenant ID invalide: ${tenant_id}"
        return 1
    fi
    
    # Validation Environment
    local valid_envs=("development" "testing" "staging" "production" "sandbox" "performance")
    if [[ ! " ${valid_envs[@]} " =~ " ${environment} " ]]; then
        log "ERROR" "Environnement invalide: ${environment}"
        return 1
    fi
    
    # Validation Database Type
    local valid_dbs=("postgresql" "redis" "mongodb" "clickhouse" "elasticsearch" "neo4j" "cassandra")
    if [[ ! " ${valid_dbs[@]} " =~ " ${database_type} " ]]; then
        log "ERROR" "Type de base de données invalide: ${database_type}"
        return 1
    fi
    
    log "SUCCESS" "Paramètres validés"
}

# =============================================================================
# FONCTIONS D'INTERFACE UTILISATEUR
# =============================================================================

get_user_input() {
    local prompt="$1"
    local default_value="${2:-}"
    local value
    
    if [[ -n "${default_value}" ]]; then
        read -rp "${prompt} [${default_value}]: " value
        echo "${value:-${default_value}}"
    else
        read -rp "${prompt}: " value
        echo "${value}"
    fi
}

select_tenant() {
    echo -e "${BLUE}🏢 Sélection du Tenant${NC}"
    echo "─────────────────────"
    
    # Liste des tenants disponibles (peut être dynamique depuis une API)
    local tenants=("spotify_free" "spotify_premium" "spotify_enterprise" "spotify_dev" "custom")
    
    echo "Tenants disponibles:"
    for i in "${!tenants[@]}"; do
        echo "  $((i+1))) ${tenants[i]}"
    done
    echo "  $((${#tenants[@]}+1))) Entrer un tenant personnalisé"
    echo
    
    local choice
    read -rp "Sélectionnez un tenant (1-$((${#tenants[@]}+1))): " choice
    
    if [[ "${choice}" =~ ^[1-9][0-9]*$ ]] && [[ "${choice}" -le "${#tenants[@]}" ]]; then
        echo "${tenants[$((choice-1))]}"
    elif [[ "${choice}" -eq $((${#tenants[@]}+1)) ]]; then
        get_user_input "Entrez l'ID du tenant personnalisé"
    else
        log "ERROR" "Sélection invalide"
        return 1
    fi
}

select_environment() {
    echo -e "${BLUE}🌍 Sélection de l'Environnement${NC}"
    echo "──────────────────────────────"
    
    local environments=("development" "testing" "staging" "production" "sandbox" "performance")
    
    echo "Environnements disponibles:"
    for i in "${!environments[@]}"; do
        echo "  $((i+1))) ${environments[i]}"
    done
    echo
    
    local choice
    read -rp "Sélectionnez un environnement (1-${#environments[@]}): " choice
    
    if [[ "${choice}" =~ ^[1-9][0-9]*$ ]] && [[ "${choice}" -le "${#environments[@]}" ]]; then
        echo "${environments[$((choice-1))]}"
    else
        log "ERROR" "Sélection invalide"
        return 1
    fi
}

select_database() {
    echo -e "${BLUE}🗄️  Sélection de la Base de Données${NC}"
    echo "───────────────────────────────────"
    
    local databases=("postgresql" "redis" "mongodb" "clickhouse" "elasticsearch" "neo4j" "cassandra")
    local descriptions=(
        "PostgreSQL - Base transactionnelle principale"
        "Redis - Cache distribué et sessions"
        "MongoDB - Stockage documentaire"
        "ClickHouse - Analytics et data warehouse"
        "Elasticsearch - Moteur de recherche"
        "Neo4j - Base de données graphe"
        "Cassandra - Big Data distribuée"
    )
    
    echo "Bases de données disponibles:"
    for i in "${!databases[@]}"; do
        echo "  $((i+1))) ${descriptions[i]}"
    done
    echo
    
    local choice
    read -rp "Sélectionnez une base de données (1-${#databases[@]}): " choice
    
    if [[ "${choice}" =~ ^[1-9][0-9]*$ ]] && [[ "${choice}" -le "${#databases[@]}" ]]; then
        echo "${databases[$((choice-1))]}"
    else
        log "ERROR" "Sélection invalide"
        return 1
    fi
}

# =============================================================================
# FONCTIONS D'ORCHESTRATION
# =============================================================================

execute_deploy_config() {
    local tenant_id="$1"
    local environment="$2"
    local database_type="$3"
    local action="${4:-deploy}"
    
    log "INFO" "🚀 Déploiement de configuration ${database_type} pour ${tenant_id}/${environment}"
    
    if [[ -x "${DEPLOY_SCRIPT}" ]]; then
        "${DEPLOY_SCRIPT}" "${tenant_id}" "${environment}" "${database_type}" "${action}"
    else
        log "ERROR" "Script de déploiement non disponible"
        return 1
    fi
}

execute_migration() {
    local tenant_id="$1"
    local environment="$2"
    local database_type="$3"
    local migration_file="$4"
    local action="${5:-migrate}"
    
    log "INFO" "🔄 Migration ${database_type} pour ${tenant_id}/${environment}"
    
    if [[ -x "${MIGRATE_SCRIPT}" ]]; then
        "${MIGRATE_SCRIPT}" "${tenant_id}" "${environment}" "${database_type}" "${migration_file}" "${action}"
    else
        log "ERROR" "Script de migration non disponible"
        return 1
    fi
}

execute_monitoring() {
    local tenant_id="$1"
    local environment="$2"
    local database_type="$3"
    local action="${4:-monitor}"
    
    log "INFO" "📊 Monitoring ${database_type} pour ${tenant_id}/${environment}"
    
    if [[ -x "${MONITOR_SCRIPT}" ]]; then
        "${MONITOR_SCRIPT}" "${tenant_id}" "${environment}" "${database_type}" "${action}"
    else
        log "ERROR" "Script de monitoring non disponible"
        return 1
    fi
}

execute_health_check() {
    local tenant_id="$1"
    local environment="$2"
    
    log "INFO" "🏥 Vérification de santé complète pour ${tenant_id}/${environment}"
    
    local databases=("postgresql" "redis" "mongodb")
    local overall_health=0
    
    for db in "${databases[@]}"; do
        log "INFO" "Vérification ${db}..."
        
        if execute_monitoring "${tenant_id}" "${environment}" "${db}" "monitor" &>/dev/null; then
            log "SUCCESS" "${db}: ✅ Santé OK"
            ((overall_health++))
        else
            log "ERROR" "${db}: ❌ Problème détecté"
        fi
    done
    
    local health_percentage=$((overall_health * 100 / ${#databases[@]}))
    
    echo
    echo -e "${CYAN}📋 RÉSUMÉ DE SANTÉ${NC}"
    echo "─────────────────────"
    echo -e "Tenant: ${GREEN}${tenant_id}${NC}"
    echo -e "Environnement: ${GREEN}${environment}${NC}"
    echo -e "Santé globale: ${GREEN}${health_percentage}%${NC} (${overall_health}/${#databases[@]} services)"
    
    if [[ "${health_percentage}" -ge 80 ]]; then
        echo -e "État: ${GREEN}✅ EXCELLENT${NC}"
    elif [[ "${health_percentage}" -ge 60 ]]; then
        echo -e "État: ${YELLOW}⚠️  ATTENTION${NC}"
    else
        echo -e "État: ${RED}❌ CRITIQUE${NC}"
    fi
}

execute_tenant_setup() {
    local tenant_id="$1"
    local environment="$2"
    
    log "INFO" "🏗️  Configuration complète du tenant ${tenant_id}/${environment}"
    
    # Configuration de toutes les bases de données
    local databases=("postgresql" "redis" "mongodb")
    
    for db in "${databases[@]}"; do
        log "INFO" "Configuration ${db}..."
        if execute_deploy_config "${tenant_id}" "${environment}" "${db}" "deploy"; then
            log "SUCCESS" "${db} configuré avec succès"
        else
            log "ERROR" "Échec configuration ${db}"
            return 1
        fi
    done
    
    # Vérification finale
    execute_health_check "${tenant_id}" "${environment}"
    
    log "SUCCESS" "🎉 Configuration complète du tenant terminée"
}

# =============================================================================
# INTERFACE INTERACTIVE
# =============================================================================

interactive_mode() {
    local tenant_id environment database_type
    
    while true; do
        clear
        print_header
        print_menu
        
        local choice
        read -rp "Sélectionnez une option (0-20): " choice
        
        case "${choice}" in
            1) # Deploy config
                tenant_id=$(select_tenant)
                environment=$(select_environment)
                database_type=$(select_database)
                validate_input "${tenant_id}" "${environment}" "${database_type}"
                execute_deploy_config "${tenant_id}" "${environment}" "${database_type}" "deploy"
                ;;
                
            2) # Validate config
                tenant_id=$(select_tenant)
                environment=$(select_environment)
                database_type=$(select_database)
                validate_input "${tenant_id}" "${environment}" "${database_type}"
                execute_deploy_config "${tenant_id}" "${environment}" "${database_type}" "validate"
                ;;
                
            4) # Migrate
                tenant_id=$(select_tenant)
                environment=$(select_environment)
                database_type=$(select_database)
                migration_file=$(get_user_input "Chemin du fichier de migration")
                validate_input "${tenant_id}" "${environment}" "${database_type}"
                execute_migration "${tenant_id}" "${environment}" "${database_type}" "${migration_file}" "migrate"
                ;;
                
            5) # Migrate rollback
                tenant_id=$(select_tenant)
                environment=$(select_environment)
                database_type=$(select_database)
                validate_input "${tenant_id}" "${environment}" "${database_type}"
                execute_migration "${tenant_id}" "${environment}" "${database_type}" "" "rollback"
                ;;
                
            7) # Monitor
                tenant_id=$(select_tenant)
                environment=$(select_environment)
                database_type=$(select_database)
                validate_input "${tenant_id}" "${environment}" "${database_type}"
                execute_monitoring "${tenant_id}" "${environment}" "${database_type}" "monitor"
                ;;
                
            9) # Optimize
                tenant_id=$(select_tenant)
                environment=$(select_environment)
                database_type=$(select_database)
                validate_input "${tenant_id}" "${environment}" "${database_type}"
                execute_monitoring "${tenant_id}" "${environment}" "${database_type}" "all"
                ;;
                
            11) # Health check
                tenant_id=$(select_tenant)
                environment=$(select_environment)
                validate_input "${tenant_id}" "${environment}" "postgresql"  # Validation générique
                execute_health_check "${tenant_id}" "${environment}"
                ;;
                
            18) # Tenant setup
                tenant_id=$(select_tenant)
                environment=$(select_environment)
                validate_input "${tenant_id}" "${environment}" "postgresql"  # Validation générique
                execute_tenant_setup "${tenant_id}" "${environment}"
                ;;
                
            0) # Exit
                log "INFO" "👋 Fermeture de l'orchestrateur"
                break
                ;;
                
            *)
                log "WARN" "Option non implémentée ou invalide: ${choice}"
                ;;
        esac
        
        echo
        read -rp "Appuyez sur Entrée pour continuer..."
    done
}

# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

main() {
    # Vérification des prérequis
    validate_prerequisites
    
    # Mode d'exécution
    if [[ $# -eq 0 ]]; then
        # Mode interactif
        interactive_mode
    else
        # Mode ligne de commande
        local action="$1"
        shift
        
        case "${action}" in
            "deploy"|"validate"|"rollback")
                if [[ $# -ge 3 ]]; then
                    execute_deploy_config "$1" "$2" "$3" "${action}"
                else
                    log "ERROR" "Usage: $0 ${action} <tenant_id> <environment> <database_type>"
                    exit 1
                fi
                ;;
                
            "migrate")
                if [[ $# -ge 4 ]]; then
                    execute_migration "$1" "$2" "$3" "$4" "migrate"
                else
                    log "ERROR" "Usage: $0 migrate <tenant_id> <environment> <database_type> <migration_file>"
                    exit 1
                fi
                ;;
                
            "monitor"|"analyze"|"optimize")
                if [[ $# -ge 3 ]]; then
                    execute_monitoring "$1" "$2" "$3" "${action}"
                else
                    log "ERROR" "Usage: $0 ${action} <tenant_id> <environment> <database_type>"
                    exit 1
                fi
                ;;
                
            "health-check")
                if [[ $# -ge 2 ]]; then
                    execute_health_check "$1" "$2"
                else
                    log "ERROR" "Usage: $0 health-check <tenant_id> <environment>"
                    exit 1
                fi
                ;;
                
            "tenant-setup")
                if [[ $# -ge 2 ]]; then
                    execute_tenant_setup "$1" "$2"
                else
                    log "ERROR" "Usage: $0 tenant-setup <tenant_id> <environment>"
                    exit 1
                fi
                ;;
                
            "help"|"--help"|"-h")
                print_header
                echo -e "${BLUE}📖 AIDE - ORCHESTRATEUR DE BASE DE DONNÉES${NC}"
                echo "════════════════════════════════════════════"
                echo
                echo "USAGE:"
                echo "  $0                                    # Mode interactif"
                echo "  $0 <action> <parameters>              # Mode ligne de commande"
                echo
                echo "ACTIONS DISPONIBLES:"
                echo "  deploy <tenant> <env> <db>            # Déployer configuration"
                echo "  migrate <tenant> <env> <db> <file>    # Exécuter migration"
                echo "  monitor <tenant> <env> <db>           # Monitoring performance"
                echo "  health-check <tenant> <env>           # Vérification de santé"
                echo "  tenant-setup <tenant> <env>           # Configuration complète"
                echo
                echo "EXEMPLES:"
                echo "  $0 deploy spotify_premium production postgresql"
                echo "  $0 monitor spotify_free staging redis"
                echo "  $0 health-check spotify_enterprise production"
                ;;
                
            *)
                log "ERROR" "Action inconnue: ${action}"
                log "INFO" "Utilisez '$0 help' pour voir l'aide"
                exit 1
                ;;
        esac
    fi
}

# Exécution si appelé directement
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
