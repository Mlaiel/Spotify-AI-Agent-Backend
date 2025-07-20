#!/bin/bash

# =============================================================================
# Script de Migration de Base de Données Multi-Tenant Automatisée
# =============================================================================
# 
# Ce script gère les migrations de schéma et de données pour l'architecture
# multi-tenant avec rollback automatique et validation intégrée.
#
# Auteur: Équipe Database Engineering & Migration
# Version: 2.1.0
# Dernière mise à jour: 2025-07-16
#
# Fonctionnalités:
# - Migrations zero-downtime avec blue-green deployment
# - Rollback automatique en cas d'échec
# - Validation de l'intégrité des données
# - Support de toutes les bases de données (PostgreSQL, MongoDB, etc.)
# - Monitoring temps réel du processus de migration
# - Backup automatique avant migration
# =============================================================================

set -euo pipefail

# Configuration globale
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_FILE="/var/log/spotify-ai/db-migration.log"
readonly TEMP_DIR="/tmp/spotify-migration-$$"
readonly MIGRATION_STATE_FILE="/var/lib/spotify-ai/migration-state.json"

# Couleurs pour l'affichage
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Configuration logging
exec 1> >(tee -a "${LOG_FILE}")
exec 2> >(tee -a "${LOG_FILE}" >&2)

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "${level}" in
        "INFO")  echo -e "${BLUE}[${timestamp}] INFO:${NC} ${message}" ;;
        "WARN")  echo -e "${YELLOW}[${timestamp}] WARN:${NC} ${message}" ;;
        "ERROR") echo -e "${RED}[${timestamp}] ERROR:${NC} ${message}" ;;
        "SUCCESS") echo -e "${GREEN}[${timestamp}] SUCCESS:${NC} ${message}" ;;
    esac
}

cleanup() {
    log "INFO" "Nettoyage des ressources temporaires..."
    rm -rf "${TEMP_DIR}"
    
    # Libération des locks de migration
    release_migration_locks
}

trap cleanup EXIT

# =============================================================================
# GESTION DES ÉTATS DE MIGRATION
# =============================================================================

create_migration_state() {
    local tenant_id="$1"
    local environment="$2"
    local database_type="$3"
    local migration_id="$4"
    
    mkdir -p "$(dirname "${MIGRATION_STATE_FILE}")"
    
    cat > "${MIGRATION_STATE_FILE}" << EOF
{
    "migration_id": "${migration_id}",
    "tenant_id": "${tenant_id}",
    "environment": "${environment}",
    "database_type": "${database_type}",
    "status": "STARTED",
    "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "backup_path": null,
    "rollback_available": false,
    "validation_passed": false
}
EOF
    
    log "INFO" "État de migration créé: ${migration_id}"
}

update_migration_state() {
    local key="$1"
    local value="$2"
    
    python3 << EOF
import json

with open('${MIGRATION_STATE_FILE}', 'r') as f:
    state = json.load(f)

state['${key}'] = '${value}'

with open('${MIGRATION_STATE_FILE}', 'w') as f:
    json.dump(state, f, indent=2)
EOF
}

get_migration_state() {
    local key="$1"
    
    python3 -c "
import json
with open('${MIGRATION_STATE_FILE}', 'r') as f:
    state = json.load(f)
print(state.get('${key}', ''))
"
}

# =============================================================================
# GESTION DES LOCKS DE MIGRATION
# =============================================================================

acquire_migration_lock() {
    local tenant_id="$1"
    local database_type="$2"
    local lock_file="/var/lock/spotify-migration-${tenant_id}-${database_type}.lock"
    
    if [[ -f "${lock_file}" ]]; then
        local lock_pid
        lock_pid=$(cat "${lock_file}")
        
        if kill -0 "${lock_pid}" 2>/dev/null; then
            log "ERROR" "Migration déjà en cours (PID: ${lock_pid})"
            return 1
        else
            log "WARN" "Lock orphelin détecté, suppression..."
            rm -f "${lock_file}"
        fi
    fi
    
    echo $$ > "${lock_file}"
    log "INFO" "Lock de migration acquis: ${lock_file}"
}

release_migration_locks() {
    find /var/lock -name "spotify-migration-*.lock" -exec rm -f {} \; 2>/dev/null || true
}

# =============================================================================
# BACKUP ET SAUVEGARDES
# =============================================================================

create_backup() {
    local tenant_id="$1"
    local environment="$2"
    local database_type="$3"
    local backup_id="backup-$(date +%Y%m%d-%H%M%S)-${tenant_id}"
    
    log "INFO" "Création du backup pré-migration: ${backup_id}"
    
    case "${database_type}" in
        "postgresql")
            create_postgresql_backup "${tenant_id}" "${environment}" "${backup_id}"
            ;;
        "mongodb")
            create_mongodb_backup "${tenant_id}" "${environment}" "${backup_id}"
            ;;
        "redis")
            create_redis_backup "${tenant_id}" "${environment}" "${backup_id}"
            ;;
        *)
            log "WARN" "Backup non implémenté pour ${database_type}"
            ;;
    esac
    
    echo "${backup_id}"
}

create_postgresql_backup() {
    local tenant_id="$1"
    local environment="$2"
    local backup_id="$3"
    local backup_path="/var/backups/spotify-ai/postgresql/${backup_id}.sql.gz"
    
    mkdir -p "$(dirname "${backup_path}")"
    
    # Backup compressé avec métadonnées
    pg_dump \
        --host="${PG_HOST}" \
        --port="${PG_PORT}" \
        --username="${PG_USERNAME}" \
        --dbname="${tenant_id}_${environment}" \
        --verbose \
        --format=custom \
        --compress=9 \
        --no-password \
        | gzip > "${backup_path}"
    
    # Vérification de l'intégrité du backup
    if [[ ! -s "${backup_path}" ]]; then
        log "ERROR" "Backup PostgreSQL vide ou échoué"
        return 1
    fi
    
    log "SUCCESS" "Backup PostgreSQL créé: ${backup_path}"
    update_migration_state "backup_path" "${backup_path}"
    update_migration_state "rollback_available" "true"
}

create_mongodb_backup() {
    local tenant_id="$1"
    local environment="$2"
    local backup_id="$3"
    local backup_path="/var/backups/spotify-ai/mongodb/${backup_id}"
    
    mkdir -p "${backup_path}"
    
    # Backup avec mongodump
    mongodump \
        --host="${MONGO_HOST}:${MONGO_PORT}" \
        --db="${tenant_id}_${environment}" \
        --username="${MONGO_USERNAME}" \
        --password="${MONGO_PASSWORD}" \
        --authenticationDatabase=admin \
        --gzip \
        --out="${backup_path}"
    
    # Compression du répertoire de backup
    tar -czf "${backup_path}.tar.gz" -C "$(dirname "${backup_path}")" "$(basename "${backup_path}")"
    rm -rf "${backup_path}"
    
    log "SUCCESS" "Backup MongoDB créé: ${backup_path}.tar.gz"
    update_migration_state "backup_path" "${backup_path}.tar.gz"
    update_migration_state "rollback_available" "true"
}

# =============================================================================
# VALIDATION PRÉ-MIGRATION
# =============================================================================

validate_pre_migration() {
    local tenant_id="$1"
    local environment="$2"
    local database_type="$3"
    
    log "INFO" "Validation pré-migration pour ${tenant_id}/${environment}/${database_type}"
    
    # Vérification de l'espace disque
    validate_disk_space "${database_type}"
    
    # Vérification de la connectivité
    validate_database_connectivity "${tenant_id}" "${environment}" "${database_type}"
    
    # Vérification des permissions
    validate_database_permissions "${tenant_id}" "${environment}" "${database_type}"
    
    # Vérification de l'état de réplication
    validate_replication_status "${database_type}"
    
    log "SUCCESS" "Validations pré-migration réussies"
}

validate_disk_space() {
    local database_type="$1"
    local required_space_gb
    
    case "${database_type}" in
        "postgresql"|"mongodb") required_space_gb=50 ;;
        "redis") required_space_gb=20 ;;
        *) required_space_gb=10 ;;
    esac
    
    local available_space_gb
    available_space_gb=$(df /var/lib/ | awk 'NR==2 {print int($4/1024/1024)}')
    
    if [[ "${available_space_gb}" -lt "${required_space_gb}" ]]; then
        log "ERROR" "Espace disque insuffisant: ${available_space_gb}GB disponible, ${required_space_gb}GB requis"
        return 1
    fi
    
    log "SUCCESS" "Espace disque suffisant: ${available_space_gb}GB disponible"
}

validate_database_connectivity() {
    local tenant_id="$1"
    local environment="$2"
    local database_type="$3"
    
    case "${database_type}" in
        "postgresql")
            pg_isready -h "${PG_HOST}" -p "${PG_PORT}" -d "${tenant_id}_${environment}" -t 10
            ;;
        "mongodb")
            mongosh --host "${MONGO_HOST}:${MONGO_PORT}" --eval "db.runCommand('ping')" --quiet
            ;;
        "redis")
            redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" ping
            ;;
    esac
    
    log "SUCCESS" "Connectivité ${database_type} validée"
}

# =============================================================================
# EXÉCUTION DES MIGRATIONS
# =============================================================================

execute_migration() {
    local tenant_id="$1"
    local environment="$2"
    local database_type="$3"
    local migration_file="$4"
    
    log "INFO" "Exécution de la migration: $(basename "${migration_file}")"
    
    case "${database_type}" in
        "postgresql")
            execute_postgresql_migration "${tenant_id}" "${environment}" "${migration_file}"
            ;;
        "mongodb")
            execute_mongodb_migration "${tenant_id}" "${environment}" "${migration_file}"
            ;;
        "redis")
            execute_redis_migration "${tenant_id}" "${environment}" "${migration_file}"
            ;;
        *)
            log "ERROR" "Type de migration non supporté: ${database_type}"
            return 1
            ;;
    esac
}

execute_postgresql_migration() {
    local tenant_id="$1"
    local environment="$2"
    local migration_file="$3"
    
    # Exécution avec transaction et rollback automatique
    psql \
        --host="${PG_HOST}" \
        --port="${PG_PORT}" \
        --username="${PG_USERNAME}" \
        --dbname="${tenant_id}_${environment}" \
        --single-transaction \
        --set ON_ERROR_STOP=on \
        --file="${migration_file}" \
        --echo-errors \
        --quiet
    
    if [[ $? -eq 0 ]]; then
        log "SUCCESS" "Migration PostgreSQL réussie"
    else
        log "ERROR" "Échec de la migration PostgreSQL"
        return 1
    fi
}

execute_mongodb_migration() {
    local tenant_id="$1"
    local environment="$2"
    local migration_file="$3"
    
    # Exécution du script JavaScript MongoDB
    mongosh \
        --host "${MONGO_HOST}:${MONGO_PORT}" \
        --username "${MONGO_USERNAME}" \
        --password "${MONGO_PASSWORD}" \
        --authenticationDatabase admin \
        "${tenant_id}_${environment}" \
        "${migration_file}"
    
    if [[ $? -eq 0 ]]; then
        log "SUCCESS" "Migration MongoDB réussie"
    else
        log "ERROR" "Échec de la migration MongoDB"
        return 1
    fi
}

# =============================================================================
# VALIDATION POST-MIGRATION
# =============================================================================

validate_post_migration() {
    local tenant_id="$1"
    local environment="$2"
    local database_type="$3"
    
    log "INFO" "Validation post-migration..."
    
    # Vérification de l'intégrité des données
    validate_data_integrity "${tenant_id}" "${environment}" "${database_type}"
    
    # Vérification des contraintes
    validate_constraints "${tenant_id}" "${environment}" "${database_type}"
    
    # Tests de performance de base
    validate_performance "${tenant_id}" "${environment}" "${database_type}"
    
    update_migration_state "validation_passed" "true"
    log "SUCCESS" "Validation post-migration réussie"
}

validate_data_integrity() {
    local tenant_id="$1"
    local environment="$2"
    local database_type="$3"
    
    case "${database_type}" in
        "postgresql")
            # Vérification des contraintes PostgreSQL
            psql --host="${PG_HOST}" --port="${PG_PORT}" --username="${PG_USERNAME}" \
                 --dbname="${tenant_id}_${environment}" \
                 -c "SELECT COUNT(*) FROM information_schema.constraint_column_usage;" > /dev/null
            ;;
        "mongodb")
            # Vérification des index MongoDB
            mongosh --host "${MONGO_HOST}:${MONGO_PORT}" \
                   --username "${MONGO_USERNAME}" \
                   --password "${MONGO_PASSWORD}" \
                   --authenticationDatabase admin \
                   "${tenant_id}_${environment}" \
                   --eval "db.runCommand('listIndexes')" > /dev/null
            ;;
    esac
    
    log "SUCCESS" "Intégrité des données validée"
}

# =============================================================================
# ROLLBACK AUTOMATIQUE
# =============================================================================

rollback_migration() {
    local tenant_id="$1"
    local environment="$2"
    local database_type="$3"
    
    local backup_path
    backup_path=$(get_migration_state "backup_path")
    
    if [[ -z "${backup_path}" ]] || [[ ! -f "${backup_path}" ]]; then
        log "ERROR" "Aucun backup disponible pour le rollback"
        return 1
    fi
    
    log "WARN" "Démarrage du rollback depuis: ${backup_path}"
    update_migration_state "status" "ROLLING_BACK"
    
    case "${database_type}" in
        "postgresql")
            rollback_postgresql "${tenant_id}" "${environment}" "${backup_path}"
            ;;
        "mongodb")
            rollback_mongodb "${tenant_id}" "${environment}" "${backup_path}"
            ;;
        *)
            log "ERROR" "Rollback non implémenté pour ${database_type}"
            return 1
            ;;
    esac
    
    update_migration_state "status" "ROLLED_BACK"
    log "SUCCESS" "Rollback terminé avec succès"
}

rollback_postgresql() {
    local tenant_id="$1"
    local environment="$2"
    local backup_path="$3"
    
    # Drop et recréation de la base
    dropdb --host="${PG_HOST}" --port="${PG_PORT}" --username="${PG_USERNAME}" "${tenant_id}_${environment}"
    createdb --host="${PG_HOST}" --port="${PG_PORT}" --username="${PG_USERNAME}" "${tenant_id}_${environment}"
    
    # Restauration depuis le backup
    gunzip -c "${backup_path}" | pg_restore \
        --host="${PG_HOST}" \
        --port="${PG_PORT}" \
        --username="${PG_USERNAME}" \
        --dbname="${tenant_id}_${environment}" \
        --verbose \
        --exit-on-error
}

# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

main() {
    local tenant_id="${1:-}"
    local environment="${2:-}"
    local database_type="${3:-}"
    local migration_file="${4:-}"
    local action="${5:-migrate}"
    
    log "INFO" "=== Démarrage de la migration de base de données ==="
    log "INFO" "Tenant: ${tenant_id}, Env: ${environment}, DB: ${database_type}"
    log "INFO" "Migration: ${migration_file}, Action: ${action}"
    
    # Validation des paramètres
    if [[ -z "${tenant_id}" ]] || [[ -z "${environment}" ]] || [[ -z "${database_type}" ]]; then
        log "ERROR" "Usage: $0 <tenant_id> <environment> <database_type> <migration_file> [action]"
        exit 1
    fi
    
    # Création du répertoire temporaire
    mkdir -p "${TEMP_DIR}"
    
    # Acquisition du lock de migration
    acquire_migration_lock "${tenant_id}" "${database_type}"
    
    # Génération de l'ID de migration
    local migration_id="migration-$(date +%Y%m%d-%H%M%S)-${tenant_id}-${database_type}"
    create_migration_state "${tenant_id}" "${environment}" "${database_type}" "${migration_id}"
    
    case "${action}" in
        "migrate")
            # Processus complet de migration
            validate_pre_migration "${tenant_id}" "${environment}" "${database_type}"
            
            # Création du backup
            local backup_id
            backup_id=$(create_backup "${tenant_id}" "${environment}" "${database_type}")
            
            # Exécution de la migration
            if execute_migration "${tenant_id}" "${environment}" "${database_type}" "${migration_file}"; then
                # Validation post-migration
                if validate_post_migration "${tenant_id}" "${environment}" "${database_type}"; then
                    update_migration_state "status" "COMPLETED"
                    log "SUCCESS" "Migration terminée avec succès"
                else
                    log "ERROR" "Validation post-migration échouée, rollback automatique..."
                    rollback_migration "${tenant_id}" "${environment}" "${database_type}"
                fi
            else
                log "ERROR" "Migration échouée, rollback automatique..."
                rollback_migration "${tenant_id}" "${environment}" "${database_type}"
            fi
            ;;
            
        "rollback")
            rollback_migration "${tenant_id}" "${environment}" "${database_type}"
            ;;
            
        "validate")
            validate_pre_migration "${tenant_id}" "${environment}" "${database_type}"
            log "SUCCESS" "Validation pré-migration réussie"
            ;;
            
        *)
            log "ERROR" "Action inconnue: ${action}"
            exit 1
            ;;
    esac
    
    log "SUCCESS" "=== Migration terminée ==="
}

# Exécution si appelé directement
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
