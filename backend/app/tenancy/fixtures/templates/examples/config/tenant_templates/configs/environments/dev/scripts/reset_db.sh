#!/bin/bash
# =============================================================================
# Script de Reset de Base de Données pour l'Environnement de Développement
# =============================================================================
# 
# Ce script remet à zéro la base de données de développement et la recrée
# avec les données initiales
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

# Chargement des variables d'environnement
load_environment() {
    local env_file="$ENV_DIR/secrets/.env"
    
    if [ -f "$env_file" ]; then
        log_info "Chargement des variables d'environnement..."
        set -a  # automatically export all variables
        source "$env_file"
        set +a
        log_success "Variables d'environnement chargées"
    else
        log_warning "Fichier .env non trouvé, utilisation des valeurs par défaut"
    fi
    
    # Variables par défaut si non définies
    export DEV_DB_HOST="${DEV_DB_HOST:-localhost}"
    export DEV_DB_PORT="${DEV_DB_PORT:-5432}"
    export DEV_DB_NAME="${DEV_DB_NAME:-spotify_ai_agent_dev}"
    export DEV_DB_USER="${DEV_DB_USER:-postgres}"
    export DEV_DB_PASSWORD="${DEV_DB_PASSWORD:-postgres}"
    export TEST_DB_NAME="${TEST_DB_NAME:-spotify_ai_agent_test}"
}

# Fonction de confirmation
confirm_reset() {
    local db_name="$1"
    local reset_type="$2"
    
    echo
    log_warning "ATTENTION: Cette opération va ${reset_type} la base de données '$db_name'"
    log_warning "Toutes les données existantes seront perdues !"
    echo
    
    if [[ "${FORCE_RESET:-false}" != "true" ]]; then
        read -p "Êtes-vous sûr de vouloir continuer ? (tapez 'oui' pour confirmer): " confirmation
        
        if [[ "$confirmation" != "oui" ]]; then
            log_info "Opération annulée par l'utilisateur"
            exit 0
        fi
    else
        log_info "Mode force activé, continuation automatique"
    fi
}

# Fonction de sauvegarde de la base de données
backup_database() {
    local db_name="$1"
    
    log_info "Création d'une sauvegarde de la base de données '$db_name'..."
    
    local backup_dir="$PROJECT_ROOT/backups"
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_file="$backup_dir/${db_name}_backup_${timestamp}.sql"
    
    # Création du répertoire de sauvegarde
    mkdir -p "$backup_dir"
    
    # Vérification de l'existence de la base de données
    if PGPASSWORD="$DEV_DB_PASSWORD" psql -h "$DEV_DB_HOST" -p "$DEV_DB_PORT" -U "$DEV_DB_USER" -lqt | cut -d \| -f 1 | grep -qw "$db_name"; then
        # Création de la sauvegarde
        if PGPASSWORD="$DEV_DB_PASSWORD" pg_dump -h "$DEV_DB_HOST" -p "$DEV_DB_PORT" -U "$DEV_DB_USER" "$db_name" > "$backup_file"; then
            log_success "Sauvegarde créée: $backup_file"
            echo "$backup_file"
        else
            log_error "Erreur lors de la création de la sauvegarde"
            return 1
        fi
    else
        log_info "Base de données '$db_name' n'existe pas, pas de sauvegarde nécessaire"
        echo ""
    fi
}

# Fonction de suppression de la base de données
drop_database() {
    local db_name="$1"
    
    log_info "Suppression de la base de données '$db_name'..."
    
    # Vérification de l'existence de la base de données
    if PGPASSWORD="$DEV_DB_PASSWORD" psql -h "$DEV_DB_HOST" -p "$DEV_DB_PORT" -U "$DEV_DB_USER" -lqt | cut -d \| -f 1 | grep -qw "$db_name"; then
        # Fermeture des connexions actives
        PGPASSWORD="$DEV_DB_PASSWORD" psql -h "$DEV_DB_HOST" -p "$DEV_DB_PORT" -U "$DEV_DB_USER" -d postgres -c "
            SELECT pg_terminate_backend(pg_stat_activity.pid)
            FROM pg_stat_activity
            WHERE pg_stat_activity.datname = '$db_name'
            AND pid <> pg_backend_pid();
        " 2>/dev/null || true
        
        # Suppression de la base de données
        if PGPASSWORD="$DEV_DB_PASSWORD" dropdb -h "$DEV_DB_HOST" -p "$DEV_DB_PORT" -U "$DEV_DB_USER" "$db_name"; then
            log_success "Base de données '$db_name' supprimée"
        else
            log_error "Erreur lors de la suppression de la base de données '$db_name'"
            return 1
        fi
    else
        log_info "Base de données '$db_name' n'existe pas"
    fi
}

# Fonction de création de la base de données
create_database() {
    local db_name="$1"
    
    log_info "Création de la base de données '$db_name'..."
    
    # Création de la base de données
    if PGPASSWORD="$DEV_DB_PASSWORD" createdb -h "$DEV_DB_HOST" -p "$DEV_DB_PORT" -U "$DEV_DB_USER" "$db_name"; then
        log_success "Base de données '$db_name' créée"
    else
        log_error "Erreur lors de la création de la base de données '$db_name'"
        return 1
    fi
}

# Fonction d'exécution des migrations
run_migrations() {
    log_info "Exécution des migrations de base de données..."
    
    cd "$PROJECT_ROOT"
    
    # Activation de l'environnement virtuel
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    # Vérification de la présence d'Alembic
    if python3 -c "import alembic" 2>/dev/null; then
        cd backend
        
        # Exécution des migrations
        if alembic upgrade head; then
            log_success "Migrations appliquées avec succès"
        else
            log_error "Erreur lors de l'application des migrations"
            return 1
        fi
    else
        log_warning "Alembic non installé. Migrations ignorées."
    fi
}

# Fonction de chargement des données initiales
load_seed_data() {
    log_info "Chargement des données initiales..."
    
    local seed_files=(
        "$PROJECT_ROOT/fixtures/seed_data.sql"
        "$ENV_DIR/../../../fixtures/seed_data.sql"
        "$PROJECT_ROOT/backend/fixtures/seed_data.sql"
    )
    
    local seed_loaded=false
    
    for seed_file in "${seed_files[@]}"; do
        if [ -f "$seed_file" ]; then
            log_info "Chargement du fichier de données: $seed_file"
            
            if PGPASSWORD="$DEV_DB_PASSWORD" psql -h "$DEV_DB_HOST" -p "$DEV_DB_PORT" -U "$DEV_DB_USER" -d "$DEV_DB_NAME" -f "$seed_file"; then
                log_success "Données chargées depuis: $seed_file"
                seed_loaded=true
                break
            else
                log_warning "Erreur lors du chargement de: $seed_file"
            fi
        fi
    done
    
    if ! $seed_loaded; then
        log_warning "Aucun fichier de données initiales trouvé ou chargé"
    fi
}

# Fonction de chargement des données de développement
load_dev_data() {
    log_info "Chargement des données de développement..."
    
    cd "$PROJECT_ROOT"
    
    # Activation de l'environnement virtuel
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    # Recherche de scripts de données de développement
    local dev_data_scripts=(
        "backend/scripts/load_dev_data.py"
        "scripts/load_dev_data.py"
        "fixtures/load_dev_data.py"
    )
    
    for script in "${dev_data_scripts[@]}"; do
        if [ -f "$script" ]; then
            log_info "Exécution du script de données de développement: $script"
            
            if python3 "$script"; then
                log_success "Données de développement chargées depuis: $script"
                return 0
            else
                log_warning "Erreur lors de l'exécution de: $script"
            fi
        fi
    done
    
    log_info "Aucun script de données de développement trouvé"
}

# Fonction de validation de la base de données
validate_database() {
    local db_name="$1"
    
    log_info "Validation de la base de données '$db_name'..."
    
    # Test de connexion
    if PGPASSWORD="$DEV_DB_PASSWORD" psql -h "$DEV_DB_HOST" -p "$DEV_DB_PORT" -U "$DEV_DB_USER" -d "$db_name" -c "SELECT 1;" &> /dev/null; then
        log_success "Connexion à la base de données réussie"
    else
        log_error "Impossible de se connecter à la base de données"
        return 1
    fi
    
    # Vérification des tables principales
    local table_count=$(PGPASSWORD="$DEV_DB_PASSWORD" psql -h "$DEV_DB_HOST" -p "$DEV_DB_PORT" -U "$DEV_DB_USER" -d "$db_name" -t -c "
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_type = 'BASE TABLE';
    " | tr -d ' ')
    
    log_info "Nombre de tables trouvées: $table_count"
    
    if [ "$table_count" -gt 0 ]; then
        log_success "Base de données validée avec $table_count tables"
    else
        log_warning "Aucune table trouvée dans la base de données"
    fi
}

# Fonction de reset complet
reset_development_database() {
    local db_name="$DEV_DB_NAME"
    
    confirm_reset "$db_name" "réinitialiser complètement"
    
    # Sauvegarde optionnelle
    if [[ "${SKIP_BACKUP:-false}" != "true" ]]; then
        backup_database "$db_name"
    fi
    
    # Reset de la base de données
    drop_database "$db_name"
    create_database "$db_name"
    run_migrations
    load_seed_data
    
    if [[ "${LOAD_DEV_DATA:-true}" == "true" ]]; then
        load_dev_data
    fi
    
    validate_database "$db_name"
    
    log_success "Reset de la base de données de développement terminé"
}

# Fonction de reset de la base de données de test
reset_test_database() {
    local db_name="$TEST_DB_NAME"
    
    confirm_reset "$db_name" "réinitialiser"
    
    # Reset de la base de données de test (pas de sauvegarde nécessaire)
    drop_database "$db_name"
    create_database "$db_name"
    run_migrations
    
    validate_database "$db_name"
    
    log_success "Reset de la base de données de test terminé"
}

# Fonction de reset des données uniquement
reset_data_only() {
    local db_name="$DEV_DB_NAME"
    
    confirm_reset "$db_name" "vider toutes les données de"
    
    log_info "Suppression de toutes les données..."
    
    # Récupération de la liste des tables
    local tables=$(PGPASSWORD="$DEV_DB_PASSWORD" psql -h "$DEV_DB_HOST" -p "$DEV_DB_PORT" -U "$DEV_DB_USER" -d "$db_name" -t -c "
        SELECT string_agg(tablename, ', ')
        FROM pg_tables 
        WHERE schemaname = 'public' 
        AND tablename != 'alembic_version';
    " | tr -d ' ')
    
    if [ -n "$tables" ]; then
        # Suppression des données avec TRUNCATE
        PGPASSWORD="$DEV_DB_PASSWORD" psql -h "$DEV_DB_HOST" -p "$DEV_DB_PORT" -U "$DEV_DB_USER" -d "$db_name" -c "
            TRUNCATE TABLE $tables RESTART IDENTITY CASCADE;
        "
        log_success "Toutes les données supprimées"
        
        # Rechargement des données
        load_seed_data
        if [[ "${LOAD_DEV_DATA:-true}" == "true" ]]; then
            load_dev_data
        fi
    else
        log_info "Aucune table trouvée à vider"
    fi
    
    log_success "Reset des données terminé"
}

# Fonction d'affichage de l'aide
show_help() {
    echo "Usage: $0 [OPTION]"
    echo
    echo "Reset de la base de données de développement"
    echo
    echo "OPTIONS:"
    echo "  --full              Reset complet (suppression et recréation)"
    echo "  --test              Reset de la base de données de test"
    echo "  --data-only         Suppression des données uniquement (garde la structure)"
    echo "  --skip-backup       Ignore la sauvegarde automatique"
    echo "  --skip-dev-data     Ne charge pas les données de développement"
    echo "  --force             Force l'exécution sans confirmation"
    echo "  --help              Affiche cette aide"
    echo
    echo "VARIABLES D'ENVIRONNEMENT:"
    echo "  DEV_DB_HOST         Host de la base de données (défaut: localhost)"
    echo "  DEV_DB_PORT         Port de la base de données (défaut: 5432)"
    echo "  DEV_DB_NAME         Nom de la base de données (défaut: spotify_ai_agent_dev)"
    echo "  DEV_DB_USER         Utilisateur de la base de données (défaut: postgres)"
    echo "  DEV_DB_PASSWORD     Mot de passe de la base de données (défaut: postgres)"
    echo
    echo "EXEMPLES:"
    echo "  $0 --full                    # Reset complet avec sauvegarde"
    echo "  $0 --data-only --force       # Suppression des données sans confirmation"
    echo "  $0 --test                    # Reset de la base de données de test"
    echo
}

# Fonction principale
main() {
    echo -e "${BLUE}==================================================================${NC}"
    echo -e "${BLUE}   Reset de Base de Données - Environnement de Développement${NC}"
    echo -e "${BLUE}   Spotify AI Agent - Fahed Mlaiel & Team${NC}"
    echo -e "${BLUE}==================================================================${NC}"
    echo
    
    # Chargement des variables d'environnement
    load_environment
    
    # Analyse des arguments
    local reset_type="full"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --full)
                reset_type="full"
                shift
                ;;
            --test)
                reset_type="test"
                shift
                ;;
            --data-only)
                reset_type="data-only"
                shift
                ;;
            --skip-backup)
                export SKIP_BACKUP="true"
                shift
                ;;
            --skip-dev-data)
                export LOAD_DEV_DATA="false"
                shift
                ;;
            --force)
                export FORCE_RESET="true"
                shift
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
    
    # Vérification de la connexion PostgreSQL
    if ! command -v psql &> /dev/null; then
        log_error "PostgreSQL client non installé"
        exit 1
    fi
    
    if ! PGPASSWORD="$DEV_DB_PASSWORD" psql -h "$DEV_DB_HOST" -p "$DEV_DB_PORT" -U "$DEV_DB_USER" -d postgres -c "SELECT 1;" &> /dev/null; then
        log_error "Impossible de se connecter à PostgreSQL"
        log_error "Vérifiez que PostgreSQL est démarré et que les paramètres de connexion sont corrects"
        exit 1
    fi
    
    # Exécution du reset selon le type demandé
    case $reset_type in
        full)
            reset_development_database
            ;;
        test)
            reset_test_database
            ;;
        data-only)
            reset_data_only
            ;;
    esac
    
    echo
    log_success "Opération de reset terminée avec succès !"
}

# Exécution du script principal
main "$@"
