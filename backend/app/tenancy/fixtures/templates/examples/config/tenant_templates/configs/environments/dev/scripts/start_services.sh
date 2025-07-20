#!/bin/bash
# =============================================================================
# Script de Démarrage des Services pour l'Environnement de Développement
# =============================================================================
# 
# Ce script démarre tous les services nécessaires pour l'environnement
# de développement du projet Spotify AI Agent
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

log_service() {
    echo -e "${PURPLE}[SERVICE]${NC} $1"
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${CYAN}[DEBUG]${NC} $1"
    fi
}

# Variables globales
SERVICES_STARTED=()
CLEANUP_REQUIRED=false

# Fonction de nettoyage
cleanup() {
    if [[ "$CLEANUP_REQUIRED" == "true" ]]; then
        log_info "Nettoyage en cours..."
        
        # Arrêt des services démarrés
        for service in "${SERVICES_STARTED[@]}"; do
            case $service in
                "api")
                    stop_api_service
                    ;;
                "redis")
                    stop_redis_service
                    ;;
                "postgresql")
                    stop_postgresql_service
                    ;;
                "monitoring")
                    stop_monitoring_services
                    ;;
            esac
        done
        
        log_info "Nettoyage terminé"
    fi
}

# Configuration du trap pour le nettoyage
trap cleanup EXIT INT TERM

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
    export DEV_API_HOST="${DEV_FASTAPI_HOST:-0.0.0.0}"
    export DEV_API_PORT="${DEV_FASTAPI_PORT:-8000}"
    export DEV_API_WORKERS="${DEV_FASTAPI_WORKERS:-1}"
    export DEV_DB_HOST="${DEV_DB_HOST:-localhost}"
    export DEV_DB_PORT="${DEV_DB_PORT:-5432}"
    export DEV_REDIS_HOST="${DEV_REDIS_HOST:-localhost}"
    export DEV_REDIS_PORT="${DEV_REDIS_PORT:-6379}"
    export DEV_HOT_RELOAD="${DEV_FASTAPI_RELOAD:-true}"
    export DEV_DEBUG="${DEBUG:-true}"
}

# Fonction de vérification des prérequis
check_prerequisites() {
    log_info "Vérification des prérequis..."
    
    # Vérification de Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 non installé"
        exit 1
    fi
    
    # Vérification de l'environnement virtuel
    if [ ! -d "$PROJECT_ROOT/venv" ]; then
        log_error "Environnement virtuel non trouvé. Exécutez d'abord setup_dev.sh"
        exit 1
    fi
    
    log_success "Prérequis vérifiés"
}

# Fonction de vérification de la connectivité des services
check_service_connectivity() {
    log_info "Vérification de la connectivité des services..."
    
    local services_ok=true
    
    # Vérification PostgreSQL
    if ! nc -z "$DEV_DB_HOST" "$DEV_DB_PORT" 2>/dev/null; then
        log_warning "PostgreSQL non accessible sur $DEV_DB_HOST:$DEV_DB_PORT"
        services_ok=false
    else
        log_debug "PostgreSQL accessible"
    fi
    
    # Vérification Redis
    if ! nc -z "$DEV_REDIS_HOST" "$DEV_REDIS_PORT" 2>/dev/null; then
        log_warning "Redis non accessible sur $DEV_REDIS_HOST:$DEV_REDIS_PORT"
        services_ok=false
    else
        log_debug "Redis accessible"
    fi
    
    if [[ "$services_ok" == "false" && "${IGNORE_SERVICE_CHECKS:-false}" != "true" ]]; then
        log_error "Certains services ne sont pas accessibles"
        log_info "Démarrez les services requis ou utilisez --ignore-services pour continuer"
        exit 1
    fi
}

# Fonction de démarrage du service PostgreSQL local
start_postgresql_service() {
    if [[ "${START_POSTGRESQL:-false}" == "true" ]]; then
        log_service "Démarrage de PostgreSQL..."
        
        # Tentative de démarrage selon l'OS
        if command -v systemctl &> /dev/null; then
            sudo systemctl start postgresql || log_warning "Impossible de démarrer PostgreSQL via systemctl"
        elif command -v brew &> /dev/null && brew services list | grep -q postgresql; then
            brew services start postgresql || log_warning "Impossible de démarrer PostgreSQL via brew"
        elif command -v pg_ctl &> /dev/null; then
            pg_ctl start -D /usr/local/var/postgres || log_warning "Impossible de démarrer PostgreSQL via pg_ctl"
        else
            log_warning "Impossible de démarrer PostgreSQL automatiquement"
            return 1
        fi
        
        # Attente que le service soit prêt
        local attempts=0
        while ! nc -z "$DEV_DB_HOST" "$DEV_DB_PORT" 2>/dev/null && [ $attempts -lt 30 ]; do
            sleep 1
            ((attempts++))
        done
        
        if nc -z "$DEV_DB_HOST" "$DEV_DB_PORT" 2>/dev/null; then
            log_success "PostgreSQL démarré"
            SERVICES_STARTED+=("postgresql")
        else
            log_error "PostgreSQL n'a pas pu être démarré"
            return 1
        fi
    fi
}

# Fonction de démarrage du service Redis local
start_redis_service() {
    if [[ "${START_REDIS:-false}" == "true" ]]; then
        log_service "Démarrage de Redis..."
        
        # Tentative de démarrage selon l'OS
        if command -v systemctl &> /dev/null; then
            sudo systemctl start redis || sudo systemctl start redis-server || log_warning "Impossible de démarrer Redis via systemctl"
        elif command -v brew &> /dev/null && brew services list | grep -q redis; then
            brew services start redis || log_warning "Impossible de démarrer Redis via brew"
        elif command -v redis-server &> /dev/null; then
            redis-server --daemonize yes || log_warning "Impossible de démarrer Redis en daemon"
        else
            log_warning "Impossible de démarrer Redis automatiquement"
            return 1
        fi
        
        # Attente que le service soit prêt
        local attempts=0
        while ! nc -z "$DEV_REDIS_HOST" "$DEV_REDIS_PORT" 2>/dev/null && [ $attempts -lt 30 ]; do
            sleep 1
            ((attempts++))
        done
        
        if nc -z "$DEV_REDIS_HOST" "$DEV_REDIS_PORT" 2>/dev/null; then
            log_success "Redis démarré"
            SERVICES_STARTED+=("redis")
        else
            log_error "Redis n'a pas pu être démarré"
            return 1
        fi
    fi
}

# Fonction de préparation de l'environnement Python
prepare_python_environment() {
    log_info "Préparation de l'environnement Python..."
    
    cd "$PROJECT_ROOT"
    
    # Activation de l'environnement virtuel
    source venv/bin/activate
    
    # Mise à jour des dépendances si nécessaire
    if [[ "${UPDATE_DEPS:-false}" == "true" ]]; then
        log_info "Mise à jour des dépendances..."
        pip install --upgrade pip
        
        if [ -f "backend/requirements-dev.txt" ]; then
            pip install -r backend/requirements-dev.txt
        fi
        
        if [ -f "backend/requirements.txt" ]; then
            pip install -r backend/requirements.txt
        fi
        
        log_success "Dépendances mises à jour"
    fi
    
    # Configuration des variables d'environnement Python
    export PYTHONPATH="$PROJECT_ROOT/backend:$PROJECT_ROOT/backend/app:$PYTHONPATH"
    export ENVIRONMENT="development"
    
    log_success "Environnement Python préparé"
}

# Fonction d'exécution des migrations de base de données
run_database_migrations() {
    if [[ "${SKIP_MIGRATIONS:-false}" != "true" ]]; then
        log_info "Exécution des migrations de base de données..."
        
        cd "$PROJECT_ROOT/backend"
        
        # Vérification de la présence d'Alembic
        if python3 -c "import alembic" 2>/dev/null; then
            # Application des migrations
            if alembic upgrade head; then
                log_success "Migrations appliquées avec succès"
            else
                log_warning "Erreur lors de l'application des migrations"
                if [[ "${IGNORE_MIGRATION_ERRORS:-false}" != "true" ]]; then
                    return 1
                fi
            fi
        else
            log_warning "Alembic non installé. Migrations ignorées."
        fi
    else
        log_info "Migrations ignorées (SKIP_MIGRATIONS=true)"
    fi
}

# Fonction de démarrage du service API principal
start_api_service() {
    log_service "Démarrage du service API principal..."
    
    cd "$PROJECT_ROOT/backend"
    
    # Configuration du serveur
    local uvicorn_args=(
        "app.main:app"
        "--host" "$DEV_API_HOST"
        "--port" "$DEV_API_PORT"
    )
    
    # Configuration du hot reload
    if [[ "$DEV_HOT_RELOAD" == "true" ]]; then
        uvicorn_args+=("--reload")
        uvicorn_args+=("--reload-dir" "app")
        uvicorn_args+=("--reload-dir" "config")
    fi
    
    # Configuration du nombre de workers
    if [[ "$DEV_API_WORKERS" -gt 1 && "$DEV_HOT_RELOAD" != "true" ]]; then
        uvicorn_args+=("--workers" "$DEV_API_WORKERS")
    fi
    
    # Configuration du logging
    if [[ "$DEV_DEBUG" == "true" ]]; then
        uvicorn_args+=("--log-level" "debug")
    else
        uvicorn_args+=("--log-level" "info")
    fi
    
    # Démarrage en arrière-plan ou premier plan
    if [[ "${BACKGROUND_MODE:-false}" == "true" ]]; then
        # Démarrage en arrière-plan
        log_info "Démarrage du serveur API en arrière-plan..."
        
        # Fichier de PID pour le suivi
        local pid_file="$PROJECT_ROOT/.api_service.pid"
        
        # Démarrage avec nohup
        nohup uvicorn "${uvicorn_args[@]}" > "$PROJECT_ROOT/logs/api_service.log" 2>&1 &
        local api_pid=$!
        echo $api_pid > "$pid_file"
        
        # Attente que le service soit prêt
        local attempts=0
        while ! nc -z "$DEV_API_HOST" "$DEV_API_PORT" 2>/dev/null && [ $attempts -lt 30 ]; do
            sleep 1
            ((attempts++))
        done
        
        if nc -z "$DEV_API_HOST" "$DEV_API_PORT" 2>/dev/null; then
            log_success "Service API démarré en arrière-plan (PID: $api_pid)"
            log_info "Logs disponibles dans: $PROJECT_ROOT/logs/api_service.log"
            SERVICES_STARTED+=("api")
        else
            log_error "Le service API n'a pas pu démarrer"
            return 1
        fi
    else
        # Démarrage en premier plan
        log_info "Démarrage du serveur API..."
        log_info "URL d'accès: http://$DEV_API_HOST:$DEV_API_PORT"
        log_info "Documentation API: http://$DEV_API_HOST:$DEV_API_PORT/docs"
        log_info "Appuyez sur Ctrl+C pour arrêter"
        echo
        
        # Marquer pour le nettoyage
        CLEANUP_REQUIRED=true
        SERVICES_STARTED+=("api")
        
        # Démarrage du serveur
        uvicorn "${uvicorn_args[@]}"
    fi
}

# Fonction de démarrage des services de monitoring
start_monitoring_services() {
    if [[ "${START_MONITORING:-false}" == "true" ]]; then
        log_service "Démarrage des services de monitoring..."
        
        # Démarrage de Prometheus si configuré
        if [[ "${DEV_PROMETHEUS_ENABLED:-false}" == "true" ]]; then
            local prometheus_port="${DEV_PROMETHEUS_PORT:-9090}"
            
            if command -v prometheus &> /dev/null; then
                log_info "Démarrage de Prometheus sur le port $prometheus_port..."
                
                # Configuration Prometheus pour développement
                local prometheus_config="$PROJECT_ROOT/config/prometheus/dev.yml"
                
                if [ -f "$prometheus_config" ]; then
                    nohup prometheus --config.file="$prometheus_config" --web.listen-address=":$prometheus_port" > "$PROJECT_ROOT/logs/prometheus.log" 2>&1 &
                    local prometheus_pid=$!
                    echo $prometheus_pid > "$PROJECT_ROOT/.prometheus.pid"
                    
                    log_success "Prometheus démarré (PID: $prometheus_pid)"
                    log_info "Interface Prometheus: http://localhost:$prometheus_port"
                else
                    log_warning "Configuration Prometheus non trouvée: $prometheus_config"
                fi
            else
                log_warning "Prometheus non installé"
            fi
        fi
        
        SERVICES_STARTED+=("monitoring")
    fi
}

# Fonction d'arrêt du service API
stop_api_service() {
    local pid_file="$PROJECT_ROOT/.api_service.pid"
    
    if [ -f "$pid_file" ]; then
        local api_pid=$(cat "$pid_file")
        if kill -0 "$api_pid" 2>/dev/null; then
            log_info "Arrêt du service API (PID: $api_pid)..."
            kill "$api_pid"
            rm -f "$pid_file"
            log_success "Service API arrêté"
        fi
    fi
}

# Fonction d'arrêt du service Redis
stop_redis_service() {
    if command -v redis-cli &> /dev/null; then
        log_info "Arrêt de Redis..."
        redis-cli shutdown || true
    fi
}

# Fonction d'arrêt du service PostgreSQL
stop_postgresql_service() {
    if command -v systemctl &> /dev/null; then
        sudo systemctl stop postgresql || true
    elif command -v brew &> /dev/null; then
        brew services stop postgresql || true
    fi
}

# Fonction d'arrêt des services de monitoring
stop_monitoring_services() {
    local prometheus_pid_file="$PROJECT_ROOT/.prometheus.pid"
    
    if [ -f "$prometheus_pid_file" ]; then
        local prometheus_pid=$(cat "$prometheus_pid_file")
        if kill -0 "$prometheus_pid" 2>/dev/null; then
            log_info "Arrêt de Prometheus (PID: $prometheus_pid)..."
            kill "$prometheus_pid"
            rm -f "$prometheus_pid_file"
        fi
    fi
}

# Fonction d'affichage du statut des services
show_service_status() {
    echo
    log_info "Statut des services:"
    echo "===================="
    
    # API Service
    if nc -z "$DEV_API_HOST" "$DEV_API_PORT" 2>/dev/null; then
        echo -e "API Service:     ${GREEN}✓ Actif${NC} (http://$DEV_API_HOST:$DEV_API_PORT)"
    else
        echo -e "API Service:     ${RED}✗ Inactif${NC}"
    fi
    
    # PostgreSQL
    if nc -z "$DEV_DB_HOST" "$DEV_DB_PORT" 2>/dev/null; then
        echo -e "PostgreSQL:      ${GREEN}✓ Actif${NC} ($DEV_DB_HOST:$DEV_DB_PORT)"
    else
        echo -e "PostgreSQL:      ${RED}✗ Inactif${NC}"
    fi
    
    # Redis
    if nc -z "$DEV_REDIS_HOST" "$DEV_REDIS_PORT" 2>/dev/null; then
        echo -e "Redis:           ${GREEN}✓ Actif${NC} ($DEV_REDIS_HOST:$DEV_REDIS_PORT)"
    else
        echo -e "Redis:           ${RED}✗ Inactif${NC}"
    fi
    
    # Prometheus
    local prometheus_port="${DEV_PROMETHEUS_PORT:-9090}"
    if nc -z "localhost" "$prometheus_port" 2>/dev/null; then
        echo -e "Prometheus:      ${GREEN}✓ Actif${NC} (http://localhost:$prometheus_port)"
    else
        echo -e "Prometheus:      ${RED}✗ Inactif${NC}"
    fi
    
    echo
}

# Fonction d'affichage de l'aide
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Démarre les services de développement pour Spotify AI Agent"
    echo
    echo "OPTIONS:"
    echo "  --background         Démarre les services en arrière-plan"
    echo "  --start-db           Démarre PostgreSQL automatiquement"
    echo "  --start-redis        Démarre Redis automatiquement"
    echo "  --start-monitoring   Démarre les services de monitoring"
    echo "  --update-deps        Met à jour les dépendances avant démarrage"
    echo "  --skip-migrations    Ignore les migrations de base de données"
    echo "  --ignore-services    Ignore les vérifications de connectivité"
    echo "  --status             Affiche le statut des services et quitte"
    echo "  --debug              Active le mode debug verbose"
    echo "  --help               Affiche cette aide"
    echo
    echo "VARIABLES D'ENVIRONNEMENT:"
    echo "  DEV_API_HOST         Host de l'API (défaut: 0.0.0.0)"
    echo "  DEV_API_PORT         Port de l'API (défaut: 8000)"
    echo "  DEV_API_WORKERS      Nombre de workers (défaut: 1)"
    echo "  DEV_HOT_RELOAD       Active le hot reload (défaut: true)"
    echo "  DEBUG                Active le mode debug (défaut: true)"
    echo
    echo "EXEMPLES:"
    echo "  $0                          # Démarre en mode interactif"
    echo "  $0 --background             # Démarre en arrière-plan"
    echo "  $0 --start-db --start-redis # Démarre avec services locaux"
    echo "  $0 --status                 # Affiche seulement le statut"
    echo
}

# Fonction principale
main() {
    echo -e "${BLUE}==================================================================${NC}"
    echo -e "${BLUE}   Démarrage des Services - Environnement de Développement${NC}"
    echo -e "${BLUE}   Spotify AI Agent - Fahed Mlaiel & Team${NC}"
    echo -e "${BLUE}==================================================================${NC}"
    echo
    
    # Analyse des arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --background)
                export BACKGROUND_MODE="true"
                shift
                ;;
            --start-db)
                export START_POSTGRESQL="true"
                shift
                ;;
            --start-redis)
                export START_REDIS="true"
                shift
                ;;
            --start-monitoring)
                export START_MONITORING="true"
                shift
                ;;
            --update-deps)
                export UPDATE_DEPS="true"
                shift
                ;;
            --skip-migrations)
                export SKIP_MIGRATIONS="true"
                shift
                ;;
            --ignore-services)
                export IGNORE_SERVICE_CHECKS="true"
                shift
                ;;
            --status)
                load_environment
                show_service_status
                exit 0
                ;;
            --debug)
                export DEBUG="true"
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
    
    # Exécution des étapes de démarrage
    load_environment
    check_prerequisites
    check_service_connectivity
    
    # Démarrage des services externes si demandé
    start_postgresql_service
    start_redis_service
    
    # Préparation de l'environnement
    prepare_python_environment
    run_database_migrations
    
    # Démarrage des services de monitoring
    start_monitoring_services
    
    # Affichage du statut avant démarrage de l'API
    if [[ "${BACKGROUND_MODE:-false}" == "true" ]]; then
        show_service_status
    fi
    
    # Démarrage du service API principal
    start_api_service
    
    # Affichage final du statut en mode arrière-plan
    if [[ "${BACKGROUND_MODE:-false}" == "true" ]]; then
        show_service_status
        log_success "Tous les services démarrés en arrière-plan"
        log_info "Utilisez '$0 --status' pour vérifier le statut"
    fi
}

# Exécution du script principal
main "$@"
