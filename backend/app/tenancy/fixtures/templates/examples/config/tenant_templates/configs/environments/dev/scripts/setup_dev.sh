#!/bin/bash
# =============================================================================
# Script de Configuration Initiale pour l'Environnement de Développement
# =============================================================================
# 
# Ce script configure automatiquement l'environnement de développement
# pour le projet Spotify AI Agent
#
# Développé par l'équipe d'experts dirigée par Fahed Mlaiel
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)"
ENV_DIR="$(dirname "$SCRIPT_DIR")"
DEV_ENV="development"

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

# Fonction de vérification des prérequis
check_prerequisites() {
    log_info "Vérification des prérequis..."
    
    local missing_tools=()
    
    # Vérification de Python
    if ! command -v python3 &> /dev/null; then
        missing_tools+=("python3")
    fi
    
    # Vérification de pip
    if ! command -v pip3 &> /dev/null; then
        missing_tools+=("pip3")
    fi
    
    # Vérification de PostgreSQL
    if ! command -v psql &> /dev/null; then
        log_warning "PostgreSQL client non trouvé. Installation recommandée."
    fi
    
    # Vérification de Redis
    if ! command -v redis-cli &> /dev/null; then
        log_warning "Redis client non trouvé. Installation recommandée."
    fi
    
    # Vérification de Docker (optionnel)
    if ! command -v docker &> /dev/null; then
        log_warning "Docker non trouvé. Installation recommandée pour le développement avec conteneurs."
    fi
    
    # Vérification de git
    if ! command -v git &> /dev/null; then
        missing_tools+=("git")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Outils manquants: ${missing_tools[*]}"
        log_error "Veuillez installer ces outils avant de continuer."
        exit 1
    fi
    
    log_success "Tous les prérequis sont satisfaits"
}

# Fonction de création de l'environnement virtuel Python
setup_python_environment() {
    log_info "Configuration de l'environnement Python..."
    
    cd "$PROJECT_ROOT"
    
    # Création de l'environnement virtuel s'il n'existe pas
    if [ ! -d "venv" ]; then
        log_info "Création de l'environnement virtuel Python..."
        python3 -m venv venv
        log_success "Environnement virtuel créé"
    else
        log_info "Environnement virtuel déjà existant"
    fi
    
    # Activation de l'environnement virtuel
    source venv/bin/activate
    
    # Mise à jour de pip
    log_info "Mise à jour de pip..."
    pip install --upgrade pip
    
    # Installation des dépendances de développement
    if [ -f "backend/requirements-dev.txt" ]; then
        log_info "Installation des dépendances de développement..."
        pip install -r backend/requirements-dev.txt
        log_success "Dépendances de développement installées"
    fi
    
    # Installation des dépendances principales
    if [ -f "backend/requirements.txt" ]; then
        log_info "Installation des dépendances principales..."
        pip install -r backend/requirements.txt
        log_success "Dépendances principales installées"
    fi
}

# Fonction de configuration des variables d'environnement
setup_environment_variables() {
    log_info "Configuration des variables d'environnement..."
    
    local env_file="$ENV_DIR/secrets/.env"
    local env_example="$ENV_DIR/secrets/.env.example"
    
    # Copie du fichier d'exemple s'il n'existe pas
    if [ ! -f "$env_file" ]; then
        if [ -f "$env_example" ]; then
            log_info "Création du fichier .env depuis l'exemple..."
            cp "$env_example" "$env_file"
            log_warning "Fichier .env créé. Veuillez le modifier avec vos paramètres."
        else
            log_warning "Fichier .env.example non trouvé"
        fi
    else
        log_info "Fichier .env déjà existant"
    fi
    
    # Génération de clés sécurisées si nécessaire
    if [ -f "$env_file" ]; then
        # Vérification si les clés par défaut sont encore présentes
        if grep -q "your-dev-.*-secret" "$env_file"; then
            log_warning "Des clés par défaut détectées dans .env"
            log_info "Génération de nouvelles clés sécurisées..."
            
            # Génération de nouvelles clés
            local new_jwt_secret=$(openssl rand -hex 32)
            local new_app_secret=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
            
            # Remplacement des clés par défaut
            sed -i.bak "s/your-dev-jwt-secret-key-change-this-in-production/$new_jwt_secret/g" "$env_file"
            sed -i.bak "s/your-dev-secret-key-change-this-in-production/$new_app_secret/g" "$env_file"
            
            log_success "Nouvelles clés générées et configurées"
        fi
    fi
}

# Fonction de configuration de la base de données
setup_database() {
    log_info "Configuration de la base de données de développement..."
    
    # Variables de base de données
    local db_host="${DEV_DB_HOST:-localhost}"
    local db_port="${DEV_DB_PORT:-5432}"
    local db_name="${DEV_DB_NAME:-spotify_ai_agent_dev}"
    local db_user="${DEV_DB_USER:-postgres}"
    local db_password="${DEV_DB_PASSWORD:-postgres}"
    
    # Vérification de la connexion PostgreSQL
    if command -v psql &> /dev/null; then
        log_info "Vérification de la connexion PostgreSQL..."
        
        # Test de connexion
        if PGPASSWORD="$db_password" psql -h "$db_host" -p "$db_port" -U "$db_user" -d postgres -c "SELECT 1;" &> /dev/null; then
            log_success "Connexion PostgreSQL réussie"
            
            # Création de la base de données si elle n'existe pas
            if ! PGPASSWORD="$db_password" psql -h "$db_host" -p "$db_port" -U "$db_user" -lqt | cut -d \| -f 1 | grep -qw "$db_name"; then
                log_info "Création de la base de données $db_name..."
                PGPASSWORD="$db_password" createdb -h "$db_host" -p "$db_port" -U "$db_user" "$db_name"
                log_success "Base de données $db_name créée"
            else
                log_info "Base de données $db_name déjà existante"
            fi
        else
            log_warning "Impossible de se connecter à PostgreSQL"
            log_info "Veuillez vérifier que PostgreSQL est démarré et accessible"
        fi
    else
        log_warning "Client PostgreSQL non installé. Configuration de la base de données ignorée."
    fi
}

# Fonction de configuration de Redis
setup_redis() {
    log_info "Configuration de Redis pour le développement..."
    
    local redis_host="${DEV_REDIS_HOST:-localhost}"
    local redis_port="${DEV_REDIS_PORT:-6379}"
    
    # Vérification de la connexion Redis
    if command -v redis-cli &> /dev/null; then
        log_info "Vérification de la connexion Redis..."
        
        if redis-cli -h "$redis_host" -p "$redis_port" ping &> /dev/null; then
            log_success "Connexion Redis réussie"
        else
            log_warning "Impossible de se connecter à Redis"
            log_info "Veuillez vérifier que Redis est démarré et accessible"
        fi
    else
        log_warning "Client Redis non installé. Configuration Redis ignorée."
    fi
}

# Fonction de configuration des répertoires
setup_directories() {
    log_info "Création des répertoires nécessaires..."
    
    local dirs=(
        "$PROJECT_ROOT/logs"
        "$PROJECT_ROOT/cache"
        "$PROJECT_ROOT/uploads"
        "$PROJECT_ROOT/downloads"
        "$PROJECT_ROOT/temp"
        "$PROJECT_ROOT/data"
        "$PROJECT_ROOT/fixtures"
        "$PROJECT_ROOT/tests/output"
        "$PROJECT_ROOT/tests/reports"
        "$ENV_DIR/secrets/keys"
    )
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "Répertoire créé: $dir"
        fi
    done
    
    log_success "Tous les répertoires nécessaires sont créés"
}

# Fonction de génération des clés de développement
generate_dev_keys() {
    log_info "Génération des clés de développement..."
    
    local keys_dir="$ENV_DIR/secrets/keys"
    
    # Génération d'une clé JWT
    if [ ! -f "$keys_dir/jwt_dev.key" ]; then
        openssl rand -hex 32 > "$keys_dir/jwt_dev.key"
        log_info "Clé JWT générée: jwt_dev.key"
    fi
    
    # Génération d'une clé de chiffrement
    if [ ! -f "$keys_dir/encryption_dev.key" ]; then
        python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" > "$keys_dir/encryption_dev.key"
        log_info "Clé de chiffrement générée: encryption_dev.key"
    fi
    
    # Génération d'une clé secrète d'application
    if [ ! -f "$keys_dir/app_secret_dev.key" ]; then
        python3 -c "import secrets; print(secrets.token_urlsafe(32))" > "$keys_dir/app_secret_dev.key"
        log_info "Clé secrète d'application générée: app_secret_dev.key"
    fi
    
    # Sécurisation des permissions
    chmod 600 "$keys_dir"/*.key
    
    log_success "Clés de développement générées et sécurisées"
}

# Fonction de validation de la configuration
validate_configuration() {
    log_info "Validation de la configuration..."
    
    # Validation du fichier de configuration principal
    local config_file="$ENV_DIR/dev.yml"
    if [ -f "$config_file" ]; then
        log_info "Fichier de configuration trouvé: dev.yml"
        
        # Validation YAML basique
        if python3 -c "import yaml; yaml.safe_load(open('$config_file'))" 2>/dev/null; then
            log_success "Fichier de configuration YAML valide"
        else
            log_error "Erreur dans le fichier de configuration YAML"
            return 1
        fi
    else
        log_error "Fichier de configuration principal non trouvé"
        return 1
    fi
    
    # Validation des overrides
    local overrides_dir="$ENV_DIR/overrides"
    if [ -d "$overrides_dir" ]; then
        for override_file in "$overrides_dir"/*.yml; do
            if [ -f "$override_file" ]; then
                if python3 -c "import yaml; yaml.safe_load(open('$override_file'))" 2>/dev/null; then
                    log_info "Override valide: $(basename "$override_file")"
                else
                    log_warning "Erreur dans l'override: $(basename "$override_file")"
                fi
            fi
        done
    fi
    
    log_success "Validation de la configuration terminée"
}

# Fonction d'initialisation des migrations de base de données
init_database_migrations() {
    log_info "Initialisation des migrations de base de données..."
    
    cd "$PROJECT_ROOT"
    
    # Activation de l'environnement virtuel
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    # Vérification de la présence d'Alembic
    if python3 -c "import alembic" 2>/dev/null; then
        # Initialisation d'Alembic si pas déjà fait
        if [ ! -d "backend/migrations" ]; then
            log_info "Initialisation d'Alembic..."
            cd backend
            alembic init migrations
            log_success "Alembic initialisé"
        else
            log_info "Migrations Alembic déjà configurées"
        fi
        
        # Application des migrations
        log_info "Application des migrations..."
        cd backend
        alembic upgrade head 2>/dev/null || log_warning "Aucune migration à appliquer ou erreur"
    else
        log_warning "Alembic non installé. Migrations ignorées."
    fi
}

# Fonction de configuration des hooks de développement
setup_development_hooks() {
    log_info "Configuration des hooks de développement..."
    
    cd "$PROJECT_ROOT"
    
    # Configuration des hooks Git si le projet est sous Git
    if [ -d ".git" ]; then
        log_info "Configuration des hooks Git..."
        
        # Hook pre-commit pour la qualité du code
        local pre_commit_hook=".git/hooks/pre-commit"
        if [ ! -f "$pre_commit_hook" ]; then
            cat > "$pre_commit_hook" << 'EOF'
#!/bin/bash
# Hook pre-commit pour vérification de la qualité du code

echo "Vérification de la qualité du code..."

# Activation de l'environnement virtuel
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Exécution de flake8 si disponible
if command -v flake8 &> /dev/null; then
    echo "Exécution de flake8..."
    flake8 backend/app --max-line-length=88 --extend-ignore=E203,W503
fi

# Exécution de black si disponible
if command -v black &> /dev/null; then
    echo "Vérification du formatage avec black..."
    black --check backend/app
fi

echo "Vérification terminée."
EOF
            chmod +x "$pre_commit_hook"
            log_success "Hook pre-commit configuré"
        fi
    fi
}

# Fonction de test de l'installation
test_installation() {
    log_info "Test de l'installation..."
    
    cd "$PROJECT_ROOT"
    
    # Activation de l'environnement virtuel
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    # Test d'import des modules principaux
    if python3 -c "import fastapi, sqlalchemy, redis" 2>/dev/null; then
        log_success "Modules principaux importés avec succès"
    else
        log_warning "Certains modules principaux ne peuvent pas être importés"
    fi
    
    # Test de connexion à la base de données
    local db_url="${DATABASE_URL:-postgresql://postgres:postgres@localhost:5432/spotify_ai_agent_dev}"
    if python3 -c "import sqlalchemy; sqlalchemy.create_engine('$db_url').connect()" 2>/dev/null; then
        log_success "Connexion à la base de données réussie"
    else
        log_warning "Impossible de se connecter à la base de données"
    fi
    
    log_success "Test de l'installation terminé"
}

# Fonction d'affichage du résumé
show_summary() {
    echo
    log_success "Configuration de l'environnement de développement terminée !"
    echo
    echo -e "${BLUE}Résumé de la configuration:${NC}"
    echo "================================"
    echo "• Environnement Python: $(python3 --version)"
    echo "• Répertoire du projet: $PROJECT_ROOT"
    echo "• Configuration: $ENV_DIR"
    echo "• Fichier d'environnement: $ENV_DIR/secrets/.env"
    echo
    echo -e "${YELLOW}Prochaines étapes:${NC}"
    echo "1. Modifiez le fichier .env avec vos paramètres spécifiques"
    echo "2. Vérifiez que PostgreSQL et Redis sont démarrés"
    echo "3. Exécutez './start_services.sh' pour démarrer l'application"
    echo "4. Visitez http://localhost:8000/docs pour l'API documentation"
    echo
    echo -e "${GREEN}Configuration terminée avec succès !${NC}"
}

# Fonction principale
main() {
    echo -e "${BLUE}==================================================================${NC}"
    echo -e "${BLUE}   Configuration de l'Environnement de Développement${NC}"
    echo -e "${BLUE}   Spotify AI Agent - Fahed Mlaiel & Team${NC}"
    echo -e "${BLUE}==================================================================${NC}"
    echo
    
    # Vérification des arguments
    if [[ $# -gt 0 && "$1" == "--help" ]]; then
        echo "Usage: $0 [options]"
        echo
        echo "Options:"
        echo "  --help           Affiche cette aide"
        echo "  --skip-db        Ignore la configuration de la base de données"
        echo "  --skip-redis     Ignore la configuration de Redis"
        echo "  --skip-deps      Ignore l'installation des dépendances"
        echo
        exit 0
    fi
    
    # Exécution des étapes de configuration
    check_prerequisites
    
    if [[ ! " $* " =~ " --skip-deps " ]]; then
        setup_python_environment
    fi
    
    setup_directories
    setup_environment_variables
    generate_dev_keys
    
    if [[ ! " $* " =~ " --skip-db " ]]; then
        setup_database
        init_database_migrations
    fi
    
    if [[ ! " $* " =~ " --skip-redis " ]]; then
        setup_redis
    fi
    
    validate_configuration
    setup_development_hooks
    test_installation
    show_summary
}

# Exécution du script principal
main "$@"
