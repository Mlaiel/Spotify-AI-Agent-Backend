#!/bin/bash

# Script d'installation automatisé pour Alertmanager Receivers Config Ultra-Avancé
# Author: Spotify AI Agent Team - Fahed Mlaiel
# Usage: ./install.sh [options]

set -euo pipefail

# Configuration par défaut
INSTALL_DIR="/opt/alertmanager-receivers-config"
PYTHON_VERSION="3.11"
DOCKER_INSTALL="yes"
DEV_MODE="no"
BACKUP_CONFIG="yes"
VERBOSE="no"

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction d'affichage avec couleurs
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Fonction d'aide
show_help() {
    cat << EOF
Installation d'Alertmanager Receivers Config Ultra-Avancé

Usage: $0 [OPTIONS]

Options:
    -d, --install-dir DIR     Répertoire d'installation (défaut: $INSTALL_DIR)
    -p, --python-version VER  Version Python (défaut: $PYTHON_VERSION)
    --no-docker              Ne pas installer Docker
    --dev-mode               Mode développement
    --no-backup              Ne pas sauvegarder la config existante
    -v, --verbose            Mode verbeux
    -h, --help               Affiche cette aide

Exemples:
    $0                                    # Installation standard
    $0 --dev-mode --verbose             # Installation en mode développement
    $0 -d /usr/local/alertmanager       # Installation dans un répertoire spécifique

EOF
}

# Parse des arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        -p|--python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --no-docker)
            DOCKER_INSTALL="no"
            shift
            ;;
        --dev-mode)
            DEV_MODE="yes"
            shift
            ;;
        --no-backup)
            BACKUP_CONFIG="no"
            shift
            ;;
        -v|--verbose)
            VERBOSE="yes"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Option inconnue: $1"
            ;;
    esac
done

# Activation du mode verbeux
if [[ "$VERBOSE" == "yes" ]]; then
    set -x
fi

# Vérifications préliminaires
check_prerequisites() {
    print_info "Vérification des prérequis..."
    
    # Vérification de l'OS
    if [[ "$OSTYPE" != "linux-gnu"* && "$OSTYPE" != "darwin"* ]]; then
        print_error "OS non supporté: $OSTYPE"
    fi
    
    # Vérification des droits root/sudo
    if [[ $EUID -eq 0 ]]; then
        print_warning "Exécution en tant que root détectée"
    elif ! sudo -n true 2>/dev/null; then
        print_error "Droits sudo requis pour l'installation"
    fi
    
    # Vérification de l'espace disque (minimum 2GB)
    available_space=$(df / | awk 'NR==2 {print $4}')
    required_space=2097152  # 2GB en KB
    
    if [[ $available_space -lt $required_space ]]; then
        print_error "Espace disque insuffisant. Requis: 2GB, disponible: $(($available_space/1024/1024))GB"
    fi
    
    print_success "Prérequis vérifiés"
}

# Installation des dépendances système
install_system_dependencies() {
    print_info "Installation des dépendances système..."
    
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        sudo apt-get update
        sudo apt-get install -y \
            curl \
            wget \
            git \
            build-essential \
            libssl-dev \
            libffi-dev \
            python3-dev \
            python3-pip \
            python3-venv \
            postgresql-client \
            redis-tools \
            jq
            
    elif command -v yum &> /dev/null; then
        # CentOS/RHEL
        sudo yum update -y
        sudo yum install -y \
            curl \
            wget \
            git \
            gcc \
            openssl-devel \
            libffi-devel \
            python3-devel \
            python3-pip \
            postgresql \
            redis \
            jq
            
    elif command -v brew &> /dev/null; then
        # macOS
        brew update
        brew install \
            python@$PYTHON_VERSION \
            postgresql \
            redis \
            jq
            
    else
        print_error "Gestionnaire de paquets non supporté"
    fi
    
    print_success "Dépendances système installées"
}

# Installation de Docker
install_docker() {
    if [[ "$DOCKER_INSTALL" == "no" ]]; then
        print_info "Installation de Docker ignorée"
        return
    fi
    
    print_info "Installation de Docker..."
    
    if ! command -v docker &> /dev/null; then
        # Installation de Docker
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        rm get-docker.sh
        
        # Ajout de l'utilisateur au groupe docker
        sudo usermod -aG docker $USER
        
        print_warning "Vous devez vous reconnecter pour utiliser Docker sans sudo"
    else
        print_info "Docker déjà installé"
    fi
    
    # Installation de Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    else
        print_info "Docker Compose déjà installé"
    fi
    
    print_success "Docker installé"
}

# Installation de Python et des dépendances
install_python() {
    print_info "Installation de Python $PYTHON_VERSION..."
    
    # Vérification de la version Python
    if command -v python$PYTHON_VERSION &> /dev/null; then
        python_version=$(python$PYTHON_VERSION --version | cut -d' ' -f2)
        print_info "Python $python_version détecté"
    else
        print_error "Python $PYTHON_VERSION non trouvé"
    fi
    
    # Mise à jour de pip
    python$PYTHON_VERSION -m pip install --upgrade pip
    
    print_success "Python configuré"
}

# Sauvegarde de la configuration existante
backup_existing_config() {
    if [[ "$BACKUP_CONFIG" == "no" ]]; then
        return
    fi
    
    if [[ -d "$INSTALL_DIR" ]]; then
        print_info "Sauvegarde de la configuration existante..."
        
        backup_dir="${INSTALL_DIR}.backup.$(date +%Y%m%d_%H%M%S)"
        sudo cp -r "$INSTALL_DIR" "$backup_dir"
        
        print_success "Configuration sauvegardée dans $backup_dir"
    fi
}

# Création du répertoire d'installation
create_install_directory() {
    print_info "Création du répertoire d'installation: $INSTALL_DIR"
    
    sudo mkdir -p "$INSTALL_DIR"
    sudo chown $USER:$USER "$INSTALL_DIR"
    
    print_success "Répertoire d'installation créé"
}

# Clonage/Copie des fichiers
copy_application_files() {
    print_info "Copie des fichiers de l'application..."
    
    # Copie de tous les fichiers du répertoire courant
    cp -r ./* "$INSTALL_DIR/"
    
    # Création des répertoires nécessaires
    mkdir -p "$INSTALL_DIR"/{data,logs,temp,config,backups}
    mkdir -p "$INSTALL_DIR"/config/{environments,security,templates}
    
    # Configuration des permissions
    chmod +x "$INSTALL_DIR"/*.sh
    
    print_success "Fichiers de l'application copiés"
}

# Installation des dépendances Python
install_python_dependencies() {
    print_info "Installation des dépendances Python..."
    
    cd "$INSTALL_DIR"
    
    # Création de l'environnement virtuel
    python$PYTHON_VERSION -m venv venv
    source venv/bin/activate
    
    # Installation des dépendances
    pip install --upgrade pip
    pip install -r requirements.txt
    
    if [[ "$DEV_MODE" == "yes" ]]; then
        if [[ -f "requirements-dev.txt" ]]; then
            pip install -r requirements-dev.txt
        fi
    fi
    
    print_success "Dépendances Python installées"
}

# Configuration des services système
configure_systemd_service() {
    print_info "Configuration du service systemd..."
    
    cat > /tmp/alertmanager-receivers-config.service << EOF
[Unit]
Description=Alertmanager Receivers Config Ultra-Avancé
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment=PATH=$INSTALL_DIR/venv/bin
ExecStart=$INSTALL_DIR/venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8080
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    sudo mv /tmp/alertmanager-receivers-config.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable alertmanager-receivers-config
    
    print_success "Service systemd configuré"
}

# Configuration de la base de données
setup_database() {
    print_info "Configuration de la base de données..."
    
    # Création de la base de données PostgreSQL (si PostgreSQL est installé localement)
    if command -v psql &> /dev/null; then
        sudo -u postgres createdb alertmanager_config 2>/dev/null || true
        sudo -u postgres psql -c "CREATE USER alertmanager WITH PASSWORD 'secure_password';" 2>/dev/null || true
        sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE alertmanager_config TO alertmanager;" 2>/dev/null || true
    fi
    
    print_success "Base de données configurée"
}

# Configuration initiale
initial_configuration() {
    print_info "Configuration initiale..."
    
    cd "$INSTALL_DIR"
    
    # Génération des clés de sécurité
    if [[ ! -f "config/security/secret.key" ]]; then
        mkdir -p config/security
        openssl rand -base64 32 > config/security/secret.key
        chmod 600 config/security/secret.key
    fi
    
    # Configuration par défaut
    if [[ ! -f "config/app.yml" ]]; then
        cat > config/app.yml << EOF
# Configuration par défaut
app:
  name: "Alertmanager Receivers Config"
  version: "2.1.0"
  debug: $([[ "$DEV_MODE" == "yes" ]] && echo "true" || echo "false")
  
database:
  url: "postgresql://alertmanager:secure_password@localhost:5432/alertmanager_config"
  
redis:
  url: "redis://localhost:6379/0"
  
security:
  encryption_enabled: true
  secret_key_file: "config/security/secret.key"
  
monitoring:
  prometheus_enabled: true
  metrics_port: 9090
EOF
    fi
    
    print_success "Configuration initiale terminée"
}

# Tests post-installation
run_post_install_tests() {
    print_info "Exécution des tests post-installation..."
    
    cd "$INSTALL_DIR"
    source venv/bin/activate
    
    # Test d'import Python
    python -c "import main; print('Import réussi')" || print_error "Échec de l'import Python"
    
    # Test de configuration
    python -c "from config import load_config; load_config(); print('Configuration valide')" || print_warning "Configuration nécessite une révision"
    
    print_success "Tests post-installation terminés"
}

# Instructions finales
show_final_instructions() {
    print_success "Installation terminée avec succès!"
    
    echo ""
    echo -e "${GREEN}Configuration installée dans:${NC} $INSTALL_DIR"
    echo ""
    echo -e "${YELLOW}Prochaines étapes:${NC}"
    echo "1. Démarrer les services:"
    echo "   sudo systemctl start alertmanager-receivers-config"
    echo ""
    echo "2. Démarrer avec Docker Compose:"
    echo "   cd $INSTALL_DIR && make compose-up"
    echo ""
    echo "3. Vérifier le statut:"
    echo "   curl http://localhost:8080/health"
    echo ""
    echo "4. Accéder aux interfaces:"
    echo "   - API: http://localhost:8080"
    echo "   - Métriques: http://localhost:9090"
    echo "   - Grafana: http://localhost:3000"
    echo ""
    echo -e "${YELLOW}Documentation:${NC}"
    echo "   - README: $INSTALL_DIR/README.md"
    echo "   - Configuration: $INSTALL_DIR/config/"
    echo "   - Logs: $INSTALL_DIR/logs/"
    echo ""
    
    if [[ "$DEV_MODE" == "yes" ]]; then
        echo -e "${BLUE}Mode développement activé:${NC}"
        echo "   - make dev        # Lancer en mode dev"
        echo "   - make test       # Lancer les tests"
        echo "   - make lint       # Vérifier la qualité"
        echo ""
    fi
}

# Fonction principale
main() {
    print_info "=== Installation d'Alertmanager Receivers Config Ultra-Avancé ==="
    print_info "Auteur: Spotify AI Agent Team - Fahed Mlaiel"
    echo ""
    
    check_prerequisites
    backup_existing_config
    install_system_dependencies
    install_docker
    install_python
    create_install_directory
    copy_application_files
    install_python_dependencies
    configure_systemd_service
    setup_database
    initial_configuration
    run_post_install_tests
    show_final_instructions
    
    print_success "Installation complète terminée!"
}

# Exécution du script principal
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
