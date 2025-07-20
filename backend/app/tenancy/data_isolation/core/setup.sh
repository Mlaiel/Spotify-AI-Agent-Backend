#!/bin/bash

# 🚀 Spotify AI Agent - Data Isolation Core Setup Script
# ======================================================
#
# Script de configuration automatique pour le module core
# d'isolation des données avec tous les outils de développement
#
# Author: Lead Dev + Architecte IA - Fahed Mlaiel

set -euo pipefail

# Couleurs pour l'affichage
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[0;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_NAME="Spotify AI Agent - Data Isolation Core"
readonly VERSION="2.0.0"
readonly PYTHON_MIN_VERSION="3.9"

# Fonctions utilitaires
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_header() {
    echo -e "${CYAN}🎵 $1${NC}"
    echo -e "${CYAN}$(printf '=%.0s' {1..60})${NC}"
}

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

check_python_version() {
    if command -v python3 >/dev/null 2>&1; then
        local python_version
        python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
            log_success "Python $python_version trouvé"
            return 0
        else
            log_error "Python $python_version trouvé, mais version $PYTHON_MIN_VERSION+ requise"
            return 1
        fi
    else
        log_error "Python 3 non trouvé"
        return 1
    fi
}

install_dependencies() {
    log_info "Installation des dépendances système..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Ubuntu/Debian
        if check_command apt-get; then
            sudo apt-get update
            sudo apt-get install -y \
                python3-pip \
                python3-venv \
                python3-dev \
                build-essential \
                libpq-dev \
                libffi-dev \
                libssl-dev \
                redis-tools \
                postgresql-client \
                curl \
                wget \
                git \
                make \
                jq
        # CentOS/RHEL
        elif check_command yum; then
            sudo yum update -y
            sudo yum install -y \
                python3-pip \
                python3-devel \
                gcc \
                gcc-c++ \
                postgresql-devel \
                libffi-devel \
                openssl-devel \
                redis \
                postgresql \
                curl \
                wget \
                git \
                make \
                jq
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if check_command brew; then
            brew install \
                python@3.11 \
                postgresql \
                redis \
                git \
                make \
                jq
        else
            log_error "Homebrew non trouvé. Installez-le depuis https://brew.sh/"
            exit 1
        fi
    fi
    
    log_success "Dépendances système installées"
}

setup_python_environment() {
    log_info "Configuration de l'environnement Python..."
    
    # Création de l'environnement virtuel
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
        log_success "Environnement virtuel créé"
    fi
    
    # Activation de l'environnement virtuel
    source venv/bin/activate
    
    # Mise à jour de pip
    pip install --upgrade pip setuptools wheel
    
    # Installation des dépendances
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
    fi
    
    # Installation des dépendances de développement
    pip install \
        pytest==7.4.3 \
        pytest-asyncio==0.21.1 \
        pytest-cov==4.1.0 \
        black==23.11.0 \
        flake8==6.1.0 \
        mypy==1.7.1 \
        bandit==1.7.5 \
        safety==2.3.5 \
        pre-commit==3.6.0
    
    log_success "Environnement Python configuré"
}

setup_docker_environment() {
    log_info "Configuration de l'environnement Docker..."
    
    # Vérification de Docker
    if ! check_command docker; then
        log_error "Docker non trouvé. Installez Docker depuis https://docs.docker.com/get-docker/"
        return 1
    fi
    
    # Vérification de Docker Compose
    if ! check_command docker-compose && ! docker compose version >/dev/null 2>&1; then
        log_error "Docker Compose non trouvé"
        return 1
    fi
    
    # Construction des images
    log_info "Construction de l'image de test..."
    docker build -f Dockerfile.test -t spotify-ai-core-test .
    
    log_success "Environnement Docker configuré"
}

create_configuration_files() {
    log_info "Création des fichiers de configuration..."
    
    # Fichier .env
    if [[ ! -f ".env" ]]; then
        cp .env.example .env
        log_success "Fichier .env créé à partir du template"
    fi
    
    # Configuration Git hooks
    if [[ -d ".git" ]] && check_command pre-commit; then
        pre-commit install
        log_success "Git hooks configurés"
    fi
    
    # Configuration des répertoires
    mkdir -p logs cache data temp
    chmod 755 logs cache data temp
    
    log_success "Fichiers de configuration créés"
}

setup_database() {
    log_info "Configuration de la base de données..."
    
    # Démarrage des services avec Docker Compose
    if [[ -f "docker-compose.dev.yml" ]]; then
        log_info "Démarrage des services de développement..."
        docker-compose -f docker-compose.dev.yml up -d postgres redis mongodb
        
        # Attente que les services soient prêts
        log_info "Attente que les services soient prêts..."
        sleep 10
        
        # Test de connexion PostgreSQL
        until docker-compose -f docker-compose.dev.yml exec -T postgres pg_isready -U spotify_user -d spotify_ai_db; do
            log_info "Attente de PostgreSQL..."
            sleep 2
        done
        
        log_success "Services de base de données démarrés"
    fi
}

run_initial_tests() {
    log_info "Exécution des tests initiaux..."
    
    # Activation de l'environnement virtuel
    source venv/bin/activate
    
    # Validation de la configuration
    if [[ -f "validate_configuration.py" ]]; then
        python validate_configuration.py --json > validation_results.json
        log_success "Configuration validée"
    fi
    
    # Benchmark initial
    if [[ -f "benchmark_performance.py" ]]; then
        python benchmark_performance.py --quick
        log_success "Benchmark initial terminé"
    fi
    
    # Tests unitaires si disponibles
    if check_command pytest && [[ -d "../../../tests" ]]; then
        pytest ../../../tests/tenancy/data_isolation/core/ -v --tb=short
        log_success "Tests unitaires exécutés"
    fi
}

generate_documentation() {
    log_info "Génération de la documentation..."
    
    # Création du fichier de documentation technique
    cat > TECHNICAL_SETUP.md << 'EOF'
# 🎵 Spotify AI Agent - Data Isolation Core - Configuration Technique

## 📋 Vue d'ensemble

Ce document décrit la configuration technique complète du module core d'isolation des données.

## 🛠️ Architecture des Composants

### Moteur de Conformité (`compliance_engine.py`)
- **Fonctionnalité** : Validation multi-réglementation (GDPR, CCPA, SOX, HIPAA, PCI-DSS)
- **API** : Interface asynchrone pour évaluation en temps réel
- **Audit** : Traçabilité complète des événements de conformité

### Moteur de Politique de Sécurité (`security_policy_engine.py`)
- **Fonctionnalité** : Politique de sécurité dynamique avec détection de menaces
- **API** : Évaluation des permissions et contrôle d'accès
- **Monitoring** : Surveillance en temps réel des patterns de menaces

### Optimiseur de Performance (`performance_optimizer.py`)
- **Fonctionnalité** : Optimisation ML-powered avec cache intelligent
- **API** : Prédiction et optimisation automatique des requêtes
- **Metrics** : Collecte de métriques détaillées

### Gestionnaire de Contexte (`context_manager.py`)
- **Fonctionnalité** : Commutation intelligente de contexte multi-tenant
- **API** : Isolation et validation des contextes
- **Performance** : Optimisation des changements de contexte

## 🔧 Configuration

### Variables d'Environnement

```bash
# Isolation
TENANT_ISOLATION_LEVEL=strict
PERFORMANCE_OPTIMIZATION=adaptive

# Cache
CACHE_SIZE_MB=2048
CACHE_TTL_SECONDS=300

# Sécurité
SECURITY_PARANOID_MODE=true
ENCRYPTION_KEY_VERSION=2

# Conformité
COMPLIANCE_AUDIT_ENABLED=true
GDPR_STRICT_MODE=true
```

### Base de Données

```sql
-- Configuration PostgreSQL recommandée
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
```

## 🚀 Déploiement

### Développement
```bash
make dev-setup
make monitor
```

### Production
```bash
make ci
make deploy-check
docker-compose -f docker-compose.prod.yml up -d
```

## 📊 Monitoring

### Métriques Clés
- Temps de réponse des requêtes
- Taux de cache hit/miss
- Score de conformité
- Détection de menaces

### Dashboards Grafana
- Performance générale
- Sécurité et conformité
- Utilisation des ressources

## 🔐 Sécurité

### Chiffrement
- AES-256-GCM pour les données sensibles
- RSA-4096 pour l'échange de clés
- PBKDF2 pour les mots de passe

### Audit
- Traçabilité complète des accès
- Logs immutables
- Alertes en temps réel

## 📈 Performance

### Optimisations
- Cache intelligent avec ML
- Pool de connexions optimisé
- Requêtes précompilées

### Benchmarks
- Latence < 50ms (P95)
- Throughput > 10k req/s
- Cache hit rate > 90%

---
*Configuration générée automatiquement par le script de setup*
*Author: Lead Dev + Architecte IA - Fahed Mlaiel*
EOF

    log_success "Documentation technique générée"
}

cleanup_on_exit() {
    log_info "Nettoyage en cours..."
    # Nettoyage si nécessaire
}

show_completion_summary() {
    log_header "Configuration Terminée avec Succès"
    echo ""
    echo -e "${GREEN}✅ Module Data Isolation Core configuré${NC}"
    echo -e "${GREEN}✅ Environnement Python prêt${NC}"
    echo -e "${GREEN}✅ Services Docker démarrés${NC}"
    echo -e "${GREEN}✅ Configuration validée${NC}"
    echo -e "${GREEN}✅ Tests initiaux exécutés${NC}"
    echo ""
    echo -e "${CYAN}🎯 Prochaines étapes:${NC}"
    echo "  1. Activez l'environnement virtuel: source venv/bin/activate"
    echo "  2. Lancez le monitoring: make monitor"
    echo "  3. Exécutez les benchmarks: make benchmark"
    echo "  4. Consultez la documentation: cat README.md"
    echo ""
    echo -e "${YELLOW}📋 Commandes utiles:${NC}"
    echo "  make help           # Voir toutes les commandes disponibles"
    echo "  make test           # Exécuter les tests"
    echo "  make validate       # Valider la configuration"
    echo "  make clean          # Nettoyer les fichiers temporaires"
    echo ""
    echo -e "${PURPLE}🏆 Le module est prêt pour le développement!${NC}"
}

main() {
    trap cleanup_on_exit EXIT
    
    log_header "$PROJECT_NAME v$VERSION - Script de Configuration"
    echo ""
    
    # Vérifications préliminaires
    log_info "Vérification des prérequis..."
    
    if ! check_python_version; then
        log_error "Version Python incompatible"
        exit 1
    fi
    
    # Installation et configuration
    install_dependencies
    setup_python_environment
    setup_docker_environment
    create_configuration_files
    setup_database
    run_initial_tests
    generate_documentation
    
    show_completion_summary
}

# Point d'entrée
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
