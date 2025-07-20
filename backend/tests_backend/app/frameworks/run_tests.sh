#!/bin/bash

# 🧪 Script d'Exécution Tests Frameworks - Test Runner Script
# ==========================================================
# 
# Script automatisé pour exécuter tous les tests des frameworks
# avec différents modes et configurations
#
# Usage:
#   ./run_tests.sh [mode] [options]
#
# Modes:
#   unit        - Tests unitaires uniquement
#   integration - Tests d'intégration
#   performance - Tests de performance
#   all         - Tous les tests (défaut)
#   coverage    - Tests avec rapport de couverture
#
# Développé par: Toute l'équipe d'experts

set -e  # Arrêter en cas d'erreur

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
TESTS_DIR="$PROJECT_ROOT/backend/tests_backend/app/frameworks"
COVERAGE_DIR="$PROJECT_ROOT/coverage_reports"
LOG_DIR="$PROJECT_ROOT/test_logs"

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

# Créer répertoires nécessaires
setup_directories() {
    log_info "Création des répertoires de test..."
    mkdir -p "$COVERAGE_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "$TESTS_DIR/__pycache__"
}

# Vérifier dépendances
check_dependencies() {
    log_info "Vérification des dépendances..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 n'est pas installé"
        exit 1
    fi
    
    if ! python3 -c "import pytest" 2>/dev/null; then
        log_error "pytest n'est pas installé. Installation..."
        pip install pytest pytest-asyncio pytest-mock pytest-cov pytest-benchmark
    fi
    
    log_success "Dépendances vérifiées"
}

# Configuration environnement
setup_environment() {
    log_info "Configuration de l'environnement de test..."
    
    export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/backend:$PYTHONPATH"
    export TEST_MODE="$1"
    export CI="${CI:-false}"
    export FAST_TESTS="${FAST_TESTS:-false}"
    export SKIP_EXTERNAL_TESTS="${SKIP_EXTERNAL_TESTS:-false}"
    
    # Configuration Redis de test
    export REDIS_URL="redis://localhost:6379/15"
    
    # Configuration base de données de test
    export DATABASE_URL="sqlite:///test_frameworks.db"
    
    log_success "Environnement configuré pour mode: $1"
}

# Nettoyage avant tests
cleanup_before_tests() {
    log_info "Nettoyage des artefacts précédents..."
    
    # Supprimer fichiers de cache
    find "$TESTS_DIR" -name "*.pyc" -delete 2>/dev/null || true
    find "$TESTS_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # Supprimer base de données de test
    rm -f "$PROJECT_ROOT/test_frameworks.db" 2>/dev/null || true
    
    # Nettoyer logs précédents
    rm -f "$LOG_DIR"/*.log 2>/dev/null || true
    
    log_success "Nettoyage terminé"
}

# Tests unitaires
run_unit_tests() {
    log_info "🧪 Exécution des tests unitaires..."
    
    pytest \
        "$TESTS_DIR" \
        -v \
        --tb=short \
        -m "not integration and not performance and not external" \
        --asyncio-mode=auto \
        --log-file="$LOG_DIR/unit_tests.log" \
        --log-file-level=INFO \
        "$@"
}

# Tests d'intégration
run_integration_tests() {
    log_info "🔗 Exécution des tests d'intégration..."
    
    pytest \
        "$TESTS_DIR/test_integration.py" \
        -v \
        --tb=short \
        -m "integration" \
        --asyncio-mode=auto \
        --log-file="$LOG_DIR/integration_tests.log" \
        --log-file-level=INFO \
        "$@"
}

# Tests de performance
run_performance_tests() {
    log_info "⚡ Exécution des tests de performance..."
    
    pytest \
        "$TESTS_DIR" \
        -v \
        --tb=short \
        -m "performance" \
        --asyncio-mode=auto \
        --benchmark-only \
        --benchmark-sort=mean \
        --benchmark-json="$LOG_DIR/benchmark_results.json" \
        --log-file="$LOG_DIR/performance_tests.log" \
        "$@"
}

# Tests avec couverture
run_coverage_tests() {
    log_info "📊 Exécution des tests avec couverture..."
    
    pytest \
        "$TESTS_DIR" \
        -v \
        --tb=short \
        --cov="backend/app/frameworks" \
        --cov-report=html:"$COVERAGE_DIR/html" \
        --cov-report=term-missing \
        --cov-report=json:"$COVERAGE_DIR/coverage.json" \
        --cov-fail-under=85 \
        --asyncio-mode=auto \
        --log-file="$LOG_DIR/coverage_tests.log" \
        "$@"
}

# Tests par framework spécifique
run_framework_tests() {
    local framework="$1"
    shift
    
    log_info "🎯 Exécution des tests pour framework: $framework"
    
    case "$framework" in
        "core")
            pytest "$TESTS_DIR/test_core.py" -v --asyncio-mode=auto "$@"
            ;;
        "hybrid")
            pytest "$TESTS_DIR/test_hybrid_backend.py" -v --asyncio-mode=auto "$@"
            ;;
        "ml")
            pytest "$TESTS_DIR/test_ml_frameworks.py" -v --asyncio-mode=auto "$@"
            ;;
        "security")
            pytest "$TESTS_DIR/test_security.py" -v --asyncio-mode=auto "$@"
            ;;
        "monitoring")
            pytest "$TESTS_DIR/test_monitoring.py" -v --asyncio-mode=auto "$@"
            ;;
        "microservices")
            pytest "$TESTS_DIR/test_microservices.py" -v --asyncio-mode=auto "$@"
            ;;
        *)
            log_error "Framework inconnu: $framework"
            log_info "Frameworks disponibles: core, hybrid, ml, security, monitoring, microservices"
            exit 1
            ;;
    esac
}

# Tests complets
run_all_tests() {
    log_info "🚀 Exécution de tous les tests..."
    
    # Tests unitaires d'abord
    run_unit_tests --maxfail=5
    
    # Tests d'intégration
    run_integration_tests --maxfail=3
    
    # Tests de performance (optionnel en CI)
    if [[ "${CI}" != "true" ]]; then
        run_performance_tests --maxfail=1 || log_warning "Certains tests de performance ont échoué"
    fi
}

# Tests rapides
run_fast_tests() {
    log_info "⚡ Exécution des tests rapides..."
    
    export FAST_TESTS="true"
    export SKIP_EXTERNAL_TESTS="true"
    
    pytest \
        "$TESTS_DIR" \
        -v \
        --tb=line \
        -m "not slow and not external" \
        --asyncio-mode=auto \
        --maxfail=10 \
        -x \
        "$@"
}

# Tests avec retry pour CI
run_ci_tests() {
    log_info "🤖 Exécution des tests CI avec retry..."
    
    pytest \
        "$TESTS_DIR" \
        -v \
        --tb=short \
        --asyncio-mode=auto \
        --maxfail=5 \
        --reruns=2 \
        --reruns-delay=1 \
        --cov="backend/app/frameworks" \
        --cov-report=xml:"$COVERAGE_DIR/coverage.xml" \
        --cov-fail-under=80 \
        --junitxml="$LOG_DIR/junit.xml" \
        "$@"
}

# Rapport final
generate_report() {
    log_info "📋 Génération du rapport final..."
    
    local report_file="$LOG_DIR/test_report.txt"
    
    cat > "$report_file" << EOF
🧪 RAPPORT DE TESTS FRAMEWORKS SPOTIFY AI AGENT
==============================================

Date: $(date)
Mode: ${TEST_MODE:-all}
Environnement: ${CI:+CI}${CI:-local}

FICHIERS DE TESTS:
- Tests Core Framework: $(wc -l < "$TESTS_DIR/test_core.py" 2>/dev/null || echo "0") lignes
- Tests Hybrid Backend: $(wc -l < "$TESTS_DIR/test_hybrid_backend.py" 2>/dev/null || echo "0") lignes  
- Tests ML Frameworks: $(wc -l < "$TESTS_DIR/test_ml_frameworks.py" 2>/dev/null || echo "0") lignes
- Tests Security: $(wc -l < "$TESTS_DIR/test_security.py" 2>/dev/null || echo "0") lignes
- Tests Monitoring: $(wc -l < "$TESTS_DIR/test_monitoring.py" 2>/dev/null || echo "0") lignes
- Tests Microservices: $(wc -l < "$TESTS_DIR/test_microservices.py" 2>/dev/null || echo "0") lignes
- Tests Integration: $(wc -l < "$TESTS_DIR/test_integration.py" 2>/dev/null || echo "0") lignes

LOGS GÉNÉRÉS:
$(ls -la "$LOG_DIR"/*.log 2>/dev/null || echo "Aucun log généré")

COUVERTURE:
$(ls -la "$COVERAGE_DIR" 2>/dev/null || echo "Aucun rapport de couverture")

EOF
    
    log_success "Rapport généré: $report_file"
    cat "$report_file"
}

# Aide
show_help() {
    cat << EOF
🧪 Script d'Exécution Tests Frameworks - Spotify AI Agent

USAGE:
    $0 [MODE] [OPTIONS]

MODES:
    unit            Tests unitaires uniquement
    integration     Tests d'intégration
    performance     Tests de performance  
    coverage        Tests avec couverture de code
    fast            Tests rapides (sans tests lents)
    ci              Tests optimisés pour CI/CD
    all             Tous les tests (défaut)
    
    framework FRAMEWORK    Tests d'un framework spécifique
                          (core|hybrid|ml|security|monitoring|microservices)

OPTIONS:
    --help, -h      Afficher cette aide
    --verbose, -v   Mode verbeux
    --maxfail=N     Arrêter après N échecs
    --workers=N     Nombre de workers parallèles
    
VARIABLES D'ENVIRONNEMENT:
    CI=true                 Mode CI (désactive tests lents)
    FAST_TESTS=true        Tests rapides uniquement
    SKIP_EXTERNAL_TESTS=true   Ignorer tests nécessitant services externes
    TEST_WORKERS=N         Nombre de workers (défaut: 1 en CI, 4 sinon)

EXEMPLES:
    $0 unit                     # Tests unitaires
    $0 integration --verbose   # Tests intégration mode verbeux
    $0 framework ml            # Tests framework ML seulement
    $0 coverage                # Tests avec couverture
    $0 fast                    # Tests rapides
    
    CI=true $0 ci              # Tests optimisés CI
    FAST_TESTS=true $0 all     # Tous les tests en mode rapide

STRUCTURE TESTS:
    backend/tests_backend/app/frameworks/
    ├── __init__.py              # Configuration globale
    ├── conftest.py             # Fixtures pytest
    ├── test_core.py            # Tests orchestrateur
    ├── test_hybrid_backend.py  # Tests Django/FastAPI
    ├── test_ml_frameworks.py   # Tests ML/IA
    ├── test_security.py       # Tests sécurité
    ├── test_monitoring.py     # Tests monitoring
    ├── test_microservices.py  # Tests microservices
    └── test_integration.py    # Tests intégration

RAPPORTS GÉNÉRÉS:
    coverage_reports/           # Rapports de couverture
    test_logs/                 # Logs des tests
    
Développé par l'équipe d'experts Spotify AI Agent
EOF
}

# Point d'entrée principal
main() {
    local mode="${1:-all}"
    shift || true
    
    case "$mode" in
        "--help"|"-h"|"help")
            show_help
            exit 0
            ;;
        "unit")
            setup_directories
            check_dependencies
            setup_environment "unit"
            cleanup_before_tests
            run_unit_tests "$@"
            ;;
        "integration")
            setup_directories
            check_dependencies
            setup_environment "integration"
            cleanup_before_tests
            run_integration_tests "$@"
            ;;
        "performance")
            setup_directories
            check_dependencies
            setup_environment "performance"
            cleanup_before_tests
            run_performance_tests "$@"
            ;;
        "coverage")
            setup_directories
            check_dependencies
            setup_environment "coverage"
            cleanup_before_tests
            run_coverage_tests "$@"
            ;;
        "fast")
            setup_directories
            check_dependencies
            setup_environment "fast"
            cleanup_before_tests
            run_fast_tests "$@"
            ;;
        "ci")
            setup_directories
            check_dependencies
            setup_environment "ci"
            cleanup_before_tests
            run_ci_tests "$@"
            ;;
        "framework")
            local framework="$1"
            shift || true
            if [[ -z "$framework" ]]; then
                log_error "Framework requis pour mode 'framework'"
                show_help
                exit 1
            fi
            setup_directories
            check_dependencies
            setup_environment "framework"
            cleanup_before_tests
            run_framework_tests "$framework" "$@"
            ;;
        "all"|"")
            setup_directories
            check_dependencies
            setup_environment "all"
            cleanup_before_tests
            run_all_tests "$@"
            ;;
        *)
            log_error "Mode inconnu: $mode"
            show_help
            exit 1
            ;;
    esac
    
    # Rapport final
    generate_report
    
    log_success "🎉 Tests terminés avec succès!"
}

# Gestion des signaux
trap 'log_error "Test interrompu par signal"; exit 130' INT TERM

# Exécution
main "$@"
