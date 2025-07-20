#!/bin/bash

# 🎵 Spotify AI Agent - Script de Test Automation
# ==============================================
# 
# Script principal pour automatiser l'exécution 
# des tests du module Spleeter avec CI/CD.
#
# 🎖️ Développé par l'équipe d'experts enterprise

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
TEST_DIR="$BACKEND_DIR/tests_backend/spleeter"
SRC_DIR="$BACKEND_DIR/spleeter"

# Couleurs pour l'affichage
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Configuration des tests
PYTHON_CMD="python3"
PYTEST_CMD="pytest"
COVERAGE_MIN=85
TIMEOUT_UNIT=30
TIMEOUT_INTEGRATION=120
TIMEOUT_PERFORMANCE=300
TIMEOUT_STRESS=600

# Variables d'environnement pour les tests
export PYTHONPATH="$BACKEND_DIR:$PYTHONPATH"
export SPLEETER_TEST_MODE="true"
export SPLEETER_LOG_LEVEL="DEBUG"
export SPLEETER_CACHE_DISABLED="true"

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
    echo -e "\n${PURPLE}🎵 $1${NC}"
    echo -e "${PURPLE}$(printf '=%.0s' {1..50})${NC}"
}

# Vérification des prérequis
check_requirements() {
    log_header "Vérification des prérequis"
    
    # Vérifier Python
    if ! command -v $PYTHON_CMD &> /dev/null; then
        log_error "Python n'est pas installé ou non trouvé dans PATH"
        exit 1
    fi
    log_success "Python: $($PYTHON_CMD --version)"
    
    # Vérifier pytest
    if ! $PYTHON_CMD -m pytest --version &> /dev/null; then
        log_warning "pytest non trouvé, installation..."
        $PYTHON_CMD -m pip install pytest pytest-asyncio pytest-cov
    fi
    log_success "pytest: $($PYTHON_CMD -m pytest --version | head -n1)"
    
    # Vérifier la structure des répertoires
    if [[ ! -d "$TEST_DIR" ]]; then
        log_error "Répertoire de tests non trouvé: $TEST_DIR"
        exit 1
    fi
    
    if [[ ! -d "$SRC_DIR" ]]; then
        log_error "Répertoire source non trouvé: $SRC_DIR"
        exit 1
    fi
    
    log_success "Structure des répertoires validée"
}

# Installation des dépendances de test
install_test_dependencies() {
    log_header "Installation des dépendances de test"
    
    local requirements=(
        "pytest>=7.0.0"
        "pytest-asyncio>=0.21.0"
        "pytest-cov>=4.0.0"
        "pytest-mock>=3.10.0"
        "pytest-benchmark>=4.0.0"
        "pytest-html>=3.1.0"
        "pytest-xdist>=3.0.0"
        "pytest-timeout>=2.1.0"
        "coverage[toml]>=7.0.0"
        "black>=23.0.0"
        "flake8>=6.0.0"
        "mypy>=1.0.0"
        "isort>=5.12.0"
    )
    
    for req in "${requirements[@]}"; do
        log_info "Installation: $req"
        $PYTHON_CMD -m pip install "$req" --quiet
    done
    
    log_success "Toutes les dépendances installées"
}

# Nettoyage avant tests
cleanup_before_tests() {
    log_header "Nettoyage avant tests"
    
    # Supprimer les caches Python
    find "$TEST_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$SRC_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$TEST_DIR" -name "*.pyc" -delete 2>/dev/null || true
    find "$SRC_DIR" -name "*.pyc" -delete 2>/dev/null || true
    
    # Supprimer les anciens rapports
    rm -rf "$PROJECT_ROOT/coverage_html" 2>/dev/null || true
    rm -f "$PROJECT_ROOT/coverage.xml" 2>/dev/null || true
    rm -f "$PROJECT_ROOT/test_report.html" 2>/dev/null || true
    rm -f "$PROJECT_ROOT/.coverage" 2>/dev/null || true
    
    log_success "Nettoyage terminé"
}

# Tests unitaires
run_unit_tests() {
    log_header "Exécution des tests unitaires"
    
    cd "$TEST_DIR"
    
    local test_files=(
        "test_core.py"
        "test_models.py"
        "test_processor.py"
        "test_cache.py"
        "test_utils.py"
        "test_monitoring.py"
        "test_exceptions.py"
    )
    
    local failed_tests=()
    
    for test_file in "${test_files[@]}"; do
        if [[ -f "$test_file" ]]; then
            log_info "Test: $test_file"
            if ! $PYTHON_CMD -m pytest "$test_file" \
                -v --tb=short \
                --timeout="$TIMEOUT_UNIT" \
                -m "not slow and not performance and not stress"; then
                failed_tests+=("$test_file")
                log_error "Échec: $test_file"
            else
                log_success "Réussi: $test_file"
            fi
        else
            log_warning "Fichier non trouvé: $test_file"
        fi
    done
    
    if [[ ${#failed_tests[@]} -eq 0 ]]; then
        log_success "Tous les tests unitaires ont réussi"
        return 0
    else
        log_error "Tests échoués: ${failed_tests[*]}"
        return 1
    fi
}

# Tests d'intégration
run_integration_tests() {
    log_header "Exécution des tests d'intégration"
    
    cd "$TEST_DIR"
    
    if [[ ! -f "test_integration.py" ]]; then
        log_warning "Fichier test_integration.py non trouvé"
        return 0
    fi
    
    log_info "Exécution des tests d'intégration..."
    if $PYTHON_CMD -m pytest test_integration.py \
        -v --tb=short \
        --timeout="$TIMEOUT_INTEGRATION" \
        -m "integration"; then
        log_success "Tests d'intégration réussis"
        return 0
    else
        log_error "Échec des tests d'intégration"
        return 1
    fi
}

# Tests de performance
run_performance_tests() {
    log_header "Exécution des tests de performance"
    
    cd "$TEST_DIR"
    
    if [[ ! -f "test_performance.py" ]]; then
        log_warning "Fichier test_performance.py non trouvé"
        return 0
    fi
    
    log_info "Exécution des benchmarks de performance..."
    if $PYTHON_CMD -m pytest test_performance.py \
        -v --tb=short \
        --timeout="$TIMEOUT_PERFORMANCE" \
        -m "performance and not stress" \
        --benchmark-only; then
        log_success "Tests de performance réussis"
        return 0
    else
        log_error "Échec des tests de performance"
        return 1
    fi
}

# Tests de stress
run_stress_tests() {
    log_header "Exécution des tests de stress"
    
    cd "$TEST_DIR"
    
    if [[ ! -f "test_performance.py" ]]; then
        log_warning "Fichier test_performance.py non trouvé"
        return 0
    fi
    
    log_info "Exécution des tests de stress..."
    if $PYTHON_CMD -m pytest test_performance.py \
        -v --tb=short \
        --timeout="$TIMEOUT_STRESS" \
        -m "stress"; then
        log_success "Tests de stress réussis"
        return 0
    else
        log_error "Échec des tests de stress"
        return 1
    fi
}

# Analyse de couverture
run_coverage_analysis() {
    log_header "Analyse de couverture de code"
    
    cd "$TEST_DIR"
    
    log_info "Génération du rapport de couverture..."
    
    # Tests avec couverture
    $PYTHON_CMD -m pytest \
        --cov="$SRC_DIR" \
        --cov-report=term-missing \
        --cov-report=html:"$PROJECT_ROOT/coverage_html" \
        --cov-report=xml:"$PROJECT_ROOT/coverage.xml" \
        --cov-fail-under="$COVERAGE_MIN" \
        -m "not slow and not performance and not stress" \
        test_core.py test_models.py test_processor.py test_cache.py \
        test_utils.py test_monitoring.py test_exceptions.py
    
    local coverage_result=$?
    
    if [[ $coverage_result -eq 0 ]]; then
        log_success "Couverture de code: ≥${COVERAGE_MIN}%"
        log_info "Rapport HTML: $PROJECT_ROOT/coverage_html/index.html"
        log_info "Rapport XML: $PROJECT_ROOT/coverage.xml"
    else
        log_error "Couverture de code insuffisante (< ${COVERAGE_MIN}%)"
    fi
    
    return $coverage_result
}

# Vérifications de qualité de code
run_code_quality_checks() {
    log_header "Vérifications de qualité de code"
    
    local checks_passed=0
    local total_checks=4
    
    # Black (formatage)
    log_info "Vérification du formatage (black)..."
    if black --check --line-length=100 "$SRC_DIR" "$TEST_DIR" &>/dev/null; then
        log_success "Formatage: OK"
        ((checks_passed++))
    else
        log_warning "Formatage: Corrections nécessaires"
        log_info "Correction automatique: black --line-length=100 $SRC_DIR $TEST_DIR"
    fi
    
    # isort (imports)
    log_info "Vérification des imports (isort)..."
    if isort --check-only --profile=black "$SRC_DIR" "$TEST_DIR" &>/dev/null; then
        log_success "Imports: OK"
        ((checks_passed++))
    else
        log_warning "Imports: Corrections nécessaires"
        log_info "Correction automatique: isort --profile=black $SRC_DIR $TEST_DIR"
    fi
    
    # flake8 (linting)
    log_info "Vérification du linting (flake8)..."
    if flake8 --max-line-length=100 --ignore=E203,W503 "$SRC_DIR" "$TEST_DIR" &>/dev/null; then
        log_success "Linting: OK"
        ((checks_passed++))
    else
        log_warning "Linting: Avertissements détectés"
    fi
    
    # mypy (types)
    log_info "Vérification des types (mypy)..."
    if mypy --ignore-missing-imports "$SRC_DIR" &>/dev/null; then
        log_success "Types: OK"
        ((checks_passed++))
    else
        log_warning "Types: Annotations manquantes ou incorrectes"
    fi
    
    log_info "Vérifications réussies: $checks_passed/$total_checks"
    
    if [[ $checks_passed -eq $total_checks ]]; then
        log_success "Toutes les vérifications de qualité ont réussi"
        return 0
    else
        log_warning "Certaines vérifications de qualité ont échoué"
        return 1
    fi
}

# Génération de rapport détaillé
generate_detailed_report() {
    log_header "Génération du rapport détaillé"
    
    cd "$TEST_DIR"
    
    log_info "Génération du rapport HTML..."
    $PYTHON_CMD -m pytest \
        --html="$PROJECT_ROOT/test_report.html" \
        --self-contained-html \
        --tb=short \
        -v \
        -m "not slow and not performance and not stress"
    
    log_success "Rapport généré: $PROJECT_ROOT/test_report.html"
}

# Tests de smoke (vérification rapide)
run_smoke_tests() {
    log_header "Tests de smoke (vérification rapide)"
    
    cd "$TEST_DIR"
    
    log_info "Exécution des tests de base..."
    if $PYTHON_CMD -m pytest \
        -k "test_initialization or test_basic" \
        --maxfail=1 \
        --timeout=10 \
        -v; then
        log_success "Tests de smoke réussis"
        return 0
    else
        log_error "Échec des tests de smoke"
        return 1
    fi
}

# Pipeline CI/CD complet
run_ci_pipeline() {
    log_header "Pipeline CI/CD complet"
    
    local start_time=$(date +%s)
    local failed_stages=()
    
    # Étape 1: Prérequis
    if ! check_requirements; then
        failed_stages+=("requirements")
    fi
    
    # Étape 2: Installation des dépendances
    if ! install_test_dependencies; then
        failed_stages+=("dependencies")
    fi
    
    # Étape 3: Nettoyage
    if ! cleanup_before_tests; then
        failed_stages+=("cleanup")
    fi
    
    # Étape 4: Tests de smoke
    if ! run_smoke_tests; then
        failed_stages+=("smoke")
    fi
    
    # Étape 5: Tests unitaires
    if ! run_unit_tests; then
        failed_stages+=("unit")
    fi
    
    # Étape 6: Tests d'intégration
    if ! run_integration_tests; then
        failed_stages+=("integration")
    fi
    
    # Étape 7: Couverture de code
    if ! run_coverage_analysis; then
        failed_stages+=("coverage")
    fi
    
    # Étape 8: Qualité de code
    if ! run_code_quality_checks; then
        failed_stages+=("quality")
    fi
    
    # Étape 9: Rapport détaillé
    if ! generate_detailed_report; then
        failed_stages+=("report")
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_header "Résumé du pipeline CI/CD"
    log_info "Durée totale: ${duration}s"
    
    if [[ ${#failed_stages[@]} -eq 0 ]]; then
        log_success "🎉 Pipeline CI/CD réussi - Toutes les étapes ont réussi!"
        return 0
    else
        log_error "❌ Pipeline CI/CD échoué - Étapes échouées: ${failed_stages[*]}"
        return 1
    fi
}

# Tests de performance uniquement
run_performance_only() {
    log_header "Tests de performance et stress"
    
    local failed_tests=()
    
    if ! run_performance_tests; then
        failed_tests+=("performance")
    fi
    
    if ! run_stress_tests; then
        failed_tests+=("stress")
    fi
    
    if [[ ${#failed_tests[@]} -eq 0 ]]; then
        log_success "Tous les tests de performance ont réussi"
        return 0
    else
        log_error "Tests échoués: ${failed_tests[*]}"
        return 1
    fi
}

# Affichage de l'aide
show_help() {
    cat << EOF
🎵 Spotify AI Agent - Script de Test Automation

Usage: $0 [OPTION]

Options disponibles:
  --unit              Exécute les tests unitaires uniquement
  --integration       Exécute les tests d'intégration uniquement
  --performance       Exécute les tests de performance uniquement
  --stress            Exécute les tests de stress uniquement
  --coverage          Exécute l'analyse de couverture uniquement
  --quality           Exécute les vérifications de qualité uniquement
  --smoke             Exécute les tests de smoke uniquement
  --ci                Exécute le pipeline CI/CD complet
  --performance-only  Exécute tous les tests de performance
  --install-deps      Installe les dépendances de test
  --cleanup           Nettoie les fichiers temporaires
  --help              Affiche cette aide

Variables d'environnement:
  COVERAGE_MIN        Couverture minimale requise (défaut: $COVERAGE_MIN)
  TIMEOUT_UNIT        Timeout pour tests unitaires (défaut: $TIMEOUT_UNIT)
  TIMEOUT_INTEGRATION Timeout pour tests d'intégration (défaut: $TIMEOUT_INTEGRATION)
  TIMEOUT_PERFORMANCE Timeout pour tests de performance (défaut: $TIMEOUT_PERFORMANCE)
  TIMEOUT_STRESS      Timeout pour tests de stress (défaut: $TIMEOUT_STRESS)

Exemples:
  $0 --ci              # Pipeline complet
  $0 --unit            # Tests unitaires seulement
  $0 --performance     # Tests de performance seulement
  $0 --smoke           # Vérification rapide

EOF
}

# Fonction principale
main() {
    case "${1:-}" in
        --unit)
            check_requirements
            cleanup_before_tests
            run_unit_tests
            ;;
        --integration)
            check_requirements
            cleanup_before_tests
            run_integration_tests
            ;;
        --performance)
            check_requirements
            cleanup_before_tests
            run_performance_tests
            ;;
        --stress)
            check_requirements
            cleanup_before_tests
            run_stress_tests
            ;;
        --coverage)
            check_requirements
            cleanup_before_tests
            run_coverage_analysis
            ;;
        --quality)
            check_requirements
            run_code_quality_checks
            ;;
        --smoke)
            check_requirements
            run_smoke_tests
            ;;
        --ci)
            run_ci_pipeline
            ;;
        --performance-only)
            check_requirements
            cleanup_before_tests
            run_performance_only
            ;;
        --install-deps)
            install_test_dependencies
            ;;
        --cleanup)
            cleanup_before_tests
            ;;
        --help)
            show_help
            ;;
        "")
            log_info "Exécution des tests de base..."
            check_requirements
            cleanup_before_tests
            run_unit_tests
            ;;
        *)
            log_error "Option inconnue: $1"
            show_help
            exit 1
            ;;
    esac
}

# Point d'entrée du script
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
