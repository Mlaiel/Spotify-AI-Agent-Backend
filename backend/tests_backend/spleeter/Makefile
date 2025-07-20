# 🎵 Spotify AI Agent - Makefile Tests Spleeter
# ============================================
# 
# Makefile pour automatiser l'exécution des tests
# du module Spleeter avec différentes options.
#
# 🎖️ Développé par l'équipe d'experts enterprise

# Variables de configuration
PYTHON = python3
PYTEST = pytest
TEST_DIR = /workspaces/Achiri/spotify-ai-agent/backend/tests_backend/spleeter
SRC_DIR = /workspaces/Achiri/spotify-ai-agent/backend/spleeter
COVERAGE_MIN = 85
PYTEST_ARGS = -v --tb=short --strict-markers

# Couleurs pour l'affichage
BLUE = \033[0;34m
GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
NC = \033[0m # No Color

.PHONY: help test test-unit test-integration test-performance test-stress test-all
.PHONY: test-fast test-slow coverage coverage-html coverage-xml
.PHONY: lint format check-style install-deps clean
.PHONY: test-cache test-core test-models test-processor test-utils test-monitoring test-exceptions

# Aide par défaut
help:
	@echo "$(BLUE)🎵 Spotify AI Agent - Tests Spleeter$(NC)"
	@echo "$(BLUE)=====================================$(NC)"
	@echo ""
	@echo "$(GREEN)Commandes de test disponibles :$(NC)"
	@echo "  $(YELLOW)test$(NC)              - Exécute tous les tests basiques"
	@echo "  $(YELLOW)test-unit$(NC)         - Tests unitaires uniquement"
	@echo "  $(YELLOW)test-integration$(NC)  - Tests d'intégration"
	@echo "  $(YELLOW)test-performance$(NC)  - Tests de performance"
	@echo "  $(YELLOW)test-stress$(NC)       - Tests de stress"
	@echo "  $(YELLOW)test-all$(NC)          - Tous les tests (long)"
	@echo "  $(YELLOW)test-fast$(NC)         - Tests rapides seulement"
	@echo "  $(YELLOW)test-slow$(NC)         - Tests lents seulement"
	@echo ""
	@echo "$(GREEN)Tests par module :$(NC)"
	@echo "  $(YELLOW)test-core$(NC)         - Tests du moteur principal"
	@echo "  $(YELLOW)test-models$(NC)       - Tests gestion modèles"
	@echo "  $(YELLOW)test-processor$(NC)    - Tests traitement audio"
	@echo "  $(YELLOW)test-cache$(NC)        - Tests système cache"
	@echo "  $(YELLOW)test-utils$(NC)        - Tests utilitaires"
	@echo "  $(YELLOW)test-monitoring$(NC)   - Tests monitoring"
	@echo "  $(YELLOW)test-exceptions$(NC)   - Tests gestion erreurs"
	@echo ""
	@echo "$(GREEN)Couverture de code :$(NC)"
	@echo "  $(YELLOW)coverage$(NC)          - Rapport de couverture console"
	@echo "  $(YELLOW)coverage-html$(NC)     - Rapport HTML détaillé"
	@echo "  $(YELLOW)coverage-xml$(NC)      - Rapport XML pour CI/CD"
	@echo ""
	@echo "$(GREEN)Qualité de code :$(NC)"
	@echo "  $(YELLOW)lint$(NC)              - Vérifications linting"
	@echo "  $(YELLOW)format$(NC)            - Formatage automatique"
	@echo "  $(YELLOW)check-style$(NC)       - Vérification style"
	@echo ""
	@echo "$(GREEN)Utilitaires :$(NC)"
	@echo "  $(YELLOW)install-deps$(NC)      - Installation dépendances test"
	@echo "  $(YELLOW)clean$(NC)             - Nettoyage fichiers temporaires"

# Installation des dépendances de test
install-deps:
	@echo "$(BLUE)📦 Installation des dépendances de test...$(NC)"
	$(PYTHON) -m pip install pytest pytest-asyncio pytest-cov pytest-mock pytest-benchmark
	$(PYTHON) -m pip install black flake8 mypy isort
	$(PYTHON) -m pip install coverage[toml] pytest-html pytest-xdist
	@echo "$(GREEN)✅ Dépendances installées$(NC)"

# Tests de base (rapides)
test:
	@echo "$(BLUE)🧪 Exécution des tests de base...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) $(PYTEST_ARGS) \
		--ignore=test_performance.py \
		--ignore=test_integration.py \
		-m "not slow and not performance and not stress"

# Tests unitaires seulement
test-unit:
	@echo "$(BLUE)🔬 Tests unitaires...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) $(PYTEST_ARGS) \
		test_core.py test_models.py test_processor.py test_cache.py \
		test_utils.py test_monitoring.py test_exceptions.py \
		-m "not slow"

# Tests d'intégration
test-integration:
	@echo "$(BLUE)🔗 Tests d'intégration...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) $(PYTEST_ARGS) \
		test_integration.py \
		-m "integration"

# Tests de performance
test-performance:
	@echo "$(BLUE)⚡ Tests de performance...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) $(PYTEST_ARGS) \
		test_performance.py \
		-m "performance" \
		--benchmark-only

# Tests de stress
test-stress:
	@echo "$(BLUE)💪 Tests de stress...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) $(PYTEST_ARGS) \
		test_performance.py \
		-m "stress" \
		--timeout=300

# Tous les tests (complet)
test-all:
	@echo "$(BLUE)🎯 Exécution de tous les tests...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) $(PYTEST_ARGS) \
		--timeout=600 \
		--maxfail=5

# Tests rapides seulement
test-fast:
	@echo "$(BLUE)🚀 Tests rapides...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) $(PYTEST_ARGS) \
		-m "not slow and not performance and not stress" \
		--timeout=30

# Tests lents seulement
test-slow:
	@echo "$(BLUE)🐌 Tests lents...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) $(PYTEST_ARGS) \
		-m "slow or performance or stress" \
		--timeout=300

# Tests par module spécifique
test-core:
	@echo "$(BLUE)🚀 Tests moteur principal...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) $(PYTEST_ARGS) test_core.py -v

test-models:
	@echo "$(BLUE)🤖 Tests gestion modèles...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) $(PYTEST_ARGS) test_models.py -v

test-processor:
	@echo "$(BLUE)🎵 Tests traitement audio...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) $(PYTEST_ARGS) test_processor.py -v

test-cache:
	@echo "$(BLUE)💾 Tests système cache...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) $(PYTEST_ARGS) test_cache.py -v

test-utils:
	@echo "$(BLUE)🔧 Tests utilitaires...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) $(PYTEST_ARGS) test_utils.py -v

test-monitoring:
	@echo "$(BLUE)📊 Tests monitoring...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) $(PYTEST_ARGS) test_monitoring.py -v

test-exceptions:
	@echo "$(BLUE)⚠️ Tests gestion erreurs...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) $(PYTEST_ARGS) test_exceptions.py -v

# Couverture de code - Console
coverage:
	@echo "$(BLUE)📊 Analyse de couverture de code...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) \
		--cov=$(SRC_DIR) \
		--cov-report=term-missing \
		--cov-fail-under=$(COVERAGE_MIN) \
		-m "not slow and not performance and not stress"

# Couverture de code - HTML
coverage-html:
	@echo "$(BLUE)📊 Génération rapport HTML de couverture...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) \
		--cov=$(SRC_DIR) \
		--cov-report=html:../../../coverage_html \
		--cov-report=term \
		--cov-fail-under=$(COVERAGE_MIN) \
		-m "not slow and not performance and not stress"
	@echo "$(GREEN)📄 Rapport HTML généré dans coverage_html/$(NC)"

# Couverture de code - XML (pour CI/CD)
coverage-xml:
	@echo "$(BLUE)📊 Génération rapport XML de couverture...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) \
		--cov=$(SRC_DIR) \
		--cov-report=xml:../../../coverage.xml \
		--cov-report=term \
		--cov-fail-under=$(COVERAGE_MIN) \
		-m "not slow and not performance and not stress"

# Linting avec flake8
lint:
	@echo "$(BLUE)🔍 Vérification linting...$(NC)"
	flake8 $(SRC_DIR) --max-line-length=100 --ignore=E203,W503
	flake8 $(TEST_DIR) --max-line-length=100 --ignore=E203,W503

# Formatage automatique
format:
	@echo "$(BLUE)✨ Formatage automatique du code...$(NC)"
	black $(SRC_DIR) --line-length=100
	black $(TEST_DIR) --line-length=100
	isort $(SRC_DIR) --profile=black
	isort $(TEST_DIR) --profile=black

# Vérification du style
check-style:
	@echo "$(BLUE)👮 Vérification du style...$(NC)"
	black $(SRC_DIR) --check --line-length=100
	black $(TEST_DIR) --check --line-length=100
	isort $(SRC_DIR) --check-only --profile=black
	isort $(TEST_DIR) --check-only --profile=black

# Vérification des types avec mypy
type-check:
	@echo "$(BLUE)🔬 Vérification des types...$(NC)"
	mypy $(SRC_DIR) --ignore-missing-imports
	@echo "$(GREEN)✅ Vérification des types terminée$(NC)"

# Tests parallèles (plus rapide)
test-parallel:
	@echo "$(BLUE)⚡ Tests en parallèle...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) $(PYTEST_ARGS) \
		-n auto \
		-m "not slow and not performance and not stress"

# Tests avec reporting détaillé
test-report:
	@echo "$(BLUE)📋 Tests avec rapport détaillé...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) $(PYTEST_ARGS) \
		--html=../../../test_report.html \
		--self-contained-html \
		-m "not slow and not performance and not stress"
	@echo "$(GREEN)📄 Rapport généré dans test_report.html$(NC)"

# Benchmark spécifique
benchmark:
	@echo "$(BLUE)⏱️ Benchmarks de performance...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) \
		test_performance.py::TestPerformanceBenchmarks \
		--benchmark-only \
		--benchmark-sort=mean \
		--benchmark-html=../../../benchmark_report.html

# Tests de sécurité
test-security:
	@echo "$(BLUE)🛡️ Tests de sécurité...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) $(PYTEST_ARGS) \
		-k "security or validation or sanitize" \
		test_utils.py test_exceptions.py

# Tests de régression
test-regression:
	@echo "$(BLUE)🔄 Tests de régression...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) $(PYTEST_ARGS) \
		--tb=line \
		--maxfail=1 \
		-m "not slow"

# Debug d'un test spécifique
test-debug:
	@echo "$(BLUE)🐛 Mode debug activé...$(NC)"
	@echo "Usage: make test-debug TEST=test_file.py::test_function"
	cd $(TEST_DIR) && $(PYTEST) -vvv -s --tb=long $(TEST)

# Nettoyage des fichiers temporaires
clean:
	@echo "$(BLUE)🧹 Nettoyage...$(NC)"
	find $(TEST_DIR) -type f -name "*.pyc" -delete
	find $(TEST_DIR) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find $(SRC_DIR) -type f -name "*.pyc" -delete
	find $(SRC_DIR) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf coverage_html
	rm -rf htmlcov
	rm -f coverage.xml
	rm -f test_report.html
	rm -f benchmark_report.html
	@echo "$(GREEN)✅ Nettoyage terminé$(NC)"

# Vérification complète (CI/CD)
check-all: install-deps format lint type-check coverage
	@echo "$(GREEN)✅ Toutes les vérifications passées$(NC)"

# Tests de smoke (vérification rapide)
smoke-test:
	@echo "$(BLUE)💨 Tests de smoke...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) $(PYTEST_ARGS) \
		-k "test_initialization or test_basic" \
		--maxfail=1 \
		--timeout=10

# Tests d'acceptation
acceptance-test:
	@echo "$(BLUE)✅ Tests d'acceptation...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) $(PYTEST_ARGS) \
		test_integration.py::TestEndToEndScenarios \
		--timeout=120

# Profiling des tests
profile-tests:
	@echo "$(BLUE)📈 Profiling des tests...$(NC)"
	cd $(TEST_DIR) && $(PYTHON) -m cProfile -o test_profile.prof \
		-m pytest test_performance.py::TestPerformanceBenchmarks -v
	@echo "$(GREEN)📊 Profil sauvé dans test_profile.prof$(NC)"

# Tests avec watchdog (développement)
watch-tests:
	@echo "$(BLUE)👀 Mode watch activé (Ctrl+C pour arrêter)...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) $(PYTEST_ARGS) \
		-f \
		-m "not slow and not performance and not stress"

# Affichage de la configuration
show-config:
	@echo "$(BLUE)⚙️ Configuration des tests :$(NC)"
	@echo "  Python: $(shell $(PYTHON) --version)"
	@echo "  Pytest: $(shell $(PYTEST) --version)"
	@echo "  Répertoire tests: $(TEST_DIR)"
	@echo "  Répertoire source: $(SRC_DIR)"
	@echo "  Couverture minimale: $(COVERAGE_MIN)%"
	@echo "  Arguments pytest: $(PYTEST_ARGS)"

# Tests avec métriques détaillées
test-metrics:
	@echo "$(BLUE)📊 Tests avec métriques détaillées...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) $(PYTEST_ARGS) \
		--durations=10 \
		--tb=short \
		-v \
		-m "not slow"

# Tests de compatibilité
test-compatibility:
	@echo "$(BLUE)🔧 Tests de compatibilité...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) $(PYTEST_ARGS) \
		-k "compatibility or version or dependency" \
		test_utils.py test_exceptions.py

# Validation complète avant commit
pre-commit: format lint type-check test coverage
	@echo "$(GREEN)✅ Validation pre-commit réussie$(NC)"

# Tests de charge
load-test:
	@echo "$(BLUE)🏋️ Tests de charge...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) $(PYTEST_ARGS) \
		test_performance.py::TestStressTests \
		--timeout=600 \
		-v

# Tests avec couverture différentielle
diff-coverage:
	@echo "$(BLUE)🔍 Couverture différentielle...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) \
		--cov=$(SRC_DIR) \
		--cov-report=term \
		--cov-branch \
		--cov-context=test \
		-m "not slow"

# Export des résultats (JSON)
test-export:
	@echo "$(BLUE)📤 Export des résultats...$(NC)"
	cd $(TEST_DIR) && $(PYTEST) \
		--json-report --json-report-file=../../../test_results.json \
		-m "not slow and not performance and not stress"

# Cibles par défaut
.DEFAULT_GOAL := help
