# Makefile pour les Schémas Spotify AI Agent
# Automatisation des tâches de développement et déploiement

.PHONY: help setup validate test docs openapi metrics build deploy clean monitor status install dev-setup

# Configuration
PYTHON := python3
PIP := pip3
SCHEMA_DIR := .
SCRIPTS_DIR := scripts

# Couleurs pour l'affichage
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m

# Cible par défaut
help:
	@echo "$(BLUE)🎯 Spotify AI Agent - Schémas Management$(NC)"
	@echo ""
	@echo "$(GREEN)Commandes disponibles:$(NC)"
	@echo "  $(YELLOW)setup$(NC)       - Configuration initiale de l'environnement"
	@echo "  $(YELLOW)install$(NC)     - Installation des dépendances Python"
	@echo "  $(YELLOW)validate$(NC)    - Validation de tous les schémas"
	@echo "  $(YELLOW)test$(NC)        - Exécution des tests complets"
	@echo "  $(YELLOW)docs$(NC)        - Génération de la documentation"
	@echo "  $(YELLOW)openapi$(NC)     - Export de la spécification OpenAPI"
	@echo "  $(YELLOW)metrics$(NC)     - Analyse des métriques des schémas"
	@echo "  $(YELLOW)build$(NC)       - Construction du package complet"
	@echo "  $(YELLOW)deploy$(NC)      - Déploiement des schémas"
	@echo "  $(YELLOW)clean$(NC)       - Nettoyage des fichiers temporaires"
	@echo "  $(YELLOW)monitor$(NC)     - Monitoring continu des schémas"
	@echo "  $(YELLOW)status$(NC)      - Affichage de l'état du système"
	@echo "  $(YELLOW)dev-setup$(NC)   - Configuration pour développement"
	@echo ""
	@echo "$(GREEN)Exemples:$(NC)"
	@echo "  make setup && make validate  # Installation et validation"
	@echo "  make build                   # Build complet"
	@echo "  make deploy                  # Déploiement en production"

# Configuration initiale
setup:
	@echo "$(BLUE)🔧 Configuration de l'environnement...$(NC)"
	@mkdir -p docs dist tests logs
	@$(SCRIPTS_DIR)/deploy.sh setup
	@echo "$(GREEN)✅ Environnement configuré$(NC)"

# Installation des dépendances
install:
	@echo "$(BLUE)📦 Installation des dépendances...$(NC)"
	@$(PIP) install --upgrade pip
	@$(PIP) install pydantic typing_extensions email_validator
	@$(PIP) install jinja2 asyncio pathlib
	@echo "$(GREEN)✅ Dépendances installées$(NC)"

# Configuration pour développement
dev-setup: install
	@echo "$(BLUE)🛠️ Configuration développement...$(NC)"
	@$(PIP) install pytest pytest-asyncio black flake8 mypy
	@$(PIP) install jupyter notebook ipython
	@echo "$(GREEN)✅ Environnement de développement prêt$(NC)"

# Validation des schémas
validate:
	@echo "$(BLUE)🔍 Validation des schémas...$(NC)"
	@$(PYTHON) $(SCRIPTS_DIR)/validate_schemas.py validate
	@echo "$(GREEN)✅ Validation terminée$(NC)"

# Tests complets
test: validate
	@echo "$(BLUE)🧪 Exécution des tests...$(NC)"
	@$(SCRIPTS_DIR)/deploy.sh test
	@echo "$(GREEN)✅ Tests terminés$(NC)"

# Génération de documentation
docs:
	@echo "$(BLUE)📚 Génération de la documentation...$(NC)"
	@$(PYTHON) $(SCRIPTS_DIR)/validate_schemas.py docs --output docs/
	@echo "$(GREEN)✅ Documentation générée dans docs/$(NC)"

# Export OpenAPI
openapi:
	@echo "$(BLUE)📋 Export OpenAPI...$(NC)"
	@$(PYTHON) $(SCRIPTS_DIR)/validate_schemas.py openapi --output openapi.json
	@echo "$(GREEN)✅ Spécification OpenAPI exportée$(NC)"

# Analyse des métriques
metrics:
	@echo "$(BLUE)📊 Analyse des métriques...$(NC)"
	@$(PYTHON) $(SCRIPTS_DIR)/validate_schemas.py metrics --output metrics_report.json
	@echo "$(GREEN)✅ Rapport de métriques généré$(NC)"

# Build complet
build: clean validate docs openapi metrics
	@echo "$(BLUE)🚀 Construction du package...$(NC)"
	@$(SCRIPTS_DIR)/deploy.sh build
	@echo "$(GREEN)✅ Package construit avec succès$(NC)"

# Déploiement
deploy: build
	@echo "$(BLUE)🚀 Déploiement...$(NC)"
	@$(SCRIPTS_DIR)/deploy.sh deploy
	@echo "$(GREEN)✅ Déploiement terminé$(NC)"

# Nettoyage
clean:
	@echo "$(BLUE)🧹 Nettoyage...$(NC)"
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@find . -name "*.pyo" -delete 2>/dev/null || true
	@find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
	@rm -rf dist/*.tar.gz 2>/dev/null || true
	@echo "$(GREEN)✅ Nettoyage terminé$(NC)"

# Monitoring continu
monitor:
	@echo "$(BLUE)👁️ Démarrage du monitoring...$(NC)"
	@$(SCRIPTS_DIR)/deploy.sh monitor

# Statut du système
status:
	@echo "$(BLUE)📊 État du système:$(NC)"
	@$(SCRIPTS_DIR)/deploy.sh status

# Formatage du code
format:
	@echo "$(BLUE)✨ Formatage du code...$(NC)"
	@black . --line-length 88
	@echo "$(GREEN)✅ Code formaté$(NC)"

# Vérification du style
lint:
	@echo "$(BLUE)🔍 Vérification du style...$(NC)"
	@flake8 . --max-line-length=88 --extend-ignore=E203,W503
	@echo "$(GREEN)✅ Style vérifié$(NC)"

# Vérification des types
typecheck:
	@echo "$(BLUE)🔍 Vérification des types...$(NC)"
	@mypy . --ignore-missing-imports
	@echo "$(GREEN)✅ Types vérifiés$(NC)"

# Vérifications complètes
check: lint typecheck validate
	@echo "$(GREEN)✅ Toutes les vérifications passées$(NC)"

# Installation en mode développement
dev-install: dev-setup
	@echo "$(BLUE)🔧 Installation en mode développement...$(NC)"
	@$(PIP) install -e .
	@echo "$(GREEN)✅ Installation développement terminée$(NC)"

# Tests de performance
perf-test:
	@echo "$(BLUE)⚡ Tests de performance...$(NC)"
	@$(PYTHON) -c "
import time
from schemas import create_sample_alert, create_sample_notification, create_sample_ml_model

print('🚀 Tests de performance des schémas')
print('=' * 50)

# Test création d'alertes
start = time.time()
alerts = [create_sample_alert() for _ in range(1000)]
end = time.time()
print(f'Création 1000 alertes: {end-start:.3f}s ({1000/(end-start):.0f}/sec)')

# Test sérialisation
start = time.time()
json_data = [alert.json() for alert in alerts[:100]]
end = time.time()
print(f'Sérialisation 100 alertes: {end-start:.3f}s ({100/(end-start):.0f}/sec)')

# Test validation
start = time.time()
for i in range(100):
    alert = create_sample_alert()
    assert alert.is_active == True
end = time.time()
print(f'Validation 100 alertes: {end-start:.3f}s ({100/(end-start):.0f}/sec)')

print('✅ Tests de performance terminés')
"

# Génération de rapport complet
report: clean validate docs openapi metrics
	@echo "$(BLUE)📋 Génération du rapport complet...$(NC)"
	@echo "# Rapport Schémas Spotify AI Agent" > report.md
	@echo "Généré le: $$(date)" >> report.md
	@echo "" >> report.md
	@echo "## Validation" >> report.md
	@$(PYTHON) $(SCRIPTS_DIR)/validate_schemas.py validate >> report.md 2>&1 || true
	@echo "" >> report.md
	@echo "## Métriques" >> report.md
	@cat metrics_report.json >> report.md 2>/dev/null || echo "Pas de métriques disponibles" >> report.md
	@echo "$(GREEN)✅ Rapport généré: report.md$(NC)"

# Sauvegarde
backup:
	@echo "$(BLUE)💾 Sauvegarde des schémas...$(NC)"
	@tar -czf "backup-schemas-$$(date +%Y%m%d-%H%M%S).tar.gz" \
		--exclude='dist' --exclude='logs' --exclude='__pycache__' \
		--exclude='*.pyc' --exclude='.git' .
	@echo "$(GREEN)✅ Sauvegarde créée$(NC)"

# Restauration depuis backup
restore:
	@echo "$(YELLOW)⚠️ Restauration depuis sauvegarde$(NC)"
	@echo "Veuillez spécifier le fichier: make restore FILE=backup-schemas-YYYYMMDD-HHMMSS.tar.gz"

# Aide détaillée
help-detailed:
	@echo "$(BLUE)🎯 Spotify AI Agent - Guide détaillé des schémas$(NC)"
	@echo ""
	@echo "$(GREEN)Structure du projet:$(NC)"
	@echo "  📁 base/          - Schémas de base et mixins"
	@echo "  📁 alerts/        - Gestion des alertes"
	@echo "  📁 notifications/ - Système de notifications"  
	@echo "  📁 ml/            - Modèles IA/ML"
	@echo "  📁 scripts/       - Scripts utilitaires"
	@echo ""
	@echo "$(GREEN)Workflow recommandé:$(NC)"
	@echo "  1. make setup      # Configuration initiale"
	@echo "  2. make validate   # Validation des schémas"
	@echo "  3. make test       # Tests complets"
	@echo "  4. make build      # Construction"
	@echo "  5. make deploy     # Déploiement"
	@echo ""
	@echo "$(GREEN)Développement:$(NC)"
	@echo "  make dev-setup     # Configuration développement"
	@echo "  make check         # Vérifications complètes"
	@echo "  make format        # Formatage du code"
	@echo "  make perf-test     # Tests de performance"

# Cibles pour CI/CD
ci-setup: install

ci-test: validate test

ci-build: build

ci-deploy: deploy

# All-in-one pour développement rapide
quick: validate docs
	@echo "$(GREEN)✅ Validation et documentation rapides terminées$(NC)"

# All-in-one pour production
prod: clean build deploy
	@echo "$(GREEN)✅ Déploiement production terminé$(NC)"
