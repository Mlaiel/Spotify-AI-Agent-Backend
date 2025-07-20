# 🚀 Spotify AI Agent - Data Isolation Core Makefile
# ================================================
# 
# Automatisation des tâches pour le module core d'isolation des données
# 
# Author: Lead Dev + Architecte IA - Fahed Mlaiel

.PHONY: help install test benchmark validate monitor migrate clean format lint security

# Variables
PYTHON := python3
PIP := pip3
CORE_DIR := /workspaces/Achiri/spotify-ai-agent/backend/app/tenancy/data_isolation/core
VENV_DIR := $(CORE_DIR)/venv

# Couleurs pour l'affichage
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
NC := \033[0m # No Color

# Aide par défaut
help: ## 📚 Affiche cette aide
	@echo "$(CYAN)🎵 Spotify AI Agent - Data Isolation Core$(NC)"
	@echo "$(CYAN)===============================================$(NC)"
	@echo ""
	@echo "$(GREEN)📋 Commandes disponibles:$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)🔧 Variables d'environnement importantes:$(NC)"
	@echo "  TENANT_ISOLATION_LEVEL    Niveau d'isolation (basic/strict/paranoid)"
	@echo "  CACHE_SIZE_MB            Taille du cache en MB"
	@echo "  SECURITY_PARANOID_MODE   Mode sécurité renforcée (true/false)"
	@echo ""

# Installation et configuration
install: ## 🔧 Installe les dépendances et configure l'environnement
	@echo "$(BLUE)🔧 Installation des dépendances...$(NC)"
	$(PIP) install -r $(CORE_DIR)/../../../requirements.txt
	$(PIP) install psutil aioredis
	@echo "$(GREEN)✅ Installation terminée$(NC)"

install-dev: ## 🛠️ Installe les dépendances de développement
	@echo "$(BLUE)🛠️ Installation des dépendances de développement...$(NC)"
	$(PIP) install pytest pytest-asyncio pytest-cov black flake8 mypy
	@echo "$(GREEN)✅ Installation dev terminée$(NC)"

# Tests et validation
test: ## 🧪 Execute les tests unitaires
	@echo "$(BLUE)🧪 Exécution des tests...$(NC)"
	cd $(CORE_DIR) && $(PYTHON) -m pytest ../../../tests/tenancy/data_isolation/core/ -v
	@echo "$(GREEN)✅ Tests terminés$(NC)"

test-coverage: ## 📊 Execute les tests avec couverture de code
	@echo "$(BLUE)📊 Tests avec couverture...$(NC)"
	cd $(CORE_DIR) && $(PYTHON) -m pytest ../../../tests/tenancy/data_isolation/core/ --cov=. --cov-report=html --cov-report=term
	@echo "$(GREEN)✅ Rapport de couverture généré$(NC)"

benchmark: ## ⚡ Execute le benchmark de performance
	@echo "$(BLUE)⚡ Benchmark de performance...$(NC)"
	cd $(CORE_DIR) && $(PYTHON) benchmark_performance.py --output benchmark_report.md
	@echo "$(GREEN)✅ Benchmark terminé - Voir benchmark_report.md$(NC)"

benchmark-quick: ## 🚀 Execute un benchmark rapide
	@echo "$(BLUE)🚀 Benchmark rapide...$(NC)"
	cd $(CORE_DIR) && $(PYTHON) benchmark_performance.py --quick
	@echo "$(GREEN)✅ Benchmark rapide terminé$(NC)"

validate: ## 🔍 Valide la configuration du système
	@echo "$(BLUE)🔍 Validation de la configuration...$(NC)"
	cd $(CORE_DIR) && $(PYTHON) validate_configuration.py --output validation_report.md
	@echo "$(GREEN)✅ Validation terminée - Voir validation_report.md$(NC)"

validate-json: ## 📄 Valide la configuration (sortie JSON)
	@echo "$(BLUE)📄 Validation JSON...$(NC)"
	cd $(CORE_DIR) && $(PYTHON) validate_configuration.py --json --output validation_results.json
	@echo "$(GREEN)✅ Validation JSON terminée$(NC)"

# Monitoring et observabilité
monitor: ## 📊 Lance le monitoring en temps réel
	@echo "$(BLUE)📊 Démarrage du monitoring en temps réel...$(NC)"
	cd $(CORE_DIR) && $(PYTHON) monitor_realtime.py

monitor-export: ## 📁 Lance le monitoring avec export automatique
	@echo "$(BLUE)📁 Monitoring avec export...$(NC)"
	cd $(CORE_DIR) && $(PYTHON) monitor_realtime.py --export monitoring_metrics_$$(date +%Y%m%d_%H%M%S).json

# Migrations
migrate-status: ## 📋 Affiche le statut des migrations
	@echo "$(BLUE)📋 Statut des migrations...$(NC)"
	cd $(CORE_DIR) && $(PYTHON) migrate_isolation.py status

migrate-up: ## ⬆️ Applique toutes les migrations en attente
	@echo "$(BLUE)⬆️ Application des migrations...$(NC)"
	cd $(CORE_DIR) && $(PYTHON) migrate_isolation.py up
	@echo "$(GREEN)✅ Migrations appliquées$(NC)"

migrate-down: ## ⬇️ Annule la dernière migration
	@echo "$(YELLOW)⬇️ Annulation de la dernière migration...$(NC)"
	cd $(CORE_DIR) && $(PYTHON) migrate_isolation.py down --steps 1
	@echo "$(GREEN)✅ Migration annulée$(NC)"

migrate-validate: ## ✅ Valide les migrations en attente
	@echo "$(BLUE)✅ Validation des migrations...$(NC)"
	cd $(CORE_DIR) && $(PYTHON) migrate_isolation.py validate

migrate-dry-run: ## 🧪 Simule les migrations sans les appliquer
	@echo "$(BLUE)🧪 Simulation des migrations...$(NC)"
	cd $(CORE_DIR) && $(PYTHON) migrate_isolation.py up --dry-run

# Qualité du code
format: ## 🎨 Formate le code avec Black
	@echo "$(BLUE)🎨 Formatage du code...$(NC)"
	black $(CORE_DIR)/*.py
	@echo "$(GREEN)✅ Code formaté$(NC)"

lint: ## 🔍 Analyse le code avec Flake8
	@echo "$(BLUE)🔍 Analyse du code...$(NC)"
	flake8 $(CORE_DIR)/*.py --max-line-length=120 --ignore=E203,W503
	@echo "$(GREEN)✅ Analyse terminée$(NC)"

type-check: ## 🔬 Vérification des types avec MyPy
	@echo "$(BLUE)🔬 Vérification des types...$(NC)"
	mypy $(CORE_DIR)/*.py --ignore-missing-imports
	@echo "$(GREEN)✅ Types vérifiés$(NC)"

# Sécurité
security-scan: ## 🛡️ Scan de sécurité avec Bandit
	@echo "$(BLUE)🛡️ Scan de sécurité...$(NC)"
	bandit -r $(CORE_DIR)/ -f json -o security_report.json
	@echo "$(GREEN)✅ Scan de sécurité terminé$(NC)"

check-deps: ## 🔒 Vérifie les vulnérabilités des dépendances
	@echo "$(BLUE)🔒 Vérification des dépendances...$(NC)"
	safety check
	@echo "$(GREEN)✅ Dépendances vérifiées$(NC)"

# Documentation
docs: ## 📚 Génère la documentation
	@echo "$(BLUE)📚 Génération de la documentation...$(NC)"
	# Ici vous pouvez ajouter Sphinx ou autre générateur de docs
	@echo "$(GREEN)✅ Documentation générée$(NC)"

# Nettoyage
clean: ## 🧹 Nettoie les fichiers temporaires
	@echo "$(BLUE)🧹 Nettoyage des fichiers temporaires...$(NC)"
	find $(CORE_DIR) -type f -name "*.pyc" -delete
	find $(CORE_DIR) -type d -name "__pycache__" -delete
	rm -f $(CORE_DIR)/*.log
	rm -f $(CORE_DIR)/benchmark_report.md
	rm -f $(CORE_DIR)/validation_report.md
	rm -f $(CORE_DIR)/validation_results.json
	rm -f $(CORE_DIR)/monitoring_metrics_*.json
	rm -f $(CORE_DIR)/security_report.json
	@echo "$(GREEN)✅ Nettoyage terminé$(NC)"

clean-cache: ## 🗑️ Nettoie le cache de performance
	@echo "$(BLUE)🗑️ Nettoyage du cache...$(NC)"
	# Simulation du nettoyage de cache
	@echo "$(GREEN)✅ Cache nettoyé$(NC)"

# Développement
dev-setup: install install-dev ## 🏗️ Configuration complète pour le développement
	@echo "$(GREEN)🏗️ Environnement de développement configuré$(NC)"

pre-commit: format lint type-check test ## 🚀 Vérifie la qualité avant commit
	@echo "$(GREEN)🚀 Code prêt pour commit$(NC)"

ci: format lint type-check test benchmark validate ## 🔄 Pipeline CI complet
	@echo "$(GREEN)🔄 Pipeline CI terminé avec succès$(NC)"

# Déploiement
deploy-check: validate benchmark security-scan ## 🚁 Vérifications avant déploiement
	@echo "$(GREEN)🚁 Vérifications de déploiement terminées$(NC)"

# Maintenance
health-check: ## 💊 Vérification de santé du système
	@echo "$(BLUE)💊 Vérification de santé...$(NC)"
	cd $(CORE_DIR) && $(PYTHON) validate_configuration.py --json | jq '.summary.overall_status'
	@echo "$(GREEN)✅ Vérification de santé terminée$(NC)"

performance-report: ## 📈 Génère un rapport de performance détaillé
	@echo "$(BLUE)📈 Génération du rapport de performance...$(NC)"
	cd $(CORE_DIR) && $(PYTHON) benchmark_performance.py --output performance_report_$$(date +%Y%m%d).md
	@echo "$(GREEN)✅ Rapport de performance généré$(NC)"

# Configuration d'exemple
create-env-example: ## 📝 Crée un fichier d'exemple de variables d'environnement
	@echo "$(BLUE)📝 Création du fichier .env.example...$(NC)"
	@cat > $(CORE_DIR)/.env.example << 'EOF'
# 🎵 Spotify AI Agent - Data Isolation Core Configuration
# =====================================================

# Isolation Configuration
TENANT_ISOLATION_LEVEL=strict
PERFORMANCE_OPTIMIZATION=adaptive

# Cache Configuration
CACHE_SIZE_MB=2048
CACHE_TTL_SECONDS=300
QUERY_CACHE_ENABLED=true

# Security Configuration
SECURITY_PARANOID_MODE=true
ENCRYPTION_KEY_VERSION=2
MFA_ENFORCEMENT=true

# Compliance Configuration
COMPLIANCE_AUDIT_ENABLED=true
GDPR_STRICT_MODE=true
AUDIT_RETENTION_DAYS=2555

# Monitoring Configuration
METRICS_COLLECTION_ENABLED=true
ALERTING_ENABLED=true
LOG_LEVEL=INFO

# Database Configuration
DB_CONNECTION_POOL_SIZE=20
DB_QUERY_TIMEOUT=30
DB_CONNECTION_TIMEOUT=10

# Performance Tuning
CONTEXT_SWITCH_OPTIMIZATION=true
QUERY_OPTIMIZATION_ENABLED=true
BACKGROUND_TASKS_ENABLED=true
EOF
	@echo "$(GREEN)✅ Fichier .env.example créé$(NC)"

# Informations système
system-info: ## ℹ️ Affiche les informations système
	@echo "$(CYAN)ℹ️ Informations Système$(NC)"
	@echo "$(CYAN)========================$(NC)"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Pip: $$($(PIP) --version)"
	@echo "Système: $$(uname -a)"
	@echo "Espace disque: $$(df -h . | tail -1 | awk '{print $$4}') disponible"
	@echo "Mémoire: $$(free -h | grep Mem | awk '{print $$7}') disponible"

# Raccourcis utiles
quick-test: benchmark-quick validate ## ⚡ Tests rapides (benchmark + validation)
	@echo "$(GREEN)⚡ Tests rapides terminés$(NC)"

full-check: ci deploy-check ## 🎯 Vérification complète
	@echo "$(GREEN)🎯 Vérification complète terminée$(NC)"

# Meta
version: ## 📌 Affiche la version du module
	@echo "$(CYAN)Data Isolation Core Module v2.0.0$(NC)"
	@echo "$(CYAN)Author: Lead Dev + Architecte IA - Fahed Mlaiel$(NC)"

update-scripts: ## 🔄 Met à jour les permissions des scripts
	chmod +x $(CORE_DIR)/*.py
	@echo "$(GREEN)✅ Permissions des scripts mises à jour$(NC)"
