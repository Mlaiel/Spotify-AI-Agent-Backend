# Makefile pour le module Alertmanager Receivers Config Ultra-Avancé
# Author: Spotify AI Agent Team - Fahed Mlaiel

.PHONY: help build run test clean docker-build docker-run docker-stop install lint format check-security deploy monitoring backup

# Variables
DOCKER_IMAGE = alertmanager-receivers-config
DOCKER_TAG = latest
COMPOSE_FILE = docker-compose.yml
PYTHON_VERSION = 3.11

# Couleurs pour l'affichage
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
NC = \033[0m # No Color

help: ## Affiche l'aide
	@echo "$(GREEN)Alertmanager Receivers Config Ultra-Avancé$(NC)"
	@echo "$(YELLOW)Commandes disponibles:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# Installation et environnement
install: ## Installe les dépendances Python
	@echo "$(GREEN)Installation des dépendances...$(NC)"
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	@echo "$(GREEN)✓ Installation terminée$(NC)"

venv: ## Crée un environnement virtuel Python
	@echo "$(GREEN)Création de l'environnement virtuel...$(NC)"
	python$(PYTHON_VERSION) -m venv venv
	@echo "$(GREEN)✓ Environnement virtuel créé$(NC)"
	@echo "$(YELLOW)Activez-le avec: source venv/bin/activate$(NC)"

# Qualité du code
lint: ## Vérifie la qualité du code avec flake8, mypy, etc.
	@echo "$(GREEN)Vérification de la qualité du code...$(NC)"
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	mypy . --ignore-missing-imports
	@echo "$(GREEN)✓ Qualité du code vérifiée$(NC)"

format: ## Formate le code avec black et isort
	@echo "$(GREEN)Formatage du code...$(NC)"
	black . --line-length 120
	isort . --profile black
	@echo "$(GREEN)✓ Code formaté$(NC)"

check-security: ## Vérifie la sécurité du code avec bandit
	@echo "$(GREEN)Vérification de sécurité...$(NC)"
	bandit -r . -f json -o security-report.json
	safety check --json --output safety-report.json
	@echo "$(GREEN)✓ Sécurité vérifiée$(NC)"

# Tests
test: ## Lance tous les tests
	@echo "$(GREEN)Lancement des tests...$(NC)"
	python -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)✓ Tests terminés$(NC)"

test-unit: ## Lance uniquement les tests unitaires
	@echo "$(GREEN)Tests unitaires...$(NC)"
	python -m pytest tests/unit/ -v
	@echo "$(GREEN)✓ Tests unitaires terminés$(NC)"

test-integration: ## Lance les tests d'intégration
	@echo "$(GREEN)Tests d'intégration...$(NC)"
	python -m pytest tests/integration/ -v
	@echo "$(GREEN)✓ Tests d'intégration terminés$(NC)"

test-performance: ## Lance les tests de performance
	@echo "$(GREEN)Tests de performance...$(NC)"
	python -m pytest tests/performance/ -v --benchmark-only
	@echo "$(GREEN)✓ Tests de performance terminés$(NC)"

# Docker
docker-build: ## Construit l'image Docker
	@echo "$(GREEN)Construction de l'image Docker...$(NC)"
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "$(GREEN)✓ Image Docker construite$(NC)"

docker-build-dev: ## Construit l'image Docker pour le développement
	@echo "$(GREEN)Construction de l'image Docker de développement...$(NC)"
	docker build --target development -t $(DOCKER_IMAGE):dev .
	@echo "$(GREEN)✓ Image Docker de développement construite$(NC)"

docker-run: ## Lance le conteneur Docker
	@echo "$(GREEN)Lancement du conteneur Docker...$(NC)"
	docker run -d --name $(DOCKER_IMAGE) -p 8080:8080 -p 9090:9090 $(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "$(GREEN)✓ Conteneur Docker lancé$(NC)"

docker-stop: ## Arrête et supprime le conteneur Docker
	@echo "$(GREEN)Arrêt du conteneur Docker...$(NC)"
	docker stop $(DOCKER_IMAGE) || true
	docker rm $(DOCKER_IMAGE) || true
	@echo "$(GREEN)✓ Conteneur Docker arrêté$(NC)"

# Docker Compose
compose-up: ## Lance la stack complète avec Docker Compose
	@echo "$(GREEN)Lancement de la stack Docker Compose...$(NC)"
	docker-compose -f $(COMPOSE_FILE) up -d
	@echo "$(GREEN)✓ Stack Docker Compose lancée$(NC)"
	@echo "$(YELLOW)Services disponibles:$(NC)"
	@echo "  - Alertmanager Config: http://localhost:8080"
	@echo "  - Métriques Prometheus: http://localhost:9090"
	@echo "  - Grafana: http://localhost:3000 (admin/admin_secure_password)"
	@echo "  - Kibana: http://localhost:5601"
	@echo "  - Jaeger: http://localhost:16686"

compose-down: ## Arrête la stack Docker Compose
	@echo "$(GREEN)Arrêt de la stack Docker Compose...$(NC)"
	docker-compose -f $(COMPOSE_FILE) down
	@echo "$(GREEN)✓ Stack Docker Compose arrêtée$(NC)"

compose-logs: ## Affiche les logs de la stack
	@echo "$(GREEN)Logs de la stack Docker Compose...$(NC)"
	docker-compose -f $(COMPOSE_FILE) logs -f

compose-build: ## Reconstruit les services Docker Compose
	@echo "$(GREEN)Reconstruction des services...$(NC)"
	docker-compose -f $(COMPOSE_FILE) build --no-cache
	@echo "$(GREEN)✓ Services reconstruits$(NC)"

# Développement
dev: ## Lance l'environnement de développement
	@echo "$(GREEN)Lancement de l'environnement de développement...$(NC)"
	python -m uvicorn main:app --reload --host 0.0.0.0 --port 8080

dev-docker: ## Lance l'environnement de développement dans Docker
	@echo "$(GREEN)Lancement de l'environnement de développement Docker...$(NC)"
	docker-compose -f docker-compose.dev.yml up -d
	@echo "$(GREEN)✓ Environnement de développement Docker lancé$(NC)"

# Monitoring et observabilité
monitoring: ## Lance les outils de monitoring
	@echo "$(GREEN)Lancement des outils de monitoring...$(NC)"
	docker-compose -f $(COMPOSE_FILE) up -d prometheus grafana kibana jaeger
	@echo "$(GREEN)✓ Outils de monitoring lancés$(NC)"

health-check: ## Vérifie la santé de tous les services
	@echo "$(GREEN)Vérification de la santé des services...$(NC)"
	@curl -f http://localhost:8080/health && echo "$(GREEN)✓ Alertmanager Config OK$(NC)" || echo "$(RED)✗ Alertmanager Config KO$(NC)"
	@curl -f http://localhost:9090/-/healthy && echo "$(GREEN)✓ Prometheus OK$(NC)" || echo "$(RED)✗ Prometheus KO$(NC)"
	@curl -f http://localhost:3000/api/health && echo "$(GREEN)✓ Grafana OK$(NC)" || echo "$(RED)✗ Grafana KO$(NC)"
	@curl -f http://localhost:5601/api/status && echo "$(GREEN)✓ Kibana OK$(NC)" || echo "$(RED)✗ Kibana KO$(NC)"

metrics: ## Affiche les métriques principales
	@echo "$(GREEN)Métriques principales:$(NC)"
	@curl -s http://localhost:9090/api/v1/query?query=up | jq -r '.data.result[] | "\(.metric.job): \(.value[1])"'

# Backup et restauration
backup: ## Sauvegarde la configuration et les données
	@echo "$(GREEN)Sauvegarde en cours...$(NC)"
	mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	docker-compose exec -T postgres pg_dump -U alertmanager alertmanager_config > backups/$(shell date +%Y%m%d_%H%M%S)/postgres.sql
	docker-compose exec -T redis redis-cli --rdb /data/dump.rdb
	docker cp alertmanager-redis:/data/dump.rdb backups/$(shell date +%Y%m%d_%H%M%S)/redis.rdb
	@echo "$(GREEN)✓ Sauvegarde terminée$(NC)"

restore: ## Restaure depuis une sauvegarde (usage: make restore BACKUP_DIR=20231201_120000)
	@echo "$(GREEN)Restauration depuis $(BACKUP_DIR)...$(NC)"
	@if [ -z "$(BACKUP_DIR)" ]; then echo "$(RED)Erreur: spécifiez BACKUP_DIR$(NC)"; exit 1; fi
	docker-compose exec -T postgres psql -U alertmanager -d alertmanager_config < backups/$(BACKUP_DIR)/postgres.sql
	docker cp backups/$(BACKUP_DIR)/redis.rdb alertmanager-redis:/data/dump.rdb
	docker-compose restart redis
	@echo "$(GREEN)✓ Restauration terminée$(NC)"

# Nettoyage
clean: ## Nettoie les fichiers temporaires
	@echo "$(GREEN)Nettoyage des fichiers temporaires...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@echo "$(GREEN)✓ Nettoyage terminé$(NC)"

clean-docker: ## Nettoie les images et volumes Docker
	@echo "$(GREEN)Nettoyage Docker...$(NC)"
	docker system prune -f
	docker volume prune -f
	@echo "$(GREEN)✓ Nettoyage Docker terminé$(NC)"

clean-all: clean clean-docker ## Nettoyage complet
	@echo "$(GREEN)✓ Nettoyage complet terminé$(NC)"

# Déploiement
deploy-staging: ## Déploie sur l'environnement de staging
	@echo "$(GREEN)Déploiement sur staging...$(NC)"
	docker-compose -f docker-compose.staging.yml up -d
	@echo "$(GREEN)✓ Déploiement staging terminé$(NC)"

deploy-prod: ## Déploie sur l'environnement de production
	@echo "$(GREEN)Déploiement sur production...$(NC)"
	@echo "$(YELLOW)Êtes-vous sûr de vouloir déployer en production ? [y/N]$(NC)"
	@read confirm && [ "$$confirm" = "y" ] || exit 1
	docker-compose -f docker-compose.prod.yml up -d
	@echo "$(GREEN)✓ Déploiement production terminé$(NC)"

# Documentation
docs: ## Génère la documentation
	@echo "$(GREEN)Génération de la documentation...$(NC)"
	sphinx-build -b html docs/ docs/_build/html
	@echo "$(GREEN)✓ Documentation générée$(NC)"
	@echo "$(YELLOW)Ouvrez docs/_build/html/index.html$(NC)"

docs-serve: ## Lance le serveur de documentation
	@echo "$(GREEN)Serveur de documentation...$(NC)"
	cd docs/_build/html && python -m http.server 8000

# Outils utiles
logs: ## Affiche les logs du service principal
	@echo "$(GREEN)Logs du service principal...$(NC)"
	docker-compose logs -f alertmanager-receivers-config

shell: ## Lance un shell dans le conteneur principal
	@echo "$(GREEN)Shell dans le conteneur principal...$(NC)"
	docker-compose exec alertmanager-receivers-config /bin/bash

psql: ## Lance psql pour se connecter à PostgreSQL
	@echo "$(GREEN)Connexion à PostgreSQL...$(NC)"
	docker-compose exec postgres psql -U alertmanager -d alertmanager_config

redis-cli: ## Lance redis-cli pour se connecter à Redis
	@echo "$(GREEN)Connexion à Redis...$(NC)"
	docker-compose exec redis redis-cli -a redis_secure_password

# Par défaut
.DEFAULT_GOAL := help
