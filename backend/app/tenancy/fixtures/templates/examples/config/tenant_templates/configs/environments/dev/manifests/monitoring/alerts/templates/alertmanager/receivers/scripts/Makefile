# Advanced Scripts System Makefile
# Version: 3.0.0
# Developed by Spotify AI Agent Team
# Lead Developer & AI Architect: Fahed Mlaiel

.PHONY: help build test deploy clean setup lint security docs monitoring backup restore

# ============================================================================
# Configuration Variables
# ============================================================================

# Project information
PROJECT_NAME := advanced-scripts-system
VERSION := 3.0.0
BUILD_DATE := $(shell date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF := $(shell git rev-parse --short HEAD)

# Docker configuration
DOCKER_REGISTRY := localhost:5000
DOCKER_IMAGE := $(DOCKER_REGISTRY)/$(PROJECT_NAME)
DOCKER_TAG := $(VERSION)

# Environment configuration
ENV := development
COMPOSE_FILE := docker-compose.yml
COMPOSE_PROFILES := development

# Directories
SRC_DIR := .
BUILD_DIR := build
DIST_DIR := dist
DOCS_DIR := docs
LOGS_DIR := logs
DATA_DIR := data

# Python configuration
PYTHON := python3
PIP := pip3
PYTEST := pytest
BLACK := black
FLAKE8 := flake8
MYPY := mypy
BANDIT := bandit

# Kubernetes configuration
KUBECTL := kubectl
HELM := helm
NAMESPACE := scripts-system

# Default target
.DEFAULT_GOAL := help

# ============================================================================
# Help and Information
# ============================================================================

help: ## Show this help message
	@echo "Advanced Scripts System - Build & Deployment Automation"
	@echo "========================================================="
	@echo ""
	@echo "Usage: make [TARGET] [ARGS...]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Environment variables:"
	@echo "  ENV              Environment (development|testing|production)"
	@echo "  DOCKER_REGISTRY  Docker registry URL"
	@echo "  VERSION          Project version"
	@echo ""
	@echo "Examples:"
	@echo "  make setup                    # Setup development environment"
	@echo "  make build ENV=production     # Build production image"
	@echo "  make deploy ENV=staging       # Deploy to staging"
	@echo "  make test                     # Run all tests"

info: ## Display project information
	@echo "Project: $(PROJECT_NAME)"
	@echo "Version: $(VERSION)"
	@echo "Build Date: $(BUILD_DATE)"
	@echo "VCS Ref: $(VCS_REF)"
	@echo "Environment: $(ENV)"
	@echo "Docker Image: $(DOCKER_IMAGE):$(DOCKER_TAG)"

# ============================================================================
# Setup and Installation
# ============================================================================

setup: ## Setup development environment
	@echo "Setting up development environment..."
	@$(PYTHON) -m venv venv
	@source venv/bin/activate && $(PIP) install --upgrade pip setuptools wheel
	@source venv/bin/activate && $(PIP) install -r requirements.txt
	@source venv/bin/activate && $(PIP) install -r requirements-dev.txt
	@$(MAKE) setup-dirs
	@$(MAKE) setup-configs
	@$(MAKE) setup-git-hooks
	@echo "Development environment setup complete!"

setup-dirs: ## Create necessary directories
	@mkdir -p $(BUILD_DIR) $(DIST_DIR) $(DOCS_DIR) $(LOGS_DIR) $(DATA_DIR)
	@mkdir -p $(DATA_DIR)/{postgres,redis,elasticsearch,prometheus,grafana,minio,rabbitmq,vault,registry}
	@mkdir -p configs/{traefik,vault,redis,monitoring}
	@mkdir -p templates/{deployment,service,configmap,secret}

setup-configs: ## Setup configuration files
	@echo "Setting up configuration files..."
	@cp -n configs/templates/* configs/ 2>/dev/null || true
	@sed -i 's/{{PROJECT_NAME}}/$(PROJECT_NAME)/g' configs/*.yml configs/*.yaml 2>/dev/null || true
	@sed -i 's/{{VERSION}}/$(VERSION)/g' configs/*.yml configs/*.yaml 2>/dev/null || true

setup-git-hooks: ## Setup Git hooks
	@echo "Setting up Git hooks..."
	@cp scripts/git-hooks/* .git/hooks/ 2>/dev/null || true
	@chmod +x .git/hooks/* 2>/dev/null || true

install: ## Install dependencies
	@echo "Installing dependencies..."
	@$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	@echo "Installing development dependencies..."
	@$(PIP) install -r requirements-dev.txt

# ============================================================================
# Code Quality and Testing
# ============================================================================

lint: ## Run code linting
	@echo "Running code linting..."
	@$(BLACK) --check --diff $(SRC_DIR)
	@$(FLAKE8) $(SRC_DIR)
	@$(MYPY) $(SRC_DIR)

format: ## Format code
	@echo "Formatting code..."
	@$(BLACK) $(SRC_DIR)
	@isort $(SRC_DIR)

security: ## Run security checks
	@echo "Running security checks..."
	@$(BANDIT) -r $(SRC_DIR) -f json -o security-report.json
	@safety check --json --output safety-report.json
	@semgrep --config=auto $(SRC_DIR) --json --output=semgrep-report.json

test: ## Run all tests
	@echo "Running tests..."
	@$(PYTEST) --cov=$(SRC_DIR) --cov-report=xml --cov-report=html --cov-report=term

test-unit: ## Run unit tests
	@echo "Running unit tests..."
	@$(PYTEST) tests/unit/

test-integration: ## Run integration tests
	@echo "Running integration tests..."
	@$(PYTEST) tests/integration/

test-e2e: ## Run end-to-end tests
	@echo "Running end-to-end tests..."
	@$(PYTEST) tests/e2e/

test-performance: ## Run performance tests
	@echo "Running performance tests..."
	@locust --headless --users 10 --spawn-rate 2 --run-time 1m --host http://localhost:8000

# ============================================================================
# Docker Operations
# ============================================================================

build: ## Build Docker image
	@echo "Building Docker image..."
	@docker build \
		--target production \
		--build-arg BUILD_DATE=$(BUILD_DATE) \
		--build-arg VCS_REF=$(VCS_REF) \
		--build-arg VERSION=$(VERSION) \
		-t $(DOCKER_IMAGE):$(DOCKER_TAG) \
		-t $(DOCKER_IMAGE):latest \
		.

build-dev: ## Build development Docker image
	@echo "Building development Docker image..."
	@docker build \
		--target development \
		--build-arg BUILD_DATE=$(BUILD_DATE) \
		--build-arg VCS_REF=$(VCS_REF) \
		--build-arg VERSION=$(VERSION) \
		-t $(DOCKER_IMAGE):dev \
		.

build-test: ## Build test Docker image
	@echo "Building test Docker image..."
	@docker build \
		--target testing \
		--build-arg BUILD_DATE=$(BUILD_DATE) \
		--build-arg VCS_REF=$(VCS_REF) \
		--build-arg VERSION=$(VERSION) \
		-t $(DOCKER_IMAGE):test \
		.

build-all: ## Build all Docker images
	@$(MAKE) build
	@$(MAKE) build-dev
	@$(MAKE) build-test

push: ## Push Docker image to registry
	@echo "Pushing Docker image to registry..."
	@docker push $(DOCKER_IMAGE):$(DOCKER_TAG)
	@docker push $(DOCKER_IMAGE):latest

pull: ## Pull Docker image from registry
	@echo "Pulling Docker image from registry..."
	@docker pull $(DOCKER_IMAGE):$(DOCKER_TAG)

# ============================================================================
# Docker Compose Operations
# ============================================================================

up: ## Start all services
	@echo "Starting all services..."
	@docker-compose -f $(COMPOSE_FILE) --profile $(COMPOSE_PROFILES) up -d

up-build: ## Build and start all services
	@echo "Building and starting all services..."
	@docker-compose -f $(COMPOSE_FILE) --profile $(COMPOSE_PROFILES) up -d --build

down: ## Stop all services
	@echo "Stopping all services..."
	@docker-compose -f $(COMPOSE_FILE) down

restart: ## Restart all services
	@$(MAKE) down
	@$(MAKE) up

logs: ## Show logs for all services
	@docker-compose -f $(COMPOSE_FILE) logs -f

logs-api: ## Show API logs
	@docker-compose -f $(COMPOSE_FILE) logs -f scripts-api

logs-worker: ## Show worker logs
	@docker-compose -f $(COMPOSE_FILE) logs -f scripts-worker

ps: ## Show service status
	@docker-compose -f $(COMPOSE_FILE) ps

scale-workers: ## Scale worker services (usage: make scale-workers WORKERS=4)
	@docker-compose -f $(COMPOSE_FILE) up -d --scale scripts-worker=$(or $(WORKERS),2)

# ============================================================================
# Database Operations
# ============================================================================

db-migrate: ## Run database migrations
	@echo "Running database migrations..."
	@docker-compose -f $(COMPOSE_FILE) exec scripts-api alembic upgrade head

db-rollback: ## Rollback database migration
	@echo "Rolling back database migration..."
	@docker-compose -f $(COMPOSE_FILE) exec scripts-api alembic downgrade -1

db-reset: ## Reset database
	@echo "Resetting database..."
	@docker-compose -f $(COMPOSE_FILE) down -v
	@docker volume rm scripts_postgres_data 2>/dev/null || true
	@$(MAKE) up

db-seed: ## Seed database with sample data
	@echo "Seeding database..."
	@docker-compose -f $(COMPOSE_FILE) exec scripts-api python scripts/seed_database.py

db-backup: ## Backup database
	@echo "Backing up database..."
	@mkdir -p backups
	@docker-compose -f $(COMPOSE_FILE) exec postgres pg_dump -U scripts_user scripts_db > backups/db_backup_$(shell date +%Y%m%d_%H%M%S).sql

db-restore: ## Restore database from backup (usage: make db-restore BACKUP=backup_file.sql)
	@echo "Restoring database from backup..."
	@docker-compose -f $(COMPOSE_FILE) exec -T postgres psql -U scripts_user scripts_db < $(BACKUP)

# ============================================================================
# Kubernetes Operations
# ============================================================================

k8s-deploy: ## Deploy to Kubernetes
	@echo "Deploying to Kubernetes..."
	@$(KUBECTL) create namespace $(NAMESPACE) --dry-run=client -o yaml | $(KUBECTL) apply -f -
	@$(KUBECTL) apply -f k8s/ -n $(NAMESPACE)

k8s-delete: ## Delete from Kubernetes
	@echo "Deleting from Kubernetes..."
	@$(KUBECTL) delete -f k8s/ -n $(NAMESPACE) --ignore-not-found=true

k8s-status: ## Show Kubernetes deployment status
	@echo "Kubernetes deployment status:"
	@$(KUBECTL) get all -n $(NAMESPACE)

k8s-logs: ## Show Kubernetes logs
	@$(KUBECTL) logs -f deployment/scripts-api -n $(NAMESPACE)

helm-install: ## Install Helm chart
	@echo "Installing Helm chart..."
	@$(HELM) upgrade --install $(PROJECT_NAME) ./helm/$(PROJECT_NAME) \
		--namespace $(NAMESPACE) \
		--create-namespace \
		--set image.tag=$(VERSION) \
		--set environment=$(ENV)

helm-uninstall: ## Uninstall Helm chart
	@echo "Uninstalling Helm chart..."
	@$(HELM) uninstall $(PROJECT_NAME) -n $(NAMESPACE)

# ============================================================================
# Monitoring and Observability
# ============================================================================

monitoring: ## Start monitoring stack
	@echo "Starting monitoring stack..."
	@docker-compose -f $(COMPOSE_FILE) up -d prometheus grafana jaeger elasticsearch

monitoring-down: ## Stop monitoring stack
	@echo "Stopping monitoring stack..."
	@docker-compose -f $(COMPOSE_FILE) stop prometheus grafana jaeger elasticsearch

dashboards: ## Import Grafana dashboards
	@echo "Importing Grafana dashboards..."
	@scripts/import_dashboards.sh

metrics: ## Show application metrics
	@echo "Application metrics:"
	@curl -s http://localhost:8000/metrics || echo "Application not running"

health: ## Check application health
	@echo "Application health:"
	@curl -s http://localhost:8000/health || echo "Application not running"

# ============================================================================
# Security Operations
# ============================================================================

security-scan: ## Run comprehensive security scan
	@echo "Running comprehensive security scan..."
	@$(MAKE) security
	@docker run --rm -v $(PWD):/src aquasec/trivy fs /src
	@docker build --target security-scan -t $(DOCKER_IMAGE):security .
	@docker run --rm $(DOCKER_IMAGE):security

vulnerability-scan: ## Scan for vulnerabilities
	@echo "Scanning for vulnerabilities..."
	@docker run --rm -v $(PWD):/app securecodewarrior/github-action-add-sarif /app

secrets-scan: ## Scan for secrets
	@echo "Scanning for secrets..."
	@docker run --rm -v $(PWD):/app trufflesecurity/trufflehog filesystem /app

# ============================================================================
# Documentation
# ============================================================================

docs: ## Generate documentation
	@echo "Generating documentation..."
	@sphinx-build -b html docs/ docs/_build/html

docs-serve: ## Serve documentation
	@echo "Serving documentation..."
	@python -m http.server 8080 --directory docs/_build/html

docs-api: ## Generate API documentation
	@echo "Generating API documentation..."
	@python scripts/generate_api_docs.py

# ============================================================================
# Backup and Restore
# ============================================================================

backup: ## Create full system backup
	@echo "Creating full system backup..."
	@mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	@$(MAKE) db-backup
	@docker run --rm -v scripts_redis_data:/data -v $(PWD)/backups:/backup redis:7-alpine tar czf /backup/redis_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz -C /data .
	@tar czf backups/configs_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz configs/
	@tar czf backups/data_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz data/

restore: ## Restore from backup (usage: make restore BACKUP_DIR=backup_directory)
	@echo "Restoring from backup..."
	@echo "Please specify BACKUP_DIR parameter"

# ============================================================================
# Deployment Operations
# ============================================================================

deploy-dev: ## Deploy to development environment
	@echo "Deploying to development environment..."
	@$(MAKE) ENV=development COMPOSE_PROFILES=development up-build

deploy-staging: ## Deploy to staging environment
	@echo "Deploying to staging environment..."
	@$(MAKE) ENV=staging build
	@$(MAKE) ENV=staging push
	@$(MAKE) ENV=staging k8s-deploy

deploy-prod: ## Deploy to production environment
	@echo "Deploying to production environment..."
	@$(MAKE) ENV=production build
	@$(MAKE) ENV=production security-scan
	@$(MAKE) ENV=production push
	@$(MAKE) ENV=production k8s-deploy

rollback: ## Rollback deployment
	@echo "Rolling back deployment..."
	@$(KUBECTL) rollout undo deployment/scripts-api -n $(NAMESPACE)

# ============================================================================
# Development Utilities
# ============================================================================

shell: ## Access application shell
	@docker-compose -f $(COMPOSE_FILE) exec scripts-api bash

db-shell: ## Access database shell
	@docker-compose -f $(COMPOSE_FILE) exec postgres psql -U scripts_user scripts_db

redis-shell: ## Access Redis shell
	@docker-compose -f $(COMPOSE_FILE) exec redis redis-cli

worker-shell: ## Access worker shell
	@docker-compose -f $(COMPOSE_FILE) exec scripts-worker bash

# ============================================================================
# Performance and Load Testing
# ============================================================================

load-test: ## Run load test
	@echo "Running load test..."
	@locust --headless --users 100 --spawn-rate 10 --run-time 5m --host http://localhost:8000

stress-test: ## Run stress test
	@echo "Running stress test..."
	@docker run --rm -v $(PWD)/tests:/tests --network host loadimpact/k6 run /tests/stress-test.js

benchmark: ## Run benchmarks
	@echo "Running benchmarks..."
	@python scripts/benchmark.py

# ============================================================================
# Clean Up Operations
# ============================================================================

clean: ## Clean up generated files
	@echo "Cleaning up..."
	@rm -rf $(BUILD_DIR) $(DIST_DIR) .pytest_cache .coverage htmlcov
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete

clean-docker: ## Clean up Docker resources
	@echo "Cleaning up Docker resources..."
	@docker system prune -f
	@docker volume prune -f
	@docker network prune -f

clean-all: ## Clean everything
	@$(MAKE) clean
	@$(MAKE) clean-docker
	@docker-compose -f $(COMPOSE_FILE) down -v --remove-orphans

# ============================================================================
# CI/CD Integration
# ============================================================================

ci-setup: ## Setup CI environment
	@echo "Setting up CI environment..."
	@$(MAKE) install
	@$(MAKE) install-dev

ci-test: ## Run CI tests
	@echo "Running CI tests..."
	@$(MAKE) lint
	@$(MAKE) security
	@$(MAKE) test
	@$(MAKE) build-test

ci-deploy: ## CI deployment
	@echo "Running CI deployment..."
	@$(MAKE) build
	@$(MAKE) push
	@$(MAKE) deploy-$(ENV)

# ============================================================================
# Debug and Troubleshooting
# ============================================================================

debug: ## Show debug information
	@echo "Debug Information:"
	@echo "=================="
	@echo "Docker version: $(shell docker --version)"
	@echo "Docker Compose version: $(shell docker-compose --version)"
	@echo "Python version: $(shell $(PYTHON) --version)"
	@echo "Git branch: $(shell git rev-parse --abbrev-ref HEAD)"
	@echo "Git commit: $(VCS_REF)"
	@echo ""
	@echo "Environment:"
	@echo "ENV=$(ENV)"
	@echo "DOCKER_IMAGE=$(DOCKER_IMAGE)"
	@echo "DOCKER_TAG=$(DOCKER_TAG)"
	@echo ""
	@echo "Services status:"
	@$(MAKE) ps

troubleshoot: ## Run troubleshooting checks
	@echo "Running troubleshooting checks..."
	@scripts/troubleshoot.sh

# ============================================================================
# Release Management
# ============================================================================

release: ## Create a new release
	@echo "Creating release $(VERSION)..."
	@git tag -a v$(VERSION) -m "Release $(VERSION)"
	@git push origin v$(VERSION)
	@$(MAKE) build
	@$(MAKE) push

release-notes: ## Generate release notes
	@echo "Generating release notes..."
	@scripts/generate_release_notes.sh

# ============================================================================
# Environment-specific targets
# ============================================================================

.env: ## Create .env file from template
	@cp .env.template .env
	@echo "Created .env file from template. Please edit as needed."

check-env: ## Check environment configuration
	@echo "Checking environment configuration..."
	@scripts/check_env.sh

# ============================================================================
# IDE Integration
# ============================================================================

vscode: ## Setup VS Code configuration
	@mkdir -p .vscode
	@cp scripts/vscode/* .vscode/

# ============================================================================
# Maintenance
# ============================================================================

update-deps: ## Update dependencies
	@echo "Updating dependencies..."
	@$(PIP) list --outdated --format=json | jq -r '.[] | .name' | xargs -I {} $(PIP) install --upgrade {}
	@$(PIP) freeze > requirements.txt

# ============================================================================
# Help for specific commands
# ============================================================================

help-docker: ## Show Docker-related commands
	@echo "Docker Commands:"
	@grep -E '^[a-zA-Z_-]*docker[a-zA-Z_-]*:.*?## .*$$|^build.*:.*?## .*$$|^push.*:.*?## .*$$|^pull.*:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

help-k8s: ## Show Kubernetes-related commands
	@echo "Kubernetes Commands:"
	@grep -E '^[a-zA-Z_-]*k8s[a-zA-Z_-]*:.*?## .*$$|^[a-zA-Z_-]*helm[a-zA-Z_-]*:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

help-test: ## Show testing-related commands
	@echo "Testing Commands:"
	@grep -E '^[a-zA-Z_-]*test[a-zA-Z_-]*:.*?## .*$$|^[a-zA-Z_-]*security[a-zA-Z_-]*:.*?## .*$$|^lint.*:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
