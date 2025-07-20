# Enterprise Tenant Templates Management Makefile
# Ultra-Advanced Industrial Multi-Tenant Architecture
# Developed by Expert Team led by Fahed Mlaiel

.PHONY: help install test deploy validate clean backup restore docs lint security performance

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE=\033[0;34m
GREEN=\033[0;32m
YELLOW=\033[1;33m
RED=\033[0;31m
NC=\033[0m # No Color

# Expert team banner
define BANNER
echo -e "$(BLUE)"
echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║               Enterprise Tenant Template Management System                   ║"
echo "║          Ultra-Advanced Industrial Multi-Tenant Architecture                 ║"
echo "║                                                                               ║"
echo "║                    Developed by Expert Team led by Fahed Mlaiel              ║"
echo "║                                                                               ║"
echo "║  • Lead Dev + AI Architect: Fahed Mlaiel                                     ║"
echo "║  • Senior Backend Developer: Python/FastAPI High Performance                 ║"
echo "║  • ML Engineer: Intelligent Optimization & Predictions                       ║"
echo "║  • DBA & Data Engineer: Multi-Database & Sharding                            ║"
echo "║  • Security Specialist: End-to-End Encryption & Compliance                   ║"
echo "║  • Microservices Architect: Event-Driven & CQRS                              ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo -e "$(NC)"
endef

# Variables
PYTHON := python3
PIP := pip3
DOCKER := docker
DOCKER_COMPOSE := docker-compose
KUBECTL := kubectl
SCRIPTS_DIR := .
CONFIGS_DIR := ./configs
SECRETS_DIR := ./secrets
BACKUPS_DIR := ./backups
LOGS_DIR := ./logs

help: ## Show this help message
	@$(BANNER)
	@echo -e "$(GREEN)Available targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies and setup environment
	@echo -e "$(YELLOW)📦 Installing dependencies...$(NC)"
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@$(PIP) install pytest pytest-asyncio pytest-cov
	@echo -e "$(GREEN)✅ Dependencies installed$(NC)"

setup: ## Setup project environment
	@echo -e "$(YELLOW)🔧 Setting up environment...$(NC)"
	@mkdir -p $(SECRETS_DIR) $(BACKUPS_DIR) $(LOGS_DIR)
	@chmod 700 $(SECRETS_DIR)
	@if [ ! -f $(SECRETS_DIR)/encryption.key ]; then \
		echo -e "$(YELLOW)🔑 Generating encryption key...$(NC)"; \
		$(PYTHON) -c "from cryptography.fernet import Fernet; key = Fernet.generate_key(); open('$(SECRETS_DIR)/encryption.key', 'wb').write(key)"; \
		chmod 600 $(SECRETS_DIR)/encryption.key; \
	fi
	@echo -e "$(GREEN)✅ Environment setup completed$(NC)"

validate: ## Validate all YAML templates
	@echo -e "$(YELLOW)🔍 Validating YAML templates...$(NC)"
	@for template in free.yaml standard.yaml premium.yaml enterprise.yaml enterprise_plus.yaml; do \
		echo -e "$(BLUE)Validating $$template...$(NC)"; \
		$(PYTHON) -c "import yaml; yaml.safe_load(open('$$template'))" || exit 1; \
	done
	@echo -e "$(GREEN)✅ All templates validated$(NC)"

test: setup validate ## Run comprehensive test suite
	@echo -e "$(YELLOW)🧪 Running test suite...$(NC)"
	@$(PYTHON) test_suite.py
	@echo -e "$(GREEN)✅ Test suite completed$(NC)"

test-unit: ## Run unit tests only
	@echo -e "$(YELLOW)🧪 Running unit tests...$(NC)"
	@pytest tests/unit/ -v --cov=. --cov-report=html
	@echo -e "$(GREEN)✅ Unit tests completed$(NC)"

test-integration: ## Run integration tests
	@echo -e "$(YELLOW)🧪 Running integration tests...$(NC)"
	@pytest tests/integration/ -v
	@echo -e "$(GREEN)✅ Integration tests completed$(NC)"

test-performance: ## Run performance tests
	@echo -e "$(YELLOW)⚡ Running performance tests...$(NC)"
	@$(PYTHON) test_suite.py --performance-only
	@echo -e "$(GREEN)✅ Performance tests completed$(NC)"

deploy: setup validate ## Deploy the enterprise tenant template system
	@echo -e "$(YELLOW)🚀 Deploying enterprise tenant template system...$(NC)"
	@./deploy.sh deploy
	@echo -e "$(GREEN)✅ Deployment completed$(NC)"

deploy-dev: ## Deploy development environment
	@echo -e "$(YELLOW)🚀 Deploying development environment...$(NC)"
	@./deploy.sh deploy --environment=development
	@echo -e "$(GREEN)✅ Development deployment completed$(NC)"

deploy-staging: ## Deploy staging environment
	@echo -e "$(YELLOW)🚀 Deploying staging environment...$(NC)"
	@./deploy.sh deploy --environment=staging
	@echo -e "$(GREEN)✅ Staging deployment completed$(NC)"

deploy-production: ## Deploy production environment
	@echo -e "$(YELLOW)🚀 Deploying production environment...$(NC)"
	@./deploy.sh deploy --environment=production
	@echo -e "$(GREEN)✅ Production deployment completed$(NC)"

rollback: ## Rollback deployment
	@echo -e "$(YELLOW)🔄 Rolling back deployment...$(NC)"
	@./deploy.sh rollback
	@echo -e "$(GREEN)✅ Rollback completed$(NC)"

health: ## Check system health
	@echo -e "$(YELLOW)❤️ Checking system health...$(NC)"
	@./deploy.sh health
	@echo -e "$(GREEN)✅ Health check completed$(NC)"

backup: ## Create system backup
	@echo -e "$(YELLOW)💾 Creating system backup...$(NC)"
	@./deploy.sh backup
	@echo -e "$(GREEN)✅ Backup completed$(NC)"

restore: ## Restore from backup
	@echo -e "$(YELLOW)🔄 Restoring from backup...$(NC)"
	@read -p "Enter backup timestamp (YYYYMMDD_HHMMSS): " backup_time; \
	./deploy.sh restore $$backup_time
	@echo -e "$(GREEN)✅ Restore completed$(NC)"

# Template Management Commands
create-template: ## Create template from YAML
	@echo -e "$(YELLOW)📄 Creating template from YAML...$(NC)"
	@read -p "Enter YAML file name: " yaml_file; \
	read -p "Enter template name (optional): " template_name; \
	$(PYTHON) tenant_utils.py create-from-yaml "$$yaml_file" --name "$$template_name"

list-templates: ## List all templates
	@echo -e "$(YELLOW)📋 Listing templates...$(NC)"
	@$(PYTHON) tenant_utils.py list --format table

list-templates-json: ## List templates in JSON format
	@$(PYTHON) tenant_utils.py list --format json

export-template: ## Export template to YAML
	@echo -e "$(YELLOW)📤 Exporting template...$(NC)"
	@read -p "Enter template ID: " template_id; \
	read -p "Enter output file: " output_file; \
	$(PYTHON) tenant_utils.py export "$$template_id" "$$output_file"

clone-template: ## Clone existing template
	@echo -e "$(YELLOW)🔄 Cloning template...$(NC)"
	@read -p "Enter source template ID: " source_id; \
	read -p "Enter new template name: " new_name; \
	read -p "Enter tier upgrade (optional): " upgrade; \
	if [ -n "$$upgrade" ]; then \
		$(PYTHON) tenant_utils.py clone "$$source_id" "$$new_name" --upgrade "$$upgrade"; \
	else \
		$(PYTHON) tenant_utils.py clone "$$source_id" "$$new_name"; \
	fi

validate-template: ## Validate specific template
	@echo -e "$(YELLOW)🔍 Validating template...$(NC)"
	@read -p "Enter template ID: " template_id; \
	$(PYTHON) tenant_utils.py validate "$$template_id"

optimize-template: ## AI-optimize template
	@echo -e "$(YELLOW)🤖 Optimizing template with AI...$(NC)"
	@read -p "Enter template ID: " template_id; \
	$(PYTHON) tenant_utils.py optimize "$$template_id"

compliance-report: ## Generate compliance report
	@echo -e "$(YELLOW)📊 Generating compliance report...$(NC)"
	@read -p "Enter template ID: " template_id; \
	read -p "Enter output file (optional): " output_file; \
	if [ -n "$$output_file" ]; then \
		$(PYTHON) tenant_utils.py compliance "$$template_id" --output "$$output_file"; \
	else \
		$(PYTHON) tenant_utils.py compliance "$$template_id"; \
	fi

batch-create: ## Batch create templates
	@echo -e "$(YELLOW)📦 Batch creating templates...$(NC)"
	@$(PYTHON) tenant_utils.py batch-create batch_deployment.json

cleanup: ## Cleanup old templates (dry run)
	@echo -e "$(YELLOW)🧹 Cleaning up old templates (dry run)...$(NC)"
	@$(PYTHON) tenant_utils.py cleanup --days 30

cleanup-execute: ## Cleanup old templates (execute)
	@echo -e "$(YELLOW)🧹 Cleaning up old templates (executing)...$(NC)"
	@read -p "Are you sure? This will permanently delete old templates (y/N): " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		$(PYTHON) tenant_utils.py cleanup --days 30 --execute; \
	else \
		echo "Cleanup cancelled"; \
	fi

# Docker Commands
docker-build: ## Build Docker images
	@echo -e "$(YELLOW)🐳 Building Docker images...$(NC)"
	@$(DOCKER) build -t tenant-templates:latest .
	@echo -e "$(GREEN)✅ Docker images built$(NC)"

docker-run: ## Run with Docker Compose
	@echo -e "$(YELLOW)🐳 Starting services with Docker Compose...$(NC)"
	@$(DOCKER_COMPOSE) up -d
	@echo -e "$(GREEN)✅ Services started$(NC)"

docker-stop: ## Stop Docker services
	@echo -e "$(YELLOW)🐳 Stopping Docker services...$(NC)"
	@$(DOCKER_COMPOSE) down
	@echo -e "$(GREEN)✅ Services stopped$(NC)"

docker-logs: ## View Docker logs
	@$(DOCKER_COMPOSE) logs -f

# Kubernetes Commands
k8s-deploy: ## Deploy to Kubernetes
	@echo -e "$(YELLOW)☸️ Deploying to Kubernetes...$(NC)"
	@$(KUBECTL) apply -f k8s/
	@echo -e "$(GREEN)✅ Kubernetes deployment completed$(NC)"

k8s-status: ## Check Kubernetes status
	@$(KUBECTL) get pods,services,deployments -l app=tenant-templates

k8s-logs: ## View Kubernetes logs
	@$(KUBECTL) logs -l app=tenant-templates -f

k8s-delete: ## Delete Kubernetes resources
	@$(KUBECTL) delete -f k8s/

# Code Quality Commands
lint: ## Run code linting
	@echo -e "$(YELLOW)🔍 Running code linting...$(NC)"
	@flake8 . --max-line-length=120 --exclude=venv,__pycache__,.git
	@black --check .
	@isort --check-only .
	@echo -e "$(GREEN)✅ Code linting completed$(NC)"

format: ## Format code
	@echo -e "$(YELLOW)✨ Formatting code...$(NC)"
	@black .
	@isort .
	@echo -e "$(GREEN)✅ Code formatting completed$(NC)"

type-check: ## Run type checking
	@echo -e "$(YELLOW)🔍 Running type checking...$(NC)"
	@mypy . --ignore-missing-imports
	@echo -e "$(GREEN)✅ Type checking completed$(NC)"

# Security Commands
security: ## Run security scans
	@echo -e "$(YELLOW)🔒 Running security scans...$(NC)"
	@bandit -r . -f json -o security_report.json || true
	@safety check --json --output safety_report.json || true
	@echo -e "$(GREEN)✅ Security scans completed$(NC)"

security-report: ## Generate security report
	@echo -e "$(YELLOW)📋 Generating security report...$(NC)"
	@if [ -f security_report.json ]; then \
		echo "Bandit Security Report:"; \
		cat security_report.json | jq '.results[] | .issue_text' 2>/dev/null || cat security_report.json; \
	fi
	@if [ -f safety_report.json ]; then \
		echo "Safety Security Report:"; \
		cat safety_report.json | jq '.vulnerabilities[] | .advisory' 2>/dev/null || cat safety_report.json; \
	fi

# Performance Commands
performance: ## Run performance benchmarks
	@echo -e "$(YELLOW)⚡ Running performance benchmarks...$(NC)"
	@$(PYTHON) -m pytest tests/performance/ -v --benchmark-only
	@echo -e "$(GREEN)✅ Performance benchmarks completed$(NC)"

load-test: ## Run load tests
	@echo -e "$(YELLOW)🔥 Running load tests...$(NC)"
	@locust -f tests/load/locustfile.py --headless -u 100 -r 10 -t 60s --host=http://localhost:8000
	@echo -e "$(GREEN)✅ Load tests completed$(NC)"

# Documentation Commands
docs: ## Generate documentation
	@echo -e "$(YELLOW)📚 Generating documentation...$(NC)"
	@mkdir -p docs/generated
	@$(PYTHON) -c "
import sys; sys.path.append('.')
from tenant_utils import TenantTemplateUtility
help(TenantTemplateUtility)
" > docs/generated/api_docs.txt
	@echo -e "$(GREEN)✅ Documentation generated$(NC)"

docs-serve: ## Serve documentation locally
	@echo -e "$(YELLOW)📚 Serving documentation...$(NC)"
	@cd docs && python -m http.server 8080

# Monitoring Commands
monitor: ## Start monitoring dashboard
	@echo -e "$(YELLOW)📊 Starting monitoring dashboard...$(NC)"
	@$(DOCKER_COMPOSE) -f docker-compose.monitoring.yml up -d
	@echo -e "$(GREEN)✅ Monitoring dashboard started$(NC)"
	@echo -e "$(BLUE)Access Grafana at: http://localhost:3000$(NC)"
	@echo -e "$(BLUE)Access Prometheus at: http://localhost:9090$(NC)"

monitor-stop: ## Stop monitoring dashboard
	@$(DOCKER_COMPOSE) -f docker-compose.monitoring.yml down

logs: ## View application logs
	@tail -f $(LOGS_DIR)/*.log

# Clean Commands
clean: ## Clean temporary files
	@echo -e "$(YELLOW)🧹 Cleaning temporary files...$(NC)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache .coverage htmlcov
	@rm -f *.log security_report.json safety_report.json
	@echo -e "$(GREEN)✅ Cleanup completed$(NC)"

clean-all: clean ## Clean everything including Docker
	@echo -e "$(YELLOW)🧹 Cleaning everything...$(NC)"
	@$(DOCKER) system prune -f
	@$(DOCKER) volume prune -f
	@echo -e "$(GREEN)✅ Complete cleanup finished$(NC)"

# Development Commands
dev-setup: install setup ## Setup development environment
	@echo -e "$(YELLOW)🛠️ Setting up development environment...$(NC)"
	@$(PIP) install -r requirements-dev.txt
	@pre-commit install
	@echo -e "$(GREEN)✅ Development environment ready$(NC)"

dev-run: ## Run development server
	@echo -e "$(YELLOW)🚀 Starting development server...$(NC)"
	@uvicorn main:app --reload --host 0.0.0.0 --port 8000

dev-test: ## Run tests in development mode
	@$(PYTHON) -m pytest -v --tb=short

# Release Commands
version: ## Show version information
	@echo -e "$(BLUE)Enterprise Tenant Templates Management System$(NC)"
	@echo -e "$(BLUE)Version: 2.0.0$(NC)"
	@echo -e "$(BLUE)Expert Team: Fahed Mlaiel and Associates$(NC)"
	@echo -e "$(BLUE)Architecture: Ultra-Advanced Industrial Multi-Tenant$(NC)"

release: test lint security ## Prepare release
	@echo -e "$(YELLOW)🎉 Preparing release...$(NC)"
	@git tag -a v2.0.0 -m "Enterprise Tenant Templates v2.0.0"
	@echo -e "$(GREEN)✅ Release prepared$(NC)"

# All-in-one commands
all: install setup validate test deploy ## Run complete setup and deployment
	@echo -e "$(GREEN)🎉 Complete setup and deployment finished!$(NC)"

ci: install validate lint type-check test security ## Run CI pipeline
	@echo -e "$(GREEN)✅ CI pipeline completed$(NC)"

# Help for specific categories
help-templates: ## Show template management commands
	@echo -e "$(GREEN)Template Management Commands:$(NC)"
	@echo -e "  $(BLUE)create-template$(NC)    Create template from YAML"
	@echo -e "  $(BLUE)list-templates$(NC)     List all templates"
	@echo -e "  $(BLUE)export-template$(NC)    Export template to YAML"
	@echo -e "  $(BLUE)clone-template$(NC)     Clone existing template"
	@echo -e "  $(BLUE)validate-template$(NC)  Validate specific template"
	@echo -e "  $(BLUE)optimize-template$(NC)  AI-optimize template"
	@echo -e "  $(BLUE)compliance-report$(NC)  Generate compliance report"
	@echo -e "  $(BLUE)batch-create$(NC)       Batch create templates"
	@echo -e "  $(BLUE)cleanup$(NC)            Cleanup old templates"

help-deploy: ## Show deployment commands
	@echo -e "$(GREEN)Deployment Commands:$(NC)"
	@echo -e "  $(BLUE)deploy$(NC)             Deploy enterprise system"
	@echo -e "  $(BLUE)deploy-dev$(NC)         Deploy development environment"
	@echo -e "  $(BLUE)deploy-staging$(NC)     Deploy staging environment"
	@echo -e "  $(BLUE)deploy-production$(NC)  Deploy production environment"
	@echo -e "  $(BLUE)rollback$(NC)           Rollback deployment"
	@echo -e "  $(BLUE)health$(NC)             Check system health"
	@echo -e "  $(BLUE)backup$(NC)             Create system backup"
	@echo -e "  $(BLUE)restore$(NC)            Restore from backup"

help-dev: ## Show development commands
	@echo -e "$(GREEN)Development Commands:$(NC)"
	@echo -e "  $(BLUE)dev-setup$(NC)          Setup development environment"
	@echo -e "  $(BLUE)dev-run$(NC)            Run development server"
	@echo -e "  $(BLUE)dev-test$(NC)           Run tests in development mode"
	@echo -e "  $(BLUE)lint$(NC)               Run code linting"
	@echo -e "  $(BLUE)format$(NC)             Format code"
	@echo -e "  $(BLUE)type-check$(NC)         Run type checking"
	@echo -e "  $(BLUE)security$(NC)           Run security scans"
