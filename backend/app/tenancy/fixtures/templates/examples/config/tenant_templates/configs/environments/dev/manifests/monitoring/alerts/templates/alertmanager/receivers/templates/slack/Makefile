# ==================================================================================
# SPOTIFY AI AGENT - ENTERPRISE TEMPLATE SYSTEM MAKEFILE
# ==================================================================================
# Description: Industrial-grade automation for template development and deployment
# Version: 2.0.0
# Author: Fahed Mlaiel (Lead Dev + AI Architect)
# Last Modified: 2025-07-18
# ==================================================================================

.PHONY: help install validate test deploy rollback clean dev prod staging
.DEFAULT_GOAL := help

# Configuration
PROJECT_NAME := spotify-ai-agent-templates
VERSION := 2.0.0
PYTHON := python3
PIP := pip3
DOCKER := docker
KUBECTL := kubectl

# Directories
TEMPLATE_DIR := .
CONFIG_DIR := $(TEMPLATE_DIR)
BACKUP_DIR := /var/backups/spotify-ai-agent/templates
LOG_DIR := /var/log/spotify-ai-agent

# Environment variables
ENVIRONMENT ?= development
DRY_RUN ?= false
FORCE ?= false

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# ==================================================================================
# HELP
# ==================================================================================

help: ## Show this help message
	@echo "$(BLUE)Spotify AI Agent - Enterprise Template System v$(VERSION)$(NC)"
	@echo "================================================================"
	@echo ""
	@echo "$(GREEN)Available commands:$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Environment variables:$(NC)"
	@echo "  ENVIRONMENT    Target environment (development|staging|production)"
	@echo "  DRY_RUN        Perform dry run without making changes (true|false)"
	@echo "  FORCE          Force operations despite warnings (true|false)"
	@echo ""
	@echo "$(GREEN)Examples:$(NC)"
	@echo "  make install                    # Install dependencies"
	@echo "  make validate                   # Validate templates"
	@echo "  make deploy ENVIRONMENT=staging # Deploy to staging"
	@echo "  make rollback                   # Rollback to latest backup"
	@echo ""

# ==================================================================================
# INSTALLATION AND SETUP
# ==================================================================================

install: ## Install dependencies and setup environment
	@echo "$(BLUE)Installing dependencies...$(NC)"
	@$(PIP) install --upgrade pip
	@$(PIP) install jinja2 pyyaml redis aiofiles click
	@$(PIP) install yamllint pytest pytest-asyncio
	@echo "$(GREEN)Dependencies installed successfully$(NC)"

install-dev: install ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	@$(PIP) install black flake8 mypy pre-commit
	@$(PIP) install pytest-cov pytest-mock
	@echo "$(GREEN)Development dependencies installed$(NC)"

setup: install ## Setup the complete environment
	@echo "$(BLUE)Setting up environment...$(NC)"
	@mkdir -p $(BACKUP_DIR)
	@mkdir -p $(LOG_DIR)
	@chmod +x deploy.sh rollback.sh
	@echo "$(GREEN)Environment setup completed$(NC)"

# ==================================================================================
# VALIDATION AND TESTING
# ==================================================================================

validate: ## Validate all templates and configuration
	@echo "$(BLUE)Validating templates and configuration...$(NC)"
	@$(PYTHON) template_validator.py --config template_config.yaml --templates-dir $(TEMPLATE_DIR)
	@yamllint template_config.yaml
	@yamllint locales/*.yaml
	@echo "$(GREEN)Validation completed successfully$(NC)"

validate-syntax: ## Validate template syntax only
	@echo "$(BLUE)Validating template syntax...$(NC)"
	@$(PYTHON) -c "\
import os; \
from jinja2 import Environment, FileSystemLoader, TemplateSyntaxError; \
env = Environment(loader=FileSystemLoader('$(TEMPLATE_DIR)')); \
[env.get_template(f) for f in os.listdir('$(TEMPLATE_DIR)') if f.endswith('.j2')]; \
print('✅ All templates have valid syntax')"

validate-config: ## Validate configuration files
	@echo "$(BLUE)Validating configuration...$(NC)"
	@yamllint template_config.yaml
	@$(PYTHON) -c "\
import yaml; \
config = yaml.safe_load(open('template_config.yaml')); \
required = ['template_engine', 'security', 'performance']; \
assert all(k in config for k in required), 'Missing required sections'; \
print('✅ Configuration is valid')"

lint: ## Run linting on Python files
	@echo "$(BLUE)Running linting...$(NC)"
	@flake8 *.py --max-line-length=100 --ignore=E501,W503
	@black --check *.py
	@mypy *.py --ignore-missing-imports
	@echo "$(GREEN)Linting completed$(NC)"

format: ## Format Python code
	@echo "$(BLUE)Formatting Python code...$(NC)"
	@black *.py
	@echo "$(GREEN)Code formatting completed$(NC)"

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	@$(PYTHON) -m pytest tests/ -v --cov=. --cov-report=term-missing
	@echo "$(GREEN)Tests completed$(NC)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	@$(PYTHON) -m pytest tests/unit/ -v
	@echo "$(GREEN)Unit tests completed$(NC)"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	@$(PYTHON) -m pytest tests/integration/ -v
	@echo "$(GREEN)Integration tests completed$(NC)"

# ==================================================================================
# DEPLOYMENT
# ==================================================================================

deploy: validate ## Deploy templates to specified environment
	@echo "$(BLUE)Deploying to $(ENVIRONMENT) environment...$(NC)"
	@./deploy.sh --environment $(ENVIRONMENT) $(if $(filter true,$(DRY_RUN)),--dry-run) $(if $(filter true,$(FORCE)),--force)
	@echo "$(GREEN)Deployment completed$(NC)"

deploy-dev: ## Deploy to development environment
	@$(MAKE) deploy ENVIRONMENT=development

deploy-staging: ## Deploy to staging environment
	@$(MAKE) deploy ENVIRONMENT=staging

deploy-prod: ## Deploy to production environment
	@$(MAKE) deploy ENVIRONMENT=production FORCE=false

deploy-dry-run: ## Perform dry run deployment
	@$(MAKE) deploy DRY_RUN=true

rollback: ## Rollback to latest backup
	@echo "$(BLUE)Rolling back to latest backup...$(NC)"
	@./rollback.sh --environment $(ENVIRONMENT) $(if $(filter true,$(DRY_RUN)),--dry-run)
	@echo "$(GREEN)Rollback completed$(NC)"

rollback-to: ## Rollback to specific backup (usage: make rollback-to BACKUP=backup_name)
	@echo "$(BLUE)Rolling back to backup: $(BACKUP)...$(NC)"
	@./rollback.sh --environment $(ENVIRONMENT) $(BACKUP) $(if $(filter true,$(DRY_RUN)),--dry-run)
	@echo "$(GREEN)Rollback completed$(NC)"

list-backups: ## List available backups
	@./rollback.sh --list

# ==================================================================================
# DEVELOPMENT
# ==================================================================================

dev: ## Start development environment
	@echo "$(BLUE)Starting development environment...$(NC)"
	@$(PYTHON) template_manager.py --config template_config.yaml --debug
	@echo "$(GREEN)Development server started$(NC)"

watch: ## Watch for template changes and auto-reload
	@echo "$(BLUE)Watching for changes...$(NC)"
	@while true; do \
		inotifywait -r -e modify,create,delete $(TEMPLATE_DIR)/ && \
		$(MAKE) validate-syntax; \
	done

generate-sample: ## Generate sample alert data for testing
	@echo "$(BLUE)Generating sample alert data...$(NC)"
	@$(PYTHON) -c "\
import json; \
sample = { \
    'alert': { \
        'alert_id': 'test-123', \
        'title': 'High CPU Usage', \
        'severity': 'warning', \
        'created_at': '2025-01-18T10:00:00Z', \
        'context': { \
            'service_name': 'web-api', \
            'cpu_usage': 85.5, \
            'threshold': 80.0 \
        } \
    }, \
    'environment': 'staging', \
    'tenant_id': 'test-tenant' \
}; \
print(json.dumps(sample, indent=2))" > sample_alert.json
	@echo "$(GREEN)Sample data generated: sample_alert.json$(NC)"

test-render: ## Test template rendering with sample data
	@echo "$(BLUE)Testing template rendering...$(NC)"
	@$(MAKE) generate-sample
	@$(PYTHON) -c "\
import json; \
import sys; \
sys.path.append('.'); \
from template_manager import SlackTemplateManager; \
manager = SlackTemplateManager(config_path='template_config.yaml'); \
with open('sample_alert.json') as f: data = json.load(f); \
result = manager.render_template('warning', data, 'en', 'text'); \
print('Rendered template:'); \
print('=' * 50); \
print(result); \
print('=' * 50)"
	@echo "$(GREEN)Template rendering test completed$(NC)"

# ==================================================================================
# MONITORING AND METRICS
# ==================================================================================

metrics: ## Display template usage metrics
	@echo "$(BLUE)Fetching template metrics...$(NC)"
	@curl -s http://localhost:8080/metrics | grep template_ || echo "No metrics available"

health: ## Check system health
	@echo "$(BLUE)Checking system health...$(NC)"
	@curl -s http://localhost:8080/health || echo "Service not available"
	@$(PYTHON) -c "\
import redis; \
r = redis.Redis(host='localhost', port=6379, db=0); \
r.ping(); \
print('✅ Redis connection OK')" || echo "❌ Redis not available"

logs: ## Show recent logs
	@echo "$(BLUE)Recent template system logs:$(NC)"
	@tail -n 50 $(LOG_DIR)/template-manager.log 2>/dev/null || echo "No logs available"

monitor: ## Start monitoring dashboard
	@echo "$(BLUE)Starting monitoring dashboard...$(NC)"
	@open http://localhost:8080/dashboard 2>/dev/null || echo "Dashboard URL: http://localhost:8080/dashboard"

# ==================================================================================
# MAINTENANCE
# ==================================================================================

clean: ## Clean temporary files and caches
	@echo "$(BLUE)Cleaning temporary files...$(NC)"
	@rm -rf __pycache__/
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@rm -f *.pyc
	@rm -f sample_alert.json
	@rm -f .deployment_manifest
	@echo "$(GREEN)Cleanup completed$(NC)"

clean-logs: ## Clean old log files
	@echo "$(BLUE)Cleaning old logs...$(NC)"
	@find $(LOG_DIR) -name "*.log" -mtime +30 -delete 2>/dev/null || true
	@echo "$(GREEN)Old logs cleaned$(NC)"

clean-backups: ## Clean old backup files
	@echo "$(BLUE)Cleaning old backups...$(NC)"
	@find $(BACKUP_DIR) -name "*.tar.gz" -mtime +90 -delete 2>/dev/null || true
	@echo "$(GREEN)Old backups cleaned$(NC)"

backup-now: ## Create immediate backup
	@echo "$(BLUE)Creating backup...$(NC)"
	@mkdir -p $(BACKUP_DIR)
	@tar -czf $(BACKUP_DIR)/manual_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz .
	@echo "$(GREEN)Backup created$(NC)"

# ==================================================================================
# DOCKER OPERATIONS
# ==================================================================================

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	@$(DOCKER) build -t $(PROJECT_NAME):$(VERSION) .
	@$(DOCKER) tag $(PROJECT_NAME):$(VERSION) $(PROJECT_NAME):latest
	@echo "$(GREEN)Docker image built$(NC)"

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	@$(DOCKER) run -d \
		--name $(PROJECT_NAME) \
		-p 8080:8080 \
		-v $(PWD):/app/templates \
		$(PROJECT_NAME):latest
	@echo "$(GREEN)Docker container started$(NC)"

docker-stop: ## Stop Docker container
	@echo "$(BLUE)Stopping Docker container...$(NC)"
	@$(DOCKER) stop $(PROJECT_NAME) || true
	@$(DOCKER) rm $(PROJECT_NAME) || true
	@echo "$(GREEN)Docker container stopped$(NC)"

# ==================================================================================
# KUBERNETES OPERATIONS
# ==================================================================================

k8s-deploy: ## Deploy to Kubernetes
	@echo "$(BLUE)Deploying to Kubernetes...$(NC)"
	@$(KUBECTL) apply -f k8s/
	@$(KUBECTL) rollout status deployment/template-manager
	@echo "$(GREEN)Kubernetes deployment completed$(NC)"

k8s-status: ## Check Kubernetes status
	@echo "$(BLUE)Kubernetes status:$(NC)"
	@$(KUBECTL) get pods -l app=template-manager
	@$(KUBECTL) get services -l app=template-manager

k8s-logs: ## Show Kubernetes logs
	@echo "$(BLUE)Kubernetes logs:$(NC)"
	@$(KUBECTL) logs -l app=template-manager --tail=50

# ==================================================================================
# SECURITY
# ==================================================================================

security-scan: ## Run security scan
	@echo "$(BLUE)Running security scan...$(NC)"
	@grep -r -i "password\|secret\|key\|token" . --include="*.j2" --include="*.py" --include="*.yaml" || echo "No hardcoded secrets found"
	@find . -type f -perm /o+w -name "*.j2" -o -name "*.py" | head -10 || echo "No world-writable files found"
	@echo "$(GREEN)Security scan completed$(NC)"

audit: ## Run security audit
	@echo "$(BLUE)Running security audit...$(NC)"
	@pip-audit || echo "pip-audit not available"
	@$(MAKE) security-scan
	@echo "$(GREEN)Security audit completed$(NC)"

# ==================================================================================
# DOCUMENTATION
# ==================================================================================

docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@mkdir -p docs/
	@echo "# Template System Documentation" > docs/README.md
	@echo "" >> docs/README.md
	@echo "Generated on: $(shell date)" >> docs/README.md
	@echo "Version: $(VERSION)" >> docs/README.md
	@echo "$(GREEN)Documentation generated$(NC)"

# ==================================================================================
# INFORMATION
# ==================================================================================

info: ## Show system information
	@echo "$(BLUE)System Information:$(NC)"
	@echo "Project: $(PROJECT_NAME)"
	@echo "Version: $(VERSION)"
	@echo "Environment: $(ENVIRONMENT)"
	@echo "Template Directory: $(TEMPLATE_DIR)"
	@echo "Python Version: $(shell $(PYTHON) --version)"
	@echo "Current User: $(shell whoami)"
	@echo "Current Directory: $(shell pwd)"
	@echo "Git Branch: $(shell git branch --show-current 2>/dev/null || echo 'Not a git repository')"
	@echo "Git Commit: $(shell git rev-parse --short HEAD 2>/dev/null || echo 'Not available')"

version: ## Show version information
	@echo "$(GREEN)$(PROJECT_NAME) v$(VERSION)$(NC)"

# ==================================================================================
# ALIASES
# ==================================================================================

# Short aliases for common commands
v: validate
d: deploy-dev
s: deploy-staging
p: deploy-prod
t: test
c: clean
h: help
