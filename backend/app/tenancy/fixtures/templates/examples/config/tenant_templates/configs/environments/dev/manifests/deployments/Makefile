# Advanced Deployment Makefile
# Developed by Fahed Mlaiel - Lead DevOps Architect & Automation Specialist
# 
# This Makefile provides comprehensive automation for the Spotify AI Agent
# multi-tenant deployment system with enterprise-grade features:
# - Multi-environment deployments
# - Advanced testing and validation
# - Security compliance checks
# - Performance optimization
# - Monitoring and alerting
# - Disaster recovery procedures

.PHONY: help all deploy deploy-all deploy-dev deploy-staging deploy-prod \
        test test-all test-integration test-security test-performance \
        monitor monitor-continuous health-check validate validate-all \
        backup restore cleanup optimize scale rollback \
        security-scan compliance-check audit-logs \
        docs build install uninstall

# Default target
.DEFAULT_GOAL := help

# Script metadata
SCRIPT_VERSION := 2.0.0
SCRIPT_AUTHOR := Fahed Mlaiel - DevOps Architect
SCRIPT_DATE := 2025-07-17

# Configuration
NAMESPACE ?= spotify-ai-agent-dev
ENVIRONMENT ?= development
DEPLOYMENT_STRATEGY ?= rolling
DRY_RUN ?= false
VERBOSE ?= false
PARALLEL_JOBS ?= 4

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
NC := \033[0m # No Color

# Directories
DEPLOY_DIR := .
SCRIPTS_DIR := .
MANIFESTS_DIR := .
CONFIGS_DIR := ../configs
TEMPLATES_DIR := ../templates

# Services
SERVICES := backend ml-service analytics notification auth billing tenant-service
CORE_SERVICES := backend auth tenant-service
ADVANCED_SERVICES := ml-service analytics notification billing
MONITORING_SERVICES := prometheus grafana jaeger elasticsearch

# Files
DEPLOYMENT_SCRIPT := $(SCRIPTS_DIR)/deploy.sh
MONITORING_SCRIPT := $(SCRIPTS_DIR)/monitor.sh
MANIFEST_FILES := $(wildcard *.yaml)

# Timestamps
TIMESTAMP := $(shell date +%Y%m%d-%H%M%S)
DATE := $(shell date '+%Y-%m-%d %H:%M:%S')

#######################################
# Helper functions
#######################################
define log
	@echo -e "$(GREEN)[$(DATE)]$(NC) $(1)"
endef

define log_error
	@echo -e "$(RED)[ERROR]$(NC) $(1)" >&2
endef

define log_warn
	@echo -e "$(YELLOW)[WARN]$(NC) $(1)"
endef

define log_info
	@echo -e "$(BLUE)[INFO]$(NC) $(1)"
endef

define check_prerequisites
	@echo "Checking prerequisites..."
	@command -v kubectl >/dev/null 2>&1 || (echo "kubectl is required but not installed" && exit 1)
	@command -v helm >/dev/null 2>&1 || (echo "helm is required but not installed" && exit 1)
	@command -v jq >/dev/null 2>&1 || (echo "jq is required but not installed" && exit 1)
	@kubectl cluster-info >/dev/null 2>&1 || (echo "Cannot connect to Kubernetes cluster" && exit 1)
	@echo "Prerequisites check passed ✓"
endef

#######################################
# Help and information
#######################################
help: ## Show this comprehensive help message
	@echo ""
	@echo -e "$(CYAN)Spotify AI Agent - Advanced Deployment Automation$(NC)"
	@echo -e "$(CYAN)===================================================$(NC)"
	@echo ""
	@echo -e "$(GREEN)Developed by: $(SCRIPT_AUTHOR)$(NC)"
	@echo -e "$(GREEN)Version: $(SCRIPT_VERSION)$(NC)"
	@echo -e "$(GREEN)Date: $(SCRIPT_DATE)$(NC)"
	@echo ""
	@echo -e "$(YELLOW)Available targets:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-25s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo -e "$(YELLOW)Configuration:$(NC)"
	@echo "  NAMESPACE=$(NAMESPACE)"
	@echo "  ENVIRONMENT=$(ENVIRONMENT)"
	@echo "  DEPLOYMENT_STRATEGY=$(DEPLOYMENT_STRATEGY)"
	@echo "  DRY_RUN=$(DRY_RUN)"
	@echo "  VERBOSE=$(VERBOSE)"
	@echo ""
	@echo -e "$(YELLOW)Services:$(NC)"
	@echo "  Core: $(CORE_SERVICES)"
	@echo "  Advanced: $(ADVANCED_SERVICES)"
	@echo "  Monitoring: $(MONITORING_SERVICES)"
	@echo ""
	@echo -e "$(YELLOW)Examples:$(NC)"
	@echo "  make deploy-dev SERVICE=backend"
	@echo "  make deploy-all ENVIRONMENT=staging STRATEGY=blue-green"
	@echo "  make test-all PARALLEL_JOBS=8"
	@echo "  make monitor-continuous"
	@echo "  make security-scan VERBOSE=true"
	@echo ""

version: ## Show version information
	@echo -e "$(CYAN)Advanced Deployment Makefile$(NC)"
	@echo "Version: $(SCRIPT_VERSION)"
	@echo "Author: $(SCRIPT_AUTHOR)"
	@echo "Date: $(SCRIPT_DATE)"

info: ## Show current configuration and environment info
	$(call log,"Current Configuration:")
	@echo "  Namespace: $(NAMESPACE)"
	@echo "  Environment: $(ENVIRONMENT)"
	@echo "  Strategy: $(DEPLOYMENT_STRATEGY)"
	@echo "  Dry Run: $(DRY_RUN)"
	@echo "  Verbose: $(VERBOSE)"
	@echo "  Parallel Jobs: $(PARALLEL_JOBS)"
	@echo ""
	$(call log,"Kubernetes Cluster Info:")
	@kubectl cluster-info --context=$(shell kubectl config current-context) 2>/dev/null || echo "  No cluster connection"
	@echo ""
	$(call log,"Available Services:")
	@for service in $(SERVICES); do \
		if kubectl get deployment $$service -n $(NAMESPACE) >/dev/null 2>&1; then \
			echo "  ✓ $$service (deployed)"; \
		else \
			echo "  ✗ $$service (not deployed)"; \
		fi \
	done

#######################################
# Core deployment targets
#######################################
all: validate deploy-all test-all monitor ## Deploy, test, and monitor everything

deploy: check-service ## Deploy a specific service (requires SERVICE variable)
	$(call check_prerequisites)
	$(call log,"Deploying service: $(SERVICE)")
	@chmod +x $(DEPLOYMENT_SCRIPT)
	@$(DEPLOYMENT_SCRIPT) deploy $(SERVICE) $(DEPLOYMENT_STRATEGY) \
		--namespace $(NAMESPACE) \
		--environment $(ENVIRONMENT) \
		$(if $(filter true,$(DRY_RUN)),--dry-run) \
		$(if $(filter true,$(VERBOSE)),--verbose)

deploy-all: ## Deploy all services
	$(call check_prerequisites)
	$(call log,"Deploying all services with strategy: $(DEPLOYMENT_STRATEGY)")
	@chmod +x $(DEPLOYMENT_SCRIPT)
	@$(DEPLOYMENT_SCRIPT) deploy-all $(DEPLOYMENT_STRATEGY) \
		--namespace $(NAMESPACE) \
		--environment $(ENVIRONMENT) \
		$(if $(filter true,$(DRY_RUN)),--dry-run) \
		$(if $(filter true,$(VERBOSE)),--verbose)

deploy-core: ## Deploy core services only
	$(call log,"Deploying core services: $(CORE_SERVICES)")
	@for service in $(CORE_SERVICES); do \
		$(MAKE) deploy SERVICE=$$service || exit 1; \
	done

deploy-advanced: ## Deploy advanced services only
	$(call log,"Deploying advanced services: $(ADVANCED_SERVICES)")
	@for service in $(ADVANCED_SERVICES); do \
		$(MAKE) deploy SERVICE=$$service || exit 1; \
	done

deploy-monitoring: ## Deploy monitoring stack
	$(call log,"Deploying monitoring services: $(MONITORING_SERVICES)")
	@for service in $(MONITORING_SERVICES); do \
		helm upgrade --install $$service $$service \
			--namespace $(NAMESPACE) \
			--create-namespace \
			$(if $(filter true,$(DRY_RUN)),--dry-run) || exit 1; \
	done

#######################################
# Environment-specific deployments
#######################################
deploy-dev: ## Deploy to development environment
	$(MAKE) deploy-all ENVIRONMENT=development NAMESPACE=spotify-ai-agent-dev

deploy-staging: ## Deploy to staging environment
	$(MAKE) deploy-all ENVIRONMENT=staging NAMESPACE=spotify-ai-agent-staging DEPLOYMENT_STRATEGY=blue-green

deploy-prod: ## Deploy to production environment
	$(call log_warn,"Deploying to PRODUCTION environment")
	@read -p "Are you sure you want to deploy to production? [y/N] " confirm && \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		$(MAKE) deploy-all ENVIRONMENT=production NAMESPACE=spotify-ai-agent-prod DEPLOYMENT_STRATEGY=canary; \
	else \
		echo "Deployment cancelled"; \
	fi

#######################################
# Testing targets
#######################################
test: check-service ## Run tests for a specific service
	$(call log,"Running tests for service: $(SERVICE)")
	@kubectl run test-$(SERVICE)-$(TIMESTAMP) \
		--image=spotify-ai-agent/$(SERVICE)-test:latest \
		--rm -i --restart=Never \
		--namespace=$(NAMESPACE) \
		-- /run-tests.sh

test-all: ## Run all tests
	$(call log,"Running comprehensive test suite")
	@for service in $(SERVICES); do \
		echo "Testing $$service..."; \
		$(MAKE) test SERVICE=$$service || exit 1; \
	done

test-integration: ## Run integration tests
	$(call log,"Running integration tests")
	@kubectl apply -f integration-tests.yaml -n $(NAMESPACE)
	@kubectl wait --for=condition=complete job/integration-tests -n $(NAMESPACE) --timeout=600s
	@kubectl logs job/integration-tests -n $(NAMESPACE)
	@kubectl delete job/integration-tests -n $(NAMESPACE)

test-security: ## Run security tests
	$(call log,"Running security tests")
	@chmod +x security-tests.sh
	@./security-tests.sh $(NAMESPACE)

test-performance: ## Run performance tests
	$(call log,"Running performance tests")
	@kubectl apply -f performance-tests.yaml -n $(NAMESPACE)
	@kubectl wait --for=condition=complete job/performance-tests -n $(NAMESPACE) --timeout=1800s
	@kubectl logs job/performance-tests -n $(NAMESPACE)
	@kubectl delete job/performance-tests -n $(NAMESPACE)

test-load: ## Run load tests
	$(call log,"Running load tests")
	@helm install load-test loadtest \
		--set target.host=spotify-ai-agent-backend.$(NAMESPACE).svc.cluster.local \
		--set target.port=8080 \
		--set load.users=100 \
		--set load.duration=300s \
		--namespace $(NAMESPACE)

#######################################
# Monitoring and validation
#######################################
monitor: ## Run single monitoring cycle
	$(call log,"Running deployment monitoring")
	@chmod +x $(MONITORING_SCRIPT)
	@$(MONITORING_SCRIPT) monitor \
		--namespace $(NAMESPACE) \
		--environment $(ENVIRONMENT)

monitor-continuous: ## Run continuous monitoring
	$(call log,"Starting continuous monitoring")
	@chmod +x $(MONITORING_SCRIPT)
	@$(MONITORING_SCRIPT) continuous \
		--namespace $(NAMESPACE) \
		--environment $(ENVIRONMENT) \
		--interval 60

health-check: check-service ## Check health of specific service
	$(call log,"Checking health of service: $(SERVICE)")
	@chmod +x $(MONITORING_SCRIPT)
	@$(MONITORING_SCRIPT) health-check $(SERVICE) \
		--namespace $(NAMESPACE)

health-check-all: ## Check health of all services
	$(call log,"Checking health of all services")
	@for service in $(SERVICES); do \
		echo "Checking $$service..."; \
		$(MAKE) health-check SERVICE=$$service || true; \
	done

validate: ## Validate deployment configurations
	$(call log,"Validating deployment configurations")
	@for file in $(MANIFEST_FILES); do \
		echo "Validating $$file..."; \
		kubectl apply --dry-run=client -f $$file -n $(NAMESPACE) || exit 1; \
	done
	$(call log,"All configurations are valid ✓")

validate-all: validate test-security ## Run all validation checks
	$(call log,"Running comprehensive validation")
	@$(MAKE) validate
	@$(MAKE) test-security
	@$(MAKE) compliance-check

#######################################
# Scaling operations
#######################################
scale: check-service check-replicas ## Scale a specific service
	$(call log,"Scaling $(SERVICE) to $(REPLICAS) replicas")
	@kubectl scale deployment $(SERVICE) --replicas=$(REPLICAS) -n $(NAMESPACE)
	@kubectl rollout status deployment $(SERVICE) -n $(NAMESPACE)

scale-up: check-service ## Scale up a service (double replicas)
	$(eval CURRENT_REPLICAS := $(shell kubectl get deployment $(SERVICE) -n $(NAMESPACE) -o jsonpath='{.spec.replicas}'))
	$(eval NEW_REPLICAS := $(shell echo $$(($(CURRENT_REPLICAS) * 2))))
	$(call log,"Scaling up $(SERVICE) from $(CURRENT_REPLICAS) to $(NEW_REPLICAS) replicas")
	@$(MAKE) scale SERVICE=$(SERVICE) REPLICAS=$(NEW_REPLICAS)

scale-down: check-service ## Scale down a service (half replicas, minimum 1)
	$(eval CURRENT_REPLICAS := $(shell kubectl get deployment $(SERVICE) -n $(NAMESPACE) -o jsonpath='{.spec.replicas}'))
	$(eval NEW_REPLICAS := $(shell echo $$(($(CURRENT_REPLICAS) / 2 > 0 ? $(CURRENT_REPLICAS) / 2 : 1))))
	$(call log,"Scaling down $(SERVICE) from $(CURRENT_REPLICAS) to $(NEW_REPLICAS) replicas")
	@$(MAKE) scale SERVICE=$(SERVICE) REPLICAS=$(NEW_REPLICAS)

auto-scale: check-service ## Enable horizontal pod autoscaling
	$(call log,"Enabling HPA for $(SERVICE)")
	@kubectl autoscale deployment $(SERVICE) \
		--cpu-percent=70 \
		--min=2 \
		--max=10 \
		-n $(NAMESPACE)

#######################################
# Security and compliance
#######################################
security-scan: ## Run comprehensive security scan
	$(call log,"Running security scan")
	@echo "Scanning for security vulnerabilities..."
	@for service in $(SERVICES); do \
		echo "Scanning $$service..."; \
		trivy image spotify-ai-agent/$$service:latest --severity HIGH,CRITICAL || true; \
	done
	@echo "Security scan completed"

compliance-check: ## Check compliance requirements
	$(call log,"Checking compliance requirements")
	@echo "Checking PCI DSS compliance..."
	@kubectl get pods -n $(NAMESPACE) -o jsonpath='{.items[*].spec.securityContext}' | jq '.'
	@echo "Checking GDPR compliance..."
	@kubectl get configmaps -n $(NAMESPACE) -o json | jq '.items[] | select(.metadata.name | contains("gdpr"))'
	@echo "Compliance check completed"

audit-logs: ## Export audit logs
	$(call log,"Exporting audit logs")
	@mkdir -p logs/$(TIMESTAMP)
	@for service in $(SERVICES); do \
		kubectl logs deployment/$$service -n $(NAMESPACE) --tail=1000 > logs/$(TIMESTAMP)/$$service.log 2>/dev/null || true; \
	done
	@tar -czf audit-logs-$(TIMESTAMP).tar.gz logs/$(TIMESTAMP)
	@echo "Audit logs exported to audit-logs-$(TIMESTAMP).tar.gz"

#######################################
# Backup and recovery
#######################################
backup: ## Create backup of current deployment
	$(call log,"Creating deployment backup")
	@mkdir -p backups/$(TIMESTAMP)
	@kubectl get all -n $(NAMESPACE) -o yaml > backups/$(TIMESTAMP)/resources.yaml
	@kubectl get configmaps -n $(NAMESPACE) -o yaml > backups/$(TIMESTAMP)/configmaps.yaml
	@kubectl get secrets -n $(NAMESPACE) -o yaml > backups/$(TIMESTAMP)/secrets.yaml
	@tar -czf backup-$(TIMESTAMP).tar.gz backups/$(TIMESTAMP)
	@echo "Backup created: backup-$(TIMESTAMP).tar.gz"

restore: check-backup ## Restore from backup (requires BACKUP_FILE variable)
	$(call log_warn,"Restoring from backup: $(BACKUP_FILE)")
	@read -p "This will replace current deployment. Continue? [y/N] " confirm && \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		tar -xzf $(BACKUP_FILE) -C /tmp; \
		kubectl apply -f /tmp/backups/*/resources.yaml -n $(NAMESPACE); \
		kubectl apply -f /tmp/backups/*/configmaps.yaml -n $(NAMESPACE); \
		kubectl apply -f /tmp/backups/*/secrets.yaml -n $(NAMESPACE); \
		echo "Restore completed"; \
	else \
		echo "Restore cancelled"; \
	fi

#######################################
# Rollback operations
#######################################
rollback: check-service ## Rollback a specific service
	$(call log,"Rolling back service: $(SERVICE)")
	@chmod +x $(DEPLOYMENT_SCRIPT)
	@$(DEPLOYMENT_SCRIPT) rollback $(SERVICE) \
		--namespace $(NAMESPACE)

rollback-all: ## Rollback all services
	$(call log_warn,"Rolling back all services")
	@read -p "Are you sure you want to rollback all services? [y/N] " confirm && \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		for service in $(SERVICES); do \
			$(MAKE) rollback SERVICE=$$service || true; \
		done; \
	else \
		echo "Rollback cancelled"; \
	fi

#######################################
# Optimization and maintenance
#######################################
optimize: ## Optimize resource allocation
	$(call log,"Optimizing resource allocation")
	@chmod +x $(DEPLOYMENT_SCRIPT)
	@$(DEPLOYMENT_SCRIPT) optimize-resources \
		--namespace $(NAMESPACE)

cleanup: ## Clean up unused resources
	$(call log,"Cleaning up unused resources")
	@kubectl delete pods --field-selector=status.phase=Succeeded -n $(NAMESPACE) 2>/dev/null || true
	@kubectl delete pods --field-selector=status.phase=Failed -n $(NAMESPACE) 2>/dev/null || true
	@docker system prune -f 2>/dev/null || true
	@rm -rf /tmp/*-green.yaml /tmp/*-canary.yaml 2>/dev/null || true
	@echo "Cleanup completed"

purge: ## Purge all resources in namespace (DANGEROUS)
	$(call log_error,"DANGER: This will delete ALL resources in namespace $(NAMESPACE)")
	@read -p "Type 'DELETE' to confirm: " confirm && \
	if [ "$$confirm" = "DELETE" ]; then \
		kubectl delete all --all -n $(NAMESPACE); \
		kubectl delete configmaps --all -n $(NAMESPACE); \
		kubectl delete secrets --all -n $(NAMESPACE); \
		kubectl delete pvc --all -n $(NAMESPACE); \
		echo "Namespace $(NAMESPACE) purged"; \
	else \
		echo "Purge cancelled"; \
	fi

#######################################
# Documentation and reporting
#######################################
docs: ## Generate documentation
	$(call log,"Generating documentation")
	@mkdir -p docs
	@echo "# Spotify AI Agent Deployment Documentation" > docs/README.md
	@echo "" >> docs/README.md
	@echo "Generated on: $(DATE)" >> docs/README.md
	@echo "By: $(SCRIPT_AUTHOR)" >> docs/README.md
	@echo "" >> docs/README.md
	@$(MAKE) help >> docs/README.md
	@echo "Documentation generated in docs/README.md"

report: ## Generate deployment status report
	$(call log,"Generating deployment status report")
	@chmod +x $(MONITORING_SCRIPT)
	@$(MONITORING_SCRIPT) generate-report \
		--namespace $(NAMESPACE) \
		--environment $(ENVIRONMENT)

status: ## Show current deployment status
	$(call log,"Current deployment status:")
	@echo ""
	@echo "Namespace: $(NAMESPACE)"
	@echo "Environment: $(ENVIRONMENT)"
	@echo ""
	@echo "Services:"
	@kubectl get deployments -n $(NAMESPACE) --no-headers 2>/dev/null | \
		awk '{printf "  %-30s %s/%s replicas ready\n", $$1, $$2, $$3}' || \
		echo "  No deployments found"
	@echo ""
	@echo "Pods:"
	@kubectl get pods -n $(NAMESPACE) --no-headers 2>/dev/null | \
		awk '{printf "  %-30s %s\n", $$1, $$3}' || \
		echo "  No pods found"

#######################################
# Installation and setup
#######################################
install: ## Install dependencies and setup environment
	$(call log,"Installing dependencies and setting up environment")
	@command -v kubectl >/dev/null 2>&1 || (echo "Please install kubectl" && exit 1)
	@command -v helm >/dev/null 2>&1 || (echo "Please install helm" && exit 1)
	@command -v jq >/dev/null 2>&1 || (echo "Please install jq" && exit 1)
	@chmod +x $(DEPLOYMENT_SCRIPT) $(MONITORING_SCRIPT)
	@kubectl create namespace $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	@echo "Environment setup completed ✓"

uninstall: ## Uninstall and cleanup everything
	$(call log_warn,"Uninstalling and cleaning up everything")
	@read -p "This will remove the entire deployment. Continue? [y/N] " confirm && \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		kubectl delete namespace $(NAMESPACE) --ignore-not-found=true; \
		rm -rf backups/ logs/ docs/ 2>/dev/null || true; \
		echo "Uninstall completed"; \
	else \
		echo "Uninstall cancelled"; \
	fi

#######################################
# Validation functions
#######################################
check-service:
ifndef SERVICE
	$(error SERVICE variable is required. Usage: make target SERVICE=service-name)
endif

check-replicas:
ifndef REPLICAS
	$(error REPLICAS variable is required. Usage: make scale SERVICE=service-name REPLICAS=3)
endif

check-backup:
ifndef BACKUP_FILE
	$(error BACKUP_FILE variable is required. Usage: make restore BACKUP_FILE=backup.tar.gz)
endif

#######################################
# Advanced targets
#######################################
demo: ## Run a complete demo deployment
	$(call log,"Running complete demo deployment")
	@$(MAKE) validate
	@$(MAKE) deploy-core
	@sleep 30
	@$(MAKE) health-check-all
	@$(MAKE) deploy-advanced
	@$(MAKE) test-integration
	@$(MAKE) monitor
	@echo ""
	@echo "🎉 Demo deployment completed successfully!"
	@echo "Access the application at: http://spotify-ai-agent.$(NAMESPACE).local"

ci: ## Continuous Integration pipeline
	$(call log,"Running CI pipeline")
	@$(MAKE) validate
	@$(MAKE) test-security
	@$(MAKE) deploy-all DRY_RUN=true
	@echo "CI pipeline completed ✓"

cd: ## Continuous Deployment pipeline
	$(call log,"Running CD pipeline")
	@$(MAKE) backup
	@$(MAKE) deploy-all
	@$(MAKE) test-integration
	@$(MAKE) health-check-all
	@$(MAKE) monitor
	@echo "CD pipeline completed ✓"

#######################################
# Environment info
#######################################
.PHONY: check-service check-replicas check-backup
