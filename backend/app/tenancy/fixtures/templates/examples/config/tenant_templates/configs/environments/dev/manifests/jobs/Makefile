# Advanced Kubernetes Jobs Management Makefile for Spotify AI Agent
# Ultra-Advanced Enterprise-Grade Job Orchestration and Automation
#
# This Makefile provides comprehensive job management workflows:
# - Automated job creation and lifecycle management
# - Advanced testing and validation pipelines
# - Performance monitoring and optimization
# - Security scanning and compliance checking
# - Multi-tenant job isolation and resource management
# - CI/CD integration and deployment automation

.PHONY: help install validate test deploy monitor cleanup security performance
.DEFAULT_GOAL := help

# Configuration Variables
NAMESPACE ?= spotify-ai-agent-dev
ENVIRONMENT ?= development
DRY_RUN ?= false
PARALLEL_JOBS ?= 4
LOG_LEVEL ?= INFO

# Job Types
JOB_TYPES := ml-training data-etl tenant-backup security-scan billing-reporting

# Tenant Configuration
TENANT_ID ?= enterprise-client-001
PRIORITY ?= normal
RESOURCE_TIER ?= medium

# Paths
SCRIPT_DIR := $(shell pwd)
JOBS_SCRIPT := $(SCRIPT_DIR)/manage-jobs.sh
MANIFESTS_DIR := $(SCRIPT_DIR)
LOG_DIR := /tmp/spotify-ai-jobs-logs

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[1;37m
NC := \033[0m

##@ General Commands

help: ## Display this help message
	@echo ""
	@echo "$(CYAN)Advanced Kubernetes Jobs Management System$(NC)"
	@echo "$(WHITE)Ultra-Advanced Enterprise-Grade Job Orchestration$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make create-ml-job TENANT_ID=premium-001 PRIORITY=high RESOURCE_TIER=xlarge"
	@echo "  make list-jobs FILTER=running TENANT_ID=enterprise-001"
	@echo "  make monitor-job JOB_NAME=ml-training-premium-001-20250717-143022"
	@echo "  make cleanup-jobs MAX_AGE=48"
	@echo "  make security-scan-all"
	@echo "  make deploy-all-jobs"
	@echo ""

##@ Installation and Setup

install: ## Install dependencies and setup environment
	@echo "$(BLUE)Installing dependencies...$(NC)"
	@command -v kubectl >/dev/null 2>&1 || { echo "$(RED)kubectl is required but not installed$(NC)"; exit 1; }
	@command -v jq >/dev/null 2>&1 || { echo "$(RED)jq is required but not installed$(NC)"; exit 1; }
	@command -v yq >/dev/null 2>&1 || { echo "$(RED)yq is required but not installed$(NC)"; exit 1; }
	@command -v curl >/dev/null 2>&1 || { echo "$(RED)curl is required but not installed$(NC)"; exit 1; }
	@command -v openssl >/dev/null 2>&1 || { echo "$(RED)openssl is required but not installed$(NC)"; exit 1; }
	@mkdir -p $(LOG_DIR)
	@chmod +x $(JOBS_SCRIPT)
	@echo "$(GREEN)Dependencies installed successfully$(NC)"

check-cluster: ## Check Kubernetes cluster connectivity
	@echo "$(BLUE)Checking Kubernetes cluster connectivity...$(NC)"
	@kubectl cluster-info --request-timeout=10s >/dev/null || { echo "$(RED)Cannot connect to Kubernetes cluster$(NC)"; exit 1; }
	@kubectl get namespace $(NAMESPACE) >/dev/null 2>&1 || kubectl create namespace $(NAMESPACE)
	@echo "$(GREEN)Kubernetes cluster connection verified$(NC)"

##@ Job Creation and Management

create-ml-job: check-cluster ## Create ML training job
	@echo "$(BLUE)Creating ML training job...$(NC)"
	@$(JOBS_SCRIPT) create ml-training $(TENANT_ID) $(PRIORITY) $(RESOURCE_TIER)

create-etl-job: check-cluster ## Create data ETL job
	@echo "$(BLUE)Creating data ETL job...$(NC)"
	@$(JOBS_SCRIPT) create data-etl $(TENANT_ID) $(PRIORITY) $(RESOURCE_TIER)

create-backup-job: check-cluster ## Create tenant backup job
	@echo "$(BLUE)Creating tenant backup job...$(NC)"
	@$(JOBS_SCRIPT) create tenant-backup $(TENANT_ID) $(PRIORITY) $(RESOURCE_TIER)

create-security-job: check-cluster ## Create security scan job
	@echo "$(BLUE)Creating security scan job...$(NC)"
	@$(JOBS_SCRIPT) create security-scan $(TENANT_ID) $(PRIORITY) $(RESOURCE_TIER)

create-billing-job: check-cluster ## Create billing reporting job
	@echo "$(BLUE)Creating billing reporting job...$(NC)"
	@$(JOBS_SCRIPT) create billing-reporting $(TENANT_ID) $(PRIORITY) $(RESOURCE_TIER)

create-all-jobs: create-ml-job create-etl-job create-backup-job create-security-job create-billing-job ## Create all job types

##@ Job Monitoring and Status

list-jobs: check-cluster ## List jobs with optional filter (FILTER=all|running|completed|failed)
	@echo "$(BLUE)Listing jobs...$(NC)"
	@$(JOBS_SCRIPT) list $(FILTER) $(TENANT_ID)

status-job: check-cluster ## Get job status (requires JOB_NAME)
	@if [ -z "$(JOB_NAME)" ]; then echo "$(RED)JOB_NAME is required$(NC)"; exit 1; fi
	@echo "$(BLUE)Getting status for job: $(JOB_NAME)$(NC)"
	@$(JOBS_SCRIPT) status $(JOB_NAME)

monitor-job: check-cluster ## Monitor job execution (requires JOB_NAME)
	@if [ -z "$(JOB_NAME)" ]; then echo "$(RED)JOB_NAME is required$(NC)"; exit 1; fi
	@echo "$(BLUE)Monitoring job: $(JOB_NAME)$(NC)"
	@$(JOBS_SCRIPT) monitor $(JOB_NAME)

monitor-all: check-cluster ## Monitor all running jobs
	@echo "$(BLUE)Monitoring all running jobs...$(NC)"
	@for job in $$(kubectl get jobs -n $(NAMESPACE) --field-selector=status.active=1 -o jsonpath='{.items[*].metadata.name}'); do \
		echo "$(CYAN)Monitoring job: $$job$(NC)"; \
		$(JOBS_SCRIPT) monitor $$job & \
	done
	@wait

logs-job: check-cluster ## Get job logs (requires JOB_NAME)
	@if [ -z "$(JOB_NAME)" ]; then echo "$(RED)JOB_NAME is required$(NC)"; exit 1; fi
	@echo "$(BLUE)Getting logs for job: $(JOB_NAME)$(NC)"
	@$(JOBS_SCRIPT) logs $(JOB_NAME)

##@ Job Cleanup and Maintenance

delete-job: check-cluster ## Delete specific job (requires JOB_NAME, optional FORCE=true)
	@if [ -z "$(JOB_NAME)" ]; then echo "$(RED)JOB_NAME is required$(NC)"; exit 1; fi
	@echo "$(BLUE)Deleting job: $(JOB_NAME)$(NC)"
	@$(JOBS_SCRIPT) delete $(JOB_NAME) $(FORCE)

cleanup-jobs: check-cluster ## Cleanup completed jobs older than MAX_AGE hours (default: 24)
	@echo "$(BLUE)Cleaning up completed jobs older than $(or $(MAX_AGE),24) hours...$(NC)"
	@$(JOBS_SCRIPT) cleanup $(or $(MAX_AGE),24)

cleanup-failed: check-cluster ## Cleanup all failed jobs
	@echo "$(BLUE)Cleaning up failed jobs...$(NC)"
	@for job in $$(kubectl get jobs -n $(NAMESPACE) --field-selector=status.failed=1 -o jsonpath='{.items[*].metadata.name}'); do \
		echo "$(YELLOW)Deleting failed job: $$job$(NC)"; \
		$(JOBS_SCRIPT) delete $$job true; \
	done

cleanup-all: cleanup-jobs cleanup-failed ## Cleanup all completed and failed jobs

##@ Validation and Testing

validate-manifests: ## Validate all job manifests
	@echo "$(BLUE)Validating job manifests...$(NC)"
	@for manifest in $(MANIFESTS_DIR)/*-job.yaml; do \
		if [ -f "$$manifest" ]; then \
			echo "$(CYAN)Validating: $$manifest$(NC)"; \
			$(JOBS_SCRIPT) validate "$$manifest"; \
		fi; \
	done
	@echo "$(GREEN)All manifests validated successfully$(NC)"

dry-run-all: ## Perform dry run of all job types
	@echo "$(BLUE)Performing dry run of all job types...$(NC)"
	@export DRY_RUN=true; \
	for job_type in $(JOB_TYPES); do \
		echo "$(CYAN)Dry run for: $$job_type$(NC)"; \
		$(JOBS_SCRIPT) create $$job_type $(TENANT_ID) $(PRIORITY) $(RESOURCE_TIER); \
	done
	@echo "$(GREEN)Dry run completed successfully$(NC)"

test-job-creation: check-cluster ## Test job creation workflow
	@echo "$(BLUE)Testing job creation workflow...$(NC)"
	@export DRY_RUN=true; \
	$(JOBS_SCRIPT) create ml-training test-tenant normal small
	@echo "$(GREEN)Job creation test passed$(NC)"

test-job-validation: ## Test job manifest validation
	@echo "$(BLUE)Testing job manifest validation...$(NC)"
	@$(MAKE) validate-manifests
	@echo "$(GREEN)Job validation test passed$(NC)"

test-all: test-job-creation test-job-validation ## Run all tests

##@ Security and Compliance

security-scan-jobs: check-cluster ## Scan all job manifests for security issues
	@echo "$(BLUE)Scanning job manifests for security issues...$(NC)"
	@for manifest in $(MANIFESTS_DIR)/*-job.yaml; do \
		if [ -f "$$manifest" ]; then \
			echo "$(CYAN)Security scanning: $$manifest$(NC)"; \
			# Check for privileged containers \
			if grep -q "privileged.*true" "$$manifest"; then \
				echo "$(RED)WARNING: Privileged container found in $$manifest$(NC)"; \
			fi; \
			# Check for hostNetwork \
			if grep -q "hostNetwork.*true" "$$manifest"; then \
				echo "$(RED)WARNING: hostNetwork enabled in $$manifest$(NC)"; \
			fi; \
			# Check for runAsRoot \
			if grep -q "runAsUser.*0" "$$manifest"; then \
				echo "$(RED)WARNING: Container running as root in $$manifest$(NC)"; \
			fi; \
		fi; \
	done
	@echo "$(GREEN)Security scan completed$(NC)"

compliance-check: check-cluster ## Check compliance requirements
	@echo "$(BLUE)Checking compliance requirements...$(NC)"
	@echo "$(CYAN)Checking resource limits...$(NC)"
	@for manifest in $(MANIFESTS_DIR)/*-job.yaml; do \
		if [ -f "$$manifest" ]; then \
			if ! grep -q "resources:" "$$manifest"; then \
				echo "$(YELLOW)WARNING: No resource limits in $$manifest$(NC)"; \
			fi; \
		fi; \
	done
	@echo "$(CYAN)Checking security contexts...$(NC)"
	@for manifest in $(MANIFESTS_DIR)/*-job.yaml; do \
		if [ -f "$$manifest" ]; then \
			if ! grep -q "securityContext:" "$$manifest"; then \
				echo "$(YELLOW)WARNING: No security context in $$manifest$(NC)"; \
			fi; \
		fi; \
	done
	@echo "$(GREEN)Compliance check completed$(NC)"

security-scan-all: security-scan-jobs compliance-check ## Run complete security and compliance scan

##@ Performance and Optimization

performance-test: check-cluster ## Run performance tests on job execution
	@echo "$(BLUE)Running performance tests...$(NC)"
	@start_time=$$(date +%s); \
	$(MAKE) create-ml-job TENANT_ID=perf-test RESOURCE_TIER=small; \
	job_name=$$(kubectl get jobs -n $(NAMESPACE) -l tenant-id=perf-test --sort-by='.metadata.creationTimestamp' -o jsonpath='{.items[-1].metadata.name}'); \
	if [ -n "$$job_name" ]; then \
		$(JOBS_SCRIPT) monitor $$job_name; \
		end_time=$$(date +%s); \
		duration=$$((end_time - start_time)); \
		echo "$(GREEN)Performance test completed in $$duration seconds$(NC)"; \
		$(JOBS_SCRIPT) delete $$job_name true; \
	fi

resource-optimization: check-cluster ## Analyze and optimize resource usage
	@echo "$(BLUE)Analyzing resource usage...$(NC)"
	@kubectl top nodes 2>/dev/null || echo "$(YELLOW)Metrics server not available$(NC)"
	@kubectl top pods -n $(NAMESPACE) 2>/dev/null || echo "$(YELLOW)Pod metrics not available$(NC)"
	@echo "$(CYAN)Job resource analysis:$(NC)"
	@kubectl get jobs -n $(NAMESPACE) -o custom-columns=NAME:.metadata.name,CPU-REQ:.spec.template.spec.containers[0].resources.requests.cpu,MEM-REQ:.spec.template.spec.containers[0].resources.requests.memory,CPU-LIM:.spec.template.spec.containers[0].resources.limits.cpu,MEM-LIM:.spec.template.spec.containers[0].resources.limits.memory 2>/dev/null || true

monitor-performance: check-cluster ## Monitor system performance during job execution
	@echo "$(BLUE)Monitoring system performance...$(NC)"
	@while true; do \
		clear; \
		echo "$(CYAN)=== System Performance Monitor ===$(NC)"; \
		echo "$(WHITE)Timestamp: $$(date)$(NC)"; \
		echo ""; \
		echo "$(YELLOW)Active Jobs:$(NC)"; \
		kubectl get jobs -n $(NAMESPACE) --field-selector=status.active=1 -o wide 2>/dev/null || true; \
		echo ""; \
		echo "$(YELLOW)Node Resources:$(NC)"; \
		kubectl top nodes 2>/dev/null || echo "Metrics not available"; \
		echo ""; \
		echo "$(YELLOW)Pod Resources:$(NC)"; \
		kubectl top pods -n $(NAMESPACE) 2>/dev/null || echo "Metrics not available"; \
		sleep 30; \
	done

##@ Multi-Tenant Operations

create-tenant-jobs: check-cluster ## Create jobs for specific tenant (requires TENANT_ID)
	@if [ -z "$(TENANT_ID)" ]; then echo "$(RED)TENANT_ID is required$(NC)"; exit 1; fi
	@echo "$(BLUE)Creating jobs for tenant: $(TENANT_ID)$(NC)"
	@for job_type in $(JOB_TYPES); do \
		echo "$(CYAN)Creating $$job_type job for $(TENANT_ID)$(NC)"; \
		$(JOBS_SCRIPT) create $$job_type $(TENANT_ID) $(PRIORITY) $(RESOURCE_TIER); \
		sleep 5; \
	done

list-tenant-jobs: check-cluster ## List jobs for specific tenant (requires TENANT_ID)
	@if [ -z "$(TENANT_ID)" ]; then echo "$(RED)TENANT_ID is required$(NC)"; exit 1; fi
	@echo "$(BLUE)Listing jobs for tenant: $(TENANT_ID)$(NC)"
	@$(JOBS_SCRIPT) list all $(TENANT_ID)

cleanup-tenant-jobs: check-cluster ## Cleanup jobs for specific tenant (requires TENANT_ID)
	@if [ -z "$(TENANT_ID)" ]; then echo "$(RED)TENANT_ID is required$(NC)"; exit 1; fi
	@echo "$(BLUE)Cleaning up jobs for tenant: $(TENANT_ID)$(NC)"
	@for job in $$(kubectl get jobs -n $(NAMESPACE) -l tenant-id=$(TENANT_ID) -o jsonpath='{.items[*].metadata.name}'); do \
		echo "$(YELLOW)Deleting job: $$job$(NC)"; \
		$(JOBS_SCRIPT) delete $$job true; \
	done

##@ CI/CD Integration

deploy-dev: check-cluster ## Deploy jobs in development environment
	@echo "$(BLUE)Deploying jobs in development environment...$(NC)"
	@export ENVIRONMENT=development; \
	$(MAKE) validate-manifests && \
	$(MAKE) dry-run-all && \
	echo "$(GREEN)Development deployment ready$(NC)"

deploy-staging: check-cluster ## Deploy jobs in staging environment
	@echo "$(BLUE)Deploying jobs in staging environment...$(NC)"
	@export ENVIRONMENT=staging; \
	export NAMESPACE=spotify-ai-agent-staging; \
	$(MAKE) validate-manifests && \
	$(MAKE) security-scan-all && \
	$(MAKE) test-all && \
	echo "$(GREEN)Staging deployment ready$(NC)"

deploy-prod: check-cluster ## Deploy jobs in production environment
	@echo "$(BLUE)Deploying jobs in production environment...$(NC)"
	@export ENVIRONMENT=production; \
	export NAMESPACE=spotify-ai-agent-prod; \
	$(MAKE) validate-manifests && \
	$(MAKE) security-scan-all && \
	$(MAKE) compliance-check && \
	$(MAKE) test-all && \
	echo "$(GREEN)Production deployment ready$(NC)"

deploy-all: deploy-dev deploy-staging ## Deploy to all environments (excluding production)

##@ Backup and Recovery

backup-job-configs: ## Backup job configurations
	@echo "$(BLUE)Backing up job configurations...$(NC)"
	@backup_dir="/tmp/job-configs-backup-$$(date +%Y%m%d-%H%M%S)"; \
	mkdir -p "$$backup_dir"; \
	cp $(MANIFESTS_DIR)/*.yaml "$$backup_dir/" 2>/dev/null || true; \
	cp $(JOBS_SCRIPT) "$$backup_dir/"; \
	cp $(MAKEFILE_LIST) "$$backup_dir/"; \
	tar -czf "$$backup_dir.tar.gz" -C "$$(dirname $$backup_dir)" "$$(basename $$backup_dir)"; \
	rm -rf "$$backup_dir"; \
	echo "$(GREEN)Backup created: $$backup_dir.tar.gz$(NC)"

restore-job-configs: ## Restore job configurations from backup (requires BACKUP_FILE)
	@if [ -z "$(BACKUP_FILE)" ]; then echo "$(RED)BACKUP_FILE is required$(NC)"; exit 1; fi
	@if [ ! -f "$(BACKUP_FILE)" ]; then echo "$(RED)Backup file not found: $(BACKUP_FILE)$(NC)"; exit 1; fi
	@echo "$(BLUE)Restoring job configurations from: $(BACKUP_FILE)$(NC)"
	@restore_dir="/tmp/job-configs-restore-$$(date +%Y%m%d-%H%M%S)"; \
	mkdir -p "$$restore_dir"; \
	tar -xzf "$(BACKUP_FILE)" -C "$$restore_dir"; \
	cp "$$restore_dir"/*/*.yaml $(MANIFESTS_DIR)/ 2>/dev/null || true; \
	cp "$$restore_dir"/*/manage-jobs.sh $(SCRIPT_DIR)/ 2>/dev/null || true; \
	chmod +x $(JOBS_SCRIPT); \
	rm -rf "$$restore_dir"; \
	echo "$(GREEN)Job configurations restored successfully$(NC)"

##@ Reporting and Analytics

job-summary-report: check-cluster ## Generate job summary report
	@echo "$(BLUE)Generating job summary report...$(NC)"
	@report_file="/tmp/job-summary-$$(date +%Y%m%d-%H%M%S).txt"; \
	echo "=== Spotify AI Agent Jobs Summary Report ===" > "$$report_file"; \
	echo "Generated: $$(date)" >> "$$report_file"; \
	echo "Namespace: $(NAMESPACE)" >> "$$report_file"; \
	echo "" >> "$$report_file"; \
	echo "=== Job Statistics ===" >> "$$report_file"; \
	echo "Total Jobs: $$(kubectl get jobs -n $(NAMESPACE) --no-headers 2>/dev/null | wc -l)" >> "$$report_file"; \
	echo "Running Jobs: $$(kubectl get jobs -n $(NAMESPACE) --field-selector=status.active=1 --no-headers 2>/dev/null | wc -l)" >> "$$report_file"; \
	echo "Completed Jobs: $$(kubectl get jobs -n $(NAMESPACE) --field-selector=status.succeeded=1 --no-headers 2>/dev/null | wc -l)" >> "$$report_file"; \
	echo "Failed Jobs: $$(kubectl get jobs -n $(NAMESPACE) --field-selector=status.failed=1 --no-headers 2>/dev/null | wc -l)" >> "$$report_file"; \
	echo "" >> "$$report_file"; \
	echo "=== Job Details ===" >> "$$report_file"; \
	kubectl get jobs -n $(NAMESPACE) -o wide >> "$$report_file" 2>/dev/null || true; \
	echo "$(GREEN)Report generated: $$report_file$(NC)"; \
	cat "$$report_file"

performance-report: check-cluster ## Generate performance analysis report
	@echo "$(BLUE)Generating performance analysis report...$(NC)"
	@report_file="/tmp/performance-report-$$(date +%Y%m%d-%H%M%S).txt"; \
	echo "=== Performance Analysis Report ===" > "$$report_file"; \
	echo "Generated: $$(date)" >> "$$report_file"; \
	echo "Namespace: $(NAMESPACE)" >> "$$report_file"; \
	echo "" >> "$$report_file"; \
	echo "=== Node Resources ===" >> "$$report_file"; \
	kubectl top nodes >> "$$report_file" 2>/dev/null || echo "Metrics not available" >> "$$report_file"; \
	echo "" >> "$$report_file"; \
	echo "=== Pod Resources ===" >> "$$report_file"; \
	kubectl top pods -n $(NAMESPACE) >> "$$report_file" 2>/dev/null || echo "Metrics not available" >> "$$report_file"; \
	echo "" >> "$$report_file"; \
	echo "=== Resource Quotas ===" >> "$$report_file"; \
	kubectl describe quota -n $(NAMESPACE) >> "$$report_file" 2>/dev/null || echo "No quotas defined" >> "$$report_file"; \
	echo "$(GREEN)Performance report generated: $$report_file$(NC)"; \
	cat "$$report_file"

##@ Maintenance and Utilities

update-manifests: ## Update all job manifests with latest configurations
	@echo "$(BLUE)Updating job manifests...$(NC)"
	@for manifest in $(MANIFESTS_DIR)/*-job.yaml; do \
		if [ -f "$$manifest" ]; then \
			echo "$(CYAN)Updating: $$manifest$(NC)"; \
			# Update image tags to latest \
			sed -i 's/:v[0-9]\+\.[0-9]\+\.[0-9]\+/:v2.1.0/g' "$$manifest"; \
			# Update namespace \
			sed -i 's/namespace: .*/namespace: $(NAMESPACE)/g' "$$manifest"; \
		fi; \
	done
	@echo "$(GREEN)Manifests updated successfully$(NC)"

check-prerequisites: ## Check all prerequisites and dependencies
	@echo "$(BLUE)Checking prerequisites...$(NC)"
	@$(MAKE) install
	@$(MAKE) check-cluster
	@$(MAKE) validate-manifests
	@echo "$(GREEN)All prerequisites satisfied$(NC)"

clean-logs: ## Clean up log files
	@echo "$(BLUE)Cleaning up log files...$(NC)"
	@rm -rf $(LOG_DIR)/*
	@rm -f /tmp/*job*.log /tmp/*job*.txt /tmp/*job*.json
	@echo "$(GREEN)Log files cleaned up$(NC)"

show-config: ## Show current configuration
	@echo "$(CYAN)=== Current Configuration ===$(NC)"
	@echo "$(WHITE)Namespace:$(NC) $(NAMESPACE)"
	@echo "$(WHITE)Environment:$(NC) $(ENVIRONMENT)"
	@echo "$(WHITE)Tenant ID:$(NC) $(TENANT_ID)"
	@echo "$(WHITE)Priority:$(NC) $(PRIORITY)"
	@echo "$(WHITE)Resource Tier:$(NC) $(RESOURCE_TIER)"
	@echo "$(WHITE)Dry Run:$(NC) $(DRY_RUN)"
	@echo "$(WHITE)Parallel Jobs:$(NC) $(PARALLEL_JOBS)"
	@echo "$(WHITE)Log Level:$(NC) $(LOG_LEVEL)"
	@echo "$(WHITE)Script Dir:$(NC) $(SCRIPT_DIR)"
	@echo "$(WHITE)Log Dir:$(NC) $(LOG_DIR)"

##@ Emergency Operations

emergency-stop: check-cluster ## Emergency stop all running jobs
	@echo "$(RED)EMERGENCY: Stopping all running jobs...$(NC)"
	@for job in $$(kubectl get jobs -n $(NAMESPACE) --field-selector=status.active=1 -o jsonpath='{.items[*].metadata.name}'); do \
		echo "$(RED)Emergency stopping job: $$job$(NC)"; \
		kubectl delete job $$job -n $(NAMESPACE) --grace-period=0 --force; \
	done
	@echo "$(RED)Emergency stop completed$(NC)"

emergency-cleanup: check-cluster ## Emergency cleanup of all jobs and resources
	@echo "$(RED)EMERGENCY: Cleaning up all jobs and resources...$(NC)"
	@kubectl delete jobs --all -n $(NAMESPACE) --grace-period=0 --force
	@kubectl delete pods --all -n $(NAMESPACE) --grace-period=0 --force
	@echo "$(RED)Emergency cleanup completed$(NC)"

health-check: check-cluster ## Perform comprehensive health check
	@echo "$(BLUE)Performing health check...$(NC)"
	@echo "$(CYAN)Cluster Status:$(NC)"
	@kubectl cluster-info
	@echo ""
	@echo "$(CYAN)Namespace Status:$(NC)"
	@kubectl get all -n $(NAMESPACE)
	@echo ""
	@echo "$(CYAN)Resource Usage:$(NC)"
	@kubectl top nodes 2>/dev/null || echo "Metrics not available"
	@kubectl top pods -n $(NAMESPACE) 2>/dev/null || echo "Pod metrics not available"
	@echo "$(GREEN)Health check completed$(NC)"
