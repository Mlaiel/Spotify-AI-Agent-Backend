#!/usr/bin/env python3
"""
Enterprise Template Migration Script
Advanced migration system with version control, rollback, and data transformation

Lead Developer & AI Architect: Fahed Mlaiel
"""

import asyncio
import json
import logging
import os
import sys
import shutil
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum
import yaml
import argparse
import uuid
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('migration.log')
    ]
)
logger = logging.getLogger(__name__)

class MigrationStatus(str, Enum):
    """Migration status enumeration"""
    PENDING = "pending"
    ANALYZING = "analyzing"
    MIGRATING = "migrating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

class MigrationStrategy(str, Enum):
    """Migration strategy enumeration"""
    IN_PLACE = "in_place"
    COPY_AND_MIGRATE = "copy_and_migrate"
    PARALLEL = "parallel"
    STAGED = "staged"

@dataclass
class MigrationRule:
    """Migration rule definition"""
    rule_id: str
    name: str
    description: str
    source_version: str
    target_version: str
    transformation_function: str
    rollback_function: Optional[str] = None
    validation_function: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    priority: int = 0

@dataclass
class MigrationPlan:
    """Migration execution plan"""
    migration_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_version: str = ""
    target_version: str = ""
    strategy: MigrationStrategy = MigrationStrategy.COPY_AND_MIGRATE
    templates: List[str] = field(default_factory=list)
    
    # Migration rules and steps
    migration_rules: List[MigrationRule] = field(default_factory=list)
    migration_steps: List[Dict[str, Any]] = field(default_factory=list)
    
    # Configuration
    backup_enabled: bool = True
    validation_enabled: bool = True
    rollback_on_failure: bool = True
    parallel_execution: bool = False
    
    # Timing
    estimated_duration_minutes: int = 0
    timeout_minutes: int = 60

@dataclass
class MigrationResult:
    """Migration execution result"""
    migration_id: str
    status: MigrationStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Results tracking
    templates_migrated: List[str] = field(default_factory=list)
    templates_failed: List[str] = field(default_factory=list)
    transformation_results: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    
    # Error information
    error_message: Optional[str] = None
    rollback_performed: bool = False
    
    # Metrics
    migration_duration_seconds: float = 0.0
    backup_size_bytes: int = 0
    templates_processed: int = 0

class TemplateMigrator:
    """Advanced template migration system"""
    
    def __init__(self, migration_plan: MigrationPlan):
        self.plan = migration_plan
        self.current_migration: Optional[MigrationResult] = None
        
        # Initialize paths
        self.base_path = Path(__file__).parent.parent
        self.templates_path = self.base_path / "templates"
        self.config_path = self.base_path / "config"
        self.backup_path = self.base_path / "migrations" / "backups"
        self.logs_path = self.base_path / "migrations" / "logs"
        
        # Create directories
        self.backup_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)
        
        # Load migration rules
        self.migration_rules = self._load_migration_rules()
        self.transformation_functions = self._load_transformation_functions()
        
        logger.info("Template migrator initialized", 
                   migration_id=self.plan.migration_id,
                   source_version=self.plan.source_version,
                   target_version=self.plan.target_version)
    
    def _load_migration_rules(self) -> List[MigrationRule]:
        """Load migration rules from configuration"""
        # Mock migration rules
        # In real implementation, these would be loaded from configuration files
        return [
            MigrationRule(
                rule_id="v1_0_to_v1_1",
                name="Version 1.0 to 1.1 Migration",
                description="Add new fields and update structure for v1.1",
                source_version="1.0.0",
                target_version="1.1.0",
                transformation_function="transform_v1_0_to_v1_1"
            ),
            MigrationRule(
                rule_id="v1_1_to_v1_2",
                name="Version 1.1 to 1.2 Migration",
                description="Update security settings and add compliance fields",
                source_version="1.1.0",
                target_version="1.2.0",
                transformation_function="transform_v1_1_to_v1_2"
            ),
            MigrationRule(
                rule_id="v1_2_to_v1_3",
                name="Version 1.2 to 1.3 Migration",
                description="Refactor configuration structure and add new features",
                source_version="1.2.0",
                target_version="1.3.0",
                transformation_function="transform_v1_2_to_v1_3"
            )
        ]
    
    def _load_transformation_functions(self) -> Dict[str, Callable]:
        """Load transformation functions"""
        return {
            "transform_v1_0_to_v1_1": self._transform_v1_0_to_v1_1,
            "transform_v1_1_to_v1_2": self._transform_v1_1_to_v1_2,
            "transform_v1_2_to_v1_3": self._transform_v1_2_to_v1_3,
            "add_security_fields": self._add_security_fields,
            "update_feature_flags": self._update_feature_flags,
            "restructure_limits": self._restructure_limits,
            "add_compliance_metadata": self._add_compliance_metadata
        }
    
    async def migrate_templates(self) -> MigrationResult:
        """Execute template migration"""
        logger.info("Starting template migration", 
                   migration_id=self.plan.migration_id)
        
        # Initialize migration result
        self.current_migration = MigrationResult(
            migration_id=self.plan.migration_id,
            status=MigrationStatus.PENDING,
            started_at=datetime.now(timezone.utc)
        )
        
        try:
            # Analyze migration requirements
            await self._analyze_migration()
            
            # Create backup if enabled
            if self.plan.backup_enabled:
                await self._create_migration_backup()
            
            # Execute migration strategy
            await self._execute_migration_strategy()
            
            # Validate migration results
            if self.plan.validation_enabled:
                await self._validate_migration()
            
            # Mark as completed
            self.current_migration.status = MigrationStatus.COMPLETED
            self.current_migration.completed_at = datetime.now(timezone.utc)
            
            # Calculate duration
            duration = (self.current_migration.completed_at - self.current_migration.started_at)
            self.current_migration.migration_duration_seconds = duration.total_seconds()
            
            logger.info("Template migration completed successfully",
                       migration_id=self.plan.migration_id,
                       duration_seconds=self.current_migration.migration_duration_seconds)
            
        except Exception as e:
            logger.error("Template migration failed", 
                        migration_id=self.plan.migration_id,
                        error=str(e))
            
            self.current_migration.status = MigrationStatus.FAILED
            self.current_migration.error_message = str(e)
            self.current_migration.completed_at = datetime.now(timezone.utc)
            
            # Rollback if enabled
            if self.plan.rollback_on_failure:
                try:
                    await self._rollback_migration()
                except Exception as rollback_error:
                    logger.error("Rollback failed", 
                               migration_id=self.plan.migration_id,
                               error=str(rollback_error))
            
            raise
        
        return self.current_migration
    
    async def _analyze_migration(self):
        """Analyze migration requirements and create execution plan"""
        logger.info("Analyzing migration requirements", 
                   migration_id=self.plan.migration_id)
        
        self.current_migration.status = MigrationStatus.ANALYZING
        
        # Determine migration path
        migration_path = self._find_migration_path(
            self.plan.source_version, 
            self.plan.target_version
        )
        
        if not migration_path:
            raise ValueError(f"No migration path found from {self.plan.source_version} to {self.plan.target_version}")
        
        # Build migration steps
        self.plan.migration_steps = []
        for rule in migration_path:
            step = {
                "rule_id": rule.rule_id,
                "name": rule.name,
                "source_version": rule.source_version,
                "target_version": rule.target_version,
                "transformation_function": rule.transformation_function,
                "templates": self.plan.templates.copy()
            }
            self.plan.migration_steps.append(step)
        
        # Estimate duration
        self.plan.estimated_duration_minutes = len(migration_path) * len(self.plan.templates) * 2
        
        logger.info("Migration analysis completed", 
                   migration_path_length=len(migration_path),
                   estimated_duration_minutes=self.plan.estimated_duration_minutes)
    
    def _find_migration_path(self, source_version: str, target_version: str) -> List[MigrationRule]:
        """Find migration path between versions"""
        # Simple linear path finding
        # In real implementation, this would use graph algorithms for complex version trees
        
        path = []
        current_version = source_version
        
        while current_version != target_version:
            found_rule = None
            
            for rule in self.migration_rules:
                if rule.source_version == current_version:
                    # Check if this rule moves us closer to target
                    if self._is_version_closer(rule.target_version, target_version, current_version):
                        found_rule = rule
                        break
            
            if not found_rule:
                break
            
            path.append(found_rule)
            current_version = found_rule.target_version
        
        return path if current_version == target_version else []
    
    def _is_version_closer(self, candidate: str, target: str, current: str) -> bool:
        """Check if candidate version is closer to target than current"""
        # Simple version comparison
        # In real implementation, this would use proper semantic versioning
        try:
            candidate_parts = [int(x) for x in candidate.split('.')]
            target_parts = [int(x) for x in target.split('.')]
            current_parts = [int(x) for x in current.split('.')]
            
            # Calculate distance
            candidate_distance = sum(abs(c - t) for c, t in zip(candidate_parts, target_parts))
            current_distance = sum(abs(c - t) for c, t in zip(current_parts, target_parts))
            
            return candidate_distance < current_distance
        except (ValueError, IndexError):
            return False
    
    async def _create_migration_backup(self):
        """Create backup before migration"""
        logger.info("Creating migration backup", 
                   migration_id=self.plan.migration_id)
        
        backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.backup_path / f"migration_{backup_timestamp}_{self.plan.migration_id}"
        backup_dir.mkdir(exist_ok=True)
        
        total_size = 0
        
        try:
            # Backup template registry
            registry_path = self.config_path / "template_registry.json"
            if registry_path.exists():
                backup_registry_path = backup_dir / "template_registry.json"
                shutil.copy2(registry_path, backup_registry_path)
                total_size += registry_path.stat().st_size
            
            # Backup templates
            for template_id in self.plan.templates:
                template_path = self._get_template_path(template_id)
                if template_path.exists():
                    backup_template_path = backup_dir / template_path.name
                    shutil.copy2(template_path, backup_template_path)
                    total_size += template_path.stat().st_size
            
            # Create backup manifest
            manifest = {
                "backup_id": str(uuid.uuid4()),
                "migration_id": self.plan.migration_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_version": self.plan.source_version,
                "target_version": self.plan.target_version,
                "templates": self.plan.templates,
                "backup_size_bytes": total_size
            }
            
            manifest_path = backup_dir / "backup_manifest.json"
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)
            
            self.current_migration.backup_size_bytes = total_size
            
            logger.info("Migration backup created successfully", 
                       backup_dir=str(backup_dir),
                       size_bytes=total_size)
                       
        except Exception as e:
            logger.error("Migration backup failed", error=str(e))
            raise
    
    async def _execute_migration_strategy(self):
        """Execute migration according to strategy"""
        logger.info("Executing migration strategy", 
                   strategy=self.plan.strategy.value)
        
        self.current_migration.status = MigrationStatus.MIGRATING
        
        if self.plan.strategy == MigrationStrategy.IN_PLACE:
            await self._execute_in_place_migration()
        elif self.plan.strategy == MigrationStrategy.COPY_AND_MIGRATE:
            await self._execute_copy_and_migrate()
        elif self.plan.strategy == MigrationStrategy.PARALLEL:
            await self._execute_parallel_migration()
        elif self.plan.strategy == MigrationStrategy.STAGED:
            await self._execute_staged_migration()
        else:
            raise ValueError(f"Unsupported migration strategy: {self.plan.strategy}")
    
    async def _execute_in_place_migration(self):
        """Execute in-place migration"""
        for step in self.plan.migration_steps:
            logger.info("Executing migration step", step_name=step["name"])
            
            for template_id in step["templates"]:
                try:
                    await self._migrate_single_template(template_id, step)
                    self.current_migration.templates_migrated.append(template_id)
                except Exception as e:
                    logger.error("Template migration failed", 
                               template_id=template_id, 
                               error=str(e))
                    self.current_migration.templates_failed.append(template_id)
                    raise
    
    async def _execute_copy_and_migrate(self):
        """Execute copy-and-migrate strategy"""
        # Create working directory
        working_dir = self.base_path / "migrations" / "working" / self.plan.migration_id
        working_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Copy templates to working directory
            for template_id in self.plan.templates:
                template_path = self._get_template_path(template_id)
                if template_path.exists():
                    working_template_path = working_dir / template_path.name
                    shutil.copy2(template_path, working_template_path)
            
            # Migrate in working directory
            for step in self.plan.migration_steps:
                logger.info("Executing migration step", step_name=step["name"])
                
                for template_id in step["templates"]:
                    try:
                        await self._migrate_single_template_in_directory(
                            template_id, step, working_dir
                        )
                        self.current_migration.templates_migrated.append(template_id)
                    except Exception as e:
                        logger.error("Template migration failed", 
                                   template_id=template_id, 
                                   error=str(e))
                        self.current_migration.templates_failed.append(template_id)
                        raise
            
            # Copy migrated templates back
            for template_id in self.plan.templates:
                working_template_path = working_dir / f"{template_id}.json"
                if working_template_path.exists():
                    template_path = self._get_template_path(template_id)
                    shutil.copy2(working_template_path, template_path)
            
        finally:
            # Cleanup working directory
            shutil.rmtree(working_dir, ignore_errors=True)
    
    async def _execute_parallel_migration(self):
        """Execute parallel migration"""
        tasks = []
        
        for step in self.plan.migration_steps:
            logger.info("Executing migration step in parallel", step_name=step["name"])
            
            step_tasks = []
            for template_id in step["templates"]:
                task = asyncio.create_task(
                    self._migrate_single_template(template_id, step)
                )
                step_tasks.append((template_id, task))
            
            # Wait for step completion
            for template_id, task in step_tasks:
                try:
                    await task
                    self.current_migration.templates_migrated.append(template_id)
                except Exception as e:
                    logger.error("Template migration failed", 
                               template_id=template_id, 
                               error=str(e))
                    self.current_migration.templates_failed.append(template_id)
                    raise
    
    async def _execute_staged_migration(self):
        """Execute staged migration"""
        stage_size = max(1, len(self.plan.templates) // 3)  # 3 stages
        
        for step in self.plan.migration_steps:
            logger.info("Executing staged migration step", step_name=step["name"])
            
            templates = step["templates"]
            for i in range(0, len(templates), stage_size):
                stage_templates = templates[i:i + stage_size]
                stage_number = i // stage_size + 1
                
                logger.info("Executing migration stage", 
                           stage=stage_number, 
                           templates=len(stage_templates))
                
                for template_id in stage_templates:
                    try:
                        await self._migrate_single_template(template_id, step)
                        self.current_migration.templates_migrated.append(template_id)
                    except Exception as e:
                        logger.error("Template migration failed", 
                                   template_id=template_id, 
                                   error=str(e))
                        self.current_migration.templates_failed.append(template_id)
                        raise
                
                # Validate stage before proceeding
                if self.plan.validation_enabled:
                    await self._validate_migration_stage(stage_templates)
    
    async def _migrate_single_template(self, template_id: str, step: Dict[str, Any]):
        """Migrate a single template"""
        logger.debug("Migrating template", template_id=template_id, step=step["name"])
        
        # Load template
        template_path = self._get_template_path(template_id)
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = json.load(f)
        
        # Apply transformation
        transformation_function_name = step["transformation_function"]
        if transformation_function_name in self.transformation_functions:
            transformation_function = self.transformation_functions[transformation_function_name]
            
            original_content = template_content.copy()
            transformed_content = await transformation_function(template_content, step)
            
            # Record transformation result
            self.current_migration.transformation_results[template_id] = {
                "step": step["name"],
                "function": transformation_function_name,
                "changes_applied": self._calculate_changes(original_content, transformed_content),
                "transformed_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Save transformed template
            with open(template_path, 'w', encoding='utf-8') as f:
                json.dump(transformed_content, f, indent=2)
            
            logger.debug("Template migrated successfully", template_id=template_id)
            
        else:
            raise ValueError(f"Transformation function not found: {transformation_function_name}")
    
    async def _migrate_single_template_in_directory(self, template_id: str, step: Dict[str, Any], 
                                                   working_dir: Path):
        """Migrate a single template in working directory"""
        # Load template from working directory
        template_path = working_dir / f"{template_id}.json"
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found in working directory: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = json.load(f)
        
        # Apply transformation
        transformation_function_name = step["transformation_function"]
        if transformation_function_name in self.transformation_functions:
            transformation_function = self.transformation_functions[transformation_function_name]
            
            original_content = template_content.copy()
            transformed_content = await transformation_function(template_content, step)
            
            # Record transformation result
            self.current_migration.transformation_results[template_id] = {
                "step": step["name"],
                "function": transformation_function_name,
                "changes_applied": self._calculate_changes(original_content, transformed_content),
                "transformed_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Save transformed template
            with open(template_path, 'w', encoding='utf-8') as f:
                json.dump(transformed_content, f, indent=2)
            
        else:
            raise ValueError(f"Transformation function not found: {transformation_function_name}")
    
    def _calculate_changes(self, original: Dict[str, Any], transformed: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate changes between original and transformed content"""
        changes = {
            "fields_added": [],
            "fields_removed": [],
            "fields_modified": [],
            "structure_changes": []
        }
        
        # Simple change detection
        original_keys = set(self._flatten_dict(original).keys())
        transformed_keys = set(self._flatten_dict(transformed).keys())
        
        changes["fields_added"] = list(transformed_keys - original_keys)
        changes["fields_removed"] = list(original_keys - transformed_keys)
        
        # Check for modified values
        common_keys = original_keys & transformed_keys
        original_flat = self._flatten_dict(original)
        transformed_flat = self._flatten_dict(transformed)
        
        for key in common_keys:
            if original_flat[key] != transformed_flat[key]:
                changes["fields_modified"].append(key)
        
        return changes
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    # Transformation Functions
    
    async def _transform_v1_0_to_v1_1(self, template_content: Dict[str, Any], 
                                     step: Dict[str, Any]) -> Dict[str, Any]:
        """Transform template from version 1.0 to 1.1"""
        logger.debug("Applying v1.0 to v1.1 transformation")
        
        # Add version field
        template_content["version"] = "1.1.0"
        
        # Add metadata section
        if "metadata" not in template_content:
            template_content["metadata"] = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "migration_applied": "v1_0_to_v1_1"
            }
        
        # Add feature flags section
        if "feature_flags" not in template_content:
            template_content["feature_flags"] = {
                "new_analytics": True,
                "enhanced_security": False
            }
        
        return template_content
    
    async def _transform_v1_1_to_v1_2(self, template_content: Dict[str, Any], 
                                     step: Dict[str, Any]) -> Dict[str, Any]:
        """Transform template from version 1.1 to 1.2"""
        logger.debug("Applying v1.1 to v1.2 transformation")
        
        # Update version
        template_content["version"] = "1.2.0"
        
        # Add security section
        if "security" not in template_content:
            template_content["security"] = {
                "encryption_required": True,
                "audit_enabled": True,
                "compliance_frameworks": ["GDPR"]
            }
        
        # Update metadata
        if "metadata" in template_content:
            template_content["metadata"]["updated_at"] = datetime.now(timezone.utc).isoformat()
            template_content["metadata"]["migration_applied"] = "v1_1_to_v1_2"
        
        # Restructure limits if present
        if "limits" in template_content:
            await self._restructure_limits(template_content, step)
        
        return template_content
    
    async def _transform_v1_2_to_v1_3(self, template_content: Dict[str, Any], 
                                     step: Dict[str, Any]) -> Dict[str, Any]:
        """Transform template from version 1.2 to 1.3"""
        logger.debug("Applying v1.2 to v1.3 transformation")
        
        # Update version
        template_content["version"] = "1.3.0"
        
        # Add compliance metadata
        await self._add_compliance_metadata(template_content, step)
        
        # Update feature flags
        await self._update_feature_flags(template_content, step)
        
        # Update metadata
        if "metadata" in template_content:
            template_content["metadata"]["updated_at"] = datetime.now(timezone.utc).isoformat()
            template_content["metadata"]["migration_applied"] = "v1_2_to_v1_3"
        
        return template_content
    
    async def _add_security_fields(self, template_content: Dict[str, Any], 
                                  step: Dict[str, Any]) -> Dict[str, Any]:
        """Add security fields to template"""
        if "security" not in template_content:
            template_content["security"] = {}
        
        security = template_content["security"]
        
        # Add default security settings
        security.setdefault("encryption_required", False)
        security.setdefault("audit_enabled", False)
        security.setdefault("access_control", "rbac")
        security.setdefault("compliance_frameworks", [])
        
        return template_content
    
    async def _update_feature_flags(self, template_content: Dict[str, Any], 
                                   step: Dict[str, Any]) -> Dict[str, Any]:
        """Update feature flags in template"""
        if "feature_flags" not in template_content:
            template_content["feature_flags"] = {}
        
        feature_flags = template_content["feature_flags"]
        
        # Add new feature flags for v1.3
        feature_flags["advanced_monitoring"] = True
        feature_flags["real_time_sync"] = False
        feature_flags["multi_region_support"] = False
        
        return template_content
    
    async def _restructure_limits(self, template_content: Dict[str, Any], 
                                 step: Dict[str, Any]) -> Dict[str, Any]:
        """Restructure limits section"""
        if "limits" in template_content:
            limits = template_content["limits"]
            
            # Restructure limits to new format
            if isinstance(limits, dict) and "max_users" in limits:
                new_limits = {
                    "user_limits": {
                        "max_users": limits.get("max_users", 10),
                        "max_active_sessions": limits.get("max_active_sessions", 5)
                    },
                    "resource_limits": {
                        "storage_gb": limits.get("storage_gb", 1),
                        "bandwidth_mbps": limits.get("bandwidth_mbps", 10)
                    },
                    "api_limits": {
                        "requests_per_hour": limits.get("requests_per_hour", 1000),
                        "requests_per_day": limits.get("requests_per_day", 10000)
                    }
                }
                template_content["limits"] = new_limits
        
        return template_content
    
    async def _add_compliance_metadata(self, template_content: Dict[str, Any], 
                                      step: Dict[str, Any]) -> Dict[str, Any]:
        """Add compliance metadata"""
        if "compliance" not in template_content:
            template_content["compliance"] = {}
        
        compliance = template_content["compliance"]
        
        # Add compliance fields
        compliance.setdefault("data_classification", "internal")
        compliance.setdefault("retention_policy", "standard")
        compliance.setdefault("geographic_restrictions", [])
        compliance.setdefault("regulatory_requirements", [])
        
        return template_content
    
    async def _validate_migration(self):
        """Validate migration results"""
        logger.info("Validating migration results", 
                   migration_id=self.plan.migration_id)
        
        for template_id in self.current_migration.templates_migrated:
            try:
                validation_result = await self._validate_migrated_template(template_id)
                self.current_migration.validation_results[template_id] = validation_result
                
                if not validation_result["valid"]:
                    raise ValueError(f"Validation failed for template {template_id}: {validation_result['errors']}")
                    
            except Exception as e:
                logger.error("Template validation failed", 
                           template_id=template_id, 
                           error=str(e))
                raise
        
        logger.info("Migration validation completed successfully")
    
    async def _validate_migrated_template(self, template_id: str) -> Dict[str, Any]:
        """Validate a migrated template"""
        template_path = self._get_template_path(template_id)
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = json.load(f)
            
            # Basic validation
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
            # Check version
            if "version" not in template_content:
                validation_result["errors"].append("Missing version field")
                validation_result["valid"] = False
            elif template_content["version"] != self.plan.target_version:
                validation_result["errors"].append(f"Version mismatch: expected {self.plan.target_version}, got {template_content.get('version')}")
                validation_result["valid"] = False
            
            # Check required fields for target version
            required_fields = self._get_required_fields_for_version(self.plan.target_version)
            for field in required_fields:
                if field not in template_content:
                    validation_result["warnings"].append(f"Missing recommended field: {field}")
            
            return validation_result
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": []
            }
    
    def _get_required_fields_for_version(self, version: str) -> List[str]:
        """Get required fields for a specific version"""
        version_requirements = {
            "1.1.0": ["version", "metadata", "feature_flags"],
            "1.2.0": ["version", "metadata", "feature_flags", "security"],
            "1.3.0": ["version", "metadata", "feature_flags", "security", "compliance"]
        }
        
        return version_requirements.get(version, [])
    
    async def _validate_migration_stage(self, templates: List[str]):
        """Validate migration stage"""
        logger.info("Validating migration stage", templates=len(templates))
        
        for template_id in templates:
            validation_result = await self._validate_migrated_template(template_id)
            if not validation_result["valid"]:
                raise ValueError(f"Stage validation failed for template {template_id}")
    
    async def _rollback_migration(self):
        """Rollback migration"""
        logger.info("Rolling back migration", 
                   migration_id=self.plan.migration_id)
        
        self.current_migration.status = MigrationStatus.ROLLED_BACK
        self.current_migration.rollback_performed = True
        
        # Find most recent backup
        backup_dir = self._find_migration_backup()
        
        if backup_dir:
            await self._restore_from_migration_backup(backup_dir)
        else:
            logger.warning("No backup found for rollback")
        
        logger.info("Migration rollback completed")
    
    def _find_migration_backup(self) -> Optional[Path]:
        """Find migration backup directory"""
        # Look for backup with matching migration ID
        for backup_dir in self.backup_path.iterdir():
            if backup_dir.is_dir() and self.plan.migration_id in backup_dir.name:
                manifest_path = backup_dir / "backup_manifest.json"
                if manifest_path.exists():
                    return backup_dir
        
        return None
    
    async def _restore_from_migration_backup(self, backup_dir: Path):
        """Restore from migration backup"""
        logger.info("Restoring from backup", backup_dir=str(backup_dir))
        
        # Load backup manifest
        manifest_path = backup_dir / "backup_manifest.json"
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        # Restore template registry
        backup_registry_path = backup_dir / "template_registry.json"
        if backup_registry_path.exists():
            registry_path = self.config_path / "template_registry.json"
            shutil.copy2(backup_registry_path, registry_path)
        
        # Restore templates
        for template_id in manifest["templates"]:
            backup_template_path = backup_dir / f"{template_id}.json"
            if backup_template_path.exists():
                template_path = self._get_template_path(template_id)
                shutil.copy2(backup_template_path, template_path)
        
        logger.info("Backup restoration completed")
    
    def _get_template_path(self, template_id: str) -> Path:
        """Get path to template file"""
        # Load registry to find template path
        registry_path = self.config_path / "template_registry.json"
        
        if registry_path.exists():
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry = json.load(f)
            
            template_info = registry.get("templates", {}).get(template_id)
            if template_info and "path" in template_info:
                return self.base_path / template_info["path"]
        
        # Fallback to default path pattern
        return self.templates_path / f"{template_id}.json"

async def main():
    """Main migration function"""
    parser = argparse.ArgumentParser(description="Migrate templates between versions")
    parser.add_argument("--templates", nargs="+", 
                       help="Template IDs to migrate (default: all)")
    parser.add_argument("--source-version", required=True, 
                       help="Source version")
    parser.add_argument("--target-version", required=True, 
                       help="Target version")
    parser.add_argument("--strategy", choices=["in_place", "copy_and_migrate", "parallel", "staged"],
                       default="copy_and_migrate", help="Migration strategy")
    parser.add_argument("--no-backup", action="store_true", 
                       help="Skip backup creation")
    parser.add_argument("--no-validation", action="store_true", 
                       help="Skip validation")
    parser.add_argument("--no-rollback", action="store_true", 
                       help="Disable automatic rollback on failure")
    parser.add_argument("--timeout", type=int, default=60, 
                       help="Migration timeout in minutes")
    
    args = parser.parse_args()
    
    # Get templates to migrate
    if args.templates:
        template_ids = args.templates
    else:
        # Get all templates from registry
        config_path = Path(__file__).parent.parent / "config"
        registry_path = config_path / "template_registry.json"
        
        if registry_path.exists():
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry = json.load(f)
            template_ids = list(registry.get("templates", {}).keys())
        else:
            print("❌ Template registry not found")
            sys.exit(1)
    
    # Create migration plan
    plan = MigrationPlan(
        source_version=args.source_version,
        target_version=args.target_version,
        strategy=MigrationStrategy(args.strategy),
        templates=template_ids,
        backup_enabled=not args.no_backup,
        validation_enabled=not args.no_validation,
        rollback_on_failure=not args.no_rollback,
        timeout_minutes=args.timeout
    )
    
    # Create migrator and run migration
    migrator = TemplateMigrator(plan)
    
    try:
        print(f"🔄 Migrating {len(template_ids)} templates from {args.source_version} to {args.target_version}...")
        result = await migrator.migrate_templates()
        
        # Print migration summary
        print("\n" + "="*60)
        print("MIGRATION SUMMARY")
        print("="*60)
        print(f"Migration ID: {result.migration_id}")
        print(f"Status: {result.status.value}")
        print(f"Source Version: {plan.source_version}")
        print(f"Target Version: {plan.target_version}")
        print(f"Strategy: {plan.strategy.value}")
        print(f"Started: {result.started_at}")
        print(f"Completed: {result.completed_at}")
        if result.completed_at:
            duration = result.migration_duration_seconds
            print(f"Duration: {duration:.2f} seconds")
        print(f"Templates Migrated: {len(result.templates_migrated)}")
        print(f"Templates Failed: {len(result.templates_failed)}")
        
        if result.templates_migrated:
            print("\nSuccessfully Migrated:")
            for template_id in result.templates_migrated:
                print(f"  ✓ {template_id}")
        
        if result.templates_failed:
            print("\nFailed Templates:")
            for template_id in result.templates_failed:
                print(f"  ✗ {template_id}")
        
        if result.rollback_performed:
            print("\n⚠️  Rollback was performed due to migration failure")
        
        if result.error_message:
            print(f"\nError: {result.error_message}")
        
        print("="*60)
        
        # Exit with appropriate code
        sys.exit(0 if result.status == MigrationStatus.COMPLETED else 1)
        
    except Exception as e:
        logger.error("Migration failed with exception", error=str(e))
        print(f"\n❌ Migration failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
