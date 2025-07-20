#!/usr/bin/env python3
"""
Enterprise Template Validation Script
Advanced validation system with schema checking, dependency analysis, and security scanning

Lead Developer & AI Architect: Fahed Mlaiel
"""

import asyncio
import json
import logging
import os
import sys
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
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
        logging.FileHandler('validation.log')
    ]
)
logger = logging.getLogger(__name__)

class ValidationLevel(str, Enum):
    """Validation level enumeration"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    ENTERPRISE = "enterprise"

class ValidationResult(str, Enum):
    """Validation result enumeration"""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    ERROR = "error"

@dataclass
class ValidationIssue:
    """Validation issue information"""
    severity: ValidationResult
    category: str
    message: str
    location: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    suggestion: Optional[str] = None
    rule_id: Optional[str] = None

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    template_id: str
    validation_level: ValidationLevel
    overall_result: ValidationResult
    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Issue tracking
    issues: List[ValidationIssue] = field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0
    
    # Detailed results
    syntax_validation: Dict[str, Any] = field(default_factory=dict)
    schema_validation: Dict[str, Any] = field(default_factory=dict)
    dependency_validation: Dict[str, Any] = field(default_factory=dict)
    security_validation: Dict[str, Any] = field(default_factory=dict)
    performance_validation: Dict[str, Any] = field(default_factory=dict)
    compliance_validation: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics
    validation_duration_seconds: float = 0.0
    file_size_bytes: int = 0
    complexity_score: int = 0
    security_score: int = 0
    
    def add_issue(self, issue: ValidationIssue):
        """Add validation issue"""
        self.issues.append(issue)
        if issue.severity == ValidationResult.ERROR:
            self.error_count += 1
        elif issue.severity == ValidationResult.WARNING:
            self.warning_count += 1
        
        # Update overall result
        if issue.severity == ValidationResult.ERROR:
            self.overall_result = ValidationResult.FAILED
        elif issue.severity == ValidationResult.WARNING and self.overall_result == ValidationResult.PASSED:
            self.overall_result = ValidationResult.WARNING

class TemplateValidator:
    """Advanced template validation system"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        
        # Initialize paths
        self.base_path = Path(__file__).parent.parent
        self.templates_path = self.base_path / "templates"
        self.config_path = self.base_path / "config"
        self.schemas_path = self.base_path / "schemas"
        
        # Load validation rules
        self.validation_rules = self._load_validation_rules()
        self.security_rules = self._load_security_rules()
        self.compliance_rules = self._load_compliance_rules()
        
        logger.info("Template validator initialized", 
                   validation_level=validation_level.value)
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules configuration"""
        return {
            "syntax": {
                "check_json_syntax": True,
                "check_yaml_syntax": True,
                "validate_encoding": True,
                "check_file_extensions": True
            },
            "schema": {
                "validate_structure": True,
                "check_required_fields": True,
                "validate_data_types": True,
                "check_field_constraints": True
            },
            "dependencies": {
                "check_circular_dependencies": True,
                "validate_dependency_versions": True,
                "check_missing_dependencies": True,
                "verify_dependency_availability": True
            },
            "security": {
                "scan_for_secrets": True,
                "check_injection_vulnerabilities": True,
                "validate_permissions": True,
                "check_encryption_requirements": True
            },
            "performance": {
                "check_file_size": True,
                "validate_complexity": True,
                "check_resource_usage": True,
                "analyze_performance_impact": True
            },
            "compliance": {
                "check_gdpr_compliance": True,
                "validate_data_retention": True,
                "check_audit_requirements": True,
                "verify_access_controls": True
            }
        }
    
    def _load_security_rules(self) -> Dict[str, Any]:
        """Load security validation rules"""
        return {
            "secrets_patterns": [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']',
                r'[A-Za-z0-9]{32,}',  # Potential API keys
                r'-----BEGIN [A-Z ]+-----',  # Certificates/Keys
            ],
            "injection_patterns": [
                r'eval\s*\(',
                r'exec\s*\(',
                r'system\s*\(',
                r'__import__',
                r'subprocess',
                r'<script[^>]*>',
                r'javascript:',
                r'data:text/html',
                r'SQL\s+(?:INSERT|UPDATE|DELETE|DROP)',
            ],
            "dangerous_functions": [
                "eval", "exec", "compile", "open", "file",
                "input", "raw_input", "__import__", "reload"
            ],
            "suspicious_urls": [
                r'http://[^/]+',  # Non-HTTPS URLs
                r'ftp://[^/]+',   # FTP URLs
                r'file://[^/]+',  # File URLs
            ]
        }
    
    def _load_compliance_rules(self) -> Dict[str, Any]:
        """Load compliance validation rules"""
        return {
            "gdpr": {
                "required_fields": ["data_controller", "lawful_basis"],
                "data_categories": ["personal", "sensitive", "public"],
                "retention_policies": True,
                "consent_mechanisms": True
            },
            "soc2": {
                "security_controls": True,
                "access_controls": True,
                "monitoring": True,
                "incident_response": True
            },
            "iso27001": {
                "risk_assessment": True,
                "security_policies": True,
                "asset_management": True,
                "access_control": True
            }
        }
    
    async def validate_template(self, template_id: str) -> ValidationReport:
        """Validate a single template"""
        logger.info("Validating template", template_id=template_id)
        
        start_time = datetime.now()
        report = ValidationReport(
            template_id=template_id,
            validation_level=self.validation_level,
            overall_result=ValidationResult.PASSED
        )
        
        try:
            # Get template path
            template_path = self._get_template_path(template_id)
            if not template_path.exists():
                report.add_issue(ValidationIssue(
                    severity=ValidationResult.ERROR,
                    category="file",
                    message=f"Template file not found: {template_path}",
                    rule_id="FILE_NOT_FOUND"
                ))
                return report
            
            # Get file info
            report.file_size_bytes = template_path.stat().st_size
            
            # Load template content
            template_content = self._load_template_content(template_path)
            
            # Perform validations based on level
            await self._validate_syntax(template_path, template_content, report)
            
            if self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.ENTERPRISE]:
                await self._validate_schema(template_id, template_content, report)
                await self._validate_dependencies(template_id, template_content, report)
            
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.ENTERPRISE]:
                await self._validate_security(template_content, report)
                await self._validate_performance(template_content, report)
            
            if self.validation_level == ValidationLevel.ENTERPRISE:
                await self._validate_compliance(template_content, report)
            
            # Calculate metrics
            report.complexity_score = self._calculate_complexity_score(template_content)
            report.security_score = self._calculate_security_score(report)
            
        except Exception as e:
            logger.error("Template validation failed", template_id=template_id, error=str(e))
            report.add_issue(ValidationIssue(
                severity=ValidationResult.ERROR,
                category="system",
                message=f"Validation system error: {str(e)}",
                rule_id="SYSTEM_ERROR"
            ))
        
        # Calculate duration
        end_time = datetime.now()
        report.validation_duration_seconds = (end_time - start_time).total_seconds()
        
        logger.info("Template validation completed", 
                   template_id=template_id,
                   result=report.overall_result.value,
                   issues=len(report.issues),
                   duration=report.validation_duration_seconds)
        
        return report
    
    async def validate_multiple_templates(self, template_ids: List[str]) -> Dict[str, ValidationReport]:
        """Validate multiple templates"""
        logger.info("Validating multiple templates", count=len(template_ids))
        
        reports = {}
        
        # Create validation tasks
        tasks = []
        for template_id in template_ids:
            task = asyncio.create_task(self.validate_template(template_id))
            tasks.append((template_id, task))
        
        # Wait for all validations to complete
        for template_id, task in tasks:
            try:
                report = await task
                reports[template_id] = report
            except Exception as e:
                logger.error("Failed to validate template", template_id=template_id, error=str(e))
                # Create error report
                reports[template_id] = ValidationReport(
                    template_id=template_id,
                    validation_level=self.validation_level,
                    overall_result=ValidationResult.ERROR
                )
                reports[template_id].add_issue(ValidationIssue(
                    severity=ValidationResult.ERROR,
                    category="system",
                    message=f"Validation failed: {str(e)}",
                    rule_id="VALIDATION_FAILED"
                ))
        
        return reports
    
    def _load_template_content(self, template_path: Path) -> Dict[str, Any]:
        """Load template content from file"""
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if template_path.suffix.lower() == '.json':
                return json.loads(content)
            elif template_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(content)
            else:
                raise ValueError(f"Unsupported file format: {template_path.suffix}")
                
        except Exception as e:
            raise ValueError(f"Failed to load template content: {str(e)}")
    
    async def _validate_syntax(self, template_path: Path, template_content: Dict[str, Any], 
                              report: ValidationReport):
        """Validate template syntax"""
        logger.debug("Validating syntax", template_path=str(template_path))
        
        try:
            # Check file extension
            if template_path.suffix.lower() not in ['.json', '.yaml', '.yml']:
                report.add_issue(ValidationIssue(
                    severity=ValidationResult.WARNING,
                    category="syntax",
                    message=f"Unexpected file extension: {template_path.suffix}",
                    rule_id="INVALID_EXTENSION"
                ))
            
            # Check encoding
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for BOM
                if content.startswith('\ufeff'):
                    report.add_issue(ValidationIssue(
                        severity=ValidationResult.WARNING,
                        category="syntax",
                        message="File contains Byte Order Mark (BOM)",
                        suggestion="Remove BOM for better compatibility",
                        rule_id="BOM_DETECTED"
                    ))
                
                # Check for null bytes
                if '\x00' in content:
                    report.add_issue(ValidationIssue(
                        severity=ValidationResult.ERROR,
                        category="syntax",
                        message="File contains null bytes",
                        rule_id="NULL_BYTES"
                    ))
                
            except UnicodeDecodeError as e:
                report.add_issue(ValidationIssue(
                    severity=ValidationResult.ERROR,
                    category="syntax",
                    message=f"Invalid UTF-8 encoding: {str(e)}",
                    rule_id="ENCODING_ERROR"
                ))
            
            # Validate JSON/YAML structure
            if isinstance(template_content, dict):
                await self._validate_json_structure(template_content, report)
            else:
                report.add_issue(ValidationIssue(
                    severity=ValidationResult.ERROR,
                    category="syntax",
                    message="Template root must be an object/dictionary",
                    rule_id="INVALID_ROOT_TYPE"
                ))
            
            report.syntax_validation = {
                "status": "completed",
                "issues_found": len([i for i in report.issues if i.category == "syntax"])
            }
            
        except Exception as e:
            report.add_issue(ValidationIssue(
                severity=ValidationResult.ERROR,
                category="syntax",
                message=f"Syntax validation failed: {str(e)}",
                rule_id="SYNTAX_ERROR"
            ))
    
    async def _validate_json_structure(self, template_content: Dict[str, Any], 
                                     report: ValidationReport):
        """Validate JSON structure"""
        # Check for required top-level fields
        required_fields = ["id", "name", "version", "type"]
        for field in required_fields:
            if field not in template_content:
                report.add_issue(ValidationIssue(
                    severity=ValidationResult.WARNING,
                    category="syntax",
                    message=f"Missing recommended field: {field}",
                    suggestion=f"Add '{field}' field to template",
                    rule_id="MISSING_FIELD"
                ))
        
        # Check for empty values
        for key, value in template_content.items():
            if value == "" or value == [] or value == {}:
                report.add_issue(ValidationIssue(
                    severity=ValidationResult.WARNING,
                    category="syntax",
                    message=f"Empty value for field: {key}",
                    suggestion=f"Provide value for '{key}' or remove the field",
                    rule_id="EMPTY_VALUE"
                ))
    
    async def _validate_schema(self, template_id: str, template_content: Dict[str, Any], 
                              report: ValidationReport):
        """Validate template against schema"""
        logger.debug("Validating schema", template_id=template_id)
        
        try:
            # Get template type
            template_type = template_content.get("type", "unknown")
            
            # Load schema for template type
            schema = self._load_schema_for_type(template_type)
            
            if schema:
                # Validate against schema
                validation_errors = self._validate_against_schema(template_content, schema)
                
                for error in validation_errors:
                    report.add_issue(ValidationIssue(
                        severity=ValidationResult.ERROR,
                        category="schema",
                        message=error["message"],
                        location=error.get("path"),
                        rule_id="SCHEMA_VIOLATION"
                    ))
            else:
                report.add_issue(ValidationIssue(
                    severity=ValidationResult.WARNING,
                    category="schema",
                    message=f"No schema found for template type: {template_type}",
                    rule_id="NO_SCHEMA"
                ))
            
            report.schema_validation = {
                "status": "completed",
                "template_type": template_type,
                "schema_found": schema is not None,
                "issues_found": len([i for i in report.issues if i.category == "schema"])
            }
            
        except Exception as e:
            report.add_issue(ValidationIssue(
                severity=ValidationResult.ERROR,
                category="schema",
                message=f"Schema validation failed: {str(e)}",
                rule_id="SCHEMA_ERROR"
            ))
    
    def _load_schema_for_type(self, template_type: str) -> Optional[Dict[str, Any]]:
        """Load schema for template type"""
        # Mock schema loading
        # In real implementation, this would load actual JSON schemas
        schemas = {
            "tenant": {
                "type": "object",
                "required": ["id", "name", "tier", "limits"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "tier": {"type": "string", "enum": ["free", "professional", "enterprise"]},
                    "limits": {"type": "object"}
                }
            },
            "user": {
                "type": "object",
                "required": ["id", "email", "tier"],
                "properties": {
                    "id": {"type": "string"},
                    "email": {"type": "string", "format": "email"},
                    "tier": {"type": "string", "enum": ["free", "premium", "enterprise", "vip"]}
                }
            }
        }
        
        return schemas.get(template_type)
    
    def _validate_against_schema(self, template_content: Dict[str, Any], 
                                schema: Dict[str, Any]) -> List[Dict[str, str]]:
        """Validate content against schema"""
        # Mock schema validation
        # In real implementation, this would use jsonschema library
        errors = []
        
        # Check required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in template_content:
                errors.append({
                    "message": f"Required field missing: {field}",
                    "path": f"/{field}"
                })
        
        return errors
    
    async def _validate_dependencies(self, template_id: str, template_content: Dict[str, Any], 
                                   report: ValidationReport):
        """Validate template dependencies"""
        logger.debug("Validating dependencies", template_id=template_id)
        
        try:
            dependencies = template_content.get("dependencies", [])
            
            # Check for circular dependencies
            circular_deps = self._check_circular_dependencies(template_id, dependencies)
            if circular_deps:
                report.add_issue(ValidationIssue(
                    severity=ValidationResult.ERROR,
                    category="dependencies",
                    message=f"Circular dependency detected: {' -> '.join(circular_deps)}",
                    rule_id="CIRCULAR_DEPENDENCY"
                ))
            
            # Check if dependencies exist
            for dep_id in dependencies:
                if not self._dependency_exists(dep_id):
                    report.add_issue(ValidationIssue(
                        severity=ValidationResult.ERROR,
                        category="dependencies",
                        message=f"Dependency not found: {dep_id}",
                        suggestion=f"Ensure template '{dep_id}' exists",
                        rule_id="MISSING_DEPENDENCY"
                    ))
            
            # Check version compatibility
            for dep_id in dependencies:
                compatibility_issues = self._check_version_compatibility(template_id, dep_id)
                for issue in compatibility_issues:
                    report.add_issue(ValidationIssue(
                        severity=ValidationResult.WARNING,
                        category="dependencies",
                        message=issue,
                        rule_id="VERSION_COMPATIBILITY"
                    ))
            
            report.dependency_validation = {
                "status": "completed",
                "dependencies_count": len(dependencies),
                "issues_found": len([i for i in report.issues if i.category == "dependencies"])
            }
            
        except Exception as e:
            report.add_issue(ValidationIssue(
                severity=ValidationResult.ERROR,
                category="dependencies",
                message=f"Dependency validation failed: {str(e)}",
                rule_id="DEPENDENCY_ERROR"
            ))
    
    def _check_circular_dependencies(self, template_id: str, dependencies: List[str], 
                                   visited: Optional[Set[str]] = None) -> Optional[List[str]]:
        """Check for circular dependencies"""
        if visited is None:
            visited = set()
        
        if template_id in visited:
            return [template_id]
        
        visited.add(template_id)
        
        for dep_id in dependencies:
            # Load dependency template to check its dependencies
            dep_path = self._get_template_path(dep_id)
            if dep_path.exists():
                try:
                    dep_content = self._load_template_content(dep_path)
                    dep_dependencies = dep_content.get("dependencies", [])
                    
                    circular_path = self._check_circular_dependencies(dep_id, dep_dependencies, visited.copy())
                    if circular_path:
                        return [template_id] + circular_path
                except Exception:
                    # If we can't load the dependency, skip circular check
                    pass
        
        return None
    
    def _dependency_exists(self, dep_id: str) -> bool:
        """Check if dependency template exists"""
        dep_path = self._get_template_path(dep_id)
        return dep_path.exists()
    
    def _check_version_compatibility(self, template_id: str, dep_id: str) -> List[str]:
        """Check version compatibility between templates"""
        # Mock version compatibility check
        # In real implementation, this would check version ranges
        issues = []
        
        # Example compatibility check
        if "old" in dep_id and "new" in template_id:
            issues.append(f"Potential compatibility issue between {template_id} and {dep_id}")
        
        return issues
    
    async def _validate_security(self, template_content: Dict[str, Any], report: ValidationReport):
        """Validate security aspects"""
        logger.debug("Validating security")
        
        try:
            # Convert content to string for pattern matching
            content_str = json.dumps(template_content, indent=2)
            
            # Check for secrets
            for pattern in self.security_rules["secrets_patterns"]:
                matches = re.finditer(pattern, content_str, re.IGNORECASE)
                for match in matches:
                    report.add_issue(ValidationIssue(
                        severity=ValidationResult.ERROR,
                        category="security",
                        message=f"Potential secret detected: {match.group()[:20]}...",
                        suggestion="Remove hardcoded secrets, use environment variables",
                        rule_id="SECRET_DETECTED"
                    ))
            
            # Check for injection vulnerabilities
            for pattern in self.security_rules["injection_patterns"]:
                matches = re.finditer(pattern, content_str, re.IGNORECASE)
                for match in matches:
                    report.add_issue(ValidationIssue(
                        severity=ValidationResult.ERROR,
                        category="security",
                        message=f"Potential injection vulnerability: {match.group()}",
                        suggestion="Sanitize input and use safe alternatives",
                        rule_id="INJECTION_RISK"
                    ))
            
            # Check for dangerous functions
            for func in self.security_rules["dangerous_functions"]:
                if func in content_str:
                    report.add_issue(ValidationIssue(
                        severity=ValidationResult.WARNING,
                        category="security",
                        message=f"Dangerous function used: {func}",
                        suggestion="Review usage and consider safer alternatives",
                        rule_id="DANGEROUS_FUNCTION"
                    ))
            
            # Check for non-HTTPS URLs
            for pattern in self.security_rules["suspicious_urls"]:
                matches = re.finditer(pattern, content_str)
                for match in matches:
                    report.add_issue(ValidationIssue(
                        severity=ValidationResult.WARNING,
                        category="security",
                        message=f"Non-HTTPS URL detected: {match.group()}",
                        suggestion="Use HTTPS URLs for better security",
                        rule_id="INSECURE_URL"
                    ))
            
            report.security_validation = {
                "status": "completed",
                "issues_found": len([i for i in report.issues if i.category == "security"])
            }
            
        except Exception as e:
            report.add_issue(ValidationIssue(
                severity=ValidationResult.ERROR,
                category="security",
                message=f"Security validation failed: {str(e)}",
                rule_id="SECURITY_ERROR"
            ))
    
    async def _validate_performance(self, template_content: Dict[str, Any], report: ValidationReport):
        """Validate performance aspects"""
        logger.debug("Validating performance")
        
        try:
            # Check template size
            content_size = len(json.dumps(template_content))
            if content_size > 1024 * 1024:  # 1MB
                report.add_issue(ValidationIssue(
                    severity=ValidationResult.WARNING,
                    category="performance",
                    message=f"Large template size: {content_size} bytes",
                    suggestion="Consider splitting large templates",
                    rule_id="LARGE_TEMPLATE"
                ))
            
            # Check nesting depth
            max_depth = self._calculate_nesting_depth(template_content)
            if max_depth > 10:
                report.add_issue(ValidationIssue(
                    severity=ValidationResult.WARNING,
                    category="performance",
                    message=f"Deep nesting detected: {max_depth} levels",
                    suggestion="Reduce nesting depth for better performance",
                    rule_id="DEEP_NESTING"
                ))
            
            # Check for large arrays
            large_arrays = self._find_large_arrays(template_content)
            for path, size in large_arrays:
                if size > 1000:
                    report.add_issue(ValidationIssue(
                        severity=ValidationResult.WARNING,
                        category="performance",
                        message=f"Large array at {path}: {size} items",
                        suggestion="Consider pagination or splitting large arrays",
                        rule_id="LARGE_ARRAY"
                    ))
            
            report.performance_validation = {
                "status": "completed",
                "content_size": content_size,
                "max_nesting_depth": max_depth,
                "issues_found": len([i for i in report.issues if i.category == "performance"])
            }
            
        except Exception as e:
            report.add_issue(ValidationIssue(
                severity=ValidationResult.ERROR,
                category="performance",
                message=f"Performance validation failed: {str(e)}",
                rule_id="PERFORMANCE_ERROR"
            ))
    
    def _calculate_nesting_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth"""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._calculate_nesting_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._calculate_nesting_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth
    
    def _find_large_arrays(self, obj: Any, path: str = "") -> List[Tuple[str, int]]:
        """Find large arrays in template"""
        large_arrays = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                large_arrays.extend(self._find_large_arrays(value, new_path))
        elif isinstance(obj, list):
            large_arrays.append((path, len(obj)))
            for i, item in enumerate(obj):
                large_arrays.extend(self._find_large_arrays(item, f"{path}[{i}]"))
        
        return large_arrays
    
    async def _validate_compliance(self, template_content: Dict[str, Any], report: ValidationReport):
        """Validate compliance aspects"""
        logger.debug("Validating compliance")
        
        try:
            # Check GDPR compliance
            if "personal_data" in json.dumps(template_content).lower():
                gdpr_fields = self.compliance_rules["gdpr"]["required_fields"]
                for field in gdpr_fields:
                    if field not in template_content:
                        report.add_issue(ValidationIssue(
                            severity=ValidationResult.WARNING,
                            category="compliance",
                            message=f"GDPR: Missing field '{field}' for personal data handling",
                            suggestion=f"Add '{field}' field for GDPR compliance",
                            rule_id="GDPR_MISSING_FIELD"
                        ))
            
            # Check data retention policies
            if "data_retention" not in template_content and "retention" in json.dumps(template_content).lower():
                report.add_issue(ValidationIssue(
                    severity=ValidationResult.WARNING,
                    category="compliance",
                    message="Data retention policy not specified",
                    suggestion="Add data retention policy for compliance",
                    rule_id="NO_RETENTION_POLICY"
                ))
            
            # Check audit requirements
            if "audit" in template_content:
                audit_config = template_content["audit"]
                if not isinstance(audit_config, dict) or not audit_config.get("enabled", False):
                    report.add_issue(ValidationIssue(
                        severity=ValidationResult.WARNING,
                        category="compliance",
                        message="Audit logging not properly configured",
                        suggestion="Enable and configure audit logging",
                        rule_id="AUDIT_NOT_CONFIGURED"
                    ))
            
            report.compliance_validation = {
                "status": "completed",
                "issues_found": len([i for i in report.issues if i.category == "compliance"])
            }
            
        except Exception as e:
            report.add_issue(ValidationIssue(
                severity=ValidationResult.ERROR,
                category="compliance",
                message=f"Compliance validation failed: {str(e)}",
                rule_id="COMPLIANCE_ERROR"
            ))
    
    def _calculate_complexity_score(self, template_content: Dict[str, Any]) -> int:
        """Calculate template complexity score"""
        try:
            score = 0
            
            # Base score
            score += len(json.dumps(template_content)) // 100  # 1 point per 100 characters
            
            # Nesting penalty
            max_depth = self._calculate_nesting_depth(template_content)
            score += max_depth * 2
            
            # Object count
            def count_objects(obj):
                count = 0
                if isinstance(obj, dict):
                    count += 1
                    for value in obj.values():
                        count += count_objects(value)
                elif isinstance(obj, list):
                    for item in obj:
                        count += count_objects(item)
                return count
            
            score += count_objects(template_content)
            
            return min(score, 100)  # Cap at 100
            
        except Exception:
            return 0
    
    def _calculate_security_score(self, report: ValidationReport) -> int:
        """Calculate security score based on issues"""
        base_score = 100
        
        for issue in report.issues:
            if issue.category == "security":
                if issue.severity == ValidationResult.ERROR:
                    base_score -= 20
                elif issue.severity == ValidationResult.WARNING:
                    base_score -= 5
        
        return max(base_score, 0)
    
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
    
    def generate_validation_summary(self, reports: Dict[str, ValidationReport]) -> Dict[str, Any]:
        """Generate validation summary"""
        total_templates = len(reports)
        passed_templates = len([r for r in reports.values() if r.overall_result == ValidationResult.PASSED])
        warning_templates = len([r for r in reports.values() if r.overall_result == ValidationResult.WARNING])
        failed_templates = len([r for r in reports.values() if r.overall_result == ValidationResult.FAILED])
        error_templates = len([r for r in reports.values() if r.overall_result == ValidationResult.ERROR])
        
        total_issues = sum(len(r.issues) for r in reports.values())
        total_errors = sum(r.error_count for r in reports.values())
        total_warnings = sum(r.warning_count for r in reports.values())
        
        return {
            "summary": {
                "total_templates": total_templates,
                "passed": passed_templates,
                "warnings": warning_templates,
                "failed": failed_templates,
                "errors": error_templates,
                "success_rate": (passed_templates / total_templates * 100) if total_templates > 0 else 0
            },
            "issues": {
                "total_issues": total_issues,
                "total_errors": total_errors,
                "total_warnings": total_warnings
            },
            "validation_level": self.validation_level.value,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

async def main():
    """Main validation function"""
    parser = argparse.ArgumentParser(description="Validate templates")
    parser.add_argument("--templates", nargs="+", 
                       help="Template IDs to validate (default: all)")
    parser.add_argument("--level", choices=["basic", "standard", "strict", "enterprise"],
                       default="standard", help="Validation level")
    parser.add_argument("--output", help="Output file for validation report")
    parser.add_argument("--format", choices=["json", "yaml", "text"], 
                       default="text", help="Output format")
    parser.add_argument("--verbose", action="store_true", 
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create validator
    validator = TemplateValidator(ValidationLevel(args.level))
    
    # Get templates to validate
    if args.templates:
        template_ids = args.templates
    else:
        # Get all templates from registry
        registry_path = validator.config_path / "template_registry.json"
        if registry_path.exists():
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry = json.load(f)
            template_ids = list(registry.get("templates", {}).keys())
        else:
            print("❌ Template registry not found")
            sys.exit(1)
    
    # Validate templates
    try:
        print(f"🔍 Validating {len(template_ids)} templates at {args.level} level...")
        reports = await validator.validate_multiple_templates(template_ids)
        
        # Generate summary
        summary = validator.generate_validation_summary(reports)
        
        # Output results
        if args.format == "text":
            print_text_report(summary, reports)
        elif args.format == "json":
            output_data = {
                "summary": summary,
                "reports": {tid: asdict(report) for tid, report in reports.items()}
            }
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2, default=str)
                print(f"📄 Report saved to {args.output}")
            else:
                print(json.dumps(output_data, indent=2, default=str))
        elif args.format == "yaml":
            import yaml
            output_data = {
                "summary": summary,
                "reports": {tid: asdict(report) for tid, report in reports.items()}
            }
            if args.output:
                with open(args.output, 'w') as f:
                    yaml.dump(output_data, f, default_flow_style=False)
                print(f"📄 Report saved to {args.output}")
            else:
                print(yaml.dump(output_data, default_flow_style=False))
        
        # Exit with appropriate code
        if summary["summary"]["failed"] > 0 or summary["summary"]["errors"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        sys.exit(1)

def print_text_report(summary: Dict[str, Any], reports: Dict[str, ValidationReport]):
    """Print text validation report"""
    print("\n" + "="*80)
    print("TEMPLATE VALIDATION REPORT")
    print("="*80)
    
    # Summary
    print(f"📊 SUMMARY")
    print(f"  Total Templates: {summary['summary']['total_templates']}")
    print(f"  ✅ Passed: {summary['summary']['passed']}")
    print(f"  ⚠️  Warnings: {summary['summary']['warnings']}")
    print(f"  ❌ Failed: {summary['summary']['failed']}")
    print(f"  💥 Errors: {summary['summary']['errors']}")
    print(f"  🎯 Success Rate: {summary['summary']['success_rate']:.1f}%")
    print()
    
    print(f"🐛 ISSUES")
    print(f"  Total Issues: {summary['issues']['total_issues']}")
    print(f"  Errors: {summary['issues']['total_errors']}")
    print(f"  Warnings: {summary['issues']['total_warnings']}")
    print()
    
    # Individual reports
    print("📋 INDIVIDUAL RESULTS")
    for template_id, report in reports.items():
        status_emoji = {
            ValidationResult.PASSED: "✅",
            ValidationResult.WARNING: "⚠️",
            ValidationResult.FAILED: "❌",
            ValidationResult.ERROR: "💥"
        }
        
        print(f"  {status_emoji[report.overall_result]} {template_id}")
        print(f"    Result: {report.overall_result.value}")
        print(f"    Issues: {len(report.issues)} (Errors: {report.error_count}, Warnings: {report.warning_count})")
        print(f"    Complexity: {report.complexity_score}/100")
        print(f"    Security: {report.security_score}/100")
        print(f"    Duration: {report.validation_duration_seconds:.2f}s")
        
        if report.issues:
            print("    Issues:")
            for issue in report.issues[:5]:  # Show first 5 issues
                severity_emoji = {
                    ValidationResult.ERROR: "❌",
                    ValidationResult.WARNING: "⚠️"
                }
                print(f"      {severity_emoji.get(issue.severity, '•')} [{issue.category}] {issue.message}")
            
            if len(report.issues) > 5:
                print(f"      ... and {len(report.issues) - 5} more issues")
        
        print()
    
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
