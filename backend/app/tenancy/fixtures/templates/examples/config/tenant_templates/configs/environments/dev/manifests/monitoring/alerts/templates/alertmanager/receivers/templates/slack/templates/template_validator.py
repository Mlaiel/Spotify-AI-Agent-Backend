#!/usr/bin/env python3
"""
Enterprise Template Validation & Testing Framework - Advanced Industrial Grade
Developed by: Fahed Mlaiel (Lead Dev + AI Architect)

This module provides comprehensive validation, testing, and quality assurance
for Slack alert templates with enterprise-grade features.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
import jinja2
from jinja2 import Environment, FileSystemLoader, meta
import aiofiles
import yaml
import pytest
from unittest.mock import Mock
import re
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Comprehensive validation result with detailed feedback"""
    template_path: str
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0


@dataclass
class TemplateTestCase:
    """Template test case definition"""
    name: str
    template_path: str
    context_data: Dict[str, Any]
    expected_output: Optional[str] = None
    expected_patterns: List[str] = field(default_factory=list)
    should_fail: bool = False
    performance_threshold_ms: float = 100.0
    accessibility_check: bool = True


class TemplateValidator(ABC):
    """Abstract base class for template validators"""
    
    @abstractmethod
    async def validate(self, template_path: str, template_content: str) -> ValidationResult:
        """Validate template and return detailed results"""
        pass


class SyntaxValidator(TemplateValidator):
    """Validates Jinja2 template syntax"""
    
    def __init__(self):
        self.env = Environment()
    
    async def validate(self, template_path: str, template_content: str) -> ValidationResult:
        """Validate Jinja2 syntax"""
        result = ValidationResult(template_path=template_path, is_valid=True)
        
        try:
            # Parse template to check syntax
            ast = self.env.parse(template_content)
            
            # Check for undefined variables
            undefined_vars = meta.find_undeclared_variables(ast)
            if undefined_vars:
                result.warnings.extend([
                    f"Undefined variable: {var}" for var in undefined_vars
                ])
            
            # Check for complex expressions that might be slow
            complex_expressions = self._find_complex_expressions(template_content)
            if complex_expressions:
                result.warnings.extend([
                    f"Complex expression found: {expr}" for expr in complex_expressions
                ])
            
        except jinja2.TemplateSyntaxError as e:
            result.is_valid = False
            result.errors.append(f"Syntax error: {str(e)}")
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Unexpected error: {str(e)}")
        
        return result
    
    def _find_complex_expressions(self, content: str) -> List[str]:
        """Find potentially complex or slow expressions"""
        complex_patterns = [
            r'\{\%\s*for\s+\w+\s+in\s+.*\|\s*length\s*>\s*\d+',  # Large loops
            r'\{\{.*\|.*\|.*\|.*\}\}',  # Multiple chained filters
            r'\{\%.*recursive.*\%\}',  # Recursive macros
        ]
        
        found = []
        for pattern in complex_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            found.extend(matches)
        
        return found


class ContentValidator(TemplateValidator):
    """Validates template content quality and best practices"""
    
    def __init__(self):
        self.required_fields = [
            'alert.alert_id',
            'alert.title',
            'alert.severity',
            'environment'
        ]
        self.recommended_fields = [
            'alert.created_at',
            'alert.context.service_name',
            'dashboard_url',
            'metrics_url'
        ]
    
    async def validate(self, template_path: str, template_content: str) -> ValidationResult:
        """Validate content quality and completeness"""
        result = ValidationResult(template_path=template_path, is_valid=True)
        
        # Check for required fields
        missing_required = []
        for field in self.required_fields:
            if field not in template_content:
                missing_required.append(field)
        
        if missing_required:
            result.errors.extend([
                f"Missing required field: {field}" for field in missing_required
            ])
            result.is_valid = False
        
        # Check for recommended fields
        missing_recommended = []
        for field in self.recommended_fields:
            if field not in template_content:
                missing_recommended.append(field)
        
        if missing_recommended:
            result.warnings.extend([
                f"Missing recommended field: {field}" for field in missing_recommended
            ])
        
        # Check template length
        if len(template_content) > 10000:
            result.warnings.append("Template is very long, consider breaking it down")
        
        # Check for hardcoded values
        hardcoded_patterns = [
            r'https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?!/\{\{)',  # Hardcoded URLs
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP addresses
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email addresses
        ]
        
        for pattern in hardcoded_patterns:
            matches = re.findall(pattern, template_content)
            if matches:
                result.warnings.extend([
                    f"Hardcoded value found: {match}" for match in matches
                ])
        
        # Check for accessibility
        accessibility_issues = self._check_accessibility(template_content)
        if accessibility_issues:
            result.warnings.extend(accessibility_issues)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(result, template_content)
        result.quality_score = quality_score
        
        return result
    
    def _check_accessibility(self, content: str) -> List[str]:
        """Check for accessibility issues"""
        issues = []
        
        # Check for excessive emoji usage
        emoji_count = len(re.findall(r'[\U00010000-\U0010ffff]', content))
        if emoji_count > 20:
            issues.append(f"Excessive emoji usage ({emoji_count}), may impact accessibility")
        
        # Check for color-only information
        color_indicators = re.findall(r'red|green|yellow|blue|orange', content, re.IGNORECASE)
        if color_indicators and len(color_indicators) > 3:
            issues.append("Color-only indicators detected, consider adding text alternatives")
        
        return issues
    
    def _calculate_quality_score(self, result: ValidationResult, content: str) -> float:
        """Calculate overall template quality score"""
        score = 100.0
        
        # Deduct for errors
        score -= len(result.errors) * 20
        
        # Deduct for warnings
        score -= len(result.warnings) * 5
        
        # Bonus for good practices
        if 'ai_insights' in content:
            score += 5
        if 'business_impact' in content:
            score += 5
        if 'escalation' in content:
            score += 3
        
        return max(0.0, min(100.0, score))


class PerformanceValidator(TemplateValidator):
    """Validates template rendering performance"""
    
    def __init__(self, max_render_time_ms: float = 100.0):
        self.max_render_time_ms = max_render_time_ms
        self.env = Environment()
    
    async def validate(self, template_path: str, template_content: str) -> ValidationResult:
        """Validate template rendering performance"""
        result = ValidationResult(template_path=template_path, is_valid=True)
        
        try:
            # Prepare test context
            test_context = self._create_test_context()
            
            # Measure rendering time
            start_time = datetime.utcnow()
            template = self.env.from_string(template_content)
            rendered = template.render(**test_context)
            end_time = datetime.utcnow()
            
            render_time_ms = (end_time - start_time).total_seconds() * 1000
            result.metrics['render_time_ms'] = render_time_ms
            result.metrics['output_length'] = len(rendered)
            
            # Check performance thresholds
            if render_time_ms > self.max_render_time_ms:
                result.warnings.append(
                    f"Slow rendering: {render_time_ms:.2f}ms (threshold: {self.max_render_time_ms}ms)"
                )
            
            # Check output size
            if len(rendered) > 40000:  # Slack message limit
                result.errors.append(f"Output too large: {len(rendered)} characters")
                result.is_valid = False
            
        except Exception as e:
            result.errors.append(f"Performance test failed: {str(e)}")
            result.is_valid = False
        
        return result
    
    def _create_test_context(self) -> Dict[str, Any]:
        """Create comprehensive test context for performance testing"""
        return {
            'alert': {
                'alert_id': 'test-alert-123',
                'title': 'Test Alert',
                'description': 'This is a test alert for performance validation',
                'severity': 'critical',
                'status': 'firing',
                'created_at': datetime.utcnow().isoformat(),
                'duration': 300,
                'context': {
                    'service_name': 'test-service',
                    'component': 'test-component',
                    'instance_id': 'i-test123',
                    'cluster_name': 'test-cluster',
                    'region': 'us-east-1'
                },
                'metrics': {
                    'cpu_usage': '85%',
                    'memory_usage': '78%',
                    'error_rate': '2.3%'
                },
                'ai_insights': {
                    'recommended_actions': [
                        'Scale up the service',
                        'Check for memory leaks',
                        'Review recent deployments'
                    ],
                    'confidence_score': 87
                },
                'business_impact': {
                    'level': 'high',
                    'affected_users': '10,000+',
                    'estimated_cost': '$500/hour'
                }
            },
            'environment': 'production',
            'tenant_id': 'test-tenant',
            'dashboard_url': 'https://dashboard.example.com',
            'metrics_url': 'https://metrics.example.com',
            'logs_url': 'https://logs.example.com',
            'tracing_url': 'https://tracing.example.com'
        }


class SecurityValidator(TemplateValidator):
    """Validates template security and prevents injection attacks"""
    
    def __init__(self):
        self.dangerous_patterns = [
            r'\{\{.*\|.*safe.*\}\}',  # |safe filter usage
            r'\{\%.*raw.*\%\}',  # raw blocks
            r'javascript:',  # JavaScript URLs
            r'<script[^>]*>',  # Script tags
            r'eval\s*\(',  # eval() calls
        ]
    
    async def validate(self, template_path: str, template_content: str) -> ValidationResult:
        """Validate template security"""
        result = ValidationResult(template_path=template_path, is_valid=True)
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            matches = re.findall(pattern, template_content, re.IGNORECASE)
            if matches:
                result.errors.extend([
                    f"Security risk detected: {match}" for match in matches
                ])
                result.is_valid = False
        
        # Check for potential XSS vulnerabilities
        xss_patterns = [
            r'\{\{.*\|.*safe.*\}\}',
            r'href\s*=\s*["\']?\s*\{\{',
            r'src\s*=\s*["\']?\s*\{\{'
        ]
        
        for pattern in xss_patterns:
            matches = re.findall(pattern, template_content, re.IGNORECASE)
            if matches:
                result.warnings.extend([
                    f"Potential XSS risk: {match}" for match in matches
                ])
        
        return result


class TemplateTestRunner:
    """Comprehensive template testing framework"""
    
    def __init__(self, template_dir: str):
        self.template_dir = Path(template_dir)
        self.validators = [
            SyntaxValidator(),
            ContentValidator(),
            PerformanceValidator(),
            SecurityValidator()
        ]
        self.test_cases = []
    
    async def validate_all_templates(self) -> Dict[str, List[ValidationResult]]:
        """Validate all templates in the directory"""
        results = {}
        
        # Find all template files
        template_files = list(self.template_dir.glob('**/*.j2'))
        
        for template_file in template_files:
            try:
                async with aiofiles.open(template_file, 'r') as f:
                    content = await f.read()
                
                template_results = []
                for validator in self.validators:
                    result = await validator.validate(str(template_file), content)
                    template_results.append(result)
                
                results[str(template_file)] = template_results
                
            except Exception as e:
                logger.error(f"Failed to validate {template_file}: {str(e)}")
                error_result = ValidationResult(
                    template_path=str(template_file),
                    is_valid=False,
                    errors=[f"Failed to read template: {str(e)}"]
                )
                results[str(template_file)] = [error_result]
        
        return results
    
    async def run_test_cases(self, test_cases: List[TemplateTestCase]) -> List[Dict[str, Any]]:
        """Run comprehensive test cases"""
        results = []
        
        for test_case in test_cases:
            result = await self._run_single_test(test_case)
            results.append(result)
        
        return results
    
    async def _run_single_test(self, test_case: TemplateTestCase) -> Dict[str, Any]:
        """Run a single test case"""
        result = {
            'test_name': test_case.name,
            'template_path': test_case.template_path,
            'passed': False,
            'errors': [],
            'metrics': {}
        }
        
        try:
            # Load template
            template_path = self.template_dir / test_case.template_path
            async with aiofiles.open(template_path, 'r') as f:
                template_content = await f.read()
            
            # Setup Jinja environment
            env = Environment()
            template = env.from_string(template_content)
            
            # Measure performance
            start_time = datetime.utcnow()
            rendered = template.render(**test_case.context_data)
            end_time = datetime.utcnow()
            
            render_time_ms = (end_time - start_time).total_seconds() * 1000
            result['metrics']['render_time_ms'] = render_time_ms
            result['metrics']['output_length'] = len(rendered)
            
            # Check performance threshold
            if render_time_ms > test_case.performance_threshold_ms:
                result['errors'].append(
                    f"Performance threshold exceeded: {render_time_ms:.2f}ms > {test_case.performance_threshold_ms}ms"
                )
            
            # Check expected patterns
            for pattern in test_case.expected_patterns:
                if not re.search(pattern, rendered, re.IGNORECASE):
                    result['errors'].append(f"Expected pattern not found: {pattern}")
            
            # Check expected output
            if test_case.expected_output and test_case.expected_output.strip() not in rendered:
                result['errors'].append("Expected output not found")
            
            # Accessibility check
            if test_case.accessibility_check:
                accessibility_issues = self._check_accessibility(rendered)
                if accessibility_issues:
                    result['errors'].extend(accessibility_issues)
            
            result['passed'] = len(result['errors']) == 0
            result['rendered_output'] = rendered
            
        except Exception as e:
            if test_case.should_fail:
                result['passed'] = True
                result['expected_failure'] = str(e)
            else:
                result['errors'].append(f"Test execution failed: {str(e)}")
        
        return result
    
    def _check_accessibility(self, content: str) -> List[str]:
        """Check rendered content for accessibility issues"""
        issues = []
        
        # Check for proper structure
        if content.count('\n') < 3:
            issues.append("Content may lack proper structure")
        
        # Check for excessive length
        if len(content) > 10000:
            issues.append("Content may be too long for easy reading")
        
        return issues
    
    async def generate_test_report(self, 
                                 validation_results: Dict[str, List[ValidationResult]],
                                 test_results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive test report"""
        
        report = []
        report.append("# Spotify AI Agent Template Validation Report")
        report.append(f"Generated: {datetime.utcnow().isoformat()}")
        report.append("")
        
        # Validation summary
        total_templates = len(validation_results)
        valid_templates = sum(1 for results in validation_results.values() 
                            if all(r.is_valid for r in results))
        
        report.append("## Validation Summary")
        report.append(f"- Total Templates: {total_templates}")
        report.append(f"- Valid Templates: {valid_templates}")
        report.append(f"- Invalid Templates: {total_templates - valid_templates}")
        report.append("")
        
        # Quality scores
        quality_scores = []
        for results in validation_results.values():
            for result in results:
                if hasattr(result, 'quality_score') and result.quality_score > 0:
                    quality_scores.append(result.quality_score)
        
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            report.append(f"- Average Quality Score: {avg_quality:.1f}/100")
            report.append("")
        
        # Test results summary
        if test_results:
            passed_tests = sum(1 for result in test_results if result['passed'])
            report.append("## Test Results Summary")
            report.append(f"- Total Tests: {len(test_results)}")
            report.append(f"- Passed Tests: {passed_tests}")
            report.append(f"- Failed Tests: {len(test_results) - passed_tests}")
            report.append("")
        
        # Detailed validation results
        report.append("## Detailed Validation Results")
        for template_path, results in validation_results.items():
            report.append(f"### {template_path}")
            
            for result in results:
                validator_name = result.__class__.__name__.replace('ValidationResult', 'Validator')
                report.append(f"#### {validator_name}")
                report.append(f"- Valid: {'✅' if result.is_valid else '❌'}")
                
                if result.errors:
                    report.append("- Errors:")
                    for error in result.errors:
                        report.append(f"  - {error}")
                
                if result.warnings:
                    report.append("- Warnings:")
                    for warning in result.warnings:
                        report.append(f"  - {warning}")
                
                if hasattr(result, 'quality_score') and result.quality_score > 0:
                    report.append(f"- Quality Score: {result.quality_score:.1f}/100")
                
                report.append("")
        
        # Detailed test results
        if test_results:
            report.append("## Detailed Test Results")
            for result in test_results:
                report.append(f"### {result['test_name']}")
                report.append(f"- Status: {'✅ PASSED' if result['passed'] else '❌ FAILED'}")
                report.append(f"- Template: {result['template_path']}")
                
                if result.get('metrics'):
                    report.append("- Metrics:")
                    for metric, value in result['metrics'].items():
                        report.append(f"  - {metric}: {value}")
                
                if result.get('errors'):
                    report.append("- Errors:")
                    for error in result['errors']:
                        report.append(f"  - {error}")
                
                report.append("")
        
        return "\n".join(report)


# Example test cases
def create_default_test_cases() -> List[TemplateTestCase]:
    """Create default test cases for template validation"""
    
    base_context = {
        'alert': {
            'alert_id': 'test-123',
            'title': 'Test Critical Alert',
            'description': 'This is a test critical alert',
            'severity': 'critical',
            'status': 'firing',
            'created_at': datetime.utcnow().isoformat(),
            'context': {
                'service_name': 'test-service',
                'component': 'api-gateway',
                'instance_id': 'i-test123'
            }
        },
        'environment': 'production',
        'dashboard_url': 'https://dashboard.test.com',
        'metrics_url': 'https://metrics.test.com'
    }
    
    return [
        TemplateTestCase(
            name="Critical Alert - English Text",
            template_path="critical_en_text.j2",
            context_data=base_context,
            expected_patterns=[
                r'CRITICAL.*ALERT',
                r'test-service',
                r'test-123'
            ]
        ),
        TemplateTestCase(
            name="Critical Alert - French Text",
            template_path="critical_fr_text.j2",
            context_data=base_context,
            expected_patterns=[
                r'ALERTE.*CRITIQUE',
                r'test-service'
            ]
        ),
        TemplateTestCase(
            name="Warning Alert Performance",
            template_path="warning_en_text.j2",
            context_data={**base_context, 'alert': {**base_context['alert'], 'severity': 'warning'}},
            performance_threshold_ms=50.0
        )
    ]


async def main():
    """Main function for running template validation"""
    template_dir = Path(__file__).parent
    
    # Initialize test runner
    runner = TemplateTestRunner(str(template_dir))
    
    # Run validation
    print("Running template validation...")
    validation_results = await runner.validate_all_templates()
    
    # Run test cases
    print("Running test cases...")
    test_cases = create_default_test_cases()
    test_results = await runner.run_test_cases(test_cases)
    
    # Generate report
    print("Generating report...")
    report = await runner.generate_test_report(validation_results, test_results)
    
    # Save report
    report_path = template_dir / "validation_report.md"
    async with aiofiles.open(report_path, 'w') as f:
        await f.write(report)
    
    print(f"Validation complete. Report saved to: {report_path}")
    print("\n" + "="*50)
    print(report[:1000] + "..." if len(report) > 1000 else report)


if __name__ == "__main__":
    asyncio.run(main())
