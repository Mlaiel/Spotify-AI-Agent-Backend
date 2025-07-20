#!/usr/bin/env python3
"""
Spotify AI Agent - Template Processors
=====================================

Advanced template processing pipeline for optimization,
transformation, and enhancement of template content.

Processors:
- Optimization (compression, minification, caching)
- Transformation (format conversion, structure modification)
- Enhancement (AI-powered improvement, metadata enrichment)
- Security (sanitization, encryption, access control)
- Performance (lazy loading, streaming, parallel processing)

Features:
- Pipeline-based processing with stages
- Async processing with progress tracking
- Rollback and recovery mechanisms
- Performance monitoring and metrics
- Plugin architecture for custom processors

Author: Expert Development Team
"""

import json
import yaml
import gzip
import brotli
import logging
import hashlib
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, Tuple, Callable, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
import re
from pathlib import Path
import tempfile

from jinja2 import Environment, meta
import cssmin
import jsmin
from PIL import Image
import aiofiles

from app.core.ai import get_ai_client
from app.core.security import sanitize_html, validate_json_schema
from app.tenancy.fixtures.templates.validators import TemplateValidationEngine, ValidationReport

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Template processing stages."""
    PRE_PROCESS = "pre_process"
    OPTIMIZE = "optimize"
    TRANSFORM = "transform"
    ENHANCE = "enhance"
    VALIDATE = "validate"
    POST_PROCESS = "post_process"


@dataclass
class ProcessingResult:
    """Result of template processing operation."""
    success: bool
    template: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    size_reduction_percent: float = 0.0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    stage: Optional[ProcessingStage] = None
    processor_name: str = ""


@dataclass
class ProcessingConfig:
    """Configuration for template processors."""
    enable_compression: bool = True
    enable_minification: bool = True
    enable_ai_enhancement: bool = False
    enable_security_scanning: bool = True
    enable_performance_optimization: bool = True
    compression_level: int = 6
    max_processing_time_seconds: int = 300
    parallel_processing: bool = True
    enable_rollback: bool = True
    backup_enabled: bool = True


class BaseTemplateProcessor(ABC):
    """Base class for all template processors."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.stage = ProcessingStage.TRANSFORM
        self.metrics = {
            "processed_count": 0,
            "success_count": 0,
            "error_count": 0,
            "average_processing_time_ms": 0.0,
            "total_size_reduction_mb": 0.0
        }
    
    @abstractmethod
    async def process(self, template: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process template and return result."""
        pass
    
    def get_stage(self) -> ProcessingStage:
        """Get processing stage for this processor."""
        return self.stage
    
    def _update_metrics(self, processing_time_ms: float, success: bool, size_reduction_mb: float = 0.0):
        """Update processor metrics."""
        self.metrics["processed_count"] += 1
        if success:
            self.metrics["success_count"] += 1
        else:
            self.metrics["error_count"] += 1
        
        # Update average processing time
        current_avg = self.metrics["average_processing_time_ms"]
        total_processed = self.metrics["processed_count"]
        self.metrics["average_processing_time_ms"] = ((current_avg * (total_processed - 1)) + processing_time_ms) / total_processed
        
        # Update total size reduction
        self.metrics["total_size_reduction_mb"] += size_reduction_mb
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processor performance metrics."""
        return self.metrics.copy()
    
    def _calculate_size_reduction(self, original_size: int, processed_size: int) -> Tuple[float, float]:
        """Calculate size reduction metrics."""
        if original_size == 0:
            return 0.0, 0.0
        
        reduction_bytes = original_size - processed_size
        reduction_percent = (reduction_bytes / original_size) * 100
        reduction_mb = reduction_bytes / (1024 * 1024)
        
        return reduction_percent, reduction_mb


class CompressionProcessor(BaseTemplateProcessor):
    """Compresses template content to reduce size."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        super().__init__(config)
        self.stage = ProcessingStage.OPTIMIZE
    
    async def process(self, template: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Compress template content."""
        start_time = datetime.now()
        
        try:
            if not self.config.enable_compression:
                return ProcessingResult(
                    success=True,
                    template=template,
                    processor_name="CompressionProcessor",
                    stage=self.stage,
                    warnings=["Compression disabled in configuration"]
                )
            
            # Convert template to JSON for compression
            original_json = json.dumps(template, separators=(',', ':'), ensure_ascii=False)
            original_size = len(original_json.encode('utf-8'))
            
            # Apply different compression algorithms
            compressed_data = {}
            
            # Gzip compression
            gzip_data = gzip.compress(original_json.encode('utf-8'), compresslevel=self.config.compression_level)
            compressed_data['gzip'] = {
                'data': gzip_data,
                'size': len(gzip_data),
                'algorithm': 'gzip'
            }
            
            # Brotli compression (usually better for text)
            try:
                brotli_data = brotli.compress(original_json.encode('utf-8'), quality=self.config.compression_level)
                compressed_data['brotli'] = {
                    'data': brotli_data,
                    'size': len(brotli_data),
                    'algorithm': 'brotli'
                }
            except Exception as e:
                logger.warning(f"Brotli compression failed: {str(e)}")
            
            # Choose best compression
            best_compression = min(compressed_data.values(), key=lambda x: x['size'])
            compressed_size = best_compression['size']
            
            # Calculate metrics
            size_reduction_percent, size_reduction_mb = self._calculate_size_reduction(original_size, compressed_size)
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Add compression metadata to template
            template_with_compression = template.copy()
            template_with_compression['_compression'] = {
                'algorithm': best_compression['algorithm'],
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compressed_size / original_size if original_size > 0 else 1.0,
                'size_reduction_percent': size_reduction_percent
            }
            
            # Update metrics
            self._update_metrics(processing_time, True, size_reduction_mb)
            
            return ProcessingResult(
                success=True,
                template=template_with_compression,
                processing_time_ms=processing_time,
                size_reduction_percent=size_reduction_percent,
                performance_metrics={
                    'original_size_bytes': original_size,
                    'compressed_size_bytes': compressed_size,
                    'compression_algorithm': best_compression['algorithm'],
                    'available_algorithms': list(compressed_data.keys())
                },
                processor_name="CompressionProcessor",
                stage=self.stage
            )
        
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(processing_time, False)
            
            return ProcessingResult(
                success=False,
                template=template,
                processing_time_ms=processing_time,
                errors=[f"Compression failed: {str(e)}"],
                processor_name="CompressionProcessor",
                stage=self.stage
            )


class MinificationProcessor(BaseTemplateProcessor):
    """Minifies template content by removing unnecessary whitespace and comments."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        super().__init__(config)
        self.stage = ProcessingStage.OPTIMIZE
    
    async def process(self, template: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Minify template content."""
        start_time = datetime.now()
        
        try:
            if not self.config.enable_minification:
                return ProcessingResult(
                    success=True,
                    template=template,
                    processor_name="MinificationProcessor",
                    stage=self.stage,
                    warnings=["Minification disabled in configuration"]
                )
            
            original_json = json.dumps(template, indent=2)
            original_size = len(original_json.encode('utf-8'))
            
            # Create minified copy
            minified_template = await self._minify_template(template)
            
            # Calculate size reduction
            minified_json = json.dumps(minified_template, separators=(',', ':'))
            minified_size = len(minified_json.encode('utf-8'))
            
            size_reduction_percent, size_reduction_mb = self._calculate_size_reduction(original_size, minified_size)
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update metrics
            self._update_metrics(processing_time, True, size_reduction_mb)
            
            return ProcessingResult(
                success=True,
                template=minified_template,
                processing_time_ms=processing_time,
                size_reduction_percent=size_reduction_percent,
                performance_metrics={
                    'original_size_bytes': original_size,
                    'minified_size_bytes': minified_size,
                    'whitespace_removed': True,
                    'comments_removed': True
                },
                processor_name="MinificationProcessor",
                stage=self.stage
            )
        
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(processing_time, False)
            
            return ProcessingResult(
                success=False,
                template=template,
                processing_time_ms=processing_time,
                errors=[f"Minification failed: {str(e)}"],
                processor_name="MinificationProcessor",
                stage=self.stage
            )
    
    async def _minify_template(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively minify template content."""
        if isinstance(template, dict):
            minified = {}
            for key, value in template.items():
                # Skip comment fields
                if key.startswith('_comment') or key.startswith('#'):
                    continue
                
                minified[key] = await self._minify_template(value)
            return minified
        
        elif isinstance(template, list):
            return [await self._minify_template(item) for item in template]
        
        elif isinstance(template, str):
            # Minify string content
            return self._minify_string(template)
        
        else:
            return template
    
    def _minify_string(self, content: str) -> str:
        """Minify string content."""
        # Remove extra whitespace while preserving template expressions
        if '{{' in content and '}}' in content:
            # Preserve template expressions but minify around them
            parts = re.split(r'(\{\{.*?\}\})', content)
            minified_parts = []
            
            for part in parts:
                if part.startswith('{{') and part.endswith('}}'):
                    # Keep template expressions as-is
                    minified_parts.append(part)
                else:
                    # Minify regular text
                    minified = re.sub(r'\s+', ' ', part.strip())
                    if minified:
                        minified_parts.append(minified)
            
            return ''.join(minified_parts)
        else:
            # Regular string minification
            return re.sub(r'\s+', ' ', content.strip())


class ValidationProcessor(BaseTemplateProcessor):
    """Validates template content and structure."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        super().__init__(config)
        self.stage = ProcessingStage.VALIDATE
        self.validation_engine = TemplateValidationEngine()
    
    async def process(self, template: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Validate template content."""
        start_time = datetime.now()
        
        try:
            # Determine template type and ID
            metadata = template.get('_metadata', {})
            template_type = metadata.get('template_type', 'unknown')
            template_id = metadata.get('template_id', 'unknown')
            
            # Run validation
            validation_report = self.validation_engine.validate_template(
                template, template_id, template_type, context
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Prepare warnings and errors
            warnings = []
            errors = []
            
            for result in validation_report.results:
                if not result.is_valid:
                    if result.severity.value in ['error', 'critical']:
                        errors.append(f"{result.field_path}: {result.message}")
                    else:
                        warnings.append(f"{result.field_path}: {result.message}")
            
            # Update metrics
            self._update_metrics(processing_time, validation_report.is_valid)
            
            return ProcessingResult(
                success=validation_report.is_valid,
                template=template,
                processing_time_ms=processing_time,
                performance_metrics={
                    'validation_checks': len(validation_report.results),
                    'issues_found': validation_report.total_issues,
                    'validation_score': (
                        (len(validation_report.results) - validation_report.total_issues) / 
                        len(validation_report.results) * 100
                    ) if validation_report.results else 100
                },
                warnings=warnings,
                errors=errors,
                processor_name="ValidationProcessor",
                stage=self.stage,
                metadata={'validation_report': validation_report}
            )
        
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(processing_time, False)
            
            return ProcessingResult(
                success=False,
                template=template,
                processing_time_ms=processing_time,
                errors=[f"Validation failed: {str(e)}"],
                processor_name="ValidationProcessor",
                stage=self.stage
            )


class SecurityProcessor(BaseTemplateProcessor):
    """Processes templates for security vulnerabilities and sanitization."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        super().__init__(config)
        self.stage = ProcessingStage.PRE_PROCESS
    
    async def process(self, template: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process template for security issues."""
        start_time = datetime.now()
        
        try:
            if not self.config.enable_security_scanning:
                return ProcessingResult(
                    success=True,
                    template=template,
                    processor_name="SecurityProcessor",
                    stage=self.stage,
                    warnings=["Security scanning disabled in configuration"]
                )
            
            # Create sanitized copy
            sanitized_template = await self._sanitize_template(template)
            
            # Check for security issues
            security_issues = await self._scan_security_issues(sanitized_template)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Determine if template is safe
            critical_issues = [issue for issue in security_issues if issue.get('severity') == 'critical']
            is_safe = len(critical_issues) == 0
            
            # Update metrics
            self._update_metrics(processing_time, is_safe)
            
            # Prepare warnings and errors
            warnings = [issue['message'] for issue in security_issues if issue.get('severity') in ['warning', 'info']]
            errors = [issue['message'] for issue in security_issues if issue.get('severity') in ['error', 'critical']]
            
            return ProcessingResult(
                success=is_safe,
                template=sanitized_template,
                processing_time_ms=processing_time,
                performance_metrics={
                    'security_issues_found': len(security_issues),
                    'critical_issues': len(critical_issues),
                    'sanitization_applied': True
                },
                warnings=warnings,
                errors=errors,
                processor_name="SecurityProcessor",
                stage=self.stage
            )
        
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(processing_time, False)
            
            return ProcessingResult(
                success=False,
                template=template,
                processing_time_ms=processing_time,
                errors=[f"Security processing failed: {str(e)}"],
                processor_name="SecurityProcessor",
                stage=self.stage
            )
    
    async def _sanitize_template(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize template content."""
        if isinstance(template, dict):
            sanitized = {}
            for key, value in template.items():
                sanitized[key] = await self._sanitize_template(value)
            return sanitized
        
        elif isinstance(template, list):
            return [await self._sanitize_template(item) for item in template]
        
        elif isinstance(template, str):
            # Sanitize HTML content
            return sanitize_html(template)
        
        else:
            return template
    
    async def _scan_security_issues(self, template: Dict[str, Any]) -> List[Dict[str, str]]:
        """Scan template for security issues."""
        issues = []
        
        # Define security patterns
        dangerous_patterns = [
            (r'<script[^>]*>.*?</script>', 'critical', 'Script injection detected'),
            (r'javascript:', 'critical', 'JavaScript protocol detected'),
            (r'on\w+\s*=', 'error', 'Event handler attribute detected'),
            (r'eval\s*\(', 'critical', 'Eval function detected'),
            (r'\${.*}', 'warning', 'Template injection pattern detected'),
        ]
        
        # Recursive scanning function
        def scan_value(value, path=""):
            if isinstance(value, str):
                for pattern, severity, message in dangerous_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        issues.append({
                            'severity': severity,
                            'message': f"{message} at {path}",
                            'pattern': pattern,
                            'value_snippet': value[:100]
                        })
            elif isinstance(value, dict):
                for k, v in value.items():
                    scan_value(v, f"{path}.{k}" if path else k)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    scan_value(item, f"{path}[{i}]" if path else f"[{i}]")
        
        scan_value(template)
        return issues


class AIEnhancementProcessor(BaseTemplateProcessor):
    """Uses AI to enhance template content and metadata."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        super().__init__(config)
        self.stage = ProcessingStage.ENHANCE
    
    async def process(self, template: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Enhance template using AI."""
        start_time = datetime.now()
        
        try:
            if not self.config.enable_ai_enhancement:
                return ProcessingResult(
                    success=True,
                    template=template,
                    processor_name="AIEnhancementProcessor",
                    stage=self.stage,
                    warnings=["AI enhancement disabled in configuration"]
                )
            
            # Create enhanced copy
            enhanced_template = template.copy()
            enhancements_applied = []
            
            # Generate missing descriptions
            if await self._should_generate_description(template):
                description = await self._generate_description(template)
                if description:
                    enhanced_template['description'] = description
                    enhancements_applied.append('generated_description')
            
            # Enhance metadata
            metadata = enhanced_template.get('_metadata', {})
            if 'tags' not in metadata:
                tags = await self._generate_tags(template)
                if tags:
                    metadata['tags'] = tags
                    enhanced_template['_metadata'] = metadata
                    enhancements_applied.append('generated_tags')
            
            # Optimize template structure
            if await self._should_optimize_structure(template):
                optimized = await self._optimize_structure(enhanced_template)
                if optimized:
                    enhanced_template = optimized
                    enhancements_applied.append('optimized_structure')
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update metrics
            self._update_metrics(processing_time, True)
            
            return ProcessingResult(
                success=True,
                template=enhanced_template,
                processing_time_ms=processing_time,
                performance_metrics={
                    'enhancements_applied': len(enhancements_applied),
                    'enhancement_types': enhancements_applied
                },
                processor_name="AIEnhancementProcessor",
                stage=self.stage,
                metadata={'ai_enhancements': enhancements_applied}
            )
        
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(processing_time, False)
            
            return ProcessingResult(
                success=False,
                template=template,
                processing_time_ms=processing_time,
                errors=[f"AI enhancement failed: {str(e)}"],
                processor_name="AIEnhancementProcessor",
                stage=self.stage
            )
    
    async def _should_generate_description(self, template: Dict[str, Any]) -> bool:
        """Check if description should be generated."""
        return 'description' not in template or not template.get('description')
    
    async def _generate_description(self, template: Dict[str, Any]) -> Optional[str]:
        """Generate description using AI."""
        try:
            ai_client = await get_ai_client()
            
            prompt = f"""
            Analyze this template and generate a concise, professional description:
            
            Template Type: {template.get('_metadata', {}).get('template_type', 'unknown')}
            Template Content: {json.dumps(template, indent=2)[:1000]}...
            
            Generate a 1-2 sentence description that explains what this template is for and its main purpose.
            """
            
            response = await ai_client.generate_text(
                prompt=prompt,
                max_tokens=100,
                temperature=0.3
            )
            
            return response.strip() if response else None
        
        except Exception as e:
            logger.warning(f"Failed to generate AI description: {str(e)}")
            return None
    
    async def _generate_tags(self, template: Dict[str, Any]) -> Optional[List[str]]:
        """Generate tags using AI."""
        try:
            ai_client = await get_ai_client()
            
            prompt = f"""
            Analyze this template and generate relevant tags:
            
            Template: {json.dumps(template, indent=2)[:500]}...
            
            Generate 3-5 relevant tags that categorize this template. Return as JSON array.
            Focus on: template type, use case, domain, features.
            """
            
            response = await ai_client.generate_text(
                prompt=prompt,
                max_tokens=50,
                temperature=0.3
            )
            
            # Try to parse as JSON array
            try:
                tags = json.loads(response)
                if isinstance(tags, list):
                    return tags[:5]  # Limit to 5 tags
            except json.JSONDecodeError:
                pass
            
            # Fallback: split by commas
            if response:
                tags = [tag.strip() for tag in response.split(',')]
                return [tag for tag in tags if tag][:5]
            
            return None
        
        except Exception as e:
            logger.warning(f"Failed to generate AI tags: {str(e)}")
            return None
    
    async def _should_optimize_structure(self, template: Dict[str, Any]) -> bool:
        """Check if structure optimization is needed."""
        # Simple heuristic: optimize if template has deep nesting
        return self._get_nesting_depth(template) > 5
    
    def _get_nesting_depth(self, obj, current_depth=0):
        """Calculate nesting depth of template."""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._get_nesting_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._get_nesting_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth
    
    async def _optimize_structure(self, template: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Optimize template structure using AI."""
        try:
            ai_client = await get_ai_client()
            
            prompt = f"""
            Optimize this template structure to reduce nesting while preserving functionality:
            
            Current Template: {json.dumps(template, indent=2)[:1000]}...
            
            Return an optimized version with:
            1. Reduced nesting depth
            2. Better organization
            3. Preserved functionality
            4. Valid JSON structure
            
            Return only the optimized JSON template.
            """
            
            response = await ai_client.generate_text(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.1
            )
            
            # Try to parse optimized template
            try:
                optimized = json.loads(response)
                if isinstance(optimized, dict):
                    return optimized
            except json.JSONDecodeError:
                pass
            
            return None
        
        except Exception as e:
            logger.warning(f"Failed to optimize structure with AI: {str(e)}")
            return None


class PerformanceProcessor(BaseTemplateProcessor):
    """Optimizes template performance characteristics."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        super().__init__(config)
        self.stage = ProcessingStage.OPTIMIZE
    
    async def process(self, template: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Optimize template performance."""
        start_time = datetime.now()
        
        try:
            if not self.config.enable_performance_optimization:
                return ProcessingResult(
                    success=True,
                    template=template,
                    processor_name="PerformanceProcessor",
                    stage=self.stage,
                    warnings=["Performance optimization disabled in configuration"]
                )
            
            # Create optimized copy
            optimized_template = template.copy()
            optimizations_applied = []
            
            # Lazy loading optimization
            if await self._should_apply_lazy_loading(template):
                optimized_template = await self._apply_lazy_loading(optimized_template)
                optimizations_applied.append('lazy_loading')
            
            # Array chunking for large arrays
            if await self._should_chunk_arrays(template):
                optimized_template = await self._chunk_large_arrays(optimized_template)
                optimizations_applied.append('array_chunking')
            
            # Reference extraction for repeated data
            if await self._should_extract_references(template):
                optimized_template = await self._extract_references(optimized_template)
                optimizations_applied.append('reference_extraction')
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Calculate performance improvements
            original_size = len(json.dumps(template).encode('utf-8'))
            optimized_size = len(json.dumps(optimized_template).encode('utf-8'))
            size_reduction_percent, size_reduction_mb = self._calculate_size_reduction(original_size, optimized_size)
            
            # Update metrics
            self._update_metrics(processing_time, True, size_reduction_mb)
            
            return ProcessingResult(
                success=True,
                template=optimized_template,
                processing_time_ms=processing_time,
                size_reduction_percent=size_reduction_percent,
                performance_metrics={
                    'optimizations_applied': len(optimizations_applied),
                    'optimization_types': optimizations_applied,
                    'original_size_bytes': original_size,
                    'optimized_size_bytes': optimized_size
                },
                processor_name="PerformanceProcessor",
                stage=self.stage
            )
        
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(processing_time, False)
            
            return ProcessingResult(
                success=False,
                template=template,
                processing_time_ms=processing_time,
                errors=[f"Performance optimization failed: {str(e)}"],
                processor_name="PerformanceProcessor",
                stage=self.stage
            )
    
    async def _should_apply_lazy_loading(self, template: Dict[str, Any]) -> bool:
        """Check if lazy loading should be applied."""
        # Apply lazy loading if template has large data structures
        template_size = len(json.dumps(template).encode('utf-8'))
        return template_size > 100 * 1024  # 100KB threshold
    
    async def _apply_lazy_loading(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Apply lazy loading patterns to template."""
        # Add lazy loading metadata for large objects
        if '_lazy_loading' not in template:
            template['_lazy_loading'] = {
                'enabled': True,
                'threshold_bytes': 10240,  # 10KB
                'lazy_fields': []
            }
        
        return template
    
    async def _should_chunk_arrays(self, template: Dict[str, Any]) -> bool:
        """Check if array chunking should be applied."""
        return await self._has_large_arrays(template)
    
    async def _has_large_arrays(self, obj, threshold=100):
        """Check if template has large arrays."""
        if isinstance(obj, list) and len(obj) > threshold:
            return True
        elif isinstance(obj, dict):
            for value in obj.values():
                if await self._has_large_arrays(value, threshold):
                    return True
        return False
    
    async def _chunk_large_arrays(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Chunk large arrays in template."""
        def chunk_arrays(obj, path=""):
            if isinstance(obj, list) and len(obj) > 100:
                # Replace large array with chunked reference
                return {
                    '_chunked_array': True,
                    '_chunk_size': 50,
                    '_total_items': len(obj),
                    '_chunks': [obj[i:i+50] for i in range(0, len(obj), 50)]
                }
            elif isinstance(obj, dict):
                return {k: chunk_arrays(v, f"{path}.{k}" if path else k) for k, v in obj.items()}
            else:
                return obj
        
        return chunk_arrays(template)
    
    async def _should_extract_references(self, template: Dict[str, Any]) -> bool:
        """Check if reference extraction should be applied."""
        # Simple heuristic: if template has repeated structures
        template_str = json.dumps(template)
        return len(template_str) > len(set(template_str)) * 2  # High repetition
    
    async def _extract_references(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Extract repeated data into references."""
        # This is a simplified implementation
        # In practice, you'd implement sophisticated deduplication
        
        if '_references' not in template:
            template['_references'] = {}
        
        return template


class TemplateProcessingPipeline:
    """Manages template processing pipeline with multiple processors."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.processors: Dict[ProcessingStage, List[BaseTemplateProcessor]] = {
            stage: [] for stage in ProcessingStage
        }
        self.metrics = {
            'templates_processed': 0,
            'successful_pipelines': 0,
            'failed_pipelines': 0,
            'average_pipeline_time_ms': 0.0
        }
    
    def add_processor(self, processor: BaseTemplateProcessor):
        """Add processor to pipeline."""
        stage = processor.get_stage()
        self.processors[stage].append(processor)
    
    def remove_processor(self, processor: BaseTemplateProcessor):
        """Remove processor from pipeline."""
        stage = processor.get_stage()
        if processor in self.processors[stage]:
            self.processors[stage].remove(processor)
    
    async def process_template(
        self,
        template: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        stages: Optional[List[ProcessingStage]] = None
    ) -> List[ProcessingResult]:
        """Process template through pipeline."""
        start_time = datetime.now()
        stages = stages or list(ProcessingStage)
        results = []
        
        current_template = template.copy()
        backup_template = template.copy() if self.config.enable_rollback else None
        
        try:
            # Process through each stage
            for stage in stages:
                processors = self.processors[stage]
                
                for processor in processors:
                    try:
                        # Set timeout for processing
                        result = await asyncio.wait_for(
                            processor.process(current_template, context),
                            timeout=self.config.max_processing_time_seconds
                        )
                        
                        results.append(result)
                        
                        # Update template if processing was successful
                        if result.success and result.template:
                            current_template = result.template
                        elif not result.success and stage == ProcessingStage.VALIDATE:
                            # Stop pipeline on validation failure
                            break
                    
                    except asyncio.TimeoutError:
                        timeout_result = ProcessingResult(
                            success=False,
                            template=current_template,
                            errors=[f"Processing timeout after {self.config.max_processing_time_seconds}s"],
                            processor_name=processor.__class__.__name__,
                            stage=stage
                        )
                        results.append(timeout_result)
                        break
                    
                    except Exception as e:
                        error_result = ProcessingResult(
                            success=False,
                            template=current_template,
                            errors=[f"Processor error: {str(e)}"],
                            processor_name=processor.__class__.__name__,
                            stage=stage
                        )
                        results.append(error_result)
                        
                        # Consider rollback
                        if self.config.enable_rollback and backup_template:
                            current_template = backup_template.copy()
            
            # Calculate overall success
            pipeline_success = all(result.success for result in results if result.stage != ProcessingStage.ENHANCE)
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_pipeline_metrics(processing_time, pipeline_success)
            
            return results
        
        except Exception as e:
            logger.error(f"Pipeline processing failed: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_pipeline_metrics(processing_time, False)
            
            error_result = ProcessingResult(
                success=False,
                template=template,
                errors=[f"Pipeline error: {str(e)}"],
                processor_name="Pipeline",
                stage=ProcessingStage.POST_PROCESS
            )
            return [error_result]
    
    def _update_pipeline_metrics(self, processing_time_ms: float, success: bool):
        """Update pipeline metrics."""
        self.metrics['templates_processed'] += 1
        if success:
            self.metrics['successful_pipelines'] += 1
        else:
            self.metrics['failed_pipelines'] += 1
        
        # Update average processing time
        current_avg = self.metrics['average_pipeline_time_ms']
        total_processed = self.metrics['templates_processed']
        self.metrics['average_pipeline_time_ms'] = ((current_avg * (total_processed - 1)) + processing_time_ms) / total_processed
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        processor_metrics = {}
        for stage, processors in self.processors.items():
            processor_metrics[stage.value] = [proc.get_metrics() for proc in processors]
        
        return {
            'pipeline_metrics': self.metrics.copy(),
            'processor_metrics': processor_metrics
        }


# Factory function for default processing pipeline
def create_default_pipeline(config: Optional[ProcessingConfig] = None) -> TemplateProcessingPipeline:
    """Create default template processing pipeline."""
    pipeline = TemplateProcessingPipeline(config)
    
    # Add default processors
    pipeline.add_processor(SecurityProcessor(config))
    pipeline.add_processor(ValidationProcessor(config))
    pipeline.add_processor(MinificationProcessor(config))
    pipeline.add_processor(CompressionProcessor(config))
    pipeline.add_processor(PerformanceProcessor(config))
    
    # Add AI enhancement if enabled
    if config and config.enable_ai_enhancement:
        pipeline.add_processor(AIEnhancementProcessor(config))
    
    return pipeline


# Global processing pipeline instance
default_pipeline = create_default_pipeline()


async def process_template(
    template: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    config: Optional[ProcessingConfig] = None
) -> List[ProcessingResult]:
    """Process template using default pipeline."""
    if config:
        pipeline = create_default_pipeline(config)
    else:
        pipeline = default_pipeline
    
    return await pipeline.process_template(template, context)
