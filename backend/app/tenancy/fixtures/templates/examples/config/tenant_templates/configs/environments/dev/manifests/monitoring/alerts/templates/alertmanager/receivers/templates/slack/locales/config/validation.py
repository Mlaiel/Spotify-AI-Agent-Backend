"""
Comprehensive Validation System for Slack Alert Configuration.

This module provides extensive validation capabilities for all components
of the Slack alert system including configuration validation, schema
validation, input sanitization, and business rule validation.

Author: Fahed Mlaiel
Version: 1.0.0
"""

import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse

from pydantic import BaseModel, Field, validator, ValidationError as PydanticValidationError
import validators

from .constants import (
    VALIDATION_RULES,
    SUPPORTED_LOCALES,
    ALERT_PRIORITIES,
    MAX_MESSAGE_LENGTH,
    MAX_TEMPLATE_SIZE
)
from .exceptions import (
    ValidationError,
    SchemaValidationError,
    InputValidationError,
    SlackValidationError
)


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    validated_data: Optional[Dict[str, Any]] = None


class SlackTokenModel(BaseModel):
    """Pydantic model for Slack token validation."""
    token: str = Field(..., min_length=40, max_length=200)
    
    @validator('token')
    def validate_token_format(cls, v):
        pattern = VALIDATION_RULES['slack_token']['pattern']
        if not re.match(pattern, v):
            raise ValueError('Invalid Slack token format')
        return v


class SlackWebhookModel(BaseModel):
    """Pydantic model for Slack webhook URL validation."""
    webhook_url: str = Field(..., max_length=500)
    
    @validator('webhook_url')
    def validate_webhook_url(cls, v):
        pattern = VALIDATION_RULES['slack_webhook']['pattern']
        if not re.match(pattern, v):
            raise ValueError('Invalid Slack webhook URL format')
        return v


class ChannelConfigModel(BaseModel):
    """Pydantic model for channel configuration validation."""
    channel_id: str = Field(..., min_length=9, max_length=11)
    channel_name: str = Field(..., min_length=1, max_length=80)
    webhook_url: str = Field(..., max_length=500)
    token: str = Field(..., min_length=40, max_length=200)
    priority_level: str = Field(default="medium")
    rate_limit: int = Field(default=100, ge=1, le=10000)
    enabled: bool = Field(default=True)
    
    @validator('channel_id')
    def validate_channel_id(cls, v):
        pattern = VALIDATION_RULES['channel_id']['pattern']
        if not re.match(pattern, v):
            raise ValueError('Invalid Slack channel ID format')
        return v
    
    @validator('priority_level')
    def validate_priority_level(cls, v):
        if v not in ALERT_PRIORITIES:
            raise ValueError(f'Priority level must be one of: {", ".join(ALERT_PRIORITIES)}')
        return v
    
    @validator('webhook_url')
    def validate_webhook_url(cls, v):
        pattern = VALIDATION_RULES['slack_webhook']['pattern']
        if not re.match(pattern, v):
            raise ValueError('Invalid Slack webhook URL format')
        return v
    
    @validator('token')
    def validate_token(cls, v):
        pattern = VALIDATION_RULES['slack_token']['pattern']
        if not re.match(pattern, v):
            raise ValueError('Invalid Slack token format')
        return v


class TenantConfigModel(BaseModel):
    """Pydantic model for tenant configuration validation."""
    tenant_id: str = Field(..., min_length=3, max_length=50)
    environment: str = Field(..., min_length=1, max_length=20)
    default_locale: str = Field(default="en_US")
    channels: Dict[str, ChannelConfigModel] = Field(default_factory=dict)
    global_settings: Dict[str, Any] = Field(default_factory=dict)
    security_settings: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('tenant_id')
    def validate_tenant_id(cls, v):
        pattern = VALIDATION_RULES['tenant_id']['pattern']
        if not re.match(pattern, v):
            raise ValueError('Invalid tenant ID format')
        return v
    
    @validator('default_locale')
    def validate_locale(cls, v):
        if v not in SUPPORTED_LOCALES:
            raise ValueError(f'Locale must be one of: {", ".join(SUPPORTED_LOCALES[:10])}...')
        return v
    
    @validator('channels')
    def validate_channels(cls, v):
        if not v:
            raise ValueError('At least one channel configuration is required')
        return v


class TemplateModel(BaseModel):
    """Pydantic model for template validation."""
    name: str = Field(..., min_length=1, max_length=100)
    content: str = Field(..., min_length=1, max_length=MAX_TEMPLATE_SIZE)
    version: str = Field(default="1.0.0")
    author: str = Field(default="Unknown")
    description: str = Field(default="")
    variables: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    
    @validator('name')
    def validate_template_name(cls, v):
        pattern = VALIDATION_RULES['template_name']['pattern']
        if not re.match(pattern, v):
            raise ValueError('Invalid template name format')
        return v
    
    @validator('content')
    def validate_template_content(cls, v):
        # Check for potentially dangerous patterns
        dangerous_patterns = [
            r'__import__',
            r'exec\s*\(',
            r'eval\s*\(',
            r'open\s*\(',
            r'file\s*\('
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f'Template contains potentially dangerous pattern: {pattern}')
        
        return v


class ConfigValidator:
    """
    Comprehensive configuration validator for Slack alert system.
    
    Provides validation for all types of configuration data including
    tenant configs, templates, locales, and business rules.
    """

    def __init__(self):
        """Initialize the configuration validator."""
        self.logger = logging.getLogger(__name__)
        
        # Compiled regex patterns for performance
        self.compiled_patterns = {
            name: re.compile(rule['pattern']) 
            for name, rule in VALIDATION_RULES.items() 
            if 'pattern' in rule
        }
        
        # Business rules
        self.business_rules = {
            'max_channels_per_tenant': 50,
            'max_templates_per_tenant': 100,
            'max_rate_limit': 10000,
            'min_rate_limit': 1,
            'required_channel_fields': ['channel_id', 'webhook_url', 'token'],
            'allowed_environments': ['development', 'staging', 'production'],
            'max_security_settings': 20,
            'max_global_settings': 30
        }

    async def validate_tenant_config(self, config_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate complete tenant configuration.
        
        Args:
            config_data: Configuration data to validate
            
        Returns:
            Validation result with errors and warnings
        """
        try:
            result = ValidationResult(
                is_valid=True,
                errors=[],
                warnings=[],
                suggestions=[]
            )
            
            # Validate using Pydantic model
            try:
                validated_model = TenantConfigModel(**config_data)
                result.validated_data = validated_model.dict()
            except PydanticValidationError as e:
                result.is_valid = False
                for error in e.errors():
                    field_path = " -> ".join(str(loc) for loc in error['loc'])
                    result.errors.append(f"{field_path}: {error['msg']}")
                return result
            
            # Additional business rule validation
            await self._validate_business_rules(config_data, result)
            
            # Performance recommendations
            await self._add_performance_suggestions(config_data, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Tenant config validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                suggestions=[]
            )

    async def validate_channel_config(self, channel_data: Dict[str, Any]) -> ValidationResult:
        """Validate individual channel configuration."""
        try:
            result = ValidationResult(
                is_valid=True,
                errors=[],
                warnings=[],
                suggestions=[]
            )
            
            # Validate using Pydantic model
            try:
                validated_model = ChannelConfigModel(**channel_data)
                result.validated_data = validated_model.dict()
            except PydanticValidationError as e:
                result.is_valid = False
                for error in e.errors():
                    field_path = " -> ".join(str(loc) for loc in error['loc'])
                    result.errors.append(f"{field_path}: {error['msg']}")
                return result
            
            # Additional validation
            await self._validate_slack_connectivity(channel_data, result)
            await self._validate_channel_permissions(channel_data, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Channel config validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                suggestions=[]
            )

    async def validate_template(self, template_data: Dict[str, Any]) -> ValidationResult:
        """Validate template configuration."""
        try:
            result = ValidationResult(
                is_valid=True,
                errors=[],
                warnings=[],
                suggestions=[]
            )
            
            # Validate using Pydantic model
            try:
                validated_model = TemplateModel(**template_data)
                result.validated_data = validated_model.dict()
            except PydanticValidationError as e:
                result.is_valid = False
                for error in e.errors():
                    field_path = " -> ".join(str(loc) for loc in error['loc'])
                    result.errors.append(f"{field_path}: {error['msg']}")
                return result
            
            # Template-specific validation
            await self._validate_template_syntax(template_data.get('content', ''), result)
            await self._validate_template_variables(template_data, result)
            await self._validate_template_security(template_data.get('content', ''), result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Template validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                suggestions=[]
            )

    async def validate_locale_code(self, locale_code: str) -> ValidationResult:
        """Validate locale code format and support."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[]
        )
        
        try:
            # Format validation
            pattern = VALIDATION_RULES['locale_code']['pattern']
            if not re.match(pattern, locale_code):
                result.is_valid = False
                result.errors.append(f"Invalid locale code format: {locale_code}")
                return result
            
            # Support validation
            if locale_code not in SUPPORTED_LOCALES:
                result.is_valid = False
                result.errors.append(f"Unsupported locale: {locale_code}")
                
                # Suggest similar locales
                language = locale_code.split('_')[0]
                similar_locales = [loc for loc in SUPPORTED_LOCALES if loc.startswith(f"{language}_")]
                if similar_locales:
                    result.suggestions.append(f"Did you mean one of: {', '.join(similar_locales[:3])}")
            
            result.validated_data = {"locale_code": locale_code}
            return result
            
        except Exception as e:
            self.logger.error(f"Locale validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                suggestions=[]
            )

    async def validate_slack_token(self, token: str) -> ValidationResult:
        """Validate Slack token format and authenticity."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[]
        )
        
        try:
            # Format validation
            pattern = self.compiled_patterns['slack_token']
            if not pattern.match(token):
                result.is_valid = False
                result.errors.append("Invalid Slack token format")
                result.suggestions.append("Slack tokens should start with 'xoxb-', 'xoxp-', 'xoxa-', or 'xoxs-'")
                return result
            
            # Length validation
            max_length = VALIDATION_RULES['slack_token']['max_length']
            if len(token) > max_length:
                result.is_valid = False
                result.errors.append(f"Token too long: {len(token)} > {max_length}")
            
            # Token type detection
            token_type = self._detect_slack_token_type(token)
            if token_type:
                result.validated_data = {"token": token, "token_type": token_type}
                result.suggestions.append(f"Detected {token_type} token")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Slack token validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                suggestions=[]
            )

    async def validate_slack_webhook(self, webhook_url: str) -> ValidationResult:
        """Validate Slack webhook URL."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[]
        )
        
        try:
            # Format validation
            pattern = self.compiled_patterns['slack_webhook']
            if not pattern.match(webhook_url):
                result.is_valid = False
                result.errors.append("Invalid Slack webhook URL format")
                result.suggestions.append("Webhook URL should be from hooks.slack.com/services/...")
                return result
            
            # URL structure validation
            parsed_url = urlparse(webhook_url)
            if parsed_url.scheme != 'https':
                result.is_valid = False
                result.errors.append("Webhook URL must use HTTPS")
            
            if parsed_url.netloc != 'hooks.slack.com':
                result.is_valid = False
                result.errors.append("Webhook URL must be from hooks.slack.com")
            
            # Path validation
            path_parts = parsed_url.path.strip('/').split('/')
            if len(path_parts) != 4 or path_parts[0] != 'services':
                result.is_valid = False
                result.errors.append("Invalid webhook URL path structure")
            
            result.validated_data = {"webhook_url": webhook_url}
            return result
            
        except Exception as e:
            self.logger.error(f"Slack webhook validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                suggestions=[]
            )

    async def validate_create_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data for creating new configuration."""
        validation_result = await self.validate_tenant_config(data)
        
        if not validation_result.is_valid:
            raise ValidationError(
                f"Configuration validation failed: {'; '.join(validation_result.errors)}"
            )
        
        return validation_result.validated_data

    async def validate_update_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data for updating existing configuration."""
        # For updates, we allow partial data
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[],
            validated_data={}
        )
        
        # Validate each field that's present
        if 'default_locale' in data:
            locale_result = await self.validate_locale_code(data['default_locale'])
            if not locale_result.is_valid:
                result.errors.extend(locale_result.errors)
                result.is_valid = False
            else:
                result.validated_data['default_locale'] = data['default_locale']
        
        if 'channels' in data:
            for channel_name, channel_data in data['channels'].items():
                channel_result = await self.validate_channel_config(channel_data)
                if not channel_result.is_valid:
                    result.errors.append(f"Channel '{channel_name}': {'; '.join(channel_result.errors)}")
                    result.is_valid = False
            
            if result.is_valid:
                result.validated_data['channels'] = data['channels']
        
        # Validate other fields
        for field in ['global_settings', 'security_settings']:
            if field in data:
                if isinstance(data[field], dict):
                    result.validated_data[field] = data[field]
                else:
                    result.errors.append(f"{field} must be a dictionary")
                    result.is_valid = False
        
        if not result.is_valid:
            raise ValidationError(
                f"Update validation failed: {'; '.join(result.errors)}"
            )
        
        return result.validated_data

    async def sanitize_input(self, input_data: str, max_length: int = 1000) -> str:
        """Sanitize user input for security."""
        try:
            # Remove potentially dangerous characters
            sanitized = re.sub(r'[<>"\';()&+]', '', input_data)
            
            # Remove control characters
            sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
            
            # Limit length
            sanitized = sanitized[:max_length]
            
            # Strip whitespace
            sanitized = sanitized.strip()
            
            return sanitized
            
        except Exception as e:
            self.logger.error(f"Input sanitization failed: {e}")
            return ""

    async def validate_message_content(self, content: str) -> ValidationResult:
        """Validate Slack message content."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[]
        )
        
        try:
            # Length validation
            if len(content) > MAX_MESSAGE_LENGTH:
                result.is_valid = False
                result.errors.append(f"Message too long: {len(content)} > {MAX_MESSAGE_LENGTH}")
            
            # Check for valid Slack formatting
            self._validate_slack_formatting(content, result)
            
            # Security checks
            self._validate_content_security(content, result)
            
            result.validated_data = {"content": content}
            return result
            
        except Exception as e:
            self.logger.error(f"Message content validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                suggestions=[]
            )

    # Private helper methods
    
    async def _validate_business_rules(self, config_data: Dict[str, Any], result: ValidationResult) -> None:
        """Validate business rules for configuration."""
        try:
            # Check maximum channels
            channels = config_data.get('channels', {})
            max_channels = self.business_rules['max_channels_per_tenant']
            if len(channels) > max_channels:
                result.warnings.append(f"Many channels configured ({len(channels)} > {max_channels} recommended)")
            
            # Check environment
            environment = config_data.get('environment', '')
            allowed_envs = self.business_rules['allowed_environments']
            if environment not in allowed_envs:
                result.warnings.append(f"Unusual environment: {environment}. Typical: {', '.join(allowed_envs)}")
            
            # Check security settings
            security_settings = config_data.get('security_settings', {})
            max_security = self.business_rules['max_security_settings']
            if len(security_settings) > max_security:
                result.warnings.append(f"Many security settings ({len(security_settings)} > {max_security} typical)")
            
        except Exception as e:
            self.logger.warning(f"Business rules validation failed: {e}")

    async def _add_performance_suggestions(self, config_data: Dict[str, Any], result: ValidationResult) -> None:
        """Add performance optimization suggestions."""
        try:
            channels = config_data.get('channels', {})
            
            # Check rate limits
            high_rate_channels = [
                name for name, channel in channels.items() 
                if isinstance(channel, dict) and channel.get('rate_limit', 0) > 1000
            ]
            
            if high_rate_channels:
                result.suggestions.append(
                    f"Consider lowering rate limits for channels: {', '.join(high_rate_channels[:3])}"
                )
            
            # Check number of enabled channels
            enabled_channels = [
                name for name, channel in channels.items()
                if isinstance(channel, dict) and channel.get('enabled', True)
            ]
            
            if len(enabled_channels) > 20:
                result.suggestions.append(
                    "Consider disabling unused channels to improve performance"
                )
            
        except Exception as e:
            self.logger.warning(f"Performance suggestions failed: {e}")

    async def _validate_slack_connectivity(self, channel_data: Dict[str, Any], result: ValidationResult) -> None:
        """Validate Slack connectivity settings."""
        try:
            # This would typically test actual connectivity
            # For now, we just validate the configuration format
            
            webhook_url = channel_data.get('webhook_url', '')
            token = channel_data.get('token', '')
            
            if webhook_url and token:
                result.warnings.append("Both webhook URL and token provided - webhook will take precedence")
            
            if not webhook_url and not token:
                result.errors.append("Either webhook URL or token must be provided")
            
        except Exception as e:
            self.logger.warning(f"Slack connectivity validation failed: {e}")

    async def _validate_channel_permissions(self, channel_data: Dict[str, Any], result: ValidationResult) -> None:
        """Validate channel permissions and access."""
        try:
            # Check priority level implications
            priority = channel_data.get('priority_level', 'medium')
            rate_limit = channel_data.get('rate_limit', 100)
            
            if priority in ['critical', 'high'] and rate_limit < 10:
                result.warnings.append(
                    f"Low rate limit ({rate_limit}) for {priority} priority channel may cause delays"
                )
            
        except Exception as e:
            self.logger.warning(f"Channel permissions validation failed: {e}")

    async def _validate_template_syntax(self, content: str, result: ValidationResult) -> None:
        """Validate template syntax."""
        try:
            # Check for balanced braces
            open_braces = content.count('{{')
            close_braces = content.count('}}')
            
            if open_braces != close_braces:
                result.errors.append(f"Unbalanced template braces: {open_braces} open, {close_braces} close")
            
            # Check for valid variable names
            variable_pattern = re.compile(r'{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}')
            variables = variable_pattern.findall(content)
            
            invalid_vars = [var for var in variables if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var)]
            if invalid_vars:
                result.errors.extend([f"Invalid variable name: {var}" for var in invalid_vars])
            
        except Exception as e:
            self.logger.warning(f"Template syntax validation failed: {e}")

    async def _validate_template_variables(self, template_data: Dict[str, Any], result: ValidationResult) -> None:
        """Validate template variables."""
        try:
            content = template_data.get('content', '')
            declared_variables = template_data.get('variables', [])
            
            # Extract variables from content
            variable_pattern = re.compile(r'{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}')
            used_variables = set(variable_pattern.findall(content))
            declared_set = set(declared_variables)
            
            # Check for undeclared variables
            undeclared = used_variables - declared_set
            if undeclared:
                result.warnings.extend([f"Undeclared variable: {var}" for var in undeclared])
            
            # Check for unused declared variables
            unused = declared_set - used_variables
            if unused:
                result.suggestions.extend([f"Unused declared variable: {var}" for var in unused])
            
        except Exception as e:
            self.logger.warning(f"Template variables validation failed: {e}")

    async def _validate_template_security(self, content: str, result: ValidationResult) -> None:
        """Validate template security."""
        try:
            # Check for dangerous patterns
            dangerous_patterns = [
                (r'__import__', 'Python import statements'),
                (r'exec\s*\(', 'Python exec calls'),
                (r'eval\s*\(', 'Python eval calls'),
                (r'<script', 'JavaScript code'),
                (r'javascript:', 'JavaScript URLs')
            ]
            
            for pattern, description in dangerous_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    result.errors.append(f"Potentially dangerous content: {description}")
            
        except Exception as e:
            self.logger.warning(f"Template security validation failed: {e}")

    def _detect_slack_token_type(self, token: str) -> Optional[str]:
        """Detect Slack token type from prefix."""
        token_types = {
            'xoxb-': 'Bot User OAuth Token',
            'xoxp-': 'User OAuth Token',
            'xoxa-': 'App-Level Token',
            'xoxs-': 'Socket Mode Token'
        }
        
        for prefix, token_type in token_types.items():
            if token.startswith(prefix):
                return token_type
        
        return None

    def _validate_slack_formatting(self, content: str, result: ValidationResult) -> None:
        """Validate Slack-specific formatting."""
        try:
            # Check for valid Slack mentions
            mention_pattern = re.compile(r'<@[UW][A-Z0-9]+(\|[^>]+)?>')
            invalid_mentions = re.findall(r'<@[^UW][^>]*>', content)
            if invalid_mentions:
                result.warnings.extend([f"Invalid mention format: {mention}" for mention in invalid_mentions[:3]])
            
            # Check for valid channel references
            channel_pattern = re.compile(r'<#[A-Z0-9]+(\|[^>]+)?>')
            invalid_channels = re.findall(r'<#[^A-Z0-9][^>]*>', content)
            if invalid_channels:
                result.warnings.extend([f"Invalid channel reference: {ref}" for ref in invalid_channels[:3]])
            
            # Check for valid emoji format
            emoji_pattern = re.compile(r':[a-zA-Z0-9_+-]+:')
            emoji_matches = emoji_pattern.findall(content)
            if emoji_matches:
                result.suggestions.append(f"Found {len(emoji_matches)} emoji references")
            
        except Exception as e:
            self.logger.warning(f"Slack formatting validation failed: {e}")

    def _validate_content_security(self, content: str, result: ValidationResult) -> None:
        """Validate content for security issues."""
        try:
            # Check for potential XSS
            xss_patterns = [
                r'<script[^>]*>',
                r'javascript:',
                r'on\w+\s*=',
                r'<iframe[^>]*>'
            ]
            
            for pattern in xss_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    result.errors.append("Potential XSS content detected")
                    break
            
            # Check for SQL injection patterns
            sql_patterns = [
                r'(\'\s*(or|and)\s*\')',
                r'union\s+select',
                r'drop\s+table',
                r'delete\s+from'
            ]
            
            for pattern in sql_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    result.warnings.append("Potential SQL injection pattern detected")
                    break
            
        except Exception as e:
            self.logger.warning(f"Content security validation failed: {e}")
