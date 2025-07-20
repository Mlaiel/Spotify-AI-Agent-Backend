# Validation Module - Spotify AI Agent

## Overview

The validation module provides comprehensive data validation and sanitization capabilities for the Spotify AI Agent's schema system. It includes custom validators, data sanitizers, and configuration rules to ensure data integrity and security across all system components.

## Key Features

### 🔒 Advanced Validation Rules
- **Tenant ID Validation**: Pattern matching, reserved word checking, length limits
- **Alert Message Validation**: Content sanitization, malicious code detection
- **Correlation ID Validation**: Format verification and consistency checks
- **Metadata Validation**: Size limits, JSON structure verification
- **Tag Validation**: Key-value pair rules, naming conventions
- **Recipient Validation**: Channel-specific format validation (email, Slack, SMS)

### 🛡️ Security Sanitization
- **HTML Sanitization**: Removal of dangerous tags and attributes
- **SQL Injection Protection**: Keyword detection and prevention
- **XSS Prevention**: Script and event handler removal
- **Input Normalization**: Whitespace and control character handling

### ⚙️ Configuration Validators
- **JSON Template Validation**: Jinja2 syntax verification
- **Cron Expression Validation**: Schedule format checking
- **URL Validation**: Protocol and domain restrictions
- **Time Range Validation**: Logical consistency checks

## Architecture

```
validation/
├── __init__.py              # Main validation classes and decorators
├── README.md               # This documentation
├── README.fr.md           # French documentation
├── README.de.md           # German documentation
├── custom_validators.py    # Domain-specific validators
├── security_rules.py      # Security validation rules
├── configuration.py       # Configuration validation schemas
└── performance.py         # Performance-optimized validators
```

## Usage Examples

### Basic Field Validation

```python
from pydantic import BaseModel
from schemas.validation import (
    validate_tenant_id_field, 
    validate_alert_message_field,
    validate_metadata_field
)

class AlertSchema(BaseModel):
    tenant_id: str
    message: str
    metadata: Dict[str, Any]
    
    # Apply validators
    _validate_tenant_id = validate_tenant_id_field()
    _validate_message = validate_alert_message_field()
    _validate_metadata = validate_metadata_field()
```

### Custom Validation Rules

```python
from schemas.validation import ValidationRules

# Validate tenant ID
try:
    clean_tenant_id = ValidationRules.validate_tenant_id("my-tenant-2024")
except ValueError as e:
    print(f"Validation error: {e}")

# Validate alert message
clean_message = ValidationRules.validate_alert_message(
    "System alert: High CPU usage detected"
)

# Validate recipients by channel
email_recipients = ValidationRules.validate_recipients_list(
    ["admin@example.com", "ops@company.com"],
    NotificationChannel.EMAIL
)
```

### Data Sanitization

```python
from schemas.validation import DataSanitizer

# Sanitize HTML content
clean_html = DataSanitizer.sanitize_html(
    "<p>Safe content</p><script>alert('xss')</script>"
)
# Result: "<p>Safe content</p>"

# Normalize whitespace
clean_text = DataSanitizer.normalize_whitespace(
    "  Multiple   spaces\t\nand\r\nlinebreaks  "
)
# Result: "Multiple spaces and linebreaks"

# Truncate long text
short_text = DataSanitizer.truncate_text(
    "Very long text content here...", 
    max_length=20, 
    suffix="..."
)
```

### Complex Validation Scenarios

```python
from schemas.validation import ValidationRules

# Validate JSON template with variables
template = """
{
    "alert": "{{ alert.name }}",
    "severity": "{{ alert.level }}",
    "tenant": "{{ tenant.id }}"
}
"""
ValidationRules.validate_json_template(template)

# Validate cron expression
ValidationRules.validate_cron_expression("0 */6 * * *")  # Every 6 hours

# Validate webhook URL
from pydantic import HttpUrl
webhook_url = ValidationRules.validate_webhook_url(
    HttpUrl("https://api.external-service.com/webhook")
)
```

## Validation Patterns

The module includes predefined patterns for common validation scenarios:

- **TENANT_ID**: `^[a-z0-9_-]+$`
- **CORRELATION_ID**: `^[a-zA-Z0-9_-]+$`
- **EMAIL**: RFC 5322 compliant pattern
- **PHONE**: International format support
- **URL**: HTTP/HTTPS with domain restrictions
- **VERSION**: Semantic versioning pattern
- **HEX_COLOR**: 6-digit hexadecimal colors

## Performance Considerations

### Compiled Patterns
All regex patterns are pre-compiled for optimal performance in high-throughput scenarios.

### Caching
Validation results are cached where appropriate to avoid redundant processing.

### Batch Validation
Support for validating multiple items in a single operation:

```python
# Validate multiple recipients at once
recipients = ["user1@example.com", "user2@example.com", "user3@example.com"]
validated = ValidationRules.validate_recipients_list(recipients, NotificationChannel.EMAIL)
```

## Error Handling

### Validation Errors
All validation errors provide detailed messages for debugging:

```python
try:
    ValidationRules.validate_tenant_id("INVALID-TENANT!")
except ValueError as e:
    # Error: "Tenant ID must contain only letters, numbers, hyphens and underscores"
    handle_validation_error(e)
```

### Security Violations
Security-related validation failures are logged and can trigger alerts:

```python
# This will raise an error and log a security event
ValidationRules.validate_alert_message("<script>malicious_code()</script>")
```

## Integration with Pydantic

### Custom Validators
Use the provided decorators for seamless Pydantic integration:

```python
from pydantic import BaseModel
from schemas.validation import validate_tenant_id_field, validate_time_range_fields

class TenantAlert(BaseModel):
    tenant_id: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    
    _validate_tenant = validate_tenant_id_field()
    _validate_time_range = validate_time_range_fields('start_time', 'end_time')
```

### Root Validators
For cross-field validation logic:

```python
from pydantic import root_validator
from schemas.validation import ValidationRules

class ComplexSchema(BaseModel):
    field1: str
    field2: str
    
    @root_validator
    def validate_consistency(cls, values):
        # Custom inter-field validation logic
        if values.get('field1') and values.get('field2'):
            # Ensure consistency between fields
            pass
        return values
```

## Configuration

### Environment Variables
- `VALIDATION_STRICT_MODE`: Enable/disable strict validation
- `MAX_VALIDATION_CACHE_SIZE`: Validation cache size limit
- `VALIDATION_LOG_LEVEL`: Logging level for validation events

### Customization
Extend validation rules for specific use cases:

```python
from schemas.validation import ValidationRules

class CustomValidationRules(ValidationRules):
    @classmethod
    def validate_custom_field(cls, value: str) -> str:
        # Custom validation logic
        return super().validate_tenant_id(value)
```

## Security Best Practices

1. **Input Sanitization**: Always sanitize user input before processing
2. **Whitelist Validation**: Use positive validation (allowed patterns) over blacklists
3. **Length Limits**: Enforce reasonable limits on all string fields
4. **Content Filtering**: Check for malicious patterns in user content
5. **URL Restrictions**: Validate and restrict webhook URLs to prevent SSRF
6. **Template Safety**: Validate template syntax to prevent injection attacks

## Testing

The validation module includes comprehensive test coverage for all validation scenarios. Tests are centralized according to project guidelines.

```bash
# Run validation tests (when available)
make test-validation

# Run security validation tests
make test-security

# Run performance benchmarks
make benchmark-validation
```

## Contributing

When adding new validation rules:

1. Follow the existing pattern structure
2. Include comprehensive error messages
3. Add performance benchmarks for complex validators
4. Document security implications
5. Provide usage examples

## Author

This module was developed as part of the Spotify AI Agent project, focusing on enterprise-grade data validation and security.

---

For more information, see the main project documentation or contact the development team.
