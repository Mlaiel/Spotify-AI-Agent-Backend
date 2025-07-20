# Spotify AI Agent - Template System

## Overview

The Spotify AI Agent Template System is an enterprise-grade template management platform designed for multi-tenant environments. It provides comprehensive template lifecycle management, including creation, validation, processing, migration, and deployment.

## Features

### 🚀 Core Features
- **Multi-tenant template isolation** with secure data separation
- **Advanced template engine** with Jinja2 and custom filters
- **Comprehensive validation** with security, schema, and business logic checks
- **High-performance caching** with Redis and LRU eviction strategies
- **Template versioning** with automated migration support
- **AI-powered enhancements** for template optimization and generation

### 🏗️ Architecture Components

#### Template Engine (`engine.py`)
- High-performance template rendering with caching
- Custom Jinja2 filters and functions
- Security-first template processing
- Real-time template compilation and validation

#### Template Manager (`manager.py`)
- Enterprise template lifecycle management
- Template discovery and metadata management
- Backup, restore, import/export capabilities
- Advanced search and filtering

#### Template Generators (`generators.py`)
- Dynamic template generation for different categories:
  - **Tenant Templates**: Initialization, configuration, billing
  - **User Templates**: Profiles, preferences, onboarding
  - **Content Templates**: Types, workflows, analytics
  - **AI Session Templates**: Configurations, prompts
  - **Collaboration Templates**: Spaces, permissions

#### Template Validators (`validators.py`)
- **Schema Validation**: Structure and type checking
- **Security Validation**: XSS, injection, sensitive data detection
- **Business Logic Validation**: Rules and consistency checks
- **Performance Validation**: Size, complexity, optimization
- **Compliance Validation**: GDPR, data retention policies

#### Template Loaders (`loaders.py`)
- Multi-source loading support:
  - **File System**: Local and network storage
  - **Database**: PostgreSQL with versioning
  - **Remote**: HTTP/HTTPS with caching
  - **Redis**: High-speed cache storage
  - **Git Repository**: Version-controlled templates
- Fallback chains and error recovery
- Performance monitoring and metrics

#### Template Processors (`processors.py`)
- **Compression**: Gzip and Brotli optimization
- **Minification**: Whitespace and comment removal
- **Security**: Sanitization and vulnerability scanning
- **AI Enhancement**: Content improvement and optimization
- **Performance**: Lazy loading and structure optimization

#### Template Migrations (`migrations.py`)
- Version-based migration chains
- Schema evolution and data transformation
- Security updates and compliance migrations
- Rollback and recovery mechanisms
- Multi-tenant migration support

## Quick Start

### Installation

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Configure Environment**
```bash
export REDIS_URL="redis://localhost:6379"
export DATABASE_URL="postgresql://user:pass@localhost/db"
export AI_API_KEY="your-ai-api-key"
```

### Basic Usage

#### 1. Initialize Template System
```python
from app.tenancy.fixtures.templates import TemplateEngine, TemplateManager

# Initialize engine and manager
engine = TemplateEngine()
manager = TemplateManager()

# Setup default loaders
from app.tenancy.fixtures.templates.loaders import setup_default_loaders
setup_default_loaders("/path/to/templates")
```

#### 2. Generate Templates
```python
from app.tenancy.fixtures.templates.generators import get_template_generator

# Generate tenant initialization template
tenant_generator = get_template_generator("tenant")
template = tenant_generator.generate_tenant_init_template(
    tier="professional",
    features=["advanced_ai", "collaboration"],
    integrations=["spotify", "slack"]
)
```

#### 3. Validate Templates
```python
from app.tenancy.fixtures.templates.validators import TemplateValidationEngine

validator = TemplateValidationEngine()
report = validator.validate_template(template, "tenant_001", "tenant_init")

if report.is_valid:
    print("Template is valid!")
else:
    print(f"Validation failed with {report.total_issues} issues")
```

#### 4. Process Templates
```python
from app.tenancy.fixtures.templates.processors import process_template

results = await process_template(template, context={"tenant_id": "tenant_001"})
for result in results:
    if result.success:
        print(f"{result.processor_name}: Optimized by {result.size_reduction_percent:.1f}%")
```

#### 5. Migrate Templates
```python
from app.tenancy.fixtures.templates.migrations import migration_manager

results, migrated_templates = await migration_manager.migrate_templates_to_version(
    templates=[template],
    target_version="1.3.0"
)
```

## Template Categories

### Tenant Templates
- **Initialization**: Multi-tier configuration with limits and features
- **Configuration**: Branding, notifications, compliance settings
- **Permissions**: Role-based access control and custom permissions
- **Billing**: Subscription management and usage tracking
- **Integrations**: Spotify, Slack, Teams, Google Workspace

### User Templates
- **Profile**: User information and music preferences
- **Preferences**: Interface, notifications, AI settings
- **Settings**: Security, API access, data management
- **Roles**: Permission assignments and context roles
- **Onboarding**: Step-by-step user introduction flow

### Content Templates
- **Types**: Playlist, track analysis, music reviews
- **Workflows**: Auto-categorization and AI enhancement
- **Analytics**: Performance metrics and insights

### AI Session Templates
- **Configuration**: Model settings and safety parameters
- **Prompts**: Optimized prompts for different use cases
- **Context**: Memory and conversation management

### Collaboration Templates
- **Spaces**: Music discovery and creative projects
- **Permissions**: Access control and moderation
- **Workflows**: Real-time collaboration features

## Advanced Configuration

### Security Configuration
```python
from app.tenancy.fixtures.templates.validators import SecurityValidator

security_config = {
    "enable_xss_protection": True,
    "enable_injection_detection": True,
    "sensitive_data_scanning": True,
    "encryption_required": True
}

security_validator = SecurityValidator(security_config)
```

### Performance Optimization
```python
from app.tenancy.fixtures.templates.processors import ProcessingConfig

performance_config = ProcessingConfig(
    enable_compression=True,
    enable_minification=True,
    enable_performance_optimization=True,
    compression_level=6,
    parallel_processing=True
)
```

### Custom Template Loaders
```python
from app.tenancy.fixtures.templates.loaders import BaseTemplateLoader

class CustomLoader(BaseTemplateLoader):
    async def load_template(self, identifier: str, **kwargs):
        # Custom loading logic
        pass

# Register custom loader
from app.tenancy.fixtures.templates.loaders import loader_manager
loader_manager.register_loader("custom", CustomLoader())
```

### Migration Management
```python
from app.tenancy.fixtures.templates.migrations import BaseMigration

class CustomMigration(BaseMigration):
    async def migrate_up(self, template, context=None):
        # Migration logic
        return template
    
    async def migrate_down(self, template, context=None):
        # Rollback logic
        return template

# Register custom migration
migration_manager.register_custom_migration(CustomMigration())
```

## API Reference

### Template Engine
```python
class TemplateEngine:
    async def render_template(self, template_content: str, context: Dict[str, Any]) -> str
    async def render_template_from_file(self, template_path: str, context: Dict[str, Any]) -> str
    async def compile_template(self, template_content: str) -> CompiledTemplate
    def clear_cache(self) -> None
    def get_metrics(self) -> Dict[str, Any]
```

### Template Manager
```python
class TemplateManager:
    async def create_template(self, template_data: Dict[str, Any], metadata: TemplateMetadata) -> str
    async def get_template(self, template_id: str) -> Optional[Dict[str, Any]]
    async def update_template(self, template_id: str, template_data: Dict[str, Any]) -> bool
    async def delete_template(self, template_id: str) -> bool
    async def search_templates(self, query: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]
    async def backup_templates(self, backup_name: str) -> str
    async def restore_templates(self, backup_path: str) -> bool
```

### Validation Engine
```python
class TemplateValidationEngine:
    def validate_template(self, template: Dict[str, Any], template_id: str, template_type: str) -> ValidationReport
    def add_validator(self, validator: BaseValidator) -> None
    def remove_validator(self, validator_class: type) -> None
```

## Performance Monitoring

### Metrics Collection
```python
# Engine metrics
engine_metrics = engine.get_metrics()
print(f"Cache hit rate: {engine_metrics['cache_hit_rate']:.2f}%")
print(f"Average render time: {engine_metrics['average_render_time_ms']:.2f}ms")

# Loader metrics
loader_metrics = loader_manager.get_loader_metrics()
for loader_name, metrics in loader_metrics.items():
    print(f"{loader_name}: {metrics['loads_successful']}/{metrics['loads_total']} successful")

# Processor metrics
processor_metrics = default_pipeline.get_pipeline_metrics()
print(f"Pipeline success rate: {processor_metrics['pipeline_metrics']['successful_pipelines']}%")
```

### Performance Optimization
- Enable Redis caching for frequently accessed templates
- Use compression for large templates
- Implement lazy loading for complex templates
- Monitor validation performance and adjust thresholds
- Use background processing for non-critical operations

## Security Best Practices

### Template Security
1. **Input Validation**: All template inputs are validated against schemas
2. **XSS Protection**: Automatic sanitization of HTML content
3. **Injection Prevention**: Detection and blocking of malicious patterns
4. **Sensitive Data**: Automatic detection and masking of sensitive information
5. **Access Control**: Role-based permissions for template operations

### Data Protection
- Templates containing personal data are automatically flagged
- GDPR compliance validation for EU tenants
- Encryption of sensitive template data
- Audit logging for all template operations
- Secure backup and restore procedures

## Troubleshooting

### Common Issues

#### Template Not Found
```python
# Check loader configuration
loader_status = loader_manager.get_loader_metrics()
print("Loader status:", loader_status)

# Verify template path
template_path = "/path/to/templates/tenant/init.json"
if not Path(template_path).exists():
    print("Template file not found")
```

#### Validation Failures
```python
# Get detailed validation report
report = validator.validate_template(template, "template_id", "template_type")
for result in report.results:
    if not result.is_valid:
        print(f"Error: {result.message} at {result.field_path}")
```

#### Performance Issues
```python
# Check cache statistics
cache_stats = engine.get_cache_stats()
if cache_stats['hit_rate'] < 0.8:
    print("Consider increasing cache size or TTL")

# Monitor processing times
if cache_stats['average_render_time_ms'] > 100:
    print("Templates may be too complex - consider optimization")
```

#### Migration Problems
```python
# Check migration status
status = migration_manager.get_migration_status(templates)
print("Migration status:", status)

# Validate before migration
for template in templates:
    needs_migration = await check_migration_needed(template)
    if needs_migration:
        print(f"Template needs migration: {template.get('_metadata', {}).get('template_version')}")
```

## Contributing

### Development Setup
1. Clone the repository
2. Install development dependencies: `pip install -r requirements-dev.txt`
3. Run tests: `pytest tests/`
4. Check code quality: `flake8 app/tenancy/fixtures/templates/`

### Adding New Features
1. Create feature branch
2. Implement with comprehensive tests
3. Update documentation
4. Submit pull request

### Testing
```bash
# Run all tests
pytest tests/tenancy/fixtures/templates/

# Run specific test categories
pytest tests/tenancy/fixtures/templates/test_engine.py
pytest tests/tenancy/fixtures/templates/test_validators.py
pytest tests/tenancy/fixtures/templates/test_generators.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Documentation: [Internal Wiki](https://wiki.company.com/spotify-ai-agent)
- Issues: [GitHub Issues](https://github.com/company/spotify-ai-agent/issues)
- Slack: #spotify-ai-agent-support
- Email: ai-agent-support@company.com
