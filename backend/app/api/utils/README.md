# 🎵 Spotify AI Agent - Enterprise Utilities Package

## 🏆 Overview

The **Enterprise Utilities Package** is a comprehensive suite of production-ready utilities designed for the Spotify AI Agent backend. Built with enterprise-grade standards, this package provides essential tools for data transformation, security, performance monitoring, and much more.

## 🎯 Key Features

### 🔄 **Data Transformation & Validation**
- Advanced data structure validation with schema support
- Secure input sanitization with XSS protection
- Deep merging with configurable strategies
- JSON serialization for complex data types
- Dictionary manipulation and filtering utilities

### 📝 **String Processing & Text Analytics**
- Intelligent multilingual slugification
- Case conversion (camel, snake, pascal)
- Pattern extraction (emails, URLs, phone numbers)
- Secure hashing and random generation
- Sensitive data masking and text statistics

### ⏰ **DateTime Management**
- Multi-format automatic parsing
- Timezone handling with zoneinfo
- Date humanization ("2 hours ago")
- Business calendar with holidays
- Date range validation and business hours

### 🔐 **Cryptographic Security**
- AES-256 encryption (GCM/CBC modes)
- RSA-2048 asymmetric encryption
- Secure password hashing (Argon2, bcrypt, scrypt)
- HMAC and digital signatures
- Cryptographically secure token generation

### 📁 **File Management**
- Secure file upload with MIME validation
- Compression/decompression (gzip, bz2, zip, tar)
- Audio/image metadata extraction with EXIF
- Large file streaming with chunking
- Automatic cleanup and space management

### ⚡ **Performance Monitoring**
- Real-time metrics collection
- Detailed profiling with cProfile integration
- High-performance caching with TTL
- Bottleneck detection and analysis
- Rate limiting and memory optimization

### 🌐 **Network Utilities**
- Enterprise-grade asynchronous HTTP client
- Automatic health checks and monitoring
- Advanced URL/domain/IP validation
- DNS resolution and SSL certificate validation
- Real-time connectivity monitoring

### ✅ **Validation Framework**
- Email validation with deliverability checks
- International phone number validation with carrier info
- Password strength scoring with security recommendations
- Business-specific metadata validation
- Audio/image file validation with security

### 🎨 **Multi-Format Export & Templates**
- Export to JSON, XML, CSV, YAML, Markdown
- Dynamic Jinja2 template system
- Currency, percentage, and duration formatting
- Text table generation and report creation
- Code beautification and data presentation

## 🚀 Quick Start

### Installation

```python
# Import the complete utils package
from backend.app.api.utils import *
```

### Basic Usage Examples

#### Data Transformation
```python
# Validate and sanitize user input
validated_data = validate_data_structure(user_input, schema)
sanitized = sanitize_input(validated_data)

# Deep merge configurations
merged_config = deep_merge(default_config, user_config)

# Flatten nested dictionaries
flat_data = flatten_dict(nested_data)
```

#### Cryptographic Operations
```python
# Secure encryption
encryptor = SecureEncryption()
encrypted_data = encryptor.encrypt_json(sensitive_data)
decrypted_data = encryptor.decrypt_json(encrypted_data)

# Password hashing
password_hash = hash_password(user_password, 'argon2')
is_valid = verify_password(user_password, password_hash, 'argon2')

# Token generation
api_key = generate_api_key('spotify', 32)
session_id = generate_session_id()
```

#### Performance Monitoring
```python
# Monitor function performance
@monitor_performance()
@memoize(maxsize=256, ttl=3600)
async def process_audio_file(file_path: str):
    return await heavy_audio_processing(file_path)

# Benchmark functions
@benchmark(iterations=1000)
def data_processing_function(data):
    return transform_data(data)
```

#### Network Operations
```python
# Enterprise HTTP client
async with EnterpriseHttpClient() as client:
    # Health check
    health = await check_http_health('https://api.spotify.com/health')
    
    # API calls with automatic retry
    response = await client.get_json('https://api.spotify.com/v1/tracks')
    
    # POST with JSON data
    result = await client.post_json(api_url, payload_data)
```

#### File Management
```python
# Secure file upload
upload_manager = FileUploadManager('/uploads')
file_info = upload_manager.save_uploaded_file(file_data, 'audio.mp3')

# Get file metadata
metadata = get_file_metadata('/path/to/audio.mp3')
print(f"Duration: {metadata.get('duration')} seconds")

# Compress files
compressed_path = compress_file('/path/to/large_file.txt', compression='gzip')
```

#### Validation
```python
# Email validation with deliverability
email_result = validate_email('user@example.com', check_deliverability=True)

# Phone number validation
phone_result = validate_phone('+33123456789', region='FR')

# Password strength validation
password_result = validate_user_password('MySecureP@ssw0rd!')
print(f"Strength: {password_result['strength']}")

# Audio metadata validation
metadata_result = validate_audio_metadata({
    'title': 'Song Title',
    'artist': 'Artist Name',
    'duration': 240.5
})
```

## 📚 Module Documentation

### Core Modules

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `data_transform` | Data transformation and validation | `transform_data`, `validate_data_structure`, `deep_merge` |
| `string_utils` | String processing and text analytics | `slugify`, `extract_emails`, `mask_sensitive_data` |
| `datetime_utils` | Date and time management | `format_datetime`, `humanize_datetime`, `convert_timezone` |
| `crypto_utils` | Cryptographic operations | `SecureEncryption`, `hash_password`, `generate_secure_token` |
| `file_utils` | File management and processing | `FileUploadManager`, `get_file_metadata`, `compress_file` |
| `performance_utils` | Performance monitoring and optimization | `monitor_performance`, `PerformanceMonitor`, `memoize` |
| `network_utils` | Network communication and validation | `EnterpriseHttpClient`, `check_http_health`, `validate_url` |
| `validators` | Data validation framework | `validate_email`, `validate_audio_metadata`, `DataValidator` |
| `formatters` | Multi-format export and templates | `format_json`, `TemplateFormatter`, `MultiFormatExporter` |

## 🛡️ Security Features

- **XSS Protection**: Built-in sanitization using `bleach`
- **Secure Random Generation**: Cryptographically secure tokens and keys
- **Constant Time Comparison**: Protection against timing attacks
- **Input Validation**: Comprehensive validation for all user inputs
- **File Security**: MIME type validation and secure upload handling

## 🚀 Performance Features

- **Intelligent Caching**: LRU cache with TTL support
- **Rate Limiting**: Distributed rate limiting for API protection
- **Memory Monitoring**: Real-time memory usage tracking
- **Profiling Integration**: Built-in performance profiling
- **Asynchronous Support**: Native async/await support throughout

## 🔧 Configuration

### Environment Variables

```bash
# Cache settings
CACHE_DEFAULT_TTL=3600
CACHE_MAX_SIZE=1000

# File upload settings
MAX_FILE_SIZE_MB=100
UPLOAD_DIRECTORY=/tmp/uploads

# Security settings
ENCRYPTION_KEY_LENGTH=32
TOKEN_EXPIRY_HOURS=24

# Performance settings
RATE_LIMIT_REQUESTS_PER_MINUTE=100
MONITORING_ENABLED=true
```

### Custom Configuration

```python
from backend.app.api.utils import NetworkConfig, PerformanceMonitor

# Configure network client
network_config = NetworkConfig(
    timeout=30.0,
    max_retries=3,
    verify_ssl=True
)

# Configure performance monitoring
perf_monitor = PerformanceMonitor(max_history=2000)
```

## 🧪 Testing

The utilities package includes comprehensive test coverage:

```bash
# Run all tests
pytest tests/

# Run specific module tests
pytest tests/test_crypto_utils.py
pytest tests/test_validators.py

# Run with coverage
pytest --cov=backend.app.api.utils tests/
```

## 📊 Monitoring & Metrics

### Performance Metrics

```python
from backend.app.api.utils import performance_monitor

# Get function statistics
stats = performance_monitor.get_stats('function_name')
print(f"Average execution time: {stats['avg']:.3f}s")
print(f"95th percentile: {stats['p95']:.3f}s")

# System monitoring
system_monitor = SystemMonitor()
system_monitor.start_monitoring()
current_metrics = system_monitor.get_current_metrics()
```

### Health Checks

```python
# Monitor endpoint health
connectivity_monitor = ConnectivityMonitor()
connectivity_monitor.add_endpoint('https://api.spotify.com')
await connectivity_monitor.start_monitoring()

# Get overall status
status = connectivity_monitor.get_overall_status()
print(f"Overall health: {status['overall_health']:.1f}%")
```

## 🤝 Contributing

### Development Guidelines

1. **Code Quality**: Follow PEP 8 and use type hints
2. **Security**: All inputs must be validated and sanitized
3. **Performance**: Include monitoring and optimization
4. **Documentation**: Comprehensive docstrings required
5. **Testing**: Minimum 90% test coverage

### Adding New Utilities

1. Create new module in appropriate category
2. Follow existing patterns and conventions
3. Add comprehensive tests
4. Update `__init__.py` exports
5. Document in README

## 📝 License

MIT License - see LICENSE file for details.

## 👨‍💻 Enterprise Team

**Developed by the Spotify AI Agent Enterprise Team**

- **Lead Developer & AI Architect**: Advanced system design and ML integration
- **Senior Backend Developer**: Core utilities and API design
- **Machine Learning Engineer**: ML-specific utilities and data processing
- **Database & Data Engineer**: Data transformation and storage utilities
- **Backend Security Specialist**: Cryptographic utilities and security features
- **Microservices Architect**: Network utilities and distributed systems

---

**Special attribution: Fahed Mlaiel** - Enterprise Architecture and Technical Excellence

## 🔗 Links

- [API Documentation](./docs/api.md)
- [Performance Guide](./docs/performance.md)
- [Security Guide](./docs/security.md)
- [Migration Guide](./docs/migration.md)

---

*Built with ❤️ for enterprise-grade applications*
