# 🎵 Enterprise API Core Tests - Spotify AI Agent

## 📋 Overview

This package contains a comprehensive enterprise-grade test suite for the API Core module of the Spotify AI Agent. These tests cover all critical aspects of the system with advanced testing patterns and industrial validation.

## 🏗️ Test Architecture

### Test Structure

```
tests_backend/app/api/core/
├── __init__.py                 # Test package
├── conftest.py                # Advanced pytest configuration
├── README.md                  # This documentation (EN)
├── README.fr.md               # French documentation
├── README.de.md               # German documentation
├── test_config.py             # Configuration tests
├── test_context.py            # Request context tests
├── test_exceptions.py         # Error handling tests
├── test_factory.py            # Factory and DI tests
├── test_monitoring.py         # Monitoring tests
├── test_response.py           # Response formatting tests
└── test_integration.py        # Complete integration tests
```

### Test Categories

| Category | Description | Files |
|----------|-------------|-------|
| **Unit** | Unit tests for each component | All test_*.py files |
| **Integration** | Integration tests between components | test_integration.py |
| **Performance** | Performance tests and benchmarks | @pytest.mark.performance sections |
| **Security** | Security and vulnerability tests | @pytest.mark.security sections |
| **E2E** | Complete end-to-end tests | @pytest.mark.e2e sections |

## 🚀 Running Tests

### Prerequisites

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Specific test dependencies
pip install pytest pytest-asyncio pytest-benchmark pytest-mock pytest-cov
```

### Basic Commands

```bash
# All core module tests
pytest tests_backend/app/api/core/

# Specific tests
pytest tests_backend/app/api/core/test_config.py
pytest tests_backend/app/api/core/test_integration.py

# Tests by category
pytest -m unit                    # Unit tests
pytest -m integration            # Integration tests
pytest -m performance           # Performance tests
pytest -m security              # Security tests
```

### Advanced Options

```bash
# Tests with coverage
pytest tests_backend/app/api/core/ --cov=app.api.core --cov-report=html

# Tests with benchmarks
pytest tests_backend/app/api/core/ --benchmark-only

# Parallel tests
pytest tests_backend/app/api/core/ -n auto

# Verbose tests
pytest tests_backend/app/api/core/ -v -s

# Fast tests (without slow ones)
pytest tests_backend/app/api/core/ -m "not slow"
```

## 📊 Test Coverage

### Tested Modules

| Module | Coverage | Tests | Description |
|--------|----------|-------|-------------|
| **config.py** | 95%+ | 200+ | Configuration management |
| **context.py** | 95%+ | 180+ | Request context handling |
| **factory.py** | 95%+ | 220+ | Dependency injection |
| **exceptions.py** | 95%+ | 190+ | Exception management |
| **response.py** | 95%+ | 210+ | Response formatting |
| **monitoring.py** | 95%+ | 250+ | Monitoring & metrics |

### Test Types by Module

#### test_config.py
- ✅ Multi-environment configuration validation
- ✅ Configuration factory tests
- ✅ Hot reload tests
- ✅ Schema validation tests
- ✅ Configuration security tests
- ✅ Loading performance tests

#### test_context.py
- ✅ Thread-safe request context tests
- ✅ Asynchronous isolation tests
- ✅ Context middleware tests
- ✅ Concurrent performance tests
- ✅ Context propagation tests
- ✅ Context cleanup tests

#### test_factory.py
- ✅ Singleton and transient pattern tests
- ✅ Dependency injection tests
- ✅ Component lifecycle tests
- ✅ Dependency resolution tests
- ✅ Factory configuration tests
- ✅ Injection performance tests

#### test_exceptions.py
- ✅ Exception hierarchy tests
- ✅ Error handler tests
- ✅ Error formatting tests
- ✅ Error security tests
- ✅ Error correlation tests
- ✅ Handling performance tests

#### test_response.py
- ✅ Response formatting tests
- ✅ JSON serialization tests
- ✅ Response metadata tests
- ✅ Pagination tests
- ✅ Response validation tests
- ✅ Formatting performance tests

#### test_monitoring.py
- ✅ Metrics collection tests
- ✅ Health check tests
- ✅ System alert tests
- ✅ Performance monitoring tests
- ✅ Prometheus metrics tests
- ✅ Data aggregation tests

#### test_integration.py
- ✅ Complete integration tests
- ✅ End-to-end flow tests
- ✅ Load and stress tests
- ✅ Error recovery tests
- ✅ Data consistency tests
- ✅ Global performance tests

## 🔧 Test Configuration

### conftest.py

The `conftest.py` file provides:

- **Global fixtures**: Test configuration, clean environment
- **Service mocks**: External services, databases
- **Performance profilers**: Memory, CPU, response time
- **Pytest configuration**: Marks, hooks, test collection
- **Test utilities**: Custom assertions, helpers

### Main Fixtures

```python
# Test configuration
@pytest.fixture
def test_config():
    """Complete configuration for tests"""

# Clean environment
@pytest.fixture
def clean_environment():
    """Isolated environment for each test"""

# Service mocks
@pytest.fixture
def mock_external_services():
    """Mock all external services"""

# Performance profiling
@pytest.fixture
def memory_profiler():
    """Memory profiler for performance tests"""
```

### Test Marks

```python
@pytest.mark.unit          # Unit tests
@pytest.mark.integration   # Integration tests
@pytest.mark.performance   # Performance tests
@pytest.mark.security      # Security tests
@pytest.mark.e2e          # End-to-end tests
@pytest.mark.slow         # Slow tests
@pytest.mark.network      # Tests requiring network
```

## 📈 Metrics and Reporting

### Code Coverage

```bash
# Generate coverage report
pytest tests_backend/app/api/core/ --cov=app.api.core --cov-report=html --cov-report=term

# Open HTML report
open htmlcov/index.html
```

### Performance Benchmarks

```bash
# Run benchmarks
pytest tests_backend/app/api/core/ --benchmark-only --benchmark-save=core_benchmarks

# Compare benchmarks
pytest-benchmark compare core_benchmarks
```

### Test Reports

```bash
# JUnit XML report
pytest tests_backend/app/api/core/ --junitxml=reports/junit.xml

# HTML report
pytest tests_backend/app/api/core/ --html=reports/report.html --self-contained-html
```

## 🛡️ Security Tests

### Tested Vulnerabilities

- **Code injection**: SQL, NoSQL, commands
- **XSS**: Reflected, stored, DOM-based
- **Directory traversal**: Path traversal, file inclusion
- **Denial of service**: Resource exhaustion, infinite loops
- **Information disclosure**: Stack traces, detailed errors
- **Authentication**: Bypass, privilege escalation

### Security Tools

```python
# Automated security tests
@pytest.mark.security
def test_xss_prevention():
    """Test XSS prevention in responses"""

@pytest.mark.security
def test_sql_injection_protection():
    """Test SQL injection protection"""
```

## ⚡ Performance Tests

### Monitored Metrics

- **Response time**: < 100ms for simple operations
- **Memory**: < 50MB per request
- **CPU**: < 50% peak usage
- **Concurrency**: 100+ simultaneous requests
- **Throughput**: 1000+ requests/second

### Benchmarks

```python
# Automatic benchmark
def test_config_loading_performance(benchmark):
    """Configuration loading benchmark"""
    result = benchmark(load_config)
    assert result is not None

# Load test
@pytest.mark.performance
def test_concurrent_requests():
    """Test 100 concurrent requests"""
```

## 🚨 Alerts and Monitoring

### Alert Thresholds

- **Response time**: > 1000ms
- **Error rate**: > 5%
- **CPU usage**: > 80%
- **Memory usage**: > 80%
- **Test failures**: > 1%

### Notifications

Tests can be configured to send notifications on critical failures:

```bash
# Tests with notification
pytest tests_backend/app/api/core/ --notify-on-failure
```

## 🔍 Debug and Troubleshooting

### Test Debugging

```bash
# Debug mode with breakpoints
pytest tests_backend/app/api/core/ --pdb

# Specific tests with logs
pytest tests_backend/app/api/core/test_config.py::TestAPIConfig::test_config_loading -s -v

# Detailed profiling
pytest tests_backend/app/api/core/ --profile
```

### Common Issues

| Issue | Solution |
|-------|----------|
| Slow tests | Use `pytest -m "not slow"` |
| Intermittent failures | Check cleanup fixtures |
| Memory errors | Increase limits or optimize |
| Timeouts | Adjust timeouts in conftest.py |

## 📝 Contributing

### Adding New Tests

1. **Choose appropriate file** based on tested component
2. **Use existing fixtures** from conftest.py
3. **Add appropriate marks** (@pytest.mark.*)
4. **Document the test** with clear docstring
5. **Check coverage** with --cov

### Code Standards

```python
# Test template
@pytest.mark.unit
def test_feature_name(clean_environment, test_config):
    """Clear and concise test description
    
    This function tests [functionality] with [conditions]
    and verifies that [expected result].
    """
    # Arrange
    setup_data = create_test_data()
    
    # Act
    result = function_under_test(setup_data)
    
    # Assert
    assert result.is_valid()
    assert result.data == expected_data
```

## 📚 Additional Documentation

- [Configuration Guide](../../../config/README.md)
- [API Core Architecture](../README.md)
- [Testing Standards](../../../../docs/testing-standards.md)
- [Performance Guide](../../../../docs/performance-guide.md)

---

**Developed by Fahed Mlaiel** - Enterprise Testing Expert  
Version 2.0.0 - Ultra-Advanced Tests for Spotify AI Agent