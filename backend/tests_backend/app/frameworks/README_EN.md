# 🧪 Frameworks Testing Suite - User Guide
# =========================================

## 📁 Test Structure

```
backend/tests_backend/app/frameworks/
├── __init__.py                 # Global pytest configuration
├── conftest.py                # Test fixtures and utilities  
├── run_tests.sh              # Automated test execution script
├── test_core.py              # Framework Orchestrator tests (400+ lines)
├── test_hybrid_backend.py    # Django/FastAPI hybrid tests (600+ lines)
├── test_ml_frameworks.py     # Spotify ML/AI tests (800+ lines)
├── test_security.py          # Enterprise security tests (700+ lines)
├── test_monitoring.py        # Monitoring/observability tests (800+ lines)
├── test_microservices.py     # Microservices architecture tests (900+ lines)
└── test_integration.py       # End-to-end integration tests (600+ lines)
```

## 🚀 Quick Start

### Tests by Mode
```bash
# Unit tests only
./run_tests.sh unit

# Integration tests
./run_tests.sh integration

# Performance tests
./run_tests.sh performance

# All tests
./run_tests.sh all

# Tests with coverage
./run_tests.sh coverage

# Fast tests (development)
./run_tests.sh fast
```

### Tests by Framework
```bash
# Core Framework Orchestrator
./run_tests.sh framework core

# Hybrid Backend Framework
./run_tests.sh framework hybrid

# ML/AI Framework
./run_tests.sh framework ml

# Security Framework
./run_tests.sh framework security

# Monitoring Framework
./run_tests.sh framework monitoring

# Microservices Framework
./run_tests.sh framework microservices
```

### CI/CD Tests
```bash
# Optimized CI mode
CI=true ./run_tests.sh ci

# Fast tests for development
FAST_TESTS=true ./run_tests.sh fast

# Skip external tests
SKIP_EXTERNAL_TESTS=true ./run_tests.sh all
```

## 📊 Test Coverage

### By Framework

| Framework | Tests | Coverage | Features |
|-----------|-------|----------|----------|
| **Core Orchestrator** | 15+ tests | >90% | Circuit breaker, health check, lifecycle |
| **Hybrid Backend** | 20+ tests | >85% | Django/FastAPI, middleware, sessions |
| **ML Frameworks** | 25+ tests | >90% | Spotify ML, MFCC, BERT, model versioning |
| **Security** | 25+ tests | >95% | JWT/OAuth2, crypto, rate limiting, audit |
| **Monitoring** | 30+ tests | >85% | Prometheus, Jaeger, alerting, health |
| **Microservices** | 35+ tests | >90% | Service mesh, discovery, load balancing |
| **Integration** | 20+ tests | >80% | End-to-end, business scenarios |

### Test Types

#### 🔬 Unit Tests
- Isolated component testing
- External dependency mocking
- Business logic validation
- Edge case testing

#### 🔗 Integration Tests
- Inter-framework communication
- End-to-end workflows
- Real business scenarios
- Resilience testing

#### ⚡ Performance Tests
- Latency benchmarks
- Load testing
- Memory optimization
- Horizontal scalability

#### 🛡️ Security Tests
- Authentication/authorization
- Encryption/decryption
- Rate limiting
- Security auditing

## 🔧 Configuration

### Environment Variables
```bash
# Development mode
export TEST_MODE="development"
export FAST_TESTS="true"
export SKIP_EXTERNAL_TESTS="true"

# CI/CD mode
export CI="true"
export TEST_WORKERS="4"
export COVERAGE_THRESHOLD="85"

# External services
export REDIS_URL="redis://localhost:6379/15"
export DATABASE_URL="sqlite:///test_frameworks.db"
```

### pytest Markers
```python
@pytest.mark.unit          # Unit test
@pytest.mark.integration   # Integration test
@pytest.mark.performance   # Performance test
@pytest.mark.slow          # Slow test (>1s)
@pytest.mark.external      # Requires external services
@pytest.mark.security      # Security test
@pytest.mark.ml            # ML/AI test
```

## 📈 Metrics and Reports

### Generated Reports
```
coverage_reports/
├── html/                   # Interactive HTML report
├── coverage.json          # JSON data
└── coverage.xml           # XML format (CI)

test_logs/
├── unit_tests.log         # Unit tests logs
├── integration_tests.log  # Integration logs  
├── performance_tests.log  # Performance logs
├── benchmark_results.json # Benchmark results
└── test_report.txt       # Final report
```

### Key Metrics
- **Global coverage**: >85%
- **Execution time**: <5min (fast), <15min (all)
- **Success rate**: >95%
- **Performance**: <100ms average latency

## 🎯 Specialized Test Scenarios

### 🤖 ML Engineer Expert - ML Tests
```python
# Spotify model tests
test_spotify_recommendation_model()
test_audio_analysis_mfcc_extraction()
test_bert_sentiment_analysis()
test_model_versioning_mlops()
```

### 🔒 Security Expert - Security Tests
```python
# Authentication tests
test_jwt_token_lifecycle()
test_oauth2_spotify_integration()
test_rate_limiting_redis()
test_encryption_fernet_rsa()
```

### 📊 DevOps Expert - Monitoring Tests
```python
# Observability tests
test_prometheus_metrics_collection()
test_jaeger_distributed_tracing()
test_intelligent_alerting()
test_health_monitoring_concurrent()
```

### 🏗️ Architecture Expert - Microservices Tests
```python
# Distributed architecture tests
test_service_discovery_multi_backend()
test_load_balancing_weighted()
test_service_mesh_mtls()
test_message_broker_patterns()
```

## 🚀 Recommended Workflows

### Local Development
```bash
# 1. Fast tests during development
./run_tests.sh fast

# 2. Framework-specific tests
./run_tests.sh framework [framework_name]

# 3. Complete tests before commit
./run_tests.sh coverage
```

### CI/CD Pipeline
```bash
# 1. Fast tests (PR validation)
CI=true ./run_tests.sh fast

# 2. Complete tests (main merge)
CI=true ./run_tests.sh ci

# 3. Performance tests (release)
./run_tests.sh performance
```

### Debug and Troubleshooting
```bash
# Verbose tests with details
./run_tests.sh unit --verbose -s

# Specific test with debug
pytest test_core.py::TestFrameworkOrchestrator::test_lifecycle -vvv -s

# Tests with profiling
./run_tests.sh performance --benchmark-json=results.json
```

## 🔍 Available Fixtures and Mocks

### Global Fixtures (conftest.py)
- `clean_frameworks`: Cleanup between tests
- `mock_redis`: Mocked Redis
- `mock_consul`: Mocked Consul  
- `mock_http_client`: Mocked HTTP client
- `sample_audio_data`: Test audio data
- `test_user_data`: Test user data

### Specialized Mocks
- `MockSpotifyAPI`: Mocked Spotify API
- `MockPrometheusRegistry`: Mocked Prometheus registry
- `MockJaegerTracer`: Mocked Jaeger tracer
- `MockMLModel`: Mocked ML models

## 📚 Resources and Documentation

### Internal Documentation
- Each test file contains its documentation
- Usage examples in docstrings
- Recommended test patterns

### Quality Standards
- Minimum coverage: 85%
- Atomic and idempotent tests
- Descriptive and consistent naming
- Complex case documentation

## 🛠️ Advanced Features

### Test Automation
- Intelligent test script with multiple modes
- Automatic dependency checking
- Environment setup and cleanup
- Comprehensive error handling

### Performance Monitoring
- Benchmark integration with pytest-benchmark
- Memory usage tracking
- Execution time optimization
- Scalability testing

### CI/CD Integration
- XML/JSON report generation
- Parallel test execution
- Retry mechanisms for flaky tests
- Coverage threshold enforcement

### Mock Infrastructure
- Complete external service mocking
- Realistic test data generation
- Network isolation
- State management between tests

## 🎵 Spotify AI Agent Specifics

### Audio Processing Tests
- MFCC feature extraction validation
- Audio format compatibility testing
- Real-time processing performance
- Audio quality metrics

### Recommendation Engine Tests
- Algorithm accuracy validation
- Cold start problem handling
- Personalization effectiveness
- Real-time recommendation generation

### Spotify API Integration Tests
- OAuth2 flow validation
- API rate limiting handling
- Data synchronization testing
- Error recovery mechanisms

---

**Developed by the Spotify AI Agent Expert Team** 🎵🤖

## Contributing

### Adding New Tests
1. Follow the existing test structure
2. Use appropriate pytest markers
3. Include comprehensive documentation
4. Maintain >85% coverage
5. Add performance benchmarks for critical paths

### Test Naming Convention
- `test_[component]_[functionality]_[scenario]()`
- Use descriptive names that explain the test purpose
- Group related tests in classes
- Prefix integration tests with `test_integration_`

### Best Practices
- Keep tests atomic and independent
- Use factories for test data generation
- Mock external dependencies
- Test both success and failure scenarios
- Include performance assertions for critical operations
