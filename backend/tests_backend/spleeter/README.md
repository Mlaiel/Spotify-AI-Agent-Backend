# 🎵 Spotify AI Agent - Spleeter Tests Complete Guide
# ================================================
# 
# Complete guide for running and maintaining
# tests for the enterprise Spleeter module.
#
# 🎖️ Developed by the enterprise expert team

# 🎵 Spleeter Tests - Complete Guide

Welcome to the complete test suite for the **Spleeter** module of the Spotify AI agent! 🚀

This documentation will guide you through installation, configuration, and execution of all available test types.

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [🛠️ Installation](#️-installation)
- [🚀 Quick Start](#-quick-start)
- [📊 Test Types](#-test-types)
- [⚙️ Configuration](#️-configuration)
- [🔧 Makefile Commands](#-makefile-commands)
- [🐍 Python Script](#-python-script)
- [📈 Code Coverage](#-code-coverage)
- [🏭 CI/CD](#-cicd)
- [🔍 Debugging](#-debugging)
- [📝 Contributing](#-contributing)

## 🎯 Overview

The Spleeter test suite includes **11 test modules** covering all system aspects:

### 📁 Test Structure

```
tests_backend/spleeter/
├── 📋 conftest.py              # Global configuration & fixtures
├── 🧪 test_core.py             # Core engine tests
├── 🤖 test_models.py           # Model management tests
├── 🎵 test_processor.py        # Audio processing tests
├── 💾 test_cache.py            # Cache system tests
├── 🔧 test_utils.py            # Utilities tests
├── 📊 test_monitoring.py       # Monitoring tests
├── ⚠️ test_exceptions.py       # Error handling tests
├── 🔗 test_integration.py      # Integration tests
├── ⚡ test_performance.py      # Performance tests
├── 🛠️ test_helpers.py          # Test utilities
├── 📋 Makefile                 # Make automation
├── 🐍 run_tests.sh             # Bash automation script
├── ⚙️ pyproject.toml           # Pytest configuration
└── 📖 README.en.md             # This documentation
```

### 🎖️ Test Coverage

- **Unit Tests**: Individual component validation
- **Integration Tests**: Complete workflow validation
- **Performance Tests**: Benchmarks and optimizations
- **Stress Tests**: High load validation
- **Security Tests**: Security control validation
- **Regression Tests**: Regression prevention

## 🛠️ Installation

### System Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3 python3-pip ffmpeg libsndfile1

# macOS
brew install python ffmpeg libsndfile

# Windows (with chocolatey)
choco install python ffmpeg
```

### Python Dependencies Installation

```bash
# Automatic installation via script
cd /workspaces/Achiri/spotify-ai-agent/backend/tests_backend/spleeter
chmod +x run_tests.sh
./run_tests.sh --install-deps

# Or manual installation
pip install pytest pytest-asyncio pytest-cov pytest-mock pytest-benchmark
pip install black flake8 mypy isort coverage[toml]
pip install pytest-html pytest-xdist pytest-timeout
```

### Installation via Makefile

```bash
# Complete dependency installation
make install-deps

# Configuration verification
make show-config
```

## 🚀 Quick Start

### Quick Tests (< 1 minute)

```bash
# Via Makefile
make test-fast

# Via bash script
./run_tests.sh --smoke

# Via direct pytest
pytest -m "not slow and not performance and not stress" --timeout=30
```

### Complete Tests

```bash
# Complete CI/CD pipeline
./run_tests.sh --ci

# All tests via Makefile
make test-all

# Tests with coverage
make coverage-html
```

### Specific Tests

```bash
# Tests by module
make test-core          # Core engine
make test-models        # Model management
make test-processor     # Audio processing
make test-cache         # Cache system

# Tests by type
make test-unit          # Unit tests
make test-integration   # Integration tests
make test-performance   # Performance tests
```

## 📊 Test Types

### 🧪 Unit Tests

Individual validation of each component:

```bash
# Unit test execution
pytest test_core.py test_models.py test_processor.py \
       test_cache.py test_utils.py test_monitoring.py test_exceptions.py \
       -v -m "not slow"
```

**Coverage**:
- ✅ Engine initialization
- ✅ Basic audio separation
- ✅ ML model management
- ✅ Multi-level cache
- ✅ Utilities and validation
- ✅ System monitoring
- ✅ Exception handling

### 🔗 Integration Tests

End-to-end workflow validation:

```bash
# Integration tests
pytest test_integration.py -v -m "integration" --timeout=120
```

**Scenarios**:
- ✅ Complete separation with cache
- ✅ Batch processing with monitoring
- ✅ Robust error handling
- ✅ Concurrent workflows

### ⚡ Performance Tests

Benchmarks and optimizations:

```bash
# Performance tests
pytest test_performance.py -v -m "performance" --benchmark-only
```

**Metrics**:
- ⏱️ Audio separation time
- 📊 Memory usage
- 🚀 Batch throughput
- 💾 Cache performance
- 📈 Monitoring overhead

### 💪 Stress Tests

High load validation:

```bash
# Stress tests
pytest test_performance.py -v -m "stress" --timeout=600
```

**Scenarios**:
- 🔥 Maximum CPU load
- 💧 Memory stress
- 🌊 Concurrent processing
- 🎯 Long duration stability

## ⚙️ Configuration

### Pytest Configuration (pyproject.toml)

Centralized configuration in `pyproject.toml` includes:

```toml
[tool.pytest.ini_options]
markers = [
    "slow: slow tests",
    "fast: fast tests", 
    "unit: unit tests",
    "integration: integration tests",
    "performance: performance tests",
    "stress: stress tests"
]

addopts = [
    "-v",
    "--tb=short",
    "--strict-markers",
    "--durations=10",
    "--maxfail=3"
]
```

### Environment Variables

```bash
export SPLEETER_TEST_MODE="true"
export SPLEETER_LOG_LEVEL="DEBUG"
export SPLEETER_CACHE_DISABLED="true"
export COVERAGE_MIN="85"
```

### Coverage Configuration

```toml
[tool.coverage.run]
source = ["backend/spleeter"]
branch = true
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
fail_under = 85
show_missing = true
```

## 🔧 Makefile Commands

### Main Commands

```bash
make help              # Display complete help
make test              # Basic tests (fast)
make test-all          # All tests (complete)
make coverage          # Console coverage analysis
make coverage-html     # HTML coverage report
make lint              # Linting checks
make format            # Automatic formatting
make clean             # Cleanup temporary files
```

### Tests by Module

```bash
make test-core         # Core engine tests
make test-models       # Model management tests  
make test-processor    # Audio processing tests
make test-cache        # Cache system tests
make test-utils        # Utilities tests
make test-monitoring   # Monitoring tests
make test-exceptions   # Error handling tests
```

### Tests by Type

```bash
make test-unit         # Unit tests
make test-integration  # Integration tests
make test-performance  # Performance tests
make test-stress       # Stress tests
make test-fast         # Fast tests only
make test-slow         # Slow tests only
```

### Advanced Utilities

```bash
make test-parallel     # Parallel tests
make test-report       # Tests with HTML report
make benchmark         # Specific benchmarks
make test-security     # Security tests
make test-regression   # Regression tests
make smoke-test        # Smoke tests
make acceptance-test   # Acceptance tests
```

## 🐍 Python Script

### Script Usage

```bash
# Make script executable
chmod +x run_tests.sh

# Complete help
./run_tests.sh --help

# Complete CI/CD pipeline
./run_tests.sh --ci

# Specific tests
./run_tests.sh --unit
./run_tests.sh --integration
./run_tests.sh --performance
./run_tests.sh --stress
```

### Available Options

| Option | Description |
|--------|-------------|
| `--ci` | Complete CI/CD pipeline |
| `--unit` | Unit tests only |
| `--integration` | Integration tests only |
| `--performance` | Performance tests only |
| `--stress` | Stress tests only |
| `--coverage` | Coverage analysis only |
| `--quality` | Quality checks only |
| `--smoke` | Smoke tests (quick check) |
| `--install-deps` | Install dependencies |
| `--cleanup` | Cleanup temporary files |

### Script Configuration

```bash
# Environment variables for customization
export COVERAGE_MIN=90           # Minimum coverage
export TIMEOUT_UNIT=45          # Unit test timeout
export TIMEOUT_INTEGRATION=180  # Integration test timeout
export TIMEOUT_PERFORMANCE=450  # Performance test timeout
export TIMEOUT_STRESS=900       # Stress test timeout
```

## 📈 Code Coverage

### Report Generation

```bash
# Console report
make coverage

# Detailed HTML report
make coverage-html
# Automatic opening: coverage_html/index.html

# XML report (for CI/CD)
make coverage-xml
# Generated file: coverage.xml
```

### Coverage Thresholds

- **Global Coverage**: ≥ 85%
- **Per-File Coverage**: ≥ 80%
- **Branch Coverage**: ≥ 75%

### Coverage Analysis

```bash
# Differential coverage
make diff-coverage

# Coverage with context
pytest --cov=../../spleeter \
       --cov-context=test \
       --cov-branch
```

## 🏭 CI/CD

### GitHub Actions

The GitHub Actions workflow includes:

```yaml
# .github/workflows/spleeter-tests.yml
name: 🎵 Spleeter Tests CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    - cron: '0 2 * * *'  # Daily tests

jobs:
  smoke-tests:    # Quick verification tests
  unit-tests:     # Multi-platform unit tests
  integration-tests: # Integration tests with services
  performance-tests: # Performance tests and benchmarks
  stress-tests:   # Load and stress tests
  security-tests: # Security tests
  code-quality:   # Code quality checks
  final-report:   # Consolidated report
```

### Local Configuration

```bash
# Local CI/CD pipeline simulation
./run_tests.sh --ci

# Pre-commit verification
make pre-commit

# Complete validation
make check-all
```

### Continuous Integration

```bash
# Fast tests for development
make test-fast

# Complete tests for release
make test-all coverage-html

# Pre-commit verification
make pre-commit
```

## 🔍 Debugging

### Specific Test Debug

```bash
# Verbose debug mode
make test-debug TEST=test_core.py::test_engine_initialization

# Debug with pdb
pytest test_core.py::test_engine_initialization -vvv -s --pdb

# Performance profiling
make profile-tests
```

### Debug Logs

```bash
# Detailed log activation
export SPLEETER_LOG_LEVEL="DEBUG"
pytest test_core.py -vvv -s --log-cli-level=DEBUG
```

### Watch Mode Tests

```bash
# Watch mode for development
make watch-tests

# Continuous tests with pytest-watch
pip install pytest-watch
ptw -- -m "not slow"
```

### Failure Debugging

```bash
# Tests with extended failure information
pytest --tb=long --capture=no

# Failure report generation
pytest --html=failure_report.html --self-contained-html
```

## 📊 Metrics and Monitoring

### Collected Metrics

- **Performance**: Execution time, throughput
- **Resources**: CPU, memory, I/O
- **Quality**: Coverage, complexity
- **Reliability**: Success rate, stability

### Available Reports

```bash
# Complete HTML report
make test-report

# Performance metrics
make test-metrics

# JSON results export
make test-export
```

### Continuous Monitoring

```bash
# Tests with resource monitoring
pytest test_performance.py::TestResourceMonitoring -v

# Memory tracking
pytest --memray

# CPU profiling
pytest --profile
```

## 🔧 Maintenance

### Dependency Updates

```bash
# Version checking
pip list --outdated

# Secure update
pip install --upgrade -r requirements/testing.txt

# Compatibility verification
./run_tests.sh --smoke
```

### Periodic Cleanup

```bash
# Complete cleanup
make clean

# Cache removal
rm -rf .pytest_cache __pycache__ .coverage

# Artifact cleanup
rm -rf coverage_html test_report.html
```

### Test Optimization

```bash
# Parallel tests
make test-parallel

# Fixture optimization
pytest --setup-show

# Duration analysis
pytest --durations=0
```

## 🔐 Security

### Security Tests

```bash
# Specific security tests
make test-security

# Vulnerability scan
bandit -r ../../spleeter/

# Dependency verification
safety check
```

### Input Validation

```bash
# Validation tests
pytest -k "validation or sanitize" test_utils.py

# Injection tests
pytest -k "security" test_exceptions.py
```

## 📝 Contributing

### Adding New Tests

1. **Create test file** in appropriate directory
2. **Use fixtures** defined in `conftest.py`
3. **Add appropriate markers** (`@pytest.mark.unit`, etc.)
4. **Document tests** with clear docstrings
5. **Verify coverage** with `make coverage`

### Test Structure

```python
import pytest
from unittest.mock import Mock, patch

class TestNewFeature:
    """Tests for new functionality."""
    
    @pytest.mark.unit
    @pytest.mark.fast
    def test_basic_functionality(self, mock_engine):
        """Test basic functionality."""
        # Arrange
        input_data = "test_input"
        expected = "expected_output"
        
        # Act
        result = mock_engine.process(input_data)
        
        # Assert
        assert result == expected
    
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_integration_workflow(self):
        """Complete integration test."""
        # Integration test with real dependencies
        pass
```

### Test Guidelines

- **Naming**: `test_<functionality>_<scenario>`
- **Isolation**: Each test must be independent
- **Mocking**: Use mocks for external dependencies
- **Assertions**: Clear and specific assertions
- **Documentation**: Explanatory docstrings
- **Markers**: Appropriate pytest marker usage

### Pre-Commit Validation

```bash
# Complete validation before commit
make pre-commit

# Individual checks
make format    # Code formatting
make lint      # Linting checks
make type-check # Type checking
make test-fast # Fast tests
```

## 🎯 Quality Objectives

### Performance Targets

- **Unit Tests**: < 30 seconds total
- **Integration Tests**: < 2 minutes total  
- **Performance Tests**: < 5 minutes total
- **Audio Separation**: < 5 seconds/file

### Coverage Targets

- **Global Coverage**: ≥ 85%
- **Branch Coverage**: ≥ 75%
- **Critical Modules**: ≥ 90%

### Reliability Targets

- **Success Rate**: ≥ 99%
- **Stability**: 0 flaky tests
- **Reproducibility**: 100%

## 🚀 Performance and Optimizations

### Test Optimizations

```bash
# Parallel tests
pytest -n auto

# Result caching
pytest --cache-clear  # Reset cache
pytest --lf          # Last failed only
pytest --ff          # Failed first

# Memory optimization
pytest --maxfail=1 --tb=no
```

### Performance Monitoring

```bash
# Detailed profiling
pytest --profile --profile-svg

# Memory monitoring
pytest --memray --memray-bin-path=memory_profile.bin

# Bottleneck analysis
pytest --durations=20
```

## 📞 Support and Help

### Resources

- **Documentation**: This README and docstrings
- **Examples**: Existing test files
- **Configuration**: `pyproject.toml`, `conftest.py`
- **Scripts**: `Makefile`, `run_tests.sh`

### Help Commands

```bash
# Makefile help
make help

# Bash script help
./run_tests.sh --help

# Pytest help
pytest --help

# Active configuration
make show-config
```

### Common Troubleshooting

**Failing tests**:
```bash
# Verbose debug mode
pytest -vvv -s --tb=long

# Isolated tests
pytest test_file.py::test_function -v
```

**Slow performance**:
```bash
# Profiling
make profile-tests

# Fast tests only
make test-fast
```

**Coverage issues**:
```bash
# Detailed report
make coverage-html

# Missing files
coverage report --show-missing
```

---

## 🎉 Conclusion

This complete test suite ensures the quality, performance, and reliability of the Spleeter module. Use this documentation as a reference for all your testing needs!

**Happy testing! 🚀🎵**

---

*Developed with ❤️ by the enterprise expert team for the Spotify AI agent*
