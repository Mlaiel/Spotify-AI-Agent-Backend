# 🎵 Spotify AI Agent - Advanced Spleeter Module

[![Enterprise Grade](https://img.shields.io/badge/Enterprise-Grade-gold.svg)](https://github.com/spotify-ai-agent)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🚀 Overview

**Advanced Enterprise Spleeter Module** - Industrial-grade audio separation engine built for high-performance, scalable music source separation. This module provides state-of-the-art AI-powered audio processing capabilities with enterprise features including multi-level caching, GPU optimization, batch processing, and comprehensive monitoring.

**🎖️ Architect & Lead Developer:** [Fahed Mlaiel](https://github.com/fahed-mlaiel)  
**🏢 Organization:** Enterprise AI Solutions Team

---

## ✨ Key Features

### 🎯 **Core Capabilities**
- **🤖 Advanced AI Models**: Support for 2-stems, 4-stems, and 5-stems separation
- **⚡ GPU Acceleration**: Optimized CUDA/TensorFlow GPU processing
- **🔄 Async Processing**: Non-blocking operations with asyncio
- **📦 Batch Operations**: High-throughput parallel processing
- **🎛️ Audio Preprocessing**: Advanced filtering, normalization, noise reduction

### 🏗️ **Enterprise Architecture**
- **💾 Multi-Level Caching**: Memory (L1) → Disk+SQLite (L2) → Redis (L3)
- **📊 Performance Monitoring**: Real-time metrics, health checks, alerting
- **🔧 Model Management**: Automatic download, validation, versioning
- **🛡️ Security & Validation**: Input sanitization, resource limits, error handling
- **📈 Scalability**: Microservices-ready, horizontal scaling support

### 🎵 **Audio Processing Excellence**
- **📻 Format Support**: WAV, FLAC, MP3, OGG, M4A, AAC, WMA
- **🔊 Quality Options**: Up to 192kHz/32-bit processing
- **🎚️ Dynamic Processing**: Loudness normalization, silence detection
- **📋 Metadata Extraction**: Comprehensive audio analysis and tagging

---

## 🛠️ Installation

### Prerequisites
```bash
# Python 3.8+ required
python --version  # Should be 3.8+

# Install core dependencies
pip install tensorflow>=2.8.0
pip install librosa>=0.9.0
pip install soundfile>=0.10.0
pip install numpy>=1.21.0
pip install asyncio-mqtt
pip install aioredis
```

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/spotify-ai-agent/backend.git
cd backend/spleeter

# Install dependencies
pip install -r requirements.txt

# Initialize the module
python -c "from spleeter import SpleeterEngine; print('✅ Installation successful!')"
```

---

## 🚀 Quick Start

### Basic Usage
```python
import asyncio
from spleeter import SpleeterEngine

async def separate_audio():
    # Initialize engine
    engine = SpleeterEngine()
    await engine.initialize()
    
    # Separate audio
    result = await engine.separate(
        audio_path="song.wav",
        model_name="spleeter:2stems-16kHz",
        output_dir="output/"
    )
    
    print(f"✅ Separation complete: {result.output_files}")

# Run
asyncio.run(separate_audio())
```

### Advanced Configuration
```python
from spleeter import SpleeterEngine, SpleeterConfig

# Enterprise configuration
config = SpleeterConfig(
    # Performance
    enable_gpu=True,
    batch_size=8,
    worker_threads=4,
    
    # Caching
    cache_enabled=True,
    cache_size_mb=2048,
    redis_url="redis://localhost:6379",
    
    # Audio settings
    default_sample_rate=44100,
    enable_preprocessing=True,
    normalize_loudness=True,
    
    # Monitoring
    enable_monitoring=True,
    metrics_export_interval=60
)

engine = SpleeterEngine(config=config)
```

---

## 📚 API Reference

### SpleeterEngine
Main processing engine with enterprise capabilities.

#### Methods

**`async initialize()`**
Initialize the engine and load models.

**`async separate(audio_path, model_name, output_dir, **options)`**
Separate audio into stems.
- `audio_path`: Input audio file path
- `model_name`: Model identifier (e.g., "spleeter:2stems-16kHz")
- `output_dir`: Output directory for separated stems
- `options`: Additional processing options

**`async batch_separate(audio_files, **options)`**
Process multiple files in parallel.

**`get_available_models()`**
List all available separation models.

**`get_processing_stats()`**
Retrieve performance statistics.

### ModelManager
Advanced model management system.

#### Methods

**`async download_model(model_name, force=False)`**
Download and cache a specific model.

**`list_local_models()`**
List locally available models.

**`validate_model(model_name)`**
Verify model integrity.

### CacheManager
Multi-level caching system.

#### Methods

**`async get(key, cache_level="auto")`**
Retrieve cached data.

**`async set(key, data, ttl=3600, cache_level="auto")`**
Store data in cache.

**`get_cache_stats()`**
Cache performance statistics.

---

## 🔧 Configuration

### Environment Variables
```bash
# GPU Configuration
SPLEETER_ENABLE_GPU=true
SPLEETER_GPU_MEMORY_GROWTH=true

# Cache Configuration
SPLEETER_CACHE_DIR=/var/cache/spleeter
SPLEETER_REDIS_URL=redis://localhost:6379/0

# Model Configuration
SPLEETER_MODELS_DIR=/opt/spleeter/models
SPLEETER_AUTO_DOWNLOAD=true

# Monitoring
SPLEETER_ENABLE_MONITORING=true
SPLEETER_METRICS_PORT=9090
```

### Configuration File (config.yaml)
```yaml
spleeter:
  performance:
    enable_gpu: true
    batch_size: 8
    worker_threads: 4
    memory_limit_mb: 8192
  
  cache:
    enabled: true
    memory_size_mb: 512
    disk_size_mb: 2048
    redis_url: "redis://localhost:6379/0"
    ttl_hours: 24
  
  audio:
    default_sample_rate: 44100
    supported_formats: ["wav", "flac", "mp3", "ogg"]
    enable_preprocessing: true
    normalize_loudness: true
    
  monitoring:
    enabled: true
    export_interval: 60
    health_check_interval: 30
    alert_thresholds:
      memory_usage: 85
      gpu_usage: 90
      error_rate: 5
```

---

## 📊 Performance & Monitoring

### Real-time Metrics
The module provides comprehensive monitoring capabilities:

- **🎯 Processing Metrics**: Success rate, processing time, throughput
- **💾 Resource Usage**: CPU, memory, GPU utilization
- **🔄 Cache Performance**: Hit rates, cache efficiency
- **🚨 Health Monitoring**: System status, error tracking, alerts

### Monitoring Dashboard
```python
from spleeter.monitoring import get_stats_summary

# Get comprehensive stats
stats = get_stats_summary()
print(f"Success Rate: {stats['processing_stats']['success_rate']}%")
print(f"Cache Hit Rate: {stats['processing_stats']['cache_hit_rate']}%")
print(f"System Health: {stats['system_health']['status']}")
```

### Performance Optimization Tips

1. **🎯 GPU Usage**: Enable GPU acceleration for 2-3x speed improvement
2. **💾 Caching**: Configure appropriate cache sizes for your workload
3. **📦 Batch Processing**: Use batch operations for multiple files
4. **🔧 Model Selection**: Choose appropriate model complexity vs. speed
5. **⚡ Preprocessing**: Enable preprocessing for better separation quality

---

## 🛡️ Security & Best Practices

### Input Validation
- **File Format Validation**: Automatic format detection and validation
- **Size Limits**: Configurable file size and duration limits
- **Path Security**: Protection against path traversal attacks
- **Resource Limits**: Memory and processing time constraints

### Error Handling
```python
from spleeter.exceptions import (
    AudioProcessingError, 
    ModelError, 
    ValidationError
)

try:
    result = await engine.separate("audio.wav", "spleeter:2stems-16kHz")
except AudioProcessingError as e:
    print(f"Audio processing failed: {e}")
    print(f"Error context: {e.context}")
except ModelError as e:
    print(f"Model error: {e}")
except ValidationError as e:
    print(f"Validation failed: {e}")
```

---

## 📈 Scalability & Production

### Horizontal Scaling
```python
# Multi-instance setup
from spleeter import SpleeterCluster

cluster = SpleeterCluster(
    nodes=["worker1:8080", "worker2:8080", "worker3:8080"],
    load_balancer="round_robin",
    shared_cache="redis://cache-cluster:6379"
)

# Distributed processing
result = await cluster.separate_distributed(
    audio_files=["song1.wav", "song2.wav", "song3.wav"],
    model_name="spleeter:4stems-16kHz"
)
```

### Docker Deployment
```dockerfile
FROM tensorflow/tensorflow:2.8.0-gpu

COPY . /app/spleeter
WORKDIR /app

RUN pip install -r requirements.txt
CMD ["python", "-m", "spleeter.server"]
```

### Kubernetes Configuration
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spleeter-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: spleeter
  template:
    metadata:
      labels:
        app: spleeter
    spec:
      containers:
      - name: spleeter
        image: spleeter:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
          requests:
            memory: "4Gi"
```

---

## 🧪 Testing & Quality Assurance

### Running Tests
```bash
# Unit tests
python -m pytest tests/test_core.py -v

# Integration tests
python -m pytest tests/test_integration.py -v

# Performance tests
python -m pytest tests/test_performance.py --benchmark-only

# Full test suite
python -m pytest tests/ --cov=spleeter --cov-report=html
```

### Quality Metrics
- **✅ Code Coverage**: >95%
- **🎯 Performance**: <2x real-time processing
- **🛡️ Security**: OWASP compliance
- **📊 Reliability**: 99.9% uptime target

---

## 🤝 Contributing

### Development Setup
```bash
# Development installation
git clone https://github.com/spotify-ai-agent/backend.git
cd backend/spleeter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Code Standards
- **Style**: Black formatting, PEP 8 compliance
- **Type Hints**: Comprehensive type annotations
- **Documentation**: Docstring coverage >90%
- **Testing**: Test coverage >95%

---

## 📋 Changelog

### v2.0.0 (Current)
- ✨ Complete enterprise architecture rewrite
- 🚀 Async/await support throughout
- 💾 Multi-level caching system
- 📊 Comprehensive monitoring
- 🛡️ Enhanced security and validation
- 🎯 GPU optimization improvements
- 📦 Batch processing capabilities

### v1.x (Legacy)
- Basic Spleeter integration
- Synchronous processing only
- Limited error handling

---

## 📞 Support & Contact

### Documentation
- **📖 Full Documentation**: [docs.spotify-ai-agent.com](https://docs.spotify-ai-agent.com)
- **🔧 API Reference**: [api.spotify-ai-agent.com](https://api.spotify-ai-agent.com)
- **💡 Examples**: [github.com/spotify-ai-agent/examples](https://github.com/spotify-ai-agent/examples)

### Community
- **💬 Discord**: [discord.gg/spotify-ai](https://discord.gg/spotify-ai)
- **📧 Email**: support@spotify-ai-agent.com
- **🐛 Issues**: [GitHub Issues](https://github.com/spotify-ai-agent/backend/issues)

### Enterprise Support
For enterprise customers, dedicated support is available with SLA guarantees.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Spleeter Team**: Original Spleeter library by Deezer Research
- **TensorFlow Team**: ML framework and GPU optimization
- **Open Source Community**: Various audio processing libraries

---

**⭐ If you find this module useful, please consider starring the repository!**

---

*Built with ❤️ by the Spotify AI Agent Enterprise Team*  
*Lead Architect: [Fahed Mlaiel](https://github.com/fahed-mlaiel)*
