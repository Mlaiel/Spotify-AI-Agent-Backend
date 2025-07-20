# Spotify AI Agent – Logging Module (EN)

This module provides a full-stack, production-grade logging and monitoring system for AI, SaaS, and microservices platforms.

## Features
- Centralized, dynamic logger configuration (JSON, rotation, Sentry-ready)
- Structured logging (JSON, context, correlation/trace ID)
- Performance logging (latency, throughput, Prometheus-ready)
- Audit logging (GDPR/SOX, security, AI actions)
- Error tracking (Sentry/ELK, context enrichment)
- Log aggregation (multi-service, export JSON/CSV)
- Asynchronous logging (FastAPI, Celery, streaming)

## Key Files
- `logger_config.py`: Central config, JSON/rotation/Sentry
- `structured_logger.py`: Structured/context logging
- `performance_logger.py`: Latency/throughput, decorators
- `audit_logger.py`: Audit trail, compliance, security
- `error_tracker.py`: Error tracking, Sentry/ELK
- `log_aggregator.py`: Aggregation, export, multi-service
- `async_logger.py`: Async logging for microservices
- `__init__.py`: Exposes all modules for direct import

## Usage Example
```python
from .logger_config import setup_logging
from .structured_logger import StructuredLogger
setup_logging()
logger = StructuredLogger()
logger.info("User login", context={"user_id": 123})
```

## Industrial-Ready
- Strict typing, robust error handling
- No TODOs, no placeholders
- Easily integrable in APIs, microservices, analytics pipelines
- Extensible for Sentry, ELK, Prometheus, Datadog

