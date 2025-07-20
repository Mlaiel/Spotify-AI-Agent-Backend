# Documentation (EN)

# Spotify AI Agent – Advanced Middleware Suite

## Overview
This directory contains the full-stack, production-grade middleware suite for the Spotify AI Agent backend. Each middleware is designed to be plug-and-play, highly secure, observable, and ready for enterprise use. All logic is business-ready, with no TODOs or placeholders.

---

## Architecture Diagram
```
[Request]
   │
   ▼
[RequestIdMiddleware] → [CorsMiddleware] → [SecurityHeaders] → [ErrorHandler]
   │
   ▼
[PerformanceMonitor] → [AuthMiddleware] → [I18NMiddleware] → [RateLimiting]
   │
   ▼
[LoggingMiddleware]
   │
   ▼
[API Business Logic]
```

---

## Middleware List & Features

### 1. RequestIdMiddleware
- Generates, propagates, and logs unique request/correlation/trace IDs (UUID4, Snowflake, etc.)
- Enables distributed tracing (OpenTelemetry)
- Exposes IDs in headers and logs for full auditability

### 2. CorsMiddleware
- Dynamic, context-aware CORS with strict security
- Whitelist/blacklist, subdomain and regex support, rate limiting per origin
- Preflight optimization, analytics, and geoip support

### 3. SecurityHeaders
- Sets advanced security headers (CORS, CSP, HSTS, XSS, Referrer, Permissions Policy)
- Dynamic adaptation per environment and endpoint
- Monitors/reporting for header violations

### 4. ErrorHandler
- Full error classification (system, business, AI, Spotify, etc.)
- Circuit breaker, fallback, auto-healing, Sentry/Prometheus integration
- Localized error messages, audit, and analytics

### 5. PerformanceMonitor
- Real-time latency, CPU, memory, DB, cache, AI metrics
- Anomaly detection, profiling, auto-optimization suggestions
- Prometheus/Grafana/Jaeger integration

### 6. AuthMiddleware
- OAuth2, JWT, API Key, role-based, session, device/IP protection
- Security audit, session management, brute-force protection

### 7. I18NMiddleware
- Auto language detection, 25+ languages, fallback, RTL support
- Dynamic loading, analytics, multi-level cache

### 8. RateLimiting
- Adaptive, ML-based, per-user/IP/endpoint, DDoS detection
- Quotas for AI/Spotify APIs, analytics, IP blocking

### 9. LoggingMiddleware
- Structured, distributed, and business logging
- Security audit, tracing, error tracking, alerting

---

## Usage Example
```python
from fastapi import FastAPI
from app.api.middleware import MIDDLEWARE_STACK

app = FastAPI()

for middleware in MIDDLEWARE_STACK:
    app.add_middleware(middleware)
```

---

## Security & Compliance
- All middleware are GDPR/CCPA ready
- No sensitive data leakage (stack traces sanitized)
- Full audit trail and monitoring

## Observability
- Prometheus, Sentry, OpenTelemetry, Grafana, Jaeger supported out-of-the-box
- All metrics and traces are labeled with request/correlation IDs

## Extensibility
- Each middleware is modular and can be configured or extended per environment
- Factories and decorators provided for advanced use cases

---

## Authors & Roles
- Lead Dev & AI Architect
- Senior Backend Developer
- ML Engineer
- DBA & Data Engineer
- Security Specialist
- Microservices Architect

---

## See Also
- [README.fr.md](./README.fr.md) (Français)
- [README.de.md](./README.de.md) (Deutsch)

