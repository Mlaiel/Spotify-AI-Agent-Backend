# Troubleshooting – Spotify AI Backend (EN)

This section details error handling, resilience, and backend troubleshooting tools.

## 1. Error Handling Strategy
- Custom exceptions (Python, FastAPI, ML)
- Error code mapping (API, DB, ML, security)
- Localized error messages (i18n)

## 2. Logging & Alerting
- Structured logging (level, context, trace)
- Automated alerts (Sentry, Prometheus)
- Log/error correlation (trace ID)

## 3. Troubleshooting Tools
- Diagnostic scripts (`scripts/maintenance/diagnostic.sh`)
- Automatic dump on crash
- Error dashboards (Grafana, Kibana)

## 4. Example Managed Errors
- Authentication, quotas, DB, ML, external APIs
- Timeout, rate limit, invalid payload

## 5. Best Practices
- Never expose raw stacktrace in production
- Always log request ID, user, context
- Document frequent errors in this file

> **Tip:** All scripts and dashboards are provided in `scripts/maintenance` and `config/monitoring`.
