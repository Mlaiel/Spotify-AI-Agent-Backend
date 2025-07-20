# Enterprise Configuration System Docker Deployment
# Multi-stage build for production-ready deployment

FROM python:3.11-slim as base

# Metadata
LABEL maintainer="Fahed Mlaiel"
LABEL version="1.0.0"
LABEL description="Enterprise Configuration System - Production Container"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Change ownership
RUN chown -R appuser:appuser /app

USER appuser

# Development command
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Production Python dependencies only
RUN pip install --no-cache-dir \
    gunicorn \
    uvicorn[standard] \
    prometheus-client

# Copy application code
COPY --chown=appuser:appuser . .

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appuser /app/logs /app/data

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["gunicorn", "app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

# Monitoring stage with additional tools
FROM production as monitoring

USER root

# Install monitoring tools
RUN apt-get update && apt-get install -y \
    htop \
    iotop \
    netcat-openbsd \
    redis-tools \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install monitoring Python packages
RUN pip install --no-cache-dir \
    prometheus-client \
    grafana-api \
    structlog

USER appuser

# Add monitoring scripts
COPY --chown=appuser:appuser scripts/monitoring/ /app/scripts/monitoring/

# Monitoring command
CMD ["python", "scripts/monitoring/system_monitor.py"]
