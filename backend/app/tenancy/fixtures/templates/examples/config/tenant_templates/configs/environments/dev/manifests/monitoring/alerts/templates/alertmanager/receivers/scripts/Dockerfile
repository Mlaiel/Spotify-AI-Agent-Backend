# Multi-stage Dockerfile for Advanced Scripts System
# Version: 3.0.0
# Developed by Spotify AI Agent Team
# Lead Developer & AI Architect: Fahed Mlaiel

# ============================================================================
# Stage 1: Base Dependencies
# ============================================================================
FROM python:3.11-slim-bullseye AS base-dependencies

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    gnupg2 \
    lsb-release \
    ca-certificates \
    apt-transport-https \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd --create-home --shell /bin/bash --user-group scripts-user

# ============================================================================
# Stage 2: Python Dependencies
# ============================================================================
FROM base-dependencies AS python-dependencies

# Copy requirements
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ============================================================================
# Stage 3: Development Stage
# ============================================================================
FROM python-dependencies AS development

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    nano \
    htop \
    tree \
    jq \
    httpie \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    flake8 \
    mypy \
    bandit \
    safety \
    pre-commit

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=scripts-user:scripts-user . /app/

# Switch to application user
USER scripts-user

# Expose development port
EXPOSE 8000

# Development command
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ============================================================================
# Stage 4: Testing Stage
# ============================================================================
FROM python-dependencies AS testing

# Install test dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    pytest-mock \
    pytest-xdist \
    coverage \
    locust

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=scripts-user:scripts-user . /app/

# Copy test configurations
COPY --chown=scripts-user:scripts-user pytest.ini /app/
COPY --chown=scripts-user:scripts-user .coveragerc /app/

# Switch to application user
USER scripts-user

# Run tests by default
CMD ["pytest", "--cov=.", "--cov-report=xml", "--cov-report=html", "--cov-report=term"]

# ============================================================================
# Stage 5: Security Scanning
# ============================================================================
FROM python-dependencies AS security-scan

# Install security tools
RUN pip install --no-cache-dir \
    bandit \
    safety \
    semgrep \
    pip-audit

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=scripts-user:scripts-user . /app/

# Switch to application user
USER scripts-user

# Security scan command
CMD ["bandit", "-r", ".", "-f", "json", "-o", "security-report.json"]

# ============================================================================
# Stage 6: Production Base
# ============================================================================
FROM python:3.11-slim-bullseye AS production-base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    ENV=production

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    ca-certificates \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

# Create application user
RUN useradd --create-home --shell /bin/bash --user-group scripts-user

# Copy Python dependencies from previous stage
COPY --from=python-dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-dependencies /usr/local/bin /usr/local/bin

# ============================================================================
# Stage 7: Production Application
# ============================================================================
FROM production-base AS production

# Install Docker client for pipeline management
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add - && \
    echo "deb [arch=amd64] https://download.docker.com/linux/debian bullseye stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# Install kubectl for Kubernetes management
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    chmod +x kubectl && \
    mv kubectl /usr/local/bin/

# Install additional tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openssh-client \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/configs /app/templates /app/scripts && \
    chown -R scripts-user:scripts-user /app

# Copy application code
COPY --chown=scripts-user:scripts-user . /app/

# Copy configuration templates
COPY --chown=scripts-user:scripts-user ./configs/ /app/configs/
COPY --chown=scripts-user:scripts-user ./templates/ /app/templates/

# Install application in development mode
RUN pip install -e .

# Create startup script
RUN cat > /app/start.sh << 'EOF'
#!/bin/bash
set -e

# Wait for dependencies
echo "Waiting for dependencies..."
if [ ! -z "$DATABASE_URL" ]; then
    python -c "import time; import psycopg2; time.sleep(10); psycopg2.connect('$DATABASE_URL')"
fi

# Run database migrations if needed
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    python -m alembic upgrade head
fi

# Start the application
echo "Starting Scripts System..."
exec "$@"
EOF

RUN chmod +x /app/start.sh && chown scripts-user:scripts-user /app/start.sh

# Switch to application user
USER scripts-user

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose application port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/app/start.sh"]

# Default command
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# ============================================================================
# Stage 8: Monitoring & Observability
# ============================================================================
FROM production AS monitoring

# Install monitoring tools
RUN pip install --no-cache-dir \
    prometheus-client \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-instrumentation \
    opentelemetry-exporter-prometheus \
    jaeger-client \
    elastic-apm

# Copy monitoring configurations
COPY --chown=scripts-user:scripts-user ./monitoring/ /app/monitoring/

# Expose monitoring ports
EXPOSE 8000 9090 14268

# Override command for monitoring
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--log-config", "monitoring/logging.yaml"]

# ============================================================================
# Stage 9: Distributed Deployment
# ============================================================================
FROM production AS distributed

# Install distributed computing dependencies
RUN pip install --no-cache-dir \
    celery \
    redis \
    flower \
    dask \
    ray

# Create worker startup script
RUN cat > /app/start-worker.sh << 'EOF'
#!/bin/bash
set -e

echo "Starting Celery worker..."
exec celery -A main.celery worker --loglevel=info --concurrency=4
EOF

RUN chmod +x /app/start-worker.sh && chown scripts-user:scripts-user /app/start-worker.sh

# Create scheduler startup script
RUN cat > /app/start-scheduler.sh << 'EOF'
#!/bin/bash
set -e

echo "Starting Celery beat scheduler..."
exec celery -A main.celery beat --loglevel=info
EOF

RUN chmod +x /app/start-scheduler.sh && chown scripts-user:scripts-user /app/start-scheduler.sh

# Create flower monitoring script
RUN cat > /app/start-flower.sh << 'EOF'
#!/bin/bash
set -e

echo "Starting Flower monitoring..."
exec celery -A main.celery flower --port=5555
EOF

RUN chmod +x /app/start-flower.sh && chown scripts-user:scripts-user /app/start-flower.sh

# Expose additional ports
EXPOSE 5555

# ============================================================================
# Stage 10: Edge Computing
# ============================================================================
FROM alpine:3.18 AS edge

# Install minimal dependencies
RUN apk add --no-cache \
    python3 \
    py3-pip \
    curl \
    git \
    bash

# Create application user
RUN adduser -D -s /bin/bash scripts-user

# Install minimal Python dependencies
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn \
    aiohttp \
    structlog \
    pydantic

# Set working directory
WORKDIR /app

# Copy minimal application
COPY --chown=scripts-user:scripts-user ./core/ /app/core/
COPY --chown=scripts-user:scripts-user ./main.py /app/
COPY --chown=scripts-user:scripts-user ./requirements-edge.txt /app/

# Install edge requirements
RUN pip3 install --no-cache-dir -r requirements-edge.txt

# Switch to application user
USER scripts-user

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=60s --timeout=5s --start-period=30s --retries=2 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================================================
# Build Arguments and Labels
# ============================================================================
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=3.0.0

LABEL maintainer="Fahed Mlaiel <fahed.mlaiel@spotify.com>" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="Advanced Scripts System" \
      org.label-schema.description="Enterprise-grade scripts management system with AI automation" \
      org.label-schema.url="https://github.com/spotify/ai-agent" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/spotify/ai-agent" \
      org.label-schema.vendor="Spotify AI Agent Team" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0" \
      org.opencontainers.image.title="Advanced Scripts System" \
      org.opencontainers.image.description="Enterprise-grade scripts management system" \
      org.opencontainers.image.authors="Fahed Mlaiel <fahed.mlaiel@spotify.com>" \
      org.opencontainers.image.vendor="Spotify AI Agent Team" \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.source="https://github.com/spotify/ai-agent" \
      org.opencontainers.image.revision=$VCS_REF

# ============================================================================
# Usage Examples:
# 
# Development:
# docker build --target development -t scripts-system:dev .
# docker run -p 8000:8000 -v $(pwd):/app scripts-system:dev
#
# Testing:
# docker build --target testing -t scripts-system:test .
# docker run scripts-system:test
#
# Security Scan:
# docker build --target security-scan -t scripts-system:security .
# docker run scripts-system:security
#
# Production:
# docker build --target production -t scripts-system:latest .
# docker run -p 8000:8000 scripts-system:latest
#
# Monitoring:
# docker build --target monitoring -t scripts-system:monitoring .
# docker run -p 8000:8000 -p 9090:9090 scripts-system:monitoring
#
# Distributed Worker:
# docker build --target distributed -t scripts-system:worker .
# docker run scripts-system:worker /app/start-worker.sh
#
# Edge Computing:
# docker build --target edge -t scripts-system:edge .
# docker run -p 8000:8000 scripts-system:edge
# ============================================================================
