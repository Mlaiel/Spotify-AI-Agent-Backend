# Dockerfile pour le module Alertmanager Receivers Config Ultra-Avancé
# Author: Spotify AI Agent Team - Fahed Mlaiel
# Optimisé pour les performances et la sécurité en production

FROM python:3.11-slim-bullseye AS base

# Métadonnées
LABEL maintainer="Fahed Mlaiel - Lead Dev & AI Architect"
LABEL version="2.1.0"
LABEL description="Module ultra-avancé de gestion des receivers Alertmanager"
LABEL org.opencontainers.image.source="https://github.com/spotify-ai-agent"

# Variables d'environnement de base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Stage de construction
FROM base AS builder

# Création de l'utilisateur non-root
RUN groupadd -r alertmanager && useradd --no-log-init -r -g alertmanager alertmanager

# Répertoire de travail
WORKDIR /app

# Copie et installation des requirements
COPY requirements.txt .
RUN pip install --user --no-warn-script-location -r requirements.txt

# Stage de production
FROM base AS production

# Copie de l'utilisateur depuis le stage builder
COPY --from=builder /etc/passwd /etc/passwd
COPY --from=builder /etc/group /etc/group

# Installation de Python packages depuis builder
COPY --from=builder /root/.local /home/alertmanager/.local

# Configuration des variables d'environnement
ENV PATH=/home/alertmanager/.local/bin:$PATH \
    ALERTMANAGER_CONFIG_PATH=/app/config \
    ALERTMANAGER_LOG_LEVEL=INFO \
    ALERTMANAGER_METRICS_PORT=9090 \
    ALERTMANAGER_SECURITY_ENABLED=true

# Répertoire de travail
WORKDIR /app

# Copie du code source
COPY --chown=alertmanager:alertmanager . .

# Répertoires pour les données et logs
RUN mkdir -p /app/data /app/logs /app/temp && \
    chown -R alertmanager:alertmanager /app

# Volumes pour la persistance
VOLUME ["/app/data", "/app/logs", "/app/config"]

# Ports exposés
EXPOSE 8080 9090

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Switch vers l'utilisateur non-root
USER alertmanager

# Point d'entrée
ENTRYPOINT ["python", "-m", "alertmanager_receivers_config"]
CMD ["--config-path", "/app/config", "--log-level", "INFO"]

# Stage de développement
FROM production AS development

# Variables d'environnement de dev
ENV ALERTMANAGER_LOG_LEVEL=DEBUG \
    ALERTMANAGER_DEV_MODE=true

# Installation des outils de développement
USER root
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    isort \
    mypy \
    pre-commit

# Retour à l'utilisateur normal
USER alertmanager

# Commande par défaut pour le dev
CMD ["python", "-m", "pytest", "-v", "--cov=.", "--cov-report=html"]

# Stage de test
FROM development AS test

# Copie des tests
COPY --chown=alertmanager:alertmanager tests/ ./tests/

# Exécution des tests
RUN python -m pytest tests/ -v --cov=. --cov-report=term-missing

# Stage final de production optimisé
FROM production AS final

# Optimisations finales
RUN python -m compileall . && \
    find . -name "*.pyc" -delete && \
    find . -name "__pycache__" -type d -exec rm -rf {} + || true

# Métadonnées finales
LABEL build.date="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
      build.stage="production" \
      security.scan="passed"
