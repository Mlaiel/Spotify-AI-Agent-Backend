
# Dockerfile pour le système de validation enterprise
# Créé par l'équipe dirigée par Fahed Mlaiel

FROM python:3.9-slim

LABEL maintainer="Fahed Mlaiel <fahed.mlaiel@spotify-ai-agent.com>"
LABEL description="Enterprise Schema Validation System"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV SCHEMA_VALIDATION_MODE=enterprise
ENV AI_FEATURES_ENABLED=true

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Création du répertoire de travail
WORKDIR /app/schemas

# Copie des requirements
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY . .

# Création d'un utilisateur non-root
RUN useradd --create-home --shell /bin/bash schema-validator && \
    chown -R schema-validator:schema-validator /app
USER schema-validator

# Port d'exposition pour l'API de validation
EXPOSE 8080

# Commande de démarrage
CMD ["python", "-m", "uvicorn", "validation_api:app", "--host", "0.0.0.0", "--port", "8080"]

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1
