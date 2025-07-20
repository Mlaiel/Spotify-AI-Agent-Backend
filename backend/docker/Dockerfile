# Dockerfile – Development
# ------------------------
# Rollen: Lead Dev, Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Security Specialist, Microservices Architect
# Multi-Stage, Caching, Dev-Tools, Hot-Reload, Non-Root

FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# System-Dependencies
RUN apt-get update && apt-get install -y build-essential libpq-dev curl git && rm -rf /var/lib/apt/lists/*

# Python-Dependencies
COPY ../requirements/ /app/requirements/
RUN pip install --upgrade pip && pip install -r requirements/development.txt

# App-Code
COPY .. /app

# User für Security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Entrypoint
CMD ["uvicorn", "app.asgi:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
