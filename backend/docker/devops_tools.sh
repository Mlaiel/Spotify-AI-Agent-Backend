#!/bin/bash
# devops_tools.sh – Spotify AI Agent
# ----------------------------------
# DevOps- und Infrastruktur-Tools für Build, Test, Monitoring, Security, ML/AI
# Rollen: DevOps, Lead Dev, Architecte IA, Security Specialist

set -e

# Build-Tools
pip install --upgrade pip setuptools wheel

# Linting & Test
pip install flake8 pytest coverage
flake8 app/
pytest --maxfail=1 --disable-warnings --cov=app

# Security-Scan
pip install bandit safety
bandit -r app/
safety check

# Monitoring/Observability
pip install prometheus_client opentelemetry-sdk sentry-sdk

# ML/AI-Tools
pip install tensorflow torch transformers

# Docker-Tools
if ! command -v trivy &> /dev/null; then
  echo "[INFO] Installiere Trivy..."
  apt-get update && apt-get install -y wget && wget -qO- https://aquasecurity.github.io/trivy-repo/deb/public.key | apt-key add - && echo "deb https://aquasecurity.github.io/trivy-repo/deb stable main" > /etc/apt/sources.list.d/trivy.list && apt-get update && apt-get install -y trivy
fi

# Fertig
