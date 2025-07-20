#!/bin/bash
# setup_dev_env.sh – Spotify AI Agent
# -----------------------------------
# Automatisiert die Einrichtung der Dev-Umgebung (Python, Node, Docker, ML/AI, Security, Pre-Commit, Compliance)
# Rollen: Lead Dev, Backend Senior, ML Engineer, Security Specialist

set -e

# Python-Umgebung
python3 -m venv .venv
echo "[INFO] Aktiviere venv..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r ../../requirements/development.txt

# Node.js (für Frontend/Tools)
if ! command -v node &> /dev/null; then
  echo "[INFO] Installiere Node.js..."
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi

# Docker-Check
if ! command -v docker &> /dev/null; then
  echo "[ERROR] Docker nicht installiert!"
  exit 1
fi

# Pre-Commit Hooks
pip install pre-commit
pre-commit install

# Security-Tools
pip install bandit safety

# ML/AI-Tools
pip install torch tensorflow transformers

# Compliance-Tools
pip install python-json-logger

echo "[OK] Dev-Umgebung eingerichtet."
