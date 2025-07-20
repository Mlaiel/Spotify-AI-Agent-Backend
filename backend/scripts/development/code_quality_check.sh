#!/bin/bash
# code_quality_check.sh – Spotify AI Agent
# ---------------------------------------
# Führt alle Code-Qualitätsprüfungen, Security- und Compliance-Checks aus.
# Rollen: Lead Dev, Backend Senior, ML Engineer, Security Specialist

set -e

# Linting
flake8 ../../app
black --check ../../app
isort --check ../../app

# Typprüfung
mypy ../../app

# Security-Scan
bandit -r ../../app
safety check

# Testabdeckung
pytest --maxfail=1 --disable-warnings --cov=../../app

# ML/AI-Checks (Beispiel)
pip install --quiet torch tensorflow transformers
python -c "import torch; print(torch.__version__)"
python -c "import tensorflow as tf; print(tf.__version__)"

# Compliance (z.B. DSGVO/GDPR)
grep -r -i 'personal' ../../app || true

echo "[OK] Code-Qualitätsprüfung abgeschlossen."
