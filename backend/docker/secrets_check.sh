#!/bin/bash
# secrets_check.sh – Spotify AI Agent
# ----------------------------------
# Prüft, ob alle Secrets/Umgebungsvariablen korrekt gesetzt sind
# Rollen: Security Specialist, Lead Dev, Architecte IA

set -e

ENV_FILE="../config/environments/.env.production"
if [ ! -f "$ENV_FILE" ]; then
  echo "[ERROR] $ENV_FILE nicht gefunden!" >&2
  exit 1
fi

REQUIRED_VARS=(POSTGRES_USER POSTGRES_PASSWORD REDIS_URL SECRET_KEY SENTRY_DSN)

for v in "${REQUIRED_VARS[@]}"; do
  if ! grep -q "^$v=" "$ENV_FILE"; then
    echo "[ERROR] $v fehlt in $ENV_FILE!" >&2
    exit 1
  fi
done

echo "[OK] Alle erforderlichen Secrets/Umgebungsvariablen sind gesetzt."
