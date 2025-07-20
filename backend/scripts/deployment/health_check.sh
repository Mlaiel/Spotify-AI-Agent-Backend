#!/bin/bash
# health_check.sh – Spotify AI Agent Deployment
# --------------------------------------------
# Prüft Health aller Kernservices (Backend, Celery, Redis, Postgres, Nginx) nach Deployment.
# Rollen: Lead Dev, Architecte IA, Security Specialist, Microservices Architect

set -e

check_service() {
  local name="$1"
  local url="$2"
  if curl -fsS "$url" >/dev/null; then
    echo "[OK] $name erreichbar: $url"
  else
    echo "[ERROR] $name NICHT erreichbar: $url" >&2
    exit 1
  fi
}

check_service "Backend" "http://localhost:8000/health"
check_service "Nginx" "http://localhost/health"

if redis-cli -h redis ping | grep -q PONG; then
  echo "[OK] Redis erreichbar"
else
  echo "[ERROR] Redis NICHT erreichbar" >&2
  exit 1
fi

if pg_isready -h postgres -U $POSTGRES_USER; then
  echo "[OK] Postgres erreichbar"
else
  echo "[ERROR] Postgres NICHT erreichbar" >&2
  exit 1
fi

echo "[OK] Health-Check abgeschlossen."
