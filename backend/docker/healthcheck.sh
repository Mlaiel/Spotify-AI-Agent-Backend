#!/bin/bash
# healthcheck.sh – Spotify AI Agent
# ---------------------------------
# Health-Check für alle Kernservices (Backend, Celery, Redis, Postgres, Nginx)
# Rollen: Lead Dev, Architecte IA, Backend Senior, Security Specialist, Microservices Architect

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

# Backend
check_service "Backend" "http://localhost:8000/health"
# Nginx
check_service "Nginx" "http://localhost/health"
# Redis
if redis-cli -h redis ping | grep -q PONG; then
  echo "[OK] Redis erreichbar"
else
  echo "[ERROR] Redis NICHT erreichbar" >&2
  exit 1
fi
# Postgres
if pg_isready -h postgres -U $POSTGRES_USER; then
  echo "[OK] Postgres erreichbar"
else
  echo "[ERROR] Postgres NICHT erreichbar" >&2
  exit 1
fi

exit 0
