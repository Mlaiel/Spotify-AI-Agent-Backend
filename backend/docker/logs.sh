#!/bin/bash
# logs.sh – Spotify AI Agent
# -------------------------
# Zeigt und filtert Logs aller Kernservices (Backend, Celery, Redis, Postgres, Nginx)
# Rollen: DevOps, Lead Dev, Security Specialist

set -e

SERVICES=(backend celery-worker celery-beat redis postgres nginx)

for s in "${SERVICES[@]}"; do
  if docker ps --format '{{.Names}}' | grep -q "$s"; then
    echo "--- LOGS: $s ---"
    docker logs "$s" --tail=100
    echo ""
  fi
done
