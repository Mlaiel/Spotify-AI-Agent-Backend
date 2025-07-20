#!/bin/bash
# update_stack.sh – Spotify AI Agent
# ---------------------------------
# Aktualisiert Images, führt Rolling-Update und prüft Stack-Status
# Rollen: DevOps, Lead Dev, Security Specialist

set -e

echo "[INFO] Pull neue Images..."
docker-compose pull

echo "[INFO] Starte Rolling-Update..."
docker-compose up -d --remove-orphans

sleep 5

./test_stack.sh

echo "[OK] Stack-Update abgeschlossen."
