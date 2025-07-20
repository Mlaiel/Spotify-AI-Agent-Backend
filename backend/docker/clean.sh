#!/bin/bash
# clean.sh – Spotify AI Agent
# --------------------------
# Löscht alte Docker-Container, Images, Volumes, Netzwerke (Cleanup)
# Rollen: DevOps, Lead Dev, Security Specialist

set -e

echo "[INFO] Stoppe und entferne alle Container..."
docker ps -aq | xargs -r docker stop
sleep 2
docker ps -aq | xargs -r docker rm

echo "[INFO] Entferne nicht verwendete Images..."
docker images -q | xargs -r docker rmi -f

echo "[INFO] Entferne nicht verwendete Volumes..."
docker volume ls -q | xargs -r docker volume rm

echo "[INFO] Entferne nicht verwendete Netzwerke..."
docker network ls -q | grep -v 'bridge\|host\|none' | xargs -r docker network rm

echo "[OK] Docker Cleanup abgeschlossen."
