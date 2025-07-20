#!/bin/bash
# prometheus_exporter.sh – Spotify AI Agent
# ----------------------------------------
# Startet einen Prometheus Node Exporter für Monitoring
# Rollen: DevOps, Lead Dev, Architecte IA, Security Specialist

set -e

docker run -d --name prometheus-node-exporter --restart unless-stopped \
  -p 9100:9100 prom/node-exporter:latest

echo "[OK] Prometheus Node Exporter läuft auf Port 9100."
