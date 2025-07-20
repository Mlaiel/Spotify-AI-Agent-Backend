"""
Script de backup automatique PostgreSQL
- Dump sécurisé, rotation, logs, prêt pour cron/CI/CD
"""

import os
import logging
import subprocess
from datetime import datetime

BACKUP_DIR = os.getenv("PG_BACKUP_DIR", "./backups/postgresql/")
DB_NAME = os.getenv("POSTGRES_DB", "spotify_ai_agent")
USER = os.getenv("POSTGRES_USER", "postgres")
HOST = os.getenv("POSTGRES_HOST", "localhost")
PORT = os.getenv("POSTGRES_PORT", "5432")
PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
MAX_BACKUPS = int(os.getenv("PG_MAX_BACKUPS", 7)

os.makedirs(BACKUP_DIR, exist_ok=True)

fname = f"{DB_NAME}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
path = os.path.join(BACKUP_DIR, fname)
cmd = [
    "pg_dump",
    f"-U{USER}",
    f"-h{HOST}",
    f"-p{PORT}",
    DB_NAME,
    "-f", path
]
try:
    subprocess.run(cmd, check=True, env={**os.environ, "PGPASSWORD": PASSWORD})
    logging.info(f"Backup PostgreSQL créé: {path}")
except Exception as e:
    logging.error(f"Erreur backup PostgreSQL: {e}")
    raise

# Rotation des backups
backups = sorted([f for f in os.listdir(BACKUP_DIR) if f.endswith('.sql')])
while len(backups) > MAX_BACKUPS:
    to_remove = backups.pop(0)
    os.remove(os.path.join(BACKUP_DIR, to_remove)
    logging.info(f"Backup supprimé (rotation): {to_remove}")

print("Backup automatique terminé.")
