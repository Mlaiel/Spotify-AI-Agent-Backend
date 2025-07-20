"""
Backup Manager PostgreSQL
========================
- Dump, restore, automatisation, logs, sécurité
- Prêt pour CI/CD, microservices, audit
"""

import os
import logging
import subprocess
from datetime import datetime

class BackupManager:
    BACKUP_DIR = os.getenv("PG_BACKUP_DIR", "./backups/postgresql/")
    DB_NAME = os.getenv("POSTGRES_DB", "spotify_ai_agent")
    USER = os.getenv("POSTGRES_USER", "postgres")
    HOST = os.getenv("POSTGRES_HOST", "localhost")
    PORT = os.getenv("POSTGRES_PORT", "5432")

    def __init__(self):
        os.makedirs(self.BACKUP_DIR, exist_ok=True)

    def backup(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{self.DB_NAME}_backup_{ts}.sql"
        path = os.path.join(self.BACKUP_DIR, fname)
        cmd = [
            "pg_dump",
            f"-U{self.USER}",
            f"-h{self.HOST}",
            f"-p{self.PORT}",
            self.DB_NAME,
            "-f", path
        ]
        try:
            subprocess.run(cmd, check=True, env={**os.environ, "PGPASSWORD": os.getenv("POSTGRES_PASSWORD", "postgres")})
            logging.info(f"Backup PostgreSQL créé: {path}")
            return path
        except Exception as e:
            logging.error(f"Erreur backup PostgreSQL: {e}")
            raise

    def restore(self, backup_file):
        cmd = [
            "psql",
            f"-U{self.USER}",
            f"-h{self.HOST}",
            f"-p{self.PORT}",
            self.DB_NAME,
            "-f", backup_file
        ]
        try:
            subprocess.run(cmd, check=True, env={**os.environ, "PGPASSWORD": os.getenv("POSTGRES_PASSWORD", "postgres")})
            logging.info(f"Backup PostgreSQL restauré: {backup_file}")
        except Exception as e:
            logging.error(f"Erreur restauration PostgreSQL: {e}")
            raise

# Exemple :
# bm = BackupManager()
# bm.backup()
# bm.restore("./backups/postgresql/spotify_ai_agent_backup_20250710_120000.sql")
