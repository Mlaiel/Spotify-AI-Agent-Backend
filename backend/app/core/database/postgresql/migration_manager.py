"""
Migration Manager PostgreSQL
===========================
- Versionning, scripts, rollback, logs
- Sécurité, audit, automatisation
- Prêt pour CI/CD, microservices
"""

import os
import logging
from .connection_pool import get_pg_conn

class MigrationManager:
    MIGRATIONS_DIR = os.getenv("PG_MIGRATIONS_DIR", "./migrations/postgresql/")

    def __init__(self):
        os.makedirs(self.MIGRATIONS_DIR, exist_ok=True)

    def apply_migrations(self):
        files = sorted([f for f in os.listdir(self.MIGRATIONS_DIR) if f.endswith('.sql')])
        with get_pg_conn() as conn:
            with conn.cursor() as cur:
                for fname in files:
                    path = os.path.join(self.MIGRATIONS_DIR, fname)
                    with open(path, 'r') as f:
                        sql = f.read()
                        try:
                            cur.execute(sql)
                            logging.info(f"Migration appliquée: {fname}")
                        except Exception as e:
                            conn.rollback()
                            logging.error(f"Erreur migration {fname}: {e}")
                            raise
                conn.commit()

    def create_migration(self, name):
        idx = len(os.listdir(self.MIGRATIONS_DIR)) + 1
        fname = f"{idx:03d}_{name}.sql"
        path = os.path.join(self.MIGRATIONS_DIR, fname)
        with open(path, 'w') as f:
            f.write('-- SQL migration script\n')
        logging.info(f"Migration créée: {fname}")
        return path

# Exemple :
# mm = MigrationManager()
# mm.create_migration("add_users_table")
# mm.apply_migrations()
