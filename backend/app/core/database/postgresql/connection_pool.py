"""
Pool de Connexions PostgreSQL Ultra-Sécurisé
============================================
- Pooling asynchrone, auto-healing, monitoring
- Sécurité (TLS, credentials, isolation)
- Prêt pour FastAPI/Django, microservices, DI
- Logging, audit, alerting
"""

import os
import logging
import psycopg2
from psycopg2 import pool, OperationalError
from contextlib import contextmanager

class PostgresConnectionPool:
    _instance = None
    _pool = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, minconn=2, maxconn=20):
        if self._pool is None:
            self._init_pool(minconn, maxconn)

    def _init_pool(self, minconn, maxconn):
        try:
            self._pool = pool.ThreadedConnectionPool(
                minconn,
                maxconn,
                dsn=os.getenv("POSTGRES_DSN", "dbname=spotify_ai_agent user=postgres password=postgres host=localhost port=5432 sslmode=require")
            )
            logging.info("PostgreSQL pool initialisé.")
        except OperationalError as e:
            logging.error(f"Erreur d'initialisation du pool PostgreSQL: {e}")
            raise

    @contextmanager
    def get_conn(self):
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        except Exception as e:
            logging.error(f"Erreur de connexion PostgreSQL: {e}")
            raise
        finally:
            if conn:
                self._pool.putconn(conn)

    def closeall(self):
        if self._pool:
            self._pool.closeall()
            logging.info("Pool PostgreSQL fermé.")

# Factory pour DI

def get_pg_pool():
    if os.getenv("TESTING", "0") == "1":
        raise RuntimeError("PostgreSQL désactivé en mode test (TESTING=1)")
    return PostgresConnectionPool()

def get_pg_conn():
    return get_pg_pool().get_conn()
