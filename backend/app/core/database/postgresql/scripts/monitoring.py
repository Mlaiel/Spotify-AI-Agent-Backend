"""
Script de monitoring PostgreSQL
- Vérifie la santé, la taille, les connexions, les verrous, les requêtes lentes
"""

import os
import psycopg2

DB_NAME = os.getenv("POSTGRES_DB", "spotify_ai_agent")
USER = os.getenv("POSTGRES_USER", "postgres")
PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
HOST = os.getenv("POSTGRES_HOST", "localhost")
PORT = os.getenv("POSTGRES_PORT", "5432")

conn = psycopg2.connect(dbname=DB_NAME, user=USER, password=PASSWORD, host=HOST, port=PORT)
cur = conn.cursor()

cur.execute("SELECT count(*) FROM pg_stat_activity;")
print(f"Connexions actives: {cur.fetchone()[0]}")

cur.execute("SELECT pg_size_pretty(pg_database_size(%s);", (DB_NAME,)
print(f"Taille base: {cur.fetchone()[0]}")

cur.execute("SELECT * FROM pg_locks WHERE NOT granted;")
locks = cur.fetchall()
print(f"Verrous non accordés: {len(locks)}")

cur.execute("SELECT pid, query, state, now() - query_start as duration FROM pg_stat_activity WHERE state='active' AND now() - query_start > interval '5 minutes';")
slow = cur.fetchall()
print(f"Requêtes lentes (>5min): {len(slow)}")

cur.close()
conn.close()
