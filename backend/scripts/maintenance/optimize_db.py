"""
optimize_db.py – Spotify AI Agent
--------------------------------
Optimiert alle Datenbanken (Postgres, MongoDB, Redis) für Performance, Security, ML/AI und Compliance.
Rollen: DBA/Data Engineer, Backend Senior, Security Specialist, Architecte IA
"""

import os
import logging
import psycopg2
from pymongo import MongoClient
import redis

logging.basicConfig(level=logging.INFO)

# Postgres Optimierung
try:
    conn = psycopg2.connect(os.environ["POSTGRES_DSN"])
    cur = conn.cursor()
    cur.execute("VACUUM FULL;")
    cur.execute("REINDEX DATABASE spotify_ai_agent;")
    cur.execute("ANALYZE;")
    conn.commit()
    cur.close()
    conn.close()
    logging.info("[OK] Postgres optimiert.")
except Exception as e:
    logging.warning(f"[WARN] Postgres: {e}")

# MongoDB Optimierung
try:
    mongo = MongoClient(os.environ["MONGO_URI"])
    db = mongo["spotify_ai_agent"]
    db.command("compact", "users")
    logging.info("[OK] MongoDB optimiert.")
except Exception as e:
    logging.warning(f"[WARN] MongoDB: {e}")

# Redis Optimierung
try:
    r = redis.Redis.from_url(os.environ["REDIS_URL"])
    r.bgrewriteaof()
    logging.info("[OK] Redis optimiert.")
except Exception as e:
    logging.warning(f"[WARN] Redis: {e}")
