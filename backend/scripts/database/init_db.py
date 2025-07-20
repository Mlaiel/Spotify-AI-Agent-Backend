"""
init_db.py – Spotify AI Agent
----------------------------
Initialisiert alle Datenbanken (Postgres, MongoDB, Redis) mit Grundstrukturen, Usern und Security-Policies.
Rollen: DBA/Data Engineer, Security Specialist, Lead Dev
"""

import os
import logging
import psycopg2
from pymongo import MongoClient
import redis

logging.basicConfig(level=logging.INFO)

# Postgres Initialisierung
try:
    conn = psycopg2.connect(os.environ["POSTGRES_DSN"])
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS citext;")
    cur.execute("CREATE ROLE app_user WITH LOGIN PASSWORD %s;", (os.environ["APP_DB_PASSWORD"],)
    cur.execute("CREATE DATABASE spotify_ai_agent;")
    conn.commit()
    cur.close()
    conn.close()
    logging.info("[OK] Postgres initialisiert.")
except Exception as e:
    logging.warning(f"[WARN] Postgres: {e}")

# MongoDB Initialisierung
try:
    mongo = MongoClient(os.environ["MONGO_URI"])
    db = mongo["spotify_ai_agent"]
    db.create_collection("users", capped=False)
    db.create_collection("logs", capped=False)
    logging.info("[OK] MongoDB initialisiert.")
except Exception as e:
    logging.warning(f"[WARN] MongoDB: {e}")

# Redis Initialisierung
try:
    r = redis.Redis.from_url(os.environ["REDIS_URL"])
    r.set("init", "ok")
    logging.info("[OK] Redis initialisiert.")
except Exception as e:
    logging.warning(f"[WARN] Redis: {e}")
