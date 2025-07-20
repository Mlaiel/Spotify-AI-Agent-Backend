"""
seed_data.py – Spotify AI Agent
------------------------------
Befüllt alle Datenbanken (Postgres, MongoDB, Redis) mit Seed-Daten für Entwicklung, Test und Demo.
Rollen: DBA/Data Engineer, ML Engineer, Lead Dev
"""

import os
import logging
import psycopg2
from pymongo import MongoClient
import redis

logging.basicConfig(level=logging.INFO)

# Postgres Seed
try:
    conn = psycopg2.connect(os.environ["POSTGRES_DSN"])
    cur = conn.cursor()
    cur.execute("INSERT INTO users (username, email) VALUES ('demo', 'demo@spotify.com') ON CONFLICT DO NOTHING;")
    conn.commit()
    cur.close()
    conn.close()
    logging.info("[OK] Postgres Seed abgeschlossen.")
except Exception as e:
    logging.warning(f"[WARN] Postgres: {e}")

# MongoDB Seed
try:
    mongo = MongoClient(os.environ["MONGO_URI"])
    db = mongo["spotify_ai_agent"]
    db.users.insert_one({"username": "demo", "email": "demo@spotify.com"})
    logging.info("[OK] MongoDB Seed abgeschlossen.")
except Exception as e:
    logging.warning(f"[WARN] MongoDB: {e}")

# Redis Seed
try:
    r = redis.Redis.from_url(os.environ["REDIS_URL"])
    r.set("demo_user", "demo@spotify.com")
    logging.info("[OK] Redis Seed abgeschlossen.")
except Exception as e:
    logging.warning(f"[WARN] Redis: {e}")
